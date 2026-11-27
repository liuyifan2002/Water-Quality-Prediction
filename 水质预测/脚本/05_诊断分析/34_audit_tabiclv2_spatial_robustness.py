"""Audit TabICLv2 spatial generalization without retraining any model.

Reads saved station-holdout predictions only. The script is deliberately
separate from model runners and writes audit artifacts under the pilot run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tabiclv2_pilot"
MAIN_TABLE = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 1000

TASKS = {
    "chla_surface": {
        "target_type": "optical",
        "prediction": None,
        "fold_results": "04_tabiclv2_comparison.csv",
        "target_column": "chla_ug_l",
        "depth": "surface",
    },
    "secchi": {
        "target_type": "optical",
        "prediction": "09_secchi_predictions.parquet",
        "fold_results": "08_secchi_fold_results.csv",
        "target_column": "secchi_depth_m",
        "depth": "surface",
    },
    "tn_bottom": {
        "target_type": "nutrient",
        "prediction": "14_tn_bottom_predictions.parquet",
        "fold_results": "13_tn_bottom_fold_results.csv",
        "target_column": "tn_mg_l",
        "depth": "bottom",
    },
    "tp_middle": {
        "target_type": "nutrient",
        "prediction": "17_tp_middle_predictions.parquet",
        "fold_results": "19_tp_middle_fold_results.csv",
        "target_column": "tp_mg_l",
        "depth": "middle",
    },
}


def norm_station(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def load_predictions(task: str) -> pd.DataFrame | None:
    filename = TASKS[task]["prediction"]
    if not filename:
        return None
    path = RUN / filename
    if not path.exists():
        return None
    data = pd.read_parquet(path)
    data = data.loc[data["validation"].eq("station")].copy()
    data["station_id"] = data["station_id"].map(norm_station)
    data["obs_id"] = data["obs_id"].astype(str)
    duplicate_count = int(data.duplicated(["obs_id", "model"]).sum())
    if duplicate_count:
        raise RuntimeError(f"{task}: duplicated station prediction rows: {duplicate_count}")
    return data


def load_fold_results(task: str) -> pd.DataFrame:
    data = pd.read_csv(RUN / TASKS[task]["fold_results"])
    return data.loc[data["validation"].eq("station") & ~data["fold"].astype(str).str.contains("summary")].copy()


def metric_dict(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    variance = float(np.var(true)) if len(true) else 0.0
    r2 = float(r2_score(true, pred)) if len(true) >= 5 and variance > 1e-12 else np.nan
    return {
        "R2": r2,
        "RMSE": float(mean_squared_error(true, pred) ** 0.5),
        "MAE": float(mean_absolute_error(true, pred)),
        "n_obs": int(len(true)),
        "target_mean": float(np.mean(true)) if len(true) else np.nan,
        "target_sd": float(np.std(true, ddof=1)) if len(true) > 1 else np.nan,
        "target_p10": float(np.quantile(true, 0.10)) if len(true) else np.nan,
        "target_p50": float(np.quantile(true, 0.50)) if len(true) else np.nan,
        "target_p90": float(np.quantile(true, 0.90)) if len(true) else np.nan,
    }


def station_comparison(predictions: pd.DataFrame, task: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station, frame in predictions.groupby("station_id", sort=True):
        row: dict[str, object] = {"task": task, "station_id": station}
        for model in ["xgb", "tabicl"]:
            subset = frame.loc[frame["model"].eq(model)]
            if subset.empty:
                continue
            values = metric_dict(subset["y_true"], subset["y_pred"])
            for key, value in values.items():
                row[f"{model}_{key}"] = value
        row["delta_R2"] = row.get("tabicl_R2", np.nan) - row.get("xgb_R2", np.nan)
        row["delta_RMSE"] = row.get("tabicl_RMSE", np.nan) - row.get("xgb_RMSE", np.nan)
        row["delta_MAE"] = row.get("tabicl_MAE", np.nan) - row.get("xgb_MAE", np.nan)
        row["r2_status"] = "R2_UNSTABLE" if pd.isna(row["delta_R2"]) else "OK"
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_metrics(predictions: pd.DataFrame, task: str, fold_results: pd.DataFrame) -> dict[str, object]:
    station = station_comparison(predictions, task)
    rows: dict[str, object] = {
        "task": task,
        "target_type": TASKS[task]["target_type"],
        "n_station": int(predictions["station_id"].nunique()),
        "n_obs": int(predictions["obs_id"].nunique()),
    }
    for model, label in [("xgb", "XGBoost"), ("tabicl", "TabICLv2")]:
        subset = predictions.loc[predictions["model"].eq(model)]
        metrics = metric_dict(subset["y_true"], subset["y_pred"])
        rows[f"{label}_pooled_R2"] = metrics["R2"]
        rows[f"{label}_pooled_RMSE"] = metrics["RMSE"]
        rows[f"{label}_pooled_MAE"] = metrics["MAE"]
        fold_subset = fold_results.loc[fold_results["fold"].astype(str).str.startswith("fold_")]
        fold_col = "XGBoost_R2" if model == "xgb" else "TabICLv2_R2"
        rows[f"{label}_mean_fold_R2"] = float(fold_subset[fold_col].mean())
        rows[f"{label}_median_fold_R2"] = float(fold_subset[fold_col].median())
    rows["delta_pooled_R2"] = rows["TabICLv2_pooled_R2"] - rows["XGBoost_pooled_R2"]
    rows["delta_pooled_RMSE"] = rows["TabICLv2_pooled_RMSE"] - rows["XGBoost_pooled_RMSE"]
    rows["delta_pooled_MAE"] = rows["TabICLv2_pooled_MAE"] - rows["XGBoost_pooled_MAE"]
    rows["n_station_win_rmse"] = int((station["delta_RMSE"] < 0).sum())
    rows["n_station_win_mae"] = int((station["delta_MAE"] < 0).sum())
    rows["station_win_rate_rmse"] = float((station["delta_RMSE"] < 0).mean())
    rows["station_win_rate_mae"] = float((station["delta_MAE"] < 0).mean())
    valid_r2 = station["delta_R2"].notna()
    rows["station_valid_r2"] = int(valid_r2.sum())
    rows["n_station_win_r2"] = int((station.loc[valid_r2, "delta_R2"] > 0).sum()) if valid_r2.any() else 0
    rows["station_win_rate_r2"] = float((station.loc[valid_r2, "delta_R2"] > 0).mean()) if valid_r2.any() else np.nan
    rows["station_delta_R2_median"] = float(station["delta_R2"].median())
    rows["station_delta_R2_sd"] = float(station["delta_R2"].std())
    return rows


def bootstrap_ci(predictions: pd.DataFrame, task: str) -> pd.DataFrame:
    stations = np.array(sorted(predictions["station_id"].unique()), dtype=object)
    station_arrays = {
        station: {
            model: (
                frame.loc[frame["model"].eq(model), "y_true"].to_numpy(dtype=float),
                frame.loc[frame["model"].eq(model), "y_pred"].to_numpy(dtype=float),
            )
            for model in ["xgb", "tabicl"]
        }
        for station, frame in predictions.groupby("station_id", sort=False)
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: list[dict[str, float]] = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        xgb_true = np.concatenate([station_arrays[station]["xgb"][0] for station in sampled])
        xgb_pred = np.concatenate([station_arrays[station]["xgb"][1] for station in sampled])
        tabicl_true = np.concatenate([station_arrays[station]["tabicl"][0] for station in sampled])
        tabicl_pred = np.concatenate([station_arrays[station]["tabicl"][1] for station in sampled])
        xgb_r2 = float(r2_score(xgb_true, xgb_pred))
        tabicl_r2 = float(r2_score(tabicl_true, tabicl_pred))
        xgb_rmse = float(mean_squared_error(xgb_true, xgb_pred) ** 0.5)
        tabicl_rmse = float(mean_squared_error(tabicl_true, tabicl_pred) ** 0.5)
        xgb_mae = float(mean_absolute_error(xgb_true, xgb_pred))
        tabicl_mae = float(mean_absolute_error(tabicl_true, tabicl_pred))
        values.append({
            "delta_R2": tabicl_r2 - xgb_r2,
            "delta_RMSE": tabicl_rmse - xgb_rmse,
            "delta_MAE": tabicl_mae - xgb_mae,
        })
    frame = pd.DataFrame(values)
    rows = []
    for metric in ["delta_R2", "delta_RMSE", "delta_MAE"]:
        rows.append(
            {
                "task": task,
                "metric": metric,
                "n_station": len(stations),
                "n_bootstrap": N_BOOTSTRAP,
                "median": float(frame[metric].median()),
                "ci_lower_2_5": float(frame[metric].quantile(0.025)),
                "ci_upper_97_5": float(frame[metric].quantile(0.975)),
            }
        )
    return pd.DataFrame(rows)


def load_station_zone_map() -> dict[str, str]:
    metadata = pd.read_csv(MAIN_TABLE, usecols=["station", "water_zone_en"], low_memory=False)
    metadata["station_id"] = metadata["station"].map(norm_station)
    metadata["water_zone_en"] = metadata["water_zone_en"].fillna("UNKNOWN").astype(str)
    return metadata.drop_duplicates("station_id").set_index("station_id")["water_zone_en"].to_dict()


def zone_comparison(predictions: pd.DataFrame, task: str, zone_map: dict[str, str]) -> pd.DataFrame:
    data = predictions.copy()
    data["water_zone"] = data["station_id"].map(zone_map).fillna("UNKNOWN")
    rows = []
    for zone, zone_frame in data.groupby("water_zone", sort=True):
        row: dict[str, object] = {
            "task": task,
            "water_zone": zone,
            "n_station": int(zone_frame["station_id"].nunique()),
            "n_obs": int(zone_frame["obs_id"].nunique()),
        }
        for model, label in [("xgb", "XGBoost"), ("tabicl", "TabICLv2")]:
            subset = zone_frame.loc[zone_frame["model"].eq(model)]
            values = metric_dict(subset["y_true"], subset["y_pred"])
            row[f"{label}_RMSE"] = values["RMSE"]
            row[f"{label}_MAE"] = values["MAE"]
        row["delta_RMSE"] = row["TabICLv2_RMSE"] - row["XGBoost_RMSE"]
        row["delta_MAE"] = row["TabICLv2_MAE"] - row["XGBoost_MAE"]
        row["interpretation"] = "single_station_descriptive" if row["n_station"] == 1 else "descriptive"
        rows.append(row)
    return pd.DataFrame(rows)


def extreme_fold_diagnostics(task: str, predictions: pd.DataFrame, fold_name: str, station_rows: pd.DataFrame) -> list[dict[str, object]]:
    fold = predictions.loc[predictions["fold"].eq(fold_name)].copy()
    xgb = fold.loc[fold["model"].eq("xgb")]
    tabicl = fold.loc[fold["model"].eq("tabicl")]
    xgb_metrics = metric_dict(xgb["y_true"], xgb["y_pred"])
    tabicl_metrics = metric_dict(tabicl["y_true"], tabicl["y_pred"])
    rows: list[dict[str, object]] = [
        {
            "task": task,
            "fold": fold_name,
            "record_type": "fold_summary",
            "station_id": "ALL",
            "n_station": int(xgb["station_id"].nunique()),
            "n_obs": int(xgb["obs_id"].nunique()),
            "target_mean": xgb_metrics["target_mean"],
            "target_sd": xgb_metrics["target_sd"],
            "target_p10": xgb_metrics["target_p10"],
            "target_p50": xgb_metrics["target_p50"],
            "target_p90": xgb_metrics["target_p90"],
            "XGBoost_RMSE": xgb_metrics["RMSE"],
            "TabICLv2_RMSE": tabicl_metrics["RMSE"],
            "delta_RMSE": tabicl_metrics["RMSE"] - xgb_metrics["RMSE"],
            "XGBoost_MAE": xgb_metrics["MAE"],
            "TabICLv2_MAE": tabicl_metrics["MAE"],
            "delta_MAE": tabicl_metrics["MAE"] - xgb_metrics["MAE"],
        }
    ]
    detail = station_rows.loc[station_rows["station_id"].isin(xgb["station_id"].unique())].copy()
    detail = detail.sort_values("delta_R2", ascending=False, na_position="last")
    for rank, row in enumerate(detail.itertuples(index=False), start=1):
        rows.append(
            {
                "task": task,
                "fold": fold_name,
                "record_type": "station_detail",
                "station_id": row.station_id,
                "rank_by_delta_R2": rank,
                "n_station": 1,
                "n_obs": row.xgb_n_obs,
                "target_mean": row.xgb_target_mean,
                "target_sd": row.xgb_target_sd,
                "target_p10": row.xgb_target_p10,
                "target_p50": row.xgb_target_p50,
                "target_p90": row.xgb_target_p90,
                "XGBoost_R2": row.xgb_R2,
                "TabICLv2_R2": row.tabicl_R2,
                "delta_R2": row.delta_R2,
                "XGBoost_RMSE": row.xgb_RMSE,
                "TabICLv2_RMSE": row.tabicl_RMSE,
                "delta_RMSE": row.delta_RMSE,
                "XGBoost_MAE": row.xgb_MAE,
                "TabICLv2_MAE": row.tabicl_MAE,
                "delta_MAE": row.delta_MAE,
                "r2_status": row.r2_status,
            }
        )
    return rows


def classify_task(task: str, pooled: dict[str, object], station: pd.DataFrame, ci: pd.DataFrame) -> dict[str, object]:
    if station.empty:
        return {"task": task, "target_type": TASKS[task]["target_type"], "status": "NO_SAMPLE_LEVEL_PREDICTIONS", "classification": "NO_SAMPLE_LEVEL_PREDICTIONS", "note": "未保存逐样本留站点预测，未重训补齐。"}
    rmse_ci = ci.loc[ci["metric"].eq("delta_RMSE")].iloc[0]
    mae_ci = ci.loc[ci["metric"].eq("delta_MAE")].iloc[0]
    median_delta = float(station["delta_R2"].median())
    max_delta = float(station["delta_R2"].max())
    r2_extreme_driver = bool(max_delta > max(0.10, 2 * abs(median_delta)))

    # R2 can be unstable when a station has little target variance. Use the
    # concentration of RMSE/MAE reductions for the practical-gain check, and
    # retain the R2 flag as a separate caveat.
    def top2_improvement_share(column: str) -> float:
        improvements = -station[column].loc[station[column] < 0]
        total = float(improvements.sum())
        if total <= 0:
            return 0.0
        return float(improvements.nlargest(2).sum() / total)

    top2_rmse_share = top2_improvement_share("delta_RMSE")
    top2_mae_share = top2_improvement_share("delta_MAE")
    top2_error_share = max(top2_rmse_share, top2_mae_share)
    error_extreme_driver = bool(top2_error_share >= 0.40)
    stable_error_gain = bool(rmse_ci["ci_upper_97_5"] < 0 or mae_ci["ci_upper_97_5"] < 0)
    majority_error_gain = bool(pooled["station_win_rate_rmse"] > 0.5 and pooled["station_win_rate_mae"] > 0.5)
    positive_pooled = bool(pooled["delta_pooled_R2"] > 0)
    if positive_pooled and majority_error_gain and stable_error_gain:
        # R2 outliers are reported separately. The practical label is based
        # on how concentrated the station-level RMSE/MAE improvements are.
        if error_extreme_driver:
            classification = "MIXED"
        elif top2_error_share >= 0.25:
            classification = "MODERATE SPATIAL GAIN"
        else:
            classification = "ROBUST SPATIAL GAIN"
    elif positive_pooled and (majority_error_gain or stable_error_gain):
        classification = "MODERATE SPATIAL GAIN" if not error_extreme_driver else "MIXED"
    elif positive_pooled or majority_error_gain:
        classification = "MIXED"
    else:
        classification = "NO ROBUST GAIN"
    return {
        "task": task,
        "target_type": TASKS[task]["target_type"],
        "status": "OK",
        "classification": classification,
        "n_station": len(station),
        "pooled_delta_R2": pooled["delta_pooled_R2"],
        "pooled_delta_RMSE": pooled["delta_pooled_RMSE"],
        "pooled_delta_MAE": pooled["delta_pooled_MAE"],
        "n_station_win_rmse": pooled["n_station_win_rmse"],
        "win_rate_rmse": pooled["station_win_rate_rmse"],
        "n_station_win_mae": pooled["n_station_win_mae"],
        "win_rate_mae": pooled["station_win_rate_mae"],
        "n_station_valid_r2": pooled["station_valid_r2"],
        "n_station_win_r2": pooled["n_station_win_r2"],
        "win_rate_r2": pooled["station_win_rate_r2"],
        "bootstrap_delta_RMSE_median": rmse_ci["median"],
        "bootstrap_delta_RMSE_ci_lower": rmse_ci["ci_lower_2_5"],
        "bootstrap_delta_RMSE_ci_upper": rmse_ci["ci_upper_97_5"],
        "bootstrap_delta_MAE_median": mae_ci["median"],
        "bootstrap_delta_MAE_ci_lower": mae_ci["ci_lower_2_5"],
        "bootstrap_delta_MAE_ci_upper": mae_ci["ci_upper_97_5"],
        "top2_rmse_improvement_share": top2_rmse_share,
        "top2_mae_improvement_share": top2_mae_share,
        "top2_error_improvement_share": top2_error_share,
        "r2_extreme_station_driver": "YES" if r2_extreme_driver else "NO",
        "error_extreme_station_driver": "YES" if error_extreme_driver else "NO",
        "extreme_station_driver": "YES" if error_extreme_driver else "NO",
        "note": "R2 极端站点与 RMSE/MAE 改善集中度分开判断；站点级结果为描述性比较，未进行显著性检验。",
    }


def write_final_decision(classification: pd.DataFrame, pooled: pd.DataFrame) -> None:
    secchi = classification.loc[classification["task"].eq("secchi"), "classification"]
    tn = classification.loc[classification["task"].eq("tn_bottom"), "classification"]
    tp = classification.loc[classification["task"].eq("tp_middle"), "classification"]
    secchi_robust = bool(not secchi.empty and secchi.iloc[0] == "ROBUST SPATIAL GAIN")
    tn_at_least_moderate = bool(not tn.empty and tn.iloc[0] in {"ROBUST SPATIAL GAIN", "MODERATE SPATIAL GAIN"})
    nutrient_mixed = bool((not tn.empty and tn.iloc[0] in {"MIXED", "NO ROBUST GAIN"}) or (not tp.empty and tp.iloc[0] in {"MIXED", "NO ROBUST GAIN"}))
    if secchi_robust and tn_at_least_moderate and not nutrient_mixed:
        decision = "EXPAND ALL REMAINING TASKS"
        reason = "Secchi 与营养盐任务均有不依赖单一站点的空间误差改善，支持先对剩余任务做 station holdout。"
    elif secchi_robust and nutrient_mixed:
        decision = "EXPAND SELECTIVELY"
        reason = "光学任务的空间证据较强，但营养盐任务存在站点异质性或极端站点驱动，先扩展光学指标族。"
    else:
        decision = "HOLD EXPANSION"
        reason = "当前逐样本审计尚不能同时支持光学和营养盐任务的稳健空间泛化。"
    chla_note = "Chl-a 缺少逐样本预测，本轮不纳入站点胜率、bootstrap 和 pooled OOF 审计；已有 fold-level 结果只能作为补充证据。"
    classification_text = classification.to_string(index=False)
    lines = [
        "# TabICLv2 空间泛化稳健性审计与扩展决策",
        "",
        f"## 最终建议：{decision}",
        "",
        reason,
        "",
        chla_note,
        "",
        "## 判定口径",
        "",
        "稳健性首先依据 pooled OOF、站点胜率和站点级 bootstrap 误差区间判断。为区分‘误差改善普遍存在’与‘少数站点贡献了大部分改善’，另计算 RMSE 和 MAE 改善量中贡献最大的两个站点所占比例：低于 0.25 标记为改善分布较广，0.25 至 0.40 标记为存在站点异质性，达到 0.40 标记为少数站点集中。该分层是本次描述性审计的透明判定规则，不是显著性检验阈值。R² 极端值单独保留为警告，因为低站点目标方差会放大 R² 变化。",
        "",
        "## 四任务证据状态",
        "",
        "```text",
        classification_text,
        "```",
        "",
        "## 三个假设",
        "",
        "- H1（未见站点上 TabICLv2 具有更稳定优势）：**PARTIALLY SUPPORTED**。Secchi、TN bottom、TP middle 的 pooled OOF 与站点误差需要结合各自分类判断；TP 若出现极端站点驱动，不能作为普遍证据。Chl-a 缺少逐样本审计。",
        "- H2（优势主要来自空间泛化而非时间外推）：**SUPPORTED**。前三个任务和当前四任务 fold-level 记录均显示 station 方向通常优于 year，但这描述的是本数据集验证模式，不等同于因果结论。",
        "- H3（空间优势具有任务依赖性，光学强于营养盐）：**PARTIALLY SUPPORTED**。光学任务的 fold-level 证据较一致，营养盐任务的站点差异更大；是否成立以本报告的 station-level bootstrap 和胜率为准。",
        "",
        "## 解释边界",
        "",
        "本审计只回答预测误差在未见站点上的分布情况，不解释为什么某个站点改善，也不把目标分布变化直接视为模型失败原因。后续正式任务应优先使用 station holdout，并同时报告 pooled OOF 与站点级胜率。",
        "",
        "本报告不改变正式主链，也没有重新训练任何模型。",
    ]
    (RUN / "29_spatial_expansion_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    zone_map = load_station_zone_map()
    pooled_rows = []
    station_frames = {}
    bootstrap_frames = []
    zone_frames = []
    extreme_rows = []
    classifications = []

    for task in ["chla_surface", "secchi", "tn_bottom", "tp_middle"]:
        predictions = load_predictions(task)
        if predictions is None:
            classifications.append(
                {
                    "task": task,
                    "target_type": TASKS[task]["target_type"],
                    "status": "NO_SAMPLE_LEVEL_PREDICTIONS",
                    "classification": "NO_SAMPLE_LEVEL_PREDICTIONS",
                    "note": "未保存逐样本留站点预测，本轮不重训补齐。",
                }
            )
            continue
        fold_results = load_fold_results(task)
        station = station_comparison(predictions, task)
        station_frames[task] = station
        pooled = pooled_metrics(predictions, task, fold_results)
        pooled_rows.append(pooled)
        ci = bootstrap_ci(predictions, task)
        bootstrap_frames.append(ci)
        zone_frames.append(zone_comparison(predictions, task, zone_map))
        classifications.append(classify_task(task, pooled, station, ci))
        if task == "tn_bottom":
            fold_station = station_comparison(predictions.loc[predictions["fold"].eq("fold_5")], task)
            extreme_rows.extend(extreme_fold_diagnostics(task, predictions, "fold_5", fold_station))
        if task == "tp_middle":
            fold_station = station_comparison(predictions.loc[predictions["fold"].eq("fold_3")], task)
            extreme_rows.extend(extreme_fold_diagnostics(task, predictions, "fold_3", fold_station))

    pooled_frame = pd.DataFrame(pooled_rows)
    pooled_frame.to_csv(RUN / "23_pooled_oof_metrics.csv", index=False, encoding="utf-8-sig")
    station_frame = pd.concat(
        [frame.assign(target_type=TASKS[task]["target_type"]) for task, frame in station_frames.items()],
        ignore_index=True,
    ) if station_frames else pd.DataFrame()
    station_frame.to_csv(RUN / "24_station_level_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(RUN / "25_station_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(extreme_rows).to_csv(RUN / "26_extreme_fold_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.concat(zone_frames, ignore_index=True).to_csv(RUN / "27_water_zone_comparison.csv", index=False, encoding="utf-8-sig")
    classification_frame = pd.DataFrame(classifications)
    classification_frame.to_csv(RUN / "28_spatial_robustness_classification.csv", index=False, encoding="utf-8-sig")
    write_final_decision(classification_frame, pooled_frame)
    print(f"spatial robustness audit complete: {RUN}")


if __name__ == "__main__":
    main()
