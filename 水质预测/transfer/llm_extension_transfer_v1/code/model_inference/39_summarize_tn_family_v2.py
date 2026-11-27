"""Summarize TN_FAMILY_V2 without fitting any model."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tn_family_v2"
PREDICTION_DIR = RUN / "predictions"
OLD_RUN = ROOT / "runs" / "tabiclv2_pilot"
DEPTHS = ("surface", "middle", "bottom")
TASKS = [f"tn_{depth}" for depth in DEPTHS]
MODELS = ("xgboost", "tabiclv2")
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
BOOTSTRAP_SEED = 20260921
N_BOOTSTRAP = 1000


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def prediction_path(task: str, model: str, fold: str) -> Path:
    return PREDICTION_DIR / f"{task}_{model}_{fold}.csv"


def load_v2_predictions(task: str, model: str) -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        path = prediction_path(task, model, fold)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"obs_id": "string", "station_id": "string"})
        required = {"task", "obs_id", "station_id", "date", "depth", "fold", "y_true", "y_pred", "residual", "model", "feature_set", "split_version"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"{path} 缺少字段: {sorted(required - set(frame.columns))}")
        if frame["obs_id"].duplicated().any():
            raise RuntimeError(f"{path} 内 obs_id 重复")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result["obs_id"].duplicated().any():
        raise RuntimeError(f"{task}/{model}: pooled OOF 中 obs_id 重复")
    return result


def pooled_table(paired_by_task: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    comparisons = []
    for task, paired in paired_by_task.items():
        depth = task.removeprefix("tn_")
        for model, prediction_column in (("xgboost", "y_pred_xgb"), ("tabiclv2", "y_pred_tabicl")):
            values = metric_values(paired["y_true"].to_numpy(float), paired[prediction_column].to_numpy(float))
            rows.append(
                {
                    "task": task,
                    "depth": depth,
                    "model": model,
                    "n_obs": len(paired),
                    "n_station": paired["station_id"].nunique(),
                    **values,
                }
            )
        xgb = rows[-2]
        tabicl = rows[-1]
        comparisons.append(
            {
                "task": task,
                "depth": depth,
                "XGB_R2": xgb["R2"],
                "TabICL_R2": tabicl["R2"],
                "delta_R2": tabicl["R2"] - xgb["R2"],
                "XGB_RMSE": xgb["RMSE"],
                "TabICL_RMSE": tabicl["RMSE"],
                "delta_RMSE": tabicl["RMSE"] - xgb["RMSE"],
                "XGB_MAE": xgb["MAE"],
                "TabICL_MAE": tabicl["MAE"],
                "delta_MAE": tabicl["MAE"] - xgb["MAE"],
                "n_obs": xgb["n_obs"],
                "n_station": xgb["n_station"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def build_paired(task: str) -> pd.DataFrame:
    xgb = load_v2_predictions(task, "xgboost").rename(columns={"y_true": "y_true_xgb", "y_pred": "y_pred_xgb"})
    tabicl = load_v2_predictions(task, "tabiclv2").rename(columns={"y_true": "y_true_tabicl", "y_pred": "y_pred_tabicl"})
    paired = xgb.merge(
        tabicl[["obs_id", "station_id", "date", "depth", "fold", "y_true_tabicl", "y_pred_tabicl"]],
        on=["obs_id", "station_id", "date", "depth", "fold"],
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(xgb) or not np.allclose(paired["y_true_xgb"], paired["y_true_tabicl"]):
        raise RuntimeError(f"{task}: XGBoost/TabICLv2 逐样本目标不一致")
    return paired.rename(columns={"y_true_xgb": "y_true"})


def station_metrics(paired_by_task: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    for task, paired in paired_by_task.items():
        for station_id, group in paired.groupby("station_id", sort=True):
            y_true = group["y_true"].to_numpy(float)
            xgb = group["y_pred_xgb"].to_numpy(float)
            tabicl = group["y_pred_tabicl"].to_numpy(float)
            r2_reliable = len(group) >= 5 and np.nanstd(y_true) > 1e-12
            xgb_values = metric_values(y_true, xgb)
            tabicl_values = metric_values(y_true, tabicl)
            xgb_r2 = xgb_values["R2"] if r2_reliable else np.nan
            tabicl_r2 = tabicl_values["R2"] if r2_reliable else np.nan
            rows.append(
                {
                    "task": task,
                    "depth": task.removeprefix("tn_"),
                    "station_id": station_id,
                    "n_obs": len(group),
                    "target_mean": float(np.mean(y_true)),
                    "target_sd": float(np.std(y_true, ddof=1)) if len(group) > 1 else 0.0,
                    "XGB_R2": xgb_r2,
                    "TabICL_R2": tabicl_r2,
                    "delta_R2": tabicl_r2 - xgb_r2 if r2_reliable else np.nan,
                    "XGB_RMSE": xgb_values["RMSE"],
                    "TabICL_RMSE": tabicl_values["RMSE"],
                    "delta_RMSE": tabicl_values["RMSE"] - xgb_values["RMSE"],
                    "XGB_MAE": xgb_values["MAE"],
                    "TabICL_MAE": tabicl_values["MAE"],
                    "delta_MAE": tabicl_values["MAE"] - xgb_values["MAE"],
                    "r2_status": "OK" if r2_reliable else "R2_UNSTABLE",
                }
            )
        station = pd.DataFrame([row for row in rows if row["task"] == task])
        valid_r2 = station.loc[station["r2_status"].eq("OK")]
        summaries.append(
            {
                "task": task,
                "depth": task.removeprefix("tn_"),
                "n_station": len(station),
                "n_station_win_rmse": int((station["delta_RMSE"] < 0).sum()),
                "win_rate_rmse": float((station["delta_RMSE"] < 0).mean()),
                "n_station_win_mae": int((station["delta_MAE"] < 0).sum()),
                "win_rate_mae": float((station["delta_MAE"] < 0).mean()),
                "n_station_valid_r2": len(valid_r2),
                "n_station_win_r2": int((valid_r2["delta_R2"] > 0).sum()),
                "win_rate_r2": float((valid_r2["delta_R2"] > 0).mean()) if len(valid_r2) else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def bootstrap(paired_by_task: dict[str, pd.DataFrame], n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for task, paired in paired_by_task.items():
        stations = np.array(sorted(paired["station_id"].astype(str).unique()))
        station_indices = {
            station: np.flatnonzero(paired["station_id"].astype(str).to_numpy() == station)
            for station in stations
        }
        y_true_all = paired["y_true"].to_numpy(float)
        xgb_all = paired["y_pred_xgb"].to_numpy(float)
        tabicl_all = paired["y_pred_tabicl"].to_numpy(float)
        values = []
        for iteration in range(n_bootstrap):
            sampled = rng.choice(stations, size=len(stations), replace=True)
            indices = np.concatenate([station_indices[station] for station in sampled])
            y_true = y_true_all[indices]
            xgb = metric_values(y_true, xgb_all[indices])
            tabicl = metric_values(y_true, tabicl_all[indices])
            values.append(
                {
                    "delta_R2": tabicl["R2"] - xgb["R2"],
                    "delta_RMSE": tabicl["RMSE"] - xgb["RMSE"],
                    "delta_MAE": tabicl["MAE"] - xgb["MAE"],
                }
            )
        frame = pd.DataFrame(values)
        for metric in ["delta_R2", "delta_RMSE", "delta_MAE"]:
            rows.append(
                {
                    "task": task,
                    "depth": task.removeprefix("tn_"),
                    "metric": metric,
                    "n_bootstrap": n_bootstrap,
                    "seed": BOOTSTRAP_SEED,
                    "median": float(frame[metric].median()),
                    "low_2p5": float(frame[metric].quantile(0.025)),
                    "high_97p5": float(frame[metric].quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def fold_consistency(paired_by_task: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for task, paired in paired_by_task.items():
        for fold, group in paired.groupby("fold", sort=True):
            xgb = metric_values(group["y_true"].to_numpy(float), group["y_pred_xgb"].to_numpy(float))
            tabicl = metric_values(group["y_true"].to_numpy(float), group["y_pred_tabicl"].to_numpy(float))
            rows.append(
                {
                    "task": task,
                    "depth": task.removeprefix("tn_"),
                    "fold": fold,
                    "n_obs": len(group),
                    "n_station": group["station_id"].nunique(),
                    "XGB_R2": xgb["R2"],
                    "TabICL_R2": tabicl["R2"],
                    "delta_R2": tabicl["R2"] - xgb["R2"],
                    "XGB_RMSE": xgb["RMSE"],
                    "TabICL_RMSE": tabicl["RMSE"],
                    "delta_RMSE": tabicl["RMSE"] - xgb["RMSE"],
                    "XGB_MAE": xgb["MAE"],
                    "TabICL_MAE": tabicl["MAE"],
                    "delta_MAE": tabicl["MAE"] - xgb["MAE"],
                    "error_direction": "improved" if tabicl["RMSE"] < xgb["RMSE"] and tabicl["MAE"] < xgb["MAE"] else "not_both_improved",
                }
            )
    return pd.DataFrame(rows)


def classify(depth: str, comparison: pd.Series, station_summary: pd.Series, boot: pd.DataFrame) -> dict[str, object]:
    rmse = boot.loc[boot["metric"].eq("delta_RMSE")].iloc[0]
    mae = boot.loc[boot["metric"].eq("delta_MAE")].iloc[0]
    both_error_improved = comparison["delta_RMSE"] < 0 and comparison["delta_MAE"] < 0
    robust = (
        both_error_improved
        and station_summary["win_rate_rmse"] >= 0.60
        and station_summary["win_rate_mae"] >= 0.60
        and rmse["high_97p5"] < 0
        and mae["high_97p5"] < 0
    )
    some_gain = comparison["delta_RMSE"] < 0 or comparison["delta_MAE"] < 0 or comparison["delta_R2"] > 0
    moderate = not robust and some_gain and station_summary["win_rate_rmse"] >= 0.50 and station_summary["win_rate_mae"] >= 0.50
    if robust:
        classification = "ROBUST"
    elif moderate:
        classification = "MODERATE"
    elif some_gain:
        classification = "MIXED"
    else:
        classification = "NO_GAIN"
    return {
        "depth": depth,
        "XGB_R2": comparison["XGB_R2"],
        "TabICL_R2": comparison["TabICL_R2"],
        "delta_R2": comparison["delta_R2"],
        "delta_RMSE": comparison["delta_RMSE"],
        "delta_MAE": comparison["delta_MAE"],
        "RMSE_win_rate": station_summary["win_rate_rmse"],
        "MAE_win_rate": station_summary["win_rate_mae"],
        "bootstrap_RMSE_median": rmse["median"],
        "bootstrap_RMSE_low": rmse["low_2p5"],
        "bootstrap_RMSE_high": rmse["high_97p5"],
        "bootstrap_MAE_median": mae["median"],
        "bootstrap_MAE_low": mae["low_2p5"],
        "bootstrap_MAE_high": mae["high_97p5"],
        "classification": classification,
    }


def load_old_tn_bottom() -> pd.DataFrame | None:
    path = OLD_RUN / "14_tn_bottom_predictions.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={"obs_id": "string", "station_id": "string"})
    if "validation" in frame.columns:
        frame = frame.loc[frame["validation"].astype(str).isin({"station", "station_holdout"})].copy()
    if frame.empty:
        return None
    frame["model_std"] = frame["model"].astype(str).str.lower().map({"xgb": "xgboost", "xgboost": "xgboost", "tabicl": "tabiclv2", "tabiclv2": "tabiclv2"})
    frame = frame.loc[frame["model_std"].notna()].copy()
    return frame


def summarize_old_direction() -> dict[str, object] | None:
    old = load_old_tn_bottom()
    if old is None:
        return None
    xgb = old.loc[old["model_std"].eq("xgboost")].rename(columns={"y_true": "y_true_xgb", "y_pred": "y_pred_xgb"})
    tab = old.loc[old["model_std"].eq("tabiclv2")].rename(columns={"y_true": "y_true_tabicl", "y_pred": "y_pred_tabicl"})
    keys = ["obs_id", "station_id", "fold"]
    paired = xgb.merge(tab[keys + ["y_true_tabicl", "y_pred_tabicl"]], on=keys, validate="one_to_one")
    paired = paired.rename(columns={"y_true_xgb": "y_true"})
    pooled_xgb = metric_values(paired["y_true"].to_numpy(float), paired["y_pred_xgb"].to_numpy(float))
    pooled_tab = metric_values(paired["y_true"].to_numpy(float), paired["y_pred_tabicl"].to_numpy(float))
    station = []
    for station_id, group in paired.groupby("station_id", sort=True):
        xgb_values = metric_values(group["y_true"].to_numpy(float), group["y_pred_xgb"].to_numpy(float))
        tab_values = metric_values(group["y_true"].to_numpy(float), group["y_pred_tabicl"].to_numpy(float))
        station.append({"station_id": station_id, "delta_RMSE": tab_values["RMSE"] - xgb_values["RMSE"], "delta_MAE": tab_values["MAE"] - xgb_values["MAE"]})
    station_frame = pd.DataFrame(station)
    return {
        "split_label": "old_tn_bottom_pilot",
        "n_obs": len(paired),
        "n_station": paired["station_id"].nunique(),
        "XGB_R2": pooled_xgb["R2"],
        "TabICL_R2": pooled_tab["R2"],
        "delta_R2": pooled_tab["R2"] - pooled_xgb["R2"],
        "delta_RMSE": pooled_tab["RMSE"] - pooled_xgb["RMSE"],
        "delta_MAE": pooled_tab["MAE"] - pooled_xgb["MAE"],
        "RMSE_win_rate": float((station_frame["delta_RMSE"] < 0).mean()),
        "MAE_win_rate": float((station_frame["delta_MAE"] < 0).mean()),
    }


def write_replication(new_row: pd.Series, new_station_summary: pd.Series) -> pd.DataFrame:
    old = summarize_old_direction()
    new = {
        "split_label": "TN_FAMILY_V2_UNIFIED_STATION_SPLIT",
        "n_obs": new_row["n_obs"],
        "n_station": new_row["n_station"],
        "XGB_R2": new_row["XGB_R2"],
        "TabICL_R2": new_row["TabICL_R2"],
        "delta_R2": new_row["delta_R2"],
        "delta_RMSE": new_row["delta_RMSE"],
        "delta_MAE": new_row["delta_MAE"],
        "RMSE_win_rate": new_station_summary["win_rate_rmse"],
        "MAE_win_rate": new_station_summary["win_rate_mae"],
    }
    rows = []
    if old is not None:
        rows.append(old)
    rows.append(new)
    frame = pd.DataFrame(rows)
    if old is None:
        frame["replication_classification"] = "OLD_RESULT_UNAVAILABLE"
    else:
        old_direction = old["delta_RMSE"] < 0 and old["delta_MAE"] < 0 and old["RMSE_win_rate"] > 0.5 and old["MAE_win_rate"] > 0.5
        new_direction = new["delta_RMSE"] < 0 and new["delta_MAE"] < 0 and new["RMSE_win_rate"] > 0.5 and new["MAE_win_rate"] > 0.5
        classification = "REPLICATED" if old_direction and new_direction else "PARTIALLY_REPLICATED" if old_direction != new_direction or old["delta_R2"] * new["delta_R2"] >= 0 else "NOT_REPLICATED"
        frame["replication_classification"] = classification
    return frame


def write_decision(depth_table: pd.DataFrame, replication: pd.DataFrame, station_summary: pd.DataFrame, bootstrap_table: pd.DataFrame, fold_table: pd.DataFrame) -> str:
    supported = int(depth_table["classification"].isin(["ROBUST", "MODERATE"]).sum())
    no_gain = int(depth_table["classification"].eq("NO_GAIN").sum())
    if supported >= 2 and no_gain <= 1:
        family_decision = "TN_FAMILY_SUPPORTED"
    elif supported == 1:
        family_decision = "TN_FAMILY_PARTIAL"
    else:
        family_decision = "TN_FAMILY_NOT_SUPPORTED"
    replication_class = str(replication["replication_classification"].iloc[0])
    lines = [
        "# TN_FAMILY_V2 决策报告",
        "",
        f"- 生成时间：`{datetime.now().isoformat(timespec='seconds')}`",
        "- 协议：`TN_FAMILY_V2_UNIFIED_STATION_SPLIT`",
        f"- TN family 总判定：**{family_decision}**",
        f"- 旧 TN bottom 方向重复验证：**{replication_class}**",
        "",
        "## 三层结果",
        "",
        "| 水层 | XGBoost R² | TabICLv2 R² | ΔR² | ΔRMSE | ΔMAE | RMSE胜率 | MAE胜率 | 分类 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in depth_table.itertuples(index=False):
        lines.append(f"| {row.depth} | {row.XGB_R2:.4f} | {row.TabICL_R2:.4f} | {row.delta_R2:+.4f} | {row.delta_RMSE:+.4f} | {row.delta_MAE:+.4f} | {row.RMSE_win_rate:.3f} | {row.MAE_win_rate:.3f} | {row.classification} |")
    lines += [
        "",
        "## 判定口径",
        "",
        "- ROBUST：pooled ΔRMSE 与 ΔMAE 均小于 0，站点 RMSE/MAE 胜率均至少 0.60，且两项 bootstrap 95% 区间上限均小于 0。",
        "- MODERATE：存在总体误差改善，站点两项误差胜率均不低于 0.50，但未同时满足 ROBUST 条件。",
        "- MIXED：存在局部改善，但总体误差或站点胜率证据不一致。",
        "- NO_GAIN：没有稳定的总体和站点级改善证据。",
        "",
        "## 旧 TN bottom 对照",
        "",
        replication.to_string(index=False),
        "",
        "## 结论",
        "",
        f"TabICLv2 的空间泛化优势在 TN 三个水层中属于 `{family_decision}`。该判断基于统一的 84 个共同站点和同一 canonical station fold，不与旧 pilot 的绝对 R² 直接等同。",
        "",
        "TN_FAMILY_V2 完成后停止新增模型实验，不自动进入 TP 或其他算法路线。",
    ]
    (RUN / "14_tn_family_v2_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return family_decision


def main() -> int:
    paired_by_task = {task: build_paired(task) for task in TASKS}
    pooled_models, pooled_comparison = pooled_table(paired_by_task)
    station_frame, station_summary = station_metrics(paired_by_task)
    bootstrap_table = bootstrap(paired_by_task)
    fold_table = fold_consistency(paired_by_task)
    classification_rows = []
    for row in pooled_comparison.itertuples(index=False):
        comparison = pd.Series(row._asdict())
        summary = station_summary.loc[station_summary["task"].eq(row.task)].iloc[0]
        boot = bootstrap_table.loc[bootstrap_table["task"].eq(row.task)]
        classification_rows.append(classify(row.depth, comparison, summary, boot))
    depth_table = pd.DataFrame(classification_rows)
    replication = write_replication(
        pooled_comparison.loc[pooled_comparison["task"].eq("tn_bottom")].iloc[0],
        station_summary.loc[station_summary["task"].eq("tn_bottom")].iloc[0],
    )

    pooled_models.to_csv(RUN / "08_tn_family_v2_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    pooled_comparison.to_csv(RUN / "08_tn_family_v2_pooled_comparison.csv", index=False, encoding="utf-8-sig")
    station_frame.to_csv(RUN / "09_tn_family_v2_station_metrics.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(RUN / "09_tn_family_v2_station_win_rates.csv", index=False, encoding="utf-8-sig")
    bootstrap_table.to_csv(RUN / "10_tn_family_v2_bootstrap.csv", index=False, encoding="utf-8-sig")
    fold_table.to_csv(RUN / "11_tn_family_v2_fold_consistency.csv", index=False, encoding="utf-8-sig")
    replication.to_csv(RUN / "12_tn_bottom_replication.csv", index=False, encoding="utf-8-sig")
    depth_table.to_csv(RUN / "13_tn_depth_generalization.csv", index=False, encoding="utf-8-sig")
    decision = write_decision(depth_table, replication, station_summary, bootstrap_table, fold_table)

    print("TN_FAMILY_V2_SUMMARY")
    print(depth_table.to_string(index=False))
    print(f"TN_FAMILY_DECISION={decision}")
    print(f"TN_BOTTOM_REPLICATION={replication['replication_classification'].iloc[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
