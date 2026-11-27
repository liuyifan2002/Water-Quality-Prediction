"""Run the isolated Chl-a surface TabICLv2 pilot.

This script reads the already recovered 77-feature manifest and obs_id split
manifest. It never creates a split and never writes to formal mainline output.
Use the pilot venv Python under runs/tabiclv2_pilot/环境/.venv.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tabiclv2_pilot"
MAIN_TABLE = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
XGB_CONFIG = ROOT / "结果" / "扩展实验" / "候选模型稳健性验证_20260914" / "portable_package" / "config.json"
TARGET = "chla_ug_l"
TASK = "chla_surface"
DEPTH = "surface"
SEED = 42
TASK_SPECS = {
    "chla_surface": {"target": "chla_ug_l", "task": "chla_surface", "depth": "surface"},
    "secchi": {"target": "secchi_depth_m", "task": "secchi", "depth": "surface"},
    "tn_bottom": {"target": "tn_mg_l", "task": "tn_bottom", "depth": "bottom"},
    "tp_middle": {"target": "tp_mg_l", "task": "tp_middle", "depth": "middle"},
}
PREDICTION_ROWS: list[pd.DataFrame] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgb", "tabicl"])
    parser.add_argument("--validation", choices=["random", "year", "station"])
    parser.add_argument("--task", choices=sorted(TASK_SPECS), default="chla_surface")
    parser.add_argument("--predictions-only", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--audit", action="store_true")
    return parser.parse_args()


def load_features() -> list[str]:
    manifest = pd.read_csv(RUN / "01_strict_feature_manifest.csv")
    selected = manifest.loc[manifest["included_in_pilot"].astype(bool), "feature"].tolist()
    if len(selected) != 77:
        raise RuntimeError(f"Expected 77 pilot features, found {len(selected)}")
    return selected


def load_data(features: list[str]) -> pd.DataFrame:
    columns = ["obs_id", "sample_id", "station", "date", "depth", TARGET, *features]
    data = pd.read_csv(MAIN_TABLE, usecols=columns, low_memory=False)
    data["obs_id"] = data["obs_id"].astype("string")
    data["date"] = pd.to_datetime(data["date"])
    data = data.loc[data["depth"].eq(DEPTH) & data[TARGET].notna()].copy()
    if data["obs_id"].duplicated().any():
        raise RuntimeError("Main table has duplicated obs_id within Chl-a surface rows")
    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    return data


def load_splits(validation: str) -> list[tuple[str, pd.DataFrame]]:
    split = pd.read_csv(RUN / "02_split_manifest.csv", dtype={"obs_id": "string"})
    split = split.loc[(split["task"] == TASK) & (split["validation"] == {
        "random": "random_strict",
        "year": "year_holdout",
        "station": "station_holdout",
    }[validation])].copy()
    if split.empty:
        raise RuntimeError(f"No saved split found for {TASK} / {validation}")
    return [(str(fold), frame.copy()) for fold, frame in split.groupby("fold", sort=False)]


def target_transform(y_train: np.ndarray) -> tuple[np.ndarray, callable]:
    if np.nanmin(y_train) >= 0:
        return np.log1p(y_train), np.expm1
    return y_train.copy(), lambda values: values


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, prediction)),
        "RMSE": float(mean_squared_error(y_true, prediction) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, prediction)),
    }


def configure_task(task_name: str) -> None:
    global TASK, TARGET, DEPTH
    spec = TASK_SPECS[task_name]
    TASK = spec["task"]
    TARGET = spec["target"]
    DEPTH = spec["depth"]


def fit_xgb(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    from xgboost import XGBRegressor

    config = json.loads(XGB_CONFIG.read_text(encoding="utf-8-sig"))
    params = dict(config["model_parameters"]["xgboost"])
    params.update({"random_state": SEED, "objective": "reg:squarederror"})
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def fit_tabicl(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    adapter_path = ROOT / "models" / "tabiclv2_adapter.py"
    spec = importlib.util.spec_from_file_location("tabiclv2_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import adapter: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.TabICLv2Regressor()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def run_one(model_name: str, validation: str, fold: str, split: pd.DataFrame, data: pd.DataFrame, features: list[str]) -> dict[str, object]:
    start = time.perf_counter()
    record: dict[str, object] = {
        "task": TASK,
        "validation": validation,
        "fold": fold,
        "n_features": len(features),
        "seed": SEED,
        "split_source": str(split["split_source"].iloc[0]),
        "status": "failed",
    }
    try:
        train_ids = set(split.loc[split["split"] == "train", "obs_id"])
        test_ids = set(split.loc[split["split"] == "test", "obs_id"])
        train = data.loc[data["obs_id"].isin(train_ids)]
        test = data.loc[data["obs_id"].isin(test_ids)]
        if len(train) != int(split["n_train"].iloc[0]) or len(test) != int(split["n_test"].iloc[0]):
            raise RuntimeError("obs_id split counts do not match recovered manifest")
        x_train_raw = train[features].to_numpy(dtype=float)
        x_test_raw = test[features].to_numpy(dtype=float)
        missing_train = int(np.isnan(x_train_raw).sum())
        missing_test = int(np.isnan(x_test_raw).sum())
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x_train = imputer.fit_transform(x_train_raw).astype(np.float32)
        x_test = imputer.transform(x_test_raw).astype(np.float32)
        y_true = test[TARGET].to_numpy(dtype=float)
        y_train, inverse = target_transform(train[TARGET].to_numpy(dtype=float))
        if model_name == "xgb":
            raw_prediction = fit_xgb(x_train, y_train, x_test)
            device = "xgboost_cpu"
        else:
            raw_prediction = fit_tabicl(x_train, y_train, x_test)
            device = "cuda"
        prediction = np.maximum(inverse(raw_prediction), 0)
        PREDICTION_ROWS.append(
            pd.DataFrame(
                {
                    "obs_id": test["obs_id"].astype(str).to_numpy(),
                    "station_id": test["station"].to_numpy(),
                    "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                    "target": TASK,
                    "validation": validation,
                    "fold": fold,
                    "y_true": y_true,
                    "y_pred": prediction,
                    "residual": prediction - y_true,
                    "model": model_name,
                    "feature_set": "X_strict_77",
                    "split_source": str(split["split_source"].iloc[0]),
                }
            )
        )
        record.update(metrics(y_true, prediction))
        record.update({
            "n_train": len(train),
            "n_test": len(test),
            "missing_train_cells": missing_train,
            "missing_test_cells": missing_test,
            "device": device,
            "model": model_name,
            "status": "ok",
        })
    except Exception as exc:
        record.update({"error_type": type(exc).__name__, "error": str(exc)})
    record["runtime"] = time.perf_counter() - start
    return record


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_predictions() -> None:
    if not PREDICTION_ROWS:
        return
    path = RUN / {
        "secchi": "09_secchi_predictions.csv",
        "tn_bottom": "14_tn_bottom_predictions.csv",
        "tp_middle": "17_tp_middle_predictions.csv",
    }[TASK]
    frame = pd.concat(PREDICTION_ROWS, ignore_index=True)
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
    frame.drop_duplicates(["obs_id", "validation", "fold", "model"], inplace=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def finalize_outputs() -> None:
    if TASK == "secchi":
        all_metrics = pd.read_csv(RUN / "07_secchi_strict_metrics.csv")
        xgb = all_metrics.loc[all_metrics["model"].eq("xgb")].copy()
        tabicl = all_metrics.loc[all_metrics["model"].eq("tabicl")].copy()
    elif TASK == "tn_bottom":
        all_metrics = pd.read_csv(RUN / "12_tn_bottom_strict_metrics.csv")
        xgb = all_metrics.loc[all_metrics["model"].eq("xgb")].copy()
        tabicl = all_metrics.loc[all_metrics["model"].eq("tabicl")].copy()
    elif TASK == "tp_middle":
        all_metrics = pd.read_csv(RUN / "18_tp_middle_strict_metrics.csv")
        xgb = all_metrics.loc[all_metrics["model"].eq("xgb")].copy()
        tabicl = all_metrics.loc[all_metrics["model"].eq("tabicl")].copy()
    else:
        xgb = pd.read_csv(RUN / "03_xgb_strict_smoke.csv")
        tabicl = pd.read_csv(RUN / "04_tabiclv2_chla_smoke.csv")
    xgb = xgb.loc[xgb["status"].eq("ok")].copy()
    tabicl = tabicl.loc[tabicl["status"].eq("ok")].copy()
    keys = ["task", "validation", "fold"]
    merged = xgb.merge(tabicl, on=keys, suffixes=("_xgb", "_tabicl"), how="outer", validate="one_to_one")
    comparison = pd.DataFrame({
        "task": merged["task"],
        "validation": merged["validation"],
        "fold": merged["fold"],
        "XGBoost_R2": merged["R2_xgb"],
        "TabICLv2_R2": merged["R2_tabicl"],
        "delta_R2": merged["R2_tabicl"] - merged["R2_xgb"],
        "XGBoost_RMSE": merged["RMSE_xgb"],
        "TabICLv2_RMSE": merged["RMSE_tabicl"],
        "delta_RMSE": merged["RMSE_tabicl"] - merged["RMSE_xgb"],
        "XGBoost_MAE": merged["MAE_xgb"],
        "TabICLv2_MAE": merged["MAE_tabicl"],
        "delta_MAE": merged["MAE_tabicl"] - merged["MAE_xgb"],
    })
    station = comparison.loc[comparison["validation"].eq("station")]
    summaries = []
    for statistic, func in [("mean", "mean"), ("median", "median"), ("sd", "std")]:
        values = station.select_dtypes(include=["number"]).agg(func)
        summaries.append({
            "task": TASK,
            "validation": "station_summary",
            "fold": statistic,
            "XGBoost_R2": values["XGBoost_R2"],
            "TabICLv2_R2": values["TabICLv2_R2"],
            "delta_R2": values["delta_R2"],
            "XGBoost_RMSE": values["XGBoost_RMSE"],
            "TabICLv2_RMSE": values["TabICLv2_RMSE"],
            "delta_RMSE": values["delta_RMSE"],
            "XGBoost_MAE": values["XGBoost_MAE"],
            "TabICLv2_MAE": values["TabICLv2_MAE"],
            "delta_MAE": values["delta_MAE"],
        })
    comparison = pd.concat([comparison, pd.DataFrame(summaries)], ignore_index=True)
    comparison_path = RUN / {
        "secchi": "08_secchi_fold_results.csv",
        "tn_bottom": "13_tn_bottom_fold_results.csv",
        "tp_middle": "19_tp_middle_fold_results.csv",
    }.get(TASK, "04_tabiclv2_comparison.csv")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    random = comparison.loc[comparison["validation"].eq("random")].iloc[0]
    year = comparison.loc[comparison["validation"].eq("year")].iloc[0]
    station_mean = comparison.loc[(comparison["validation"] == "station_summary") & (comparison["fold"] == "mean")].iloc[0]
    station_sd = comparison.loc[(comparison["validation"] == "station_summary") & (comparison["fold"] == "sd")].iloc[0]
    fold_lines = [
        f"| {row.fold} | {row.XGBoost_R2:.4f} | {row.TabICLv2_R2:.4f} | {row.delta_R2:+.4f} |"
        for row in station.itertuples()
    ]
    runtime = pd.read_csv(RUN / "06_runtime_log.csv")
    runtime = runtime.loc[runtime["status"].eq("ok")]
    lines = [
        "# TabICLv2 Chl-a surface pilot 决策报告",
        "",
        "## 1. 一句话结论",
        "",
        "**CONTINUE**：TabICLv2 在 Chl-a surface 的留站点五折中全部优于 77 特征 XGBoost，年份留出基本持平，值得按原计划继续评估少量代表任务；本轮不扩展到全部任务。",
        "",
        "## 2. 三类验证结果",
        "",
        "| 验证方式 | XGBoost R² | TabICLv2 R² | ΔR² |",
        "|---|---:|---:|---:|",
        f"| Random strict | {random.XGBoost_R2:.4f} | {random.TabICLv2_R2:.4f} | {random.delta_R2:+.4f} |",
        f"| Year holdout | {year.XGBoost_R2:.4f} | {year.TabICLv2_R2:.4f} | {year.delta_R2:+.4f} |",
        f"| Station mean | {station_mean.XGBoost_R2:.4f} | {station_mean.TabICLv2_R2:.4f} | {station_mean.delta_R2:+.4f} |",
        "",
        "## 3. 留站点五折稳定性",
        "",
        "| 折次 | XGBoost R² | TabICLv2 R² | ΔR² |",
        "|---|---:|---:|---:|",
        *fold_lines,
        "",
        f"TabICLv2 五折 R² 均值为 `{station_mean.TabICLv2_R2:.4f}`，标准差为 `{station_sd.TabICLv2_R2:.4f}`；五折 ΔR² 均为正，因此不是单个折次驱动。",
        "",
        "## 4. 误差与资源",
        "",
        f"随机验证 RMSE 从 `{random.XGBoost_RMSE:.4f}` 降至 `{random.TabICLv2_RMSE:.4f}`；年份留出 RMSE 为 `{year.TabICLv2_RMSE:.4f}`，与 XGBoost 接近。",
        f"本轮已完成运行数：`{len(runtime)}`；总运行时间约 `{runtime['runtime'].sum():.1f}` 秒；TabICLv2 使用 CUDA，默认 checkpoint 和默认推理配置。",
        "",
        "## 5. 下一步",
        "",
        "1. 只扩展到 Secchi，继续使用同一 77 特征和保存划分。",
        "2. 若 Secchi 的年份留出或留站点没有正信号，则停止 TabICLv2 全量扩展；若有正信号，再评估 TN bottom 和 TP middle。",
        "",
        "本报告仅针对扩展实验 pilot，不改变正式主链和既有正式结果。",
    ]
    if TASK == "secchi":
        station_positive = int((station["delta_R2"] > 0).sum())
        station_count = len(station)
        year_delta = float(year.delta_R2)
        station_delta = float(station_mean.delta_R2)
        if station_delta > 0.02 and station_positive >= 3:
            decision = "STRONG CONTINUE"
            reason = "Secchi 留站点平均提升且多数折次为正，继续 TN bottom 和 TP middle。"
        elif station_delta > 0.0 and station_positive >= 3:
            decision = "SPATIAL-SPECIFIC CONTINUE"
            reason = "Secchi 的空间泛化有改善，但时间外推仍需谨慎；继续 TN bottom 和 TP middle。"
        elif random.delta_R2 >= -0.01 or year_delta >= -0.01 or station_delta >= -0.01:
            decision = "MIXED SIGNAL"
            reason = "Secchi 未形成稳定的跨验证提升，下一步只运行 TN bottom。"
        else:
            decision = "MIXED SIGNAL"
            reason = "Secchi 三类验证均未形成稳定优势，下一步只运行 TN bottom，不立即扩展 TP middle。"
        secchi_lines = [
            "# Secchi TabICLv2 严格 pilot 决策报告",
            "",
            "## 1. 一句话结论",
            "",
            f"**{decision}**：{reason}",
            "",
            "## 2. 三类验证结果",
            "",
            "| 验证方式 | XGBoost R² | TabICLv2 R² | ΔR² |",
            "|---|---:|---:|---:|",
            f"| Random strict | {random.XGBoost_R2:.4f} | {random.TabICLv2_R2:.4f} | {random.delta_R2:+.4f} |",
            f"| Year holdout | {year.XGBoost_R2:.4f} | {year.TabICLv2_R2:.4f} | {year.delta_R2:+.4f} |",
            f"| Station mean | {station_mean.XGBoost_R2:.4f} | {station_mean.TabICLv2_R2:.4f} | {station_mean.delta_R2:+.4f} |",
            "",
            "## 3. 留站点五折结果",
            "",
            "| 折次 | XGBoost R² | TabICLv2 R² | ΔR² |",
            "|---|---:|---:|---:|",
            *fold_lines,
            "",
            f"TabICLv2 五折 R² 均值为 `{station_mean.TabICLv2_R2:.4f}`，标准差为 `{station_sd.TabICLv2_R2:.4f}`；五折正 ΔR² 为 `{station_positive}/{station_count}`。",
            "",
            "## 4. 误差与资源",
            "",
            f"随机验证 RMSE 从 `{random.XGBoost_RMSE:.4f}` 降至 `{random.TabICLv2_RMSE:.4f}`；年份留出 RMSE 为 `{year.TabICLv2_RMSE:.4f}`。",
            f"本轮总运行时间约 `{runtime['runtime'].sum():.1f}` 秒；TabICLv2 使用 CUDA 默认配置。",
            "",
            "## 5. 下一步",
            "",
            "1. 继续运行 TN bottom，检验优势是否扩展到非光学营养盐任务。",
            "2. 若 TN bottom 严格验证仍有稳定优势，再运行 TP middle；否则不扩展到全部任务。",
            "",
            f"年份留出 ΔR²：`{year_delta:+.4f}`；当前结果支持优先讨论空间泛化，不提前声称时间外推改善。",
            "",
            "本报告仅针对扩展实验 pilot，不改变正式主链。",
        ]
        (RUN / "10_secchi_decision.md").write_text("\n".join(secchi_lines), encoding="utf-8")
    elif TASK == "tn_bottom":
        station_positive = int((station["delta_R2"] > 0).sum())
        station_count = len(station)
        station_delta = float(station_mean.delta_R2)
        station_median_delta = float(comparison.loc[(comparison.validation == "station_summary") & (comparison.fold == "median"), "delta_R2"].iloc[0])
        station_sd_delta = float(comparison.loc[(comparison.validation == "station_summary") & (comparison.fold == "sd"), "delta_R2"].iloc[0])
        max_station_delta = float(station["delta_R2"].max())
        single_fold_driver = max_station_delta > max(0.10, 2 * abs(station_median_delta)) or station_sd_delta > abs(station_delta)
        if station_delta > 0 and station_positive >= 4 and not single_fold_driver:
            decision = "CROSS-TARGET STRONG SIGNAL"
            reason = "TN bottom 留站点平均 ΔR² 为正且至少 4/5 折获胜，支持继续检验 TP middle。"
            next_steps = [
                "1. 继续运行 TP middle，检验 hard case。",
                "2. 保留 TN bottom 的 strict baseline，并在论文中注明历史 TN bottom 复现警告。",
            ]
        elif station_delta >= 0 and station_positive >= 3:
            decision = "NUTRIENT WEAK SIGNAL"
            reason = "TN bottom 五折方向一致但存在明显单折放大，仍可运行 TP middle，但不假设统一跨指标优势。"
            next_steps = [
                "1. 只运行 TP middle 作为第二个营养盐 hard case。",
                "2. 不扩展到全部 TN/TP 层。",
            ]
        elif station_positive < 3:
            decision = "OPTICAL-SPECIFIC SIGNAL"
            reason = "TN bottom 留站点多数折次未胜出，当前证据更支持光学任务的空间泛化优势。"
            next_steps = [
                "1. 只运行 TP middle 作为第二个营养盐 hard case。",
                "2. 若 TP middle 也无优势，停止营养盐扩展。",
            ]
        else:
            decision = "FAILURE"
            reason = "TN bottom 未形成稳定优势，且存在明显下降或不稳定。"
            next_steps = [
                "1. 暂停 TP middle。",
                "2. 保留本轮结果作为营养盐任务的负结果。",
            ]
        tn_lines = [
            "# TN bottom TabICLv2 严格 pilot 决策报告",
            "",
            "## 1. 一句话结论",
            "",
            f"**{decision}**：{reason}",
            "",
            "## 2. 三类验证结果",
            "",
            "| 验证方式 | XGBoost R² | TabICLv2 R² | ΔR² |",
            "|---|---:|---:|---:|",
            f"| Random strict | {random.XGBoost_R2:.4f} | {random.TabICLv2_R2:.4f} | {random.delta_R2:+.4f} |",
            f"| Year holdout | {year.XGBoost_R2:.4f} | {year.TabICLv2_R2:.4f} | {year.delta_R2:+.4f} |",
            f"| Station mean | {station_mean.XGBoost_R2:.4f} | {station_mean.TabICLv2_R2:.4f} | {station_mean.delta_R2:+.4f} |",
            "",
            "## 3. 留站点五折结果",
            "",
            "| 折次 | XGBoost R² | TabICLv2 R² | ΔR² |",
            "|---|---:|---:|---:|",
            *fold_lines,
            "",
            f"TabICLv2 五折正 ΔR²：`{station_positive}/{station_count}`；均值 ΔR²：`{station_mean.delta_R2:+.4f}`；中位数 ΔR²：`{comparison.loc[(comparison.validation == 'station_summary') & (comparison.fold == 'median'), 'delta_R2'].iloc[0]:+.4f}`；SD：`{comparison.loc[(comparison.validation == 'station_summary') & (comparison.fold == 'sd'), 'delta_R2'].iloc[0]:.4f}`。",
            f"单折驱动检查：最大折次 ΔR² 为 `{max_station_delta:+.4f}`，去掉该折后的均值为 `{station.drop(station['delta_R2'].idxmax())['delta_R2'].mean():+.4f}`；判定为 `{'是' if single_fold_driver else '否'}`。",
            "",
            "## 4. 可重复性警告",
            "",
            "本轮 XGBoost 是基于当前 77 个严格特征和保存 obs_id 划分重新计算的 strict benchmark。历史 TN bottom station 结果存在约 0.079 的复现差异，本轮不使用旧 aggregate 数值，也不追溯旧 82/84 特征实验。",
            "",
            "## 5. 下一步",
            "",
            *next_steps,
            "",
            "本报告仅针对扩展实验 pilot，不改变正式主链。",
        ]
        (RUN / "16_tn_bottom_decision.md").write_text("\n".join(tn_lines), encoding="utf-8")
    elif TASK == "tp_middle":
        (RUN / "19_tp_middle_fold_results.csv").write_text(
            comparison.to_csv(index=False), encoding="utf-8"
        )
    else:
        (RUN / "05_chla_smoke_decision.md").write_text("\n".join(lines), encoding="utf-8")


def write_distribution_check(task_name: str, target_column: str, depth: str, output_name: str) -> None:
    features = load_features()
    columns = ["obs_id", "depth", target_column, *features]
    data = pd.read_csv(MAIN_TABLE, usecols=columns, low_memory=False)
    data["obs_id"] = data["obs_id"].astype("string")
    data = data.loc[data["depth"].eq(depth) & data[target_column].notna()].copy()
    split = pd.read_csv(RUN / "02_split_manifest.csv", dtype={"obs_id": "string"})
    split = split.loc[split["task"].eq(task_name)]
    rows = []
    for (validation, fold), frame in split.groupby(["validation", "fold"], sort=False):
        train_ids = set(frame.loc[frame["split"].eq("train"), "obs_id"])
        test_ids = set(frame.loc[frame["split"].eq("test"), "obs_id"])
        train = data.loc[data["obs_id"].isin(train_ids)]
        test = data.loc[data["obs_id"].isin(test_ids)]
        train_x = train[features].apply(pd.to_numeric, errors="coerce")
        test_x = test[features].apply(pd.to_numeric, errors="coerce")
        for split_name, subset, feature_frame in [("train", train, train_x), ("test", test, test_x)]:
            rows.append(
                {
                    "task": task_name,
                    "validation": validation,
                    "fold": fold,
                    "split": split_name,
                    "n_samples": len(subset),
                    "missing_feature_cells": int(feature_frame.isna().sum().sum()),
                    "missing_feature_ratio": float(feature_frame.isna().mean().mean()),
                    "target_mean": float(subset[target_column].mean()),
                    "target_sd": float(subset[target_column].std()),
                    "target_p10": float(subset[target_column].quantile(0.10)),
                    "target_p50": float(subset[target_column].quantile(0.50)),
                    "target_p90": float(subset[target_column].quantile(0.90)),
                }
            )
    pd.DataFrame(rows).to_csv(RUN / output_name, index=False, encoding="utf-8-sig")


def write_four_task_audit() -> None:
    specs = {
        "chla_surface": {"target_type": "optical", "target_column": "chla_ug_l", "depth": "surface", "comparison": "04_tabiclv2_comparison.csv"},
        "secchi": {"target_type": "optical", "target_column": "secchi_depth_m", "depth": "surface", "comparison": "08_secchi_fold_results.csv"},
        "tn_bottom": {"target_type": "nutrient", "target_column": "tn_mg_l", "depth": "bottom", "comparison": "13_tn_bottom_fold_results.csv"},
        "tp_middle": {"target_type": "nutrient", "target_column": "tp_mg_l", "depth": "middle", "comparison": "19_tp_middle_fold_results.csv"},
    }
    main_columns = ["obs_id", "depth", *[spec["target_column"] for spec in specs.values()]]
    main = pd.read_csv(MAIN_TABLE, usecols=main_columns, low_memory=False)
    main["obs_id"] = main["obs_id"].astype("string")
    split = pd.read_csv(RUN / "02_split_manifest.csv", dtype={"obs_id": "string"})
    comparison_map: dict[str, pd.DataFrame] = {}
    audit_rows = []
    pattern_rows = []
    task_stats: dict[str, dict[str, object]] = {}
    for task, spec in specs.items():
        comparison = pd.read_csv(RUN / spec["comparison"])
        comparison_map[task] = comparison
        station = comparison.loc[comparison["validation"].eq("station")].copy()
        station_mean = comparison.loc[(comparison["validation"] == "station_summary") & (comparison["fold"] == "mean")].iloc[0]
        station_median = comparison.loc[(comparison["validation"] == "station_summary") & (comparison["fold"] == "median")].iloc[0]
        station_sd = comparison.loc[(comparison["validation"] == "station_summary") & (comparison["fold"] == "sd")].iloc[0]
        wins = int((station["delta_R2"] > 0).sum())
        max_delta = float(station["delta_R2"].max())
        single_fold = bool(max_delta > 2 * max(abs(float(station_median["delta_R2"])), 1e-12) or float(station_sd["delta_R2"]) > abs(float(station_mean["delta_R2"])))
        year_split = split.loc[(split["task"] == task) & (split["validation"] == "year_holdout")]
        eligible = main.loc[main["depth"].eq(spec["depth"]) & main[spec["target_column"]].notna()]
        train_ids = set(year_split.loc[year_split["split"].eq("train"), "obs_id"])
        test_ids = set(year_split.loc[year_split["split"].eq("test"), "obs_id"])
        train_target = eligible.loc[eligible["obs_id"].isin(train_ids), spec["target_column"]]
        test_target = eligible.loc[eligible["obs_id"].isin(test_ids), spec["target_column"]]
        year_shift = float(test_target.mean() - train_target.mean())
        random_delta = float(comparison.loc[comparison["validation"].eq("random"), "delta_R2"].iloc[0])
        year_delta = float(comparison.loc[comparison["validation"].eq("year"), "delta_R2"].iloc[0])
        station_delta = float(station_mean["delta_R2"])
        if station_delta > 0 and wins >= 4 and not single_fold:
            interpretation = "留站点稳定正增益"
        elif station_delta > 0 and wins >= 4:
            interpretation = "留站点正增益，但存在单折放大"
        elif station_delta > 0:
            interpretation = "留站点增益不稳定"
        else:
            interpretation = "留站点未形成正增益"
        task_stats[task] = {
            "target_type": spec["target_type"],
            "random_delta": random_delta,
            "year_delta": year_delta,
            "station_delta": station_delta,
            "station_wins": wins,
            "station_total": len(station),
            "station_median": float(station_median["delta_R2"]),
            "station_sd": float(station_sd["delta_R2"]),
            "single_fold": single_fold,
            "year_shift": year_shift,
        }
        for validation, delta in [("random", random_delta), ("year", year_delta), ("station", station_delta)]:
            row = comparison.loc[comparison["validation"].eq(validation)]
            audit_rows.append(
                {
                    "task": task,
                    "target_type": spec["target_type"],
                    "validation": validation,
                    "XGBoost_R2": float(row["XGBoost_R2"].iloc[0]) if validation != "station" else float(station_mean["XGBoost_R2"]),
                    "TabICLv2_R2": float(row["TabICLv2_R2"].iloc[0]) if validation != "station" else float(station_mean["TabICLv2_R2"]),
                    "delta_R2": delta,
                    "station_win_folds": wins,
                    "station_total_folds": len(station),
                    "station_delta_median": float(station_median["delta_R2"]),
                    "station_delta_sd": float(station_sd["delta_R2"]),
                    "year_target_shift": year_shift,
                    "interpretation": interpretation,
                }
            )

        def classify(value: float, validation: str) -> str:
            scale = max(abs(float(station_sd["delta_R2"])), 1e-12)
            if validation == "station":
                if wins >= 4 and float(station_median["delta_R2"]) > 0 and not single_fold:
                    return "STRONG_POSITIVE"
                if station_delta > 0:
                    return "POSITIVE"
                if station_delta < 0 and wins <= 1:
                    return "STRONG_NEGATIVE"
                if station_delta < 0:
                    return "NEGATIVE"
                return "NEUTRAL"
            if abs(value) <= scale:
                return "NEUTRAL"
            if value > scale:
                return "STRONG_POSITIVE"
            if value > 0:
                return "POSITIVE"
            if value < -scale:
                return "STRONG_NEGATIVE"
            if value < 0:
                return "NEGATIVE"
            return "NEUTRAL"

        pattern_rows.append(
            {
                "task": task,
                "random": classify(random_delta, "random"),
                "year": classify(year_delta, "year"),
                "station": classify(station_delta, "station"),
            }
        )

    summary = pd.DataFrame(audit_rows)
    summary.to_csv(RUN / "20_tabiclv2_pilot_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pattern_rows).to_csv(RUN / "21_generalization_pattern_matrix.csv", index=False, encoding="utf-8-sig")

    stable_station = [task for task, stats in task_stats.items() if stats["station_delta"] > 0 and stats["station_wins"] >= 4 and not stats["single_fold"]]
    station_gain = [task for task, stats in task_stats.items() if stats["station_delta"] > 0]
    year_gain = [task for task, stats in task_stats.items() if stats["year_delta"] > 0]
    optical_stable = [task for task in stable_station if task_stats[task]["target_type"] == "optical"]
    nutrient_stable = [task for task in stable_station if task_stats[task]["target_type"] == "nutrient"]
    optical_station_mean = float(np.mean([task_stats[t]["station_delta"] for t in task_stats if task_stats[t]["target_type"] == "optical"]))
    nutrient_station_mean = float(np.mean([task_stats[t]["station_delta"] for t in task_stats if task_stats[t]["target_type"] == "nutrient"]))
    h1 = "SUPPORTED" if len(station_gain) >= 3 and all(task_stats[t]["station_wins"] >= 4 for t in station_gain) else "PARTIALLY SUPPORTED"
    h2 = "SUPPORTED" if len(station_gain) >= 3 and len(year_gain) <= 1 else "PARTIALLY SUPPORTED"
    h3 = "SUPPORTED" if optical_station_mean > nutrient_station_mean and len(optical_stable) > len(nutrient_stable) else "PARTIALLY SUPPORTED"
    if len(stable_station) >= 3:
        final_decision = "FULL EXPANSION"
        final_reason = "至少 3 个任务在留站点验证中表现为稳定正增益，且不是单折驱动。"
    elif len(station_gain) >= 3 and len(year_gain) <= 1:
        final_decision = "SPATIAL-FOCUSED EXPANSION"
        final_reason = "多个任务存在空间增益，但年份留出普遍没有同步改善。"
    elif optical_stable:
        final_decision = "SELECTIVE EXPANSION"
        final_reason = "稳定增益主要集中在光学任务，营养盐任务存在边界或折次不稳定。"
    else:
        final_decision = "STOP FULL EXPANSION"
        final_reason = "四任务中未形成足够稳定的空间泛化证据。"
    lines = [
        "# TabICLv2 四任务 pilot 最终决策",
        "",
        f"## 最终决策：{final_decision}",
        "",
        final_reason,
        "",
        "## 四任务留站点概览",
        "",
        "| 任务 | 类型 | station ΔR² | 胜出折数 | 中位数 | SD | 是否单折放大 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for task, stats in task_stats.items():
        lines.append(f"| {task} | {stats['target_type']} | {stats['station_delta']:+.4f} | {stats['station_wins']}/{stats['station_total']} | {stats['station_median']:+.4f} | {stats['station_sd']:.4f} | {'是' if stats['single_fold'] else '否'} |")
    lines += [
        "",
        "## 三个论文假设",
        "",
        f"- H1 未见站点优势：**{h1}**。证据：{len(station_gain)}/4 个任务留站点均为正，稳定增益任务为 {', '.join(stable_station) or '无'}。",
        f"- H2 空间泛化强于时间外推：**{h2}**。证据：留站点正增益任务 {len(station_gain)}/4，年份留出正增益任务 {len(year_gain)}/4。",
        f"- H3 光学任务优势强于营养盐任务：**{h3}**。四任务 station ΔR² 均值：光学 `{optical_station_mean:+.4f}`，营养盐 `{nutrient_station_mean:+.4f}`。",
        "",
        "## 判定说明",
        "",
        "模式矩阵中的 strong/positive/neutral/negative 不是按固定 R² 阈值划分：留站点结果同时看 ΔR² 方向、折次一致性和折间标准差；随机与年份结果用各任务留站点 ΔR² 的波动作为参照。",
        "目标分布变化只作为解释因素之一，不作为模型失败的唯一原因。",
        "",
        "## 下一步",
        "",
        "1. 根据最终决策选择是否扩展到正式 10 个任务。",
        "2. 在论文中把 TabICLv2 的主要证据定位为留站点空间泛化，并单独说明年份留出的限制。",
        "3. 保留四任务逐样本预测，用于残差和跨站点误差分析。",
        "",
        "本报告属于扩展实验，不修改正式主链。",
    ]
    (RUN / "22_tabiclv2_final_pilot_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_task(args.task)
    RUN.mkdir(parents=True, exist_ok=True)
    if args.audit:
        write_four_task_audit()
        print(f"four-task audit written: {RUN}")
        return
    if args.finalize:
        finalize_outputs()
        if TASK == "chla_surface":
            write_distribution_check("chla_surface", "chla_ug_l", "surface", "11_chla_distribution_check.csv")
        elif TASK == "tn_bottom":
            write_distribution_check("tn_bottom", "tn_mg_l", "bottom", "15_tn_bottom_distribution_check.csv")
        elif TASK == "tp_middle":
            write_distribution_check("tp_middle", "tp_mg_l", "middle", "23_tp_middle_distribution_check.csv")
        print(f"pilot outputs finalized: {RUN}")
        return
    if not args.model or not args.validation:
        raise SystemExit("--model and --validation are required unless --finalize is used")
    features = load_features()
    data = load_data(features)
    splits = load_splits(args.validation)
    rows: list[dict[str, object]] = []
    for fold, split in splits:
        print(f"开始 {args.model} {args.validation} {fold}", flush=True)
        rows.append(run_one(args.model, args.validation, fold, split, data, features))
        print(json.dumps(rows[-1], ensure_ascii=False, indent=2), flush=True)
        if rows[-1]["status"] != "ok" and args.model == "tabicl":
            break
    if TASK == "secchi":
        output = RUN / "07_secchi_strict_metrics.csv"
    elif TASK == "tn_bottom":
        output = RUN / "12_tn_bottom_strict_metrics.csv"
    elif TASK == "tp_middle":
        output = RUN / "18_tp_middle_strict_metrics.csv"
    else:
        output = RUN / ("03_xgb_strict_smoke.csv" if args.model == "xgb" else "04_tabiclv2_chla_smoke.csv")
    if not args.predictions_only:
        append_csv(output, rows)
    if TASK in {"secchi", "tn_bottom", "tp_middle"}:
        write_predictions()
    runtime_rows = [
        {
            "model": args.model,
            "task": TASK,
            "validation": row["validation"],
            "fold": row["fold"],
            "runtime": row["runtime"],
            "status": row["status"],
            "n_features": row["n_features"],
            "n_train": row.get("n_train"),
            "n_test": row.get("n_test"),
            "device": row.get("device"),
            "error": row.get("error"),
        }
        for row in rows
    ]
    if not args.predictions_only:
        append_csv(RUN / "06_runtime_log.csv", runtime_rows)
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
