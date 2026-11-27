"""Run the Chl-a family station-holdout extension for the TabICLv2 pilot.

This extension reuses the existing 77-feature manifest, the saved five-fold
station split manifest and the fixed model functions from the Chl-a pilot.
It writes only under runs/tabiclv2_pilot and never changes formal results.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tabiclv2_pilot"
MAIN_TABLE = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
SPLIT_MANIFEST = (
    ROOT
    / "结果"
    / "扩展实验"
    / "候选模型稳健性验证_20260914"
    / "回传结果_20260914"
    / "outputs_station_5fold"
    / "split_manifest.csv"
)
FEATURE_SET = "X_strict_77"
REPRO_TOLERANCE_STOP = 0.03

TASKS = {
    "chla_surface": {"depth": "surface", "metric": "30_chla_surface_station_metrics.csv", "prediction": "33_chla_surface_station_predictions.csv"},
    "chla_middle": {"depth": "middle", "metric": "31_chla_middle_station_metrics.csv", "prediction": "34_chla_middle_station_predictions.csv"},
    "chla_bottom": {"depth": "bottom", "metric": "32_chla_bottom_station_metrics.csv", "prediction": "35_chla_bottom_station_predictions.csv"},
}


def load_pilot_module():
    path = ROOT / "脚本" / "04_模型训练" / "33_run_tabiclv2_chla_smoke.py"
    spec = importlib.util.spec_from_file_location("tabiclv2_chla_pilot_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pilot base: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_pilot_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    return parser.parse_args()


def load_data(task: str, features: list[str]) -> pd.DataFrame:
    target = "chla_ug_l"
    depth = TASKS[task]["depth"]
    columns = ["obs_id", "station", "date", "depth", target, *features]
    data = pd.read_csv(MAIN_TABLE, usecols=columns, low_memory=False)
    data["obs_id"] = data["obs_id"].astype("string")
    data["station"] = data["station"].astype("string")
    data["date"] = pd.to_datetime(data["date"])
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.loc[data["depth"].eq(depth) & data[target].notna()].copy()
    if data["obs_id"].duplicated().any():
        raise RuntimeError(f"{task}: duplicated obs_id in main table")
    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return data


def load_station_splits(task: str) -> dict[str, pd.DataFrame]:
    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError(f"Saved station split manifest not found: {SPLIT_MANIFEST}")
    split = pd.read_csv(SPLIT_MANIFEST, dtype={"obs_id": "string"})
    subset = split.loc[
        split["evaluation_mode"].eq("station_5fold")
        & split["target"].eq("chla")
        & split["depth"].eq(TASKS[task]["depth"])
    ].copy()
    subset["obs_id"] = subset["obs_id"].astype("string")
    subset["fold"] = subset["fold"].astype(int)
    if sorted(subset["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"{task}: expected saved station folds 1-5")
    if subset.duplicated(["fold", "split", "obs_id"]).any():
        raise RuntimeError(f"{task}: duplicated obs_id in saved split manifest")
    result = {}
    for fold, frame in subset.groupby("fold", sort=True):
        frame = frame.copy()
        frame["fold_name"] = f"fold_{int(fold)}"
        result[f"fold_{int(fold)}"] = frame
    return result


def run_one(task: str, model_name: str, fold: str, split: pd.DataFrame, data: pd.DataFrame, features: list[str], target: str = "chla_ug_l") -> tuple[dict[str, object], pd.DataFrame]:
    started = time.perf_counter()
    train_ids = set(split.loc[split["split"].eq("train"), "obs_id"])
    test_ids = set(split.loc[split["split"].eq("test"), "obs_id"])
    if train_ids & test_ids:
        raise RuntimeError(f"{task}/{fold}: train and test obs_id overlap")
    train = data.loc[data["obs_id"].isin(train_ids)].copy()
    test = data.loc[data["obs_id"].isin(test_ids)].copy()
    if len(train) != len(train_ids) or len(test) != len(test_ids):
        missing = (train_ids | test_ids) - set(data["obs_id"])
        raise RuntimeError(f"{task}/{fold}: split ids missing from main table: {len(missing)}")

    x_train_raw = train[features].to_numpy(dtype=float)
    x_test_raw = test[features].to_numpy(dtype=float)
    missing_train = int(np.isnan(x_train_raw).sum())
    missing_test = int(np.isnan(x_test_raw).sum())
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(x_train_raw).astype(np.float32)
    x_test = imputer.transform(x_test_raw).astype(np.float32)
    y_train, inverse = BASE.target_transform(train[target].to_numpy(dtype=float))
    y_true = test[target].to_numpy(dtype=float)

    if model_name == "xgb":
        raw_prediction = BASE.fit_xgb(x_train, y_train, x_test)
        device = "xgboost_cpu"
    else:
        raw_prediction = BASE.fit_tabicl(x_train, y_train, x_test)
        device = "cuda"
    prediction = np.maximum(inverse(raw_prediction), 0)
    scores = BASE.metrics(y_true, prediction)
    split_source = "候选模型稳健性验证保存 station_5fold obs_id split_manifest"
    records = pd.DataFrame(
        {
            "obs_id": test["obs_id"].astype(str).to_numpy(),
            "station_id": test["station"].astype(str).to_numpy(),
            "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "depth": TASKS[task]["depth"],
            "target": task,
            "validation": "station",
            "fold": fold,
            "y_true": y_true,
            "y_pred": prediction,
            "residual": prediction - y_true,
            "model": model_name,
            "feature_set": FEATURE_SET,
            "split_source": split_source,
        }
    )
    result = {
        "task": task,
        "validation": "station",
        "fold": fold,
        "XGBoost_R2": scores["R2"] if model_name == "xgb" else np.nan,
        "TabICLv2_R2": scores["R2"] if model_name == "tabicl" else np.nan,
        "XGBoost_RMSE": scores["RMSE"] if model_name == "xgb" else np.nan,
        "TabICLv2_RMSE": scores["RMSE"] if model_name == "tabicl" else np.nan,
        "XGBoost_MAE": scores["MAE"] if model_name == "xgb" else np.nan,
        "TabICLv2_MAE": scores["MAE"] if model_name == "tabicl" else np.nan,
        "R2": scores["R2"],
        "RMSE": scores["RMSE"],
        "MAE": scores["MAE"],
        "model": model_name,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(features),
        "missing_train_cells": missing_train,
        "missing_test_cells": missing_test,
        "feature_set": FEATURE_SET,
        "split_source": split_source,
        "device": device,
        "seed": int(BASE.SEED),
        "runtime": time.perf_counter() - started,
        "status": "ok",
    }
    return result, records


def combine_metrics(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    result = []
    for fold, group in frame.groupby("fold", sort=True):
        row = {"task": group["task"].iloc[0], "validation": "station", "fold": fold}
        for column in ["R2", "RMSE", "MAE"]:
            xgb = group.loc[group["model"].eq("xgb"), column].iloc[0]
            tabicl = group.loc[group["model"].eq("tabicl"), column].iloc[0]
            row[f"XGBoost_{column}"] = xgb
            row[f"TabICLv2_{column}"] = tabicl
            row[f"delta_{column}"] = tabicl - xgb
        for column in ["n_train", "n_test", "n_features", "missing_train_cells", "missing_test_cells", "feature_set", "split_source", "seed"]:
            row[column] = group[column].iloc[0]
        row["status"] = "ok"
        result.append(row)
    return pd.DataFrame(result)


def check_surface_reproducibility(new_metrics: pd.DataFrame) -> None:
    old_path = RUN / "04_tabiclv2_comparison.csv"
    old = pd.read_csv(old_path)
    old = old.loc[old["validation"].eq("station") & old["fold"].astype(str).str.startswith("fold_")].copy()
    current = new_metrics.loc[new_metrics["fold"].astype(str).str.startswith("fold_")].copy()
    merged = old.merge(current, on=["task", "validation", "fold"], suffixes=("_old", "_new"), validate="one_to_one")
    rows = []
    for metric in ["XGBoost_R2", "TabICLv2_R2"]:
        difference = (merged[f"{metric}_new"] - merged[f"{metric}_old"]).abs()
        rows.append(
            {
                "metric": metric,
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
                "status": "STOP" if difference.max() > REPRO_TOLERANCE_STOP else ("ACCEPTABLE" if difference.max() <= 0.01 else "CAUTION"),
            }
        )
    check = pd.DataFrame(rows)
    check.to_csv(RUN / "chla_surface_repro_check.csv", index=False, encoding="utf-8-sig")
    if (check["status"] == "STOP").any():
        raise RuntimeError("Chl-a surface reproduction difference exceeded 0.03; Phase A stopped before middle/bottom")


def run_task(task: str, features: list[str]) -> pd.DataFrame:
    data = load_data(task, features)
    splits = load_station_splits(task)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold, split in splits.items():
        for model_name in ["xgb", "tabicl"]:
            print(f"开始 {task} {model_name} station {fold}", flush=True)
            row, prediction = run_one(task, model_name, fold, split, data, features)
            metric_rows.append(row)
            prediction_frames.append(prediction)
            print(
                f"完成 {task} {model_name} {fold}: R2={row['XGBoost_R2'] if model_name == 'xgb' else row['TabICLv2_R2']:.4f}, runtime={row['runtime']:.1f}s",
                flush=True,
            )
    metrics = combine_metrics(metric_rows)
    metrics.to_csv(RUN / TASKS[task]["metric"], index=False, encoding="utf-8-sig")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions.duplicated(["obs_id", "model"]).any():
        raise RuntimeError(f"{task}: duplicate sample-level predictions")
    predictions.to_csv(RUN / TASKS[task]["prediction"], index=False, encoding="utf-8-sig")
    if task == "chla_surface":
        check_surface_reproducibility(metrics)
    return metrics


def main() -> None:
    args = parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    features = BASE.load_features()
    if len(features) != 77:
        raise RuntimeError(f"Expected 77 strict features, found {len(features)}")
    all_metrics = []
    for task in args.tasks:
        metrics = run_task(task, features)
        all_metrics.append(metrics)
    pd.concat(all_metrics, ignore_index=True).to_csv(RUN / "chla_phase_a_run_log.csv", index=False, encoding="utf-8-sig")
    print("Chl-a Phase A station family complete", flush=True)


if __name__ == "__main__":
    main()
