"""Strict time and held-station validation for the zone-aware model.

The validation deliberately excludes target-derived lag, rolling, and difference
features. It therefore tests whether zone-aware learning improves medium/longer
term prediction from information available without future water-quality labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
FORMAL_TRAINING_SCRIPT = ROOT / "\u811a\u672c" / "04_\u6a21\u578b\u8bad\u7ec3" / "05_train_full_station_models.py"
DEFAULT_DATASET = ROOT / "\u6570\u636e" / "\u5904\u7406\u540e\u6570\u636e" / "\u6b63\u5f0f\u4e3b\u8868" / "model_table_3depth_2014_2024.csv"
DEFAULT_REPORT_DIR = ROOT / "\u7ed3\u679c" / "\u6269\u5c55\u5b9e\u9a8c" / "\u5206\u6d77\u533a\u4e25\u683c\u9a8c\u8bc1_20260906"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate global versus zone-aware models by time and held station.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--targets", nargs="*", choices=["chla", "secchi", "tn", "tp"], default=["chla", "secchi", "tn", "tp"])
    parser.add_argument("--depths", nargs="*", choices=["surface", "middle", "bottom"], default=["surface", "middle", "bottom"])
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--test-start-year", type=int, default=2023)
    parser.add_argument("--test-end-year", type=int, default=2024)
    parser.add_argument("--station-test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-zone-train-rows", type=int, default=300)
    parser.add_argument("--min-zone-stations", type=int, default=3)
    parser.add_argument("--min-test-rows", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def load_formal_module():
    spec = importlib.util.spec_from_file_location("formal_full_station_training", FORMAL_TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load formal training script: {FORMAL_TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def strict_external_features(formal, data: pd.DataFrame, target_col: str) -> list[str]:
    features = formal.build_feature_sets(data, target_col)["f1_history"]
    target_history_prefix = f"{target_col}_"
    selected = [feature for feature in features if not feature.startswith(target_history_prefix)]
    if target_col in selected:
        raise ValueError("Target cannot be used as an input feature.")
    return selected


def fit_predict(train: pd.DataFrame, prediction_data: pd.DataFrame, target_col: str, features: list[str], args: argparse.Namespace) -> np.ndarray:
    y = train[target_col].to_numpy(dtype=float)
    if np.nanmin(y) >= 0:
        y_fit, inverse = np.log1p(y), np.expm1
    else:
        y_fit, inverse = y, lambda values: values
    model = XGBRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror", random_state=args.random_state, n_jobs=args.n_jobs,
    )
    model.fit(train[features], y_fit)
    return np.maximum(inverse(model.predict(prediction_data[features])), 0)


def evaluate(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "bias": float(np.mean(prediction - y_true)),
    }


def held_stations(data: pd.DataFrame, fraction: float, random_state: int) -> pd.DataFrame:
    station_zone = data[["station", "water_zone_en"]].drop_duplicates().sort_values(["water_zone_en", "station"])
    rng = np.random.default_rng(random_state)
    test_stations: list[str] = []
    for _, group in station_zone.groupby("water_zone_en", sort=True):
        stations = group["station"].to_numpy(dtype=str)
        count = max(1, int(np.ceil(len(stations) * fraction)))
        count = min(count, len(stations))
        test_stations.extend(rng.choice(stations, size=count, replace=False).tolist())
    manifest = station_zone.copy()
    manifest["spatial_split"] = np.where(manifest["station"].isin(set(test_stations)), "test_station", "train_station")
    return manifest


def run_one_task(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_key: str,
    target_col: str,
    depth: str,
    features: list[str],
    mode: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    global_prediction = fit_predict(train, test, target_col, features, args)
    zone_prediction = global_prediction.copy()
    source = np.full(len(test), "global_fallback", dtype=object)
    decisions: list[dict[str, object]] = []
    for zone, zone_test in test.groupby("water_zone_en", sort=True):
        zone_train = train[train["water_zone_en"].eq(zone)].copy()
        eligible = len(zone_train) >= args.min_zone_train_rows and zone_train["station"].nunique() >= args.min_zone_stations
        decisions.append({
            "evaluation_mode": mode, "target": target_key, "depth": depth, "water_zone_en": zone,
            "train_rows": len(zone_train), "test_rows": len(zone_test), "train_stations": int(zone_train["station"].nunique()),
            "test_stations": int(zone_test["station"].nunique()), "decision": "zone_model" if eligible else "global_fallback",
        })
        if eligible:
            positions = zone_test.index.to_numpy(dtype=int)
            zone_prediction[positions] = fit_predict(zone_train, zone_test, target_col, features, args)
            source[positions] = "zone_model"

    output = test[["obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "year", "month", "depth"]].copy()
    output["evaluation_mode"] = mode
    output["target"] = target_key
    output["observed"] = test[target_col].to_numpy(dtype=float)
    output["prediction_global_external"] = global_prediction
    output["prediction_zone_aware_external"] = zone_prediction
    output["prediction_source"] = source

    metrics: list[dict[str, object]] = []
    for model_name, prediction in (("global_external", global_prediction), ("zone_aware_external", zone_prediction)):
        metrics.append({
            "evaluation_mode": mode, "target": target_key, "depth": depth, "model_variant": model_name,
            "water_zone_en": "all", "rows": len(output), "feature_count": len(features),
            "target_history_features": "excluded", "same_day_water_quality_covariates": "excluded",
            **evaluate(output["observed"].to_numpy(dtype=float), prediction),
        })
        for zone, group in output.groupby("water_zone_en", sort=True):
            if len(group) < args.min_test_rows:
                continue
            positions = group.index.to_numpy(dtype=int)
            metrics.append({
                "evaluation_mode": mode, "target": target_key, "depth": depth, "model_variant": model_name,
                "water_zone_en": zone, "rows": len(group), "feature_count": len(features),
                "target_history_features": "excluded", "same_day_water_quality_covariates": "excluded",
                **evaluate(group["observed"].to_numpy(dtype=float), prediction[positions]),
            })
    return metrics, output, decisions


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    formal = load_formal_module()
    data = formal.prepare_dataset(args.dataset)
    data["water_zone_en"] = data["water_zone_en"].fillna("unknown").astype(str)
    station_manifest = held_stations(data, args.station_test_fraction, args.random_state)
    held_station_set = set(station_manifest.loc[station_manifest["spatial_split"].eq("test_station"), "station"])

    metric_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    decisions: list[dict[str, object]] = []
    feature_manifest: list[dict[str, object]] = []
    for target_key in args.targets:
        target_col = formal.TARGET_MAP[target_key]
        features = strict_external_features(formal, data, target_col)
        feature_manifest.append({"target": target_key, "features_json": json.dumps(features, ensure_ascii=False), "target_history_features": "excluded"})
        for depth in args.depths:
            subset = data[(data["depth"].eq(depth)) & data[target_col].notna()].copy().sort_values(["date", "station"]).reset_index(drop=True)
            split_specs = {
                "time_holdout_2023_2024": (
                    subset[subset["year"] <= args.train_end_year].copy(),
                    subset[subset["year"].between(args.test_start_year, args.test_end_year)].copy(),
                ),
                "held_station": (
                    subset[~subset["station"].isin(held_station_set)].copy(),
                    subset[subset["station"].isin(held_station_set)].copy(),
                ),
            }
            for mode, (train, test) in split_specs.items():
                train, test = train.reset_index(drop=True), test.reset_index(drop=True)
                if len(train) < 200 or len(test) < args.min_test_rows:
                    continue
                rows, output, task_decisions = run_one_task(train, test, target_key, target_col, depth, features, mode, args)
                metric_rows.extend(rows)
                predictions.append(output)
                decisions.extend(task_decisions)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.report_dir / "分海区严格验证_指标.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(decisions).to_csv(args.report_dir / "分区训练决策.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_manifest).to_csv(args.report_dir / "严格验证特征清单.csv", index=False, encoding="utf-8-sig")
    station_manifest.to_csv(args.report_dir / "留站点划分清单.csv", index=False, encoding="utf-8-sig")
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(args.report_dir / "分海区严格验证_预测.csv", index=False, encoding="utf-8-sig")
    (args.report_dir / "实验说明.md").write_text(
        "# 分海区模型严格验证\n\n本实验使用目标自身历史值以外的特征。时间留出为 2014-2022 年训练、2023-2024 年测试；留站点验证按海区抽取约 20% 站点，所有被留出站点不进入训练。两种验证均比较全域模型与分海区模型。\n",
        encoding="utf-8",
    )
    summary = metrics[metrics["water_zone_en"].eq("all")]
    print(summary[["evaluation_mode", "target", "depth", "model_variant", "rows", "r2", "rmse", "mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
