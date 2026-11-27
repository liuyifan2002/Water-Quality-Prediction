"""Compare global and water-zone-specific f1_history XGBoost models.

The experiment keeps the formal target definitions and XGBoost parameters. For
each target-depth task, it creates one shared random 80/20 split. A zone-specific
model is fitted only when its training subset has enough observations and
stations; small zones fall back to the global model trained on the same outer
training data. No contemporaneous water-quality feature is allowed.
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
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
FORMAL_TRAINING_SCRIPT = ROOT / "\u811a\u672c" / "04_\u6a21\u578b\u8bad\u7ec3" / "05_train_full_station_models.py"
DEFAULT_DATASET = ROOT / "\u6570\u636e" / "\u5904\u7406\u540e\u6570\u636e" / "\u6b63\u5f0f\u4e3b\u8868" / "model_table_3depth_2014_2024.csv"
DEFAULT_REPORT_DIR = ROOT / "\u7ed3\u679c" / "\u6269\u5c55\u5b9e\u9a8c" / "\u5206\u6d77\u533a\u5efa\u6a21_20260906"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run water-zone-specific versus global full-station A/B.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--targets", nargs="*", choices=["chla", "secchi", "tn", "tp"], default=["chla", "secchi", "tn", "tp"])
    parser.add_argument("--depths", nargs="*", choices=["surface", "middle", "bottom"], default=["surface", "middle", "bottom"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-zone-train-rows", type=int, default=300)
    parser.add_argument("--min-zone-stations", type=int, default=3)
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


def transform_target(values: pd.Series) -> tuple[np.ndarray, callable]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.nanmin(numeric) >= 0:
        return np.log1p(numeric), np.expm1
    return numeric, lambda array: array


def fit_predict(train: pd.DataFrame, prediction_data: pd.DataFrame, target_col: str, features: list[str], args: argparse.Namespace) -> np.ndarray:
    y_train, inverse = transform_target(train[target_col])
    model = XGBRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror", random_state=args.random_state, n_jobs=args.n_jobs,
    )
    model.fit(train[features], y_train)
    return np.maximum(inverse(model.predict(prediction_data[features])), 0)


def evaluate(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "bias": float(np.mean(prediction - y_true)),
    }


def assert_feature_safety(features: list[str], target_col: str, target_map: dict[str, str]) -> None:
    forbidden = set(target_map.values()) | {"turbidity", "turbidity_ntu", "salinity", "do", "ph"}
    overlap = sorted(set(features) & forbidden)
    if overlap or target_col in features:
        raise ValueError(f"Forbidden same-day water-quality input: {overlap or [target_col]}")


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    formal = load_formal_module()
    data = formal.prepare_dataset(args.dataset)
    data["water_zone_en"] = data["water_zone_en"].fillna("unknown").astype(str)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    zone_decisions: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []

    for target_key in args.targets:
        target_col = formal.TARGET_MAP[target_key]
        features = formal.build_feature_sets(data, target_col)["f1_history"]
        assert_feature_safety(features, target_col, formal.TARGET_MAP)
        for depth in args.depths:
            subset = data[(data["depth"] == depth) & data[target_col].notna()].copy().sort_values(["date", "station"]).reset_index(drop=True)
            train, test = train_test_split(subset, test_size=args.test_size, random_state=args.random_state, shuffle=True)
            train, test = train.reset_index(drop=True), test.reset_index(drop=True)
            global_prediction = fit_predict(train, test, target_col, features, args)
            zone_prediction = global_prediction.copy()
            source = np.full(len(test), "global_fallback", dtype=object)

            for zone, zone_test in test.groupby("water_zone_en", sort=True):
                zone_train = train[train["water_zone_en"].eq(zone)].copy()
                eligible = len(zone_train) >= args.min_zone_train_rows and zone_train["station"].nunique() >= args.min_zone_stations
                zone_decisions.append({
                    "target": target_key, "depth": depth, "water_zone_en": zone,
                    "train_rows": len(zone_train), "test_rows": len(zone_test), "stations": int(zone_train["station"].nunique()),
                    "decision": "zone_model" if eligible else "global_fallback",
                    "min_zone_train_rows": args.min_zone_train_rows, "min_zone_stations": args.min_zone_stations,
                })
                if eligible:
                    positions = zone_test.index.to_numpy(dtype=int)
                    zone_prediction[positions] = fit_predict(zone_train, zone_test, target_col, features, args)
                    source[positions] = "zone_model"

            output = test[["obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "year", "month", "depth"]].copy()
            output["target"] = target_key
            output["observed"] = test[target_col].to_numpy(dtype=float)
            output["prediction_global_f1"] = global_prediction
            output["prediction_zone_aware_f1"] = zone_prediction
            output["prediction_source"] = source
            prediction_frames.append(output)
            manifests.append({
                "target": target_key, "depth": depth, "features_json": json.dumps(features, ensure_ascii=False),
                "same_day_water_quality_covariates": "excluded", "outer_split": "random_80_20", "split_random_state": args.random_state,
            })

            for model_name, prediction in (("global_f1_history", global_prediction), ("zone_aware_f1_history", zone_prediction)):
                metric_rows.append({
                    "target": target_key, "depth": depth, "model_variant": model_name, "water_zone_en": "all",
                    "rows": len(output), "feature_count": len(features), "same_day_water_quality_covariates": "excluded",
                    **evaluate(output["observed"].to_numpy(dtype=float), prediction),
                })
                for zone, group in output.groupby("water_zone_en", sort=True):
                    if len(group) < 20:
                        continue
                    positions = group.index.to_numpy(dtype=int)
                    metric_rows.append({
                        "target": target_key, "depth": depth, "model_variant": model_name, "water_zone_en": zone,
                        "rows": len(group), "feature_count": len(features), "same_day_water_quality_covariates": "excluded",
                        **evaluate(group["observed"].to_numpy(dtype=float), prediction[positions]),
                    })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.report_dir / "分海区建模_指标.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(zone_decisions).to_csv(args.report_dir / "分区训练决策.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifests).to_csv(args.report_dir / "特征与划分清单.csv", index=False, encoding="utf-8-sig")
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(args.report_dir / "分海区建模_预测.csv", index=False, encoding="utf-8-sig")
    (args.report_dir / "实验说明.md").write_text(
        "# 分海区建模 A/B\n\n本目录为扩展实验，不覆盖正式主结果。水区采用原始 `water_zone_en`。只有训练样本不少于 300 且不少于 3 个站点的水区才训练本地模型；其余样本回退至同一外层训练集拟合的全域模型。\n",
        encoding="utf-8",
    )
    summary = metrics[metrics["water_zone_en"].eq("all")]
    print(summary[["target", "depth", "model_variant", "rows", "r2", "rmse", "mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
