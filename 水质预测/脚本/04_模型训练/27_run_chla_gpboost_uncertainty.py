"""Chl-a GPBoost point prediction with quantile and split-conformal uncertainty.

This is an extension experiment. It uses only f1_history features, a sample_id
outer split shared across water layers, and never uses contemporaneous water
quality as an input. GPBoost is retained only because its prior Chl-a signal was
positive; it is not a four-target replacement model.
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
DEFAULT_REPORT_DIR = ROOT / "\u7ed3\u679c" / "\u6269\u5c55\u5b9e\u9a8c" / "\u53f6\u7eff\u7d20\u4e0d\u786e\u5b9a\u6027\u9884\u6d4b_20260905"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Chl-a GPBoost uncertainty extension experiment.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
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


def sample_split(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = np.sort(data["sample_id"].astype(str).unique())
    train_ids, test_ids = train_test_split(sample_ids, test_size=test_size, random_state=random_state, shuffle=True)
    train = data[data["sample_id"].astype(str).isin(set(train_ids))].copy()
    test = data[data["sample_id"].astype(str).isin(set(test_ids))].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def encode_station_groups(train: pd.DataFrame, prediction_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = pd.concat([train["station"], prediction_data["station"]], ignore_index=True).astype(str)
    codes, _ = pd.factorize(values)
    return codes[: len(train)], codes[len(train) :]


def gpboost_fit_predict(train: pd.DataFrame, prediction_data: pd.DataFrame, target_col: str, features: list[str], args: argparse.Namespace) -> np.ndarray:
    try:
        import gpboost as gpb
    except ImportError as exc:
        raise SystemExit("Missing gpboost. Install it in the current extension environment.") from exc
    y_train, inverse = transform_target(train[target_col])
    train_groups, prediction_groups = encode_station_groups(train, prediction_data)
    train_set = gpb.Dataset(train[features], label=y_train, free_raw_data=False)
    gp_model = gpb.GPModel(group_data=train_groups)
    params = {
        "objective": "regression_l2",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": 15,
        "feature_fraction": args.colsample_bytree,
        "bagging_freq": 0,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 20,
        "seed": args.random_state,
        "num_threads": args.n_jobs,
        "verbose": -1,
    }
    model = gpb.train(params=params, train_set=train_set, num_boost_round=args.n_estimators, gp_model=gp_model)
    prediction = model.predict(data=prediction_data[features], group_data_pred=prediction_groups, predict_var=True)
    return np.maximum(inverse(np.asarray(prediction["response_mean"])), 0)


def quantile_fit_predict(train: pd.DataFrame, prediction_data: pd.DataFrame, target_col: str, features: list[str], alpha: float, args: argparse.Namespace) -> np.ndarray:
    y_train, inverse = transform_target(train[target_col])
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    model.fit(train[features], y_train)
    return np.maximum(inverse(model.predict(prediction_data[features])), 0)


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return np.nan
    rank = int(np.ceil((len(values) + 1) * (1 - alpha))) - 1
    return float(values[min(max(rank, 0), len(values) - 1)])


def metrics(y_true: np.ndarray, prediction: np.ndarray, lower: np.ndarray | None = None, upper: np.ndarray | None = None, high_threshold: float | None = None) -> dict[str, float]:
    output = {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "bias": float(np.mean(prediction - y_true)),
    }
    if lower is not None and upper is not None:
        output["interval_coverage"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
        output["interval_mean_width"] = float(np.mean(upper - lower))
    else:
        output["interval_coverage"] = np.nan
        output["interval_mean_width"] = np.nan
    if high_threshold is not None:
        high = y_true >= high_threshold
        output["high_value_rows"] = int(high.sum())
        output["high_value_underprediction_rate"] = float(np.mean(prediction[high] < y_true[high])) if high.any() else np.nan
        output["high_value_mae"] = float(mean_absolute_error(y_true[high], prediction[high])) if high.any() else np.nan
    return output


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    formal = load_formal_module()
    data = formal.prepare_dataset(args.dataset)
    target_col = formal.TARGET_MAP["chla"]
    features = formal.build_feature_sets(data, target_col)["f1_history"] + ["depth_order"]
    if target_col in features:
        raise ValueError("The Chl-a target cannot be used as a predictor.")
    full = data[data[target_col].notna()].copy().sort_values(["date", "station", "depth_order"]).reset_index(drop=True)
    outer_train, outer_test = sample_split(full, args.test_size, args.random_state)
    core_train, calibration = sample_split(outer_train, args.calibration_fraction, args.random_state + 1)

    gpboost_calibration = gpboost_fit_predict(core_train, calibration, target_col, features, args)
    conformal_radii = {}
    high_thresholds = {}
    for depth in ("surface", "middle", "bottom"):
        cal_mask = calibration["depth"].eq(depth).to_numpy()
        conformal_radii[depth] = conformal_quantile(np.abs(calibration.loc[cal_mask, target_col].to_numpy(dtype=float) - gpboost_calibration[cal_mask]), alpha=0.2)
        high_thresholds[depth] = float(outer_train.loc[outer_train["depth"].eq(depth), target_col].quantile(0.9))

    gpboost_prediction = gpboost_fit_predict(outer_train, outer_test, target_col, features, args)
    q10 = quantile_fit_predict(outer_train, outer_test, target_col, features, 0.1, args)
    q50 = quantile_fit_predict(outer_train, outer_test, target_col, features, 0.5, args)
    q90 = quantile_fit_predict(outer_train, outer_test, target_col, features, 0.9, args)
    q_lower, q_upper = np.minimum(q10, q90), np.maximum(q10, q90)
    conformal_radius = outer_test["depth"].map(conformal_radii).to_numpy(dtype=float)
    c_lower = np.maximum(gpboost_prediction - conformal_radius, 0)
    c_upper = gpboost_prediction + conformal_radius

    prediction = outer_test[["obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "year", "month", "depth"]].copy()
    prediction["observed"] = outer_test[target_col].to_numpy(dtype=float)
    prediction["gpboost_prediction"] = gpboost_prediction
    prediction["quantile_q10"], prediction["quantile_q50"], prediction["quantile_q90"] = q10, q50, q90
    prediction["quantile_lower"], prediction["quantile_upper"] = q_lower, q_upper
    prediction["conformal_lower"], prediction["conformal_upper"] = c_lower, c_upper
    prediction["conformal_radius"] = conformal_radius
    prediction["high_value_threshold_train_p90"] = prediction["depth"].map(high_thresholds)

    metric_rows: list[dict[str, object]] = []
    for depth in ("surface", "middle", "bottom"):
        mask = prediction["depth"].eq(depth).to_numpy()
        y = prediction.loc[mask, "observed"].to_numpy(dtype=float)
        threshold = high_thresholds[depth]
        for model_name, point, lower, upper in (
            ("gpboost_point", gpboost_prediction[mask], None, None),
            ("xgboost_quantile_median", q50[mask], q_lower[mask], q_upper[mask]),
            ("gpboost_split_conformal", gpboost_prediction[mask], c_lower[mask], c_upper[mask]),
        ):
            metric_rows.append({
                "target": "chla", "depth": depth, "model_variant": model_name, "rows": int(mask.sum()),
                "outer_split": "sample_id_random_80_20_shared_across_depths",
                "inner_calibration": "sample_id_random_20pct_of_outer_train",
                "same_day_water_quality_covariates": "excluded", "high_value_threshold_train_p90": threshold,
                **metrics(y, point, lower, upper, threshold),
            })

    prediction.to_csv(args.report_dir / "叶绿素不确定性预测.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(args.report_dir / "叶绿素不确定性指标.csv", index=False, encoding="utf-8-sig")
    outer_train_ids = set(outer_train["sample_id"].astype(str).unique())
    calibration_ids = set(calibration["sample_id"].astype(str).unique())
    split_manifest = pd.DataFrame({"sample_id": sorted(full["sample_id"].astype(str).unique())})
    split_manifest["outer_split"] = np.where(split_manifest["sample_id"].isin(outer_train_ids), "train", "test")
    split_manifest["inner_role"] = np.where(
        split_manifest["outer_split"].eq("test"),
        "not_applicable",
        np.where(split_manifest["sample_id"].isin(calibration_ids), "calibration", "core_train"),
    )
    split_manifest.to_csv(args.report_dir / "样本划分清单.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"depth": list(conformal_radii), "conformal_radius": list(conformal_radii.values()), "high_value_threshold_train_p90": [high_thresholds[d] for d in conformal_radii]}).to_csv(args.report_dir / "共形校准参数.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features}).to_csv(args.report_dir / "特征清单.csv", index=False, encoding="utf-8-sig")
    (args.report_dir / "实验说明.md").write_text(
        "# Chl-a 不确定性预测\n\n本目录为扩展实验，不覆盖正式主链。GPBoost 用于点预测；XGBoost 0.1/0.5/0.9 分位数用于条件区间；GPBoost split conformal 区间使用外层训练集内独立校准样本。高值阈值为各水层外层训练集 P90。\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
