from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from itertools import product


DEFAULT_DATASET = Path("数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv")
DEFAULT_REPORT_DIR = Path("结果/正式结果/full_station")

TARGET_MAP = {
    "chla": "chla_ug_l",
    "secchi": "secchi_depth_m",
    "tn": "tn_mg_l",
    "tp": "tp_mg_l",
}

DEPTH_CHOICES = ["surface", "middle", "bottom"]
FEATURE_SET_CHOICES = ["baseline", "f1_history", "f2_remote", "f3_history_remote"]

BASELINE_FEATURES = [
    "lon",
    "lat",
    "year_offset",
    "month_sin",
    "month_cos",
    "is_wet_season",
    "quarter",
    "day_of_year",
    "water_zone_code",
]

HISTORY_BASE_FEATURES = ["days_since_prev_sample"]
HISTORY_SUFFIXES = ["_lag1", "_lag2", "_roll3", "_delta1"]

PROCESS_BASE_FEATURES = [
    "t2m",
    "u10",
    "v10",
    "tcc",
    "wind_speed",
    "wind_speed_sq",
    "wind_dir_sin",
    "wind_dir_cos",
    "rain_3d",
    "rain_14d",
    "rain_month",
    "tidal_range",
    "ntl_avg_rad",
    "viirs_count",
    "t2m_x_rain14d",
    "wind_speed_x_rain14d",
    "rain14d_x_tidal_range",
    "t2m_x_tidal_range",
    "ssrd_sum_3d",
    "ssrd_mean_3d",
    "ssrd_sum_7d",
    "ssrd_mean_7d",
    "ssrd_sum_14d",
    "ssrd_mean_14d",
    "sst_mean_3d",
    "sst_max_3d",
    "sst_anom_3d",
    "sst_mean_7d",
    "sst_max_7d",
    "sst_anom_7d",
    "sst_mean_14d",
    "sst_max_14d",
    "sst_anom_14d",
    "sst_mean_7d_x_ssrd_mean_7d",
    "sst_anom_7d_x_ssrd_mean_7d",
    "rain14d_x_ssrd_mean_7d",
]

PROCESS_LAG_BASES = [
    "t2m",
    "tcc",
    "wind_speed",
    "rain_3d",
    "rain_14d",
    "rain_month",
    "tidal_range",
    "ntl_avg_rad",
]

REMOTE_FEATURES = [
    "landsat_hq_blue",
    "landsat_hq_green",
    "landsat_hq_red",
    "landsat_hq_nir",
    "landsat_hq_swir1",
    "landsat_hq_swir2",
    "landsat_hq_ndvi",
    "landsat_hq_ndwi",
    "landsat_hq_mndwi",
    "landsat_hq_ndti",
    "landsat_image_count",
    "landsat_clear_pixel_count",
    "landsat_strict_valid_gee",
    "landsat_hq",
    "landsat_available",
    "s2_hq_b2_mean",
    "s2_hq_b3_mean",
    "s2_hq_b4_mean",
    "s2_hq_b5_mean",
    "s2_hq_b6_mean",
    "s2_hq_b8_mean",
    "s2_hq_b8a_mean",
    "s2_hq_b11_mean",
    "s2_hq_b12_mean",
    "s2_hq_ndwi_mean",
    "s2_hq_mndwi_mean",
    "s2_hq_ndci_mean",
    "s2_hq_ndti_mean",
    "s2_has_images",
    "s2_image_count",
    "s2_strict_valid",
    "s2_valid_pixel_count",
    "s2_hq",
    "s2_available",
    "hq_satellite_available_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the full-station, three-depth XGBoost models with interpretable target history features."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--targets", nargs="*", choices=sorted(TARGET_MAP), default=sorted(TARGET_MAP))
    parser.add_argument("--depths", nargs="*", choices=DEPTH_CHOICES, default=DEPTH_CHOICES)
    parser.add_argument("--feature-sets", nargs="*", choices=FEATURE_SET_CHOICES, default=FEATURE_SET_CHOICES)
    parser.add_argument(
        "--split-mode",
        choices=["random", "time"],
        default="random",
        help="正式主链默认采用固定随机比例留出；time 仅用于复现历史时间外推口径。",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="随机留出测试集比例。")
    parser.add_argument("--train-end-year", type=int, default=2022, help="仅 split-mode=time 时使用。")
    parser.add_argument("--test-start-year", type=int, default=2023, help="仅 split-mode=time 时使用。")
    parser.add_argument("--test-end-year", type=int, default=2024, help="仅 split-mode=time 时使用。")
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--min-test-rows", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--use-log1p-target", action="store_true", default=True)
    parser.add_argument("--tune", action="store_true", help="Tune XGBoost hyperparameters using a time-ordered validation split inside the training period.")
    parser.add_argument("--val-start-year", type=int, default=2022, help="First year of the validation split used for tuning.")
    parser.add_argument("--tune-n-estimators", nargs="*", type=int, default=[300, 500])
    parser.add_argument("--tune-max-depth", nargs="*", type=int, default=[3, 4, 5])
    parser.add_argument("--tune-learning-rate", nargs="*", type=float, default=[0.03, 0.05, 0.08])
    parser.add_argument("--tune-subsample", nargs="*", type=float, default=[0.8, 1.0])
    parser.add_argument("--tune-colsample-bytree", nargs="*", type=float, default=[0.8, 1.0])
    return parser.parse_args()


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.replace("\ufeff", "") for c in df.columns})


def normalize_bool_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        lowered = out[col].dropna().astype(str).str.strip().str.lower()
        if lowered.empty:
            continue
        if set(lowered.unique()).issubset({"true", "false", "0", "1"}):
            out[col] = (
                out[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
            )
    return out


def prepare_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run 脚本/02_特征构建/04_build_model_table.py first.")

    data = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    data = normalize_columns(data)
    data = normalize_bool_like_columns(data)

    data["water_zone_en"] = data["water_zone_en"].fillna("unknown")
    data["water_zone_code"] = pd.Categorical(data["water_zone_en"]).codes.astype("int16")
    data["quarter"] = pd.to_numeric(data["quarter"], errors="coerce")
    data["is_wet_season"] = pd.to_numeric(data["is_wet_season"], errors="coerce")

    numeric_candidates = set(
        BASELINE_FEATURES + HISTORY_BASE_FEATURES + PROCESS_BASE_FEATURES + PROCESS_LAG_BASES + REMOTE_FEATURES + list(TARGET_MAP.values())
    )
    for target_col in TARGET_MAP.values():
        for suffix in HISTORY_SUFFIXES:
            numeric_candidates.add(f"{target_col}{suffix}")
    for base_col in PROCESS_LAG_BASES:
        for suffix in HISTORY_SUFFIXES:
            numeric_candidates.add(f"{base_col}{suffix}")
    for col in numeric_candidates:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def available_feature_list(data: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in data.columns]


def build_feature_sets(data: pd.DataFrame, target_col: str) -> dict[str, list[str]]:
    baseline = available_feature_list(data, BASELINE_FEATURES)
    history = available_feature_list(
        data,
        HISTORY_BASE_FEATURES + [f"{target_col}{suffix}" for suffix in HISTORY_SUFFIXES],
    )
    process = available_feature_list(data, PROCESS_BASE_FEATURES)
    process_lags = available_feature_list(data, [f"{base}{suffix}" for base in PROCESS_LAG_BASES for suffix in HISTORY_SUFFIXES])
    remote = available_feature_list(data, REMOTE_FEATURES)

    return {
        "baseline": baseline,
        "f1_history": baseline + [f for f in history + process + process_lags if f not in baseline],
        "f2_remote": remote,
        "f3_history_remote": baseline + [f for f in history + process + process_lags + remote if f not in baseline],
    }


def split_for_evaluation(subset: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the formal evaluation split without changing target or feature values."""
    if args.split_mode == "random":
        train, test = train_test_split(
            subset,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=True,
        )
        return (
            train.sort_values(["date", "station"]).reset_index(drop=True),
            test.sort_values(["date", "station"]).reset_index(drop=True),
        )

    train = subset[subset["year"] <= args.train_end_year].copy()
    test = subset[subset["year"].between(args.test_start_year, args.test_end_year)].copy()
    return train, test


def transform_target(train_values: pd.Series, use_log1p: bool) -> tuple[np.ndarray, callable]:
    values = pd.to_numeric(train_values, errors="coerce").to_numpy(dtype=float)
    if use_log1p and np.nanmin(values) >= 0:
        return np.log1p(values), np.expm1
    return values, lambda arr: arr


def make_model(args: argparse.Namespace, overrides: dict[str, object] | None = None):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("Missing xgboost. Install it first with `pip install xgboost`.") from exc

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "objective": "reg:squarederror",
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
    }
    if overrides:
        params.update(overrides)
    return XGBRegressor(**params)


def tuning_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    combos: list[dict[str, object]] = []
    for n_estimators, max_depth, learning_rate, subsample, colsample_bytree in product(
        args.tune_n_estimators,
        args.tune_max_depth,
        args.tune_learning_rate,
        args.tune_subsample,
        args.tune_colsample_bytree,
    ):
        combos.append(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }
        )
    return combos


def select_model_params(
    train: pd.DataFrame,
    target_col: str,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    base_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }
    default_meta = {"tuned": False, "validation_rows": 0, "validation_r2": np.nan}
    if not args.tune:
        return base_params, default_meta

    inner_train = train[train["year"] < args.val_start_year].copy()
    inner_val = train[train["year"] >= args.val_start_year].copy()
    if len(inner_train) < args.min_train_rows or len(inner_val) < args.min_test_rows:
        return base_params, default_meta

    y_inner, inverse_transform = transform_target(inner_train[target_col], args.use_log1p_target)
    y_val = inner_val[target_col].to_numpy(dtype=float)

    best_params = base_params.copy()
    best_score = -np.inf
    for candidate in tuning_grid(args):
        model = make_model(args, candidate)
        model.fit(inner_train[features], y_inner)
        pred_val = inverse_transform(model.predict(inner_val[features]))
        pred_val = np.maximum(pred_val, 0)
        score = float(r2_score(y_val, pred_val))
        if score > best_score:
            best_score = score
            best_params = candidate.copy()

    return best_params, {"tuned": True, "validation_rows": len(inner_val), "validation_r2": best_score}


def fit_one_combo(
    data: pd.DataFrame,
    target_key: str,
    depth: str,
    feature_set_name: str,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    target_col = TARGET_MAP[target_key]
    subset = data[(data["depth"] == depth) & data[target_col].notna()].copy()
    subset = subset.sort_values(["date", "station"]).reset_index(drop=True)

    train, test = split_for_evaluation(subset, args)

    result: dict[str, object] = {
        "target": target_key,
        "target_column": target_col,
        "depth": depth,
        "feature_set": feature_set_name,
        "feature_count": len(features),
        "features_json": json.dumps(features, ensure_ascii=False),
        "split_mode": args.split_mode,
        "test_size": args.test_size if args.split_mode == "random" else np.nan,
        "split_random_state": args.random_state if args.split_mode == "random" else np.nan,
        "train_end_year": args.train_end_year if args.split_mode == "time" else np.nan,
        "test_start_year": args.test_start_year if args.split_mode == "time" else np.nan,
        "test_end_year": args.test_end_year if args.split_mode == "time" else np.nan,
        "train_rows": len(train),
        "test_rows": len(test),
    }

    if not features:
        result["status"] = "skipped_no_features"
        return result, pd.DataFrame()
    if len(train) < args.min_train_rows or len(test) < args.min_test_rows:
        result["status"] = "skipped_insufficient_rows"
        return result, pd.DataFrame()

    selected_params, tuning_meta = select_model_params(train, target_col, features, args)
    result.update(
        {
            "tuned": tuning_meta["tuned"],
            "validation_rows": tuning_meta["validation_rows"],
            "validation_r2": tuning_meta["validation_r2"],
            "model_params_json": json.dumps(selected_params, ensure_ascii=False),
        }
    )

    model = make_model(args, selected_params)
    y_train, inverse_transform = transform_target(train[target_col], args.use_log1p_target)
    model.fit(train[features], y_train)
    pred = inverse_transform(model.predict(test[features]))
    pred = np.maximum(pred, 0)
    y = test[target_col].to_numpy()

    result.update(
        {
            "status": "ok",
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
            "nse": float(nse(y, pred)),
        }
    )

    pred_df = test[
        ["obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "year", "month", "depth", target_col]
    ].copy()
    pred_df["target"] = target_key
    pred_df["feature_set"] = feature_set_name
    pred_df["observed"] = pred_df[target_col]
    pred_df["predicted"] = pred
    pred_df = pred_df.drop(columns=[target_col])

    return result, pred_df


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_dataset(args.dataset)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for target_key in args.targets:
        target_col = TARGET_MAP[target_key]
        feature_sets = build_feature_sets(data, target_col)

        for depth in args.depths:
            for feature_set_name in args.feature_sets:
                features = feature_sets[feature_set_name]
                manifest_rows.append(
                    {
                        "target": target_key,
                        "target_column": target_col,
                        "depth": depth,
                        "feature_set": feature_set_name,
                        "feature_count": len(features),
                        "features_json": json.dumps(features, ensure_ascii=False),
                    }
                )
                result, pred_df = fit_one_combo(data, target_key, depth, feature_set_name, features, args)
                metric_rows.append(result)
                if not pred_df.empty:
                    prediction_frames.append(pred_df)

    metrics = pd.DataFrame(metric_rows)
    manifest = pd.DataFrame(manifest_rows).drop_duplicates()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    metrics_out = args.report_dir / "full_station_training_metrics.csv"
    manifest_out = args.report_dir / "full_station_feature_manifest.csv"
    predictions_out = args.report_dir / "full_station_predictions.csv"

    metrics.to_csv(metrics_out, index=False, encoding="utf-8-sig")
    manifest.to_csv(manifest_out, index=False, encoding="utf-8-sig")
    if not predictions.empty:
        predictions.to_csv(predictions_out, index=False, encoding="utf-8-sig")

    ok = metrics[metrics["status"] == "ok"].copy()
    print(f"Wrote {metrics_out}")
    print(f"Wrote {manifest_out}")
    if not predictions.empty:
        print(f"Wrote {predictions_out}")
    if ok.empty:
        print("No successful model runs.")
    else:
        print(ok[["target", "depth", "feature_set", "train_rows", "test_rows", "rmse", "mae", "r2", "nse"]].to_string(index=False))


if __name__ == "__main__":
    main()
