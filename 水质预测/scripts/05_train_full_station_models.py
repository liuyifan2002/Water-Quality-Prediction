from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATASET = Path("data/processed/model_table_3depth_2014_2024.csv")
DEFAULT_REPORT_DIR = Path("reports/full_station")

TARGET_MAP = {
    "chla": "chla_ug_l",
    "secchi": "secchi_depth_m",
    "tn": "tn_mg_l",
    "tp": "tp_mg_l",
}

DEPTH_CHOICES = ["surface", "middle", "bottom"]
FEATURE_SET_CHOICES = ["baseline", "f1_process", "f2_remote", "f3_fusion"]

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

PROCESS_CANDIDATES = [
    "temperature_c",
    "salinity_psu",
    "ph",
    "do_mg_l",
    "do_pct_saturation",
    "turbidity_ntu",
    "suspended_solids_mg_l",
    "bod5_mg_l",
    "ammonia_n_mg_l",
    "nitrate_n_mg_l",
    "nitrite_n_mg_l",
    "orthophosphate_p_mg_l",
    "tin_mg_l",
    "tkn_mg_l",
    "unionised_ammonia_mg_l",
    "vss_mg_l",
    "phaeo_pigments_ug_l",
    "silica_mg_l",
    "chla_ug_l",
    "secchi_depth_m",
    "tn_mg_l",
    "tp_mg_l",
]

REMOTE_FEATURES = [
    "landsat_blue",
    "landsat_green",
    "landsat_red",
    "landsat_nir",
    "landsat_swir1",
    "landsat_swir2",
    "landsat_ndvi",
    "landsat_ndwi",
    "landsat_mndwi",
    "landsat_ndti",
    "landsat_valid_count",
    "landsat_clear_pixel_count",
    "landsat_strict_valid_gee",
    "landsat_missing",
    "landsat_imputed",
    "landsat_no_data",
    "landsat_available",
    "s2_b2_mean",
    "s2_b3_mean",
    "s2_b4_mean",
    "s2_b5_mean",
    "s2_b6_mean",
    "s2_b8_mean",
    "s2_b8a_mean",
    "s2_b11_mean",
    "s2_b12_mean",
    "s2_ndwi_mean",
    "s2_mndwi_mean",
    "s2_ndci_mean",
    "s2_ndti_mean",
    "s2_has_images",
    "s2_image_count",
    "s2_strict_valid",
    "s2_valid_pixel_count",
    "s2_available",
    "satellite_any_available",
    "satellite_available_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the full-station, three-depth XGBoost skeleton for Chl-a/Secchi/TN/TP."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--targets", nargs="*", choices=sorted(TARGET_MAP), default=sorted(TARGET_MAP))
    parser.add_argument("--depths", nargs="*", choices=DEPTH_CHOICES, default=DEPTH_CHOICES)
    parser.add_argument("--feature-sets", nargs="*", choices=FEATURE_SET_CHOICES, default=FEATURE_SET_CHOICES)
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--test-start-year", type=int, default=2023)
    parser.add_argument("--test-end-year", type=int, default=2024)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--min-test-rows", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
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
        raise FileNotFoundError(f"Dataset not found: {path}. Run scripts/04_build_model_table.py first.")

    data = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    data = normalize_columns(data)
    data = normalize_bool_like_columns(data)

    data["water_zone_en"] = data["water_zone_en"].fillna("unknown")
    data["water_zone_code"] = pd.Categorical(data["water_zone_en"]).codes.astype("int16")
    data["quarter"] = pd.to_numeric(data["quarter"], errors="coerce")
    data["is_wet_season"] = pd.to_numeric(data["is_wet_season"], errors="coerce")

    numeric_candidates = set(BASELINE_FEATURES + PROCESS_CANDIDATES + REMOTE_FEATURES + list(TARGET_MAP.values()))
    for col in numeric_candidates:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def available_feature_list(data: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in data.columns]


def build_feature_sets(data: pd.DataFrame, target_col: str) -> dict[str, list[str]]:
    baseline = available_feature_list(data, BASELINE_FEATURES)
    process = available_feature_list(data, [f for f in PROCESS_CANDIDATES if f != target_col])
    remote = available_feature_list(data, REMOTE_FEATURES)

    return {
        "baseline": baseline,
        "f1_process": baseline + [f for f in process if f not in baseline],
        "f2_remote": remote,
        "f3_fusion": baseline + [f for f in process + remote if f not in baseline],
    }


def make_model(args: argparse.Namespace):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("Missing xgboost. Install it first with `pip install xgboost`.") from exc

    return XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )


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

    train = subset[subset["year"] <= args.train_end_year].copy()
    test = subset[subset["year"].between(args.test_start_year, args.test_end_year)].copy()

    result: dict[str, object] = {
        "target": target_key,
        "target_column": target_col,
        "depth": depth,
        "feature_set": feature_set_name,
        "feature_count": len(features),
        "features_json": json.dumps(features, ensure_ascii=False),
        "train_rows": len(train),
        "test_rows": len(test),
    }

    if not features:
        result["status"] = "skipped_no_features"
        return result, pd.DataFrame()
    if len(train) < args.min_train_rows or len(test) < args.min_test_rows:
        result["status"] = "skipped_insufficient_rows"
        return result, pd.DataFrame()

    model = make_model(args)
    model.fit(train[features], train[target_col])
    pred = model.predict(test[features])
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
