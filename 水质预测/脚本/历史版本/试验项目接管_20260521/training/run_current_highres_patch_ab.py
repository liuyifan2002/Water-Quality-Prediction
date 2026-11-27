from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.run_chla_ocean_colour_ab_experiment import (
    BASE_FEATURES,
    OCEAN_COLOUR_FEATURES,
    split_label,
)
from scripts.training.run_chla_ocean_colour_precision_optimization import (
    find_engineered_features,
)


CURRENT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "current"
CURRENT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
CURRENT_CONFIG_DIR = PROJECT_ROOT / "experiments" / "configs" / "current"
CURRENT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"
CURRENT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "current"

BASE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chla_ocean_colour_precision_optimization_20260517"
    / "chla_ocean_colour_precision_optimization_dataset_20260517.csv"
)
HIGHRES_DATASET = CURRENT_PROCESSED_DIR / "highres_patch_features_2014_2024.csv"
REQUEST_TABLE = CURRENT_PROCESSED_DIR / "gee_station_date_requests.csv"
BOTTOM_MECH_TABLE = CURRENT_PROCESSED_DIR / "bottom_mechanism_features_2014_2024.csv"

CHLA_PRED_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "chla_ocean_colour_precision_maximize_20260518"
    / "predictions.csv"
)
SECCHI_LIT_PRED_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "literature_inspired_precision_boost_20260518"
    / "predictions.csv"
)
R2_BREAKTHROUGH_PRED_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "outputs"
    / "r2_0p6_breakthrough_20260518"
    / "predictions.csv"
)

TIME_SPLITS = {
    "train": (2014, 2020),
    "validation": (2021, 2022),
    "test": (2023, 2024),
}
DEPTH_ORDER = ["surface", "middle", "bottom"]
TARGETS = ["chla", "secchi"]
CAT_COLUMNS = ["station", "water_zone_en", "month", "quarter", "is_wet_season"]
ID_COLUMNS = [
    "obs_id",
    "sample_id",
    "station",
    "date",
    "depth",
    "water_zone_en",
    "split",
]

CURRENT_BEST = {
    ("chla", "surface"): {
        "r2": 0.3298422706637335,
        "candidate": "chla_surface_station_calibration",
        "prediction_path": R2_BREAKTHROUGH_PRED_PATH,
        "model_type": "station_calibration",
    },
    ("chla", "middle"): {
        "r2": 0.3645565409147751,
        "candidate": "ensemble_lgbm_rf_avg_middle",
        "prediction_path": CHLA_PRED_PATH,
        "model_type": "ensemble_lgbm_rf_avg",
    },
    ("chla", "bottom"): {
        "r2": 0.3317795164749872,
        "candidate": "impute_knn_bottom",
        "prediction_path": CHLA_PRED_PATH,
        "model_type": "impute_knn_lightgbm",
    },
    ("secchi", "surface"): {
        "r2": 0.4080,
        "candidate": "secchi_surface_stacking_ridge",
        "prediction_path": SECCHI_LIT_PRED_PATH,
        "model_type": "stacking_ridge",
    },
    ("secchi", "middle"): {
        "r2": 0.344727909771647,
        "candidate": "secchi_middle_station_calibration",
        "prediction_path": R2_BREAKTHROUGH_PRED_PATH,
        "model_type": "station_calibration",
    },
    ("secchi", "bottom"): {
        "r2": 0.3574282975687144,
        "candidate": "secchi_bottom_station_calibration",
        "prediction_path": R2_BREAKTHROUGH_PRED_PATH,
        "model_type": "station_calibration",
    },
}

TARGET_SPECS = {
    "chla": {"column": "chla_ug_l", "history_prefix": "chla_ug_l"},
    "secchi": {"column": "secchi_depth_m", "history_prefix": "secchi_depth_m"},
}

SCENARIOS = {
    "s2_patch": {"feature_set_name": "current_best + sentinel2_highres_patch"},
    "landsat_patch": {"feature_set_name": "current_best + landsat_highres_patch"},
    "highres_patch": {"feature_set_name": "current_best + sentinel2_landsat_highres_patch"},
    "bottom_mechanism": {"feature_set_name": "current_best + bottom_mechanism_features"},
}


@dataclass
class CandidateResult:
    target: str
    depth: str
    scenario: str
    candidate_name: str
    model_type: str
    params: dict[str, Any]
    features: list[str]
    train_pred: np.ndarray
    validation_pred: np.ndarray
    test_pred: np.ndarray
    feature_importance: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run current sensor-specific highres patch and mechanism A/B experiments."
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        CURRENT_OUTPUT_DIR,
        CURRENT_REPORT_DIR,
        CURRENT_CONFIG_DIR,
        CURRENT_PROCESSED_DIR,
        CURRENT_INTERIM_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b.abs() > 1e-12)


def split_frame(
    df: pd.DataFrame, target_col: str, depth: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = df[(df["depth"] == depth) & df[target_col].notna()].copy()
    sub = sub[sub["split"].notna()].copy()
    return (
        sub[sub["split"] == "train"].copy(),
        sub[sub["split"] == "validation"].copy(),
        sub[sub["split"] == "test"].copy(),
    )


def current_best_prediction_table() -> pd.DataFrame:
    frames = []
    for (target, depth), meta in CURRENT_BEST.items():
        pred = pd.read_csv(meta["prediction_path"], low_memory=False)
        pred["date"] = pd.to_datetime(pred["date"])
        if "target" not in pred.columns:
            pred["target"] = target
        sub = pred[
            (pred["target"] == target)
            & (pred["depth"] == depth)
            & (pred["candidate"] == meta["candidate"])
        ].copy()
        sub["target"] = target
        sub["depth"] = depth
        frames.append(
            sub[
                [
                    "obs_id",
                    "sample_id",
                    "station",
                    "date",
                    "depth",
                    "target",
                    "split",
                    "candidate",
                    "y_true",
                    "y_pred",
                ]
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def curated_highres_features(columns: list[str]) -> list[str]:
    wanted_stats = ["mean", "stdDev", "median", "p10", "p90", "valid_ratio", "count"]
    selected: list[str] = []
    metadata = [
        "s2_image_id",
        "s2_image_date",
        "s2_date_diff_abs",
        "s2_cloudy_pixel_percentage",
        "s2_image_count_in_window",
        "s2_window_days_used",
        "landsat_image_id",
        "landsat_image_date",
        "landsat_date_diff_abs",
        "landsat_cloud_cover",
        "landsat_image_count_in_window",
        "landsat_window_days_used",
        "landsat_spacecraft_id",
    ]
    for col in columns:
        if col in metadata:
            selected.append(col)
            continue
        if col.startswith(("s2_3x3_", "s2_5x5_", "landsat_3x3_", "landsat_5x5_")):
            parts = col.split("_")
            if len(parts) < 4:
                continue
            stat = "_".join(parts[3:])
            if stat in wanted_stats or stat in {"missing_ratio"}:
                selected.append(col)
    return sorted(dict.fromkeys(selected))


def load_base_dataset() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATASET, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "split" not in df.columns:
        df["split"] = df["year"].map(split_label)
    return df.replace([np.inf, -np.inf], np.nan)


def load_request_map() -> pd.DataFrame:
    req = pd.read_csv(REQUEST_TABLE, usecols=["request_id", "sample_id", "station", "date"])
    req["date"] = pd.to_datetime(req["date"])
    return req.drop_duplicates(subset=["sample_id", "station", "date"])


def load_highres_table() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = pd.read_csv(HIGHRES_DATASET, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    cols = curated_highres_features(df.columns.tolist())
    keep = ["request_id", "sample_id", "station", "date", "lon", "lat", "split"] + cols
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    feature_groups = {
        "s2_patch": [c for c in cols if c.startswith("s2_")],
        "landsat_patch": [c for c in cols if c.startswith("landsat_")],
    }
    feature_groups["highres_patch"] = feature_groups["s2_patch"] + feature_groups["landsat_patch"]
    return out, feature_groups


def load_bottom_mechanism() -> tuple[pd.DataFrame, list[str]]:
    mech = pd.read_csv(BOTTOM_MECH_TABLE, low_memory=False)
    mech["date"] = pd.to_datetime(mech["date"])
    usable = []
    for col in mech.columns:
        if col in {
            "obs_id",
            "sample_id",
            "station",
            "date",
            "depth",
            "year",
            "month",
            "is_wet_season",
            "lon",
            "lat",
            "zone",
        }:
            continue
        if mech[col].notna().sum() > 0:
            usable.append(col)
    keep = ["obs_id", "sample_id", "station", "date", "depth"] + usable
    return mech[keep].copy(), usable


def merge_model_dataset() -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    base = load_base_dataset()
    request_map = load_request_map()
    highres, highres_features = load_highres_table()
    mech, mech_features = load_bottom_mechanism()

    merged = base.merge(
        request_map,
        on=["sample_id", "station", "date"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        highres,
        on=["request_id", "sample_id", "station", "date", "lon", "lat", "split"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        mech,
        on=["obs_id", "sample_id", "station", "date", "depth"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_mech"),
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged.to_csv(
        CURRENT_INTERIM_DIR / "current_model_dataset_with_highres_and_mechanism.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return merged, highres_features, mech_features


def build_common_features(
    df: pd.DataFrame,
    target: str,
    extra_features: list[str] | None = None,
) -> list[str]:
    spec = TARGET_SPECS[target]
    history_prefix = spec["history_prefix"]
    base = [col for col in BASE_FEATURES if col in df.columns and not col.startswith("chla_ug_l_")]
    history = [
        f"{history_prefix}_lag1",
        f"{history_prefix}_lag2",
        f"{history_prefix}_roll3",
        "days_since_prev_sample",
        "station",
        "water_zone_en",
    ]
    history = [col for col in history if col in df.columns]
    ocean = [col for col in OCEAN_COLOUR_FEATURES if col in df.columns]
    engineered = [col for col in find_engineered_features(df) if col in df.columns]
    if target != "chla":
        engineered = [col for col in engineered if not col.startswith("chla_")]
    features = list(dict.fromkeys(base + history + ocean + engineered + (extra_features or [])))
    out: list[str] = []
    for col in features:
        if col in df.columns and df[col].notna().sum() > 0 and col not in out:
            out.append(col)
    return out


def build_stable_core_features(
    df: pd.DataFrame,
    target: str,
    extra_features: list[str] | None = None,
) -> list[str]:
    spec = TARGET_SPECS[target]
    hp = spec["history_prefix"]
    stable = [
        "lon",
        "lat",
        "month_sin",
        "month_cos",
        "day_of_year",
        "days_since_prev_sample",
        "water_zone_en",
        "station",
        f"{hp}_lag1",
        f"{hp}_lag2",
        f"{hp}_roll3",
        "oc_zsd",
        "oc_zsd_3d",
        "oc_zsd_7d",
        "oc_kd490",
        "oc_kd490_7d",
        "oc_chl",
        "valid_count_7d",
    ]
    if target == "chla":
        stable += [
            col
            for col in [
                "chla_history_trend_lag1_lag2",
                "chla_history_lag1_minus_roll3",
                "chla_lag1_per_day_gap",
            ]
            if col in df.columns
        ]
    if extra_features:
        stable += [
            col
            for col in extra_features
            if any(
                token in col
                for token in [
                    "NDCI_mean",
                    "NDWI_mean",
                    "MNDWI_mean",
                    "B3_mean",
                    "B4_mean",
                    "B5_mean",
                    "B8_mean",
                    "LST_K_mean",
                    "valid_ratio",
                ]
            )
        ]
    out: list[str] = []
    for col in stable:
        if col in df.columns and df[col].notna().sum() > 0 and col not in out:
            out.append(col)
    return out


def make_preprocessor(
    train_df: pd.DataFrame,
    features: list[str],
    numeric_mode: str = "median",
) -> ColumnTransformer:
    cat = [col for col in features if col in CAT_COLUMNS or train_df[col].dtype == "object"]
    num = [col for col in features if col not in cat]
    if numeric_mode == "knn":
        num_pipe: Any = KNNImputer(n_neighbors=5, weights="distance")
    else:
        num_pipe = SimpleImputer(strategy="median")
    return ColumnTransformer(
        [
            ("num", num_pipe, num),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def fit_lgbm_pipeline(
    train_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    numeric_mode: str = "median",
) -> Pipeline:
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=18,
        max_depth=6,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    pipe = Pipeline(
        [("preprocessor", make_preprocessor(train_df, features, numeric_mode)), ("model", model)]
    )
    pipe.fit(train_df[features], train_df[target_col])
    return pipe


def fit_rf_pipeline(train_df: pd.DataFrame, features: list[str], target_col: str) -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=350,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    pipe = Pipeline(
        [("preprocessor", make_preprocessor(train_df, features, "median")), ("model", model)]
    )
    pipe.fit(train_df[features], train_df[target_col])
    return pipe


def fit_xgb_pipeline(train_df: pd.DataFrame, features: list[str], target_col: str) -> Pipeline:
    model = XGBRegressor(
        n_estimators=420,
        learning_rate=0.035,
        max_depth=4,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    pipe = Pipeline(
        [("preprocessor", make_preprocessor(train_df, features, "median")), ("model", model)]
    )
    pipe.fit(train_df[features], train_df[target_col])
    return pipe


def high_value_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_y: np.ndarray,
) -> dict[str, float]:
    p90 = float(np.nanquantile(train_y, 0.90))
    mask = y_true >= p90
    return {
        "p90_rmse": rmse(y_true[mask], y_pred[mask]) if mask.any() else np.nan,
        "p90_underestimation_rate": float(np.mean(y_pred[mask] < y_true[mask]))
        if mask.any()
        else np.nan,
    }


def fit_station_calibration(
    df: pd.DataFrame,
    target: str,
    depth: str,
    features: list[str],
    scenario: str,
) -> CandidateResult:
    target_col = TARGET_SPECS[target]["column"]
    train, val, test = split_frame(df, target_col, depth)
    features = [col for col in features if train[col].notna().sum() > 0]
    pipe = fit_lgbm_pipeline(train, features, target_col, "median")
    train_pred = np.maximum(pipe.predict(train[features]), 0)
    val_pred = np.maximum(pipe.predict(val[features]), 0)
    test_pred = np.maximum(pipe.predict(test[features]), 0)
    residual = val[target_col].to_numpy() - val_pred
    bias_table = (
        pd.DataFrame({"station": val["station"].to_numpy(), "residual": residual})
        .groupby("station")
        .agg(mean_residual=("residual", "mean"), n=("residual", "size"))
        .reset_index()
    )
    shrink = 12.0
    bias_table["station_bias"] = bias_table["mean_residual"] * bias_table["n"] / (
        bias_table["n"] + shrink
    )
    default_bias = float(np.mean(residual))
    station_bias = dict(zip(bias_table["station"], bias_table["station_bias"]))
    val_pred = np.maximum(
        val_pred + val["station"].map(station_bias).fillna(default_bias).to_numpy(),
        0,
    )
    test_pred = np.maximum(
        test_pred + test["station"].map(station_bias).fillna(default_bias).to_numpy(),
        0,
    )
    raw_model = pipe.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": features[: len(raw_model.feature_importances_)],
            "importance": raw_model.feature_importances_[: len(features)],
        }
    )
    return CandidateResult(
        target=target,
        depth=depth,
        scenario=scenario,
        candidate_name=f"{target}_{depth}_{scenario}_station_calibration",
        model_type="station_calibration",
        params={
            "base_model": "LightGBM",
            "shrink": shrink,
            "n_estimators": 500,
            "learning_rate": 0.035,
            "num_leaves": 18,
            "max_depth": 6,
            "min_child_samples": 35,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
        features=features,
        train_pred=train_pred,
        validation_pred=val_pred,
        test_pred=test_pred,
        feature_importance=importance,
    )


def fit_ensemble_lgbm_rf(
    df: pd.DataFrame,
    target: str,
    depth: str,
    full_features: list[str],
    stable_features: list[str],
    scenario: str,
) -> CandidateResult:
    target_col = TARGET_SPECS[target]["column"]
    train, val, test = split_frame(df, target_col, depth)
    full_features = [col for col in full_features if train[col].notna().sum() > 0]
    stable_features = [col for col in stable_features if train[col].notna().sum() > 0]
    lgbm = fit_lgbm_pipeline(train, full_features, target_col, "median")
    rf = fit_rf_pipeline(train, stable_features, target_col)
    train_lgbm = np.maximum(lgbm.predict(train[full_features]), 0)
    val_lgbm = np.maximum(lgbm.predict(val[full_features]), 0)
    test_lgbm = np.maximum(lgbm.predict(test[full_features]), 0)
    train_rf = np.maximum(rf.predict(train[stable_features]), 0)
    val_rf = np.maximum(rf.predict(val[stable_features]), 0)
    test_rf = np.maximum(rf.predict(test[stable_features]), 0)
    best_weight = 0.5
    best_r2 = -np.inf
    for weight in [0.25, 0.35, 0.50, 0.65, 0.75]:
        pred = weight * val_lgbm + (1 - weight) * val_rf
        score = float(r2_score(val[target_col], pred))
        if score > best_r2:
            best_r2 = score
            best_weight = weight
    train_pred = np.maximum(best_weight * train_lgbm + (1 - best_weight) * train_rf, 0)
    val_pred = np.maximum(best_weight * val_lgbm + (1 - best_weight) * val_rf, 0)
    test_pred = np.maximum(best_weight * test_lgbm + (1 - best_weight) * test_rf, 0)
    raw_model = lgbm.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": full_features[: len(raw_model.feature_importances_)],
            "importance": raw_model.feature_importances_[: len(full_features)],
        }
    )
    return CandidateResult(
        target=target,
        depth=depth,
        scenario=scenario,
        candidate_name=f"{target}_{depth}_{scenario}_ensemble_lgbm_rf_avg",
        model_type="ensemble_lgbm_rf_avg",
        params={
            "blend_weight_lgbm": best_weight,
            "blend_weight_rf": 1 - best_weight,
            "lgbm_n_estimators": 500,
            "rf_n_estimators": 350,
            "rf_min_samples_leaf": 2,
        },
        features=list(dict.fromkeys(full_features + stable_features)),
        train_pred=train_pred,
        validation_pred=val_pred,
        test_pred=test_pred,
        feature_importance=importance,
    )


def time_aware_oof(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    features: list[str],
    family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oof = np.full(len(train), np.nan)
    years = [2017, 2018, 2019, 2020]
    for year in years:
        fit_mask = train["year"] < year
        pred_mask = train["year"] == year
        if fit_mask.sum() < 500 or pred_mask.sum() == 0:
            continue
        if family == "lightgbm":
            pipe = fit_lgbm_pipeline(train.loc[fit_mask].copy(), features, target_col, "median")
        elif family == "xgboost":
            pipe = fit_xgb_pipeline(train.loc[fit_mask].copy(), features, target_col)
        else:
            pipe = fit_rf_pipeline(train.loc[fit_mask].copy(), features, target_col)
        oof[np.where(pred_mask)[0]] = np.maximum(
            pipe.predict(train.loc[pred_mask, features]),
            0,
        )
    missing = np.isnan(oof)
    if missing.any():
        fill_pipe = fit_lgbm_pipeline(train.copy(), features, target_col, "median")
        oof[missing] = np.maximum(fill_pipe.predict(train.loc[missing, features]), 0)
    if family == "lightgbm":
        pipe = fit_lgbm_pipeline(train.copy(), features, target_col, "median")
    elif family == "xgboost":
        pipe = fit_xgb_pipeline(train.copy(), features, target_col)
    else:
        pipe = fit_rf_pipeline(train.copy(), features, target_col)
    return (
        oof,
        np.maximum(pipe.predict(val[features]), 0),
        np.maximum(pipe.predict(test[features]), 0),
    )


def fit_stacking_ridge(
    df: pd.DataFrame,
    target: str,
    depth: str,
    features: list[str],
    scenario: str,
) -> CandidateResult:
    target_col = TARGET_SPECS[target]["column"]
    train, val, test = split_frame(df, target_col, depth)
    features = [col for col in features if train[col].notna().sum() > 0]
    families = ["lightgbm", "xgboost", "random_forest"]
    train_meta = []
    val_meta = []
    test_meta = []
    for family in families:
        oof, val_pred, test_pred = time_aware_oof(train, val, test, target_col, features, family)
        train_meta.append(oof)
        val_meta.append(val_pred)
        test_meta.append(test_pred)
    x_train = np.vstack(train_meta).T
    x_val = np.vstack(val_meta).T
    x_test = np.vstack(test_meta).T
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_s, train[target_col])
    train_pred = np.maximum(ridge.predict(x_train_s), 0)
    val_pred = np.maximum(ridge.predict(x_val_s), 0)
    test_pred = np.maximum(ridge.predict(x_test_s), 0)
    importance = pd.DataFrame(
        {"feature": [f"meta_{name}" for name in families], "importance": ridge.coef_}
    )
    return CandidateResult(
        target=target,
        depth=depth,
        scenario=scenario,
        candidate_name=f"{target}_{depth}_{scenario}_stacking_ridge",
        model_type="stacking_ridge",
        params={"base_models": families, "meta_model": "Ridge", "alpha": 1.0},
        features=features,
        train_pred=train_pred,
        validation_pred=val_pred,
        test_pred=test_pred,
        feature_importance=importance,
    )


def fit_impute_knn_lgbm(
    df: pd.DataFrame,
    target: str,
    depth: str,
    features: list[str],
    scenario: str,
) -> CandidateResult:
    target_col = TARGET_SPECS[target]["column"]
    train, val, test = split_frame(df, target_col, depth)
    features = [col for col in features if train[col].notna().sum() > 0]
    pipe = fit_lgbm_pipeline(train, features, target_col, "knn")
    train_pred = np.maximum(pipe.predict(train[features]), 0)
    val_pred = np.maximum(pipe.predict(val[features]), 0)
    test_pred = np.maximum(pipe.predict(test[features]), 0)
    raw_model = pipe.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": features[: len(raw_model.feature_importances_)],
            "importance": raw_model.feature_importances_[: len(features)],
        }
    )
    return CandidateResult(
        target=target,
        depth=depth,
        scenario=scenario,
        candidate_name=f"{target}_{depth}_{scenario}_impute_knn_lgbm",
        model_type="impute_knn_lightgbm",
        params={
            "numeric_imputer": "KNNImputer",
            "knn_n_neighbors": 5,
            "knn_weights": "distance",
            "n_estimators": 500,
            "learning_rate": 0.035,
            "num_leaves": 18,
            "max_depth": 6,
            "min_child_samples": 35,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
        features=features,
        train_pred=train_pred,
        validation_pred=val_pred,
        test_pred=test_pred,
        feature_importance=importance,
    )


def metrics_row(
    target: str,
    depth: str,
    split: str,
    candidate_name: str,
    model_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_y: np.ndarray,
    features: list[str],
    scenario: str,
) -> dict[str, Any]:
    metrics = {
        "target": target,
        "depth_layer": depth,
        "split": split,
        "scenario": scenario,
        "candidate": candidate_name,
        "model_type": model_type,
        "n_samples": int(len(y_true)),
        "feature_count": int(len(features)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }
    metrics.update(high_value_metrics(y_true, y_pred, train_y))
    return metrics


def candidate_prediction_frame(df: pd.DataFrame, result: CandidateResult) -> pd.DataFrame:
    target_col = TARGET_SPECS[result.target]["column"]
    train, val, test = split_frame(df, target_col, result.depth)
    frames = []
    for split, frame, pred in [
        ("train", train, result.train_pred),
        ("validation", val, result.validation_pred),
        ("test", test, result.test_pred),
    ]:
        out = frame[[c for c in ID_COLUMNS if c in frame.columns]].copy()
        out["target"] = result.target
        out["candidate"] = result.candidate_name
        out["model_type"] = result.model_type
        out["scenario"] = result.scenario
        out["y_true"] = frame[target_col].to_numpy()
        out["y_pred"] = pred
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def compare_station_level(
    target: str,
    depth: str,
    baseline_test: pd.DataFrame,
    candidate_test: pd.DataFrame,
    candidate_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = baseline_test[
        ["obs_id", "sample_id", "station", "date", "depth", "target", "y_true", "y_pred"]
    ].rename(columns={"y_pred": "current_best_pred"})
    cand = candidate_test[
        ["obs_id", "sample_id", "station", "date", "depth", "target", "y_true", "y_pred"]
    ].rename(columns={"y_pred": "candidate_pred"})
    merged = base.merge(
        cand,
        on=["obs_id", "sample_id", "station", "date", "depth", "target", "y_true"],
        how="inner",
        validate="one_to_one",
    )
    merged["absolute_error_current_best"] = (
        merged["current_best_pred"] - merged["y_true"]
    ).abs()
    merged["absolute_error_candidate"] = (merged["candidate_pred"] - merged["y_true"]).abs()
    merged["error_improved"] = (
        merged["absolute_error_candidate"] < merged["absolute_error_current_best"]
    )
    station_rows = []
    for station, sub in merged.groupby("station"):
        current_r2 = float(r2_score(sub["y_true"], sub["current_best_pred"])) if len(sub) > 1 else np.nan
        candidate_r2 = float(r2_score(sub["y_true"], sub["candidate_pred"])) if len(sub) > 1 else np.nan
        current_rmse = rmse(sub["y_true"].to_numpy(), sub["current_best_pred"].to_numpy())
        candidate_rmse = rmse(sub["y_true"].to_numpy(), sub["candidate_pred"].to_numpy())
        current_mae = float(mean_absolute_error(sub["y_true"], sub["current_best_pred"]))
        candidate_mae = float(mean_absolute_error(sub["y_true"], sub["candidate_pred"]))
        current_bias = float(np.mean(sub["current_best_pred"] - sub["y_true"]))
        candidate_bias = float(np.mean(sub["candidate_pred"] - sub["y_true"]))
        p90 = float(np.nanquantile(sub["y_true"], 0.90))
        mask = sub["y_true"] >= p90
        current_p90_rmse = (
            rmse(sub.loc[mask, "y_true"], sub.loc[mask, "current_best_pred"]) if mask.any() else np.nan
        )
        candidate_p90_rmse = (
            rmse(sub.loc[mask, "y_true"], sub.loc[mask, "candidate_pred"]) if mask.any() else np.nan
        )
        improved = bool((candidate_rmse < current_rmse) or (candidate_r2 > current_r2))
        station_rows.append(
            {
                "target": target,
                "depth_layer": depth,
                "station": station,
                "candidate_label": candidate_label,
                "n_test": int(len(sub)),
                "current_best_r2": current_r2,
                "candidate_r2": candidate_r2,
                "delta_r2": candidate_r2 - current_r2 if pd.notna(current_r2) and pd.notna(candidate_r2) else np.nan,
                "current_best_rmse": current_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse": candidate_rmse - current_rmse,
                "current_best_mae": current_mae,
                "candidate_mae": candidate_mae,
                "delta_mae": candidate_mae - current_mae,
                "current_best_bias": current_bias,
                "candidate_bias": candidate_bias,
                "delta_bias": candidate_bias - current_bias,
                "current_best_p90_rmse": current_p90_rmse,
                "candidate_p90_rmse": candidate_p90_rmse,
                "delta_p90_rmse": candidate_p90_rmse - current_p90_rmse
                if pd.notna(current_p90_rmse) and pd.notna(candidate_p90_rmse)
                else np.nan,
                "improved": improved,
                "recommended_model_for_station": candidate_label if improved else "current_best",
                "reason": "candidate_lower_rmse_or_higher_r2" if improved else "keep_current_best",
            }
        )
    merged["candidate_label"] = candidate_label
    return pd.DataFrame(station_rows), merged


def fit_for_current_best_family(
    df: pd.DataFrame,
    target: str,
    depth: str,
    features: list[str],
    scenario: str,
) -> CandidateResult:
    best = CURRENT_BEST[(target, depth)]
    if target == "chla" and depth == "middle":
        stable = build_stable_core_features(df, target, features)
        return fit_ensemble_lgbm_rf(df, target, depth, features, stable, scenario)
    if best["model_type"] == "station_calibration":
        return fit_station_calibration(df, target, depth, features, scenario)
    if best["model_type"] == "impute_knn_lightgbm":
        return fit_impute_knn_lgbm(df, target, depth, features, scenario)
    if best["model_type"] == "stacking_ridge":
        return fit_stacking_ridge(df, target, depth, features, scenario)
    if best["model_type"] == "ensemble_lgbm_rf_avg":
        stable = build_stable_core_features(df, target, features)
        return fit_ensemble_lgbm_rf(df, target, depth, features, stable, scenario)
    raise ValueError(best["model_type"])


def current_best_metric_map() -> dict[tuple[str, str], dict[str, float]]:
    baseline_predictions = current_best_prediction_table()
    out: dict[tuple[str, str], dict[str, float]] = {}
    for (target, depth), sub in baseline_predictions[baseline_predictions["split"] == "test"].groupby(
        ["target", "depth"]
    ):
        y_true = sub["y_true"].to_numpy()
        y_pred = sub["y_pred"].to_numpy()
        high = high_value_metrics(y_true, y_pred, y_true)
        out[(target, depth)] = {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": rmse(y_true, y_pred),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "bias": float(np.mean(y_pred - y_true)),
            "p90_rmse": high["p90_rmse"],
            "p90_underestimation_rate": high["p90_underestimation_rate"],
        }
    return out


def run_patch_scenario(
    df: pd.DataFrame,
    feature_subset: list[str],
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline_predictions = current_best_prediction_table()
    metric_rows = []
    prediction_frames = []
    importance_frames = []
    station_compare_frames = []
    comparison_rows = []
    manifest: dict[str, Any] = {"scenario": scenario, "models": {}}

    for target in TARGETS:
        for depth in DEPTH_ORDER:
            features = build_common_features(df, target, extra_features=feature_subset)
            result = fit_for_current_best_family(df, target, depth, features, scenario)
            manifest["models"][f"{target}::{depth}"] = {
                "candidate_name": result.candidate_name,
                "model_type": result.model_type,
                "params": result.params,
                "features": result.features,
                "feature_count": len(result.features),
            }
            prediction_frame = candidate_prediction_frame(df, result)
            prediction_frames.append(prediction_frame)
            imp = result.feature_importance.copy()
            imp["target"] = target
            imp["depth_layer"] = depth
            imp["candidate"] = result.candidate_name
            imp["scenario"] = scenario
            importance_frames.append(imp)

            target_col = TARGET_SPECS[target]["column"]
            train, val, test = split_frame(df, target_col, depth)
            metric_rows.extend(
                [
                    metrics_row(
                        target,
                        depth,
                        "train",
                        result.candidate_name,
                        result.model_type,
                        train[target_col].to_numpy(),
                        result.train_pred,
                        train[target_col].to_numpy(),
                        result.features,
                        scenario,
                    ),
                    metrics_row(
                        target,
                        depth,
                        "validation",
                        result.candidate_name,
                        result.model_type,
                        val[target_col].to_numpy(),
                        result.validation_pred,
                        train[target_col].to_numpy(),
                        result.features,
                        scenario,
                    ),
                    metrics_row(
                        target,
                        depth,
                        "test",
                        result.candidate_name,
                        result.model_type,
                        test[target_col].to_numpy(),
                        result.test_pred,
                        train[target_col].to_numpy(),
                        result.features,
                        scenario,
                    ),
                ]
            )

            baseline_test = baseline_predictions[
                (baseline_predictions["target"] == target)
                & (baseline_predictions["depth"] == depth)
                & (baseline_predictions["split"] == "test")
            ].copy()
            candidate_test = prediction_frame[prediction_frame["split"] == "test"].copy()
            station_cmp, _ = compare_station_level(
                target,
                depth,
                baseline_test,
                candidate_test,
                scenario,
            )
            station_compare_frames.append(station_cmp)

            current_rmse = rmse(baseline_test["y_true"].to_numpy(), baseline_test["y_pred"].to_numpy())
            current_mae = float(mean_absolute_error(baseline_test["y_true"], baseline_test["y_pred"]))
            current_val = baseline_predictions[
                (baseline_predictions["target"] == target)
                & (baseline_predictions["depth"] == depth)
                & (baseline_predictions["split"] == "validation")
            ].copy()
            candidate_val = prediction_frame[prediction_frame["split"] == "validation"].copy()
            candidate_r2 = float(r2_score(candidate_test["y_true"], candidate_test["y_pred"]))
            candidate_val_r2 = float(r2_score(candidate_val["y_true"], candidate_val["y_pred"]))
            baseline_val_r2 = float(r2_score(current_val["y_true"], current_val["y_pred"]))
            improved_ratio = float(station_cmp["improved"].mean()) if len(station_cmp) else 0.0
            candidate_high = high_value_metrics(
                candidate_test["y_true"].to_numpy(),
                candidate_test["y_pred"].to_numpy(),
                train[target_col].to_numpy(),
            )
            baseline_high = high_value_metrics(
                baseline_test["y_true"].to_numpy(),
                baseline_test["y_pred"].to_numpy(),
                train[target_col].to_numpy(),
            )
            comparison_rows.append(
                {
                    "target": target,
                    "depth_layer": depth,
                    "scenario": scenario,
                    "current_best_candidate": CURRENT_BEST[(target, depth)]["candidate"],
                    "candidate_model": result.candidate_name,
                    "current_best_model_type": CURRENT_BEST[(target, depth)]["model_type"],
                    "candidate_model_type": result.model_type,
                    "current_best_r2": CURRENT_BEST[(target, depth)]["r2"],
                    "candidate_r2": candidate_r2,
                    "delta_r2": candidate_r2 - CURRENT_BEST[(target, depth)]["r2"],
                    "current_best_validation_r2": baseline_val_r2,
                    "candidate_validation_r2": candidate_val_r2,
                    "delta_validation_r2": candidate_val_r2 - baseline_val_r2,
                    "current_best_rmse": current_rmse,
                    "candidate_rmse": rmse(candidate_test["y_true"].to_numpy(), candidate_test["y_pred"].to_numpy()),
                    "delta_rmse": rmse(candidate_test["y_true"].to_numpy(), candidate_test["y_pred"].to_numpy()) - current_rmse,
                    "current_best_mae": current_mae,
                    "candidate_mae": float(mean_absolute_error(candidate_test["y_true"], candidate_test["y_pred"])),
                    "delta_mae": float(mean_absolute_error(candidate_test["y_true"], candidate_test["y_pred"])) - current_mae,
                    "current_best_p90_rmse": baseline_high["p90_rmse"],
                    "candidate_p90_rmse": candidate_high["p90_rmse"],
                    "delta_p90_rmse": candidate_high["p90_rmse"] - baseline_high["p90_rmse"]
                    if pd.notna(candidate_high["p90_rmse"]) and pd.notna(baseline_high["p90_rmse"])
                    else np.nan,
                    "feature_count": len(result.features),
                    "station_improved_ratio": improved_ratio,
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    importance_df = pd.concat(importance_frames, ignore_index=True)
    station_cmp_df = pd.concat(station_compare_frames, ignore_index=True)
    comparison_df = pd.DataFrame(comparison_rows)
    return metrics_df, pred_df, importance_df, station_cmp_df, comparison_df, manifest


def classify_candidate(row: pd.Series) -> tuple[str, bool]:
    delta_r2 = float(row["delta_r2"])
    delta_val_r2 = float(row["delta_validation_r2"])
    improved_ratio = float(row["station_improved_ratio"])
    delta_p90 = row["delta_p90_rmse"]
    if delta_r2 >= 0.03 and delta_val_r2 > 0:
        return "mainline_update", True
    if delta_r2 > 0 and delta_val_r2 <= 0:
        return "suspicious", False
    if delta_r2 < 0.03 and improved_ratio >= 0.60:
        return "station_specific_auxiliary", False
    if pd.notna(delta_p90) and delta_p90 < 0 and delta_r2 <= 0:
        return "high_value_auxiliary", False
    return "retain_current_best", False


def update_scorecard_and_registry(
    scenario_comparisons: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    registry_rows = []
    for target in TARGETS:
        for depth in DEPTH_ORDER:
            best = CURRENT_BEST[(target, depth)]
            candidates = []
            for scenario, comp in scenario_comparisons.items():
                sub = comp[(comp["target"] == target) & (comp["depth_layer"] == depth)]
                if not sub.empty:
                    row = sub.iloc[0].copy()
                    row["scenario"] = scenario
                    status, replace = classify_candidate(row)
                    row["selection_status"] = status
                    row["eligible_mainline"] = replace
                    candidates.append(row)
                    registry_rows.append(row.to_dict())
            selected_model = best["candidate"]
            selected_feature_set = "retain_previous_best"
            final_selected_r2 = best["r2"]
            whether_updated = False
            reason = "retain_current_best"
            eligible = [c for c in candidates if bool(c["eligible_mainline"])]
            if eligible:
                chosen = max(eligible, key=lambda x: float(x["candidate_r2"]))
                selected_model = str(chosen["candidate_model"])
                selected_feature_set = SCENARIOS[str(chosen["scenario"])]["feature_set_name"]
                final_selected_r2 = float(chosen["candidate_r2"])
                whether_updated = True
                reason = str(chosen["selection_status"])
            rows.append(
                {
                    "target": target,
                    "depth_layer": depth,
                    "previous_best_r2": float(best["r2"]),
                    "current_candidate_r2": max([float(c["candidate_r2"]) for c in candidates], default=float(best["r2"])),
                    "final_selected_r2": final_selected_r2,
                    "selected_model": selected_model,
                    "selected_feature_set": selected_feature_set,
                    "whether_updated": whether_updated,
                    "reason": reason,
                }
            )
    scorecard = pd.DataFrame(rows)
    registry = pd.DataFrame(registry_rows)
    scorecard.to_csv(CURRENT_OUTPUT_DIR / "current_best_scorecard.csv", index=False, encoding="utf-8-sig")
    scorecard.to_csv(CURRENT_PROCESSED_DIR / "current_best_scorecard.csv", index=False, encoding="utf-8-sig")
    registry.to_csv(CURRENT_OUTPUT_DIR / "current_candidate_registry.csv", index=False, encoding="utf-8-sig")
    return scorecard, registry


def build_station_level_outputs(
    baseline_predictions: pd.DataFrame,
    scenario_prediction_tables: dict[str, pd.DataFrame],
    scorecard: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    pred_frames = []
    for target in TARGETS:
        for depth in DEPTH_ORDER:
            baseline = baseline_predictions[
                (baseline_predictions["target"] == target)
                & (baseline_predictions["depth"] == depth)
                & (baseline_predictions["split"] == "test")
            ].copy()
            merged = baseline[
                ["obs_id", "sample_id", "station", "date", "depth", "target", "y_true", "y_pred"]
            ].rename(columns={"y_pred": "current_best_pred"})
            scenario_cols = {
                "s2_patch": "s2_patch_pred",
                "landsat_patch": "landsat_patch_pred",
                "highres_patch": "highres_patch_pred",
                "bottom_mechanism": "mechanism_pred",
            }
            for scenario, pred_col in scenario_cols.items():
                table = scenario_prediction_tables.get(scenario)
                if table is None:
                    continue
                sub = table[
                    (table["target"] == target)
                    & (table["depth"] == depth)
                    & (table["split"] == "test")
                ][["obs_id", "sample_id", "station", "date", "depth", "target", "y_pred"]].rename(
                    columns={"y_pred": pred_col}
                )
                merged = merged.merge(
                    sub,
                    on=["obs_id", "sample_id", "station", "date", "depth", "target"],
                    how="left",
                    validate="one_to_one",
                )
            for station, sub in merged.groupby("station"):
                model_metrics = {}
                for label, pred_col in [
                    ("current_best", "current_best_pred"),
                    ("s2_patch", "s2_patch_pred"),
                    ("landsat_patch", "landsat_patch_pred"),
                    ("highres_patch", "highres_patch_pred"),
                    ("mechanism", "mechanism_pred"),
                ]:
                    valid = sub[pred_col].notna()
                    if valid.sum() < 2:
                        model_metrics[label] = {
                            "r2": np.nan,
                            "rmse": np.nan,
                            "mae": np.nan,
                            "bias": np.nan,
                        }
                        continue
                    y_true = sub.loc[valid, "y_true"].to_numpy()
                    y_pred = sub.loc[valid, pred_col].to_numpy()
                    model_metrics[label] = {
                        "r2": float(r2_score(y_true, y_pred)),
                        "rmse": rmse(y_true, y_pred),
                        "mae": float(mean_absolute_error(y_true, y_pred)),
                        "bias": float(np.mean(y_pred - y_true)),
                    }
                choices = {
                    key: val["rmse"]
                    for key, val in model_metrics.items()
                    if pd.notna(val["rmse"])
                }
                recommended = min(choices, key=choices.get) if choices else "current_best"
                current_rmse = model_metrics["current_best"]["rmse"]
                candidate_best_r2 = max(
                    [v["r2"] for k, v in model_metrics.items() if k != "current_best" and pd.notna(v["r2"])],
                    default=np.nan,
                )
                comparison_rows.append(
                    {
                        "target": target,
                        "depth_layer": depth,
                        "station": station,
                        "n_test": int(len(sub)),
                        "current_best_r2": model_metrics["current_best"]["r2"],
                        "s2_patch_r2": model_metrics["s2_patch"]["r2"],
                        "landsat_patch_r2": model_metrics["landsat_patch"]["r2"],
                        "highres_patch_r2": model_metrics["highres_patch"]["r2"],
                        "mechanism_r2": model_metrics["mechanism"]["r2"],
                        "best_station_r2": candidate_best_r2
                        if pd.notna(candidate_best_r2)
                        else model_metrics["current_best"]["r2"],
                        "current_best_rmse": model_metrics["current_best"]["rmse"],
                        "s2_patch_rmse": model_metrics["s2_patch"]["rmse"],
                        "landsat_patch_rmse": model_metrics["landsat_patch"]["rmse"],
                        "highres_patch_rmse": model_metrics["highres_patch"]["rmse"],
                        "mechanism_rmse": model_metrics["mechanism"]["rmse"],
                        "current_best_mae": model_metrics["current_best"]["mae"],
                        "s2_patch_mae": model_metrics["s2_patch"]["mae"],
                        "landsat_patch_mae": model_metrics["landsat_patch"]["mae"],
                        "highres_patch_mae": model_metrics["highres_patch"]["mae"],
                        "mechanism_mae": model_metrics["mechanism"]["mae"],
                        "current_best_bias": model_metrics["current_best"]["bias"],
                        "s2_patch_bias": model_metrics["s2_patch"]["bias"],
                        "landsat_patch_bias": model_metrics["landsat_patch"]["bias"],
                        "highres_patch_bias": model_metrics["highres_patch"]["bias"],
                        "mechanism_bias": model_metrics["mechanism"]["bias"],
                        "recommended_model_for_station": recommended,
                        "improved": bool(recommended != "current_best"),
                        "reason": "lower_station_rmse" if recommended != "current_best" else "keep_current_best",
                    }
                )
                chosen_col = {
                    "current_best": "current_best_pred",
                    "s2_patch": "s2_patch_pred",
                    "landsat_patch": "landsat_patch_pred",
                    "highres_patch": "highres_patch_pred",
                    "mechanism": "mechanism_pred",
                }[recommended]
                station_frame = sub.copy()
                station_frame["recommended_model_for_station"] = recommended
                station_frame["best_pred"] = station_frame[chosen_col]
                station_frame["global_pred"] = station_frame["current_best_pred"]
                station_frame["grouped_pred"] = station_frame["highres_patch_pred"]
                station_frame["model_used"] = recommended
                station_frame["absolute_error_global"] = (
                    station_frame["global_pred"] - station_frame["y_true"]
                ).abs()
                station_frame["absolute_error_grouped"] = (
                    station_frame["best_pred"] - station_frame["y_true"]
                ).abs()
                station_frame["error_improved"] = (
                    station_frame["absolute_error_grouped"]
                    < station_frame["absolute_error_global"]
                )
                pred_frames.append(station_frame)
    station_cmp = pd.DataFrame(comparison_rows)
    station_pred = pd.concat(pred_frames, ignore_index=True)
    station_cmp.to_csv(
        CURRENT_OUTPUT_DIR / "station_level_prediction_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_pred.to_csv(
        CURRENT_OUTPUT_DIR / "station_level_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return station_cmp, station_pred


def update_cleanup_log(lines: list[str]) -> None:
    path = CURRENT_REPORT_DIR / "cleanup_log.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# 清理与汇总日志\n"
    path.write_text(text.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parse_args()
    ensure_dirs()

    df, highres_feature_groups, mechanism_features = merge_model_dataset()
    baseline_predictions = current_best_prediction_table()

    scenario_prediction_tables: dict[str, pd.DataFrame] = {}
    scenario_manifests: dict[str, Any] = {}
    scenario_comparisons: dict[str, pd.DataFrame] = {}

    scenario_output_map = {
        "s2_patch": {
            "metrics": CURRENT_OUTPUT_DIR / "s2_patch_ab_metrics.csv",
            "predictions": CURRENT_OUTPUT_DIR / "s2_patch_predictions.csv",
            "importance": CURRENT_OUTPUT_DIR / "s2_patch_feature_importance.csv",
            "station": CURRENT_OUTPUT_DIR / "s2_patch_station_level_comparison.csv",
        },
        "landsat_patch": {
            "metrics": CURRENT_OUTPUT_DIR / "landsat_patch_ab_metrics.csv",
            "predictions": CURRENT_OUTPUT_DIR / "landsat_patch_predictions.csv",
            "importance": CURRENT_OUTPUT_DIR / "landsat_patch_feature_importance.csv",
            "station": CURRENT_OUTPUT_DIR / "landsat_patch_station_level_comparison.csv",
        },
        "highres_patch": {
            "metrics": CURRENT_OUTPUT_DIR / "highres_patch_ab_metrics.csv",
            "predictions": CURRENT_OUTPUT_DIR / "highres_patch_predictions.csv",
            "importance": CURRENT_OUTPUT_DIR / "highres_patch_feature_importance.csv",
            "station": CURRENT_OUTPUT_DIR / "highres_patch_station_level_comparison.csv",
            "comparison": CURRENT_OUTPUT_DIR / "highres_patch_comparison_to_current_best.csv",
        },
    }

    for scenario in ["s2_patch", "landsat_patch", "highres_patch"]:
        feature_subset = highres_feature_groups[scenario]
        (
            metrics_df,
            pred_df,
            importance_df,
            station_cmp_df,
            comparison_df,
            manifest,
        ) = run_patch_scenario(df, feature_subset, scenario)
        scenario_output_map[scenario]["metrics"].write_text("", encoding="utf-8") if False else None
        metrics_df.to_csv(scenario_output_map[scenario]["metrics"], index=False, encoding="utf-8-sig")
        pred_df.to_csv(scenario_output_map[scenario]["predictions"], index=False, encoding="utf-8-sig")
        importance_df.to_csv(scenario_output_map[scenario]["importance"], index=False, encoding="utf-8-sig")
        station_cmp_df.to_csv(scenario_output_map[scenario]["station"], index=False, encoding="utf-8-sig")
        if "comparison" in scenario_output_map[scenario]:
            comparison_df.to_csv(
                scenario_output_map[scenario]["comparison"], index=False, encoding="utf-8-sig"
            )
        scenario_prediction_tables[scenario] = pred_df
        scenario_manifests[scenario] = manifest
        scenario_comparisons[scenario] = comparison_df

    mech_metrics, mech_pred, mech_importance, mech_station, mech_manifest = (None, None, None, None, None)
    if mechanism_features:
        mech_result = fit_impute_knn_lgbm(
            df,
            "chla",
            "bottom",
            build_common_features(df, "chla", extra_features=mechanism_features),
            "bottom_mechanism",
        )
        mech_pred = candidate_prediction_frame(df, mech_result)
        target_col = TARGET_SPECS["chla"]["column"]
        train, val, test = split_frame(df, target_col, "bottom")
        mech_metrics = pd.DataFrame(
            [
                metrics_row(
                    "chla",
                    "bottom",
                    "train",
                    mech_result.candidate_name,
                    mech_result.model_type,
                    train[target_col].to_numpy(),
                    mech_result.train_pred,
                    train[target_col].to_numpy(),
                    mech_result.features,
                    "bottom_mechanism",
                ),
                metrics_row(
                    "chla",
                    "bottom",
                    "validation",
                    mech_result.candidate_name,
                    mech_result.model_type,
                    val[target_col].to_numpy(),
                    mech_result.validation_pred,
                    train[target_col].to_numpy(),
                    mech_result.features,
                    "bottom_mechanism",
                ),
                metrics_row(
                    "chla",
                    "bottom",
                    "test",
                    mech_result.candidate_name,
                    mech_result.model_type,
                    test[target_col].to_numpy(),
                    mech_result.test_pred,
                    train[target_col].to_numpy(),
                    mech_result.features,
                    "bottom_mechanism",
                ),
            ]
        )
        mech_importance = mech_result.feature_importance.copy()
        mech_importance["target"] = "chla"
        mech_importance["depth_layer"] = "bottom"
        mech_importance["candidate"] = mech_result.candidate_name
        mech_importance["scenario"] = "bottom_mechanism"
        baseline_bottom = baseline_predictions[
            (baseline_predictions["target"] == "chla")
            & (baseline_predictions["depth"] == "bottom")
            & (baseline_predictions["split"] == "test")
        ].copy()
        mech_station, _ = compare_station_level(
            "chla",
            "bottom",
            baseline_bottom,
            mech_pred[mech_pred["split"] == "test"].copy(),
            "bottom_mechanism",
        )
        mech_comparison = {
            "target": "chla",
            "depth_layer": "bottom",
            "scenario": "bottom_mechanism",
            "current_best_candidate": CURRENT_BEST[("chla", "bottom")]["candidate"],
            "candidate_model": mech_result.candidate_name,
            "current_best_model_type": CURRENT_BEST[("chla", "bottom")]["model_type"],
            "candidate_model_type": mech_result.model_type,
            "current_best_r2": CURRENT_BEST[("chla", "bottom")]["r2"],
            "candidate_r2": float(
                r2_score(
                    mech_pred[mech_pred["split"] == "test"]["y_true"],
                    mech_pred[mech_pred["split"] == "test"]["y_pred"],
                )
            ),
            "delta_r2": float(
                r2_score(
                    mech_pred[mech_pred["split"] == "test"]["y_true"],
                    mech_pred[mech_pred["split"] == "test"]["y_pred"],
                )
                - CURRENT_BEST[("chla", "bottom")]["r2"]
            ),
            "current_best_validation_r2": float(
                r2_score(
                    baseline_predictions[
                        (baseline_predictions["target"] == "chla")
                        & (baseline_predictions["depth"] == "bottom")
                        & (baseline_predictions["split"] == "validation")
                    ]["y_true"],
                    baseline_predictions[
                        (baseline_predictions["target"] == "chla")
                        & (baseline_predictions["depth"] == "bottom")
                        & (baseline_predictions["split"] == "validation")
                    ]["y_pred"],
                )
            ),
            "candidate_validation_r2": float(
                r2_score(
                    mech_pred[mech_pred["split"] == "validation"]["y_true"],
                    mech_pred[mech_pred["split"] == "validation"]["y_pred"],
                )
            ),
            "delta_validation_r2": float(
                r2_score(
                    mech_pred[mech_pred["split"] == "validation"]["y_true"],
                    mech_pred[mech_pred["split"] == "validation"]["y_pred"],
                )
                - r2_score(
                    baseline_predictions[
                        (baseline_predictions["target"] == "chla")
                        & (baseline_predictions["depth"] == "bottom")
                        & (baseline_predictions["split"] == "validation")
                    ]["y_true"],
                    baseline_predictions[
                        (baseline_predictions["target"] == "chla")
                        & (baseline_predictions["depth"] == "bottom")
                        & (baseline_predictions["split"] == "validation")
                    ]["y_pred"],
                )
            ),
            "current_best_rmse": rmse(
                baseline_bottom["y_true"].to_numpy(),
                baseline_bottom["y_pred"].to_numpy(),
            ),
            "candidate_rmse": rmse(
                mech_pred[mech_pred["split"] == "test"]["y_true"].to_numpy(),
                mech_pred[mech_pred["split"] == "test"]["y_pred"].to_numpy(),
            ),
            "delta_rmse": rmse(
                mech_pred[mech_pred["split"] == "test"]["y_true"].to_numpy(),
                mech_pred[mech_pred["split"] == "test"]["y_pred"].to_numpy(),
            )
            - rmse(baseline_bottom["y_true"].to_numpy(), baseline_bottom["y_pred"].to_numpy()),
            "current_best_mae": float(
                mean_absolute_error(
                    baseline_bottom["y_true"].to_numpy(),
                    baseline_bottom["y_pred"].to_numpy(),
                )
            ),
            "candidate_mae": float(
                mean_absolute_error(
                    mech_pred[mech_pred["split"] == "test"]["y_true"].to_numpy(),
                    mech_pred[mech_pred["split"] == "test"]["y_pred"].to_numpy(),
                )
            ),
            "delta_mae": float(
                mean_absolute_error(
                    mech_pred[mech_pred["split"] == "test"]["y_true"].to_numpy(),
                    mech_pred[mech_pred["split"] == "test"]["y_pred"].to_numpy(),
                )
                - mean_absolute_error(
                    baseline_bottom["y_true"].to_numpy(),
                    baseline_bottom["y_pred"].to_numpy(),
                )
            ),
            "current_best_p90_rmse": high_value_metrics(
                baseline_bottom["y_true"].to_numpy(),
                baseline_bottom["y_pred"].to_numpy(),
                train[target_col].to_numpy(),
            )["p90_rmse"],
            "candidate_p90_rmse": high_value_metrics(
                mech_pred[mech_pred["split"] == "test"]["y_true"].to_numpy(),
                mech_pred[mech_pred["split"] == "test"]["y_pred"].to_numpy(),
                train[target_col].to_numpy(),
            )["p90_rmse"],
            "delta_p90_rmse": high_value_metrics(
                mech_pred[mech_pred["split"] == "test"]["y_true"].to_numpy(),
                mech_pred[mech_pred["split"] == "test"]["y_pred"].to_numpy(),
                train[target_col].to_numpy(),
            )["p90_rmse"]
            - high_value_metrics(
                baseline_bottom["y_true"].to_numpy(),
                baseline_bottom["y_pred"].to_numpy(),
                train[target_col].to_numpy(),
            )["p90_rmse"],
            "feature_count": len(mech_result.features),
            "station_improved_ratio": float(mech_station["improved"].mean()) if len(mech_station) else 0.0,
        }
        mech_comparison_df = pd.DataFrame([mech_comparison])
        mech_metrics.to_csv(
            CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_ab_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mech_pred.to_csv(
            CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mech_importance.to_csv(
            CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_feature_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mech_station.to_csv(
            CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_station_level_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
        scenario_prediction_tables["bottom_mechanism"] = mech_pred
        scenario_comparisons["bottom_mechanism"] = mech_comparison_df
        mech_manifest = {
            "scenario": "bottom_mechanism",
            "models": {
                "chla::bottom": {
                    "candidate_name": mech_result.candidate_name,
                    "model_type": mech_result.model_type,
                    "params": mech_result.params,
                    "features": mech_result.features,
                    "feature_count": len(mech_result.features),
                }
            },
        }
    else:
        pd.DataFrame().to_csv(
            CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_ab_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame().to_csv(
            CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame().to_csv(
            CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_feature_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame().to_csv(
            CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_station_level_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )

    scorecard, registry = update_scorecard_and_registry(scenario_comparisons)
    station_cmp, station_pred = build_station_level_outputs(
        baseline_predictions,
        scenario_prediction_tables,
        scorecard,
    )

    config = {
        "base_dataset": str(BASE_DATASET.relative_to(PROJECT_ROOT)),
        "highres_patch_dataset": str(HIGHRES_DATASET.relative_to(PROJECT_ROOT)),
        "bottom_mechanism_dataset": str(BOTTOM_MECH_TABLE.relative_to(PROJECT_ROOT)),
        "time_splits": TIME_SPLITS,
        "highres_feature_groups": highres_feature_groups,
        "mechanism_feature_count": len(mechanism_features),
        "scenario_manifests": scenario_manifests,
        "mechanism_manifest": mech_manifest,
        "selection_rules": {
            "mainline_update": "整体 test R² 提升 >= 0.03 且 validation 同步提升",
            "station_specific_auxiliary": "整体提升不足 0.03，但 improved station 比例 >= 60%",
            "high_value_auxiliary": "整体 R² 未提升，但 P90 RMSE 更低",
            "suspicious": "test 提升但 validation 未提升",
            "retain_current_best": "没有达到主线替代条件",
        },
    }
    (CURRENT_CONFIG_DIR / "current_highres_patch_ab.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    update_cleanup_log(
        [
            "## 本轮 current 高分 patch / bottom mechanism A/B",
            "",
            "- 更新 `experiments/outputs/current/s2_patch_ab_metrics.csv`、`s2_patch_predictions.csv`、`s2_patch_feature_importance.csv`、`s2_patch_station_level_comparison.csv`。",
            "- 更新 `experiments/outputs/current/landsat_patch_ab_metrics.csv`、`landsat_patch_predictions.csv`、`landsat_patch_feature_importance.csv`、`landsat_patch_station_level_comparison.csv`。",
            "- 更新 `experiments/outputs/current/highres_patch_ab_metrics.csv`、`highres_patch_predictions.csv`、`highres_patch_feature_importance.csv`、`highres_patch_station_level_comparison.csv`。",
            "- 更新 `experiments/outputs/current/chla_bottom_mechanism_ab_metrics.csv` 及其逐站点比较输出。",
            "- 更新 `experiments/outputs/current/station_level_prediction_comparison.csv`、`station_level_predictions.csv`。",
            "- 按最新规则重写 `current_best_scorecard.csv`，避免把小幅 uplift 直接写成主线刷新。",
        ]
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "s2_metrics_rows": int(len(pd.read_csv(CURRENT_OUTPUT_DIR / "s2_patch_ab_metrics.csv"))),
                "landsat_metrics_rows": int(len(pd.read_csv(CURRENT_OUTPUT_DIR / "landsat_patch_ab_metrics.csv"))),
                "highres_metrics_rows": int(len(pd.read_csv(CURRENT_OUTPUT_DIR / "highres_patch_ab_metrics.csv"))),
                "mechanism_metrics_rows": int(
                    len(pd.read_csv(CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_ab_metrics.csv"))
                ),
                "scorecard_rows": int(len(scorecard)),
                "station_rows": int(len(station_cmp)),
                "candidate_registry_rows": int(len(registry)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
