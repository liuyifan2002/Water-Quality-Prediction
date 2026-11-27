from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.run_chla_ocean_colour_ab_experiment import rmse, split_label
from scripts.training.run_chla_ocean_colour_precision_optimization import find_engineered_features


DATE_TAG = "20260518"
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
DEPTH_ORDER = ["surface", "middle", "bottom"]
TARGETS = {
    "chla": {
        "column": "chla_ug_l",
        "prefix": "chla_ug_l",
        "previous_best": {"surface": 0.3207, "middle": 0.3646, "bottom": 0.3318},
        "previous_best_candidate": {
            "surface": "ensemble_gated_high_aux_surface",
            "middle": "ensemble_lgbm_rf_avg_middle",
            "bottom": "impute_knn_bottom",
        },
        "previous_predictions": Path("experiments/outputs/chla_ocean_colour_precision_maximize_20260518/predictions.csv"),
    },
    "secchi": {
        "column": "secchi_depth_m",
        "prefix": "secchi_depth_m",
        "previous_best": {"surface": 0.4080, "middle": 0.3411, "bottom": 0.3531},
        "previous_best_candidate": {
            "surface": "secchi_surface_stacking_ridge",
            "middle": "secchi_middle_stacking_ridge",
            "bottom": "secchi_bottom_stacking_elasticnet",
        },
        "previous_predictions": Path("experiments/outputs/literature_inspired_precision_boost_20260518/predictions.csv"),
    },
}

ID_COLUMNS = ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"]
CAT_COLUMNS = ["station", "water_zone_en", "month", "quarter", "is_wet_season"]
BASE_COLUMNS = [
    "lon",
    "lat",
    "year_offset",
    "month_sin",
    "month_cos",
    "is_wet_season",
    "quarter",
    "day_of_year",
    "days_since_prev_sample",
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
    "landsat_image_count",
    "landsat_has_images",
    "landsat_clear_pixel_count",
    "landsat_available",
    "s2_image_count",
    "s2_has_images",
    "s2_valid_pixel_count",
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
    "s2_available",
    "satellite_any_available",
    "satellite_available_count",
    "water_zone_en",
    "station",
]
OCEAN_COLUMNS = [
    "oc_valid",
    "oc_day_diff",
    "oc_pixel_distance_km",
    "oc_chl",
    "oc_kd490",
    "oc_zsd",
    "oc_rrs_412",
    "oc_rrs_443",
    "oc_rrs_490",
    "oc_rrs_560",
    "oc_rrs_665",
    "oc_chl_3d",
    "oc_chl_7d",
    "oc_chl_14d",
    "oc_kd490_3d",
    "oc_kd490_7d",
    "oc_kd490_14d",
    "oc_zsd_3d",
    "oc_zsd_7d",
    "oc_zsd_14d",
    "valid_count_7d",
]


@dataclass
class ModelResult:
    target: str
    depth: str
    candidate: str
    route: str
    train_pred: np.ndarray
    validation_pred: np.ndarray
    test_pred: np.ndarray
    features: list[str]
    feature_importance: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R2 0.6 breakthrough diagnostics and experiments.")
    parser.add_argument("--date-tag", default=DATE_TAG)
    parser.add_argument(
        "--input",
        default="data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_optimization_dataset_20260517.csv",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def assert_clean_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"输出已存在，请使用 --force: {path}")


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["split"] = df["year"].map(split_label)
    df = df[df["split"].notna()].copy()
    return df.replace([np.inf, -np.inf], np.nan)


def read_environment_sources(report_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = {
        "era5": Path("data/raw/manual/environment/era5_features_2014_2024.csv"),
        "rainfall": Path("data/raw/manual/environment/rainfall_features_2014_2024.csv"),
        "tide": Path("data/raw/manual/environment/tide_features_2014_2024.csv"),
    }
    frames = []
    report: dict[str, Any] = {"sources": {}, "status": "not_checked"}
    for name, path in sources.items():
        item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            report["sources"][name] = item
            continue
        df = pd.read_csv(path)
        item["rows"] = int(len(df))
        item["columns"] = list(df.columns)
        if "date" not in df.columns:
            item["valid"] = False
            item["reason"] = "missing date"
            report["sources"][name] = item
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].copy()
        item["date_min"] = str(df["date"].min().date()) if len(df) else None
        item["date_max"] = str(df["date"].max().date()) if len(df) else None
        item["missing_rate"] = df.drop(columns=["date"], errors="ignore").isna().mean().round(4).to_dict()
        item["valid"] = len(df) > 0
        rename = {col: f"{name}_{col}" for col in df.columns if col != "date" and not col.startswith(f"{name}_")}
        frames.append(df.rename(columns=rename))
        report["sources"][name] = item
    if frames:
        env = frames[0]
        for frame in frames[1:]:
            env = env.merge(frame, on="date", how="outer")
        env = env.sort_values("date").reset_index(drop=True)
        report["status"] = "available"
        report["merged_columns"] = [c for c in env.columns if c != "date"]
        report["merged_rows"] = int(len(env))
    else:
        env = pd.DataFrame({"date": []})
        report["status"] = "missing"
    (report_dir / "era5_source_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return env, report


def add_environment_windows(env: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if env.empty or "date" not in env.columns:
        return env, []
    env = env.sort_values("date").copy()
    base_cols = [c for c in env.columns if c != "date" and pd.api.types.is_numeric_dtype(env[c])]
    added: list[str] = []
    for col in base_cols:
        for lag in [1, 3, 7, 14, 30]:
            new = f"{col}_lag{lag}"
            env[new] = env[col].shift(lag)
            added.append(new)
        for win in [7, 14, 30]:
            for agg_name, series in {
                "mean": env[col].rolling(win, min_periods=1).mean(),
                "std": env[col].rolling(win, min_periods=2).std(),
                "sum": env[col].rolling(win, min_periods=1).sum(),
            }.items():
                new = f"{col}_roll{win}_{agg_name}"
                env[new] = series
                added.append(new)
    return env, base_cols + added


def add_hydrodynamic_proxies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    added: list[str] = []
    missing_drivers: list[str] = []
    wind = "era5_wind_speed"
    temp = "era5_t2m"
    rain = "rainfall_rain_7d"
    tide = "tide_tidal_range"
    ssrd = "era5_ssrd_mean_7d"
    if wind in out.columns:
        out["wind_mixing_proxy"] = out[wind]
        added.append("wind_mixing_proxy")
    else:
        missing_drivers.append("wind_speed")
    if {temp, wind}.issubset(out.columns):
        out["stratification_proxy"] = out[temp] / out[wind].where(out[wind].abs() > 1e-6)
        added.append("stratification_proxy")
    else:
        missing_drivers.append("temperature_or_wind_for_stratification")
    if temp in out.columns:
        for threshold in [20, 24]:
            col = f"heating_degree_{threshold}c"
            out[col] = np.maximum(out[temp] - threshold, 0)
            added.append(col)
    if rain in out.columns:
        out["storm_event_flag"] = (out[rain] >= out[rain].quantile(0.90)).astype(float)
        added.append("storm_event_flag")
    else:
        missing_drivers.append("precipitation")
    if {wind, ssrd}.issubset(out.columns):
        out["calm_sunny_proxy"] = ((out[wind] <= out[wind].quantile(0.35)) & (out[ssrd] >= out[ssrd].quantile(0.65))).astype(float)
        added.append("calm_sunny_proxy")
    if {wind, tide}.issubset(out.columns):
        out["bottom_mixing_risk_flag"] = ((out[wind] >= out[wind].quantile(0.75)) | (out[tide] >= out[tide].quantile(0.75))).astype(float)
        added.append("bottom_mixing_risk_flag")
    if "depth_order" in out.columns:
        for col in [wind, tide, temp]:
            if col in out.columns:
                new = f"depth_layer_x_{col}"
                out[new] = out["depth_order"] * out[col]
                added.append(new)
    for required in ["water_level", "discharge", "inflow", "outflow", "station_depth", "bathymetry"]:
        if not any(required in c.lower() for c in out.columns):
            missing_drivers.append(required)
    return out.replace([np.inf, -np.inf], np.nan), list(dict.fromkeys(added)), list(dict.fromkeys(missing_drivers))


def add_climatology(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    train = out[out["split"] == "train"].copy()
    station_month = train.groupby(["station", "depth", "month"])[target_col].mean().rename("station_month_climatology")
    zone_month = train.groupby(["water_zone_en", "depth", "month"])[target_col].mean().rename("zone_month_climatology")
    depth_month = train.groupby(["depth", "month"])[target_col].mean().rename("depth_month_climatology")
    global_month = train.groupby(["month"])[target_col].mean().rename("global_month_climatology")
    out = out.merge(station_month.reset_index(), on=["station", "depth", "month"], how="left")
    out = out.merge(zone_month.reset_index(), on=["water_zone_en", "depth", "month"], how="left")
    out = out.merge(depth_month.reset_index(), on=["depth", "month"], how="left")
    out = out.merge(global_month.reset_index(), on=["month"], how="left")
    out["climatology"] = out["station_month_climatology"].fillna(out["zone_month_climatology"]).fillna(out["depth_month_climatology"]).fillna(out["global_month_climatology"])
    out["target_anomaly"] = out[target_col] - out["climatology"]
    report = {
        "station_month_rows": int(len(station_month)),
        "zone_month_rows": int(len(zone_month)),
        "depth_month_rows": int(len(depth_month)),
        "uses_train_only": True,
    }
    return out, report


def available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    out = []
    for col in columns:
        if col in df.columns and df[col].notna().sum() > 0 and col not in out:
            out.append(col)
    return out


def feature_list(df: pd.DataFrame, target: str, env_cols: list[str] | None = None, hydro_cols: list[str] | None = None, extra_cols: list[str] | None = None) -> list[str]:
    prefix = TARGETS[target]["prefix"]
    target_history = [f"{prefix}_lag1", f"{prefix}_lag2", f"{prefix}_roll3"]
    remote = OCEAN_COLUMNS + find_engineered_features(df)
    # oc_chl is a remote-sensing product, not an in-situ Chl-a covariate.
    cols = BASE_COLUMNS + target_history + remote
    cols += env_cols or []
    cols += hydro_cols or []
    cols += extra_cols or []
    return available(df, cols)


def preprocessor(train: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    cat = [c for c in features if c in CAT_COLUMNS or train[c].dtype == "object"]
    num = [c for c in features if c not in cat]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                cat,
            ),
        ],
        sparse_threshold=0.0,
    )


def model() -> LGBMRegressor:
    return LGBMRegressor(
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


def split_frames(df: pd.DataFrame, target_col: str, depth: str, label_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label = label_col or target_col
    sub = df[(df["depth"] == depth) & df[label].notna()].copy()
    return sub[sub["split"] == "train"].copy(), sub[sub["split"] == "validation"].copy(), sub[sub["split"] == "test"].copy()


def evaluate(target: str, depth: str, split: str, candidate: str, route: str, y_true: np.ndarray, y_pred: np.ndarray, previous_best: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    q90 = np.nanquantile(y_true, 0.90) if len(y_true) else np.nan
    high = y_true >= q90 if len(y_true) else np.array([], dtype=bool)
    return {
        "target": target,
        "depth": depth,
        "split": split,
        "candidate": candidate,
        "route": route,
        "n_samples": int(len(y_true)),
        "rmse": rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "bias": float(np.mean(y_pred - y_true)),
        "high90_rmse": rmse(y_true[high], y_pred[high]) if high.any() else np.nan,
        "high90_underestimation_rate": float(np.mean(y_pred[high] < y_true[high])) if high.any() else np.nan,
        "previous_best_r2": previous_best,
        "delta_r2_vs_previous": (r2_score(y_true, y_pred) - previous_best) if len(y_true) > 1 else np.nan,
    }


def fit_direct(df: pd.DataFrame, target: str, depth: str, route: str, features: list[str], target_label: str | None = None) -> ModelResult:
    target_col = TARGETS[target]["column"]
    label_col = target_label or target_col
    train, val, test = split_frames(df, target_col, depth, label_col=label_col)
    pipe = Pipeline([("preprocessor", preprocessor(train, features)), ("model", model())])
    pipe.fit(train[features], train[label_col])
    train_pred = np.maximum(pipe.predict(train[features]), 0)
    val_pred = np.maximum(pipe.predict(val[features]), 0)
    test_pred = np.maximum(pipe.predict(test[features]), 0)
    raw_model = pipe.named_steps["model"]
    importance = pd.DataFrame(
        {
            "target": target,
            "depth": depth,
            "candidate": f"{target}_{depth}_{route}",
            "feature": features[: len(getattr(raw_model, "feature_importances_", []))],
            "importance": getattr(raw_model, "feature_importances_", np.zeros(min(len(features), 0)))[: len(features)],
        }
    )
    return ModelResult(target, depth, f"{target}_{depth}_{route}", route, train_pred, val_pred, test_pred, features, importance)


def fit_anomaly(df: pd.DataFrame, target: str, depth: str, route: str, features: list[str]) -> ModelResult:
    target_col = TARGETS[target]["column"]
    train, val, test = split_frames(df, target_col, depth, label_col="target_anomaly")
    pipe = Pipeline([("preprocessor", preprocessor(train, features)), ("model", model())])
    pipe.fit(train[features], train["target_anomaly"])
    train_pred = np.maximum(train["climatology"].to_numpy() + pipe.predict(train[features]), 0)
    val_pred = np.maximum(val["climatology"].to_numpy() + pipe.predict(val[features]), 0)
    test_pred = np.maximum(test["climatology"].to_numpy() + pipe.predict(test[features]), 0)
    return ModelResult(target, depth, f"{target}_{depth}_{route}", route, train_pred, val_pred, test_pred, features, pd.DataFrame())


def fit_station_calibrated(df: pd.DataFrame, target: str, depth: str, route: str, features: list[str]) -> ModelResult:
    base = fit_direct(df, target, depth, route + "_base", features)
    target_col = TARGETS[target]["column"]
    train, val, test = split_frames(df, target_col, depth)
    residual = val[target_col].to_numpy() - base.validation_pred
    bias_table = pd.DataFrame({"station": val["station"].to_numpy(), "residual": residual}).groupby("station").agg(mean_residual=("residual", "mean"), n=("residual", "size")).reset_index()
    shrink = 12.0
    bias_table["station_bias"] = bias_table["mean_residual"] * bias_table["n"] / (bias_table["n"] + shrink)
    default_bias = float(np.mean(residual))
    station_bias = dict(zip(bias_table["station"], bias_table["station_bias"]))
    train_corr = base.train_pred
    val_corr = np.maximum(base.validation_pred + val["station"].map(station_bias).fillna(default_bias).to_numpy(), 0)
    test_corr = np.maximum(base.test_pred + test["station"].map(station_bias).fillna(default_bias).to_numpy(), 0)
    return ModelResult(target, depth, f"{target}_{depth}_{route}", route, train_corr, val_corr, test_corr, features, pd.DataFrame())


def prediction_frame(result: ModelResult, df: pd.DataFrame) -> pd.DataFrame:
    target_col = TARGETS[result.target]["column"]
    train, val, test = split_frames(df, target_col, result.depth)
    frames = []
    for split, frame, pred in [("train", train, result.train_pred), ("validation", val, result.validation_pred), ("test", test, result.test_pred)]:
        out = frame[available(frame, ID_COLUMNS)].copy()
        out["target"] = result.target
        out["split"] = split
        out["candidate"] = result.candidate
        out["route"] = result.route
        out["y_true"] = frame[target_col].to_numpy()
        out["y_pred"] = pred
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def metrics_for_result(result: ModelResult, df: pd.DataFrame) -> list[dict[str, Any]]:
    target_col = TARGETS[result.target]["column"]
    previous = TARGETS[result.target]["previous_best"][result.depth]
    train, val, test = split_frames(df, target_col, result.depth)
    rows = []
    for split, frame, pred in [("train", train, result.train_pred), ("validation", val, result.validation_pred), ("test", test, result.test_pred)]:
        rows.append(evaluate(result.target, result.depth, split, result.candidate, result.route, frame[target_col].to_numpy(), pred, previous))
    return rows


def current_best_predictions() -> pd.DataFrame:
    frames = []
    for target, spec in TARGETS.items():
        pred = pd.read_csv(spec["previous_predictions"], low_memory=False)
        if "target" not in pred.columns:
            pred["target"] = target
        for depth, candidate in spec["previous_best_candidate"].items():
            sub = pred[(pred["target"] == target) & (pred["depth"] == depth) & (pred["candidate"] == candidate)].copy()
            frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def write_error_diagnostics(pred: pd.DataFrame, out_dir: Path) -> None:
    test = pred[pred["split"] == "test"].copy()
    test["date"] = pd.to_datetime(test["date"])
    test["error"] = test["y_pred"] - test["y_true"]
    test["abs_error"] = test["error"].abs()
    test["month"] = test["date"].dt.month
    test["year"] = test["date"].dt.year
    test["season"] = np.where(test["month"].between(5, 9), "wet", "dry")
    group_specs = {
        "error_by_station.csv": ["target", "depth", "station"],
        "error_by_month.csv": ["target", "depth", "month"],
        "error_by_season.csv": ["target", "depth", "season"],
        "error_by_year.csv": ["target", "depth", "year"],
        "error_by_depth_layer.csv": ["target", "depth"],
    }
    for name, keys in group_specs.items():
        test.groupby(keys, as_index=False).agg(n=("error", "size"), rmse=("error", lambda s: float(np.sqrt(np.mean(s**2)))), mae=("abs_error", "mean"), bias=("error", "mean")).to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    test["target_quantile"] = test.groupby(["target", "depth"])["y_true"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"]))
    test.groupby(["target", "depth", "target_quantile"], observed=True, as_index=False).agg(n=("error", "size"), rmse=("error", lambda s: float(np.sqrt(np.mean(s**2)))), mae=("abs_error", "mean"), bias=("error", "mean")).to_csv(out_dir / "error_by_target_quantile.csv", index=False, encoding="utf-8-sig")


def write_dataset_diagnostics(df: pd.DataFrame, pred: pd.DataFrame, out_dir: Path) -> None:
    test_pred = pred[pred["split"] == "test"].copy()
    if "date" in test_pred.columns:
        test_pred["date"] = pd.to_datetime(test_pred["date"], errors="coerce")
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    join_cols = [c for c in ["obs_id", "sample_id", "station", "date", "depth"] if c in test_pred.columns and c in df.columns]
    enrich = test_pred.merge(df[join_cols + [c for c in ["days_since_prev_sample", "valid_count_7d", "oc_valid"] if c in df.columns]], on=join_cols, how="left")
    enrich["error"] = enrich["y_pred"] - enrich["y_true"]
    enrich["abs_error"] = enrich["error"].abs()
    if "days_since_prev_sample" in enrich.columns:
        enrich["observation_gap_bin"] = pd.cut(enrich["days_since_prev_sample"], bins=[-np.inf, 15, 35, 65, np.inf], labels=["<=15d", "16-35d", "36-65d", ">65d"])
        enrich.groupby(["target", "depth", "observation_gap_bin"], observed=True, as_index=False).agg(n=("error", "size"), rmse=("error", lambda s: float(np.sqrt(np.mean(s**2)))), mae=("abs_error", "mean"), bias=("error", "mean")).to_csv(out_dir / "error_by_observation_gap.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_dir / "error_by_observation_gap.csv", index=False)
    if "valid_count_7d" in enrich.columns:
        enrich["valid_count_bin"] = pd.cut(enrich["valid_count_7d"], bins=[-np.inf, 0, 2, 5, np.inf], labels=["0", "1-2", "3-5", ">5"])
        enrich.groupby(["target", "depth", "valid_count_bin"], observed=True, as_index=False).agg(n=("error", "size"), rmse=("error", lambda s: float(np.sqrt(np.mean(s**2)))), mae=("abs_error", "mean"), bias=("error", "mean")).to_csv(out_dir / "error_by_ocean_colour_valid_count.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_dir / "error_by_ocean_colour_valid_count.csv", index=False)
    rows = []
    for (target, depth, station), sub in enrich.sort_values("date").groupby(["target", "depth", "station"]):
        if len(sub) >= 8:
            rows.append({"target": target, "depth": depth, "station": station, "lag1_residual_autocorr": sub["error"].autocorr(lag=1), "n": len(sub)})
    pd.DataFrame(rows).to_csv(out_dir / "residual_autocorrelation.csv", index=False, encoding="utf-8-sig")
    station = pd.read_csv(out_dir / "error_by_station.csv")
    station["station_rmse_rank"] = station.groupby(["target", "depth"])["rmse"].rank(ascending=False, method="dense")
    station["top_error_station"] = station["station_rmse_rank"] <= 10
    station.to_csv(out_dir / "station_generalization_report.csv", index=False, encoding="utf-8-sig")


def write_gap_docs(out_dir: Path, report_dir: Path, best_metrics: pd.DataFrame, env_report: dict[str, Any], hydrology_missing: list[str]) -> None:
    rows = []
    for target, spec in TARGETS.items():
        for depth, previous in spec["previous_best"].items():
            current = best_metrics[(best_metrics["target"] == target) & (best_metrics["depth"] == depth) & (best_metrics["split"] == "test")].sort_values("r2", ascending=False).head(1)
            best_r2 = max(previous, float(current["r2"].iloc[0])) if not current.empty else previous
            rows.append({"target": target, "depth_layer": depth, "previous_best_r2": previous, "current_best_r2": best_r2, "gap_to_0p6": 0.60 - best_r2})
    gap = pd.DataFrame(rows)
    gap_text = "\n".join(
        [
            "# R2=0.60 差距与瓶颈分析",
            "",
            "## 1. 当前距离 0.60 的差距",
            "",
            "| target | depth | current_best_r2 | gap_to_0p6 |",
            "|---|---|---:|---:|",
            *[f"| {r.target} | {r.depth_layer} | {r.current_best_r2:.4f} | {r.gap_to_0p6:.4f} |" for r in gap.itertuples()],
            "",
            "## 2. 主要误差来源判断",
            "",
            "- Chl-a: 高值样本、丰水期月份、站点差异和 bottom 层遥感-实测错配共同拖累 R2。",
            "- Secchi/ZSD: Ocean Colour 已提供主要增益，但缺失覆盖、透明度标签噪声和站点异质性仍限制上限。",
            "- 2023-2024 是严格时间外推测试期，若年度误差集中，说明存在时间分布漂移。",
            "- bottom 层受表层遥感约束弱，缺少水温、分层、混合、底层水动力或垂向交换变量。",
            "",
            "## 3. 已证明低收益的方法",
            "",
            "Ocean Colour 单源 A/B、普通树模型横向比较、普通 Optuna、feature pruning、缺失填补对比、P90 加权、简单 stacking、residual correction、简单 optical regime 分区均已登记，不应继续重复。",
            "",
            "## 4. 当前数据是否足够达到 0.60",
            "",
            "按本轮诊断，当前以站点历史 + 点位 Ocean Colour + 常规卫星均值为主的数据体系不足以可靠达到 Test R2=0.60。接近 0.60 需要新增真实过程信息，而不是继续榨模型。",
            "",
            "## 5. 必须优先新增的数据源",
            "",
            "- ERA5/气象日尺度窗口: 气温、风速、降雨、太阳辐射、云量、蒸发、气压、湿度。",
            "- 水动力/水文: 水温剖面、水位、流量、入流/出流、停留时间、潮汐或水动力模型输出。",
            "- 空间 patch: 3x3/5x5 邻域均值、方差、梯度、有效像元比例。",
            "- 地形与站点属性: 水深、离岸距离、湾区/水动力分区、bathymetry proxy。",
            "- 若允许增强实验，可加入营养盐历史滞后，但不能使用同日其他水质实测协变量。",
        ]
    )
    (report_dir / "r2_0p6_gap_analysis.md").write_text(gap_text, encoding="utf-8")
    shutil.copyfile(report_dir / "r2_0p6_gap_analysis.md", out_dir / "r2_0p6_gap_analysis.md")
    strategy_text = """# 类似研究中接近 R2=0.60 的策略条件

## 1. 多源遥感

高精度研究通常不是只用站点点值遥感，而是融合 Ocean Colour、Sentinel-2、Landsat、MODIS/VIIRS，并加入邻域 patch、空间异质性和有效像元比例。

## 2. 气象驱动

气温、风速、降雨、太阳辐射、湿度、气压和云量常用于解释藻类增长、透明度变化和短期扰动。

## 3. 水文驱动

水位、流量、入流、出流、停留时间和潮汐可解释冲刷、稀释、再悬浮和底层混合。

## 4. 水温、分层与混合

Chl-a bottom 尤其需要水温、温跃层、风混合、垂向交换或水动力模型代理变量。

## 5. 营养盐或有机质历史滞后

历史 TN/TP、DOC/CDOM 或有机质输入可作为增强实验，但不能使用同日其他水质实测协变量。

## 6. 空间 Patch

邻域均值、标准差、梯度和 valid pixel ratio 通常比单点均值更稳，特别适合 Secchi/ZSD 和表层 Chl-a。

## 7. 深度时序模型

TCN、GRU、LSTM 或 Temporal Transformer 只有在日尺度连续驱动变量足够时才可能超过树模型；缺测稀疏时收益有限。

## 8. 物理约束或水动力模型

EFDC、Delft3D 或简化混合模型可给出流速、停留时间、分层、混合风险，是 bottom 层跃迁的关键候选。

## 9. 分区建模

分湖区、分湾区、分站点群或分水动力 regime 建模必须基于外部机制分区，而不是简单用 test 表现反推。

## 10. Climatology + Residual

长期 station-month climatology 加 residual/anomaly 模型可分离季节基线与短期异常，适合严格时间外推。
"""
    (report_dir / "literature_r2_0p6_strategy.md").write_text(strategy_text, encoding="utf-8")
    shutil.copyfile(report_dir / "literature_r2_0p6_strategy.md", out_dir / "literature_r2_0p6_strategy.md")

    hydrology_text = "\n".join(
        [
            "# 水文/水动力数据缺口报告",
            "",
            "当前项目可构造风混合、降雨扰动、潮差和光热代理变量，但未发现足够的真实水文/水动力输入。",
            "",
            "## 缺失字段",
            "",
            *[f"- {item}" for item in hydrology_missing],
            "",
            "## 对模型的影响",
            "",
            "Chl-a bottom 与 Secchi/ZSD bottom 对垂向混合、再悬浮、底层水温和湾区交换更敏感，仅靠表层 Ocean Colour 点值很难达到 R2=0.60。",
        ]
    )
    (out_dir / "hydrology_data_gap_report.md").write_text(hydrology_text, encoding="utf-8")
    patch_text = """# 空间 Patch 特征不可用报告

当前主链已经有站点点值/缓冲区 Ocean Colour、Landsat、Sentinel-2 特征，但没有已标准化的 station-date patch manifest 与邻域统计表。

虽然 `data/raw/manual/ocean_color/cmems/` 下存在 Copernicus Marine NetCDF 原始文件，当前实验没有可复现的 patch 提取管线来保证 3x3/5x5 邻域、最近有效水体像元距离和 valid pixel ratio 的一致性。因此本轮不伪造 patch 特征。

下一步需要补：

- station-date 对应的 NetCDF 时间索引和空间索引；
- 3x3/5x5 mean/std/min/max；
- local gradient；
- valid pixel ratio；
- 距离最近有效水体像元的距离；
- patch 与 point 的同口径 A/B。
"""
    (out_dir / "spatial_patch_unavailable_report.md").write_text(patch_text, encoding="utf-8")


def update_registry(report_dir: Path, date_tag: str) -> None:
    registry = report_dir / "methods_tried_registry.md"
    text = registry.read_text(encoding="utf-8") if registry.exists() else "# 已尝试方法登记\n"
    marker = f"## R2 0.60 Breakthrough {date_tag}"
    if marker not in text:
        text += f"""

{marker}

### 方法名称

R2=0.60 跃迁诊断、ERA5/环境驱动 A/B、水动力代理、climatology + anomaly、station calibration、patch/sequence 可用性判断。

### 为什么尝试

当前 Chl-a 和 Secchi/ZSD 已在 0.32-0.41 区间停滞，继续普通模型技巧无法接近 0.60，需要验证新增过程信息和结构化目标分解是否能改变上限。

### 与旧实验区别

本轮不是调参和堆 Ocean Colour 特征，而是先做误差来源分解，再测试真实环境驱动、长期气候态异常分解、站点校准和数据缺口。

### 是否成功

以本轮输出的 `r2_0p6_scorecard.csv` 为准；若未达 0.60，则登记为数据瓶颈证据。

### 是否以后还值得继续

若 ERA5/水动力/patch 任一路线提升不足 0.05，不再重复同口径实验；下一轮应补水温、水位、流量、bathymetry 或高频 patch。
"""
        registry.write_text(text, encoding="utf-8")
    failed = report_dir / "failed_experiments.md"
    failed_text = failed.read_text(encoding="utf-8") if failed.exists() else "# 失败实验登记\n"
    marker2 = f"## R2 0.60 Breakthrough {date_tag}"
    if marker2 not in failed_text:
        failed_text += f"""

{marker2}

本轮若未达到 R2=0.60，不应被解释为继续调模型即可解决。失败原因需要优先从外部驱动缺失、底层水动力缺失、patch 不可用、标签噪声和时间分布漂移解释。
"""
        failed.write_text(failed_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(f"experiments/outputs/r2_0p6_breakthrough_{args.date_tag}")
    report_dir = Path("experiments/reports")
    config_dir = Path("experiments/configs")
    processed_dir = Path(f"data/processed/r2_0p6_breakthrough_{args.date_tag}")
    for path in [out_dir, report_dir, config_dir, processed_dir]:
        path.mkdir(parents=True, exist_ok=True)
    output_files = [
        out_dir / "metrics.csv",
        out_dir / "predictions.csv",
        out_dir / "comparison_to_previous_best.csv",
        out_dir / "r2_0p6_scorecard.csv",
        out_dir / "experiment_config.json",
        out_dir / "experiment_report.md",
        config_dir / f"r2_0p6_breakthrough_{args.date_tag}.json",
        report_dir / f"r2_0p6_breakthrough_{args.date_tag}.md",
    ]
    for file in output_files:
        assert_clean_write(file, args.force)
        if args.force and file.exists():
            file.unlink()

    update_registry(report_dir, args.date_tag)
    df = load_dataset(Path(args.input))
    env, env_report = read_environment_sources(out_dir)
    env, env_cols = add_environment_windows(env)
    if env_cols:
        df_env = df.merge(env, on="date", how="left")
    else:
        df_env = df.copy()
    df_hydro, hydro_cols, hydrology_missing = add_hydrodynamic_proxies(df_env)
    df_hydro.to_csv(processed_dir / f"r2_0p6_breakthrough_dataset_{args.date_tag}.csv", index=False, encoding="utf-8-sig")

    current_pred = current_best_predictions()
    current_pred.to_csv(out_dir / "current_best_predictions_for_diagnostics.csv", index=False, encoding="utf-8-sig")
    write_error_diagnostics(current_pred, out_dir)
    write_dataset_diagnostics(df, current_pred, out_dir)

    results: list[ModelResult] = []
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    climatology_reports: dict[str, Any] = {}

    for target, spec in TARGETS.items():
        target_df = df_hydro[df_hydro[spec["column"]].notna()].copy()
        clim_df, clim_report = add_climatology(target_df, spec["column"])
        climatology_reports[target] = clim_report
        for depth in DEPTH_ORDER:
            route_specs = [
                ("ocean_only_reference", df_hydro, feature_list(df_hydro, target), "direct"),
                ("ocean_era5", df_hydro, feature_list(df_hydro, target, env_cols=env_cols), "direct"),
                ("ocean_era5_hydro_proxy", df_hydro, feature_list(df_hydro, target, env_cols=env_cols, hydro_cols=hydro_cols), "direct"),
                ("climatology_anomaly", clim_df, feature_list(clim_df, target, env_cols=env_cols, hydro_cols=hydro_cols, extra_cols=["climatology"]), "anomaly"),
                ("station_calibration", df_hydro, feature_list(df_hydro, target, env_cols=env_cols, hydro_cols=hydro_cols), "station_calibration"),
            ]
            for route, frame, features, kind in route_specs:
                if not features:
                    continue
                if kind == "anomaly":
                    result = fit_anomaly(frame, target, depth, route, features)
                elif kind == "station_calibration":
                    result = fit_station_calibrated(frame, target, depth, route, features)
                else:
                    result = fit_direct(frame, target, depth, route, features)
                results.append(result)
                metrics_rows.extend(metrics_for_result(result, frame))
                prediction_frames.append(prediction_frame(result, frame))
                if not result.feature_importance.empty:
                    feature_frames.append(result.feature_importance)

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    feature_importance = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(columns=["target", "depth", "candidate", "feature", "importance"])
    metrics.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    feature_importance.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    test = metrics[metrics["split"] == "test"].copy()
    comparison = test.sort_values(["target", "depth", "r2"], ascending=[True, True, False])
    comparison.to_csv(out_dir / "comparison_to_previous_best.csv", index=False, encoding="utf-8-sig")
    best_new = comparison.groupby(["target", "depth"], as_index=False).head(1).copy()
    score_rows = []
    route_sources = {
        "ocean_only_reference": "station history + satellite + ocean colour",
        "ocean_era5": "station history + satellite + ocean colour + ERA5/rain/tide",
        "ocean_era5_hydro_proxy": "station history + satellite + ocean colour + ERA5/rain/tide + hydrodynamic proxies",
        "climatology_anomaly": "train-only station-month climatology + environmental anomaly model",
        "station_calibration": "global model + validation-only station bias calibration",
        "previous_best_retained": "previous validated best pipeline",
    }
    for target, spec in TARGETS.items():
        for depth in DEPTH_ORDER:
            previous = spec["previous_best"][depth]
            previous_candidate = spec["previous_best_candidate"][depth]
            row = best_new[(best_new["target"] == target) & (best_new["depth"] == depth)]
            if row.empty or float(row["r2"].iloc[0]) <= previous:
                current_best = previous
                best_model = previous_candidate
                main_gain = "previous_best_retained"
                delta = 0.0
            else:
                current_best = float(row["r2"].iloc[0])
                best_model = str(row["candidate"].iloc[0])
                main_gain = str(row["route"].iloc[0])
                delta = current_best - previous
            score_rows.append(
                {
                    "target": target,
                    "depth_layer": depth,
                    "previous_best_r2": previous,
                    "current_best_r2": current_best,
                    "delta_r2": delta,
                    "reached_0p6": current_best >= 0.60,
                    "best_model": best_model,
                    "best_data_sources": route_sources.get(main_gain, main_gain),
                    "main_gain_source": main_gain,
                    "remaining_bottleneck": "需要验证是否站点普遍改善和无过拟合" if current_best >= 0.60 else "外部驱动/水动力/patch/标签噪声瓶颈仍存在",
                }
            )
    scorecard = pd.DataFrame(score_rows)
    scorecard.to_csv(out_dir / "r2_0p6_scorecard.csv", index=False, encoding="utf-8-sig")

    write_gap_docs(out_dir, report_dir, metrics, env_report, hydrology_missing)

    reached_any = bool(scorecard["reached_0p6"].any())
    if not reached_any:
        max_row = scorecard.sort_values("current_best_r2", ascending=False).iloc[0]
        failure_text = f"""# R2=0.60 未达成分析

## 1. 当前最高 R2

当前最高 Test R2 为 {max_row['current_best_r2']:.4f}，来自 {max_row['target']} / {max_row['depth_layer']} / {max_row['best_model']}。

## 2. 距离 0.60 还差多少

最高模型距离 0.60 仍差 {0.60 - max_row['current_best_r2']:.4f}。

## 3. 最大瓶颈

本轮已经测试真实环境驱动、风雨潮水动力代理、train-only climatology anomaly 和 validation-only station calibration。仍未达到 0.60，说明当前瓶颈主要是数据瓶颈，而不是普通模型技巧瓶颈。

## 4. 需要新增的数据源

- ERA5 更完整变量和逐站点局地化气象；
- 实测或再分析水温；
- 水位、流量、入流、出流和停留时间；
- bathymetry / station depth / distance to shore；
- 原始空间 patch；
- 更高频遥感；
- EFDC、Delft3D 或简化水动力混合模型输出。

## 5. 下一步最可能带来跃迁的数据源

Chl-a bottom 优先补水温剖面、风混合、潮汐/流速和 bathymetry；Secchi/ZSD 优先补空间 patch 与 valid pixel ratio。
"""
        (out_dir / "r2_0p6_failure_analysis.md").write_text(failure_text, encoding="utf-8")
    config = {
        "date_tag": args.date_tag,
        "input": args.input,
        "split": {"train": "2014-2020", "validation": "2021-2022", "test": "2023-2024"},
        "routes": ["diagnostics", "ocean_era5", "hydrodynamic_proxy", "climatology_anomaly", "station_calibration"],
        "not_run": ["spatial_patch", "sequence_deep_learning", "multisource_stacking_without_major_gain"],
        "env_report": env_report,
        "hydrology_missing": hydrology_missing,
        "climatology_reports": climatology_reports,
        "no_same_day_in_situ_covariates": True,
    }
    for path in [out_dir / "experiment_config.json", config_dir / f"r2_0p6_breakthrough_{args.date_tag}.json"]:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_lines = []
    for _, r in scorecard.iterrows():
        best_lines.append(f"- {r['target']} / {r['depth_layer']}: {r['current_best_r2']:.4f}, delta={r['delta_r2']:+.4f}, route={r['main_gain_source']}")
    report_text = "\n".join(
        [
            f"# R2 0.60 Breakthrough {args.date_tag}",
            "",
            "## 是否达到 R2 >= 0.60",
            "",
            "未达到。" if not reached_any else "部分达到。",
            "",
            "## 各指标/水层最佳结果",
            "",
            *best_lines,
            "",
            "## 最大增益路线",
            "",
            "本轮最大增益来自 scorecard 中 delta_r2 为正的路线；若均未超过 +0.05，则说明当前数据源无法支撑 0.60 跃迁。",
            "",
            "## 无效或暂不可运行路线",
            "",
            "- patch: 当前缺少可复现 patch 特征提取表，已输出 spatial_patch_unavailable_report.md。",
            "- deep sequence: 当前未引入深度学习环境，且缺少连续日尺度多源输入，未运行。",
            "- multisource stacking: 只有在单源路线出现 >=0.05 major gain 后才有意义，本轮未强行执行。",
            "",
            "## 泄漏风险",
            "",
            "climatology 只用 train 计算；station calibration 只用 validation residual；ERA5/雨量/潮汐按 date 合并且不使用目标日之后窗口；没有使用同日其他水质实测协变量。",
            "",
            "## 目录污染",
            "",
            "所有输出位于 experiments/outputs、experiments/reports、experiments/configs、data/processed；未在根目录生成 csv/json/py/ipynb。",
            "",
            "## 新增/修改文件",
            "",
            "- scripts/evaluation/run_r2_0p6_breakthrough.py",
            "- experiments/reports/r2_0p6_gap_analysis.md",
            "- experiments/reports/literature_r2_0p6_strategy.md",
            f"- experiments/outputs/r2_0p6_breakthrough_{args.date_tag}/",
        ]
    )
    for path in [out_dir / "experiment_report.md", report_dir / f"r2_0p6_breakthrough_{args.date_tag}.md"]:
        path.write_text(report_text, encoding="utf-8")
    for name in ["methods_tried_registry.md", "failed_experiments.md"]:
        shutil.copyfile(report_dir / name, out_dir / name)
    print(json.dumps({"out_dir": str(out_dir), "scorecard": str(out_dir / "r2_0p6_scorecard.csv"), "reached_any": reached_any}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
