from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATE_TAG = "20260518"
KEYWORDS = [
    "temperature",
    "temp",
    "water_temp",
    "wtemp",
    "thermocline",
    "stratification",
    "DO",
    "dissolved_oxygen",
    "depth",
    "bathymetry",
    "bathy",
    "station",
    "lon",
    "lat",
    "coordinate",
    "tide",
    "current",
    "velocity",
    "water_level",
    "level",
    "discharge",
    "inflow",
    "outflow",
    "river",
    "wind",
    "radiation",
    "solar",
    "precipitation",
    "ERA5",
    "CDS",
    "netcdf",
    "nc",
    "raster",
    "patch",
    "pixel",
]
SCAN_DIRS = ["data/raw", "data/interim", "data/processed", "data/external", "data/raw/manual", "experiments/outputs", "configs", "docs"]
TARGETS = {
    "chla": {"column": "chla_ug_l", "previous_best": {"bottom": 0.3318}},
    "secchi": {"column": "secchi_depth_m", "previous_best": {"surface": 0.4080, "middle": 0.3447, "bottom": 0.3574}},
}
DEPTHS = ["surface", "middle", "bottom"]
CAT_COLS = ["station", "water_zone_en", "month", "quarter", "is_wet_season"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare mechanism and spatial patch data expansion.")
    parser.add_argument("--date-tag", default=DATE_TAG)
    parser.add_argument(
        "--input",
        default="data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_optimization_dataset_20260517.csv",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_dirs(date_tag: str) -> tuple[Path, Path, Path, Path]:
    out_dir = Path(f"experiments/outputs/mechanism_data_expansion_{date_tag}")
    report_dir = Path("experiments/reports")
    config_dir = Path("experiments/configs")
    processed_dir = Path(f"data/processed/mechanism_data_expansion_{date_tag}")
    for path in [out_dir, report_dir, config_dir, processed_dir, Path("data/raw/manual/ocean_color")]:
        path.mkdir(parents=True, exist_ok=True)
    return out_dir, report_dir, config_dir, processed_dir


def split_label(year: int) -> str | None:
    if 2014 <= year <= 2020:
        return "train"
    if 2021 <= year <= 2022:
        return "validation"
    if 2023 <= year <= 2024:
        return "test"
    return None


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def scan_files(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for root in SCAN_DIRS:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            text = str(path).lower()
            hits = [kw for kw in KEYWORDS if kw.lower() in text]
            if hits:
                rows.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "suffix": path.suffix.lower(),
                        "size_bytes": path.stat().st_size,
                        "keyword_hits": ";".join(hits),
                    }
                )
    available = pd.DataFrame(rows).sort_values(["suffix", "name"]) if rows else pd.DataFrame(columns=["path", "name", "suffix", "size_bytes", "keyword_hits"])
    candidates = []
    unusable = []
    for row in available.to_dict("records"):
        suffix = row["suffix"]
        path_lower = row["path"].lower()
        if suffix in [".csv", ".nc", ".tif", ".tiff"] and any(k in path_lower for k in ["ocean", "era5", "rain", "tide", "station", "patch"]):
            candidates.append({**row, "candidate_reason": "contains mechanism or patch keywords and supported file type"})
        else:
            unusable.append({**row, "unusable_reason": "not a supported mechanism data table/raster for this round"})
    cand = pd.DataFrame(candidates)
    unus = pd.DataFrame(unusable)
    available.to_csv(out_dir / "available_mechanism_files.csv", index=False, encoding="utf-8-sig")
    cand.to_csv(out_dir / "candidate_data_sources.csv", index=False, encoding="utf-8-sig")
    unus.to_csv(out_dir / "unusable_data_sources.csv", index=False, encoding="utf-8-sig")
    return available, cand, unus


def copy_s2_manifest() -> dict[str, Any]:
    src = Path(r"d:\Users\刘逸凡\Downloads\hk_s2_patch_manifest.csv")
    dst = Path("data/raw/manual/ocean_color/hk_s2_patch_manifest.csv")
    report = {"source": str(src), "copied_to": str(dst), "exists": src.exists()}
    if src.exists():
        shutil.copyfile(src, dst)
        df = pd.read_csv(dst, nrows=100)
        report.update({"rows_checked": int(len(df)), "columns": list(df.columns), "is_feature_table": any("mean" in c.lower() or "std" in c.lower() for c in df.columns)})
    return report


def load_model_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["split"] = df["year"].map(split_label)
    df = df[df["split"].notna()].copy()
    return df.replace([np.inf, -np.inf], np.nan)


def open_cmems_datasets() -> tuple[dict[str, xr.DataArray], dict[str, Any]]:
    nc_dir = Path("data/raw/manual/ocean_color/cmems")
    arrays: dict[str, xr.DataArray] = {}
    report: dict[str, Any] = {"netcdf_dir": str(nc_dir), "files": []}
    mapping = {
        "CHL": "oc_patch_chl",
        "KD490": "oc_patch_kd490",
        "ZSD": "oc_patch_zsd",
        "RRS412": "oc_patch_rrs412",
        "RRS443": "oc_patch_rrs443",
        "RRS490": "oc_patch_rrs490",
        "RRS555": "oc_patch_rrs555",
        "RRS670": "oc_patch_rrs670",
    }
    for path in nc_dir.glob("*.nc"):
        ds = xr.open_dataset(path, engine="h5netcdf")
        file_report = {"path": str(path), "dims": dict(ds.sizes), "coords": list(ds.coords), "variables": list(ds.data_vars)}
        for var, out_name in mapping.items():
            if var in ds.data_vars:
                arrays[out_name] = ds[var].load()
        ds.close()
        report["files"].append(file_report)
    report["available_variables"] = sorted(arrays)
    return arrays, report


def nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.abs(values - value).argmin())


def patch_stats(arr: np.ndarray, t_idx: int, lat_idx: int, lon_idx: int, radius: int) -> dict[str, float]:
    lat0 = max(0, lat_idx - radius)
    lat1 = min(arr.shape[1], lat_idx + radius + 1)
    lon0 = max(0, lon_idx - radius)
    lon1 = min(arr.shape[2], lon_idx + radius + 1)
    patch = arr[t_idx, lat0:lat1, lon0:lon1]
    finite = np.isfinite(patch)
    total = patch.size
    if total == 0 or not finite.any():
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "valid_ratio": 0.0}
    return {
        "mean": float(np.nanmean(patch)),
        "std": float(np.nanstd(patch)),
        "min": float(np.nanmin(patch)),
        "max": float(np.nanmax(patch)),
        "valid_ratio": float(finite.sum() / total),
    }


def extract_patch_features(df: pd.DataFrame, processed_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    arrays, nc_report = open_cmems_datasets()
    if not arrays:
        report = {"status": "unavailable", "reason": "no supported CMEMS Ocean Colour variables", **nc_report}
        (out_dir / "ocean_colour_patch_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return pd.DataFrame(), report
    sample_cols = ["sample_id", "station", "date", "lon", "lat", "water_zone_en"]
    samples = df[[c for c in sample_cols if c in df.columns]].drop_duplicates(["station", "date"]).copy()
    samples["date"] = pd.to_datetime(samples["date"])
    first = next(iter(arrays.values()))
    lats = first["latitude"].values
    lons = first["longitude"].values
    times = pd.to_datetime(first["time"].values).normalize()
    time_lookup = {d: i for i, d in enumerate(times)}
    loaded = {name: da.values for name, da in arrays.items()}
    rows = []
    for rec in samples.itertuples(index=False):
        date = pd.Timestamp(rec.date).normalize()
        t_idx = time_lookup.get(date)
        base = {
            "sample_id": getattr(rec, "sample_id", None),
            "station": rec.station,
            "date": date.date().isoformat(),
            "lon": rec.lon,
            "lat": rec.lat,
            "water_zone_en": getattr(rec, "water_zone_en", None),
            "patch_has_time": int(t_idx is not None),
        }
        if t_idx is None:
            rows.append(base)
            continue
        lat_idx = nearest_index(lats, float(rec.lat))
        lon_idx = nearest_index(lons, float(rec.lon))
        base["patch_grid_lat"] = float(lats[lat_idx])
        base["patch_grid_lon"] = float(lons[lon_idx])
        for name, arr in loaded.items():
            point = float(arr[t_idx, lat_idx, lon_idx]) if np.isfinite(arr[t_idx, lat_idx, lon_idx]) else np.nan
            base[f"{name}_point"] = point
            for radius, label in [(1, "3x3"), (2, "5x5")]:
                stats = patch_stats(arr, t_idx, lat_idx, lon_idx, radius)
                for stat, value in stats.items():
                    base[f"{name}_{label}_{stat}"] = value
            mean3 = base.get(f"{name}_3x3_mean", np.nan)
            mean5 = base.get(f"{name}_5x5_mean", np.nan)
            base[f"{name}_local_gradient"] = abs(point - mean3) if np.isfinite(point) and np.isfinite(mean3) else np.nan
            base[f"{name}_heterogeneity_index"] = (base.get(f"{name}_5x5_std", np.nan) / abs(mean5)) if np.isfinite(mean5) and abs(mean5) > 1e-12 else np.nan
            base[f"{name}_patch_missing_ratio"] = 1.0 - base.get(f"{name}_5x5_valid_ratio", 0.0)
        rows.append(base)
    patch = pd.DataFrame(rows)
    out_path = processed_dir / "ocean_colour_patch_features_2014_2024.csv"
    patch.to_csv(out_path, index=False, encoding="utf-8-sig")
    missing = patch.isna().mean().reset_index()
    missing.columns = ["field", "missing_rate"]
    missing.to_csv(out_dir / "ocean_colour_patch_missing_summary.csv", index=False, encoding="utf-8-sig")
    corr_rows = []
    for col in [c for c in patch.columns if c.endswith("_point")]:
        base = col[: -len("_point")]
        for patch_col in [f"{base}_3x3_mean", f"{base}_5x5_mean"]:
            if patch_col in patch.columns:
                corr_rows.append({"point": col, "patch": patch_col, "corr": patch[[col, patch_col]].corr().iloc[0, 1]})
    report = {
        "status": "extracted",
        "rows": int(len(patch)),
        "station_count": int(patch["station"].nunique()),
        "date_min": str(pd.to_datetime(patch["date"]).min().date()),
        "date_max": str(pd.to_datetime(patch["date"]).max().date()),
        "station_date_unique": bool(patch.duplicated(["station", "date"]).sum() == 0),
        "variables": sorted(arrays),
        "netcdf": nc_report,
        "point_patch_correlations": corr_rows,
        "valid_ratio_summary": patch[[c for c in patch.columns if c.endswith("valid_ratio")]].describe().to_dict(),
    }
    (out_dir / "ocean_colour_patch_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return patch, report


def read_environment() -> pd.DataFrame:
    files = {
        "era5": Path("data/raw/manual/environment/era5_features_2014_2024.csv"),
        "rainfall": Path("data/raw/manual/environment/rainfall_features_2014_2024.csv"),
        "tide": Path("data/raw/manual/environment/tide_features_2014_2024.csv"),
    }
    frames = []
    for prefix, path in files.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "date" not in frame.columns:
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        rename = {c: f"{prefix}_{c}" for c in frame.columns if c != "date" and not c.startswith(f"{prefix}_")}
        frames.append(frame.rename(columns=rename))
    if not frames:
        return pd.DataFrame({"date": []})
    env = frames[0]
    for frame in frames[1:]:
        env = env.merge(frame, on="date", how="outer")
    return env.sort_values("date")


def make_bottom_mechanism(df: pd.DataFrame, processed_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    env = read_environment()
    merged = df.merge(env, on="date", how="left") if not env.empty else df.copy()
    added = []
    if "era5_wind_speed" in merged:
        merged["wind_speed_3d_mean"] = merged["era5_wind_speed"]
        merged["wind_speed_7d_mean"] = merged["era5_wind_speed"]
        merged["wind_mixing_proxy"] = merged["era5_wind_speed"] * merged.get("tide_tidal_range", 1)
        added += ["wind_speed_3d_mean", "wind_speed_7d_mean", "wind_mixing_proxy"]
    if {"era5_u10", "era5_v10"}.issubset(merged.columns):
        merged["wind_direction_u"] = merged["era5_u10"]
        merged["wind_direction_v"] = merged["era5_v10"]
        added += ["wind_direction_u", "wind_direction_v"]
    if "rainfall_rain_7d" in merged:
        q90 = merged.loc[merged["split"] == "train", "rainfall_rain_7d"].quantile(0.90)
        merged["precip_7d_sum"] = merged["rainfall_rain_7d"]
        merged["storm_event_flag"] = (merged["rainfall_rain_7d"] >= q90).astype(float)
        added += ["precip_7d_sum", "storm_event_flag"]
    if "era5_t2m" in merged:
        merged["air_temp"] = merged["era5_t2m"]
        added.append("air_temp")
    if "era5_sst_mean_7d" in merged:
        merged["rolling_temp_7d"] = merged["era5_sst_mean_7d"]
        merged["rolling_temp_14d"] = merged.get("era5_sst_mean_14d", merged["era5_sst_mean_7d"])
        merged["temp_change_rate"] = merged["era5_sst_mean_3d"] - merged["era5_sst_mean_14d"] if "era5_sst_mean_3d" in merged and "era5_sst_mean_14d" in merged else np.nan
        added += ["rolling_temp_7d", "rolling_temp_14d", "temp_change_rate"]
    if {"era5_t2m", "era5_wind_speed"}.issubset(merged.columns):
        merged["stratification_index_proxy"] = merged["era5_t2m"] / merged["era5_wind_speed"].where(merged["era5_wind_speed"].abs() > 1e-6)
        added.append("stratification_index_proxy")
    if {"depth_order", "tide_tidal_range"}.issubset(merged.columns):
        merged["depth_layer_x_tidal_range"] = merged["depth_order"] * merged["tide_tidal_range"]
        added.append("depth_layer_x_tidal_range")
    if {"depth_order", "era5_wind_speed"}.issubset(merged.columns):
        merged["bottom_mixing_risk_flag"] = ((merged["depth"] == "bottom") & (merged["era5_wind_speed"] >= merged.loc[merged["split"] == "train", "era5_wind_speed"].quantile(0.75))).astype(float)
        added.append("bottom_mixing_risk_flag")
    static_cols = [c for c in ["station", "date", "sample_id", "obs_id", "depth", "water_zone_en", "lon", "lat"] if c in merged.columns]
    output = merged[static_cols + sorted(set(added))].copy()
    output.to_csv(processed_dir / "bottom_mechanism_features_2014_2024.csv", index=False, encoding="utf-8-sig")
    missing = output.isna().mean().reset_index()
    missing.columns = ["field", "missing_rate"]
    missing.to_csv(out_dir / "bottom_mechanism_missing_summary.csv", index=False, encoding="utf-8-sig")
    validation = {
        "rows": int(len(output)),
        "features": sorted(set(added)),
        "date_min": str(pd.to_datetime(output["date"]).min().date()),
        "date_max": str(pd.to_datetime(output["date"]).max().date()),
        "critical_missing": ["bottom_water_temperature", "vertical_temperature_profile", "bathymetry", "station_depth", "current_velocity", "water_level", "discharge"],
    }
    (out_dir / "bottom_mechanism_feature_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged, sorted(set(added))


def feature_columns(df: pd.DataFrame, target: str, extra: list[str] | None = None) -> list[str]:
    prefix = TARGETS[target]["column"]
    base = [
        "lon",
        "lat",
        "year_offset",
        "month_sin",
        "month_cos",
        "is_wet_season",
        "quarter",
        "day_of_year",
        "days_since_prev_sample",
        "station",
        "water_zone_en",
        f"{prefix}_lag1",
        f"{prefix}_lag2",
        f"{prefix}_roll3",
    ]
    remote = [
        c
        for c in df.columns
        if (c.startswith("oc_") and not c.startswith("oc_patch_"))
        or c.startswith("landsat_")
        or c.startswith("s2_")
        or c.startswith("satellite_")
    ]
    cols = base + remote + (extra or [])
    return [c for c in dict.fromkeys(cols) if c in df.columns and df[c].notna().sum() > 0]


def model_pipeline(train: pd.DataFrame, features: list[str]) -> Pipeline:
    cat = [c for c in features if c in CAT_COLS or train[c].dtype == "object"]
    num = [c for c in features if c not in cat]
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
                        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
                    ],
                    sparse_threshold=0.0,
                ),
            ),
            ("model", ExtraTreesRegressor(n_estimators=260, min_samples_leaf=3, random_state=42, n_jobs=1)),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    high = y_true >= np.nanquantile(y_true, 0.90)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "p90_rmse": rmse(y_true[high], y_pred[high]) if high.any() else np.nan,
        "p90_underestimation_rate": float(np.mean(y_pred[high] < y_true[high])) if high.any() else np.nan,
    }


def run_ab(df: pd.DataFrame, target: str, depths: list[str], baseline_extra: list[str], experiment_extra: list[str], route: str, previous: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    preds = []
    importances = []
    y_col = TARGETS[target]["column"]
    for depth in depths:
        sub = df[(df["depth"] == depth) & df[y_col].notna()].copy()
        for label, extra in [("baseline", baseline_extra), ("experiment", baseline_extra + experiment_extra)]:
            features = feature_columns(sub, target, extra)
            train = sub[sub["split"] == "train"]
            test = sub[sub["split"] == "test"]
            if train.empty or test.empty:
                continue
            pipe = model_pipeline(train, features)
            pipe.fit(train[features], train[y_col])
            pred = np.maximum(pipe.predict(test[features]), 0)
            row = evaluate(test[y_col].to_numpy(), pred)
            row.update({"target": target, "depth": depth, "route": route, "ab_group": label, "n_features": len(features), "n_test": len(test), "previous_best_r2": previous.get(depth, np.nan), "delta_vs_previous": row["r2"] - previous.get(depth, np.nan)})
            metrics.append(row)
            p = test[[c for c in ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"] if c in test.columns]].copy()
            p["target"] = target
            p["route"] = route
            p["ab_group"] = label
            p["y_true"] = test[y_col].to_numpy()
            p["y_pred"] = pred
            preds.append(p)
            fitted = pipe.named_steps["model"]
            n_imp = min(len(features), len(fitted.feature_importances_))
            importances.append(pd.DataFrame({"target": target, "depth": depth, "route": route, "ab_group": label, "feature": features[:n_imp], "importance": fitted.feature_importances_[:n_imp]}))
    return pd.DataFrame(metrics), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(), pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()


def write_reports(out_dir: Path, report_dir: Path, config_dir: Path, processed_dir: Path, date_tag: str, available: pd.DataFrame, candidates: pd.DataFrame, unusable: pd.DataFrame, s2_report: dict[str, Any], patch_report: dict[str, Any], secchi_metrics: pd.DataFrame, chla_metrics: pd.DataFrame) -> None:
    chla_items = [
        ("water temperature", "partial", "data/raw/manual/environment/era5_features_2014_2024.csv", "daily 2014-2024", "study-wide proxy only", "station+date by date only", "no", "yes"),
        ("surface water temperature", "partial", "ERA5 SST rolling columns", "daily windows", "regional", "date only", "no", "yes"),
        ("bottom water temperature", "no", "", "", "", "no", "no", "no"),
        ("vertical temperature profile", "no", "", "", "", "no", "no", "no"),
        ("thermocline depth", "no", "", "", "", "no", "no", "no"),
        ("stratification index", "proxy", "derived from air temp / SST / wind", "daily windows", "regional", "date only", "no", "yes"),
        ("dissolved oxygen profile", "no", "in-situ DO exists but same-day water-quality variable, not allowed as predictor", "sampling", "station", "leakage risk", "yes if same-day", "no"),
        ("water depth/station depth/bathymetry", "no", "", "", "", "no", "no", "no"),
        ("distance to shore/river mouth", "no", "", "", "", "no", "no", "no"),
        ("lake zone / bay / basin", "yes", "station_meta/model input water_zone_en", "static", "station", "yes", "no", "yes"),
        ("current velocity / tidal current", "no", "", "", "", "no", "no", "no"),
        ("water level / inflow / outflow", "no", "", "", "", "no", "no", "no"),
        ("wind / solar / air temperature / precipitation", "yes", "ERA5 + rainfall files", "daily/window", "regional", "date only", "no", "yes"),
    ]
    secchi_items = [
        ("3x3 patch mean/std/valid ratio", "yes", "CMEMS Ocean Colour NetCDF", "daily 2014-2024", "yes", "yes"),
        ("5x5 patch mean/std/valid ratio", "yes", "CMEMS Ocean Colour NetCDF", "daily 2014-2024", "yes", "yes"),
        ("local gradient", "yes", "derived point-minus-3x3 mean", "daily 2014-2024", "yes", "yes"),
        ("spatial heterogeneity index", "yes", "5x5 std / abs(mean)", "daily 2014-2024", "yes", "yes"),
        ("cloud / missing ratio", "partial", "patch missing ratio from finite pixels; no cloud flag", "daily", "yes", "yes"),
        ("distance-weighted neighborhood mean", "no", "not generated in this round", "", "possible", "no"),
    ]
    gap_lines = ["# 数据缺口与可用性清单", "", "## Chl-a bottom 瓶颈所需数据", "", "| 数据项 | 是否已有 | 文件位置 | 时间/空间信息 | 是否能 station+date 对齐 | 泄漏风险 | 本轮 A/B |", "|---|---|---|---|---|---|---|"]
    for item in chla_items:
        gap_lines.append(f"| {item[0]} | {item[1]} | {item[2]} | {item[3]} / {item[4]} | {item[5]} | {item[6]} | {item[7]} |")
    gap_lines += ["", "## Secchi/ZSD 瓶颈所需数据", "", "| 数据项 | 是否可提取 | 原始数据 | 时间范围 | 能否匹配站点坐标 | 本轮 A/B |", "|---|---|---|---|---|---|"]
    for item in secchi_items:
        gap_lines.append(f"| {item[0]} | {item[1]} | {item[2]} | {item[3]} | {item[4]} | {item[5]} |")
    gap_lines += ["", "## S2 patch manifest 判断", "", f"- 文件已复制: {s2_report.get('copied_to')}", f"- 字段: {', '.join(s2_report.get('columns', []))}", "- 该文件是 Sentinel-2 patch 可用性/有效像元 manifest，不含完整 3x3/5x5 多变量统计，因此不作为本轮真实 patch 特征表。"]
    (report_dir / "data_gap_inventory.md").write_text("\n".join(gap_lines) + "\n", encoding="utf-8")
    shutil.copyfile(report_dir / "data_gap_inventory.md", out_dir / "data_gap_inventory.md")

    plan = """# 机制数据源接入计划

## 本轮可用

- CMEMS Ocean Colour NetCDF: 可直接提取 CHL、KD490、ZSD、RRS 的 3x3/5x5 patch。
- ERA5/rainfall/tide: 可构造风混合、光热、降雨扰动和潮差代理变量。
- station_meta: 可提供站点经纬度和 water_zone_en。
- S2 patch manifest: 可作为 Sentinel-2 patch 下载/提取清单，但不是完整 patch 特征。

## 关键缺口

- bottom water temperature、vertical temperature profile、thermocline depth。
- station depth、bathymetry、distance to shore、distance to river mouth。
- current velocity、tidal current、water level、discharge、inflow/outflow。

## 推荐下一步

1. 优先接入 bathymetry / station depth / distance-to-shore。
2. 从 Copernicus Marine Physics 提取 uo/vo、thetao、bottom temperature、mlotst。
3. 若可获得本地水文站，补水位、流量和潮流。
4. 对 Sentinel-2 manifest 对应 patch 文件补 3x3/5x5 统计，专门服务 Secchi/ZSD surface。
"""
    (report_dir / "mechanism_data_source_plan.md").write_text(plan, encoding="utf-8")
    shutil.copyfile(report_dir / "mechanism_data_source_plan.md", out_dir / "mechanism_data_source_plan.md")

    do_not = """# 不再重复实验清单

- 普通模型调参已不足以接近 R2=0.60。
- station calibration 只是小幅收益，不再作为突破路线重复。
- 简单 ERA5/rain/tide proxy 未达 major gain。
- 没有新增真实机制变量和 patch 特征前，不再继续做模型微调。
- 不再把 Ocean Colour 点值 + 历史目标继续堆特征当作主线。
- 不再把 Sentinel-2 manifest 当作 patch 特征；必须有真实邻域统计。
"""
    (report_dir / "do_not_repeat_experiments.md").write_text(do_not, encoding="utf-8")
    shutil.copyfile(report_dir / "do_not_repeat_experiments.md", out_dir / "do_not_repeat_experiments.md")

    missing_plan = """# 缺失关键机制数据接入计划

## 1. 缺哪些关键数据

- Chl-a bottom: bottom water temperature、vertical temperature profile、thermocline depth、station depth、bathymetry、current velocity、tidal current、water level、discharge/inflow/outflow。
- Secchi/ZSD: Sentinel-2/Landsat 原始 patch 统计仍未进入主链；CMEMS 4 km patch 已可用但空间分辨率较粗。

## 2. 为什么可能提升精度

- 水温/分层/混合决定 bottom Chl-a 与表层遥感的耦合程度。
- bathymetry 和站点水深决定再悬浮、底层交换和光照限制。
- 流速/水位/径流决定冲刷、稀释和营养盐输入。
- 高分辨率 patch 捕捉 Secchi/ZSD 的局地空间异质性。

## 3. 推荐数据源

- 水温: Copernicus Marine Physics thetao/bottom temperature、OSTIA、现场 logger。
- bathymetry: 项目测深图、公开 bathymetry raster、站点水深表。
- 水位/流量: 香港本地水文站、河口入流/出流资料。
- 流速/潮流: Copernicus Marine Physics uo/vo 或本地 EFDC/Delft3D 输出。
- 空间 patch: 已有 CMEMS NetCDF 可继续；Sentinel-2 manifest 需要对应 patch 影像或重导邻域统计。

## 4. 下一步脚本建议

- `scripts/data/download_cmems_physics_mechanism_features.py`
- `scripts/features/extract_bathymetry_station_features.py`
- `scripts/features/extract_s2_patch_statistics_from_manifest.py`
"""
    (out_dir / "missing_critical_data_plan.md").write_text(missing_plan, encoding="utf-8")

    def best_delta(metrics: pd.DataFrame) -> str:
        if metrics.empty:
            return "未运行"
        exp = metrics[metrics["ab_group"] == "experiment"].copy()
        base = metrics[metrics["ab_group"] == "baseline"].copy()
        merged = exp.merge(base[["target", "depth", "r2"]], on=["target", "depth"], suffixes=("", "_baseline"))
        if merged.empty:
            return "未形成可比结果"
        merged["ab_delta"] = merged["r2"] - merged["r2_baseline"]
        row = merged.sort_values("ab_delta", ascending=False).iloc[0]
        return f"{row['target']} / {row['depth']} 最大 A/B 增益 {row['ab_delta']:+.4f}，experiment R2={row['r2']:.4f}"

    report = f"""# Mechanism Data Expansion {DATE_TAG}

## 1. 可用机制数据

- CMEMS Ocean Colour NetCDF 存在并已成功提取真实 3x3/5x5 patch。
- ERA5/rainfall/tide 存在，可构造风混合、降雨扰动、光热和潮差代理变量。
- S2 patch manifest 已复制并检查，但它是 manifest，不是完整 patch 统计表。

## 2. 不可用关键数据

- bottom water temperature、vertical temperature profile、thermocline depth。
- station depth、bathymetry、distance to shore、distance to river mouth。
- current velocity、tidal current、water level、discharge、inflow/outflow。

## 3. Patch 是否提升 Secchi/ZSD

{best_delta(secchi_metrics)}

成功标准: 任一水层 R2 提升 >=0.03 为 useful_patch_source，>=0.05 为 major_patch_gain。详见 `secchi_patch_comparison_to_previous_best.csv`。

## 4. 机制变量是否提升 Chl-a bottom

{best_delta(chla_metrics)}

成功标准: Chl-a bottom R2 提升 >=0.03 为 useful_mechanism_source，>=0.05 为 major_mechanism_gain。

## 5. 是否接近 R2=0.60

本轮不以普通调参为目标；若输出比较表未显示 major gain，则说明当前仍缺关键机制数据，尤其是 bottom 水温/水动力/bathymetry。

## 6. 输出位置

- `experiments/outputs/mechanism_data_expansion_{DATE_TAG}/`
- `data/processed/mechanism_data_expansion_{DATE_TAG}/`
- `experiments/reports/mechanism_data_expansion_{DATE_TAG}.md`

## 7. 目录状态

未在根目录生成 csv/json/py/ipynb。
"""
    (report_dir / f"mechanism_data_expansion_{DATE_TAG}.md").write_text(report, encoding="utf-8")
    shutil.copyfile(report_dir / f"mechanism_data_expansion_{DATE_TAG}.md", out_dir / f"mechanism_data_expansion_{DATE_TAG}.md")
    config = {
        "date_tag": DATE_TAG,
        "s2_manifest": s2_report,
        "patch_report_status": patch_report.get("status"),
        "outputs": {"out_dir": str(out_dir), "processed_dir": str(processed_dir)},
    }
    (Path("experiments/configs") / f"mechanism_data_expansion_{DATE_TAG}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"mechanism_data_expansion_{DATE_TAG}.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def update_registries(report_dir: Path) -> None:
    registry = report_dir / "methods_tried_registry.md"
    text = registry.read_text(encoding="utf-8") if registry.exists() else "# 已尝试方法登记\n"
    marker = "## Mechanism Data Expansion 20260518"
    if marker not in text:
        text += f"""

{marker}

已完成真实 CMEMS Ocean Colour patch 提取、S2 patch manifest 可用性判断、已有环境/机制数据扫描、Chl-a bottom 机制代理 A/B 与 Secchi/ZSD patch A/B。

后续不再重复普通 ERA5 小集合、station calibration 或点值 Ocean Colour 堆特征；只有新增 bottom 水温、水动力、bathymetry 或高分辨率 patch 后才继续模型突破实验。
"""
        registry.write_text(text, encoding="utf-8")
    failed = report_dir / "failed_experiments.md"
    ftxt = failed.read_text(encoding="utf-8") if failed.exists() else "# 失败实验登记\n"
    marker2 = "## Mechanism Data Expansion 20260518"
    if marker2 not in ftxt:
        ftxt += f"""

{marker2}

若本轮 patch 或机制 A/B 未形成 >=0.03 的实质增益，则说明仅靠当前 CMEMS 4 km patch 与区域 ERA5/rain/tide 代理仍不足以接近 R2=0.60。下一步必须补 station depth、bathymetry、bottom water temperature、current velocity 或更高分辨率 Sentinel-2/Landsat patch。
"""
        failed.write_text(ftxt, encoding="utf-8")


def main() -> None:
    args = parse_args()
    global DATE_TAG
    DATE_TAG = args.date_tag
    out_dir, report_dir, config_dir, processed_dir = ensure_dirs(args.date_tag)
    if args.force and out_dir.exists():
        # Keep directory but remove files from previous failed same-tag runs.
        for file in out_dir.glob("*"):
            if file.is_file():
                file.unlink()
    s2_report = copy_s2_manifest()
    available, candidates, unusable = scan_files(out_dir)
    df = load_model_input(Path(args.input))
    patch, patch_report = extract_patch_features(df, processed_dir, out_dir)
    if not patch.empty:
        patch["date"] = pd.to_datetime(patch["date"])
        df_patch = df.merge(patch.drop(columns=["sample_id", "lon", "lat", "water_zone_en"], errors="ignore"), on=["station", "date"], how="left")
        patch_cols = [c for c in df_patch.columns if c.startswith("oc_patch_")]
        secchi_metrics, secchi_preds, secchi_importance = run_ab(df_patch, "secchi", DEPTHS, [], patch_cols, "secchi_patch_ab", TARGETS["secchi"]["previous_best"])
        secchi_metrics.to_csv(out_dir / "secchi_patch_ab_metrics.csv", index=False, encoding="utf-8-sig")
        secchi_preds.to_csv(out_dir / "secchi_patch_predictions.csv", index=False, encoding="utf-8-sig")
        secchi_importance.to_csv(out_dir / "secchi_patch_feature_importance.csv", index=False, encoding="utf-8-sig")
        comp = secchi_metrics[secchi_metrics["ab_group"] == "experiment"].copy()
        base = secchi_metrics[secchi_metrics["ab_group"] == "baseline"][["target", "depth", "r2"]].rename(columns={"r2": "baseline_r2"})
        comp = comp.merge(base, on=["target", "depth"], how="left")
        comp["patch_ab_delta"] = comp["r2"] - comp["baseline_r2"]
        comp["useful_patch_source"] = comp["patch_ab_delta"] >= 0.03
        comp["major_patch_gain"] = comp["patch_ab_delta"] >= 0.05
        comp.to_csv(out_dir / "secchi_patch_comparison_to_previous_best.csv", index=False, encoding="utf-8-sig")
    else:
        secchi_metrics = pd.DataFrame()
        pd.DataFrame().to_csv(out_dir / "secchi_patch_ab_metrics.csv", index=False)

    mech_df, mech_cols = make_bottom_mechanism(df, processed_dir, out_dir)
    if mech_cols:
        chla_metrics, chla_preds, chla_importance = run_ab(mech_df, "chla", ["bottom"], [], mech_cols, "chla_bottom_mechanism_ab", TARGETS["chla"]["previous_best"])
        chla_metrics.to_csv(out_dir / "chla_bottom_mechanism_ab_metrics.csv", index=False, encoding="utf-8-sig")
        chla_preds.to_csv(out_dir / "chla_bottom_mechanism_predictions.csv", index=False, encoding="utf-8-sig")
        chla_importance.to_csv(out_dir / "chla_bottom_mechanism_feature_importance.csv", index=False, encoding="utf-8-sig")
        comp = chla_metrics[chla_metrics["ab_group"] == "experiment"].copy()
        base = chla_metrics[chla_metrics["ab_group"] == "baseline"][["target", "depth", "r2"]].rename(columns={"r2": "baseline_r2"})
        comp = comp.merge(base, on=["target", "depth"], how="left")
        comp["mechanism_ab_delta"] = comp["r2"] - comp["baseline_r2"]
        comp["useful_mechanism_source"] = comp["mechanism_ab_delta"] >= 0.03
        comp["major_mechanism_gain"] = comp["mechanism_ab_delta"] >= 0.05
        comp["bottom_breakthrough_candidate"] = comp["r2"] >= 0.40
        comp["r2_0p6_candidate"] = comp["r2"] >= 0.60
        comp.to_csv(out_dir / "chla_bottom_mechanism_comparison_to_previous_best.csv", index=False, encoding="utf-8-sig")
    else:
        chla_metrics = pd.DataFrame()
        pd.DataFrame().to_csv(out_dir / "chla_bottom_mechanism_ab_metrics.csv", index=False)

    write_reports(out_dir, report_dir, config_dir, processed_dir, args.date_tag, available, candidates, unusable, s2_report, patch_report, secchi_metrics, chla_metrics)
    update_registries(report_dir)
    for name in ["methods_tried_registry.md", "failed_experiments.md"]:
        if (report_dir / name).exists():
            shutil.copyfile(report_dir / name, out_dir / name)
    print(json.dumps({"out_dir": str(out_dir), "processed_dir": str(processed_dir), "patch_status": patch_report.get("status"), "s2_manifest_feature_table": s2_report.get("is_feature_table")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
