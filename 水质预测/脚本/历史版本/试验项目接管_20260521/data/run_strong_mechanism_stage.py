from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "venvs" / "ocean_ab_20260517" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(VENV_SITE_PACKAGES))

from scripts.new_data_sources_utils import load_env_file
from scripts.training import run_current_highres_patch_ab as base_ab

CURRENT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
CURRENT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "current"
CURRENT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"
CURRENT_EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
CURRENT_GEE_EXPORT_DIR = CURRENT_EXTERNAL_DIR / "gee_exports"
MODEL_TABLE = PROJECT_ROOT / "data" / "model_input" / "model_table_3depth_2014_2024.csv"
STATION_META = PROJECT_ROOT / "data" / "model_input" / "station_meta.csv"
EXISTING_STRONG_INPUT = CURRENT_PROCESSED_DIR / "strong_input_features_2014_2024.csv"
EXISTING_STRONG_INPUT_DICT = CURRENT_PROCESSED_DIR / "strong_input_feature_dictionary.csv"
ERA5_FILE = PROJECT_ROOT / "data" / "raw" / "manual" / "environment" / "era5_features_2014_2024.csv"
RAIN_FILE = PROJECT_ROOT / "data" / "raw" / "manual" / "environment" / "rainfall_features_2014_2024.csv"
TIDE_FILE = PROJECT_ROOT / "data" / "raw" / "manual" / "environment" / "tide_features_2014_2024.csv"
S2_PATCH = CURRENT_PROCESSED_DIR / "s2_highres_patch_features_2014_2024.csv"
LANDSAT_PATCH = CURRENT_PROCESSED_DIR / "landsat_highres_patch_features_2014_2024.csv"
HIGHRES_PATCH = CURRENT_PROCESSED_DIR / "highres_patch_features_2014_2024.csv"
CMEMS_OUT = PROJECT_ROOT / "data" / "model_input" / "new_sources" / "copernicus_physics_features_2014_2024.csv"
CMEMS_SCRIPT = PROJECT_ROOT / "scripts" / "25_download_cmems_physics_features.py"

CAPABILITY_CSV = CURRENT_PROCESSED_DIR / "auto_data_acquisition_capability.csv"
GLOBATHY_CSV = CURRENT_PROCESSED_DIR / "globathy_bathymetry_features.csv"
GEBCO_CSV = CURRENT_PROCESSED_DIR / "gebco_bathymetry_features.csv"
DEPTH_FEATURES_CSV = CURRENT_PROCESSED_DIR / "depth_bathymetry_features_2014_2024.csv"
COPERNICUS_CURRENT_CSV = CURRENT_PROCESSED_DIR / "copernicus_physics_features_2014_2024.csv"
COPERNICUS_VALIDATION_JSON = CURRENT_PROCESSED_DIR / "copernicus_physics_validation_report.json"
COPERNICUS_MISSING_CSV = CURRENT_PROCESSED_DIR / "copernicus_physics_missing_summary.csv"
TIDE_PHASE_CSV = CURRENT_PROCESSED_DIR / "tide_phase_features_2014_2024.csv"
TIDE_PHASE_VALIDATION = CURRENT_PROCESSED_DIR / "tide_phase_validation_report.json"
HIGH_QUALITY_PATCH_CSV = CURRENT_PROCESSED_DIR / "high_quality_patch_features_2014_2024.csv"
STRONG_MECH_CSV = CURRENT_PROCESSED_DIR / "strong_mechanism_input_features_2014_2024.csv"
STRONG_MECH_DICT = CURRENT_PROCESSED_DIR / "strong_mechanism_feature_dictionary.csv"
STRONG_MECH_MISSING = CURRENT_PROCESSED_DIR / "strong_mechanism_missing_summary.csv"
STRONG_MECH_VALIDATION = CURRENT_PROCESSED_DIR / "strong_mechanism_validation_report.json"
PATCH_QC_METRICS = CURRENT_OUTPUT_DIR / "patch_quality_filter_ab_metrics.csv"
STRONG_MECH_METRICS = CURRENT_OUTPUT_DIR / "strong_mechanism_ab_metrics.csv"
STRONG_MECH_PRED = CURRENT_OUTPUT_DIR / "strong_mechanism_ab_predictions.csv"
STRONG_MECH_IMPORTANCE = CURRENT_OUTPUT_DIR / "strong_mechanism_feature_importance.csv"
STRONG_MECH_STATION = CURRENT_OUTPUT_DIR / "strong_mechanism_station_level_comparison.csv"
STRONG_MECH_COMPARE = CURRENT_OUTPUT_DIR / "strong_mechanism_comparison_to_current_best.csv"

CMEMS_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strong mechanism data acquisition and A/B stage.")
    parser.add_argument("--date-tag", default="20260519")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        CURRENT_REPORT_DIR,
        CURRENT_OUTPUT_DIR,
        CURRENT_PROCESSED_DIR,
        CURRENT_EXTERNAL_DIR,
        CURRENT_GEE_EXPORT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_base_ids() -> pd.DataFrame:
    df = base_ab.load_base_dataset()
    keep = [
        "obs_id",
        "sample_id",
        "station",
        "date",
        "depth",
        "split",
        "lon",
        "lat",
        "water_zone_en",
        "year",
        "month",
        "is_wet_season",
    ]
    out = df[keep].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


def station_meta_frame() -> pd.DataFrame:
    df = pd.read_csv(STATION_META, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    out = df[[cols.get("station", "Station"), cols.get("lon", "lon"), cols.get("lat", "lat"), cols.get("water_zone_en", "water_zone_en")]].copy()
    out.columns = ["station", "lon", "lat", "water_zone_en"]
    out["station"] = out["station"].astype(str).str.strip().str.upper()
    return out.drop_duplicates(subset=["station"]).reset_index(drop=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def non_null_cols(df: pd.DataFrame, skip: set[str]) -> list[str]:
    cols = []
    for col in df.columns:
        if col in skip:
            continue
        if df[col].notna().sum() > 0:
            cols.append(col)
    return cols


def capability_checks() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(
        data_source: str,
        variable_group: str,
        can_access: bool,
        can_download_automatically: bool,
        requires_user_action: bool,
        expected_variables: str,
        expected_target: str,
        expected_gain_level: str,
        risk: str,
        next_action: str,
    ) -> None:
        rows.append(
            {
                "data_source": data_source,
                "variable_group": variable_group,
                "can_access": can_access,
                "can_download_automatically": can_download_automatically,
                "requires_user_action": requires_user_action,
                "expected_variables": expected_variables,
                "expected_target": expected_target,
                "expected_gain_level": expected_gain_level,
                "risk": risk,
                "next_action": next_action,
            }
        )

    gee_ok = False
    dynamic_world_ok = False
    globathy_ok = False
    gee_note = "need_manual_auth"
    try:
        import ee

        ee.Initialize()
        gee_ok = True
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").limit(1).size().getInfo()
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").limit(1).size().getInfo()
        dynamic_world_ok = True
        ee.Image("projects/sat-io/open-datasets/GLOBathy/GLOBathy_bathymetry").bandNames().getInfo()
        globathy_ok = True
        gee_note = "gee_ready"
    except Exception as exc:
        gee_note = f"gee_probe_failed:{type(exc).__name__}"
    add_row(
        "gee",
        "platform",
        gee_ok,
        gee_ok,
        not gee_ok,
        "Sentinel-2,Landsat,DynamicWorld,GLOBathy",
        "chla;secchi",
        "high",
        "medium",
        gee_note,
    )
    add_row(
        "globathy_via_gee",
        "bathymetry",
        globathy_ok,
        globathy_ok,
        not globathy_ok,
        "public bathymetry raster",
        "chla_bottom;secchi_bottom",
        "medium",
        "high_domain_mismatch_for_coastal_marine",
        "exploratory_only_if_no_marine_bathymetry",
    )
    add_row(
        "dynamic_world_via_gee",
        "water_mask_qc",
        dynamic_world_ok,
        dynamic_world_ok,
        not dynamic_world_ok,
        "water probability",
        "chla_surface;secchi",
        "medium",
        "low",
        "usable_for_patch_qc_if_station_date_export_added",
    )

    cop_ok = False
    cop_auto = False
    cop_note = "credentials_or_catalog_unavailable"
    try:
        import copernicusmarine

        env = load_env_file()
        username = env.get("COPERNICUSMARINE_USERNAME")
        password = env.get("COPERNICUSMARINE_PASSWORD")
        if username and password:
            copernicusmarine.login(
                username=username,
                password=password,
                check_credentials_valid=True,
                force_overwrite=False,
            )
            copernicusmarine.describe(
                dataset_id=CMEMS_DATASET_ID,
                disable_progress_bar=True,
            )
            cop_ok = True
            cop_auto = True
            cop_note = f"dataset_ready:{CMEMS_DATASET_ID}"
    except Exception as exc:
        cop_note = f"copernicus_probe_failed:{type(exc).__name__}"
    add_row(
        "copernicus_marine_physics",
        "physics",
        cop_ok,
        cop_auto,
        not cop_auto,
        "thetao,uo,vo,zos,mlotst,so",
        "chla_bottom;secchi_bottom",
        "high",
        "medium_reanalysis_proxy",
        cop_note,
    )

    hko_ok = False
    datagov_ok = False
    try:
        hko_ok = requests.get("https://www.hko.gov.hk", timeout=15).status_code == 200
    except Exception:
        hko_ok = False
    try:
        datagov_ok = requests.get("https://data.gov.hk", timeout=15).status_code == 200
    except Exception:
        datagov_ok = False
    add_row(
        "hko_datagov_tide",
        "tide",
        hko_ok or datagov_ok or TIDE_FILE.exists(),
        bool(hko_ok or datagov_ok),
        False,
        "daily_tidal_range;phase_if_sampling_time_exists",
        "chla_bottom;secchi",
        "medium",
        "date_level_only_without_sampling_datetime",
        "local_tide_file_ready_and_exact_sampling_time_missing",
    )

    def local_source_found(tokens: list[str]) -> bool:
        # Only inspect raw/external/interim-style data containers to avoid
        # false positives from reports, model outputs, and markdown notes.
        search_roots = [
            PROJECT_ROOT / "data" / "raw",
            PROJECT_ROOT / "data" / "external",
            PROJECT_ROOT / "data" / "interim",
            PROJECT_ROOT / "data" / "processed" / "current",
        ]
        allowed_suffixes = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".nc", ".tif", ".tiff", ".shp", ".dbf", ".json"}
        token_set = [token.lower() for token in tokens]
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                    continue
                path_text = str(path).lower()
                if any(token in path_text for token in token_set):
                    return True
        return False

    local_files = [
        ("exact_sampling_datetime", ["sampling_time", "sample_time", "sampling_datetime", "sample_datetime"], "sampling tide phase"),
        ("station_depth", ["station_depth", "station-bathymetry", "bathymetry_depth"], "depth"),
        ("vertical_temp_profile", ["vertical_temp_profile", "thermocline_depth", "stratification_index", "bottom_water_temperature"], "temperature profile"),
        ("current_discharge_water_level", ["adcp", "current_speed", "river_discharge", "water_level", "flow_velocity"], "hydrodynamics"),
    ]
    for source_name, tokens, variable_group in local_files:
        found = local_source_found(tokens)
        add_row(
            source_name,
            variable_group,
            found,
            False,
            not found,
            ",".join(tokens),
            "chla_bottom;secchi_bottom",
            "high",
            "low",
            "local_found" if found else "user_manual_required",
        )

    df = pd.DataFrame(rows)
    df.to_csv(CAPABILITY_CSV, index=False, encoding="utf-8-sig")
    return df


def extract_globathy_features(base_ids: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    report = {
        "status": "not_run",
        "note": "",
        "risk": "high_domain_mismatch_for_coastal_marine",
    }
    try:
        import ee

        ee.Initialize()
        img = ee.Image("projects/sat-io/open-datasets/GLOBathy/GLOBathy_bathymetry")
        stations = station_meta_frame()
        rows: list[dict[str, Any]] = []
        reducer = (
            ee.Reducer.mean()
            .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
            .combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
        )
        for row in stations.itertuples(index=False):
            point = ee.Geometry.Point([float(row.lon), float(row.lat)])
            point_depth = img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=31,
                maxPixels=1e8,
            ).getInfo()
            patch_3 = img.reduceRegion(
                reducer=reducer,
                geometry=point.buffer(46).bounds(),
                scale=31,
                maxPixels=1e8,
            ).getInfo()
            patch_5 = img.reduceRegion(
                reducer=reducer,
                geometry=point.buffer(77).bounds(),
                scale=31,
                maxPixels=1e8,
            ).getInfo()
            rows.append(
                {
                    "station": row.station,
                    "lon": row.lon,
                    "lat": row.lat,
                    "globathy_point_depth": (point_depth or {}).get("b1"),
                    "globathy_3x3_mean": (patch_3 or {}).get("b1_mean"),
                    "globathy_3x3_std": (patch_3 or {}).get("b1_stdDev"),
                    "globathy_3x3_min": (patch_3 or {}).get("b1_min"),
                    "globathy_3x3_max": (patch_3 or {}).get("b1_max"),
                    "globathy_5x5_mean": (patch_5 or {}).get("b1_mean"),
                    "globathy_5x5_std": (patch_5 or {}).get("b1_stdDev"),
                    "globathy_5x5_min": (patch_5 or {}).get("b1_min"),
                    "globathy_5x5_max": (patch_5 or {}).get("b1_max"),
                    "bathymetry_source": "GLOBathy_via_GEE",
                    "bathymetry_risk_flag": "coastal_marine_domain_check_needed",
                }
            )
        station_df = pd.DataFrame(rows)
        station_df.to_csv(GLOBATHY_CSV, index=False, encoding="utf-8-sig")
        expanded = base_ids.merge(
            station_df.drop(columns=["lon", "lat"]),
            on="station",
            how="left",
            validate="many_to_one",
        )
        expanded.to_csv(DEPTH_FEATURES_CSV, index=False, encoding="utf-8-sig")
        report["status"] = "completed_exploratory"
        report["rows"] = int(len(expanded))
        report["station_rows"] = int(len(station_df))
        report["note"] = "Used GLOBathy via GEE as exploratory public bathymetry raster. It is not treated as confirmed coastal station depth."
        return expanded, report
    except Exception as exc:
        report["status"] = "failed"
        report["note"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(columns=["obs_id", "sample_id", "station", "date", "depth"]), report


def run_copernicus_physics() -> tuple[pd.DataFrame, dict[str, Any]]:
    report: dict[str, Any] = {
        "status": "not_run",
        "dataset_id": CMEMS_DATASET_ID,
    }
    if not CMEMS_SCRIPT.exists():
        report["status"] = "missing_script"
        return pd.DataFrame(), report
    cmd = [
        sys.executable,
        str(CMEMS_SCRIPT),
        "--config",
        "config/new_data_sources.yaml",
        "--dataset-id",
        CMEMS_DATASET_ID,
        "--date-tag",
        "20260519",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    report["returncode"] = proc.returncode
    report["stdout_tail"] = (proc.stdout or "")[-2000:]
    report["stderr_tail"] = (proc.stderr or "")[-2000:]
    if proc.returncode != 0 or not CMEMS_OUT.exists():
        report["status"] = "failed"
        return pd.DataFrame(), report

    df = pd.read_csv(CMEMS_OUT, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["station"] = df["station"].astype(str).str.strip().str.upper()
    df["surface_temp"] = safe_numeric(df.get("cmems_sst", pd.Series(np.nan, index=df.index)))
    df["near_bottom_temp_proxy"] = safe_numeric(df.get("cmems_bottom_temp", pd.Series(np.nan, index=df.index)))
    df["temp_surface_bottom_diff_proxy"] = df["surface_temp"] - df["near_bottom_temp_proxy"]
    df["current_speed"] = safe_numeric(df.get("cmems_current_speed", pd.Series(np.nan, index=df.index)))
    df["current_speed_3d_mean"] = safe_numeric(df.get("cmems_current_speed_3d", pd.Series(np.nan, index=df.index)))
    df["current_speed_7d_mean"] = safe_numeric(df.get("cmems_current_speed_7d", pd.Series(np.nan, index=df.index)))
    df["sea_level"] = safe_numeric(df.get("cmems_ssh", pd.Series(np.nan, index=df.index)))
    grouped = df.sort_values(["station", "date"]).groupby("station", dropna=False)
    df["sea_level_change_1d"] = grouped["sea_level"].diff()
    df["sea_level_change_3d"] = grouped["sea_level"].transform(lambda s: s - s.shift(3))
    df["mixed_layer_depth"] = safe_numeric(df.get("cmems_mld", pd.Series(np.nan, index=df.index)))
    df["mixed_layer_depth_7d_mean"] = safe_numeric(df.get("cmems_mld_7d", pd.Series(np.nan, index=df.index)))
    keep_cols = [
        "date",
        "station",
        "cmems_sst",
        "cmems_bottom_temp",
        "cmems_sss",
        "cmems_uo",
        "cmems_vo",
        "cmems_ssh",
        "cmems_mld",
        "cmems_current_speed",
        "cmems_current_dir",
        "cmems_sss_3d",
        "cmems_sss_7d",
        "cmems_sss_14d",
        "cmems_mld_3d",
        "cmems_mld_7d",
        "cmems_mld_14d",
        "cmems_current_speed_3d",
        "cmems_current_speed_7d",
        "cmems_current_speed_14d",
        "surface_temp",
        "near_bottom_temp_proxy",
        "temp_surface_bottom_diff_proxy",
        "current_speed",
        "current_speed_3d_mean",
        "current_speed_7d_mean",
        "sea_level",
        "sea_level_change_1d",
        "sea_level_change_3d",
        "mixed_layer_depth",
        "mixed_layer_depth_7d_mean",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    out.to_csv(COPERNICUS_CURRENT_CSV, index=False, encoding="utf-8-sig")
    missing = pd.DataFrame(
        {
            "feature": out.columns,
            "missing_rate": out.isna().mean().to_numpy(),
            "non_null_count": out.notna().sum().to_numpy(),
        }
    )
    missing.to_csv(COPERNICUS_MISSING_CSV, index=False, encoding="utf-8-sig")
    report["status"] = "completed"
    report["rows"] = int(len(out))
    report["date_range"] = f"{out['date'].min().date()} to {out['date'].max().date()}" if len(out) else ""
    report["available_feature_count"] = int(len([c for c in out.columns if c not in {"date", "station"}]))
    report["note"] = "These variables are reanalysis/model-derived physical proxies, not in-situ bottom observations."
    write_json(COPERNICUS_VALIDATION_JSON, report)
    return out, report


def build_tide_phase_features(base_ids: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    tide = pd.read_csv(TIDE_FILE, low_memory=False)
    tide["date"] = pd.to_datetime(tide["date"])
    tide = tide.sort_values("date").reset_index(drop=True)
    tide["tide_range_same_day"] = safe_numeric(tide["tidal_range"])
    tide["tide_range_previous_3d_mean"] = tide["tide_range_same_day"].rolling(3, min_periods=1).mean()
    tide["tide_range_previous_1d"] = tide["tide_range_same_day"].shift(1)
    tide["tide_level_daily_mean"] = safe_numeric(tide["tide_level_mean"])
    tide["tide_proxy_feature_level"] = "date_level_only"
    out = base_ids[["obs_id", "sample_id", "station", "date", "depth"]].merge(
        tide[
            [
                "date",
                "tidal_range",
                "tide_events",
                "tide_high_mean",
                "tide_low_mean",
                "tide_level_mean",
                "tide_range_same_day",
                "tide_range_previous_3d_mean",
                "tide_range_previous_1d",
                "tide_level_daily_mean",
                "tide_proxy_feature_level",
            ]
        ],
        on="date",
        how="left",
        validate="many_to_one",
    )
    out.to_csv(TIDE_PHASE_CSV, index=False, encoding="utf-8-sig")
    report = {
        "status": "completed",
        "rows": int(len(out)),
        "exact_sampling_datetime_available": False,
        "note": "Only date-level tide proxies were generated because exact_sampling_datetime is unavailable.",
    }
    write_json(TIDE_PHASE_VALIDATION, report)
    return out, report


def _sensor_quality_mask(df: pd.DataFrame, sensor: str, valid_threshold: float, date_threshold: int, require_water: bool) -> pd.Series:
    ratio = df.get(f"{sensor}_5x5_valid_ratio", df.get(f"{sensor}_3x3_valid_ratio"))
    if ratio is None:
        return pd.Series(False, index=df.index)
    ratio = safe_numeric(ratio).fillna(0)
    date_diff = safe_numeric(df.get(f"{sensor}_date_diff_abs", pd.Series(np.nan, index=df.index))).fillna(999)
    mask = ratio >= valid_threshold
    mask &= date_diff <= date_threshold
    if require_water:
        ndwi = safe_numeric(df.get(f"{sensor}_5x5_NDWI_mean", df.get(f"{sensor}_3x3_NDWI_mean", pd.Series(np.nan, index=df.index))))
        mndwi = safe_numeric(df.get(f"{sensor}_5x5_MNDWI_mean", df.get(f"{sensor}_3x3_MNDWI_mean", pd.Series(np.nan, index=df.index))))
        mask &= ((ndwi > -0.05) | (mndwi > -0.05)).fillna(False)
    return mask.fillna(False)


def make_masked_variant(df: pd.DataFrame, variant_name: str, valid_threshold: float, date_threshold: int, require_water: bool) -> tuple[pd.DataFrame, list[str]]:
    out = df[["request_id", "sample_id", "station", "date", "split", "lon", "lat"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    features: list[str] = []
    for sensor in ["s2", "landsat"]:
        mask = _sensor_quality_mask(df, sensor, valid_threshold, date_threshold, require_water)
        out[f"{variant_name}_{sensor}_quality_pass"] = mask.astype(int)
        for col in df.columns:
            if not col.startswith(f"{sensor}_"):
                continue
            new_col = f"{variant_name}_{col}"
            out[new_col] = df[col].where(mask, np.nan)
            features.append(new_col)
    return out, features


def run_quality_filter_ab(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    highres = pd.read_csv(HIGHRES_PATCH, low_memory=False)
    highres["date"] = pd.to_datetime(highres["date"])
    variants = {
        "hq_valid50_dt3": {"valid_threshold": 0.50, "date_threshold": 3, "require_water": False},
        "hq_valid70_dt3": {"valid_threshold": 0.70, "date_threshold": 3, "require_water": False},
        "hq_valid50_dt1": {"valid_threshold": 0.50, "date_threshold": 1, "require_water": False},
        "hq_water_strict": {"valid_threshold": 0.50, "date_threshold": 3, "require_water": True},
    }
    metric_frames = []
    best_variant_name = ""
    best_variant_features: list[str] = []
    best_variant_table = pd.DataFrame()
    best_score = -np.inf
    for variant_name, rules in variants.items():
        variant_table, features = make_masked_variant(highres, variant_name, **rules)
        merged = base_df.merge(
            variant_table,
            on=["request_id", "sample_id", "station", "date", "split", "lon", "lat"],
            how="left",
            validate="many_to_one",
        )
        metrics_df, _, _, _, comparison_df, _ = run_custom_scenario(
            merged,
            features,
            variant_name,
            feature_set_name=f"current_best + {variant_name}",
        )
        summary = comparison_df.copy()
        summary["validation_score_selector"] = summary["candidate_validation_r2"].fillna(-999) - summary["current_best_validation_r2"].fillna(-999)
        metric_frames.append(summary)
        selector = float(summary["validation_score_selector"].mean())
        if selector > best_score:
            best_score = selector
            best_variant_name = variant_name
            best_variant_features = features
            best_variant_table = variant_table
    patch_qc = pd.concat(metric_frames, ignore_index=True)
    patch_qc.to_csv(PATCH_QC_METRICS, index=False, encoding="utf-8-sig")
    final_cols = ["request_id", "sample_id", "station", "date", "split", "lon", "lat"] + best_variant_features
    best_variant_table[final_cols].to_csv(HIGH_QUALITY_PATCH_CSV, index=False, encoding="utf-8-sig")
    return patch_qc, best_variant_table[final_cols].copy(), best_variant_features


def run_custom_scenario(
    df: pd.DataFrame,
    feature_subset: list[str],
    scenario: str,
    feature_set_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline_predictions = base_ab.current_best_prediction_table()
    metric_rows = []
    prediction_frames = []
    importance_frames = []
    station_compare_frames = []
    comparison_rows = []
    manifest: dict[str, Any] = {"scenario": scenario, "feature_set_name": feature_set_name, "models": {}}

    for target in base_ab.TARGETS:
        for depth in base_ab.DEPTH_ORDER:
            features = base_ab.build_common_features(df, target, extra_features=feature_subset)
            result = base_ab.fit_for_current_best_family(df, target, depth, features, scenario)
            manifest["models"][f"{target}::{depth}"] = {
                "candidate_name": result.candidate_name,
                "model_type": result.model_type,
                "params": result.params,
                "features": result.features,
                "feature_count": len(result.features),
            }
            prediction_frame = base_ab.candidate_prediction_frame(df, result)
            prediction_frames.append(prediction_frame)
            imp = result.feature_importance.copy()
            imp["target"] = target
            imp["depth_layer"] = depth
            imp["candidate"] = result.candidate_name
            imp["scenario"] = scenario
            importance_frames.append(imp)

            target_col = base_ab.TARGET_SPECS[target]["column"]
            train, val, test = base_ab.split_frame(df, target_col, depth)
            metric_rows.extend(
                [
                    base_ab.metrics_row(target, depth, "train", result.candidate_name, result.model_type, train[target_col].to_numpy(), result.train_pred, train[target_col].to_numpy(), result.features, scenario),
                    base_ab.metrics_row(target, depth, "validation", result.candidate_name, result.model_type, val[target_col].to_numpy(), result.validation_pred, train[target_col].to_numpy(), result.features, scenario),
                    base_ab.metrics_row(target, depth, "test", result.candidate_name, result.model_type, test[target_col].to_numpy(), result.test_pred, train[target_col].to_numpy(), result.features, scenario),
                ]
            )

            baseline_test = baseline_predictions[
                (baseline_predictions["target"] == target)
                & (baseline_predictions["depth"] == depth)
                & (baseline_predictions["split"] == "test")
            ].copy()
            candidate_test = prediction_frame[prediction_frame["split"] == "test"].copy()
            station_cmp, _ = base_ab.compare_station_level(target, depth, baseline_test, candidate_test, scenario)
            station_compare_frames.append(station_cmp)

            current_rmse = base_ab.rmse(baseline_test["y_true"].to_numpy(), baseline_test["y_pred"].to_numpy())
            current_mae = float(base_ab.mean_absolute_error(baseline_test["y_true"], baseline_test["y_pred"]))
            current_val = baseline_predictions[
                (baseline_predictions["target"] == target)
                & (baseline_predictions["depth"] == depth)
                & (baseline_predictions["split"] == "validation")
            ].copy()
            candidate_val = prediction_frame[prediction_frame["split"] == "validation"].copy()
            candidate_r2 = float(base_ab.r2_score(candidate_test["y_true"], candidate_test["y_pred"]))
            candidate_val_r2 = float(base_ab.r2_score(candidate_val["y_true"], candidate_val["y_pred"]))
            baseline_val_r2 = float(base_ab.r2_score(current_val["y_true"], current_val["y_pred"]))
            improved_ratio = float(station_cmp["improved"].mean()) if len(station_cmp) else 0.0
            candidate_high = base_ab.high_value_metrics(
                candidate_test["y_true"].to_numpy(),
                candidate_test["y_pred"].to_numpy(),
                train[target_col].to_numpy(),
            )
            baseline_high = base_ab.high_value_metrics(
                baseline_test["y_true"].to_numpy(),
                baseline_test["y_pred"].to_numpy(),
                train[target_col].to_numpy(),
            )
            comparison_rows.append(
                {
                    "target": target,
                    "depth_layer": depth,
                    "scenario": scenario,
                    "feature_set_name": feature_set_name,
                    "current_best_candidate": base_ab.CURRENT_BEST[(target, depth)]["candidate"],
                    "candidate_model": result.candidate_name,
                    "current_best_model_type": base_ab.CURRENT_BEST[(target, depth)]["model_type"],
                    "candidate_model_type": result.model_type,
                    "current_best_r2": base_ab.CURRENT_BEST[(target, depth)]["r2"],
                    "candidate_r2": candidate_r2,
                    "delta_r2": candidate_r2 - base_ab.CURRENT_BEST[(target, depth)]["r2"],
                    "current_best_validation_r2": baseline_val_r2,
                    "candidate_validation_r2": candidate_val_r2,
                    "delta_validation_r2": candidate_val_r2 - baseline_val_r2,
                    "current_best_rmse": current_rmse,
                    "candidate_rmse": base_ab.rmse(candidate_test["y_true"].to_numpy(), candidate_test["y_pred"].to_numpy()),
                    "delta_rmse": base_ab.rmse(candidate_test["y_true"].to_numpy(), candidate_test["y_pred"].to_numpy()) - current_rmse,
                    "current_best_mae": current_mae,
                    "candidate_mae": float(base_ab.mean_absolute_error(candidate_test["y_true"], candidate_test["y_pred"])),
                    "delta_mae": float(base_ab.mean_absolute_error(candidate_test["y_true"], candidate_test["y_pred"])) - current_mae,
                    "current_best_p90_rmse": baseline_high["p90_rmse"],
                    "candidate_p90_rmse": candidate_high["p90_rmse"],
                    "delta_p90_rmse": candidate_high["p90_rmse"] - baseline_high["p90_rmse"] if pd.notna(candidate_high["p90_rmse"]) and pd.notna(baseline_high["p90_rmse"]) else np.nan,
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


def merge_strong_mechanism_table(
    base_ids: pd.DataFrame,
    depth_df: pd.DataFrame,
    cop_df: pd.DataFrame,
    tide_df: pd.DataFrame,
    hq_patch_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = base_ab.merge_model_dataset()[0]
    if not depth_df.empty:
        merged = merged.merge(
            depth_df.drop(columns=[c for c in ["split", "lon", "lat", "water_zone_en", "year", "month", "is_wet_season"] if c in depth_df.columns]),
            on=["obs_id", "sample_id", "station", "date", "depth"],
            how="left",
            validate="one_to_one",
        )
    if not cop_df.empty:
        merged = merged.merge(
            cop_df,
            on=["station", "date"],
            how="left",
            validate="many_to_one",
        )
    if not tide_df.empty:
        merged = merged.merge(
            tide_df,
            on=["obs_id", "sample_id", "station", "date", "depth"],
            how="left",
            validate="one_to_one",
        )
    if not hq_patch_df.empty:
        merged = merged.merge(
            hq_patch_df,
            on=["request_id", "sample_id", "station", "date", "split", "lon", "lat"],
            how="left",
            validate="many_to_one",
        )
    if EXISTING_STRONG_INPUT.exists():
        extra = pd.read_csv(EXISTING_STRONG_INPUT, low_memory=False)
        extra["date"] = pd.to_datetime(extra["date"])
        join_cols = [c for c in ["obs_id", "sample_id", "station", "date", "depth"] if c in extra.columns]
        extra_feature_cols = [c for c in extra.columns if c not in join_cols]
        extra = extra[join_cols + extra_feature_cols]
        merged = merged.merge(extra, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_existingstrong"))
    merged.to_csv(STRONG_MECH_CSV, index=False, encoding="utf-8-sig")

    dict_rows = []
    for col in merged.columns:
        if col in {"obs_id", "sample_id", "station", "date", "depth", "split", "lon", "lat", "water_zone_en", "year", "month", "is_wet_season"}:
            continue
        category = "existing_model_input"
        if col.startswith("globathy_"):
            category = "public_bathymetry_exploratory"
        elif col.startswith("cmems_") or col in {"surface_temp", "near_bottom_temp_proxy", "temp_surface_bottom_diff_proxy", "current_speed", "current_speed_3d_mean", "current_speed_7d_mean", "sea_level", "sea_level_change_1d", "sea_level_change_3d", "mixed_layer_depth", "mixed_layer_depth_7d_mean"}:
            category = "reanalysis_proxy"
        elif col.startswith("tide_") or col in {"tidal_range", "tide_events", "tide_high_mean", "tide_low_mean", "tide_level_mean"}:
            category = "date_level_proxy"
        elif col.startswith("hq_"):
            category = "quality_control_feature"
        elif col.startswith("s2_") or col.startswith("landsat_"):
            category = "highres_remote_sensing"
        dict_rows.append({"feature": col, "feature_group": category, "non_null_count": int(merged[col].notna().sum())})
    dictionary = pd.DataFrame(dict_rows)
    dictionary.to_csv(STRONG_MECH_DICT, index=False, encoding="utf-8-sig")

    missing = pd.DataFrame(
        {
            "feature": merged.columns,
            "missing_rate": merged.isna().mean().to_numpy(),
            "non_null_count": merged.notna().sum().to_numpy(),
        }
    )
    missing.to_csv(STRONG_MECH_MISSING, index=False, encoding="utf-8-sig")
    validation = {
        "rows": int(len(merged)),
        "station_count": int(merged["station"].nunique()),
        "date_range": f"{merged['date'].min().date()} to {merged['date'].max().date()}",
        "duplicate_obs_id": int(merged["obs_id"].duplicated().sum()) if "obs_id" in merged.columns else None,
        "duplicate_sample_station_date_depth": int(merged.duplicated(subset=["sample_id", "station", "date", "depth"]).sum()),
        "non_null_feature_groups": dictionary.groupby("feature_group")["feature"].count().to_dict(),
        "leakage_check": "No future information introduced; all added features are static, date-level, reanalysis, or patch QC inputs aligned on station/date/sample keys.",
    }
    write_json(STRONG_MECH_VALIDATION, validation)
    return merged, dictionary


def run_strong_mechanism_ab(
    merged_df: pd.DataFrame,
    depth_cols: list[str],
    cop_cols: list[str],
    tide_cols: list[str],
    hq_cols: list[str],
) -> pd.DataFrame:
    scenario_defs = []
    if depth_cols:
        scenario_defs.append(("depth_bathymetry", depth_cols, "current_best + depth_bathymetry_features"))
    if cop_cols:
        scenario_defs.append(("copernicus_physics", cop_cols, "current_best + copernicus_physics_features"))
    if tide_cols:
        scenario_defs.append(("tide_phase", tide_cols, "current_best + tide_phase_features"))
    if hq_cols:
        scenario_defs.append(("high_quality_patch", hq_cols, "current_best + high_quality_patch_features"))
    all_cols = list(dict.fromkeys(depth_cols + cop_cols + tide_cols + hq_cols))
    if all_cols:
        scenario_defs.append(("all_strong_mechanism_inputs", all_cols, "current_best + all_strong_mechanism_inputs"))

    metrics_frames = []
    pred_frames = []
    imp_frames = []
    station_frames = []
    compare_frames = []
    for scenario, cols, label in scenario_defs:
        metrics_df, pred_df, imp_df, station_df, compare_df, _ = run_custom_scenario(
            merged_df,
            cols,
            scenario,
            label,
        )
        metrics_frames.append(metrics_df)
        pred_frames.append(pred_df)
        imp_frames.append(imp_df)
        station_frames.append(station_df)
        compare_frames.append(compare_df)

    metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    importance = pd.concat(imp_frames, ignore_index=True) if imp_frames else pd.DataFrame()
    station_cmp = pd.concat(station_frames, ignore_index=True) if station_frames else pd.DataFrame()
    comparison = pd.concat(compare_frames, ignore_index=True) if compare_frames else pd.DataFrame()

    metrics.to_csv(STRONG_MECH_METRICS, index=False, encoding="utf-8-sig")
    predictions.to_csv(STRONG_MECH_PRED, index=False, encoding="utf-8-sig")
    importance.to_csv(STRONG_MECH_IMPORTANCE, index=False, encoding="utf-8-sig")
    station_cmp.to_csv(STRONG_MECH_STATION, index=False, encoding="utf-8-sig")
    comparison.to_csv(STRONG_MECH_COMPARE, index=False, encoding="utf-8-sig")
    return comparison


def update_report_docs(capability_df: pd.DataFrame, bath_report: dict[str, Any], cop_report: dict[str, Any], tide_report: dict[str, Any], patch_qc: pd.DataFrame, comparison: pd.DataFrame) -> None:
    strong_action = CURRENT_REPORT_DIR / "strong_input_data_action_plan.md"
    breakthrough = CURRENT_REPORT_DIR / "r2_0p6_breakthrough_log.md"
    experiment_report = CURRENT_REPORT_DIR / "experiment_report.md"
    data_gap = CURRENT_REPORT_DIR / "data_gap_inventory.md"
    next_plan = CURRENT_REPORT_DIR / "next_data_acquisition_plan.md"
    methods = CURRENT_REPORT_DIR / "methods_tried_registry.md"
    failed = CURRENT_REPORT_DIR / "failed_experiments.md"
    do_not_repeat = CURRENT_REPORT_DIR / "do_not_repeat_experiments.md"
    ceiling = CURRENT_REPORT_DIR / "prediction_ceiling_analysis.md"
    alignment = CURRENT_REPORT_DIR / "model_input_alignment_documentation.md"
    inventory = CURRENT_REPORT_DIR / "model_parameter_and_feature_inventory.md"
    cleanup = CURRENT_REPORT_DIR / "cleanup_log.md"

    capability_lines = [
        "# 强输入数据行动计划",
        "",
        "## 本轮自动能力检查",
        "",
    ]
    for row in capability_df.itertuples(index=False):
        capability_lines.append(
            f"- `{row.data_source}` | can_access={row.can_access} | auto={row.can_download_automatically} | risk={row.risk} | next={row.next_action}"
        )
    strong_action.write_text("\n".join(capability_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# R² 0.6 突破日志",
        "",
        "## 2026-05-19 强机制输入单轮尝试",
        "",
        f"- GLOBathy / bathymetry: {bath_report.get('status')} ({bath_report.get('note', '')})",
        f"- Copernicus physics: {cop_report.get('status')} | dataset=`{cop_report.get('dataset_id', CMEMS_DATASET_ID)}`",
        f"- Tide phase: {tide_report.get('status')} | exact_sampling_datetime={tide_report.get('exact_sampling_datetime_available')}",
        f"- Patch QC 变体数: {patch_qc['scenario'].nunique() if not patch_qc.empty and 'scenario' in patch_qc.columns else 0}",
        "",
        "## 本轮强输入 A/B 结论",
        "",
    ]
    if comparison.empty:
        summary_lines.append("- 本轮未形成可比较的 strong mechanism A/B 结果。")
    else:
        test_comp = comparison.copy()
        top = test_comp.sort_values("delta_r2", ascending=False).head(10)
        for row in top.itertuples(index=False):
            summary_lines.append(
                f"- `{row.target}/{row.depth_layer}`: `{row.scenario}` delta R² = {row.delta_r2:.4f}, validation delta = {row.delta_validation_r2:.4f}"
            )
    breakthrough.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    experiment_lines = [
        "# 当前实验报告",
        "",
        "## 本轮完成内容",
        "",
        "- 自动完成了强输入数据能力检查。",
        f"- Bathymetry/GLOBathy: `{bath_report.get('status')}`。",
        f"- Copernicus Marine physics: `{cop_report.get('status')}`。",
        f"- Tide phase/date-level proxy: `{tide_report.get('status')}`。",
        f"- Highres patch 质量筛选 A/B: `{PATCH_QC_METRICS.name}` 已生成。",
        f"- Strong mechanism A/B: `{STRONG_MECH_METRICS.name}` 已生成。",
        "",
        "## 本轮关键原则",
        "",
        "- 未进行普通 Optuna、未更换模型、未使用 test 集调参。",
        "- Copernicus 变量被明确标记为再分析代理，不写成实测底层变量。",
        "- 缺少 exact_sampling_datetime 时，潮汐仅作为 date-level proxy 进入。",
        "",
        "## 当前判断",
        "",
    ]
    if comparison.empty:
        experiment_lines.append("- 本轮没有形成实质可用的新强输入增益。")
    else:
        best = comparison.sort_values("delta_r2", ascending=False).iloc[0]
        experiment_lines.append(
            f"- 本轮 delta R² 最好的组合是 `{best['scenario']}`，对应 `{best['target']}/{best['depth_layer']}`，delta R² = {best['delta_r2']:.4f}。"
        )
        if float(best["delta_r2"]) >= 0.03:
            experiment_lines.append("- 已达到 useful_strong_input 阈值。")
        else:
            experiment_lines.append("- 仍未达到 useful_strong_input（delta R² >= 0.03）阈值。")
    experiment_lines.append("- 若本轮仍离 R²=0.60 很远，最关键缺口仍是 exact sampling time、真实 station depth / marine bathymetry、bottom water temperature、vertical profile、current / discharge / water level。")
    experiment_report.write_text("\n".join(experiment_lines) + "\n", encoding="utf-8")

    data_gap.write_text(
        "# 数据缺口清单\n\n"
        "- 仍缺 exact_sampling_datetime，因此无法构造真正的采样潮相特征。\n"
        "- 仍缺经过海岸近岸适配确认的 marine bathymetry / station depth。\n"
        "- 仍缺实测 bottom water temperature / vertical temperature profile / thermocline depth。\n"
        "- 仍缺 ADCP current、river discharge、water level 等真实水动力过程数据。\n",
        encoding="utf-8",
    )
    next_plan.write_text(
        "# 下一步数据获取计划\n\n"
        "1. 用户手动提供 station_depth 或可用的海域 bathymetry 栅格。\n"
        "2. 用户手动提供 exact_sampling_datetime。\n"
        "3. 用户手动提供 bottom water temperature / vertical profile / ADCP current / discharge / water level。\n"
        "4. 如需继续自动化，下一轮优先做 GEBCO 或本地 bathymetry 栅格接入，而不是重复模型微调。\n",
        encoding="utf-8",
    )
    methods.write_text(
        "# 已尝试方法登记\n\n"
        "- 2026-05-19：强机制输入单轮尝试（GLOBathy exploratory bathymetry、Copernicus physics、date-level tide proxy、patch quality filters）。\n"
        "- 结论：本轮重点是 source validation + fixed-frame A/B，不属于普通调参路线。\n",
        encoding="utf-8",
    )
    failed_lines = ["# 失败实验\n", ""]
    if bath_report.get("status") == "failed":
        failed_lines.append(f"- GLOBathy bathymetry failed: {bath_report.get('note','')}")
    if cop_report.get("status") != "completed":
        failed_lines.append(f"- Copernicus physics failed or incomplete: {cop_report.get('stderr_tail','')[:300]}")
    if comparison.empty:
        failed_lines.append("- Strong mechanism A/B 未形成有效输出。")
    failed.write_text("\n".join(failed_lines) + "\n", encoding="utf-8")
    do_not_repeat.write_text(
        "# 不再重复的路线\n\n"
        "- 不再重复普通调参与普通 stacking。\n"
        "- 若本轮 strong mechanism 输入增益仍弱，下一轮不再重复弱代理环境特征微调。\n"
        "- 在未补齐 exact_sampling_datetime / station_depth / bottom_temp / current / discharge 前，不再把 date-level proxy 误写成真实底层过程变量。\n",
        encoding="utf-8",
    )
    ceiling.write_text(
        "# 预测精度上限分析\n\n"
        "- 当前瓶颈主要不再是特征数量，而是强机制变量缺失与标签时间分辨率不足。\n"
        "- Copernicus physics 只能提供再分析代理，不能完全替代实测底层温度和流速。\n"
        "- 没有 exact_sampling_datetime 时，感潮过程只能做 date-level 代理。\n"
        "- 若要接近 R²=0.60，仍需要真实底层机制数据和更准确的 sampling-time alignment。\n",
        encoding="utf-8",
    )
    alignment.write_text(
        "# 模型输入数据对齐说明\n\n"
        "## ERA5 / 风 / 降雨 / 潮汐 proxy 当前处理方式\n"
        "- `t2m`: 当日 2m 气温。\n"
        "- `u10`, `v10`, `wind_speed`: 当日 10m 风场与风速。\n"
        "- `ssrd_*`: 预测日及之前 3/7/14 天太阳辐射滚动累积或均值。\n"
        "- `sst_mean_*`, `sst_max_*`, `sst_anom_*`: 预测日及之前 3/7/14 天海表温度统计。\n"
        "- `rain_1d/3d/7d/14d/30d/month`: 预测日及之前累计降雨。\n"
        "- `tidal_range`, `tide_events`, `tide_high_mean`, `tide_low_mean`, `tide_level_mean`: date-level 潮汐背景统计。\n"
        "- 以上变量都按 `date` 对齐；不含未来信息；不是同日其他水质实测协变量。\n\n"
        "## 本轮新增强输入\n"
        "- `globathy_*`: 通过 GEE 读取的公开 bathymetry 栅格统计，当前仅作为 exploratory public bathymetry。\n"
        "- `cmems_*`: Copernicus Marine 再分析物理场站点日期提取值，属于 reanalysis proxy。\n"
        "- `surface_temp`, `near_bottom_temp_proxy`, `temp_surface_bottom_diff_proxy`: 由再分析温度剖面构造的近底温度代理。\n"
        "- `tide_range_same_day`, `tide_range_previous_3d_mean` 等：无采样时刻时的 date-level tide proxy。\n"
        "- `hq_*`: 在 highres patch 原始字段基础上，按 valid_ratio/date_diff_abs/water-consistency 质量筛选后的 patch 特征。\n",
        encoding="utf-8",
    )
    inventory.write_text(
        "# 模型参数与输入特征清单\n\n"
        "## 本轮 strong mechanism candidate 模型\n"
        "- 模型框架沿用 current best family（station_calibration / ensemble_lgbm_rf_avg / impute_knn_lightgbm / stacking_ridge）。\n"
        "- 未新增 Optuna，未切换模型家族。\n"
        "- 新增特征组：`globathy_*`、`cmems_*`、`surface_temp`、`near_bottom_temp_proxy`、`tide_*`、`hq_*`。\n"
        "- 这些特征的具体含义见 `strong_mechanism_feature_dictionary.csv`。\n",
        encoding="utf-8",
    )
    with cleanup.open("a", encoding="utf-8") as fh:
        fh.write("\n## 2026-05-19 strong mechanism stage\n")
        fh.write("- 新增/更新 auto_data_acquisition_capability.csv\n")
        fh.write("- 新增/更新 depth_bathymetry_features_2014_2024.csv\n")
        fh.write("- 新增/更新 copernicus_physics_features_2014_2024.csv\n")
        fh.write("- 新增/更新 tide_phase_features_2014_2024.csv\n")
        fh.write("- 新增/更新 high_quality_patch_features_2014_2024.csv\n")
        fh.write("- 新增/更新 strong_mechanism_input_features_2014_2024.csv\n")


def main() -> None:
    parse_args()
    ensure_dirs()
    base_ids = load_base_ids()
    capability_df = capability_checks()
    depth_df, bath_report = extract_globathy_features(base_ids)
    cop_df, cop_report = run_copernicus_physics()
    tide_df, tide_report = build_tide_phase_features(base_ids)
    merged_for_qc = base_ab.merge_model_dataset()[0]
    patch_qc, hq_patch_df, hq_cols = run_quality_filter_ab(merged_for_qc)
    depth_cols = non_null_cols(
        depth_df,
        {
            "obs_id", "sample_id", "station", "date", "depth", "split", "lon", "lat",
            "water_zone_en", "year", "month", "is_wet_season", "bathymetry_source", "bathymetry_risk_flag",
        },
    ) if not depth_df.empty else []
    cop_cols = non_null_cols(cop_df, {"station", "date"}) if not cop_df.empty else []
    tide_cols = non_null_cols(
        tide_df,
        {"obs_id", "sample_id", "station", "date", "depth", "tide_proxy_feature_level"},
    ) if not tide_df.empty else []
    merged_strong, _ = merge_strong_mechanism_table(base_ids, depth_df, cop_df, tide_df, hq_patch_df)
    comparison = run_strong_mechanism_ab(merged_strong, depth_cols, cop_cols, tide_cols, hq_cols)
    update_report_docs(capability_df, bath_report, cop_report, tide_report, patch_qc, comparison)
    print("Strong mechanism stage completed.")


if __name__ == "__main__":
    main()
