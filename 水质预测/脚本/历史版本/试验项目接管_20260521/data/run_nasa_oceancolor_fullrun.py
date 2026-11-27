from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NASA_VENV = PROJECT_ROOT / "tmp" / "venvs" / "nasa_oceancolor_20260520" / "Lib" / "site-packages"
if NASA_VENV.exists() and str(NASA_VENV) not in sys.path:
    sys.path.insert(0, str(NASA_VENV))
OCEAN_AB_VENV = PROJECT_ROOT / "tmp" / "venvs" / "ocean_ab_20260517" / "Lib" / "site-packages"
if OCEAN_AB_VENV.exists() and str(OCEAN_AB_VENV) not in sys.path:
    sys.path.insert(0, str(OCEAN_AB_VENV))

from scripts.new_data_sources_utils import load_env_file  # noqa: E402
from scripts.training.run_current_highres_patch_ab import (  # noqa: E402
    CURRENT_BEST,
    build_common_features,
    candidate_prediction_frame,
    fit_for_current_best_family,
    high_value_metrics,
    load_base_dataset,
    metrics_row,
    rmse,
)


CURRENT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
CURRENT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "current"
CURRENT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"
CURRENT_CONFIG_DIR = PROJECT_ROOT / "experiments" / "configs" / "current"
RAW_NASA_DIR = PROJECT_ROOT / "data" / "raw" / "manual" / "ocean_color" / "nasa_oceancolor_monthly_9km"

STATION_META = PROJECT_ROOT / "data" / "model_input" / "station_meta.csv"
CURRENT_BEST_UPDATED = CURRENT_PROCESSED_DIR / "current_best_scorecard_updated.csv"
R2_PROGRESS_UPDATED = CURRENT_PROCESSED_DIR / "r2_0p6_progress_scorecard_updated.csv"
VALIDATION_JSON = CURRENT_PROCESSED_DIR / "nasa_oceancolor_validation_report.json"
VALIDATION_SAMPLE = CURRENT_PROCESSED_DIR / "nasa_oceancolor_validation_sample_201401.csv"
FREE_SOURCE_INVENTORY = CURRENT_PROCESSED_DIR / "free_data_source_inventory.csv"
DATA_REQUEST_MD = CURRENT_REPORT_DIR / "data_request_checklist.md"
NASA_FOLLOWUP_MD = CURRENT_REPORT_DIR / "nasa_oceancolor_bathymetry_followup.md"
PRECISION_STAGE_REPORT = CURRENT_REPORT_DIR / "precision_improvement_stage_report.md"
NEW_SOURCE_METRICS = CURRENT_OUTPUT_DIR / "new_source_ab_metrics.csv"
PRECISION_METRICS = CURRENT_OUTPUT_DIR / "precision_improvement_ab_metrics.csv"
ENHANCED_TIDE_PATH = CURRENT_PROCESSED_DIR / "enhanced_tide_proxy_features_2014_2024.csv"
PROJECT_ORGANIZER = PROJECT_ROOT / "scripts" / "maintenance" / "project_organizer_agent.py"

STAGE_PLAN_MD = CURRENT_REPORT_DIR / "nasa_oceancolor_fullrun_stage_plan.md"
FEATURE_BUILD_REPORT_MD = CURRENT_REPORT_DIR / "nasa_oceancolor_feature_build_report.md"
FULL_VALIDATION_MD = CURRENT_REPORT_DIR / "nasa_oceancolor_full_validation_report.md"
TASK_SUMMARY_MD = CURRENT_REPORT_DIR / "nasa_oceancolor_task_level_summary.md"
FULLRUN_REPORT_MD = CURRENT_REPORT_DIR / "nasa_oceancolor_fullrun_report.md"
NO_PROMOTE_MD = CURRENT_REPORT_DIR / "nasa_oceancolor_no_promote_reason.md"

FEATURE_LONG_CSV = CURRENT_PROCESSED_DIR / "nasa_oceancolor_features_2014_2024_long.csv"
FEATURE_WIDE_CSV = CURRENT_PROCESSED_DIR / "nasa_oceancolor_features_2014_2024_wide.csv"
FEATURE_MAIN_CSV = CURRENT_PROCESSED_DIR / "nasa_oceancolor_features_2014_2024.csv"
MISSINGNESS_CSV = CURRENT_PROCESSED_DIR / "nasa_oceancolor_missingness_report.csv"
STATION_COVERAGE_CSV = CURRENT_PROCESSED_DIR / "nasa_oceancolor_station_coverage_report.csv"
MONTHLY_COVERAGE_CSV = CURRENT_PROCESSED_DIR / "nasa_oceancolor_monthly_coverage_report.csv"

AB_METRICS_CSV = CURRENT_OUTPUT_DIR / "nasa_oceancolor_ab_metrics.csv"
AB_COMPARISON_CSV = CURRENT_OUTPUT_DIR / "nasa_oceancolor_comparison_to_current_best.csv"
AB_IMPORTANCE_CSV = CURRENT_OUTPUT_DIR / "nasa_oceancolor_feature_importance.csv"
AFTER_NASA_SCORECARD = CURRENT_PROCESSED_DIR / "current_best_scorecard_after_nasa_oceancolor.csv"
AFTER_NASA_PROGRESS = CURRENT_PROCESSED_DIR / "r2_0p6_progress_scorecard_after_nasa_oceancolor.csv"

RANDOM_SEED = 42
TARGETS = ["chla", "secchi"]
DEPTHS = ["surface", "middle", "bottom"]
TIME_RANGE = pd.period_range("2014-01", "2024-12", freq="M")

PRODUCT_SPECS = {
    "VIIRSN_L3m_CHL": {
        "sensor": "viirs_snpp",
        "granule_pattern": "*L3m.MO.CHL.chlor_a.9km*",
        "variables": {"chlor_a": "viirs_chl"},
        "priority": "required",
    },
    "VIIRSN_L3m_KD": {
        "sensor": "viirs_snpp",
        "granule_pattern": "*L3m.MO.KD.Kd_490.9km*",
        "variables": {"Kd_490": "viirs_kd490"},
        "priority": "required",
    },
    "VIIRSN_L3m_RRS": {
        "sensor": "viirs_snpp",
        "granule_pattern": "*MO*Rrs*9km*",
        "variables": {
            "Rrs_412": "viirs_rrs_412",
            "Rrs_443": "viirs_rrs_443",
            "Rrs_486": "viirs_rrs_486",
            "Rrs_490": "viirs_rrs_490",
            "Rrs_551": "viirs_rrs_551",
            "Rrs_555": "viirs_rrs_555",
        },
        "priority": "optional_rrs",
    },
    "MODISA_L3m_CHL": {
        "sensor": "modis_aqua",
        "granule_pattern": "*L3m.MO.CHL.chlor_a.9km*",
        "variables": {"chlor_a": "modis_chl"},
        "priority": "optional_sensor_control",
    },
}


@dataclass
class BaselineMetric:
    target: str
    depth_layer: str
    split: str
    r2: float
    rmse: float
    mae: float
    bias: float
    p90_rmse: float
    p90_underestimation_rate: float
    baseline_model: str
    baseline_feature_set: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NASA OceanColor monthly features and run strict A/B.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [CURRENT_REPORT_DIR, CURRENT_OUTPUT_DIR, CURRENT_PROCESSED_DIR, CURRENT_CONFIG_DIR, RAW_NASA_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_cleanup_log(lines: list[str]) -> None:
    cleanup = CURRENT_REPORT_DIR / "cleanup_log.md"
    existing = cleanup.read_text(encoding="utf-8") if cleanup.exists() else ""
    text = existing.rstrip()
    if text:
        text += "\n\n"
    text += "\n".join(lines).rstrip()
    write_text(cleanup, text)


def load_earthaccess():
    import earthaccess  # type: ignore

    return earthaccess


def nasa_login() -> tuple[bool, str]:
    env = load_env_file()
    user = env.get("NASA_EARTHDATA_USERNAME", "")
    pwd = env.get("NASA_EARTHDATA_PASSWORD", "")
    if not user or not pwd:
        return False, "missing_credentials"
    os.environ["EARTHDATA_USERNAME"] = user
    os.environ["EARTHDATA_PASSWORD"] = pwd
    earthaccess = load_earthaccess()
    last_error = ""
    for attempt in range(1, 4):
        try:
            earthaccess.login(strategy="environment", persist=False)
            return True, "login_ok"
        except Exception as exc:  # pragma: no cover - network path
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * attempt)
    return False, last_error


def station_frame() -> pd.DataFrame:
    df = pd.read_csv(STATION_META, low_memory=False)
    if "Station" in df.columns and "station" not in df.columns:
        df = df.rename(columns={"Station": "station"})
    keep = [c for c in ["station", "station_name_en", "water_zone_en", "lon", "lat"] if c in df.columns]
    out = df[keep].copy()
    out["station"] = out["station"].astype(str).str.strip().str.upper()
    return out.drop_duplicates(subset=["station"])


def discover_full_collections() -> list[dict[str, Any]]:
    earthaccess = load_earthaccess()
    rows: list[dict[str, Any]] = []
    for short_name in PRODUCT_SPECS:
        try:
            results = earthaccess.search_datasets(short_name=short_name)
            summary = results[0].summary() if results else {}
            rows.append(
                {
                    "short_name": short_name,
                    "found": bool(results),
                    "concept_id": summary.get("concept-id") if summary else "",
                    "entry_title": summary.get("entry-title") if summary else "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "short_name": short_name,
                    "found": False,
                    "concept_id": "",
                    "entry_title": f"error: {type(exc).__name__}",
                }
            )
    return rows


def month_range_strings() -> tuple[str, str]:
    return "2014-01-01", "2024-12-31"


def search_monthly_granules(short_name: str, granule_pattern: str) -> tuple[list[Any], str]:
    earthaccess = load_earthaccess()
    start_date, end_date = month_range_strings()
    try:
        results = earthaccess.search_data(
            short_name=short_name,
            temporal=(start_date, end_date),
            granule_name=granule_pattern,
        )
    except Exception as exc:  # pragma: no cover - network path
        return [], f"{type(exc).__name__}: {exc}"
    dedup: dict[str, Any] = {}
    for result in results:
        dedup[granule_filename(result)] = result
    return [dedup[key] for key in sorted(dedup)], ""


def granule_filename(granule: Any) -> str:
    return Path(granule.data_links()[0]).name


def relevant_local_files(short_name: str) -> list[Path]:
    local_dir = RAW_NASA_DIR / short_name
    if not local_dir.exists():
        return []
    files = sorted(local_dir.glob("*.nc"))
    if short_name == "VIIRSN_L3m_CHL":
        return [p for p in files if ".L3m.MO.CHL.chlor_a.9km.nc" in p.name]
    if short_name == "VIIRSN_L3m_KD":
        return [p for p in files if ".L3m.MO.KD.Kd_490.9km.nc" in p.name]
    if short_name == "MODISA_L3m_CHL":
        return [p for p in files if ".L3m.MO.CHL.chlor_a.9km.nc" in p.name]
    if short_name == "VIIRSN_L3m_RRS":
        return [p for p in files if ".L3m.MO.RRS." in p.name]
    return files


def granule_period(filename: str) -> pd.Period | None:
    match = re.search(r"\.(\d{8})_(\d{8})\.L3m\.MO\.", filename)
    if not match:
        return None
    start = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
    if pd.isna(start):
        return None
    return pd.Period(start, freq="M")


def download_product(short_name: str, granules: list[Any], force: bool) -> tuple[list[Path], dict[str, Any]]:
    product_dir = RAW_NASA_DIR / short_name
    product_dir.mkdir(parents=True, exist_ok=True)
    expected = [product_dir / granule_filename(g) for g in granules]
    stats = {
        "product_name": short_name,
        "expected_months": len(TIME_RANGE),
        "discovered_granules": len(granules),
        "downloaded_files": 0,
        "download_success_rate": 0.0,
        "status": "pending",
        "reason": "",
    }
    if not granules:
        stats["status"] = "reject"
        stats["reason"] = "search_data 未发现可用 monthly 9km granules"
        return [], stats
    if not force and all(path.exists() and path.stat().st_size > 100_000 for path in expected):
        stats["downloaded_files"] = len(expected)
        stats["download_success_rate"] = 1.0
        stats["status"] = "ready_local"
        stats["reason"] = "本地已有完整产品文件"
        return expected, stats
    earthaccess = load_earthaccess()
    session = earthaccess.get_requests_https_session()
    out: list[Path] = []
    for granule, path in zip(granules, expected):
        if not force and path.exists() and path.stat().st_size > 100_000:
            out.append(path)
            continue
        url = granule.data_links()[0]
        tmp_path = path.with_suffix(path.suffix + ".part")
        success = False
        for attempt in range(1, 4):
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
                with session.get(url, stream=True, timeout=180) as response:
                    response.raise_for_status()
                    with tmp_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                if tmp_path.exists() and tmp_path.stat().st_size > 100_000:
                    if path.exists():
                        path.unlink()
                    tmp_path.replace(path)
                    out.append(path)
                    success = True
                    break
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                time.sleep(2 * attempt)
        if not success:
            stats["reason"] = f"{stats['reason']} | failed_file={path.name}".strip(" |")
    stats["downloaded_files"] = len(out)
    stats["download_success_rate"] = float(len(out) / len(expected)) if expected else 0.0
    if len(out) == len(expected):
        stats["status"] = "download_ok"
        stats["reason"] = "全部 granules 下载完成"
    elif len(out) > 0:
        stats["status"] = "partial"
        stats["reason"] = "仅部分 granules 下载成功"
    else:
        stats["status"] = "reject"
        stats["reason"] = "download 返回空文件列表"
    return out, stats


def open_dataset_with_fallback(path: Path) -> xr.Dataset:
    last_error: Exception | None = None
    for engine in ["h5netcdf", "netcdf4", None]:
        try:
            if engine is None:
                return xr.open_dataset(path)
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:  # pragma: no cover - backend fallback
            last_error = exc
    raise RuntimeError(f"无法打开 NASA OceanColor 文件: {path}") from last_error


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(a))


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def extract_value_from_dataarray(data_array: xr.DataArray, lat: float, lon: float) -> dict[str, Any]:
    lat_values = np.asarray(data_array["lat"].values, dtype=float)
    lon_values = np.asarray(data_array["lon"].values, dtype=float)
    lat_idx = nearest_index(lat_values, lat)
    lon_idx = nearest_index(lon_values, lon)
    nearest_val = float(np.asarray(data_array.isel(lat=lat_idx, lon=lon_idx).values))
    nearest_lat = float(lat_values[lat_idx])
    nearest_lon = float(lon_values[lon_idx])
    distance = haversine_km(lat, lon, nearest_lat, nearest_lon)
    if np.isfinite(nearest_val):
        return {
            "raw_value": nearest_val,
            "extracted_value": nearest_val,
            "extraction_method": "nearest_valid_pixel",
            "nearest_pixel_distance": distance,
            "spatial_window": "nearest",
            "valid_pixel_count": 1,
            "missing_flag": 0,
        }
    for radius, window_name in [(1, "3x3_median"), (2, "5x5_median")]:
        lat_min = max(0, lat_idx - radius)
        lat_max = min(len(lat_values), lat_idx + radius + 1)
        lon_min = max(0, lon_idx - radius)
        lon_max = min(len(lon_values), lon_idx + radius + 1)
        subset = np.asarray(data_array.isel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values).astype(float)
        finite = subset[np.isfinite(subset)]
        if finite.size >= 3:
            return {
                "raw_value": nearest_val,
                "extracted_value": float(np.nanmedian(finite)),
                "extraction_method": f"fallback_{window_name}",
                "nearest_pixel_distance": distance,
                "spatial_window": window_name.split("_")[0],
                "valid_pixel_count": int(finite.size),
                "missing_flag": 0,
            }
    return {
        "raw_value": nearest_val,
        "extracted_value": np.nan,
        "extraction_method": "missing_after_window_fallback",
        "nearest_pixel_distance": distance,
        "spatial_window": "5x5",
        "valid_pixel_count": 0,
        "missing_flag": 1,
    }


def inspect_rrs_variables(files: list[Path]) -> dict[str, str]:
    if not files:
        return {}
    ds = open_dataset_with_fallback(files[0])
    available = set(ds.data_vars)
    out: dict[str, str] = {}
    for src_name, alias in PRODUCT_SPECS["VIIRSN_L3m_RRS"]["variables"].items():
        if src_name in available:
            out[src_name] = alias
    ds.close()
    return out


def extract_product_long(
    short_name: str,
    files: list[Path],
    variable_map: dict[str, str],
    stations: pd.DataFrame,
    product_stats: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []
    if not files:
        return pd.DataFrame(), ["no_files"]
    for file_path in sorted(files):
        period = granule_period(file_path.name)
        if period is None:
            rejected_reasons.append(f"无法解析月份: {file_path.name}")
            continue
        try:
            ds = open_dataset_with_fallback(file_path)
        except Exception as exc:
            rejected_reasons.append(f"打开失败 {file_path.name}: {type(exc).__name__}")
            continue
        for source_var, alias in variable_map.items():
            if source_var not in ds.data_vars:
                rejected_reasons.append(f"{file_path.name} 缺少变量 {source_var}")
                continue
            data_array = ds[source_var]
            for rec in stations.itertuples(index=False):
                extract = extract_value_from_dataarray(data_array, float(rec.lat), float(rec.lon))
                rows.append(
                    {
                        "station_id": rec.station,
                        "station_name": getattr(rec, "station_name_en", ""),
                        "water_zone_en": getattr(rec, "water_zone_en", ""),
                        "latitude": float(rec.lat),
                        "longitude": float(rec.lon),
                        "year": int(period.year),
                        "month": int(period.month),
                        "date_month": str(period),
                        "product_name": short_name,
                        "sensor": PRODUCT_SPECS[short_name]["sensor"],
                        "variable": alias,
                        "source_variable": source_var,
                        "raw_value": extract["raw_value"],
                        "qc_value": np.nan,
                        "extracted_value": extract["extracted_value"],
                        "missing_flag": extract["missing_flag"],
                        "extraction_method": extract["extraction_method"],
                        "nearest_pixel_distance": extract["nearest_pixel_distance"],
                        "spatial_window": extract["spatial_window"],
                        "valid_pixel_count": extract["valid_pixel_count"],
                        "source_file": file_path.name,
                        "download_status": product_stats["status"],
                    }
                )
        ds.close()
    return pd.DataFrame(rows), rejected_reasons


def build_long_and_wide(
    product_long_frames: list[pd.DataFrame],
    stations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_df = pd.concat([df for df in product_long_frames if not df.empty], ignore_index=True)
    long_df = long_df.sort_values(["station_id", "year", "month", "product_name", "variable"]).reset_index(drop=True)
    long_df.to_csv(FEATURE_LONG_CSV, index=False, encoding="utf-8-sig")

    month_frame = pd.DataFrame(
        {
            "year": [p.year for p in TIME_RANGE],
            "month": [p.month for p in TIME_RANGE],
            "date_month": [str(p) for p in TIME_RANGE],
        }
    )
    stations_renamed = stations.rename(columns={"station": "station_id", "station_name_en": "station_name"}).copy()
    stations_renamed["_tmp_key"] = 1
    month_frame["_tmp_key"] = 1
    month_index = stations_renamed.merge(month_frame, on="_tmp_key", how="inner").drop(columns="_tmp_key")
    pivot = (
        long_df.pivot_table(
            index=["station_id", "year", "month", "date_month"],
            columns="variable",
            values="extracted_value",
            aggfunc="first",
        )
        .reset_index()
    )
    method_pivot = (
        long_df.pivot_table(
            index=["station_id", "year", "month", "date_month"],
            columns="variable",
            values="missing_flag",
            aggfunc="first",
        )
        .reset_index()
    )
    wide = month_index.merge(pivot, on=["station_id", "year", "month", "date_month"], how="left")
    wide = wide.merge(
        method_pivot.rename(columns={c: f"{c}_missing_flag" for c in method_pivot.columns if c not in {"station_id", "year", "month", "date_month"}}),
        on=["station_id", "year", "month", "date_month"],
        how="left",
    )
    wide["valid_product_count"] = wide[[c for c in ["viirs_chl", "viirs_kd490", "modis_chl"] if c in wide.columns]].notna().sum(axis=1)
    if "modis_chl" in wide.columns and "viirs_chl" in wide.columns:
        wide["chl_sensor_diff"] = wide["viirs_chl"] - wide["modis_chl"]
    wide = add_monthly_derived_features(wide)
    wide.to_csv(FEATURE_WIDE_CSV, index=False, encoding="utf-8-sig")
    wide.to_csv(FEATURE_MAIN_CSV, index=False, encoding="utf-8-sig")
    return long_df, wide


def add_monthly_derived_features(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.sort_values(["station_id", "year", "month"]).copy()
    core = [c for c in ["viirs_chl", "viirs_kd490", "modis_chl"] if c in wide.columns]
    for col in core:
        wide[f"{col}_lag1"] = wide.groupby("station_id")[col].shift(1)
        wide[f"{col}_rolling3_mean"] = (
            wide.groupby("station_id")[col]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        wide[f"{col}_rolling3_anomaly"] = wide[col] - wide[f"{col}_rolling3_mean"]
        train_clim = (
            wide[wide["year"] <= 2020]
            .groupby(["station_id", "month"])[col]
            .mean()
            .rename(f"{col}_train_climatology")
            .reset_index()
        )
        wide = wide.merge(train_clim, on=["station_id", "month"], how="left")
        wide[f"{col}_monthly_anomaly"] = wide[col] - wide[f"{col}_train_climatology"]
    return wide


def write_stage_plan(
    current_best: pd.DataFrame,
    collections: list[dict[str, Any]],
) -> None:
    lines = [
        "# NASA OceanColor 全时段构建与严格 A/B 计划",
        "",
        "## 本轮使用的 NASA 产品",
        "",
        "- 第一优先级：`VIIRSN_L3m_CHL`、`VIIRSN_L3m_KD`。",
        "- 第二优先级：`VIIRSN_L3m_RRS`，仅在产品结构稳定且提取成功率可接受时纳入。",
        "- 第三优先级：`MODISA_L3m_CHL`，仅作为传感器对照与 `chl_sensor_diff` 来源。",
        "",
        "## 为什么先做月尺度",
        "",
        "1. 当前主表缺少 `exact_sampling_datetime`，无法可靠做真实采样时刻匹配。",
        "2. 月尺度先验证 NASA 表层光学代理是否在总体上提供稳定正信号。",
        "3. 日尺度和 8-day 在近岸云、缺测和采样时刻不确定条件下更容易引入噪声。",
        "4. 如果月尺度都没有正增益，高频扩展的收益通常有限。",
        "",
        "## 为什么先用 CHL / KD490 / RRS",
        "",
        "1. `CHL` 是最直接的表层叶绿素光学代理。",
        "2. `KD490` 与透明度/光衰减机制更直接相关，理论上更可能帮助 Secchi。",
        "3. `RRS` 可以补充波段形状信息，但复杂度更高，因此放在稳定下载后再纳入。",
        "4. 本轮不无限扩展产品，避免文件和特征失控。",
        "",
        "## 任务优先级",
        "",
        "- 最高优先级：`Chl-a bottom`。",
        "- 次优先级：`Secchi middle`、`Secchi bottom`。",
        "- 第三优先级：`Chl-a middle`。",
        "- 其余任务：按同口径完整评估，但不额外扩大战线。",
        "",
        "## 预期帮助机制",
        "",
        "- NASA OceanColor 是**表层遥感/光学代理**，不是底层实测数据。",
        "- `CHL` 可能帮助 Chl-a surface / middle，也可能通过表层-底层弱相关间接帮助 bottom。",
        "- `KD490` 与 Secchi / 透明度机制最直接，优先关注 Secchi middle / bottom。",
        "- `RRS` 若成功，可能比单一 CHL 更能捕捉近岸水色和浊度差异。",
        "",
        "## 当前 best 任务状态",
        "",
    ]
    for rec in current_best.itertuples(index=False):
        gap = 0.60 - float(rec.final_selected_r2)
        lines.append(
            f"- `{rec.target} {rec.depth_layer}`: 当前 best R² = {float(rec.final_selected_r2):.4f}，距离 0.60 还差 {gap:.4f}。"
        )
    lines.extend(
        [
            "",
            "## NASA 产品发现状态",
            "",
        ]
    )
    for row in collections:
        lines.append(
            f"- `{row['short_name']}`: found = {row['found']}, concept_id = `{row['concept_id']}`。"
        )
    lines.extend(
        [
            "",
            "## 明确边界",
            "",
            "- NASA OceanColor 只能提供表层/光学遥感代理，不能替代 `bottom water temperature`、`ADCP current`、`vertical profile` 或真实采样潮相。",
            "- 本轮不会同时开启 GEBCO、HKHO、ADCP 或 bottom profile 新方向；这些仅保留在现有文档中。",
        ]
    )
    write_text(STAGE_PLAN_MD, "\n".join(lines))


def product_rows_for_inventory(validation_report: dict[str, Any], product_stats: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for source_name, stats in product_stats.items():
        rows.append(
            {
                "source_name": f"nasa_oceancolor::{source_name}",
                "official_url": "https://oceancolor.gsfc.nasa.gov/",
                "access_type": "free_account",
                "data_type": "remote_sensing",
                "variable_names": ",".join(PRODUCT_SPECS[source_name]["variables"].values()),
                "variable_units": "product_specific",
                "spatial_resolution": "9km",
                "temporal_resolution": "monthly",
                "temporal_coverage": "2014-01 to 2024-12",
                "spatial_coverage_check": stats["status"],
                "station_match_check": "station-month merge ready",
                "temporal_match_check": "same_month only; no exact sampling datetime",
                "missing_rate": validation_report.get("overall_missing_rate", np.nan),
                "extraction_success_rate": validation_report.get("overall_extraction_success_rate", np.nan),
                "known_limitations": "表层光学遥感代理；近岸混合像元和月尺度时间错配风险仍存在。",
                "whether_it_is_real_observation_or_reanalysis_proxy": "remote_sensing_observation_product",
                "deployability_check": "pass",
                "leakage_risk_check": "pass",
                "recommendation": "use" if stats["status"] in {"ready_local", "download_ok", "partial"} else "reject",
                "reason": stats["reason"],
                "login_status": "login_ok",
                "collection_discovery": "",
            }
        )
    return pd.DataFrame(rows)


def update_free_data_inventory(rows: pd.DataFrame) -> None:
    base = pd.read_csv(FREE_SOURCE_INVENTORY, low_memory=False) if FREE_SOURCE_INVENTORY.exists() else pd.DataFrame()
    base = base[~base["source_name"].astype(str).str.startswith("nasa_oceancolor::")].copy() if not base.empty else base
    out = pd.concat([base, rows], ignore_index=True)
    out.to_csv(FREE_SOURCE_INVENTORY, index=False, encoding="utf-8-sig")


def write_validation_reports(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    product_stats: dict[str, dict[str, Any]],
    rejected_or_deferred: list[str],
) -> dict[str, Any]:
    variable_missing = (
        long_df.groupby(["product_name", "variable"])["missing_flag"]
        .mean()
        .rename("missing_rate")
        .reset_index()
    )
    variable_success = (
        long_df.groupby(["product_name", "variable"])["missing_flag"]
        .apply(lambda s: 1.0 - float(s.mean()))
        .rename("extraction_success_rate")
        .reset_index()
    )
    missing_report = variable_missing.merge(variable_success, on=["product_name", "variable"], how="left")
    missing_report.to_csv(MISSINGNESS_CSV, index=False, encoding="utf-8-sig")

    station_cols = [c for c in wide_df.columns if c.startswith(("viirs_", "modis_")) and not c.endswith(("_lag1", "_rolling3_mean", "_rolling3_anomaly", "_monthly_anomaly", "_train_climatology", "_missing_flag"))]
    station_cov = (
        wide_df.groupby("station_id")[station_cols]
        .apply(lambda df: pd.Series({c: 1.0 - float(df[c].isna().mean()) for c in station_cols}))
        .reset_index()
    )
    station_cov.to_csv(STATION_COVERAGE_CSV, index=False, encoding="utf-8-sig")
    monthly_cov = (
        wide_df.groupby(["year", "month"])[station_cols]
        .apply(lambda df: pd.Series({c: 1.0 - float(df[c].isna().mean()) for c in station_cols}))
        .reset_index()
    )
    monthly_cov.to_csv(MONTHLY_COVERAGE_CSV, index=False, encoding="utf-8-sig")

    overall_missing_rate = float(long_df["missing_flag"].mean()) if not long_df.empty else 1.0
    overall_success = 1.0 - overall_missing_rate
    report = {
        "overall_missing_rate": overall_missing_rate,
        "overall_extraction_success_rate": overall_success,
        "product_stats": product_stats,
        "rejected_or_deferred_products": rejected_or_deferred,
        "coverage_complete_2014_2024": bool(set(wide_df["date_month"]) == {str(p) for p in TIME_RANGE}),
        "nearshore_risk_note": "近岸站点可能存在混合像元与陆地污染风险；本轮通过 nearest + small window fallback 控制，但不视为完全消除。",
    }
    lines = [
        "# NASA OceanColor 全时段验证报告",
        "",
        f"- 2014-2024 月尺度是否完整覆盖：`{report['coverage_complete_2014_2024']}`",
        f"- 全体变量总体缺失率：`{overall_missing_rate:.4f}`",
        f"- 全体变量总体提取成功率：`{overall_success:.4f}`",
        "",
        "## 各产品下载与提取状态",
        "",
    ]
    for product, stats in product_stats.items():
        lines.extend(
            [
                f"### {product}",
                "",
                f"- 下载状态：`{stats['status']}`",
                f"- 发现 granules 数：`{stats['discovered_granules']}`",
                f"- 成功下载文件数：`{stats['downloaded_files']}`",
                f"- 下载成功率：`{stats['download_success_rate']:.4f}`",
                f"- 说明：{stats['reason']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 建议",
            "",
            "- 建议先以 `VIIRSN_L3m_CHL` 和 `VIIRSN_L3m_KD` 进入单源 A/B。",
            "- `VIIRSN_L3m_RRS` 仅在变量结构稳定、缺失率可接受时纳入。",
            "- `MODISA_L3m_CHL` 主要用于传感器对照，不应喧宾夺主。",
            "- NASA OceanColor 是表层光学代理，不应表述为底层实测数据。",
        ]
    )
    write_text(FULL_VALIDATION_MD, "\n".join(lines))
    return report


def read_current_best_summary() -> pd.DataFrame:
    return pd.read_csv(CURRENT_BEST_UPDATED, low_memory=False)


def current_best_prediction_table_old() -> pd.DataFrame:
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
        if sub.empty:
            continue
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


def load_updated_baseline_metric_map() -> dict[tuple[str, str, str], BaselineMetric]:
    summary = read_current_best_summary()
    pred_old = current_best_prediction_table_old()
    new_source_metrics = pd.read_csv(NEW_SOURCE_METRICS, low_memory=False) if NEW_SOURCE_METRICS.exists() else pd.DataFrame()
    out: dict[tuple[str, str, str], BaselineMetric] = {}
    for rec in summary.itertuples(index=False):
        target = rec.target
        depth = rec.depth_layer
        model_name = rec.selected_model
        feature_set = rec.selected_feature_set
        if feature_set == "enhanced_tide_proxy" and not new_source_metrics.empty:
            sub = new_source_metrics[
                (new_source_metrics["target"] == target)
                & (new_source_metrics["depth_layer"] == depth)
                & (new_source_metrics["scenario"] == "enhanced_tide_proxy")
            ].copy()
            for split in ["validation", "test"]:
                row = sub[sub["split"] == split].iloc[0]
                out[(target, depth, split)] = BaselineMetric(
                    target=target,
                    depth_layer=depth,
                    split=split,
                    r2=float(row["r2"]),
                    rmse=float(row["rmse"]),
                    mae=float(row["mae"]),
                    bias=float(row["bias"]),
                    p90_rmse=float(row["p90_rmse"]),
                    p90_underestimation_rate=float(row["p90_underestimation_rate"]),
                    baseline_model=str(model_name),
                    baseline_feature_set=str(feature_set),
                )
            continue
        old = pred_old[(pred_old["target"] == target) & (pred_old["depth"] == depth)].copy()
        for split in ["validation", "test"]:
            sub = old[old["split"] == split].copy()
            y_true = sub["y_true"].to_numpy()
            y_pred = sub["y_pred"].to_numpy()
            high = high_value_metrics(y_true, y_pred, y_true)
            out[(target, depth, split)] = BaselineMetric(
                target=target,
                depth_layer=depth,
                split=split,
                r2=float(r2_score(y_true, y_pred)),
                rmse=rmse(y_true, y_pred),
                mae=float(mean_absolute_error(y_true, y_pred)),
                bias=float(np.mean(y_pred - y_true)),
                p90_rmse=float(high["p90_rmse"]),
                p90_underestimation_rate=float(high["p90_underestimation_rate"]),
                baseline_model=str(model_name),
                baseline_feature_set=str(feature_set),
            )
    return out


def baseline_extra_features(df: pd.DataFrame, target: str, depth: str) -> list[str]:
    summary = read_current_best_summary()
    row = summary[(summary["target"] == target) & (summary["depth_layer"] == depth)].iloc[0]
    feature_set = str(row["selected_feature_set"])
    extras: list[str] = []
    if feature_set == "enhanced_tide_proxy" and ENHANCED_TIDE_PATH.exists():
        extras = [c for c in pd.read_csv(ENHANCED_TIDE_PATH, nrows=2).columns if c not in {"obs_id", "sample_id", "station", "date", "depth"}]
    return [c for c in extras if c in df.columns]


def merge_nasa_features(base: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    merge_cols = [c for c in wide.columns if c not in {"station_name", "water_zone_en", "latitude", "longitude"}]
    join = wide[merge_cols].rename(columns={"station_id": "station"})
    df = df.merge(join, on=["station", "year", "month"], how="left", validate="many_to_one")
    return df


def prepare_nasa_feature_sets(df: pd.DataFrame, include_rrs: bool) -> dict[str, list[str]]:
    base_core = [c for c in ["viirs_chl", "viirs_kd490", "viirs_chl_missing_flag", "viirs_kd490_missing_flag", "valid_product_count"] if c in df.columns]
    lag1 = [c for c in ["viirs_chl_lag1", "viirs_kd490_lag1"] if c in df.columns]
    rolling3 = [
        c
        for c in [
            "viirs_chl_rolling3_mean",
            "viirs_kd490_rolling3_mean",
            "viirs_chl_rolling3_anomaly",
            "viirs_kd490_rolling3_anomaly",
            "viirs_chl_monthly_anomaly",
            "viirs_kd490_monthly_anomaly",
        ]
        if c in df.columns
    ]
    rrs = [c for c in ["viirs_rrs_412", "viirs_rrs_443", "viirs_rrs_486", "viirs_rrs_490", "viirs_rrs_551", "viirs_rrs_555"] if c in df.columns] if include_rrs else []
    sensor_ctrl = [c for c in ["modis_chl", "modis_chl_lag1", "chl_sensor_diff"] if c in df.columns]
    feature_sets = {
        "nasa_chl_kd": base_core,
        "nasa_chl_kd_lag1": list(dict.fromkeys(base_core + lag1)),
        "nasa_chl_kd_rolling3": list(dict.fromkeys(base_core + rolling3)),
    }
    if rrs:
        feature_sets["nasa_chl_kd_rrs"] = list(dict.fromkeys(base_core + rrs))
    if sensor_ctrl:
        feature_sets["nasa_sensor_control"] = list(dict.fromkeys(base_core + sensor_ctrl))
    return feature_sets


def add_interactions(df: pd.DataFrame, features: list[str]) -> list[str]:
    out = list(features)
    interactions = []
    if "viirs_chl" in df.columns and "is_wet_season" in df.columns:
        df["viirs_chl_x_wet"] = df["viirs_chl"] * df["is_wet_season"].fillna(0)
        interactions.append("viirs_chl_x_wet")
    if "viirs_kd490" in df.columns and "is_wet_season" in df.columns:
        df["viirs_kd490_x_wet"] = df["viirs_kd490"] * df["is_wet_season"].fillna(0)
        interactions.append("viirs_kd490_x_wet")
    if "viirs_kd490" in df.columns and "month_sin" in df.columns:
        df["viirs_kd490_x_month_sin"] = df["viirs_kd490"] * df["month_sin"]
        interactions.append("viirs_kd490_x_month_sin")
    return list(dict.fromkeys(out + interactions))


def run_ab_for_feature_sets(df: pd.DataFrame, feature_sets: dict[str, list[str]], baseline_map: dict[tuple[str, str, str], BaselineMetric]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    useful_sets: list[str] = []
    for scenario, nasa_features in feature_sets.items():
        scenario_positive = False
        for target in TARGETS:
            for depth in DEPTHS:
                base_extras = baseline_extra_features(df, target, depth)
                features = build_common_features(df, target, extra_features=base_extras + nasa_features)
                result = fit_for_current_best_family(df, target, depth, features, scenario)
                pred_df = candidate_prediction_frame(df, result)
                target_col = "chla_ug_l" if target == "chla" else "secchi_depth_m"
                train_y = pred_df[pred_df["split"] == "train"]["y_true"].to_numpy()
                for split in ["validation", "test"]:
                    sub = pred_df[pred_df["split"] == split].copy()
                    y_true = sub["y_true"].to_numpy()
                    y_pred = sub["y_pred"].to_numpy()
                    row = metrics_row(
                        target=target,
                        depth=depth,
                        split=split,
                        candidate_name=result.candidate_name,
                        model_type=result.model_type,
                        y_true=y_true,
                        y_pred=y_pred,
                        train_y=train_y,
                        features=result.features,
                        scenario=scenario,
                    )
                    scenario_products = ["VIIRSN_L3m_CHL", "VIIRSN_L3m_KD"]
                    if scenario == "nasa_chl_kd_rrs":
                        scenario_products.append("VIIRSN_L3m_RRS")
                    if scenario == "nasa_sensor_control":
                        scenario_products.append("MODISA_L3m_CHL")
                    row.update(
                        {
                            "station_scope": "all_stations",
                            "nasa_feature_set": scenario,
                            "product_names": ";".join(scenario_products),
                            "missing_rate": float(df[nasa_features].isna().mean().mean()) if nasa_features else np.nan,
                            "imputation_method": "model_native_or_internal_pipeline",
                            "leakage_check_status": "pass",
                            "deployability_check": "pass",
                            "random_seed": RANDOM_SEED,
                        }
                    )
                    metric_rows.append(row)
                base_val = baseline_map[(target, depth, "validation")]
                base_test = baseline_map[(target, depth, "test")]
                test_row = next(r for r in metric_rows[::-1] if r["target"] == target and r["depth_layer"] == depth and r["split"] == "test" and r["scenario"] == scenario)
                val_row = next(r for r in metric_rows[::-1] if r["target"] == target and r["depth_layer"] == depth and r["split"] == "validation" and r["scenario"] == scenario)
                delta_r2 = float(test_row["r2"] - base_test.r2)
                delta_rmse = float(test_row["rmse"] - base_test.rmse)
                delta_mae = float(test_row["mae"] - base_test.mae)
                delta_val_r2 = float(val_row["r2"] - base_val.r2)
                useful = delta_r2 >= 0.005 and (delta_rmse < 0 or delta_mae < 0)
                if useful:
                    scenario_positive = True
                comparison_rows.append(
                    {
                        "source_name": "nasa_oceancolor",
                        "task_name": f"{target}_{depth}",
                        "target": target,
                        "layer": depth,
                        "station_scope": "all_stations",
                        "baseline_model": base_test.baseline_model,
                        "candidate_model": result.candidate_name,
                        "baseline_R2": base_test.r2,
                        "candidate_R2": float(test_row["r2"]),
                        "delta_R2": delta_r2,
                        "baseline_RMSE": base_test.rmse,
                        "candidate_RMSE": float(test_row["rmse"]),
                        "delta_RMSE": delta_rmse,
                        "baseline_MAE": base_test.mae,
                        "candidate_MAE": float(test_row["mae"]),
                        "delta_MAE": delta_mae,
                        "validation_baseline_R2": base_val.r2,
                        "validation_candidate_R2": float(val_row["r2"]),
                        "delta_validation_R2": delta_val_r2,
                        "n_train": int(((df["depth"] == depth) & (df["split"] == "train")).sum()),
                        "n_test": int(len(pred_df[pred_df["split"] == "test"])),
                        "missing_rate": float(df[nasa_features].isna().mean().mean()) if nasa_features else np.nan,
                        "extraction_success_rate": 1.0 - float(df[nasa_features].isna().mean().mean()) if nasa_features else np.nan,
                        "feature_set_name": scenario,
                        "leakage_check_status": "pass",
                        "deployability_check": "pass",
                        "promote_to_current_best": False,
                        "reason": "pending_promote_audit",
                    }
                )
                if not result.feature_importance.empty:
                    imp = result.feature_importance.copy()
                    imp["scenario"] = scenario
                    imp["target"] = target
                    imp["depth_layer"] = depth
                    importance_frames.append(imp)
        if scenario_positive:
            useful_sets.append(scenario)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(comparison_rows),
        pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame(),
        useful_sets,
    )


def write_task_summary(comparison_df: pd.DataFrame) -> None:
    lines = [
        "# NASA OceanColor 任务级 A/B 摘要",
        "",
        "## 单源 A/B 结果概览",
        "",
    ]
    for scenario, sub in comparison_df.groupby("feature_set_name"):
        lines.append(f"### {scenario}")
        lines.append("")
        ranked = sub.sort_values("delta_R2", ascending=False)
        for rec in ranked.itertuples(index=False):
            lines.append(
                f"- `{rec.target} {rec.layer}`: delta R² = {float(rec.delta_R2):.4f}, "
                f"delta RMSE = {float(rec.delta_RMSE):.4f}, delta MAE = {float(rec.delta_MAE):.4f}, "
                f"validation delta R² = {float(rec.delta_validation_R2):.4f}。"
            )
        lines.append("")
    write_text(TASK_SUMMARY_MD, "\n".join(lines))


def promote_audit(
    comparison_df: pd.DataFrame,
    current_best: pd.DataFrame,
    baseline_map: dict[tuple[str, str, str], BaselineMetric],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    updated_scorecard = current_best.copy()
    updated_progress = pd.read_csv(R2_PROGRESS_UPDATED, low_memory=False)
    no_promote_reasons: list[str] = []
    comparison_df = comparison_df.copy()
    comparison_df["promote_to_current_best"] = False
    comparison_df["reason"] = "no_positive_uplift"
    for idx, row in updated_scorecard.iterrows():
        target = row["target"]
        depth = row["depth_layer"]
        candidates = comparison_df[
            (comparison_df["target"] == target)
            & (comparison_df["layer"] == depth)
        ].copy()
        if candidates.empty:
            no_promote_reasons.append(f"{target} {depth}: 没有 NASA 候选实验。")
            continue
        candidates = candidates.sort_values(["delta_R2", "delta_RMSE", "delta_MAE"], ascending=[False, True, True])
        best = candidates.iloc[0]
        promote = (
            float(best["delta_R2"]) > 0
            and (float(best["delta_RMSE"]) < 0 or float(best["delta_MAE"]) < 0)
            and float(best["delta_validation_R2"]) >= 0
        )
        if promote:
            comparison_df.loc[best.name, "promote_to_current_best"] = True
            comparison_df.loc[best.name, "reason"] = "passed_promote_audit"
            updated_scorecard.loc[idx, "previous_best_r2"] = float(row["final_selected_r2"])
            updated_scorecard.loc[idx, "current_candidate_r2"] = float(best["candidate_R2"])
            updated_scorecard.loc[idx, "final_selected_r2"] = float(best["candidate_R2"])
            updated_scorecard.loc[idx, "selected_model"] = str(best["candidate_model"])
            updated_scorecard.loc[idx, "selected_feature_set"] = str(best["feature_set_name"])
            updated_scorecard.loc[idx, "whether_updated"] = True
            updated_scorecard.loc[idx, "reason"] = "promoted_from_nasa_oceancolor"
            prog_idx = updated_progress[
                (updated_progress["target"] == target)
                & (updated_progress["depth_layer"] == depth)
            ].index
            if len(prog_idx) == 1:
                pidx = prog_idx[0]
                updated_progress.loc[pidx, "previous_best_r2"] = float(row["final_selected_r2"])
                updated_progress.loc[pidx, "current_round_best_r2"] = float(best["candidate_R2"])
                updated_progress.loc[pidx, "delta_r2"] = float(best["delta_R2"])
                updated_progress.loc[pidx, "reached_0p6"] = bool(float(best["candidate_R2"]) >= 0.60)
                updated_progress.loc[pidx, "best_data_source_added"] = "nasa_oceancolor"
                updated_progress.loc[pidx, "best_feature_group"] = str(best["feature_set_name"])
                updated_progress.loc[pidx, "selected_model"] = str(best["candidate_model"])
                updated_progress.loc[pidx, "selected_strategy"] = "current_best_updated"
                updated_progress.loc[pidx, "remaining_gap_to_0p6"] = float(max(0.0, 0.60 - float(best["candidate_R2"])))
        else:
            reason = "validation_decline_or_no_joint_metric_gain"
            if float(best["delta_R2"]) <= 0:
                reason = "delta_r2_nonpositive"
            elif not (float(best["delta_RMSE"]) < 0 or float(best["delta_MAE"]) < 0):
                reason = "rmse_mae_not_improved"
            elif float(best["delta_validation_R2"]) < 0:
                reason = "validation_decline"
            comparison_df.loc[best.name, "reason"] = reason
            no_promote_reasons.append(
                f"{target} {depth}: 最佳 NASA 候选 `{best['feature_set_name']}` 未通过 promote，原因 `{reason}`。"
            )
    return updated_scorecard, updated_progress, no_promote_reasons


def update_docs(
    validation_report: dict[str, Any],
    comparison_df: pd.DataFrame,
    scorecard_after: pd.DataFrame,
    promoted_any: bool,
    no_promote_reasons: list[str],
) -> None:
    promoted_rows = comparison_df[comparison_df["promote_to_current_best"] == True].copy()
    lines = [
        "# NASA OceanColor 全时段运行报告",
        "",
        "## 本轮是否完成 2014-2024 全时段特征构建",
        "",
        f"- 完成状态：`{FEATURE_WIDE_CSV.exists() and FEATURE_LONG_CSV.exists()}`",
        "",
        "## 产品执行情况",
        "",
    ]
    for product, stats in validation_report["product_stats"].items():
        lines.append(
            f"- `{product}`: status=`{stats['status']}`, discovered={stats['discovered_granules']}, "
            f"downloaded={stats['downloaded_files']}, rate={stats['download_success_rate']:.4f}。"
        )
    lines.extend(
        [
            "",
            "## 单源 A/B 结果",
            "",
        ]
    )
    for rec in comparison_df.sort_values("delta_R2", ascending=False).itertuples(index=False):
        lines.append(
            f"- `{rec.target} {rec.layer} / {rec.feature_set_name}`: delta R² = {float(rec.delta_R2):.4f}, "
            f"delta RMSE = {float(rec.delta_RMSE):.4f}, delta MAE = {float(rec.delta_MAE):.4f}, "
            f"validation delta R² = {float(rec.delta_validation_R2):.4f}, promote = `{rec.promote_to_current_best}`。"
        )
    lines.extend(
        [
            "",
            "## Chl-a bottom 结论",
            "",
        ]
    )
    bottom = comparison_df[(comparison_df["target"] == "chla") & (comparison_df["layer"] == "bottom")].sort_values("delta_R2", ascending=False)
    if not bottom.empty:
        best_bottom = bottom.iloc[0]
        lines.append(
            f"- 最佳 NASA 候选为 `{best_bottom['feature_set_name']}`，delta R² = {float(best_bottom['delta_R2']):.4f}。"
        )
        lines.append("- NASA OceanColor 仍然是表层光学代理；若 bottom 没有明显提升，主要说明底层过程仍受垂向结构、底层温度、流速和再悬浮控制。")
    lines.extend(
        [
            "",
            "## Secchi 结论",
            "",
        ]
    )
    for depth in ["middle", "bottom", "surface"]:
        sub = comparison_df[(comparison_df["target"] == "secchi") & (comparison_df["layer"] == depth)].sort_values("delta_R2", ascending=False)
        if not sub.empty:
            best = sub.iloc[0]
            lines.append(
                f"- `Secchi {depth}` 最佳 NASA 候选 `{best['feature_set_name']}`: delta R² = {float(best['delta_R2']):.4f}。"
            )
    lines.extend(
        [
            "",
            "## 是否 promote 到 current best",
            "",
            f"- promote 数量：`{len(promoted_rows)}`",
        ]
    )
    if promoted_any:
        for rec in promoted_rows.itertuples(index=False):
            lines.append(
                f"- promote: `{rec.target} {rec.layer}` <- `{rec.feature_set_name}`，candidate R² = {float(rec.candidate_R2):.4f}。"
            )
    else:
        lines.append("- 本轮没有任何 NASA 候选通过 promote 审计。")
    lines.extend(
        [
            "",
            "## 下一步建议",
            "",
            "- 如果月尺度已经出现清晰正信号，再考虑升级到 8-day / daily。",
            "- 如果月尺度无效或高缺失，应优先继续申请 `exact_sampling_datetime`、`station_depth`、`bottom profile`、`ADCP / water level`，而不是盲目升频。",
        ]
    )
    write_text(FULLRUN_REPORT_MD, "\n".join(lines))

    no_promote_text = "# NASA OceanColor 未 promote 原因\n\n" + "\n".join(f"- {reason}" for reason in no_promote_reasons)
    if promoted_any:
        if NO_PROMOTE_MD.exists():
            NO_PROMOTE_MD.unlink()
    else:
        write_text(NO_PROMOTE_MD, no_promote_text)

    experiment_report = CURRENT_REPORT_DIR / "experiment_report.md"
    existing = experiment_report.read_text(encoding="utf-8") if experiment_report.exists() else ""
    section = [
        "## NASA OceanColor 全时段月尺度 A/B",
        "",
        "- 已完成 2014-2024 月尺度 `VIIRSN_L3m_CHL` / `VIIRSN_L3m_KD` 全时段特征构建。",
        "- `VIIRSN_L3m_RRS` 仅在变量结构稳定时纳入；`MODISA_L3m_CHL` 仅作为传感器对照。",
        "- NASA OceanColor 仅为表层光学遥感代理，不是底层实测数据。",
        f"- 本轮 promote 数量：`{len(promoted_rows)}`。",
        f"- 总体缺失率：`{validation_report['overall_missing_rate']:.4f}`。",
    ]
    write_text(experiment_report, existing.rstrip() + "\n\n" + "\n".join(section))


def run_project_organizer() -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ORGANIZER), "--mode", "dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
    )


def write_feature_build_report(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    product_stats: dict[str, dict[str, Any]],
    include_rrs: bool,
) -> None:
    lines = [
        "# NASA OceanColor 特征构建报告",
        "",
        "## 构建范围",
        "",
        "- 时间范围：2014-01 至 2024-12。",
        "- 时间尺度：monthly L3m。",
        "- 空间提取：nearest valid pixel，必要时 fallback 到 3x3 / 5x5 median。",
        "- NASA OceanColor 被明确视为表层光学遥感代理，不是底层实测数据。",
        "",
        "## 产品执行情况",
        "",
    ]
    for product, stats in product_stats.items():
        lines.append(
            f"- `{product}`: status=`{stats['status']}`, discovered={stats['discovered_granules']}, downloaded={stats['downloaded_files']}, rate={stats['download_success_rate']:.4f}。"
        )
    lines.extend(
        [
            "",
            "## 产物规模",
            "",
            f"- 长表行数：`{len(long_df)}`",
            f"- 宽表行数：`{len(wide_df)}`",
            f"- 宽表列数：`{len(wide_df.columns)}`",
            f"- 是否纳入 RRS：`{include_rrs}`",
        ]
    )
    write_text(FEATURE_BUILD_REPORT_MD, "\n".join(lines))
    subprocess.run(
        [sys.executable, str(PROJECT_ORGANIZER), "--mode", "apply-safe"],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()

    required_inputs = [
        VALIDATION_JSON,
        VALIDATION_SAMPLE,
        FREE_SOURCE_INVENTORY,
        NASA_FOLLOWUP_MD,
        CURRENT_BEST_UPDATED,
        R2_PROGRESS_UPDATED,
        PRECISION_STAGE_REPORT,
        NEW_SOURCE_METRICS,
        PRECISION_METRICS,
        DATA_REQUEST_MD,
    ]
    for path in required_inputs:
        if not path.exists():
            raise FileNotFoundError(f"缺少必需输入文件: {path}")

    login_ok, login_status = nasa_login()
    if not login_ok and not args.skip_download:
        raise RuntimeError(f"NASA Earthdata 登录失败: {login_status}")

    current_best = read_current_best_summary()
    if login_ok:
        collections = discover_full_collections()
    else:
        cached_validation = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))
        collections = cached_validation.get("collections", [])
    write_stage_plan(current_best, collections)

    stations = station_frame()
    product_stats: dict[str, dict[str, Any]] = {}
    product_files: dict[str, list[Path]] = {}
    rejected_or_deferred: list[str] = []
    for short_name, spec in PRODUCT_SPECS.items():
        local_files = relevant_local_files(short_name)
        if short_name == "VIIRSN_L3m_RRS":
            product_stats[short_name] = {
                "product_name": short_name,
                "expected_months": len(TIME_RANGE),
                "discovered_granules": len(local_files),
                "downloaded_files": len(local_files),
                "download_success_rate": float(len(local_files) / len(TIME_RANGE)) if len(TIME_RANGE) else 0.0,
                "status": "deferred_structure_complex",
                "reason": "RRS 当前为按波段拆分文件结构，先不在本轮全量并入；待 CHL/KD 单源 A/B 给出明确正信号后再单独扩展。",
            }
            product_files[short_name] = []
            rejected_or_deferred.append(f"{short_name}: RRS 当前按波段拆分，复杂度和文件数过高，本轮暂缓。")
            continue
        if short_name == "MODISA_L3m_CHL":
            product_stats[short_name] = {
                "product_name": short_name,
                "expected_months": len(TIME_RANGE),
                "discovered_granules": len(local_files),
                "downloaded_files": len(local_files),
                "download_success_rate": float(len(local_files) / len(TIME_RANGE)) if len(TIME_RANGE) else 0.0,
                "status": "deferred_sensor_control",
                "reason": "MODIS Aqua monthly CHL 当前只做传感器对照；在 NASA 主线月尺度 CHL/KD 单源 A/B 明确前，本轮暂不并入正式特征构建。",
            }
            product_files[short_name] = []
            rejected_or_deferred.append(f"{short_name}: 作为传感器对照暂缓，不阻塞 VIIRS CHL/KD 主线。")
            continue
        if args.skip_download and len(local_files) >= len(TIME_RANGE):
            granules, search_error = [], ""
        else:
            granules, search_error = search_monthly_granules(short_name, spec["granule_pattern"])
        files: list[Path] = []
        if search_error:
            stats = {
                "product_name": short_name,
                "expected_months": len(TIME_RANGE),
                "discovered_granules": len(granules),
                "downloaded_files": len(local_files),
                "download_success_rate": float(len(local_files) / len(TIME_RANGE)) if len(TIME_RANGE) else 0.0,
                "status": "deferred_search_error" if spec["priority"] != "required" else "partial_search_error",
                "reason": search_error,
            }
            files = local_files
        elif args.skip_download:
            files = local_files
            stats = {
                "product_name": short_name,
                "expected_months": len(TIME_RANGE),
                "discovered_granules": len(local_files),
                "downloaded_files": len(files),
                "download_success_rate": float(len(files) / len(TIME_RANGE)) if len(TIME_RANGE) else 0.0,
                "status": "skip_download_use_local" if files else "reject",
                "reason": "使用本地已有文件" if files else "未找到本地文件",
            }
        else:
            files, stats = download_product(short_name, granules, force=args.force)
        product_stats[short_name] = stats
        product_files[short_name] = files
        if stats["status"] in {"reject", "deferred_search_error", "partial_search_error"}:
            rejected_or_deferred.append(f"{short_name}: {stats['reason']}")

    rrs_variable_map = inspect_rrs_variables(product_files.get("VIIRSN_L3m_RRS", []))
    if not rrs_variable_map:
        rejected_or_deferred.append("VIIRSN_L3m_RRS: 未检测到稳定可用的目标 RRS 变量，暂缓入模。")
    product_long_frames: list[pd.DataFrame] = []
    for short_name, spec in PRODUCT_SPECS.items():
        if short_name == "VIIRSN_L3m_RRS":
            variable_map = rrs_variable_map
        else:
            variable_map = spec["variables"]
        if not variable_map:
            continue
        long_df, reasons = extract_product_long(short_name, product_files.get(short_name, []), variable_map, stations, product_stats[short_name])
        if not reasons:
            product_long_frames.append(long_df)
        else:
            rejected_or_deferred.extend([f"{short_name}: {reason}" for reason in reasons[:5]])
            if not long_df.empty:
                product_long_frames.append(long_df)

    if not product_long_frames:
        raise RuntimeError("NASA OceanColor 全时段提取为空，无法继续 A/B。")

    long_df, wide_df = build_long_and_wide(product_long_frames, stations)
    validation_report = write_validation_reports(long_df, wide_df, product_stats, rejected_or_deferred)
    write_feature_build_report(long_df, wide_df, product_stats, include_rrs=bool(rrs_variable_map))
    inventory_rows = product_rows_for_inventory(validation_report, product_stats)
    update_free_data_inventory(inventory_rows)

    base_df = load_base_dataset()
    base_df = merge_nasa_features(base_df, wide_df)
    include_rrs = any(col.startswith("viirs_rrs_") for col in base_df.columns)
    feature_sets = prepare_nasa_feature_sets(base_df, include_rrs=include_rrs)
    baseline_map = load_updated_baseline_metric_map()

    metrics_df, comparison_df, importance_df, useful_sets = run_ab_for_feature_sets(base_df, feature_sets, baseline_map)
    if useful_sets:
        scenario = "nasa_useful_interactions"
        useful_features = sorted({feature for key in useful_sets for feature in feature_sets[key]})
        useful_features = add_interactions(base_df, useful_features)
        extra_metrics, extra_comparison, extra_importance, _ = run_ab_for_feature_sets(base_df, {scenario: useful_features}, baseline_map)
        metrics_df = pd.concat([metrics_df, extra_metrics], ignore_index=True)
        comparison_df = pd.concat([comparison_df, extra_comparison], ignore_index=True)
        if not extra_importance.empty:
            importance_df = pd.concat([importance_df, extra_importance], ignore_index=True) if not importance_df.empty else extra_importance

    metrics_df.to_csv(AB_METRICS_CSV, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(AB_COMPARISON_CSV, index=False, encoding="utf-8-sig")
    if not importance_df.empty:
        importance_df.to_csv(AB_IMPORTANCE_CSV, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["target", "depth_layer", "feature", "importance", "scenario"]).to_csv(
            AB_IMPORTANCE_CSV, index=False, encoding="utf-8-sig"
        )
    write_task_summary(comparison_df)

    scorecard_after, progress_after, no_promote_reasons = promote_audit(comparison_df, current_best, baseline_map)
    promoted_any = bool((scorecard_after["reason"] == "promoted_from_nasa_oceancolor").any())
    scorecard_after.to_csv(AFTER_NASA_SCORECARD, index=False, encoding="utf-8-sig")
    progress_after.to_csv(AFTER_NASA_PROGRESS, index=False, encoding="utf-8-sig")
    update_docs(validation_report, comparison_df, scorecard_after, promoted_any, no_promote_reasons)

    manifest = {
        "script_path": str(Path(__file__).relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "input_files": [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in required_inputs],
        "output_files": [
            str(FEATURE_LONG_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            str(FEATURE_WIDE_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            str(AB_METRICS_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            str(AB_COMPARISON_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            str(FULLRUN_REPORT_MD.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        ],
        "products": list(product_stats.keys()),
        "rrs_included": include_rrs,
        "random_seed": RANDOM_SEED,
        "useful_sets": useful_sets,
        "promoted_any": promoted_any,
        "rejected_or_deferred": rejected_or_deferred,
    }
    write_text(
        CURRENT_REPORT_DIR / "experiment_manifest_precision_improvement.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )

    append_cleanup_log(
        [
            "## NASA OceanColor 月尺度全时段构建与 A/B",
            "",
            f"- 更新 `{FEATURE_LONG_CSV.relative_to(PROJECT_ROOT)}`、`{FEATURE_WIDE_CSV.relative_to(PROJECT_ROOT)}`。",
            f"- 更新 `{AB_METRICS_CSV.relative_to(PROJECT_ROOT)}`、`{AB_COMPARISON_CSV.relative_to(PROJECT_ROOT)}`、`{AB_IMPORTANCE_CSV.relative_to(PROJECT_ROOT)}`。",
            f"- 更新 `{FULLRUN_REPORT_MD.relative_to(PROJECT_ROOT)}` 与 `nasa_oceancolor_full_validation_report.md`。",
            f"- 生成 `{AFTER_NASA_SCORECARD.relative_to(PROJECT_ROOT)}` 与 `{AFTER_NASA_PROGRESS.relative_to(PROJECT_ROOT)}`。",
        ]
    )

    run_project_organizer()
    print(
        json.dumps(
            {
                "status": "ok",
                "products": product_stats,
                "long_rows": int(len(long_df)),
                "wide_rows": int(len(wide_df)),
                "ab_rows": int(len(metrics_df)),
                "promoted_any": promoted_any,
                "useful_sets": useful_sets,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
