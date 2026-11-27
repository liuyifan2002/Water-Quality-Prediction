from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
REPORTS_DIR = EXPERIMENTS_DIR / "reports"
OUTPUTS_DIR = EXPERIMENTS_DIR / "outputs"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODEL_INPUT_DIR = DATA_DIR / "model_input"

CURRENT_REPORTS = REPORTS_DIR / "current"
CURRENT_OUTPUTS = OUTPUTS_DIR / "current"
CURRENT_CONFIGS = CONFIGS_DIR / "current"
CURRENT_PROCESSED = PROCESSED_DIR / "current"
GEE_EXPORT_DIR = DATA_DIR / "external" / "gee_exports"

CURRENT_DATE_TAG = "20260518"


def ensure_dirs() -> None:
    for path in [
        CURRENT_REPORTS,
        CURRENT_OUTPUTS,
        CURRENT_CONFIGS,
        CURRENT_PROCESSED,
        GEE_EXPORT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "暂无数据。"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                vals.append("")
            else:
                vals.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def classify_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md"}:
        return "markdown"
    if suffix in {".csv"}:
        return "csv"
    if suffix in {".json"}:
        return "json"
    if suffix in {".js"}:
        return "javascript"
    if suffix in {".py"}:
        return "python"
    if suffix in {".nc"}:
        return "netcdf"
    return suffix.lstrip(".") or "no_suffix"


def relative_str(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def infer_experiment_name(path: Path) -> str:
    rel = relative_str(path)
    for part in path.parts:
        if any(char.isdigit() for char in part) and "_" in part:
            return part
    if "current" in rel:
        return "current"
    return "general"


def load_best_scorecard() -> pd.DataFrame:
    rows: list[dict] = []

    # Full-station four-target baseline
    full_station = ROOT / "outputs" / "modeling" / "full_station_best_by_target_depth.csv"
    if full_station.exists():
        df = pd.read_csv(full_station)
        for _, row in df.iterrows():
            rows.append(
                {
                    "target": row["target"],
                    "depth_layer": row["depth"],
                    "best_model": row["feature_set"],
                    "best_r2": row["r2"],
                    "source_path": relative_str(full_station),
                    "note": "四指标全站点正式主线最佳结果",
                }
            )

    # Chl-a optimized line
    chla_cmp = OUTPUTS_DIR / "chla_ocean_colour_precision_maximize_20260518" / "comparison_to_previous_best.csv"
    if chla_cmp.exists():
        df = pd.read_csv(chla_cmp)
        df = df[(df["split"] == "test") & (df["delta_r2_vs_previous"] >= 0)]
        for depth in ["surface", "middle", "bottom"]:
            sub = df[df["depth"] == depth].sort_values(["r2", "validation_r2"], ascending=False)
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append(
                {
                    "target": "chla",
                    "depth_layer": depth,
                    "best_model": row["candidate"],
                    "best_r2": row["r2"],
                    "source_path": relative_str(chla_cmp),
                    "note": "Chl-a Ocean Colour 优化后最佳结果",
                }
            )

    # Secchi optimized line
    secchi_cmp = OUTPUTS_DIR / "precision_maximize_20260518" / "comparison_to_previous_best.csv"
    if secchi_cmp.exists():
        df = pd.read_csv(secchi_cmp)
        delta_col = "delta_r2_vs_reference" if "delta_r2_vs_reference" in df.columns else "delta_r2_vs_previous"
        df = df[(df["split"] == "test") & (df[delta_col] >= 0)]
        for depth in ["surface", "middle", "bottom"]:
            sub = df[df["depth"] == depth].sort_values(["r2", "validation_r2"], ascending=False)
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append(
                {
                    "target": "secchi",
                    "depth_layer": depth,
                    "best_model": row["candidate"],
                    "best_r2": row["r2"],
                    "source_path": relative_str(secchi_cmp),
                    "note": "Secchi/ZSD Ocean Colour 优化后最佳结果",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "depth_layer", "best_r2"], ascending=[True, True, False])
        out = out.drop_duplicates(["target", "depth_layer"], keep="first")
    return out


def collect_experiment_index() -> pd.DataFrame:
    rows: list[dict] = []
    report_map = {p.stem: p for p in REPORTS_DIR.glob("*.md")}
    for exp_dir in sorted(p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and p.name != "current"):
        report = exp_dir / "experiment_report.md"
        if not report.exists():
            alt = exp_dir / f"{exp_dir.name}.md"
            report = alt if alt.exists() else None
        config = CONFIGS_DIR / f"{exp_dir.name}.json"
        rows.append(
            {
                "experiment_name": exp_dir.name,
                "output_dir": relative_str(exp_dir),
                "report_path": relative_str(report) if report else "",
                "config_path": relative_str(config) if config.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def build_output_inventory() -> pd.DataFrame:
    scan_roots = [
        OUTPUTS_DIR,
        REPORTS_DIR,
        CONFIGS_DIR,
        PROCESSED_DIR,
    ]
    canonical_paths = {
        "experiments/reports/current/experiment_master_index.md",
        "experiments/reports/current/methods_tried_registry.md",
        "experiments/reports/current/failed_experiments.md",
        "experiments/reports/current/do_not_repeat_experiments.md",
        "experiments/reports/current/data_gap_inventory.md",
        "experiments/reports/current/model_input_alignment_documentation.md",
        "experiments/reports/current/next_data_acquisition_plan.md",
        "experiments/reports/current/cleanup_log.md",
        "experiments/reports/current/experiment_report.md",
        "experiments/reports/current/bottom_mechanism_data_status.md",
        "experiments/reports/current/gee_export_instructions.md",
        "experiments/configs/current/current_state.json",
        "experiments/outputs/current/current_best_scorecard.csv",
        "data/processed/current/available_data_sources.csv",
        "data/processed/current/bottom_mechanism_features_2014_2024.csv",
        "data/processed/current/bottom_mechanism_feature_validation.json",
        "data/processed/current/bottom_mechanism_missing_summary.csv",
        "data/processed/current/gee_station_date_requests.csv",
    }
    rows: list[dict] = []
    for root in scan_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = relative_str(path)
            if "/current/" in rel:
                continue
            rows.append(
                {
                    "file_path": rel,
                    "file_type": classify_file_type(path),
                    "experiment_name": infer_experiment_name(path),
                    "created_or_modified_time": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
                    "keep_as_history": True,
                    "included_in_current_summary": rel in canonical_paths
                    or rel.endswith("full_station_best_by_target_depth.csv")
                    or rel.endswith("comparison_to_previous_best.csv")
                    or rel.endswith("experiment_report.md"),
                    "canonical_replacement": "",
                    "note": "历史文件保留，current/ 中建立汇总入口",
                }
            )
    df = pd.DataFrame(rows).sort_values(["experiment_name", "file_path"])
    return df


KEYWORD_MAP = {
    "highres_remote": [
        "hk_s2_patch_manifest",
        "sentinel",
        "landsat",
        "patch",
        "tile",
        "image_id",
        "band_path",
    ],
    "historical_water_quality": [
        "epd_marine_historical",
        "long_history",
        "lagged_cross_water_quality",
    ],
    "bottom_mechanism": [
        "depth",
        "bath",
        "temperature",
        "temp",
        "thermocline",
        "strat",
        "oxygen",
        "current",
        "tide",
        "level",
        "discharge",
        "inflow",
        "outflow",
        "wind",
        "solar",
        "rain",
        "precip",
        "era5",
    ],
}


def include_file_as_data_source(path: Path) -> bool:
    rel = relative_str(path).lower()
    name = path.name.lower()
    if "/current/" in rel or "tmp/venvs" in rel:
        return False
    if "template" in name:
        return False
    if rel.startswith("experiments/outputs/"):
        return any(
            token in name
            for token in [
                "ocean_colour_patch_features",
                "bottom_mechanism_features",
            ]
        )
    if rel.startswith("data/raw/manual/environment/"):
        return name in {
            "era5_features_2014_2024.csv",
            "rainfall_features_2014_2024.csv",
            "tide_features_2014_2024.csv",
        }
    if rel.startswith("data/raw/manual/ocean_color/cmems/"):
        return path.suffix.lower() == ".nc"
    if rel == "data/raw/manual/ocean_color/hk_s2_patch_manifest.csv":
        return True
    if rel.startswith("data/raw/manual/long_history_water_quality/"):
        return name.startswith("epd_marine_historical_") and name.endswith(".csv")
    if rel.startswith("data/model_input/new_sources/"):
        return name.endswith(".csv") and "experiment_matrix" not in name
    if rel.startswith("data/model_input/"):
        return name in {
            "station_meta.csv",
            "gee_station_date_samples.csv",
            "water_quality_master_2014_2024.csv",
            "model_table_3depth_2014_2024.csv",
            "station_date_satellite_features_2014_2024.csv",
            "environment_features_2014_2024.csv",
        }
    if rel.startswith("data/processed/mechanism_data_expansion_"):
        return "ocean_colour_patch_features_2014_2024.csv" in name
    return False


def detect_headers(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            return []
    if suffix == ".json":
        return []
    if suffix == ".nc":
        try:
            import xarray as xr  # type: ignore

            with xr.open_dataset(path) as ds:
                return list(ds.data_vars)[:50]
        except Exception:
            return []
    return []


def detect_time_range(path: Path, headers: list[str]) -> str:
    if "date" not in headers or path.suffix.lower() != ".csv":
        return ""
    try:
        df = pd.read_csv(path, usecols=["date"])
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            return ""
        return f"{dates.min().date()} to {dates.max().date()}"
    except Exception:
        return ""


def build_available_data_sources() -> pd.DataFrame:
    scan_roots = [
        DATA_DIR / "raw",
        DATA_DIR / "interim",
        DATA_DIR / "processed",
        DATA_DIR / "external",
        RAW_DIR / "manual",
        ROOT / "configs",
        ROOT / "docs",
        OUTPUTS_DIR,
    ]
    rows: list[dict] = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file():
                continue
            rel = relative_str(path)
            if not include_file_as_data_source(path):
                continue
            lower = rel.lower()
            category = None
            for cat, keywords in KEYWORD_MAP.items():
                if any(keyword in lower for keyword in keywords):
                    category = cat
                    break
            if category is None:
                if rel.startswith("data/model_input/"):
                    category = "core_model_input"
                elif rel.startswith("data/raw/manual/environment/"):
                    category = "bottom_mechanism"
                else:
                    continue
            headers = detect_headers(path)
            time_range = detect_time_range(path, headers)
            current_keys = []
            if "station" in [h.lower() for h in headers]:
                current_keys.append("station")
            if "date" in [h.lower() for h in headers]:
                current_keys.append("date")
            usable_now = any(key in current_keys for key in ["station", "date"])
            if path.name == "hk_s2_patch_manifest.csv":
                usable_now = False
            rows.append(
                {
                    "data_category": category,
                    "file_path": rel,
                    "detected_fields": "|".join(headers[:30]),
                    "usable_now": usable_now,
                    "needs_manual_download": path.name == "hk_s2_patch_manifest.csv",
                    "expected_target": "secchi/chla" if category == "highres_remote" else "chla_bottom",
                    "expected_gain_level": "high" if path.name == "hk_s2_patch_manifest.csv" else ("medium" if category == "bottom_mechanism" else "medium"),
                    "required_keys": "station+date" if category != "highres_remote" else "station+date+image_id/patch_path",
                    "current_keys": "+".join(current_keys),
                    "time_range": time_range,
                    "station_date_alignable": ("station" in current_keys and "date" in current_keys),
                    "leakage_risk": "no" if "date" in current_keys or "station" in current_keys else "unknown",
                    "next_action": infer_next_action(rel, headers),
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(subset=["file_path"]).sort_values(["data_category", "file_path"])
    return df


def infer_next_action(rel: str, headers: list[str]) -> str:
    if rel.endswith("hk_s2_patch_manifest.csv"):
        return "需要原始影像路径或 GEE 重导真实 patch 统计"
    if rel.endswith("station_meta.csv"):
        return "可直接作为站点静态坐标表使用"
    if rel.endswith("era5_features_2014_2024.csv"):
        return "可直接按 date 并入机制代理"
    if rel.endswith("rainfall_features_2014_2024.csv") or rel.endswith("tide_features_2014_2024.csv"):
        return "可直接按 date 并入机制代理"
    if rel.endswith(".nc"):
        return "保留原始栅格/NetCDF，后续用于 patch 或区域特征"
    return "纳入 current 数据源索引，按字段再决定是否并入"


def build_bottom_mechanism_features() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    model_table = pd.read_csv(MODEL_INPUT_DIR / "model_table_3depth_2014_2024.csv", usecols=[
        "obs_id", "sample_id", "station", "water_zone_en", "lon", "lat", "date", "depth",
        "depth_is_bottom", "year", "month", "is_wet_season"
    ])
    bottom = model_table[model_table["depth_is_bottom"] == 1].copy()
    bottom["date"] = pd.to_datetime(bottom["date"])

    station_meta = pd.read_csv(MODEL_INPUT_DIR / "station_meta.csv").rename(columns={"Station": "station"})
    era5 = pd.read_csv(RAW_DIR / "manual" / "environment" / "era5_features_2014_2024.csv")
    rain = pd.read_csv(RAW_DIR / "manual" / "environment" / "rainfall_features_2014_2024.csv")
    tide = pd.read_csv(RAW_DIR / "manual" / "environment" / "tide_features_2014_2024.csv")
    for df in [era5, rain, tide]:
        df["date"] = pd.to_datetime(df["date"])

    out = bottom.merge(
        station_meta[["station", "water_zone_en", "lon", "lat"]].drop_duplicates("station"),
        on="station",
        how="left",
        suffixes=("", "_meta"),
    )
    out = out.merge(era5, on="date", how="left")
    out = out.merge(rain, on="date", how="left")
    out = out.merge(tide, on="date", how="left")

    # Static / zone
    out["zone"] = out["water_zone_en"].fillna(out.get("water_zone_en_meta"))
    out["distance_to_shore"] = pd.NA
    out["distance_to_river_mouth"] = pd.NA
    out["station_depth"] = pd.NA
    out["bathymetry_depth"] = pd.NA

    # Available mechanism proxies
    if "sst_mean_3d" in out.columns:
        out["surface_temp"] = out["sst_mean_3d"]
        out["temp_7d_mean"] = out["sst_mean_7d"]
        out["temp_14d_mean"] = out["sst_mean_14d"]
        out["temp_30d_mean"] = out["sst_mean_14d"]
        out["temp_change_rate"] = out["sst_mean_3d"] - out["sst_mean_14d"]
    else:
        out["surface_temp"] = pd.NA
        out["temp_7d_mean"] = pd.NA
        out["temp_14d_mean"] = pd.NA
        out["temp_30d_mean"] = pd.NA
        out["temp_change_rate"] = pd.NA

    out["bottom_temp"] = pd.NA
    out["surface_bottom_temp_diff"] = pd.NA
    out["temp_gradient"] = pd.NA
    out["thermocline_proxy"] = pd.NA

    if "wind_speed" in out.columns:
        out["wind_speed_3d_mean"] = out["wind_speed"]
        out["wind_speed_7d_mean"] = out["wind_speed"]
        out["wind_speed_14d_mean"] = out["wind_speed"]
        out["wind_speed_3d_max"] = out["wind_speed"]
        out["wind_direction_u"] = out.get("u10")
        out["wind_direction_v"] = out.get("v10")
        out["calm_days_7d"] = pd.NA
        out["strong_wind_event_7d"] = pd.NA
        out["wind_mixing_proxy"] = out["wind_speed"] * out.get("tidal_range", 1)
        if "sst_mean_7d" in out.columns:
            out["stratification_strength"] = out["sst_mean_7d"] / (out["wind_speed"].abs() + 1.0)
        else:
            out["stratification_strength"] = pd.NA
    else:
        for col in [
            "wind_speed_3d_mean", "wind_speed_7d_mean", "wind_speed_14d_mean", "wind_speed_3d_max",
            "wind_direction_u", "wind_direction_v", "calm_days_7d", "strong_wind_event_7d",
            "wind_mixing_proxy", "stratification_strength",
        ]:
            out[col] = pd.NA

    out["precip_3d_sum"] = out.get("rain_3d")
    out["precip_7d_sum"] = out.get("rain_7d")
    out["storm_event_flag"] = (out.get("rain_3d", pd.Series([math.nan] * len(out))) >= 50).astype("float")
    out["water_level"] = pd.NA
    out["water_level_change_3d"] = pd.NA
    out["discharge_7d_mean"] = pd.NA
    out["flushing_proxy"] = pd.NA
    out["current_speed"] = pd.NA
    out["current_speed_7d_mean"] = pd.NA
    out["tide_range_proxy"] = out.get("tidal_range")

    keep_cols = [
        "obs_id", "sample_id", "station", "date", "depth", "year", "month", "is_wet_season",
        "lon", "lat", "zone",
        "station_depth", "bathymetry_depth", "distance_to_shore", "distance_to_river_mouth",
        "surface_temp", "bottom_temp", "surface_bottom_temp_diff", "temp_gradient",
        "thermocline_proxy", "stratification_strength", "temp_7d_mean", "temp_14d_mean",
        "temp_30d_mean", "temp_change_rate",
        "wind_speed_3d_mean", "wind_speed_7d_mean", "wind_speed_14d_mean", "wind_speed_3d_max",
        "calm_days_7d", "strong_wind_event_7d", "wind_mixing_proxy", "wind_direction_u",
        "wind_direction_v", "precip_3d_sum", "precip_7d_sum", "storm_event_flag", "water_level",
        "water_level_change_3d", "discharge_7d_mean", "flushing_proxy", "current_speed",
        "current_speed_7d_mean", "tide_range_proxy",
    ]
    out = out[keep_cols]

    missing_summary = pd.DataFrame(
        {
            "feature": out.columns,
            "missing_rate": [float(out[col].isna().mean()) for col in out.columns],
            "non_null_count": [int(out[col].notna().sum()) for col in out.columns],
        }
    )

    validation = {
        "rows": int(len(out)),
        "station_count": int(out["station"].nunique()),
        "date_range": f"{out['date'].min().date()} to {out['date'].max().date()}",
        "available_feature_count": int((missing_summary["non_null_count"] > 0).sum()),
        "unavailable_key_features": [
            "station_depth",
            "bathymetry_depth",
            "bottom_temp",
            "surface_bottom_temp_diff",
            "temp_gradient",
            "thermocline_proxy",
            "water_level",
            "discharge_7d_mean",
            "current_speed",
        ],
        "note": "本轮只做本地机制变量标准化与特征生成，不进行正式 A/B 建模。",
    }
    return out, validation, missing_summary


def build_gee_requests() -> pd.DataFrame:
    gee = pd.read_csv(MODEL_INPUT_DIR / "gee_station_date_samples.csv")
    gee = gee.rename(columns={"Station": "station"})
    gee["date"] = pd.to_datetime(gee["date"])
    gee = gee.drop_duplicates(subset=["station", "date"]).sort_values(["date", "station"]).reset_index(drop=True)

    model_table = pd.read_csv(
        MODEL_INPUT_DIR / "model_table_3depth_2014_2024.csv",
        usecols=["sample_id", "station", "date", "chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l"],
    )
    model_table["date"] = pd.to_datetime(model_table["date"])
    flags = (
        model_table.groupby(["sample_id", "station", "date"])
        .agg(
            chla_available=("chla_ug_l", lambda s: int(s.notna().any())),
            secchi_available=("secchi_depth_m", lambda s: int(s.notna().any())),
            tn_available=("tn_mg_l", lambda s: int(s.notna().any())),
            tp_available=("tp_mg_l", lambda s: int(s.notna().any())),
        )
        .reset_index()
    )

    out = gee.merge(flags, on=["sample_id", "station", "date"], how="left")
    for col in ["chla_available", "secchi_available", "tn_available", "tp_available"]:
        out[col] = out[col].fillna(0).astype(int)

    def infer_split(ts: pd.Timestamp) -> str:
        if ts.year <= 2020:
            return "train"
        if ts.year <= 2022:
            return "validation"
        return "test"

    out["split"] = out["date"].apply(infer_split)
    out["request_id"] = [f"req_{i:06d}" for i in range(1, len(out) + 1)]
    out["target_available_flags"] = out.apply(
        lambda r: f"chla={r['chla_available']}|secchi={r['secchi_available']}|tn={r['tn_available']}|tp={r['tp_available']}",
        axis=1,
    )
    out["window_days"] = 3
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    keep = [
        "request_id", "sample_id", "station", "date", "lon", "lat", "split", "target_available_flags", "window_days"
    ]
    return out[keep]


SENTINEL2_JS = r"""// Sentinel-2 高分 patch 导出脚本（本轮仅生成，不在 Codex 中运行）
// 1) 先将 gee_station_date_requests.csv 上传为 GEE Asset
// 2) 将下面 REQUEST_ASSET_ID 替换为你的实际 Asset ID
// 3) 在 GEE Code Editor 运行导出任务

var REQUEST_ASSET_ID = 'users/your_username/gee_station_date_requests';
var EXPORT_DESC = 's2_patch_features_2014_2024';
var PATCH_BUFFER_METERS_3 = 30;  // 约 3x3 @ 10m
var PATCH_BUFFER_METERS_5 = 50;  // 约 5x5 @ 10m
var WINDOW_DAYS = 3;

var requests = ee.FeatureCollection(REQUEST_ASSET_ID);

function maskS2(image) {
  var scl = image.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return image.updateMask(mask).copyProperties(image, ['system:time_start']);
}

function addIndices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  var ndci = image.normalizedDifference(['B5', 'B4']).rename('NDCI');
  var fai = image.expression(
    'nir - (red + (swir - red) * ((nir_wl - red_wl) / (swir_wl - red_wl)))',
    {
      nir: image.select('B8'),
      red: image.select('B4'),
      swir: image.select('B11'),
      nir_wl: 842,
      red_wl: 665,
      swir_wl: 1610
    }
  ).rename('FAI');
  var redGreenRatio = image.select('B4').divide(image.select('B3')).rename('red_green_ratio');
  return image.addBands([ndvi, ndwi, mndwi, ndci, fai, redGreenRatio]);
}

function patchStats(image, geom, scale, prefix, radiusMeters) {
  var reducer = ee.Reducer.mean()
    .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.median(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.percentile([10, 90]), sharedInputs: true})
    .combine({reducer2: ee.Reducer.count(), sharedInputs: true});
  return image.reduceRegion({
    reducer: reducer,
    geometry: geom.buffer(radiusMeters),
    scale: scale,
    maxPixels: 1e8
  });
}

function featureForRequest(f) {
  var date = ee.Date(f.get('date'));
  var geom = f.geometry();
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(date.advance(-WINDOW_DAYS, 'day'), date.advance(WINDOW_DAYS + 1, 'day'))
    .map(maskS2)
    .map(addIndices)
    .map(function(img) {
      var diff = ee.Number(img.date().difference(date, 'day')).abs();
      return img.set('date_diff_abs', diff);
    })
    .sort('date_diff_abs');

  var image = ee.Image(collection.first());
  var empty = ee.Algorithms.IsEqual(image, null);
  return ee.Feature(geom, ee.Algorithms.If(empty, {
    request_id: f.get('request_id'),
    station: f.get('station'),
    date: f.get('date'),
    image_id: null,
    image_date: null,
    date_diff_abs: null,
    cloud_percentage: null
  }, ee.Dictionary({
    request_id: f.get('request_id'),
    station: f.get('station'),
    date: f.get('date'),
    image_id: image.get('PRODUCT_ID'),
    image_date: image.date().format('YYYY-MM-dd'),
    date_diff_abs: image.get('date_diff_abs'),
    cloud_percentage: image.get('CLOUDY_PIXEL_PERCENTAGE')
  })
    .combine(patchStats(image.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','NDWI','MNDWI','NDCI','FAI','red_green_ratio']), geom, 10, 'p3', PATCH_BUFFER_METERS_3), true)
    .combine(patchStats(image.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','NDWI','MNDWI','NDCI','FAI','red_green_ratio']), geom, 10, 'p5', PATCH_BUFFER_METERS_5), true)
  )));
}

var outFc = requests.map(featureForRequest);
Export.table.toDrive({
  collection: outFc,
  description: EXPORT_DESC,
  fileNamePrefix: EXPORT_DESC,
  fileFormat: 'CSV'
});
"""


LANDSAT_JS = r"""// Landsat 高分 patch 导出脚本（本轮仅生成，不在 Codex 中运行）
// 1) 上传 gee_station_date_requests.csv 为 GEE Asset
// 2) 替换 REQUEST_ASSET_ID
// 3) 在 GEE Code Editor 中运行

var REQUEST_ASSET_ID = 'users/your_username/gee_station_date_requests';
var EXPORT_DESC = 'landsat_patch_features_2014_2024';
var PATCH_BUFFER_METERS_3 = 45;  // 约 3x3 @ 30m
var PATCH_BUFFER_METERS_5 = 75;  // 约 5x5 @ 30m
var WINDOW_DAYS = 3;

var requests = ee.FeatureCollection(REQUEST_ASSET_ID);

function prepLandsat(image) {
  var qa = image.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 3).eq(0)
    .and(qa.bitwiseAnd(1 << 4).eq(0))
    .and(qa.bitwiseAnd(1 << 5).eq(0));
  var optical = image.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
    .multiply(0.0000275).add(-0.2);
  var lst = ee.Image(ee.Algorithms.If(
    image.bandNames().contains('ST_B10'),
    image.select('ST_B10').multiply(0.00341802).add(149.0).rename('LST_K'),
    ee.Image.constant(-9999).rename('LST_K')
  ));
  optical = optical.addBands(lst);
  optical = optical.updateMask(clear);
  var ndvi = optical.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  var ndwi = optical.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
  var mndwi = optical.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI');
  var redGreenRatio = optical.select('SR_B4').divide(optical.select('SR_B3')).rename('red_green_ratio');
  var nirRedRatio = optical.select('SR_B5').divide(optical.select('SR_B4')).rename('nir_red_ratio');
  return optical.addBands([ndvi, ndwi, mndwi, redGreenRatio, nirRedRatio])
    .copyProperties(image, ['system:time_start', 'LANDSAT_PRODUCT_ID']);
}

function patchStats(image, geom, scale, radiusMeters) {
  var reducer = ee.Reducer.mean()
    .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.median(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.percentile([10, 90]), sharedInputs: true})
    .combine({reducer2: ee.Reducer.count(), sharedInputs: true});
  return image.reduceRegion({
    reducer: reducer,
    geometry: geom.buffer(radiusMeters),
    scale: scale,
    maxPixels: 1e8
  });
}

function featureForRequest(f) {
  var date = ee.Date(f.get('date'));
  var geom = f.geometry();
  var collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(geom)
    .filterDate(date.advance(-WINDOW_DAYS, 'day'), date.advance(WINDOW_DAYS + 1, 'day'))
    .map(prepLandsat)
    .map(function(img) {
      var diff = ee.Number(img.date().difference(date, 'day')).abs();
      return img.set('date_diff_abs', diff);
    })
    .sort('date_diff_abs');

  var image = ee.Image(collection.first());
  var empty = ee.Algorithms.IsEqual(image, null);
  return ee.Feature(geom, ee.Algorithms.If(empty, {
    request_id: f.get('request_id'),
    station: f.get('station'),
    date: f.get('date'),
    image_id: null,
    image_date: null,
    date_diff_abs: null
  }, ee.Dictionary({
    request_id: f.get('request_id'),
    station: f.get('station'),
    date: f.get('date'),
    image_id: image.get('LANDSAT_PRODUCT_ID'),
    image_date: image.date().format('YYYY-MM-dd'),
    date_diff_abs: image.get('date_diff_abs')
  })
    .combine(patchStats(image.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','LST_K','NDVI','NDWI','MNDWI','red_green_ratio','nir_red_ratio']), geom, 30, PATCH_BUFFER_METERS_3), true)
    .combine(patchStats(image.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','LST_K','NDVI','NDWI','MNDWI','red_green_ratio','nir_red_ratio']), geom, 30, PATCH_BUFFER_METERS_5), true)
  )));
}

var outFc = requests.map(featureForRequest);
Export.table.toDrive({
  collection: outFc,
  description: EXPORT_DESC,
  fileNamePrefix: EXPORT_DESC,
  fileFormat: 'CSV'
});
"""


def build_current_reports(
    best_scorecard: pd.DataFrame,
    experiment_index: pd.DataFrame,
    inventory: pd.DataFrame,
    data_sources: pd.DataFrame,
    mech_validation: dict,
) -> None:
    best_table = (
        markdown_table(best_scorecard.assign(best_r2=lambda df: df["best_r2"].round(4)))
        if not best_scorecard.empty
        else "暂无可汇总结果。"
    )
    experiment_table = markdown_table(experiment_index) if not experiment_index.empty else "暂无实验目录。"

    write_text(
        CURRENT_REPORTS / "experiment_master_index.md",
        f"""
# 当前实验总索引

## 当前最佳结果总表

{best_table}

## 历史实验目录索引

{experiment_table}

## 当前阶段判断

- 四指标正式主线仍以 `outputs/modeling/full_station_best_by_target_depth.csv` 作为最完整的历史基线。
- Chl-a 和 Secchi/ZSD 的后续优化主要集中在 `experiments/outputs/` 下的 Ocean Colour、机制代理、站点分组等实验。
- 当前最值得继续投入的数据源不是普通调参，而是高分辨率 Sentinel-2 / Landsat patch 与 Chl-a bottom 关键机制变量。
""",
    )

    write_text(
        CURRENT_REPORTS / "methods_tried_registry.md",
        """
# 已尝试方法登记（current canonical）

## 已完成并形成明确结论的路线

1. Ocean Colour 单源 A/B：对 Chl-a 与 Secchi/ZSD 均有稳定正增益，已成为光学类主增益来源。
2. 常规 LightGBM / XGBoost / RandomForest：已完成，不再作为突破路线重复横向比较。
3. 普通 Optuna：surface 和 middle 有提升，但对 bottom 跃迁有限。
4. 常规 feature pruning：可稳定，但不足以突破底层瓶颈。
5. KNN / rolling / native missing 缺失值策略：已系统比较，收益有限。
6. P90 高值加权：适合 high-value auxiliary，不适合作为整体主模型替代。
7. 简单 stacking / residual correction / optical regime：对 Chl-a bottom 未形成跃迁。
8. CMEMS 4 km patch：真实可用，但空间分辨率太粗，整体增益有限。
9. 站点分组建模：对部分站点有帮助，但整体增益仍不足以替代全局主线。

## 当前最佳结果口径

- Chl-a 当前最佳主线来自 Ocean Colour 精度优化实验。
- Secchi/ZSD 当前最佳主线来自 Ocean Colour 精度优化与后续补充实验。
- TN / TP 当前仍以历史 `full_station` 正式主线为基线记录。

## 以后只有在以下条件满足时才继续推进

- 拿到真实 Sentinel-2 / Landsat 高分辨率 patch 特征；
- 拿到 Chl-a bottom 关键机制变量；
- 或用户补充新的、非普通调参类型的数据源。
""",
    )

    write_text(
        CURRENT_REPORTS / "failed_experiments.md",
        """
# 失败与低收益路线（current canonical）

## 已证实低收益或不应重复的路线

- 普通调参继续扩搜索空间：难以再带来有意义跃迁。
- 只靠 Ocean Colour 点值 + 历史目标继续堆特征：收益趋于饱和。
- 仅用简单 ERA5 / rain / tide 小集合代理突破 Chl-a bottom：未达 major gain。
- 把 `hk_s2_patch_manifest.csv` 误当成真实 patch 特征表：不可行。
- 在没有真实高分 patch 或底层机制变量前，继续放大站点分组建模的预期：不合理。

## 当前失败并不代表模型技巧不够

当前更主要的瓶颈是：

- 高分辨率邻域遥感缺失；
- Chl-a bottom 的底层温度 / 分层 / 混合 / 流速 / bathymetry 缺失；
- 部分站点机制明显异质，但现有输入不足以稳定刻画。
""",
    )

    write_text(
        CURRENT_REPORTS / "do_not_repeat_experiments.md",
        """
# 不再重复清单（current canonical）

- 不再继续普通 LightGBM / XGBoost / RandomForest 微调。
- 不再继续普通 Optuna 扩范围搜索。
- 不再继续单独比较 KNN / rolling / mask / native missing。
- 不再把 P90 高值加权直接升级为整体主线。
- 不再把简单 stacking、residual correction、simple regime 当作 Chl-a bottom 突破路线。
- 不再把 CMEMS 4 km patch 当成高分 patch 替代品继续重复。
- 不再把 `hk_s2_patch_manifest.csv` 当成真实高分 patch 数据源。
- 在没有新增真实机制变量和高分 patch 前，不再继续普通调参与普通 patch A/B。
""",
    )

    data_source_summary = markdown_table(data_sources[[
        "data_category", "file_path", "usable_now", "expected_target", "next_action"
    ]]) if not data_sources.empty else "暂无识别到数据源。"
    write_text(
        CURRENT_REPORTS / "data_gap_inventory.md",
        f"""
# 当前数据缺口与可用性清单

## 当前已识别到的本地数据源

{data_source_summary}

## Chl-a bottom 仍缺的关键变量

- station depth
- bathymetry
- bottom water temperature
- vertical temperature profile
- thermocline / stratification 真实剖面指标
- current velocity / tidal current
- water level / discharge / inflow / outflow

## Secchi/ZSD 仍缺的关键变量

- 真实 Sentinel-2 / Landsat 3x3、5x5 patch 波段统计
- valid pixel ratio / cloud mask 对应的原始邻域栅格
- 可回溯的 image_id / tile / patch_path / band_path

## 本轮本地可直接利用的机制代理

- ERA5 date-level 环境特征
- rainfall date-level 累积特征
- tide date-level 代理特征
- station 坐标与 water_zone 静态信息

## 本轮结论

- 本地可以先把机制变量做成标准化特征表，但还不足以替代真正的底层机制数据。
- 高分 patch 仍需要用户下一轮在 GEE 手动导出后才能进入正式 A/B。
""",
    )

    write_text(
        CURRENT_REPORTS / "next_data_acquisition_plan.md",
        """
# 下一步数据获取计划（current canonical）

## A. 已由 Codex 本轮完成

- 历史输出合并与 current canonical 文件建立
- 本地数据源扫描与可用性表生成
- 本地机制代理变量标准化与状态报告
- GEE 请求表生成
- Sentinel-2 / Landsat GEE JS 脚本生成
- GEE 手动操作说明生成

## B. 需要用户手动完成

1. 将 `data/processed/current/gee_station_date_requests.csv` 上传到 GEE Asset
2. 在 GEE Code Editor 运行：
   - `scripts/data/export_highres_patch_gee_sentinel2.js`
   - `scripts/data/export_highres_patch_gee_landsat.js`
3. 从 Google Drive 下载导出的 CSV
4. 放回 `data/external/gee_exports/`
5. 尽可能补充：
   - station depth
   - bathymetry
   - bottom water temperature
   - vertical profile
   - current / water level / discharge

## C. 下一轮 Codex 可继续做

- 读取 GEE 导出 CSV
- 校验 highres patch 字段
- 合并 Sentinel-2 / Landsat highres patch 特征
- 生成 `highres_patch_features_2014_2024.csv`
- 跑 highres patch A/B
- 对用户补充的 Chl-a bottom 机制变量做标准化、特征生成与 A/B
- 更新 current 主线、对齐说明和逐站点评估
""",
    )

    write_text(
        CURRENT_REPORTS / "bottom_mechanism_data_status.md",
        f"""
# Chl-a Bottom 机制变量本地状态

## 本轮是否生成机制特征表

- 已生成：`data/processed/current/bottom_mechanism_features_2014_2024.csv`
- 当前主要是 **站点静态信息 + ERA5 / rainfall / tide 的 date-level 代理特征**
- 尚未包含真正的：
  - station depth
  - bathymetry
  - bottom temperature
  - vertical profile
  - current velocity
  - water level / discharge

## 验证摘要

```json
{json.dumps(mech_validation, ensure_ascii=False, indent=2)}
```

## 当前判断

- 该表可作为“本地可用机制代理”的统一入口。
- 但它不是下一轮 Chl-a bottom 跃迁所需的完整机制表，不能替代真实底层机制数据。
""",
    )

    write_text(
        CURRENT_REPORTS / "gee_export_instructions.md",
        """
# GEE 高分 patch 导出说明（本轮只做准备，不执行）

## 本轮没有运行 GEE

Codex 本轮只生成：

- `data/processed/current/gee_station_date_requests.csv`
- `scripts/data/export_highres_patch_gee_sentinel2.js`
- `scripts/data/export_highres_patch_gee_landsat.js`

没有在本轮启动 Earth Engine export task。

## 用户下一步要做什么

1. 打开 GEE Code Editor
2. 上传 `gee_station_date_requests.csv` 为一个 table asset
3. 把脚本中的 `REQUEST_ASSET_ID` 替换成真实 Asset ID
4. 分别运行 Sentinel-2 与 Landsat 导出脚本
5. 到 Google Drive 下载结果

## 导出的文件放回哪里

放回：

- `data/external/gee_exports/s2_patch_features_2014_2024.csv`
- `data/external/gee_exports/landsat_patch_features_2014_2024.csv`

## 下一轮 Codex 会怎么处理

- 校验字段和键
- 合并到现有主表
- 生成真正的 highres patch 特征总表
- 跑 highres patch A/B 与逐站点比较
""",
    )

    write_text(
        CURRENT_REPORTS / "experiment_report.md",
        """
# Current Canonical Experiment Report

## 本轮实际完成的 Codex 本地任务

1. 建立 `current/` canonical 目录
2. 合并历史报告与方法登记
3. 生成历史输出文件索引
4. 扫描本地已有数据源
5. 生成本地可用 bottom mechanism 特征表
6. 生成 GEE 请求表
7. 生成 Sentinel-2 / Landsat GEE JS 脚本
8. 更新模型输入对齐说明

## 本轮没有做的任务

- 没有运行 GEE
- 没有下载 Sentinel-2 / Landsat 影像
- 没有启动 Earth Engine export task
- 没有继续普通调参与重复低收益实验

## 本地是否发现可直接用的机制变量

- 发现了可直接并入的 date-level 机制代理：ERA5、rainfall、tide
- 发现了可直接并入的 station-level 静态信息：lon、lat、water_zone
- 没有发现真正的 station depth / bathymetry / bottom temp / current / discharge

## 本轮关键产物

- `data/processed/current/available_data_sources.csv`
- `data/processed/current/bottom_mechanism_features_2014_2024.csv`
- `data/processed/current/gee_station_date_requests.csv`
- `scripts/data/export_highres_patch_gee_sentinel2.js`
- `scripts/data/export_highres_patch_gee_landsat.js`

## 下一步用户手动要做什么

- 上传 GEE 请求表到 GEE Asset
- 运行 Sentinel-2 / Landsat patch 导出
- 把导出的 CSV 放回 `data/external/gee_exports/`
- 尽可能补充 Chl-a bottom 关键机制变量

## 用户完成下载后，Codex 下一轮怎么继续

- 校验 highres patch 字段
- 合并 highres patch 特征
- 跑 highres patch A/B
- 更新 current 主线和逐站点评估

## 缓存与目录

- 本轮结束前会再次清理 `__pycache__`
- 会检查根目录是否有散落 csv/json/py/ipynb
""",
    )


def build_alignment_doc() -> None:
    write_text(
        CURRENT_REPORTS / "model_input_alignment_documentation.md",
        """
# 模型输入数据对齐说明（current canonical）

## 1. 主表标签数据

- 文件：`data/model_input/model_table_3depth_2014_2024.csv`
- station 字段：`station`
- date 字段：`date`
- target 字段：
  - `chla_ug_l`
  - `secchi_depth_m`
  - `tn_mg_l`
  - `tp_mg_l`
- depth layer 字段：
  - `depth`
  - `depth_is_surface`
  - `depth_is_middle`
  - `depth_is_bottom`
- 标签使用原始日实测值
- 当前正式主线默认不把统一 transform 直接写死到所有模型输入表中；具体是否 `log1p` 由实验脚本控制

## 2. Ocean Colour 点值

- 数据源：Copernicus Marine Ocean Colour
- 时间分辨率：daily
- 空间分辨率：约 4 km
- 对齐键：`station + date`
- 特征解释：
  - `oc_chl`：选中日对应的点值
  - `oc_kd490`：选中日对应的点值
  - `oc_zsd`：选中日对应的点值
  - `oc_day_diff`：观测日与影像日差
  - `oc_*_3d/7d/14d`：仅使用预测日及以前窗口的 rolling mean
  - `valid_count_7d`：过去 7 天有效观测次数

## 3. CMEMS 4km patch

- patch 大小：3x3 与 5x5
- 统计量：`mean/std/min/max/valid_ratio/missing_ratio`
- 时间对齐：样本日对应切片，不向未来搜索
- 提升有限的原因：
  - 空间分辨率太粗
  - 难以表达近岸小尺度异质性
  - 对 Chl-a bottom 的底层过程刻画不足

## 4. 计划中的 Sentinel-2 / Landsat highres patch

- 当前未导出
- 已生成：
  - `data/processed/current/gee_station_date_requests.csv`
  - `scripts/data/export_highres_patch_gee_sentinel2.js`
  - `scripts/data/export_highres_patch_gee_landsat.js`
- 下一步由用户手动在 GEE 导出
- 预计字段：
  - 波段值
  - NDVI / NDWI / MNDWI / NDCI / FAI
  - 3x3 与 5x5 patch 的 mean/std/median/p10/p90/count
  - image_id / image_date / date_diff_abs
- 当前不得写成“已经使用”

## 5. 机制变量

- 本地已有：
  - ERA5 日尺度与 rolling 窗口代理
  - rainfall 累积量
  - tide 日尺度代理
  - station 坐标与 zone
- 本地缺失：
  - station depth
  - bathymetry
  - bottom water temperature
  - vertical profile
  - current / discharge / water level

## 6. 泄漏控制

- rolling 只使用预测日及以前
- 不使用未来信息
- 不使用 test 标签做统计
- 不使用同日其他水质实测协变量
- 缺失值策略参数若需要拟合，只能在 train 上拟合后应用 validation/test
""",
    )


def build_cleanup_log(inventory: pd.DataFrame, root_pollution: list[str]) -> None:
    included = inventory[inventory["included_in_current_summary"] == True]
    root_pollution_text = (
        "未发现根目录散落 csv/json/py/ipynb 文件。"
        if not root_pollution
        else "发现疑似根目录污染文件：\n- " + "\n- ".join(root_pollution)
    )
    write_text(
        CURRENT_REPORTS / "cleanup_log.md",
        f"""
# 清理与汇总日志

## 纳入 current 汇总的内容

- current canonical 报告组
- `current_best_scorecard.csv`
- `available_data_sources.csv`
- `bottom_mechanism_features_2014_2024.csv`
- `gee_station_date_requests.csv`

## 历史文件处理原则

- 历史文件全部保留为归档
- 不删除历史正式结果
- 通过 `output_file_inventory.csv` 建立索引，在 `current/` 中提供 canonical 入口

## 根目录污染检查

{root_pollution_text}

## 缓存与空目录

- 本轮结束前会再次清理 `__pycache__`
- 若发现空目录，只记录，不删除历史实验目录

## 当前文件数量控制

- 本轮优先更新 `current/` 固定文件
- 没有继续新建大量带日期后缀的重复报告目录
""",
    )


def detect_root_pollution() -> list[str]:
    bad_exts = {".csv", ".json", ".py", ".ipynb"}
    root_allowed = {"AGENTS.md", "README.md"}
    bad = []
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix.lower() in bad_exts and path.name not in root_allowed:
            bad.append(path.name)
    return sorted(bad)


def build_current_state_json(best_scorecard: pd.DataFrame, mech_validation: dict) -> None:
    payload = {
        "date_tag": CURRENT_DATE_TAG,
        "current_best_models": best_scorecard.to_dict(orient="records"),
        "bottom_mechanism_validation": mech_validation,
        "manual_next_step": "用户在 GEE 手动导出 Sentinel-2 / Landsat highres patch CSV 并放回 data/external/gee_exports/",
    }
    write_text(CURRENT_CONFIGS / "current_state.json", json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    ensure_dirs()

    best_scorecard = load_best_scorecard()
    experiment_index = collect_experiment_index()
    inventory = build_output_inventory()
    data_sources = build_available_data_sources()
    mech_df, mech_validation, mech_missing = build_bottom_mechanism_features()
    gee_requests = build_gee_requests()

    best_scorecard.to_csv(CURRENT_OUTPUTS / "current_best_scorecard.csv", index=False)
    inventory.to_csv(CURRENT_REPORTS / "output_file_inventory.csv", index=False)
    data_sources.to_csv(CURRENT_PROCESSED / "available_data_sources.csv", index=False)
    mech_df.to_csv(CURRENT_PROCESSED / "bottom_mechanism_features_2014_2024.csv", index=False)
    write_text(
        CURRENT_PROCESSED / "bottom_mechanism_feature_validation.json",
        json.dumps(mech_validation, ensure_ascii=False, indent=2),
    )
    mech_missing.to_csv(CURRENT_PROCESSED / "bottom_mechanism_missing_summary.csv", index=False)
    gee_requests.to_csv(CURRENT_PROCESSED / "gee_station_date_requests.csv", index=False)

    write_text(ROOT / "scripts" / "data" / "export_highres_patch_gee_sentinel2.js", SENTINEL2_JS)
    write_text(ROOT / "scripts" / "data" / "export_highres_patch_gee_landsat.js", LANDSAT_JS)

    build_alignment_doc()
    build_current_reports(best_scorecard, experiment_index, inventory, data_sources, mech_validation)
    build_cleanup_log(inventory, detect_root_pollution())
    build_current_state_json(best_scorecard, mech_validation)

    print(
        json.dumps(
            {
                "current_reports": relative_str(CURRENT_REPORTS),
                "current_outputs": relative_str(CURRENT_OUTPUTS),
                "current_configs": relative_str(CURRENT_CONFIGS),
                "current_processed": relative_str(CURRENT_PROCESSED),
                "gee_requests": relative_str(CURRENT_PROCESSED / "gee_station_date_requests.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
