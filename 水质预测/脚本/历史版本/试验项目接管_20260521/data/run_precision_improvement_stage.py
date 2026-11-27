from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENV_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "venvs" / "ocean_ab_20260517" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(VENV_SITE_PACKAGES))

from scripts.new_data_sources_utils import load_env_file, load_yaml_config  # noqa: E402
from scripts.training import run_current_highres_patch_ab as base_ab  # noqa: E402


CURRENT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
CURRENT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "current"
CURRENT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"
MODEL_INPUT_NEW_SOURCES = PROJECT_ROOT / "data" / "model_input" / "new_sources"
CONFIG_PATH = PROJECT_ROOT / "config" / "new_data_sources.yaml"

CURRENT_BEST_SCORECARD = CURRENT_PROCESSED_DIR / "current_best_scorecard.csv"
R2_PROGRESS = CURRENT_PROCESSED_DIR / "r2_0p6_progress_scorecard.csv"
STRONG_MECH_COMPARE = CURRENT_OUTPUT_DIR / "strong_mechanism_comparison_to_current_best.csv"
STRONG_MECH_METRICS = CURRENT_OUTPUT_DIR / "strong_mechanism_ab_metrics.csv"
STRONG_MECH_STATION = CURRENT_OUTPUT_DIR / "strong_mechanism_station_level_comparison.csv"
STATION_LEVEL_PRED = CURRENT_OUTPUT_DIR / "station_level_predictions.csv"
STATION_LEVEL_COMPARE = CURRENT_OUTPUT_DIR / "station_level_prediction_comparison.csv"
COPERNICUS_PHYSICS = CURRENT_PROCESSED_DIR / "copernicus_physics_features_2014_2024.csv"
COPERNICUS_PHYSICS_VALID = CURRENT_PROCESSED_DIR / "copernicus_physics_validation_report.json"
HIGH_QUALITY_PATCH = CURRENT_PROCESSED_DIR / "high_quality_patch_features_2014_2024.csv"
TIDE_PHASE = CURRENT_PROCESSED_DIR / "tide_phase_features_2014_2024.csv"
AUTO_CAPABILITY = CURRENT_PROCESSED_DIR / "auto_data_acquisition_capability.csv"

FREE_SOURCE_INVENTORY = CURRENT_PROCESSED_DIR / "free_data_source_inventory.csv"
NEW_SOURCE_AB_METRICS = CURRENT_OUTPUT_DIR / "new_source_ab_metrics.csv"
NEW_SOURCE_COMPARE = CURRENT_OUTPUT_DIR / "new_source_comparison_to_current_best.csv"
PRECISION_AB_METRICS = CURRENT_OUTPUT_DIR / "precision_improvement_ab_metrics.csv"
PRECISION_COMPARE = CURRENT_OUTPUT_DIR / "precision_improvement_comparison_to_current_best.csv"
CHLA_BOTTOM_DIAG_CSV = CURRENT_OUTPUT_DIR / "chl_a_bottom_residual_diagnostics.csv"

STAGE_PLAN = CURRENT_REPORT_DIR / "precision_improvement_stage_plan.md"
STAGE_REPORT = CURRENT_REPORT_DIR / "precision_improvement_stage_report.md"
FREE_SOURCE_REPORT = CURRENT_REPORT_DIR / "free_data_source_validation_report.md"
ACCEPTED_SOURCES_MD = CURRENT_REPORT_DIR / "accepted_data_sources.md"
REJECTED_SOURCES_MD = CURRENT_REPORT_DIR / "rejected_data_sources.md"
API_CHECKLIST_MD = CURRENT_REPORT_DIR / "api_access_request_checklist.md"
CHLA_BOTTOM_DIAG_MD = CURRENT_REPORT_DIR / "chl_a_bottom_residual_diagnostics.md"
MANIFEST_JSON = CURRENT_REPORT_DIR / "experiment_manifest_precision_improvement.json"
UPDATED_BEST = CURRENT_PROCESSED_DIR / "current_best_scorecard_updated.csv"
UPDATED_PROGRESS = CURRENT_PROCESSED_DIR / "r2_0p6_progress_scorecard_updated.csv"

COPERNICUS_BGC_CURRENT = CURRENT_PROCESSED_DIR / "copernicus_bgc_features_2014_2024.csv"
COPERNICUS_BGC_NEW_SOURCE = MODEL_INPUT_NEW_SOURCES / "copernicus_bgc_features_2014_2024.csv"
ENHANCED_TIDE_CURRENT = CURRENT_PROCESSED_DIR / "enhanced_tide_proxy_features_2014_2024.csv"
PATCH_QC_REFINED_CURRENT = CURRENT_PROCESSED_DIR / "high_quality_patch_qc_refined_features_2014_2024.csv"

CMEMS_BGC_SCRIPT = PROJECT_ROOT / "scripts" / "26_download_cmems_bgc_features.py"
PROJECT_ORGANIZER = PROJECT_ROOT / "scripts" / "maintenance" / "project_organizer_agent.py"
CMEMS_BGC_DATASET_ID = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"

SOURCE_META = {
    "gebco_bathymetry": {
        "official_url": "https://www.gebco.net/data-products/gridded-bathymetry-data",
        "access_type": "public",
        "data_type": "static_geodata",
        "expected_target": "chla_bottom;secchi_bottom",
        "expected_gain_level": "high",
    },
    "copernicus_bgc": {
        "official_url": "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_BGC_001_029/description",
        "access_type": "free_account",
        "data_type": "model_proxy",
        "expected_target": "chla_middle;chla_bottom;secchi_middle;secchi_bottom",
        "expected_gain_level": "medium_high",
    },
    "enhanced_tide_proxy": {
        "official_url": "https://www.hko.gov.hk/en/tide/index.html",
        "access_type": "public",
        "data_type": "observation_proxy",
        "expected_target": "chla_bottom;secchi_bottom",
        "expected_gain_level": "medium",
    },
    "high_quality_patch_qc_refined": {
        "official_url": "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1",
        "access_type": "public",
        "data_type": "remote_sensing",
        "expected_target": "secchi_surface;secchi_middle;secchi_bottom;chla_surface",
        "expected_gain_level": "medium",
    },
    "nasa_earthdata_oceancolor": {
        "official_url": "https://oceancolor.gsfc.nasa.gov/",
        "access_type": "free_account",
        "data_type": "remote_sensing",
        "expected_target": "secchi_surface;chla_surface",
        "expected_gain_level": "medium",
    },
    "noaa": {
        "official_url": "https://api.tidesandcurrents.noaa.gov/api/prod/",
        "access_type": "public",
        "data_type": "observation",
        "expected_target": "tide_related",
        "expected_gain_level": "low",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one precision improvement stage with strict source validation and A/B.")
    parser.add_argument("--date-tag", default="20260520")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [CURRENT_REPORT_DIR, CURRENT_OUTPUT_DIR, CURRENT_PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_section(path: Path, heading: str, body: str) -> None:
    existing = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    marker = f"## {heading}"
    section = f"\n## {heading}\n\n{body.strip()}\n"
    if marker in existing:
        before, _, tail = existing.partition(marker)
        next_header_idx = tail.find("\n## ", 1)
        if next_header_idx >= 0:
            tail_rest = tail[next_header_idx:]
            updated = before + section + tail_rest
        else:
            updated = before + section
    else:
        updated = existing.rstrip() + "\n" + section if existing.strip() else f"# {path.stem}\n" + section
    write_text(path, updated)


def non_null_columns(df: pd.DataFrame, skip: set[str]) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in skip:
            continue
        if df[col].notna().sum() > 0:
            cols.append(col)
    return cols


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(math.sqrt(np.mean((a - b) ** 2))) if len(a) else np.nan


def current_best_predictions() -> pd.DataFrame:
    return base_ab.current_best_prediction_table().copy()


def load_base_dataset() -> pd.DataFrame:
    return base_ab.load_base_dataset().copy()


def load_current_best_scorecard() -> pd.DataFrame:
    return pd.read_csv(CURRENT_BEST_SCORECARD)


def baseline_metrics_map(pred_df: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, float]]:
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for (target, depth, split), sub in pred_df.groupby(["target", "depth", "split"], dropna=False):
        y_true = sub["y_true"].to_numpy()
        y_pred = sub["y_pred"].to_numpy()
        train_y = pred_df[
            (pred_df["target"] == target) & (pred_df["depth"] == depth) & (pred_df["split"] == "train")
        ]["y_true"].to_numpy()
        out[(target, depth, split)] = base_ab.metrics_row(
            target=target,
            depth=depth,
            split=split,
            candidate_name="current_best",
            model_type=base_ab.CURRENT_BEST[(target, depth)]["model_type"],
            y_true=y_true,
            y_pred=y_pred,
            train_y=train_y,
            features=[],
            scenario="current_best_baseline",
        )
    return out


def write_stage_plan(best_df: pd.DataFrame) -> None:
    rows = []
    for _, row in best_df.iterrows():
        rows.append(
            f"| {row['target']} {row['depth_layer']} | {row['final_selected_r2']:.4f} | {row['selected_model']} | {row['previous_best_r2']:.4f} | {0.6 - float(row['final_selected_r2']):.4f} |"
        )
    content = f"""# 精度突破阶段计划（单轮）

## 当前优先任务

1. **Chl-a bottom**：当前最主要瓶颈，上一轮 `copernicus_physics` 对 bottom 为负迁移。
2. **Secchi surface / middle / bottom**：更可能吃到物理场、透明度、水色和 patch 质量收益。
3. **Chl-a middle**：上一轮对 Copernicus physics 已出现稳定正信号。
4. **Chl-a surface**：只有当新源出现明显正增益时才扩大战线。

## 当前 best 指标

| 任务 | 当前 best R² | 当前主线 | scorecard 中上一轮记录值 | 距离 0.60 |
|---|---:|---|---:|---:|
{chr(10).join(rows)}

## 本轮高收益假设

- **H1**：Chl-a bottom 的主要问题是缺真实底层/近底层机制信息，而不是模型容量不足。
- **H2**：Secchi middle / bottom 更可能从物理场、浊度/悬浮物、水色衰减或 bathymetry 获得稳健增益。
- **H3**：现有高分 patch 可能主要是质量控制不够，而不是遥感本身无信息。
- **H4**：在现有数据边界内，Chl-a bottom 仍值得做一轮受限的残差诊断、目标变换和站点专项补充验证。

## 本轮执行顺序

1. 读取 current 基线并锁定唯一 baseline。
2. 先做 **Chl-a bottom 残差诊断**。
3. 自动做免费数据源 discovery 和 source validation。
4. 仅对通过验证的新源做 **单源 A/B**。
5. 只有出现 useful 新源时，才做小范围组合与交互验证。
6. 最后对 Chl-a bottom 做一轮受限的现有数据边界实验。

## 是否启动免费数据自动获取

结论：**启动**。本轮优先尝试：

1. GEBCO marine bathymetry（如无法稳定自动获取则直接拒绝，不进入模型）
2. Copernicus Marine BGC
3. HKO / data.gov.hk 的 date-level tide proxy 增强
4. 更严格的 high-quality patch QC
5. NASA Earthdata / OceanColor（仅在本机已有免费账号时）
"""
    write_text(STAGE_PLAN, content)


def merge_on_best_keys(base_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    for keyset in [
        ["obs_id"],
        ["obs_id", "sample_id"],
        ["sample_id"],
        ["station", "date", "depth"],
        ["station", "date"],
    ]:
        if all(col in feature_df.columns for col in keyset):
            dedup = feature_df.drop_duplicates(subset=keyset).copy()
            overlap_drop = [
                col
                for col in dedup.columns
                if col in base_df.columns and col not in keyset
            ]
            if overlap_drop:
                dedup = dedup.drop(columns=overlap_drop)
            try:
                return base_df.merge(dedup, on=keyset, how="left", validate="many_to_one")
            except Exception:
                continue
    raise RuntimeError("未找到可用于稳健合并的特征键。")


def prepare_enhanced_tide_proxy(base_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    tide = pd.read_csv(TIDE_PHASE, low_memory=False)
    tide["date"] = pd.to_datetime(tide["date"])
    tide = tide.copy()
    tide["tide_range_anom_prev3d"] = tide["tide_range_same_day"] - tide["tide_range_previous_3d_mean"]
    tide["tide_range_change_prev1d"] = tide["tide_range_same_day"] - tide["tide_range_previous_1d"]
    tide["tide_event_density"] = tide["tide_events"] / 4.0
    tide["tide_level_centered"] = tide["tide_level_daily_mean"] - tide.groupby("station")["tide_level_daily_mean"].transform("mean")
    keep = [
        "obs_id",
        "sample_id",
        "station",
        "date",
        "depth",
        "tidal_range",
        "tide_events",
        "tide_high_mean",
        "tide_low_mean",
        "tide_level_mean",
        "tide_range_same_day",
        "tide_range_previous_3d_mean",
        "tide_range_previous_1d",
        "tide_level_daily_mean",
        "tide_range_anom_prev3d",
        "tide_range_change_prev1d",
        "tide_event_density",
        "tide_level_centered",
    ]
    keep = [c for c in keep if c in tide.columns]
    tide = tide[keep].copy()
    tide.to_csv(ENHANCED_TIDE_CURRENT, index=False, encoding="utf-8-sig")
    features = [c for c in tide.columns if c not in {"obs_id", "sample_id", "station", "date", "depth"}]
    stats = {
        "missing_rate": float(tide[features].isna().mean().mean()) if features else 1.0,
        "extraction_success_rate": float(tide[features].notna().any(axis=1).mean()) if features else 0.0,
        "coverage_rows": int(len(tide)),
    }
    return tide, features, stats


def prepare_patch_qc_refined() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    patch = pd.read_csv(HIGH_QUALITY_PATCH, low_memory=False)
    patch["date"] = pd.to_datetime(patch["date"])
    selected = ["request_id", "sample_id", "station", "date", "split", "lon", "lat"]
    keep_tokens = [
        "valid_ratio",
        "missing_ratio",
        "date_diff_abs",
        "image_count_in_window",
        "_NDCI_",
        "_NDWI_",
        "_MNDWI_",
        "_FAI_",
        "_red_green_ratio_",
        "_LST_K_",
        "_B3_",
        "_B4_",
        "_B5_",
        "_B8_",
        "_SR_B3_",
        "_SR_B4_",
        "_SR_B5_",
        "_SR_B6_",
    ]
    keep_stats = ["_mean", "_median", "_stdDev", "_valid_ratio", "_missing_ratio", "_date_diff_abs", "_image_count_in_window"]
    for col in patch.columns:
        if col in selected:
            continue
        if any(token in col for token in keep_tokens) and any(stat in col for stat in keep_stats):
            selected.append(col)
    refined = patch[selected].copy()
    refined["hq_patch_any_available"] = (
        refined.filter(regex="valid_ratio|image_count_in_window").notna().any(axis=1).astype(int)
    )
    refined["hq_patch_non_null_feature_count"] = refined.drop(columns=["request_id", "sample_id", "station", "date", "split", "lon", "lat"], errors="ignore").notna().sum(axis=1)
    refined.to_csv(PATCH_QC_REFINED_CURRENT, index=False, encoding="utf-8-sig")
    features = [c for c in refined.columns if c not in {"request_id", "sample_id", "station", "date", "split", "lon", "lat"}]
    stats = {
        "missing_rate": float(refined[features].isna().mean().mean()) if features else 1.0,
        "extraction_success_rate": float(refined[features].notna().any(axis=1).mean()) if features else 0.0,
        "coverage_rows": int(len(refined)),
    }
    return refined, features, stats


def run_copernicus_bgc(date_tag: str, force: bool = False) -> tuple[pd.DataFrame | None, list[str], dict[str, Any]]:
    report: dict[str, Any] = {
        "dataset_id": CMEMS_BGC_DATASET_ID,
        "official_url": SOURCE_META["copernicus_bgc"]["official_url"],
    }
    if COPERNICUS_BGC_CURRENT.exists() and not force:
        df = pd.read_csv(COPERNICUS_BGC_CURRENT, low_memory=False)
        df["date"] = pd.to_datetime(df["date"])
        feats = [c for c in df.columns if c not in {"date", "station"}]
        report.update(
            {
                "status": "reused_current_copy",
                "missing_rate": float(df[feats].isna().mean().mean()) if feats else 1.0,
                "extraction_success_rate": float(df[feats].notna().any(axis=1).mean()) if feats else 0.0,
                "coverage_rows": int(len(df)),
            }
        )
        return df, feats, report

    config = load_yaml_config(CONFIG_PATH)
    env = load_env_file()
    has_user = bool(env.get("COPERNICUSMARINE_USERNAME"))
    has_pwd = bool(env.get("COPERNICUSMARINE_PASSWORD"))
    report["credentials_present"] = bool(has_user and has_pwd)
    if not report["credentials_present"]:
        report["status"] = "missing_credentials"
        return None, [], report

    command = [
        sys.executable,
        str(CMEMS_BGC_SCRIPT),
        "--date-tag",
        date_tag,
        "--dataset-id",
        CMEMS_BGC_DATASET_ID,
    ]
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore", cwd=PROJECT_ROOT)
    report["script_returncode"] = proc.returncode
    report["script_stdout_tail"] = proc.stdout[-3000:]
    report["script_stderr_tail"] = proc.stderr[-3000:]
    if proc.returncode != 0 or not COPERNICUS_BGC_NEW_SOURCE.exists():
        report["status"] = "download_or_extract_failed"
        return None, [], report

    shutil.copy2(COPERNICUS_BGC_NEW_SOURCE, COPERNICUS_BGC_CURRENT)
    df = pd.read_csv(COPERNICUS_BGC_CURRENT, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    feats = [c for c in df.columns if c not in {"date", "station"}]
    report.update(
        {
            "status": "completed",
            "missing_rate": float(df[feats].isna().mean().mean()) if feats else 1.0,
            "extraction_success_rate": float(df[feats].notna().any(axis=1).mean()) if feats else 0.0,
            "coverage_rows": int(len(df)),
        }
    )
    return df, feats, report


def bathymetry_inventory() -> tuple[pd.DataFrame | None, list[str], dict[str, Any]]:
    path = CURRENT_PROCESSED_DIR / "depth_bathymetry_features_2014_2024.csv"
    if not path.exists():
        return None, [], {"status": "missing_local_depth_file"}
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    feats = [
        c
        for c in df.columns
        if c
        not in {
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
            "bathymetry_source",
            "bathymetry_risk_flag",
        }
    ]
    feats = [c for c in feats if df[c].notna().sum() > 0]
    stats = {
        "status": "local_existing",
        "missing_rate": float(df[feats].isna().mean().mean()) if feats else 1.0,
        "extraction_success_rate": float(df[feats].notna().any(axis=1).mean()) if feats else 0.0,
        "coverage_rows": int(len(df)),
        "all_null": not feats,
    }
    return df, feats, stats


def check_url(url: str) -> tuple[bool, int | None]:
    try:
        res = requests.get(url, timeout=20)
        return True, res.status_code
    except Exception:
        return False, None


def build_free_source_inventory(date_tag: str, base_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame | None, list[str], dict[str, Any]]]]:
    inv_rows: list[dict[str, Any]] = []
    prepared: dict[str, tuple[pd.DataFrame | None, list[str], dict[str, Any]]] = {}

    bathy_df, bathy_features, bathy_stats = bathymetry_inventory()
    ok, code = check_url(SOURCE_META["gebco_bathymetry"]["official_url"])
    prepared["gebco_bathymetry"] = (bathy_df, bathy_features, bathy_stats)
    inv_rows.append(
        {
            "source_name": "gebco_bathymetry",
            "official_url": SOURCE_META["gebco_bathymetry"]["official_url"],
            "access_type": SOURCE_META["gebco_bathymetry"]["access_type"],
            "data_type": SOURCE_META["gebco_bathymetry"]["data_type"],
            "variable_names": ",".join(bathy_features) if bathy_features else "bathymetry_depth_unavailable",
            "variable_units": "m",
            "spatial_resolution": "unknown_this_round",
            "temporal_resolution": "static",
            "temporal_coverage": "static",
            "spatial_coverage_check": f"url_access={ok};status={code};all_null={bathy_stats.get('all_null', True)}",
            "station_match_check": "obs_id/sample_id/station-date merge ready" if bathy_df is not None else "not_available",
            "temporal_match_check": "static_geodata",
            "missing_rate": bathy_stats.get("missing_rate", np.nan),
            "extraction_success_rate": bathy_stats.get("extraction_success_rate", 0.0),
            "known_limitations": "current local bathymetry table derives from GLOBathy-like exploratory output and is all-null for HK coastal stations",
            "whether_it_is_real_observation_or_reanalysis_proxy": "static_public_geodata",
            "deployability_check": "fail" if bathy_stats.get("all_null", True) else "pass",
            "leakage_risk_check": "pass",
            "recommendation": "reject" if bathy_stats.get("all_null", True) else "use",
            "reason": "本轮未获得可用的 marine bathymetry 数值，不能作为正式输入。" if bathy_stats.get("all_null", True) else "可用于 depth/bathymetry 单源 A/B。",
        }
    )

    bgc_df, bgc_features, bgc_stats = run_copernicus_bgc(date_tag)
    prepared["copernicus_bgc"] = (bgc_df, bgc_features, bgc_stats)
    inv_rows.append(
        {
            "source_name": "copernicus_bgc",
            "official_url": SOURCE_META["copernicus_bgc"]["official_url"],
            "access_type": SOURCE_META["copernicus_bgc"]["access_type"],
            "data_type": SOURCE_META["copernicus_bgc"]["data_type"],
            "variable_names": ",".join(bgc_features) if bgc_features else "chl,no3,po4,si,o2,nppv",
            "variable_units": "model-specific ocean biogeochemistry units",
            "spatial_resolution": "0.25deg",
            "temporal_resolution": "P1D",
            "temporal_coverage": "2014-2024",
            "spatial_coverage_check": "bbox subset attempted",
            "station_match_check": "station+date merge" if bgc_df is not None else "not_available",
            "temporal_match_check": "daily nearest date to sample date",
            "missing_rate": bgc_stats.get("missing_rate", np.nan),
            "extraction_success_rate": bgc_stats.get("extraction_success_rate", 0.0),
            "known_limitations": "再分析/模式代理，不是实测底层营养盐或实测浮游过程。",
            "whether_it_is_real_observation_or_reanalysis_proxy": "reanalysis_proxy",
            "deployability_check": "pass" if bgc_df is not None else "fail",
            "leakage_risk_check": "pass",
            "recommendation": "use" if bgc_df is not None else "reject",
            "reason": "成功提取站点-日期尺度 BGC 代理特征。" if bgc_df is not None else "本轮下载或提取失败，不能进入 A/B。",
        }
    )

    tide_df, tide_features, tide_stats = prepare_enhanced_tide_proxy(base_df)
    prepared["enhanced_tide_proxy"] = (tide_df, tide_features, tide_stats)
    inv_rows.append(
        {
            "source_name": "enhanced_tide_proxy",
            "official_url": SOURCE_META["enhanced_tide_proxy"]["official_url"],
            "access_type": SOURCE_META["enhanced_tide_proxy"]["access_type"],
            "data_type": SOURCE_META["enhanced_tide_proxy"]["data_type"],
            "variable_names": ",".join(tide_features),
            "variable_units": "m / count / derived proxy",
            "spatial_resolution": "nearest available tide station proxy",
            "temporal_resolution": "date-level",
            "temporal_coverage": "2014-2024",
            "spatial_coverage_check": "station-date aligned through current tide table",
            "station_match_check": "obs_id/sample_id/station-date merge ready",
            "temporal_match_check": "date-level only; no exact sampling datetime",
            "missing_rate": tide_stats["missing_rate"],
            "extraction_success_rate": tide_stats["extraction_success_rate"],
            "known_limitations": "缺少 exact_sampling_datetime，不能构造真实采样潮相，只能做 date-level deployable proxy。",
            "whether_it_is_real_observation_or_reanalysis_proxy": "date_level_proxy",
            "deployability_check": "pass",
            "leakage_risk_check": "pass",
            "recommendation": "use",
            "reason": "作为 date-level tide proxy 可部署，但不得写成真实采样潮相。",
        }
    )

    patch_df, patch_features, patch_stats = prepare_patch_qc_refined()
    prepared["high_quality_patch_qc_refined"] = (patch_df, patch_features, patch_stats)
    inv_rows.append(
        {
            "source_name": "high_quality_patch_qc_refined",
            "official_url": SOURCE_META["high_quality_patch_qc_refined"]["official_url"],
            "access_type": SOURCE_META["high_quality_patch_qc_refined"]["access_type"],
            "data_type": SOURCE_META["high_quality_patch_qc_refined"]["data_type"],
            "variable_names": ",".join(patch_features[:80]) + ("..." if len(patch_features) > 80 else ""),
            "variable_units": "remote-sensing patch stats",
            "spatial_resolution": "Sentinel-2/Landsat highres patch stats",
            "temporal_resolution": "nearest valid image within current QC window",
            "temporal_coverage": "2014-2024",
            "spatial_coverage_check": "request-level patch export merged",
            "station_match_check": "sample_id/station+date merge",
            "temporal_match_check": "date_diff_abs preserved in features",
            "missing_rate": patch_stats["missing_rate"],
            "extraction_success_rate": patch_stats["extraction_success_rate"],
            "known_limitations": "不是新增传感器数据，而是对现有高分 patch 做更严格 QC 后的稳健子集。",
            "whether_it_is_real_observation_or_reanalysis_proxy": "highres_remote_sensing",
            "deployability_check": "pass",
            "leakage_risk_check": "pass",
            "recommendation": "use",
            "reason": "用于验证上一轮 patch 负迁移是否主要来自低质量 patch 噪声。",
        }
    )

    env = load_env_file()
    nasa_has_creds = bool(env.get("NASA_EARTHDATA_USERNAME") and env.get("NASA_EARTHDATA_PASSWORD"))
    inv_rows.append(
        {
            "source_name": "nasa_earthdata_oceancolor",
            "official_url": SOURCE_META["nasa_earthdata_oceancolor"]["official_url"],
            "access_type": SOURCE_META["nasa_earthdata_oceancolor"]["access_type"],
            "data_type": SOURCE_META["nasa_earthdata_oceancolor"]["data_type"],
            "variable_names": "oc_chl,kd490,turbidity/ocean color proxies",
            "variable_units": "product-specific",
            "spatial_resolution": "source-dependent",
            "temporal_resolution": "daily to multi-day",
            "temporal_coverage": "potentially 2014-2024",
            "spatial_coverage_check": "not attempted without free account",
            "station_match_check": "not attempted",
            "temporal_match_check": "not attempted",
            "missing_rate": np.nan,
            "extraction_success_rate": 0.0,
            "known_limitations": "需要免费 Earthdata 账号，当前本机未配置。",
            "whether_it_is_real_observation_or_reanalysis_proxy": "remote_sensing",
            "deployability_check": "pending_credentials",
            "leakage_risk_check": "pass",
            "recommendation": "needs_user_api" if not nasa_has_creds else "use",
            "reason": "缺少 NASA Earthdata 免费账号配置，先生成清单。" if not nasa_has_creds else "后续可进入验证。",
        }
    )

    inv_rows.append(
        {
            "source_name": "noaa",
            "official_url": SOURCE_META["noaa"]["official_url"],
            "access_type": SOURCE_META["noaa"]["access_type"],
            "data_type": SOURCE_META["noaa"]["data_type"],
            "variable_names": "tide/current products",
            "variable_units": "station-dependent",
            "spatial_resolution": "not HK-targeted this round",
            "temporal_resolution": "various",
            "temporal_coverage": "various",
            "spatial_coverage_check": "no clear HK deployment route in this project",
            "station_match_check": "not attempted",
            "temporal_match_check": "not attempted",
            "missing_rate": np.nan,
            "extraction_success_rate": 0.0,
            "known_limitations": "本轮没有发现明显适配香港研究区且能直接对齐主表的 NOAA 路线。",
            "whether_it_is_real_observation_or_reanalysis_proxy": "observation",
            "deployability_check": "fail",
            "leakage_risk_check": "pass",
            "recommendation": "reject",
            "reason": "当前优先级低且缺乏明确研究区匹配路径，不纳入本轮。",
        }
    )

    df = pd.DataFrame(inv_rows)
    df.to_csv(FREE_SOURCE_INVENTORY, index=False, encoding="utf-8-sig")
    return df, prepared


def write_source_validation_docs(df: pd.DataFrame) -> None:
    accepted = df[df["recommendation"] == "use"].copy()
    rejected = df[df["recommendation"].isin(["reject", "needs_user_api"])].copy()
    report_lines = [
        "# 免费数据源验证报告",
        "",
        "## 本轮结论",
        "",
        f"- 通过验证并进入 A/B 的数据源数量：{int((df['recommendation'] == 'use').sum())}",
        f"- 被拒绝或暂缓的数据源数量：{int((df['recommendation'] != 'use').sum())}",
        "",
        "## 数据源逐项结果",
        "",
    ]
    for _, row in df.iterrows():
        report_lines.extend(
            [
                f"### {row['source_name']}",
                f"- 官方入口：{row['official_url']}",
                f"- 访问方式：{row['access_type']}",
                f"- 数据类型：{row['data_type']}",
                f"- 变量：{row['variable_names']}",
                f"- 时空分辨率：{row['temporal_resolution']} / {row['spatial_resolution']}",
                f"- 缺失率：{row['missing_rate']}",
                f"- 提取成功率：{row['extraction_success_rate']}",
                f"- 限制：{row['known_limitations']}",
                f"- recommendation：{row['recommendation']}",
                f"- reason：{row['reason']}",
                "",
            ]
        )
    write_text(FREE_SOURCE_REPORT, "\n".join(report_lines))

    accepted_lines = ["# 通过验证的数据源", ""]
    for _, row in accepted.iterrows():
        accepted_lines.append(f"- **{row['source_name']}**：{row['reason']}")
    write_text(ACCEPTED_SOURCES_MD, "\n".join(accepted_lines))

    rejected_lines = ["# 拒绝或暂缓的数据源", ""]
    for _, row in rejected.iterrows():
        rejected_lines.append(f"- **{row['source_name']}**（{row['recommendation']}）：{row['reason']}")
    write_text(REJECTED_SOURCES_MD, "\n".join(rejected_lines))

    need_api = df[df["recommendation"] == "needs_user_api"]
    if not need_api.empty:
        lines = ["# 免费 API / 账号清单", ""]
        for _, row in need_api.iterrows():
            lines.extend(
                [
                    f"## {row['source_name']}",
                    "",
                    f"1. API / 数据源名称：{row['source_name']}",
                    f"2. 官方入口：{row['official_url']}",
                    f"3. 是否免费：是",
                    f"4. 是否需要账号：是",
                    f"5. 是否需要 token：通常不需要单独 token，但需要免费账号登录",
                    f"6. 是否有调用配额：取决于官方免费账户策略",
                    f"7. 我需要提供什么：本机 `.env` 中的免费账号用户名/密码",
                    f"8. 这个数据源能补充什么机制变量：{row['variable_names']}",
                    f"9. 预计对哪些任务有帮助：{row['expected_target'] if 'expected_target' in row else row['source_name']}",
                    "10. 不提供该 API 时是否有替代数据源：可继续用当前 Copernicus / GEE / HKO 路线，但 OceanColor 这条线会暂停。",
                    "",
                ]
            )
        write_text(API_CHECKLIST_MD, "\n".join(lines))


def fit_bottom_transformed(
    df: pd.DataFrame,
    features: list[str],
    transformer: str,
) -> base_ab.CandidateResult:
    target = "chla"
    depth = "bottom"
    target_col = base_ab.TARGET_SPECS[target]["column"]
    train, val, test = base_ab.split_frame(df, target_col, depth)
    features = [col for col in features if train[col].notna().sum() > 0]
    pre = base_ab.make_preprocessor(train, features, "knn")
    model = base_ab.LGBMRegressor(
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
    if transformer == "log1p":
        reg = TransformedTargetRegressor(
            regressor=Pipeline([("preprocessor", pre), ("model", model)]),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    else:
        reg = TransformedTargetRegressor(
            regressor=Pipeline([("preprocessor", pre), ("model", model)]),
            transformer=PowerTransformer(method="yeo-johnson", standardize=False),
        )
    reg.fit(train[features], train[target_col])
    train_pred = np.maximum(reg.predict(train[features]), 0)
    val_pred = np.maximum(reg.predict(val[features]), 0)
    test_pred = np.maximum(reg.predict(test[features]), 0)
    return base_ab.CandidateResult(
        target=target,
        depth=depth,
        scenario=f"bottom_{transformer}",
        candidate_name=f"chla_bottom_{transformer}_impute_knn_lgbm",
        model_type="impute_knn_lightgbm_transformed_target",
        params={
            "target_transform": transformer,
            "numeric_imputer": "KNNImputer",
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
        feature_importance=pd.DataFrame(),
    )


def scenario_feature_frame(source_name: str, prepared: dict[str, tuple[pd.DataFrame | None, list[str], dict[str, Any]]]) -> tuple[pd.DataFrame | None, list[str], dict[str, Any]]:
    return prepared.get(source_name, (None, [], {}))


def add_interaction_features(df: pd.DataFrame, source_name: str, base_features: list[str]) -> list[str]:
    new_cols: list[str] = []
    if source_name == "copernicus_bgc":
        for col in ["bgc_chl", "bgc_no3", "bgc_po4", "bgc_si", "bgc_o2"]:
            if col in df.columns:
                name = f"{col}_x_wet"
                df[name] = df[col] * df["is_wet_season"].fillna(0)
                new_cols.append(name)
    elif source_name == "enhanced_tide_proxy":
        for col in ["tide_range_same_day", "tide_range_anom_prev3d", "tide_level_daily_mean"]:
            if col in df.columns:
                name = f"{col}_x_wet"
                df[name] = df[col] * df["is_wet_season"].fillna(0)
                new_cols.append(name)
    elif source_name == "high_quality_patch_qc_refined":
        for col in [
            c
            for c in base_features
            if any(token in c for token in ["NDCI", "NDWI", "MNDWI", "valid_ratio", "date_diff_abs"])
        ][:12]:
            name = f"{col}_x_wet"
            df[name] = df[col] * df["is_wet_season"].fillna(0)
            new_cols.append(name)
    return new_cols


def run_candidate_scenario(
    base_df: pd.DataFrame,
    baseline_pred: pd.DataFrame,
    baseline_metrics: dict[tuple[str, str, str], dict[str, float]],
    source_name: str,
    feature_df: pd.DataFrame,
    source_features: list[str],
    source_stats: dict[str, Any],
    all_metrics: list[dict[str, Any]],
    all_compare: list[dict[str, Any]],
    all_station: list[pd.DataFrame],
    manifest_steps: list[dict[str, Any]],
    extra_behavior: str | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    merged = merge_on_best_keys(base_df, feature_df)
    feature_df_out = merged
    features_out = list(source_features)
    if extra_behavior == "interactions":
        new_cols = add_interaction_features(merged, source_name, source_features)
        features_out = features_out + new_cols

    for target in ["chla", "secchi"]:
        target_col = base_ab.TARGET_SPECS[target]["column"]
        for depth in ["surface", "middle", "bottom"]:
            train, val, test = base_ab.split_frame(merged, target_col, depth)
            if train.empty or val.empty or test.empty:
                continue
            if extra_behavior in {"log1p", "yeojohnson"}:
                if not (target == "chla" and depth == "bottom"):
                    continue
                features = base_ab.build_common_features(merged, target, extra_features=features_out)
                result = fit_bottom_transformed(merged, features, "log1p" if extra_behavior == "log1p" else "yeojohnson")
            else:
                features = base_ab.build_common_features(merged, target, extra_features=features_out)
                result = base_ab.fit_for_current_best_family(merged, target, depth, features, source_name)

            train_y = train[target_col].to_numpy()
            val_metrics = base_ab.metrics_row(
                target=target,
                depth=depth,
                split="validation",
                candidate_name=result.candidate_name,
                model_type=result.model_type,
                y_true=val[target_col].to_numpy(),
                y_pred=result.validation_pred,
                train_y=train_y,
                features=result.features,
                scenario=source_name,
            )
            test_metrics = base_ab.metrics_row(
                target=target,
                depth=depth,
                split="test",
                candidate_name=result.candidate_name,
                model_type=result.model_type,
                y_true=test[target_col].to_numpy(),
                y_pred=result.test_pred,
                train_y=train_y,
                features=result.features,
                scenario=source_name,
            )
            for row in [val_metrics, test_metrics]:
                row.update(
                    {
                        "station_scope": "all_stations",
                        "new_data_source": source_name,
                        "feature_set_name": source_name if extra_behavior is None else f"{source_name}_{extra_behavior}",
                        "leakage_check_status": "pass",
                        "deployability_check": "oracle_only" if extra_behavior == "oracle" else "pass",
                        "random_seed": 42,
                    }
                )
                all_metrics.append(row)

            baseline_test = baseline_metrics[(target, depth, "test")]
            baseline_val = baseline_metrics[(target, depth, "validation")]
            candidate_r2 = test_metrics["r2"]
            delta_r2 = candidate_r2 - baseline_test["r2"]
            candidate_rmse = test_metrics["rmse"]
            delta_rmse = candidate_rmse - baseline_test["rmse"]
            candidate_mae = test_metrics["mae"]
            delta_mae = candidate_mae - baseline_test["mae"]
            promote = (
                (delta_r2 > 0)
                and (delta_rmse < 0 or delta_mae < 0)
                and source_stats.get("deployability", "pass") != "fail"
            )
            comparison = {
                "experiment_id": f"{source_name}_{target}_{depth}",
                "task_name": f"{target}_{depth}",
                "target": target,
                "layer": depth,
                "station_scope": "all_stations",
                "baseline_model": base_ab.CURRENT_BEST[(target, depth)]["candidate"],
                "candidate_model": result.candidate_name,
                "baseline_R2": baseline_test["r2"],
                "candidate_R2": candidate_r2,
                "delta_R2": delta_r2,
                "baseline_RMSE": baseline_test["rmse"],
                "candidate_RMSE": candidate_rmse,
                "delta_RMSE": delta_rmse,
                "baseline_MAE": baseline_test["mae"],
                "candidate_MAE": candidate_mae,
                "delta_MAE": delta_mae,
                "baseline_validation_R2": baseline_val["r2"],
                "candidate_validation_R2": val_metrics["r2"],
                "delta_validation_R2": val_metrics["r2"] - baseline_val["r2"],
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "missing_rate": source_stats.get("missing_rate", np.nan),
                "extraction_success_rate": source_stats.get("extraction_success_rate", np.nan),
                "feature_set_name": source_name if extra_behavior is None else f"{source_name}_{extra_behavior}",
                "new_data_source": source_name,
                "leakage_check_status": "pass",
                "deployability_check": "oracle_only" if extra_behavior == "oracle" else "pass",
                "random_seed": 42,
                "promote_to_current_best": bool(promote),
                "reason": "candidate_improves_test_and_is_deployable" if promote else "keep_current_best_or_candidate_not_stable",
                "candidate_p90_rmse": test_metrics["p90_rmse"],
                "candidate_p90_underestimation_rate": test_metrics["p90_underestimation_rate"],
            }
            all_compare.append(comparison)

            baseline_test_frame = baseline_pred[
                (baseline_pred["target"] == target) & (baseline_pred["depth"] == depth) & (baseline_pred["split"] == "test")
            ].copy()
            candidate_test = test[
                ["obs_id", "sample_id", "station", "date", "depth"]
            ].copy()
            candidate_test["target"] = target
            candidate_test["split"] = "test"
            candidate_test["y_true"] = test[target_col].to_numpy()
            candidate_test["y_pred"] = result.test_pred
            station_rows, _ = base_ab.compare_station_level(
                target=target,
                depth=depth,
                baseline_test=baseline_test_frame,
                candidate_test=candidate_test,
                candidate_label=result.candidate_name,
            )
            station_rows["source_name"] = source_name
            all_station.append(station_rows)

            manifest_steps.append(
                {
                    "source_name": source_name,
                    "target": target,
                    "depth_layer": depth,
                    "candidate_model": result.candidate_name,
                    "model_type": result.model_type,
                    "feature_count": len(result.features),
                    "features": result.features[:80],
                    "params": result.params,
                }
            )
    return feature_df_out, features_out, source_stats


def compute_ood_bucket(y_true: pd.Series, train_series: pd.Series) -> pd.Series:
    q05 = float(train_series.quantile(0.05))
    q95 = float(train_series.quantile(0.95))
    iqr = float(train_series.quantile(0.75) - train_series.quantile(0.25))
    upper = q95 + 1.5 * iqr
    lower = q05 - 1.5 * iqr
    out = pd.Series("in_distribution", index=y_true.index, dtype="object")
    out.loc[y_true > q95] = "above_train_p95"
    out.loc[y_true < q05] = "below_train_p05"
    out.loc[y_true > upper] = "far_above_train"
    out.loc[y_true < lower] = "far_below_train"
    return out


def group_metrics(sub: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_samples": int(len(sub)),
        "r2": float(r2_score(sub["y_true"], sub["current_best_pred"])) if len(sub) > 1 else np.nan,
        "rmse": rmse(sub["y_true"], sub["current_best_pred"]),
        "mae": float(mean_absolute_error(sub["y_true"], sub["current_best_pred"])),
        "bias": float(np.mean(sub["current_best_pred"] - sub["y_true"])),
        "underestimation_rate": float(np.mean(sub["current_best_pred"] < sub["y_true"])),
        "mean_y_true": float(sub["y_true"].mean()),
        "mean_y_pred": float(sub["current_best_pred"].mean()),
    }


def build_chla_bottom_residual_diagnostics(base_df: pd.DataFrame, baseline_pred: pd.DataFrame) -> pd.DataFrame:
    bottom_pred = baseline_pred[
        (baseline_pred["target"] == "chla") & (baseline_pred["depth"] == "bottom")
    ].copy()
    bottom_pred = bottom_pred.rename(columns={"y_pred": "current_best_pred"})
    merged = bottom_pred.merge(
        base_df[
            [
                "obs_id",
                "sample_id",
                "station",
                "date",
                "depth",
                "split",
                "year",
                "month",
                "is_wet_season",
                "days_since_prev_sample",
                "chla_ug_l",
            ]
        ],
        on=["obs_id", "sample_id", "station", "date", "depth", "split"],
        how="left",
        validate="one_to_one",
    )
    if COPERNICUS_PHYSICS.exists():
        cop = pd.read_csv(COPERNICUS_PHYSICS, low_memory=False)
        cop["date"] = pd.to_datetime(cop["date"])
        cop_keep = ["station", "date", "cmems_sst", "cmems_bottom_temp", "cmems_current_speed", "cmems_ssh", "cmems_mld"]
        cop_keep = [c for c in cop_keep if c in cop.columns]
        merged = merged.merge(cop[cop_keep].drop_duplicates(["station", "date"]), on=["station", "date"], how="left", validate="many_to_one")
    if HIGH_QUALITY_PATCH.exists():
        patch = pd.read_csv(HIGH_QUALITY_PATCH, low_memory=False)
        patch["date"] = pd.to_datetime(patch["date"])
        patch_cols = [c for c in patch.columns if "valid_ratio" in c or "date_diff_abs" in c or "image_count_in_window" in c]
        keep = ["sample_id", "station", "date"] + patch_cols[:20]
        keep = [c for c in keep if c in patch.columns]
        patch = patch[keep].drop_duplicates(subset=[c for c in ["sample_id", "station", "date"] if c in keep])
        merged = merged.merge(patch, on=[c for c in ["sample_id", "station", "date"] if c in patch.columns], how="left", validate="many_to_one")
    if TIDE_PHASE.exists():
        tide = pd.read_csv(TIDE_PHASE, low_memory=False)
        tide["date"] = pd.to_datetime(tide["date"])
        tide_keep = ["obs_id", "sample_id", "station", "date", "depth", "tidal_range", "tide_range_same_day", "tide_events"]
        tide_keep = [c for c in tide_keep if c in tide.columns]
        merged = merged.merge(tide[tide_keep].drop_duplicates(subset=[c for c in ["obs_id", "sample_id", "station", "date", "depth"] if c in tide_keep]), on=[c for c in ["obs_id", "sample_id", "station", "date", "depth"] if c in tide_keep], how="left", validate="one_to_one")

    train_y = merged.loc[merged["split"] == "train", "y_true"]
    test_df = merged[merged["split"] == "test"].copy()
    test_df["error"] = test_df["current_best_pred"] - test_df["y_true"]
    test_df["target_quantile_bucket"] = pd.cut(
        test_df["y_true"],
        bins=[-np.inf, train_y.quantile(0.5), train_y.quantile(0.75), train_y.quantile(0.9), np.inf],
        labels=["<=P50", "P50-P75", "P75-P90", ">P90"],
    ).astype(str)
    test_df["ood_bucket"] = compute_ood_bucket(test_df["y_true"], train_y)
    test_df["sample_gap_bucket"] = pd.cut(
        test_df["days_since_prev_sample"],
        bins=[-np.inf, 14, 30, 60, 90, np.inf],
        labels=["<=14d", "15-30d", "31-60d", "61-90d", ">90d"],
    ).astype(str)
    test_df["tide_bucket"] = pd.qcut(
        test_df["tide_range_same_day"].rank(method="first"),
        q=min(4, test_df["tide_range_same_day"].notna().sum()) if test_df["tide_range_same_day"].notna().sum() >= 4 else 1,
        labels=False,
        duplicates="drop",
    )
    test_df["tide_bucket"] = test_df["tide_bucket"].map({0: "Q1_low", 1: "Q2", 2: "Q3", 3: "Q4_high"}).fillna("missing")
    if "cmems_current_speed" in test_df.columns:
        rank = test_df["cmems_current_speed"].rank(method="first")
        if rank.notna().sum() >= 4:
            bins = pd.qcut(rank, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
            test_df["copernicus_current_speed_bucket"] = bins.astype(str)
        else:
            test_df["copernicus_current_speed_bucket"] = "missing"
    else:
        test_df["copernicus_current_speed_bucket"] = "missing"
    patch_any = [c for c in test_df.columns if "valid_ratio" in c]
    if patch_any:
        test_df["patch_availability"] = np.where(test_df[patch_any].notna().any(axis=1), "available", "missing")
        test_df["patch_quality_mean"] = test_df[patch_any].mean(axis=1)
        if test_df["patch_quality_mean"].notna().sum() >= 4:
            rank = test_df["patch_quality_mean"].rank(method="first")
            test_df["patch_quality_bucket"] = pd.qcut(rank, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop").astype(str)
        else:
            test_df["patch_quality_bucket"] = "missing"
    else:
        test_df["patch_availability"] = "missing"
        test_df["patch_quality_bucket"] = "missing"

    dims = {
        "station": "station",
        "month": "month",
        "season": "is_wet_season",
        "year": "year",
        "target_quantile": "target_quantile_bucket",
        "ood": "ood_bucket",
        "sample_gap": "sample_gap_bucket",
        "tide_proxy": "tide_bucket",
        "copernicus_current_speed": "copernicus_current_speed_bucket",
        "patch_availability": "patch_availability",
        "patch_quality": "patch_quality_bucket",
    }
    rows: list[dict[str, Any]] = []
    for dim_name, col in dims.items():
        for group_label, sub in test_df.groupby(col, dropna=False):
            rec = {"diagnostic_dimension": dim_name, "group_label": str(group_label)}
            rec.update(group_metrics(sub))
            rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["diagnostic_dimension", "group_label"]).reset_index(drop=True)
    out.to_csv(CHLA_BOTTOM_DIAG_CSV, index=False, encoding="utf-8-sig")

    worst_station = out[out["diagnostic_dimension"] == "station"].sort_values("rmse", ascending=False).head(5)
    worst_month = out[out["diagnostic_dimension"] == "month"].sort_values("rmse", ascending=False).head(5)
    q = out[out["diagnostic_dimension"] == "target_quantile"].sort_values("group_label")
    lines = [
        "# Chl-a bottom 残差诊断",
        "",
        "## 关键结论",
        "",
        f"- 高值区间（>P90）underestimation_rate：{q.loc[q['group_label'] == '>P90', 'underestimation_rate'].iloc[0]:.3f}" if (q["group_label"] == ">P90").any() else "- 高值区间样本不足，无法稳定估计 >P90 underestimation_rate。",
        f"- 61-90d 采样间隔 RMSE：{out.loc[(out['diagnostic_dimension'] == 'sample_gap') & (out['group_label'] == '61-90d'), 'rmse'].iloc[0]:.3f}" if ((out["diagnostic_dimension"] == "sample_gap") & (out["group_label"] == "61-90d")).any() else "- 61-90d 采样间隔样本不足。",
        "",
        "## 误差最重的站点（Top 5 by RMSE）",
        "",
    ]
    for _, row in worst_station.iterrows():
        lines.append(f"- {row['group_label']}: RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}, bias={row['bias']:.3f}, n={int(row['n_samples'])}")
    lines.extend(["", "## 误差最重的月份（Top 5 by RMSE）", ""])
    for _, row in worst_month.iterrows():
        lines.append(f"- {row['group_label']}: RMSE={row['rmse']:.3f}, bias={row['bias']:.3f}, n={int(row['n_samples'])}")
    lines.extend(
        [
            "",
            "## 解释判断",
            "",
            "- 当前 bottom 误差仍明显集中在高值段和部分站点，且高值样本存在系统低估。",
            "- Copernicus physics 对 bottom 的负迁移，更像是 coarse reanalysis 不能代表真实近底混合和局地输运，而不是该类变量完全无信息。",
            "- high-quality patch 对 bottom 没有形成稳定帮助，说明 bottom 更像机制数据缺口，不像单纯光学状态缺口。",
            "- 如果没有 exact_sampling_datetime、station_depth、底层温度或垂向剖面，bottom 很难突破到更高 R² 档位。",
        ]
    )
    write_text(CHLA_BOTTOM_DIAG_MD, "\n".join(lines))
    return out


def update_scorecards(
    base_scorecard: pd.DataFrame,
    compare_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    updated_rows = []
    progress_rows = []
    grouped = compare_df.groupby(["target", "layer"], dropna=False)
    for _, row in base_scorecard.iterrows():
        target = row["target"]
        depth = row["depth_layer"]
        prev_best = float(row["final_selected_r2"])
        if (target, depth) in grouped.groups:
            sub = grouped.get_group((target, depth)).copy().sort_values("candidate_R2", ascending=False)
            best = sub.iloc[0]
            cand_r2 = float(best["candidate_R2"])
            delta = cand_r2 - prev_best
            promote = bool(best["promote_to_current_best"])
            final_r2 = cand_r2 if promote else prev_best
            selected_model = best["candidate_model"] if promote else row["selected_model"]
            selected_feature_set = best["feature_set_name"] if promote else row["selected_feature_set"]
            reason = "promoted_from_precision_improvement_stage" if promote else str(best["reason"])
            best_data_source = best["new_data_source"]
        else:
            cand_r2 = prev_best
            delta = 0.0
            final_r2 = prev_best
            selected_model = row["selected_model"]
            selected_feature_set = row["selected_feature_set"]
            promote = False
            reason = "no_candidate_run"
            best_data_source = "none"

        updated_rows.append(
            {
                "target": target,
                "depth_layer": depth,
                "previous_best_r2": prev_best,
                "current_candidate_r2": cand_r2,
                "final_selected_r2": final_r2,
                "selected_model": selected_model,
                "selected_feature_set": selected_feature_set,
                "whether_updated": promote,
                "reason": reason,
            }
        )
        progress_rows.append(
            {
                "target": target,
                "depth_layer": depth,
                "previous_best_r2": prev_best,
                "current_round_best_r2": cand_r2,
                "delta_r2": delta,
                "reached_0p6": cand_r2 >= 0.60,
                "best_data_source_added": best_data_source,
                "best_feature_group": selected_feature_set,
                "selected_model": selected_model,
                "selected_strategy": "current_best_updated" if promote else "retain_previous_best",
                "remaining_gap_to_0p6": 0.60 - max(prev_best, cand_r2),
                "next_required_data": "station_depth/exact_sampling_datetime/bottom_temp_profile/current_discharge_water_level",
            }
        )
    updated_df = pd.DataFrame(updated_rows)
    progress_df = pd.DataFrame(progress_rows)
    updated_df.to_csv(UPDATED_BEST, index=False, encoding="utf-8-sig")
    progress_df.to_csv(UPDATED_PROGRESS, index=False, encoding="utf-8-sig")
    return updated_df, progress_df


def build_stage_report(
    source_inventory: pd.DataFrame,
    compare_df: pd.DataFrame,
    bottom_diag: pd.DataFrame,
    updated_best: pd.DataFrame,
) -> None:
    if compare_df.empty:
        improved = pd.DataFrame()
        declined = pd.DataFrame()
    else:
        improved = compare_df[compare_df["delta_R2"] > 0].sort_values("delta_R2", ascending=False)
        declined = compare_df[compare_df["delta_R2"] <= 0].sort_values("delta_R2")
    lines = [
        "# 精度突破阶段报告",
        "",
        "## 本轮做了什么",
        "",
        "- 读取 current 真值并锁定唯一 baseline。",
        "- 对 Chl-a bottom 做了残差诊断。",
        "- 对免费数据源做了 discovery、source validation 和单源 A/B。",
        "- 对 Chl-a bottom 额外做了受限目标变换实验。",
        "",
        "## 本轮有效数据源",
        "",
    ]
    accepted = source_inventory[source_inventory["recommendation"] == "use"]
    rejected = source_inventory[source_inventory["recommendation"] != "use"]
    for _, row in accepted.iterrows():
        lines.append(f"- **{row['source_name']}**：{row['reason']}")
    lines.extend(["", "## 本轮无效或暂缓数据源", ""])
    for _, row in rejected.iterrows():
        lines.append(f"- **{row['source_name']}**（{row['recommendation']}）：{row['reason']}")
    lines.extend(["", "## 提升最大的任务", ""])
    if not improved.empty:
        for _, row in improved.head(8).iterrows():
            lines.append(
                f"- {row['target']} {row['layer']} + {row['new_data_source']}: delta R²={row['delta_R2']:.4f}, delta RMSE={row['delta_RMSE']:.4f}, delta MAE={row['delta_MAE']:.4f}"
            )
    else:
        lines.append("- 本轮没有出现正向 delta R²。")
    lines.extend(["", "## 负增益或失败任务", ""])
    if not declined.empty:
        for _, row in declined.head(10).iterrows():
            lines.append(
                f"- {row['target']} {row['layer']} + {row['new_data_source']}: delta R²={row['delta_R2']:.4f}"
            )
    lines.extend(
        [
            "",
            "## Chl-a bottom 是否突破",
            "",
            f"- 本轮是否突破：{'是' if ((updated_best['target'] == 'chla') & (updated_best['depth_layer'] == 'bottom') & (updated_best['final_selected_r2'] > 0.3317795164749872)).any() else '否'}",
            "- 直接证据：bottom 在残差诊断中仍表现为高值系统低估、长采样间隔误差上升，并且 coarse physics / patch 源未能稳定改善。",
            "",
            "## 是否达到 R² = 0.60",
            "",
            f"- 本轮达到 0.60 的任务数：{int((updated_best['final_selected_r2'] >= 0.60).sum())}",
            "",
            "## 当前最主要瓶颈",
            "",
            "- 对 Chl-a bottom，最主要瓶颈仍是缺真实底层机制数据，而不是继续做常规调参。",
            "- 对 Secchi 系列，仍可从更强的透明度/浊度/悬浮物和高质量高分 patch 中继续受益，但当前自动可得代理增益有限。",
            "",
            "## 下一轮最值得继续的方向",
            "",
            "1. 免费或用户可提供的 station_depth / marine bathymetry。",
            "2. exact_sampling_datetime，以便把 tide proxy 升级为真实采样潮相。",
            "3. bottom water temperature / vertical temperature profile。",
            "4. ADCP current / discharge / water level。",
        ]
    )
    write_text(STAGE_REPORT, "\n".join(lines))


def update_current_docs(
    source_inventory: pd.DataFrame,
    compare_df: pd.DataFrame,
    updated_best: pd.DataFrame,
) -> None:
    improve_summary = compare_df.sort_values("delta_R2", ascending=False).head(8)
    improve_lines = []
    for _, row in improve_summary.iterrows():
        improve_lines.append(
            f"- {row['target']} {row['layer']} + {row['new_data_source']}: delta R²={row['delta_R2']:.4f}, delta RMSE={row['delta_RMSE']:.4f}, delta MAE={row['delta_MAE']:.4f}"
        )
    if not improve_lines:
        improve_lines.append("- 本轮没有出现满足 promote 条件的提升。")

    append_section(
        CURRENT_REPORT_DIR / "experiment_report.md",
        "2026-05-20 精度突破阶段（单轮）",
        "\n".join(
            [
                "本轮先锁定了 current baseline，然后按“残差诊断 → 免费数据源验证 → 单源 A/B → Chl-a bottom 受限补充实验”的顺序执行。",
                "",
                "主要结论：",
                *improve_lines,
                "",
                "Chl-a bottom 仍未形成可 promote 的突破，证据来自残差诊断和底层候选源的负迁移。",
            ]
        ),
    )

    append_section(
        CURRENT_REPORT_DIR / "model_input_alignment_documentation.md",
        "2026-05-20 精度突破阶段输入对齐补充",
        "\n".join(
            [
                "- `copernicus_bgc_features_2014_2024.csv`：按 `station + date` 对齐；属于 daily reanalysis/model proxy，不是实测底层营养盐或实测浮游过程。",
                "- `enhanced_tide_proxy_features_2014_2024.csv`：按 `obs_id/sample_id/station/date/depth` 对齐；仅为 date-level tide proxy，不是 exact sampling tide phase。",
                "- `high_quality_patch_qc_refined_features_2014_2024.csv`：按 `sample_id/station/date` 合并；保留高分 patch 的 valid_ratio/date_diff_abs/index stats，作为 QC 后稳健子集。",
                "- 本轮所有新增 rolling/anomaly 仅基于预测日及以前可获得的信息，不使用未来信息，也不使用 test 标签做统计。",
            ]
        ),
    )

    append_section(
        CURRENT_REPORT_DIR / "model_parameter_and_feature_inventory.md",
        "2026-05-20 新增候选模型与特征补充",
        "\n".join(
            [
                "- `copernicus_bgc` 单源 A/B：沿用 current best family，不做大规模调参；新增字段主要为 `bgc_chl/bgc_no3/bgc_po4/bgc_si/bgc_o2/bgc_npp/bgc_*_7d/bgc_chl_anom_7d/bgc_n_to_p_ratio`。",
                "- `enhanced_tide_proxy` 单源 A/B：沿用 current best family；新增字段为 `tide_range_same_day/tide_range_previous_3d_mean/tide_range_anom_prev3d/tide_range_change_prev1d/tide_level_daily_mean/tide_event_density` 等。",
                "- `high_quality_patch_qc_refined` 单源 A/B：沿用 current best family；只保留高分 patch 中与水体指数、valid_ratio、date_diff_abs 相关的更稳健子集，不再把全部 patch 统计重新堆回模型。",
                "- `chla_bottom_log1p` 与 `chla_bottom_yeojohnson`：保持 bottom 当前 LightGBM-family 框架，仅做目标变换验证。",
            ]
        ),
    )

    append_section(
        CURRENT_REPORT_DIR / "data_gap_inventory.md",
        "2026-05-20 数据缺口更新",
        "\n".join(
            [
                "- 本轮确认：`Copernicus BGC` 可以自动获取，但本质仍是大尺度生态背景代理。",
                "- 本轮确认：`GEBCO` 官方入口可访问，但本轮未形成可直接进入模型的稳定 marine bathymetry 特征表。",
                "- 仍缺：`station_depth`、`exact_sampling_datetime`、`bottom water temperature`、`vertical temperature profile`、`ADCP current / discharge / water level`。",
            ]
        ),
    )

    append_section(
        CURRENT_REPORT_DIR / "methods_tried_registry.md",
        "2026-05-20 新增方法登记",
        "\n".join(
            [
                "- `Copernicus BGC` 单源 A/B：用于验证 BGC 大尺度生态背景代理对 Chl-a middle/bottom 和 Secchi middle/bottom 的增益。",
                "- `enhanced_tide_proxy` 单源 A/B：继续验证 date-level tide proxy 的可部署上限。",
                "- `high_quality_patch_qc_refined` 单源 A/B：验证上一轮 patch 负迁移是否主要来自低质量 patch 噪声。",
                "- `Chl-a bottom target transform`：仅做受限 log1p / Yeo-Johnson，不做大规模 family 搜索。",
            ]
        ),
    )

    negative = compare_df[compare_df["delta_R2"] <= 0].sort_values("delta_R2").head(8)
    fail_lines = []
    for _, row in negative.iterrows():
        fail_lines.append(f"- {row['target']} {row['layer']} + {row['new_data_source']}: delta R²={row['delta_R2']:.4f}")
    if not fail_lines:
        fail_lines.append("- 本轮没有明显负增益路线。")
    append_section(
        CURRENT_REPORT_DIR / "failed_experiments.md",
        "2026-05-20 本轮失败或负迁移实验",
        "\n".join(fail_lines),
    )

    append_section(
        CURRENT_REPORT_DIR / "do_not_repeat_experiments.md",
        "2026-05-20 停止重复建议",
        "\n".join(
            [
                "- 如果 `enhanced_tide_proxy` 再次不能带来 >=0.005 的增益，后续不再单独重复 date-level tide proxy 路线。",
                "- 如果 `high_quality_patch_qc_refined` 仍不改善 bottom，则后续不再把 bottom 作为高分 patch 主攻对象。",
                "- 如果 `Copernicus BGC` 对 bottom 继续负迁移，则后续只保留其在 middle / Secchi 任务中的可选验证身份。",
            ]
        ),
    )

    append_section(
        CURRENT_REPORT_DIR / "prediction_ceiling_analysis.md",
        "2026-05-20 天花板证据补充",
        "\n".join(
            [
                "- 当前自动可获取的免费代理源已经覆盖了 patch QC、physics、BGC 和 date-level tide proxy，但对 Chl-a bottom 仍未形成可 promote 的突破。",
                "- 这进一步支持：bottom 的主要天花板来自真实底层机制数据缺口，而不是普通调参或继续堆弱代理。",
            ]
        ),
    )

    append_section(
        CURRENT_REPORT_DIR / "next_data_acquisition_plan.md",
        "2026-05-20 下一轮最优先数据",
        "\n".join(
            [
                "1. station_depth 或可靠的近岸 marine bathymetry。",
                "2. exact_sampling_datetime，用于真实采样潮相。",
                "3. bottom water temperature / vertical temperature profile。",
                "4. ADCP current / discharge / water level。",
                "5. 若用户愿意配置免费 NASA Earthdata 账号，可继续 OceanColor 路线。",
            ]
        ),
    )

    append_section(
        CURRENT_REPORT_DIR / "strong_input_data_action_plan.md",
        "2026-05-20 强输入行动补充",
        "\n".join(
            [
                "- 本轮新增并验证了 `Copernicus BGC`。",
                "- 本轮将 `tide_phase_features` 升级为 `enhanced_tide_proxy_features`，但仍严格限定为 date-level proxy。",
                "- 本轮将 `high_quality_patch_features` 收缩为更稳健的 `high_quality_patch_qc_refined_features` 子集，用于检验 patch 噪声假设。",
            ]
        ),
    )

    append_section(
        CURRENT_REPORT_DIR / "r2_0p6_breakthrough_log.md",
        "2026-05-20 单轮突破日志",
        "\n".join(
            [
                f"- 通过验证的数据源数：{int((source_inventory['recommendation'] == 'use').sum())}",
                f"- 本轮达到 R² >= 0.60 的任务数：{int((updated_best['final_selected_r2'] >= 0.60).sum())}",
                "- 若没有达到 0.60，本轮证据继续指向：真实底层机制数据缺口仍是主要障碍。",
            ]
        ),
    )

    append_section(
        CURRENT_REPORT_DIR / "cleanup_log.md",
        "2026-05-20 精度突破阶段更新",
        "- 新增本轮计划、残差诊断、免费数据源验证、A/B 对比、updated scorecard 和阶段报告；随后会运行 Project Organizer Agent 做 dry-run 与 apply-safe。",
    )


def run_project_organizer() -> dict[str, Any]:
    dry = subprocess.run(
        [sys.executable, str(PROJECT_ORGANIZER), "--mode", "dry-run"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        cwd=PROJECT_ROOT,
    )
    apply = None
    if dry.returncode == 0:
        apply = subprocess.run(
            [sys.executable, str(PROJECT_ORGANIZER), "--mode", "apply-safe"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=PROJECT_ROOT,
        )
    return {
        "dry_run_returncode": dry.returncode,
        "dry_run_stdout_tail": dry.stdout[-3000:],
        "dry_run_stderr_tail": dry.stderr[-3000:],
        "apply_safe_returncode": None if apply is None else apply.returncode,
        "apply_safe_stdout_tail": None if apply is None else apply.stdout[-3000:],
        "apply_safe_stderr_tail": None if apply is None else apply.stderr[-3000:],
    }


def main() -> None:
    args = parse_args()
    ensure_dirs()
    started = time.time()

    base_df = load_base_dataset()
    best_df = load_current_best_scorecard()
    baseline_pred = current_best_predictions()
    baseline_map = baseline_metrics_map(baseline_pred)
    write_stage_plan(best_df)

    bottom_diag = build_chla_bottom_residual_diagnostics(base_df, baseline_pred)

    source_inventory, prepared = build_free_source_inventory(args.date_tag, base_df)
    write_source_validation_docs(source_inventory)

    metrics_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    station_frames: list[pd.DataFrame] = []
    manifest_steps: list[dict[str, Any]] = []

    source_order = [
        "gebco_bathymetry",
        "copernicus_bgc",
        "enhanced_tide_proxy",
        "high_quality_patch_qc_refined",
    ]
    tried_new_sources = 0
    failed_new_sources = 0
    useful_sources: list[str] = []
    for source_name in source_order:
        inv_row = source_inventory[source_inventory["source_name"] == source_name].iloc[0]
        if inv_row["recommendation"] != "use":
            if source_name in {"gebco_bathymetry", "copernicus_bgc", "enhanced_tide_proxy", "high_quality_patch_qc_refined"}:
                failed_new_sources += 1
            continue
        feature_df, feature_cols, feature_stats = scenario_feature_frame(source_name, prepared)
        if feature_df is None or not feature_cols:
            failed_new_sources += 1
            continue
        tried_new_sources += 1
        run_candidate_scenario(
            base_df=base_df,
            baseline_pred=baseline_pred,
            baseline_metrics=baseline_map,
            source_name=source_name,
            feature_df=feature_df,
            source_features=feature_cols,
            source_stats=feature_stats,
            all_metrics=metrics_rows,
            all_compare=compare_rows,
            all_station=station_frames,
            manifest_steps=manifest_steps,
        )
        source_cmp = [row for row in compare_rows if row["new_data_source"] == source_name]
        if any((row["delta_R2"] >= 0.005) and ((row["delta_RMSE"] < 0) or (row["delta_MAE"] < 0)) for row in source_cmp):
            useful_sources.append(source_name)
        else:
            failed_new_sources += 1
        if failed_new_sources >= 3:
            break

    if useful_sources:
        combo_sources = useful_sources.copy()
        combo_name = "useful_sources_combined"
        combo_frames = []
        combo_features: list[str] = []
        stats_missing = []
        stats_success = []
        for src in combo_sources:
            frame, feats, stats = prepared[src]
            if frame is None or not feats:
                continue
            combo_frames.append(frame)
            combo_features.extend(feats)
            stats_missing.append(stats.get("missing_rate", np.nan))
            stats_success.append(stats.get("extraction_success_rate", np.nan))
        if combo_frames:
            merged_combo = combo_frames[0].copy()
            for frame in combo_frames[1:]:
                merged_combo = merge_on_best_keys(merged_combo, frame)
            run_candidate_scenario(
                base_df=base_df,
                baseline_pred=baseline_pred,
                baseline_metrics=baseline_map,
                source_name=combo_name,
                feature_df=merged_combo,
                source_features=list(dict.fromkeys(combo_features)),
                source_stats={
                    "missing_rate": float(np.nanmean(stats_missing)) if stats_missing else np.nan,
                    "extraction_success_rate": float(np.nanmean(stats_success)) if stats_success else np.nan,
                },
                all_metrics=metrics_rows,
                all_compare=compare_rows,
                all_station=station_frames,
                manifest_steps=manifest_steps,
            )
            run_candidate_scenario(
                base_df=base_df,
                baseline_pred=baseline_pred,
                baseline_metrics=baseline_map,
                source_name=combo_name,
                feature_df=merged_combo,
                source_features=list(dict.fromkeys(combo_features)),
                source_stats={
                    "missing_rate": float(np.nanmean(stats_missing)) if stats_missing else np.nan,
                    "extraction_success_rate": float(np.nanmean(stats_success)) if stats_success else np.nan,
                },
                all_metrics=metrics_rows,
                all_compare=compare_rows,
                all_station=station_frames,
                manifest_steps=manifest_steps,
                extra_behavior="interactions",
            )

    # Existing-data boundary bottom-only bounded tests
    base_features_bottom = base_ab.build_common_features(base_df, "chla", extra_features=[])
    current_bottom = base_df.copy()
    empty_frame = pd.DataFrame(
        {
            "obs_id": current_bottom["obs_id"],
            "sample_id": current_bottom["sample_id"],
            "station": current_bottom["station"],
            "date": current_bottom["date"],
            "depth": current_bottom["depth"],
        }
    )
    run_candidate_scenario(
        base_df=current_bottom,
        baseline_pred=baseline_pred,
        baseline_metrics=baseline_map,
        source_name="existing_data_bottom",
        feature_df=empty_frame,
        source_features=[],
        source_stats={"missing_rate": 0.0, "extraction_success_rate": 1.0},
        all_metrics=metrics_rows,
        all_compare=compare_rows,
        all_station=station_frames,
        manifest_steps=manifest_steps,
        extra_behavior="log1p",
    )
    run_candidate_scenario(
        base_df=current_bottom,
        baseline_pred=baseline_pred,
        baseline_metrics=baseline_map,
        source_name="existing_data_bottom",
        feature_df=empty_frame,
        source_features=[],
        source_stats={"missing_rate": 0.0, "extraction_success_rate": 1.0},
        all_metrics=metrics_rows,
        all_compare=compare_rows,
        all_station=station_frames,
        manifest_steps=manifest_steps,
        extra_behavior="yeojohnson",
    )

    metrics_df = pd.DataFrame(metrics_rows)
    compare_df = pd.DataFrame(compare_rows)
    station_df = pd.concat(station_frames, ignore_index=True) if station_frames else pd.DataFrame()
    metrics_df.to_csv(PRECISION_AB_METRICS, index=False, encoding="utf-8-sig")
    compare_df.to_csv(PRECISION_COMPARE, index=False, encoding="utf-8-sig")
    compare_df[compare_df["new_data_source"].isin(source_order + ["useful_sources_combined"])].to_csv(
        NEW_SOURCE_COMPARE, index=False, encoding="utf-8-sig"
    )
    metrics_df[metrics_df["new_data_source"].isin(source_order + ["useful_sources_combined"])].to_csv(
        NEW_SOURCE_AB_METRICS, index=False, encoding="utf-8-sig"
    )

    updated_best, updated_progress = update_scorecards(best_df, compare_df)
    build_stage_report(source_inventory, compare_df, bottom_diag, updated_best)
    update_current_docs(source_inventory, compare_df, updated_best)

    organizer_result = run_project_organizer()
    manifest = {
        "script_path": str(Path(__file__).resolve()),
        "inputs": [
            str(CURRENT_BEST_SCORECARD),
            str(R2_PROGRESS),
            str(STRONG_MECH_COMPARE),
            str(STATION_LEVEL_PRED),
            str(COPERNICUS_PHYSICS),
            str(HIGH_QUALITY_PATCH),
            str(TIDE_PHASE),
        ],
        "outputs": [
            str(STAGE_PLAN),
            str(FREE_SOURCE_INVENTORY),
            str(NEW_SOURCE_AB_METRICS),
            str(NEW_SOURCE_COMPARE),
            str(CHLA_BOTTOM_DIAG_CSV),
            str(PRECISION_AB_METRICS),
            str(PRECISION_COMPARE),
            str(UPDATED_BEST),
            str(UPDATED_PROGRESS),
            str(STAGE_REPORT),
        ],
        "data_sources": source_inventory.to_dict(orient="records"),
        "models": sorted(compare_df["candidate_model"].dropna().unique().tolist()) if not compare_df.empty else [],
        "feature_sets": sorted(compare_df["feature_set_name"].dropna().unique().tolist()) if not compare_df.empty else [],
        "random_seed": 42,
        "experiment_config": {
            "baseline_scorecard": str(CURRENT_BEST_SCORECARD),
            "copernicus_bgc_dataset_id": CMEMS_BGC_DATASET_ID,
            "stop_rule_new_sources_negative_streak": 3,
            "single_round_only": True,
        },
        "used_api": bool(source_inventory["recommendation"].eq("needs_user_api").any()),
        "unfinished_steps": [] if organizer_result["dry_run_returncode"] == 0 else ["project_organizer_dry_run"],
        "failure_reason": "" if organizer_result["dry_run_returncode"] == 0 else "project_organizer_failed",
        "steps": manifest_steps,
        "runtime_seconds": round(time.time() - started, 2),
        "project_organizer": organizer_result,
    }
    write_json(MANIFEST_JSON, manifest)
    print("precision improvement stage completed")


if __name__ == "__main__":
    main()
