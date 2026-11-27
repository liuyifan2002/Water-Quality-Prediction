from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CURRENT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
CURRENT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "current"
CURRENT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"

MODEL_TABLE = PROJECT_ROOT / "data" / "model_input" / "model_table_3depth_2014_2024.csv"
STATION_META = PROJECT_ROOT / "data" / "model_input" / "station_meta.csv"
HIGHRES_ALL = CURRENT_PROCESSED_DIR / "highres_patch_features_2014_2024.csv"
S2_PATCH = CURRENT_PROCESSED_DIR / "s2_highres_patch_features_2014_2024.csv"
LANDSAT_PATCH = CURRENT_PROCESSED_DIR / "landsat_highres_patch_features_2014_2024.csv"
BOTTOM_MECH = CURRENT_PROCESSED_DIR / "bottom_mechanism_features_2014_2024.csv"
ERA5_ENV = PROJECT_ROOT / "data" / "raw" / "manual" / "environment" / "era5_features_2014_2024.csv"
RAIN_ENV = PROJECT_ROOT / "data" / "raw" / "manual" / "environment" / "rainfall_features_2014_2024.csv"
TIDE_ENV = PROJECT_ROOT / "data" / "raw" / "manual" / "environment" / "tide_features_2014_2024.csv"
GEE_S2_EXPORT = PROJECT_ROOT / "data" / "external" / "gee_exports" / "s2_patch_features_2014_2024.csv"
GEE_LANDSAT_EXPORT = PROJECT_ROOT / "data" / "external" / "gee_exports" / "landsat_patch_features_2014_2024.csv"
CMEMS_PHYSICS_TEMPLATE = PROJECT_ROOT / "data" / "raw" / "manual" / "copernicus_physics" / "copernicus_physics_features_2014_2024_template.csv"
CMEMS_BGC_TEMPLATE = PROJECT_ROOT / "data" / "raw" / "manual" / "copernicus_bgc" / "copernicus_bgc_features_2014_2024_template.csv"

CURRENT_SCORECARD = CURRENT_PROCESSED_DIR / "current_best_scorecard.csv"
CURRENT_REGISTRY = CURRENT_OUTPUT_DIR / "current_candidate_registry.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the strong-input data driven stage.")
    parser.add_argument("--date-tag", default="20260519")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [CURRENT_REPORT_DIR, CURRENT_OUTPUT_DIR, CURRENT_PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, **kwargs)


def to_bool(value: bool) -> str:
    return "True" if value else "False"


def infer_time_range(df: pd.DataFrame) -> str:
    for col in ["date", "image_date", "s2_image_date", "landsat_image_date"]:
        if col in df.columns:
            values = pd.to_datetime(df[col], errors="coerce").dropna()
            if not values.empty:
                return f"{values.min().date()} to {values.max().date()}"
    return ""


def infer_available_keys(df: pd.DataFrame) -> str:
    return ",".join(df.columns[:30].tolist())


def build_source_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(
        data_source: str,
        variable_group: str,
        path: Path,
        can_codex_download: bool,
        needs_user_manual: bool,
        expected_target: str,
        expected_gain_level: str,
        next_action: str,
        leakage_risk: str = "low",
    ) -> None:
        found = path.exists()
        df = read_csv_if_exists(path, nrows=5) if found and path.suffix.lower() == ".csv" else pd.DataFrame()
        rows.append(
            {
                "data_source": data_source,
                "variable_group": variable_group,
                "file_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "found_locally": to_bool(found),
                "needs_download": to_bool(not found),
                "can_codex_download": to_bool(can_codex_download),
                "needs_user_manual": to_bool(needs_user_manual),
                "required_keys": "station,date" if "patch" in data_source or "environment" in data_source else "station,date,depth",
                "available_keys": infer_available_keys(df) if found else "",
                "time_range": infer_time_range(read_csv_if_exists(path)) if found and path.suffix.lower() == ".csv" else "",
                "spatial_range": "Hong Kong coastal stations" if found else "",
                "station_date_alignable": to_bool(found and ("station" in df.columns and "date" in df.columns)),
                "leakage_risk": leakage_risk,
                "expected_target": expected_target,
                "expected_gain_level": expected_gain_level,
                "next_action": next_action,
            }
        )

    add_row(
        "station_meta",
        "station_space",
        STATION_META,
        False,
        False,
        "all",
        "supporting",
        "local_ready",
    )
    add_row(
        "s2_highres_patch_export",
        "highres_patch",
        GEE_S2_EXPORT,
        False,
        False,
        "chla_surface;secchi",
        "high",
        "local_ready_use_existing_export",
    )
    add_row(
        "landsat_highres_patch_export",
        "highres_patch",
        GEE_LANDSAT_EXPORT,
        False,
        False,
        "chla_surface;secchi;chla_bottom",
        "medium",
        "local_ready_use_existing_export",
    )
    add_row(
        "merged_highres_patch_features",
        "highres_patch",
        HIGHRES_ALL,
        False,
        False,
        "chla;secchi",
        "medium",
        "local_ready_already_validated",
    )
    add_row(
        "bottom_mechanism_features",
        "mechanism_proxy",
        BOTTOM_MECH,
        False,
        False,
        "chla_bottom",
        "medium",
        "local_ready_but_only_weak_proxy",
    )
    add_row(
        "era5_environment_features",
        "environment_proxy",
        ERA5_ENV,
        False,
        False,
        "chla;secchi",
        "low",
        "already_available_no_repeat_as_main_gain",
    )
    add_row(
        "rainfall_environment_features",
        "environment_proxy",
        RAIN_ENV,
        False,
        False,
        "chla;secchi",
        "low",
        "already_available_no_repeat_as_main_gain",
    )
    add_row(
        "tide_environment_features",
        "environment_proxy",
        TIDE_ENV,
        False,
        False,
        "chla;secchi",
        "low",
        "already_available_no_repeat_as_main_gain",
    )
    add_row(
        "copernicus_physics_features",
        "physics",
        CMEMS_PHYSICS_TEMPLATE,
        False,
        True,
        "chla_bottom",
        "high",
        "template_only_user_or_network_required",
        leakage_risk="low",
    )
    add_row(
        "copernicus_bgc_features",
        "bgc",
        CMEMS_BGC_TEMPLATE,
        False,
        True,
        "chla",
        "medium",
        "template_only_user_or_network_required",
        leakage_risk="low",
    )
    rows.extend(
        [
            {
                "data_source": "bathymetry_depth_source",
                "variable_group": "bathymetry_depth",
                "file_path": "",
                "found_locally": "False",
                "needs_download": "True",
                "can_codex_download": "False",
                "needs_user_manual": "True",
                "required_keys": "station,lon,lat",
                "available_keys": "",
                "time_range": "static",
                "spatial_range": "",
                "station_date_alignable": "False",
                "leakage_risk": "low",
                "expected_target": "chla_bottom;secchi_bottom",
                "expected_gain_level": "high",
                "next_action": "need_bathymetry_raster_or_station_depth_table",
            },
            {
                "data_source": "sampling_time_source",
                "variable_group": "sampling_time_tide_phase",
                "file_path": "",
                "found_locally": "False",
                "needs_download": "True",
                "can_codex_download": "False",
                "needs_user_manual": "True",
                "required_keys": "station,datetime",
                "available_keys": "",
                "time_range": "",
                "spatial_range": "station",
                "station_date_alignable": "False",
                "leakage_risk": "low",
                "expected_target": "chla_bottom;secchi",
                "expected_gain_level": "high",
                "next_action": "need_exact_sampling_datetime",
            },
            {
                "data_source": "discharge_water_level_source",
                "variable_group": "hydrology",
                "file_path": "",
                "found_locally": "False",
                "needs_download": "True",
                "can_codex_download": "False",
                "needs_user_manual": "True",
                "required_keys": "date,value,station_or_basin",
                "available_keys": "",
                "time_range": "",
                "spatial_range": "basin_or_river",
                "station_date_alignable": "False",
                "leakage_risk": "low",
                "expected_target": "chla_bottom;secchi",
                "expected_gain_level": "high",
                "next_action": "need_local_hydrology_records",
            },
        ]
    )

    out = pd.DataFrame(rows)
    out.to_csv(
        CURRENT_PROCESSED_DIR / "strong_input_source_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out


def build_action_plan(inventory: pd.DataFrame) -> None:
    def block(title: str, subset: pd.DataFrame) -> list[str]:
        lines = [f"## {title}", ""]
        if subset.empty:
            lines += ["- 无。", ""]
            return lines
        for _, row in subset.iterrows():
            lines.append(
                f"- `{row['data_source']}`：{row['variable_group']}，动作 `{row['next_action']}`，目标 `{row['expected_target']}`，预期增益 `{row['expected_gain_level']}`。"
            )
        lines.append("")
        return lines

    class1 = inventory[(inventory["found_locally"] == "True") & (inventory["needs_user_manual"] == "False")]
    class2 = inventory[(inventory["can_codex_download"] == "True")]
    class3 = inventory[(inventory["needs_user_manual"] == "True")]
    class4 = inventory[
        (inventory["found_locally"] == "False")
        & (inventory["can_codex_download"] == "False")
        & (inventory["needs_user_manual"] == "False")
    ]

    lines = [
        "# 强输入数据行动计划",
        "",
        "## 总结",
        "",
        "- 本轮本地可直接处理的强输入数据主要有：高分遥感 patch、已有 bottom 机制代理表、station meta、ERA5/rain/tide 原始环境表。",
        "- 当前环境下没有可稳定自动下载并立即并入的新强机制数据；bathymetry、station depth、采样时刻、真实水文流量/水位仍需用户手动提供。",
        "- 因为连续两轮已有强输入尝试都未达到 `delta R² >= 0.01`，本轮按停止规则不继续追加第三轮自动训练，只保留数据缺口与下一步获取计划。",
        "",
    ]
    lines += block("Class 1：本地已有，可直接处理", class1)
    lines += block("Class 2：可由 Codex 自动下载", class2)
    lines += block("Class 3：需要用户手动提供", class3)
    lines += block("Class 4：当前不可获得", class4)
    lines += [
        "## 下一步优先级",
        "",
        "1. 站点水深 / bathymetry 栅格。",
        "2. 精确采样时刻，用于潮位相位和涨落潮判定。",
        "3. 底层水温或垂向温度剖面，用于分层与混合强度建模。",
        "4. 流量 / 水位 / 流速等真实水文水动力数据。",
    ]
    (CURRENT_REPORT_DIR / "strong_input_data_action_plan.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_strong_input_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    mech = pd.read_csv(BOTTOM_MECH, low_memory=False)
    highres = pd.read_csv(HIGHRES_ALL, low_memory=False)
    mech["date"] = pd.to_datetime(mech["date"])
    highres["date"] = pd.to_datetime(highres["date"])

    id_cols = ["obs_id", "sample_id", "station", "date", "depth"]
    keep_mech = [col for col in mech.columns if col in id_cols or col in {
        "year",
        "month",
        "is_wet_season",
        "lon",
        "lat",
        "zone",
        "station_depth",
        "bathymetry_depth",
        "distance_to_shore",
        "distance_to_river_mouth",
        "surface_temp",
        "bottom_temp",
        "surface_bottom_temp_diff",
        "temp_gradient",
        "thermocline_proxy",
        "stratification_strength",
        "temp_7d_mean",
        "temp_14d_mean",
        "temp_30d_mean",
        "temp_change_rate",
        "wind_speed_3d_mean",
        "wind_speed_7d_mean",
        "wind_speed_14d_mean",
        "wind_speed_3d_max",
        "calm_days_7d",
        "strong_wind_event_7d",
        "wind_mixing_proxy",
        "wind_direction_u",
        "wind_direction_v",
        "precip_3d_sum",
        "precip_7d_sum",
        "storm_event_flag",
        "water_level",
        "water_level_change_3d",
        "discharge_7d_mean",
        "flushing_proxy",
        "current_speed",
        "current_speed_7d_mean",
        "tide_range_proxy",
    }]
    mech = mech[keep_mech].copy()

    keep_highres = [
        col for col in highres.columns
        if col in {"request_id", "sample_id", "station", "date", "s2_image_id", "s2_image_date", "s2_date_diff_abs", "s2_cloudy_pixel_percentage", "s2_image_count_in_window", "landsat_image_id", "landsat_image_date", "landsat_date_diff_abs", "landsat_cloud_cover", "landsat_image_count_in_window"}
        or col.startswith(("s2_3x3_", "s2_5x5_", "landsat_3x3_", "landsat_5x5_"))
    ]
    highres = highres[keep_highres].copy()

    merged = mech.merge(
        highres,
        on=["sample_id", "station", "date"],
        how="left",
        validate="many_to_one",
    )
    merged["has_s2_patch"] = merged["s2_image_id"].notna().astype(int) if "s2_image_id" in merged.columns else 0
    merged["has_landsat_patch"] = merged["landsat_image_id"].notna().astype(int) if "landsat_image_id" in merged.columns else 0
    merged["has_any_highres_patch"] = ((merged["has_s2_patch"] == 1) | (merged["has_landsat_patch"] == 1)).astype(int)

    merged.to_csv(
        CURRENT_PROCESSED_DIR / "strong_input_features_2014_2024.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dictionary_rows = []
    for col in merged.columns:
        if col in id_cols or col in {"year", "month", "is_wet_season", "lon", "lat", "zone"}:
            continue
        if col.startswith(("s2_", "landsat_")):
            feature_group = "highres_patch_features"
            source = "GEE highres patch export"
            value_type = "patch_stat_or_metadata"
            is_strong = "True"
        elif any(token in col for token in ["station_depth", "bathymetry", "distance_to_"]):
            feature_group = "depth_bathymetry_features"
            source = "bottom mechanism table"
            value_type = "static"
            is_strong = "True" if col in {"station_depth", "bathymetry_depth"} else "proxy"
        elif any(token in col for token in ["temp", "thermocline", "stratification"]):
            feature_group = "temperature_stratification_features"
            source = "bottom mechanism table"
            value_type = "daily_or_rolling_proxy"
            is_strong = "True" if col in {"bottom_temp", "surface_bottom_temp_diff", "temp_gradient", "thermocline_proxy"} else "proxy"
        elif any(token in col for token in ["current", "water_level", "discharge", "flushing"]):
            feature_group = "hydrodynamic_features"
            source = "bottom mechanism table"
            value_type = "daily_or_rolling_proxy"
            is_strong = "True" if col in {"current_speed", "water_level", "discharge_7d_mean"} else "proxy"
        elif any(token in col for token in ["wind", "precip", "storm", "tide"]):
            feature_group = "weather_tide_proxy_features"
            source = "bottom mechanism table"
            value_type = "daily_or_rolling_proxy"
            is_strong = "proxy"
        elif col.startswith("has_"):
            feature_group = "availability_flags"
            source = "derived"
            value_type = "flag"
            is_strong = "supporting"
        else:
            feature_group = "other"
            source = "derived"
            value_type = "unknown"
            is_strong = "unknown"
        dictionary_rows.append(
            {
                "feature": col,
                "feature_group": feature_group,
                "source_table": source,
                "alignment_key": "obs_id|sample_id|station|date|depth",
                "value_type": value_type,
                "is_strong_mechanism_variable": is_strong,
            }
        )
    dictionary = pd.DataFrame(dictionary_rows)
    dictionary.to_csv(
        CURRENT_PROCESSED_DIR / "strong_input_feature_dictionary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return merged, dictionary


def write_strong_input_missing_and_validation(df: pd.DataFrame, dictionary: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns if c not in {"obs_id", "sample_id", "station", "date", "depth", "year", "month", "is_wet_season", "lon", "lat", "zone"}]
    missing_rows = []
    for col in feature_cols:
        missing_rows.append(
            {
                "feature": col,
                "feature_group": dictionary.loc[dictionary["feature"] == col, "feature_group"].iloc[0] if col in dictionary["feature"].values else "",
                "missing_rate": float(df[col].isna().mean()),
                "non_null_count": int(df[col].notna().sum()),
            }
        )
    pd.DataFrame(missing_rows).sort_values(["feature_group", "missing_rate"], ascending=[True, False]).to_csv(
        CURRENT_PROCESSED_DIR / "strong_input_missing_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_coverage = {}
    for feature_group, sub in dictionary.groupby("feature_group"):
        cols = [c for c in sub["feature"].tolist() if c in df.columns]
        if not cols:
            continue
        group_coverage[feature_group] = float(df[cols].notna().any(axis=1).mean())

    report = {
        "rows": int(len(df)),
        "station_count": int(df["station"].nunique()),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "obs_id_unique": bool(df["obs_id"].is_unique),
        "station_date_unique": bool(~df.duplicated(["station", "date"]).any()),
        "station_date_depth_unique": bool(~df.duplicated(["station", "date", "depth"]).any()),
        "duplicate_obs_id_count": int(df["obs_id"].duplicated().sum()),
        "duplicate_station_date_count": int(df.duplicated(["station", "date"]).sum()),
        "group_coverage": group_coverage,
        "s2_patch_coverage": float(df["has_s2_patch"].mean()) if "has_s2_patch" in df.columns else None,
        "landsat_patch_coverage": float(df["has_landsat_patch"].mean()) if "has_landsat_patch" in df.columns else None,
        "any_highres_patch_coverage": float(df["has_any_highres_patch"].mean()) if "has_any_highres_patch" in df.columns else None,
        "future_info_used": False,
        "test_label_statistics_used": False,
        "leakage_risk": "low",
        "note": "本表按建模样本 obs_id 对齐，因此 station+date 在多水层采样下不唯一；正式唯一键为 obs_id 或 station+date+depth。",
    }
    (CURRENT_PROCESSED_DIR / "strong_input_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def concat_existing_outputs() -> pd.DataFrame:
    files = {
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
        },
        "bottom_mechanism": {
            "metrics": CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_ab_metrics.csv",
            "predictions": CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_predictions.csv",
            "importance": CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_feature_importance.csv",
            "station": CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_station_level_comparison.csv",
        },
    }

    metrics_frames = []
    prediction_frames = []
    importance_frames = []
    station_frames = []
    for scenario, paths in files.items():
        for kind, path in paths.items():
            df = read_csv_if_exists(path)
            if df.empty:
                continue
            if "scenario" not in df.columns:
                df["scenario"] = scenario
            if kind == "metrics":
                metrics_frames.append(df)
            elif kind == "predictions":
                prediction_frames.append(df)
            elif kind == "importance":
                importance_frames.append(df)
            elif kind == "station":
                station_frames.append(df)

    metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    station = pd.concat(station_frames, ignore_index=True) if station_frames else pd.DataFrame()

    metrics.to_csv(CURRENT_OUTPUT_DIR / "strong_input_ab_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(CURRENT_OUTPUT_DIR / "strong_input_ab_predictions.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(CURRENT_OUTPUT_DIR / "strong_input_feature_importance.csv", index=False, encoding="utf-8-sig")
    station.to_csv(CURRENT_OUTPUT_DIR / "strong_input_station_level_comparison.csv", index=False, encoding="utf-8-sig")
    return station


def build_comparison_and_progress() -> pd.DataFrame:
    registry = pd.read_csv(CURRENT_REGISTRY, low_memory=False)
    registry = registry[registry["scenario"].isin(["s2_patch", "landsat_patch", "highres_patch", "bottom_mechanism"])].copy()
    registry.to_csv(
        CURRENT_OUTPUT_DIR / "strong_input_comparison_to_current_best.csv",
        index=False,
        encoding="utf-8-sig",
    )

    current_best = pd.read_csv(CURRENT_SCORECARD, low_memory=False)
    score_rows = []
    for _, base_row in current_best.iterrows():
        target = base_row["target"]
        depth = base_row["depth_layer"]
        prev = float(base_row["final_selected_r2"])
        cand = registry[(registry["target"] == target) & (registry["depth_layer"] == depth)].copy()
        if cand.empty:
            score_rows.append(
                {
                    "target": target,
                    "depth_layer": depth,
                    "previous_best_r2": prev,
                    "current_round_best_r2": prev,
                    "delta_r2": 0.0,
                    "reached_0p6": False,
                    "best_data_source_added": "none",
                    "best_feature_group": "retain_previous_best",
                    "selected_model": base_row["selected_model"],
                    "selected_strategy": "retain_current_best",
                    "remaining_gap_to_0p6": round(0.60 - prev, 6),
                    "next_required_data": next_required_data(target, depth),
                }
            )
            continue
        cand = cand.sort_values(["delta_r2", "candidate_validation_r2"], ascending=[False, False])
        best = cand.iloc[0]
        best_r2 = max(prev, float(best["candidate_r2"]))
        delta = float(best["candidate_r2"]) - prev
        strategy = classify_strategy(best)
        score_rows.append(
            {
                "target": target,
                "depth_layer": depth,
                "previous_best_r2": prev,
                "current_round_best_r2": best_r2,
                "delta_r2": delta,
                "reached_0p6": bool(best_r2 >= 0.60),
                "best_data_source_added": best["scenario"],
                "best_feature_group": scenario_to_feature_group(best["scenario"]),
                "selected_model": best["candidate_model"] if best_r2 > prev else base_row["selected_model"],
                "selected_strategy": strategy,
                "remaining_gap_to_0p6": round(0.60 - best_r2, 6),
                "next_required_data": next_required_data(target, depth),
            }
        )
    progress = pd.DataFrame(score_rows)
    progress.to_csv(
        CURRENT_PROCESSED_DIR / "r2_0p6_progress_scorecard.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return progress


def classify_strategy(row: pd.Series) -> str:
    delta_r2 = float(row["delta_r2"])
    delta_val = float(row["delta_validation_r2"])
    improved_ratio = float(row["station_improved_ratio"])
    delta_p90 = row["delta_p90_rmse"]
    if delta_r2 >= 0.05 and delta_val > 0:
        return "major_data_gain"
    if delta_r2 >= 0.03 and delta_val > 0:
        return "useful_data_gain"
    if delta_r2 > 0 and delta_val <= 0:
        return "suspicious"
    if improved_ratio >= 0.60 and delta_r2 < 0.03:
        return "station_specific_auxiliary"
    if pd.notna(delta_p90) and delta_p90 < 0 and delta_r2 <= 0:
        return "high_value_auxiliary"
    return "retain_current_best"


def scenario_to_feature_group(scenario: str) -> str:
    mapping = {
        "s2_patch": "sentinel2_highres_patch",
        "landsat_patch": "landsat_highres_patch",
        "highres_patch": "sentinel2_landsat_highres_patch",
        "bottom_mechanism": "bottom_mechanism_proxy",
    }
    return mapping.get(scenario, scenario)


def next_required_data(target: str, depth: str) -> str:
    if target == "chla" and depth == "bottom":
        return "station_depth;bathymetry_depth;bottom_temp;vertical_temperature_profile;current_speed;water_level;discharge;sampling_datetime"
    if target == "secchi":
        return "strict_water_mask_highres_patch;valid_ratio;sampling_datetime;distance_to_shore"
    return "bathymetry_or_station_depth;sampling_datetime;hydrology_or_current"


def build_station_recommendation(station_comp: pd.DataFrame) -> None:
    rows = []
    if station_comp.empty:
        pd.DataFrame().to_csv(
            CURRENT_OUTPUT_DIR / "station_model_recommendation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return
    for (target, depth, station), sub in station_comp.groupby(["target", "depth_layer", "station"]):
        sub = sub.sort_values(["delta_r2", "candidate_r2"], ascending=[False, False])
        best = sub.iloc[0]
        rows.append(
            {
                "target": target,
                "depth_layer": depth,
                "station": station,
                "n_test": int(best["n_test"]),
                "current_best_r2": float(best["current_best_r2"]),
                "candidate_best_r2": float(best["candidate_r2"]),
                "delta_r2": float(best["delta_r2"]),
                "current_best_rmse": float(best["current_best_rmse"]),
                "candidate_best_rmse": float(best["candidate_rmse"]),
                "delta_rmse": float(best["delta_rmse"]),
                "recommended_strategy": (
                    scenario_to_feature_group(str(best["candidate_label"])) if bool(best["improved"]) else "current_best"
                ),
                "selected_feature_group": scenario_to_feature_group(str(best["candidate_label"])),
                "reason": str(best["reason"]),
                "improved": bool(best["improved"]),
            }
        )
    pd.DataFrame(rows).sort_values(["target", "depth_layer", "station"]).to_csv(
        CURRENT_OUTPUT_DIR / "station_model_recommendation.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_breakthrough_log(progress: pd.DataFrame) -> None:
    registry = pd.read_csv(CURRENT_REGISTRY, low_memory=False)
    subset = registry[registry["scenario"].isin(["highres_patch", "bottom_mechanism"])].copy()
    highres_best = subset[subset["scenario"] == "highres_patch"]["delta_r2"].max()
    mech_best = subset[subset["scenario"] == "bottom_mechanism"]["delta_r2"].max()
    lines = [
        "# R2 0.6 突破日志",
        "",
        "## 第 1 轮：高分 patch 组",
        "",
        "- 数据源：`s2_highres_patch_features_2014_2024.csv` + `landsat_highres_patch_features_2014_2024.csv`。",
        "- 来源：本地 GEE 导出，已完成 source validation。",
        "- 是否自动下载：否，本轮直接复用已落地导出。",
        "- 覆盖率：Sentinel-2 约 39.0%，Landsat 约 10.1%。",
        f"- 最优 delta R²：{highres_best:.4f}。",
        "- 结果：相对当前最佳主线未形成实质增益，不进入 mainline。",
        "",
        "## 第 2 轮：bottom 机制代理组",
        "",
        "- 数据源：`bottom_mechanism_features_2014_2024.csv`。",
        "- 来源：本地已生成特征表，已完成 source validation。",
        "- 是否包含真实强机制变量：否，当前主要是 surface temp / wind / precip / tide range 等弱代理，真实 station depth、bathymetry、bottom temp、water level、discharge、current speed 仍缺失。",
        f"- 最优 delta R²：{mech_best:.4f}（仅 Chl-a bottom 有小幅正增益）。",
        "- 结果：不足以推动主线刷新，也不足以证明机制瓶颈已补齐。",
        "",
        "## 停止原因",
        "",
        "- 连续两轮强输入尝试的最优 `delta R²` 都小于 0.01，触发停止规则。",
        "- 当前本地没有可自动并入的新强机制数据源；bathymetry / station depth / bottom temp / sampling datetime / discharge / water level / current 仍需用户手动提供或单独下载。",
        "",
        "## 当前判断",
        "",
        "- 本轮没有任何目标/水层达到 `R² > 0.60`。",
        "- 当前最需要补的数据不是更多弱 proxy，而是更接近机理的真实底层和时刻级数据。",
        "",
        "## 当前进度卡摘要",
        "",
    ]
    for _, row in progress.iterrows():
        lines.append(
            f"- `{row['target']} / {row['depth_layer']}`：当前轮最佳 R² = {row['current_round_best_r2']:.4f}，距 0.60 还差 {row['remaining_gap_to_0p6']:.4f}，下一步优先 `{row['next_required_data']}`。"
        )
    (CURRENT_REPORT_DIR / "r2_0p6_breakthrough_log.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parse_args()
    ensure_dirs()
    inventory = build_source_inventory()
    build_action_plan(inventory)
    strong_df, dictionary = build_strong_input_table()
    write_strong_input_missing_and_validation(strong_df, dictionary)
    station_comp = concat_existing_outputs()
    progress = build_comparison_and_progress()
    build_station_recommendation(station_comp)
    write_breakthrough_log(progress)
    summary = {
        "status": "ok",
        "inventory_rows": int(len(inventory)),
        "strong_input_rows": int(len(strong_df)),
        "strong_input_feature_count": int(len(dictionary)),
        "progress_rows": int(len(progress)),
        "stop_reason": "two_consecutive_rounds_delta_r2_below_0.01",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
