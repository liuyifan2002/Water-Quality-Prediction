from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from new_data_sources_utils import ROOT, date_tag_or_today, load_yaml_config, station_meta_frame, write_json


def build_sources() -> dict[str, dict[str, Any]]:
    return {
        "sentinel3_olci": {
            "path": ROOT / "data" / "model_input" / "new_sources" / "sentinel3_olci_features_2014_2024.csv",
            "template": ROOT / "data" / "raw" / "manual" / "ocean_color" / "sentinel3_olci_features_2014_2024_template.csv",
            "required": ["date", "station"],
            "expected": [
                "date",
                "station",
                "olci_valid",
                "olci_day_diff",
                "olci_chla",
                "olci_tsm",
                "olci_kd490",
                "olci_rrs_443",
                "olci_rrs_490",
                "olci_rrs_510",
                "olci_rrs_560",
                "olci_rrs_665",
                "olci_rrs_681",
                "olci_rrs_709",
                "olci_ndci",
                "olci_mci",
                "olci_fai",
                "olci_quality_flag",
            ],
            "target_range": (2014, 2024),
            "station_metric": "station",
            "valid_col": "olci_valid",
        },
        "ocean_color_kd490_tss": {
            "path": ROOT / "data" / "model_input" / "new_sources" / "ocean_color_kd490_tss_features_2014_2024.csv",
            "template": ROOT / "data" / "raw" / "manual" / "ocean_color" / "ocean_color_kd490_tss_features_2014_2024_template.csv",
            "required": ["date", "station"],
            "expected": [
                "date",
                "station",
                "oc_valid",
                "oc_day_diff",
                "oc_chl",
                "oc_kd490",
                "oc_spm",
                "oc_zsd",
                "oc_pp",
                "oc_rrs_400",
                "oc_rrs_412",
                "oc_rrs_443",
                "oc_rrs_490",
                "oc_rrs_510",
                "oc_rrs_560",
                "oc_rrs_620",
                "oc_rrs_665",
                "oc_rrs_681",
                "oc_rrs_709",
                "oc_ndci",
                "oc_mci",
                "oc_fai",
                "oc_chl_3d",
                "oc_chl_7d",
                "oc_chl_14d",
                "oc_kd490_3d",
                "oc_kd490_7d",
                "oc_kd490_14d",
                "oc_spm_3d",
                "oc_spm_7d",
                "oc_spm_14d",
                "oc_zsd_3d",
                "oc_zsd_7d",
                "oc_zsd_14d",
                "valid_count_7d",
                "oc_missing_reason",
                "oc_pixel_distance_km",
                "oc_source_dataset_id",
                "oc_temporal_resolution",
            ],
            "target_range": (2014, 2024),
            "station_metric": "station",
            "valid_col": "oc_valid",
        },
        "copernicus_physics": {
            "path": ROOT / "data" / "model_input" / "new_sources" / "copernicus_physics_features_2014_2024.csv",
            "template": ROOT / "data" / "raw" / "manual" / "copernicus_physics" / "copernicus_physics_features_2014_2024_template.csv",
            "required": ["date", "station"],
            "expected": [
                "date",
                "station",
                "cmems_sst",
                "cmems_sss",
                "cmems_mld",
                "cmems_ssh",
                "cmems_uo",
                "cmems_vo",
                "cmems_current_speed",
                "cmems_current_dir",
                "cmems_bottom_temp",
                "cmems_sss_3d",
                "cmems_sss_7d",
                "cmems_sss_14d",
                "cmems_mld_3d",
                "cmems_mld_7d",
                "cmems_mld_14d",
                "cmems_current_speed_3d",
                "cmems_current_speed_7d",
                "cmems_current_speed_14d",
                "cmems_plume_index",
            ],
            "target_range": (2014, 2024),
            "station_metric": "station",
            "valid_col": None,
        },
        "copernicus_bgc": {
            "path": ROOT / "data" / "model_input" / "new_sources" / "copernicus_bgc_features_2014_2024.csv",
            "template": ROOT / "data" / "raw" / "manual" / "copernicus_bgc" / "copernicus_bgc_features_2014_2024_template.csv",
            "required": ["date", "station"],
            "expected": [
                "date",
                "station",
                "bgc_chl",
                "bgc_no3",
                "bgc_po4",
                "bgc_si",
                "bgc_o2",
                "bgc_npp",
                "bgc_phyto",
                "bgc_zoo",
                "bgc_no3_7d",
                "bgc_po4_7d",
                "bgc_npp_7d",
                "bgc_chl_anom_7d",
                "bgc_n_to_p_ratio",
            ],
            "target_range": (2014, 2024),
            "station_metric": "station",
            "valid_col": None,
        },
        "lagged_cross_water_quality": {
            "path": ROOT / "data" / "model_input" / "new_sources" / "lagged_cross_water_quality_features_2014_2024.csv",
            "template": ROOT / "data" / "raw" / "manual" / "long_history_water_quality" / "lagged_cross_water_quality_features_2014_2024_template.csv",
            "required": ["obs_id", "sample_id", "station", "date", "depth"],
            "expected": [
                "obs_id",
                "sample_id",
                "station",
                "date",
                "depth",
                "tn_lag1_for_chla",
                "tp_lag1_for_chla",
                "secchi_lag1_for_chla",
                "do_lag1_for_chla",
                "salinity_lag1_for_chla",
                "turbidity_lag1_for_chla",
                "tn_roll3_for_chla",
                "tp_roll3_for_chla",
                "secchi_roll3_for_chla",
                "nutrient_proxy_lag1",
            ],
            "target_range": (2014, 2024),
            "station_metric": "station",
            "valid_col": None,
        },
        "long_history_chla": {
            "path": ROOT / "data" / "model_input" / "new_sources" / "long_history_chla_features_1986_2024.csv",
            "template": ROOT / "data" / "raw" / "manual" / "long_history_water_quality" / "long_history_chla_features_1986_2024_template.csv",
            "required": ["station", "date", "depth"],
            "expected": [
                "station",
                "date",
                "depth",
                "long_chla_lag1",
                "long_chla_lag2",
                "long_chla_roll3",
                "long_tn_lag1",
                "long_tp_lag1",
                "long_secchi_lag1",
                "long_station_month_median_chla",
                "long_station_month_p90_chla",
                "long_zone_month_median_chla",
                "long_zone_month_p90_chla",
            ],
            "target_range": (1986, 2024),
            "station_metric": "station",
            "valid_col": None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验新增 Chl-a 数据源输入文件。")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--date-tag", default="20260516")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def safe_missing_rate(series: pd.Series) -> float:
    return float(series.isna().mean()) if len(series) else float("nan")


def constant_flag(series: pd.Series) -> bool:
    non_null = series.dropna()
    return bool(not non_null.empty and non_null.nunique(dropna=True) == 1)


def validate_source(name: str, spec: dict[str, Any], station_count: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    real_path: Path = spec["path"]
    template_path: Path = spec["template"]
    required = list(spec["required"])
    expected = list(spec["expected"])
    target_start, target_end = spec["target_range"]
    valid_col = spec.get("valid_col")

    if not real_path.exists():
        readiness = "template_only" if template_path.exists() else "missing_real_input"
        row = {
            "source": name,
            "input_path": str(real_path),
            "template_path": str(template_path),
            "exists": False,
            "template_exists": template_path.exists(),
            "readiness_level": readiness,
            "reason": "未发现真实输入文件；模板文件不会被当作真实数据。",
            "rows": 0,
            "columns": 0,
            "date_min": None,
            "date_max": None,
            "station_coverage_rate": 0.0,
            "valid_ratio": np.nan,
            "covers_required_range": False,
            "required_fields_present": False,
            "missing_required_fields": required,
            "missing_expected_fields": expected,
            "all_empty_columns": [],
            "constant_columns": [],
        }
        return row, [], []

    try:
        df = pd.read_csv(real_path, low_memory=False)
    except Exception as exc:
        row = {
            "source": name,
            "input_path": str(real_path),
            "template_path": str(template_path),
            "exists": True,
            "template_exists": template_path.exists(),
            "readiness_level": "read_error",
            "reason": f"文件读取失败: {type(exc).__name__}: {exc}",
            "rows": 0,
            "columns": 0,
            "date_min": None,
            "date_max": None,
            "station_coverage_rate": 0.0,
            "valid_ratio": np.nan,
            "covers_required_range": False,
            "required_fields_present": False,
            "missing_required_fields": required,
            "missing_expected_fields": expected,
            "all_empty_columns": [],
            "constant_columns": [],
        }
        return row, [], []

    missing_required = [col for col in required if col not in df.columns]
    missing_expected = [col for col in expected if col not in df.columns]
    parsed_dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
    date_parse_ok = bool(parsed_dates.notna().any()) if "date" in df.columns else False
    date_min = parsed_dates.min().date().isoformat() if date_parse_ok else None
    date_max = parsed_dates.max().date().isoformat() if date_parse_ok else None

    covers_range = False
    if date_parse_ok:
        years = parsed_dates.dt.year.dropna()
        covers_range = bool((years.min() <= target_start) and (years.max() >= target_end))

    station_coverage_rate = 0.0
    if "station" in df.columns and station_count > 0:
        station_coverage_rate = float(df["station"].dropna().astype(str).str.upper().nunique() / station_count)

    valid_ratio = np.nan
    if valid_col and valid_col in df.columns:
        valid_numeric = pd.to_numeric(df[valid_col], errors="coerce")
        if valid_numeric.notna().any():
            valid_ratio = float(valid_numeric.fillna(0).mean())

    field_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    all_empty: list[str] = []
    all_constant: list[str] = []
    for col in df.columns:
        miss = safe_missing_rate(df[col])
        is_empty = bool(df[col].isna().all())
        is_constant = constant_flag(df[col])
        if is_empty:
            all_empty.append(col)
        if is_constant:
            all_constant.append(col)
        numeric = pd.to_numeric(df[col], errors="coerce")
        field_rows.append(
            {
                "source": name,
                "field": col,
                "rows": len(df),
                "non_null_rows": int(df[col].notna().sum()),
                "missing_rate": miss,
                "is_required": col in required,
                "is_expected": col in expected,
                "is_all_empty": is_empty,
                "is_constant": is_constant,
                "numeric_min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "numeric_max": float(numeric.max()) if numeric.notna().any() else np.nan,
            }
        )
        missing_rows.append(
            {
                "source": name,
                "field": col,
                "missing_rate": miss,
                "non_null_rows": int(df[col].notna().sum()),
            }
        )

    core_expected = set(required)
    if valid_col:
        core_expected.add(valid_col)

    blocking_all_empty = [col for col in all_empty if col in core_expected]

    if missing_required:
        readiness = "not_ready"
        reason = "缺少关键字段。"
    elif len(df) == 0:
        readiness = "not_ready"
        reason = "文件为空。"
    elif not date_parse_ok:
        readiness = "not_ready"
        reason = "date 字段无法解析。"
    elif blocking_all_empty:
        readiness = "needs_review"
        reason = "关键字段中存在全空列，需要确认提取过程。"
    elif not covers_range:
        readiness = "needs_review"
        reason = "时间范围未覆盖目标区间。"
    elif valid_col and valid_col in df.columns and pd.notna(valid_ratio) and valid_ratio == 0:
        readiness = "needs_review"
        reason = f"{valid_col} 全为 0，说明当前数据不可用。"
    elif missing_expected:
        readiness = "ready_for_ab_experiment"
        reason = "关键字段通过，部分建议字段缺失；可进入 A/B，但需在报告中说明变量覆盖边界。"
    elif all_empty:
        readiness = "ready_for_ab_experiment"
        reason = "关键字段通过，部分可选字段为空；可进入 A/B，但需在报告中说明变量覆盖边界。"
    else:
        readiness = "ready_for_ab_experiment"
        reason = "关键字段、时间范围和基础质量检查通过。"

    row = {
        "source": name,
        "input_path": str(real_path),
        "template_path": str(template_path),
        "exists": True,
        "template_exists": template_path.exists(),
        "readiness_level": readiness,
        "reason": reason,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "date_min": date_min,
        "date_max": date_max,
        "station_coverage_rate": station_coverage_rate,
        "valid_ratio": valid_ratio,
        "target_start_year": target_start,
        "target_end_year": target_end,
        "covers_required_range": covers_range,
        "required_fields_present": len(missing_required) == 0,
        "missing_required_fields": missing_required,
        "missing_expected_fields": missing_expected,
        "all_empty_columns": all_empty,
        "constant_columns": all_constant,
        "max_missing_rate": float(pd.Series([r["missing_rate"] for r in missing_rows]).max()) if missing_rows else np.nan,
    }
    return row, field_rows, missing_rows


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    out_dir = args.out_dir or (ROOT / "outputs" / "modeling" / "experiments" / f"chla_new_data_source_validation_{date_tag_or_today(args.date_tag)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    station_count = len(station_meta_frame(config))
    sources = build_sources()
    readiness_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    report_sources: dict[str, Any] = {}

    for name, spec in sources.items():
        row, fields, missing = validate_source(name, spec, station_count)
        readiness_rows.append(row)
        field_rows.extend(fields)
        missing_rows.extend(missing)
        report_sources[name] = row

    readiness_df = pd.DataFrame(readiness_rows).sort_values("source")
    field_df = pd.DataFrame(field_rows)
    missing_df = pd.DataFrame(missing_rows)

    readiness_df.to_csv(out_dir / "new_data_source_readiness_table.csv", index=False, encoding="utf-8-sig")
    if not field_df.empty:
        field_df.sort_values(["source", "field"]).to_csv(out_dir / "new_data_source_field_summary.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["source", "field", "rows", "non_null_rows", "missing_rate"]).to_csv(
            out_dir / "new_data_source_field_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if not missing_df.empty:
        missing_df.sort_values(["source", "missing_rate", "field"], ascending=[True, False, True]).to_csv(
            out_dir / "new_data_source_missing_rate.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame(columns=["source", "field", "missing_rate", "non_null_rows"]).to_csv(
            out_dir / "new_data_source_missing_rate.csv",
            index=False,
            encoding="utf-8-sig",
        )

    ready_sources = readiness_df.loc[readiness_df["readiness_level"] == "ready_for_ab_experiment", "source"].tolist()
    template_only_sources = readiness_df.loc[readiness_df["readiness_level"] == "template_only", "source"].tolist()
    missing_real_input_sources = readiness_df.loc[readiness_df["readiness_level"] == "missing_real_input", "source"].tolist()

    report = {
        "label": "Chl-a 新增数据源校验报告",
        "output_dir": str(out_dir),
        "ready_sources": ready_sources,
        "template_only_sources": template_only_sources,
        "missing_real_input_sources": missing_real_input_sources,
        "source_count": len(readiness_rows),
        "sources": report_sources,
    }
    write_json(out_dir / "new_data_source_validation_report.json", report)

    print(f"Saved validation outputs to {out_dir}")
    print(f"ready_for_ab_experiment: {', '.join(ready_sources) if ready_sources else 'none'}")


if __name__ == "__main__":
    main()
