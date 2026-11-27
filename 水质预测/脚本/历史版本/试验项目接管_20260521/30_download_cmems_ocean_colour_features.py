from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from new_data_sources_utils import (
    ROOT,
    credentials_present,
    date_tag_or_today,
    ensure_dir,
    experiment_dir,
    haversine_km,
    load_env_file,
    load_yaml_config,
    parse_csv_like_list,
    run_script,
    sample_frame,
    station_meta_frame,
    write_json,
)


VARIABLE_ALIASES = {
    "oc_chl": ["chl", "CHL"],
    "oc_kd490": ["kd490", "KD490"],
    "oc_spm": ["spm", "SPM", "tsm", "TSM"],
    "oc_zsd": ["zsd", "ZSD", "transparency", "secchi_depth"],
    "oc_pp": ["pp", "PP", "npp", "NPP", "primary_production"],
    "oc_rrs_400": ["rrs400", "RRS400"],
    "oc_rrs_412": ["rrs412", "RRS412"],
    "oc_rrs_443": ["rrs443", "RRS443"],
    "oc_rrs_490": ["rrs490", "RRS490"],
    "oc_rrs_510": ["rrs510", "RRS510"],
    "oc_rrs_560": ["rrs560", "RRS560", "rrs555", "RRS555"],
    "oc_rrs_620": ["rrs620", "RRS620"],
    "oc_rrs_665": ["rrs665", "RRS665", "rrs670", "RRS670"],
    "oc_rrs_681": ["rrs681", "RRS681"],
    "oc_rrs_709": ["rrs709", "RRS709"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并提取 Copernicus Marine Ocean Colour 站点特征。")
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--date-tag", default="20260516")
    parser.add_argument("--bbox-buffer", type=float, default=0.2)
    parser.add_argument("--variables", default="")
    parser.add_argument("--temporal-resolution", choices=["daily", "monthly", "auto"], default="auto")
    parser.add_argument("--spatial-mode", choices=["nearest", "mean3x3"], default="nearest")
    return parser.parse_args()


def discovery_recommendation_path(date_tag: str) -> Path:
    return ROOT / "outputs" / "modeling" / "experiments" / f"cmems_ocean_colour_discovery_{date_tag}" / "cmems_ocean_colour_recommendation.json"


def choose_dataset_ids(args: argparse.Namespace, recommendation: dict[str, Any]) -> list[str]:
    if args.dataset_id:
        return parse_csv_like_list(args.dataset_id)
    if args.temporal_resolution == "monthly":
        return list(recommendation.get("recommended_monthly_dataset_bundle", []))
    bundle = list(recommendation.get("recommended_daily_dataset_bundle", []))
    if bundle:
        return bundle
    single = str(recommendation.get("recommended_daily_dataset_id", "")) or str(recommendation.get("recommended_monthly_dataset_id", ""))
    return [single] if single else []


def choose_variables(args: argparse.Namespace, recommendation: dict[str, Any]) -> list[str]:
    manual = parse_csv_like_list(args.variables)
    if manual:
        return manual
    return list(recommendation.get("recommended_variables", []))


def infer_temporal_resolution(dataset_id: str, manual: str) -> str:
    if manual != "auto":
        return manual
    if dataset_id.endswith("P1M"):
        return "monthly"
    return "daily"


def open_dataset(path: Path):
    try:
        import xarray as xr  # type: ignore
    except Exception as exc:
        raise RuntimeError("需要 xarray/netCDF4 才能提取 Ocean Colour 特征。") from exc

    last_exc: Exception | None = None
    for engine in ["h5netcdf", "netcdf4"]:
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:  # pragma: no cover - fallback path
            last_exc = exc
    raise RuntimeError(f"无法打开 NetCDF 文件: {path}") from last_exc


def subset_download(
    dataset_id: str,
    out_file: Path,
    bbox: dict[str, Any],
    start_date: str,
    end_date: str,
    variables: list[str],
    username: str,
    password: str,
) -> None:
    try:
        import copernicusmarine  # type: ignore
    except Exception as exc:
        raise RuntimeError("需要安装 copernicusmarine 才能下载 Ocean Colour 数据。") from exc

    copernicusmarine.login(username=username, password=password, force_overwrite=False, check_credentials_valid=True)
    copernicusmarine.subset(
        dataset_id=dataset_id,
        username=username,
        password=password,
        variables=variables,
        minimum_longitude=float(bbox["min_lon"]),
        maximum_longitude=float(bbox["max_lon"]),
        minimum_latitude=float(bbox["min_lat"]),
        maximum_latitude=float(bbox["max_lat"]),
        start_datetime=start_date,
        end_datetime=end_date,
        output_filename=out_file.name,
        file_format="netcdf",
        output_directory=str(out_file.parent),
        overwrite=True,
    )


def sanitize_dataset_id(dataset_id: str) -> str:
    return dataset_id.replace("/", "_").replace(":", "_")


def variables_for_dataset(dataset_id: str, requested_variables: list[str]) -> list[str]:
    dataset_lower = dataset_id.lower()
    requested_upper = {item.upper() for item in requested_variables}
    if "plankton" in dataset_lower:
        base = ["CHL"]
        return base
    if "transp" in dataset_lower:
        base = [item for item in ["KD490", "ZSD"] if item in requested_upper]
        return base or ["KD490", "ZSD"]
    if "reflectance" in dataset_lower or "optics" in dataset_lower:
        base = [item for item in ["RRS412", "RRS443", "RRS490", "RRS555", "RRS670"] if item in requested_upper or item in {"RRS555", "RRS670"}]
        return base or ["RRS412", "RRS443", "RRS490", "RRS555", "RRS670"]
    return requested_variables


def find_var(ds, aliases: list[str]) -> str | None:
    available = {var.lower(): var for var in ds.data_vars}
    for alias in aliases:
        if alias.lower() in available:
            return available[alias.lower()]
    return None


def safe_index_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    denom = a + b
    if denom == 0:
        return None
    return float((a - b) / denom)


def point_extract(ds, lon: float, lat: float, spatial_mode: str) -> tuple[dict[str, float | None], float]:
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    method = "nearest"
    point = ds.interp({lon_name: lon, lat_name: lat}, method=method)

    out: dict[str, float | None] = {}
    for out_name, aliases in VARIABLE_ALIASES.items():
        source = find_var(ds, aliases)
        if not source:
            continue
        value = point[source]
        raw = np.asarray(value.values).astype(float)
        if raw.size == 0 or not np.isfinite(raw).any():
            out[out_name] = None
            continue
        if spatial_mode == "mean3x3" and raw.ndim >= 2:
            out[out_name] = float(np.nanmean(raw))
        else:
            out[out_name] = float(raw.reshape(-1)[0])

    nearest_lon = float(point[lon_name].values) if lon_name in point.coords else lon
    nearest_lat = float(point[lat_name].values) if lat_name in point.coords else lat
    return out, haversine_km(lon, lat, nearest_lon, nearest_lat)


def compute_indices(row: dict[str, Any], notes: dict[str, list[str]]) -> None:
    row["oc_ndci"] = safe_index_ratio(row.get("oc_rrs_709"), row.get("oc_rrs_665"))

    if row.get("oc_rrs_709") is not None and row.get("oc_rrs_681") is not None:
        row["oc_mci"] = float(row["oc_rrs_709"] - row["oc_rrs_681"])
        notes.setdefault("mci_mode_log", []).append("simplified_red_edge_peak")
    else:
        row["oc_mci"] = None
        notes.setdefault("mci_mode_log", []).append("missing_required_bands")

    row["oc_fai"] = None
    notes.setdefault("fai_mode_log", []).append("not_generated_missing_nir_swir")


def month_climatology_features(df: pd.DataFrame, base: str) -> None:
    if base not in df.columns:
        return
    monthly_mean = df.groupby(["station", "month"], dropna=False)[base].transform("mean")
    df[f"{base}_monthly_anom"] = df[base] - monthly_mean


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    env = load_env_file()
    date_tag = date_tag_or_today(args.date_tag)
    log_dir = ensure_dir(experiment_dir("chla_new_data_download", date_tag))
    raw_dir = ensure_dir(args.out_dir or (ROOT / config["outputs"]["ocean_color_cmems_raw_dir"]))
    new_sources_dir = ensure_dir(ROOT / config["outputs"]["model_input_new_sources_dir"])
    ocean_color_out = new_sources_dir / "ocean_color_kd490_tss_features_2014_2024.csv"
    sentinel_out = new_sources_dir / "sentinel3_olci_features_2014_2024.csv"
    report_path = log_dir / "cmems_ocean_colour_report.json"

    username_key = config.get("copernicus_marine", {}).get("username_env_var", "COPERNICUSMARINE_USERNAME")
    password_key = config.get("copernicus_marine", {}).get("password_env_var", "COPERNICUSMARINE_PASSWORD")
    has_creds, missing_keys = credentials_present(env, [username_key, password_key])

    recommendation = {}
    rec_path = discovery_recommendation_path(date_tag)
    if rec_path.exists():
        recommendation = json.loads(rec_path.read_text(encoding="utf-8"))

    dataset_ids = choose_dataset_ids(args, recommendation)
    variables = choose_variables(args, recommendation)
    temporal_resolution_used = infer_temporal_resolution(dataset_ids[0], args.temporal_resolution) if dataset_ids else infer_temporal_resolution("", args.temporal_resolution)
    stations = station_meta_frame(config)
    samples = sample_frame(config, args.start_date, args.end_date)

    bbox = dict(config["study_bbox"])
    bbox["min_lon"] = bbox["min_lon"] - args.bbox_buffer
    bbox["max_lon"] = bbox["max_lon"] + args.bbox_buffer
    bbox["min_lat"] = bbox["min_lat"] - args.bbox_buffer
    bbox["max_lat"] = bbox["max_lat"] + args.bbox_buffer

    report: dict[str, Any] = {
        "label": "Copernicus Marine Ocean Colour 下载与提取",
        "dry_run": args.dry_run,
        "skip_download": args.skip_download,
        "dataset_id": dataset_ids[0] if dataset_ids else "",
        "dataset_ids": dataset_ids,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "bbox": bbox,
        "variables": variables,
        "raw_dir": str(raw_dir),
        "output_csv": str(ocean_color_out),
        "olci_output_csv": str(sentinel_out),
        "temporal_resolution": temporal_resolution_used,
        "spatial_mode": args.spatial_mode,
        "sample_rows": int(len(samples)),
        "credentials_present": has_creds,
        "missing_credentials": missing_keys,
    }

    if args.dry_run:
        report["status"] = "dry_run_only"
        report["recommended_dataset_from_discovery"] = recommendation.get("recommended_daily_dataset_bundle", [])
        report["missing_years"] = []
        report["fallback_to_monthly"] = temporal_resolution_used == "monthly"
        write_json(report_path, report)
        print("Dry-run only. Planned CMEMS Ocean Colour download/extraction.")
        return

    if not args.skip_download and not has_creds:
        report["status"] = "missing_credentials"
        write_json(report_path, report)
        raise RuntimeError(f"未配置 Copernicus Marine 账号，缺少: {missing_keys}")

    if not dataset_ids:
        report["status"] = "missing_dataset_id"
        write_json(report_path, report)
        raise RuntimeError("未提供 Ocean Colour dataset_id，且 discovery 推荐文件中也没有可用 dataset_id。")

    dataset_objects: list[tuple[str, Any]] = []
    missing_local_files: list[str] = []
    for dataset_id in dataset_ids:
        out_file = raw_dir / f"{sanitize_dataset_id(dataset_id)}_{args.start_date}_{args.end_date}.nc"
        dataset_variables = variables_for_dataset(dataset_id, variables)
        if not args.skip_download and (args.force or not out_file.exists()):
            subset_download(dataset_id, out_file, bbox, args.start_date, args.end_date, dataset_variables, env[username_key], env[password_key])
        if not out_file.exists():
            missing_local_files.append(str(out_file))
            continue
        dataset_objects.append((dataset_id, open_dataset(out_file)))

    if missing_local_files:
        report["status"] = "missing_download_outputs"
        report["missing_local_files"] = missing_local_files
        write_json(report_path, report)
        raise RuntimeError("Ocean Colour subset 已请求，但没有生成预期 NetCDF 文件。请检查 dataset_id、变量名或服务可用性。")

    if not dataset_objects:
        report["status"] = "no_dataset_loaded"
        write_json(report_path, report)
        raise RuntimeError("没有成功加载任何 Ocean Colour 数据集。")

    time_name = "time"
    notes: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []

    for _, sample in samples.iterrows():
        station = sample["station"]
        station_row = stations[stations["station"] == station]
        if station_row.empty:
            continue
        lon = float(station_row.iloc[0]["lon"])
        lat = float(station_row.iloc[0]["lat"])
        sample_date = pd.Timestamp(sample["date"]).normalize()
        dataset_values: dict[str, float | None] = {}
        quality_entries: list[tuple[str, int, float]] = []
        for dataset_id, ds in dataset_objects:
            ds_time = ds.sel({time_name: sample_date}, method="nearest")
            chosen_date = pd.Timestamp(ds_time[time_name].values).normalize()
            day_diff = int((chosen_date - sample_date).days)
            values, pixel_distance_km = point_extract(ds_time, lon, lat, args.spatial_mode)
            for key, value in values.items():
                if dataset_values.get(key) is None and value is not None:
                    dataset_values[key] = value
            if any(v is not None for v in values.values()):
                quality_entries.append((dataset_id, abs(day_diff), pixel_distance_km))

        if quality_entries:
            best_dataset_id, best_day_diff, best_pixel_distance = sorted(quality_entries, key=lambda x: (x[1], x[2]))[0]
            day_diff = best_day_diff
            pixel_distance_km = best_pixel_distance
            source_dataset_id = ",".join([item[0] for item in quality_entries])
            valid_flag = 1
            missing_reason = None
        else:
            day_diff = None
            pixel_distance_km = None
            source_dataset_id = ",".join(dataset_ids)
            valid_flag = 0
            missing_reason = "all_variables_nan"

        row: dict[str, Any] = {
            "date": sample_date.date().isoformat(),
            "station": station,
            "oc_valid": valid_flag,
            "oc_day_diff": day_diff,
            "oc_missing_reason": missing_reason,
            "oc_pixel_distance_km": pixel_distance_km,
            "oc_source_dataset_id": source_dataset_id,
            "oc_temporal_resolution": temporal_resolution_used,
        }
        row.update(dataset_values)
        compute_indices(row, notes)
        rows.append(row)

    oc_df = pd.DataFrame(rows).sort_values(["station", "date"]).reset_index(drop=True)
    oc_df["date"] = pd.to_datetime(oc_df["date"])
    oc_df["month"] = oc_df["date"].dt.month

    if temporal_resolution_used == "daily":
        for base in ["oc_chl", "oc_kd490", "oc_spm", "oc_zsd"]:
            if base in oc_df.columns:
                grouped = oc_df.groupby("station", dropna=False)[base]
                for window in [3, 7, 14]:
                    oc_df[f"{base}_{window}d"] = grouped.transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        if "oc_valid" in oc_df.columns:
            oc_df["valid_count_7d"] = oc_df.groupby("station", dropna=False)["oc_valid"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    else:
        for base in ["oc_chl", "oc_kd490", "oc_spm", "oc_zsd"]:
            month_climatology_features(oc_df, base)
        notes.setdefault("resolution_notes", []).append("only_monthly_product_available")

    oc_df["date"] = oc_df["date"].dt.date.astype(str)
    oc_df.to_csv(ocean_color_out, index=False, encoding="utf-8-sig")

    olci_cols = {
        "oc_valid": "olci_valid",
        "oc_day_diff": "olci_day_diff",
        "oc_chl": "olci_chla",
        "oc_spm": "olci_tsm",
        "oc_kd490": "olci_kd490",
        "oc_rrs_443": "olci_rrs_443",
        "oc_rrs_490": "olci_rrs_490",
        "oc_rrs_510": "olci_rrs_510",
        "oc_rrs_560": "olci_rrs_560",
        "oc_rrs_665": "olci_rrs_665",
        "oc_rrs_681": "olci_rrs_681",
        "oc_rrs_709": "olci_rrs_709",
        "oc_ndci": "olci_ndci",
        "oc_mci": "olci_mci",
        "oc_fai": "olci_fai",
        "oc_missing_reason": "olci_quality_flag",
    }
    available_olci = [col for col in olci_cols if col in oc_df.columns]
    if available_olci:
        olci_df = oc_df[["date", "station"] + available_olci].rename(columns=olci_cols)
        olci_df.to_csv(sentinel_out, index=False, encoding="utf-8-sig")

    report["status"] = "completed"
    report["rows"] = int(len(oc_df))
    report["missing_rate"] = oc_df.isna().mean().round(4).to_dict() if not oc_df.empty else {}
    report["valid_sample_count"] = oc_df.notna().sum().to_dict() if not oc_df.empty else {}
    report["mci_mode_log"] = sorted(set(notes.get("mci_mode_log", [])))
    report["fai_mode_log"] = sorted(set(notes.get("fai_mode_log", [])))
    report["resolution_notes"] = sorted(set(notes.get("resolution_notes", [])))
    report["missing_years"] = []
    write_json(report_path, report)

    validator = run_script("21_validate_new_chla_data_sources.py", ["--date-tag", date_tag])
    print(validator.stdout.strip())
    if validator.returncode != 0:
        raise RuntimeError(validator.stderr.strip() or "新增数据源校验失败。")
    print(f"Saved {ocean_color_out}")


if __name__ == "__main__":
    main()
