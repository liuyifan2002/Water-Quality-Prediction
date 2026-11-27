from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from new_data_sources_utils import (
    ROOT,
    angle_degrees,
    credentials_present,
    date_tag_or_today,
    ensure_dir,
    experiment_dir,
    load_env_file,
    load_yaml_config,
    parse_date,
    run_script,
    sample_frame,
    station_meta_frame,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare Copernicus Marine physics features.")
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--date-tag", default="20260516")
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--interpolation", choices=["nearest", "linear"], default="nearest")
    return parser.parse_args()


def open_dataset(path: Path):
    try:
        import xarray as xr  # type: ignore

        for engine in ["h5netcdf", "scipy", None]:
            try:
                if engine is None:
                    return xr.open_dataset(path)
                return xr.open_dataset(path, engine=engine)
            except Exception:
                continue
        return xr.open_dataset(path)
    except Exception as exc:
        raise RuntimeError("需要 xarray/netCDF4 才能提取 CMEMS 物理场。") from exc


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
        raise RuntimeError("需要安装 copernicusmarine 才能下载 Copernicus Marine 数据。") from exc

    copernicusmarine.login(
        username=username,
        password=password,
        force_overwrite=False,
        check_credentials_valid=True,
    )
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
        output_directory=str(out_file.parent),
        overwrite=True,
        disable_progress_bar=True,
    )


def extract_features(nc_path: Path, stations: pd.DataFrame, samples: pd.DataFrame, interpolation: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    ds = open_dataset(nc_path)
    coord_map = {
        "lon": "longitude" if "longitude" in ds.coords else "lon",
        "lat": "latitude" if "latitude" in ds.coords else "lat",
        "time": "time",
        "depth": "depth" if "depth" in ds.coords else None,
    }
    available = {key: key for key in ds.data_vars}
    rows = []
    for _, sample in samples.iterrows():
        station = sample["station"]
        sample_date = pd.Timestamp(sample["date"]).normalize()
        station_row = stations[stations["station"] == station]
        if station_row.empty:
            continue
        lon = float(station_row.iloc[0]["lon"])
        lat = float(station_row.iloc[0]["lat"])
        ds_day = ds.sel({coord_map["time"]: sample_date}, method="nearest")
        method = "nearest" if interpolation == "nearest" else "linear"
        point = ds_day.interp({coord_map["lon"]: lon, coord_map["lat"]: lat}, method=method)
        row = {"date": sample_date.date().isoformat(), "station": station}
        for out_var in [
            "cmems_sst",
            "cmems_bottom_temp",
            "cmems_sss",
            "cmems_uo",
            "cmems_vo",
            "cmems_ssh",
            "cmems_mld",
        ]:
            row[out_var] = np.nan

        if "thetao" in available:
            thetao = point["thetao"]
            if coord_map["depth"] and coord_map["depth"] in thetao.dims:
                row["cmems_sst"] = float(thetao.isel({coord_map["depth"]: 0}).values)
                row["cmems_bottom_temp"] = float(thetao.isel({coord_map["depth"]: -1}).values)
            else:
                row["cmems_sst"] = float(thetao.values)
                row["cmems_bottom_temp"] = float(thetao.values)

        if "so" in available:
            values = point["so"]
            row["cmems_sss"] = float(
                values.isel({coord_map["depth"]: 0}).values
                if coord_map["depth"] and coord_map["depth"] in values.dims
                else values.values
            )
        if "uo" in available:
            values = point["uo"]
            row["cmems_uo"] = float(
                values.isel({coord_map["depth"]: 0}).values
                if coord_map["depth"] and coord_map["depth"] in values.dims
                else values.values
            )
        if "vo" in available:
            values = point["vo"]
            row["cmems_vo"] = float(
                values.isel({coord_map["depth"]: 0}).values
                if coord_map["depth"] and coord_map["depth"] in values.dims
                else values.values
            )
        if "zos" in available:
            row["cmems_ssh"] = float(point["zos"].values)
        if "mlotst" in available:
            row["cmems_mld"] = float(point["mlotst"].values)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["station", "date"]).reset_index(drop=True)
    if df.empty:
        return df, {"available_vars": list(available), "note": "提取结果为空。"}
    df["cmems_current_speed"] = np.sqrt(df["cmems_uo"] ** 2 + df["cmems_vo"] ** 2)
    df["cmems_current_dir"] = angle_degrees(df["cmems_uo"], df["cmems_vo"])
    grouped = df.groupby("station", dropna=False)
    for base in ["cmems_sss", "cmems_mld", "cmems_current_speed"]:
        for window in [3, 7, 14]:
            df[f"{base}_{window}d"] = grouped[base].transform(lambda s: s.rolling(window, min_periods=1).mean())
    df["cmems_plume_index"] = np.nan
    return df, {"available_vars": list(available), "plume_index_generated": False, "plume_index_note": "当前未定义稳定 plume_index 公式，暂不生成。"}


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    env = load_env_file()
    date_tag = date_tag_or_today(args.date_tag)
    log_dir = ensure_dir(experiment_dir("chla_new_data_download", date_tag))
    raw_dir = ensure_dir(args.out_dir or (ROOT / config["outputs"]["copernicus_physics_raw_dir"]))
    new_sources_dir = ensure_dir(ROOT / config["outputs"]["model_input_new_sources_dir"])
    out_csv = new_sources_dir / "copernicus_physics_features_2014_2024.csv"
    log_path = log_dir / "cmems_physics_report.json"
    dataset_id = args.dataset_id

    username_key = config.get("copernicus_marine", {}).get("username_env_var", "COPERNICUSMARINE_USERNAME")
    password_key = config.get("copernicus_marine", {}).get("password_env_var", "COPERNICUSMARINE_PASSWORD")
    has_creds, missing_keys = credentials_present(env, [username_key, password_key])
    stations = station_meta_frame(config)
    samples = sample_frame(config, args.start_date, args.end_date)
    bbox = config["study_bbox"]
    variables = ["thetao", "so", "uo", "vo", "zos", "mlotst"]

    report: dict[str, Any] = {
        "label": "Copernicus Marine 物理场下载与提取",
        "dry_run": args.dry_run,
        "skip_download": args.skip_download,
        "dataset_id": dataset_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "bbox": bbox,
        "variables": variables,
        "raw_dir": str(raw_dir),
        "output_csv": str(out_csv),
        "interpolation": args.interpolation,
        "sample_rows": int(len(samples)),
        "credentials_present": has_creds,
        "missing_credentials": missing_keys,
    }

    nc_path = raw_dir / f"cmems_physics_{args.start_date}_{args.end_date}.nc"
    if args.dry_run:
        report["status"] = "dry_run_only"
        write_json(log_path, report)
        print("Dry-run only. Planned Copernicus Marine physics download/extraction.")
        return

    if not args.skip_download and not has_creds:
        report["status"] = "missing_credentials"
        write_json(log_path, report)
        raise RuntimeError(f"未配置 Copernicus Marine 账号：{missing_keys}")
    if not args.skip_download and not dataset_id:
        report["status"] = "missing_dataset_id"
        write_json(log_path, report)
        raise RuntimeError("未提供 Copernicus Marine physics dataset_id。")
    if not args.skip_download:
        subset_download(dataset_id, nc_path, bbox, args.start_date, args.end_date, variables, env[username_key], env[password_key])
    elif not nc_path.exists():
        report["status"] = "missing_local_netcdf"
        write_json(log_path, report)
        raise RuntimeError(f"未找到本地 NetCDF：{nc_path}")

    extracted, extra = extract_features(nc_path, stations, samples, args.interpolation)
    extracted.to_csv(out_csv, index=False, encoding="utf-8-sig")
    report.update(extra)
    report["status"] = "completed"
    report["rows"] = int(len(extracted))
    report["missing_rate"] = extracted.isna().mean().round(4).to_dict() if not extracted.empty else {}
    write_json(log_path, report)
    validator = run_script("21_validate_new_chla_data_sources.py", ["--date-tag", date_tag])
    print(validator.stdout.strip())
    if validator.returncode != 0:
        raise RuntimeError(validator.stderr.strip() or "新增数据源校验失败。")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
