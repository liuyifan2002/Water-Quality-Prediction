from __future__ import annotations

import argparse
import calendar
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import xarray as xr


RAW_DIR = Path("data/raw/era5")
DEFAULT_OUT = Path("data/processed/era5_monthly_2014_2024.csv")
VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_cloud_cover"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并聚合 ERA5 single levels 香港区域月均气象。")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-download", action="store_true", help="已有 NetCDF 时只做聚合。")
    parser.add_argument("--mode", choices=["timeseries", "grid"], default="timeseries")
    return parser.parse_args()


def download_month(year: int, month: int, out_path: Path) -> None:
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("缺少 cdsapi，请先安装并配置 CDS API：pip install cdsapi") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"已存在，跳过下载：{out_path}")
        return

    client_kwargs = {}
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        client_kwargs = {"url": os.environ["CDSAPI_URL"], "key": os.environ["CDSAPI_KEY"]}
    client = cdsapi.Client(**client_kwargs)
    _, last_day = calendar.monthrange(year, month)
    request = {
        "product_type": "reanalysis",
        "variable": VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, last_day + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [23.0, 113.5, 22.0, 115.0]
    }
    for attempt in range(1, 4):
        try:
            print(f"下载 ERA5 {year}-{month:02d}，第 {attempt} 次尝试")
            client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
            return
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"下载 {year}-{month:02d} 失败，准备重试：{exc}")
            time.sleep(30 * attempt)


def download_timeseries_year(year: int, out_path: Path) -> None:
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("缺少 cdsapi，请先安装并配置 CDS API：pip install cdsapi") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"已存在，跳过下载：{out_path}")
        return

    client_kwargs = {}
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        client_kwargs = {"url": os.environ["CDSAPI_URL"], "key": os.environ["CDSAPI_KEY"]}
    client = cdsapi.Client(**client_kwargs)
    request = {
        "variable": VARIABLES,
        "location": {"longitude": 114.16, "latitude": 22.32},
        "date": [f"{year}-01-01/{year}-12-31"],
        "data_format": "netcdf"
    }
    for attempt in range(1, 4):
        try:
            print(f"下载 ERA5 time-series {year}，第 {attempt} 次尝试")
            client.retrieve("reanalysis-era5-single-levels-timeseries", request, str(out_path))
            return
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"下载 {year} 失败，准备重试：{exc}")
            time.sleep(30 * attempt)


def aggregate_monthly(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        ds = xr.open_dataset(path, engine="h5netcdf")
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        keep_vars = [name for name in ["t2m", "u10", "v10", "tcc"] if name in ds.data_vars]
        if len(keep_vars) != 4:
            raise ValueError(f"{path} 缺少 ERA5 变量，当前变量：{list(ds.data_vars)}")

        monthly = (
            ds[keep_vars]
            .mean(dim=[dim for dim in ["latitude", "longitude"] if dim in ds.dims])
            .resample({time_name: "MS"})
            .mean()
        )
        frame = monthly.to_dataframe().reset_index()
        frame["year"] = frame[time_name].dt.year
        frame["month"] = frame[time_name].dt.month
        frame = frame[["year", "month", "t2m", "u10", "v10", "tcc"]]
        frame["t2m"] = frame["t2m"] - 273.15
        frames.append(frame)
        ds.close()

    return pd.concat(frames, ignore_index=True).sort_values(["year", "month"])


def extract_timeseries_zip(path: Path) -> Path:
    if not zipfile.is_zipfile(path):
        return path
    out_dir = path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        nc_names = [name for name in zf.namelist() if name.lower().endswith(".nc")]
        if len(nc_names) != 1:
            raise ValueError(f"{path} 中 NetCDF 文件数量异常：{nc_names}")
        nc_path = out_dir / Path(nc_names[0]).name
        if not nc_path.exists():
            zf.extract(nc_names[0], out_dir)
            extracted = out_dir / nc_names[0]
            if extracted != nc_path:
                extracted.replace(nc_path)
        return nc_path


def main() -> None:
    args = parse_args()
    if os.environ.get("CDSAPI_IGNORE_PROXY") == "1":
        for key in list(os.environ):
            if "proxy" in key.lower():
                os.environ.pop(key, None)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    if args.mode == "timeseries":
        for year in range(args.start_year, args.end_year + 1):
            path = args.raw_dir / f"era5_hk_timeseries_{year}.zip"
            paths.append(path)
            if not args.skip_download:
                download_timeseries_year(year, path)
    else:
        for year in range(args.start_year, args.end_year + 1):
            for month in range(1, 13):
                path = args.raw_dir / f"era5_hk_single_levels_{year}_{month:02d}.nc"
                paths.append(path)
                if not args.skip_download:
                    download_month(year, month, path)

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 ERA5 NetCDF：{missing}")

    nc_paths = [extract_timeseries_zip(path) for path in paths]
    out = aggregate_monthly(nc_paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"写出 {args.out}：{len(out)} 个年月")
    print(out.head().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
