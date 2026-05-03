from __future__ import annotations

import argparse
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RAW_DIR = Path("data/raw/era5_chla_drivers")
OUT = Path("data/processed/era5_ssrd_sst_daily_2014_2024.csv")
FEATURE_OUT = Path("data/processed/era5_ssrd_sst_window_features_2014_2024.csv")
WATER = Path("data/processed/water_quality_surface_2014_2024.csv")

VARIABLES = [
    "surface_solar_radiation_downwards",
    "sea_surface_temperature",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并整理 Chl-a 用 ERA5 ssrd/sst 日尺度与窗口特征。")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--daily-out", type=Path, default=OUT)
    parser.add_argument("--feature-out", type=Path, default=FEATURE_OUT)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--longitude", type=float, default=114.16)
    parser.add_argument("--latitude", type=float, default=22.20)
    return parser.parse_args()


def setup_cds_env() -> None:
    cds = Path("scripts/.cdsapirc")
    if cds.exists():
        lines = cds.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("url:"):
                os.environ.setdefault("CDSAPI_URL", line.split(":", 1)[1].strip())
            if line.startswith("key:"):
                os.environ.setdefault("CDSAPI_KEY", line.split(":", 1)[1].strip())


def download_year(year: int, path: Path, lon: float, lat: float) -> None:
    import cdsapi

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"已存在，跳过下载: {path}")
        return
    client_kwargs = {}
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        client_kwargs = {"url": os.environ["CDSAPI_URL"], "key": os.environ["CDSAPI_KEY"]}
    client = cdsapi.Client(**client_kwargs)
    request = {
        "variable": VARIABLES,
        "location": {"longitude": lon, "latitude": lat},
        "date": [f"{year}-01-01/{year}-12-31"],
        "data_format": "netcdf",
    }
    for attempt in range(1, 4):
        try:
            print(f"下载 ERA5 Chl-a drivers {year}, 第 {attempt} 次")
            client.retrieve("reanalysis-era5-single-levels-timeseries", request, str(path))
            return
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"{year} 下载失败，准备重试: {exc}")
            time.sleep(30 * attempt)


def extract_zip(path: Path) -> Path:
    if not zipfile.is_zipfile(path):
        return path
    out_dir = path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        nc_names = [name for name in zf.namelist() if name.lower().endswith(".nc")]
        if len(nc_names) != 1:
            raise ValueError(f"{path} 中 NetCDF 数量异常: {nc_names}")
        nc_path = out_dir / Path(nc_names[0]).name
        if not nc_path.exists():
            zf.extract(nc_names[0], out_dir)
            extracted = out_dir / nc_names[0]
            if extracted != nc_path:
                extracted.replace(nc_path)
        return nc_path


def aggregate_daily(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        ds = xr.open_dataset(path, engine="h5netcdf")
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        frame = ds.to_dataframe().reset_index()
        ds.close()
        frame = frame.rename(columns={time_name: "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["date"] = frame["datetime"].dt.floor("D")
        rename = {}
        if "ssrd" in frame.columns:
            rename["ssrd"] = "ssrd_j_m2"
        if "sst" in frame.columns:
            rename["sst"] = "sst_k"
        frame = frame.rename(columns=rename)
        if "ssrd_j_m2" not in frame.columns or "sst_k" not in frame.columns:
            raise ValueError(f"{path} 变量不足，当前列: {frame.columns.tolist()}")
        daily = (
            frame.groupby("date")
            .agg(
                ssrd_sum_j_m2=("ssrd_j_m2", "sum"),
                ssrd_mean_j_m2=("ssrd_j_m2", "mean"),
                sst_mean_c=("sst_k", lambda s: float(np.nanmean(s) - 273.15)),
                sst_max_c=("sst_k", lambda s: float(np.nanmax(s) - 273.15)),
                sst_min_c=("sst_k", lambda s: float(np.nanmin(s) - 273.15)),
            )
            .reset_index()
        )
        frames.append(daily)
    daily_all = pd.concat(frames, ignore_index=True).sort_values("date")
    daily_all["doy"] = daily_all["date"].dt.dayofyear
    clim = daily_all.groupby("doy")["sst_mean_c"].mean()
    daily_all["sst_anom_c"] = daily_all["sst_mean_c"] - daily_all["doy"].map(clim)
    return daily_all.drop(columns=["doy"])


def window_stat(daily: pd.DataFrame, sample_date: pd.Timestamp, col: str, days: int, func: str) -> float:
    start = sample_date - pd.Timedelta(days=days)
    end = sample_date - pd.Timedelta(days=1)
    sub = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    if sub.empty:
        return np.nan
    if func == "sum":
        return float(sub[col].sum())
    if func == "mean":
        return float(sub[col].mean())
    if func == "max":
        return float(sub[col].max())
    raise ValueError(func)


def build_windows(daily: pd.DataFrame) -> pd.DataFrame:
    water = pd.read_csv(WATER, parse_dates=["date"])
    samples = water[["date"]].drop_duplicates().sort_values("date")
    rows = []
    for date in samples["date"]:
        row = {"date": date}
        for days in [3, 7, 14]:
            row[f"ssrd_sum_{days}d"] = window_stat(daily, date, "ssrd_sum_j_m2", days, "sum")
            row[f"ssrd_mean_{days}d"] = window_stat(daily, date, "ssrd_mean_j_m2", days, "mean")
            row[f"sst_mean_{days}d"] = window_stat(daily, date, "sst_mean_c", days, "mean")
            row[f"sst_max_{days}d"] = window_stat(daily, date, "sst_max_c", days, "max")
            row[f"sst_anom_{days}d"] = window_stat(daily, date, "sst_anom_c", days, "mean")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    setup_cds_env()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in range(args.start_year, args.end_year + 1):
        path = args.raw_dir / f"era5_ssrd_sst_hk_water_{year}.zip"
        paths.append(path)
        if not args.skip_download:
            download_year(year, path, args.longitude, args.latitude)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 ERA5 ssrd/sst 文件: {missing}")
    nc_paths = [extract_zip(path) for path in paths]
    daily = aggregate_daily(nc_paths)
    features = build_windows(daily)
    args.daily_out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.daily_out, index=False, encoding="utf-8-sig")
    features.to_csv(args.feature_out, index=False, encoding="utf-8-sig")
    print(f"写出 {args.daily_out}: {daily.shape}")
    print(f"写出 {args.feature_out}: {features.shape}")
    print(features.head().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
