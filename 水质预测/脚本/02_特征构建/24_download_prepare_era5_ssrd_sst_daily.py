from __future__ import annotations

import argparse
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RAW_DIR = Path("数据/原始数据/era5_ssrd_sst_offshore_v2")
OUT = Path("数据/处理后数据/衍生版本/era5_ssrd_sst_daily_2014_2024.csv")
FEATURE_OUT = Path("数据/处理后数据/衍生版本/era5_ssrd_sst_window_features_2014_2024.csv")
WATER = Path("数据/处理后数据/衍生版本/water_quality_surface_2014_2024.csv")

VARIABLES = [
    "surface_solar_radiation_downwards",
    "sea_surface_temperature",
]

OFFSHORE_POINTS = [
    {"label": "west_offshore", "longitude": 113.75, "latitude": 22.00},
    {"label": "south_offshore", "longitude": 114.00, "latitude": 22.00},
    {"label": "east_offshore", "longitude": 114.25, "latitude": 22.00},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare daily ERA5 ssrd/sst series and sample-date window features for surface Chl-a modeling."
    )
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--daily-out", type=Path, default=OUT)
    parser.add_argument("--feature-out", type=Path, default=FEATURE_OUT)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def setup_cds_env() -> None:
    cds = Path("待确认_未归类/.cdsapirc")
    if cds.exists():
        lines = cds.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("url:"):
                os.environ.setdefault("CDSAPI_URL", line.split(":", 1)[1].strip())
            if line.startswith("key:"):
                os.environ.setdefault("CDSAPI_KEY", line.split(":", 1)[1].strip())


def download_year_point(year: int, path: Path, point: dict[str, float | str]) -> None:
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit(
            "Missing cdsapi. Install it in the active Python environment before downloading ERA5 ssrd/sst."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Already exists, skip download: {path}")
        return

    client_kwargs = {}
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        client_kwargs = {"url": os.environ["CDSAPI_URL"], "key": os.environ["CDSAPI_KEY"]}

    client = cdsapi.Client(**client_kwargs)
    request = {
        "variable": VARIABLES,
        "location": {"longitude": point["longitude"], "latitude": point["latitude"]},
        "date": [f"{year}-01-01/{year}-12-31"],
        "data_format": "netcdf",
    }
    for attempt in range(1, 4):
        try:
            print(f"Downloading ERA5 ssrd/sst for {year} at {point['label']}, attempt {attempt}")
            client.retrieve("reanalysis-era5-single-levels-timeseries", request, str(path))
            return
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"Download failed for {year} at {point['label']}, retrying: {exc}")
            time.sleep(30 * attempt)


def extract_zip(path: Path) -> Path:
    if not zipfile.is_zipfile(path):
        return path

    out_dir = path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        nc_names = [name for name in zf.namelist() if name.lower().endswith(".nc")]
        if len(nc_names) != 1:
            raise ValueError(f"Unexpected NetCDF payload in {path}: {nc_names}")
        nc_path = out_dir / Path(nc_names[0]).name
        if not nc_path.exists():
            zf.extract(nc_names[0], out_dir)
            extracted = out_dir / nc_names[0]
            if extracted != nc_path:
                extracted.replace(nc_path)
        return nc_path


def resolve_existing_year_path(raw_dir: Path, year: int, label: str) -> Path:
    direct_candidates = [
        raw_dir / f"era5_ssrd_sst_{label}_{year}.zip",
        raw_dir / f"era5_ssrd_sst_{label}_{year}.nc",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    extracted_candidates = sorted(raw_dir.glob(f"*{label}*{year}*/**/*.nc"))
    if extracted_candidates:
        return extracted_candidates[0]

    raise FileNotFoundError(f"No ERA5 raw file found for {year} at {label} under {raw_dir}")


def safe_nanmean(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


def safe_nanmax(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmax(arr))


def aggregate_daily(paths_by_point: list[tuple[str, Path]]) -> pd.DataFrame:
    point_frames = []
    for label, path in paths_by_point:
        ds = xr.open_dataset(path, engine="h5netcdf")
        required_vars = {"ssrd", "sst"}
        missing = required_vars.difference(ds.data_vars)
        if missing:
            available = sorted(ds.data_vars)
            ds.close()
            raise ValueError(
                f"{path} does not contain required ERA5 variables {sorted(missing)}. "
                f"Available variables: {available}. "
                "The current raw files look like the old t2m/u10/v10/tcc package instead of ssrd/sst."
            )

        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        frame = ds.to_dataframe().reset_index()
        ds.close()

        frame = frame.rename(columns={time_name: "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["date"] = frame["datetime"].dt.floor("D")
        frame = frame.rename(columns={"ssrd": "ssrd_j_m2", "sst": "sst_k"})

        daily = (
            frame.groupby("date")
            .agg(
                ssrd_sum_j_m2=("ssrd_j_m2", "sum"),
                ssrd_mean_j_m2=("ssrd_j_m2", "mean"),
                sst_mean_c=("sst_k", lambda s: safe_nanmean(s) - 273.15 if not np.isnan(safe_nanmean(s)) else np.nan),
                sst_max_c=("sst_k", lambda s: safe_nanmax(s) - 273.15 if not np.isnan(safe_nanmax(s)) else np.nan),
                sst_min_c=("sst_k", lambda s: float(np.nanmin(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))) - 273.15 if not np.isnan(safe_nanmean(s)) else np.nan),
            )
            .reset_index()
        )
        daily["point_label"] = label
        point_frames.append(daily)

    point_daily = pd.concat(point_frames, ignore_index=True).sort_values(["date", "point_label"])
    daily_all = (
        point_daily.groupby("date", as_index=False)
        .agg(
            ssrd_sum_j_m2=("ssrd_sum_j_m2", "mean"),
            ssrd_mean_j_m2=("ssrd_mean_j_m2", "mean"),
            sst_mean_c=("sst_mean_c", safe_nanmean),
            sst_max_c=("sst_max_c", safe_nanmax),
            sst_min_c=("sst_min_c", lambda s: float(np.nanmin(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))) if not np.isnan(safe_nanmean(s)) else np.nan),
            sst_valid_point_count=("sst_mean_c", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            ssrd_valid_point_count=("ssrd_mean_j_m2", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
        .sort_values("date")
    )
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

    paths_by_point: list[tuple[str, Path]] = []
    for year in range(args.start_year, args.end_year + 1):
        for point in OFFSHORE_POINTS:
            label = str(point["label"])
            download_path = args.raw_dir / f"era5_ssrd_sst_{label}_{year}.zip"
            if not args.skip_download:
                download_year_point(year, download_path, point)
            paths_by_point.append((label, resolve_existing_year_path(args.raw_dir, year, label)))

    missing = [p for _, p in paths_by_point if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ERA5 raw files: {missing}")

    nc_paths = [(label, extract_zip(path)) for label, path in paths_by_point]
    daily = aggregate_daily(nc_paths)
    features = build_windows(daily)

    args.daily_out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.daily_out, index=False, encoding="utf-8-sig")
    features.to_csv(args.feature_out, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.daily_out}: {daily.shape}")
    print(f"Wrote {args.feature_out}: {features.shape}")
    print(features.head().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
