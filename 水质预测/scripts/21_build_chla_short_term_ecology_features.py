from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ERA5_DIR = Path("data/raw/era5")
RAIN_RAW = Path("data/raw/daily_HKO_RF_ALL.csv")
DATASET = Path("data/processed/model_dataset_surface_2014_2024_enhanced.csv")
SHORT_OUT = Path("data/processed/chla_short_term_ecology_features_2014_2024.csv")
MERGED_OUT = Path("data/processed/model_dataset_surface_2014_2024_chla_ecology.csv")
REPORT_OUT = Path("reports/chla_optimization/chla_short_term_feature_report.csv")


def read_era5_hourly() -> pd.DataFrame:
    frames = []
    for path in sorted(ERA5_DIR.glob("era5_hk_timeseries_*/*.nc")):
        ds = xr.open_dataset(path, engine="h5netcdf")
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        frame = ds[["t2m", "u10", "v10", "tcc"]].to_dataframe().reset_index()
        ds.close()
        frame = frame.rename(columns={time_name: "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["date"] = frame["datetime"].dt.floor("D")
        frame["t2m_c"] = frame["t2m"] - 273.15
        frame["wind_speed_hourly"] = np.sqrt(frame["u10"] ** 2 + frame["v10"] ** 2)
        frame["solar_proxy"] = (1 - frame["tcc"]).clip(0, 1)
        frames.append(frame[["datetime", "date", "t2m_c", "wind_speed_hourly", "tcc", "solar_proxy"]])
    hourly = pd.concat(frames, ignore_index=True).sort_values("datetime")
    return hourly


def daily_era5(hourly: pd.DataFrame) -> pd.DataFrame:
    daily = (
        hourly.groupby("date")
        .agg(
            t2m_day_mean=("t2m_c", "mean"),
            t2m_day_max=("t2m_c", "max"),
            t2m_day_min=("t2m_c", "min"),
            wind_day_mean=("wind_speed_hourly", "mean"),
            wind_day_max=("wind_speed_hourly", "max"),
            low_wind_hours=("wind_speed_hourly", lambda s: int((s < 2.0).sum())),
            tcc_day_mean=("tcc", "mean"),
            solar_proxy_day_mean=("solar_proxy", "mean"),
        )
        .reset_index()
    )
    daily["low_wind_day"] = ((daily["wind_day_mean"] < 2.0) | (daily["low_wind_hours"] >= 18)).astype(int)
    daily["sunny_proxy_day"] = (daily["solar_proxy_day_mean"] >= 0.65).astype(int)

    clim = daily.copy()
    clim["doy"] = clim["date"].dt.dayofyear
    doy_mean = clim.groupby("doy")["t2m_day_mean"].mean()
    daily["doy"] = daily["date"].dt.dayofyear
    daily["t2m_anom"] = daily["t2m_day_mean"] - daily["doy"].map(doy_mean)
    return daily.drop(columns=["doy"])


def read_rainfall() -> pd.DataFrame:
    rain = pd.read_csv(RAIN_RAW, skiprows=2)
    rain = rain.rename(
        columns={
            "年/Year": "year",
            "月/Month": "month",
            "日/Day": "day",
            "數值/Value": "rain_mm",
            "數據完整性/data Completeness": "rain_completeness",
        }
    )
    for col in ["year", "month", "day"]:
        rain[col] = pd.to_numeric(rain[col], errors="coerce")
    rain = rain.dropna(subset=["year", "month", "day"]).copy()
    rain[["year", "month", "day"]] = rain[["year", "month", "day"]].astype(int)
    date_text = (
        rain["year"].astype(str).str.zfill(4)
        + "-"
        + rain["month"].astype(str).str.zfill(2)
        + "-"
        + rain["day"].astype(str).str.zfill(2)
    )
    rain["date"] = pd.to_datetime(date_text, errors="coerce")
    rain = rain.dropna(subset=["date"]).copy()
    rain["rain_mm"] = pd.to_numeric(rain["rain_mm"], errors="coerce").fillna(0)
    rain = rain[(rain["date"].dt.year >= 2014) & (rain["date"].dt.year <= 2024)].copy()
    return rain[["date", "rain_mm", "rain_completeness"]]


def window_sum(daily: pd.DataFrame, sample_date: pd.Timestamp, col: str, days: int) -> float:
    start = sample_date - pd.Timedelta(days=days)
    end = sample_date - pd.Timedelta(days=1)
    sub = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    return float(sub[col].sum()) if len(sub) else np.nan


def window_mean(daily: pd.DataFrame, sample_date: pd.Timestamp, col: str, days: int) -> float:
    start = sample_date - pd.Timedelta(days=days)
    end = sample_date - pd.Timedelta(days=1)
    sub = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    return float(sub[col].mean()) if len(sub) else np.nan


def window_max(daily: pd.DataFrame, sample_date: pd.Timestamp, col: str, days: int) -> float:
    start = sample_date - pd.Timedelta(days=days)
    end = sample_date - pd.Timedelta(days=1)
    sub = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    return float(sub[col].max()) if len(sub) else np.nan


def consecutive_low_wind(daily: pd.DataFrame, sample_date: pd.Timestamp) -> int:
    day = sample_date - pd.Timedelta(days=1)
    count = 0
    lookup = daily.set_index("date")["low_wind_day"]
    while day in lookup.index and int(lookup.loc[day]) == 1:
        count += 1
        day -= pd.Timedelta(days=1)
    return count


def build_features() -> pd.DataFrame:
    water = pd.read_csv(DATASET, parse_dates=["date"])
    samples = water[["date"]].drop_duplicates().sort_values("date")

    era_daily = daily_era5(read_era5_hourly())
    rain = read_rainfall()
    daily = era_daily.merge(rain, on="date", how="left")
    daily["rain_mm"] = daily["rain_mm"].fillna(0)

    rows = []
    for sample_date in samples["date"]:
        row = {"date": sample_date}
        for days in [7, 14]:
            row[f"t2m_mean_{days}d"] = window_mean(daily, sample_date, "t2m_day_mean", days)
            row[f"t2m_max_{days}d"] = window_max(daily, sample_date, "t2m_day_max", days)
            row[f"t2m_anom_mean_{days}d"] = window_mean(daily, sample_date, "t2m_anom", days)
            row[f"wind_mean_{days}d"] = window_mean(daily, sample_date, "wind_day_mean", days)
            row[f"wind_max_{days}d"] = window_max(daily, sample_date, "wind_day_max", days)
            row[f"low_wind_hours_{days}d"] = window_sum(daily, sample_date, "low_wind_hours", days)
            row[f"low_wind_days_{days}d"] = window_sum(daily, sample_date, "low_wind_day", days)
            row[f"tcc_mean_{days}d"] = window_mean(daily, sample_date, "tcc_day_mean", days)
            row[f"solar_proxy_mean_{days}d"] = window_mean(daily, sample_date, "solar_proxy_day_mean", days)
            row[f"sunny_proxy_days_{days}d"] = window_sum(daily, sample_date, "sunny_proxy_day", days)
            row[f"rain_sum_{days}d"] = window_sum(daily, sample_date, "rain_mm", days)
            row[f"rain_max_{days}d"] = window_max(daily, sample_date, "rain_mm", days)
            row[f"heavy_rain_days_{days}d"] = window_sum(daily.assign(heavy=(daily["rain_mm"] >= 25).astype(int)), sample_date, "heavy", days)
        row["consecutive_low_wind_days"] = consecutive_low_wind(daily, sample_date)
        rows.append(row)
    return pd.DataFrame(rows)


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["landsat_strict_valid"] = (
        (out["valid_count"] >= 2)
        & (out["landsat_imputed"] == 0)
        & (out["landsat_no_data"] == 0)
    ).astype(int)
    out["landsat_moderate_valid"] = (
        (out["valid_count"] >= 1)
        & (out["landsat_no_data"] == 0)
    ).astype(int)
    out["landsat_low_quality"] = (
        (out["valid_count"] <= 0)
        | (out["landsat_imputed"] == 1)
        | (out["landsat_no_data"] == 1)
    ).astype(int)
    out["landsat_valid_count_log1p"] = np.log1p(out["valid_count"].clip(lower=0))
    return out


def main() -> None:
    SHORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    features = build_features()
    features.to_csv(SHORT_OUT, index=False, encoding="utf-8-sig")

    base = pd.read_csv(DATASET, parse_dates=["date"])
    merged = base.merge(features, on="date", how="left")
    merged = add_quality_flags(merged)
    merged.to_csv(MERGED_OUT, index=False, encoding="utf-8-sig")

    short_cols = [c for c in features.columns if c != "date"]
    report = pd.DataFrame(
        {
            "item": [
                "short_term_feature_count",
                "era5_source_variables",
                "solar_radiation_status",
                "water_temperature_status",
                "landsat_qc_added",
                "sentinel2_status",
            ],
            "value": [
                len(short_cols),
                "t2m,u10,v10,tcc hourly point time-series",
                "not downloaded; solar_proxy_mean uses 1 - total_cloud_cover",
                "not downloaded; t2m short windows are air-temperature proxy",
                "landsat_strict_valid, landsat_moderate_valid, landsat_low_quality, landsat_valid_count_log1p",
                "no exported Sentinel-2 table found locally; GEE script should be rerun with stricter QC",
            ],
        }
    )
    report.to_csv(REPORT_OUT, index=False, encoding="utf-8-sig")
    print(f"写出 {SHORT_OUT}: {features.shape}")
    print(f"写出 {MERGED_OUT}: {merged.shape}")
    print(f"写出 {REPORT_OUT}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
