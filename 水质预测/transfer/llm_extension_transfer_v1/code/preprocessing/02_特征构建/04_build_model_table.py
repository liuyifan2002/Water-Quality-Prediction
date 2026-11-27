from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WATER = Path("数据/处理后数据/正式主表/water_quality_master_2014_2024.csv")
DEFAULT_SATELLITE = Path("数据/处理后数据/正式主表/station_date_satellite_features_2014_2024.csv")
DEFAULT_ERA5 = Path("数据/处理后数据/正式主表/era5_monthly_2014_2024.csv")
DEFAULT_ERA5_DAILY = Path("数据/处理后数据/衍生版本/era5_ssrd_sst_window_features_2014_2024.csv")
DEFAULT_RAIN = Path("数据/处理后数据/正式主表/rainfall_features_2014_2024.csv")
DEFAULT_TIDE = Path("数据/处理后数据/正式主表/tide_features_2014_2024.csv")
DEFAULT_NTL = Path("数据/处理后数据/正式主表/viirs_ntl_station_monthly_2014_2024.csv")
DEFAULT_OUT = Path("数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv")

TARGET_HISTORY_COLS = [
    "chla_ug_l",
    "secchi_depth_m",
    "tn_mg_l",
    "tp_mg_l",
]

TARGET_HISTORY_STEPS = [1, 2, 3, 7]
TARGET_ROLL_WINDOWS = [3, 5]

ENV_CURRENT_COLS = [
    "t2m",
    "u10",
    "v10",
    "tcc",
    "rain_3d",
    "rain_14d",
    "rain_month",
    "tidal_range",
    "ntl_avg_rad",
    "viirs_count",
]

ERA5_WINDOW_COLS = [
    "ssrd_sum_3d",
    "ssrd_mean_3d",
    "ssrd_sum_7d",
    "ssrd_mean_7d",
    "ssrd_sum_14d",
    "ssrd_mean_14d",
    "sst_mean_3d",
    "sst_max_3d",
    "sst_anom_3d",
    "sst_mean_7d",
    "sst_max_7d",
    "sst_anom_7d",
    "sst_mean_14d",
    "sst_max_14d",
    "sst_anom_14d",
]

LANDSAT_SIGNAL_COLS = [
    "landsat_blue",
    "landsat_green",
    "landsat_red",
    "landsat_nir",
    "landsat_swir1",
    "landsat_swir2",
    "landsat_ndvi",
    "landsat_ndwi",
    "landsat_mndwi",
    "landsat_ndti",
]

S2_SIGNAL_COLS = [
    "s2_b2_mean",
    "s2_b3_mean",
    "s2_b4_mean",
    "s2_b5_mean",
    "s2_b6_mean",
    "s2_b8_mean",
    "s2_b8a_mean",
    "s2_b11_mean",
    "s2_b12_mean",
    "s2_ndwi_mean",
    "s2_mndwi_mean",
    "s2_ndci_mean",
    "s2_ndti_mean",
]

ENV_LAG_BASE_COLS = [
    "t2m",
    "tcc",
    "wind_speed",
    "rain_3d",
    "rain_14d",
    "rain_month",
    "tidal_range",
    "ntl_avg_rad",
]

ENV_LAG_STEPS = [1, 2, 3]
ENV_ROLL_WINDOWS = [3, 5]

OVERLAP_COLUMNS = [
    "station",
    "date",
    "year",
    "month",
    "water_zone_en",
    "station_name_en",
    "lon",
    "lat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the three-depth model table by merging water quality and satellite features on sample_id."
    )
    parser.add_argument("--water", type=Path, default=DEFAULT_WATER)
    parser.add_argument("--satellite", type=Path, default=DEFAULT_SATELLITE)
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--era5-daily-window", type=Path, default=DEFAULT_ERA5_DAILY)
    parser.add_argument("--rain", type=Path, default=DEFAULT_RAIN)
    parser.add_argument("--tide", type=Path, default=DEFAULT_TIDE)
    parser.add_argument("--ntl", type=Path, default=DEFAULT_NTL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path, parse_dates=parse_dates, low_memory=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.replace("\ufeff", "") for c in df.columns})


def normalize_bool_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    candidate_suffixes = ("_available", "_missing", "_imputed", "_no_data", "_strict_valid")
    for col in out.columns:
        if not col.endswith(candidate_suffixes):
            continue
        lowered = out[col].dropna().astype(str).str.strip().str.lower()
        if lowered.empty:
            continue
        if set(lowered.unique()).issubset({"true", "false", "0", "1"}):
            out[col] = (
                out[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .astype("boolean")
            )
    return out


def read_optional_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Skip optional feature file: {path}")
        return None
    return pd.read_csv(path, parse_dates=parse_dates, low_memory=False)


def validate_required(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def compare_overlap_columns(water: pd.DataFrame, sat: pd.DataFrame) -> None:
    merged = water[["sample_id", *OVERLAP_COLUMNS]].merge(
        sat[["sample_id", *[c for c in OVERLAP_COLUMNS if c in sat.columns]]],
        on="sample_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_sat"),
    )

    mismatches: list[str] = []
    for col in OVERLAP_COLUMNS:
        sat_col = f"{col}_sat"
        if sat_col not in merged.columns:
            continue

        left = merged[col]
        right = merged[sat_col]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            same = pd.Series(
                np.isclose(left.astype(float), right.astype(float), equal_nan=True),
                index=merged.index,
            )
        else:
            same = (left == right) | (left.isna() & right.isna())
        bad = merged[right.notna() & ~same]
        if not bad.empty:
            preview = bad[["sample_id", col, sat_col]].head(10).to_string(index=False)
            mismatches.append(f"Column '{col}' mismatch examples:\n{preview}")

    if mismatches:
        raise ValueError("Satellite/base overlap columns do not match.\n" + "\n\n".join(mismatches))


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["is_wet_season"] = out["month"].isin([4, 5, 6, 7, 8, 9]).astype("int8")
    out["month_sin"] = out["month"].map(lambda m: math.sin(2 * math.pi * (m - 1) / 12))
    out["month_cos"] = out["month"].map(lambda m: math.cos(2 * math.pi * (m - 1) / 12))
    out["depth_is_surface"] = out["depth"].eq("surface").astype("int8")
    out["depth_is_middle"] = out["depth"].eq("middle").astype("int8")
    out["depth_is_bottom"] = out["depth"].eq("bottom").astype("int8")
    out["year_offset"] = out["year"] - int(out["year"].min())
    return out


def merge_optional_features(
    df: pd.DataFrame,
    era5: pd.DataFrame | None,
    era5_daily: pd.DataFrame | None,
    rain: pd.DataFrame | None,
    tide: pd.DataFrame | None,
    ntl: pd.DataFrame | None,
) -> pd.DataFrame:
    out = df.copy()

    if era5 is not None:
        era5 = normalize_columns(era5)
        validate_required(era5, {"year", "month"}, "ERA5 feature table")
        out = out.merge(era5, on=["year", "month"], how="left", validate="many_to_one")

    if era5_daily is not None:
        era5_daily = normalize_columns(era5_daily)
        validate_required(era5_daily, {"date"}, "ERA5 daily-window feature table")
        era5_daily["date"] = pd.to_datetime(era5_daily["date"], errors="raise")
        out = out.merge(era5_daily, on="date", how="left", validate="many_to_one")

    if rain is not None:
        rain = normalize_columns(rain)
        validate_required(rain, {"date"}, "Rainfall feature table")
        rain["date"] = pd.to_datetime(rain["date"], errors="raise")
        out = out.merge(rain, on="date", how="left", validate="many_to_one")

    if tide is not None:
        tide = normalize_columns(tide)
        validate_required(tide, {"date"}, "Tide feature table")
        tide["date"] = pd.to_datetime(tide["date"], errors="raise")
        out = out.merge(tide, on="date", how="left", validate="many_to_one")

    if ntl is not None:
        ntl = normalize_columns(ntl)
        validate_required(ntl, {"station", "year", "month"}, "VIIRS NTL feature table")
        out = out.merge(ntl, on=["station", "year", "month"], how="left", validate="many_to_one")

    return out


def add_remote_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "landsat_image_count",
        "landsat_clear_pixel_count",
        "landsat_strict_valid_gee",
        "s2_image_count",
        "s2_valid_pixel_count",
        "s2_strict_valid",
        *LANDSAT_SIGNAL_COLS,
        *S2_SIGNAL_COLS,
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    landsat_hq_mask = pd.Series(False, index=out.index)
    if {"landsat_strict_valid_gee", "landsat_clear_pixel_count"}.issubset(out.columns):
        landsat_hq_mask = (
            out["landsat_strict_valid_gee"].fillna(0).ge(1)
            & out["landsat_clear_pixel_count"].fillna(0).ge(40)
        )
        out["landsat_hq"] = landsat_hq_mask.astype("int8")
        for col in LANDSAT_SIGNAL_COLS:
            if col in out.columns:
                out[col.replace("landsat_", "landsat_hq_")] = out[col].where(landsat_hq_mask)

    s2_hq_mask = pd.Series(False, index=out.index)
    if {"s2_strict_valid", "s2_valid_pixel_count"}.issubset(out.columns):
        s2_hq_mask = (
            out["s2_strict_valid"].fillna(0).ge(1)
            & out["s2_valid_pixel_count"].fillna(0).ge(8)
        )
        out["s2_hq"] = s2_hq_mask.astype("int8")
        for col in S2_SIGNAL_COLS:
            if col in out.columns:
                out[col.replace("s2_", "s2_hq_")] = out[col].where(s2_hq_mask)

    out["hq_satellite_available_count"] = (
        landsat_hq_mask.astype("int8") + s2_hq_mask.astype("int8")
    ).astype("int8")

    landsat_hq_cols = [col.replace("landsat_", "landsat_hq_") for col in LANDSAT_SIGNAL_COLS if col.replace("landsat_", "landsat_hq_") in out.columns]
    if landsat_hq_cols:
        sample_level = (
            out[["sample_id", "station", "date", *landsat_hq_cols]]
            .drop_duplicates(subset=["sample_id"])
            .sort_values(["station", "date"])
            .reset_index(drop=True)
        )
        for col in landsat_hq_cols:
            sample_level[f"{col}_filled"] = (
                sample_level.groupby("station", sort=False)[col]
                .transform(lambda s: s.interpolate(method="linear", limit=2, limit_direction="both"))
            )
            station_median = sample_level.groupby("station", sort=False)[f"{col}_filled"].transform("median")
            sample_level[f"{col}_filled"] = sample_level[f"{col}_filled"].fillna(station_median)

        filled_cols = [f"{col}_filled" for col in landsat_hq_cols]
        sample_level["landsat_hq_signal_count"] = sample_level[landsat_hq_cols].notna().sum(axis=1).astype("int16")
        sample_level["landsat_hq_filled_signal_count"] = sample_level[filled_cols].notna().sum(axis=1).astype("int16")
        sample_level["landsat_hq_fill_flag"] = (
            sample_level["landsat_hq_signal_count"].lt(len(landsat_hq_cols))
            & sample_level["landsat_hq_filled_signal_count"].gt(0)
        ).astype("int8")

        out = out.merge(
            sample_level[["sample_id", *filled_cols, "landsat_hq_signal_count", "landsat_hq_filled_signal_count", "landsat_hq_fill_flag"]],
            on="sample_id",
            how="left",
            validate="many_to_one",
        )
    return out


def add_environment_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ENV_CURRENT_COLS + ERA5_WINDOW_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if {"u10", "v10"}.issubset(out.columns):
        out["wind_speed"] = np.sqrt(out["u10"] ** 2 + out["v10"] ** 2)
        out["wind_speed_sq"] = out["wind_speed"] ** 2
        denom = out["wind_speed"].replace(0, np.nan)
        out["wind_dir_sin"] = out["v10"] / denom
        out["wind_dir_cos"] = out["u10"] / denom

    if {"t2m", "rain_14d"}.issubset(out.columns):
        out["t2m_x_rain14d"] = out["t2m"] * out["rain_14d"]
    if {"wind_speed", "rain_14d"}.issubset(out.columns):
        out["wind_speed_x_rain14d"] = out["wind_speed"] * out["rain_14d"]
    if {"rain_14d", "tidal_range"}.issubset(out.columns):
        out["rain14d_x_tidal_range"] = out["rain_14d"] * out["tidal_range"]
    if {"t2m", "tidal_range"}.issubset(out.columns):
        out["t2m_x_tidal_range"] = out["t2m"] * out["tidal_range"]
    if {"sst_mean_7d", "ssrd_mean_7d"}.issubset(out.columns):
        out["sst_mean_7d_x_ssrd_mean_7d"] = out["sst_mean_7d"] * out["ssrd_mean_7d"]
    if {"sst_anom_7d", "ssrd_mean_7d"}.issubset(out.columns):
        out["sst_anom_7d_x_ssrd_mean_7d"] = out["sst_anom_7d"] * out["ssrd_mean_7d"]
    if {"rain_14d", "ssrd_mean_7d"}.issubset(out.columns):
        out["rain14d_x_ssrd_mean_7d"] = out["rain_14d"] * out["ssrd_mean_7d"]
    if {"wind_speed", "tidal_range"}.issubset(out.columns):
        out["wind_speed_x_tidal_range"] = out["wind_speed"] * out["tidal_range"]
    if {"wind_dir_sin", "tidal_range"}.issubset(out.columns):
        out["wind_dir_sin_x_tidal_range"] = out["wind_dir_sin"] * out["tidal_range"]
    if {"wind_dir_cos", "tidal_range"}.issubset(out.columns):
        out["wind_dir_cos_x_tidal_range"] = out["wind_dir_cos"] * out["tidal_range"]
    if {"wind_speed", "sst_mean_3d"}.issubset(out.columns):
        out["wind_speed_x_sst_mean_3d"] = out["wind_speed"] * out["sst_mean_3d"]
    if {"wind_speed", "ssrd_mean_3d"}.issubset(out.columns):
        out["wind_speed_x_ssrd_mean_3d"] = out["wind_speed"] * out["ssrd_mean_3d"]

    sample_level_cols = ["sample_id", "station", "date"]
    available_env_cols = [c for c in ENV_LAG_BASE_COLS if c in out.columns]
    if not available_env_cols:
        return out

    sample_level = (
        out[sample_level_cols + available_env_cols]
        .drop_duplicates(subset=["sample_id"])
        .sort_values(["station", "date"])
        .reset_index(drop=True)
    )

    new_cols: list[str] = []
    for col in available_env_cols:
        grouped = sample_level.groupby("station", sort=False)[col]
        derived: dict[str, pd.Series] = {}
        for step in ENV_LAG_STEPS:
            derived[f"_lag{step}"] = grouped.shift(step)
        for window in ENV_ROLL_WINDOWS:
            derived[f"_roll{window}"] = grouped.transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        for suffix, series in derived.items():
            new_col = f"{col}{suffix}"
            sample_level[new_col] = series
            new_cols.append(new_col)
        delta_col = f"{col}_delta1"
        sample_level[delta_col] = sample_level[f"{col}_lag1"] - sample_level[f"{col}_lag2"]
        new_cols.append(delta_col)
        delta7_col = f"{col}_delta7"
        sample_level[delta7_col] = sample_level[f"{col}_lag1"] - grouped.shift(7)
        new_cols.append(delta7_col)

    out = out.merge(sample_level[["sample_id", *new_cols]], on="sample_id", how="left", validate="many_to_one")
    return out


def add_target_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["station", "depth", "date", "sample_no"]).copy()
    group_keys = ["station", "depth"]

    prev_date = out.groupby(group_keys, sort=False)["date"].shift(1)
    out["days_since_prev_sample"] = (out["date"] - prev_date).dt.days.astype("float64")

    for col in TARGET_HISTORY_COLS:
        if col not in out.columns:
            continue
        grouped = out.groupby(group_keys, sort=False)[col]
        for step in TARGET_HISTORY_STEPS:
            out[f"{col}_lag{step}"] = grouped.shift(step)
        for window in TARGET_ROLL_WINDOWS:
            out[f"{col}_roll{window}"] = grouped.transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        out[f"{col}_delta1"] = out[f"{col}_lag1"] - out[f"{col}_lag2"]
        out[f"{col}_delta7"] = out[f"{col}_lag1"] - out[f"{col}_lag7"]

    return out


def merge_model_table(
    water: pd.DataFrame,
    sat: pd.DataFrame,
    era5: pd.DataFrame | None,
    era5_daily: pd.DataFrame | None,
    rain: pd.DataFrame | None,
    tide: pd.DataFrame | None,
    ntl: pd.DataFrame | None,
) -> pd.DataFrame:
    compare_overlap_columns(water, sat)

    sat_feature_cols = [c for c in sat.columns if c not in {"sample_id", *OVERLAP_COLUMNS}]
    merged = water.merge(
        sat[["sample_id", *sat_feature_cols]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )

    merged = add_time_features(merged)
    merged = merge_optional_features(merged, era5=era5, era5_daily=era5_daily, rain=rain, tide=tide, ntl=ntl)
    merged = add_remote_quality_features(merged)
    merged = add_environment_features(merged)
    merged = add_target_history_features(merged)

    availability_cols = [c for c in merged.columns if c.endswith("_available")]
    if availability_cols:
        availability_frame = merged[availability_cols].astype("boolean")
        merged["satellite_any_available"] = availability_frame.fillna(False).any(axis=1).astype("int8")
        merged["satellite_available_count"] = availability_frame.fillna(False).sum(axis=1).astype("int16")

    ordered_front = [
        "obs_id",
        "sample_id",
        "station",
        "station_name_en",
        "water_zone_en",
        "lon",
        "lat",
        "date",
        "year",
        "month",
        "quarter",
        "day",
        "day_of_year",
        "year_offset",
        "month_sin",
        "month_cos",
        "is_wet_season",
        "sample_no",
        "days_since_prev_sample",
        "depth_raw",
        "depth",
        "depth_order",
        "depth_is_surface",
        "depth_is_middle",
        "depth_is_bottom",
    ]

    existing_front = [c for c in ordered_front if c in merged.columns]
    remaining = [c for c in merged.columns if c not in existing_front]
    return merged[existing_front + remaining].sort_values(["station", "date", "depth_order"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    water = normalize_columns(read_csv(args.water, parse_dates=["date"]))
    sat = normalize_bool_like_columns(normalize_columns(read_csv(args.satellite, parse_dates=["date"])))
    era5 = read_optional_csv(args.era5)
    era5_daily = read_optional_csv(args.era5_daily_window, parse_dates=["date"])
    rain = read_optional_csv(args.rain, parse_dates=["date"])
    tide = read_optional_csv(args.tide, parse_dates=["date"])
    ntl = read_optional_csv(args.ntl)

    validate_required(water, {"obs_id", "sample_id", "station", "date", "depth"}, "Water quality master table")
    validate_required(sat, {"sample_id", "station", "date"}, "Satellite feature table")

    model_table = merge_model_table(water, sat, era5=era5, era5_daily=era5_daily, rain=rain, tide=tide, ntl=ntl)
    model_table.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.out} ({len(model_table)} rows, {model_table.shape[1]} columns)")
    print(f"Stations: {model_table['station'].nunique()}, sample_ids: {model_table['sample_id'].nunique()}, obs_ids: {model_table['obs_id'].nunique()}")
    availability_cols = [c for c in model_table.columns if c.endswith('_available')]
    if availability_cols:
        print("Satellite availability (% of rows):")
        availability_frame = model_table[availability_cols].astype("boolean")
        print((availability_frame.fillna(False).mean() * 100).round(1).to_string())
    print("Top missing-rate columns (%):")
    print((model_table.isna().mean().sort_values(ascending=False).head(20) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
