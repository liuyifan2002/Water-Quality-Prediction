from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WATER = Path("data/processed/water_quality_master_2014_2024.csv")
DEFAULT_SATELLITE = Path("data/processed/station_date_satellite_features_2014_2024.csv")
DEFAULT_OUT = Path("data/processed/model_table_3depth_2014_2024.csv")

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


def merge_model_table(water: pd.DataFrame, sat: pd.DataFrame) -> pd.DataFrame:
    compare_overlap_columns(water, sat)

    sat_feature_cols = [c for c in sat.columns if c not in {"sample_id", *OVERLAP_COLUMNS}]
    merged = water.merge(
        sat[["sample_id", *sat_feature_cols]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )

    merged = add_time_features(merged)

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

    validate_required(water, {"obs_id", "sample_id", "station", "date", "depth"}, "Water quality master table")
    validate_required(sat, {"sample_id", "station", "date"}, "Satellite feature table")

    model_table = merge_model_table(water, sat)
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
