from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_IN = Path("data/processed/landsat_station_monthly_2014_2024.csv")
DEFAULT_OUT = Path("data/processed/landsat_station_monthly_2014_2024_clean.csv")
DEFAULT_REPORT = Path("reports/landsat_quality_report.csv")

SPECTRAL_COLS = [
    "blue", "green", "red", "nir", "swir1", "swir2",
    "ndvi", "ndwi", "mndwi", "ndti"
]
BAND_COLS = ["blue", "green", "red", "nir", "swir1", "swir2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗 Landsat 月尺度特征并补齐缺失。")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    if {"nir", "red"}.issubset(df.columns):
        df["ndvi"] = safe_ratio(df["nir"] - df["red"], df["nir"] + df["red"])
    if {"green", "nir"}.issubset(df.columns):
        df["ndwi"] = safe_ratio(df["green"] - df["nir"], df["green"] + df["nir"])
    if {"green", "swir1"}.issubset(df.columns):
        df["mndwi"] = safe_ratio(df["green"] - df["swir1"], df["green"] + df["swir1"])
    if {"red", "green"}.issubset(df.columns):
        df["ndti"] = safe_ratio(df["red"] - df["green"], df["red"] + df["green"])
    return df


def interpolate_by_station(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _station, group in df.sort_values(["station", "date"]).groupby("station", sort=False):
        group = group.copy()
        group[SPECTRAL_COLS] = group[SPECTRAL_COLS].interpolate(
            method="linear",
            limit=2,
            limit_direction="both"
        )
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def fill_climatology(df: pd.DataFrame) -> pd.DataFrame:
    for col in SPECTRAL_COLS:
        station_month = df.groupby(["station", "month"])[col].transform("median")
        month = df.groupby("month")[col].transform("median")
        global_median = df[col].median()
        df[col] = df[col].fillna(station_month).fillna(month).fillna(global_median)
    return df


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df = df.rename(columns={c: c.replace("\ufeff", "") for c in df.columns})
    rename = {
        "Blue": "blue",
        "Green": "green",
        "Red": "red",
        "NIR": "nir",
        "SWIR1": "swir1",
        "SWIR2": "swir2",
        "NDVI": "ndvi",
        "NDWI": "ndwi",
        "MNDWI": "mndwi",
        "NDTI": "ndti",
        "landsat_image_count": "valid_count"
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in BAND_COLS:
        if col not in df.columns:
            raise ValueError(f"缺少 Landsat 波段列：{col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["valid_count"] = pd.to_numeric(df.get("valid_count", 0), errors="coerce").fillna(0)
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["month"] = pd.to_numeric(df["month"], errors="raise").astype(int)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=15))

    df = add_indices(df)
    df["landsat_missing"] = df[BAND_COLS].isna().any(axis=1).astype(int)
    df["landsat_no_data"] = (
        df["valid_count"].eq(0) | df.get("landsat_status", pd.Series("", index=df.index)).eq("no_data")
    ).astype(int)

    before_missing = df[SPECTRAL_COLS].isna().mean().mul(100).round(1)
    df = interpolate_by_station(df)
    after_interp_missing = df[SPECTRAL_COLS].isna().mean().mul(100).round(1)
    df = fill_climatology(df)
    after_fill_missing = df[SPECTRAL_COLS].isna().mean().mul(100).round(1)
    df["landsat_imputed"] = df["landsat_missing"]

    keep_cols = [
        "station", "year", "month",
        *SPECTRAL_COLS,
        "valid_count", "landsat_missing", "landsat_imputed", "landsat_no_data"
    ]
    out = df[keep_cols].sort_values(["station", "year", "month"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    report = pd.DataFrame(
        {
            "feature": SPECTRAL_COLS,
            "missing_before_pct": before_missing.reindex(SPECTRAL_COLS).to_numpy(),
            "missing_after_interpolation_pct": after_interp_missing.reindex(SPECTRAL_COLS).to_numpy(),
            "missing_after_fill_pct": after_fill_missing.reindex(SPECTRAL_COLS).to_numpy()
        }
    )
    report.to_csv(args.report, index=False, encoding="utf-8-sig")
    print(f"写出 {args.out}：{len(out)} 行")
    print(f"写出 {args.report}")
    print("Landsat 缺失行数：", int(out["landsat_missing"].sum()))
    print("Landsat no_data 行数：", int(out["landsat_no_data"].sum()))
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
