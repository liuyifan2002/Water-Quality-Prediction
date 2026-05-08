from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_BASE_SAMPLES = Path("data/processed/gee_station_date_samples.csv")
DEFAULT_OUT = Path("data/processed/station_date_satellite_features_2014_2024.csv")

AUTO_DISCOVERY = {
    "landsat": [
        Path("data/processed/hk_landsat_station_date_features.csv"),
        Path("data/processed/landsat_station_date_features_2014_2024.csv"),
        Path("data/processed/landsat_station_monthly_2014_2024_clean.csv"),
        Path("data/processed/landsat_station_monthly_2014_2024.csv"),
    ],
    "s2": [
        Path("data/processed/hk_s2_station_date_features.csv"),
        Path("data/processed/s2_station_date_features_2014_2024.csv"),
        Path("data/processed/hk_s2_station_monthly_2017_2024.csv"),
    ],
}

DROP_COLUMNS = {"system:index", ".geo", "geo", "geometry", "longitude", "latitude"}
BASE_COLUMNS = {"sample_id", "station", "date", "year", "month", "water_zone_en", "station_name_en", "lon", "lat"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge full-station satellite features onto the station-date sample table."
    )
    parser.add_argument("--base-samples", type=Path, default=DEFAULT_BASE_SAMPLES)
    parser.add_argument(
        "--feature-file",
        action="append",
        default=[],
        help="Feature source in the form source_name=path/to/file.csv. Can be passed multiple times.",
    )
    parser.add_argument(
        "--allow-row-order-align",
        action="store_true",
        help="Only for debugging. Allows filling missing join keys by row-order alignment against the base sample table.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def to_snake(name: str) -> str:
    name = name.replace("\ufeff", "").strip()
    name = name.replace("%", "pct")
    name = name.replace("°", "deg")
    name = re.sub(r"[^0-9A-Za-z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    name = name.replace("stddev", "std_dev")
    return name


def load_base_samples(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base sample file not found: {path}. Run scripts/01_prepare_station_samples.py first.")

    base = pd.read_csv(path, parse_dates=["date"])
    base = base.rename(columns={c: c.replace("\ufeff", "") for c in base.columns})
    base = base.rename(columns={"Station": "station"})
    required = {"sample_id", "station", "date", "year", "month"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"Base sample file is missing columns: {sorted(missing)}")

    return base.sort_values(["station", "date"]).reset_index(drop=True)


def resolve_feature_specs(raw_specs: list[str]) -> list[tuple[str, Path]]:
    resolved: list[tuple[str, Path]] = []

    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --feature-file value '{spec}'. Use source_name=path/to/file.csv")
        source, path_text = spec.split("=", 1)
        source = to_snake(source)
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        resolved.append((source, path))

    if resolved:
        return resolved

    for source, candidates in AUTO_DISCOVERY.items():
        for candidate in candidates:
            if candidate.exists():
                resolved.append((source, candidate))
                break

    if not resolved:
        raise FileNotFoundError(
            "No satellite feature files were provided or auto-discovered. "
            "Use --feature-file landsat=... and/or --feature-file s2=..."
        )

    return resolved


def convert_bool_strings(series: pd.Series) -> pd.Series:
    lowered = series.dropna().astype(str).str.strip().str.lower()
    if lowered.empty:
        return series
    if set(lowered.unique()).issubset({"true", "false"}):
        return series.astype(str).str.strip().str.lower().map({"true": True, "false": False}).astype("boolean")
    return series


def maybe_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series
    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().sum() + series.isna().sum() == len(series):
        return converted
    return series


def aggregate_duplicates(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if not df.duplicated(subset=key_cols).any():
        return df

    agg: dict[str, str] = {}
    for col in df.columns:
        if col in key_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            agg[col] = "max" if pd.api.types.is_bool_dtype(df[col]) else "mean"
        else:
            agg[col] = "first"
    return df.groupby(key_cols, as_index=False).agg(agg)


def has_usable_values(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    series = df[col]
    if series.isna().all():
        return False
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    return ~series.astype(str).str.strip().isin({"", "nan", "none", "nat"}).all()


def align_to_base_by_row_order(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if len(df) != len(base):
        raise ValueError(
            "Cannot align feature file to base sample table by row order because row counts differ: "
            f"{len(df)} vs {len(base)}"
        )

    aligned = df.reset_index(drop=True).copy()
    base_reset = base.reset_index(drop=True)

    if "year" in aligned.columns and "month" in aligned.columns:
        same_ym = aligned[["year", "month"]].reset_index(drop=True).equals(base_reset[["year", "month"]])
        if not same_ym:
            raise ValueError("Feature file row-order alignment failed because year/month sequence does not match the base sample table.")

    for col in ["sample_id", "station", "date", "year", "month", "water_zone_en", "station_name_en", "lon", "lat"]:
        aligned[col] = base_reset[col].values
    return aligned


def normalize_feature_frame(
    path: Path,
    source: str,
    base: pd.DataFrame,
    allow_row_order_align: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    rename = {col: to_snake(col) for col in df.columns}
    df = df.rename(columns=rename)
    df = df.drop(columns=[c for c in df.columns if c in DROP_COLUMNS], errors="ignore")

    alias_map = {
        "station": "station",
        "station_": "station",
        "sample_id": "sample_id",
        "month_id": "month_id",
        "dates": "date",
        "date": "date",
        "year": "year",
        "month": "month",
        "lon": "lon",
        "long": "lon",
        "longitude": "lon",
        "lat": "lat",
        "latitude": "lat",
    }
    df = df.rename(columns={c: alias_map[c] for c in df.columns if c in alias_map})

    for col in list(df.columns):
        df[col] = convert_bool_strings(df[col])
        df[col] = maybe_numeric(df[col])

    station_missing = ("station" not in df.columns) or df["station"].isna().all() or df["station"].astype(str).str.strip().eq("").all()
    if station_missing:
        if not allow_row_order_align:
            raise ValueError(
                f"{path} does not contain usable station identifiers. "
                "For formal modeling, export station/date keys from GEE and merge by explicit keys. "
                "If you only want a temporary debug run, pass --allow-row-order-align."
            )
        df = align_to_base_by_row_order(df, base)

    join_cols: list[str]
    sample_id_usable = has_usable_values(df, "sample_id")
    station_usable = has_usable_values(df, "station")
    date_usable = has_usable_values(df, "date")
    year_usable = has_usable_values(df, "year")
    month_usable = has_usable_values(df, "month")

    if sample_id_usable:
        join_cols = ["sample_id"]
    elif station_usable and date_usable:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df["sample_id"] = df["station"].astype(str) + "_" + df["date"].dt.strftime("%Y%m%d")
        join_cols = ["sample_id"]
    elif station_usable and year_usable and month_usable:
        df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
        df["month"] = pd.to_numeric(df["month"], errors="raise").astype(int)
        join_cols = ["station", "year", "month"]
    elif has_usable_values(df, "month_id") and station_usable and year_usable and month_usable:
        df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
        df["month"] = pd.to_numeric(df["month"], errors="raise").astype(int)
        join_cols = ["station", "year", "month"]
    else:
        raise ValueError(
            f"{path} does not contain a supported join key. Need sample_id, station+date, or station+year+month."
        )

    drop_meta = {"lat", "lon", "year", "month", "date", "station", "sample_id", "month_id"}
    feature_cols = [c for c in df.columns if c not in drop_meta]

    rename_features = {}
    for col in feature_cols:
        if col.startswith(f"{source}_"):
            rename_features[col] = col
        else:
            rename_features[col] = f"{source}_{col}"
    df = df.rename(columns=rename_features)
    feature_cols = [rename_features[c] for c in feature_cols]

    keep_cols = join_cols + feature_cols
    df = aggregate_duplicates(df[keep_cols], join_cols)

    available_col = f"{source}_available"
    if available_col not in df.columns:
        if f"{source}_image_count" in df.columns:
            df[available_col] = pd.to_numeric(df[f"{source}_image_count"], errors="coerce").fillna(0).gt(0)
        elif f"{source}_valid_count" in df.columns:
            df[available_col] = pd.to_numeric(df[f"{source}_valid_count"], errors="coerce").fillna(0).gt(0)
        else:
            numeric_feature_cols = [
                c
                for c in feature_cols
                if pd.api.types.is_numeric_dtype(df[c]) and not c.endswith(("_window_days", "_buffer_m"))
            ]
            if numeric_feature_cols:
                df[available_col] = df[numeric_feature_cols].notna().any(axis=1)
            else:
                df[available_col] = True

    return df, join_cols


def merge_features(
    base: pd.DataFrame,
    specs: list[tuple[str, Path]],
    allow_row_order_align: bool = False,
) -> pd.DataFrame:
    merged = base.copy()
    for source, path in specs:
        features, join_cols = normalize_feature_frame(path, source, base, allow_row_order_align=allow_row_order_align)
        merged = merged.merge(features, on=join_cols, how="left", validate="one_to_one" if join_cols == ["sample_id"] else "many_to_one")
        print(f"Merged {source}: {path} using {join_cols}, columns added={len(features.columns) - len(join_cols)}")
    return merged


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    base = load_base_samples(args.base_samples)
    specs = resolve_feature_specs(args.feature_file)
    merged = merge_features(base, specs, allow_row_order_align=args.allow_row_order_align)
    merged.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.out} ({len(merged)} rows, {merged.shape[1]} columns)")
    availability_cols = [c for c in merged.columns if c.endswith("_available")]
    if availability_cols:
        print("Availability summary:")
        print((merged[availability_cols].mean().sort_index() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
