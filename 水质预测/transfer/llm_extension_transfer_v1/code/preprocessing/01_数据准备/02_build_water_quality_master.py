from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_WATER_DIR = Path(r"C:\Users\刘逸凡\Desktop\开题\水质数据")
DEFAULT_STATION_META = Path("数据/处理后数据/正式主表/station_meta.csv")
DEFAULT_OUT = Path("数据/处理后数据/正式主表/water_quality_master_2014_2024.csv")

ID_COLUMNS = {
    "Water Control Zone": "water_zone_en_raw",
    "Station": "station",
    "Dates": "date",
    "Sample No": "sample_no",
    "Depth": "depth_raw",
}

MEASUREMENT_COLUMNS = {
    "5-day Biochemical Oxygen Demand (mg/L)": "bod5_mg_l",
    "Ammonia Nitrogen (mg/L)": "ammonia_n_mg_l",
    "Chlorophyll-a (μg/L)": "chla_ug_l",
    "Dissolved Oxygen (%saturation)": "do_pct_saturation",
    "Dissolved Oxygen (mg/L)": "do_mg_l",
    "E. coli (cfu/100mL)": "e_coli_cfu_100ml",
    "Faecal Coliforms (cfu/100mL)": "faecal_coliforms_cfu_100ml",
    "Nitrate Nitrogen (mg/L)": "nitrate_n_mg_l",
    "Nitrite Nitrogen (mg/L)": "nitrite_n_mg_l",
    "Orthophosphate Phosphorus (mg/L)": "orthophosphate_p_mg_l",
    "pH": "ph",
    "Phaeo-pigments (μg/L)": "phaeo_pigments_ug_l",
    "Salinity (psu)": "salinity_psu",
    "Secchi Disc Depth (M)": "secchi_depth_m",
    "Silica (mg/L)": "silica_mg_l",
    "Suspended Solids (mg/L)": "suspended_solids_mg_l",
    "Temperature (°C)": "temperature_c",
    "Total Inorganic Nitrogen (mg/L)": "tin_mg_l",
    "Total Kjeldahl Nitrogen (mg/L)": "tkn_mg_l",
    "Total Nitrogen (mg/L)": "tn_mg_l",
    "Total Phosphorus (mg/L)": "tp_mg_l",
    "Turbidity (NTU)": "turbidity_ntu",
    "Unionised Ammonia (mg/L)": "unionised_ammonia_mg_l",
    "Volatile Suspended Solids (mg/L)": "vss_mg_l",
}

DEPTH_MAP = {
    "Surface Water": ("surface", 1),
    "Middle Water": ("middle", 2),
    "Bottom Water": ("bottom", 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 2014-2024 full-station, full-depth water quality master table."
    )
    parser.add_argument("--water-dir", type=Path, default=DEFAULT_WATER_DIR)
    parser.add_argument("--station-meta", type=Path, default=DEFAULT_STATION_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_station_meta(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Station metadata not found: {path}. Run 脚本/01_数据准备/01_prepare_station_samples.py first.")

    meta = pd.read_csv(path)
    meta = meta.rename(columns={c: c.replace("\ufeff", "") for c in meta.columns})
    required = {"Station", "lon", "lat"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Station metadata is missing columns: {sorted(missing)}")

    meta = meta.rename(columns={"Station": "station"})
    keep = [c for c in ["station", "station_name_en", "water_zone_en", "lon", "lat"] if c in meta.columns]
    return meta[keep].drop_duplicates(subset=["station"]).sort_values("station").reset_index(drop=True)


def load_water_files(water_dir: Path) -> pd.DataFrame:
    paths = sorted(water_dir.glob("marine-historical-*-en.csv"))
    if not paths:
        raise FileNotFoundError(f"No marine-historical-*-en.csv files found under {water_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def parse_censored_series(raw: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    text = raw.astype("string").str.strip()
    text = text.mask(text.isin(["", "nan", "None", "<NA>"]), pd.NA)

    censored = text.str.startswith("<", na=False)
    numeric_text = text.str.replace("<", "", regex=False).str.strip()
    numeric = pd.to_numeric(numeric_text, errors="coerce").astype("float64")
    detect_limit = numeric.where(censored)
    value = numeric.where(~censored, numeric / 2)

    return text, value, censored.astype("boolean"), detect_limit


def build_master_table(raw: pd.DataFrame, station_meta: pd.DataFrame) -> pd.DataFrame:
    raw = raw.rename(columns={**ID_COLUMNS, **MEASUREMENT_COLUMNS}).copy()

    required = {"station", "date", "sample_no", "depth_raw", *MEASUREMENT_COLUMNS.values()}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Water quality files are missing columns: {sorted(missing)}")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    if raw["date"].isna().any():
        bad_rows = raw.loc[raw["date"].isna(), ["station", "source_file"]].head(10)
        raise ValueError(f"Found invalid dates in water quality files, e.g.\n{bad_rows.to_string(index=False)}")

    raw["year"] = raw["date"].dt.year
    raw["month"] = raw["date"].dt.month
    raw["day"] = raw["date"].dt.day
    raw["day_of_year"] = raw["date"].dt.dayofyear

    raw["depth"] = raw["depth_raw"].map(lambda x: DEPTH_MAP.get(x, (str(x).strip().lower(), pd.NA))[0])
    raw["depth_order"] = raw["depth_raw"].map(lambda x: DEPTH_MAP.get(x, (None, pd.NA))[1]).astype("Int64")
    raw["sample_id"] = raw["station"].astype(str) + "_" + raw["date"].dt.strftime("%Y%m%d")
    raw["obs_id"] = raw["sample_id"] + "_" + raw["depth"]

    out = raw[
        [
            "obs_id",
            "sample_id",
            "station",
            "date",
            "year",
            "month",
            "day",
            "day_of_year",
            "sample_no",
            "depth_raw",
            "depth",
            "depth_order",
            "water_zone_en_raw",
            "source_file",
            *MEASUREMENT_COLUMNS.values(),
        ]
    ].copy()

    for clean_name in MEASUREMENT_COLUMNS.values():
        raw_text, numeric_value, censored_flag, detect_limit = parse_censored_series(out[clean_name])
        out[f"{clean_name}_raw"] = raw_text
        out[clean_name] = numeric_value
        out[f"{clean_name}_censored"] = censored_flag
        out[f"{clean_name}_detect_limit"] = detect_limit

    out = out.merge(station_meta, on="station", how="left", validate="many_to_one")
    if out[["lon", "lat"]].isna().any().any():
        missing_stations = sorted(out.loc[out["lon"].isna() | out["lat"].isna(), "station"].dropna().unique().tolist())
        raise ValueError(f"Missing station coordinates for: {missing_stations}")

    if "water_zone_en" in out.columns:
        mismatch = out[
            out["water_zone_en"].notna()
            & out["water_zone_en_raw"].notna()
            & (out["water_zone_en"] != out["water_zone_en_raw"])
        ]
        if not mismatch.empty:
            preview = mismatch[["station", "water_zone_en_raw", "water_zone_en"]].drop_duplicates().head(10)
            raise ValueError(f"Water zone mismatch between station metadata and water files:\n{preview.to_string(index=False)}")

    ordered = [
        "obs_id",
        "sample_id",
        "station",
        "station_name_en",
        "water_zone_en",
        "water_zone_en_raw",
        "lon",
        "lat",
        "date",
        "year",
        "month",
        "day",
        "day_of_year",
        "sample_no",
        "depth_raw",
        "depth",
        "depth_order",
        "source_file",
    ]
    for clean_name in MEASUREMENT_COLUMNS.values():
        ordered.extend(
            [
                f"{clean_name}_raw",
                clean_name,
                f"{clean_name}_censored",
                f"{clean_name}_detect_limit",
            ]
        )

    return out[ordered].sort_values(["station", "date", "depth_order"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    station_meta = load_station_meta(args.station_meta)
    raw = load_water_files(args.water_dir)
    master = build_master_table(raw, station_meta)
    master.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.out} ({len(master)} rows)")
    print(f"Stations: {master['station'].nunique()}, sample_ids: {master['sample_id'].nunique()}, depths: {sorted(master['depth'].dropna().unique().tolist())}")
    print("Target availability:")
    print(
        master[["chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l"]]
        .notna()
        .sum()
        .rename(
            {
                "chla_ug_l": "Chl-a",
                "secchi_depth_m": "Secchi",
                "tn_mg_l": "TN",
                "tp_mg_l": "TP",
            }
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
