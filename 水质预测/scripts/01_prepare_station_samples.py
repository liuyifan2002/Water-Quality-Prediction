from __future__ import annotations

import argparse
import struct
from pathlib import Path

import pandas as pd


DEFAULT_WATER_DIR = Path(r"C:\Users\刘逸凡\Desktop\开题\水质数据")
DEFAULT_STATION_SHP = DEFAULT_WATER_DIR / "站点位置" / "download_20251010_1507_converted.shp"
DEFAULT_OUT_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare station metadata and a station-date table for GEE extraction."
    )
    parser.add_argument("--water-dir", type=Path, default=DEFAULT_WATER_DIR)
    parser.add_argument("--station-shp", type=Path, default=DEFAULT_STATION_SHP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_dbf(dbf_path: Path) -> pd.DataFrame:
    cpg_path = dbf_path.with_suffix(".cpg")
    encoding = "utf-8"
    if cpg_path.exists():
        encoding = cpg_path.read_text(encoding="ascii", errors="ignore").strip() or encoding

    with dbf_path.open("rb") as f:
        header = f.read(32)
        record_count = struct.unpack("<I", header[4:8])[0]
        header_length = struct.unpack("<H", header[8:10])[0]
        record_length = struct.unpack("<H", header[10:12])[0]

        fields = []
        while True:
            desc = f.read(32)
            if desc[0] == 0x0D:
                break
            name = desc[:11].split(b"\x00", 1)[0].decode(encoding, errors="ignore")
            field_type = chr(desc[11])
            length = desc[16]
            decimals = desc[17]
            fields.append((name, field_type, length, decimals))

        f.seek(header_length)
        rows = []
        for _ in range(record_count):
            record = f.read(record_length)
            if not record or record[0:1] == b"*":
                continue
            pos = 1
            row = {}
            for name, field_type, length, _decimals in fields:
                raw = record[pos : pos + length]
                pos += length
                value = raw.decode(encoding, errors="ignore").strip()
                row[name] = value if value != "" else pd.NA
            rows.append(row)

    return pd.DataFrame(rows)


def load_station_meta(station_shp: Path) -> pd.DataFrame:
    try:
        import geopandas as gpd
        stations = gpd.read_file(station_shp)
        if stations.crs is None:
            raise ValueError(f"{station_shp} has no CRS. Please define it before running this script.")
        stations = stations.to_crs("EPSG:4326")
    except Exception:
        stations = _read_dbf(station_shp.with_suffix(".dbf"))

    meta = stations.rename(
        columns={
            "A_Station": "Station",
            "B_Station_": "station_name_en",
            "C_Water_Co": "water_zone_en",
            "LATITUDE": "lat_source",
            "LONGITUDE": "lon_source",
        }
    ).copy()
    required = {"Station", "station_name_en", "water_zone_en"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Missing station columns in shapefile: {sorted(missing)}")

    if "geometry" in meta.columns:
        meta["lon"] = meta.geometry.x
        meta["lat"] = meta.geometry.y
    else:
        meta["lon"] = pd.to_numeric(meta["lon_source"], errors="raise")
        meta["lat"] = pd.to_numeric(meta["lat_source"], errors="raise")
    return meta[["Station", "station_name_en", "water_zone_en", "lon", "lat"]].sort_values("Station")


def load_water_samples(water_dir: Path) -> pd.DataFrame:
    paths = sorted(water_dir.glob("marine-historical-*-en.csv"))
    if not paths:
        raise FileNotFoundError(f"No marine-historical-*-en.csv files found under {water_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["Water Control Zone", "Station", "Dates"])
        frame["source_file"] = path.name
        frames.append(frame)

    water = pd.concat(frames, ignore_index=True)
    water["date"] = pd.to_datetime(water["Dates"], errors="raise").dt.date.astype(str)
    water["year"] = pd.to_datetime(water["date"]).dt.year
    water["month"] = pd.to_datetime(water["date"]).dt.month
    return water


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    station_meta = load_station_meta(args.station_shp)
    water = load_water_samples(args.water_dir)

    water_stations = set(water["Station"].dropna().astype(str))
    meta_stations = set(station_meta["Station"].dropna().astype(str))
    if water_stations != meta_stations:
        raise ValueError(
            "Station mismatch between water data and shapefile. "
            f"Missing in shapefile: {sorted(water_stations - meta_stations)}; "
            f"extra in shapefile: {sorted(meta_stations - water_stations)}"
        )

    samples = (
        water[["Station", "date", "year", "month"]]
        .drop_duplicates()
        .merge(station_meta, on="Station", how="left", validate="many_to_one")
        .sort_values(["Station", "date"])
        .reset_index(drop=True)
    )
    samples["sample_id"] = samples["Station"] + "_" + samples["date"].str.replace("-", "", regex=False)

    station_out = args.out_dir / "station_meta.csv"
    sample_out = args.out_dir / "gee_station_date_samples.csv"
    station_meta.to_csv(station_out, index=False, encoding="utf-8-sig")
    samples[
        ["sample_id", "Station", "date", "year", "month", "water_zone_en", "station_name_en", "lon", "lat"]
    ].to_csv(sample_out, index=False, encoding="utf-8-sig")

    print(f"Wrote {station_out} ({len(station_meta)} stations)")
    print(f"Wrote {sample_out} ({len(samples)} station-date samples)")


if __name__ == "__main__":
    main()
