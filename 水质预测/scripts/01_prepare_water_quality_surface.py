from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG = Path("config/minimal_v0.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理 EPD 2014-2024 表层水质数据。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--station-months-out", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_dbf(dbf_path: Path) -> pd.DataFrame:
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
            length = desc[16]
            fields.append((name, length))

        f.seek(header_length)
        rows = []
        for _ in range(record_count):
            record = f.read(record_length)
            if not record or record[0:1] == b"*":
                continue
            pos = 1
            row = {}
            for name, length in fields:
                raw = record[pos : pos + length]
                pos += length
                value = raw.decode(encoding, errors="ignore").strip()
                row[name] = value if value else pd.NA
            rows.append(row)

    return pd.DataFrame(rows)


def load_station_meta(station_shp: Path) -> pd.DataFrame:
    try:
        import geopandas as gpd

        stations = gpd.read_file(station_shp)
        if stations.crs is None:
            raise ValueError(f"{station_shp} 没有 CRS，请先定义坐标系。")
        stations = stations.to_crs("EPSG:4326")
    except Exception:
        stations = read_dbf(station_shp.with_suffix(".dbf"))

    meta = stations.rename(
        columns={
            "A_Station": "station",
            "B_Station_": "station_name_en",
            "C_Water_Co": "water_zone_en",
            "LATITUDE": "lat_source",
            "LONGITUDE": "lon_source"
        }
    ).copy()

    if "geometry" in meta.columns:
        meta["lon"] = meta.geometry.x
        meta["lat"] = meta.geometry.y
    else:
        meta["lon"] = pd.to_numeric(meta["lon_source"], errors="raise")
        meta["lat"] = pd.to_numeric(meta["lat_source"], errors="raise")

    required = {"station", "lon", "lat"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"站点文件缺少字段：{sorted(missing)}")

    return meta[["station", "lon", "lat"]].drop_duplicates("station")


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("<", "", regex=False).str.strip(),
        errors="coerce"
    )


def load_epd_water(water_dir: Path, year_start: int, year_end: int) -> pd.DataFrame:
    frames = []
    for year in range(year_start, year_end + 1):
        path = water_dir / f"marine-historical-{year}-en.csv"
        if not path.exists():
            raise FileNotFoundError(f"找不到水质 CSV：{path}")
        frame = pd.read_csv(path)
        frame["source_year"] = year
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    year_start = int(config["year_start"])
    year_end = int(config["year_end"])
    stations = config["representative_stations"]
    layer = config["layer"]
    outputs = config["outputs"]

    water_dir = Path(config["epd_water_dir"])
    station_shp = Path(config["station_shp"])
    water_out = args.out or Path(outputs["water_quality"])
    station_months_out = args.station_months_out or Path(outputs["station_months"])
    water_out.parent.mkdir(parents=True, exist_ok=True)
    station_months_out.parent.mkdir(parents=True, exist_ok=True)

    water = load_epd_water(water_dir, year_start, year_end)
    meta = load_station_meta(station_shp)

    water = water.rename(
        columns={
            "Station": "station",
            "Dates": "date",
            "Depth": "layer",
            "Chlorophyll-a (μg/L)": "Chl_a",
            "Secchi Disc Depth (M)": "SD",
            "Total Nitrogen (mg/L)": "TN",
            "Total Phosphorus (mg/L)": "TP"
        }
    )
    water["date"] = pd.to_datetime(water["date"], errors="coerce")
    water["year"] = water["date"].dt.year
    water["month"] = water["date"].dt.month

    keep = water[
        water["station"].isin(stations)
        & water["year"].between(year_start, year_end)
        & water["layer"].eq(layer)
    ].copy()
    for col in ["Chl_a", "SD", "TN", "TP"]:
        keep[col] = to_number(keep[col])

    out = (
        keep[["date", "station", "layer", "Chl_a", "SD", "TN", "TP", "year", "month"]]
        .merge(meta, on="station", how="left", validate="many_to_one")
        [["date", "station", "lon", "lat", "layer", "Chl_a", "SD", "TN", "TP", "year", "month"]]
        .sort_values(["station", "date"])
        .reset_index(drop=True)
    )
    if out[["lon", "lat"]].isna().any().any():
        missing = out.loc[out["lon"].isna() | out["lat"].isna(), "station"].unique().tolist()
        raise ValueError(f"这些站点缺少经纬度：{missing}")

    out.to_csv(water_out, index=False, encoding="utf-8-sig")

    station_months = (
        out[["station", "lon", "lat", "year", "month"]]
        .drop_duplicates()
        .sort_values(["station", "year", "month"])
        .reset_index(drop=True)
    )
    station_months["month_id"] = (
        station_months["station"]
        + "_"
        + station_months["year"].astype(str)
        + station_months["month"].astype(str).str.zfill(2)
    )
    station_months[["month_id", "station", "lon", "lat", "year", "month"]].to_csv(
        station_months_out, index=False, encoding="utf-8-sig"
    )

    print(f"写出 {water_out}：{len(out)} 行，{out['station'].nunique()} 个站点")
    print(f"写出 {station_months_out}：{len(station_months)} 个站点-月份")
    print("目标变量缺失数：")
    print(out[["Chl_a", "SD", "TN", "TP"]].isna().sum().to_string())


if __name__ == "__main__":
    main()
