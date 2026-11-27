from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG = Path("配置/历史_v0配置.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并最小版表层模型数据集。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--era5", type=Path, default=Path("数据/处理后数据/正式主表/era5_monthly_2014_2024.csv"))
    parser.add_argument("--rain", type=Path, default=Path("数据/处理后数据/正式主表/rainfall_features_2014_2024.csv"))
    parser.add_argument("--tide", type=Path, default=Path("数据/处理后数据/正式主表/tide_features_2014_2024.csv"))
    parser.add_argument("--viirs", type=Path, default=Path("数据/处理后数据/正式主表/viirs_ntl_station_monthly_2014_2024.csv"))
    parser.add_argument("--landsat", type=Path, default=Path("数据/处理后数据/衍生版本/landsat_station_monthly_2014_2024_clean.csv"))
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def read_csv_if_exists(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"跳过 {name}：找不到 {path}")
        return None
    return pd.read_csv(path)


def normalize_landsat_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    if {"nir", "red"}.issubset(df.columns) and "ndvi" not in df.columns:
        df["ndvi"] = (df["nir"] - df["red"]) / (df["nir"] + df["red"])
    if {"green", "nir"}.issubset(df.columns) and "ndwi" not in df.columns:
        df["ndwi"] = (df["green"] - df["nir"]) / (df["green"] + df["nir"])
    if {"green", "swir1"}.issubset(df.columns) and "mndwi" not in df.columns:
        df["mndwi"] = (df["green"] - df["swir1"]) / (df["green"] + df["swir1"])
    if {"red", "green"}.issubset(df.columns) and "ndti" not in df.columns:
        df["ndti"] = (df["red"] - df["green"]) / (df["red"] + df["green"])
    return df


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    water_path = Path(config["outputs"]["water_quality"])
    out_path = args.out or Path(config["outputs"]["model_dataset"])

    if not water_path.exists():
        raise FileNotFoundError(f"请先运行 脚本/历史版本/01_prepare_water_quality_surface.py，缺少 {water_path}")

    data = pd.read_csv(water_path, parse_dates=["date"])
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    era5 = read_csv_if_exists(args.era5, "ERA5")
    if era5 is not None:
        data = data.merge(era5, on=["year", "month"], how="left", validate="many_to_one")

    rain = read_csv_if_exists(args.rain, "降水")
    if rain is not None:
        rain["date"] = pd.to_datetime(rain["date"])
        data = data.merge(rain, on="date", how="left", validate="many_to_one")

    tide = read_csv_if_exists(args.tide, "潮汐")
    if tide is not None:
        tide["date"] = pd.to_datetime(tide["date"])
        data = data.merge(tide, on="date", how="left", validate="many_to_one")

    viirs = read_csv_if_exists(args.viirs, "VIIRS 夜光")
    if viirs is not None:
        keep_cols = [c for c in ["station", "year", "month", "ntl_avg_rad", "ntl_cf_cvg"] if c in viirs.columns]
        data = data.merge(viirs[keep_cols], on=["station", "year", "month"], how="left", validate="many_to_one")

    landsat = read_csv_if_exists(args.landsat, "Landsat")
    if landsat is None and args.landsat.name.endswith("_clean.csv"):
        raw_landsat = args.landsat.with_name(args.landsat.name.replace("_clean.csv", ".csv"))
        landsat = read_csv_if_exists(raw_landsat, "Landsat 原始文件")
    if landsat is not None:
        landsat = normalize_landsat_columns(landsat)
        cols = [
            "station", "year", "month", "blue", "green", "red", "nir", "swir1", "swir2",
            "ndvi", "ndwi", "mndwi", "ndti", "valid_count",
            "landsat_missing", "landsat_imputed", "landsat_no_data"
        ]
        keep_cols = [c for c in cols if c in landsat.columns]
        data = data.merge(landsat[keep_cols], on=["station", "year", "month"], how="left", validate="many_to_one")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"写出 {out_path}：{len(data)} 行，{data.shape[1]} 列")
    print("当前缺失率最高的字段：")
    print((data.isna().mean().sort_values(ascending=False).head(20) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
