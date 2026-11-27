from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_IN = Path("数据/处理后数据/衍生版本/model_dataset_surface_2014_2024.csv")
DEFAULT_OUT = Path("数据/处理后数据/衍生版本/model_dataset_surface_2014_2024_enhanced.csv")
DEFAULT_REPORT = Path("归档_历史版本/过期输出/reports_root_csv/enhanced_feature_report.csv")


LAG_COLS = [
    "t2m", "tcc", "wind_speed",
    "rain_month", "rain_14d",
    "tidal_range", "ntl_avg_rad",
    "ndwi", "mndwi", "ndti", "valid_count"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建优化版水质模型特征。")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def add_ratio(df: pd.DataFrame, out_col: str, numerator: str, denominator: str) -> None:
    denom = df[denominator].replace(0, np.nan)
    df[out_col] = (df[numerator] / denom).replace([np.inf, -np.inf], np.nan)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, parse_dates=["date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    df["season_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["season_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    df["wind_dir_sin"] = df["v10"] / df["wind_speed"].replace(0, np.nan)
    df["wind_dir_cos"] = df["u10"] / df["wind_speed"].replace(0, np.nan)

    add_ratio(df, "blue_green", "blue", "green")
    add_ratio(df, "green_red", "green", "red")
    add_ratio(df, "red_green", "red", "green")
    add_ratio(df, "nir_red", "nir", "red")
    add_ratio(df, "red_nir", "red", "nir")

    for col in LAG_COLS:
        if col not in df.columns:
            continue
        grouped = df.groupby("station", sort=False)[col]
        df[f"{col}_lag1"] = grouped.shift(1)
        df[f"{col}_lag2"] = grouped.shift(2)
        df[f"{col}_roll3"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())

    # 滞后项只用同站点历史信息；开头月份缺失用站点中位数和全局中位数兜底。
    engineered_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_lag2") or c.endswith("_roll3")]
    engineered_cols += [
        "season_sin", "season_cos", "wind_speed", "wind_dir_sin", "wind_dir_cos",
        "blue_green", "green_red", "red_green", "nir_red", "red_nir"
    ]
    for col in engineered_cols:
        station_median = df.groupby("station")[col].transform("median")
        df[col] = df[col].fillna(station_median).fillna(df[col].median())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    report = pd.DataFrame(
        {
            "feature_group": [
                "season",
                "wind",
                "spectral_ratios",
                "lag_roll",
                "station_static"
            ],
            "feature_count": [
                2,
                3,
                5,
                len([c for c in engineered_cols if c.endswith("_lag1") or c.endswith("_lag2") or c.endswith("_roll3")]),
                3
            ],
            "purpose": [
                "用 sin/cos 表达季节周期，避免月份 12 到 1 的断点。",
                "把 U/V 风转换为风速和方向，表达混合与输运过程。",
                "补充 Landsat 水色比例特征，增强透明度、浊度和叶绿素代理信息。",
                "表达环境过程和遥感响应的滞后/平滑效应。",
                "训练阶段加入 lon/lat 和 station one-hot，显式表达空间差异。"
            ]
        }
    )
    report.to_csv(args.report, index=False, encoding="utf-8-sig")
    print(f"写出 {args.out}：{df.shape[0]} 行，{df.shape[1]} 列")
    print(f"写出 {args.report}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
