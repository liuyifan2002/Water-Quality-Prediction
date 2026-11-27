from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RAIN = Path("数据/原始数据/daily_HKO_RF_ALL.csv")
DEFAULT_WATER = Path("数据/处理后数据/衍生版本/water_quality_surface_2014_2024.csv")
DEFAULT_OUT = Path("数据/处理后数据/正式主表/rainfall_features_2014_2024.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="由 HKO 日降水生成采样日期降水特征。")
    parser.add_argument("--rain", type=Path, default=DEFAULT_RAIN)
    parser.add_argument("--water", type=Path, default=DEFAULT_WATER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rain = pd.read_csv(args.rain, skiprows=2)
    rain = rain.rename(
        columns={
            "年/Year": "year",
            "月/Month": "month",
            "日/Day": "day",
            "數值/Value": "rain"
        }
    )
    rain["date"] = pd.to_datetime(rain[["year", "month", "day"]], errors="coerce")
    rain["rain"] = pd.to_numeric(rain["rain"], errors="coerce").fillna(0)
    rain = (
        rain[["date", "year", "month", "rain"]]
        .dropna(subset=["date"])
        .groupby("date", as_index=False)["rain"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    rain["year"] = rain["date"].dt.year
    rain["month"] = rain["date"].dt.month

    samples = pd.read_csv(args.water, usecols=["date"])
    sample_dates = pd.to_datetime(samples["date"]).drop_duplicates().sort_values()
    rain_by_date = rain.set_index("date")["rain"]
    month_sum = rain.groupby(["year", "month"], as_index=False)["rain"].sum().rename(columns={"rain": "rain_month"})

    rows = []
    for date in sample_dates:
        prev_3 = pd.date_range(date - pd.Timedelta(days=3), date - pd.Timedelta(days=1), freq="D")
        prev_14 = pd.date_range(date - pd.Timedelta(days=14), date - pd.Timedelta(days=1), freq="D")
        rows.append(
            {
                "date": date,
                "year": date.year,
                "month": date.month,
                "rain_3d": rain_by_date.reindex(prev_3).sum(),
                "rain_14d": rain_by_date.reindex(prev_14).sum()
            }
        )

    features = pd.DataFrame(rows).merge(month_sum, on=["year", "month"], how="left", validate="many_to_one")
    features = features[["date", "rain_3d", "rain_14d", "rain_month"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"写出 {args.out}：{len(features)} 个采样日期")
    print(features[["rain_3d", "rain_14d", "rain_month"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
