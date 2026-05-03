from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


DEFAULT_TIDE_DIR = Path("data/raw")
DEFAULT_WATER = Path("data/processed/water_quality_surface_2014_2024.csv")
DEFAULT_OUT = Path("data/processed/tide_features_2014_2024.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="由 HKO HLT 潮汐表生成每日潮差特征。")
    parser.add_argument("--tide-dir", type=Path, default=DEFAULT_TIDE_DIR)
    parser.add_argument("--station", default="QUB")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--water", type=Path, default=DEFAULT_WATER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def parse_tide_file(path: Path, year: int) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            month = int(row[0])
            day = int(row[1])
            heights = []
            for idx in [3, 5, 7, 9]:
                if idx < len(row) and row[idx].strip():
                    heights.append(float(row[idx]))
            if heights:
                rows.append(
                    {
                        "date": pd.Timestamp(year=year, month=month, day=day),
                        "tidal_range": max(heights) - min(heights),
                        "tide_events": len(heights)
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    frames = []
    for year in range(args.start_year, args.end_year + 1):
        path = args.tide_dir / f"tide_{args.station}_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少潮汐 CSV：{path}")
        frames.append(parse_tide_file(path, year))

    tide = pd.concat(frames, ignore_index=True).drop_duplicates("date")
    samples = pd.read_csv(args.water, usecols=["date"])
    dates = pd.to_datetime(samples["date"]).drop_duplicates().sort_values().to_frame()
    out = dates.merge(tide[["date", "tidal_range"]], on="date", how="left", validate="one_to_one")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"写出 {args.out}：{len(out)} 个采样日期")
    print(out["tidal_range"].describe().round(2).to_string())
    missing = int(out["tidal_range"].isna().sum())
    if missing:
        print(f"警告：tidal_range 缺失 {missing} 条")


if __name__ == "__main__":
    main()
