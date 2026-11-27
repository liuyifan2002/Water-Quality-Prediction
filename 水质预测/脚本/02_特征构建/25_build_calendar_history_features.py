"""构建仅使用当前观测日前历史记录的日历时间滞后特征。

该脚本产生扩展实验专用衍生主表，不覆盖正式主表。历史值可用于
“监测更新式预测”（本次采样前的监测结果已经可用），不代表一次性
多年完全盲推的设定。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv")
DEFAULT_OUTPUT = Path("数据/处理后数据/衍生版本/model_table_3depth_2014_2024_calendar_history.csv")
TARGETS = ["chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l"]
HALF_LIVES = (30, 90, 180)
RECENCY_WINDOWS = (30, 90, 180, 365)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为三水层主表构建因果日历历史特征。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def weighted_history(values: np.ndarray, dates: np.ndarray, half_life: int) -> np.ndarray:
    """返回每一时刻之前观测的按真实天数指数衰减均值。"""
    result = np.full(len(values), np.nan, dtype=float)
    for current in range(1, len(values)):
        past_values = values[:current]
        past_dates = dates[:current]
        valid = np.isfinite(past_values)
        if not valid.any():
            continue
        ages = (dates[current] - past_dates[valid]).astype("timedelta64[D]").astype(float)
        weights = np.exp(-np.log(2.0) * ages / half_life)
        result[current] = float(np.average(past_values[valid], weights=weights))
    return result


def add_target_history(data: pd.DataFrame, target: str) -> pd.DataFrame:
    out = data.copy()
    prefix = f"{target}_cal"
    for column in [
        f"{prefix}_prev", f"{prefix}_days_since_prev", f"{prefix}_history_count",
        *[f"{prefix}_prev_within_{days}d" for days in RECENCY_WINDOWS],
        *[f"{prefix}_ewm_{days}d" for days in HALF_LIVES],
    ]:
        out[column] = np.nan

    for _, group in out.groupby(["station", "depth"], sort=False):
        ordered = group.sort_values("date")
        index = ordered.index
        values = pd.to_numeric(ordered[target], errors="coerce").to_numpy(dtype=float)
        dates = ordered["date"].to_numpy(dtype="datetime64[ns]")

        previous = np.concatenate(([np.nan], values[:-1]))
        previous_dates = np.concatenate((np.array([np.datetime64("NaT")]), dates[:-1]))
        age_days = (dates - previous_dates).astype("timedelta64[D]").astype(float)
        age_days[0] = np.nan
        history_count = np.cumsum(np.isfinite(values)) - np.isfinite(values)

        out.loc[index, f"{prefix}_prev"] = previous
        out.loc[index, f"{prefix}_days_since_prev"] = age_days
        out.loc[index, f"{prefix}_history_count"] = history_count
        for days in RECENCY_WINDOWS:
            out.loc[index, f"{prefix}_prev_within_{days}d"] = np.where(age_days <= days, previous, np.nan)
        for half_life in HALF_LIVES:
            out.loc[index, f"{prefix}_ewm_{half_life}d"] = weighted_history(values, dates, half_life)
    return out


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"未找到正式主表：{args.input}")
    data = pd.read_csv(args.input, parse_dates=["date"], low_memory=False)
    data.columns = [str(column).replace("\ufeff", "") for column in data.columns]
    data = data.sort_values(["station", "depth", "date", "obs_id"]).reset_index(drop=True)

    for target in TARGETS:
        if target in data.columns:
            data = add_target_history(data, target)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, encoding="utf-8-sig")
    feature_count = len([column for column in data.columns if "_cal_" in column])
    print(f"写出：{args.output}")
    print(f"日历历史特征数：{feature_count}；行数：{len(data)}")


if __name__ == "__main__":
    main()
