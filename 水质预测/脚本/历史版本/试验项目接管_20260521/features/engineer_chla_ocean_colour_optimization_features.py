from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OC_BASE_FEATURES = [
    "oc_chl",
    "oc_kd490",
    "oc_zsd",
    "oc_rrs_412",
    "oc_rrs_443",
    "oc_rrs_490",
    "oc_rrs_560",
    "oc_rrs_665",
    "oc_chl_3d",
    "oc_chl_7d",
    "oc_chl_14d",
    "oc_kd490_3d",
    "oc_kd490_7d",
    "oc_kd490_14d",
    "oc_zsd_3d",
    "oc_zsd_7d",
    "oc_zsd_14d",
    "valid_count_7d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 Chl-a Ocean Colour 精度优化实验构建无泄漏增强特征"
    )
    parser.add_argument(
        "--input",
        default="data/interim/chla_ocean_colour_ab_20260517/chla_ocean_colour_ab_dataset_20260517.csv",
        help="上一阶段 A/B 合并数据集",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/chla_ocean_colour_precision_optimization_20260517",
        help="增强数据输出目录",
    )
    parser.add_argument("--date-tag", default="20260517")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def assert_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"输出已存在，请使用 --force 覆盖: {path}")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.abs() > 1e-12)
    return numerator / denominator


def add_grouped_lag_features(df: pd.DataFrame, group_cols: list[str], feature: str) -> list[str]:
    added: list[str] = []
    if feature not in df.columns:
        return added
    group = df.groupby(group_cols, sort=False)[feature]
    for lag in [1, 2]:
        col = f"{feature}_sample_lag{lag}"
        df[col] = group.shift(lag)
        added.append(col)
    shifted = group.shift(1)
    for window in [3, 5]:
        for stat in ["mean", "std", "min", "max"]:
            col = f"{feature}_past{window}_{stat}"
            rolled = shifted.groupby([df[col] for col in group_cols], sort=False).rolling(window, min_periods=1)
            if stat == "mean":
                values = rolled.mean()
            elif stat == "std":
                values = rolled.std()
            elif stat == "min":
                values = rolled.min()
            else:
                values = rolled.max()
            df[col] = values.reset_index(level=list(range(len(group_cols))), drop=True)
            added.append(col)
    return added


def add_past_rolling_features(df: pd.DataFrame, group_cols: list[str], feature: str) -> list[str]:
    added: list[str] = []
    if feature not in df.columns:
        return added
    for col in [f"{feature}_sample_lag1", f"{feature}_sample_lag2"]:
        if col not in df.columns:
            continue
    lag1 = df.get(f"{feature}_sample_lag1")
    lag2 = df.get(f"{feature}_sample_lag2")
    if lag1 is not None and lag2 is not None:
        trend_col = f"{feature}_lag1_trend"
        df[trend_col] = lag1 - lag2
        added.append(trend_col)
    return added


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / f"chla_ocean_colour_precision_optimization_dataset_{args.date_tag}.csv"
    manifest_path = out_dir / f"chla_ocean_colour_precision_feature_manifest_{args.date_tag}.csv"
    report_path = out_dir / f"chla_ocean_colour_precision_feature_report_{args.date_tag}.json"
    for path in [output_path, manifest_path, report_path]:
        assert_can_write(path, args.force)

    df = pd.read_csv(input_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station", "date", "depth", "obs_id"]).reset_index(drop=True)

    added_features: list[dict[str, str]] = []

    for feature in OC_BASE_FEATURES:
        if feature in df.columns:
            mask_col = f"{feature}_missing_mask"
            df[mask_col] = df[feature].isna().astype("int8")
            added_features.append(
                {
                    "feature": mask_col,
                    "source": feature,
                    "feature_type": "missing_mask",
                    "leakage_note": "仅记录该样本 Ocean Colour 字段是否缺失，不使用目标变量。",
                }
            )

    for feature in ["oc_chl", "oc_kd490", "oc_zsd", "oc_rrs_490", "oc_rrs_560"]:
        new_cols = add_grouped_lag_features(df, ["station"], feature)
        new_cols += add_past_rolling_features(df, ["station"], feature)
        for col in new_cols:
            added_features.append(
                {
                    "feature": col,
                    "source": feature,
                    "feature_type": "station_past_ocean_colour",
                    "leakage_note": "按 station 和日期排序后 shift，严格只使用历史站点水色记录。",
                }
            )

    if {"oc_chl", "oc_kd490"}.issubset(df.columns):
        df["oc_chl_x_kd490"] = df["oc_chl"] * df["oc_kd490"]
        df["oc_chl_per_kd490"] = safe_ratio(df["oc_chl"], df["oc_kd490"])
        added_features += [
            {
                "feature": "oc_chl_x_kd490",
                "source": "oc_chl, oc_kd490",
                "feature_type": "ocean_colour_interaction",
                "leakage_note": "同日外部水色特征组合，不使用水质目标。",
            },
            {
                "feature": "oc_chl_per_kd490",
                "source": "oc_chl, oc_kd490",
                "feature_type": "ocean_colour_ratio",
                "leakage_note": "同日外部水色特征比例，不使用水质目标。",
            },
        ]

    if {"oc_zsd", "oc_kd490"}.issubset(df.columns):
        df["oc_zsd_per_kd490"] = safe_ratio(df["oc_zsd"], df["oc_kd490"])
        added_features.append(
            {
                "feature": "oc_zsd_per_kd490",
                "source": "oc_zsd, oc_kd490",
                "feature_type": "ocean_colour_ratio",
                "leakage_note": "同日外部水色特征比例，不使用水质目标。",
            }
        )

    for prefix in ["oc_chl", "oc_kd490", "oc_zsd"]:
        for left, right in [("3d", "7d"), ("7d", "14d")]:
            left_col = f"{prefix}_{left}"
            right_col = f"{prefix}_{right}"
            if {left_col, right_col}.issubset(df.columns):
                diff_col = f"{prefix}_change_{left}_{right}"
                ratio_col = f"{prefix}_ratio_{left}_{right}"
                df[diff_col] = df[left_col] - df[right_col]
                df[ratio_col] = safe_ratio(df[left_col], df[right_col])
                added_features += [
                    {
                        "feature": diff_col,
                        "source": f"{left_col}, {right_col}",
                        "feature_type": "short_window_change",
                        "leakage_note": "使用已制备的外部水色窗口特征，不使用目标变量。",
                    },
                    {
                        "feature": ratio_col,
                        "source": f"{left_col}, {right_col}",
                        "feature_type": "short_window_ratio",
                        "leakage_note": "使用已制备的外部水色窗口特征，不使用目标变量。",
                    },
                ]

    if {"oc_chl", "valid_count_7d"}.issubset(df.columns):
        df["oc_chl_x_valid_count_7d"] = df["oc_chl"] * df["valid_count_7d"]
        added_features.append(
            {
                "feature": "oc_chl_x_valid_count_7d",
                "source": "oc_chl, valid_count_7d",
                "feature_type": "quality_weighted_ocean_colour",
                "leakage_note": "用有效观测数量表达水色可信度，不使用目标变量。",
            }
        )

    if {"chla_ug_l_lag1", "chla_ug_l_lag2"}.issubset(df.columns):
        df["chla_history_trend_lag1_lag2"] = df["chla_ug_l_lag1"] - df["chla_ug_l_lag2"]
        added_features.append(
            {
                "feature": "chla_history_trend_lag1_lag2",
                "source": "chla_ug_l_lag1, chla_ug_l_lag2",
                "feature_type": "target_history_trend",
                "leakage_note": "仅使用目标变量自身历史值，不使用当前 Chl-a。",
            }
        )
    if {"chla_ug_l_lag1", "chla_ug_l_roll3"}.issubset(df.columns):
        df["chla_history_lag1_minus_roll3"] = df["chla_ug_l_lag1"] - df["chla_ug_l_roll3"]
        df["chla_history_lag1_ratio_roll3"] = safe_ratio(df["chla_ug_l_lag1"], df["chla_ug_l_roll3"])
        added_features += [
            {
                "feature": "chla_history_lag1_minus_roll3",
                "source": "chla_ug_l_lag1, chla_ug_l_roll3",
                "feature_type": "target_history_anomaly",
                "leakage_note": "仅使用目标变量自身历史值，不使用当前 Chl-a。",
            },
            {
                "feature": "chla_history_lag1_ratio_roll3",
                "source": "chla_ug_l_lag1, chla_ug_l_roll3",
                "feature_type": "target_history_ratio",
                "leakage_note": "仅使用目标变量自身历史值，不使用当前 Chl-a。",
            },
        ]
    if {"chla_ug_l_lag1", "days_since_prev_sample"}.issubset(df.columns):
        df["chla_lag1_per_day_gap"] = safe_ratio(df["chla_ug_l_lag1"], df["days_since_prev_sample"])
        added_features.append(
            {
                "feature": "chla_lag1_per_day_gap",
                "source": "chla_ug_l_lag1, days_since_prev_sample",
                "feature_type": "target_history_rate",
                "leakage_note": "仅使用历史 Chl-a 和采样间隔，不使用当前 Chl-a。",
            }
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    manifest = pd.DataFrame(added_features)
    if not manifest.empty:
        manifest["missing_rate"] = manifest["feature"].map(lambda col: float(df[col].isna().mean()))
        manifest["non_null_count"] = manifest["feature"].map(lambda col: int(df[col].notna().sum()))
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    report = {
        "date_tag": args.date_tag,
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "added_feature_count": int(len(added_features)),
        "added_feature_manifest": str(manifest_path),
        "leakage_control": [
            "Ocean Colour lag/rolling 特征按 station 日期排序后 shift，只使用历史水色记录。",
            "Chl-a 历史增强特征只使用 lag/rolling 字段，不使用当前目标值。",
            "缺失 mask 只表达特征可用性，不使用目标变量。",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
