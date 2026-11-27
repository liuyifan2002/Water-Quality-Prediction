from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 Ocean Colour A/B 实验数据集（仅合并与校验，不训练模型）"
    )
    parser.add_argument(
        "--model-table",
        default="data/model_input/model_table_3depth_2014_2024.csv",
        help="主建模表路径",
    )
    parser.add_argument(
        "--ocean-colour",
        default="data/model_input/new_sources/ocean_color_kd490_tss_features_2014_2024.csv",
        help="Ocean Colour 特征表路径",
    )
    parser.add_argument(
        "--out-dir",
        default="data/interim/chla_ocean_colour_ab_20260517",
        help="输出目录",
    )
    parser.add_argument("--date-tag", default="20260517", help="日期标签")
    parser.add_argument(
        "--force", action="store_true", help="覆盖已存在输出"
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def assert_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"输出已存在，请使用 --force 覆盖: {path}")


def main() -> None:
    args = parse_args()
    model_table_path = Path(args.model_table)
    ocean_colour_path = Path(args.ocean_colour)
    out_dir = Path(args.out_dir)
    ensure_output_dir(out_dir)

    dataset_path = out_dir / f"chla_ocean_colour_ab_dataset_{args.date_tag}.csv"
    merge_report_path = out_dir / f"chla_ocean_colour_ab_merge_report_{args.date_tag}.json"
    missing_path = out_dir / f"chla_ocean_colour_ab_missing_summary_{args.date_tag}.csv"

    assert_can_write(dataset_path, args.force)
    assert_can_write(merge_report_path, args.force)
    assert_can_write(missing_path, args.force)

    model_df = pd.read_csv(model_table_path, low_memory=False)
    ocean_df = pd.read_csv(ocean_colour_path, low_memory=False)

    model_df["date"] = pd.to_datetime(model_df["date"])
    ocean_df["date"] = pd.to_datetime(ocean_df["date"])

    required_model_cols = {"obs_id", "sample_id", "station", "date", "depth", "chla_ug_l"}
    missing_model_cols = sorted(required_model_cols - set(model_df.columns))
    if missing_model_cols:
        raise KeyError(f"主表缺少必要字段: {missing_model_cols}")

    required_ocean_cols = {"station", "date"}
    missing_ocean_cols = sorted(required_ocean_cols - set(ocean_df.columns))
    if missing_ocean_cols:
        raise KeyError(f"Ocean Colour 表缺少必要字段: {missing_ocean_cols}")

    duplicate_count = int(ocean_df.duplicated(subset=["station", "date"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Ocean Colour 特征表存在 station+date 重复键，无法安全合并: {duplicate_count}"
        )

    merged_df = model_df.merge(
        ocean_df,
        on=["station", "date"],
        how="left",
        suffixes=("", "_oc"),
        validate="many_to_one",
    )

    if len(merged_df) != len(model_df):
        raise ValueError(
            f"合并后样本数变化：合并前 {len(model_df)}，合并后 {len(merged_df)}"
        )

    # 清理少量重复元数据列
    redundant_cols = [
        col for col in merged_df.columns if col.endswith("_oc") and col[:-3] in merged_df.columns
    ]
    if redundant_cols:
        merged_df = merged_df.drop(columns=redundant_cols)

    merged_df = merged_df.sort_values(["date", "station", "depth", "obs_id"]).reset_index(drop=True)
    merged_df.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    missing_summary = pd.DataFrame(
        {
            "column": merged_df.columns,
            "missing_rate": merged_df.isna().mean().values,
            "non_null_count": merged_df.notna().sum().values,
        }
    ).sort_values(["missing_rate", "column"], ascending=[False, True])
    missing_summary.to_csv(missing_path, index=False, encoding="utf-8-sig")

    report = {
        "date_tag": args.date_tag,
        "model_table_path": str(model_table_path),
        "ocean_colour_path": str(ocean_colour_path),
        "output_dataset_path": str(dataset_path),
        "merge_key": ["station", "date"],
        "rows_before_merge": int(len(model_df)),
        "rows_after_merge": int(len(merged_df)),
        "duplicate_station_date_in_ocean_colour": duplicate_count,
        "ocean_colour_row_count": int(len(ocean_df)),
        "ocean_colour_date_range": [
            ocean_df["date"].min().strftime("%Y-%m-%d"),
            ocean_df["date"].max().strftime("%Y-%m-%d"),
        ],
        "merged_date_range": [
            merged_df["date"].min().strftime("%Y-%m-%d"),
            merged_df["date"].max().strftime("%Y-%m-%d"),
        ],
        "ocean_colour_columns": sorted(
            [col for col in ocean_df.columns if col not in {"station", "date", "month"}]
        ),
        "note": "当前数据集仅完成样本级合并与缺失统计，未进行任何建模。",
    }
    merge_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
