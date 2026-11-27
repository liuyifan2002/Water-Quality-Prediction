from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - 环境缺包时自动跳过
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - 环境缺包时自动跳过
    XGBRegressor = None


TARGET_COLUMN = "chla_ug_l"
DEPTH_ORDER = ["surface", "middle", "bottom"]
TIME_SPLITS = {
    "train": (2014, 2020),
    "validation": (2021, 2022),
    "test": (2023, 2024),
}
BASE_FEATURES = [
    "lon",
    "lat",
    "year_offset",
    "month_sin",
    "month_cos",
    "is_wet_season",
    "quarter",
    "day_of_year",
    "days_since_prev_sample",
    "chla_ug_l_lag1",
    "chla_ug_l_lag2",
    "chla_ug_l_roll3",
    "landsat_blue",
    "landsat_green",
    "landsat_red",
    "landsat_nir",
    "landsat_swir1",
    "landsat_swir2",
    "landsat_ndvi",
    "landsat_ndwi",
    "landsat_mndwi",
    "landsat_ndti",
    "landsat_image_count",
    "landsat_has_images",
    "landsat_clear_pixel_count",
    "landsat_strict_valid_gee",
    "landsat_available",
    "s2_image_count",
    "s2_has_images",
    "s2_valid_pixel_count",
    "s2_strict_valid",
    "s2_b2_mean",
    "s2_b3_mean",
    "s2_b4_mean",
    "s2_b5_mean",
    "s2_b6_mean",
    "s2_b8_mean",
    "s2_b8a_mean",
    "s2_b11_mean",
    "s2_b12_mean",
    "s2_ndwi_mean",
    "s2_mndwi_mean",
    "s2_ndci_mean",
    "s2_ndti_mean",
    "s2_available",
    "satellite_any_available",
    "satellite_available_count",
    "water_zone_en",
]
OCEAN_COLOUR_FEATURES = [
    "oc_valid",
    "oc_day_diff",
    "oc_pixel_distance_km",
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
IDENTITY_COLUMNS = ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"]


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 Chl-a Ocean Colour 单源 A/B 实验（固定时间外推，不修改正式主链）"
    )
    parser.add_argument(
        "--dataset",
        default="data/interim/chla_ocean_colour_ab_20260517/chla_ocean_colour_ab_dataset_20260517.csv",
        help="合并后的 A/B 数据集路径",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/outputs/chla_ocean_colour_ab_20260517",
        help="实验输出目录",
    )
    parser.add_argument(
        "--config-out",
        default="experiments/configs/chla_ocean_colour_ab_20260517.json",
        help="实验配置输出路径",
    )
    parser.add_argument(
        "--report-out",
        default="experiments/reports/chla_ocean_colour_ab_20260517.md",
        help="实验文字报告输出路径",
    )
    parser.add_argument("--date-tag", default="20260517", help="日期标签")
    parser.add_argument("--force", action="store_true", help="覆盖已存在结果")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def assert_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"输出已存在，请使用 --force 覆盖: {path}")


def build_model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if LGBMRegressor is not None:
        specs.append(
            ModelSpec(
                name="lightgbm",
                factory=lambda: LGBMRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                ),
            )
        )
    if XGBRegressor is not None:
        specs.append(
            ModelSpec(
                name="xgboost",
                factory=lambda: XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                ),
            )
        )
    specs.append(
        ModelSpec(
            name="random_forest",
            factory=lambda: RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            ),
        )
    )
    return specs


def split_label(year: int) -> str | None:
    for label, (start_year, end_year) in TIME_SPLITS.items():
        if start_year <= year <= end_year:
            return label
    return None


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100.0)


def safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-9
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
        "mape": safe_mape(y_true, y_pred),
    }


def select_available_features(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    features: list[str] = []
    for feature in candidates:
        if feature not in df.columns:
            continue
        if df[feature].notna().sum() == 0:
            continue
        features.append(feature)
    return features


def create_pipeline(train_df: pd.DataFrame, features: list[str], model: object) -> Pipeline:
    categorical_features = [
        feature
        for feature in features
        if train_df[feature].dtype == "object" or str(train_df[feature].dtype).startswith("category")
    ]
    numeric_features = [feature for feature in features if feature not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def collect_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importance = model.feature_importances_
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )


def build_missing_summary(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_set_name, feature_list in feature_sets.items():
        for feature in feature_list:
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "feature": feature,
                    "missing_rate": float(df[feature].isna().mean()),
                    "non_null_count": int(df[feature].notna().sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_set", "missing_rate", "feature"], ascending=[True, False, True])


def build_feature_manifest(feature_sets: dict[str, list[str]], df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    ocean_set = set(OCEAN_COLOUR_FEATURES)
    for feature_set_name, feature_list in feature_sets.items():
        for feature in feature_list:
            records.append(
                {
                    "feature_set": feature_set_name,
                    "feature": feature,
                    "source_group": "ocean_colour" if feature in ocean_set else "baseline",
                    "dtype": str(df[feature].dtype),
                    "missing_rate": float(df[feature].isna().mean()),
                    "non_null_count": int(df[feature].notna().sum()),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    config_path = Path(args.config_out)
    report_path = Path(args.report_out)

    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(config_path)
    ensure_parent(report_path)

    metrics_path = out_dir / "metrics.csv"
    predictions_path = out_dir / "predictions.csv"
    importance_path = out_dir / "feature_importance.csv"
    missing_summary_path = out_dir / "missing_summary.csv"
    feature_usage_path = out_dir / "feature_usage_log.csv"
    comparison_path = out_dir / "metrics_comparison.csv"

    for path in [
        metrics_path,
        predictions_path,
        importance_path,
        missing_summary_path,
        feature_usage_path,
        comparison_path,
        config_path,
        report_path,
    ]:
        assert_can_write(path, args.force)

    df = pd.read_csv(dataset_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["split"] = df["year"].map(split_label)
    df = df[df["split"].notna()].copy()

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"数据集缺少目标列: {TARGET_COLUMN}")

    base_features = select_available_features(df, BASE_FEATURES)
    oc_features = select_available_features(df, OCEAN_COLOUR_FEATURES)
    feature_sets = {
        "baseline": base_features,
        "baseline_plus_ocean_colour": base_features + [f for f in oc_features if f not in base_features],
    }

    model_specs = build_model_specs()
    if not model_specs:
        raise RuntimeError("没有可用模型，请至少安装 LightGBM / XGBoost / scikit-learn。")

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    importance_rows: list[pd.DataFrame] = []

    split_counts: dict[str, dict[str, int]] = {}
    for depth in DEPTH_ORDER:
        split_counts[depth] = {}
        depth_df = df[(df["depth"] == depth) & df[TARGET_COLUMN].notna()].copy()
        for split_name in TIME_SPLITS:
            split_counts[depth][split_name] = int((depth_df["split"] == split_name).sum())

    feature_usage_log = build_feature_manifest(df=df, feature_sets=feature_sets)
    feature_usage_log.to_csv(feature_usage_path, index=False, encoding="utf-8-sig")

    missing_summary = build_missing_summary(df=df, feature_sets=feature_sets)
    missing_summary.to_csv(missing_summary_path, index=False, encoding="utf-8-sig")

    for depth in DEPTH_ORDER:
        depth_df = df[(df["depth"] == depth) & df[TARGET_COLUMN].notna()].copy()
        train_df = depth_df[depth_df["split"] == "train"].copy()
        val_df = depth_df[depth_df["split"] == "validation"].copy()
        test_df = depth_df[depth_df["split"] == "test"].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(f"{depth} 缺少完整 train/validation/test 样本，无法继续。")

        p75_threshold = float(train_df[TARGET_COLUMN].quantile(0.75))
        p90_threshold = float(train_df[TARGET_COLUMN].quantile(0.90))

        for feature_set_name, features in feature_sets.items():
            for model_spec in model_specs:
                pipeline = create_pipeline(train_df, features, model_spec.factory())
                pipeline.fit(train_df[features], train_df[TARGET_COLUMN])

                importance_df = collect_feature_importance(pipeline)
                if not importance_df.empty:
                    importance_df["model"] = model_spec.name
                    importance_df["feature_set"] = feature_set_name
                    importance_df["depth"] = depth
                    importance_rows.append(importance_df)

                for split_name, split_df in [
                    ("train", train_df),
                    ("validation", val_df),
                    ("test", test_df),
                ]:
                    y_pred = pipeline.predict(split_df[features])
                    metric = evaluate_predictions(split_df[TARGET_COLUMN], y_pred)
                    high75_mask = split_df[TARGET_COLUMN] >= p75_threshold
                    high90_mask = split_df[TARGET_COLUMN] >= p90_threshold
                    low_mask = split_df[TARGET_COLUMN] < p75_threshold

                    metric_row = {
                        "depth": depth,
                        "split": split_name,
                        "model": model_spec.name,
                        "feature_set": feature_set_name,
                        "n_samples": int(len(split_df)),
                        "n_features": len(features),
                        "p75_threshold_train": p75_threshold,
                        "p90_threshold_train": p90_threshold,
                        **metric,
                        "high75_rmse": rmse(split_df.loc[high75_mask, TARGET_COLUMN], y_pred[high75_mask])
                        if high75_mask.any()
                        else np.nan,
                        "high75_mae": float(
                            mean_absolute_error(split_df.loc[high75_mask, TARGET_COLUMN], y_pred[high75_mask])
                        )
                        if high75_mask.any()
                        else np.nan,
                        "high90_rmse": rmse(split_df.loc[high90_mask, TARGET_COLUMN], y_pred[high90_mask])
                        if high90_mask.any()
                        else np.nan,
                        "high90_mae": float(
                            mean_absolute_error(split_df.loc[high90_mask, TARGET_COLUMN], y_pred[high90_mask])
                        )
                        if high90_mask.any()
                        else np.nan,
                        "high90_underestimation_rate": float(
                            np.mean(y_pred[high90_mask] < split_df.loc[high90_mask, TARGET_COLUMN])
                        )
                        if high90_mask.any()
                        else np.nan,
                        "low_value_mae": float(
                            mean_absolute_error(split_df.loc[low_mask, TARGET_COLUMN], y_pred[low_mask])
                        )
                        if low_mask.any()
                        else np.nan,
                    }
                    metrics_rows.append(metric_row)

                    prediction_frame = split_df[IDENTITY_COLUMNS].copy()
                    prediction_frame["split"] = split_name
                    prediction_frame["model"] = model_spec.name
                    prediction_frame["feature_set"] = feature_set_name
                    prediction_frame["y_true"] = split_df[TARGET_COLUMN].values
                    prediction_frame["y_pred"] = y_pred
                    prediction_rows.append(prediction_frame)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    importance_df = (
        pd.concat(importance_rows, ignore_index=True)
        if importance_rows
        else pd.DataFrame(columns=["feature", "importance", "model", "feature_set", "depth"])
    )

    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    baseline_metrics = test_metrics[test_metrics["feature_set"] == "baseline"].rename(
        columns={
            "rmse": "baseline_rmse",
            "mae": "baseline_mae",
            "r2": "baseline_r2",
            "smape": "baseline_smape",
            "mape": "baseline_mape",
            "high90_rmse": "baseline_high90_rmse",
            "high90_underestimation_rate": "baseline_high90_underestimation_rate",
            "low_value_mae": "baseline_low_value_mae",
        }
    )
    experiment_metrics = test_metrics[test_metrics["feature_set"] == "baseline_plus_ocean_colour"].copy()
    comparison_df = experiment_metrics.merge(
        baseline_metrics[
            [
                "depth",
                "model",
                "baseline_rmse",
                "baseline_mae",
                "baseline_r2",
                "baseline_smape",
                "baseline_mape",
                "baseline_high90_rmse",
                "baseline_high90_underestimation_rate",
                "baseline_low_value_mae",
            ]
        ],
        on=["depth", "model"],
        how="left",
    )
    comparison_df["delta_r2"] = comparison_df["r2"] - comparison_df["baseline_r2"]
    comparison_df["delta_rmse"] = comparison_df["rmse"] - comparison_df["baseline_rmse"]
    comparison_df["delta_mae"] = comparison_df["mae"] - comparison_df["baseline_mae"]
    comparison_df["delta_high90_rmse"] = comparison_df["high90_rmse"] - comparison_df["baseline_high90_rmse"]
    comparison_df["delta_high90_underestimation_rate"] = (
        comparison_df["high90_underestimation_rate"] - comparison_df["baseline_high90_underestimation_rate"]
    )
    comparison_df["delta_low_value_mae"] = comparison_df["low_value_mae"] - comparison_df["baseline_low_value_mae"]

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    config = {
        "date_tag": args.date_tag,
        "dataset_path": str(dataset_path),
        "output_dir": str(out_dir),
        "target_column": TARGET_COLUMN,
        "time_splits": TIME_SPLITS,
        "depths": DEPTH_ORDER,
        "base_features": base_features,
        "ocean_colour_features_available": oc_features,
        "feature_sets": feature_sets,
        "split_counts": split_counts,
        "models": [spec.name for spec in model_specs],
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sys.modules["sklearn"].__version__,
            "lightgbm": getattr(sys.modules.get("lightgbm"), "__version__", None),
            "xgboost": getattr(sys.modules.get("xgboost"), "__version__", None),
        },
        "notes": [
            "本实验仅针对 Chl-a，未引入同日其他水质协变量。",
            "Baseline 与 baseline_plus_ocean_colour 共享相同切分与模型，仅差异于 Ocean Colour 特征。",
            "当前 model_table 不含完整环境链，因此 baseline 定义为当前主表中可用的时间、空间、Chl-a 历史和原始卫星特征。",
        ],
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_rows = comparison_df.sort_values(["delta_r2", "r2"], ascending=[False, False]).head(6)
    report_lines = [
        "# Chl-a Ocean Colour 单源 A/B 实验",
        "",
        f"- 日期标签：`{args.date_tag}`",
        f"- 数据集：`{dataset_path}`",
        f"- 模型：`{', '.join(spec.name for spec in model_specs)}`",
        f"- 时间切分：Train 2014-2020 / Validation 2021-2022 / Test 2023-2024",
        "",
        "## 样本量",
        "",
    ]
    for depth in DEPTH_ORDER:
        counts = split_counts[depth]
        report_lines.append(
            f"- `{depth}`: train={counts['train']} / validation={counts['validation']} / test={counts['test']}"
        )
    report_lines.extend(["", "## Test 集 A/B 结果摘要", ""])
    if best_rows.empty:
        report_lines.append("- 暂无可比较结果。")
    else:
        for row in best_rows.itertuples(index=False):
            report_lines.append(
                f"- `{row.model}` / `{row.depth}`: baseline+ocean_colour 的 R²={row.r2:.4f}，"
                f"相对 baseline 变化 {row.delta_r2:+.4f}；RMSE={row.rmse:.4f}，"
                f"P90 RMSE 变化 {row.delta_high90_rmse:+.4f}。"
            )
    report_lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 这份报告仅记录 A/B 建模结果，不改写正式 outputs/modeling/。",
            "- 如果 Ocean Colour 带来稳定正增益，再进入四项指标扩展实验。",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("A/B experiment completed")
    print(json.dumps({"metrics": str(metrics_path), "comparison": str(comparison_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
