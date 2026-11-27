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
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.run_chla_ocean_colour_ab_experiment import (
    BASE_FEATURES,
    DEPTH_ORDER,
    OCEAN_COLOUR_FEATURES,
    TARGET_COLUMN,
    TIME_SPLITS,
    evaluate_predictions,
    rmse,
    split_label,
)

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None


ENGINEERED_FEATURE_PREFIXES = [
    "oc_chl_",
    "oc_kd490_",
    "oc_zsd_",
    "oc_rrs_490_",
    "oc_rrs_560_",
    "chla_history_",
    "chla_lag1_per_day_gap",
]
ENGINEERED_FEATURE_EXACT = [
    "oc_chl_x_kd490",
    "oc_chl_per_kd490",
    "oc_zsd_per_kd490",
    "oc_chl_x_valid_count_7d",
]


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], object]


@dataclass
class ExperimentSpec:
    name: str
    model_name: str
    feature_group: str
    imputer: str
    weight_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 Chl-a Ocean Colour 精度优化第一批实验"
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_optimization_dataset_20260517.csv",
    )
    parser.add_argument(
        "--engineered-manifest",
        default="data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_feature_manifest_20260517.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/outputs/chla_ocean_colour_precision_optimization_20260517",
    )
    parser.add_argument(
        "--config-out",
        default="experiments/configs/chla_ocean_colour_precision_optimization_20260517.json",
    )
    parser.add_argument(
        "--report-out",
        default="experiments/reports/chla_ocean_colour_precision_optimization_20260517.md",
    )
    parser.add_argument("--date-tag", default="20260517")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def assert_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"输出已存在，请使用 --force 覆盖: {path}")


def select_available_features(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    result: list[str] = []
    for col in candidates:
        if col not in df.columns:
            continue
        if df[col].notna().sum() == 0:
            continue
        result.append(col)
    return result


def find_engineered_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for col in df.columns:
        if col in ENGINEERED_FEATURE_EXACT:
            features.append(col)
            continue
        if col.endswith("_missing_mask"):
            features.append(col)
            continue
        if any(col.startswith(prefix) for prefix in ENGINEERED_FEATURE_PREFIXES):
            if col not in OCEAN_COLOUR_FEATURES:
                features.append(col)
    return select_available_features(df, sorted(set(features)))


def build_model_specs() -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    if LGBMRegressor is not None:
        specs["lightgbm_regularized"] = ModelSpec(
            "lightgbm_regularized",
            lambda: LGBMRegressor(
                n_estimators=700,
                learning_rate=0.035,
                num_leaves=15,
                max_depth=5,
                min_child_samples=45,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=1,
                verbose=-1,
            ),
        )
        specs["lightgbm_deeper"] = ModelSpec(
            "lightgbm_deeper",
            lambda: LGBMRegressor(
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.85,
                reg_lambda=0.5,
                random_state=42,
                n_jobs=1,
                verbose=-1,
            ),
        )
    if XGBRegressor is not None:
        specs["xgboost_regularized"] = ModelSpec(
            "xgboost_regularized",
            lambda: XGBRegressor(
                n_estimators=550,
                learning_rate=0.035,
                max_depth=4,
                min_child_weight=4,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.5,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            ),
        )
        specs["xgboost_deeper"] = ModelSpec(
            "xgboost_deeper",
            lambda: XGBRegressor(
                n_estimators=500,
                learning_rate=0.04,
                max_depth=6,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            ),
        )
    if CatBoostRegressor is not None:
        specs["catboost_regularized"] = ModelSpec(
            "catboost_regularized",
            lambda: CatBoostRegressor(
                iterations=600,
                learning_rate=0.035,
                depth=6,
                l2_leaf_reg=5,
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            ),
        )
    specs["random_forest_reference"] = ModelSpec(
        "random_forest_reference",
        lambda: RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
    )
    return specs


def build_experiment_specs(model_specs: dict[str, ModelSpec]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for model_name in ["lightgbm_regularized", "xgboost_regularized", "random_forest_reference"]:
        if model_name in model_specs:
            specs.append(ExperimentSpec("reference_" + model_name, model_name, "reference", "median", "none"))
    for model_name in ["lightgbm_regularized", "lightgbm_deeper", "xgboost_regularized", "xgboost_deeper", "catboost_regularized"]:
        if model_name in model_specs:
            specs.append(ExperimentSpec("engineered_" + model_name, model_name, "engineered", "median_indicator", "none"))
            specs.append(ExperimentSpec("engineered_p90x5_" + model_name, model_name, "engineered", "median_indicator", "p90x5"))
    if "lightgbm_regularized" in model_specs:
        specs.append(ExperimentSpec("engineered_knn_lightgbm_regularized", "lightgbm_regularized", "engineered", "knn", "none"))
        specs.append(ExperimentSpec("engineered_p85x3_lightgbm_regularized", "lightgbm_regularized", "engineered", "median_indicator", "p85x3"))
    return specs


def make_preprocessor(train_df: pd.DataFrame, features: list[str], imputer_name: str) -> ColumnTransformer:
    categorical = [
        col
        for col in features
        if train_df[col].dtype == "object" or str(train_df[col].dtype).startswith("category")
    ]
    numeric = [col for col in features if col not in categorical]
    if imputer_name == "knn":
        numeric_imputer = KNNImputer(n_neighbors=5, weights="distance")
    elif imputer_name == "median_indicator":
        numeric_imputer = SimpleImputer(strategy="median", add_indicator=True)
    else:
        numeric_imputer = SimpleImputer(strategy="median")
    return ColumnTransformer(
        transformers=[
            ("num", numeric_imputer, numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def create_pipeline(train_df: pd.DataFrame, features: list[str], imputer_name: str, model: object) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(train_df, features, imputer_name)),
            ("model", model),
        ]
    )


def sample_weights(train_y: pd.Series, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    weights = np.ones(len(train_y), dtype=float)
    if mode == "p90x5":
        weights[train_y >= train_y.quantile(0.90)] = 5.0
    elif mode == "p85x3":
        weights[train_y >= train_y.quantile(0.85)] = 3.0
    else:
        raise ValueError(f"未知权重模式: {mode}")
    return weights


def evaluate_regime(split_df: pd.DataFrame, y_pred: np.ndarray, p75: float, p90: float) -> dict[str, float]:
    y_true = split_df[TARGET_COLUMN]
    high75 = y_true >= p75
    high90 = y_true >= p90
    low = y_true < p75
    out = {}
    out["high75_rmse"] = rmse(y_true[high75], y_pred[high75]) if high75.any() else np.nan
    out["high75_mae"] = float(mean_absolute_error(y_true[high75], y_pred[high75])) if high75.any() else np.nan
    out["high90_rmse"] = rmse(y_true[high90], y_pred[high90]) if high90.any() else np.nan
    out["high90_mae"] = float(mean_absolute_error(y_true[high90], y_pred[high90])) if high90.any() else np.nan
    out["high90_underestimation_rate"] = float(np.mean(y_pred[high90] < y_true[high90])) if high90.any() else np.nan
    out["low_value_mae"] = float(mean_absolute_error(y_true[low], y_pred[low])) if low.any() else np.nan
    return out


def collect_importance(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame({"feature": names, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    config_path = Path(args.config_out)
    report_path = Path(args.report_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "metrics": out_dir / "metrics.csv",
        "predictions": out_dir / "predictions.csv",
        "feature_importance": out_dir / "feature_importance.csv",
        "experiment_config": out_dir / "experiment_config.json",
        "best_by_depth": out_dir / "best_by_depth.csv",
        "high_value_summary": out_dir / "high_value_summary.csv",
        "feature_group_manifest": out_dir / "feature_group_manifest.csv",
        "config_copy": config_path,
        "report": report_path,
    }
    for path in output_paths.values():
        assert_can_write(path, args.force)

    df = pd.read_csv(dataset_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["split"] = df["year"].map(split_label)
    df = df[df["split"].notna() & df[TARGET_COLUMN].notna()].copy()

    base_features = select_available_features(df, BASE_FEATURES)
    oc_features = select_available_features(df, OCEAN_COLOUR_FEATURES)
    engineered_features = find_engineered_features(df)
    feature_groups = {
        "reference": base_features + [col for col in oc_features if col not in base_features],
        "engineered": base_features
        + [col for col in oc_features if col not in base_features]
        + [col for col in engineered_features if col not in base_features and col not in oc_features],
    }

    model_specs = build_model_specs()
    experiment_specs = build_experiment_specs(model_specs)

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    identity_cols = ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"]
    for depth in DEPTH_ORDER:
        depth_df = df[df["depth"] == depth].copy()
        train_df = depth_df[depth_df["split"] == "train"].copy()
        val_df = depth_df[depth_df["split"] == "validation"].copy()
        test_df = depth_df[depth_df["split"] == "test"].copy()
        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(f"{depth} 样本切分不完整，停止实验。")
        p75 = float(train_df[TARGET_COLUMN].quantile(0.75))
        p90 = float(train_df[TARGET_COLUMN].quantile(0.90))

        for exp in experiment_specs:
            features = feature_groups[exp.feature_group]
            pipe = create_pipeline(train_df, features, exp.imputer, model_specs[exp.model_name].factory())
            weights = sample_weights(train_df[TARGET_COLUMN], exp.weight_mode)
            fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
            pipe.fit(train_df[features], train_df[TARGET_COLUMN], **fit_kwargs)

            imp = collect_importance(pipe)
            if not imp.empty:
                imp["depth"] = depth
                imp["experiment"] = exp.name
                imp["model"] = exp.model_name
                imp["feature_group"] = exp.feature_group
                importance_frames.append(imp)

            for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
                y_pred = pipe.predict(split_df[features])
                y_pred = np.maximum(y_pred, 0)
                metrics = evaluate_predictions(split_df[TARGET_COLUMN], y_pred)
                row = {
                    "depth": depth,
                    "split": split_name,
                    "experiment": exp.name,
                    "model": exp.model_name,
                    "feature_group": exp.feature_group,
                    "imputer": exp.imputer,
                    "weight_mode": exp.weight_mode,
                    "n_samples": int(len(split_df)),
                    "n_features": int(len(features)),
                    "p75_threshold_train": p75,
                    "p90_threshold_train": p90,
                    **metrics,
                    **evaluate_regime(split_df, y_pred, p75, p90),
                }
                metrics_rows.append(row)

                pred = split_df[identity_cols].copy()
                pred["split"] = split_name
                pred["experiment"] = exp.name
                pred["model"] = exp.model_name
                pred["feature_group"] = exp.feature_group
                pred["y_true"] = split_df[TARGET_COLUMN].values
                pred["y_pred"] = y_pred
                prediction_frames.append(pred)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    importance_df = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame(columns=["feature", "importance", "depth", "experiment", "model", "feature_group"])
    )

    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    best_by_depth = test_metrics.sort_values(["depth", "r2", "high90_rmse"], ascending=[True, False, True]).groupby(
        "depth", as_index=False
    ).head(5)
    high_value_summary = test_metrics[
        [
            "depth",
            "experiment",
            "model",
            "feature_group",
            "imputer",
            "weight_mode",
            "r2",
            "rmse",
            "mae",
            "high90_rmse",
            "high90_mae",
            "high90_underestimation_rate",
            "low_value_mae",
        ]
    ].sort_values(["depth", "high90_rmse", "r2"], ascending=[True, True, False])

    feature_group_manifest = pd.DataFrame(
        [
            {"feature_group": group, "feature": feature, "source": "engineered" if feature in engineered_features else "base_or_ocean"}
            for group, features in feature_groups.items()
            for feature in features
        ]
    )

    metrics_df.to_csv(output_paths["metrics"], index=False, encoding="utf-8-sig")
    predictions_df.to_csv(output_paths["predictions"], index=False, encoding="utf-8-sig")
    importance_df.to_csv(output_paths["feature_importance"], index=False, encoding="utf-8-sig")
    best_by_depth.to_csv(output_paths["best_by_depth"], index=False, encoding="utf-8-sig")
    high_value_summary.to_csv(output_paths["high_value_summary"], index=False, encoding="utf-8-sig")
    feature_group_manifest.to_csv(output_paths["feature_group_manifest"], index=False, encoding="utf-8-sig")

    config = {
        "date_tag": args.date_tag,
        "experiment_name": "chla_ocean_colour_precision_optimization_20260517",
        "dataset": str(dataset_path),
        "target": TARGET_COLUMN,
        "time_splits": TIME_SPLITS,
        "depths": DEPTH_ORDER,
        "feature_groups": {k: v for k, v in feature_groups.items()},
        "experiment_specs": [exp.__dict__ for exp in experiment_specs],
        "models_available": sorted(model_specs.keys()),
        "catboost_available": CatBoostRegressor is not None,
        "python": sys.version,
        "platform": platform.platform(),
        "notes": [
            "固定时间切分，无随机切分与 shuffle。",
            "P85/P90 权重阈值只基于训练集目标分布计算。",
            "增强特征来自外部 Ocean Colour、Chl-a 历史 lag/rolling 和缺失可用性，不引入同日其他水质变量。",
        ],
    }
    for path in [output_paths["experiment_config"], output_paths["config_copy"]]:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Chl-a Ocean Colour 精度优化第一批实验",
        "",
        f"- 日期标签：`{args.date_tag}`",
        f"- 数据集：`{dataset_path}`",
        f"- 输出目录：`{out_dir}`",
        f"- CatBoost 可用：`{CatBoostRegressor is not None}`",
        "",
        "## Test 集各水层 Top 结果",
        "",
    ]
    for row in best_by_depth.itertuples(index=False):
        report_lines.append(
            f"- `{row.depth}` / `{row.experiment}`: R²={row.r2:.4f}, "
            f"RMSE={row.rmse:.4f}, P90 RMSE={row.high90_rmse:.4f}, "
            f"P90低估率={row.high90_underestimation_rate:.4f}"
        )
    report_lines.extend(
        [
            "",
            "## 工程边界",
            "",
            "- 未覆盖正式 `outputs/modeling/`。",
            "- 未修改 test 集，不使用随机切分。",
            "- 未引入同日其他水质协变量。",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "metrics": str(output_paths["metrics"]),
                "best_by_depth": str(output_paths["best_by_depth"]),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
