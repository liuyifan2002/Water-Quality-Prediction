from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from scripts.training.run_chla_ocean_colour_ab_experiment import (
    BASE_FEATURES,
    DEPTH_ORDER,
    OCEAN_COLOUR_FEATURES,
    TIME_SPLITS,
    evaluate_predictions,
    rmse,
    split_label,
)
from scripts.training.run_chla_ocean_colour_precision_optimization import find_engineered_features


DATE_TAG = "20260518"
TARGETS = {
    "chla": {
        "column": "chla_ug_l",
        "history_prefix": "chla_ug_l",
        "previous_best": {
            "surface": {"r2": 0.3207, "rmse": 5.2324, "high90_rmse": 14.4626},
            "middle": {"r2": 0.3646, "rmse": 3.8751, "high90_rmse": 9.5047},
            "bottom": {"r2": 0.3318, "rmse": 3.5180, "high90_rmse": 8.1324},
        },
    },
    "secchi": {
        "column": "secchi_depth_m",
        "history_prefix": "secchi_depth_m",
        "previous_best": {
            "surface": {"r2": 0.4071, "rmse": 0.8756, "high90_rmse": 1.8743},
            "middle": {"r2": 0.3330, "rmse": 0.9346, "high90_rmse": 2.0079},
            "bottom": {"r2": 0.3449, "rmse": 0.9080, "high90_rmse": 1.8621},
        },
    },
}
IDENTITY_COLUMNS = ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"]
CAT_FEATURES = ["station", "water_zone_en", "month", "quarter", "is_wet_season"]


@dataclass
class PredictionBundle:
    target: str
    depth: str
    candidate: str
    strategy: str
    model_family: str
    feature_group: str
    train_pred: np.ndarray
    validation_pred: np.ndarray
    test_pred: np.ndarray
    features: list[str]
    importance: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行文献启发的精度提升实验")
    parser.add_argument(
        "--input",
        default="data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_optimization_dataset_20260517.csv",
    )
    parser.add_argument("--date-tag", default=DATE_TAG)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=f"experiments/outputs/literature_inspired_precision_boost_{DATE_TAG}",
    )
    parser.add_argument(
        "--processed-dir",
        default=f"data/processed/literature_inspired_precision_boost_{DATE_TAG}",
    )
    parser.add_argument(
        "--config-out",
        default=f"experiments/configs/literature_inspired_precision_boost_{DATE_TAG}.json",
    )
    parser.add_argument(
        "--report-out",
        default=f"experiments/reports/literature_inspired_precision_boost_{DATE_TAG}.md",
    )
    return parser.parse_args()


def assert_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"输出已存在，请使用 --force 覆盖: {path}")


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b.abs() > 1e-12)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["split"] = df["year"].map(split_label)
    df = df[df["split"].notna()].copy()
    df["month"] = df["month"].astype("int16")
    return df


def add_target_features(df: pd.DataFrame, target: str, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    added: list[str] = []
    lag1, lag2, roll3 = f"{prefix}_lag1", f"{prefix}_lag2", f"{prefix}_roll3"
    if {lag1, lag2}.issubset(out.columns):
        col = f"{target}_lag_trend"
        out[col] = out[lag1] - out[lag2]
        added.append(col)
    if {lag1, roll3}.issubset(out.columns):
        col = f"{target}_lag_anomaly"
        out[col] = out[lag1] - out[roll3]
        added.append(col)
        col = f"{target}_lag_ratio"
        out[col] = safe_ratio(out[lag1], out[roll3])
        added.append(col)
    if {lag1, "days_since_prev_sample"}.issubset(out.columns):
        col = f"{target}_lag_rate"
        out[col] = safe_ratio(out[lag1], out["days_since_prev_sample"])
        added.append(col)
    return out.replace([np.inf, -np.inf], np.nan), added


def available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns and df[col].notna().sum() > 0]


def build_features(df: pd.DataFrame, target: str, prefix: str, added: list[str]) -> list[str]:
    base = [col for col in BASE_FEATURES if not col.startswith("chla_ug_l_")]
    base += [f"{prefix}_lag1", f"{prefix}_lag2", f"{prefix}_roll3"]
    base += added
    base += ["station", "month"]
    ocean = OCEAN_COLOUR_FEATURES + find_engineered_features(df)
    if target != "chla":
        ocean = [col for col in ocean if not col.startswith("chla_")]
    mechanism = [
        "oc_zsd_change_3d_7d",
        "oc_zsd_change_7d_14d",
        "oc_kd490_change_3d_7d",
        "oc_kd490_change_7d_14d",
        "oc_chl_change_3d_7d",
        "oc_chl_change_7d_14d",
        "oc_chl_x_valid_count_7d",
        "oc_zsd_per_kd490",
        "oc_chl_per_kd490",
    ]
    return list(dict.fromkeys(available(df, base + ocean + mechanism)))


def split_frames(df: pd.DataFrame, target_col: str, depth: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = df[(df["depth"] == depth) & df[target_col].notna()].copy()
    return (
        sub[sub["split"] == "train"].copy(),
        sub[sub["split"] == "validation"].copy(),
        sub[sub["split"] == "test"].copy(),
    )


def sklearn_preprocessor(train: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    cat = [col for col in features if col in CAT_FEATURES or train[col].dtype == "object"]
    num = [col for col in features if col not in cat]
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def sklearn_model(family: str) -> Any:
    if family == "lightgbm":
        return LGBMRegressor(
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=18,
            max_depth=6,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
    if family == "xgboost":
        return XGBRegressor(
            n_estimators=420,
            learning_rate=0.035,
            max_depth=4,
            min_child_weight=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
    return RandomForestRegressor(n_estimators=260, min_samples_leaf=2, random_state=42, n_jobs=1)


def fit_sklearn_candidate(
    family: str,
    target: str,
    depth: str,
    candidate: str,
    strategy: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    features: list[str],
) -> PredictionBundle:
    pipe = Pipeline([("preprocessor", sklearn_preprocessor(train, features)), ("model", sklearn_model(family))])
    pipe.fit(train[features], train[target_col])
    preds = [np.maximum(pipe.predict(frame[features]), 0) for frame in [train, val, test]]
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
        importance = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    else:
        importance = pd.DataFrame(columns=["feature", "importance"])
    importance["target"] = target
    importance["depth"] = depth
    importance["candidate"] = candidate
    return PredictionBundle(target, depth, candidate, strategy, family, "ocean_mechanism", *preds, features, importance)


def catboost_pool(frame: pd.DataFrame, target_col: str | None, features: list[str]) -> Pool:
    data = frame[features].copy()
    cat_features = [col for col in features if col in CAT_FEATURES or data[col].dtype == "object"]
    for col in cat_features:
        data[col] = data[col].astype("string").fillna("missing")
    labels = frame[target_col] if target_col else None
    return Pool(data, label=labels, cat_features=cat_features)


def fit_catboost(
    target: str,
    depth: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    features: list[str],
) -> PredictionBundle:
    model = CatBoostRegressor(
        iterations=450,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=6,
        loss_function="RMSE",
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(catboost_pool(train, target_col, features), eval_set=catboost_pool(val, target_col, features), use_best_model=True)
    preds = [np.maximum(model.predict(catboost_pool(frame, None, features)), 0) for frame in [train, val, test]]
    importance = pd.DataFrame({"feature": features, "importance": model.get_feature_importance(catboost_pool(train, target_col, features))})
    importance["target"] = target
    importance["depth"] = depth
    importance["candidate"] = f"{target}_{depth}_catboost"
    return PredictionBundle(target, depth, f"{target}_{depth}_catboost", "catboost", "catboost", "ocean_mechanism", *preds, features, importance)


def metric_dict(frame: pd.DataFrame, target_col: str, pred: np.ndarray, p75: float, p90: float) -> dict[str, float]:
    y = frame[target_col]
    high90 = y >= p90
    metrics = evaluate_predictions(y, pred)
    metrics["bias"] = float(np.mean(pred - y))
    metrics["high90_rmse"] = rmse(y[high90], pred[high90]) if high90.any() else np.nan
    metrics["high90_underestimation_rate"] = float(np.mean(pred[high90] < y[high90])) if high90.any() else np.nan
    return metrics


def rows_for(bundle: PredictionBundle, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_col: str) -> list[dict[str, Any]]:
    p75, p90 = float(train[target_col].quantile(0.75)), float(train[target_col].quantile(0.90))
    rows = []
    for split, frame, pred in [("train", train, bundle.train_pred), ("validation", val, bundle.validation_pred), ("test", test, bundle.test_pred)]:
        rows.append(
            {
                "target": bundle.target,
                "depth": bundle.depth,
                "split": split,
                "candidate": bundle.candidate,
                "strategy": bundle.strategy,
                "model_family": bundle.model_family,
                "n_features": len(bundle.features),
                "n_samples": len(frame),
                **metric_dict(frame, target_col, pred, p75, p90),
            }
        )
    return rows


def prediction_frame(bundle: PredictionBundle, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = []
    for split, frame, pred in [("train", train, bundle.train_pred), ("validation", val, bundle.validation_pred), ("test", test, bundle.test_pred)]:
        tmp = frame[IDENTITY_COLUMNS].copy()
        tmp["target"] = bundle.target
        tmp["split"] = split
        tmp["candidate"] = bundle.candidate
        tmp["strategy"] = bundle.strategy
        tmp["y_true"] = frame[target_col].values
        tmp["y_pred"] = pred
        out.append(tmp)
    return pd.concat(out, ignore_index=True)


def time_aware_oof(
    target: str,
    depth: str,
    family: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oof = np.full(len(train), np.nan)
    years = [2017, 2018, 2019, 2020]
    for year in years:
        fit_mask = train["year"] < year
        pred_mask = train["year"] == year
        if fit_mask.sum() < 500 or pred_mask.sum() == 0:
            continue
        if family == "catboost":
            model = fit_catboost(target, depth, train[fit_mask], train[pred_mask], train[pred_mask], target_col, features)
            oof[np.where(pred_mask)[0]] = model.validation_pred
        else:
            pipe = Pipeline([("preprocessor", sklearn_preprocessor(train[fit_mask], features)), ("model", sklearn_model(family))])
            pipe.fit(train.loc[fit_mask, features], train.loc[fit_mask, target_col])
            oof[np.where(pred_mask)[0]] = np.maximum(pipe.predict(train.loc[pred_mask, features]), 0)
    missing = np.isnan(oof)
    if missing.any():
        fill_model = fit_sklearn_candidate("lightgbm", target, depth, "fill", "fill", train[~missing], train[missing], train[missing], target_col, features)
        oof[missing] = fill_model.validation_pred
    if family == "catboost":
        full = fit_catboost(target, depth, train, val, test, target_col, features)
        return oof, full.validation_pred, full.test_pred
    pipe = Pipeline([("preprocessor", sklearn_preprocessor(train, features)), ("model", sklearn_model(family))])
    pipe.fit(train[features], train[target_col])
    return oof, np.maximum(pipe.predict(val[features]), 0), np.maximum(pipe.predict(test[features]), 0)


def stacking_bundle(target: str, depth: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_col: str, features: list[str]) -> tuple[PredictionBundle, pd.DataFrame]:
    families = ["lightgbm", "xgboost", "random_forest", "catboost"]
    train_meta, val_meta, test_meta = [], [], []
    oof_rows = []
    for family in families:
        oof, val_pred, test_pred = time_aware_oof(target, depth, family, train, val, test, target_col, features)
        train_meta.append(oof)
        val_meta.append(val_pred)
        test_meta.append(test_pred)
        oof_rows.append(pd.DataFrame({"target": target, "depth": depth, "model_family": family, "row_index": train.index, "oof_pred": oof}))
    x_train = np.vstack(train_meta).T
    x_val = np.vstack(val_meta).T
    x_test = np.vstack(test_meta).T
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    meta_models = {
        "ridge": Ridge(alpha=1.0),
        "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=42, max_iter=10000),
    }
    best_name, best_model, best_r2 = "", None, -np.inf
    for name, model in meta_models.items():
        model.fit(x_train_s, train[target_col])
        pred = np.maximum(model.predict(x_val_s), 0)
        score = r2_score(val[target_col], pred)
        if score > best_r2:
            best_name, best_model, best_r2 = name, model, score
    assert best_model is not None
    bundle = PredictionBundle(
        target,
        depth,
        f"{target}_{depth}_stacking_{best_name}",
        "stacking",
        "stacking",
        "ocean_mechanism",
        np.maximum(best_model.predict(x_train_s), 0),
        np.maximum(best_model.predict(x_val_s), 0),
        np.maximum(best_model.predict(x_test_s), 0),
        features,
        pd.DataFrame(columns=["feature", "importance", "target", "depth", "candidate"]),
    )
    return bundle, pd.concat(oof_rows, ignore_index=True)


def residual_bundle(base: PredictionBundle, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_col: str, features: list[str]) -> PredictionBundle:
    residual = val[target_col].to_numpy() - base.validation_pred
    residual_features = [col for col in ["station", "water_zone_en", "month", "is_wet_season", "oc_chl", "oc_zsd", "oc_kd490", "valid_count_7d"] if col in features]
    train_res = val.copy()
    train_res["base_pred"] = base.validation_pred
    val_like = val.copy()
    test_like = test.copy()
    val_like["base_pred"] = base.validation_pred
    test_like["base_pred"] = base.test_pred
    residual_features = ["base_pred"] + residual_features
    pipe = Pipeline([("preprocessor", sklearn_preprocessor(train_res, residual_features)), ("model", LGBMRegressor(n_estimators=180, learning_rate=0.03, num_leaves=8, max_depth=3, random_state=42, n_jobs=1, verbose=-1))])
    pipe.fit(train_res[residual_features], residual)
    # Train prediction is not corrected because residual model is trained on validation residuals only.
    corrected_val = np.maximum(base.validation_pred + pipe.predict(val_like[residual_features]), 0)
    corrected_test = np.maximum(base.test_pred + pipe.predict(test_like[residual_features]), 0)
    return PredictionBundle(base.target, base.depth, base.candidate + "_residual_corrected", "residual_correction", "lightgbm_residual", base.feature_group, base.train_pred, corrected_val, corrected_test, features, pd.DataFrame())


def regime_bundle(target: str, depth: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_col: str, features: list[str]) -> tuple[PredictionBundle, dict[str, Any]]:
    signal_cols = [col for col in ["oc_chl", "oc_zsd", "oc_kd490", f"{target}_lag_trend"] if col in train.columns]
    signal = train[signal_cols].rank(pct=True).mean(axis=1)
    threshold = float(signal.quantile(0.70))
    definition = {"signal_columns": signal_cols, "threshold": threshold, "rule": "rank_mean >= train_p70"}

    def mask(frame: pd.DataFrame) -> pd.Series:
        return frame[signal_cols].rank(pct=True).mean(axis=1) >= threshold

    component_models: dict[str, Pipeline] = {}
    fallback_model: Pipeline | None = None
    preds = {}
    for split, frame in [("train", train), ("validation", val), ("test", test)]:
        preds[split] = np.zeros(len(frame))
    for label, subset_mask in {"high_variability": mask(train), "normal": ~mask(train)}.items():
        subset = train[subset_mask]
        if len(subset) < 500:
            subset = train
        pipe = Pipeline(
            [
                ("preprocessor", sklearn_preprocessor(subset, features)),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=350,
                        learning_rate=0.035,
                        num_leaves=24,
                        max_depth=5,
                        min_child_samples=35,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=1,
                        verbose=-1,
                    ),
                ),
            ]
        )
        pipe.fit(subset[features], subset[target_col])
        component_models[label] = pipe
        if fallback_model is None:
            fallback_model = pipe
        for split, frame, pred in [("train", train, preds["train"]), ("validation", val, preds["validation"]), ("test", test, preds["test"])]:
            current_mask = mask(frame) if label == "high_variability" else ~mask(frame)
            if current_mask.any():
                model = component_models.get(label, fallback_model)
                pred[current_mask.to_numpy()] = np.maximum(model.predict(frame.loc[current_mask, features]), 0)
    bundle = PredictionBundle(target, depth, f"{target}_{depth}_optical_regime", "regime", "lightgbm", "ocean_mechanism", preds["train"], preds["validation"], preds["test"], features, pd.DataFrame())
    return bundle, definition


def error_tables(predictions: pd.DataFrame, output_dir: Path) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    test["error"] = test["y_pred"] - test["y_true"]
    test["abs_error"] = test["error"].abs()
    by_station = test.groupby(["target", "depth", "candidate", "station"], as_index=False).agg(
        n=("error", "size"), bias=("error", "mean"), mae=("abs_error", "mean")
    )
    by_station.to_csv(output_dir / "error_by_station.csv", index=False, encoding="utf-8-sig")
    test["month"] = pd.to_datetime(test["date"]).dt.month
    test["season"] = np.where(test["month"].between(5, 9), "wet", "dry")
    by_season = test.groupby(["target", "depth", "candidate", "season"], as_index=False).agg(
        n=("error", "size"), bias=("error", "mean"), mae=("abs_error", "mean")
    )
    by_season.to_csv(output_dir / "error_by_season.csv", index=False, encoding="utf-8-sig")
    test["target_quantile"] = test.groupby(["target", "depth"])["y_true"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"]))
    by_quantile = test.groupby(["target", "depth", "candidate", "target_quantile"], observed=True, as_index=False).agg(
        n=("error", "size"), bias=("error", "mean"), mae=("abs_error", "mean")
    )
    by_quantile.to_csv(output_dir / "error_by_target_quantile.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    processed_dir = Path(args.processed_dir)
    config_path = Path(args.config_out)
    report_path = Path(args.report_out)
    for path in [out_dir, processed_dir, config_path.parent, report_path.parent]:
        path.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics": out_dir / "metrics.csv",
        "predictions": out_dir / "predictions.csv",
        "feature_importance": out_dir / "feature_importance.csv",
        "permutation_importance": out_dir / "permutation_importance.csv",
        "comparison_to_previous_best": out_dir / "comparison_to_previous_best.csv",
        "catboost_metrics": out_dir / "catboost_metrics.csv",
        "catboost_feature_importance": out_dir / "catboost_feature_importance.csv",
        "catboost_comparison": out_dir / "catboost_comparison_to_best.csv",
        "stacking_oof": out_dir / "stacking_oof_predictions.csv",
        "stacking_metrics": out_dir / "stacking_metrics.csv",
        "stacking_comparison": out_dir / "stacking_comparison_to_best.csv",
        "residual_metrics": out_dir / "residual_correction_metrics.csv",
        "residual_station": out_dir / "residual_error_by_station.csv",
        "residual_season": out_dir / "residual_error_by_season.csv",
        "regime_definition": out_dir / "regime_definition.json",
        "regime_metrics": out_dir / "regime_metrics.csv",
        "regime_comparison": out_dir / "regime_comparison_to_global_model.csv",
        "sequence_metrics": out_dir / "sequence_model_metrics.csv",
        "sequence_windows": out_dir / "sequence_window_comparison.csv",
        "era5_validation": out_dir / "era5_source_validation_report.json",
        "era5_ab": out_dir / "era5_ab_metrics.csv",
        "era5_usage": out_dir / "era5_feature_usage_log.csv",
        "experiment_config": out_dir / "experiment_config.json",
        "experiment_report": out_dir / "experiment_report.md",
        "processed_dataset": processed_dir / f"literature_inspired_precision_boost_dataset_{args.date_tag}.csv",
        "config_copy": config_path,
        "report_copy": report_path,
    }
    for path in outputs.values():
        if args.force and path.exists() and path.is_file():
            path.unlink()
        assert_can_write(path, args.force)

    df = load_data(Path(args.input))
    added_by_target: dict[str, list[str]] = {}
    for target, spec in TARGETS.items():
        df, added = add_target_features(df, target, spec["history_prefix"])
        added_by_target[target] = added
    df.to_csv(outputs["processed_dataset"], index=False, encoding="utf-8-sig")

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    regime_defs: dict[str, Any] = {}

    for target, spec in TARGETS.items():
        target_col = spec["column"]
        target_df = df[df[target_col].notna()].copy()
        features = build_features(target_df, target, spec["history_prefix"], added_by_target[target])
        for depth in DEPTH_ORDER:
            train, val, test = split_frames(target_df, target_col, depth)
            bundles: list[PredictionBundle] = []
            for family in ["lightgbm", "xgboost", "random_forest"]:
                bundles.append(fit_sklearn_candidate(family, target, depth, f"{target}_{depth}_{family}", family, train, val, test, target_col, features))
            bundles.append(fit_catboost(target, depth, train, val, test, target_col, features))
            stack, oof = stacking_bundle(target, depth, train, val, test, target_col, features)
            bundles.append(stack)
            oof_frames.append(oof)
            base = max([b for b in bundles if b.strategy in ["lightgbm", "xgboost", "random_forest", "catboost", "stacking"]], key=lambda b: r2_score(val[target_col], b.validation_pred))
            bundles.append(residual_bundle(base, train, val, test, target_col, features))
            regime, definition = regime_bundle(target, depth, train, val, test, target_col, features)
            bundles.append(regime)
            regime_defs[f"{target}_{depth}"] = definition

            for bundle in bundles:
                metrics_rows.extend(rows_for(bundle, train, val, test, target_col))
                prediction_frames.append(prediction_frame(bundle, train, val, test, target_col))
                if not bundle.importance.empty:
                    importance_frames.append(bundle.importance)

            top_for_perm = max(bundles, key=lambda b: r2_score(val[target_col], b.validation_pred))
            if top_for_perm.model_family in ["lightgbm", "xgboost", "random_forest"]:
                # A small permutation pass for the validation winner.
                pipe = Pipeline([("preprocessor", sklearn_preprocessor(train, top_for_perm.features)), ("model", sklearn_model(top_for_perm.model_family))])
                pipe.fit(train[top_for_perm.features], train[target_col])
                perm = permutation_importance(pipe, val[top_for_perm.features], val[target_col], scoring="r2", n_repeats=3, random_state=42, n_jobs=1)
                importance = pd.DataFrame(
                    {
                        "target": target,
                        "depth": depth,
                        "candidate": top_for_perm.candidate,
                        "feature": top_for_perm.features,
                        "importance_mean": perm.importances_mean[: len(top_for_perm.features)],
                        "importance_std": perm.importances_std[: len(top_for_perm.features)],
                    }
                )
                importance.to_csv(outputs["permutation_importance"], mode="a", header=not outputs["permutation_importance"].exists(), index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    oof_table = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()

    test_metrics = metrics[metrics["split"] == "test"].copy()
    val_metrics = metrics[metrics["split"] == "validation"].copy()
    comparison_rows = []
    for _, row in test_metrics.iterrows():
        prev = TARGETS[row["target"]]["previous_best"][row["depth"]]
        val_r2 = val_metrics[(val_metrics["target"] == row["target"]) & (val_metrics["depth"] == row["depth"]) & (val_metrics["candidate"] == row["candidate"])]["r2"].iloc[0]
        comparison_rows.append(
            {
                **row.to_dict(),
                "validation_r2": val_r2,
                "previous_best_r2": prev["r2"],
                "delta_r2_vs_previous": row["r2"] - prev["r2"],
                "previous_best_rmse": prev["rmse"],
                "delta_rmse_vs_previous": row["rmse"] - prev["rmse"],
                "previous_best_high90_rmse": prev["high90_rmse"],
                "delta_high90_rmse_vs_previous": row["high90_rmse"] - prev["high90_rmse"],
                "model_role": "high_value_auxiliary" if row["r2"] < prev["r2"] - 0.02 and row["high90_rmse"] < prev["high90_rmse"] else "candidate_main",
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    metrics.to_csv(outputs["metrics"], index=False, encoding="utf-8-sig")
    predictions.to_csv(outputs["predictions"], index=False, encoding="utf-8-sig")
    importance.to_csv(outputs["feature_importance"], index=False, encoding="utf-8-sig")
    comparison.to_csv(outputs["comparison_to_previous_best"], index=False, encoding="utf-8-sig")
    oof_table.to_csv(outputs["stacking_oof"], index=False, encoding="utf-8-sig")

    comparison[comparison["strategy"] == "catboost"].to_csv(outputs["catboost_metrics"], index=False, encoding="utf-8-sig")
    importance[importance["candidate"].astype(str).str.contains("catboost", na=False)].to_csv(outputs["catboost_feature_importance"], index=False, encoding="utf-8-sig")
    comparison[comparison["strategy"] == "catboost"].to_csv(outputs["catboost_comparison"], index=False, encoding="utf-8-sig")
    comparison[comparison["strategy"] == "stacking"].to_csv(outputs["stacking_metrics"], index=False, encoding="utf-8-sig")
    comparison[comparison["strategy"] == "stacking"].to_csv(outputs["stacking_comparison"], index=False, encoding="utf-8-sig")
    comparison[comparison["strategy"] == "residual_correction"].to_csv(outputs["residual_metrics"], index=False, encoding="utf-8-sig")
    comparison[comparison["strategy"] == "regime"].to_csv(outputs["regime_metrics"], index=False, encoding="utf-8-sig")
    comparison[comparison["strategy"] == "regime"].to_csv(outputs["regime_comparison"], index=False, encoding="utf-8-sig")
    Path(outputs["regime_definition"]).write_text(json.dumps(regime_defs, ensure_ascii=False, indent=2), encoding="utf-8")

    error_tables(predictions[predictions["strategy"] == "residual_correction"], out_dir)
    (out_dir / "error_by_station.csv").replace(outputs["residual_station"])
    (out_dir / "error_by_season.csv").replace(outputs["residual_season"])
    error_tables(predictions, out_dir)

    failed_rows = [
        {
            "route": "sequence_model",
            "status": "not_run",
            "reason": "隔离环境没有 TensorFlow / PyTorch。为避免引入大型深度学习依赖破坏当前可复现环境，本轮仅登记可行性，不训练 LSTM/GRU/TCN。",
            "next_step": "如需执行，建议单独创建 sequence_models_YYYYMMDD 环境与实验目录。",
        },
        {
            "route": "era5_ab",
            "status": "not_run",
            "reason": "本轮目标优先完成不新增数据源的新方法路线。ERA5 需要先做 source validation，再进入 A/B。",
            "next_step": "运行已有 ERA5 下载/校验链，生成 era5_source_validation_report.json 后再做 A/B。",
        },
    ]
    pd.DataFrame(failed_rows).to_csv(outputs["sequence_metrics"], index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows[:1]).to_csv(outputs["sequence_windows"], index=False, encoding="utf-8-sig")
    Path(outputs["era5_validation"]).write_text(json.dumps(failed_rows[1], ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(failed_rows[1:]).to_csv(outputs["era5_ab"], index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows[1:]).to_csv(outputs["era5_usage"], index=False, encoding="utf-8-sig")

    config = {
        "date_tag": args.date_tag,
        "experiment_name": f"literature_inspired_precision_boost_{args.date_tag}",
        "input": args.input,
        "targets": TARGETS,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "routes": ["catboost", "time_aware_stacking", "residual_correction", "optical_regime", "sequence_feasibility", "era5_feasibility"],
    }
    for path in [outputs["experiment_config"], outputs["config_copy"]]:
        Path(path).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_lines = []
    for (target, depth), sub in comparison.groupby(["target", "depth"]):
        best = sub.sort_values(["r2", "high90_rmse"], ascending=[False, True]).iloc[0]
        aux = sub.sort_values(["high90_rmse", "r2"], ascending=[True, False]).iloc[0]
        best_lines.append(
            f"- `{target}` / `{depth}`: best=`{best.candidate}`, R²={best.r2:.4f}, "
            f"delta={best.delta_r2_vs_previous:+.4f}; aux=`{aux.candidate}`, P90 RMSE={aux.high90_rmse:.4f}"
        )
    report = "# Literature Inspired Precision Boost 20260518\n\n" + "\n".join(best_lines)
    for path in [outputs["experiment_report"], outputs["report_copy"]]:
        Path(path).write_text(report, encoding="utf-8")

    print(json.dumps({"comparison": str(outputs["comparison_to_previous_best"]), "report": str(outputs["experiment_report"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
