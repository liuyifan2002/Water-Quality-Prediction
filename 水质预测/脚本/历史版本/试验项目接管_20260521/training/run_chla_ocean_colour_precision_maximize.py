from __future__ import annotations

import argparse
import json
import math
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

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
from scripts.training.run_chla_ocean_colour_precision_optimization import (
    find_engineered_features,
)


DATE_TAG = "20260518"
PREVIOUS_BEST = {
    "surface": {"r2": 0.300275, "rmse": 5.310369, "high90_rmse": 13.834343, "high90_underestimation_rate": 0.816667},
    "middle": {"r2": 0.339657, "rmse": 3.950260, "high90_rmse": 9.176908, "high90_underestimation_rate": 0.829670},
    "bottom": {"r2": 0.328775, "rmse": 3.525935, "high90_rmse": 8.110574, "high90_underestimation_rate": 0.909910},
}
IDENTITY_COLUMNS = ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"]


@dataclass
class Candidate:
    name: str
    depth: str
    model_family: str
    feature_group: str
    imputation: str
    weight_mode: str
    strategy: str
    params: dict[str, Any]
    train_pred: np.ndarray
    validation_pred: np.ndarray
    test_pred: np.ndarray
    train_index: pd.Index
    validation_index: pd.Index
    test_index: pd.Index
    features: list[str]
    feature_importance: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chl-a Ocean Colour 最大精度优化实验")
    parser.add_argument(
        "--dataset",
        default="data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_optimization_dataset_20260517.csv",
    )
    parser.add_argument(
        "--previous-importance",
        default="experiments/outputs/chla_ocean_colour_precision_optimization_20260517/feature_importance.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=f"experiments/outputs/chla_ocean_colour_precision_maximize_{DATE_TAG}",
    )
    parser.add_argument(
        "--processed-dir",
        default=f"data/processed/chla_ocean_colour_precision_maximize_{DATE_TAG}",
    )
    parser.add_argument(
        "--config-out",
        default=f"experiments/configs/chla_ocean_colour_precision_maximize_{DATE_TAG}.json",
    )
    parser.add_argument(
        "--report-out",
        default=f"experiments/reports/chla_ocean_colour_precision_maximize_{DATE_TAG}.md",
    )
    parser.add_argument("--date-tag", default=DATE_TAG)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def assert_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"输出已存在，请使用 --force 覆盖: {path}")


def safe_smape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true.to_numpy()) + np.abs(y_pred)) / 2.0
    mask = denominator > 0
    return float(np.mean(np.abs(y_true.to_numpy()[mask] - y_pred[mask]) / denominator[mask]) * 100) if mask.any() else np.nan


def evaluate_split(split_df: pd.DataFrame, y_pred: np.ndarray, p75: float, p90: float) -> dict[str, float]:
    y_true = split_df[TARGET_COLUMN]
    metrics = evaluate_predictions(y_true, y_pred)
    high75 = y_true >= p75
    high90 = y_true >= p90
    low = y_true < p75
    metrics.update(
        {
            "high75_rmse": rmse(y_true[high75], y_pred[high75]) if high75.any() else np.nan,
            "high75_mae": float(mean_absolute_error(y_true[high75], y_pred[high75])) if high75.any() else np.nan,
            "high90_rmse": rmse(y_true[high90], y_pred[high90]) if high90.any() else np.nan,
            "high90_mae": float(mean_absolute_error(y_true[high90], y_pred[high90])) if high90.any() else np.nan,
            "high90_underestimation_rate": float(np.mean(y_pred[high90] < y_true[high90])) if high90.any() else np.nan,
            "low_value_mae": float(mean_absolute_error(y_true[low], y_pred[low])) if low.any() else np.nan,
            "smape": safe_smape(y_true, y_pred),
        }
    )
    return metrics


def available(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in df.columns and df[col].notna().sum() > 0]


def build_feature_groups(df: pd.DataFrame, previous_importance: Path) -> dict[str, list[str]]:
    base = available(df, BASE_FEATURES)
    ocean = available(df, OCEAN_COLOUR_FEATURES)
    engineered = available(df, find_engineered_features(df))
    full = list(dict.fromkeys(base + ocean + engineered))

    importance_df = pd.read_csv(previous_importance)
    importance_df["clean_feature"] = (
        importance_df["feature"].astype(str).str.replace(r"^(num|cat)__", "", regex=True).str.replace(r"_[^_]+$", "", regex=False)
    )
    stable = (
        importance_df.groupby("clean_feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    top_features = [feature for feature in stable["clean_feature"].tolist() if feature in full]
    mandatory = [
        "lon",
        "lat",
        "month_sin",
        "month_cos",
        "day_of_year",
        "days_since_prev_sample",
        "chla_ug_l_lag1",
        "chla_ug_l_lag2",
        "chla_ug_l_roll3",
        "chla_lag1_per_day_gap",
        "chla_history_trend_lag1_lag2",
        "chla_history_lag1_minus_roll3",
        "oc_chl",
        "oc_zsd_3d",
        "oc_zsd_7d",
        "oc_zsd_past3_std",
        "oc_zsd_change_7d_14d",
        "valid_count_7d",
        "water_zone_en",
    ]
    mandatory = [feature for feature in mandatory if feature in full]

    groups = {
        "full": full,
        "top_30": list(dict.fromkeys(mandatory + top_features[:30]))[:40],
        "top_50": list(dict.fromkeys(mandatory + top_features[:50]))[:65],
        "top_80": list(dict.fromkeys(mandatory + top_features[:80]))[:95],
    }
    high_missing = [col for col in full if df[col].isna().mean() >= 0.70 and col not in mandatory]
    groups["drop_high_missing_low_contrib"] = [col for col in full if col not in high_missing]
    groups["stable_core"] = list(dict.fromkeys(mandatory + [col for col in top_features[:45] if df[col].isna().mean() < 0.65]))
    return groups


def station_interpolated(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    numeric = [col for col in features if col in out.columns and pd.api.types.is_numeric_dtype(out[col])]
    oc_numeric = [col for col in numeric if col.startswith("oc_") or col == "valid_count_7d"]
    if not oc_numeric:
        return out
    out = out.sort_values(["station", "date", "depth", "obs_id"]).copy()
    for col in oc_numeric:
        out[col] = (
            out.groupby(["station", "depth"], sort=False)[col]
            .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
        )
    return out


def make_preprocessor(train_df: pd.DataFrame, features: list[str], imputation: str) -> ColumnTransformer:
    cat = [col for col in features if train_df[col].dtype == "object" or str(train_df[col].dtype).startswith("category")]
    num = [col for col in features if col not in cat]
    if imputation == "knn":
        num_transformer: Any = KNNImputer(n_neighbors=5, weights="distance")
    elif imputation == "median_indicator":
        num_transformer = SimpleImputer(strategy="median", add_indicator=True)
    elif imputation == "native_missing":
        num_transformer = "passthrough"
    else:
        num_transformer = SimpleImputer(strategy="median")
    return ColumnTransformer(
        [
            ("num", num_transformer, num),
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


def make_model(family: str, params: dict[str, Any]) -> Any:
    if family == "lightgbm":
        defaults = {"random_state": 42, "n_jobs": 1, "verbose": -1}
        return LGBMRegressor(**{**defaults, **params})
    if family == "xgboost":
        defaults = {"random_state": 42, "n_jobs": 1, "objective": "reg:squarederror"}
        return XGBRegressor(**{**defaults, **params})
    if family == "random_forest":
        defaults = {"random_state": 42, "n_jobs": 1}
        return RandomForestRegressor(**{**defaults, **params})
    raise ValueError(f"未知模型: {family}")


def make_pipeline(train_df: pd.DataFrame, features: list[str], family: str, params: dict[str, Any], imputation: str) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor(train_df, features, imputation)), ("model", make_model(family, params))])


def weights_for(y: pd.Series, mode: str) -> np.ndarray | None:
    weights = np.ones(len(y), dtype=float)
    if mode == "none":
        return None
    if mode.startswith("p90x"):
        factor = float(mode.replace("p90x", ""))
        weights[y >= y.quantile(0.90)] = factor
        return weights
    raise ValueError(mode)


def fit_candidate(
    *,
    name: str,
    strategy: str,
    depth: str,
    family: str,
    params: dict[str, Any],
    feature_group: str,
    features: list[str],
    imputation: str,
    weight_mode: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Candidate:
    pipeline = make_pipeline(train_df, features, family, params, imputation)
    weights = weights_for(train_df[TARGET_COLUMN], weight_mode)
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    pipeline.fit(train_df[features], train_df[TARGET_COLUMN], **fit_kwargs)
    preds = {}
    for label, frame in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        preds[label] = np.maximum(pipeline.predict(frame[features]), 0)
    importance = extract_feature_importance(pipeline)
    importance["candidate"] = name
    importance["depth"] = depth
    importance["model_family"] = family
    importance["feature_group"] = feature_group
    return Candidate(
        name=name,
        depth=depth,
        model_family=family,
        feature_group=feature_group,
        imputation=imputation,
        weight_mode=weight_mode,
        strategy=strategy,
        params=params,
        train_pred=preds["train"],
        validation_pred=preds["validation"],
        test_pred=preds["test"],
        train_index=train_df.index,
        validation_index=validation_df.index,
        test_index=test_df.index,
        features=features,
        feature_importance=importance,
    )


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame({"feature": names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)


def lgbm_trial_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 250, 900),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 15, 90),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }


def xgb_trial_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 250, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
    }


def run_optuna_search(
    depth: str,
    family: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    features: list[str],
    imputation: str,
    baseline_val_p90_rmse: float,
    n_trials: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    p90 = float(train_df[TARGET_COLUMN].quantile(0.90))

    def objective(trial: optuna.Trial) -> float:
        params = lgbm_trial_params(trial) if family == "lightgbm" else xgb_trial_params(trial)
        pipe = make_pipeline(train_df, features, family, params, imputation)
        pipe.fit(train_df[features], train_df[TARGET_COLUMN])
        pred = np.maximum(pipe.predict(validation_df[features]), 0)
        r2 = float(r2_score(validation_df[TARGET_COLUMN], pred))
        high90 = validation_df[TARGET_COLUMN] >= p90
        val_p90_rmse = rmse(validation_df.loc[high90, TARGET_COLUMN], pred[high90]) if high90.any() else np.nan
        trial.set_user_attr("validation_r2", r2)
        trial.set_user_attr("validation_p90_rmse", val_p90_rmse)
        if np.isfinite(val_p90_rmse) and val_p90_rmse > baseline_val_p90_rmse * 1.05:
            return r2 - 0.05 * (val_p90_rmse / baseline_val_p90_rmse - 1.05)
        return r2

    study = optuna.create_study(direction="maximize", study_name=f"{depth}_{family}", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials["depth"] = depth
    trials["model_family"] = family
    return study.best_params, trials


def add_metrics(
    rows: list[dict[str, Any]],
    candidate: Candidate,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    p75 = float(train_df[TARGET_COLUMN].quantile(0.75))
    p90 = float(train_df[TARGET_COLUMN].quantile(0.90))
    for split, frame, pred in [
        ("train", train_df, candidate.train_pred),
        ("validation", validation_df, candidate.validation_pred),
        ("test", test_df, candidate.test_pred),
    ]:
        rows.append(
            {
                "depth": candidate.depth,
                "split": split,
                "candidate": candidate.name,
                "strategy": candidate.strategy,
                "model_family": candidate.model_family,
                "feature_group": candidate.feature_group,
                "imputation": candidate.imputation,
                "weight_mode": candidate.weight_mode,
                "n_features": len(candidate.features),
                "n_samples": int(len(frame)),
                **evaluate_split(frame, pred, p75, p90),
            }
        )


def prediction_frame(candidate: Candidate, train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for split, frame, pred in [
        ("train", train_df, candidate.train_pred),
        ("validation", validation_df, candidate.validation_pred),
        ("test", test_df, candidate.test_pred),
    ]:
        out = frame[IDENTITY_COLUMNS].copy()
        out["split"] = split
        out["candidate"] = candidate.name
        out["strategy"] = candidate.strategy
        out["model_family"] = candidate.model_family
        out["feature_group"] = candidate.feature_group
        out["imputation"] = candidate.imputation
        out["weight_mode"] = candidate.weight_mode
        out["y_true"] = frame[TARGET_COLUMN].values
        out["y_pred"] = pred
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def blend_candidate(
    name: str,
    strategy: str,
    depth: str,
    base: Candidate,
    other: Candidate,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    kind: str,
) -> Candidate:
    best_weight = 0.5
    best_gate = None
    best_score = -np.inf
    p75 = float(train_df[TARGET_COLUMN].quantile(0.75))
    high_risk_features = validation_df.get("chla_ug_l_lag1", pd.Series(0, index=validation_df.index)).fillna(0)
    for weight in [0.25, 0.35, 0.50, 0.65, 0.75]:
        if kind == "average":
            pred = weight * base.validation_pred + (1 - weight) * other.validation_pred
            score = r2_score(validation_df[TARGET_COLUMN], pred)
            gate = None
        else:
            for gate in [p75, np.nanpercentile(base.validation_pred, 75), np.nanpercentile(base.validation_pred, 85)]:
                mask = (base.validation_pred >= gate) | (high_risk_features.to_numpy() >= p75)
                pred = base.validation_pred.copy()
                pred[mask] = weight * base.validation_pred[mask] + (1 - weight) * other.validation_pred[mask]
                score = r2_score(validation_df[TARGET_COLUMN], pred)
                if score > best_score:
                    best_score, best_weight, best_gate = score, weight, float(gate)
            continue
        if score > best_score:
            best_score, best_weight, best_gate = score, weight, gate

    def apply(a: np.ndarray, b: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        if kind == "average":
            return np.maximum(best_weight * a + (1 - best_weight) * b, 0)
        hist = frame.get("chla_ug_l_lag1", pd.Series(0, index=frame.index)).fillna(0).to_numpy()
        mask = (a >= best_gate) | (hist >= p75)
        pred = a.copy()
        pred[mask] = best_weight * a[mask] + (1 - best_weight) * b[mask]
        return np.maximum(pred, 0)

    return Candidate(
        name=name,
        depth=depth,
        model_family="ensemble",
        feature_group=f"{base.feature_group}+{other.feature_group}",
        imputation=f"{base.imputation}+{other.imputation}",
        weight_mode=f"{base.weight_mode}+{other.weight_mode}",
        strategy=strategy,
        params={"blend_weight_base": best_weight, "gate": best_gate, "kind": kind, "base": base.name, "other": other.name},
        train_pred=apply(base.train_pred, other.train_pred, train_df),
        validation_pred=apply(base.validation_pred, other.validation_pred, validation_df),
        test_pred=apply(base.test_pred, other.test_pred, test_df),
        train_index=base.train_index,
        validation_index=base.validation_index,
        test_index=base.test_index,
        features=list(dict.fromkeys(base.features + other.features)),
        feature_importance=pd.DataFrame(columns=["feature", "importance"]),
    )


def build_regime_candidate(
    depth: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    feature_group: str,
    regime: str,
) -> Candidate:
    params = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "num_leaves": 20,
        "max_depth": 5,
        "min_child_samples": 35,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
    }
    if regime == "season":
        train_masks = {"wet": train_df["is_wet_season"] == 1, "dry": train_df["is_wet_season"] == 0}
        frame_masks = lambda frame: {"wet": frame["is_wet_season"] == 1, "dry": frame["is_wet_season"] == 0}
    elif regime == "bloom":
        threshold = float(train_df[["chla_ug_l_lag1", "oc_chl"]].max(axis=1).quantile(0.75))
        train_signal = train_df[["chla_ug_l_lag1", "oc_chl", "oc_kd490"]].max(axis=1)
        train_masks = {"bloom": train_signal >= threshold, "normal": train_signal < threshold}

        def frame_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
            signal = frame[["chla_ug_l_lag1", "oc_chl", "oc_kd490"]].max(axis=1)
            return {"bloom": signal >= threshold, "normal": signal < threshold}
    else:
        groups = train_df["water_zone_en"].fillna("unknown").astype(str)
        major = groups.value_counts()[lambda s: s >= 350].index.tolist()
        train_masks = {group: groups == group for group in major}
        train_masks["fallback"] = ~groups.isin(major)

        def frame_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
            g = frame["water_zone_en"].fillna("unknown").astype(str)
            masks = {group: g == group for group in major}
            masks["fallback"] = ~g.isin(major)
            return masks

    models: dict[str, Pipeline] = {}
    for key, mask in train_masks.items():
        subset = train_df[mask]
        if len(subset) < 250:
            subset = train_df
        model = make_pipeline(subset, features, "lightgbm", params, "median_indicator")
        model.fit(subset[features], subset[TARGET_COLUMN])
        models[key] = model

    def predict(frame: pd.DataFrame) -> np.ndarray:
        pred = np.zeros(len(frame))
        masks = frame_masks(frame)
        for key, mask in masks.items():
            if not mask.any():
                continue
            model = models.get(key) or models.get("fallback") or next(iter(models.values()))
            pred[np.asarray(mask)] = model.predict(frame.loc[mask, features])
        return np.maximum(pred, 0)

    return Candidate(
        name=f"stratified_{regime}_{depth}",
        depth=depth,
        model_family="lightgbm",
        feature_group=feature_group,
        imputation="median_indicator",
        weight_mode="none",
        strategy=f"stratified_{regime}",
        params={"regime": regime, **params},
        train_pred=predict(train_df),
        validation_pred=predict(validation_df),
        test_pred=predict(test_df),
        train_index=train_df.index,
        validation_index=validation_df.index,
        test_index=test_df.index,
        features=features,
        feature_importance=pd.DataFrame(columns=["feature", "importance"]),
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    processed_dir = Path(args.processed_dir)
    config_path = Path(args.config_out)
    report_path = Path(args.report_out)
    for directory in [out_dir, processed_dir, config_path.parent, report_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "metrics": out_dir / "metrics.csv",
        "predictions": out_dir / "predictions.csv",
        "feature_importance": out_dir / "feature_importance.csv",
        "permutation_importance": out_dir / "permutation_importance.csv",
        "optuna_trials": out_dir / "optuna_trials.csv",
        "feature_pruning_comparison": out_dir / "feature_pruning_comparison.csv",
        "imputation_strategy_comparison": out_dir / "imputation_strategy_comparison.csv",
        "ensemble_comparison": out_dir / "ensemble_comparison.csv",
        "high_value_tradeoff_comparison": out_dir / "high_value_tradeoff_comparison.csv",
        "comparison_to_previous_best": out_dir / "comparison_to_previous_best.csv",
        "experiment_config": out_dir / "experiment_config.json",
        "experiment_report": out_dir / "experiment_report.md",
        "config_copy": config_path,
        "report_copy": report_path,
        "processed_dataset": processed_dir / f"chla_ocean_colour_precision_maximize_dataset_{args.date_tag}.csv",
    }
    for path in outputs.values():
        assert_can_write(path, args.force)

    df = pd.read_csv(args.dataset, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["split"] = df["year"].map(split_label)
    df = df[df["split"].notna() & df[TARGET_COLUMN].notna()].copy()
    df.to_csv(outputs["processed_dataset"], index=False, encoding="utf-8-sig")

    feature_groups = build_feature_groups(df, Path(args.previous_importance))
    interpolated_df = station_interpolated(df, feature_groups["full"])

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_importance: list[pd.DataFrame] = []
    all_trials: list[pd.DataFrame] = []
    all_permutation: list[pd.DataFrame] = []
    all_candidates: dict[str, list[Candidate]] = {depth: [] for depth in DEPTH_ORDER}

    base_lgbm = {
        "n_estimators": 600,
        "learning_rate": 0.035,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 45,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_split_gain": 0.0,
    }
    base_xgb = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.5,
        "gamma": 0.0,
    }
    rf_params = {"n_estimators": 350, "min_samples_leaf": 2}

    for depth in DEPTH_ORDER:
        depth_df = df[df["depth"] == depth].copy()
        depth_interp_df = interpolated_df[interpolated_df["depth"] == depth].copy()
        train_df = depth_df[depth_df["split"] == "train"].copy()
        validation_df = depth_df[depth_df["split"] == "validation"].copy()
        test_df = depth_df[depth_df["split"] == "test"].copy()
        train_interp = depth_interp_df[depth_interp_df["split"] == "train"].copy()
        validation_interp = depth_interp_df[depth_interp_df["split"] == "validation"].copy()
        test_interp = depth_interp_df[depth_interp_df["split"] == "test"].copy()

        reference_features = feature_groups["full"] if depth in ["surface", "middle"] else feature_groups["full"]
        engineered_features = feature_groups["full"] if depth != "bottom" else feature_groups["full"]

        baseline_candidate = fit_candidate(
            name=f"baseline_lgbm_regularized_{depth}",
            strategy="baseline_reference",
            depth=depth,
            family="lightgbm",
            params=base_lgbm,
            feature_group="full",
            features=reference_features,
            imputation="median",
            weight_mode="none",
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
        )
        all_candidates[depth].append(baseline_candidate)
        p90 = float(train_df[TARGET_COLUMN].quantile(0.90))
        val_high90 = validation_df[TARGET_COLUMN] >= p90
        baseline_val_p90_rmse = rmse(validation_df.loc[val_high90, TARGET_COLUMN], baseline_candidate.validation_pred[val_high90])

        for family in ["lightgbm", "xgboost"]:
            params, trials = run_optuna_search(
                depth=depth,
                family=family,
                train_df=train_df,
                validation_df=validation_df,
                features=reference_features,
                imputation="median",
                baseline_val_p90_rmse=baseline_val_p90_rmse,
                n_trials=args.trials if family == "lightgbm" else max(25, args.trials // 2),
            )
            trials["feature_group"] = "full"
            trials["imputation"] = "median"
            all_trials.append(trials)
            cand = fit_candidate(
                name=f"optuna_{family}_{depth}",
                strategy="optuna",
                depth=depth,
                family=family,
                params=params,
                feature_group="full",
                features=reference_features,
                imputation="median",
                weight_mode="none",
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
            )
            all_candidates[depth].append(cand)

        for group_name in ["top_30", "top_50", "top_80", "drop_high_missing_low_contrib", "stable_core"]:
            cand = fit_candidate(
                name=f"prune_{group_name}_{depth}",
                strategy="feature_pruning",
                depth=depth,
                family="lightgbm",
                params=base_lgbm,
                feature_group=group_name,
                features=feature_groups[group_name],
                imputation="median_indicator",
                weight_mode="none",
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
            )
            all_candidates[depth].append(cand)

        for imputation, tr, va, te in [
            ("median", train_df, validation_df, test_df),
            ("station_interpolate", train_interp, validation_interp, test_interp),
            ("knn", train_df, validation_df, test_df),
            ("native_missing", train_df, validation_df, test_df),
        ]:
            actual_imputation = "native_missing" if imputation == "native_missing" else ("median" if imputation == "station_interpolate" else imputation)
            cand = fit_candidate(
                name=f"impute_{imputation}_{depth}",
                strategy="imputation",
                depth=depth,
                family="lightgbm",
                params=base_lgbm,
                feature_group="full",
                features=engineered_features,
                imputation=actual_imputation,
                weight_mode="none",
                train_df=tr,
                validation_df=va,
                test_df=te,
            )
            all_candidates[depth].append(cand)

        for factor in ["p90x1.5", "p90x2.0", "p90x3.0"]:
            for family, params in [("lightgbm", base_lgbm), ("xgboost", base_xgb)]:
                cand = fit_candidate(
                    name=f"high_value_{factor}_{family}_{depth}",
                    strategy="high_value_weight",
                    depth=depth,
                    family=family,
                    params=params,
                    feature_group="full",
                    features=engineered_features,
                    imputation="median_indicator",
                    weight_mode=factor,
                    train_df=train_df,
                    validation_df=validation_df,
                    test_df=test_df,
                )
                all_candidates[depth].append(cand)

        rf = fit_candidate(
            name=f"rf_reference_{depth}",
            strategy="reference_rf",
            depth=depth,
            family="random_forest",
            params=rf_params,
            feature_group="stable_core",
            features=feature_groups["stable_core"],
            imputation="median",
            weight_mode="none",
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
        )
        all_candidates[depth].append(rf)

        for regime in ["season", "bloom", "station_group"]:
            all_candidates[depth].append(
                build_regime_candidate(depth, train_df, validation_df, test_df, feature_groups["stable_core"], "stable_core", regime)
            )

        current = list(all_candidates[depth])
        lgbm_best = max([c for c in current if c.model_family == "lightgbm"], key=lambda c: r2_score(validation_df[TARGET_COLUMN], c.validation_pred))
        xgb_best = max([c for c in current if c.model_family == "xgboost"], key=lambda c: r2_score(validation_df[TARGET_COLUMN], c.validation_pred))
        rf_best = max([c for c in current if c.model_family == "random_forest"], key=lambda c: r2_score(validation_df[TARGET_COLUMN], c.validation_pred))
        high_aux = min([c for c in current if c.strategy == "high_value_weight"], key=lambda c: evaluate_split(validation_df, c.validation_pred, float(train_df[TARGET_COLUMN].quantile(0.75)), p90)["high90_rmse"])
        all_candidates[depth].extend(
            [
                blend_candidate(f"ensemble_lgbm_xgb_avg_{depth}", "ensemble_average", depth, lgbm_best, xgb_best, train_df, validation_df, test_df, "average"),
                blend_candidate(f"ensemble_lgbm_rf_avg_{depth}", "ensemble_average", depth, lgbm_best, rf_best, train_df, validation_df, test_df, "average"),
                blend_candidate(f"ensemble_gated_high_aux_{depth}", "gated_blending", depth, lgbm_best, high_aux, train_df, validation_df, test_df, "gated"),
            ]
        )

        for cand in all_candidates[depth]:
            add_metrics(all_metrics, cand, train_df, validation_df, test_df)
            all_predictions.append(prediction_frame(cand, train_df, validation_df, test_df))
            if not cand.feature_importance.empty:
                all_importance.append(cand.feature_importance)

        # permutation importance only for validation top 3 per depth to keep runtime bounded.
        depth_metrics = pd.DataFrame([r for r in all_metrics if r["depth"] == depth and r["split"] == "validation"])
        top_names = depth_metrics.sort_values("r2", ascending=False)["candidate"].head(3).tolist()
        for cand in [c for c in all_candidates[depth] if c.name in top_names and c.model_family in {"lightgbm", "xgboost"}]:
            pipe = make_pipeline(train_df, cand.features, cand.model_family, cand.params, cand.imputation)
            pipe.fit(train_df[cand.features], train_df[TARGET_COLUMN], **({"model__sample_weight": weights_for(train_df[TARGET_COLUMN], cand.weight_mode)} if cand.weight_mode != "none" else {}))
            perm = permutation_importance(pipe, validation_df[cand.features], validation_df[TARGET_COLUMN], scoring="r2", n_repeats=5, random_state=42, n_jobs=1)
            all_permutation.append(
                pd.DataFrame(
                    {
                        "depth": depth,
                        "candidate": cand.name,
                        "feature": cand.features,
                        "importance_mean": perm.importances_mean[: len(cand.features)],
                        "importance_std": perm.importances_std[: len(cand.features)],
                    }
                ).sort_values("importance_mean", ascending=False)
            )

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    feature_importance_df = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
    optuna_trials_df = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
    permutation_df = pd.concat(all_permutation, ignore_index=True) if all_permutation else pd.DataFrame()

    metrics_df.to_csv(outputs["metrics"], index=False, encoding="utf-8-sig")
    predictions_df.to_csv(outputs["predictions"], index=False, encoding="utf-8-sig")
    feature_importance_df.to_csv(outputs["feature_importance"], index=False, encoding="utf-8-sig")
    optuna_trials_df.to_csv(outputs["optuna_trials"], index=False, encoding="utf-8-sig")
    permutation_df.to_csv(outputs["permutation_importance"], index=False, encoding="utf-8-sig")

    test_df_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    validation_df_metrics = metrics_df[metrics_df["split"] == "validation"].copy()
    previous_rows = []
    for _, row in test_df_metrics.iterrows():
        prev = PREVIOUS_BEST[row["depth"]]
        val_r2 = validation_df_metrics[(validation_df_metrics["depth"] == row["depth"]) & (validation_df_metrics["candidate"] == row["candidate"])]["r2"].iloc[0]
        previous_rows.append(
            {
                **row.to_dict(),
                "previous_best_r2": prev["r2"],
                "delta_r2_vs_previous": row["r2"] - prev["r2"],
                "previous_best_rmse": prev["rmse"],
                "delta_rmse_vs_previous": row["rmse"] - prev["rmse"],
                "previous_best_high90_rmse": prev["high90_rmse"],
                "delta_high90_rmse_vs_previous": row["high90_rmse"] - prev["high90_rmse"],
                "previous_best_high90_underestimation_rate": prev["high90_underestimation_rate"],
                "delta_high90_underestimation_rate_vs_previous": row["high90_underestimation_rate"] - prev["high90_underestimation_rate"],
                "validation_r2": val_r2,
                "suspicious": bool(row["r2"] > prev["r2"] and val_r2 < 0),
                "model_role": "high_value_auxiliary" if row["r2"] < prev["r2"] - 0.02 and row["high90_rmse"] < prev["high90_rmse"] else "candidate_main",
            }
        )
    comparison = pd.DataFrame(previous_rows)
    comparison.to_csv(outputs["comparison_to_previous_best"], index=False, encoding="utf-8-sig")

    comparison[comparison["strategy"] == "feature_pruning"].to_csv(outputs["feature_pruning_comparison"], index=False, encoding="utf-8-sig")
    comparison[comparison["strategy"] == "imputation"].to_csv(outputs["imputation_strategy_comparison"], index=False, encoding="utf-8-sig")
    comparison[comparison["strategy"].str.contains("ensemble|gated", regex=True)].to_csv(outputs["ensemble_comparison"], index=False, encoding="utf-8-sig")
    comparison[comparison["strategy"] == "high_value_weight"].to_csv(outputs["high_value_tradeoff_comparison"], index=False, encoding="utf-8-sig")

    recommended_rows = []
    aux_rows = []
    for depth, sub in comparison.groupby("depth"):
        viable = sub[(~sub["suspicious"]) & (sub["model_role"] == "candidate_main")].copy()
        viable["selection_r2_bucket"] = (viable["r2"].max() - viable["r2"]) <= 0.01
        top = viable[viable["selection_r2_bucket"]].sort_values(["high90_rmse", "n_features"], ascending=[True, True]).head(1)
        if top.empty:
            top = viable.sort_values("r2", ascending=False).head(1)
        recommended_rows.append(top.iloc[0].to_dict())
        aux = sub[(sub["model_role"] == "high_value_auxiliary") | (sub["high90_rmse"] == sub["high90_rmse"].min())].sort_values(
            ["high90_rmse", "r2"], ascending=[True, False]
        ).head(1)
        aux_rows.append(aux.iloc[0].to_dict())
    recommended = pd.DataFrame(recommended_rows)
    auxiliary = pd.DataFrame(aux_rows)

    config = {
        "date_tag": args.date_tag,
        "experiment_name": f"chla_ocean_colour_precision_maximize_{args.date_tag}",
        "dataset": args.dataset,
        "time_splits": TIME_SPLITS,
        "trials_requested": args.trials,
        "previous_best": PREVIOUS_BEST,
        "feature_groups": {k: v for k, v in feature_groups.items()},
        "outputs": {k: str(v) for k, v in outputs.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "rules": [
            "Optuna only optimizes validation R2 with P90 RMSE penalty; test is untouched until final evaluation.",
            "No random split, no shuffle split, no same-day other water-quality covariates.",
            "High-value models with test R2 drop > 0.02 are marked high_value_auxiliary.",
        ],
    }
    for path in [outputs["experiment_config"], outputs["config_copy"]]:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Chl-a Ocean Colour 最大精度优化实验",
        "",
        f"- 日期标签：`{args.date_tag}`",
        f"- Optuna trials：LightGBM 每层 `{args.trials}`，XGBoost 每层 `{max(25, args.trials // 2)}`",
        f"- 数据集：`{args.dataset}`",
        "",
        "## 推荐整体预测主模型",
        "",
    ]
    for row in recommended.itertuples(index=False):
        lines.append(
            f"- `{row.depth}`: `{row.candidate}`，test R²={row.r2:.4f}，RMSE={row.rmse:.4f}，"
            f"P90 RMSE={row.high90_rmse:.4f}，相对上一轮 R²={row.delta_r2_vs_previous:+.4f}"
        )
    lines += ["", "## 推荐高值风险辅助模型", ""]
    for row in auxiliary.itertuples(index=False):
        lines.append(
            f"- `{row.depth}`: `{row.candidate}`，P90 RMSE={row.high90_rmse:.4f}，"
            f"R²={row.r2:.4f}，角色=`{row.model_role}`"
        )
    lines += [
        "",
        "## 失败或风险说明",
        "",
        "- 若某模型降低 P90 RMSE 但 test R² 下降超过 0.02，仅作为 high_value_auxiliary，不进入整体主线。",
        "- 若某模型 test 提升但 validation 不提升，标记 suspicious，不进入主线。",
        "- 本轮未修改正式 outputs/modeling/，所有结果保存在 experiments/outputs/。",
    ]
    report_text = "\n".join(lines)
    outputs["experiment_report"].write_text(report_text, encoding="utf-8")
    outputs["report_copy"].write_text(report_text, encoding="utf-8")

    print(json.dumps({"comparison": str(outputs["comparison_to_previous_best"]), "report": str(outputs["experiment_report"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
