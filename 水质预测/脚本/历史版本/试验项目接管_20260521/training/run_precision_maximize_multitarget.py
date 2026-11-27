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
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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
from scripts.training.run_chla_ocean_colour_precision_optimization import (
    find_engineered_features,
)


DATE_TAG = "20260518"
TARGETS = {
    "chla": {
        "column": "chla_ug_l",
        "depths": ["bottom"],
        "previous_best": {
            "bottom": {
                "r2": 0.331780,
                "rmse": 3.518034,
                "high90_rmse": 8.132445,
                "candidate": "impute_knn_bottom",
            }
        },
        "history_prefix": "chla_ug_l",
    },
    "secchi": {
        "column": "secchi_depth_m",
        "depths": DEPTH_ORDER,
        "previous_best": {},
        "history_prefix": "secchi_depth_m",
    },
}
IDENTITY_COLUMNS = ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"]


@dataclass
class CandidateResult:
    target: str
    depth: str
    candidate: str
    strategy: str
    model_family: str
    feature_group: str
    imputation: str
    weight_mode: str
    features: list[str]
    params: dict[str, Any]
    train_pred: np.ndarray
    validation_pred: np.ndarray
    test_pred: np.ndarray
    feature_importance: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多目标 Ocean Colour 精度最大化实验")
    parser.add_argument(
        "--input",
        default="data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_optimization_dataset_20260517.csv",
    )
    parser.add_argument("--date-tag", default=DATE_TAG)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=f"experiments/outputs/precision_maximize_{DATE_TAG}",
    )
    parser.add_argument(
        "--processed-dir",
        default=f"data/processed/precision_maximize_{DATE_TAG}",
    )
    parser.add_argument(
        "--config-out",
        default=f"experiments/configs/precision_maximize_{DATE_TAG}.json",
    )
    parser.add_argument(
        "--report-out",
        default=f"experiments/reports/precision_maximize_{DATE_TAG}.md",
    )
    return parser.parse_args()


def assert_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"输出已存在，请使用 --force 覆盖: {path}")


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b.abs() > 1e-12)


def add_target_history_features(df: pd.DataFrame, target_name: str, target_column: str, history_prefix: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    added: list[str] = []
    lag1 = f"{history_prefix}_lag1"
    lag2 = f"{history_prefix}_lag2"
    roll3 = f"{history_prefix}_roll3"
    if {lag1, lag2}.issubset(out.columns):
        col = f"{target_name}_history_trend_lag1_lag2"
        out[col] = out[lag1] - out[lag2]
        added.append(col)
    if {lag1, roll3}.issubset(out.columns):
        col = f"{target_name}_history_lag1_minus_roll3"
        out[col] = out[lag1] - out[roll3]
        added.append(col)
        col = f"{target_name}_history_lag1_ratio_roll3"
        out[col] = safe_ratio(out[lag1], out[roll3])
        added.append(col)
    if {lag1, "days_since_prev_sample"}.issubset(out.columns):
        col = f"{target_name}_lag1_per_day_gap"
        out[col] = safe_ratio(out[lag1], out["days_since_prev_sample"])
        added.append(col)
    return out.replace([np.inf, -np.inf], np.nan), added


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["split"] = df["year"].map(split_label)
    df = df[df["split"].notna()].copy()
    return df


def available(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in df.columns and df[col].notna().sum() > 0]


def feature_groups(df: pd.DataFrame, target_name: str, history_prefix: str, target_added: list[str]) -> dict[str, list[str]]:
    base_candidates = [
        col for col in BASE_FEATURES if not col.startswith("chla_ug_l_")
    ]
    base_candidates += [f"{history_prefix}_lag1", f"{history_prefix}_lag2", f"{history_prefix}_roll3"]
    base_candidates += target_added
    base = available(df, base_candidates)
    ocean = available(df, OCEAN_COLOUR_FEATURES)
    engineered = available(df, find_engineered_features(df))
    # For Secchi, keep Chl-a target-history engineered features out of the default feature groups.
    if target_name != "chla":
        engineered = [col for col in engineered if not col.startswith("chla_")]
    full = list(dict.fromkeys(base + ocean + engineered))
    stable = [
        "lon",
        "lat",
        "month_sin",
        "month_cos",
        "day_of_year",
        "days_since_prev_sample",
        "water_zone_en",
        f"{history_prefix}_lag1",
        f"{history_prefix}_lag2",
        f"{history_prefix}_roll3",
        f"{target_name}_history_trend_lag1_lag2",
        f"{target_name}_history_lag1_minus_roll3",
        f"{target_name}_lag1_per_day_gap",
        "oc_zsd",
        "oc_zsd_3d",
        "oc_zsd_7d",
        "oc_zsd_14d",
        "oc_kd490",
        "oc_kd490_7d",
        "oc_chl",
        "valid_count_7d",
        "oc_zsd_past3_std",
        "oc_zsd_change_7d_14d",
    ]
    stable = available(df, stable)
    high_missing = [col for col in full if df[col].isna().mean() > 0.70 and col not in stable]
    no_high_missing = [col for col in full if col not in high_missing]
    return {
        "baseline": base,
        "ocean_full": full,
        "stable_core": list(dict.fromkeys(stable)),
        "drop_high_missing": no_high_missing,
    }


def station_interpolate(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.sort_values(["station", "date", "depth", "obs_id"]).copy()
    numeric = [col for col in features if col in out.columns and pd.api.types.is_numeric_dtype(out[col])]
    oc_numeric = [col for col in numeric if col.startswith("oc_") or col == "valid_count_7d"]
    for col in oc_numeric:
        out[col] = out.groupby(["station", "depth"], sort=False)[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
    return out


def preprocessor(train_df: pd.DataFrame, features: list[str], imputation: str) -> ColumnTransformer:
    cat = [col for col in features if train_df[col].dtype == "object" or str(train_df[col].dtype).startswith("category")]
    num = [col for col in features if col not in cat]
    if imputation == "knn":
        num_trans: Any = KNNImputer(n_neighbors=5, weights="distance")
    elif imputation == "native_missing":
        num_trans = "passthrough"
    elif imputation == "median_indicator":
        num_trans = SimpleImputer(strategy="median", add_indicator=True)
    else:
        num_trans = SimpleImputer(strategy="median")
    return ColumnTransformer(
        [
            ("num", num_trans, num),
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
        return LGBMRegressor(random_state=42, n_jobs=1, verbose=-1, **params)
    if family == "xgboost":
        return XGBRegressor(random_state=42, n_jobs=1, objective="reg:squarederror", **params)
    if family == "random_forest":
        return RandomForestRegressor(random_state=42, n_jobs=1, **params)
    raise ValueError(family)


def make_pipeline(train_df: pd.DataFrame, features: list[str], family: str, params: dict[str, Any], imputation: str) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", preprocessor(train_df, features, imputation)),
            ("model", make_model(family, params)),
        ]
    )


def weight_vector(y: pd.Series, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    weights = np.ones(len(y), dtype=float)
    if mode == "p90x2":
        weights[y >= y.quantile(0.90)] = 2.0
    elif mode == "p90x3":
        weights[y >= y.quantile(0.90)] = 3.0
    else:
        raise ValueError(mode)
    return weights


def trial_params(trial: optuna.Trial, family: str) -> dict[str, Any]:
    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 800),
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
    return {
        "n_estimators": trial.suggest_int("n_estimators", 250, 700),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
    }


def run_optuna(
    target_name: str,
    depth: str,
    family: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str,
    features: list[str],
    imputation: str,
    trials: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    p90 = float(train[target_col].quantile(0.90))

    def objective(trial: optuna.Trial) -> float:
        params = trial_params(trial, family)
        pipe = make_pipeline(train, features, family, params, imputation)
        pipe.fit(train[features], train[target_col])
        pred = np.maximum(pipe.predict(val[features]), 0)
        r2 = float(r2_score(val[target_col], pred))
        high = val[target_col] >= p90
        p90_rmse = rmse(val.loc[high, target_col], pred[high]) if high.any() else np.nan
        trial.set_user_attr("validation_r2", r2)
        trial.set_user_attr("validation_p90_rmse", p90_rmse)
        return r2

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{target_name}_{depth}_{family}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    table = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    table["target"] = target_name
    table["depth"] = depth
    table["model_family"] = family
    return study.best_params, table


def evaluate(frame: pd.DataFrame, target_col: str, pred: np.ndarray, p75: float, p90: float) -> dict[str, float]:
    y = frame[target_col]
    high90 = y >= p90
    low = y < p75
    metrics = evaluate_predictions(y, pred)
    metrics["high90_rmse"] = rmse(y[high90], pred[high90]) if high90.any() else np.nan
    metrics["high90_mae"] = float(mean_absolute_error(y[high90], pred[high90])) if high90.any() else np.nan
    metrics["high90_underestimation_rate"] = float(np.mean(pred[high90] < y[high90])) if high90.any() else np.nan
    metrics["low_value_mae"] = float(mean_absolute_error(y[low], pred[low])) if low.any() else np.nan
    return metrics


def fit_candidate(
    *,
    target_name: str,
    depth: str,
    candidate: str,
    strategy: str,
    family: str,
    feature_group: str,
    features: list[str],
    imputation: str,
    weight_mode: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
) -> CandidateResult:
    pipe = make_pipeline(train, features, family, params, imputation)
    weights = weight_vector(train[target_col], weight_mode)
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    pipe.fit(train[features], train[target_col], **fit_kwargs)
    preds = {
        "train": np.maximum(pipe.predict(train[features]), 0),
        "validation": np.maximum(pipe.predict(val[features]), 0),
        "test": np.maximum(pipe.predict(test[features]), 0),
    }
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
        importance = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    else:
        importance = pd.DataFrame(columns=["feature", "importance"])
    importance["target"] = target_name
    importance["depth"] = depth
    importance["candidate"] = candidate
    return CandidateResult(
        target_name,
        depth,
        candidate,
        strategy,
        family,
        feature_group,
        imputation,
        weight_mode,
        features,
        params,
        preds["train"],
        preds["validation"],
        preds["test"],
        importance,
    )


def metrics_rows(result: CandidateResult, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_col: str) -> list[dict[str, Any]]:
    p75 = float(train[target_col].quantile(0.75))
    p90 = float(train[target_col].quantile(0.90))
    rows = []
    for split, frame, pred in [
        ("train", train, result.train_pred),
        ("validation", val, result.validation_pred),
        ("test", test, result.test_pred),
    ]:
        rows.append(
            {
                "target": result.target,
                "depth": result.depth,
                "split": split,
                "candidate": result.candidate,
                "strategy": result.strategy,
                "model_family": result.model_family,
                "feature_group": result.feature_group,
                "imputation": result.imputation,
                "weight_mode": result.weight_mode,
                "n_features": len(result.features),
                "n_samples": len(frame),
                **evaluate(frame, target_col, pred, p75, p90),
            }
        )
    return rows


def prediction_rows(result: CandidateResult, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_col: str) -> pd.DataFrame:
    frames = []
    for split, frame, pred in [
        ("train", train, result.train_pred),
        ("validation", val, result.validation_pred),
        ("test", test, result.test_pred),
    ]:
        out = frame[IDENTITY_COLUMNS].copy()
        out["target"] = result.target
        out["split"] = split
        out["candidate"] = result.candidate
        out["strategy"] = result.strategy
        out["y_true"] = frame[target_col].values
        out["y_pred"] = pred
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def blend(result_a: CandidateResult, result_b: CandidateResult, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_col: str) -> CandidateResult:
    best_w = 0.5
    best_r2 = -np.inf
    for w in [0.25, 0.35, 0.50, 0.65, 0.75]:
        pred = w * result_a.validation_pred + (1 - w) * result_b.validation_pred
        score = r2_score(val[target_col], pred)
        if score > best_r2:
            best_r2 = score
            best_w = w
    return CandidateResult(
        result_a.target,
        result_a.depth,
        f"ensemble_{result_a.model_family}_{result_b.model_family}_{result_a.depth}",
        "ensemble",
        "ensemble",
        f"{result_a.feature_group}+{result_b.feature_group}",
        f"{result_a.imputation}+{result_b.imputation}",
        "none",
        list(dict.fromkeys(result_a.features + result_b.features)),
        {"weight_a": best_w, "a": result_a.candidate, "b": result_b.candidate},
        np.maximum(best_w * result_a.train_pred + (1 - best_w) * result_b.train_pred, 0),
        np.maximum(best_w * result_a.validation_pred + (1 - best_w) * result_b.validation_pred, 0),
        np.maximum(best_w * result_a.test_pred + (1 - best_w) * result_b.test_pred, 0),
        pd.DataFrame(columns=["feature", "importance", "target", "depth", "candidate"]),
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    processed_dir = Path(args.processed_dir)
    config_path = Path(args.config_out)
    report_path = Path(args.report_out)
    for d in [out_dir, processed_dir, config_path.parent, report_path.parent]:
        d.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "metrics": out_dir / "metrics.csv",
        "predictions": out_dir / "predictions.csv",
        "feature_importance": out_dir / "feature_importance.csv",
        "permutation_importance": out_dir / "permutation_importance.csv",
        "optuna_trials": out_dir / "optuna_trials.csv",
        "comparison_to_previous_best": out_dir / "comparison_to_previous_best.csv",
        "experiment_config": out_dir / "experiment_config.json",
        "experiment_report": out_dir / "experiment_report.md",
        "processed_dataset": processed_dir / f"precision_maximize_dataset_{args.date_tag}.csv",
        "config_copy": config_path,
        "report_copy": report_path,
    }
    for p in output_paths.values():
        assert_can_write(p, args.force)

    df = load_dataset(Path(args.input))
    target_added_features: dict[str, list[str]] = {}
    for target_name, spec in TARGETS.items():
        df, added = add_target_history_features(df, target_name, spec["column"], spec["history_prefix"])
        target_added_features[target_name] = added
    df.to_csv(output_paths["processed_dataset"], index=False, encoding="utf-8-sig")

    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_importance: list[pd.DataFrame] = []
    all_trials: list[pd.DataFrame] = []
    all_permutation: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []

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
    rf_params = {"n_estimators": 300, "min_samples_leaf": 2}

    for target_name, spec in TARGETS.items():
        target_col = spec["column"]
        target_df = df[df[target_col].notna()].copy()
        groups = feature_groups(target_df, target_name, spec["history_prefix"], target_added_features.get(target_name, []))
        interpolated = station_interpolate(target_df, groups["ocean_full"])
        for depth in spec["depths"]:
            depth_df = target_df[target_df["depth"] == depth].copy()
            interp_depth = interpolated[interpolated["depth"] == depth].copy()
            train = depth_df[depth_df["split"] == "train"].copy()
            val = depth_df[depth_df["split"] == "validation"].copy()
            test = depth_df[depth_df["split"] == "test"].copy()
            train_i = interp_depth[interp_depth["split"] == "train"].copy()
            val_i = interp_depth[interp_depth["split"] == "validation"].copy()
            test_i = interp_depth[interp_depth["split"] == "test"].copy()

            candidates: list[CandidateResult] = []
            for group_name in ["baseline", "ocean_full", "stable_core", "drop_high_missing"]:
                candidates.append(
                    fit_candidate(
                        target_name=target_name,
                        depth=depth,
                        candidate=f"{target_name}_{depth}_{group_name}_lgbm",
                        strategy="feature_group",
                        family="lightgbm",
                        feature_group=group_name,
                        features=groups[group_name],
                        imputation="median_indicator",
                        weight_mode="none",
                        params=base_lgbm,
                        train=train,
                        val=val,
                        test=test,
                        target_col=target_col,
                    )
                )
            candidates.append(
                fit_candidate(
                    target_name=target_name,
                    depth=depth,
                    candidate=f"{target_name}_{depth}_knn_lgbm",
                    strategy="imputation",
                    family="lightgbm",
                    feature_group="ocean_full",
                    features=groups["ocean_full"],
                    imputation="knn",
                    weight_mode="none",
                    params=base_lgbm,
                    train=train,
                    val=val,
                    test=test,
                    target_col=target_col,
                )
            )
            candidates.append(
                fit_candidate(
                    target_name=target_name,
                    depth=depth,
                    candidate=f"{target_name}_{depth}_station_interpolate_lgbm",
                    strategy="imputation",
                    family="lightgbm",
                    feature_group="ocean_full",
                    features=groups["ocean_full"],
                    imputation="median",
                    weight_mode="none",
                    params=base_lgbm,
                    train=train_i,
                    val=val_i,
                    test=test_i,
                    target_col=target_col,
                )
            )

            for family, base_params in [("lightgbm", base_lgbm), ("xgboost", base_xgb)]:
                best_params, trials = run_optuna(
                    target_name,
                    depth,
                    family,
                    train,
                    val,
                    target_col,
                    groups["ocean_full"],
                    "median",
                    args.trials if family == "lightgbm" else max(15, args.trials // 2),
                )
                all_trials.append(trials)
                candidates.append(
                    fit_candidate(
                        target_name=target_name,
                        depth=depth,
                        candidate=f"{target_name}_{depth}_optuna_{family}",
                        strategy="optuna",
                        family=family,
                        feature_group="ocean_full",
                        features=groups["ocean_full"],
                        imputation="median",
                        weight_mode="none",
                        params=best_params,
                        train=train,
                        val=val,
                        test=test,
                        target_col=target_col,
                    )
                )

            for weight in ["p90x2", "p90x3"]:
                for family, params in [("lightgbm", base_lgbm), ("xgboost", base_xgb)]:
                    candidates.append(
                        fit_candidate(
                            target_name=target_name,
                            depth=depth,
                            candidate=f"{target_name}_{depth}_high_{weight}_{family}",
                            strategy="high_value_weight",
                            family=family,
                            feature_group="ocean_full",
                            features=groups["ocean_full"],
                            imputation="median_indicator",
                            weight_mode=weight,
                            params=params,
                            train=train,
                            val=val,
                            test=test,
                            target_col=target_col,
                        )
                    )

            candidates.append(
                fit_candidate(
                    target_name=target_name,
                    depth=depth,
                    candidate=f"{target_name}_{depth}_rf_stable_core",
                    strategy="random_forest",
                    family="random_forest",
                    feature_group="stable_core",
                    features=groups["stable_core"],
                    imputation="median",
                    weight_mode="none",
                    params=rf_params,
                    train=train,
                    val=val,
                    test=test,
                    target_col=target_col,
                )
            )
            lgbm_best = max([c for c in candidates if c.model_family == "lightgbm"], key=lambda c: r2_score(val[target_col], c.validation_pred))
            xgb_best = max([c for c in candidates if c.model_family == "xgboost"], key=lambda c: r2_score(val[target_col], c.validation_pred))
            rf_best = max([c for c in candidates if c.model_family == "random_forest"], key=lambda c: r2_score(val[target_col], c.validation_pred))
            candidates.extend([blend(lgbm_best, xgb_best, train, val, test, target_col), blend(lgbm_best, rf_best, train, val, test, target_col)])

            for c in candidates:
                all_metrics.extend(metrics_rows(c, train, val, test, target_col))
                all_predictions.append(prediction_rows(c, train, val, test, target_col))
                all_importance.append(c.feature_importance)

            val_metric_rows = pd.DataFrame([r for r in all_metrics if r["target"] == target_name and r["depth"] == depth and r["split"] == "validation"])
            top_for_perm = val_metric_rows.sort_values("r2", ascending=False)["candidate"].head(2).tolist()
            for c in [x for x in candidates if x.candidate in top_for_perm and x.model_family in ["lightgbm", "xgboost"]]:
                pipe = make_pipeline(train, c.features, c.model_family, c.params, c.imputation)
                pipe.fit(train[c.features], train[target_col], **({"model__sample_weight": weight_vector(train[target_col], c.weight_mode)} if c.weight_mode != "none" else {}))
                perm = permutation_importance(pipe, val[c.features], val[target_col], scoring="r2", n_repeats=3, random_state=42, n_jobs=1)
                all_permutation.append(
                    pd.DataFrame(
                        {
                            "target": target_name,
                            "depth": depth,
                            "candidate": c.candidate,
                            "feature": c.features,
                            "importance_mean": perm.importances_mean[: len(c.features)],
                            "importance_std": perm.importances_std[: len(c.features)],
                        }
                    )
                )

    metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    importance = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
    trials = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
    permutation = pd.concat(all_permutation, ignore_index=True) if all_permutation else pd.DataFrame()

    test_metrics = metrics[metrics["split"] == "test"].copy()
    val_metrics = metrics[metrics["split"] == "validation"].copy()
    for _, row in test_metrics.iterrows():
        val_r2 = val_metrics[(val_metrics["target"] == row["target"]) & (val_metrics["depth"] == row["depth"]) & (val_metrics["candidate"] == row["candidate"])]["r2"].iloc[0]
        previous = TARGETS[row["target"]]["previous_best"].get(row["depth"])
        prev_r2 = previous["r2"] if previous else np.nan
        prev_rmse = previous["rmse"] if previous else np.nan
        prev_p90 = previous["high90_rmse"] if previous else np.nan
        baseline = test_metrics[
            (test_metrics["target"] == row["target"])
            & (test_metrics["depth"] == row["depth"])
            & (test_metrics["candidate"].str.contains("_baseline_lgbm"))
        ].head(1)
        baseline_r2 = baseline["r2"].iloc[0] if previous is None and not baseline.empty else prev_r2
        baseline_rmse = baseline["rmse"].iloc[0] if previous is None and not baseline.empty else prev_rmse
        baseline_p90 = baseline["high90_rmse"].iloc[0] if previous is None and not baseline.empty else prev_p90
        comparison_rows.append(
            {
                **row.to_dict(),
                "validation_r2": val_r2,
                "reference_r2": baseline_r2,
                "delta_r2_vs_reference": row["r2"] - baseline_r2 if pd.notna(baseline_r2) else np.nan,
                "reference_rmse": baseline_rmse,
                "delta_rmse_vs_reference": row["rmse"] - baseline_rmse if pd.notna(baseline_rmse) else np.nan,
                "reference_high90_rmse": baseline_p90,
                "delta_high90_rmse_vs_reference": row["high90_rmse"] - baseline_p90 if pd.notna(baseline_p90) else np.nan,
                "suspicious": bool(pd.notna(baseline_r2) and row["r2"] > baseline_r2 and val_r2 < 0),
                "model_role": "high_value_auxiliary"
                if pd.notna(baseline_r2) and row["r2"] < baseline_r2 - 0.02 and pd.notna(baseline_p90) and row["high90_rmse"] < baseline_p90
                else "candidate_main",
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    metrics.to_csv(output_paths["metrics"], index=False, encoding="utf-8-sig")
    predictions.to_csv(output_paths["predictions"], index=False, encoding="utf-8-sig")
    importance.to_csv(output_paths["feature_importance"], index=False, encoding="utf-8-sig")
    permutation.to_csv(output_paths["permutation_importance"], index=False, encoding="utf-8-sig")
    trials.to_csv(output_paths["optuna_trials"], index=False, encoding="utf-8-sig")
    comparison.to_csv(output_paths["comparison_to_previous_best"], index=False, encoding="utf-8-sig")

    recommended = []
    auxiliary = []
    for (target, depth), sub in comparison.groupby(["target", "depth"]):
        main = sub[(sub["model_role"] == "candidate_main") & (~sub["suspicious"])].copy()
        max_r2 = main["r2"].max()
        choice = main[(max_r2 - main["r2"]) <= 0.01].sort_values(["high90_rmse", "n_features"], ascending=[True, True]).head(1)
        if choice.empty:
            choice = main.sort_values("r2", ascending=False).head(1)
        recommended.append(choice.iloc[0].to_dict())
        aux = sub.sort_values(["high90_rmse", "r2"], ascending=[True, False]).head(1)
        auxiliary.append(aux.iloc[0].to_dict())
    recommended_df = pd.DataFrame(recommended)
    auxiliary_df = pd.DataFrame(auxiliary)

    config = {
        "date_tag": args.date_tag,
        "experiment_name": f"precision_maximize_{args.date_tag}",
        "input": args.input,
        "time_splits": TIME_SPLITS,
        "targets": TARGETS,
        "outputs": {k: str(v) for k, v in output_paths.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "rules": [
            "fixed temporal split only",
            "no shuffle split",
            "no same-day other water-quality covariates",
            "test set is only used for final evaluation",
        ],
    }
    for p in [output_paths["experiment_config"], output_paths["config_copy"]]:
        p.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Precision Maximize 20260518",
        "",
        "## 推荐整体主模型",
        "",
    ]
    for row in recommended_df.sort_values(["target", "depth"]).itertuples(index=False):
        lines.append(
            f"- `{row.target}` / `{row.depth}`: `{row.candidate}`，R²={row.r2:.4f}，"
            f"RMSE={row.rmse:.4f}，P90 RMSE={row.high90_rmse:.4f}，"
            f"参考变化={row.delta_r2_vs_reference:+.4f}"
        )
    lines += ["", "## 高值辅助线", ""]
    for row in auxiliary_df.sort_values(["target", "depth"]).itertuples(index=False):
        lines.append(
            f"- `{row.target}` / `{row.depth}`: `{row.candidate}`，R²={row.r2:.4f}，"
            f"P90 RMSE={row.high90_rmse:.4f}，角色=`{row.model_role}`"
        )
    lines += [
        "",
        "## 工程说明",
        "",
        "- 所有结果位于 `experiments/outputs/precision_maximize_20260518/`。",
        "- 未覆盖历史实验，未修改正式 `outputs/modeling/`。",
        "- Chl-a bottom 使用上一轮最佳作为参考；Secchi 使用本轮 baseline 作为参考。",
    ]
    report = "\n".join(lines)
    for p in [output_paths["experiment_report"], output_paths["report_copy"]]:
        p.write_text(report, encoding="utf-8")

    print(json.dumps({"comparison": str(output_paths["comparison_to_previous_best"]), "report": str(output_paths["experiment_report"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
