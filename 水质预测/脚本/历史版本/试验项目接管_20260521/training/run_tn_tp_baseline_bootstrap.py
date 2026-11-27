from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = "20260521"

TIME_SPLIT_MAP = {
    "train": list(range(2014, 2021)),
    "validation": [2021, 2022],
    "test": [2023, 2024],
}

DEPTHS = ["surface", "middle", "bottom"]
IDENTITY_COLUMNS = ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"]

TARGETS = {
    "tn": {"column": "tn_mg_l", "history_prefix": "tn_mg_l"},
    "tp": {"column": "tp_mg_l", "history_prefix": "tp_mg_l"},
}

BASE_FEATURES = [
    "lon",
    "lat",
    "month",
    "quarter",
    "day_of_year",
    "month_sin",
    "month_cos",
    "year_offset",
    "is_wet_season",
    "days_since_prev_sample",
    "station",
    "water_zone_en",
    "depth",
    "depth_is_surface",
    "depth_is_middle",
    "depth_is_bottom",
]

OCEAN_CONTEXT_FEATURES = [
    "oc_chl",
    "oc_kd490",
    "oc_zsd",
    "oc_rrs_443",
    "oc_rrs_490",
    "oc_rrs_560",
    "oc_rrs_665",
    "oc_chl_3d",
    "oc_chl_7d",
    "oc_kd490_3d",
    "oc_kd490_7d",
    "oc_zsd_3d",
    "oc_zsd_7d",
    "valid_count_7d",
    "oc_valid",
    "tide_range_proxy",
    "surface_temp",
    "temp_7d_mean",
    "temp_14d_mean",
    "wind_speed_3d_mean",
    "wind_speed_7d_mean",
    "wind_speed_14d_mean",
    "wind_mixing_proxy",
    "precip_3d_sum",
    "precip_7d_sum",
    "storm_event_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 TN/TP 统一 baseline bootstrap")
    parser.add_argument(
        "--input",
        default="data/interim/current/current_model_dataset_with_highres_and_mechanism.csv",
    )
    parser.add_argument(
        "--metrics-out",
        default="experiments/outputs/current/tn_tp_baseline_bootstrap_metrics.csv",
    )
    parser.add_argument(
        "--scorecard-out",
        default="data/processed/current/tn_tp_baseline_bootstrap_scorecard.csv",
    )
    parser.add_argument(
        "--report-out",
        default="experiments/reports/current/tn_tp_baseline_bootstrap_report.md",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def split_label(year: int) -> str | None:
    for label, years in TIME_SPLIT_MAP.items():
        if year in years:
            return label
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["split"] = df["year"].map(split_label)
    return df[df["split"].notna()].copy()


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b.abs() > 1e-12)


def add_target_history_features(df: pd.DataFrame, target_name: str, history_prefix: str) -> tuple[pd.DataFrame, list[str]]:
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


def available(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in df.columns and df[col].notna().sum() > 0]


def feature_groups(df: pd.DataFrame, target_name: str, history_prefix: str, target_added: list[str]) -> dict[str, list[str]]:
    history_cols = [f"{history_prefix}_lag1", f"{history_prefix}_lag2", f"{history_prefix}_roll3"]
    baseline = available(df, BASE_FEATURES + history_cols + target_added)
    stable_core = available(
        df,
        BASE_FEATURES
        + history_cols
        + target_added
        + [
            "oc_kd490",
            "oc_zsd",
            "oc_zsd_3d",
            "oc_zsd_7d",
            "valid_count_7d",
            "tide_range_proxy",
            "surface_temp",
            "temp_7d_mean",
            "wind_speed_7d_mean",
            "precip_7d_sum",
        ],
    )
    mechanism = available(df, BASE_FEATURES + history_cols + target_added + OCEAN_CONTEXT_FEATURES)
    return {
        "baseline": list(dict.fromkeys(baseline)),
        "stable_core": list(dict.fromkeys(stable_core)),
        "mechanism_context": list(dict.fromkeys(mechanism)),
    }


def build_preprocessor(train_df: pd.DataFrame, features: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    cat = [col for col in features if train_df[col].dtype == "object" or str(train_df[col].dtype).startswith("category")]
    num = [col for col in features if col not in cat]
    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
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
    return preprocessor, num, cat


def make_model(model_name: str, seed: int) -> Any:
    if model_name == "lightgbm":
        return LGBMRegressor(
            random_state=seed,
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.2,
            n_jobs=1,
            verbose=-1,
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            random_state=seed,
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=1,
        )
    raise ValueError(model_name)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def fit_and_predict(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    features: list[str],
    model_name: str,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    preprocessor, _, _ = build_preprocessor(train_df, features)
    model = make_model(model_name, seed)
    pipeline = Pipeline([("pre", preprocessor), ("model", model)])
    pipeline.fit(train_df[features], train_df[target_col])
    valid_pred = pipeline.predict(valid_df[features])
    test_pred = pipeline.predict(test_df[features])
    return evaluate(valid_df[target_col].to_numpy(), valid_pred), evaluate(test_df[target_col].to_numpy(), test_pred)


def main() -> None:
    args = parse_args()
    df = load_dataset(PROJECT_ROOT / args.input)

    metrics_rows: list[dict[str, Any]] = []
    scorecard_rows: list[dict[str, Any]] = []
    report_lines = [
        "# TN/TP baseline bootstrap 报告",
        "",
        "本轮目标：在不引入新增模型大调参的前提下，确认 TN / TP 是否已经具备纳入四指标统一框架的 baseline 条件。",
        "",
    ]

    for target_name, cfg in TARGETS.items():
        target_col = cfg["column"]
        history_prefix = cfg["history_prefix"]
        report_lines.append(f"## {target_name.upper()}")
        report_lines.append("")

        for depth in DEPTHS:
            subset = df[(df["depth"] == depth) & df[target_col].notna()].copy()
            if subset.empty:
                scorecard_rows.append(
                    {
                        "target": target_name,
                        "depth_layer": depth,
                        "status": "not_ready",
                        "best_candidate": "",
                        "best_model": "",
                        "best_feature_set": "",
                        "validation_r2": np.nan,
                        "test_r2": np.nan,
                        "test_rmse": np.nan,
                        "test_mae": np.nan,
                        "reason": "target_column_missing_or_empty",
                    }
                )
                continue

            subset, added = add_target_history_features(subset, target_name, history_prefix)
            groups = feature_groups(subset, target_name, history_prefix, added)
            train_df = subset[subset["split"] == "train"].copy()
            valid_df = subset[subset["split"] == "validation"].copy()
            test_df = subset[subset["split"] == "test"].copy()

            if train_df.empty or valid_df.empty or test_df.empty:
                scorecard_rows.append(
                    {
                        "target": target_name,
                        "depth_layer": depth,
                        "status": "not_ready",
                        "best_candidate": "",
                        "best_model": "",
                        "best_feature_set": "",
                        "validation_r2": np.nan,
                        "test_r2": np.nan,
                        "test_rmse": np.nan,
                        "test_mae": np.nan,
                        "reason": "split_missing",
                    }
                )
                continue

            best: dict[str, Any] | None = None
            for feature_set_name, features in groups.items():
                if not features:
                    continue
                for model_name in ["lightgbm", "random_forest"]:
                    valid_metrics, test_metrics = fit_and_predict(
                        train_df=train_df,
                        valid_df=valid_df,
                        test_df=test_df,
                        target_col=target_col,
                        features=features,
                        model_name=model_name,
                        seed=args.random_seed,
                    )
                    row = {
                        "target": target_name,
                        "depth_layer": depth,
                        "candidate": f"{target_name}_{depth}_{feature_set_name}_{model_name}",
                        "model_name": model_name,
                        "feature_set_name": feature_set_name,
                        "feature_count": len(features),
                        "n_train": len(train_df),
                        "n_valid": len(valid_df),
                        "n_test": len(test_df),
                        "validation_r2": valid_metrics["r2"],
                        "validation_rmse": valid_metrics["rmse"],
                        "validation_mae": valid_metrics["mae"],
                        "test_r2": test_metrics["r2"],
                        "test_rmse": test_metrics["rmse"],
                        "test_mae": test_metrics["mae"],
                        "test_bias": test_metrics["bias"],
                    }
                    metrics_rows.append(row)
                    if best is None or row["validation_r2"] > best["validation_r2"]:
                        best = row

            if best is None:
                scorecard_rows.append(
                    {
                        "target": target_name,
                        "depth_layer": depth,
                        "status": "not_ready",
                        "best_candidate": "",
                        "best_model": "",
                        "best_feature_set": "",
                        "validation_r2": np.nan,
                        "test_r2": np.nan,
                        "test_rmse": np.nan,
                        "test_mae": np.nan,
                        "reason": "no_candidate_fitted",
                    }
                )
                continue

            scorecard_rows.append(
                {
                    "target": target_name,
                    "depth_layer": depth,
                    "status": "baseline_ready",
                    "best_candidate": best["candidate"],
                    "best_model": best["model_name"],
                    "best_feature_set": best["feature_set_name"],
                    "validation_r2": best["validation_r2"],
                    "test_r2": best["test_r2"],
                    "test_rmse": best["test_rmse"],
                    "test_mae": best["test_mae"],
                    "reason": "selected_by_validation_r2",
                }
            )

            report_lines.append(
                f"- `{depth}`: baseline ready，best={best['candidate']}，validation R²={best['validation_r2']:.4f}，test R²={best['test_r2']:.4f}，RMSE={best['test_rmse']:.4f}，MAE={best['test_mae']:.4f}"
            )
        report_lines.append("")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target", "depth_layer", "validation_r2"], ascending=[True, True, False])
    scorecard_df = pd.DataFrame(scorecard_rows).sort_values(["target", "depth_layer"])

    metrics_path = PROJECT_ROOT / args.metrics_out
    scorecard_path = PROJECT_ROOT / args.scorecard_out
    report_path = PROJECT_ROOT / args.report_out
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    scorecard_df.to_csv(scorecard_path, index=False, encoding="utf-8-sig")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(
        {
            "metrics_out": str(metrics_path),
            "scorecard_out": str(scorecard_path),
            "report_out": str(report_path),
            "rows_metrics": int(len(metrics_df)),
            "rows_scorecard": int(len(scorecard_df)),
        }
    )


if __name__ == "__main__":
    main()
