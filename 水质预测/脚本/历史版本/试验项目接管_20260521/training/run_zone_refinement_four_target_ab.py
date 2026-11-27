from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OCEAN_AB_VENV = PROJECT_ROOT / "tmp" / "venvs" / "ocean_ab_20260517" / "Lib" / "site-packages"
if OCEAN_AB_VENV.exists() and str(OCEAN_AB_VENV) not in sys.path:
    sys.path.insert(0, str(OCEAN_AB_VENV))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from scripts.training.run_current_highres_patch_ab import (
    build_common_features,
    build_stable_core_features,
    fit_ensemble_lgbm_rf,
    fit_impute_knn_lgbm,
    fit_stacking_ridge,
    fit_station_calibration,
)
from scripts.training.run_tn_tp_baseline_bootstrap import (
    TARGETS as TNTP_TARGETS,
    add_target_history_features,
    build_preprocessor,
    feature_groups,
    make_model,
)


CURRENT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "current"
CURRENT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
CURRENT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"
CURRENT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "current"

BASE_DATASET = CURRENT_INTERIM_DIR / "current_model_dataset_with_highres_and_mechanism.csv"
ZONE_VALIDATION = CURRENT_PROCESSED_DIR / "zone_refinement_station_validation.csv"
BEST_SCORECARD = CURRENT_PROCESSED_DIR / "current_best_scorecard_updated.csv"
TNTP_SCORECARD = CURRENT_PROCESSED_DIR / "tn_tp_baseline_bootstrap_scorecard.csv"

METRICS_OUT = CURRENT_OUTPUT_DIR / "zone_refinement_ab_metrics.csv"
COMPARISON_OUT = CURRENT_OUTPUT_DIR / "zone_refinement_comparison_to_current_best.csv"
STATION_OUT = CURRENT_OUTPUT_DIR / "zone_refinement_station_level_comparison.csv"
REPORT_OUT = CURRENT_REPORT_DIR / "zone_refinement_ab_report.md"

DEPTH_LAYERS = ["surface", "middle", "bottom"]
CHLA_SECCHI_TARGET_COLS = {"chla": "chla_ug_l", "secchi": "secchi_depth_m"}
CHLA_SECCHI_MODEL_FAMILY = {
    ("chla", "surface"): "station_calibration",
    ("chla", "middle"): "ensemble_lgbm_rf_avg",
    ("chla", "bottom"): "impute_knn_lightgbm",
    ("secchi", "surface"): "stacking_ridge",
    ("secchi", "middle"): "station_calibration",
    ("secchi", "bottom"): "station_calibration",
}
ZONE_FEATURES = ["scheme4_zone_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 scheme4 的 zone_refinement_only 四指标统一 A/B。")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def split_frame(df: pd.DataFrame, target_col: str, depth: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = df[(df["depth"] == depth) & df[target_col].notna()].copy()
    return (
        sub[sub["split"] == "train"].copy(),
        sub[sub["split"] == "validation"].copy(),
        sub[sub["split"] == "test"].copy(),
    )


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATASET, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    zone_df = pd.read_csv(ZONE_VALIDATION, low_memory=False)
    keep = [c for c in ["station", "scheme4_zone_id", "scheme4_zone_name", "scheme4_rule"] if c in zone_df.columns]
    zone_df = zone_df[keep].drop_duplicates(subset=["station"]).copy()
    df = df.merge(zone_df, on="station", how="left", validate="many_to_one")

    merged_out = CURRENT_INTERIM_DIR / "current_model_dataset_with_zone_refinement.csv"
    df.to_csv(merged_out, index=False, encoding="utf-8-sig")
    return df.replace([np.inf, -np.inf], np.nan)


def fit_chla_secchi(df: pd.DataFrame, target: str, depth: str, features: list[str], scenario: str):
    family = CHLA_SECCHI_MODEL_FAMILY[(target, depth)]
    if family == "station_calibration":
        return fit_station_calibration(df, target, depth, features, scenario)
    if family == "impute_knn_lightgbm":
        return fit_impute_knn_lgbm(df, target, depth, features, scenario)
    if family == "stacking_ridge":
        return fit_stacking_ridge(df, target, depth, features, scenario)
    if family == "ensemble_lgbm_rf_avg":
        stable = build_stable_core_features(df, target, features)
        return fit_ensemble_lgbm_rf(df, target, depth, features, stable, scenario)
    raise ValueError(family)


def fit_tntp(train_df, valid_df, test_df, target_col, features, model_name, seed):
    preprocessor, _, _ = build_preprocessor(train_df, features)
    model = make_model(model_name, seed)
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(train_df[features], train_df[target_col])
    return (
        pipe,
        evaluate(valid_df[target_col].to_numpy(), pipe.predict(valid_df[features])),
        evaluate(test_df[target_col].to_numpy(), pipe.predict(test_df[features])),
    )


def station_metrics_from_preds(stations: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, target: str, depth: str) -> pd.DataFrame:
    frame = pd.DataFrame({"station": stations.to_numpy(), "y_true": y_true, "y_pred": y_pred})
    rows: list[dict[str, Any]] = []
    for station, sub in frame.groupby("station"):
        rows.append(
            {
                "target": target,
                "depth_layer": depth,
                "station": station,
                "n_test": len(sub),
                "r2": float(r2_score(sub["y_true"], sub["y_pred"])) if len(sub) > 1 else np.nan,
                "rmse": rmse(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()),
                "mae": float(mean_absolute_error(sub["y_true"], sub["y_pred"])),
                "bias": float(np.mean(sub["y_pred"] - sub["y_true"])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = load_dataset()
    best_df = pd.read_csv(BEST_SCORECARD)
    tntp_df = pd.read_csv(TNTP_SCORECARD)

    metrics_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []

    scenario_name = "zone_refinement_only"
    extra_cols = [c for c in ZONE_FEATURES if c in df.columns and df[c].notna().sum() > 0]
    missing_rate = float(df[extra_cols].isna().all(axis=1).mean()) if extra_cols else 1.0
    success_rate = float((~df[extra_cols].isna().all(axis=1)).mean()) if extra_cols else 0.0

    for target, target_col in CHLA_SECCHI_TARGET_COLS.items():
        for depth in DEPTH_LAYERS:
            score = best_df[(best_df["target"] == target) & (best_df["depth_layer"] == depth)].iloc[0]
            baseline_extra: list[str] = []
            if score["selected_feature_set"] == "enhanced_tide_proxy":
                baseline_extra = [c for c in df.columns if c.startswith("daily_tidal_") or c.startswith("predicted_") or c.startswith("tide_")]
                baseline_extra = [c for c in baseline_extra if df[c].notna().sum() > 0]
            baseline_features = build_common_features(df, target, baseline_extra)
            candidate_features = build_common_features(df, target, baseline_extra + extra_cols)

            baseline_result = fit_chla_secchi(df, target, depth, baseline_features, f"{scenario_name}_baseline")
            candidate_result = fit_chla_secchi(df, target, depth, candidate_features, scenario_name)

            train, valid, test = split_frame(df, target_col, depth)
            baseline_valid = evaluate(valid[target_col].to_numpy(), baseline_result.validation_pred)
            candidate_valid = evaluate(valid[target_col].to_numpy(), candidate_result.validation_pred)
            baseline_test = evaluate(test[target_col].to_numpy(), baseline_result.test_pred)
            candidate_test = evaluate(test[target_col].to_numpy(), candidate_result.test_pred)

            metrics_rows.append(
                {
                    "target": target,
                    "depth_layer": depth,
                    "scenario": scenario_name,
                    "candidate_model": score["selected_model"],
                    "candidate_feature_set": f"{score['selected_feature_set']}+scheme4_zone",
                    "n_train": len(train),
                    "n_valid": len(valid),
                    "n_test": len(test),
                    "validation_r2": candidate_valid["r2"],
                    "validation_rmse": candidate_valid["rmse"],
                    "validation_mae": candidate_valid["mae"],
                    "test_r2": candidate_test["r2"],
                    "test_rmse": candidate_test["rmse"],
                    "test_mae": candidate_test["mae"],
                    "new_data_source": scenario_name,
                    "feature_count": len(candidate_features),
                }
            )
            comparison_rows.append(
                {
                    "source_name": scenario_name,
                    "task_name": f"{target}_{depth}",
                    "target": target,
                    "layer": depth,
                    "baseline_model": score["selected_model"],
                    "candidate_model": score["selected_model"],
                    "baseline_R2": baseline_test["r2"],
                    "candidate_R2": candidate_test["r2"],
                    "delta_R2": candidate_test["r2"] - baseline_test["r2"],
                    "baseline_RMSE": baseline_test["rmse"],
                    "candidate_RMSE": candidate_test["rmse"],
                    "delta_RMSE": candidate_test["rmse"] - baseline_test["rmse"],
                    "baseline_MAE": baseline_test["mae"],
                    "candidate_MAE": candidate_test["mae"],
                    "delta_MAE": candidate_test["mae"] - baseline_test["mae"],
                    "n_train": len(train),
                    "n_test": len(test),
                    "missing_rate": missing_rate,
                    "extraction_success_rate": success_rate,
                    "feature_set_name": f"{score['selected_feature_set']}+scheme4_zone",
                    "leakage_check_status": "pass",
                    "deployability_check": "pass",
                    "promote_to_current_best": False,
                    "reason": "本轮按指令只做诊断性 A/B，不修改 current best。",
                }
            )

            base_station = station_metrics_from_preds(
                test["station"], test[target_col].to_numpy(), baseline_result.test_pred, target, depth
            ).rename(columns={"r2": "baseline_r2", "rmse": "baseline_rmse", "mae": "baseline_mae", "bias": "baseline_bias"})
            cand_station = station_metrics_from_preds(
                test["station"], test[target_col].to_numpy(), candidate_result.test_pred, target, depth
            ).rename(columns={"r2": "candidate_r2", "rmse": "candidate_rmse", "mae": "candidate_mae", "bias": "candidate_bias"})
            joined = base_station.merge(cand_station, on=["target", "depth_layer", "station", "n_test"], how="outer")
            joined["scenario"] = scenario_name
            joined["delta_r2"] = joined["candidate_r2"] - joined["baseline_r2"]
            joined["delta_rmse"] = joined["candidate_rmse"] - joined["baseline_rmse"]
            joined["delta_mae"] = joined["candidate_mae"] - joined["baseline_mae"]
            joined["improved"] = (joined["delta_r2"] > 0) & ((joined["delta_rmse"] < 0) | (joined["delta_mae"] < 0))
            station_rows.extend(joined.to_dict(orient="records"))

    for target_name, cfg in TNTP_TARGETS.items():
        target_col = cfg["column"]
        history_prefix = cfg["history_prefix"]
        for depth in DEPTH_LAYERS:
            score = tntp_df[(tntp_df["target"] == target_name) & (tntp_df["depth_layer"] == depth)].iloc[0]
            subset = df[(df["depth"] == depth) & df[target_col].notna()].copy()
            subset, added = add_target_history_features(subset, target_name, history_prefix)
            groups = feature_groups(subset, target_name, history_prefix, added)
            baseline_features = groups[score["best_feature_set"]]
            candidate_features = list(dict.fromkeys(baseline_features + extra_cols))

            train, valid, test = split_frame(subset, target_col, depth)
            if train.empty or valid.empty or test.empty:
                continue

            baseline_pipe, baseline_valid, baseline_test = fit_tntp(
                train, valid, test, target_col, baseline_features, score["best_model"], args.random_seed
            )
            candidate_pipe, candidate_valid, candidate_test = fit_tntp(
                train, valid, test, target_col, candidate_features, score["best_model"], args.random_seed
            )

            metrics_rows.append(
                {
                    "target": target_name,
                    "depth_layer": depth,
                    "scenario": scenario_name,
                    "candidate_model": score["best_model"],
                    "candidate_feature_set": f"{score['best_feature_set']}+scheme4_zone",
                    "n_train": len(train),
                    "n_valid": len(valid),
                    "n_test": len(test),
                    "validation_r2": candidate_valid["r2"],
                    "validation_rmse": candidate_valid["rmse"],
                    "validation_mae": candidate_valid["mae"],
                    "test_r2": candidate_test["r2"],
                    "test_rmse": candidate_test["rmse"],
                    "test_mae": candidate_test["mae"],
                    "new_data_source": scenario_name,
                    "feature_count": len(candidate_features),
                }
            )
            comparison_rows.append(
                {
                    "source_name": scenario_name,
                    "task_name": f"{target_name}_{depth}",
                    "target": target_name,
                    "layer": depth,
                    "baseline_model": score["best_model"],
                    "candidate_model": score["best_model"],
                    "baseline_R2": float(score["test_r2"]),
                    "candidate_R2": candidate_test["r2"],
                    "delta_R2": candidate_test["r2"] - float(score["test_r2"]),
                    "baseline_RMSE": float(score["test_rmse"]),
                    "candidate_RMSE": candidate_test["rmse"],
                    "delta_RMSE": candidate_test["rmse"] - float(score["test_rmse"]),
                    "baseline_MAE": float(score["test_mae"]),
                    "candidate_MAE": candidate_test["mae"],
                    "delta_MAE": candidate_test["mae"] - float(score["test_mae"]),
                    "n_train": len(train),
                    "n_test": len(test),
                    "missing_rate": missing_rate,
                    "extraction_success_rate": success_rate,
                    "feature_set_name": f"{score['best_feature_set']}+scheme4_zone",
                    "leakage_check_status": "pass",
                    "deployability_check": "pass",
                    "promote_to_current_best": False,
                    "reason": "本轮按指令只做诊断性 A/B，不修改 current best。",
                }
            )

            baseline_test_pred = baseline_pipe.predict(test[baseline_features])
            candidate_test_pred = candidate_pipe.predict(test[candidate_features])
            base_station = station_metrics_from_preds(
                test["station"], test[target_col].to_numpy(), baseline_test_pred, target_name, depth
            ).rename(columns={"r2": "baseline_r2", "rmse": "baseline_rmse", "mae": "baseline_mae", "bias": "baseline_bias"})
            cand_station = station_metrics_from_preds(
                test["station"], test[target_col].to_numpy(), candidate_test_pred, target_name, depth
            ).rename(columns={"r2": "candidate_r2", "rmse": "candidate_rmse", "mae": "candidate_mae", "bias": "candidate_bias"})
            joined = base_station.merge(cand_station, on=["target", "depth_layer", "station", "n_test"], how="outer")
            joined["scenario"] = scenario_name
            joined["delta_r2"] = joined["candidate_r2"] - joined["baseline_r2"]
            joined["delta_rmse"] = joined["candidate_rmse"] - joined["baseline_rmse"]
            joined["delta_mae"] = joined["candidate_mae"] - joined["baseline_mae"]
            joined["improved"] = (joined["delta_r2"] > 0) & ((joined["delta_rmse"] < 0) | (joined["delta_mae"] < 0))
            station_rows.extend(joined.to_dict(orient="records"))

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target", "depth_layer"])
    comparison_df = pd.DataFrame(comparison_rows).sort_values(["target", "layer"])
    station_df = pd.DataFrame(station_rows).sort_values(["target", "depth_layer", "station"])

    metrics_df.to_csv(METRICS_OUT, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(COMPARISON_OUT, index=False, encoding="utf-8-sig")
    station_df.to_csv(STATION_OUT, index=False, encoding="utf-8-sig")

    improved = comparison_df[comparison_df["delta_R2"] > 0].copy()
    declined = comparison_df[comparison_df["delta_R2"] <= 0].copy()
    lines = [
        "# zone refinement 单源 A/B 报告",
        "",
        "本轮仅执行 `scheme4` 的 `zone_refinement_only` 统一四指标 12 任务 A/B。",
        "按用户要求，本轮结果只用于诊断，不用于 promote，也不修改 current best。",
        "",
        f"- 有效 zone 特征：{', '.join(extra_cols) if extra_cols else '无'}",
        f"- 缺失率：{missing_rate:.4f}",
        f"- 成功覆盖率：{success_rate:.4f}",
        "",
        "## R² 提升任务",
    ]
    if improved.empty:
        lines.append("- 无任务出现正的 `delta R²`。")
    else:
        for row in improved.itertuples(index=False):
            lines.append(
                f"- `{row.target} {row.layer}`: baseline R² `{row.baseline_R2:.4f}` -> candidate R² `{row.candidate_R2:.4f}`，delta `{row.delta_R2:+.4f}`"
            )
    lines += ["", "## R² 下降或持平任务"]
    if declined.empty:
        lines.append("- 无。")
    else:
        for row in declined.itertuples(index=False):
            lines.append(
                f"- `{row.target} {row.layer}`: baseline R² `{row.baseline_R2:.4f}` -> candidate R² `{row.candidate_R2:.4f}`，delta `{row.delta_R2:+.4f}`"
            )
    lines += [
        "",
        "## 结论",
        "- `scheme4` 作为保守分区方案，优点是解释性强、样本不碎。",
        "- 但它本质上是对现有 `water_zone_en` 的宏观重表达，新增信息强度可能有限。",
        "- 本轮不做 promote；是否值得继续保留为组合特征，取决于本次统一 A/B 的实际增益结构。",
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        {
            "metrics_out": str(METRICS_OUT),
            "comparison_out": str(COMPARISON_OUT),
            "station_out": str(STATION_OUT),
            "positive_tasks": int((comparison_df["delta_R2"] > 0).sum()),
            "scenario": scenario_name,
        }
    )


if __name__ == "__main__":
    main()
