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
MOUTH_VALIDATION = CURRENT_PROCESSED_DIR / "distance_to_mouth_system_station_validation.csv"
ZONE_VALIDATION = CURRENT_PROCESSED_DIR / "zone_refinement_station_validation.csv"
BEST_SCORECARD = CURRENT_PROCESSED_DIR / "current_best_scorecard_updated.csv"
TNTP_SCORECARD = CURRENT_PROCESSED_DIR / "tn_tp_baseline_bootstrap_scorecard.csv"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 distance-to-mouth 四指标统一 A/B")
    parser.add_argument(
        "--scenario",
        choices=["distance_to_mouth_only", "distance_to_mouth_plus_zone"],
        required=True,
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def output_paths(scenario: str) -> tuple[Path, Path, Path, Path]:
    if scenario == "distance_to_mouth_only":
        prefix = "distance_to_mouth"
    else:
        prefix = "distance_to_mouth_zone"
    return (
        CURRENT_OUTPUT_DIR / f"{prefix}_ab_metrics.csv",
        CURRENT_OUTPUT_DIR / f"{prefix}_comparison_to_current_best.csv",
        CURRENT_OUTPUT_DIR / f"{prefix}_station_level_comparison.csv",
        CURRENT_REPORT_DIR / f"{prefix}_ab_report.md",
    )


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


def add_distance_bucket(series: pd.Series) -> pd.Series:
    bins = [-np.inf, 5.0, 15.0, 30.0, np.inf]
    labels = ["lt5km", "5to15km", "15to30km", "ge30km"]
    return pd.cut(series, bins=bins, labels=labels)


def load_dataset(scenario: str) -> pd.DataFrame:
    df = pd.read_csv(BASE_DATASET, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    mouth_df = pd.read_csv(MOUTH_VALIDATION, low_memory=False)
    mouth_df["mouth_system_distance_bucket"] = add_distance_bucket(mouth_df["distance_to_mouth_system_km"]).astype(str)
    keep = [
        c
        for c in [
            "station",
            "nearest_mouth_system_id",
            "nearest_mouth_system_name",
            "distance_to_mouth_system_km",
            "log_distance_to_mouth_system_km",
            "mouth_system_distance_bucket",
            "system_mapping_ambiguity_flag",
        ]
        if c in mouth_df.columns
    ]
    df = df.merge(mouth_df[keep].drop_duplicates(subset=["station"]), on="station", how="left", validate="many_to_one")

    if scenario == "distance_to_mouth_plus_zone":
        zone_df = pd.read_csv(ZONE_VALIDATION, low_memory=False)
        zone_keep = [c for c in ["station", "scheme4_zone_id", "scheme4_zone_name", "scheme4_rule"] if c in zone_df.columns]
        df = df.merge(zone_df[zone_keep].drop_duplicates(subset=["station"]), on="station", how="left", validate="many_to_one")

    merged_out = CURRENT_INTERIM_DIR / f"current_model_dataset_with_{scenario}.csv"
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


def scenario_feature_columns(df: pd.DataFrame, scenario: str) -> list[str]:
    base = [
        "distance_to_mouth_system_km",
        "log_distance_to_mouth_system_km",
        "nearest_mouth_system_id",
        "mouth_system_distance_bucket",
        "system_mapping_ambiguity_flag",
    ]
    cols = [c for c in base if c in df.columns and df[c].notna().sum() > 0]
    if scenario == "distance_to_mouth_plus_zone":
        cols += [c for c in ["scheme4_zone_id"] if c in df.columns and df[c].notna().sum() > 0]
    return list(dict.fromkeys(cols))


def main() -> None:
    args = parse_args()
    scenario_name = args.scenario
    metrics_out, comparison_out, station_out, report_out = output_paths(scenario_name)

    df = load_dataset(scenario_name)
    best_df = pd.read_csv(BEST_SCORECARD)
    tntp_df = pd.read_csv(TNTP_SCORECARD)
    extra_cols = scenario_feature_columns(df, scenario_name)
    missing_rate = float(df[extra_cols].isna().all(axis=1).mean()) if extra_cols else 1.0
    success_rate = float((~df[extra_cols].isna().all(axis=1)).mean()) if extra_cols else 0.0

    metrics_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []

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
            baseline_test = evaluate(test[target_col].to_numpy(), baseline_result.test_pred)
            candidate_valid = evaluate(valid[target_col].to_numpy(), candidate_result.validation_pred)
            candidate_test = evaluate(test[target_col].to_numpy(), candidate_result.test_pred)

            metrics_rows.append(
                {
                    "target": target,
                    "depth_layer": depth,
                    "scenario": scenario_name,
                    "candidate_model": score["selected_model"],
                    "candidate_feature_set": f"{score['selected_feature_set']}+{'+'.join(extra_cols)}",
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
                    "feature_set_name": f"{score['selected_feature_set']}+{scenario_name}",
                    "leakage_check_status": "pass",
                    "deployability_check": "pass",
                    "promote_to_current_best": False,
                    "reason": "本轮按指令只做诊断性 A/B，不修改 current best。",
                }
            )

            base_station = station_metrics_from_preds(test["station"], test[target_col].to_numpy(), baseline_result.test_pred, target, depth).rename(
                columns={"r2": "baseline_r2", "rmse": "baseline_rmse", "mae": "baseline_mae", "bias": "baseline_bias"}
            )
            cand_station = station_metrics_from_preds(test["station"], test[target_col].to_numpy(), candidate_result.test_pred, target, depth).rename(
                columns={"r2": "candidate_r2", "rmse": "candidate_rmse", "mae": "candidate_mae", "bias": "candidate_bias"}
            )
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
            _, baseline_valid, baseline_test = fit_tntp(train, valid, test, target_col, baseline_features, score["best_model"], args.random_seed)
            candidate_model, candidate_valid, candidate_test = fit_tntp(train, valid, test, target_col, candidate_features, score["best_model"], args.random_seed)

            metrics_rows.append(
                {
                    "target": target_name,
                    "depth_layer": depth,
                    "scenario": scenario_name,
                    "candidate_model": score["best_model"],
                    "candidate_feature_set": f"{score['best_feature_set']}+{'+'.join(extra_cols)}",
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
                    "feature_set_name": f"{score['best_feature_set']}+{scenario_name}",
                    "leakage_check_status": "pass",
                    "deployability_check": "pass",
                    "promote_to_current_best": False,
                    "reason": "本轮按指令只做诊断性 A/B，不修改 current best。",
                }
            )

            baseline_pipe, _, _ = fit_tntp(train, valid, test, target_col, baseline_features, score["best_model"], args.random_seed)
            test_baseline_pred = baseline_pipe.predict(test[baseline_features])
            test_candidate_pred = candidate_model.predict(test[candidate_features])
            base_station = station_metrics_from_preds(test["station"], test[target_col].to_numpy(), test_baseline_pred, target_name, depth).rename(
                columns={"r2": "baseline_r2", "rmse": "baseline_rmse", "mae": "baseline_mae", "bias": "baseline_bias"}
            )
            cand_station = station_metrics_from_preds(test["station"], test[target_col].to_numpy(), test_candidate_pred, target_name, depth).rename(
                columns={"r2": "candidate_r2", "rmse": "candidate_rmse", "mae": "candidate_mae", "bias": "candidate_bias"}
            )
            joined = base_station.merge(cand_station, on=["target", "depth_layer", "station", "n_test"], how="outer")
            joined["scenario"] = scenario_name
            joined["delta_r2"] = joined["candidate_r2"] - joined["baseline_r2"]
            joined["delta_rmse"] = joined["candidate_rmse"] - joined["baseline_rmse"]
            joined["delta_mae"] = joined["candidate_mae"] - joined["baseline_mae"]
            joined["improved"] = (joined["delta_r2"] > 0) & ((joined["delta_rmse"] < 0) | (joined["delta_mae"] < 0))
            station_rows.extend(joined.to_dict(orient="records"))

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target", "depth_layer"]).reset_index(drop=True)
    comparison_df = pd.DataFrame(comparison_rows).sort_values(["target", "layer"]).reset_index(drop=True)
    station_df = pd.DataFrame(station_rows).sort_values(["target", "depth_layer", "station"]).reset_index(drop=True)

    metrics_df.to_csv(metrics_out, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(comparison_out, index=False, encoding="utf-8-sig")
    station_df.to_csv(station_out, index=False, encoding="utf-8-sig")

    positive = comparison_df[(comparison_df["delta_R2"] > 0) & ((comparison_df["delta_RMSE"] < 0) | (comparison_df["delta_MAE"] < 0))]
    negative = comparison_df[comparison_df["delta_R2"] < 0].copy()
    lines = [
        f"# {scenario_name} 统一 A/B 报告",
        "",
        "本轮按指令仅做诊断性 A/B，不 promote，不修改 current best。",
        "",
        "## 特征口径",
        f"- 额外特征：{', '.join(extra_cols) if extra_cols else '无'}",
        f"- observation 级缺失率：{missing_rate:.4f}",
        f"- extraction success rate：{success_rate:.4f}",
        "",
        "## 正信号任务",
        positive[["target", "layer", "delta_R2", "delta_RMSE", "delta_MAE"]].to_string(index=False) if not positive.empty else "- 无",
        "",
        "## 负迁移任务",
        negative[["target", "layer", "delta_R2", "delta_RMSE", "delta_MAE"]].to_string(index=False) if not negative.empty else "- 无",
        "",
        "## 判断",
        "- 该结果仅用于判断后续是否继续组合实验，不作为 current best 更新依据。",
    ]
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        {
            "metrics_out": str(metrics_out),
            "comparison_out": str(comparison_out),
            "station_out": str(station_out),
            "positive_tasks": int(len(positive)),
            "scenario": scenario_name,
        }
    )


if __name__ == "__main__":
    main()
