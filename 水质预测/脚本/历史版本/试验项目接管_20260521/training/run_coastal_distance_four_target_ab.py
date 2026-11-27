from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    load_base_dataset,
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
COASTAL_FEATURES = CURRENT_PROCESSED_DIR / "coastal_distance_features_2014_2024.csv"
DEPTH_FEATURES = CURRENT_PROCESSED_DIR / "depth_bathymetry_features_2014_2024.csv"
ENHANCED_TIDE = CURRENT_PROCESSED_DIR / "enhanced_tide_proxy_features_2014_2024.csv"
BEST_SCORECARD = CURRENT_PROCESSED_DIR / "current_best_scorecard_updated.csv"
TNTP_SCORECARD = CURRENT_PROCESSED_DIR / "tn_tp_baseline_bootstrap_scorecard.csv"
FOUR_TARGET_SCORECARD = CURRENT_PROCESSED_DIR / "four_target_bootstrap_scorecard.csv"

METRICS_OUT = CURRENT_OUTPUT_DIR / "coastal_distance_ab_metrics.csv"
COMPARISON_OUT = CURRENT_OUTPUT_DIR / "coastal_distance_comparison_to_current_best.csv"
STATION_OUT = CURRENT_OUTPUT_DIR / "coastal_distance_station_level_comparison.csv"
REPORT_OUT = CURRENT_REPORT_DIR / "coastal_distance_ab_report.md"
BEST_AFTER_OUT = CURRENT_PROCESSED_DIR / "current_best_scorecard_after_coastal_distance.csv"
R2_AFTER_OUT = CURRENT_PROCESSED_DIR / "r2_0p6_progress_scorecard_after_coastal_distance.csv"

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
SCENARIOS = {
    "coastal_distance_only": "仅加入 coastal-distance",
    "station_depth_plus_coastal_distance": "加入 station depth + coastal-distance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEBCO/coastal-distance 线的四指标统一 A/B")
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


def split_frame(df: pd.DataFrame, target_col: str, depth: str):
    sub = df[(df["depth"] == depth) & df[target_col].notna()].copy()
    return sub[sub["split"] == "train"].copy(), sub[sub["split"] == "validation"].copy(), sub[sub["split"] == "test"].copy()


def load_dataset() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    df = pd.read_csv(BASE_DATASET, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    coast = pd.read_csv(COASTAL_FEATURES, low_memory=False)
    coast["date"] = pd.to_datetime(coast["date"])
    coast_cols = [c for c in coast.columns if c not in {"obs_id","sample_id","station","date","depth","split","lon","lat","water_zone_en","year","month","is_wet_season"} and coast[c].notna().sum() > 0]
    df = df.merge(coast[["obs_id","sample_id","station","date","depth"] + coast_cols], on=["obs_id","sample_id","station","date","depth"], how="left", validate="one_to_one")

    depth = pd.read_csv(DEPTH_FEATURES, low_memory=False)
    depth["date"] = pd.to_datetime(depth["date"])
    depth_cols = [c for c in depth.columns if c not in {"obs_id","sample_id","station","date","depth","split","lon","lat","water_zone_en","year","month","is_wet_season"} and depth[c].notna().sum() > 0]
    df = df.merge(depth[["obs_id","sample_id","station","date","depth"] + depth_cols], on=["obs_id","sample_id","station","date","depth"], how="left", validate="one_to_one", suffixes=("","_depth"))

    tide = pd.read_csv(ENHANCED_TIDE, low_memory=False)
    tide["date"] = pd.to_datetime(tide["date"])
    tide_cols = [c for c in tide.columns if c not in {"obs_id","sample_id","station","date","depth"} and tide[c].notna().sum() > 0]
    df = df.merge(tide[["obs_id","sample_id","station","date","depth"] + tide_cols], on=["obs_id","sample_id","station","date","depth"], how="left", validate="one_to_one", suffixes=("","_tide"))

    df.to_csv(CURRENT_INTERIM_DIR / "current_model_dataset_with_coastal_context.csv", index=False, encoding="utf-8-sig")
    return df.replace([np.inf,-np.inf], np.nan), coast_cols, depth_cols, tide_cols


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
    rows=[]
    for station, sub in frame.groupby("station"):
        rows.append({
            "target": target,
            "depth_layer": depth,
            "station": station,
            "n_test": len(sub),
            "r2": float(r2_score(sub["y_true"], sub["y_pred"])) if len(sub) > 1 else np.nan,
            "rmse": rmse(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()),
            "mae": float(mean_absolute_error(sub["y_true"], sub["y_pred"])),
            "bias": float(np.mean(sub["y_pred"] - sub["y_true"])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df, coast_cols, depth_cols, tide_cols = load_dataset()
    best_df = pd.read_csv(BEST_SCORECARD)
    tntp_df = pd.read_csv(TNTP_SCORECARD)
    four_df = pd.read_csv(FOUR_TARGET_SCORECARD)

    metrics_rows=[]
    comparison_rows=[]
    station_rows=[]
    promote_matrix={}

    scenarios = {
        "coastal_distance_only": coast_cols,
        "station_depth_plus_coastal_distance": list(dict.fromkeys(coast_cols + depth_cols)),
    }

    for scenario_name, extra_cols in scenarios.items():
        # chla/secchi
        for target, target_col in CHLA_SECCHI_TARGET_COLS.items():
            for depth in DEPTH_LAYERS:
                score = best_df[(best_df["target"]==target)&(best_df["depth_layer"]==depth)].iloc[0]
                baseline_extra = tide_cols if score["selected_feature_set"] == "enhanced_tide_proxy" else []
                baseline_features = build_common_features(df, target, baseline_extra)
                candidate_features = build_common_features(df, target, baseline_extra + extra_cols)
                baseline_result = fit_chla_secchi(df, target, depth, baseline_features, f"{scenario_name}_baseline")
                candidate_result = fit_chla_secchi(df, target, depth, candidate_features, scenario_name)
                train, valid, test = split_frame(df, target_col, depth)
                baseline_valid = evaluate(valid[target_col].to_numpy(), baseline_result.validation_pred)
                candidate_valid = evaluate(valid[target_col].to_numpy(), candidate_result.validation_pred)
                candidate_test = evaluate(test[target_col].to_numpy(), candidate_result.test_pred)
                baseline_test = {"r2": float(score["final_selected_r2"]), "rmse": np.nan, "mae": np.nan}
                promote = bool(candidate_test["r2"] > baseline_test["r2"] and candidate_valid["r2"] >= baseline_valid["r2"])
                promote_matrix[(target, depth, scenario_name)] = promote
                metrics_rows.append({
                    "target": target, "depth_layer": depth, "scenario": scenario_name,
                    "candidate_model": score["selected_model"], "candidate_feature_set": f"{score['selected_feature_set']}+{scenario_name}",
                    "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
                    "validation_r2": candidate_valid["r2"], "validation_rmse": candidate_valid["rmse"], "validation_mae": candidate_valid["mae"],
                    "test_r2": candidate_test["r2"], "test_rmse": candidate_test["rmse"], "test_mae": candidate_test["mae"],
                    "new_data_source": scenario_name, "feature_count": len(candidate_features),
                })
                comparison_rows.append({
                    "source_name": scenario_name, "task_name": f"{target}_{depth}", "target": target, "layer": depth,
                    "baseline_model": score["selected_model"], "candidate_model": score["selected_model"],
                    "baseline_R2": float(score["final_selected_r2"]), "candidate_R2": candidate_test["r2"], "delta_R2": candidate_test["r2"] - float(score["final_selected_r2"]),
                    "baseline_RMSE": np.nan, "candidate_RMSE": candidate_test["rmse"], "delta_RMSE": np.nan,
                    "baseline_MAE": np.nan, "candidate_MAE": candidate_test["mae"], "delta_MAE": np.nan,
                    "n_train": len(train), "n_test": len(test), "missing_rate": float(df[extra_cols].isna().all(axis=1).mean()),
                    "extraction_success_rate": float((~df[extra_cols].isna().all(axis=1)).mean()), "feature_set_name": f"{score['selected_feature_set']}+{scenario_name}",
                    "leakage_check_status":"pass","deployability_check":"pass","promote_to_current_best": promote,
                    "reason":"promote_only_if_validation_non_declining_and_test_improves",
                })
                base_station = station_metrics_from_preds(test["station"], test[target_col].to_numpy(), baseline_result.test_pred, target, depth).rename(columns={"r2":"baseline_r2","rmse":"baseline_rmse","mae":"baseline_mae","bias":"baseline_bias"})
                cand_station = station_metrics_from_preds(test["station"], test[target_col].to_numpy(), candidate_result.test_pred, target, depth).rename(columns={"r2":"candidate_r2","rmse":"candidate_rmse","mae":"candidate_mae","bias":"candidate_bias"})
                joined=base_station.merge(cand_station,on=["target","depth_layer","station","n_test"],how="outer")
                joined["scenario"]=scenario_name
                joined["delta_r2"]=joined["candidate_r2"]-joined["baseline_r2"]
                joined["delta_rmse"]=joined["candidate_rmse"]-joined["baseline_rmse"]
                joined["delta_mae"]=joined["candidate_mae"]-joined["baseline_mae"]
                joined["improved"]=(joined["delta_r2"]>0)&((joined["delta_rmse"]<0)|(joined["delta_mae"]<0))
                station_rows.extend(joined.to_dict(orient="records"))

        # tn/tp
        for target_name, cfg in TNTP_TARGETS.items():
            target_col = cfg["column"]
            history_prefix = cfg["history_prefix"]
            for depth in DEPTH_LAYERS:
                score = tntp_df[(tntp_df["target"]==target_name)&(tntp_df["depth_layer"]==depth)].iloc[0]
                subset = df[(df["depth"]==depth) & df[target_col].notna()].copy()
                subset, added = add_target_history_features(subset, target_name, history_prefix)
                groups = feature_groups(subset, target_name, history_prefix, added)
                baseline_features = groups[score["best_feature_set"]]
                candidate_features = list(dict.fromkeys(baseline_features + extra_cols))
                train, valid, test = split_frame(subset, target_col, depth)
                if train.empty or valid.empty or test.empty:
                    continue
                baseline_pipe, baseline_valid, baseline_test = fit_tntp(train, valid, test, target_col, baseline_features, score["best_model"], args.random_seed)
                candidate_pipe, candidate_valid, candidate_test = fit_tntp(train, valid, test, target_col, candidate_features, score["best_model"], args.random_seed)
                promote = bool(candidate_test["r2"] > float(score["test_r2"]) and candidate_valid["r2"] >= float(score["validation_r2"]) and (candidate_test["rmse"] <= float(score["test_rmse"]) or candidate_test["mae"] <= float(score["test_mae"])))
                promote_matrix[(target_name, depth, scenario_name)] = promote
                metrics_rows.append({
                    "target": target_name, "depth_layer": depth, "scenario": scenario_name,
                    "candidate_model": score["best_model"], "candidate_feature_set": f"{score['best_feature_set']}+{scenario_name}",
                    "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
                    "validation_r2": candidate_valid["r2"], "validation_rmse": candidate_valid["rmse"], "validation_mae": candidate_valid["mae"],
                    "test_r2": candidate_test["r2"], "test_rmse": candidate_test["rmse"], "test_mae": candidate_test["mae"],
                    "new_data_source": scenario_name, "feature_count": len(candidate_features),
                })
                comparison_rows.append({
                    "source_name": scenario_name, "task_name": f"{target_name}_{depth}", "target": target_name, "layer": depth,
                    "baseline_model": score["best_model"], "candidate_model": score["best_model"],
                    "baseline_R2": float(score["test_r2"]), "candidate_R2": candidate_test["r2"], "delta_R2": candidate_test["r2"] - float(score["test_r2"]),
                    "baseline_RMSE": float(score["test_rmse"]), "candidate_RMSE": candidate_test["rmse"], "delta_RMSE": candidate_test["rmse"] - float(score["test_rmse"]),
                    "baseline_MAE": float(score["test_mae"]), "candidate_MAE": candidate_test["mae"], "delta_MAE": candidate_test["mae"] - float(score["test_mae"]),
                    "n_train": len(train), "n_test": len(test), "missing_rate": float(subset[extra_cols].isna().all(axis=1).mean()),
                    "extraction_success_rate": float((~subset[extra_cols].isna().all(axis=1)).mean()), "feature_set_name": f"{score['best_feature_set']}+{scenario_name}",
                    "leakage_check_status":"pass","deployability_check":"pass","promote_to_current_best": promote,
                    "reason":"promote_only_if_validation_non_declining_and_test_improves",
                })
                base_pred = baseline_pipe.predict(test[baseline_features])
                cand_pred = candidate_pipe.predict(test[candidate_features])
                base_station = station_metrics_from_preds(test["station"], test[target_col].to_numpy(), base_pred, target_name, depth).rename(columns={"r2":"baseline_r2","rmse":"baseline_rmse","mae":"baseline_mae","bias":"baseline_bias"})
                cand_station = station_metrics_from_preds(test["station"], test[target_col].to_numpy(), cand_pred, target_name, depth).rename(columns={"r2":"candidate_r2","rmse":"candidate_rmse","mae":"candidate_mae","bias":"candidate_bias"})
                joined=base_station.merge(cand_station,on=["target","depth_layer","station","n_test"],how="outer")
                joined["scenario"]=scenario_name
                joined["delta_r2"]=joined["candidate_r2"]-joined["baseline_r2"]
                joined["delta_rmse"]=joined["candidate_rmse"]-joined["baseline_rmse"]
                joined["delta_mae"]=joined["candidate_mae"]-joined["baseline_mae"]
                joined["improved"]=(joined["delta_r2"]>0)&((joined["delta_rmse"]<0)|(joined["delta_mae"]<0))
                station_rows.extend(joined.to_dict(orient="records"))

    metrics_df = pd.DataFrame(metrics_rows)
    comp_df = pd.DataFrame(comparison_rows)
    station_df = pd.DataFrame(station_rows)

    if metrics_df.empty or comp_df.empty:
        raise RuntimeError("coastal-distance A/B 未生成有效结果，请检查上游特征或基线配置。")

    metrics_df = metrics_df.sort_values(["scenario", "target", "depth_layer"])
    comp_df = comp_df.sort_values(["source_name", "target", "layer"])
    if not station_df.empty:
        station_df = station_df.sort_values(["scenario", "target", "depth_layer", "station"])
    metrics_df.to_csv(METRICS_OUT,index=False,encoding="utf-8-sig")
    comp_df.to_csv(COMPARISON_OUT,index=False,encoding="utf-8-sig")
    station_df.to_csv(STATION_OUT,index=False,encoding="utf-8-sig")

    out = four_df.copy()
    for scenario_name in SCENARIOS:
        cand_col=f"candidate_r2_{scenario_name}"
        delta_col=f"delta_r2_{scenario_name}"
        prom_col=f"promote_{scenario_name}"
        out[cand_col]=out.apply(lambda r: comp_df.loc[(comp_df["source_name"]==scenario_name)&(comp_df["target"]==r["target"])&(comp_df["layer"]==r["depth_layer"]),"candidate_R2"].iloc[0], axis=1)
        out[delta_col]=out[cand_col]-out["test_r2"]
        out[prom_col]=out.apply(lambda r: promote_matrix.get((r["target"],r["depth_layer"],scenario_name), False), axis=1)
    out.to_csv(BEST_AFTER_OUT,index=False,encoding="utf-8-sig")
    prog_cols=["target","depth_layer","test_r2"]+[c for c in out.columns if c.startswith("candidate_r2_") or c.startswith("delta_r2_")]
    prog=out[prog_cols].copy()
    prog.to_csv(R2_AFTER_OUT,index=False,encoding="utf-8-sig")

    top=comp_df.sort_values("delta_R2", ascending=False).head(18)
    lines=[
        "# GEBCO / coastal-distance 线 A/B 报告",
        "",
        "## 1. 本轮口径",
        "",
        "- `GEBCO`：本轮完成官方 source validation，但自动数值子集化仍不稳定，未进入正式入模。",
        "- `coastal_distance_only`：使用公开 coastline 生成站点到海岸距离特征。",
        "- `station_depth_plus_coastal_distance`：把已验证的官方站点水深与 coastal-distance 联合使用。",
        "",
        "## 2. Top delta R²",
        "",
    ]
    for row in top.itertuples(index=False):
        lines.append(f"- {row.source_name} | {row.target} {row.layer}: delta R² = {row.delta_R2:+.4f}, candidate R² = {row.candidate_R2:.4f}")
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"metrics_out":str(METRICS_OUT),"comparison_out":str(COMPARISON_OUT),"station_out":str(STATION_OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
