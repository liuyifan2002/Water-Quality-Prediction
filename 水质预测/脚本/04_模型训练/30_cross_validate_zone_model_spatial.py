"""Five-fold held-station validation for the zone-aware full-station model.

Each station is assigned to one fold within its water zone.  In a test fold,
all observations from its held stations are unavailable to both the global and
zone-aware models.  Target-derived history features remain excluded so this is
a strict spatial-transfer comparison rather than a same-station interpolation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
FORMAL_TRAINING_SCRIPT = ROOT / "\u811a\u672c" / "04_\u6a21\u578b\u8bad\u7ec3" / "05_train_full_station_models.py"
DEFAULT_DATASET = ROOT / "\u6570\u636e" / "\u5904\u7406\u540e\u6570\u636e" / "\u6b63\u5f0f\u4e3b\u8868" / "model_table_3depth_2014_2024.csv"
DEFAULT_REPORT_DIR = ROOT / "\u7ed3\u679c" / "\u6269\u5c55\u5b9e\u9a8c" / "\u5206\u6d77\u533a\u4e94\u6298\u7559\u7ad9\u9a8c\u8bc1_20260906"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five-fold held-station comparison: global versus zone-aware models.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--targets", nargs="*", choices=["chla", "secchi", "tn", "tp"], default=["chla", "secchi", "tn", "tp"])
    parser.add_argument("--depths", nargs="*", choices=["surface", "middle", "bottom"], default=["surface", "middle", "bottom"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-zone-train-rows", type=int, default=300)
    parser.add_argument("--min-zone-stations", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def load_formal_module():
    spec = importlib.util.spec_from_file_location("formal_full_station_training", FORMAL_TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load formal training script: {FORMAL_TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def strict_external_features(formal, data: pd.DataFrame, target_col: str) -> list[str]:
    features = formal.build_feature_sets(data, target_col)["f1_history"]
    target_history_prefix = f"{target_col}_"
    selected = [feature for feature in features if not feature.startswith(target_history_prefix)]
    if target_col in selected:
        raise ValueError("Target cannot be used as an input feature.")
    return selected


def make_model(args: argparse.Namespace) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )


def fit_predict(
    train: pd.DataFrame,
    prediction_data: pd.DataFrame,
    target_col: str,
    features: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    y = train[target_col].to_numpy(dtype=float)
    if np.nanmin(y) >= 0:
        y_fit, inverse = np.log1p(y), np.expm1
    else:
        y_fit, inverse = y, lambda values: values
    model = make_model(args)
    model.fit(train[features], y_fit)
    return np.maximum(inverse(model.predict(prediction_data[features])), 0)


def evaluate(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "bias": float(np.mean(prediction - y_true)),
    }


def assign_station_folds(data: pd.DataFrame, n_folds: int, random_state: int) -> pd.DataFrame:
    station_zone = data[["station", "water_zone_en"]].drop_duplicates().sort_values(["water_zone_en", "station"])
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for zone, group in station_zone.groupby("water_zone_en", sort=True):
        stations = group["station"].to_numpy(dtype=str).copy()
        rng.shuffle(stations)
        for position, station in enumerate(stations):
            rows.append({"station": station, "water_zone_en": zone, "fold": position % n_folds + 1})
    return pd.DataFrame(rows).sort_values(["fold", "water_zone_en", "station"]).reset_index(drop=True)


def run_one_task(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_key: str,
    target_col: str,
    depth: str,
    fold: int,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    global_prediction = fit_predict(train, test, target_col, features, args)
    zone_prediction = global_prediction.copy()
    source = np.full(len(test), "global_fallback", dtype=object)
    decisions: list[dict[str, object]] = []

    for zone, zone_test in test.groupby("water_zone_en", sort=True):
        zone_train = train.loc[train["water_zone_en"].eq(zone)].copy()
        eligible = len(zone_train) >= args.min_zone_train_rows and zone_train["station"].nunique() >= args.min_zone_stations
        decisions.append({
            "fold": fold,
            "target": target_key,
            "depth": depth,
            "water_zone_en": zone,
            "train_rows": len(zone_train),
            "test_rows": len(zone_test),
            "train_stations": int(zone_train["station"].nunique()),
            "test_stations": int(zone_test["station"].nunique()),
            "decision": "zone_model" if eligible else "global_fallback",
        })
        if eligible:
            positions = zone_test.index.to_numpy(dtype=int)
            zone_prediction[positions] = fit_predict(zone_train, zone_test, target_col, features, args)
            source[positions] = "zone_model"

    output = test[["obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "year", "month", "depth"]].copy()
    output["fold"] = fold
    output["target"] = target_key
    output["observed"] = test[target_col].to_numpy(dtype=float)
    output["prediction_global_external"] = global_prediction
    output["prediction_zone_aware_external"] = zone_prediction
    output["prediction_source"] = source

    metrics: list[dict[str, object]] = []
    for model_name, prediction in (("global_external", global_prediction), ("zone_aware_external", zone_prediction)):
        metrics.append({
            "fold": fold,
            "target": target_key,
            "depth": depth,
            "model_variant": model_name,
            "water_zone_en": "all",
            "rows": len(output),
            "feature_count": len(features),
            "target_history_features": "excluded",
            "same_day_water_quality_covariates": "excluded",
            **evaluate(output["observed"].to_numpy(dtype=float), prediction),
        })
    return metrics, output, decisions


def build_summary(metrics: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_metrics = metrics.loc[metrics["water_zone_en"].eq("all")].copy()
    fold_wide = all_metrics.pivot_table(
        index=["fold", "target", "depth"], columns="model_variant", values="r2", aggfunc="first"
    ).reset_index()
    fold_wide["delta_r2"] = fold_wide["zone_aware_external"] - fold_wide["global_external"]

    summary = fold_wide.groupby(["target", "depth"], as_index=False).agg(
        folds=("fold", "nunique"),
        global_r2_mean=("global_external", "mean"),
        global_r2_std=("global_external", "std"),
        zone_r2_mean=("zone_aware_external", "mean"),
        zone_r2_std=("zone_aware_external", "std"),
        delta_r2_mean=("delta_r2", "mean"),
        delta_r2_std=("delta_r2", "std"),
        positive_folds=("delta_r2", lambda values: int((values > 0).sum())),
    )

    pooled_rows: list[dict[str, object]] = []
    for (target, depth), group in predictions.groupby(["target", "depth"], sort=True):
        observed = group["observed"].to_numpy(dtype=float)
        global_metrics = evaluate(observed, group["prediction_global_external"].to_numpy(dtype=float))
        zone_metrics = evaluate(observed, group["prediction_zone_aware_external"].to_numpy(dtype=float))
        pooled_rows.append({
            "target": target,
            "depth": depth,
            "global_r2_pooled": global_metrics["r2"],
            "zone_r2_pooled": zone_metrics["r2"],
            "delta_r2_pooled": zone_metrics["r2"] - global_metrics["r2"],
            "global_rmse_pooled": global_metrics["rmse"],
            "zone_rmse_pooled": zone_metrics["rmse"],
        })
    summary = summary.merge(pd.DataFrame(pooled_rows), on=["target", "depth"], how="left")
    summary["stable_better"] = np.where(
        (summary["positive_folds"] >= 4) & (summary["delta_r2_pooled"] > 0), "yes", "no"
    )

    macro_by_fold = fold_wide.groupby("fold", as_index=False).agg(
        global_r2_mean=("global_external", "mean"),
        zone_r2_mean=("zone_aware_external", "mean"),
        delta_r2_mean=("delta_r2", "mean"),
    )
    macro_by_fold["zone_better"] = macro_by_fold["delta_r2_mean"] > 0
    return summary, macro_by_fold


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    formal = load_formal_module()
    data = formal.prepare_dataset(args.dataset)
    data["water_zone_en"] = data["water_zone_en"].fillna("unknown").astype(str)
    station_folds = assign_station_folds(data, args.n_folds, args.random_state)
    station_to_fold = station_folds.set_index("station")["fold"]

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    decisions: list[dict[str, object]] = []
    feature_manifest: list[dict[str, object]] = []
    for target_key in args.targets:
        target_col = formal.TARGET_MAP[target_key]
        features = strict_external_features(formal, data, target_col)
        feature_manifest.append({"target": target_key, "features_json": json.dumps(features, ensure_ascii=False), "target_history_features": "excluded"})
        for depth in args.depths:
            subset = data.loc[(data["depth"].eq(depth)) & data[target_col].notna()].copy()
            subset["fold"] = subset["station"].map(station_to_fold)
            subset = subset.sort_values(["fold", "date", "station"]).reset_index(drop=True)
            for fold in range(1, args.n_folds + 1):
                train = subset.loc[subset["fold"].ne(fold)].drop(columns="fold").reset_index(drop=True)
                test = subset.loc[subset["fold"].eq(fold)].drop(columns="fold").reset_index(drop=True)
                if len(train) < 200 or test.empty:
                    continue
                rows, output, task_decisions = run_one_task(train, test, target_key, target_col, depth, fold, features, args)
                metric_rows.extend(rows)
                prediction_frames.append(output)
                decisions.extend(task_decisions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary, macro_by_fold = build_summary(metrics, predictions)
    metrics.to_csv(args.report_dir / "五折留站点验证_各折指标.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.report_dir / "五折留站点验证_预测.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.report_dir / "五折留站点验证_汇总.csv", index=False, encoding="utf-8-sig")
    macro_by_fold.to_csv(args.report_dir / "五折总体比较.csv", index=False, encoding="utf-8-sig")
    station_folds.to_csv(args.report_dir / "五折站点划分清单.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(decisions).to_csv(args.report_dir / "分区训练决策.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_manifest).to_csv(args.report_dir / "严格验证特征清单.csv", index=False, encoding="utf-8-sig")
    (args.report_dir / "实验说明.md").write_text(
        "# 分海区模型五折留站点验证\n\n"
        "本实验将站点按所属海区随机分配至五折；每折中，被留出的站点全部观测均不参与训练。"
        "输入沿用正式主链的 f1_history 外部驱动特征，但剔除目标自身历史水质特征和所有同期水质协变量。"
        "每折比较全域模型与分海区模型，并同时报告逐折、五折均值和全部折外预测合并结果。"
        "若某任务至少 4/5 折 R2 提升且合并 R2 也提升，汇总表标记为 stable_better=yes。\n",
        encoding="utf-8",
    )

    macro_delta = macro_by_fold["delta_r2_mean"]
    print("Five-fold macro R2 comparison")
    print(macro_by_fold.to_string(index=False))
    print("\nTask summary")
    print(summary[["target", "depth", "global_r2_pooled", "zone_r2_pooled", "delta_r2_pooled", "positive_folds", "stable_better"]].to_string(index=False))
    print(f"\nMacro mean delta: {macro_delta.mean():.4f}; positive folds: {(macro_delta > 0).sum()}/{len(macro_delta)}")


if __name__ == "__main__":
    main()
