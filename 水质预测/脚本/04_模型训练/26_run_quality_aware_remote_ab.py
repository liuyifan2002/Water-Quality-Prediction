"""Quality-aware remote-sensing gated-fusion A/B extension experiment.

It keeps the formal main table and results unchanged. For every target-depth task,
the script compares an f1 history/process expert, a remote-sensing expert, and a
quality gate trained only from outer-training OOF errors and image-quality fields.
No same-day water-quality covariate enters either expert or the gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
FORMAL_TRAINING_SCRIPT = ROOT / "\u811a\u672c" / "04_\u6a21\u578b\u8bad\u7ec3" / "05_train_full_station_models.py"
DEFAULT_DATASET = ROOT / "\u6570\u636e" / "\u5904\u7406\u540e\u6570\u636e" / "\u6b63\u5f0f\u4e3b\u8868" / "model_table_3depth_2014_2024.csv"
DEFAULT_LANDSAT_PATCH = ROOT / "\u6570\u636e" / "\u5916\u90e8\u6570\u636e" / "\u8bd5\u9a8c\u9879\u76ee\u63a5\u7ba1_20260521" / "data_processed_current" / "landsat_highres_patch_features_2014_2024.csv"
DEFAULT_S2_PATCH = ROOT / "\u6570\u636e" / "\u5916\u90e8\u6570\u636e" / "\u8bd5\u9a8c\u9879\u76ee\u63a5\u7ba1_20260521" / "data_processed_current" / "s2_highres_patch_features_2014_2024.csv"
DEFAULT_REPORT_DIR = ROOT / "\u7ed3\u679c" / "\u6269\u5c55\u5b9e\u9a8c" / "\u8d28\u91cf\u611f\u77e5\u9065\u611f\u95e8\u63a7\u878d\u5408_20260905"

LANDSAT_REMOTE_CANDIDATES = [
    "landsat_3x3_SR_B2_mean", "landsat_3x3_SR_B3_mean", "landsat_3x3_SR_B4_mean",
    "landsat_3x3_SR_B5_mean", "landsat_3x3_SR_B6_mean", "landsat_3x3_SR_B7_mean",
    "landsat_3x3_NDWI_mean", "landsat_3x3_MNDWI_mean", "landsat_3x3_NDTI_mean",
    "landsat_3x3_nir_red_ratio_mean",
]
S2_REMOTE_CANDIDATES = [
    "s2_3x3_B2_mean", "s2_3x3_B3_mean", "s2_3x3_B4_mean", "s2_3x3_B5_mean",
    "s2_3x3_B8_mean", "s2_3x3_B8A_mean", "s2_3x3_B11_mean", "s2_3x3_B12_mean",
    "s2_3x3_NDWI_mean", "s2_3x3_MNDWI_mean", "s2_3x3_NDCI_mean",
    "s2_3x3_NDTI_mean", "s2_3x3_FAI_mean",
]
LANDSAT_QUALITY_CANDIDATES = ["landsat_image_count_in_window", "landsat_3x3_SR_B2_count"]
S2_QUALITY_CANDIDATES = ["s2_image_count_in_window", "s2_3x3_B2_count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quality-aware remote-sensing gated-fusion A/B.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--landsat-patch", type=Path, default=DEFAULT_LANDSAT_PATCH)
    parser.add_argument("--s2-patch", type=Path, default=DEFAULT_S2_PATCH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--targets", nargs="*", choices=["chla", "secchi", "tn", "tp"], default=["chla", "secchi", "tn", "tp"])
    parser.add_argument("--depths", nargs="*", choices=["surface", "middle", "bottom"], default=["surface", "middle", "bottom"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--oof-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--min-test-rows", type=int, default=50)
    return parser.parse_args()


def load_formal_module():
    spec = importlib.util.spec_from_file_location("formal_full_station_training", FORMAL_TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load formal training script: {FORMAL_TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def xgb_model(args: argparse.Namespace) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror", random_state=args.random_state, n_jobs=args.n_jobs,
    )


def transform_target(values: pd.Series) -> tuple[np.ndarray, callable]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.nanmin(numeric) >= 0:
        return np.log1p(numeric), np.expm1
    return numeric, lambda array: array


def fit_predict(train: pd.DataFrame, prediction_data: pd.DataFrame, target_col: str, features: list[str], args: argparse.Namespace) -> np.ndarray:
    model = xgb_model(args)
    y_train, inverse = transform_target(train[target_col])
    model.fit(train[features], y_train)
    return np.maximum(inverse(model.predict(prediction_data[features])), 0)


def read_patch(path: Path, remote_candidates: list[str], quality_candidates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Patch feature table not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    selected_remote = [column for column in remote_candidates if column in header]
    selected_quality = [column for column in quality_candidates if column in header]
    selected = selected_remote + [column for column in selected_quality if column not in selected_remote]
    patch = pd.read_csv(path, usecols=["sample_id"] + selected, low_memory=False)
    if patch["sample_id"].duplicated().any():
        raise ValueError(f"Patch table has duplicate sample_id: {path}")
    renamed = {column: f"patch_{column}" for column in selected}
    return patch.rename(columns=renamed), [renamed[column] for column in selected_remote]


def build_quality_features(data: pd.DataFrame) -> list[str]:
    source_columns = {
        "q_landsat_image_count": "patch_landsat_image_count_in_window",
        "q_landsat_valid_pixels": "patch_landsat_3x3_SR_B2_count",
        "q_s2_image_count": "patch_s2_image_count_in_window",
        "q_s2_valid_pixels": "patch_s2_3x3_B2_count",
    }
    for output, source in source_columns.items():
        values = pd.to_numeric(data[source], errors="coerce") if source in data.columns else pd.Series(0.0, index=data.index)
        data[output] = values.fillna(0.0)
    data["q_landsat_has_valid_remote"] = ((data["q_landsat_image_count"] >= 1) & (data["q_landsat_valid_pixels"] >= 1)).astype(float)
    data["q_s2_has_valid_remote"] = ((data["q_s2_image_count"] >= 1) & (data["q_s2_valid_pixels"] >= 1)).astype(float)
    data["q_any_valid_remote"] = np.maximum(data["q_landsat_has_valid_remote"], data["q_s2_has_valid_remote"])
    data["q_both_valid_remote"] = np.minimum(data["q_landsat_has_valid_remote"], data["q_s2_has_valid_remote"])
    return list(source_columns) + ["q_landsat_has_valid_remote", "q_s2_has_valid_remote", "q_any_valid_remote", "q_both_valid_remote"]


def merge_patch_features(data: pd.DataFrame, landsat_path: Path, s2_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    landsat, landsat_features = read_patch(landsat_path, LANDSAT_REMOTE_CANDIDATES, LANDSAT_QUALITY_CANDIDATES)
    s2, s2_features = read_patch(s2_path, S2_REMOTE_CANDIDATES, S2_QUALITY_CANDIDATES)
    merged = data.merge(landsat, on="sample_id", how="left", validate="many_to_one")
    merged = merged.merge(s2, on="sample_id", how="left", validate="many_to_one")
    if len(merged) != len(data):
        raise AssertionError("Patch merge changed model-table row count.")
    quality_features = build_quality_features(merged)
    return merged, landsat_features + s2_features, quality_features


def assert_feature_safety(features: list[str], target_col: str, target_map: dict[str, str]) -> None:
    forbidden = set(target_map.values()) | {"turbidity", "turbidity_ntu", "salinity", "do", "ph"}
    overlaps = sorted(set(features) & forbidden)
    if overlaps or target_col in features:
        raise ValueError(f"Forbidden same-day water-quality feature: {overlaps or [target_col]}")


def quality_bin(data: pd.DataFrame) -> pd.Series:
    landsat = data["q_landsat_has_valid_remote"].to_numpy(dtype=bool)
    s2 = data["q_s2_has_valid_remote"].to_numpy(dtype=bool)
    labels = np.where(landsat & s2, "both_sensors", np.where(landsat, "landsat_only", np.where(s2, "s2_only", "no_valid_remote")))
    return pd.Series(labels, index=data.index)


def oof_predictions(train: pd.DataFrame, target_col: str, f1_features: list[str], remote_features: list[str], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    f1_oof = np.full(len(train), np.nan)
    remote_oof = np.full(len(train), np.nan)
    kfold = KFold(n_splits=args.oof_folds, shuffle=True, random_state=args.random_state)
    for train_index, validation_index in kfold.split(train):
        fold_train = train.iloc[train_index]
        fold_validation = train.iloc[validation_index]
        f1_oof[validation_index] = fit_predict(fold_train, fold_validation, target_col, f1_features, args)
        remote_oof[validation_index] = fit_predict(fold_train, fold_validation, target_col, remote_features, args)
    if np.isnan(f1_oof).any() or np.isnan(remote_oof).any():
        raise AssertionError("OOF prediction is incomplete.")
    return f1_oof, remote_oof


def fit_gate_from_oof(train: pd.DataFrame, target_col: str, f1_oof: np.ndarray, remote_oof: np.ndarray, quality_features: list[str]):
    observed = train[target_col].to_numpy(dtype=float)
    remote_wins = (np.abs(remote_oof - observed) < np.abs(f1_oof - observed)).astype(int)
    valid_remote = train["q_any_valid_remote"].to_numpy(dtype=int)
    if valid_remote.sum() == 0 or len(np.unique(remote_wins[valid_remote == 1])) < 2:
        return None, remote_wins
    gate = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    gate.fit(train[quality_features], remote_wins)
    return gate, remote_wins


def gate_weights(gate, data: pd.DataFrame, quality_features: list[str]) -> np.ndarray:
    valid_remote = data["q_any_valid_remote"].to_numpy(dtype=float)
    if gate is None:
        return np.zeros(len(data), dtype=float)
    return np.clip(gate.predict_proba(data[quality_features])[:, 1] * valid_remote, 0.0, 1.0)


def make_metric_rows(test: pd.DataFrame, target_key: str, depth: str, variants: dict[str, np.ndarray], counts: dict[str, int], mean_weight: float, oof_win_rate: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_names = ["all"] + sorted(test["quality_bin"].unique().tolist())
    for variant, prediction in variants.items():
        for group_name in group_names:
            group = test if group_name == "all" else test[test["quality_bin"] == group_name]
            if len(group) < 20:
                continue
            positions = group.index.to_numpy(dtype=int)
            rows.append({
                "target": target_key, "depth": depth, "model_variant": variant, "quality_group": group_name,
                "rows": len(group), "feature_count": counts[variant], "split_mode": "random_80_20",
                "split_random_state": 42, "same_day_water_quality_covariates": "excluded",
                "gate_mean_weight": mean_weight if variant == "quality_gate_fusion" else np.nan,
                "oof_remote_win_rate": oof_win_rate if variant == "quality_gate_fusion" else np.nan,
                **evaluate(group["observed"].to_numpy(dtype=float), prediction[positions]),
            })
    return rows


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    formal = load_formal_module()
    data = formal.prepare_dataset(args.dataset)
    data, patch_remote_features, quality_features = merge_patch_features(data, args.landsat_patch, args.s2_patch)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    for target_key in args.targets:
        target_col = formal.TARGET_MAP[target_key]
        f1_features = formal.build_feature_sets(data, target_col)["f1_history"]
        remote_features = f1_features + [feature for feature in patch_remote_features + quality_features if feature not in f1_features]
        assert_feature_safety(f1_features, target_col, formal.TARGET_MAP)
        assert_feature_safety(remote_features, target_col, formal.TARGET_MAP)
        for depth in args.depths:
            subset = data[(data["depth"] == depth) & data[target_col].notna()].copy().sort_values(["date", "station"]).reset_index(drop=True)
            train, test = train_test_split(subset, test_size=args.test_size, random_state=args.random_state, shuffle=True)
            train, test = train.reset_index(drop=True), test.reset_index(drop=True)
            if len(train) < args.min_train_rows or len(test) < args.min_test_rows:
                continue

            f1_oof, remote_oof = oof_predictions(train, target_col, f1_features, remote_features, args)
            gate, remote_wins = fit_gate_from_oof(train, target_col, f1_oof, remote_oof, quality_features)
            f1_prediction = fit_predict(train, test, target_col, f1_features, args)
            remote_prediction = fit_predict(train, test, target_col, remote_features, args)
            weight = gate_weights(gate, test, quality_features)
            gated_prediction = (1.0 - weight) * f1_prediction + weight * remote_prediction

            output = test[["obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "year", "month", "depth"]].copy()
            output["target"] = target_key
            output["observed"] = test[target_col].to_numpy(dtype=float)
            output["prediction_f1_history"] = f1_prediction
            output["prediction_remote_expert"] = remote_prediction
            output["prediction_quality_gate_fusion"] = gated_prediction
            output["gate_remote_weight"] = weight
            output["quality_bin"] = quality_bin(test).to_numpy()
            for column in ["q_landsat_image_count", "q_landsat_valid_pixels", "q_s2_image_count", "q_s2_valid_pixels"]:
                output[column] = test[column].to_numpy()
            prediction_frames.append(output)

            variants = {"f1_history": f1_prediction, "remote_expert": remote_prediction, "quality_gate_fusion": gated_prediction}
            counts = {"f1_history": len(f1_features), "remote_expert": len(remote_features), "quality_gate_fusion": len(remote_features)}
            metric_rows.extend(make_metric_rows(output.reset_index(drop=True), target_key, depth, variants, counts, float(np.mean(weight)), float(np.mean(remote_wins))))

            oof = train[["obs_id", "sample_id", "station", "date"]].copy()
            oof["target"], oof["depth"] = target_key, depth
            oof["observed"], oof["prediction_f1_history"], oof["prediction_remote_expert"] = train[target_col].to_numpy(dtype=float), f1_oof, remote_oof
            oof["remote_wins_oof"], oof["quality_bin"] = remote_wins, quality_bin(train).to_numpy()
            oof_frames.append(oof)
            manifest_rows.append({
                "target": target_key, "depth": depth, "f1_history_features_json": json.dumps(f1_features, ensure_ascii=False),
                "remote_expert_features_json": json.dumps(remote_features, ensure_ascii=False),
                "gate_features_json": json.dumps(quality_features, ensure_ascii=False), "same_day_water_quality_covariates": "excluded",
            })
            if gate is not None:
                for feature, coefficient in zip(quality_features, gate.named_steps["logistic"].coef_[0], strict=True):
                    coefficient_rows.append({"target": target_key, "depth": depth, "quality_feature": feature, "standardized_log_odds": coefficient})

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.report_dir / "质量感知遥感融合_指标.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(args.report_dir / "特征与防泄漏清单.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefficient_rows).to_csv(args.report_dir / "门控系数.csv", index=False, encoding="utf-8-sig")
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(args.report_dir / "质量感知遥感融合_预测.csv", index=False, encoding="utf-8-sig")
    if oof_frames:
        pd.concat(oof_frames, ignore_index=True).to_csv(args.report_dir / "门控训练期OOF记录.csv", index=False, encoding="utf-8-sig")
    (args.report_dir / "实验说明.md").write_text(
        "# 质量感知遥感门控融合 A/B\n\n本目录为扩展实验，不覆盖正式主链结果。门控只使用影像数和有效像元数；训练标签来自外层训练集 OOF 绝对误差，不读取测试集目标。无有效影像时，预测强制回退到 f1_history 专家。\n",
        encoding="utf-8",
    )
    print(f"Wrote quality-aware remote-sensing A/B outputs to {args.report_dir}")
    if not metrics.empty:
        print(metrics[metrics["quality_group"] == "all"][["target", "depth", "model_variant", "rows", "r2", "rmse", "mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
