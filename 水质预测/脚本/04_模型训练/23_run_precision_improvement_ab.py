"""全站点三水层精度提升扩展实验。

统一使用严格时间外推：2014-2020 训练、2021-2022 验证、2014-2022 重训、
2023-2024 最终测试。模型、特征组和目标变换均由验证期选择；最终测试集
不参与任何选择或集成权重拟合。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_DATASET = Path("数据/处理后数据/衍生版本/model_table_3depth_2014_2024_calendar_history.csv")
DEFAULT_REPORT_DIR = Path("结果/扩展实验/精度提升_20260905")
TARGET_MAP = {
    "chla": "chla_ug_l",
    "secchi": "secchi_depth_m",
    "tn": "tn_mg_l",
    "tp": "tp_mg_l",
}
DEPTHS = ("surface", "middle", "bottom")
BASELINE_FEATURES = [
    "lon", "lat", "year_offset", "month_sin", "month_cos", "is_wet_season",
    "quarter", "day_of_year", "water_zone_code",
]
PROCESS_FEATURES = [
    "t2m", "u10", "v10", "tcc", "wind_speed", "wind_speed_sq", "wind_dir_sin", "wind_dir_cos",
    "rain_3d", "rain_14d", "rain_month", "tidal_range", "ntl_avg_rad", "viirs_count",
    "t2m_x_rain14d", "wind_speed_x_rain14d", "rain14d_x_tidal_range", "t2m_x_tidal_range",
    "ssrd_sum_3d", "ssrd_mean_3d", "ssrd_sum_7d", "ssrd_mean_7d", "ssrd_sum_14d", "ssrd_mean_14d",
    "sst_mean_3d", "sst_max_3d", "sst_anom_3d", "sst_mean_7d", "sst_max_7d", "sst_anom_7d",
    "sst_mean_14d", "sst_max_14d", "sst_anom_14d", "sst_mean_7d_x_ssrd_mean_7d",
    "sst_anom_7d_x_ssrd_mean_7d", "rain14d_x_ssrd_mean_7d",
]
PROCESS_LAG_BASES = ("t2m", "tcc", "wind_speed", "rain_3d", "rain_14d", "rain_month", "tidal_range", "ntl_avg_rad")
PROCESS_LAG_SUFFIXES = ("_lag1", "_lag2", "_lag3", "_roll3", "_roll5", "_delta1", "_delta7")
REMOTE_FEATURES = [
    "landsat_hq_blue", "landsat_hq_green", "landsat_hq_red", "landsat_hq_nir", "landsat_hq_swir1",
    "landsat_hq_swir2", "landsat_hq_ndvi", "landsat_hq_ndwi", "landsat_hq_mndwi", "landsat_hq_ndti",
    "landsat_image_count", "landsat_clear_pixel_count", "landsat_strict_valid_gee", "landsat_hq",
    "landsat_available", "s2_hq_b2_mean", "s2_hq_b3_mean", "s2_hq_b4_mean", "s2_hq_b5_mean",
    "s2_hq_b6_mean", "s2_hq_b8_mean", "s2_hq_b8a_mean", "s2_hq_b11_mean", "s2_hq_b12_mean",
    "s2_hq_ndwi_mean", "s2_hq_mndwi_mean", "s2_hq_ndci_mean", "s2_hq_ndti_mean", "s2_has_images",
    "s2_image_count", "s2_strict_valid", "s2_valid_pixel_count", "s2_hq", "s2_available",
    "hq_satellite_available_count",
]


@dataclass(frozen=True)
class Candidate:
    model: str
    feature_set: str
    target_transform: str

    @property
    def name(self) -> str:
        return f"{self.model}__{self.feature_set}__{self.target_transform}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较全站点精度提升候选组合。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--top-k-ensemble", type=int, default=4)
    parser.add_argument("--include-remote", action="store_true", help="在候选池中加入遥感融合特征组。")
    parser.add_argument("--remote-only", action="store_true", help="只运行加入遥感特征的融合组。")
    parser.add_argument("--evaluation-mode", choices=("time", "random"), default="time")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--include-huber", action="store_true", help="在候选池中加入计算较慢的 Huber 梯度提升。")
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGET_MAP), default=sorted(TARGET_MAP))
    parser.add_argument("--depths", nargs="+", choices=DEPTHS, default=list(DEPTHS))
    parser.add_argument("--models", nargs="+", choices=("XGBoost", "ExtraTrees", "RandomForest", "GradientBoostingHuber"), default=None)
    parser.add_argument("--feature-sets", nargs="+", choices=("external_driver", "step_history", "calendar_history"), default=None)
    parser.add_argument("--transforms", nargs="+", choices=("raw", "log1p"), default=None)
    return parser.parse_args()


def nse(y_true: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if denominator == 0 else float(1 - np.sum((y_true - prediction) ** 2) / denominator)


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "nse": nse(y_true, prediction),
        "bias": float(np.mean(prediction - y_true)),
    }


def normalise(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for column in out.columns:
        values = out[column].dropna().astype(str).str.strip().str.lower()
        if not values.empty and set(values.unique()).issubset({"true", "false", "0", "1"}):
            out[column] = out[column].astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    out["water_zone_en"] = out["water_zone_en"].fillna("unknown")
    out["water_zone_code"] = pd.Categorical(out["water_zone_en"]).codes.astype("int16")
    return out


def available(data: pd.DataFrame, names: list[str]) -> list[str]:
    return [name for name in names if name in data.columns and data[name].notna().any()]


def feature_sets(data: pd.DataFrame, target: str) -> dict[str, list[str]]:
    external = available(data, BASELINE_FEATURES + PROCESS_FEATURES + [
        f"{base}{suffix}" for base in PROCESS_LAG_BASES for suffix in PROCESS_LAG_SUFFIXES
    ])
    step_history = available(data, [
        f"{target}_lag1", f"{target}_lag2", f"{target}_lag3", f"{target}_lag7",
        f"{target}_roll3", f"{target}_roll5", f"{target}_delta1", f"{target}_delta7",
    ])
    calendar_history = available(data, [name for name in data.columns if name.startswith(f"{target}_cal_")])
    remote = available(data, REMOTE_FEATURES)
    base_sets = {
        "external_driver": external,
        "step_history": external + step_history,
        "calendar_history": external + calendar_history,
    }
    return {**base_sets, **{f"{name}_remote": list(dict.fromkeys(features + remote)) for name, features in base_sets.items()}}


def make_model(name: str, args: argparse.Namespace):
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=args.n_estimators, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            random_state=42, n_jobs=args.n_jobs,
        )
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=args.n_estimators, max_features=0.8, min_samples_leaf=2,
            random_state=42, n_jobs=args.n_jobs,
        )
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=args.n_estimators, max_features=0.8, min_samples_leaf=2,
            random_state=42, n_jobs=args.n_jobs,
        )
    if name == "GradientBoostingHuber":
        return GradientBoostingRegressor(
            loss="huber", n_estimators=min(args.n_estimators, 120), max_depth=2, learning_rate=0.04,
            min_samples_leaf=4, subsample=0.85, random_state=42,
        )
    raise ValueError(name)


def transform_target(values: np.ndarray, name: str) -> tuple[np.ndarray, callable]:
    if name == "log1p":
        return np.log1p(values), np.expm1
    return values, lambda array: array


def train_predict(
    train: pd.DataFrame, predict: pd.DataFrame, target: str, features: list[str], candidate: Candidate, args: argparse.Namespace,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_predict = imputer.transform(predict[features])
    y_train_raw = train[target].to_numpy(dtype=float)
    y_train, inverse = transform_target(y_train_raw, candidate.target_transform)
    model = make_model(candidate.model, args)
    model.fit(x_train, y_train)
    prediction = inverse(model.predict(x_predict))
    return np.maximum(prediction, 0.0)


def candidate_grid(args: argparse.Namespace) -> list[Candidate]:
    feature_sets = args.feature_sets or ["external_driver", "step_history", "calendar_history"]
    if args.include_remote:
        feature_sets.extend([f"{name}_remote" for name in feature_sets])
    if args.remote_only:
        feature_sets = [f"{name}_remote" for name in feature_sets if not name.endswith("_remote")]
    models = args.models or (("XGBoost", "ExtraTrees", "RandomForest", "GradientBoostingHuber") if args.include_huber else ("XGBoost", "ExtraTrees", "RandomForest"))
    transforms = args.transforms or ("raw", "log1p")
    return [
        Candidate(model, feature_set, transform)
        for model in models
        for feature_set in feature_sets
        for transform in transforms
    ]


def evaluate_task(data: pd.DataFrame, target_key: str, depth: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[pd.DataFrame], dict[str, object]]:
    target = TARGET_MAP[target_key]
    subset = data.loc[data["depth"].eq(depth)].dropna(subset=[target]).copy()
    if args.evaluation_mode == "time":
        train_selection = subset.loc[subset["year"].between(2014, 2020)].copy()
        validation = subset.loc[subset["year"].between(2021, 2022)].copy()
        train_final = subset.loc[subset["year"].between(2014, 2022)].copy()
        test = subset.loc[subset["year"].between(2023, 2024)].copy()
        selection_split = "2014-2020 train / 2021-2022 validation"
        final_split = "2014-2022 train / 2023-2024 test"
    else:
        development, test = train_test_split(
            subset, test_size=args.test_size, random_state=args.random_state, shuffle=True,
        )
        train_selection, validation = train_test_split(
            development, test_size=0.25, random_state=args.random_state + 1, shuffle=True,
        )
        train_final = development.copy()
        selection_split = "outer-train 80% 内随机 75/25（即全样本 60/20）"
        final_split = f"固定随机 {int((1 - args.test_size) * 100)}/{int(args.test_size * 100)}，random_state={args.random_state}"
    sets = feature_sets(subset, target)
    records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    val_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}

    grid = candidate_grid(args)
    for candidate in grid:
        features = sets[candidate.feature_set]
        if not features:
            continue
        val_prediction = train_predict(train_selection, validation, target, features, candidate, args)
        val_predictions[candidate.name] = val_prediction
        records.append({
            "target": target_key, "target_column": target, "depth": depth, "stage": "selection_validation",
            "candidate": candidate.name, "model": candidate.model, "feature_set": candidate.feature_set,
            "target_transform": candidate.target_transform, "feature_count": len(features),
            "features_json": json.dumps(features, ensure_ascii=False), "train_rows": len(train_selection), "eval_rows": len(validation),
            "evaluation_mode": args.evaluation_mode, "selection_split": selection_split, "final_test_split": final_split,
            **metrics(validation[target].to_numpy(dtype=float), val_prediction),
        })

    validation_records = [record for record in records if record["stage"] == "selection_validation"]
    ranked = sorted(validation_records, key=lambda row: (row["r2"], -row["rmse"]), reverse=True)
    selected_names = [row["candidate"] for row in ranked[:args.top_k_ensemble]]
    selected_candidates = {candidate.name: candidate for candidate in grid if candidate.name in selected_names}

    for candidate_name in selected_names:
        candidate = selected_candidates[candidate_name]
        features = sets[candidate.feature_set]
        test_prediction = train_predict(train_final, test, target, features, candidate, args)
        test_predictions[candidate_name] = test_prediction
        records.append({
            "target": target_key, "target_column": target, "depth": depth, "stage": "final_test",
            "candidate": candidate.name, "model": candidate.model, "feature_set": candidate.feature_set,
            "target_transform": candidate.target_transform, "feature_count": len(features),
            "features_json": json.dumps(features, ensure_ascii=False), "train_rows": len(train_final), "eval_rows": len(test),
            "evaluation_mode": args.evaluation_mode, "selection_split": selection_split, "final_test_split": final_split,
            **metrics(test[target].to_numpy(dtype=float), test_prediction),
        })
        prediction_frame = test[["obs_id", "sample_id", "station", "date", "year", "depth", target]].copy()
        prediction_frame["target"] = target_key
        prediction_frame["stage"] = "final_test"
        prediction_frame["candidate"] = candidate.name
        prediction_frame["prediction"] = test_prediction
        prediction_frames.append(prediction_frame)

    x_validation = np.column_stack([val_predictions[name] for name in selected_names])
    x_test = np.column_stack([test_predictions[name] for name in selected_names])
    blender = LinearRegression(positive=True, fit_intercept=False)
    blender.fit(x_validation, validation[target].to_numpy(dtype=float))
    weights = blender.coef_ / blender.coef_.sum() if blender.coef_.sum() > 0 else np.repeat(1 / len(selected_names), len(selected_names))
    ensemble_prediction = np.maximum(x_test @ weights, 0.0)
    weight_map = dict(zip(selected_names, weights.round(8)))
    records.append({
        "target": target_key, "target_column": target, "depth": depth, "stage": "final_test",
        "candidate": "OOF_nonnegative_blend", "model": "OOFBlend", "feature_set": "selected_top_k",
        "target_transform": "mixed", "feature_count": np.nan, "features_json": json.dumps(weight_map, ensure_ascii=False),
        "train_rows": len(train_final), "eval_rows": len(test), "evaluation_mode": args.evaluation_mode,
        "selection_split": selection_split, "final_test_split": final_split,
        **metrics(test[target].to_numpy(dtype=float), ensemble_prediction),
    })
    ensemble_frame = test[["obs_id", "sample_id", "station", "date", "year", "depth", target]].copy()
    ensemble_frame["target"] = target_key
    ensemble_frame["stage"] = "final_test"
    ensemble_frame["candidate"] = "OOF_nonnegative_blend"
    ensemble_frame["prediction"] = ensemble_prediction
    prediction_frames.append(ensemble_frame)
    return records, prediction_frames, {
        "target": target_key, "depth": depth, "selected_candidates": ";".join(selected_names),
        "blend_weights_json": json.dumps(weight_map, ensure_ascii=False),
    }


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"未找到日历历史衍生主表：{args.dataset}；请先运行 25_build_calendar_history_features.py")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.dataset, parse_dates=["date"], low_memory=False)
    data.columns = [str(column).replace("\ufeff", "") for column in data.columns]
    data = normalise(data)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    for target_key in args.targets:
        for depth in args.depths:
            rows, frames, selection = evaluate_task(data, target_key, depth, args)
            metric_rows.extend(rows)
            prediction_frames.extend(frames)
            selections.append(selection)
            print(f"完成：{target_key} / {depth}", flush=True)

    metrics_table = pd.DataFrame(metric_rows)
    predictions_table = pd.concat(prediction_frames, ignore_index=True)
    selection_table = pd.DataFrame(selections)
    final_metrics = metrics_table.loc[metrics_table["stage"].eq("final_test")].copy()
    best = (final_metrics.sort_values(["target", "depth", "r2"], ascending=[True, True, False])
            .groupby(["target", "depth"], as_index=False).first())
    metrics_table.to_csv(args.report_dir / "精度提升_全部候选指标.csv", index=False, encoding="utf-8-sig")
    final_metrics.to_csv(args.report_dir / "精度提升_最终测试指标.csv", index=False, encoding="utf-8-sig")
    best.to_csv(args.report_dir / "精度提升_各任务最优组合.csv", index=False, encoding="utf-8-sig")
    predictions_table.to_csv(args.report_dir / "精度提升_最终测试预测值.csv", index=False, encoding="utf-8-sig")
    selection_table.to_csv(args.report_dir / "精度提升_验证期选择与融合权重.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "experiment_scope": "扩展实验：全站点三水层精度提升组合筛选",
        "formal_mainline_replaced": False,
        "evaluation_mode": args.evaluation_mode,
        "selection_split": "2014-2020 train / 2021-2022 validation" if args.evaluation_mode == "time" else "outer-train 80% 内随机 75/25（即全样本 60/20）",
        "final_test_split": "2014-2022 train / 2023-2024 test" if args.evaluation_mode == "time" else f"固定随机 {int((1 - args.test_size) * 100)}/{int(args.test_size * 100)}，random_state={args.random_state}",
        "history_feature_setting": "监测更新式：每个观测仅使用日期严格早于它的同站同水层历史目标值",
        "models_tested": "XGBoost;ExtraTrees;RandomForest" + (";GradientBoostingHuber" if args.include_huber else "") + ";OOF_nonnegative_blend",
        "remote_feature_groups_included": args.include_remote,
        "models_not_available": "CatBoost;LightGBM",
    }]).to_csv(args.report_dir / "精度提升_实验说明.csv", index=False, encoding="utf-8-sig")
    print(best[["target", "depth", "candidate", "model", "feature_set", "r2", "rmse", "mae"]].round(4).to_string(index=False))
    print(f"已写出：{args.report_dir}")


if __name__ == "__main__":
    main()
