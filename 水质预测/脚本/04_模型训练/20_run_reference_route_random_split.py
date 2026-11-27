"""复现实验性对照：参考论文的随机划分、多源外部驱动水质建模路线。

用途：比较随机 80/20 划分下的同分布估计能力，不能替代 full_station 正式主链的
2014-2022 训练、2023-2024 测试时间外推结果。所有特征均禁止使用同日原始水质指标。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_DATASET = Path("数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv")
DEFAULT_REPORT_DIR = Path("结果/扩展实验/参考论文路线随机划分_20260905")
TARGET_MAP = {
    "chla": "chla_ug_l",
    "secchi": "secchi_depth_m",
    "tn": "tn_mg_l",
    "tp": "tp_mg_l",
}
DEPTHS = ["surface", "middle", "bottom"]

# 与正式主链一致的外部驱动、时空背景和高质量遥感字段；不含任何实测水质字段。
BASELINE_FEATURES = [
    "lon", "lat", "year_offset", "month_sin", "month_cos", "is_wet_season",
    "quarter", "day_of_year", "water_zone_code",
]
PROCESS_FEATURES = [
    "t2m", "u10", "v10", "tcc", "wind_speed", "wind_speed_sq", "wind_dir_sin",
    "wind_dir_cos", "rain_3d", "rain_14d", "rain_month", "tidal_range",
    "ntl_avg_rad", "viirs_count", "t2m_x_rain14d", "wind_speed_x_rain14d",
    "rain14d_x_tidal_range", "t2m_x_tidal_range", "ssrd_sum_3d", "ssrd_mean_3d",
    "ssrd_sum_7d", "ssrd_mean_7d", "ssrd_sum_14d", "ssrd_mean_14d", "sst_mean_3d",
    "sst_max_3d", "sst_anom_3d", "sst_mean_7d", "sst_max_7d", "sst_anom_7d",
    "sst_mean_14d", "sst_max_14d", "sst_anom_14d", "sst_mean_7d_x_ssrd_mean_7d",
    "sst_anom_7d_x_ssrd_mean_7d", "rain14d_x_ssrd_mean_7d",
]
PROCESS_LAG_BASES = ["t2m", "tcc", "wind_speed", "rain_3d", "rain_14d", "rain_month", "tidal_range", "ntl_avg_rad"]
PROCESS_LAG_SUFFIXES = ["_lag1", "_lag2", "_lag3", "_roll3", "_roll5", "_delta1", "_delta7"]
REMOTE_FEATURES = [
    "landsat_hq_blue", "landsat_hq_green", "landsat_hq_red", "landsat_hq_nir",
    "landsat_hq_swir1", "landsat_hq_swir2", "landsat_hq_ndvi", "landsat_hq_ndwi",
    "landsat_hq_mndwi", "landsat_hq_ndti", "landsat_image_count",
    "landsat_clear_pixel_count", "landsat_strict_valid_gee", "landsat_hq",
    "landsat_available", "s2_hq_b2_mean", "s2_hq_b3_mean", "s2_hq_b4_mean",
    "s2_hq_b5_mean", "s2_hq_b6_mean", "s2_hq_b8_mean", "s2_hq_b8a_mean",
    "s2_hq_b11_mean", "s2_hq_b12_mean", "s2_hq_ndwi_mean", "s2_hq_mndwi_mean",
    "s2_hq_ndci_mean", "s2_hq_ndti_mean", "s2_has_images", "s2_image_count",
    "s2_strict_valid", "s2_valid_pixel_count", "s2_hq", "s2_available",
    "hq_satellite_available_count",
]
FORBIDDEN_WATER_QUALITY_COLUMNS = {
    "temperature_c", "salinity_psu", "ph", "do_mg_l", "do_pct_saturation",
    "turbidity_ntu", "suspended_solids_mg_l", "bod5_mg_l", "ammonia_n_mg_l",
    "nitrate_n_mg_l", "nitrite_n_mg_l", "orthophosphate_p_mg_l", "tin_mg_l",
    "tkn_mg_l", "unionised_ammonia_mg_l", "vss_mg_l", "silica_mg_l",
    "phaeo_pigments_ug_l", "secchi_depth_m", "chla_ug_l", "tn_mg_l", "tp_mg_l",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行参考论文路线的随机划分外部驱动对照实验。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if denominator == 0 else 1 - np.sum((y_true - y_pred) ** 2) / denominator


def normalize_bool_like_columns(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for column in out.columns:
        values = out[column].dropna().astype(str).str.strip().str.lower()
        if not values.empty and set(values.unique()).issubset({"true", "false", "0", "1"}):
            out[column] = out[column].astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    return out


def available(data: pd.DataFrame, names: list[str]) -> list[str]:
    return [name for name in names if name in data.columns]


def feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    process_lags = [f"{base}{suffix}" for base in PROCESS_LAG_BASES for suffix in PROCESS_LAG_SUFFIXES]
    external = available(data, BASELINE_FEATURES + PROCESS_FEATURES + process_lags)
    remote = available(data, REMOTE_FEATURES)
    result = {
        "external_driver": list(dict.fromkeys(external)),
        "external_driver_remote": list(dict.fromkeys(external + remote)),
    }
    for name, features in result.items():
        forbidden = set(features) & FORBIDDEN_WATER_QUALITY_COLUMNS
        if forbidden:
            raise RuntimeError(f"{name} 检测到禁止的同日水质字段：{sorted(forbidden)}")
    return result


def make_models(args: argparse.Namespace) -> dict[str, object]:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("缺少 xgboost，请先安装：pip install xgboost") from exc
    return {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=args.n_estimators, max_features=0.8, min_samples_leaf=2,
            random_state=args.random_state, n_jobs=args.n_jobs,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=args.n_estimators, max_features=0.8, min_samples_leaf=2,
            random_state=args.random_state, n_jobs=args.n_jobs,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=args.n_estimators, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            random_state=args.random_state, n_jobs=args.n_jobs,
        ),
    }


def fit_one(
    data: pd.DataFrame, target_key: str, depth: str, set_name: str,
    requested_features: list[str], args: argparse.Namespace,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    target = TARGET_MAP[target_key]
    subset = data.loc[data["depth"] == depth].dropna(subset=[target]).copy()
    train, test = train_test_split(subset, test_size=args.test_size, random_state=args.random_state, shuffle=True)
    features = [name for name in requested_features if train[name].notna().any()]
    if not features:
        return [{"target": target_key, "depth": depth, "feature_set": set_name, "status": "skipped_no_features"}], pd.DataFrame()

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_test = imputer.transform(test[features])
    y_train_raw = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    use_log = np.nanmin(y_train_raw) >= 0
    y_train = np.log1p(y_train_raw) if use_log else y_train_raw
    inverse = np.expm1 if use_log else lambda values: values

    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for model_name, model in make_models(args).items():
        model.fit(x_train, y_train)
        pred = np.maximum(inverse(model.predict(x_test)), 0)
        rows.append({
            "target": target_key, "target_column": target, "depth": depth,
            "feature_set": set_name, "model": model_name, "feature_count": len(features),
            "features_json": json.dumps(features, ensure_ascii=False), "train_rows": len(train),
            "test_rows": len(test), "split": "random_80_20", "random_state": args.random_state,
            "target_transform": "log1p" if use_log else "none", "r2": r2_score(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
            "mae": mean_absolute_error(y_test, pred), "nse": nse(y_test, pred), "status": "ok",
        })
        pred_frame = test[["obs_id", "sample_id", "station", "date", "depth", target]].copy()
        pred_frame["target"] = target_key
        pred_frame["feature_set"] = set_name
        pred_frame["model"] = model_name
        pred_frame["prediction"] = pred
        predictions.append(pred_frame)
    return rows, pd.concat(predictions, ignore_index=True)


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"未找到主表：{args.dataset}")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.dataset, parse_dates=["date"], low_memory=False)
    data.columns = [str(column).replace("\ufeff", "") for column in data.columns]
    data = normalize_bool_like_columns(data)
    data["water_zone_en"] = data["water_zone_en"].fillna("unknown")
    data["water_zone_code"] = pd.Categorical(data["water_zone_en"]).codes.astype("int16")
    for column in set(BASELINE_FEATURES + PROCESS_FEATURES + REMOTE_FEATURES + list(TARGET_MAP.values())):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for base in PROCESS_LAG_BASES:
        for suffix in PROCESS_LAG_SUFFIXES:
            column = f"{base}{suffix}"
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")

    sets = feature_sets(data)
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for target_key in TARGET_MAP:
        for depth in DEPTHS:
            for set_name, features in sets.items():
                result_rows, result_predictions = fit_one(data, target_key, depth, set_name, features, args)
                rows.extend(result_rows)
                if not result_predictions.empty:
                    predictions.append(result_predictions)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.report_dir / "参考论文路线_随机划分模型指标.csv", index=False, encoding="utf-8-sig")
    best = (metrics.loc[metrics["status"].eq("ok")]
            .sort_values(["target", "depth", "r2"], ascending=[True, True, False])
            .groupby(["target", "depth"], as_index=False).first())
    best.to_csv(args.report_dir / "参考论文路线_随机划分最佳结果.csv", index=False, encoding="utf-8-sig")
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(args.report_dir / "参考论文路线_随机划分预测值.csv", index=False, encoding="utf-8-sig")
    manifest = pd.DataFrame([{
        "experiment_scope": "扩展实验：参考论文随机 80/20 划分的外部驱动建模对照",
        "formal_mainline_replaced": False,
        "same_day_water_quality_allowed": False,
        "same_day_water_quality_columns_forbidden": ";".join(sorted(FORBIDDEN_WATER_QUALITY_COLUMNS)),
        "external_features_available": ";".join(sets["external_driver"]),
        "reference_paper_missing_sources": "空气污染物；河流流量",
        "models": "ExtraTrees;RandomForest;XGBoost",
    }])
    manifest.to_csv(args.report_dir / "参考论文路线_随机划分实验说明.csv", index=False, encoding="utf-8-sig")
    print(best[["target", "depth", "feature_set", "model", "r2", "rmse", "mae", "test_rows"]].round(4).to_string(index=False))
    print(f"已写出：{args.report_dir}")


if __name__ == "__main__":
    main()
