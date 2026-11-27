from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATASET = Path("数据/处理后数据/衍生版本/model_dataset_surface_2014_2024_enhanced.csv")
DEFAULT_OUT_DIR = Path("结果/扩展实验/optimized")
TARGETS = ["Chl_a", "SD", "TN", "TP"]

STATIC_FEATURES = ["lon", "lat"]
BASE_PROCESS = [
    "month", "season_sin", "season_cos",
    "t2m", "u10", "v10", "tcc", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "rain_3d", "rain_14d", "rain_month",
    "tidal_range", "ntl_avg_rad"
]
LAG_PROCESS = [
    "t2m_lag1", "t2m_lag2", "t2m_roll3",
    "tcc_lag1", "tcc_lag2", "tcc_roll3",
    "wind_speed_lag1", "wind_speed_lag2", "wind_speed_roll3",
    "rain_month_lag1", "rain_month_lag2", "rain_month_roll3",
    "rain_14d_lag1", "rain_14d_lag2", "rain_14d_roll3",
    "tidal_range_lag1", "tidal_range_lag2", "tidal_range_roll3",
    "ntl_avg_rad_lag1", "ntl_avg_rad_lag2", "ntl_avg_rad_roll3"
]
BASE_REMOTE = [
    "blue", "green", "red", "nir", "swir1", "swir2",
    "ndvi", "ndwi", "mndwi", "ndti",
    "blue_green", "green_red", "red_green", "nir_red", "red_nir",
    "valid_count", "landsat_missing", "landsat_imputed", "landsat_no_data"
]
LAG_REMOTE = [
    "ndwi_lag1", "ndwi_lag2", "ndwi_roll3",
    "mndwi_lag1", "mndwi_lag2", "mndwi_roll3",
    "ndti_lag1", "ndti_lag2", "ndti_roll3",
    "valid_count_lag1", "valid_count_lag2", "valid_count_roll3"
]

FEATURE_SETS = {
    "F1_process": STATIC_FEATURES + BASE_PROCESS + LAG_PROCESS,
    "F2_remote_sensing": STATIC_FEATURES + BASE_REMOTE + LAG_REMOTE,
    "F3_fusion": STATIC_FEATURES + BASE_PROCESS + LAG_PROCESS + BASE_REMOTE + LAG_REMOTE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练优化版 XGBoost，并输出融合和消融结果。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target", default="all", choices=["all", *TARGETS])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-log", action="store_true", help="不对 Chl-a/TN/TP 做 log1p 目标变换。")
    parser.add_argument("--tune", action="store_true", help="使用 2021-2022 做小网格调参。")
    return parser.parse_args()


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "nse": float(nse(y_true, y_pred)),
    }


def build_matrix(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列：{missing}")
    x = df[features].copy()
    station_dummies = pd.get_dummies(df["station"], prefix="station", dtype=float)
    x = pd.concat([x, station_dummies], axis=1)
    return x, list(x.columns)


def make_model(params: dict | None = None):
    from xgboost import XGBRegressor

    base = {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.035,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 4,
    }
    if params:
        base.update(params)
    return XGBRegressor(**base)


def maybe_transform_target(y: pd.Series, use_log: bool) -> tuple[np.ndarray, callable]:
    if use_log:
        return np.log1p(y.to_numpy()), np.expm1
    return y.to_numpy(), lambda values: values


def tune_params(train_df: pd.DataFrame, target: str, features: list[str], use_log: bool) -> dict:
    fit_df = train_df[train_df["year"].between(2014, 2020)].copy()
    val_df = train_df[train_df["year"].between(2021, 2022)].copy()
    if fit_df.empty or val_df.empty:
        return {}
    grid = {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05],
        "min_child_weight": [1, 3, 6],
        "reg_lambda": [1.0, 3.0],
    }
    best = None
    for max_depth, learning_rate, min_child_weight, reg_lambda in product(
        grid["max_depth"], grid["learning_rate"], grid["min_child_weight"], grid["reg_lambda"]
    ):
        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "n_estimators": 350,
        }
        x_fit, cols = build_matrix(fit_df, features)
        x_val, _ = build_matrix(val_df, features)
        x_val = x_val.reindex(columns=cols, fill_value=0)
        y_fit, inv = maybe_transform_target(fit_df[target], use_log)
        model = make_model(params)
        model.fit(x_fit, y_fit)
        pred = inv(model.predict(x_val))
        score = np.sqrt(mean_squared_error(val_df[target].to_numpy(), pred))
        if best is None or score < best[0]:
            best = (score, params)
    return best[1] if best else {}


def run_one(data: pd.DataFrame, target: str, model_name: str, features: list[str], use_log: bool, tune: bool) -> tuple[dict, pd.DataFrame]:
    needed = [target, *features]
    model_data = data.dropna(subset=needed).copy()
    train = model_data[model_data["year"].between(2014, 2022)].copy()
    test = model_data[model_data["year"].between(2023, 2024)].copy()
    if train.empty or test.empty:
        raise ValueError(f"{target}/{model_name} 训练或测试数据为空")

    params = tune_params(train, target, features, use_log) if tune else {}
    x_train, cols = build_matrix(train, features)
    x_test, _ = build_matrix(test, features)
    x_test = x_test.reindex(columns=cols, fill_value=0)
    y_train, inv = maybe_transform_target(train[target], use_log)

    model = make_model(params)
    model.fit(x_train, y_train)
    pred = inv(model.predict(x_test))
    pred = np.maximum(pred, 0) if target in {"Chl_a", "SD", "TN", "TP"} else pred
    y_true = test[target].to_numpy()

    row = {
        "target": target,
        "model": model_name,
        "feature_count": len(cols),
        "train_rows": len(train),
        "test_rows": len(test),
        "log_target": use_log,
        **metrics_dict(y_true, pred),
        "params": str(params or "default"),
    }
    pred_df = test[["station", "date", "year", "month", target]].copy()
    pred_df[f"{target}_pred"] = pred
    pred_df["model"] = model_name
    return row, pred_df


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.dataset, parse_dates=["date"])
    data["year"] = data["date"].dt.year
    targets = TARGETS if args.target == "all" else [args.target]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_preds = []
    for target in targets:
        use_log = (target in {"Chl_a", "TN", "TP"}) and not args.no_log
        for model_name, features in FEATURE_SETS.items():
            row, pred_df = run_one(data, target, model_name, features, use_log, args.tune)
            all_rows.append(row)
            all_preds.append(pred_df)
            print(pd.DataFrame([row]).drop(columns=["params"]).to_string(index=False))

    metrics = pd.DataFrame(all_rows)
    preds = pd.concat(all_preds, ignore_index=True)
    suffix = "tuned" if args.tune else "default"
    metrics_path = args.out_dir / f"xgboost_optimized_{suffix}_metrics.csv"
    preds_path = args.out_dir / f"xgboost_optimized_{suffix}_predictions.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    preds.to_csv(preds_path, index=False, encoding="utf-8-sig")
    print(f"写出 {metrics_path}")
    print(f"写出 {preds_path}")


if __name__ == "__main__":
    main()
