from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATASET = Path("data/processed/model_dataset_surface_2014_2024_enhanced.csv")
DEFAULT_OUT_DIR = Path("reports/station_specific")
TARGETS = ["Chl_a", "SD", "TN", "TP"]

F1_FEATURES = [
    "month", "season_sin", "season_cos",
    "t2m", "u10", "v10", "tcc", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "rain_3d", "rain_14d", "rain_month",
    "tidal_range", "ntl_avg_rad",
    "t2m_lag1", "t2m_lag2", "t2m_roll3",
    "tcc_lag1", "tcc_lag2", "tcc_roll3",
    "wind_speed_lag1", "wind_speed_lag2", "wind_speed_roll3",
    "rain_month_lag1", "rain_month_lag2", "rain_month_roll3",
    "rain_14d_lag1", "rain_14d_lag2", "rain_14d_roll3",
    "tidal_range_lag1", "tidal_range_lag2", "tidal_range_roll3",
    "ntl_avg_rad_lag1", "ntl_avg_rad_lag2", "ntl_avg_rad_roll3",
]

F2_FEATURES = [
    "blue", "green", "red", "nir", "swir1", "swir2",
    "ndvi", "ndwi", "mndwi", "ndti",
    "blue_green", "green_red", "red_green", "nir_red", "red_nir",
    "valid_count", "landsat_missing", "landsat_imputed", "landsat_no_data",
    "ndwi_lag1", "ndwi_lag2", "ndwi_roll3",
    "mndwi_lag1", "mndwi_lag2", "mndwi_roll3",
    "ndti_lag1", "ndti_lag2", "ndti_roll3",
    "valid_count_lag1", "valid_count_lag2", "valid_count_roll3",
]

FEATURE_SETS = {
    "F1_process": F1_FEATURES,
    "F2_remote_sensing": F2_FEATURES,
    "F3_fusion": F1_FEATURES + F2_FEATURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按站点分别训练 XGBoost 水质模型。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target", default="all", choices=["all", *TARGETS])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def make_model():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=350,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=3.0,
        reg_alpha=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )


def transform_target(y: pd.Series, use_log: bool) -> tuple[np.ndarray, callable]:
    if use_log:
        return np.log1p(y.to_numpy()), np.expm1
    return y.to_numpy(), lambda values: values


def metric_row(target: str, station: str, model_name: str, train: pd.DataFrame, test: pd.DataFrame, y_pred: np.ndarray) -> dict:
    y = test[target].to_numpy()
    return {
        "target": target,
        "station": station,
        "model": model_name,
        "feature_count": len(FEATURE_SETS[model_name]),
        "train_rows": len(train),
        "test_rows": len(test),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
        "nse": float(nse(y, y_pred)),
    }


def run_one(data: pd.DataFrame, target: str, station: str, model_name: str, features: list[str]) -> tuple[dict, pd.DataFrame]:
    subset = data[data["station"].eq(station)].dropna(subset=[target, *features]).copy()
    train = subset[subset["year"].between(2014, 2022)].copy()
    test = subset[subset["year"].between(2023, 2024)].copy()
    if len(train) < 30 or len(test) < 6:
        raise ValueError(f"{station}/{target}/{model_name} 样本不足：train={len(train)}, test={len(test)}")

    use_log = target in {"Chl_a", "TN", "TP"}
    y_train, inv = transform_target(train[target], use_log)
    model = make_model()
    model.fit(train[features], y_train)
    pred = inv(model.predict(test[features]))
    pred = np.maximum(pred, 0)

    row = metric_row(target, station, model_name, train, test, pred)
    row["log_target"] = use_log
    pred_df = test[["station", "date", "year", "month", target]].copy()
    pred_df["model"] = model_name
    pred_df[f"{target}_pred"] = pred
    return row, pred_df


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.dataset, parse_dates=["date"])
    data["year"] = data["date"].dt.year
    targets = TARGETS if args.target == "all" else [args.target]
    stations = sorted(data["station"].dropna().unique())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = sorted({col for features in FEATURE_SETS.values() for col in features if col not in data.columns})
    if missing:
        raise SystemExit(f"缺少特征列：{missing}")

    rows = []
    preds = []
    for target in targets:
        for station in stations:
            for model_name, features in FEATURE_SETS.items():
                row, pred_df = run_one(data, target, station, model_name, features)
                rows.append(row)
                preds.append(pred_df)

    metrics = pd.DataFrame(rows)
    pred_all = pd.concat(preds, ignore_index=True)
    metrics_path = args.out_dir / "xgboost_station_specific_metrics.csv"
    pred_path = args.out_dir / "xgboost_station_specific_predictions.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pred_all.to_csv(pred_path, index=False, encoding="utf-8-sig")

    best = metrics.loc[metrics.groupby(["target", "station"])["rmse"].idxmin()].sort_values(["target", "station"])
    best_path = args.out_dir / "xgboost_station_specific_best_by_station.csv"
    best.to_csv(best_path, index=False, encoding="utf-8-sig")

    summary = (
        metrics.groupby(["target", "model"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            mean_nse=("nse", "mean"),
            stations=("station", "nunique"),
        )
        .sort_values(["target", "mean_rmse"])
    )
    summary_path = args.out_dir / "xgboost_station_specific_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("按站点独立建模汇总：")
    print(summary.round(4).to_string(index=False))
    print(f"写出 {metrics_path}")
    print(f"写出 {pred_path}")
    print(f"写出 {best_path}")
    print(f"写出 {summary_path}")


if __name__ == "__main__":
    main()
