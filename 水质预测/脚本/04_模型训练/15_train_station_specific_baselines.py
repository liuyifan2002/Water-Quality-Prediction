from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATASET = Path("数据/处理后数据/衍生版本/model_dataset_surface_2014_2024_enhanced.csv")
DEFAULT_OUT_DIR = Path("结果/扩展实验/station_specific")
TARGETS = ["Chl_a", "SD", "TN", "TP"]

F3_FEATURES = [
    "month", "season_sin", "season_cos",
    "t2m", "u10", "v10", "tcc", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "rain_3d", "rain_14d", "rain_month", "tidal_range", "ntl_avg_rad",
    "t2m_lag1", "t2m_lag2", "t2m_roll3",
    "tcc_lag1", "tcc_lag2", "tcc_roll3",
    "wind_speed_lag1", "wind_speed_lag2", "wind_speed_roll3",
    "rain_month_lag1", "rain_month_lag2", "rain_month_roll3",
    "rain_14d_lag1", "rain_14d_lag2", "rain_14d_roll3",
    "tidal_range_lag1", "tidal_range_lag2", "tidal_range_roll3",
    "ntl_avg_rad_lag1", "ntl_avg_rad_lag2", "ntl_avg_rad_roll3",
    "blue", "green", "red", "nir", "swir1", "swir2",
    "ndvi", "ndwi", "mndwi", "ndti",
    "blue_green", "green_red", "red_green", "nir_red", "red_nir",
    "valid_count", "landsat_missing", "landsat_imputed", "landsat_no_data",
    "ndwi_lag1", "ndwi_lag2", "ndwi_roll3",
    "mndwi_lag1", "mndwi_lag2", "mndwi_roll3",
    "ndti_lag1", "ndti_lag2", "ndti_roll3",
    "valid_count_lag1", "valid_count_lag2", "valid_count_roll3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按站点训练 Ridge 与 Random Forest 基线模型。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def metric_row(target: str, station: str, model_name: str, train: pd.DataFrame, test: pd.DataFrame, pred: np.ndarray) -> dict:
    y = test[target].to_numpy()
    return {
        "target": target,
        "station": station,
        "model": model_name,
        "train_rows": len(train),
        "test_rows": len(test),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "nse": float(nse(y, pred)),
    }


def build_models() -> dict:
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ),
    }


def transform_y(target: str, y: pd.Series) -> tuple[np.ndarray, callable]:
    if target in {"Chl_a", "TN", "TP"}:
        return np.log1p(y.to_numpy()), np.expm1
    return y.to_numpy(), lambda values: values


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.dataset, parse_dates=["date"])
    data["year"] = data["date"].dt.year
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    preds = []
    for target in TARGETS:
        for station, subset in data.groupby("station"):
            subset = subset.dropna(subset=[target, *F3_FEATURES]).copy()
            train = subset[subset["year"].between(2014, 2022)].copy()
            test = subset[subset["year"].between(2023, 2024)].copy()
            if len(train) < 30 or len(test) < 6:
                continue
            y_train, inv = transform_y(target, train[target])
            for model_name, model in build_models().items():
                model.fit(train[F3_FEATURES], y_train)
                pred = np.maximum(inv(model.predict(test[F3_FEATURES])), 0)
                rows.append(metric_row(target, station, model_name, train, test, pred))
                pred_df = test[["station", "date", "year", "month", target]].copy()
                pred_df["model"] = model_name
                pred_df[f"{target}_pred"] = pred
                preds.append(pred_df)

    metrics = pd.DataFrame(rows)
    pred_all = pd.concat(preds, ignore_index=True)
    metrics_path = args.out_dir / "station_specific_baseline_metrics.csv"
    pred_path = args.out_dir / "station_specific_baseline_predictions.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pred_all.to_csv(pred_path, index=False, encoding="utf-8-sig")

    pooled_rows = []
    for target in TARGETS:
        col = f"{target}_pred"
        sub = pred_all.dropna(subset=[target, col])
        for model_name, group in sub.groupby("model"):
            pooled_rows.append(metric_row(target, "ALL", model_name, group, group, group[col].to_numpy()))
    pooled = pd.DataFrame(pooled_rows).drop(columns=["train_rows"])
    pooled_path = args.out_dir / "station_specific_baseline_pooled_eval.csv"
    pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")

    print("站点独立基线模型 pooled evaluation：")
    print(pooled.round(4).to_string(index=False))
    print(f"写出 {metrics_path}")
    print(f"写出 {pred_path}")
    print(f"写出 {pooled_path}")


if __name__ == "__main__":
    main()
