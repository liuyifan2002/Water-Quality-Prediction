from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATASET = Path("data/processed/model_dataset_surface_2014_2024_enhanced.csv")
DEFAULT_OUT_DIR = Path("reports/station_specific")
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
    parser = argparse.ArgumentParser(description="按站点对 F3 XGBoost 做小网格调参。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def transform_y(target: str, y: pd.Series) -> tuple[np.ndarray, callable]:
    if target in {"Chl_a", "TN", "TP"}:
        return np.log1p(y.to_numpy()), np.expm1
    return y.to_numpy(), lambda values: values


def make_model(params: dict):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=params["min_child_weight"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "nse": float(nse(y, pred)),
    }


def tune_one(subset: pd.DataFrame, target: str) -> dict:
    fit = subset[subset["year"].between(2014, 2020)].copy()
    val = subset[subset["year"].between(2021, 2022)].copy()
    if len(fit) < 30 or len(val) < 12:
        return {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.03, "min_child_weight": 3, "reg_lambda": 3.0}

    grid = []
    for max_depth, learning_rate, min_child_weight, reg_lambda, n_estimators in product(
        [1, 2, 3],
        [0.02, 0.03, 0.05],
        [1, 3, 6],
        [1.0, 3.0, 6.0],
        [250, 350],
    ):
        grid.append(
            {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "min_child_weight": min_child_weight,
                "reg_lambda": reg_lambda,
                "n_estimators": n_estimators,
            }
        )

    y_fit, inv = transform_y(target, fit[target])
    y_val = val[target].to_numpy()
    best = None
    for params in grid:
        model = make_model(params)
        model.fit(fit[F3_FEATURES], y_fit)
        pred = np.maximum(inv(model.predict(val[F3_FEATURES])), 0)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        if best is None or rmse < best[0]:
            best = (rmse, params)
    return best[1]


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.dataset, parse_dates=["date"])
    data["year"] = data["date"].dt.year
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    preds = []
    for target in TARGETS:
        for station, group in data.groupby("station"):
            subset = group.dropna(subset=[target, *F3_FEATURES]).copy()
            train = subset[subset["year"].between(2014, 2022)].copy()
            test = subset[subset["year"].between(2023, 2024)].copy()
            if len(train) < 30 or len(test) < 6:
                continue
            params = tune_one(train, target)
            y_train, inv = transform_y(target, train[target])
            model = make_model(params)
            model.fit(train[F3_FEATURES], y_train)
            pred = np.maximum(inv(model.predict(test[F3_FEATURES])), 0)
            row = {
                "target": target,
                "station": station,
                "model": "F3_fusion_tuned",
                "train_rows": len(train),
                "test_rows": len(test),
                **evaluate(test[target].to_numpy(), pred),
                "params": str(params),
            }
            rows.append(row)
            pred_df = test[["station", "date", "year", "month", target]].copy()
            pred_df["model"] = "F3_fusion_tuned"
            pred_df[f"{target}_pred"] = pred
            preds.append(pred_df)

    metrics = pd.DataFrame(rows)
    pred_all = pd.concat(preds, ignore_index=True)
    metrics_path = args.out_dir / "xgboost_station_specific_f3_tuned_metrics.csv"
    pred_path = args.out_dir / "xgboost_station_specific_f3_tuned_predictions.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pred_all.to_csv(pred_path, index=False, encoding="utf-8-sig")

    pooled_rows = []
    for target in TARGETS:
        col = f"{target}_pred"
        sub = pred_all.dropna(subset=[target, col])
        y = sub[target].to_numpy()
        pred = sub[col].to_numpy()
        pooled_rows.append({"target": target, "model": "F3_fusion_tuned", "test_rows": len(sub), **evaluate(y, pred)})
    pooled = pd.DataFrame(pooled_rows)
    pooled_path = args.out_dir / "xgboost_station_specific_f3_tuned_pooled_eval.csv"
    pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")

    print("站点独立 F3 调参 pooled evaluation：")
    print(pooled.round(4).to_string(index=False))
    print(f"写出 {metrics_path}")
    print(f"写出 {pred_path}")
    print(f"写出 {pooled_path}")


if __name__ == "__main__":
    main()
