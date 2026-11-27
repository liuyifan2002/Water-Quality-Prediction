from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    parser = argparse.ArgumentParser(description="站点独立自回归增强模型：F3 特征 + 上期水质状态。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def evaluate(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "nse": float(nse(y_true, pred)),
    }


def target_transform(target: str, y: pd.Series) -> tuple[np.ndarray, callable]:
    if target in {"Chl_a", "TN", "TP"}:
        return np.log1p(y.to_numpy()), np.expm1
    return y.to_numpy(), lambda values: values


def add_target_history(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values(["station", "date"]).copy()
    grouped = out.groupby("station", sort=False)[target]
    cols = [
        f"{target}_lag1",
        f"{target}_lag2",
        f"{target}_roll3",
        f"{target}_delta1",
    ]
    out[cols[0]] = grouped.shift(1)
    out[cols[1]] = grouped.shift(2)
    out[cols[2]] = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    out[cols[3]] = out[cols[0]] - out[cols[1]]
    return out, cols


def make_models(target: str) -> dict:
    from xgboost import XGBRegressor

    common_xgb = {
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
    }
    return {
        "AR_F3_XGBoost_regularized": XGBRegressor(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=3,
            reg_lambda=6.0,
            **common_xgb,
        ),
        "AR_F3_XGBoost_shallow": XGBRegressor(
            n_estimators=220,
            max_depth=1,
            learning_rate=0.05,
            min_child_weight=6,
            reg_lambda=8.0,
            **common_xgb,
        ),
        "AR_F3_RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        ),
        "AR_F3_ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        ),
    }


def prepare_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fill_values = train[features].median(numeric_only=True)
    global_values = pd.concat([train[features], test[features]], axis=0).median(numeric_only=True)
    fill_values = fill_values.fillna(global_values).fillna(0)
    return train[features].fillna(fill_values), test[features].fillna(fill_values)


def choose_model(train: pd.DataFrame, target: str, features: list[str]) -> str:
    fit = train[train["year"].between(2014, 2020)].copy()
    val = train[train["year"].between(2021, 2022)].copy()
    if len(fit) < 30 or len(val) < 8:
        return "AR_F3_XGBoost_regularized"

    x_fit, x_val = prepare_matrix(fit, val, features)
    y_fit, inv = target_transform(target, fit[target])
    y_val = val[target].to_numpy()
    best_name = None
    best_rmse = np.inf
    for name, model in make_models(target).items():
        model.fit(x_fit, y_fit)
        pred = np.maximum(inv(model.predict(x_val)), 0)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
    return str(best_name)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.dataset, parse_dates=["date"])
    data["year"] = data["date"].dt.year

    rows = []
    pred_frames = []
    feature_rows = []

    for target in TARGETS:
        target_data, target_history = add_target_history(data, target)
        features = F3_FEATURES + target_history
        for station, group in target_data.groupby("station"):
            subset = group.dropna(subset=[target]).copy()
            train = subset[subset["year"].between(2014, 2022)].copy()
            test = subset[subset["year"].between(2023, 2024)].copy()
            if len(train) < 30 or len(test) < 6:
                continue

            selected = choose_model(train, target, features)
            x_train, x_test = prepare_matrix(train, test, features)
            y_train, inv = target_transform(target, train[target])
            model = make_models(target)[selected]
            model.fit(x_train, y_train)
            pred = np.maximum(inv(model.predict(x_test)), 0)

            metrics = evaluate(test[target].to_numpy(), pred)
            rows.append(
                {
                    "target": target,
                    "station": station,
                    "model": selected,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    **metrics,
                    "feature_count": len(features),
                    "target_history_features": ",".join(target_history),
                }
            )

            pred_df = test[["station", "date", "year", "month", target]].copy()
            pred_df["model"] = selected
            pred_df[f"{target}_pred"] = pred
            pred_df[f"{target}_lag1"] = test[f"{target}_lag1"].to_numpy()
            pred_df["abs_error"] = np.abs(test[target].to_numpy() - pred)
            pred_frames.append(pred_df)

            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame(
                    {
                        "target": target,
                        "station": station,
                        "model": selected,
                        "feature": features,
                        "importance": model.feature_importances_,
                    }
                )
                feature_rows.append(imp.sort_values("importance", ascending=False).head(15))

    metrics_all = pd.DataFrame(rows)
    preds_all = pd.concat(pred_frames, ignore_index=True)
    importance_all = pd.concat(feature_rows, ignore_index=True)

    metrics_path = args.out_dir / "station_specific_autoregressive_metrics.csv"
    preds_path = args.out_dir / "station_specific_autoregressive_predictions.csv"
    importance_path = args.out_dir / "station_specific_autoregressive_top_features.csv"
    pooled_path = args.out_dir / "station_specific_autoregressive_pooled_eval.csv"

    metrics_all.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    preds_all.to_csv(preds_path, index=False, encoding="utf-8-sig")
    importance_all.to_csv(importance_path, index=False, encoding="utf-8-sig")

    pooled_rows = []
    for target in TARGETS:
        pred_col = f"{target}_pred"
        sub = preds_all.dropna(subset=[target, pred_col])
        pooled_rows.append(
            {
                "target": target,
                "model": "AR_F3_selected_by_station",
                "test_rows": len(sub),
                **evaluate(sub[target].to_numpy(), sub[pred_col].to_numpy()),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")

    print("站点独立自回归增强模型 pooled evaluation:")
    print(pooled.round(4).to_string(index=False))
    print(f"写出 {metrics_path}")
    print(f"写出 {preds_path}")
    print(f"写出 {importance_path}")
    print(f"写出 {pooled_path}")


if __name__ == "__main__":
    main()
