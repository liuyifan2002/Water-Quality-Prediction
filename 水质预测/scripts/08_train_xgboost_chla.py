from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATASET = Path("data/processed/model_dataset_surface_2014_2024.csv")
DEFAULT_REPORT_DIR = Path("reports")

FEATURES = [
    "month",
    "t2m", "u10", "v10", "tcc",
    "rain_3d", "rain_14d", "rain_month",
    "tidal_range",
    "ntl_avg_rad",
    "blue", "green", "red", "nir", "swir1", "swir2",
    "ndwi", "mndwi", "ndti", "valid_count",
    "landsat_missing", "landsat_imputed", "landsat_no_data"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练表层水质目标变量的最小版 XGBoost 模型。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target", default="Chl_a", choices=["Chl_a", "SD", "TN", "TP"])
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    return parser.parse_args()


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def main() -> None:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("缺少 xgboost，请先安装：pip install xgboost") from exc

    args = parse_args()
    target_slug = args.target.lower()
    report = args.report or DEFAULT_REPORT_DIR / f"xgboost_{target_slug}_surface_metrics.csv"
    predictions = args.predictions or DEFAULT_REPORT_DIR / f"xgboost_{target_slug}_surface_predictions.csv"
    data = pd.read_csv(args.dataset, parse_dates=["date"])
    data["year"] = data["date"].dt.year
    missing_features = [col for col in FEATURES if col not in data.columns]
    if missing_features:
        raise SystemExit(f"模型数据集缺少特征列：{missing_features}")

    model_data = data.dropna(subset=[args.target, *FEATURES]).copy()
    train = model_data[model_data["year"].between(2014, 2022)]
    test = model_data[model_data["year"].between(2023, 2024)]
    if train.empty or test.empty:
        raise SystemExit(f"训练或测试数据为空：train={len(train)}, test={len(test)}")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4
    )
    model.fit(train[FEATURES], train[args.target])
    pred = model.predict(test[FEATURES])
    y = test[args.target].to_numpy()

    metrics = pd.DataFrame(
        [{
            "target": args.target,
            "train_rows": len(train),
            "test_rows": len(test),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": mean_absolute_error(y, pred),
            "r2": r2_score(y, pred),
            "nse": nse(y, pred)
        }]
    )
    pred_df = test[["station", "date", "year", "month", args.target]].copy()
    pred_df[f"{args.target}_pred"] = pred

    report.parent.mkdir(parents=True, exist_ok=True)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report, index=False, encoding="utf-8-sig")
    pred_df.to_csv(predictions, index=False, encoding="utf-8-sig")

    print(metrics.to_string(index=False))
    print(f"写出 {report}")
    print(f"写出 {predictions}")


if __name__ == "__main__":
    main()
