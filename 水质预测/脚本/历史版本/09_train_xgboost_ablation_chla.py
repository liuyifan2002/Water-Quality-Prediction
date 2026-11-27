from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATASET = Path("数据/处理后数据/衍生版本/model_dataset_surface_2014_2024.csv")
DEFAULT_REPORT_DIR = Path("reports")

F1_FEATURES = [
    "month",
    "t2m", "u10", "v10", "tcc",
    "rain_3d", "rain_14d", "rain_month",
    "tidal_range",
    "ntl_avg_rad"
]
F2_FEATURES = [
    "blue", "green", "red", "nir", "swir1", "swir2",
    "ndwi", "mndwi", "ndti", "valid_count",
    "landsat_missing", "landsat_imputed", "landsat_no_data"
]
FEATURE_SETS = {
    "F1_process": F1_FEATURES,
    "F2_remote_sensing": F2_FEATURES,
    "F3_fusion": F1_FEATURES + F2_FEATURES
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练表层水质目标变量的 F1/F2/F3 XGBoost 消融模型。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target", default="Chl_a", choices=["Chl_a", "SD", "TN", "TP"])
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--common-sample", action="store_true", help="F1/F2/F3 使用同一批完整样本。")
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
    if args.report is None:
        suffix = "ablation_common" if args.common_sample else "ablation"
        args.report = DEFAULT_REPORT_DIR / f"xgboost_{target_slug}_surface_{suffix}_metrics.csv"
    data = pd.read_csv(args.dataset, parse_dates=["date"])
    data["year"] = data["date"].dt.year
    if args.common_sample:
        all_features = sorted(set(F1_FEATURES + F2_FEATURES))
        missing = [col for col in all_features if col not in data.columns]
        if missing:
            raise SystemExit(f"共同样本缺少特征列：{missing}")
        data = data.dropna(subset=[args.target, *all_features]).copy()

    rows = []
    for name, features in FEATURE_SETS.items():
        missing = [col for col in features if col not in data.columns]
        if missing:
            rows.append({"model": name, "status": f"缺少特征：{missing}"})
            continue

        model_data = data.dropna(subset=[args.target, *features]).copy()
        train = model_data[model_data["year"].between(2014, 2022)]
        test = model_data[model_data["year"].between(2023, 2024)]
        if train.empty or test.empty:
            rows.append({"model": name, "status": f"训练或测试数据为空：train={len(train)}, test={len(test)}"})
            continue

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
        model.fit(train[features], train[args.target])
        pred = model.predict(test[features])
        y = test[args.target].to_numpy()
        rows.append(
            {
                "model": name,
                "target": args.target,
                "status": "ok",
                "feature_count": len(features),
                "train_rows": len(train),
                "test_rows": len(test),
                "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                "mae": mean_absolute_error(y, pred),
                "r2": r2_score(y, pred),
                "nse": nse(y, pred)
            }
        )

    report = pd.DataFrame(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print(f"写出 {args.report}")


if __name__ == "__main__":
    main()
