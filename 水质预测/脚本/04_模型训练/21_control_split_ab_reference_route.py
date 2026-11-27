"""控制 A/B：固定模型与特征，仅比较随机划分和严格时间外推的性能差异。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_DATASET = Path("数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv")
DEFAULT_REPORT_DIR = Path("结果/扩展实验/划分方式控制AB_20260905")
REFERENCE_SCRIPT = Path(__file__).with_name("20_run_reference_route_random_split.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定模型与特征，仅改变数据划分方式的控制 A/B。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def load_reference_module():
    spec = importlib.util.spec_from_file_location("reference_route", REFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载特征定义脚本：{REFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if denominator == 0 else 1 - np.sum((y_true - y_pred) ** 2) / denominator


def train_only_fill(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    fill_values = train[features].median(numeric_only=True).fillna(0.0)
    return train[features].fillna(fill_values).to_numpy(), test[features].fillna(fill_values).to_numpy()


def make_xgboost(args: argparse.Namespace):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("缺少 xgboost，请先安装：pip install xgboost") from exc
    return XGBRegressor(
        n_estimators=args.n_estimators, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        random_state=args.random_state, n_jobs=args.n_jobs,
    )


def evaluate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray]:
    x_train, x_test = train_only_fill(train, test, features)
    y_train_raw = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    use_log = np.nanmin(y_train_raw) >= 0
    y_train = np.log1p(y_train_raw) if use_log else y_train_raw
    inverse = np.expm1 if use_log else lambda values: values
    model = make_xgboost(args)
    model.fit(x_train, y_train)
    pred = np.maximum(inverse(model.predict(x_test)), 0)
    return {
        "r2": r2_score(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "mae": mean_absolute_error(y_test, pred),
        "nse": nse(y_test, pred),
        "target_transform": "log1p" if use_log else "none",
    }, pred


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"未找到主表：{args.dataset}")
    reference = load_reference_module()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.dataset, parse_dates=["date"], low_memory=False)
    data.columns = [str(column).replace("\ufeff", "") for column in data.columns]
    data = reference.normalize_bool_like_columns(data)
    data["water_zone_en"] = data["water_zone_en"].fillna("unknown")
    data["water_zone_code"] = pd.Categorical(data["water_zone_en"]).codes.astype("int16")
    numeric_columns = set(reference.BASELINE_FEATURES + reference.PROCESS_FEATURES + reference.REMOTE_FEATURES + list(reference.TARGET_MAP.values()))
    for base in reference.PROCESS_LAG_BASES:
        for suffix in reference.PROCESS_LAG_SUFFIXES:
            numeric_columns.add(f"{base}{suffix}")
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    sets = reference.feature_sets(data)
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for target_key, target in reference.TARGET_MAP.items():
        for depth in reference.DEPTHS:
            subset = data.loc[data["depth"].eq(depth)].dropna(subset=[target]).copy()
            chronological_train = subset.loc[subset["date"].dt.year.between(2014, 2022)].copy()
            chronological_test = subset.loc[subset["date"].dt.year.between(2023, 2024)].copy()
            random_train, random_test = train_test_split(
                subset, test_size=args.test_size, random_state=args.random_state, shuffle=True,
            )
            split_pairs = {
                "random_80_20": (random_train, random_test),
                "time_2014_2022_to_2023_2024": (chronological_train, chronological_test),
            }
            for set_name, requested_features in sets.items():
                # 固定完整样本中的可用字段，确保两种划分使用完全一致的特征列。
                features = [name for name in requested_features if subset[name].notna().any()]
                forbidden = set(features) & reference.FORBIDDEN_WATER_QUALITY_COLUMNS
                if forbidden:
                    raise RuntimeError(f"检测到禁止的同日水质字段：{sorted(forbidden)}")
                for split_name, (train, test) in split_pairs.items():
                    scores, pred = evaluate_split(train, test, target, features, args)
                    metrics_rows.append({
                        "target": target_key, "target_column": target, "depth": depth,
                        "feature_set": set_name, "model": "XGBoost", "split": split_name,
                        "feature_count": len(features), "features_json": json.dumps(features, ensure_ascii=False),
                        "train_rows": len(train), "test_rows": len(test), "random_state": args.random_state,
                        "n_estimators": args.n_estimators, "max_depth": 4, "learning_rate": 0.05,
                        **scores,
                    })
                    prediction_frame = test[["obs_id", "sample_id", "station", "date", "depth", target]].copy()
                    prediction_frame["target"] = target_key
                    prediction_frame["feature_set"] = set_name
                    prediction_frame["split"] = split_name
                    prediction_frame["prediction"] = pred
                    prediction_frames.append(prediction_frame)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.report_dir / "划分方式控制AB_模型指标.csv", index=False, encoding="utf-8-sig")
    random_metrics = metrics.loc[metrics["split"].eq("random_80_20")]
    time_metrics = metrics.loc[metrics["split"].eq("time_2014_2022_to_2023_2024")]
    comparison = random_metrics.merge(
        time_metrics,
        on=["target", "target_column", "depth", "feature_set", "model", "feature_count", "features_json", "random_state", "n_estimators", "max_depth", "learning_rate"],
        suffixes=("_random", "_time"),
    )
    comparison["r2_loss_time_minus_random"] = comparison["r2_random"] - comparison["r2_time"]
    comparison["rmse_increase_time_minus_random"] = comparison["rmse_time"] - comparison["rmse_random"]
    comparison.to_csv(args.report_dir / "划分方式控制AB_性能差异.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.report_dir / "划分方式控制AB_预测值.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "control_rule": "同一主表、同一任务、同一特征、同一 XGBoost 参数、同一 log1p 和训练集填补；仅改变划分方式",
        "random_split": "随机 80/20，random_state=42",
        "time_split": "2014-2022 训练，2023-2024 测试",
        "same_day_water_quality_allowed": False,
        "same_day_water_quality_overlap": "none",
    }]).to_csv(args.report_dir / "划分方式控制AB_实验说明.csv", index=False, encoding="utf-8-sig")
    show = comparison[["target", "depth", "feature_set", "r2_random", "r2_time", "r2_loss_time_minus_random", "rmse_random", "rmse_time"]]
    print(show.round(4).to_string(index=False))
    print(f"已写出：{args.report_dir}")


if __name__ == "__main__":
    main()
