"""比较独立 XGBoost、共享水层 XGBoost 与站点水层混合效应 GPBoost。

本脚本是扩展实验。所有模型共享同一 target-specific sample_id 外层划分，
从而避免同一站点同日的其他水层进入训练集而造成间接标签泄漏。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
FORMAL_TRAINING_SCRIPT = ROOT / "脚本" / "04_模型训练" / "05_train_full_station_models.py"
DEFAULT_DATASET = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
DEFAULT_REPORT_DIR = ROOT / "结果" / "扩展实验" / "分层混合效应梯度提升_20260905"

MODEL_VARIANTS = [
    "independent_xgboost",
    "pooled_depth_xgboost",
    "gpboost_station_random_effect",
    "gpboost_station_depth_random_effect",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a leakage-controlled hierarchical GPBoost A/B experiment for all full-station tasks."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--targets", nargs="*", choices=["chla", "secchi", "tn", "tp"], default=["chla", "secchi", "tn", "tp"])
    parser.add_argument("--feature-sets", nargs="*", choices=["f1_history", "f3_history_remote"], default=["f1_history"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--min-test-rows", type=int, default=50)
    return parser.parse_args()


def load_formal_training_module():
    """复用正式主链的目标映射、特征组与预处理，避免定义漂移。"""
    spec = importlib.util.spec_from_file_location("formal_full_station_training", FORMAL_TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load formal training script: {FORMAL_TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return np.nan
    return float(1 - np.sum((y_true - y_pred) ** 2) / denominator)


def transform_target(values: pd.Series) -> tuple[np.ndarray, callable]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.nanmin(numeric) >= 0:
        return np.log1p(numeric), np.expm1
    return numeric, lambda array: array


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nse": nse(y_true, y_pred),
    }


def assert_no_same_day_water_quality_features(features: list[str], target_col: str, target_map: dict[str, str]) -> None:
    forbidden = set(target_map.values()) | {
        "turbidity",
        "turbidity_ntu",
        "salinity",
        "do",
        "dissolved_oxygen",
        "ph",
    }
    direct_forbidden = sorted(set(features) & forbidden)
    if direct_forbidden:
        raise ValueError(f"Detected forbidden same-day water-quality features: {direct_forbidden}")
    if target_col in features:
        raise ValueError(f"Target column cannot be an input feature: {target_col}")


def split_by_sample_id(subset: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_ids = np.sort(subset["sample_id"].dropna().astype(str).unique())
    train_ids, test_ids = train_test_split(sample_ids, test_size=test_size, random_state=random_state, shuffle=True)
    train_id_set = set(train_ids)
    train = subset[subset["sample_id"].astype(str).isin(train_id_set)].copy()
    test = subset[~subset["sample_id"].astype(str).isin(train_id_set)].copy()
    manifest = pd.DataFrame(
        {
            "sample_id": np.concatenate([train_ids, test_ids]),
            "outer_split": ["train"] * len(train_ids) + ["test"] * len(test_ids),
        }
    )
    return train, test, manifest


def xgb_model(args: argparse.Namespace) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="reg:squarederror",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )


def gpboost_model(args: argparse.Namespace, x_train: pd.DataFrame, y_train: np.ndarray, groups_train: np.ndarray):
    try:
        import gpboost as gpb
    except ImportError as exc:
        raise SystemExit("Missing gpboost. Install it with `python -m pip install gpboost`.") from exc

    train_set = gpb.Dataset(x_train, label=y_train, free_raw_data=False)
    gp_model = gpb.GPModel(group_data=groups_train)
    params = {
        "objective": "regression_l2",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": 15,
        "feature_fraction": args.colsample_bytree,
        # GPBoost with a random-effect component does not support row bagging.
        "bagging_freq": 0,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 20,
        "seed": args.random_state,
        "num_threads": args.n_jobs,
        "verbose": -1,
    }
    return gpb.train(params=params, train_set=train_set, num_boost_round=args.n_estimators, gp_model=gp_model)


def encode_groups(train: pd.DataFrame, test: pd.DataFrame, group_level: str) -> tuple[np.ndarray, np.ndarray]:
    if group_level == "station":
        keys = pd.concat([train["station"], test["station"]], ignore_index=True).astype(str)
    elif group_level == "station_depth":
        combined = pd.concat([train, test], ignore_index=True)
        keys = combined["station"].astype(str) + "__" + combined["depth"].astype(str)
    else:
        raise ValueError(f"Unsupported group level: {group_level}")
    codes, _ = pd.factorize(keys)
    return codes[: len(train)], codes[len(train) :]


def prediction_frame(
    test: pd.DataFrame,
    target_key: str,
    target_col: str,
    feature_set: str,
    model_variant: str,
    predicted: np.ndarray,
    response_variance: np.ndarray | None = None,
) -> pd.DataFrame:
    columns = ["obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "year", "month", "depth", target_col]
    frame = test[[column for column in columns if column in test.columns]].copy()
    frame = frame.rename(columns={target_col: "observed"})
    frame["target"] = target_key
    frame["feature_set"] = feature_set
    frame["model_variant"] = model_variant
    frame["predicted"] = predicted
    if response_variance is not None:
        frame["gpboost_response_variance_log_scale"] = response_variance
    return frame


def metric_row(
    test: pd.DataFrame,
    target_key: str,
    target_col: str,
    depth: str,
    feature_set: str,
    model_variant: str,
    feature_count: int,
    metrics: dict[str, float],
    random_state: int,
) -> dict[str, object]:
    return {
        "target": target_key,
        "target_column": target_col,
        "depth": depth,
        "feature_set": feature_set,
        "model_variant": model_variant,
        "feature_count": feature_count,
        "test_rows": len(test),
        **metrics,
        "split_unit": "sample_id",
        "split_mode": "random_80_20_shared_across_depths",
        "split_random_state": random_state,
        "same_day_water_quality_covariates": "excluded",
        "status": "ok",
        "error": "",
    }


def run_feature_set(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_key: str,
    target_col: str,
    feature_set: str,
    features: list[str],
    target_map: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]]]:
    assert_no_same_day_water_quality_features(features, target_col, target_map)
    metrics_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    feature_manifest: list[dict[str, object]] = []

    pooled_features = features + ["depth_order"]
    for variant in MODEL_VARIANTS:
        feature_manifest.append(
            {
                "target": target_key,
                "feature_set": feature_set,
                "model_variant": variant,
                "features_json": json.dumps(features if variant == "independent_xgboost" else pooled_features, ensure_ascii=False),
            }
        )

    for depth in ("surface", "middle", "bottom"):
        train_depth = train[train["depth"] == depth].copy()
        test_depth = test[test["depth"] == depth].copy()
        if len(train_depth) < args.min_train_rows or len(test_depth) < args.min_test_rows:
            continue
        y_train, inverse_transform = transform_target(train_depth[target_col])
        model = xgb_model(args)
        model.fit(train_depth[features], y_train)
        predicted = np.maximum(inverse_transform(model.predict(test_depth[features])), 0)
        y_test = test_depth[target_col].to_numpy(dtype=float)
        metrics_rows.append(
            metric_row(
                test_depth,
                target_key,
                target_col,
                depth,
                feature_set,
                "independent_xgboost",
                len(features),
                evaluate(y_test, predicted),
                args.random_state,
            )
        )
        predictions.append(prediction_frame(test_depth, target_key, target_col, feature_set, "independent_xgboost", predicted))

    y_train, inverse_transform = transform_target(train[target_col])
    pooled_xgb = xgb_model(args)
    pooled_xgb.fit(train[pooled_features], y_train)
    pooled_predicted = np.maximum(inverse_transform(pooled_xgb.predict(test[pooled_features])), 0)

    gp_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    gp_errors: dict[str, str] = {}
    for model_variant, group_level in (
        ("gpboost_station_random_effect", "station"),
        ("gpboost_station_depth_random_effect", "station_depth"),
    ):
        try:
            train_groups, test_groups = encode_groups(train, test, group_level)
            hierarchical_model = gpboost_model(args, train[pooled_features], y_train, train_groups)
            gp_prediction = hierarchical_model.predict(
                data=test[pooled_features],
                group_data_pred=test_groups,
                predict_var=True,
            )
            gp_predictions[model_variant] = (
                np.maximum(inverse_transform(np.asarray(gp_prediction["response_mean"])), 0),
                np.asarray(gp_prediction["response_var"]),
            )
        except Exception as exc:  # Preserve the A/B record when a native GPBoost routine fails.
            gp_errors[model_variant] = f"{type(exc).__name__}: {exc}"

    for depth in ("surface", "middle", "bottom"):
        depth_mask = test["depth"].eq(depth).to_numpy()
        test_depth = test.loc[depth_mask].copy()
        if len(test_depth) < args.min_test_rows:
            continue
        y_test = test_depth[target_col].to_numpy(dtype=float)
        pooled_depth_predicted = pooled_predicted[depth_mask]
        metrics_rows.append(
            metric_row(
                test_depth,
                target_key,
                target_col,
                depth,
                feature_set,
                "pooled_depth_xgboost",
                len(pooled_features),
                evaluate(y_test, pooled_depth_predicted),
                args.random_state,
            )
        )
        predictions.append(prediction_frame(test_depth, target_key, target_col, feature_set, "pooled_depth_xgboost", pooled_depth_predicted))
        for model_variant in ("gpboost_station_random_effect", "gpboost_station_depth_random_effect"):
            if model_variant in gp_predictions:
                predicted, response_variance = gp_predictions[model_variant]
                predicted_depth = predicted[depth_mask]
                metrics_rows.append(
                    metric_row(
                        test_depth,
                        target_key,
                        target_col,
                        depth,
                        feature_set,
                        model_variant,
                        len(pooled_features),
                        evaluate(y_test, predicted_depth),
                        args.random_state,
                    )
                )
                predictions.append(
                    prediction_frame(
                        test_depth,
                        target_key,
                        target_col,
                        feature_set,
                        model_variant,
                        predicted_depth,
                        response_variance[depth_mask],
                    )
                )
            else:
                failed = metric_row(
                    test_depth,
                    target_key,
                    target_col,
                    depth,
                    feature_set,
                    model_variant,
                    len(pooled_features),
                    {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "nse": np.nan},
                    args.random_state,
                )
                failed["status"] = "failed_runtime"
                failed["error"] = gp_errors[model_variant]
                metrics_rows.append(failed)

    return metrics_rows, predictions, feature_manifest


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["feature_set", "model_variant"], as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            min_r2=("r2", "min"),
            mean_rmse=("rmse", "mean"),
            task_count=("r2", "size"),
            successful_task_count=("r2", "count"),
        )
        .sort_values(["feature_set", "mean_r2"], ascending=[True, False])
    )
    return summary


def markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include="number"):
        formatted[column] = formatted[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    headers = [str(column) for column in formatted.columns]
    rows = [[str(value) for value in row] for row in formatted.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def promotion_lines(metrics: pd.DataFrame) -> list[str]:
    baseline = metrics[metrics["model_variant"] == "independent_xgboost"][["target", "depth", "feature_set", "r2"]].rename(columns={"r2": "baseline_r2"})
    lines: list[str] = []
    for model_variant in ("pooled_depth_xgboost", "gpboost_station_random_effect", "gpboost_station_depth_random_effect"):
        candidate = metrics[metrics["model_variant"] == model_variant][["target", "depth", "feature_set", "r2", "status"]]
        merged = candidate.merge(baseline, on=["target", "depth", "feature_set"], how="left")
        valid = merged[merged["status"].eq("ok")].copy()
        valid["r2_delta"] = valid["r2"] - valid["baseline_r2"]
        positive = int((valid["r2_delta"] > 0).sum())
        total = len(merged)
        mean_delta = float(valid["r2_delta"].mean()) if not valid.empty else np.nan
        promoted = len(valid) == total and positive >= 8 and mean_delta > 0
        verdict = "候选可推广" if promoted else "不推广"
        lines.append(f"- `{model_variant}`：有效任务 {len(valid)}/{total}，正增益任务 {positive}/{len(valid)}，平均 ΔR2={mean_delta:.4f}，判定：**{verdict}**。")
    return lines


def write_report(report_dir: Path, summary: pd.DataFrame, metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# 分层混合效应梯度提升 A/B 结果",
        "",
        "- 性质：扩展实验，不改写正式 full_station 主结果。",
        "- 划分：以 sample_id 为单位的固定随机 80/20 划分，确保同站点同日的三水层不会跨训练集与测试集。",
        "- 防泄漏：不使用同日其他水质指标；仅使用目标变量自身历史特征及外部过程、空间、遥感信息。",
        "- 模型：独立 XGBoost、共享水层 XGBoost、station 随机效应 GPBoost、station × depth 随机效应 GPBoost。",
        "- 参数可比性：各模型固定使用相同树轮数、最大深度、学习率和列采样比例；GPBoost 的随机效应训练不支持行采样，因此其 bagging 已关闭。",
        f"- 随机种子：{args.random_state}。",
        "",
        "## 跨任务汇总",
        "",
        markdown_table(summary),
        "",
        "## 推进判定规则",
        "",
        "只有 GPBoost 在同一 feature_set 下相对独立 XGBoost 呈现跨任务稳定增益，且未出现多数任务明显退化时，才进入下一轮质量感知遥感融合实验。station 与 station × depth 两种随机效应层级必须分别报告。",
        "",
        "## 本次推广判定",
        "",
        *promotion_lines(metrics),
        "具体任务结果以 `gpboost_hierarchical_metrics.csv` 为准。",
    ]
    (report_dir / "实验结果说明.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    formal = load_formal_training_module()
    data = formal.prepare_dataset(args.dataset)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []

    for target_key in args.targets:
        target_col = formal.TARGET_MAP[target_key]
        subset = data[data[target_col].notna()].copy()
        train, test, split_manifest = split_by_sample_id(subset, args.test_size, args.random_state)
        split_manifest.insert(0, "target", target_key)
        split_frames.append(split_manifest)

        feature_sets = formal.build_feature_sets(data, target_col)
        for feature_set in args.feature_sets:
            features = feature_sets[feature_set]
            if not features:
                raise ValueError(f"No available features for {target_key}/{feature_set}")
            rows, predictions, manifest = run_feature_set(
                train,
                test,
                target_key,
                target_col,
                feature_set,
                features,
                formal.TARGET_MAP,
                args,
            )
            metric_rows.extend(rows)
            prediction_frames.extend(predictions)
            feature_rows.extend(manifest)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    split_manifest = pd.concat(split_frames, ignore_index=True)
    feature_manifest = pd.DataFrame(feature_rows)
    summary = build_summary(metrics)

    metrics.to_csv(args.report_dir / "gpboost_hierarchical_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.report_dir / "gpboost_hierarchical_predictions.csv", index=False, encoding="utf-8-sig")
    split_manifest.to_csv(args.report_dir / "gpboost_hierarchical_sample_split.csv", index=False, encoding="utf-8-sig")
    feature_manifest.to_csv(args.report_dir / "gpboost_hierarchical_feature_manifest.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.report_dir / "gpboost_hierarchical_summary.csv", index=False, encoding="utf-8-sig")
    write_report(args.report_dir, summary, metrics, args)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
