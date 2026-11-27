"""验证 TabM / TabPFN 在候选水质任务上的稳健性。

本脚本只服务扩展实验：Chl-a 三水层、透明度三水层与 TN 底层。
随机重复验证保留正式随机留出中允许使用的目标历史特征；跨年份和
留站点验证移除目标自身历史特征，避免把未来期或未见站点水质带入输入。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split


DEFAULT_TASKS = [
    "chla_surface",
    "chla_middle",
    "chla_bottom",
    "secchi_surface",
    "secchi_middle",
    "secchi_bottom",
    "tn_bottom",
]
MODELS = ["xgboost", "tabm", "tabpfn"]
MODES = ["random_seeds", "time_holdout", "station_5fold"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证候选表格模型的随机、时间与空间泛化表现。")
    parser.add_argument("--dataset", type=Path, default=Path("data/model_table.csv"))
    parser.add_argument("--model-config", type=Path, default=Path("config.json"))
    parser.add_argument("--validation-config", type=Path, default=Path("validation_config.json"))
    parser.add_argument("--runner", type=Path, default=Path("run_experiment.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=DEFAULT_TASKS)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="跳过同一输出目录中已成功的模型任务。")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_runner(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"找不到基础模型运行器：{path}")
    spec = importlib.util.spec_from_file_location("candidate_base_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础模型运行器：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def task_parts(task: str) -> tuple[str, str]:
    target, depth = task.rsplit("_", maxsplit=1)
    return target, depth


def select_features(
    runner: ModuleType,
    data: pd.DataFrame,
    target_column: str,
    feature_set: str,
    strict_external: bool,
) -> list[str]:
    features = list(runner.build_feature_sets(data, target_column)[feature_set])
    if strict_external:
        features = [name for name in features if not name.startswith(f"{target_column}_")]
    if target_column in features:
        raise ValueError(f"目标变量错误进入输入特征：{target_column}")
    if not features:
        raise ValueError(f"没有可用特征：{target_column} / {feature_set}")
    return features


def fit_predict(
    runner: ModuleType,
    model_config: dict[str, Any],
    model: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    features: list[str],
    seed: int,
) -> np.ndarray:
    y_train, inverse_transform = runner.target_transform(train[target_column].to_numpy(dtype=float))
    prediction = runner.FIT_FUNCTIONS[model](train[features], y_train, test[features], seed, model_config)
    prediction = inverse_transform(np.asarray(prediction, dtype=float))
    return np.maximum(prediction, 0.0) if np.nanmin(train[target_column].to_numpy(dtype=float)) >= 0 else prediction


def make_splits(
    subset: pd.DataFrame,
    mode: str,
    seed_values: list[int],
    validation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if mode == "random_seeds":
        rows = []
        for seed in seed_values:
            train, test = train_test_split(
                subset,
                test_size=float(validation_config["random_test_size"]),
                random_state=seed,
                shuffle=True,
            )
            rows.append({"split_id": f"seed_{seed}", "fold": np.nan, "seed": seed, "train": train, "test": test})
        return rows

    if mode == "time_holdout":
        train = subset[subset["year"] <= int(validation_config["time_train_end_year"])].copy()
        test = subset[
            subset["year"].between(
                int(validation_config["time_test_start_year"]),
                int(validation_config["time_test_end_year"]),
            )
        ].copy()
        return [{"split_id": "2014_2022_to_2023_2024", "fold": np.nan, "seed": 42, "train": train, "test": test}]

    if mode == "station_5fold":
        splitter = GroupKFold(n_splits=int(validation_config["station_n_folds"]))
        rows = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(subset, groups=subset["station"]), start=1):
            rows.append(
                {
                    "split_id": f"station_fold_{fold}",
                    "fold": fold,
                    "seed": int(validation_config["station_random_state"]),
                    "train": subset.iloc[train_idx].copy(),
                    "test": subset.iloc[test_idx].copy(),
                }
            )
        return rows

    raise ValueError(f"未知验证模式：{mode}")


def metric_row(
    runner: ModuleType,
    mode: str,
    split_spec: dict[str, Any],
    target: str,
    depth: str,
    model: str,
    feature_set: str,
    features: list[str],
    strict_external: bool,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    prediction: np.ndarray | None,
    elapsed: float,
    exc: Exception | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "evaluation_mode": mode,
        "split_id": split_spec["split_id"],
        "fold": split_spec["fold"],
        "seed": split_spec["seed"],
        "target": target,
        "depth": depth,
        "target_column": target_column,
        "model": model,
        "feature_set": feature_set,
        "feature_count": len(features),
        "target_history_features": "excluded" if strict_external else "included",
        "same_day_water_quality_covariates": "excluded",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_stations": int(train["station"].nunique()),
        "test_stations": int(test["station"].nunique()),
        "train_year_min": int(train["year"].min()),
        "train_year_max": int(train["year"].max()),
        "test_year_min": int(test["year"].min()),
        "test_year_max": int(test["year"].max()),
        "runtime_seconds": elapsed,
    }
    if exc is not None:
        row.update({"status": "error", "error_type": type(exc).__name__, "error_message": str(exc)})
        return row
    row.update({"status": "ok", "error_type": "", "error_message": "", **runner.metrics(test[target_column].to_numpy(dtype=float), prediction)})
    return row


def prediction_frame(
    test: pd.DataFrame,
    mode: str,
    split_spec: dict[str, Any],
    target: str,
    depth: str,
    model: str,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    columns = ["obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "year", "month"]
    output = test[columns].copy()
    output.insert(0, "evaluation_mode", mode)
    output.insert(1, "split_id", split_spec["split_id"])
    output.insert(2, "fold", split_spec["fold"])
    output.insert(3, "seed", split_spec["seed"])
    output["target"] = target
    output["depth"] = depth
    output["model"] = model
    output["observed"] = observed
    output["predicted"] = predicted
    output["residual"] = predicted - observed
    return output


def write_outputs(
    output_dir: Path,
    metrics_rows: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    feature_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(metrics_rows).to_csv(output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_rows).drop_duplicates().to_csv(output_dir / "feature_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(split_rows).drop_duplicates().to_csv(output_dir / "split_manifest.csv", index=False, encoding="utf-8-sig")
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(output_dir / "model_predictions.csv", index=False, encoding="utf-8-sig")


def pooled_summary(runner: ModuleType, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["evaluation_mode", "target", "depth", "model"], dropna=False):
        observed = group["observed"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        rows.append(
            {
                "evaluation_mode": keys[0],
                "target": keys[1],
                "depth": keys[2],
                "model": keys[3],
                "n_predictions": len(group),
                "n_splits": int(group["split_id"].nunique()),
                **runner.metrics(observed, predicted),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty and "xgboost" in set(summary["model"]):
        baseline = summary[summary["model"].eq("xgboost")][["evaluation_mode", "target", "depth", "r2"]].rename(columns={"r2": "xgboost_r2"})
        summary = summary.merge(baseline, on=["evaluation_mode", "target", "depth"], how="left")
        summary["r2_change_vs_xgboost"] = summary["r2"] - summary["xgboost_r2"]
    return summary


def write_summary(output_dir: Path, pooled: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    pooled.to_csv(output_dir / "candidate_validation_summary.csv", index=False, encoding="utf-8-sig")
    lines = ["# 候选模型稳健性验证结果", ""]
    if pooled.empty:
        lines.extend(["没有成功完成的模型任务。请查看 `model_errors.log`。", ""])
    else:
        lines.extend(
            [
                "## 汇总指标",
                "",
                "| 验证方式 | 指标 | 水层 | 模型 | R2 | 相对 XGBoost | 样本数 |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for row in pooled.sort_values(["evaluation_mode", "target", "depth", "model"]).itertuples():
            delta = getattr(row, "r2_change_vs_xgboost", np.nan)
            delta_text = "" if pd.isna(delta) else f"{delta:+.4f}"
            lines.append(f"| {row.evaluation_mode} | {row.target} | {row.depth} | {row.model} | {row.r2:.4f} | {delta_text} | {row.n_predictions} |")
        lines.extend(
            [
                "",
                "## 解释边界",
                "",
                "- `random_seeds` 保留目标历史特征，用于检验随机留出结果对种子的稳定性。",
                "- `time_holdout` 与 `station_5fold` 均移除目标自身历史特征，用于更严格的跨年份和未见站点泛化检验。",
                "- 所有模型均在同一任务、同一划分、同一特征集内比较；同期其他水质协变量始终排除。",
                "",
            ]
        )
    failed = metrics_df[metrics_df["status"].ne("ok")]
    if not failed.empty:
        lines.extend(["## 未完成任务", "", "| 验证方式 | 指标 | 水层 | 模型 | 错误 |", "|---|---|---|---|---|"])
        for row in failed.itertuples():
            message = str(row.error_message).replace("|", "/").replace("\n", " ")[:180]
            lines.append(f"| {row.evaluation_mode} | {row.target} | {row.depth} | {row.model} | {message} |")
        lines.append("")
    (output_dir / "结果摘要.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = load_runner(args.runner)
    model_config = runner.load_config(args.model_config)
    validation_config = load_json(args.validation_config)
    data = runner.prepare_dataset(args.dataset)
    seeds = args.random_seeds or [int(value) for value in validation_config["random_seeds"]]

    environment = runner.environment_report()
    environment["validation_modes"] = args.modes
    environment["candidate_tasks"] = args.tasks
    (args.output_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")

    old_metrics: list[dict[str, Any]] = []
    old_predictions: list[pd.DataFrame] = []
    metric_path = args.output_dir / "model_metrics.csv"
    prediction_path = args.output_dir / "model_predictions.csv"
    if args.resume and metric_path.exists():
        old_metrics = pd.read_csv(metric_path).to_dict("records")
        if prediction_path.exists():
            old_predictions = [pd.read_csv(prediction_path, parse_dates=["date"])]
    completed = {
        (str(row.get("evaluation_mode")), str(row.get("split_id")), str(row.get("target")), str(row.get("depth")), str(row.get("model")))
        for row in old_metrics
        if str(row.get("status")) == "ok"
    }

    metrics_rows = old_metrics.copy()
    predictions = old_predictions.copy()
    feature_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    errors_path = args.output_dir / "model_errors.log"
    if errors_path.exists() and not args.resume:
        errors_path = args.output_dir / "model_errors_current_run.log"

    jobs: list[dict[str, Any]] = []
    for task in args.tasks:
        target, depth = task_parts(task)
        target_column = runner.TARGET_MAP[target]
        subset = data[(data["depth"].eq(depth)) & data[target_column].notna()].copy()
        subset = subset.sort_values(["date", "station"]).reset_index(drop=True)
        feature_set = validation_config["task_feature_sets"][task]
        for mode in args.modes:
            strict_external = mode in {"time_holdout", "station_5fold"}
            features = select_features(runner, data, target_column, feature_set, strict_external)
            split_specs = make_splits(subset, mode, seeds, validation_config)
            for split_spec in split_specs:
                if split_spec["train"].empty or split_spec["test"].empty:
                    raise ValueError(f"空训练/测试集：{task} / {mode} / {split_spec['split_id']}")
                jobs.append(
                    {
                        "task": task,
                        "target": target,
                        "depth": depth,
                        "target_column": target_column,
                        "feature_set": feature_set,
                        "features": features,
                        "strict_external": strict_external,
                        "mode": mode,
                        "split_spec": split_spec,
                    }
                )

    total_jobs = len(jobs) * len(args.models)
    completed_jobs = 0
    checkpoint_interval = 3
    for job in jobs:
        split_spec = job["split_spec"]
        train = split_spec["train"].sort_values(["date", "station"]).reset_index(drop=True)
        test = split_spec["test"].sort_values(["date", "station"]).reset_index(drop=True)
        feature_rows.append(
            {
                "evaluation_mode": job["mode"],
                "target": job["target"],
                "depth": job["depth"],
                "feature_set": job["feature_set"],
                "target_history_features": "excluded" if job["strict_external"] else "included",
                "features_json": json.dumps(job["features"], ensure_ascii=False),
            }
        )
        for frame, split_name in [(train, "train"), (test, "test")]:
            split_rows.extend(
                {
                    "evaluation_mode": job["mode"],
                    "split_id": split_spec["split_id"],
                    "fold": split_spec["fold"],
                    "seed": split_spec["seed"],
                    "target": job["target"],
                    "depth": job["depth"],
                    "split": split_name,
                    "obs_id": row.obs_id,
                    "sample_id": row.sample_id,
                    "station": row.station,
                    "date": row.date,
                }
                for row in frame[["obs_id", "sample_id", "station", "date"]].itertuples(index=False)
            )
        for model in args.models:
            completed_jobs += 1
            key = (job["mode"], split_spec["split_id"], job["target"], job["depth"], model)
            if key in completed:
                print(f"[{completed_jobs}/{total_jobs}] 跳过已完成：{key}", flush=True)
                continue
            print(f"[{completed_jobs}/{total_jobs}] 开始：{job['mode']} / {job['task']} / {split_spec['split_id']} / {model}", flush=True)
            started = time.perf_counter()
            try:
                prediction = fit_predict(
                    runner, model_config, model, train, test, job["target_column"], job["features"], int(split_spec["seed"])
                )
                elapsed = time.perf_counter() - started
                metrics_rows.append(
                    metric_row(
                        runner, job["mode"], split_spec, job["target"], job["depth"], model, job["feature_set"],
                        job["features"], job["strict_external"], train, test, job["target_column"], prediction, elapsed, None,
                    )
                )
                predictions.append(
                    prediction_frame(
                        test, job["mode"], split_spec, job["target"], job["depth"], model,
                        test[job["target_column"]].to_numpy(dtype=float), prediction,
                    )
                )
                print(f"    完成：R2={metrics_rows[-1]['r2']:.4f}，耗时={elapsed:.1f}s", flush=True)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                metrics_rows.append(
                    metric_row(
                        runner, job["mode"], split_spec, job["target"], job["depth"], model, job["feature_set"],
                        job["features"], job["strict_external"], train, test, job["target_column"], None, elapsed, exc,
                    )
                )
                with errors_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n{'=' * 80}\n{key}\n{traceback.format_exc()}")
                print(f"    失败：{type(exc).__name__}: {exc}", flush=True)
            if completed_jobs % checkpoint_interval == 0:
                write_outputs(args.output_dir, metrics_rows, predictions, feature_rows, split_rows)

    metrics_df = pd.DataFrame(metrics_rows)
    all_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    pooled = pooled_summary(runner, all_predictions) if not all_predictions.empty else pd.DataFrame()
    write_summary(args.output_dir, pooled, metrics_df)
    run_status = {
        "finished": True,
        "successful_jobs": int((metrics_df["status"] == "ok").sum()),
        "failed_jobs": int((metrics_df["status"] != "ok").sum()),
        "expected_jobs": total_jobs,
        "modes": args.modes,
        "tasks": args.tasks,
    }
    (args.output_dir / "run_status.json").write_text(json.dumps(run_status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("候选模型稳健性验证完成。", flush=True)


if __name__ == "__main__":
    main()
