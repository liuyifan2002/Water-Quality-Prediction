"""Reproduce selected XGBoost runs using the saved split manifest exactly."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "结果" / "扩展实验" / "候选模型稳健性验证_20260914" / "portable_package"
SAVED = ROOT / "结果" / "扩展实验" / "候选模型稳健性验证_20260914" / "回传结果_20260914"
DEFAULT_OUTPUT = ROOT / "结果" / "扩展实验" / "论文主线审计_20260916" / "复现检查_XGBoost_保存划分"
TASKS = ["chla_surface", "secchi_surface", "tn_bottom"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use saved split manifests for XGBoost reproducibility checks.")
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--saved-root", type=Path, default=SAVED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_runner_for_repro", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_mode_folder(mode: str) -> str:
    return f"outputs_{mode}"


def task_parts(task: str) -> tuple[str, str]:
    return task.rsplit("_", maxsplit=1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = load_module(args.package / "run_experiment.py")
    config = runner.load_config(args.package / "config.json")
    validation = load_json(args.package / "validation_config.json")
    data = runner.prepare_dataset(args.package / "data" / "model_table.csv")
    data_by_obs = data.set_index("obs_id", drop=False)
    rows: list[dict[str, object]] = []
    manifest_checks: list[dict[str, object]] = []

    for mode in ["random_seeds", "time_holdout", "station_5fold"]:
        folder = args.saved_root / split_mode_folder(mode)
        saved_metrics_path = folder / "model_metrics.csv"
        saved_split_path = folder / "split_manifest.csv"
        if not saved_metrics_path.exists() or not saved_split_path.exists():
            continue
        saved_metrics = pd.read_csv(saved_metrics_path)
        split_manifest = pd.read_csv(saved_split_path)
        for task in args.tasks:
            target, depth = task_parts(task)
            target_column = runner.TARGET_MAP[target]
            feature_set = validation["task_feature_sets"][task]
            strict_external = mode != "random_seeds"
            features = list(runner.build_feature_sets(data, target_column)[feature_set])
            if strict_external:
                features = [name for name in features if not name.startswith(f"{target_column}_")]

            saved_task = saved_metrics.loc[
                saved_metrics["model"].eq("xgboost")
                & saved_metrics["target"].eq(target)
                & saved_metrics["depth"].eq(depth)
                & saved_metrics["status"].eq("ok")
            ]
            for saved in saved_task.itertuples(index=False):
                split = split_manifest.loc[
                    split_manifest["target"].eq(target)
                    & split_manifest["depth"].eq(depth)
                    & split_manifest["split_id"].eq(saved.split_id)
                ]
                train_ids = split.loc[split["split"].eq("train"), "obs_id"].tolist()
                test_ids = split.loc[split["split"].eq("test"), "obs_id"].tolist()
                missing_train = sorted(set(train_ids) - set(data_by_obs.index))
                missing_test = sorted(set(test_ids) - set(data_by_obs.index))
                if missing_train or missing_test:
                    raise ValueError(f"Saved split contains unknown obs_id for {task}/{saved.split_id}")
                train = data_by_obs.loc[train_ids].copy()
                test = data_by_obs.loc[test_ids].copy()
                y_fit, inverse = runner.target_transform(train[target_column].to_numpy(dtype=float))
                prediction = runner.FIT_FUNCTIONS["xgboost"](
                    train[features], y_fit, test[features], int(saved.seed), config
                )
                prediction = inverse(np.asarray(prediction, dtype=float))
                prediction = np.maximum(prediction, 0.0)
                metrics = runner.metrics(test[target_column].to_numpy(dtype=float), prediction)
                delta = float(metrics["r2"] - saved.r2)
                rows.append(
                    {
                        "evaluation_mode": mode,
                        "split_id": saved.split_id,
                        "target": target,
                        "depth": depth,
                        "model": "xgboost",
                        "seed": int(saved.seed),
                        "feature_set": feature_set,
                        "target_history_features": "excluded" if strict_external else "included",
                        "feature_count": len(features),
                        "saved_r2": saved.r2,
                        "reproduced_r2": metrics["r2"],
                        "abs_delta_r2": abs(delta),
                        "saved_rmse": saved.rmse,
                        "reproduced_rmse": metrics["rmse"],
                        "saved_mae": saved.mae,
                        "reproduced_mae": metrics["mae"],
                        "train_rows": len(train),
                        "test_rows": len(test),
                    }
                )
                manifest_checks.append(
                    {
                        "evaluation_mode": mode,
                        "split_id": saved.split_id,
                        "target": target,
                        "depth": depth,
                        "saved_train_rows": len(train_ids),
                        "loaded_train_rows": len(train),
                        "saved_test_rows": len(test_ids),
                        "loaded_test_rows": len(test),
                        "missing_train_obs": len(missing_train),
                        "missing_test_obs": len(missing_test),
                    }
                )

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics["status"] = pd.cut(
            metrics["abs_delta_r2"],
            bins=[-np.inf, 0.01, 0.03, 0.05, np.inf],
            labels=["通过", "可接受：环境版本差异", "需解释", "失败"],
        ).astype(str)
    metrics.to_csv(args.output_dir / "保存划分复现结果.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_checks).to_csv(args.output_dir / "保存划分完整性检查.csv", index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
