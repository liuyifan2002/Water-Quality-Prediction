"""Run TP_FAMILY_V2 using the frozen TP manifests and fixed model settings."""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tp_family_v2"
BASE_PATH = ROOT / "脚本" / "04_模型训练" / "36_run_tn_family_v2_models.py"
PROTOCOL = "TP_FAMILY_V2_UNIFIED_STATION_SPLIT"
FEATURE_SET = "X_strict_77"
DEPTHS = ("surface", "middle", "bottom")
MODELS = ("xgboost", "tabiclv2")
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))


def load_base():
    spec = importlib.util.spec_from_file_location("tp_family_v2_model_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import shared model runner: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()
BASE.RUN = RUN
BASE.PREDICTION_DIR = RUN / "predictions"
BASE.METRICS_DIR = RUN / "fold_metrics"
BASE.MAIN_TABLE = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
BASE.STRICT_MANIFEST = ROOT / "runs" / "tabiclv2_pilot" / "01_strict_feature_manifest.csv"
BASE.CANONICAL_MAP = RUN / "02_tp_family_v2_station_fold_map.csv"
BASE.SPLIT_VALIDATION = RUN / "06_split_validation.csv"
BASE.REGISTRY = RUN / "07_run_registry.csv"
BASE.TARGET_COLUMN = "tp_mg_l"
BASE.SPLIT_VERSION = PROTOCOL
BASE.FEATURE_SET = FEATURE_SET
BASE.DEPTHS = DEPTHS
BASE.MODELS = MODELS
BASE.FOLDS = FOLDS


def tp_manifest_path(depth: str) -> Path:
    return RUN / {"surface": "03_tp_surface_manifest.csv", "middle": "04_tp_middle_manifest.csv", "bottom": "05_tp_bottom_manifest.csv"}[depth]


BASE.manifest_path = tp_manifest_path


def build_registry() -> pd.DataFrame:
    rows = []
    for depth in DEPTHS:
        for model in MODELS:
            for fold in FOLDS:
                task = f"tp_{depth}"
                rows.append(
                    {
                        "task": task,
                        "depth": depth,
                        "model": model,
                        "fold": fold,
                        "status": "PENDING",
                        "start_time": "",
                        "end_time": "",
                        "runtime": np.nan,
                        "prediction_file": str(BASE.prediction_file(task, model, fold)),
                        "metrics_file": str(BASE.metrics_file(task, model, fold)),
                        "R2": np.nan,
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "n_train": np.nan,
                        "n_test": np.nan,
                        "n_features": 77,
                        "feature_set": FEATURE_SET,
                        "split_version": PROTOCOL,
                        "error": "",
                    }
                )
    return pd.DataFrame(rows, columns=BASE.registry_columns())


BASE.build_registry = build_registry


def run_models(features: list[str], data: pd.DataFrame) -> None:
    registry = BASE.load_or_create_registry()
    manifests = {depth: BASE.load_manifest(depth) for depth in DEPTHS}
    for depth in DEPTHS:
        task = f"tp_{depth}"
        task_data = data.loc[data["depth"].eq(depth) & data[BASE.TARGET_COLUMN].notna()].copy()
        for model_name in MODELS:
            for fold in FOLDS:
                row = registry.loc[registry["task"].eq(task) & registry["model"].eq(model_name) & registry["fold"].eq(fold)].iloc[0]
                if str(row["status"]) == "OK" and BASE.artifacts_valid(row):
                    print(f"SKIP {task} {model_name} {fold}: already OK", flush=True)
                    continue
                started = time.perf_counter()
                BASE.set_registry(registry, task, model_name, fold, status="RUNNING", start_time=BASE.now(), end_time="", error="")
                print(f"START {task} {model_name} {fold}", flush=True)
                try:
                    result = BASE.run_one(task, depth, model_name, fold, task_data, manifests[depth], features)
                    BASE.set_registry(
                        registry,
                        task,
                        model_name,
                        fold,
                        status="OK",
                        end_time=BASE.now(),
                        runtime=result["runtime"],
                        prediction_file=result["prediction_file"],
                        metrics_file=result["metrics_file"],
                        R2=result["R2"],
                        RMSE=result["RMSE"],
                        MAE=result["MAE"],
                        n_train=result["n_train"],
                        n_test=result["n_test"],
                        error="",
                    )
                    print(f"OK {task} {model_name} {fold} R2={result['R2']:.4f} runtime={result['runtime']:.1f}s", flush=True)
                except Exception as exc:
                    BASE.set_registry(registry, task, model_name, fold, status="FAILED", end_time=BASE.now(), runtime=time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}")
                    print(f"FAILED {task} {model_name} {fold}: {type(exc).__name__}: {exc}", flush=True)
                    raise
                finally:
                    BASE.clear_cuda()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    features, data, _, _ = BASE.run_preflight()
    if args.preflight_only or not args.run:
        return 0
    run_models(features, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
