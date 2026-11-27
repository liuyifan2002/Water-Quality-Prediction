"""按预登记样本执行完整77维P1 Permutation SHAP，逐样本落盘。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v1"
MODULE_PATH = ROOT / "脚本/05_诊断分析/88_run_tabicl_slim_explainability_v1.py"
OLD_RUNNER = ROOT / "脚本/05_诊断分析/79_run_all_tabicl_explanation_experiments.py"


def main(task: str, phase: str) -> None:
    import shap

    spec = __import__("importlib.util", fromlist=["spec_from_file_location"])
    run_spec = spec.spec_from_file_location("tabicl_slim_runner", MODULE_PATH)
    run = spec.module_from_spec(run_spec)
    sys.modules[run_spec.name] = run
    run_spec.loader.exec_module(run)
    old_spec = spec.spec_from_file_location("tabicl_original_pointwise_runner", OLD_RUNNER)
    old = spec.module_from_spec(old_spec)
    sys.modules[old_spec.name] = old
    old_spec.loader.exec_module(old)
    gate = OUT / "gates" / f"{task}_p1_decision.csv"
    if not gate.is_file() or pd.read_csv(gate).status.iloc[0] != "P1_GATE_PASS":
        raise RuntimeError(f"{task}: P1资格门槛未通过，不允许计算正式SHAP")
    base, features, train, test, x_train, x_test, adapter, inv, metadata = run.context(task)
    train_seed_index = {"chla_surface": 0, "secchi": 1, "tn_surface": 2, "tp_middle": 3}
    if task in train_seed_index:
        task_index = train_seed_index[task]
    else:
        task_index = 4 + [t for t in run.TASKS if t not in train_seed_index].index(task)
    positions = old.representatives(train, x_train, min(16, len(train)), 9001 + 100 * task_index)
    background = x_train[positions]
    background_ids = train.iloc[positions].obs_id.astype(str).tolist()
    metadata.update({"background_hash": run.hash_array(background),
        "background_obs_order_hash": run.hash_ids(background_ids),
        "background_n": len(background), "shap_seed": 42,
        "shap_max_evals": 155, "output_scale": "original_target",
        "shap_script_sha256": run.file_sha(Path(__file__))})
    bg_file = OUT / "shap" / task / "background_manifest.csv"
    if bg_file.exists():
        old_bg = pd.read_csv(bg_file)
        if old_bg.obs_id.astype(str).tolist() != background_ids:
            raise RuntimeError("背景样本清单与此前同一任务计算不一致")
    else:
        run.atomic_csv(bg_file, pd.DataFrame({"task": task, "fold": "fold_1",
            "background_order": np.arange(1, len(background_ids) + 1),
            "obs_id": background_ids, "background_hash": metadata["background_hash"]}))
    query_ids = {str(obs): i for i, obs in enumerate(test.obs_id.astype(str))}
    selected = run.sample_rows(task, "CORE_3" if phase == "SENSITIVITY_1" else phase)
    if phase == "SENSITIVITY_1":
        selected = selected.loc[selected.target_quantile.eq(0.5)].copy()
    shap_seed = 43 if phase == "SENSITIVITY_1" else 42
    calls = {"actual": 0, "seconds": 0.0, "cache_hits": 0}
    cache: dict[str, float] = {}

    def p1_predict(matrix):
        x = np.asarray(matrix, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        output = []
        for row in x:
            key = run.hash_array(row)
            if key in cache:
                calls["cache_hits"] += 1
                output.append(cache[key])
                continue
            started = time.perf_counter()
            raw = np.asarray(adapter.predict(row[None, :]), float).reshape(-1)
            calls["seconds"] += time.perf_counter() - started
            calls["actual"] += 1
            if raw.size != 1 or not np.isfinite(raw[0]):
                raise RuntimeError("P1预测非有限值或输出不是单个标量")
            value = float(max(float(inv(raw)[0]), 0.0))
            cache[key] = value
            output.append(value)
        return np.asarray(output, dtype=float)

    masker = shap.maskers.Independent(background, max_samples=16)
    output_dir = OUT / "shap" / task
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in selected.itertuples():
        path = output_dir / f"{sample.obs_id}_seed{shap_seed}.csv.gz"
        if path.exists():
            previous = pd.read_csv(path)
            if len(previous) == 77 and previous.feature.astype(str).tolist() == features \
                    and previous.protocol.iloc[0] == run.P1 \
                    and previous.context_hash.iloc[0] == metadata["context_hash"] \
                    and previous.background_hash.iloc[0] == metadata["background_hash"] \
                    and float(previous.additivity_abs_error.iloc[0]) <= run.ADD_TOL:
                print(json.dumps({"task": task, "obs_id": str(sample.obs_id),
                    "status": "REUSED_VERIFIED_CHECKPOINT"}, ensure_ascii=False), flush=True)
                continue
            raise RuntimeError(f"已有SHAP结果资格不一致，禁止覆盖: {path}")
        row = x_test[query_ids[str(sample.obs_id)]:query_ids[str(sample.obs_id)] + 1]
        if row.shape != (1, 77) or not np.isfinite(row).all():
            raise RuntimeError(f"{sample.obs_id}: SHAP查询输入不合格")
        started_calls = calls["actual"]
        started_seconds = calls["seconds"]
        started_cache_hits = calls["cache_hits"]
        started = time.perf_counter()
        explainer = shap.PermutationExplainer(p1_predict, masker, feature_names=features, seed=shap_seed)
        explanation = explainer(row, max_evals=155, silent=True)
        duration = time.perf_counter() - started
        shap_values = np.asarray(explanation.values, float).reshape(-1)
        base_value = float(np.asarray(explanation.base_values, float).reshape(-1)[0])
        prediction = float(p1_predict(row)[0])
        add_error = abs(base_value + shap_values.sum() - prediction)
        status = "COMPUTED_PASS" if len(shap_values) == 77 and add_error <= run.ADD_TOL else "ADDITIVITY_FAILED"
        result = pd.DataFrame({**metadata, "obs_id": str(sample.obs_id),
            "station_id": str(sample.station_id), "date": str(sample.date),
            "selection_quantile": float(sample.target_quantile), "selection_phase": phase,
            "selection_uses_test_label": False, "shap_seed": shap_seed,
            "feature_order": np.arange(1, 78),
            "feature": features, "feature_value": row[0], "shap_value": shap_values,
            "base_value": base_value, "model_prediction": prediction,
            "additivity_abs_error": add_error, "query_hash": run.hash_array(row),
            "actual_model_calls": calls["actual"] - started_calls,
            "exact_input_cache_hits": calls["cache_hits"] - started_cache_hits,
            "model_seconds": calls["seconds"] - started_seconds,
            "elapsed_seconds": duration, "status": status})
        tmp = path.with_suffix(path.suffix + ".tmp")
        result.to_csv(tmp, index=False, encoding="utf-8-sig", compression="gzip")
        tmp.replace(path)
        print(json.dumps({"task": task, "obs_id": str(sample.obs_id),
            "status": status, "actual_model_calls": calls["actual"] - started_calls,
            "elapsed_seconds": duration, "additivity_abs_error": add_error},
            ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--phase", choices=("CORE_3", "OPTIONAL_2", "SENSITIVITY_1"), default="CORE_3")
    arguments = parser.parse_args()
    main(arguments.task, arguments.phase)
