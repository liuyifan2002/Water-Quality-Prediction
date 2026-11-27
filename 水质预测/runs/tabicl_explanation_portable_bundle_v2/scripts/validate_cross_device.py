"""RTX4060首次P1单查询兼容性门禁；只运行5个预注册黄金query。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from portable_common import BUNDLE_ROOT, load_config, resolve, sha256, verify_static_manifest

OBSERVED_JITTER = 0.0014983664


def hash_array(x) -> str:
    a = np.ascontiguousarray(np.asarray(x))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def import_runner():
    p = resolve("workspace/脚本/05_诊断分析/79_run_all_tabicl_explanation_experiments.py")
    spec = importlib.util.spec_from_file_location("portable_tabicl_runner", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    cfg = load_config()
    cpu_n = os.cpu_count() or 4
    cpu_threads = max(1, min(8, cpu_n // 4 or 1))
    os.environ.update({"OMP_NUM_THREADS":str(cpu_threads),"MKL_NUM_THREADS":str(cpu_threads),
                       "OPENBLAS_NUM_THREADS":str(cpu_threads),"RAM_CACHE_MODE":"ON"})
    errors = verify_static_manifest()
    manifest_path = resolve("inputs/golden_query_manifest.csv")
    array_path = resolve("inputs/golden_queries.npz")
    m = pd.read_csv(manifest_path)
    if len(m) < 5:
        errors.append("黄金query不足5条")
    try:
        npz = np.load(array_path, allow_pickle=False)
    except Exception as exc:
        errors.append(f"黄金query数组无法读取：{exc!r}")
        npz = None
    if errors:
        return write_decision("CROSS_DEVICE_DIFFERENCE_REQUIRES_REVIEW", errors, [], None)

    runner = import_runner()
    checkpoint_path = resolve("model/tabicl-regressor-v2-20260212.ckpt")
    os.environ["TABICL_PRETRAINED_CHECKPOINT"] = str(checkpoint_path)
    rows = []
    contexts = {}
    try:
        for r in m.itertuples(index=False):
            key = str(r.probe_id)
            task_fold = (str(r.task), str(r.fold))
            if task_fold not in contexts:
                ctx = runner.make_context(*task_fold)
                base, torch, features, train, test, xtr, xte, imp, adapter, est, pred, env, calls = ctx
                pred.cache_enabled = False
                checks = {
                    "feature_order_hash": str(env["feature_order_hash"]) == str(r.feature_order_hash),
                    "checkpoint_sha256": str(env["checkpoint_sha256"]) == str(r.checkpoint_sha256),
                    "context_hash": str(env["context_hash"]) == str(r.train_context_hash),
                    "imputer_stats_hash": str(env["imputer_stats_hash"]) == str(r.imputer_stats_hash),
                    "model_state_hash": str(env["model_state_hash"]) == str(r.model_state_hash),
                    "train_obs_order_hash": str(env["train_obs_order_hash"]) == str(r.train_obs_order_hash),
                    "test_obs_order_hash": str(env["test_obs_order_hash"]) == str(r.test_obs_order_hash),
                    "protocol": str(r.query_shape) == "1x77" and str(r.query_dtype) == "float32",
                }
                if not all(checks.values()):
                    raise RuntimeError(f"{task_fold}: immutable split/context/checkpoint/function hash check failed: {checks}")
                contexts[task_fold] = (pred, env, checks, torch, est)
            pred, env, checks, torch, est = contexts[task_fold]
            x = np.asarray(npz[key], dtype=np.float32)
            observed_hash = hash_array(x)
            q = x.reshape(1, -1)
            if q.shape != (1, 77):
                raise RuntimeError(f"{key}: P1输入形状错误 {q.shape}")
            if observed_hash != str(r.query_input_sha256):
                raise RuntimeError(f"{key}: frozen golden query hash mismatch; prediction not called")
            calls_before = callsafe(pred)
            t0 = time.perf_counter()
            yhat = float(pred(q)[0])
            elapsed = time.perf_counter() - t0
            ref = float(r.frozen_prediction)
            delta = abs(yhat - ref)
            relative = delta / max(abs(ref), np.finfo(float).eps)
            rmse = float(r.official_pooled_rmse)
            iqr = float(r.target_iqr)
            ale_range = float(r.relevant_ale_curve_range) if np.isfinite(float(r.relevant_ale_curve_range)) else np.nan
            row_ok = all(checks.values()) and observed_hash == str(r.query_input_sha256)
            rows.append({"probe_id":key,"task":r.task,"fold":r.fold,"obs_id":r.obs_id,
                "protocol":"P1_SINGLE_QUERY","query_shape":str(q.shape),"query_dtype":str(q.dtype),
                "query_hash_expected":str(r.query_input_sha256),"query_hash_actual":observed_hash,
                "feature_order_hash_actual":env["feature_order_hash"],"split_train_hash_actual":env["train_obs_order_hash"],
                "split_test_hash_actual":env["test_obs_order_hash"],"checkpoint_sha256_actual":env["checkpoint_sha256"],
                "train_context_hash_actual":env["context_hash"],"imputer_stats_hash_actual":env["imputer_stats_hash"],
                "model_state_hash_actual":env["model_state_hash"],"reference_prediction_rtx2070":ref,
                "destination_prediction":yhat,"absolute_delta":delta,"relative_delta":relative,
                "delta_over_formal_rmse":delta/rmse if rmse else np.nan,"delta_over_target_iqr":delta/iqr if iqr else np.nan,
                "relevant_ale_curve_range":ale_range,"delta_over_ale_range":delta/ale_range if np.isfinite(ale_range) and ale_range else np.nan,
                "observed_jitter_reference":OBSERVED_JITTER,"delta_within_observed_jitter":delta <= OBSERVED_JITTER,
                "hash_checks_pass":row_ok,"elapsed_seconds":elapsed,"actual_model_calls":int(callsafe(pred)-calls_before),
                "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA",
                "prediction_function":"f_fold(X_query | X_train, y_train, pretrained_model, inference_config); P1 single query"})
    except Exception as exc:
        errors.append(f"预注册跨设备探针运行失败：{exc!r}")
    if len(rows) != len(m):
        errors.append(f"实际探针条数{len(rows)}与manifest条数{len(m)}不匹配")
    if rows:
        all_hash = all(r["hash_checks_pass"] for r in rows)
        within = all(r["delta_within_observed_jitter"] for r in rows)
        small_vs_task = all(r["delta_over_formal_rmse"] < 1 and r["delta_over_target_iqr"] < 1 for r in rows)
        has_ale = [r for r in rows if np.isfinite(r["relevant_ale_curve_range"]) and r["relevant_ale_curve_range"] > 0]
        small_vs_ale = all(r["delta_over_ale_range"] < 1 for r in has_ale)
        compatible = all_hash and within and small_vs_task and small_vs_ale and not errors
    else:
        compatible = False
    status = "CROSS_DEVICE_NUMERICALLY_COMPATIBLE" if compatible else "CROSS_DEVICE_DIFFERENCE_REQUIRES_REVIEW"
    return write_decision(status, errors, rows, len(contexts))


def callsafe(pred) -> int:
    # Predictor maintains true uncached-call count; no synthetic count is inserted.
    return int(getattr(pred, "calls", {}).get("n", 0))


def write_decision(status: str, errors: list[str], rows: list[dict], contexts: int | None) -> int:
    out_csv = BUNDLE_ROOT / "logs/cross_device_query_comparison.csv"
    out_json = BUNDLE_ROOT / "logs/cross_device_validation.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    report = {"status":status,"errors":errors,"probe_n":len(rows),"task_fold_contexts":contexts,
        "observed_jitter_reference_abs":OBSERVED_JITTER,"observed_jitter_is_theoretical_bound":False,
        "comparison_criteria":"exact immutable hashes and P1 protocol; per-query delta <= observed historical jitter and < task RMSE/IQR and supported relevant ALE range",
        "bitwise_equivalent_claimed":False,"golden_reference_overwritten":False,
        "prediction_calls_are_single_query_and_uncached":True,"cpu_threads":cpu_threads}
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "CROSS_DEVICE_NUMERICALLY_COMPATIBLE" else 3


if __name__ == "__main__":
    raise SystemExit(main())

