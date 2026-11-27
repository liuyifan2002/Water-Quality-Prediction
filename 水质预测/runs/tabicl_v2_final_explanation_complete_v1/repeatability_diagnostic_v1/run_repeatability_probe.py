"""Up to three uncached, single-query diagnostics for the recorded ALE lower point."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/tabicl_v2_final_explanation_complete_v1"
RUNNER = ROOT / "脚本/05_诊断分析/79_run_all_tabicl_explanation_experiments.py"
TASK, FOLD = "chla_surface", "fold_1"
OBS_ID, FEATURE = "NM5_20141217_surface", "ssrd_mean_14d"
LOWER = 293192.125
EXPECTED_CONTEXT = "bf980bcea72fe62c2e56b0f5a37e0308d7959ee32389b820a9825f139e39a662:af72699a0370b2ed52cb32c962ffe00f272a49df6b5781fd3a433c0e8c964aeb"
EXPECTED_CKPT = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
EXPECTED_FEATURES = "da3dc69e30b052cce1fd5cbb30351c1f5abebb82dd8162db616ae8883ada3e99"
EXPECTED_IMPUTER = "45f57668e93fac9db7a1bf9f4532a518437f996d39a17359c7c70e2423f6dc0f"
EXPECTED_STATE = "577bf26e978278d794ee406c401686c755d828c4260bb60a3abd76c8fb25b858"
THRESHOLD = 1e-6


def stable_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def numpy_hash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return stable_hash(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def rng_hash(obj) -> str:
    return stable_hash(repr(obj).encode("utf-8", errors="replace"))


def write_row(path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Match the already-recorded inference environment; no prediction setting is changed.
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("Launch this diagnostic with PYTHONHASHSEED=42 set before Python starts.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TABICL_FIXED_NORM_ORDER"] = "none,power"
    patch = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
    os.environ["PYTHONPATH"] = str(patch) + os.pathsep + os.environ.get("PYTHONPATH", "")

    spec = importlib.util.spec_from_file_location("tabicl_diag_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    base, torch, features, train, test, xtr, xte, imp, adapter, est, pred, env, calls = mod.make_context(TASK, FOLD)

    expected = {
        "context_hash": EXPECTED_CONTEXT,
        "checkpoint_sha256": EXPECTED_CKPT,
        "feature_order_hash": EXPECTED_FEATURES,
        "imputer_stats_hash": EXPECTED_IMPUTER,
        "model_state_hash": EXPECTED_STATE,
    }
    mismatches = {k: {"expected": v, "actual": env.get(k)} for k, v in expected.items() if env.get(k) != v}
    if mismatches:
        (OUT / "probe_precondition_failure.json").write_text(json.dumps({"mismatches": mismatches, "env": env}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("Reconstructed model/context failed hash preconditions; zero probe predictions were run.")

    feature_index = features.index(FEATURE)
    matches = np.flatnonzero(test.obs_id.astype(str).to_numpy() == OBS_ID)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one fixed query row for {OBS_ID}; found {len(matches)}")
    query = xte[int(matches[0])].copy()
    query[feature_index] = LOWER
    query = np.asarray(query, dtype=np.float32)
    pred.cache_enabled = False
    results_path = OUT / "diagnostic_probe_results.csv"
    before_calls = int(calls["n"])

    for i in range(1, 4):
        py_rng_before = rng_hash(random.getstate())
        np_rng_before = rng_hash(np.random.get_state())
        torch_cpu_before = numpy_hash(torch.get_rng_state().cpu().numpy())
        torch_cuda_before = numpy_hash(torch.cuda.get_rng_state_all()[0].cpu().numpy()) if torch.cuda.is_available() else "NA"
        qhash_before = numpy_hash(query)
        tic = time.perf_counter()
        error = ""
        try:
            prediction = float(pred(query[None, :])[0])
        except Exception as exc:  # retain exact error and stop, never exceed three calls
            prediction = float("nan")
            error = repr(exc)
        elapsed = time.perf_counter() - tic
        row = {
            "probe_index": i, "task": TASK, "fold": FOLD, "obs_id": OBS_ID,
            "feature_modified": FEATURE, "query_shape": "1x77", "query_dtype": str(query.dtype),
            "query_input_sha256": qhash_before, "query_hash_after_call": numpy_hash(query),
            "feature_order_hash": env["feature_order_hash"], "imputer_stats_hash": env["imputer_stats_hash"],
            "checkpoint_sha256": env["checkpoint_sha256"], "train_context_hash": env["context_hash"],
            "model_state_hash": env["model_state_hash"], "prediction_original_scale": prediction,
            "unit": "Chl-a concentration scale as emitted by frozen P1 predictor",
            "elapsed_seconds": elapsed, "actual_model_calls_this_probe": int(calls["n"]) - before_calls,
            "python_rng_before": py_rng_before, "numpy_rng_before": np_rng_before,
            "torch_cpu_rng_before": torch_cpu_before, "torch_cuda_rng_before": torch_cuda_before,
            "torch_cpu_rng_after": numpy_hash(torch.get_rng_state().cpu().numpy()),
            "torch_cuda_rng_after": numpy_hash(torch.cuda.get_rng_state_all()[0].cpu().numpy()) if torch.cuda.is_available() else "NA",
            "pid": os.getpid(), "python": platform.python_version(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "protocol": mod.POINTWISE_PROTOCOL, "cache_enabled": pred.cache_enabled, "error": error,
        }
        write_row(results_path, row)
        before_calls = int(calls["n"])
        print(json.dumps({"probe_index": i, "prediction": prediction, "elapsed_seconds": elapsed, "error": error}, ensure_ascii=False), flush=True)
        if error:
            break


if __name__ == "__main__":
    main()
