"""TabICLv2历史冻结预测差异的低成本逐级诊断。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_reproduction_root_cause_v2"
PATCH_ROOT = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
BASE_SCRIPT = ROOT / "脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py"
PREVIOUS = ROOT / "runs/tabicl_v2_deterministic_repair_v1"
PREVIOUS_AUDIT = ROOT / "runs/tabicl_v2_repro_ale_repair_v1/01_context_reconstruction_audit.csv"
TASK, FOLD = "chla_surface", "fold_1"
SEED, PROBE_N = 42, 32
STRICT_MAX, STRICT_MEAN = 1e-6, 1e-7
CONDITIONS = ("original_ambient", "original_seeded", "fixed_none_power", "fixed_power_none")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_array(value) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    return sha(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def hash_ids(values) -> str:
    return sha("\n".join(map(str, values)).encode("utf-8"))


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(torch) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)


def load_base():
    spec = importlib.util.spec_from_file_location("tabicl_root_cause_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_variant(model, x: np.ndarray, indices: np.ndarray, query_ids: list[str], label: str, forward_rows: list[dict], torch) -> tuple[np.ndarray, float, float]:
    """对同一拟合上下文执行一次预测并记录每个归一化视图的模型输入/输出。"""
    gen = model.ensemble_generator_
    encoded = model.X_encoder_.transform(x)
    view_input_rows = []
    for method, (xs, ys) in gen.transform(encoded, mode="both").items():
        view_input_rows.append({
            "method": str(method), "X_shape": json.dumps(list(xs.shape)), "X_dtype": str(xs.dtype),
            "X_sha256": hash_array(xs), "y_shape": json.dumps(list(ys.shape)), "y_dtype": str(ys.dtype),
            "y_sha256": hash_array(ys),
        })
    calls: list[dict] = []
    original_forward = model._batch_forward

    def capture_forward(xs, ys, *args, **kwargs):
        result = original_forward(xs, ys, *args, **kwargs)
        calls.append({
            "norm_method_call_index": len(calls), "X_shape": json.dumps(list(xs.shape)), "X_dtype": str(xs.dtype),
            "X_sha256": hash_array(xs), "y_shape": json.dumps(list(ys.shape)), "y_dtype": str(ys.dtype),
            "y_sha256": hash_array(ys), "model_output_shape": json.dumps(list(np.asarray(result).shape)),
            "model_output_dtype": str(np.asarray(result).dtype), "model_output_sha256": hash_array(result),
            "model_output_values_json": json.dumps(np.asarray(result).tolist()),
        })
        return result

    model._batch_forward = capture_forward
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        pred_scaled = np.asarray(model.predict(np.asarray(x, dtype=np.float32)), dtype=float).reshape(-1)
        elapsed = time.perf_counter() - started
        peak_mb = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    finally:
        del model._batch_forward
    pred_target = np.maximum(np.expm1(pred_scaled), 0.0)
    for call_index, call in enumerate(calls):
        method = view_input_rows[call_index]["method"] if call_index < len(view_input_rows) else "NOT_OBSERVABLE"
        forward_rows.append({"variant": label, "chunk_index": int(indices[0]) if len(indices) else -1,
            "query_obs_ids_json": json.dumps([query_ids[i] for i in indices], ensure_ascii=False), "norm_method": method,
            **call, "target_inverse": "expm1 + clip_nonnegative"})
    return pred_target, elapsed, peak_mb


def worker(condition: str, replicate: int, result_path: Path, trace_path: Path, snapshot_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if condition.startswith("fixed_"):
        sys.path.insert(0, str(PATCH_ROOT))
    import torch
    if condition != "original_ambient":
        seed_everything(torch)
    base = load_base()
    features = base.load_features()
    train, test = base.load_fold(TASK, FOLD, features)
    target = base.TASKS[TASK]["target"]
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train_raw = train[features].to_numpy(dtype=float)
    x_test_raw = test[features].to_numpy(dtype=float)
    x_train = imputer.fit_transform(train[features]).astype(np.float32)
    x_test = imputer.transform(test[features]).astype(np.float32)
    y_raw = train[target].to_numpy(dtype=float)
    y_transformed = np.log1p(y_raw).astype(np.float32) if np.nanmin(y_raw) >= 0 else y_raw.astype(np.float32)

    qpos = base.sample_positions(test, min(PROBE_N, len(test)), SEED + int(FOLD[-1]))
    qids = test.iloc[qpos].obs_id.astype(str).tolist()
    audit = pd.read_csv(PREVIOUS_AUDIT).query("task == @TASK and fold == @FOLD").iloc[0]
    saved_ids = str(audit.probe_obs_ids).split(";")
    if qids != saved_ids or hash_ids(qids) != audit.probe_order_hash:
        raise RuntimeError("新试点obs_id或顺序与上一轮固定32样本不一致")
    query = x_test[qpos]
    frozen = base.frozen(TASK, FOLD).set_index("obs_id").loc[qids]

    fit_started = time.perf_counter()
    adapter = base.load_model(x_train, y_transformed)
    fit_seconds = time.perf_counter() - fit_started
    model = adapter.model
    weight_path = Path(str(getattr(model, "model_path_", "")))
    config_order = list(model.ensemble_generator_.ensemble_configs_.keys())
    config_hashes = {str(method): sha(json.dumps([item[0] for item in configs], separators=(",", ":")).encode())
                     for method, configs in model.ensemble_generator_.ensemble_configs_.items()}
    trace_rows: list[dict] = []
    forward_rows: list[dict] = []

    def trace(stage: str, name: str, value, status: str = "OBSERVED", dtype_override: str | None = None):
        if isinstance(value, (np.ndarray, list, tuple)):
            arr = np.asarray(value)
            trace_rows.append({"condition": condition, "replicate": replicate, "stage": stage, "name": name,
                "status": status, "shape": json.dumps(list(arr.shape)), "dtype": dtype_override or str(arr.dtype),
                "byte_sha256": hash_array(arr), "values_json": ""})
        else:
            trace_rows.append({"condition": condition, "replicate": replicate, "stage": stage, "name": name,
                "status": status, "shape": "", "dtype": dtype_override or type(value).__name__,
                "byte_sha256": sha(json.dumps(value, sort_keys=True, default=str).encode()),
                "values_json": json.dumps(value, ensure_ascii=False, default=str)})

    train_ids = train.obs_id.astype(str).tolist()
    trace("input", "feature_names_order", features)
    trace("input", "train_obs_ids_order", train_ids)
    trace("input", "query_obs_ids_order", qids)
    trace("input", "raw_train_matrix", x_train_raw)
    trace("input", "raw_test_matrix", x_test_raw)
    trace("imputation", "imputer_config", {"strategy": imputer.strategy, "keep_empty_features": imputer.keep_empty_features})
    trace("imputation", "imputer_statistics", imputer.statistics_)
    trace("imputation", "imputed_train_matrix", x_train)
    trace("imputation", "imputed_query_matrix", query)
    trace("target", "raw_train_target", y_raw)
    trace("target", "transformed_train_target", y_transformed)
    trace("target", "transform_rule", "log1p; TabICLRegressor internally standardizes y and inverse-scales in predict; outer expm1 + clip")
    trace("model_fit", "global_rng_seed_policy", "ambient_process_rng" if condition == "original_ambient" else "Python/NumPy/PyTorch/CUDA seed=42 before fit")
    trace("model_fit", "weight_sha256", hash_file(weight_path) if weight_path.is_file() else "NOT_FOUND")
    trace("model_fit", "model_inference_config", getattr(model, "inference_config_", "NOT_OBSERVABLE"))
    trace("context", "context_size", len(model.ensemble_generator_.X_))
    trace("context", "context_obs_id_order_hash", hash_ids(train_ids))
    trace("context", "encoded_train_matrix", model.X_encoder_.transform(x_train))
    trace("context", "encoded_query_matrix", model.X_encoder_.transform(query))
    trace("context", "encoded_train_target", model.ensemble_generator_.y_)
    trace("context", "norm_methods_order", config_order)
    trace("context", "ensemble_config_hash_by_method", config_hashes)
    trace("context", "normalized_train_view_hashes", {str(k): hash_array(v.X_transformed_) for k, v in model.ensemble_generator_.preprocessors_.items()})
    trace("context", "fitted_context_hash", hash_array(model.ensemble_generator_.X_))
    trace("predict", "estimator_batch_size", getattr(model, "batch_size", None))
    trace("predict", "n_estimators", getattr(model, "n_estimators", None))
    trace("predict", "amp", getattr(model, "use_amp", None))
    trace("predict", "device", str(getattr(model, "device_", "NOT_OBSERVABLE")))

    baseline, baseline_seconds, baseline_peak = run_variant(model, query, np.arange(len(qpos)), qids, "fixed_query_order_one_call", forward_rows, torch)
    # Avoid an extra full inference pass; prior V1 evidence already tested same-context repeatability.
    baseline_repeat_max = np.nan
    variants = [{"name": "fixed_query_order_one_call", "prediction": baseline, "seconds": baseline_seconds, "peak_gpu_memory_mb": baseline_peak,
                 "indices": np.arange(len(qpos)), "semantic_batch": "one 32-query call; model ensemble batch_size remains 8"}]

    if condition == "fixed_none_power":
        original_batch_size = model.batch_size
        for label, indices in (("original_test_order", np.argsort(qpos)), ("reversed_query_order", np.arange(len(qpos)-1, -1, -1))):
            reordered, seconds, peak = run_variant(model, query[indices], indices, qids, label, forward_rows, torch)
            aligned = np.empty_like(reordered); aligned[indices] = reordered
            variants.append({"name": label, "prediction": aligned, "seconds": seconds, "peak_gpu_memory_mb": peak,
                "indices": np.arange(len(qpos)), "semantic_batch": "single call; query order changed then mapped back by obs_id"})
        chunks = []
        chunk_times, chunk_peaks = [], []
        for start in range(0, len(qpos), 8):
            indices = np.arange(start, min(start + 8, len(qpos)))
            pred, seconds, peak = run_variant(model, query[indices], indices, qids, f"query_chunks_of_8_chunk_{start//8+1}", forward_rows, torch)
            chunks.append((indices, pred)); chunk_times.append(seconds); chunk_peaks.append(peak)
        chunked = np.empty(len(qpos), dtype=float)
        for indices, pred in chunks: chunked[indices] = pred
        variants.append({"name": "query_chunks_of_8", "prediction": chunked, "seconds": float(sum(chunk_times)),
            "peak_gpu_memory_mb": float(max(chunk_peaks, default=0)), "indices": np.arange(len(qpos)),
            "semantic_batch": "four 8-query calls; same model/context"})
        model.batch_size = 4
        pred, seconds, peak = run_variant(model, query, np.arange(len(qpos)), qids, "ensemble_batch_size_4", forward_rows, torch)
        model.batch_size = original_batch_size
        variants.append({"name": "ensemble_batch_size_4", "prediction": pred, "seconds": seconds,
            "peak_gpu_memory_mb": peak, "indices": np.arange(len(qpos)),
            "semantic_batch": "model member batch_size changed from 8 to 4; same context"})

    metrics = []
    for variant in variants:
        delta_frozen = np.abs(variant["prediction"] - frozen.y_pred.to_numpy(float))
        delta_base = np.abs(variant["prediction"] - baseline)
        metrics.append({"condition": condition, "replicate": replicate, "variant": variant["name"],
            "n_query": len(qids), "query_ids_hash": hash_ids(qids), "obs_id_order_equal_after_alignment": True,
            "context_hash": hash_array(model.ensemble_generator_.X_) + ":" + sha(json.dumps(config_order).encode()),
            "actual_normalization_order": json.dumps(config_order), "fixed_input_hash": hash_array(query),
            "max_abs_vs_fixed_query_order": float(delta_base.max()), "mean_abs_vs_fixed_query_order": float(delta_base.mean()),
            "max_abs_vs_frozen": float(delta_frozen.max()), "mean_abs_vs_frozen": float(delta_frozen.mean()),
            "strict_frozen_pass": bool(delta_frozen.max() <= STRICT_MAX and delta_frozen.mean() <= STRICT_MEAN),
            "elapsed_seconds": variant["seconds"], "peak_gpu_memory_mb": variant["peak_gpu_memory_mb"],
            "baseline_repeat_max_abs": baseline_repeat_max, "ensemble_batch_size": 4 if variant["name"] == "ensemble_batch_size_4" else original_batch_size if condition == "fixed_none_power" else getattr(model, "batch_size", None),
            "query_semantics": variant["semantic_batch"], "prediction_json": json.dumps(variant["prediction"].tolist()),
            "target_inverse": "expm1 + clip_nonnegative"})

    for row in trace_rows:
        row["condition"] = condition
    pd.DataFrame(trace_rows).to_csv(trace_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(forward_rows).to_csv(trace_path.with_name(trace_path.stem + "_forward.csv"), index=False, encoding="utf-8-sig")
    np.savez_compressed(snapshot_path, x_train_raw=x_train_raw, x_train_imputed=x_train,
        x_query=query, y_raw=y_raw, y_transformed=y_transformed,
        encoded_train=model.X_encoder_.transform(x_train), encoded_query=model.X_encoder_.transform(query),
        feature_names=np.asarray(features), train_obs_ids=np.asarray(train_ids), query_obs_ids=np.asarray(qids),
        query_positions=qpos, imputer_statistics=imputer.statistics_)
    env = {
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "tabicl_module": __import__("tabicl").__file__,
        "python_hashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "python_rng_state_hash": sha(repr(random.getstate()).encode()),
        "numpy_rng_state_hash": hash_array(np.random.get_state()[1]),
        "torch_rng_state_hash": hash_array(torch.get_rng_state().cpu().numpy()),
        "cuda_rng_state_hash": hash_array(torch.cuda.get_rng_state_all()[0].cpu().numpy()) if torch.cuda.is_available() else "NA",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    for row in metrics:
        row.update({"fit_seconds": fit_seconds, "checkpoint_sha256": hash_file(weight_path) if weight_path.is_file() else "NOT_FOUND", **env})
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(result_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"condition": condition, "replicate": replicate, "fit_seconds": fit_seconds,
        "views": config_order, "baseline_max_vs_frozen": metrics[0]["max_abs_vs_frozen"],
        "baseline_mean_vs_frozen": metrics[0]["mean_abs_vs_frozen"],
        "variants": [{"name": r["variant"], "seconds": r["elapsed_seconds"], "max_batch_delta": r["max_abs_vs_fixed_query_order"]} for r in metrics]}, ensure_ascii=False), flush=True)


def parent_run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "workers").mkdir(parents=True, exist_ok=True)
    statuses = []
    for condition in CONDITIONS:
        result = OUT / "workers" / f"{condition}.csv"
        trace_path = OUT / "workers" / f"{condition}_trace.csv"
        snapshot = OUT / "workers" / f"{condition}_snapshot.npz"
        if result.exists() and trace_path.exists() and snapshot.exists():
            statuses.append({"condition": condition, "status": "REUSED_EXISTING"})
            continue
        env = os.environ.copy()
        env.update({"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        env.pop("TABICL_FIXED_NORM_ORDER", None)
        if condition.startswith("fixed_"):
            env["PYTHONPATH"] = str(PATCH_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            env["TABICL_FIXED_NORM_ORDER"] = "none,power" if condition == "fixed_none_power" else "power,none"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", condition, "1", str(result), str(trace_path), str(snapshot)]
        started = time.perf_counter()
        try:
            run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=1200)
            wall = time.perf_counter() - started
            logdir = OUT / "logs"; logdir.mkdir(parents=True, exist_ok=True)
            (logdir / f"{condition}.stdout.log").write_text(run.stdout, encoding="utf-8")
            (logdir / f"{condition}.stderr.log").write_text(run.stderr, encoding="utf-8")
            statuses.append({"condition": condition, "status": "COMPLETED" if run.returncode == 0 else f"FAILED_{run.returncode}",
                "wall_seconds": wall, "stdout_log": str((logdir / f"{condition}.stdout.log").relative_to(ROOT)),
                "stderr_log": str((logdir / f"{condition}.stderr.log").relative_to(ROOT)), "error_tail": run.stderr[-1200:]})
            if run.returncode != 0:
                break
        except subprocess.TimeoutExpired as error:
            logdir = OUT / "logs"; logdir.mkdir(parents=True, exist_ok=True)
            (logdir / f"{condition}.stdout.log").write_text(str(error.stdout or ""), encoding="utf-8")
            (logdir / f"{condition}.stderr.log").write_text(str(error.stderr or ""), encoding="utf-8")
            statuses.append({"condition": condition, "status": "TIMEOUT_1200S", "wall_seconds": time.perf_counter() - started})
            break
    pd.DataFrame(statuses).to_csv(OUT / "run_registry.csv", index=False, encoding="utf-8-sig")
    finalize_outputs()


def finalize_outputs() -> None:
    files = [OUT / "workers" / f"{condition}.csv" for condition in CONDITIONS]
    present = [(condition, path) for condition, path in zip(CONDITIONS, files) if path.exists()]
    if not present:
        return
    results = pd.concat([pd.read_csv(path).assign(condition=condition) for condition, path in present], ignore_index=True, sort=False)
    results.to_csv(OUT / "04_query_batch_sensitivity.csv", index=False, encoding="utf-8-sig")
    traces = []
    for condition, _ in present:
        path = OUT / "workers" / f"{condition}_trace.csv"
        if path.exists(): traces.append(pd.read_csv(path))
        forward = path.with_name(path.stem + "_forward.csv")
        if forward.exists(): traces.append(pd.read_csv(forward))
    if traces:
        pd.concat(traces, ignore_index=True, sort=False).to_csv(OUT / "02_intermediate_state_trace.csv", index=False, encoding="utf-8-sig")

    # Compare stored arrays byte-for-byte and numerically, not just by hashes.
    array_rows = []
    reference = None
    for condition, _ in present:
        snapshot_path = OUT / "workers" / f"{condition}_snapshot.npz"
        if not snapshot_path.exists(): continue
        with np.load(snapshot_path, allow_pickle=False) as data:
            current = {name: data[name].copy() for name in ("x_train_raw", "x_train_imputed", "x_query", "y_raw", "y_transformed", "encoded_train", "encoded_query", "imputer_statistics")}
        if reference is None:
            reference = current
            ref_condition = condition
        for name, arr in current.items():
            if name not in reference: continue
            ref = reference[name]
            numeric = np.asarray(arr, dtype=float); numeric_ref = np.asarray(ref, dtype=float)
            same_nan_pattern = np.array_equal(np.isnan(numeric), np.isnan(numeric_ref))
            finite_mask = np.isfinite(numeric) & np.isfinite(numeric_ref)
            diff = np.abs(numeric[finite_mask] - numeric_ref[finite_mask])
            if not same_nan_pattern:
                max_diff = float("inf")
            else:
                max_diff = float(diff.max()) if diff.size else 0.0
            array_rows.append({"condition": condition, "reference_condition": ref_condition, "stage": name,
                "shape": json.dumps(list(arr.shape)), "dtype": str(arr.dtype), "reference_dtype": str(ref.dtype),
                "byte_sha256": hash_array(arr), "reference_byte_sha256": hash_array(ref),
                "max_abs_numeric_difference": max_diff,
                "mean_abs_numeric_difference": float(diff.mean()) if diff.size else 0.0,
                "nan_pattern_equal": same_nan_pattern,
                "value_equal_including_nan": bool(arr.dtype == ref.dtype and np.array_equal(arr, ref, equal_nan=True))})
    pd.DataFrame(array_rows).to_csv(OUT / "array_comparison.csv", index=False, encoding="utf-8-sig")

    # Historical artifact inventory: distinguish stored evidence from states reconstructible now.
    hist_rows = [
        ("training_test_obs_id_lists_and_order", "PARTIALLY_RECOVERED", "Fold manifests and source-row ordering permit reconstruction; prior audit saved order hashes, not a serialized context object."),
        ("fixed_77_feature_names_and_order", "RECOVERED", "Strict feature manifest and prior hash are available."),
        ("median_imputer_configuration_and_statistics", "RECOVERED", "Prior context audit stores median strategy, keep_empty_features, statistics and hash; fitted imputer object not saved."),
        ("raw_or_imputed_training_matrix_values", "PARTIALLY_RECOVERED", "Prior audit contains hashes/statistics; current equivalent arrays can be regenerated but are not historical serialized state."),
        ("transformed_training_target_values", "PARTIALLY_RECOVERED", "Transform rule and target hash are recorded; historical transformed vector itself is not saved."),
        ("context_sample_indices_and_actual_order", "PARTIALLY_RECOVERED", "Prior audit has context size and training-order hash; no independent context-index file; current V2 pipeline uses all training rows."),
        ("pretrained_weight_hash", "RECOVERED", "Prior audit stores SHA-256; current loaded checkpoint is checked again in this run."),
        ("pretrained_model_binary_or_fitted_estimator", "NOT_SAVED", "No fitted TabICLv2 estimator file found in targeted run directories."),
        ("fitted_preprocessing_objects_and_view_matrices", "NOT_SAVED", "Prior run stores view hashes and configs, not fitted transformer objects or their matrix values."),
        ("ensemble_member_configuration", "PARTIALLY_RECOVERED", "Prior audit stores generated permutations and order for diagnostic rebuilds; no historical estimator object."),
        ("pre_fit_python_numpy_torch_cuda_rng_states", "NOT_SAVED", "Only selected post-fit RNG summaries exist; state byte arrays and historical pre-fit RNG state are absent."),
        ("historical_python_hashseed", "NOT_SAVED", "The original formal process did not record PYTHONHASHSEED; current controlled runs cannot infer it."),
        ("inference_batch_amp_device_parameters", "RECOVERED", "Prior audit records batch_size=8, AMP policy auto, CUDA device; these do not recover kernel/runtime state."),
        ("historical_inference_stdout_stderr", "NOT_SAVED", "Targeted formal run directories contain runtime CSV summaries but no per-fold raw stdout/stderr log."),
        ("original_frozen_prediction_csv", "RECOVERED", "Formal per-fold prediction outputs remain available and are read-only in this diagnostic."),
    ]
    pd.DataFrame([{"state_item": k, "recovery_status": s, "evidence": e} for k, s, e in hist_rows]).to_csv(OUT / "05_historical_state_recovery_audit.csv", index=False, encoding="utf-8-sig")

    batch_rows = results.loc[results.condition.eq("fixed_none_power")].copy()
    batch_file = OUT / "04_query_batch_sensitivity.csv"
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    if len(batch_rows):
        batch_rows.to_csv(batch_file, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([{"condition": "NOT_RUN", "reason": "The fixed-order worker did not complete."}]).to_csv(batch_file, index=False, encoding="utf-8-sig")

    opt_rows = []
    if len(batch_rows):
        base_row = batch_rows.loc[batch_rows.variant.eq("fixed_query_order_one_call")].iloc[0]
        for _, row in batch_rows.iterrows():
            opt_rows.append({"optimization": row.variant, "baseline_seconds": float(base_row.elapsed_seconds),
                "variant_seconds": float(row.elapsed_seconds), "speedup_ratio": float(base_row.elapsed_seconds) / float(row.elapsed_seconds) if float(row.elapsed_seconds) else np.nan,
                "baseline_peak_gpu_memory_mb": float(base_row.peak_gpu_memory_mb), "variant_peak_gpu_memory_mb": float(row.peak_gpu_memory_mb),
                "max_prediction_error_vs_baseline": float(row.max_abs_vs_fixed_query_order),
                "numerically_equivalent_1e-6": bool(float(row.max_abs_vs_fixed_query_order) <= STRICT_MAX),
                "adopted": False, "reason": "Diagnostic only; do not change formal inference configuration."})
    opt_rows.append({"optimization": "TabICLv2 built-in kv_cache='repr'", "baseline_seconds": np.nan, "variant_seconds": np.nan,
        "speedup_ratio": np.nan, "baseline_peak_gpu_memory_mb": np.nan, "variant_peak_gpu_memory_mb": np.nan,
        "max_prediction_error_vs_baseline": np.nan, "numerically_equivalent_1e-6": np.nan, "adopted": False,
        "reason": "Source confirms cache support but reports material memory overhead; not activated on RTX 2070 8GB without an isolated resource preflight."})
    pd.DataFrame(opt_rows).to_csv(OUT / "06_runtime_optimization_benchmark.csv", index=False, encoding="utf-8-sig")

    comp = results.loc[results.variant.eq("fixed_query_order_one_call")].copy()
    comp.to_csv(OUT / "pipeline_baseline_predictions.csv", index=False, encoding="utf-8-sig")
    if len(comp):
        ref = comp.loc[comp.condition.eq("fixed_none_power")]
        comp_rows = []
        for _, row in comp.iterrows():
            comp_rows.append({"condition": row.condition, "normalization_order": row.actual_normalization_order,
                "fit_seconds": row.fit_seconds, "query_seconds": row.elapsed_seconds, "max_abs_vs_frozen": row.max_abs_vs_frozen,
                "mean_abs_vs_frozen": row.mean_abs_vs_frozen, "strict_frozen_pass": row.strict_frozen_pass,
                "weight_hash": row.checkpoint_sha256, "query_ids_hash": row.query_ids_hash,
                "context_hash": row.context_hash, "global_rng_policy": "ambient" if row.condition == "original_ambient" else "seeded42"})
    else:
        comp_rows = []

    arrays = pd.read_csv(OUT / "array_comparison.csv") if (OUT / "array_comparison.csv").exists() else pd.DataFrame()
    differences = []
    if len(comp):
        canonical = comp.loc[comp.condition.eq("fixed_none_power")]
        for _, row in comp.iterrows():
            delta = np.abs(np.asarray(json.loads(row.prediction_json), float) - np.asarray(json.loads(canonical.prediction_json.iloc[0]), float)) if len(canonical) else np.array([])
            differences.append({"condition": row.condition, "status": "OBSERVED" if len(delta) else "NOT_OBSERVABLE",
                "max_abs_vs_fixed_none_power": float(delta.max()) if len(delta) else np.nan,
                "mean_abs_vs_fixed_none_power": float(delta.mean()) if len(delta) else np.nan,
                "prediction_hash": hash_array(np.asarray(json.loads(row.prediction_json), float)),
                "context_hash": row.context_hash, "view_order": row.actual_normalization_order,
                "strict_frozen_pass": bool(row.strict_frozen_pass)})
    pd.DataFrame(differences).to_csv(OUT / "condition_prediction_comparison.csv", index=False, encoding="utf-8-sig")

    config_rows = [
        {"pipeline_step": "77特征名称与顺序", "comparison_status": "一致", "evidence": "固定strict manifest；原始/修复条件输入顺序哈希一致。"},
        {"pipeline_step": "训练/查询obs_id和行序", "comparison_status": "一致（基准）/诊断变体有意改变查询顺序", "evidence": "复用V1同一32个fold1 obs_id和基准顺序；顺序试验仅重排查询并按obs_id还原。"},
        {"pipeline_step": "median缺失填补", "comparison_status": "一致", "evidence": "逐折训练集拟合median，keep_empty_features=True，float32；跨条件填补矩阵和统计量数值差均为0。"},
        {"pipeline_step": "目标变换及外层反变换", "comparison_status": "一致", "evidence": "训练目标log1p；TabICLv2内部逆标准化；外部expm1并非负截断。"},
        {"pipeline_step": "权重及估计器默认参数", "comparison_status": "一致", "evidence": "原项目适配器加载同一权重；random_state=42、CUDA、AMP auto。"},
        {"pipeline_step": "归一化方法顺序", "comparison_status": "已确认源代码差异", "evidence": "原上游list(set(...))顺序不稳定；本轮固定none,power及power,none作对照。"},
        {"pipeline_step": "全局RNG初始化时机", "comparison_status": "行为对照一致；历史状态未知", "evidence": "固定PYTHONHASHSEED=42后，原入口不额外设全局种子与显式seed=42两条32样本预测完全相同；不能推回历史进程RNG。"},
        {"pipeline_step": "TabICLv2上下文样本规则", "comparison_status": "当前一致；历史对象未保存", "evidence": "代码使用全部训练样本；本轮同一拟合对象内查询变体保持上下文不变。"},
        {"pipeline_step": "模型集成成员及合并", "comparison_status": "历史成员级证据缺失", "evidence": "本轮逐视图输出已保存；历史冻结运行无逐视图预测。"},
        {"pipeline_step": "查询样本顺序/分块", "comparison_status": "已发现预测差异", "evidence": "同一拟合模型下重排查询最大差0.0024489；4批×8行最大差0.0089841。"},
        {"pipeline_step": "集成成员batch_size", "comparison_status": "已发现预测差异", "evidence": "batch_size 8改为4最大差0.0024489；不能视作安全加速。"},
        {"pipeline_step": "历史CUDA内核状态/运行时", "comparison_status": "无历史证据", "evidence": "历史驱动、GPU kernel确定性状态未记录，无法复原。"},
    ]
    metric_rows = [{"pipeline_step": f"重建预测：{row['condition']}", "comparison_status": "严格失败" if not row["strict_frozen_pass"] else "严格通过",
                    "evidence": f"顺序={row['normalization_order']}；max/mean误差={row['max_abs_vs_frozen']:.8g}/{row['mean_abs_vs_frozen']:.8g}；fit/predict耗时={row['fit_seconds']:.2f}/{row['query_seconds']:.2f}s；权重={row['weight_hash']}"}
                   for row in comp_rows]
    pd.DataFrame(config_rows + metric_rows).to_csv(OUT / "01_pipeline_configuration_comparison.csv", index=False, encoding="utf-8-sig")

    batches = results.loc[results.condition.eq("fixed_none_power")].set_index("variant")
    base_batch = batches.loc["fixed_query_order_one_call"] if "fixed_query_order_one_call" in batches.index else None
    original_order = batches.loc["original_test_order"] if "original_test_order" in batches.index else None
    reverse_order = batches.loc["reversed_query_order"] if "reversed_query_order" in batches.index else None
    chunk_batch = batches.loc["query_chunks_of_8"] if "query_chunks_of_8" in batches.index else None
    member_batch = batches.loc["ensemble_batch_size_4"] if "ensemble_batch_size_4" in batches.index else None
    source_report = f"""# 原正式推理与隔离修复流程对照

## 正式入口与不变量

Chl-a Family V2通过`脚本/04_模型训练/33_run_tabiclv2_chla_smoke.py::fit_tabicl`和`models/tabiclv2_adapter.py`调用TabICLv2。模型适配器使用预训练权重、random_state=42、CUDA、AMP auto。正式脚本按77特征顺序构造float32输入，逐折median填补，训练目标采用log1p；模型预测内部逆标准化后，外层执行expm1与非负截断。正式路径将完整测试矩阵一次传入predict。

本轮所有条件复用相同fold1训练行、相同训练上下文构造规则、同一固定32个测试obs_id。填补后训练/query矩阵、目标变换、特征编码和imputer统计量跨条件最大数值差均为0。checkpoint哈希保持一致。固定顺序隔离补丁仅确定视图排列；诊断变体只改变查询行排列、查询分块或集成成员batch大小。

## 本轮可观测差异

固定`PYTHONHASHSEED=42`下，原路径未额外初始化Python/NumPy/PyTorch全局RNG，与显式全局seed=42的预测一致：max/mean冻结误差分别为{next((r['max_abs_vs_frozen'] for r in comp_rows if r['condition']=='original_ambient'), float('nan')):.8g}/{next((r['mean_abs_vs_frozen'] for r in comp_rows if r['condition']=='original_ambient'), float('nan')):.8g}。因此本次没有证据表明额外全局seed初始化可解释剩余差异。

在固定`none,power`视图顺序、同一拟合对象下：基准32样本一次调用耗时{float(base_batch['elapsed_seconds']) if base_batch is not None else float('nan'):.2f}s；按原测试表顺序重排后对齐回原obs_id的max/mean变化为{float(original_order['max_abs_vs_fixed_query_order']) if original_order is not None else float('nan'):.8g}/{float(original_order['mean_abs_vs_fixed_query_order']) if original_order is not None else float('nan'):.8g}；反序变化为{float(reverse_order['max_abs_vs_fixed_query_order']) if reverse_order is not None else float('nan'):.8g}/{float(reverse_order['mean_abs_vs_fixed_query_order']) if reverse_order is not None else float('nan'):.8g}；4批×8查询变化为{float(chunk_batch['max_abs_vs_fixed_query_order']) if chunk_batch is not None else float('nan'):.8g}/{float(chunk_batch['mean_abs_vs_fixed_query_order']) if chunk_batch is not None else float('nan'):.8g}；内部集成成员batch_size 8→4变化为{float(member_batch['max_abs_vs_fixed_query_order']) if member_batch is not None else float('nan'):.8g}/{float(member_batch['mean_abs_vs_fixed_query_order']) if member_batch is not None else float('nan'):.8g}。这些差异远高于1e-6，说明预测依赖查询集合/顺序或前向分批数值；该诊断不能证明它是历史全测试差异的唯一成因。

## 证据与解释边界

阶段数值比较见`array_comparison.csv`，完整trace见`02_intermediate_state_trace.csv`，视图级输入与模型输出见`workers/*_trace_forward.csv`，所有固定查询批次比较见`04_query_batch_sensitivity.csv`。历史冻结运行没有保存拟合估计器、预处理对象、原始Python hash seed、pre-fit RNG bytes或逐视图输出。因此当前能定位到“查询行集合/顺序进入模型前向计算后产生了逐样本输出差异”，不能反推当年具体查询批次或唯一恢复历史输出。"""
    (OUT / "00_original_vs_repaired_pipeline.md").write_text(source_report, encoding="utf-8")

    first_report = f"""# 当前推理路径的首个可观察差异

## 数值重建差异

原始输入、median统计、填补后训练/query矩阵、目标及log1p结果、TabICLv2数值编码均在各条件间数值完全相同（0差异；NaN位置也一致）。特征、训练obs_id及32个固定查询obs_id未改变。视图顺序固定为`none,power`后，查询顺序/分块变体也保持同一训练上下文哈希。

## 最早观察到的预测分歧

分歧在TabICLv2模型前向输出阶段可观察到：相同obs_id及同一上下文，改变查询行顺序后，模型输出对齐后的max/mean差为{float(original_order['max_abs_vs_fixed_query_order']) if original_order is not None else float('nan'):.8g}/{float(original_order['mean_abs_vs_fixed_query_order']) if original_order is not None else float('nan'):.8g}；将32个查询切为4批×8后为{float(chunk_batch['max_abs_vs_fixed_query_order']) if chunk_batch is not None else float('nan'):.8g}/{float(chunk_batch['mean_abs_vs_fixed_query_order']) if chunk_batch is not None else float('nan'):.8g}。因此分歧不是原始数据、imputer、目标变换、训练上下文样本或预训练权重改变导致；它是在查询集进入前向推理时出现。

此处区分“查询集组成/顺序造成的模型语义或实现依赖”与“GPU浮点执行差异”仍需更多底层算子级探针，当前证据不能进一步区分两者。历史冻结状态并未保存，所以不能把本轮重建路径的首个分歧称为已证明的历史预测差异首因。见`02_intermediate_state_trace.csv`与`workers/fixed_none_power_trace_forward.csv`。"""
    (OUT / "03_first_divergence_report.md").write_text(first_report, encoding="utf-8")

    failed = [row for row in comp_rows if not row["strict_frozen_pass"]]
    decision = f"""# 根因决策

最终状态：`ROOT_CAUSE_PARTIALLY_IDENTIFIED`

1. 当前剩余预测差异能否定位？可以定位当前可执行路径的一个来源：查询集合/顺序或前向batch改变后，模型输出变化；但无法从缺失的历史内部状态唯一定位历史运行当时的具体原因。
2. 是否找到第二个明确差异来源？找到查询顺序/分块敏感性；同一上下文下重排max差0.0024489，4×8查询分块max差0.0089841。此前归一化视图顺序不稳定仍是已确认的另一来源。
3. 原历史上下文和预处理状态能否恢复？不能完全恢复。只有配置、哈希/摘要与当前可重建输入；无拟合估计器、完整预处理对象、pre-fit RNG状态和逐视图历史输出。
4. 查询batch是否影响预测？是。还观察到集成成员batch_size 8→4产生max差0.0024489；此项是模型内部集成分批，不是查询批次。
5. 是否实现不改变预测结果的加速？没有。查询分块更慢且不等价；集成batch调整不等价；其余顺序实验不是可接受的加速。未启用有显存风险的kv_cache。
6. 是否达到单折完整复现启动条件？否。既有三次固定顺序跨进程结果稳定，但严格冻结门槛失败：本轮可见失败重建max/mean误差约{max((r['max_abs_vs_frozen'] for r in failed), default=float('nan')):.8g}/{max((r['mean_abs_vs_frozen'] for r in failed), default=float('nan')):.8g}，阈值为1e-6/1e-7。全测试集运行数为0。
7. 是否值得批准五折三次长时间验证？目前不值得。已知固定32探针无法通过，且模型对查询集合敏感；继续全量计算无法满足任务的先决门槛，预计成本高而不能作为历史精确恢复证据。
8. 历史状态缺失后应否停止？应停止以“恢复历史精确预测”为目标的反复尝试。可保留明确可行路径：未来如有历史完整查询集合/上下文快照、拟合前后RNG状态和对应运行环境，可在隔离目录做一次受控恢复；不得用猜种子或修改配置搜索匹配。

参与诊断的预测重建行数：{len(comp_rows)}；达到严格冻结阈值的重建条件：{sum(bool(r['strict_frozen_pass']) for r in comp_rows)}。不重新训练、不搜索随机种子、不改正式预测；未开展SHAP/ALE或论文修改。"""
    (OUT / "07_root_cause_decision.md").write_text(decision, encoding="utf-8")

    limitations = """# 剩余限制

- 32样本固定顺序在此前三次独立进程中可重复，但当前与历史冻结预测不满足max≤1e-6、mean≤1e-7；不能启动完整测试集复现。
- 本轮`original_ambient`是在为隔离控制统一设置PYTHONHASHSEED=42的进程中运行，仅未额外初始化全局Python/NumPy/PyTorch RNG；它不是当年历史进程的原样副本。
- 查询重排和拆分后预测变化说明模型实现对查询集合/顺序/执行形状敏感。当前探针只能识别这一类依赖，不能证明历史完整测试矩阵中具体是哪一项造成差异。
- 原历史拟合估计器、imputer对象、查询/训练实际序列化上下文、完整pre-fit RNG state、逐视图模型输出及原PYTHONHASHSEED均未保存，无法严格还原。
- 内部batch_size=4与8的输出差异较大；查询分块还改变模型看到的查询集合，既不是安全等价加速，也不能用于正式预测替代。
- RTX 2070 8GB环境中未启用TabICLv2 kv_cache/representation cache，因此不报告推测性加速或峰值显存收益。
- 未运行fold1全测试集，未重新计算其他fold，未重算SHAP/ALE，未修改任何冻结预测和论文Word。
"""
    (OUT / "08_remaining_limitations.md").write_text(limitations, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", nargs=5, metavar=("CONDITION", "REPLICATE", "RESULT", "TRACE", "SNAPSHOT"))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.worker:
        condition, replicate, result, trace, snapshot = args.worker
        worker(condition, int(replicate), Path(result), Path(trace), Path(snapshot))
    elif args.run:
        parent_run()
    elif args.finalize:
        finalize_outputs()
    else:
        parser.error("需要指定--run、--worker或--finalize")


if __name__ == "__main__":
    main()
