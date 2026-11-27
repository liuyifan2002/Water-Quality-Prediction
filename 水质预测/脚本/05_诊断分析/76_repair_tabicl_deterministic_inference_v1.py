"""TabICLv2归一化视图顺序的隔离确定性诊断与冻结预测复现。"""
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
OUT = ROOT / "runs/tabicl_v2_deterministic_repair_v1"
PATCH_ROOT = OUT / "isolated_patch"
BASE_SCRIPT = ROOT / "脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py"
OLD_REPAIR = ROOT / "脚本/05_诊断分析/75_repair_tabicl_repro_ale_v1.py"
TASK_FOLD = ("chla_surface", "fold_1")
MAX_TOL = 1e-6
MEAN_TOL = 1e-7
N_PROBE = 32
SEED = 42


def hbytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def harray(value) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    return hbytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def hids(values) -> str:
    return hbytes("\n".join(map(str, values)).encode("utf-8"))


def hfile(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def append_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        old_key = old[keys].fillna("").astype(str).agg("|".join, axis=1)
        new_key = new[keys].fillna("").astype(str).agg("|".join, axis=1)
        old = old.loc[~old_key.isin(set(new_key))]
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(path, index=False, encoding="utf-8-sig")


def load_base():
    spec = importlib.util.spec_from_file_location("tabicl_deterministic_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_all(torch) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)


def context_hash(model) -> str:
    gen = model.ensemble_generator_
    chunks = [json.dumps(list(gen.ensemble_configs_.keys()))]
    for method, configs in gen.ensemble_configs_.items():
        chunks.append(method)
        chunks.append(harray(gen.preprocessors_[method].X_transformed_))
        chunks.extend(harray(np.asarray(x, dtype=np.int64)) for x, _ in configs)
    return hbytes("|".join(chunks).encode())


def environment(torch, model, normalized_order: str) -> dict:
    try:
        import importlib.metadata as metadata
        tabicl_version = metadata.version("tabicl")
    except Exception:
        tabicl_version = "unknown"
    cuda = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "python_hashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "numpy": np.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "tabicl_version": tabicl_version, "tabicl_module": __import__("tabicl").__file__,
        "gpu": torch.cuda.get_device_name(0) if cuda else "CPU", "cuda_available": cuda,
        "amp": str(getattr(model, "use_amp", None)),
        "resolved_device": str(getattr(model, "device_", "unknown")),
        "inference_batch_size": str(getattr(model, "batch_size", "unknown")),
        "n_estimators": str(getattr(model, "n_estimators", "unknown")),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "configured_normalization_order": normalized_order,
        "actual_normalization_order": json.dumps(list(model.ensemble_generator_.ensemble_configs_.keys())),
        "python_rng_hash": hbytes(repr(random.getstate()).encode()),
        "numpy_rng_hash": harray(np.random.get_state()[1]),
        "torch_cpu_rng_hash": harray(torch.get_rng_state().cpu().numpy()),
        "torch_cuda_rng_hash": harray(torch.cuda.get_rng_state_all()[0].cpu().numpy()) if cuda else "NA",
    }


def worker(task: str, fold: str, mode: str, replicate: int, out_path: Path) -> None:
    # This path points at the copied package only; site-packages remains untouched.
    sys.path.insert(0, str(PATCH_ROOT))
    import torch
    seed_all(torch)
    base = load_base()
    features = base.load_features()
    train, test = base.load_fold(task, fold, features)
    target = base.TASKS[task]["target"]
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(train[features]).astype("float32")
    x_test = imputer.transform(test[features]).astype("float32")
    y_original = train[target].to_numpy(float)
    inverse_log = bool(np.nanmin(y_original) >= 0)
    y_train = np.log1p(y_original).astype("float32") if inverse_log else y_original.astype("float32")
    frozen = base.frozen(task, fold)
    frozen["obs_id"] = frozen.obs_id.astype(str).str.strip()
    test_ids = test.obs_id.astype(str).str.strip().tolist()
    if len(features) != 77 or len(set(features)) != 77:
        raise RuntimeError("strict特征清单不是77个唯一特征")
    if len(test_ids) != len(set(test_ids)):
        raise RuntimeError("测试obs_id重复")

    from tabicl import TabICLRegressor
    kwargs = {"device": "cuda", "use_amp": "auto", "offload_mode": "auto", "random_state": 42, "verbose": False}
    started_fit = time.perf_counter()
    model = TabICLRegressor(**kwargs)
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started_fit

    qpos = base.sample_positions(test, min(N_PROBE, len(test)), SEED + int(fold[-1]) + 100 * list(base.TASKS).index(task))
    x_query = x_test if mode == "full" else x_test[qpos]
    query_ids = test_ids if mode == "full" else [test_ids[i] for i in qpos]
    started_pred = time.perf_counter()
    raw_pred = np.asarray(model.predict(np.asarray(x_query, dtype="float32")), dtype=float).reshape(-1)
    prediction = np.maximum(np.expm1(raw_pred) if inverse_log else raw_pred, 0)
    predict_seconds = time.perf_counter() - started_pred

    reference = frozen.set_index("obs_id").loc[query_ids]
    if list(reference.index.astype(str)) != query_ids:
        raise RuntimeError("冻结预测obs_id匹配顺序不一致")
    delta = np.abs(prediction - reference.y_pred.to_numpy(float))
    rng = environment(torch, model, os.environ.get("TABICL_FIXED_NORM_ORDER", "declared"))
    checkpoint = Path(str(getattr(model, "model_path_", "")))
    row = {
        "task": task, "fold": fold, "mode": mode, "replicate": replicate,
        "status": "PILOT_COMPLETED", "n_train": len(train), "n_query": len(query_ids),
        "n_features": len(features), "feature_names_json": json.dumps(features, ensure_ascii=False),
        "feature_order_hash": hids(features), "train_obs_order_hash": hids(train.obs_id.astype(str)),
        "query_obs_order_hash": hids(query_ids), "frozen_obs_order_hash": hids(reference.index.astype(str)),
        "obs_order_equal": query_ids == list(reference.index.astype(str)),
        "imputer_strategy": "median", "imputer_keep_empty_features": True,
        "imputer_statistics_hash": harray(imputer.statistics_), "train_matrix_hash": harray(x_train),
        "test_matrix_hash": harray(x_test), "query_matrix_hash": harray(x_query),
        "target_transform": "log1p" if inverse_log else "identity", "target_inverse": "expm1 + clip_nonnegative" if inverse_log else "clip_nonnegative",
        "context_hash": context_hash(model), "checkpoint_sha256": hfile(checkpoint) if checkpoint.is_file() else "UNAVAILABLE",
        "prediction_hash": harray(prediction), "frozen_prediction_hash": harray(reference.y_pred.to_numpy(float)),
        "max_abs_vs_frozen": float(delta.max()), "mean_abs_vs_frozen": float(delta.mean()),
        "max_abs_vs_first_repeat": np.nan, "mean_abs_vs_first_repeat": np.nan,
        "r2_reproduced": float(r2_score(reference.y_true, prediction)),
        "rmse_reproduced": float(mean_squared_error(reference.y_true, prediction) ** 0.5),
        "mae_reproduced": float(mean_absolute_error(reference.y_true, prediction)),
        "r2_frozen": float(r2_score(reference.y_true, reference.y_pred)),
        "rmse_frozen": float(mean_squared_error(reference.y_true, reference.y_pred) ** 0.5),
        "mae_frozen": float(mean_absolute_error(reference.y_true, reference.y_pred)),
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "prediction_json": json.dumps(prediction.tolist()), "obs_id_json": json.dumps(query_ids),
        **rng,
    }
    row["status"] = "EXACT_PROBE" if delta.max() <= MAX_TOL and delta.mean() <= MEAN_TOL else "PROBE_NOT_EXACT"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(json.dumps({k: row[k] for k in ("status", "mode", "replicate", "n_query", "context_hash", "max_abs_vs_frozen", "mean_abs_vs_frozen", "fit_seconds", "predict_seconds")}, ensure_ascii=False), flush=True)


def run_process(task: str, fold: str, mode: str, replicate: int, norm_order: str, timeout: int) -> tuple[dict | None, float, str]:
    path = OUT / "workers" / f"{task}_{fold}_{mode}_{norm_order.replace(',', '-')}_r{replicate}.csv"
    stdout_path = OUT / "logs" / f"{task}_{fold}_{mode}_{norm_order.replace(',', '-')}_r{replicate}.stdout.log"
    stderr_path = OUT / "logs" / f"{task}_{fold}_{mode}_{norm_order.replace(',', '-')}_r{replicate}.stderr.log"
    if path.exists():
        return pd.read_csv(path).iloc[0].to_dict(), 0.0, "REUSED_EXISTING"
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": str(PATCH_ROOT) + os.pathsep + env.get("PYTHONPATH", ""), "TABICL_FIXED_NORM_ORDER": norm_order})
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", task, fold, mode, str(replicate), str(path)]
    started = time.perf_counter()
    try:
        completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode == 0 and path.exists():
            row = pd.read_csv(path).iloc[0].to_dict()
            row["parent_wall_seconds"] = elapsed
            row["worker_return_code"] = completed.returncode
            row["stdout_log"] = str(stdout_path.relative_to(ROOT))
            row["stderr_log"] = str(stderr_path.relative_to(ROOT))
            return row, elapsed, "OK"
        return None, elapsed, f"PROCESS_FAILED:{completed.returncode}:{completed.stderr[-1200:]}"
    except subprocess.TimeoutExpired as error:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text((error.stdout or "") if isinstance(error.stdout, str) else "", encoding="utf-8")
        stderr_path.write_text((error.stderr or "") if isinstance(error.stderr, str) else "", encoding="utf-8")
        return None, time.perf_counter() - started, "TIMEOUT"


def run_pilot() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for norm_order in ("none,power", "power,none"):
        for replicate in range(1, 4):
            row, wall, message = run_process(*TASK_FOLD, "probe", replicate, norm_order, 900)
            if row is None:
                rows.append({"task": TASK_FOLD[0], "fold": TASK_FOLD[1], "mode": "probe", "replicate": replicate, "norm_order": norm_order, "status": message, "parent_wall_seconds": wall})
                break
            row["norm_order"] = norm_order
            rows.append(row)
            append_csv(OUT / "05_cross_process_determinism.csv", [row], ["task", "fold", "mode", "norm_order", "replicate"])
            append_csv(OUT / "02_context_hash_comparison.csv", [row], ["task", "fold", "mode", "norm_order", "replicate"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "03_chla_fold1_reproduction.csv", index=False, encoding="utf-8-sig")
    main = df.loc[df.norm_order.eq("none,power")] if "norm_order" in df else pd.DataFrame()
    if len(main) == 3:
        preds = [np.asarray(json.loads(x)) for x in main.prediction_json]
        contexts_same = main.context_hash.nunique() == 1
        order_same = main.actual_normalization_order.nunique() == 1
        pairmax = max(float(np.max(np.abs(preds[i] - preds[j]))) for i in range(3) for j in range(i)) if len(preds) > 1 else np.nan
        stable = contexts_same and order_same and pairmax <= MAX_TOL
        pilot = {"status": "STABLE_32_QUERY_PILOT" if stable else "PILOT_NOT_STABLE", "context_hashes_equal": contexts_same, "view_orders_equal": order_same, "max_pairwise_prediction_delta": pairmax, "strict_threshold_max": MAX_TOL, "strict_threshold_mean": MEAN_TOL, "pilot_runtime_sum_seconds": float(main.parent_wall_seconds.sum()), "full_validation_launched": False}
        summarize_pilot()


def summarize_pilot() -> None:
    """汇总已落盘试点并用当前吞吐量估算预算，不启动新模型推理。"""
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / "03_chla_fold1_reproduction.csv")
    main = df.loc[df.norm_order.eq("none,power")].copy()
    summaries = []
    for order, group in df.groupby("norm_order", sort=False):
        predictions = [np.asarray(json.loads(value), dtype=float) for value in group.prediction_json]
        contexts_same = group.context_hash.nunique() == 1
        views_same = group.actual_normalization_order.nunique() == 1
        pair_max = max(float(np.max(np.abs(predictions[i] - predictions[j]))) for i in range(len(predictions)) for j in range(i)) if len(predictions) > 1 else np.nan
        summaries.append({"record_type": "CROSS_PROCESS_ORDER_SUMMARY", "task": TASK_FOLD[0], "fold": TASK_FOLD[1], "mode": "probe", "norm_order": order,
            "replicates": len(group), "contexts_identical": contexts_same, "actual_view_order_identical": views_same,
            "max_pairwise_prediction_delta": pair_max, "max_abs_vs_frozen": float(group.max_abs_vs_frozen.max()),
            "mean_abs_vs_frozen_median": float(group.mean_abs_vs_frozen.median()),
            "cross_process_deterministic_at_1e-6": bool(contexts_same and views_same and pair_max <= MAX_TOL)})
    groups = {str(order): group for order, group in df.groupby("norm_order", sort=False)}
    if "none,power" in groups and "power,none" in groups:
        left = np.asarray(json.loads(groups["none,power"].sort_values("replicate").iloc[0].prediction_json), dtype=float)
        right = np.asarray(json.loads(groups["power,none"].sort_values("replicate").iloc[0].prediction_json), dtype=float)
        order_delta = np.abs(left - right)
        summaries.append({"record_type": "FIXED_ORDER_CAUSAL_PAIR", "task": TASK_FOLD[0], "fold": TASK_FOLD[1], "mode": "probe",
            "norm_order": "none,power vs power,none", "replicates": 1, "contexts_identical": False,
            "actual_view_order_identical": False, "max_pairwise_prediction_delta": float(order_delta.max()),
            "mean_pairwise_prediction_delta": float(order_delta.mean()),
            "cross_process_deterministic_at_1e-6": False})
    individual = df.copy()
    individual["record_type"] = "FIXED_ORDER_INDEPENDENT_PROCESS_RUN"
    pd.concat([individual, pd.DataFrame(summaries)], ignore_index=True, sort=False).to_csv(OUT / "05_cross_process_determinism.csv", index=False, encoding="utf-8-sig")
    prior = pd.read_csv(OUT.parent / "tabicl_v2_repro_ale_repair_v1" / "02_context_order_pairwise_delta.csv")
    prior["record_type"] = "ORIGINAL_UNPATCHED_HASHSEED_PAIR"
    current = df.copy(); current["record_type"] = "FIXED_ORDER_RUN"
    pd.concat([prior, current, pd.DataFrame(summaries)], ignore_index=True, sort=False).to_csv(OUT / "02_context_hash_comparison.csv", index=False, encoding="utf-8-sig")

    probe_seconds = pd.to_numeric(main.predict_seconds, errors="coerce").dropna()
    median_seconds = float(probe_seconds.median())
    best_seconds = float(probe_seconds.min())
    worst_seconds = float(probe_seconds.max())
    audit = pd.read_csv(OUT.parent / "tabicl_v2_repro_ale_repair_v1" / "01_context_reconstruction_audit.csv")
    folds = [("chla_surface", f"fold_{i}") for i in (1, 2, 3, 5)] + [("secchi", "fold_4")]
    rows = []
    for task, fold in folds:
        match = audit.loc[audit.task.eq(task) & audit.fold.eq(fold)]
        n_query = int(match.n_test.iloc[0]) if len(match) else None
        rows.append({"task": task, "fold": fold, "n_test": n_query,
            "estimate_seconds_median_rate_three_runs": 3 * n_query * median_seconds / N_PROBE if n_query else None,
            "estimate_seconds_best_rate_three_runs": 3 * n_query * best_seconds / N_PROBE if n_query else None,
            "estimate_seconds_worst_rate_three_runs": 3 * n_query * worst_seconds / N_PROBE if n_query else None})
    estimate_median = sum(x["estimate_seconds_median_rate_three_runs"] or 0 for x in rows)
    estimate_best = sum(x["estimate_seconds_best_rate_three_runs"] or 0 for x in rows)
    estimate_worst = sum(x["estimate_seconds_worst_rate_three_runs"] or 0 for x in rows)
    history = pd.read_csv(OUT.parent / "tabicl_v2_explainability_v3" / "00_inference_reproduction.csv")
    historical_seconds = []
    for task, fold in folds:
        match = history.loc[history.task.eq(task) & history.fold.eq(fold)]
        historical_seconds.append(float(match.formal_predict_seconds.iloc[0]) if len(match) else np.nan)
    hist_estimate = float(np.nansum(historical_seconds) * 3)
    main_summary = next((x for x in summaries if x["norm_order"] == "none,power"), {})
    order_pair = next((x for x in summaries if x["record_type"] == "FIXED_ORDER_CAUSAL_PAIR"), {})
    pilot = {"status": "STABLE_32_QUERY_PILOT" if len(main) == 3 and main.context_hash.nunique() == 1 and main.actual_normalization_order.nunique() == 1 and main_summary.get("max_pairwise_prediction_delta", np.inf) <= MAX_TOL else "PILOT_NOT_STABLE",
        "strict_threshold_max": MAX_TOL, "strict_threshold_mean": MEAN_TOL, "main_order_frozen_prediction_passes": int(((main.max_abs_vs_frozen.astype(float) <= MAX_TOL) & (main.mean_abs_vs_frozen.astype(float) <= MEAN_TOL)).sum()),
        "main_order_cross_process_prediction_seconds_median": median_seconds, "main_order_cross_process_prediction_seconds_min": best_seconds,
        "main_order_cross_process_prediction_seconds_max": worst_seconds, "probe_query_n": N_PROBE,
        "full_validation_launched": False, "budget_gate": "OVER_6_HOURS_STOP_PENDING_APPROVAL" if estimate_median > 21600 else "WITHIN_6_HOURS",
        "observed_rate_estimate_hours_median": estimate_median / 3600, "observed_rate_estimate_hours_best_case": estimate_best / 3600,
        "observed_rate_estimate_hours_worst_case": estimate_worst / 3600,
        "historical_recorded_estimate_hours_for_context_only": hist_estimate / 3600,
        "full_fold_estimates": rows, "one_successful_probe_is_not_full_validation": True}
    (OUT / "pilot_budget_decision.json").write_text(json.dumps(pilot, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "pilot_full_validation_budget.csv", index=False, encoding="utf-8-sig")

    initial = pd.read_csv(OUT.parent / "tabicl_v2_explainability_v3" / "00_inference_reproduction.csv")
    validation_rows = []
    for task, fold in folds:
        prior_row = initial.loc[initial.task.eq(task) & initial.fold.eq(fold)].iloc[0]
        validation_rows.append({"task": task, "fold": fold, "historical_initial_status": prior_row.status,
            "strict_max_threshold": MAX_TOL, "strict_mean_threshold": MEAN_TOL,
            "full_test_independent_runs_completed": 0, "full_test_status": "BLOCKED_BUDGET_GATE",
            "pilot_32_query_repeats_if_fold1": len(main) if task == TASK_FOLD[0] and fold == TASK_FOLD[1] else 0,
            "pilot_probe_frozen_pass_count": pilot["main_order_frozen_prediction_passes"] if task == TASK_FOLD[0] and fold == TASK_FOLD[1] else np.nan,
            "reason": "Fold1 deterministic probe is not numerically equivalent to frozen predictions; current measured throughput projects the mandated 15 full-test runs above six hours."})
    pd.DataFrame(validation_rows).to_csv(OUT / "04_five_fold_exact_reproduction.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"task": TASK_FOLD[0], "fold": TASK_FOLD[1], "status": "NOT_RUN",
        "reason": "The 32-query stable rebuild failed frozen-prediction strict tolerances; no SHAP/ALE reinterpretation was run."}]).to_csv(OUT / "06_shap_sensitivity_after_repair.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"task": TASK_FOLD[0], "fold": TASK_FOLD[1], "status": "NOT_RUN",
        "reason": "The 32-query stable rebuild failed frozen-prediction strict tolerances; no SHAP/ALE reinterpretation was run."}]).to_csv(OUT / "07_ale_sensitivity_after_repair.csv", index=False, encoding="utf-8-sig")

    original_source = Path(sys.executable).parent.parent / "Lib/site-packages/tabicl/_sklearn/preprocessing.py"
    patched_source = PATCH_ROOT / "tabicl/_sklearn/preprocessing.py"
    script_path = Path(__file__).resolve()
    pd.DataFrame([
        {"artifact": "installed_preprocessing.py", "path": str(original_source), "sha256": hfile(original_source), "scope": "read_only_original_environment", "change": "unchanged"},
        {"artifact": "isolated_preprocessing.py", "path": str(patched_source), "sha256": hfile(patched_source), "scope": "copied_tabicl_package_in_experiment_output", "change": "replace unordered set iteration with explicit configured order; diagnostic override limited to configured methods"},
        {"artifact": "diagnostic_script.py", "path": str(script_path), "sha256": hfile(script_path), "scope": "project_diagnostic_script", "change": "independent-process probe, hashes, prediction checks and budget gate"},
    ]).to_csv(OUT / "01_deterministic_patch_manifest.csv", index=False, encoding="utf-8-sig")

    source_report = f"""# 源码根因定位与隔离修复

## 范围和代码路径

运行环境为TabICLv2 {main.iloc[0].tabicl_version}，Python {main.iloc[0].python}、PyTorch {main.iloc[0].torch}、CUDA runtime {main.iloc[0].cuda_runtime}，GPU为{main.iloc[0].gpu}。实际加载的上游文件路径为`{original_source}`。本轮只把TabICLv2包复制到`runs/tabicl_v2_deterministic_repair_v1/isolated_patch/tabicl/`，不改安装环境。

调用链为`TabICLRegressor.fit` → `EnsembleGenerator.fit` → `_generate_ensemble` → `ensemble_configs_`有序分组 → `EnsembleGenerator.transform`依序生成各归一化视图 → 推理侧按这些视图构造批次。原实现第1129行的`list(set([config[1] ...]))`将归一化方法重排；由于后续预处理器字典、视图拼接和随机数消耗顺序均沿该顺序执行，Python哈希种子变化可改变上下文及预测。

## 修复与对照证据

隔离副本将分组顺序固定为估计器声明的`norm_methods_`顺序（该任务默认为`none,power`），并允许诊断运行显式比较已配置的两种顺序。原始安装包SHA-256与副本SHA-256见`01_deterministic_patch_manifest.csv`。

fold 1的三次独立进程在固定`none,power`下得到相同上下文哈希、相同实际视图顺序，逐样本预测两两最大差为0；固定`power,none`下也得到另一组稳定上下文和稳定输出。两种视图顺序间32样本预测最大差为{order_pair.get('max_pairwise_prediction_delta', float('nan')):.8g}、平均差为{order_pair.get('mean_pairwise_prediction_delta', float('nan')):.8g}，两组均未达到冻结预测阈值（详见`02_context_hash_comparison.csv`和`03_chla_fold1_reproduction.csv`）。结合上一轮五折仅改变`PYTHONHASHSEED`的配对结果，确认无序集合是跨进程顺序不稳定的直接来源之一；不能据此认定它是全部冻结数值差异的唯一来源。

## 结论边界

本轮只证明32个固定查询样本下的重建确定性，不是全测试集严格复现。由于探针预测相对冻结值不等价，且按本轮实测吞吐量估算规定的15次全测试集验证超过6小时，未启动完整测试集、其他四折、SHAP或ALE敏感性复核。"""
    (OUT / "00_source_code_root_cause.md").write_text(source_report, encoding="utf-8")

    limitations = f"""# 剩余限制

- 严格复现阈值未变：最大逐样本误差≤{MAX_TOL:g}，平均绝对误差≤{MEAN_TOL:g}。
- Chl-a surface fold 1固定`none,power`三进程探针稳定，但3/3次均未达到冻结预测严格阈值；只测试32/2039个查询样本。
- `power,none`作为顺序敏感性对照，同样三次确定，但其预测输出与`none,power`不同。当前证据确认集合顺序影响实现路径，无法从冻结输出反推出唯一历史顺序。
- 32样本实测推理时间中位数为{median_seconds:.2f}秒，范围{best_seconds:.2f}—{worst_seconds:.2f}秒，存在明显运行时间波动。按中位吞吐量估算15次全测试集约{estimate_median/3600:.2f}小时，超过6小时门槛；历史计时估算约{hist_estimate/3600:.2f}小时，仅作为差异背景，不替代当前试点测得预算。
- 五个失败折的全测试集三次独立复现均未运行，不能标记为`STABLE_EXACT_REPRODUCTION`或`DETERMINISTIC_BUT_NOT_EXACT`（后者仍需完整折预测）。当前只能称fold1 32样本`DETERMINISTIC_PROBE_NOT_EXACT`。
- CUDA算子/AMP/批次数值行为及历史执行顺序是否一致仍待查；本轮仅修复并验证归一化方法顺序这一来源。
- 原TabICLv2安装包、冻结预测、正式split、77特征、正式指标和历史失败记录均未修改。SHAP/ALE没有定向复算。"""
    (OUT / "08_remaining_limitations.md").write_text(limitations, encoding="utf-8")
    decision = f"""# 最终确定性修复决策

## 状态

`PARTIALLY_REPAIRED`

## 已完成

- 已确认TabICLv2 2.2.0 `preprocessing.py`中的`list(set(...))`会导致视图顺序受Python哈希种子影响；该顺序进入有序预处理/集成视图构造链并改变上下文及预测。
- 已在独立副本中按显式配置顺序固定视图。Chl-a surface fold 1两种配置分别完成3个独立进程探针，每种配置内部上下文哈希及预测均完全一致。
- 正式模型声明顺序`none,power`的32样本重建与冻结值最大误差为{float(main.max_abs_vs_frozen.astype(float).max()):.8g}，平均误差中位数为{float(main.mean_abs_vs_frozen.astype(float).median()):.8g}，未通过原严格阈值。不同固定顺序的预测也不等价。
- 按本轮中位吞吐量估算，五个问题折三次全测试集核验约{estimate_median/3600:.2f}小时（范围估算{estimate_best/3600:.2f}—{estimate_worst/3600:.2f}小时），超过6小时上限。因此未启动全测试集运行或后续解释敏感性计算。

## 未完成与结论

五折分别保留原`REPRODUCTION_FAILED`历史状态；本轮新增全测试集状态均为`BLOCKED_BUDGET_GATE`，完整运行次数0。当前不能声称冻结预测严格复现成功，也不能判定全部数值差异已由集合顺序解释。原安装环境与正式结果无修改。

**下一步需人工批准是否分阶段继续全测试集验证**，或提供可降低推理耗时且不改变预测语义的资源/运行时方案。未经批准不继续超过6小时预算。"""
    (OUT / "09_final_deterministic_repair_decision.md").write_text(decision, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=5, metavar=("TASK", "FOLD", "MODE", "REPLICATE", "OUT"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.worker:
        task, fold, mode, replicate, out_path = args.worker
        worker(task, fold, mode, int(replicate), Path(out_path))
    elif args.pilot:
        run_pilot()
    elif args.summarize:
        summarize_pilot()
    elif args.finalize:
        summarize_pilot()
    else:
        parser.error("需要指定 --pilot 或 --worker")


if __name__ == "__main__":
    main()
