"""Run only fixed P1 single-query prediction repetitions for numerical jitter estimation."""
from __future__ import annotations

import hashlib
import importlib.util
import argparse
import math
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_explanation_numerical_stability_v2"
RUN = ROOT / "runs/tabicl_v2_final_explanation_complete_v1"
SOURCE = ROOT / "脚本/05_诊断分析/79_run_all_tabicl_explanation_experiments.py"
TARGETS = ("chla_surface", "tn_surface")
N_REPEATS = 10
FAILED_QUERY_ID = "NM5_20141217_surface"
FAILED_FEATURE = "ssrd_mean_14d"
EPS = np.finfo(float).eps


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(x: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(x))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def state_hash(torch, model) -> str:
    parts = []
    for key, value in sorted(model.state_dict().items()):
        a = np.ascontiguousarray(value.detach().cpu().numpy())
        digest = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()
        parts.append(f"{key}:{digest}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def state_rng_hash(torch) -> tuple[str, str]:
    cpu = array_hash(torch.get_rng_state().cpu().numpy())
    cuda = array_hash(torch.cuda.get_rng_state_all()[0].cpu().numpy()) if torch.cuda.is_available() else "NA"
    return cpu, cuda


def append_atomic(path: Path, row: dict, keys: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.is_file() and path.stat().st_size:
        old = pd.read_csv(path)
        sig_old = set(tuple(map(str, x)) for x in old[list(keys)].itertuples(index=False, name=None))
        key = tuple(str(row[k]) for k in keys)
        if key in sig_old:
            return
        new = pd.concat([old, new], ignore_index=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    new.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def frozen_predictions(task: str) -> pd.DataFrame:
    folder = ROOT / ("runs/chla_family_v2/predictions" if task == "chla_surface" else "runs/tn_family_v2/predictions")
    path = folder / f"{task}_tabiclv2_fold_1.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"obs_id": "string"})


def choose_queries(task: str, features: list[str], test: pd.DataFrame, xte: np.ndarray) -> list[dict]:
    frozen = frozen_predictions(task)
    if set(frozen.obs_id.astype(str)) != set(test.obs_id.astype(str)):
        raise RuntimeError(f"Frozen prediction obs_id set mismatch for {task}/fold_1")
    pred_by_id = frozen.set_index(frozen.obs_id.astype(str)).y_pred.astype(float)
    items: list[dict] = []
    if task == "chla_surface":
        pos = test.obs_id.astype(str).tolist().index(FAILED_QUERY_ID)
        feature_index = features.index(FAILED_FEATURE)
        failed_row = xte[pos].copy()
        # Reuse the exact already-recorded ALE lower boundary, not a newly selected value.
        repeat = pd.read_csv(RUN / "ale_1d_repeatability.csv")
        repeat = repeat.loc[(repeat.task == task) & (repeat.fold == "fold_1") & (repeat.feature == FAILED_FEATURE)].iloc[0]
        failed_row[feature_index] = float(repeat.x_lower)
        items.append({"probe_id": "chla_fold1_failed_ssrd_lower", "obs_id": FAILED_QUERY_ID,
                      "selection_rule": "existing V1 failed ALE lower-endpoint query; no new sample selection",
                      "frozen_prediction": float(pred_by_id[FAILED_QUERY_ID]), "x": failed_row})
    excluded = {x["obs_id"] for x in items}
    ordered = frozen.assign(_id=frozen.obs_id.astype(str)).sort_values(["y_pred", "_id"], kind="stable")
    low = ordered.loc[~ordered._id.isin(excluded)].iloc[0]
    high = ordered.loc[~ordered._id.isin(excluded)].iloc[-1]
    test_pos = {str(oid): i for i, oid in enumerate(test.obs_id.astype(str))}
    for label, row in (("low", low), ("high", high)):
        oid = str(row._id)
        items.append({"probe_id": f"{task}_fold1_frozen_pred_{label}", "obs_id": oid,
                      "selection_rule": f"minimum/maximum archived frozen TabICLv2 fold-1 y_pred, excluding already-fixed failed query where applicable",
                      "frozen_prediction": float(row.y_pred), "x": xte[test_pos[oid]].copy()})
    return items


def load_runner():
    spec = importlib.util.spec_from_file_location("tabicl_numerical_jitter_runner", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def target_scale(task: str) -> dict:
    folder = ROOT / ("runs/chla_family_v2/predictions" if task == "chla_surface" else "runs/tn_family_v2/predictions")
    frames = [pd.read_csv(folder / f"{task}_tabiclv2_fold_{i}.csv") for i in range(1, 6)]
    y = pd.concat([f.y_true for f in frames], ignore_index=True).astype(float).to_numpy()
    core = pd.read_csv(ROOT / "runs/final_experiment_closure_audit_v1/final_core_results_table_v2.csv")
    row = core.loc[(core.task == task) & (core.model == "TabICLv2")]
    if len(row) != 1:
        raise RuntimeError(f"Expected one official pooled metric row for {task}")
    target_name = "chla_ug_l" if task == "chla_surface" else "tn_mg_l"
    return {"target_name": target_name, "target_distribution_source": "official five-fold OOF prediction CSV y_true values",
            "target_n": int(len(y)), "target_median": float(np.median(y)),
            "target_q25": float(np.quantile(y, 0.25)), "target_q75": float(np.quantile(y, 0.75)),
            "target_iqr": float(np.quantile(y, 0.75) - np.quantile(y, 0.25)),
            "target_sd": float(np.std(y, ddof=1)), "official_pooled_rmse": float(row.iloc[0].RMSE),
            "official_pooled_mae": float(row.iloc[0].MAE)}


def do_probes() -> None:
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("Set PYTHONHASHSEED=42 before starting Python; do not set it after interpreter startup.")
    # Keep the ordinary previously recorded CUDA settings; do not enable deterministic CUDA mode.
    os.environ["TABICL_FIXED_NORM_ORDER"] = "none,power"
    patch = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
    os.environ["PYTHONPATH"] = str(patch) + os.pathsep + os.environ.get("PYTHONPATH", "")
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_runner()
    manifest_path = OUT / "probe_manifest.csv"
    predictions_path = OUT / "p1_repeat_predictions.csv"
    all_manifest = []

    for task in TARGETS:
        base, torch, features, train, test, xtr, xte, imp, adapter, est, pred, env, calls = module.make_context(task, "fold_1")
        pred.cache_enabled = False
        model_hash_before = state_hash(torch, est.model_)
        env["model_state_hash"] = model_hash_before
        if task == "chla_surface":
            expected = pd.read_csv(RUN / "environment_manifest.csv")
            expected = expected.loc[(expected.task == task) & (expected.fold == "fold_1")].iloc[-1]
            checks = {"feature_order_hash": expected.feature_order_hash,
                      "checkpoint_sha256": expected.checkpoint_sha256,
                      "imputer_stats_hash": expected.imputer_stats_hash,
                      "context_hash": expected.context_hash,
                      "model_state_hash": expected.model_state_hash}
            bad = {k: {"expected": str(v), "actual": str(env[k])} for k, v in checks.items() if str(v) != str(env[k])}
            if bad:
                raise RuntimeError(f"Chla fixed-context hash precondition failed; no Chla probes run: {bad}")
        items = choose_queries(task, features, test, xte)
        scales = target_scale(task)
        for qorder, item in enumerate(items, 1):
            x = np.asarray(item["x"], dtype=np.float32)
            if x.ndim != 1 or x.shape[0] != 77:
                raise RuntimeError(f"Bad query shape for {item['probe_id']}: {x.shape}")
            qhash = array_hash(x)
            mrow = {k: v for k, v in item.items() if k != "x"}
            mrow.update({"task": task, "fold": "fold_1", "query_order": qorder,
                         "query_shape": "1x77", "query_dtype": str(x.dtype),
                         "query_input_sha256": qhash, "feature_order_hash": env["feature_order_hash"],
                         "checkpoint_sha256": env["checkpoint_sha256"], "train_context_hash": env["context_hash"],
                         "imputer_stats_hash": env["imputer_stats_hash"], "model_state_hash": model_hash_before,
                         "norm_order": env["norm_order"], "device": env["device"], "use_amp": env["use_amp"],
                         "ensemble_batch_size": env["ensemble_batch_size"], "ensemble_members": env["ensemble_members"],
                         "torch_deterministic_algorithms": env["torch_deterministic_algorithms"],
                         "tf32_matmul": env["tf32_matmul"], "tf32_cudnn": env["tf32_cudnn"], **scales})
            all_manifest.append(mrow)
            append_atomic(manifest_path, mrow, ("probe_id",))
            for repeat_i in range(1, N_REPEATS + 1):
                if predictions_path.is_file():
                    existing = pd.read_csv(predictions_path)
                    already = ((existing.probe_id.astype(str) == str(item["probe_id"]))
                               & (existing.repeat_index.astype(int) == repeat_i)).any()
                    if already:
                        continue
                if array_hash(x) != qhash or x.shape != (77,):
                    raise RuntimeError("Query changed between repetitions; aborting this query.")
                cpu_rng_before, cuda_rng_before = state_rng_hash(torch)
                calls_before = int(calls["n"])
                tic = time.perf_counter()
                value = float(pred(x[None, :])[0])
                elapsed = time.perf_counter() - tic
                actual_calls = int(calls["n"]) - calls_before
                cpu_rng_after, cuda_rng_after = state_rng_hash(torch)
                row = {
                    "task": task, "fold": "fold_1", "probe_id": item["probe_id"], "obs_id": item["obs_id"],
                    "repeat_index": repeat_i, "prediction": value, "target_unit": scales["target_name"],
                    "query_shape": "1x77", "query_dtype": str(x.dtype), "query_input_sha256": qhash,
                    "feature_order_hash": env["feature_order_hash"], "checkpoint_sha256": env["checkpoint_sha256"],
                    "train_context_hash": env["context_hash"], "imputer_stats_hash": env["imputer_stats_hash"],
                    "model_state_hash": model_hash_before, "actual_model_calls": actual_calls,
                    "cache_enabled": False, "elapsed_seconds": elapsed,
                    "torch_cpu_rng_before": cpu_rng_before, "torch_cpu_rng_after": cpu_rng_after,
                    "torch_cuda_rng_before": cuda_rng_before, "torch_cuda_rng_after": cuda_rng_after,
                    "pid": os.getpid(), "python": platform.python_version(), "numpy": np.__version__,
                    "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                    "prediction_function": "f_fold(X_query | X_train,y_train,pretrained_model,inference_config); P1 one-query",
                }
                if actual_calls != 1:
                    row["call_count_warning"] = f"expected one uncached model call, measured {actual_calls}"
                append_atomic(predictions_path, row, ("probe_id", "repeat_index"))
                print(json.dumps({"task": task, "probe_id": item["probe_id"], "repeat": repeat_i,
                                  "prediction": value, "seconds": elapsed, "actual_calls": actual_calls}, ensure_ascii=False), flush=True)
        model_hash_after = state_hash(torch, est.model_)
        if model_hash_after != model_hash_before:
            raise RuntimeError(f"Model state changed during prediction probes for {task}: {model_hash_before} != {model_hash_after}")


def summarize() -> None:
    pred = pd.read_csv(OUT / "p1_repeat_predictions.csv")
    manifests = pd.read_csv(OUT / "probe_manifest.csv")
    summaries = []
    for probe_id, g in pred.groupby("probe_id", sort=False):
        x = g.prediction.astype(float).to_numpy()
        m = manifests.loc[manifests.probe_id == probe_id].iloc[0]
        mean = float(math.fsum(map(float, x)) / len(x))
        sd = float(math.sqrt(math.fsum((float(v) - mean) ** 2 for v in x) / (len(x) - 1))) if len(x) > 1 else 0.0
        summaries.append({"task": m.task, "fold": m.fold, "probe_id": probe_id, "obs_id": m.obs_id,
                          "n_repeats": len(x), "mean": mean, "std_sample": sd,
                          "min": float(np.min(x)), "max": float(np.max(x)), "range": float(np.ptp(x)),
                          "max_abs_deviation_from_mean": float(np.max(np.abs(x - mean))),
                          "max_pairwise_abs_delta": float(np.max(np.abs(x[:, None] - x[None, :]))),
                          "relative_range": float(np.ptp(x) / max(abs(mean), EPS)),
                          "input_hash_nunique": int(g.query_input_sha256.nunique()),
                          "context_hash_nunique": int(g.train_context_hash.nunique()),
                          "checkpoint_hash_nunique": int(g.checkpoint_sha256.nunique()),
                          "feature_order_hash_nunique": int(g.feature_order_hash.nunique()),
                          "model_state_hash_nunique": int(g.model_state_hash.nunique()),
                          "actual_calls_min": int(g.actual_model_calls.min()),
                          "actual_calls_total": int(g.actual_model_calls.sum()),
                          "all_cache_off": bool((g.cache_enabled.astype(str).str.lower() == "false").all()),
                          "target_name": m.target_name, "target_median": m.target_median,
                          "target_iqr": m.target_iqr, "target_sd": m.target_sd,
                          "official_pooled_rmse": m.official_pooled_rmse, "official_pooled_mae": m.official_pooled_mae,
                          "target_n": m.target_n})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "numerical_jitter_summary.csv", index=False, encoding="utf-8-sig")
    task_rows = []
    for r in summary.itertuples(index=False):
        task_rows.append({"task": r.task, "fold": r.fold, "probe_id": r.probe_id, "target_name": r.target_name,
                          "jitter_range": r.range, "target_median": r.target_median, "target_iqr": r.target_iqr,
                          "target_sd": r.target_sd, "official_pooled_rmse": r.official_pooled_rmse,
                          "official_pooled_mae": r.official_pooled_mae, "target_n": r.target_n,
                          "jitter_over_iqr": r.range / max(abs(r.target_iqr), EPS),
                          "jitter_over_sd": r.range / max(abs(r.target_sd), EPS),
                          "jitter_over_rmse": r.range / max(abs(r.official_pooled_rmse), EPS),
                          "jitter_over_mae": r.range / max(abs(r.official_pooled_mae), EPS)})
    # Preserve the already-frozen nonzero V1 event as historical evidence, never as a V2 rerun.
    v1_audit = json.loads((RUN / "fold_1_completion_audit.json").read_text(encoding="utf-8"))
    v1_row = next(x for x in v1_audit["ale1_checks"] if x["feature"] == "ssrd_mean_14d")
    chla_task = summary.loc[summary.task == "chla_surface"].iloc[0]
    hist_jitter = float(v1_row["max_local_ale_delta_range"])
    task_rows.append({"task": "chla_surface", "fold": "fold_1", "probe_id": "V1_historical_ssrd_repeatability_failure",
                      "target_name": chla_task.target_name, "jitter_range": hist_jitter,
                      "target_median": chla_task.target_median, "target_iqr": chla_task.target_iqr,
                      "target_sd": chla_task.target_sd, "official_pooled_rmse": chla_task.official_pooled_rmse,
                      "official_pooled_mae": chla_task.official_pooled_mae, "target_n": chla_task.target_n,
                      "jitter_over_iqr": hist_jitter / max(abs(float(chla_task.target_iqr)), EPS),
                      "jitter_over_sd": hist_jitter / max(abs(float(chla_task.target_sd)), EPS),
                      "jitter_over_rmse": hist_jitter / max(abs(float(chla_task.official_pooled_rmse)), EPS),
                      "jitter_over_mae": hist_jitter / max(abs(float(chla_task.official_pooled_mae)), EPS),
                      "evidence_status": "frozen historical V1 failure; not a V2 probe"})
    pd.DataFrame(task_rows).to_csv(OUT / "jitter_vs_task_scale.csv", index=False, encoding="utf-8-sig")

    explanation_rows = []
    shap_files = sorted((RUN / "shap_values/chla_surface/fold_1").glob("*.csv.gz"))
    for path in shap_files:
        d = pd.read_csv(path)
        obs_id = str(d.obs_id.iloc[0])
        a = d.shap_value.astype(float).abs().to_numpy()
        for metric, value in (("median_abs_shap", np.median(a)), ("p75_abs_shap", np.quantile(a, .75)),
                              ("p90_abs_shap", np.quantile(a, .90))):
            explanation_rows.append({"method": "SHAP", "sample_id": obs_id, "effect_scale_type": metric,
                                     "feature": "", "effect_scale": float(value), "evidence_status": "existing_single_SHAP_sample"})
        for _, row in d.assign(abs_shap=d.shap_value.astype(float).abs()).nlargest(10, "abs_shap").iterrows():
            explanation_rows.append({"method": "SHAP", "sample_id": obs_id, "effect_scale_type": "top10_feature_abs_shap",
                                     "feature": row.feature, "effect_scale": float(row.abs_shap),
                                     "evidence_status": "existing_single_SHAP_sample"})
    ale = pd.read_csv(RUN / "05_ale_1d_results.csv")
    ale = ale.loc[(ale.task == "chla_surface") & (ale.fold == "fold_1")
                  & ale.feature.isin(["ssrd_mean_14d", "wind_speed_x_rain14d"])]
    for feature, g in ale.groupby("feature", sort=False):
        g = g.sort_values("bin")
        curve = g.ale.astype(float).to_numpy()
        local = g.local_effect.astype(float).to_numpy()
        diff = np.diff(curve)
        vals = [("ale_curve_range", np.ptp(curve)), ("median_abs_local_effect", np.median(np.abs(local))),
                ("p90_abs_local_effect", np.quantile(np.abs(local), .90)),
                ("median_abs_bin_to_bin_change", np.median(np.abs(diff))),
                ("p90_abs_bin_to_bin_change", np.quantile(np.abs(diff), .90))]
        for metric, value in vals:
            explanation_rows.append({"method": "ALE", "sample_id": "chla_surface_fold_1",
                                     "effect_scale_type": metric, "feature": feature,
                                     "effect_scale": float(value), "evidence_status": "existing_single_ALE_curve"})
        for i, value in enumerate(np.abs(diff), 1):
            explanation_rows.append({"method": "ALE", "sample_id": "chla_surface_fold_1",
                                     "effect_scale_type": "bin_to_bin_abs_change", "feature": f"{feature}:bin_{i}_to_{i+1}",
                                     "effect_scale": float(value), "evidence_status": "existing_single_ALE_curve"})
    exp = pd.DataFrame(explanation_rows)
    exp.to_csv(OUT / "_explanation_effect_scales.csv", index=False, encoding="utf-8-sig")
    chla_jitters = summary.loc[summary.task == "chla_surface", ["probe_id", "range"]]
    chla_jitters = pd.concat([chla_jitters, pd.DataFrame([{"probe_id": "V1_historical_ssrd_repeatability_failure",
                                                           "range": hist_jitter}])], ignore_index=True)
    comparisons = []
    for _, j in chla_jitters.iterrows():
        for _, e in exp.iterrows():
            denom = abs(float(e.effect_scale))
            comparisons.append({"probe_id": j.probe_id, "prediction_jitter_range": float(j.range),
                                "method": e.method, "effect_sample_id": e.sample_id,
                                "effect_scale_type": e.effect_scale_type, "feature": e.feature,
                                "effect_scale": e.effect_scale,
                                "jitter_over_effect_scale": float(j.range) / max(denom, EPS),
                                "effect_scale_near_zero": bool(denom <= EPS),
                                "evidence_status": e.evidence_status})
    pd.DataFrame(comparisons).to_csv(OUT / "jitter_vs_explanation_scale.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"summary_rows": len(summary), "explanation_scale_rows": len(exp),
                      "jitter_vs_explanation_rows": len(comparisons), "all_queries_have_10": bool(summary.n_repeats.eq(10).all())}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.summarize_only:
        do_probes()
    summarize()


if __name__ == "__main__":
    main()
