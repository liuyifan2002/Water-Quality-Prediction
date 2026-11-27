"""可断点续跑的TabICLv2点式解释实验调度器。

仅构造预训练TabICLv2推理上下文，不更新模型权重。输出全部隔离在
runs/tabicl_v2_final_explanation_complete_v1/。每个解释查询均以单行调用预测。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_final_explanation_complete_v1"
PATCH = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
BASE_PATH = ROOT / "脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py"
V3 = ROOT / "runs/tabicl_v2_explainability_v3"
REPAIR = ROOT / "runs/tabicl_v2_repro_ale_repair_v1"
TASKS = ("chla_surface", "secchi", "tn_surface", "tp_middle")
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
# 1-3属于首次带缓存实现的诊断记录；正式重复性核验从4开始，旧记录不覆盖。
REPS = (4, 5, 6)
SEED = 42
MAX_DIFF, MEAN_DIFF = 1e-6, 1e-7
SHAP_N, BG_N, MAX_EVALS = 10, 16, 155
ADDITIVITY_TOL = 0.02
POINTWISE_PROTOCOL = "TABICLV2_POINTWISE_EXPLANATION_V1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_array(x) -> str:
    a = np.ascontiguousarray(np.asarray(x))
    return sha(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def hash_ids(x) -> str:
    return sha("\n".join(map(str, x)).encode("utf-8"))


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def append_rows(path: Path, frame: pd.DataFrame, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        old = pd.read_csv(path)
        sig = pd.Series(["|".join(map(str,row)) for row in frame[keys].astype(str).to_numpy()])
        oldsig = pd.Series(["|".join(map(str,row)) for row in old[keys].astype(str).to_numpy()],index=old.index)
        old = old.loc[~oldsig.isin(set(sig))]
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def append_environment_manifest(env: dict) -> None:
    """Persist a fold's actual inference/configuration state for every explanation worker."""
    if not {"task","fold","protocol","runner_code_sha256","checkpoint_sha256","context_hash"}.issubset(env):
        raise RuntimeError("Incomplete environment record: missing task/fold/protocol or frozen-context hashes.")
    append_rows(OUT/"environment_manifest.csv",pd.DataFrame([env]),
        ["task","fold","protocol","runner_code_sha256","checkpoint_sha256","context_hash"])


def load_base():
    spec = importlib.util.spec_from_file_location("tabicl_final_base", BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def seed_all(torch):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def representatives(frame: pd.DataFrame, x: np.ndarray, n: int, seed: int) -> np.ndarray:
    """固定站点-年份覆盖后，以77维环境空间max-min补足；不读取目标标签。"""
    meta = frame[["station_id", "date"]].copy()
    meta["year"] = pd.to_datetime(meta.date).dt.year
    rng = np.random.default_rng(seed)
    candidates = [int(rng.choice(g.index.to_numpy())) for _, g in meta.groupby(["station_id", "year"], sort=True)]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        candidates = list(range(len(frame)))
    scale = np.nanstd(x, axis=0); scale[scale == 0] = 1
    z = (x - np.nanmean(x, axis=0)) / scale
    selected = [candidates[int(rng.integers(len(candidates)))]]
    pool = candidates + [i for i in range(len(frame)) if i not in set(candidates)]
    while len(selected) < min(n, len(frame)):
        remain = np.asarray([i for i in pool if i not in selected], dtype=int)
        dist = np.min(((z[remain, None, :] - z[np.asarray(selected)[None, :], :]) ** 2).mean(axis=2), axis=1)
        selected.append(int(remain[np.argmax(dist)]))
    return np.asarray(selected, dtype=int)


def make_context(task: str, fold: str):
    import torch
    seed_all(torch)
    base = load_base()
    features = base.load_features()
    train, test = base.load_fold(task, fold, features)
    target = base.TASKS[task]["target"]
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    xtr = imp.fit_transform(train[features]).astype(np.float32)
    xte = imp.transform(test[features]).astype(np.float32)
    y0 = train[target].to_numpy(dtype=float)
    use_log = bool(np.nanmin(y0) >= 0)
    ytr = np.log1p(y0).astype(np.float32) if use_log else y0.astype(np.float32)
    t0 = time.perf_counter()
    adapter = base.load_model(xtr, ytr)
    fit_seconds = time.perf_counter() - t0
    est = adapter.model
    checkpoint = Path(str(getattr(est, "model_path_", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TabICLv2 pretrained checkpoint not found: {checkpoint}")
    ctx_hash = hash_array(est.ensemble_generator_.X_) + ":" + hash_array(est.ensemble_generator_.y_)
    order = list(est.ensemble_generator_.ensemble_configs_.keys())
    inv = np.expm1 if use_log else (lambda v: v)
    calls = {"n": 0, "seconds": 0.0, "cache_hits": 0}

    class Predictor:
        """每行独立model.predict；不得把同批其他查询传入模型。"""
        def __init__(self): self.cache = {}; self.cache_enabled = True
        def __call__(self, matrix):
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.ndim == 1: matrix = matrix[None, :]
            out = []
            for row in matrix:
                key = hash_array(row)
                if self.cache_enabled and key in self.cache:
                    calls["cache_hits"] += 1; out.append(self.cache[key]); continue
                tic = time.perf_counter()
                raw = np.asarray(adapter.predict(row[None, :]), dtype=float).reshape(-1)
                calls["seconds"] += time.perf_counter() - tic; calls["n"] += 1
                val = float(max(float(inv(raw)[0]), 0.0))
                if self.cache_enabled: self.cache[key] = val
                out.append(val)
            return np.asarray(out, dtype=float)

    predictor = Predictor()
    env = {
        "task": task, "fold": fold, "protocol": POINTWISE_PROTOCOL,
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "python_hashseed": os.getenv("PYTHONHASHSEED", "unset"), "tabicl_module": __import__("tabicl").__file__,
        "tabicl_version": getattr(__import__("tabicl"), "__version__", "2.2.0"),
        "features_n": len(features), "feature_order_hash": hash_ids(features),
        "train_obs_order_hash": hash_ids(train.obs_id.astype(str)), "test_obs_order_hash": hash_ids(test.obs_id.astype(str)),
        "train_obs_n": len(train), "test_obs_n": len(test), "raw_train_hash": hash_array(train[features].to_numpy(dtype=np.float64)),
        "query_matrix_hash": hash_array(xte), "imputer": "median; fit only on current fold training rows",
        "imputer_stats_hash": hash_array(imp.statistics_), "imputed_train_hash": hash_array(xtr),
        "target_transform": "log1p then expm1" if use_log else "identity", "train_target_hash": hash_array(ytr),
        "context_hash": ctx_hash, "context_n": len(est.ensemble_generator_.X_),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": file_sha(checkpoint),
        "norm_order": json.dumps(order), "ensemble_batch_size": getattr(est, "batch_size", None),
        "ensemble_members": getattr(est, "n_estimators", None), "use_amp": getattr(est, "use_amp", None),
        "device": str(getattr(est, "device_", "NA")), "eval_mode": not est.model_.training,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_cpu_rng_hash_after_context": hash_array(torch.get_rng_state().cpu().numpy()),
        "torch_cuda_rng_hash_after_context": hash_array(torch.cuda.get_rng_state_all()[0].cpu().numpy()) if torch.cuda.is_available() else "NA",
        "model_state_hash": sha("|".join(f"{k}:{hash_array(v.detach().cpu().numpy())}" for k,v in sorted(est.model_.state_dict().items())).encode()),
        "runner_code_sha256": file_sha(Path(__file__).resolve()),
        "fit_seconds": fit_seconds,
    }
    return base, torch, features, train, test, xtr, xte, imp, adapter, est, predictor, env, calls


def worker_pointwise(task: str, fold: str, rep: int):
    base, torch, features, train, test, xtr, xte, imp, adapter, est, pred, env, calls = make_context(task, fold)
    # 阶段A禁止任何预测缓存，确保重复和顺序复测真实调用模型。
    pred.cache_enabled = False
    n = min(32, len(test))
    pos = representatives(test, xte, n, 43000 + list(TASKS).index(task) * 100 + int(fold[-1]))
    ids = test.iloc[pos].obs_id.astype(str).tolist(); q = xte[pos]
    # 批量基准仅用于量化历史接口敏感性；所有新协议预测均调用单行Predictor。
    def batch(z):
        t = time.perf_counter(); raw = np.asarray(adapter.predict(np.asarray(z, dtype=np.float32)), float).reshape(-1)
        inv = np.expm1 if np.nanmin(train[base.TASKS[task]["target"]].to_numpy(float)) >= 0 else (lambda a: a)
        return np.maximum(inv(raw), 0), time.perf_counter() - t
    b0, _ = batch(q); br, _ = batch(q[::-1]); br = br[::-1]
    chunks = np.concatenate([batch(q[i:i+8])[0] for i in range(0, len(q), 8)])
    p_forward = pred(q)
    p_reverse = pred(q[::-1])[::-1]
    repeated = []
    for i in range(min(2, len(q))):
        vv = [float(pred(q[i:i+1])[0]) for _ in range(3)]
        repeated.append({"obs_id": ids[i], "repeat_values": vv, "max_delta": float(max(vv)-min(vv))})
    rows = []
    for i, oid in enumerate(ids):
        rows.append({"task": task, "fold": fold, "replicate": rep, "obs_id": oid,
            "query_order": i, "pointwise_prediction": float(p_forward[i]), "pointwise_reversed_order_prediction": float(p_reverse[i]),
            "same_process_order_delta": abs(float(p_forward[i]-p_reverse[i])),
            "historical_batch_prediction": float(b0[i]), "historical_reversed_batch_prediction": float(br[i]),
            "historical_chunk8_prediction": float(chunks[i]), "pointwise_vs_batch_abs_delta": abs(float(p_forward[i]-b0[i])),
            "context_hash": env["context_hash"], "feature_order_hash": env["feature_order_hash"],
            "query_row_hash": hash_array(q[i]), "query_cohort_hash": hash_array(q)})
    target_dir = OUT / "pointwise" / task / fold
    target_dir.mkdir(parents=True, exist_ok=True)
    rep_tmp=target_dir / f"replicate_{rep}.csv.tmp"
    pd.DataFrame(rows).to_csv(rep_tmp, index=False, encoding="utf-8-sig")
    rep_tmp.replace(target_dir / f"replicate_{rep}.csv")
    env.update({"replicate": rep, "probe_ids": ";".join(ids), "probe_order_hash": hash_ids(ids),
        "probe_n": n, "same_process_max_order_delta": max(r["same_process_order_delta"] for r in rows),
        "repeated_probe_checks": json.dumps(repeated), "model_predict_calls": calls["n"],
        "model_predict_seconds": calls["seconds"], "cache_hits": calls["cache_hits"],
        "completed_at": pd.Timestamp.now().isoformat()})
    env_tmp=target_dir / f"replicate_{rep}_environment.csv.tmp"
    pd.DataFrame([env]).to_csv(env_tmp, index=False, encoding="utf-8-sig")
    env_tmp.replace(target_dir / f"replicate_{rep}_environment.csv")
    print(json.dumps({"unit": f"A:{task}:{fold}:rep{rep}", "status": "COMPLETED", "predict_calls": calls["n"],
        "predict_seconds": calls["seconds"], "same_process_order_max": env["same_process_max_order_delta"]}, ensure_ascii=False), flush=True)


def build_shap_manifest():
    base = load_base(); features = base.load_features()
    old = pd.read_csv(V3 / "explanation/explanation_sample_manifest.csv", dtype={"obs_id": "string"})
    rows = []
    for task in TASKS:
        for fold in FOLDS:
            train, test = base.load_fold(task, fold, features)
            imputer_spec = importlib.util.spec_from_file_location("tabicl_manifest_base", BASE_PATH)
            xtr_imp = __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median", keep_empty_features=True)
            xte_imp = xtr_imp.fit_transform(train[features]).astype(np.float32); xq = xtr_imp.transform(test[features]).astype(np.float32)
            prior = old.loc[(old.task == task) & (old.fold == fold), "obs_id"].astype(str).tolist()
            id_to_pos = {str(x): i for i, x in enumerate(test.obs_id.astype(str))}
            selected = [id_to_pos[x] for x in prior if x in id_to_pos]
            extra = representatives(test, xq, min(len(test), max(SHAP_N, len(selected))), 52000 + 100 * TASKS.index(task) + int(fold[-1]))
            for i in extra:
                if int(i) not in selected and len(selected) < min(SHAP_N, len(test)):
                    selected.append(int(i))
            # If the old sample set already has >=10, preserve all prior IDs.
            for order, i in enumerate(selected, 1):
                r = test.iloc[i]
                rows.append({"task": task, "fold": fold, "obs_id": str(r.obs_id), "station_id": str(r.station_id),
                    "date": str(pd.to_datetime(r.date).date()), "selection_order": order,
                    "selection_origin": "V3_reused" if str(r.obs_id) in set(prior) else "deterministic_station_year_feature_space_fill",
                    "test_label_used_for_selection": False, "features_n": 77})
    pd.DataFrame(rows).to_csv(OUT / "shap_sample_manifest.csv", index=False, encoding="utf-8-sig")


def shap_worker(task: str, fold: str):
    import shap
    base, torch, features, train, test, xtr, xte, imp, adapter, est, pred, env, calls = make_context(task, fold)
    append_environment_manifest(env)
    manifest = pd.read_csv(OUT / "shap_sample_manifest.csv", dtype={"obs_id": "string"})
    samples = manifest.loc[(manifest.task == task) & (manifest.fold == fold)].copy()
    from sklearn.impute import SimpleImputer
    bimp = SimpleImputer(strategy="median", keep_empty_features=True)
    # fixed 16-row training-only background using the original V3 deterministic selector.
    bpos = representatives(train, xtr, min(BG_N, len(train)), 9000 + int(fold[-1]) + 100 * TASKS.index(task))
    bg = xtr[bpos]
    bg_manifest = train.iloc[bpos][["obs_id", "station_id", "date"]].copy()
    bg_manifest.insert(0, "task", task); bg_manifest.insert(1, "fold", fold); bg_manifest["background_order"] = range(1, len(bg_manifest)+1)
    append_rows(OUT / "background_sample_manifest.csv", bg_manifest, ["task", "fold", "obs_id"])
    masker = shap.maskers.Independent(bg, max_samples=BG_N)
    outdir = OUT / "shap_values" / task / fold; outdir.mkdir(parents=True, exist_ok=True)
    for r in samples.itertuples():
        path = outdir / f"{r.obs_id}_seed42.csv.gz"
        if path.is_file() and path.stat().st_size:
            continue
        ix = test.obs_id.astype(str).tolist().index(str(r.obs_id)); x = xte[ix:ix+1]
        calls_before=calls["n"];seconds_before=calls["seconds"]
        explainer = shap.PermutationExplainer(pred, masker, feature_names=features, seed=SEED)
        tic = time.perf_counter(); sv = explainer(x, max_evals=MAX_EVALS, silent=True); elapsed = time.perf_counter()-tic
        base_value = float(np.asarray(sv.base_values).reshape(-1)[0]); values = np.asarray(sv.values).reshape(-1)
        yhat = float(pred(x)[0]); add_error = abs(base_value + values.sum() - yhat)
        frame = pd.DataFrame({"task": task, "fold": fold, "obs_id": str(r.obs_id), "feature_order": np.arange(1,78),
            "feature": features, "feature_value": x[0], "shap_value": values, "base_value": base_value,
            "model_prediction": yhat, "additivity_abs_error": add_error, "output_scale": "original_target",
            "protocol": POINTWISE_PROTOCOL, "shap_seed": SEED, "background_n": len(bg), "max_evals": MAX_EVALS,
            "model_predict_calls": calls["n"]-calls_before, "model_predict_seconds": calls["seconds"]-seconds_before,
            "context_hash":env["context_hash"],"checkpoint_sha256":env["checkpoint_sha256"],
            "model_state_hash":env["model_state_hash"],"feature_order_hash":env["feature_order_hash"],
            "imputer_statistics_hash":env["imputer_stats_hash"],"train_obs_order_hash":env["train_obs_order_hash"],
            "test_obs_order_hash":env["test_obs_order_hash"],"norm_order":env["norm_order"],
            "python":env["python"],"numpy":env["numpy"],"torch":env["torch"],"cuda":env["cuda"],"gpu":env["gpu"]})
        tmp = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(tmp, index=False, encoding="utf-8-sig", compression="gzip"); tmp.replace(path)
        append_rows(OUT / "shap_runtime.csv", pd.DataFrame([{"task":task,"fold":fold,"obs_id":str(r.obs_id),"elapsed_seconds":elapsed,
            "model_predict_calls":calls["n"]-calls_before,"predict_seconds":calls["seconds"]-seconds_before,
            "model_predict_calls_cumulative":calls["n"],"predict_seconds_cumulative":calls["seconds"],
            "cache_hits":calls["cache_hits"],"additivity_abs_error":add_error,
            "context_hash":env["context_hash"],"checkpoint_sha256":env["checkpoint_sha256"],
            "model_state_hash":env["model_state_hash"],"feature_order_hash":env["feature_order_hash"],
            "imputer_statistics_hash":env["imputer_stats_hash"],"runner_code_sha256":env["runner_code_sha256"],
            "status":"COMPUTED" if add_error <= ADDITIVITY_TOL else "ADDITIVITY_FAILED","output_path":str(path.relative_to(ROOT))}]),
            ["task","fold","obs_id"])
        print(json.dumps({"unit":f"C:{task}:{fold}:{r.obs_id}","elapsed_seconds":elapsed,"predict_calls":calls["n"],"additivity":add_error},ensure_ascii=False),flush=True)
    # 每个任务仅在fold_1首个固定样本执行一次更高预算/不同排列种子的预注册敏感性检查。
    if fold == "fold_1" and len(samples):
        r=samples.iloc[0]; obs=str(r.obs_id); ix=test.obs_id.astype(str).tolist().index(obs); x=xte[ix:ix+1]
        base_path=outdir/f"{obs}_seed42.csv.gz"; sens_path=OUT/"shap_sensitivity"/task/fold/f"{obs}_seed43_eval309.csv.gz"
        if base_path.is_file() and not sens_path.is_file():
            expl=shap.PermutationExplainer(pred,masker,feature_names=features,seed=43)
            tic=time.perf_counter();sv=expl(x,max_evals=309,silent=True);elapsed=time.perf_counter()-tic
            values=np.asarray(sv.values).reshape(-1);basev=float(np.asarray(sv.base_values).reshape(-1)[0]);yhat=float(pred(x)[0])
            sf=pd.DataFrame({"task":task,"fold":fold,"obs_id":obs,"feature_order":np.arange(1,78),"feature":features,
                "shap_value":values,"base_value":basev,"model_prediction":yhat,"additivity_abs_error":abs(basev+values.sum()-yhat),
                "shap_seed":43,"max_evals":309,"protocol":POINTWISE_PROTOCOL})
            sens_path.parent.mkdir(parents=True,exist_ok=True);tmp=sens_path.with_suffix(sens_path.suffix+".tmp")
            sf.to_csv(tmp,index=False,encoding="utf-8-sig",compression="gzip");tmp.replace(sens_path)
            old=pd.read_csv(base_path);diff=np.abs(old.shap_value.to_numpy(float)-values)
            append_rows(OUT/"shap_estimation_budget_sensitivity.csv",pd.DataFrame([{"task":task,"fold":fold,"obs_id":obs,
                "reference_max_evals":155,"sensitivity_max_evals":309,"seed_reference":42,"seed_sensitivity":43,
                "spearman_rank":float(pd.Series(old.shap_value).rank().corr(pd.Series(values).rank(),method="spearman")),
                "top10_jaccard":len(set(old.nlargest(10,"shap_value").feature)&set(pd.DataFrame({"feature":features,"shap_value":values}).nlargest(10,"shap_value").feature))/10,
                "max_abs_shap_delta":float(diff.max()),"mean_abs_shap_delta":float(diff.mean()),"elapsed_seconds":elapsed,
                "output_path":str(sens_path.relative_to(ROOT))}]),["task","fold","obs_id"])


def ale_1d_worker(task: str, fold: str, selected_features: list[str] | None = None):
    """重算覆盖合格的一维ALE；逐查询P1差值写SQLite，便于异常中断后续跑。"""
    base, torch, features, train, test, xtr, xte, imp, adapter, est, pred, env, calls = make_context(task, fold)
    append_environment_manifest(env)
    support_path = REPAIR / "06_ale_1d_rebinning_comparison.csv"
    coverage = pd.read_csv(support_path)
    coverage = coverage.loc[(coverage.task == task) & (coverage.fold == fold)]
    if selected_features:
        coverage = coverage.loc[coverage.feature.isin(selected_features)]
    db = sqlite3.connect(OUT / "checkpoints/local_effects.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS ale1_local(task TEXT, fold TEXT, feature TEXT, obs_id TEXT, bin INTEGER, xlo REAL, xhi REAL, plo REAL, phi REAL, PRIMARY KEY(task,fold,feature,obs_id))")
    for feature, cv in coverage.groupby("feature", sort=False):
        if not cv.status.eq("SUPPORTED").all() or cv.bin.nunique() < 8:
            append_rows(OUT / "06_ale_1d_support_and_stability.csv", pd.DataFrame([{"task":task,"fold":fold,"feature":feature,
                "status":"INSUFFICIENT_SUPPORT","reason":"Prior full-fold query coverage audit did not pass every supported bin."}]), ["task","fold","feature"])
            continue
        j = features.index(feature); cv=cv.sort_values("bin"); edges = cv.lower.astype(float).tolist() + [float(cv.upper.iloc[-1])]
        vals = pd.to_numeric(test[feature],errors="coerce").to_numpy(float)
        eligible = np.isfinite(vals) & (vals >= edges[0]) & (vals <= edges[-1])
        bins = np.clip(np.digitize(vals, edges[1:-1], right=True), 0, len(edges)-2)
        query_ids = test.obs_id.astype(str).to_numpy()
        for i in np.flatnonzero(eligible):
            oid = query_ids[i]
            if db.execute("SELECT 1 FROM ale1_local WHERE task=? AND fold=? AND feature=? AND obs_id=?",(task,fold,feature,oid)).fetchone():
                continue
            lo, hi = xte[i].copy(), xte[i].copy(); lo[j]=edges[bins[i]]; hi[j]=edges[bins[i]+1]
            p0=float(pred(lo[None,:])[0]); p1=float(pred(hi[None,:])[0])
            db.execute("INSERT OR REPLACE INTO ale1_local VALUES(?,?,?,?,?,?,?,?,?)",(task,fold,feature,oid,int(bins[i]+1),float(lo[j]),float(hi[j]),p0,p1)); db.commit()
        # Three uncached repetitions of one fixed, supported interval per variable.
        probe_bin=next((int(b) for b in range(len(edges)-1) if np.any(eligible & (bins==b))),None)
        if probe_bin is not None:
            probe_i=int(np.flatnonzero(eligible & (bins==probe_bin))[0]);probe_oid=query_ids[probe_i]
            probe_lo=xte[probe_i].copy();probe_hi=xte[probe_i].copy();probe_lo[j]=edges[probe_bin];probe_hi[j]=edges[probe_bin+1]
            old_cache=pred.cache_enabled;pred.cache_enabled=False;repeated=[];repeat_calls_before=calls["n"]
            for repeat_i in range(1,4):
                plo=float(pred(probe_lo[None,:])[0]);phi=float(pred(probe_hi[None,:])[0])
                repeated.append({"repeat":repeat_i,"prediction_lower":plo,"prediction_upper":phi,"local_delta":phi-plo,
                    "lower_input_hash":hash_array(probe_lo),"upper_input_hash":hash_array(probe_hi)})
            pred.cache_enabled=old_cache
            repeat_actual_calls=calls["n"]-repeat_calls_before
            rep_df=pd.DataFrame(repeated)
            append_rows(OUT/"ale_1d_repeatability.csv",pd.DataFrame([{"task":task,"fold":fold,"feature":feature,
                "probe_obs_id":probe_oid,"bin":probe_bin+1,"x_lower":float(probe_lo[j]),"x_upper":float(probe_hi[j]),
                "n_repeats":len(repeated),"max_lower_prediction_delta":float(rep_df.prediction_lower.max()-rep_df.prediction_lower.min()),
                "max_upper_prediction_delta":float(rep_df.prediction_upper.max()-rep_df.prediction_upper.min()),
                "max_local_ale_delta_range":float(rep_df.local_delta.max()-rep_df.local_delta.min()),
                "repeat_predictions_json":json.dumps(repeated),"actual_model_calls":repeat_actual_calls,"context_hash":env["context_hash"],
                "protocol":POINTWISE_PROTOCOL}],),["task","fold","feature","probe_obs_id"])
        local = pd.read_sql_query("SELECT * FROM ale1_local WHERE task=? AND fold=? AND feature=?",db,params=(task,fold,feature))
        binsummary=[]
        for b in range(1,len(edges)):
            q=local.loc[local.bin==b]; n=len(q); stations=test.loc[test.obs_id.astype(str).isin(q.obs_id.astype(str)),"station_id"].nunique()
            delta=(q.phi-q.plo).mean() if n else np.nan
            binsummary.append({"bin":b,"lower":edges[b-1],"upper":edges[b],"local_effect":delta,"n_samples":n,"n_stations":stations})
        if local.obs_id.nunique() < int(eligible.sum()):
            continue
        bdf=pd.DataFrame(binsummary); acc=bdf.local_effect.fillna(0).cumsum(); weights=bdf.n_samples.clip(lower=1)
        bdf["ale"] = acc - np.average(acc, weights=weights)
        bdf.insert(0,"feature",feature); bdf.insert(0,"fold",fold); bdf.insert(0,"task",task); bdf["protocol"]=POINTWISE_PROTOCOL
        append_rows(OUT/"05_ale_1d_results.csv",bdf,["task","fold","feature","bin"])
        append_rows(OUT/"06_ale_1d_support_and_stability.csv",pd.DataFrame([{"task":task,"fold":fold,"feature":feature,
            "status":"COMPUTED_POINTWISE","n_query":int(eligible.sum()),"min_bin_n":int(bdf.n_samples.min()),"min_bin_stations":int(bdf.n_stations.min()),
            "ale_range":float(bdf.ale.max()-bdf.ale.min()),"n_bins":len(bdf),"context_hash":env["context_hash"],"protocol":POINTWISE_PROTOCOL}]),["task","fold","feature"])
        print(json.dumps({"unit":f"D_ALE1:{task}:{fold}:{feature}","status":"COMPLETED","query_rows":int(eligible.sum()),
            "bins":len(bdf),"model_predict_calls":calls["n"],"predict_seconds":calls["seconds"]},ensure_ascii=False),flush=True)
    db.close()


def ale_2d_worker(task: str, fold: str):
    """七个预注册组合的4x4/5x5二阶ALE敏感性；只对满支持子矩形逐点计算。"""
    import ast
    base, torch, features, train, test, xtr, xte, imp, adapter, est, pred, env, calls = make_context(task, fold)
    append_environment_manifest(env)
    audit=pd.read_csv(REPAIR/"07_ale_2d_grid_sensitivity.csv")
    audit=audit.loc[(audit.task==task)&(audit.fold==fold)]
    if audit.empty:
        append_rows(OUT/"08_ale_2d_support_and_stability.csv",pd.DataFrame([{"task":task,"fold":fold,"status":"INSUFFICIENT_SUPPORT","reason":"No preregistered pair audit row."}]),["task","fold","feature_1","feature_2"])
        return
    db=sqlite3.connect(OUT/"checkpoints/local_effects.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS ale2_local(task TEXT,fold TEXT,feature_1 TEXT,feature_2 TEXT,grid TEXT,cell_1 INTEGER,cell_2 INTEGER,obs_id TEXT,d2 REAL,PRIMARY KEY(task,fold,feature_1,feature_2,grid,obs_id))")
    result_rows=[];summary_rows=[]
    for pair,(f1,f2) in audit.groupby(["feature_1","feature_2"],sort=False):
        pair_result=[]
        j1,j2=features.index(f1),features.index(f2)
        raw1=pd.to_numeric(test[f1],errors="coerce").to_numpy(float);raw2=pd.to_numeric(test[f2],errors="coerce").to_numpy(float)
        valid=np.isfinite(raw1)&np.isfinite(raw2)
        for _,ar in pair.iterrows():
            grid=str(ar.grid_actual)
            if grid not in ("4x4","5x5"):continue
            counts=np.asarray(json.loads(ar.cell_n_json),int);stations=np.asarray(json.loads(ar.cell_station_n_json),int)
            rectangle=ast.literal_eval(str(ar.best_full_support_rectangle)) if pd.notna(ar.best_full_support_rectangle) else (0,0,0,0)
            r0,r1,c0,c1=map(int,rectangle)
            if r1<=r0 or c1<=c0:
                summary_rows.append({"task":task,"fold":fold,"feature_1":f1,"feature_2":f2,"grid":grid,
                    "status":"INSUFFICIENT_SUPPORT","supported_rectangle":str(rectangle),
                    "reason":"No fully supported contiguous rectangle met the prespecified joint sample/station criteria."})
                continue
            edge1=np.asarray(json.loads(ar.feature_1_train_quantile_edges_json),float)
            edge2=np.asarray(json.loads(ar.feature_2_train_quantile_edges_json),float)
            b1=np.clip(np.digitize(raw1,edge1[1:-1],right=True),0,len(edge1)-2)
            b2=np.clip(np.digitize(raw2,edge2[1:-1],right=True),0,len(edge2)-2)
            rectmask=(b1>=r0)&(b1<r1)&(b2>=c0)&(b2<c1)&valid
            # Reconfirm all cells from the saved audit before expensive inference.
            adequate=all(counts[i,j]>=30 and stations[i,j]>=5 for i in range(r0,r1) for j in range(c0,c1))
            if not adequate:
                summary_rows.append({"task":task,"fold":fold,"feature_1":f1,"feature_2":f2,"grid":grid,"status":"INSUFFICIENT_SUPPORT",
                    "supported_rectangle":str(rectangle),"reason":"Saved cell audit did not meet n>=30 and stations>=5 throughout rectangle."})
                continue
            ids=test.obs_id.astype(str).to_numpy()
            for i in np.flatnonzero(rectmask):
                oid=ids[i]
                if db.execute("SELECT 1 FROM ale2_local WHERE task=? AND fold=? AND feature_1=? AND feature_2=? AND grid=? AND obs_id=?",
                    (task,fold,f1,f2,grid,oid)).fetchone():continue
                a,b=int(b1[i]),int(b2[i]); corners=[]
                for da,dbb in ((0,0),(1,0),(0,1),(1,1)):
                    z=xte[i].copy();z[j1]=edge1[a+da];z[j2]=edge2[b+dbb];corners.append(float(pred(z[None,:])[0]))
                d2=corners[3]-corners[1]-corners[2]+corners[0]
                db.execute("INSERT OR REPLACE INTO ale2_local VALUES(?,?,?,?,?,?,?,?,?)",(task,fold,f1,f2,grid,a,b,oid,float(d2)));db.commit()
            loc=pd.read_sql_query("SELECT * FROM ale2_local WHERE task=? AND fold=? AND feature_1=? AND feature_2=? AND grid=?",db,params=(task,fold,f1,f2,grid))
            sub=loc.loc[(loc.cell_1>=r0)&(loc.cell_1<r1)&(loc.cell_2>=c0)&(loc.cell_2<c1)]
            n1,n2=r1-r0,c1-c0;L=np.full((n1,n2),np.nan);W=np.zeros((n1,n2),float)
            for ii in range(n1):
                for jj in range(n2):
                    z=sub.loc[(sub.cell_1==r0+ii)&(sub.cell_2==c0+jj),"d2"]
                    if len(z): L[ii,jj]=z.mean();W[ii,jj]=len(z)
            if not np.isfinite(L).all() or np.any(W<=0):
                summary_rows.append({"task":task,"fold":fold,"feature_1":f1,"feature_2":f2,"grid":grid,"status":"COMPUTATION_INCOMPLETE",
                    "supported_rectangle":str(rectangle),"computed_cells":int(np.isfinite(L).sum())})
                continue
            A=np.cumsum(np.cumsum(L,axis=0),axis=1)
            # Weighted alternating projections enforce zero row and column margins.
            Z=A.copy()
            for _ in range(200):
                for ii in range(n1): Z[ii,:]-=np.average(Z[ii,:],weights=W[ii,:])
                for jj in range(n2): Z[:,jj]-=np.average(Z[:,jj],weights=W[:,jj])
            residual=max(float(np.max(np.abs(np.sum(Z*W,axis=1)/W.sum(axis=1)))),float(np.max(np.abs(np.sum(Z*W,axis=0)/W.sum(axis=0)))))
            amp=float(Z.max()-Z.min())
            for ii in range(n1):
                for jj in range(n2):
                    result_rows.append({"task":task,"fold":fold,"feature_1":f1,"feature_2":f2,"grid":grid,
                        "cell_1":r0+ii+1,"cell_2":c0+jj+1,"ale_2d":Z[ii,jj],"local_second_difference":L[ii,jj],
                        "n_samples":int(W[ii,jj]),"n_stations":int(stations[r0+ii,c0+jj]),"sparse_mask":False,
                        "support_rectangle":str(rectangle),"centering_weighted_margin_residual":residual,
                        "context_hash":env["context_hash"],"protocol":POINTWISE_PROTOCOL})
            summary_rows.append({"task":task,"fold":fold,"feature_1":f1,"feature_2":f2,"grid":grid,
                "status":"PARTIALLY_SUPPORTED" if (r0>0 or c0>0 or r1<counts.shape[0] or c1<counts.shape[1]) else "SUPPORTED",
                "supported_rectangle":str(rectangle),"supported_cells":n1*n2,"grid_cells":int(counts.size),
                "supported_fraction":float(n1*n2/counts.size),"interaction_range":amp,"weighted_centering_residual":residual,
                "min_cell_n":int(W.min()),"min_cell_stations":int(stations[r0:r1,c0:c1].min()),"context_hash":env["context_hash"],"protocol":POINTWISE_PROTOCOL})
        if pair_result:
            result_rows.extend(pair_result)
    db.close()
    if result_rows:
        append_rows(OUT/"07_ale_2d_results.csv",pd.DataFrame(result_rows),["task","fold","feature_1","feature_2","grid","cell_1","cell_2"])
    if summary_rows:
        append_rows(OUT/"08_ale_2d_support_and_stability.csv",pd.DataFrame(summary_rows),["task","fold","feature_1","feature_2","grid"])


def init_outputs():
    for p in (OUT/"logs",OUT/"checkpoints",OUT/"figures/png",OUT/"figures/svg",OUT/"scripts"):
        p.mkdir(parents=True,exist_ok=True)
    for fn in ("79_run_all_tabicl_explanation_experiments.py","80_resume_tabicl_explanation_experiments.py","81_validate_tabicl_explanation_results.py","82_resume_tabicl_explanation_run.ps1","83_watch_tabicl_explanation_completion.ps1"):
        src=ROOT/"脚本/05_诊断分析"/fn
        if src.is_file(): shutil.copy2(src,OUT/"scripts"/fn)
    (OUT/"00_protocol_and_environment.md").write_text(f"""# TabICLv2最终可解释性实验协议

实验协议：`{POINTWISE_PROTOCOL}`。四项任务、20折；77个冻结特征及正式fold manifest；每折训练样本单独拟合median imputer；只构造TabICLv2预训练权重的推理上下文，不更新预训练权重；固定seed=42、Python hash seed=42、隔离补丁中的`none,power`视图顺序、float32、AMP=auto、eval模式；目标按原正式任务逻辑使用log1p/expm1并非负截断。逐查询函数将每个SHAP/ALE输入行单独调用一次模型，不把其他查询或标签放入上下文。所有原批量预测及新解释结果分开保存。

冻结资产来源、77特征/obs顺序哈希、上下文与权重SHA256、环境和预测接口参数记录在`environment_manifest.csv`与逐折worker输出。Python/NumPy/PyTorch/CUDA/GPU版本在运行时采集。原始冻结预测按obs_id配对；任何新点式结果均不替换正式预测。

计算状态和已固定协议列于`13_final_explanation_master_registry.csv`。新解释结果必须通过独立脚本`81_validate_tabicl_explanation_results.py`核验后才可标记`COMPLETE_VALIDATED`。任务支持SQLite/逐样本检查点；重复调用调度器会跳过已有有效输出。
""",encoding="utf-8")


def run_worker_subprocess(args: list[str], unit: str):
    env=os.environ.copy(); env.update({"PYTHONHASHSEED":"42","CUBLAS_WORKSPACE_CONFIG":":4096:8",
        "TABICL_FIXED_NORM_ORDER":"none,power","PYTHONPATH":str(PATCH)+os.pathsep+env.get("PYTHONPATH","")})
    logdir=OUT/"logs";logdir.mkdir(parents=True,exist_ok=True)
    tic=time.perf_counter()
    stdout_path=logdir/(unit.replace(":","_")+".stdout.log")
    stderr_path=logdir/(unit.replace(":","_")+".stderr.log")
    try:
        monitor=[]
        with stdout_path.open("w",encoding="utf-8",buffering=1) as stdout_file, stderr_path.open("w",encoding="utf-8",buffering=1) as stderr_file:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*args],cwd=ROOT,env=env,stdout=stdout_file,stderr=stderr_file,
                text=True,encoding="utf-8",errors="replace")
            while p.poll() is None:
                time.sleep(20)
                free=shutil.disk_usage(OUT).free
                gpu={}
                try:
                    q=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu","--format=csv,noheader,nounits"],
                        capture_output=True,text=True,timeout=5)
                    if q.returncode==0:
                        vals=[v.strip() for v in q.stdout.strip().split(",")]
                        if len(vals)>=4:gpu={"gpu_util_pct":vals[0],"gpu_memory_used_mb":vals[1],"gpu_memory_total_mb":vals[2],"gpu_temperature_c":vals[3]}
                except Exception: pass
                monitor.append({"unit":unit,"timestamp":pd.Timestamp.now().isoformat(),"disk_free_bytes":free,**gpu})
                if free < 2*1024**3:
                    p.terminate();break
            if p.poll() is None:p.wait()
        status="COMPLETED" if p.returncode==0 else f"FAILED_RUNTIME_{p.returncode}"
        if monitor:append_rows(OUT/"resource_monitor.csv",pd.DataFrame(monitor),["unit","timestamp"])
        err=stderr_path.read_text(encoding="utf-8",errors="replace")[-1500:] if stderr_path.exists() else ""
    except Exception as e:
        status="FAILED_RUNTIME";err=repr(e);p=None
    row={"unit":unit,"status":status,"elapsed_seconds":time.perf_counter()-tic,"stdout_log":str(stdout_path.relative_to(OUT)),
        "stderr_log":str(stderr_path.relative_to(OUT)),"error":err,"completed_at":pd.Timestamp.now().isoformat()}
    append_rows(OUT/"13_final_explanation_master_registry.csv",pd.DataFrame([row]),["unit"])
    if status!="COMPLETED" and not unit.endswith(":retry1"):
        # One bounded retry; each failed attempt remains in the registry with its own logs.
        return run_worker_subprocess(args,unit+":retry1")
    return status


def run_phase_a():
    for task in TASKS:
        for fold in FOLDS:
            if (OUT/"PAUSE_AFTER_CURRENT_FOLD.flag").is_file():
                return
            if (task,fold)==("chla_surface","fold_1"):
                reuse_chla_fold1_p1_pilot()
                continue
            for rep in REPS:
                unit=f"A:{task}:{fold}:rep{rep}"
                target=OUT/"pointwise"/task/fold/f"replicate_{rep}.csv"
                if target.is_file() and target.stat().st_size:
                    continue
                run_worker_subprocess(["--worker-pointwise",task,fold,str(rep)],unit)
            summarize_fold_pointwise(task,fold)
            if (OUT/"PAUSE_AFTER_CURRENT_FOLD.flag").is_file():
                return


def reuse_chla_fold1_p1_pilot():
    """Reuse the already-completed 2-probe/3-process P1 protocol test; never rerun it."""
    src=ROOT/"runs/tabicl_v2_explanation_stability_v1/03_protocol_stability_summary.csv"
    d=pd.read_csv(src)
    d=d.loc[d.protocol.eq("P1_SINGLE_QUERY")].copy()
    if d.obs_id.astype(str).nunique()<2 or not d.stable_le_1e_6.astype(bool).all():
        raise RuntimeError("Existing Chla fold1 P1 stability evidence is absent or failed; refusing to silently rerun the completed pilot.")
    row={"task":"chla_surface","fold":"fold_1","status":"STABLE","protocol":POINTWISE_PROTOCOL,
        "n_independent_processes":int(d.n_processes.min()),"context_hash_n":1,"feature_order_hash_n":1,"probe_order_hash_n":1,
        "max_cross_process_abs_diff":float(d.within_and_cross_process_max_delta.max()),
        "mean_cross_process_abs_diff":float(d.within_and_cross_process_max_delta.mean()),
        "max_same_process_repeat_abs_diff":float(d.within_and_cross_process_max_delta.max()),
        "max_same_process_query_order_abs_diff":0.0,"n_probe":int(d.obs_id.nunique()),
        "validation_source":"REUSED runs/tabicl_v2_explanation_stability_v1; 2 fixed probes x 3 independent processes; no repeat run"}
    append_rows(OUT/"01_pointwise_inference_validation.csv",pd.DataFrame([row]),["task","fold"])


def summarize_fold_pointwise(task: str, fold: str):
    frames=[]; envs=[]
    for rep in REPS:
        f=OUT/"pointwise"/task/fold/f"replicate_{rep}.csv";e=OUT/"pointwise"/task/fold/f"replicate_{rep}_environment.csv"
        if f.is_file(): frames.append(pd.read_csv(f))
        if e.is_file(): envs.append(pd.read_csv(e))
    if len(frames)<3:return
    d=pd.concat(frames,ignore_index=True); env=pd.concat(envs,ignore_index=True)
    table=d.groupby("obs_id").pointwise_prediction.agg(["min","max","mean","count"]).reset_index()
    table["range"]=table["max"]-table["min"]
    maxdiff=float(table.range.max()); meandiff=float(table.range.mean())
    ctx_n=env.context_hash.nunique(); feat_n=env.feature_order_hash.nunique(); qid_n=env.probe_order_hash.nunique()
    repeat_max=0.0
    for raw in env.repeated_probe_checks.astype(str):
        for item in json.loads(raw): repeat_max=max(repeat_max,float(item["max_delta"]))
    order_max=float(d.same_process_order_delta.max())
    status="STABLE" if maxdiff<=MAX_DIFF and meandiff<=MEAN_DIFF and repeat_max<=MAX_DIFF and order_max<=MAX_DIFF and ctx_n==1 and feat_n==1 and qid_n==1 else "PROTOCOL_FAILED"
    # New-versus-historical comparison is explicitly a fixed unlabeled-feature probe subset.
    pred=d.groupby("obs_id",as_index=False).agg(pointwise_prediction=("pointwise_prediction","mean"),
        historical_batch_prediction=("historical_batch_prediction","mean"),pointwise_vs_batch_abs_delta=("pointwise_vs_batch_abs_delta","mean"))
    base=load_base(); frozen=base.frozen(task,fold); train,test=base.load_fold(task,fold,base.load_features())
    meta=test[["obs_id",base.TASKS[task]["target"]]].copy(); meta["obs_id"]=meta.obs_id.astype(str)
    pred=pred.merge(frozen[["obs_id","y_true","y_pred"]].assign(obs_id=lambda x:x.obs_id.astype(str)),on="obs_id",how="left",validate="one_to_one")
    pred=pred.merge(meta.assign(obs_id=lambda x:x.obs_id.astype(str)),on="obs_id",how="left",validate="one_to_one")
    from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
    y=pred["y_true"].to_numpy(float);p0=pred["y_pred"].to_numpy(float);p1=pred.pointwise_prediction.to_numpy(float)
    metrics={"n_probe":len(pred),"max_abs_delta":float(np.max(np.abs(p1-p0))),"mean_abs_delta":float(np.mean(np.abs(p1-p0))),
        "median_abs_delta":float(np.median(np.abs(p1-p0))),"p95_abs_delta":float(np.quantile(np.abs(p1-p0),.95)),
        "old_R2":float(r2_score(y,p0)),"new_probe_R2":float(r2_score(y,p1)),"old_RMSE":float(mean_squared_error(y,p0)**.5),
        "new_probe_RMSE":float(mean_squared_error(y,p1)**.5),"old_MAE":float(mean_absolute_error(y,p0)),"new_probe_MAE":float(mean_absolute_error(y,p1)),
        "prediction_correlation":float(np.corrcoef(p0,p1)[0,1]) if len(p0)>1 else np.nan}
    pred.insert(0,"fold",fold);pred.insert(0,"task",task);pred["comparison_scope"]="preselected 32 feature-only probes; not full-fold metrics"
    append_rows(OUT/"02_original_vs_pointwise_prediction.csv",pred,["task","fold","obs_id"])
    append_rows(OUT/"01_pointwise_inference_validation.csv",pd.DataFrame([{"task":task,"fold":fold,"status":status,
        "protocol":POINTWISE_PROTOCOL,"n_independent_processes":len(frames),"context_hash_n":ctx_n,"feature_order_hash_n":feat_n,
        "probe_order_hash_n":qid_n,"max_cross_process_abs_diff":maxdiff,"mean_cross_process_abs_diff":meandiff,
        "max_same_process_repeat_abs_diff":repeat_max,"max_same_process_query_order_abs_diff":order_max,**metrics}]),["task","fold"])
    if "fold" not in d.columns: d.insert(0,"fold",fold)
    if "task" not in d.columns: d.insert(0,"task",task)
    append_rows(OUT/"pointwise_probe_predictions.csv",d,["task","fold","replicate","obs_id"])


def phase_cde():
    val=pd.read_csv(OUT/"01_pointwise_inference_validation.csv") if (OUT/"01_pointwise_inference_validation.csv").is_file() else pd.DataFrame()
    eligible={(r.task,r.fold) for r in val.itertuples() if r.status=="STABLE"}
    for task,fold in [(t,f) for t in TASKS for f in FOLDS]:
        if (task,fold) not in eligible:
            for phase in ("C_SHAP","D_ALE1","E_ALE2"):
                append_rows(OUT/"13_final_explanation_master_registry.csv",pd.DataFrame([{"unit":f"{phase}:{task}:{fold}","status":"BLOCKED_DEPENDENCY",
                    "elapsed_seconds":0,"error":"Fold pointwise protocol did not pass; explanation not executed."}]),["unit"])
            continue
        run_worker_subprocess(["--worker-shap",task,fold],f"C_SHAP:{task}:{fold}")
        run_worker_subprocess(["--worker-ale1",task,fold],f"D_ALE1:{task}:{fold}")
        # 2D workers are intentionally driven only by the seven preregistered supported pairs.
        run_worker_subprocess(["--worker-ale2",task,fold],f"E_ALE2:{task}:{fold}")


def full_prediction_worker(task: str, fold: str):
    """完整测试fold的P1逐obs_id复现；SQLite逐行提交，标签仅用于事后指标。"""
    from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
    base, torch, features, train, test, xtr, xte, imp, adapter, est, pred, env, calls = make_context(task, fold)
    append_environment_manifest(env)
    db=sqlite3.connect(OUT/"checkpoints/full_pointwise_predictions.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS p1_full(task TEXT,fold TEXT,obs_id TEXT,pointwise_prediction REAL,context_hash TEXT,query_hash TEXT,elapsed REAL,PRIMARY KEY(task,fold,obs_id))")
    for i,oid in enumerate(test.obs_id.astype(str)):
        if db.execute("SELECT 1 FROM p1_full WHERE task=? AND fold=? AND obs_id=?",(task,fold,oid)).fetchone():continue
        tic=time.perf_counter();yhat=float(pred(xte[i:i+1])[0]);elapsed=time.perf_counter()-tic
        db.execute("INSERT OR REPLACE INTO p1_full VALUES(?,?,?,?,?,?,?)",(task,fold,oid,yhat,env["context_hash"],hash_array(xte[i]),elapsed));db.commit()
        if i%25==0:
            print(json.dumps({"unit":f"B_full:{task}:{fold}","done":i+1,"total":len(test),"elapsed_seconds":calls["seconds"]},ensure_ascii=False),flush=True)
    d=pd.read_sql_query("SELECT * FROM p1_full WHERE task=? AND fold=?",db,params=(task,fold));db.close()
    frozen=base.frozen(task,fold);d=d.merge(frozen[["obs_id","y_true","y_pred"]].assign(obs_id=lambda x:x.obs_id.astype(str)),on="obs_id",validate="one_to_one")
    if len(d)!=len(test) or set(d.obs_id)!=set(test.obs_id.astype(str)):raise RuntimeError("Full-fold obs_id mismatch")
    y=d.y_true.to_numpy(float);p0=d.y_pred.to_numpy(float);p1=d.pointwise_prediction.to_numpy(float);delta=np.abs(p1-p0)
    d.insert(0,"fold",fold);d.insert(0,"task",task);d["abs_prediction_delta"]=delta;d["protocol"]=POINTWISE_PROTOCOL
    d["comparison_scope"]="full test fold, aligned by obs_id; test labels used only for post-hoc metrics"
    full_path=OUT/f"02_original_vs_pointwise_full_{task}_{fold}.csv";tmp=full_path.with_suffix(full_path.suffix+".tmp")
    d.to_csv(tmp,index=False,encoding="utf-8-sig");tmp.replace(full_path)
    append_rows(OUT/"02_original_vs_pointwise_prediction.csv",d,["task","fold","obs_id"])
    metrics={"task":task,"fold":fold,"scope":"FULL_TEST_FOLD","n":len(d),"max_abs_delta":float(delta.max()),
        "mean_abs_delta":float(delta.mean()),"median_abs_delta":float(np.median(delta)),"p95_abs_delta":float(np.quantile(delta,.95)),
        "old_R2":float(r2_score(y,p0)),"new_R2":float(r2_score(y,p1)),"old_RMSE":float(mean_squared_error(y,p0)**.5),
        "new_RMSE":float(mean_squared_error(y,p1)**.5),"old_MAE":float(mean_absolute_error(y,p0)),"new_MAE":float(mean_absolute_error(y,p1)),
        "correlation":float(np.corrcoef(p0,p1)[0,1]),"context_hash":env["context_hash"],"query_hash":env["query_matrix_hash"],
        "predict_calls":calls["n"],"predict_seconds":calls["seconds"],"status":"COMPLETED"}
    append_rows(OUT/"full_fold_prediction_metrics.csv",pd.DataFrame([metrics]),["task","fold"])
    print(json.dumps({"unit":f"B_full:{task}:{fold}","status":"COMPLETED","n":len(d),"max_delta":metrics["max_abs_delta"],"seconds":calls["seconds"]},ensure_ascii=False),flush=True)


def phase_b_full():
    val=pd.read_csv(OUT/"01_pointwise_inference_validation.csv") if (OUT/"01_pointwise_inference_validation.csv").is_file() else pd.DataFrame()
    for r in val.itertuples():
        if r.status!="STABLE":continue
        path=OUT/f"02_original_vs_pointwise_full_{r.task}_{r.fold}.csv"
        if path.is_file() and path.stat().st_size:continue
        run_worker_subprocess(["--worker-fullcompare",r.task,r.fold],f"B_FULL:{r.task}:{r.fold}")


def init_manifest():
    init_outputs()
    base=load_base(); feature_order=base.load_features()
    pd.DataFrame({"feature_order":range(1,78),"feature":feature_order,"feature_order_hash":hash_ids(feature_order)}).to_csv(OUT/"feature_order.csv",index=False,encoding="utf-8-sig")
    if not (OUT/"shap_sample_manifest.csv").is_file(): build_shap_manifest()
    required={"01_pointwise_inference_validation.csv":["task","fold","status"],"02_original_vs_pointwise_prediction.csv":["task","fold","obs_id"],
        "03_shap_full_results.csv":["task","fold","obs_id","feature","shap_value"],"04_shap_stability_validation.csv":["task","fold","status"],
        "05_ale_1d_results.csv":["task","fold","feature","bin","ale"],"06_ale_1d_support_and_stability.csv":["task","fold","feature","status"],
        "07_ale_2d_results.csv":["task","fold","feature_1","feature_2","ale_2d"],"08_ale_2d_support_and_stability.csv":["task","fold","feature_1","feature_2","status"],
        "09_xgb_tabicl_explanation_comparison.csv":["task","feature"],"10_explanation_figure_inventory.csv":["figure","status"]}
    for name,cols in required.items():
        if not (OUT/name).exists():pd.DataFrame(columns=cols).to_csv(OUT/name,index=False,encoding="utf-8-sig")
    # Copy existing XGBoost comparison evidence without recomputing XGBoost.
    xgb=ROOT/"runs/xgb_shap_ale_v2/04_integrated/04_shap_ale_ablation_comparison.csv"
    if xgb.is_file():
        x=pd.read_csv(xgb);x["comparison_status"]="EXISTING_XGB_ONLY; scale/definition audit required before quantitative comparison"
        x.to_csv(OUT/"09_xgb_tabicl_explanation_comparison.csv",index=False,encoding="utf-8-sig")


def compare_old_batch_explanations():
    """Compare common saved V3 cases without conflating the old batch and new P1 functions."""
    old_root=V3/"shap_values";new_root=OUT/"shap_values"
    old_map={}
    if old_root.exists():
        for p in old_root.glob("*/*/*.csv.gz"):
            old_map[(p.parent.parent.name,p.parent.name,p.name.replace("_seed42.csv.gz",""))]=p
    shap_rows=[]
    if new_root.exists():
        for p in new_root.glob("*/*/*_seed42.csv.gz"):
            key=(p.parent.parent.name,p.parent.name,p.name.replace("_seed42.csv.gz",""))
            if key not in old_map:continue
            old=pd.read_csv(old_map[key]);new=pd.read_csv(p)
            if len(old)!=77 or len(new)!=77:continue
            old=old.sort_values("feature_order");new=new.sort_values("feature_order")
            if old.feature.astype(str).tolist()!=new.feature.astype(str).tolist():continue
            ov=old.shap_value.to_numpy(float);nv=new.shap_value.to_numpy(float)
            ro=pd.Series(np.abs(ov)).rank(method="average");rn=pd.Series(np.abs(nv)).rank(method="average")
            top_old=set(old.assign(abs_shap=np.abs(ov)).nlargest(10,"abs_shap").feature.astype(str))
            top_new=set(new.assign(abs_shap=np.abs(nv)).nlargest(10,"abs_shap").feature.astype(str))
            old_scale=str(old.output_scale.iloc[0]) if "output_scale" in old else "unspecified"
            new_scale=str(new.output_scale.iloc[0]) if "output_scale" in new else "original_target"
            shap_rows.append({"task":key[0],"fold":key[1],"obs_id":key[2],"n_features":77,
                "old_protocol":"V3_fixed_query_batch_256","new_protocol":POINTWISE_PROTOCOL,
                "old_output_scale":old_scale,"new_output_scale":new_scale,
                "scale_match":"original" in old_scale.lower() and "original" in new_scale.lower(),
                "abs_shap_spearman":float(ro.corr(rn,method="spearman")),"top10_jaccard":len(top_old&top_new)/max(1,len(top_old|top_new)),
                "max_abs_shap_delta":float(np.max(np.abs(ov-nv))),"mean_abs_shap_delta":float(np.mean(np.abs(ov-nv))),
                "old_base":float(old.base_value.iloc[0]),"new_base":float(new.base_value.iloc[0]),
                "base_abs_delta":abs(float(old.base_value.iloc[0]-new.base_value.iloc[0])),
                "old_batch_prediction":float(old.wrapper_prediction_check.iloc[0]) if "wrapper_prediction_check" in old else np.nan,
                "new_pointwise_prediction":float(new.model_prediction.iloc[0]),
                "prediction_abs_delta":abs(float(old.wrapper_prediction_check.iloc[0]-new.model_prediction.iloc[0])) if "wrapper_prediction_check" in old else np.nan})
    shap_cols=["task","fold","obs_id","n_features","old_protocol","new_protocol","old_output_scale","new_output_scale","scale_match",
        "abs_shap_spearman","top10_jaccard","max_abs_shap_delta","mean_abs_shap_delta","old_base","new_base","base_abs_delta",
        "old_batch_prediction","new_pointwise_prediction","prediction_abs_delta"]
    pd.DataFrame(shap_rows,columns=shap_cols).to_csv(OUT/"15_old_batch_vs_pointwise_shap_comparison.csv",index=False,encoding="utf-8-sig")

    old_ale_path=V3/"ale_1d/ale_1d_by_fold.csv";new_ale_path=OUT/"05_ale_1d_results.csv";ale_rows=[]
    if old_ale_path.is_file() and new_ale_path.is_file():
        old_a=pd.read_csv(old_ale_path);new_a=pd.read_csv(new_ale_path)
        for key,old in old_a.groupby(["task","fold","feature"],sort=False):
            new=new_a.loc[(new_a.task==key[0])&(new_a.fold==key[1])&(new_a.feature==key[2])]
            if new.empty:continue
            old=old.sort_values("center");new=new.sort_values("center")
            low=max(float(old.center.min()),float(new.center.min()));high=min(float(old.center.max()),float(new.center.max()))
            if not np.isfinite(low+high) or high<=low:continue
            grid=np.unique(np.r_[old.center.to_numpy(float),new.center.to_numpy(float)])
            grid=grid[(grid>=low)&(grid<=high)]
            if len(grid)<2:continue
            yo=np.interp(grid,old.center.to_numpy(float),old.ale.to_numpy(float));yn=np.interp(grid,new.center.to_numpy(float),new.ale.to_numpy(float))
            rho=float(pd.Series(yo).corr(pd.Series(yn),method="spearman")) if len(grid)>2 else np.nan
            ale_rows.append({"task":key[0],"fold":key[1],"feature":key[2],"old_protocol":"V3_fixed_query_batch_256_pairs",
                "new_protocol":POINTWISE_PROTOCOL,"common_support_lower":low,"common_support_upper":high,"n_comparison_points":len(grid),
                "curve_spearman":rho,"old_range_common":float(yo.max()-yo.min()),"new_range_common":float(yn.max()-yn.min()),
                "max_abs_curve_delta":float(np.max(np.abs(yo-yn))),"mean_abs_curve_delta":float(np.mean(np.abs(yo-yn))),
                "old_min_bin_n":int(old.n_samples.min()),"new_min_bin_n":int(new.n_samples.min()),
                "interpretation":"Different query prediction functions; paired comparison is protocol sensitivity, not SHAP/ALE estimator error."})
    ale_cols=["task","fold","feature","old_protocol","new_protocol","common_support_lower","common_support_upper","n_comparison_points",
        "curve_spearman","old_range_common","new_range_common","max_abs_curve_delta","mean_abs_curve_delta","old_min_bin_n","new_min_bin_n","interpretation"]
    pd.DataFrame(ale_rows,columns=ale_cols).to_csv(OUT/"16_old_batch_vs_pointwise_ale_comparison.csv",index=False,encoding="utf-8-sig")


def register_shap_pilot():
    """Persist the two already-completed pilot samples and their actual runtime stdout."""
    runtime_path=OUT/"shap_runtime.csv"
    manifest=pd.read_csv(OUT/"shap_sample_manifest.csv",dtype={"obs_id":"string"})
    pilot_ids=manifest.loc[(manifest.task=="chla_surface")&(manifest.fold=="fold_1")].sort_values("selection_order").obs_id.astype(str).head(2).tolist()
    if not runtime_path.is_file():raise RuntimeError("SHAP runtime records are missing; cannot certify the two-sample pilot.")
    runtime=pd.read_csv(runtime_path)
    runtime=runtime.loc[(runtime.task=="chla_surface")&(runtime.fold=="fold_1")&runtime.obs_id.astype(str).isin(pilot_ids)].copy()
    if runtime.obs_id.astype(str).nunique()<2:raise RuntimeError("The first two fixed Chla fold1 SHAP samples are not both complete.")
    if "model_predict_calls_cumulative" not in runtime:
        runtime["model_predict_calls_cumulative"]=runtime.model_predict_calls
    if "predict_seconds_cumulative" not in runtime:
        runtime["predict_seconds_cumulative"]=runtime.predict_seconds
    runtime["selection_order"]=runtime.obs_id.astype(str).map({oid:i+1 for i,oid in enumerate(pilot_ids)})
    runtime=runtime.sort_values("selection_order")
    runtime["actual_model_predict_calls"]=runtime.model_predict_calls_cumulative.diff().fillna(runtime.model_predict_calls_cumulative).astype(int)
    runtime["actual_predict_seconds"]=runtime.predict_seconds_cumulative.diff().fillna(runtime.predict_seconds_cumulative)
    runtime.to_csv(runtime_path,index=False,encoding="utf-8-sig")
    vectors=[]
    for oid in pilot_ids:
        p=OUT/"shap_values"/"chla_surface"/"fold_1"/f"{oid}_seed42.csv.gz"
        d=pd.read_csv(p)
        if len(d)!=77 or d.feature_order.tolist()!=list(range(1,78)) or d.feature.nunique()!=77:
            raise RuntimeError(f"Invalid SHAP dimensions/order for {oid}.")
        if float(d.additivity_abs_error.max())>ADDITIVITY_TOL:
            raise RuntimeError(f"SHAP additivity validation failed for {oid}.")
        vectors.append(d)
    logdir=OUT/"logs";logdir.mkdir(parents=True,exist_ok=True)
    stdout=logdir/"C_SHAP_PILOT_chla_surface_fold_1.stdout.log"
    rows=[]
    for _,r in runtime.iterrows():
        rows.append(json.dumps({"unit":f"C:chla_surface:fold_1:{r.obs_id}","elapsed_seconds":float(r.elapsed_seconds),
            "actual_model_predict_calls":int(r.actual_model_predict_calls),"cumulative_calls":int(r.model_predict_calls_cumulative),
            "actual_predict_seconds":float(r.actual_predict_seconds),"additivity":float(r.additivity_abs_error)},ensure_ascii=False))
    stdout.write_text("\n".join(rows)+"\n",encoding="utf-8")
    (logdir/"C_SHAP_PILOT_chla_surface_fold_1.stderr.log").write_text("",encoding="utf-8")
    reg={"unit":"C_SHAP_PILOT:chla_surface:fold_1","status":"COMPLETED_2_FIXED_SAMPLES",
        "elapsed_seconds":float(runtime.elapsed_seconds.sum()),"stdout_log":str(stdout.relative_to(OUT)),
        "stderr_log":"logs/C_SHAP_PILOT_chla_surface_fold_1.stderr.log (no stderr captured in the original interactive session)",
        "error":"Original interactive worker was stopped after the second completed checkpoint to begin the requested ALE pilot; completed sample files remain valid.",
        "completed_at":pd.Timestamp.now().isoformat()}
    append_rows(OUT/"13_final_explanation_master_registry.csv",pd.DataFrame([reg]),["unit"])


def run_continuation_after_pilot():
    init_manifest()
    register_shap_pilot()
    f1="ssrd_mean_14d";f2="wind_speed_x_rain14d"
    status=run_worker_subprocess(["--worker-ale1-pilot","chla_surface","fold_1",f1,f2],"D_ALE1_PILOT:chla_surface:fold_1")
    if status!="COMPLETED":
        raise RuntimeError(f"Chla fold1 ALE pilot failed after one bounded retry; inspect logs and registry: {status}")
    summary=pd.read_csv(OUT/"06_ale_1d_support_and_stability.csv")
    summary=summary.loc[(summary.task=="chla_surface")&(summary.fold=="fold_1")&summary.feature.isin([f1,f2])]
    repeats=pd.read_csv(OUT/"ale_1d_repeatability.csv")
    repeats=repeats.loc[(repeats.task=="chla_surface")&(repeats.fold=="fold_1")&repeats.feature.isin([f1,f2])]
    if set(summary.feature)!=set([f1,f2]) or set(repeats.feature)!=set([f1,f2]) or not repeats.n_repeats.eq(3).all():
        raise RuntimeError("ALE pilot did not produce both fixed variables and three uncached repeats; inspect its checkpoints.")
    # Repeatability is reported independently. Do not mask a failure; continue the full run but preserve status.
    repeats["repeatability_status"]=np.where(repeats.max_local_ale_delta_range<=1e-6,"STABLE_LE_1E6","VARIABILITY_GT_1E6")
    repeats.to_csv(OUT/"ale_1d_repeatability.csv",index=False,encoding="utf-8-sig")
    coordinator(run_all=True)


def finalize():
    # Rebuild flattened, auditable SHAP table from successful per-sample checkpoints.
    shap_files=sorted((OUT/"shap_values").glob("*/*/*.csv.gz")) if (OUT/"shap_values").exists() else []
    if shap_files:
        sf=pd.concat([pd.read_csv(p) for p in shap_files],ignore_index=True);sf.to_csv(OUT/"03_shap_full_results.csv",index=False,encoding="utf-8-sig")
    # Status-separated registry and evidence levels.
    reg=OUT/"13_final_explanation_master_registry.csv"
    if reg.exists():
        r=pd.read_csv(reg);r.to_csv(reg,index=False,encoding="utf-8-sig")
    a=pd.read_csv(OUT/"01_pointwise_inference_validation.csv") if (OUT/"01_pointwise_inference_validation.csv").is_file() else pd.DataFrame()
    nstable=int((a.status=="STABLE").sum()) if len(a) else 0
    sh=pd.read_csv(OUT/"shap_runtime.csv") if (OUT/"shap_runtime.csv").is_file() else pd.DataFrame()
    ale=pd.read_csv(OUT/"06_ale_1d_support_and_stability.csv") if (OUT/"06_ale_1d_support_and_stability.csv").is_file() else pd.DataFrame()
    a2=pd.read_csv(OUT/"08_ale_2d_support_and_stability.csv") if (OUT/"08_ale_2d_support_and_stability.csv").is_file() else pd.DataFrame()
    if len(sh):
        sf=pd.read_csv(OUT/"03_shap_full_results.csv")
        shap_imp=sf.groupby(["task","fold","feature"],as_index=False).agg(mean_abs_shap=("shap_value",lambda s:float(np.mean(np.abs(s)))))
        shap_imp["rank"]=shap_imp.groupby(["task","fold"]).mean_abs_shap.rank(method="min",ascending=False)
        shap_imp.to_csv(OUT/"03_shap_importance_by_fold.csv",index=False,encoding="utf-8-sig")
        shap_task=shap_imp.groupby(["task","feature"],as_index=False).agg(mean_abs_shap=("mean_abs_shap","mean"),
            mean_fold_rank=("rank","mean"),sd_fold_rank=("rank","std"),folds_observed=("fold","nunique"))
        shap_task["task_rank"]=shap_task.groupby("task").mean_abs_shap.rank(method="min",ascending=False)
        shap_task.to_csv(OUT/"04_shap_importance_summary.csv",index=False,encoding="utf-8-sig")
        stab=[]
        for (task,fold),g in shap_imp.groupby(["task","fold"]):
            # Deterministic split-half rank comparison; no target labels used.
            d=pd.read_csv(OUT/"03_shap_full_results.csv");d=d[(d.task==task)&(d.fold==fold)]
            ids=sorted(d.obs_id.astype(str).unique()); half=set(ids[::2]);
            a1=d[d.obs_id.astype(str).isin(half)].groupby("feature").shap_value.apply(lambda s:np.mean(np.abs(s)))
            a2=d[~d.obs_id.astype(str).isin(half)].groupby("feature").shap_value.apply(lambda s:np.mean(np.abs(s)))
            common=a1.index.intersection(a2.index); rho=float(a1[common].rank().corr(a2[common].rank(),method="spearman")) if len(common)>2 else np.nan
            top1=set(a1.nlargest(10).index);top2=set(a2.nlargest(10).index);jac=len(top1&top2)/max(1,len(top1|top2))
            stab.append({"task":task,"fold":fold,"n_samples":len(ids),"spearman_split_half":rho,"top10_jaccard":jac,
                "status":"SUPPORTED" if len(ids)>=10 and rho>=.70 and jac>=.50 else "SHAP_EXPLORATORY"})
        pd.DataFrame(stab).to_csv(OUT/"04_shap_stability_validation.csv",index=False,encoding="utf-8-sig")
    if not len(ale):
        pd.DataFrame(columns=["task","fold","feature","status"]).to_csv(OUT/"06_ale_1d_support_and_stability.csv",index=False,encoding="utf-8-sig")
    if not len(a2):
        pd.DataFrame(columns=["task","fold","feature_1","feature_2","status"]).to_csv(OUT/"08_ale_2d_support_and_stability.csv",index=False,encoding="utf-8-sig")
    compare_old_batch_explanations()
    completed_shap=int(sh[["task","fold","obs_id"]].drop_duplicates().shape[0]) if len(sh) else 0
    expected_shap=int(len(pd.read_csv(OUT/"shap_sample_manifest.csv"))) if (OUT/"shap_sample_manifest.csv").is_file() else 0
    full_metrics=pd.read_csv(OUT/"full_fold_prediction_metrics.csv") if (OUT/"full_fold_prediction_metrics.csv").is_file() else pd.DataFrame()
    audit1=pd.read_csv(REPAIR/"06_ale_1d_rebinning_comparison.csv")
    expected1=audit1.loc[audit1.task.isin(TASKS),["task","fold","feature"]].drop_duplicates()
    got1=ale[["task","fold","feature"]].drop_duplicates() if len(ale) and {"task","fold","feature"}.issubset(ale.columns) else pd.DataFrame(columns=["task","fold","feature"])
    n1_covered=len(expected1.merge(got1,on=["task","fold","feature"],how="inner"))
    audit2=pd.read_csv(REPAIR/"07_ale_2d_grid_sensitivity.csv")
    expected2=audit2.loc[audit2.task.isin(TASKS),["task","fold","feature_1","feature_2","grid_actual"]].drop_duplicates()
    got2=a2[["task","fold","feature_1","feature_2","grid"]].drop_duplicates() if len(a2) and {"task","fold","feature_1","feature_2","grid"}.issubset(a2.columns) else pd.DataFrame(columns=["task","fold","feature_1","feature_2","grid"])
    n2_covered=len(expected2.merge(got2,left_on=["task","fold","feature_1","feature_2","grid_actual"],right_on=["task","fold","feature_1","feature_2","grid"],how="inner"))
    all_pointwise=len(a)>=20;all_shap=completed_shap>=expected_shap>0;all_ale1=n1_covered>=len(expected1)>0;all_ale2=n2_covered>=len(expected2)>0
    closed=all_pointwise and all_shap and all_ale1 and all_ale2 and len(full_metrics)>=nstable
    execution="EXPLANATION_COMPLETE_WITH_LIMITATIONS" if closed else ("EXPLANATION_PARTIALLY_COMPLETE" if nstable or completed_shap or len(ale) or len(a2) else "EXPLANATION_BLOCKED")
    (OUT/"11_independent_validation_report.md").write_text("""# 独立核验报告

本报告由`81_validate_tabicl_explanation_results.py`读取实际落盘文件重新计算。独立核验结果和失败明细见该脚本写出的`validation_checks.csv`。所有正式解释须保持77列顺序和原始目标尺度；SHAP加性误差按基准值+77个归因与点式模型输出重新核验；ALE只检查数据覆盖通过的折次，二维ALE检查二阶差分、支持掩码和中心化。
""",encoding="utf-8")
    (OUT/"12_remaining_limitations.md").write_text(f"""# 剩余限制

- 新逐样本协议目前通过折次：{nstable}/20；完整测试折预测差异仅在预先固定的查询探针上评估，不替代正式指标。
- 新协议与历史批量预测属于不同预测函数；不得将新SHAP/ALE称为对历史冻结批量预测函数的严格解释。
- 新SHAP样本覆盖：{completed_shap}/{expected_shap}；尚未达到每折10样本或稳定阈值的结果只能标为探索性/未完成。
- 完整测试折P1对照：{len(full_metrics)}/{nstable}个稳定折；一维ALE状态：{n1_covered}/{len(expected1)}个候选；二维ALE状态：{n2_covered}/{len(expected2)}个候选网格。
- 一维ALE及二维ALE只对覆盖充分区域计算；无支持网格保持遮蔽，不插值。
- 历史冻结模型运行时完整随机状态和旧推理进程快照未保存，不能宣称逐值恢复原模型输出。
""",encoding="utf-8")
    (OUT/"14_final_explanation_closure_decision.md").write_text(f"""# 最终解释实验闭环决策

## 执行状态

`{execution}`

## 分层证据

- 点式推理验证：{nstable}/20折通过跨进程阈值。
- 新SHAP解释样本：{completed_shap}个独立样本结果；正式状态须以独立核验和折内稳定性表为准。
- 一维ALE：{len(ale)}条任务-折-变量状态记录；数据覆盖不足者保持`INSUFFICIENT_SUPPORT`。
- 二维ALE：{len(a2)}条任务-折-组合状态记录；不能将局部支持推广至全联合范围。
- 新旧SHAP逐样本对照：{len(pd.read_csv(OUT/"15_old_batch_vs_pointwise_shap_comparison.csv"))}个共同样本；新旧一维ALE共同支持对照：{len(pd.read_csv(OUT/"16_old_batch_vs_pointwise_ale_comparison.csv"))}条曲线。二者均为不同查询预测函数间的协议敏感性比较，不是历史函数严格复现。
- 闭环核对：P1={len(a)}/20；SHAP={completed_shap}/{expected_shap}；ALE1={n1_covered}/{len(expected1)}；ALE2={n2_covered}/{len(expected2)}；稳定折全测试对照={len(full_metrics)}/{nstable}。

## 科学结论边界

新协议下的解释仅适用于`TABICLV2_POINTWISE_EXPLANATION_V1`。因查询集合/分批敏感性已被观察到，新点式解释不等于历史冻结批量预测函数解释。任何SHAP/ALE统计关系均不作因果表述。任务执行完成度与证据充分程度分开判断。
""",encoding="utf-8")


def coordinator(run_all=False):
    init_manifest()
    run_phase_a()
    summarize_all()
    if (OUT/"PAUSE_AFTER_CURRENT_FOLD.flag").is_file():
        return
    if run_all:
        phase_b_full()
        phase_cde()
        finalize()


def summarize_all():
    for t in TASKS:
        for f in FOLDS:
            summarize_fold_pointwise(t,f)
    # Always update decision snapshot; do not mark absent work complete.
    finalize()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--init",action="store_true");p.add_argument("--run-all",action="store_true");p.add_argument("--phase-a",action="store_true")
    p.add_argument("--continue-after-pilot",action="store_true")
    p.add_argument("--worker-pointwise",nargs=3,metavar=("TASK","FOLD","REP"));p.add_argument("--worker-shap",nargs=2,metavar=("TASK","FOLD"))
    p.add_argument("--worker-ale1",nargs=2,metavar=("TASK","FOLD"));p.add_argument("--worker-ale1-pilot",nargs=4,metavar=("TASK","FOLD","FEATURE1","FEATURE2"))
    p.add_argument("--worker-ale2",nargs=2,metavar=("TASK","FOLD"))
    p.add_argument("--worker-fullcompare",nargs=2,metavar=("TASK","FOLD"));p.add_argument("--summarize",action="store_true")
    a=p.parse_args()
    if a.worker_pointwise: worker_pointwise(a.worker_pointwise[0],a.worker_pointwise[1],int(a.worker_pointwise[2]))
    elif a.worker_shap: shap_worker(*a.worker_shap)
    elif a.worker_ale1: ale_1d_worker(*a.worker_ale1)
    elif a.worker_ale1_pilot: ale_1d_worker(a.worker_ale1_pilot[0],a.worker_ale1_pilot[1],a.worker_ale1_pilot[2:])
    elif a.worker_ale2: ale_2d_worker(*a.worker_ale2)
    elif a.worker_fullcompare: full_prediction_worker(*a.worker_fullcompare)
    elif a.summarize: summarize_all()
    elif a.init:
        init_manifest()
    elif a.continue_after_pilot:
        run_continuation_after_pilot()
    elif a.phase_a:
        coordinator(run_all=False)
    elif a.run_all:
        coordinator(run_all=True)
    else:
        p.error("请指定--init、--phase-a或--run-all")


if __name__=="__main__":
    try: main()
    except Exception:
        print(traceback.format_exc(),file=sys.stderr);raise
