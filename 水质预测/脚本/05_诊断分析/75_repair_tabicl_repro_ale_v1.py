"""TabICLv2推理重建稳定性与ALE支持度敏感性审计。

本脚本只读取冻结预测、训练manifest及现有解释结果。TabICLv2的fit仅构建
预训练模型的训练上下文，不更新预训练权重。所有新试算写入独立目录。
"""
from __future__ import annotations

import argparse
import functools
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
OUT = ROOT / "runs/tabicl_v2_repro_ale_repair_v1"
BASE_PATH = ROOT / "脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py"
V3 = ROOT / "runs/tabicl_v2_explainability_v3"
CLOSURE = ROOT / "runs/tabicl_v2_explainability_closure_v1"
STRICT_MAX = 1e-6
STRICT_MEAN = 1e-7
FAILED = [("chla_surface", f"fold_{i}") for i in (1, 2, 3, 5)] + [("secchi", "fold_4")]
SEED = 42
PROBE_N = 32
REPEATS = 3
TASK_ALIAS = {"chla_surface": "chla_surface", "secchi": "secchi", "tn_surface": "tn_surface", "tp_middle": "tp_middle"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def arr_hash(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return sha256_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def ids_hash(xs) -> str:
    return sha256_bytes("\n".join(map(str, xs)).encode("utf-8"))


@functools.lru_cache(maxsize=16)
def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path: str, rows) -> None:
    p = OUT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        keys = [c for c in ("task", "fold", "condition", "replicate", "feature", "feature_1", "feature_2", "grid") if c in df.columns and c in old.columns]
        if keys:
            sig = df[keys].fillna("").astype(str).agg("|".join, axis=1)
            oldsig = old[keys].fillna("").astype(str).agg("|".join, axis=1)
            old = old.loc[~oldsig.isin(set(sig))]
            df = pd.concat([old, df], ignore_index=True, sort=False)
    df.to_csv(p, index=False, encoding="utf-8-sig")


def load_base():
    spec = importlib.util.spec_from_file_location("tabicl_repair_base", BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_all_seeds(seed: int, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare(base, task: str, fold: str):
    from sklearn.impute import SimpleImputer
    features = base.load_features()
    train, test = base.load_fold(task, fold, features)
    target = base.TASKS[task]["target"]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xtr = imputer.fit_transform(train[features]).astype("float32")
    xte = imputer.transform(test[features]).astype("float32")
    y0 = train[target].to_numpy(float)
    log_target = bool(np.nanmin(y0) >= 0)
    ytr = np.log1p(y0).astype("float32") if log_target else y0.astype("float32")
    pred = base.frozen(task, fold)
    # 完全不使用水质标签选择探针：按测试特征的站点轮转顺序固定取32条。
    pos = base.sample_positions(test, min(PROBE_N, len(test)), SEED + int(fold[-1]) + 100 * list(base.TASKS).index(task))
    q = xte[pos]
    frozen = pred.set_index("obs_id").loc[test.iloc[pos].obs_id.astype(str), "y_pred"].to_numpy(float)
    meta = {
        "task": task, "fold": fold, "n_train": len(train), "n_test": len(test), "n_features": len(features),
        "feature_order_hash": ids_hash(features), "train_obs_order_hash": ids_hash(train.obs_id.astype(str)),
        "test_obs_order_hash": ids_hash(test.obs_id.astype(str)), "train_obs_set_hash": ids_hash(sorted(train.obs_id.astype(str))),
        "test_obs_set_hash": ids_hash(sorted(test.obs_id.astype(str))), "raw_train_feature_hash": arr_hash(train[features].to_numpy(dtype="float64")),
        "raw_test_feature_hash": arr_hash(test[features].to_numpy(dtype="float64")), "imputed_train_hash": arr_hash(xtr),
        "imputed_test_hash": arr_hash(xte), "imputer_strategy": "median", "imputer_keep_empty_features": True,
        "imputer_statistics_hash": arr_hash(imputer.statistics_), "imputer_statistics": json.dumps(imputer.statistics_.tolist()),
        "model_y_hash": arr_hash(ytr), "target_transform": "log1p then estimator StandardScaler" if log_target else "raw then estimator StandardScaler",
        "target_inverse": "expm1 then nonnegative clip" if log_target else "nonnegative clip", "probe_obs_ids": ";".join(test.iloc[pos].obs_id.astype(str)),
        "probe_order_hash": ids_hash(test.iloc[pos].obs_id.astype(str)), "frozen_probe_hash": arr_hash(frozen),
    }
    return features, train, test, imputer, xtr, xte, ytr, y0, pos, q, frozen, meta


def estimator(base, xtr, ytr, condition="formal", torch=None):
    from tabicl import TabICLRegressor
    kw = {"device": "cuda", "use_amp": "auto", "offload_mode": "auto", "random_state": 42, "verbose": False}
    if condition == "no_amp": kw["use_amp"] = False
    if condition == "cpu": kw.update(device="cpu", use_amp=False, offload_mode=False)
    # 与models/tabiclv2_adapter.py相同默认配置；诊断条件只覆盖明确登记的device/AMP。
    model = TabICLRegressor(**kw)
    model.fit(xtr, ytr)
    return model


def pred_original(model, x, inverse):
    raw = np.asarray(model.predict(np.asarray(x, dtype="float32")), dtype=float).reshape(-1)
    return np.maximum(np.expm1(raw) if inverse else raw, 0)


def ensemble_hash(model):
    eg = model.ensemble_generator_
    bits = []
    bits.append(str([(k, len(v)) for k, v in eg.ensemble_configs_.items()]))
    for k, v in eg.ensemble_configs_.items():
        bits.append(f"norm={k}")
        bits.append(arr_hash(eg.preprocessors_[k].X_transformed_))
        bits.extend(arr_hash(np.asarray(fs, dtype=np.int64)) for fs, _yp in v)
    for k, v in eg.preprocessors_.items():
        bits.append(f"{k}:{type(v.normalizer_).__name__}:{getattr(v, 'outlier_threshold', None)}")
    return sha256_bytes("|".join(bits).encode())


def checkpoint_hash(model):
    p = Path(str(getattr(model, "model_path_", "")))
    return file_hash(p) if p.is_file() else "UNAVAILABLE:" + str(p)


def runtime_info(torch):
    try:
        import tabicl
        tabicl_ver = getattr(tabicl, "__version__", "2.2.0")
    except Exception:
        tabicl_ver = "unknown"
    cuda = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(), "python_executable": sys.executable, "python_hashseed": os.environ.get("PYTHONHASHSEED", "randomized"),
        "torch": torch.__version__, "tabicl": tabicl_ver, "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else "CPU", "cuda_available": cuda,
        "driver_version": driver_version(), "amp_policy": "auto/default unless condition=no_amp/cpu",
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_cpu_rng_hash": arr_hash(torch.get_rng_state().cpu().numpy()),
        "torch_cuda_rng_hash": arr_hash(torch.cuda.get_rng_state_all()[0].cpu().numpy()) if cuda else "NA",
    }


def driver_version():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def context_and_input_audit(base):
    import torch
    records = []
    for task, fold in FAILED:
        features, train, test, imp, xtr, xte, ytr, y0, pos, q, frozen, meta = prepare(base, task, fold)
        rec = dict(meta)
        rec.update({"feature_names": json.dumps(features, ensure_ascii=False), "query_n": len(q), "query_feature_hash": arr_hash(q),
                    "frozen_query_prediction_hash": arr_hash(frozen), "context_hash": "pending_model_fit"})
        records.append(rec)
    save("01_context_reconstruction_audit.csv", records)
    import torch
    manifest={"created_local_date":"2026-10-21","script_sha256":file_hash(Path(__file__).resolve()),"formal_reproduction_threshold":{"max_abs":STRICT_MAX,"mean_abs":STRICT_MEAN},
        "probe_queries":PROBE_N,"repetitions_per_condition":REPEATS,"failed_folds":FAILED,"feature_count":77,"seed":SEED,"run_environment":runtime_info(torch),
        "formal_frozen_predictions_modified":False,"pretrained_weights_updated":False,"split_or_feature_changes":False}
    (OUT/"00_run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return records


def make_record(task, fold, condition, replicate, model, q, frozen, inverse, info, seed_mode, elapsed, reference=None):
    p = pred_original(model, q, inverse)
    ref = p if reference is None else reference
    d = np.abs(p - frozen)
    return {"task": task, "fold": fold, "condition": condition, "replicate": replicate,
        "status": "PROBE_OK", "n_probe": len(q), "max_abs_vs_frozen_probe": float(d.max()), "mean_abs_vs_frozen_probe": float(d.mean()),
        "max_abs_vs_first_same_condition": float(np.max(np.abs(p-ref))), "mean_abs_vs_first_same_condition": float(np.mean(np.abs(p-ref))),
        "prediction_hash": arr_hash(p), "probe_predictions_json":json.dumps(p.tolist()), "context_hash": ensemble_hash(model), "pretrained_weight_sha256": checkpoint_hash(model),
        "ensemble_config_order": json.dumps({k: [list(map(int,x[0])) for x in v] for k,v in model.ensemble_generator_.ensemble_configs_.items()}),
        "n_estimators": model.n_estimators, "feature_shuffle_method": model.feat_shuffle_method, "resolved_device": str(model.device_),
        "resolved_batch_size": str(getattr(model, "inference_config_", None)), "random_state": model.random_state,
        "seed_mode": seed_mode, "fit_and_predict_seconds": elapsed, **info}


def probe_worker(task, fold, condition, replicate, outpath):
    import torch
    base = load_base()
    if condition in ("seeded", "deterministic", "no_amp", "cpu", "hashseed_fixed"): set_all_seeds(SEED, torch)
    if condition == "deterministic":
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
    features, train, test, imp, xtr, xte, ytr, y0, pos, q, frozen, meta = prepare(base, task, fold)
    started = time.perf_counter()
    model = estimator(base, xtr, ytr, condition="formal" if condition=="hashseed_fixed" else condition, torch=torch)
    info = runtime_info(torch)
    rec = make_record(task, fold, condition, replicate, model, q, frozen, np.nanmin(y0) >= 0, info,
                      "all_python_numpy_torch_cuda_seed42" if condition in ("seeded","hashseed_fixed") else "formal-estimator-seed42-only", time.perf_counter()-started)
    rec["input_hashes_json"] = json.dumps(meta, ensure_ascii=False)
    pd.DataFrame([rec]).to_csv(outpath, index=False, encoding="utf-8-sig")


def run_hash_order_pairwise():
    """固定所有RNG，只改变PYTHONHASHSEED，逐样本比较两种集成视图顺序。"""
    seeds=(1,42); outrows=[]
    for task,fold in FAILED:
        files={}
        for seed in seeds:
            path=OUT/f"hashorder_{task}_{fold}_seed{seed}.csv"
            if not path.exists():
                env=os.environ.copy();env["PYTHONHASHSEED"]=str(seed);env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                cmd=[sys.executable,str(Path(__file__).resolve()),"--worker",task,fold,"hashseed_fixed",str(seed),str(path)]
                try:
                    cp=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
                    if cp.returncode!=0 or not path.exists():
                        outrows.append({"task":task,"fold":fold,"status":"PROCESS_FAILED","python_hashseed":seed,"error":cp.stderr[-1500:]});continue
                except subprocess.TimeoutExpired:
                    outrows.append({"task":task,"fold":fold,"status":"TIMEOUT_240S","python_hashseed":seed});continue
            files[seed]=pd.read_csv(path).iloc[0].to_dict()
        if len(files)==2:
            a,b=(files[s] for s in seeds);pa=np.asarray(json.loads(a["probe_predictions_json"]));pb=np.asarray(json.loads(b["probe_predictions_json"]));delta=np.abs(pa-pb)
            outrows.append({"task":task,"fold":fold,"status":"HASH_ORDER_PAIR_COMPLETE","n_probe":len(pa),"seed_1":1,"seed_42":42,
                "order_seed_1":json.dumps(list(json.loads(a["ensemble_config_order"]).keys())),"order_seed_42":json.dumps(list(json.loads(b["ensemble_config_order"]).keys())),
                "context_hash_seed_1":a["context_hash"],"context_hash_seed_42":b["context_hash"],"prediction_hash_seed_1":a["prediction_hash"],"prediction_hash_seed_42":b["prediction_hash"],
                "max_pairwise_abs_error":float(delta.max()),"mean_pairwise_abs_error":float(delta.mean()),"global_rng_seed_fixed":True,"pretrained_weight_hash_equal":a["pretrained_weight_sha256"]==b["pretrained_weight_sha256"]})
    pd.DataFrame(outrows).to_csv(OUT/"02_context_order_pairwise_delta.csv",index=False,encoding="utf-8-sig")


def probe_fold(base, task, fold, do_independent=True):
    import torch
    from sklearn.base import clone
    features, train, test, imp, xtr, xte, ytr, y0, pos, q, frozen, meta = prepare(base, task, fold)
    outcsv=OUT/"02_determinism_experiment_results.csv"
    if outcsv.exists():
        prior=pd.read_csv(outcsv);prior=prior[(prior.task==task)&(prior.fold==fold)]
        allrows=prior.to_dict("records")
    else: allrows=[]
    done={(str(r.get("condition")),int(r.get("replicate",0))) for r in allrows}
    def has(cond,rep): return (cond,rep) in done
    def checkpoint(row):
        allrows.append(row);done.add((str(row.get("condition")),int(row.get("replicate",0))))
        save("02_determinism_experiment_results.csv",allrows)
    # C: one independently constructed model instance per repetition in a single process.
    model = estimator(base, xtr, ytr)
    info = runtime_info(torch)
    context0 = ensemble_hash(model)
    frozen_inverse = np.nanmin(y0) >= 0
    a_preds=[]
    b_first=None
    for rep in range(1, REPEATS+1):
        t=time.perf_counter(); p=pred_original(model,q,frozen_inverse); elapsed=time.perf_counter()-t; a_preds.append(p)
        if not has("A_same_model_same_context",rep):
            r=make_record(task,fold,"A_same_model_same_context",rep,model,q,frozen,frozen_inverse,info,"unmodified formal process state",elapsed,a_preds[0]);checkpoint(r)
    # B: same pre-trained weight instance, reconstruct only the fitted ensemble context 3 times.
    from tabicl._sklearn.preprocessing import EnsembleGenerator
    encoded = model.X_encoder_.transform(xtr)
    yscaled = model.ensemble_generator_.y_
    gen0=model.ensemble_generator_
    for rep in range(1, REPEATS+1):
        gen=EnsembleGenerator(classification=False,n_estimators=model.n_estimators,norm_methods=model.norm_methods or ["none","power"],
            feat_shuffle_method=model.feat_shuffle_method,outlier_threshold=model.outlier_threshold,random_state=model.random_state)
        gen.fit(encoded,yscaled); model.ensemble_generator_=gen
        if has("B_same_loaded_weights_rebuild_context",rep): continue
        t=time.perf_counter(); p=pred_original(model,q,frozen_inverse); elapsed=time.perf_counter()-t
        if b_first is None: b_first=p.copy()
        r=make_record(task,fold,"B_same_loaded_weights_rebuild_context",rep,model,q,frozen,frozen_inverse,runtime_info(torch),"formal-estimator-seed42-only",elapsed,b_first); checkpoint(r)
    model.ensemble_generator_=gen0
    # C: independent estimator reconstruction 3x, retaining process-level environment.
    c_preds=[]
    for rep in range(1, REPEATS+1):
        if has("C_new_model_same_process",rep): continue
        t=time.perf_counter(); m=estimator(base,xtr,ytr); p=pred_original(m,q,frozen_inverse); elapsed=time.perf_counter()-t;c_preds.append(p)
        r=make_record(task,fold,"C_new_model_same_process",rep,m,q,frozen,frozen_inverse,runtime_info(torch),"formal-estimator-seed42-only",elapsed,c_preds[0]);checkpoint(r)
    # Additional RNG-initialized same-process condition (algorithm/config unchanged).
    seeded_preds=[]
    for rep in range(1, REPEATS+1):
        if has("A4_global_rng_seeded_formal",rep): continue
        set_all_seeds(SEED,torch);t=time.perf_counter();m=estimator(base,xtr,ytr);p=pred_original(m,q,frozen_inverse);elapsed=time.perf_counter()-t;seeded_preds.append(p)
        r=make_record(task,fold,"A4_global_rng_seeded_formal",rep,m,q,frozen,frozen_inverse,runtime_info(torch),"python_numpy_torch_cuda_seed42",elapsed,seeded_preds[0]);checkpoint(r)
    if do_independent:
        # D: separate interpreters under the original estimator configuration and ambient RNG policy.
        for rep in range(1, REPEATS+1):
            if has("D_independent_process_formal_ambient_rng",rep): continue
            worker=OUT/f"worker_{task}_{fold}_D{rep}.csv"
            env=os.environ.copy();env.pop("PYTHONHASHSEED",None)
            cmd=[sys.executable,str(Path(__file__).resolve()),"--worker",task,fold,"formal",str(rep),str(worker)]
            try:
                cp=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
                if cp.returncode==0 and worker.exists():
                    wr=pd.read_csv(worker).iloc[0].to_dict();wr["condition"]="D_independent_process_formal_ambient_rng";checkpoint(wr)
                else:
                    checkpoint({"task":task,"fold":fold,"condition":"D_independent_process_formal_ambient_rng","replicate":rep,"status":"PROCESS_FAILED","error":cp.stderr[-2000:]})
            except subprocess.TimeoutExpired:
                checkpoint({"task":task,"fold":fold,"condition":"D_independent_process_formal_ambient_rng","replicate":rep,"status":"TIMEOUT_240S"})
    save("02_determinism_experiment_results.csv",allrows)
    # Attach full model/context and tensor-level inputs to per-fold audit.
    ctxrow={**meta,"context_hash":context0,"pretrained_weight_sha256":checkpoint_hash(model),
            "ensemble_config_order":json.dumps({k:[list(map(int,x[0])) for x in v] for k,v in gen0.ensemble_configs_.items()}),
            "encoded_training_hash":arr_hash(model.X_encoder_.transform(xtr)),"estimator_scaled_label_hash":arr_hash(gen0.y_),
            "ensemble_generator_training_matrix_hash":arr_hash(gen0.X_),"context_sample_count":len(gen0.X_),
            "n_estimators":model.n_estimators,"feature_shuffle_method":model.feat_shuffle_method,"batch_size":getattr(model,"batch_size",None),
            "amp_setting":model.use_amp,"resolved_device":str(model.device_),"feature_view_hashes":json.dumps({k:arr_hash(v.X_transformed_) for k,v in gen0.preprocessors_.items()}),
            "rng_after_fit":json.dumps(runtime_info(torch),ensure_ascii=False)}
    save("01_context_reconstruction_audit.csv",[ctxrow])


def run_config_diagnostics():
    """按原配置、全局种子、关AMP、确定性算法及CPU逐项做受限比较。"""
    task,fold="chla_surface","fold_1"
    rows=[]
    todo=[("seeded",3,240), ("no_amp",3,240), ("deterministic",3,240), ("cpu",1,180)]
    for condition,nrep,timeout in todo:
      for rep in range(1,nrep+1):
        worker=OUT/f"config_{condition}_{task}_{fold}_r{rep}.csv"
        if worker.exists():
            rows.append(pd.read_csv(worker).iloc[0].to_dict());continue
        env=os.environ.copy();env["PYTHONHASHSEED"]="42";env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        cmd=[sys.executable,str(Path(__file__).resolve()),"--worker",task,fold,condition,str(rep),str(worker)]
        started=time.perf_counter()
        try:
            cp=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=timeout)
            if cp.returncode==0 and worker.exists():
                r=pd.read_csv(worker).iloc[0].to_dict();r["condition_wall_seconds"]=time.perf_counter()-started;rows.append(r)
            else:
                rows.append({"task":task,"fold":fold,"condition":condition,"replicate":rep,"status":"PROCESS_FAILED","error":cp.stderr[-2000:],"condition_wall_seconds":time.perf_counter()-started})
        except subprocess.TimeoutExpired:
            rows.append({"task":task,"fold":fold,"condition":condition,"replicate":rep,"status":f"TIMEOUT_{timeout}S","condition_wall_seconds":time.perf_counter()-started})
        save("02_determinism_experiment_results.csv",rows)
        # CPU is only an feasibility pilot, do not commit to repeated CPU runs.
        if condition=="cpu":break
    save("02_determinism_experiment_results.csv",rows)


def run_hashseed_candidate(seed=1):
    """检查固定PYTHONHASHSEED能否固定set派生的norm方法顺序；不改算法。"""
    task,fold="chla_surface","fold_1"
    rows=[]
    for rep in range(1,REPEATS+1):
        worker=OUT/f"config_hashseed_{seed}_{task}_{fold}_r{rep}.csv"
        if worker.exists():
            rows.append(pd.read_csv(worker).iloc[0].to_dict());continue
        env=os.environ.copy();env["PYTHONHASHSEED"]=str(seed)
        cmd=[sys.executable,str(Path(__file__).resolve()),"--worker",task,fold,"formal",str(rep),str(worker)]
        started=time.perf_counter()
        try:
            cp=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
            if cp.returncode==0 and worker.exists():
                r=pd.read_csv(worker).iloc[0].to_dict();r["condition"]="fixed_hashseed_candidate";r["condition_wall_seconds"]=time.perf_counter()-started;rows.append(r)
            else: rows.append({"task":task,"fold":fold,"condition":"fixed_hashseed_candidate","replicate":rep,"status":"PROCESS_FAILED","error":cp.stderr[-2000:]})
        except subprocess.TimeoutExpired:
            rows.append({"task":task,"fold":fold,"condition":"fixed_hashseed_candidate","replicate":rep,"status":"TIMEOUT_240S"})
        save("02_determinism_experiment_results.csv",rows)
    save("02_determinism_experiment_results.csv",rows)


def probe_all_folds():
    p=OUT/"02_determinism_experiment_results.csv"
    existing=pd.read_csv(p) if p.exists() else pd.DataFrame()
    required={"A_same_model_same_context","B_same_loaded_weights_rebuild_context","C_new_model_same_process","A4_global_rng_seeded_formal","D_independent_process_formal_ambient_rng"}
    for task,fold in FAILED:
        g=existing[(existing.task==task)&(existing.fold==fold)] if len(existing) else pd.DataFrame()
        complete=all(len(g[g.condition.eq(c)])>=REPEATS for c in required)
        if complete: continue
        env=os.environ.copy();env["PYTHONHASHSEED"]="42"
        cmd=[sys.executable,str(Path(__file__).resolve()),"--probe-fold",task,fold]
        started=time.perf_counter()
        try:
            cp=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=900)
            (OUT/f"probe_{task}_{fold}.log").write_text(cp.stdout+"\n"+cp.stderr,encoding="utf-8")
            if cp.returncode!=0:
                save("02_determinism_experiment_results.csv",[{"task":task,"fold":fold,"condition":"probe_fold_driver","replicate":1,"status":"PROCESS_FAILED","error":cp.stderr[-2000:],"wall_seconds":time.perf_counter()-started}])
        except subprocess.TimeoutExpired as e:
            tail=(e.stderr or b"").decode("utf-8",errors="replace")[-2000:] if isinstance(e.stderr,bytes) else str(e.stderr)[-2000:]
            save("02_determinism_experiment_results.csv",[{"task":task,"fold":fold,"condition":"probe_fold_driver","replicate":1,"status":"TIMEOUT_900S","error":tail,"wall_seconds":time.perf_counter()-started}])
        existing=pd.read_csv(p) if p.exists() else pd.DataFrame()


def full_fold_worker(task, fold, replicate, outpath, seeded=True):
    import torch
    base=load_base()
    if seeded: set_all_seeds(SEED,torch)
    features,train,test,imp,xtr,xte,ytr,y0,pos,q,frozen,meta=prepare(base,task,fold)
    start=time.perf_counter();model=estimator(base,xtr,ytr);fitsec=time.perf_counter()-start
    start=time.perf_counter();p=pred_original(model,xte,np.nanmin(y0)>=0);predsec=time.perf_counter()-start
    old=base.frozen(task,fold);aligned=pd.DataFrame({"obs_id":test.obs_id.astype(str),"y_pred_new":p}).merge(old[["obs_id","y_pred"]],on="obs_id",validate="one_to_one")
    delta=np.abs(aligned.y_pred_new-aligned.y_pred)
    rec={**meta,"replicate":replicate,"condition":"global_rng_seed42+PYTHONHASHSEED42+formal_cuda_amp_auto","status":"EXACT_REPRODUCTION" if delta.max()<=STRICT_MAX and delta.mean()<=STRICT_MEAN and test.obs_id.astype(str).tolist()==old.obs_id.astype(str).tolist() else "REPRODUCTION_FAILED",
        "test_n":len(test),"max_abs_error":float(delta.max()),"mean_abs_error":float(delta.mean()),"max_threshold":STRICT_MAX,"mean_threshold":STRICT_MEAN,
        "ordered_obs_ids_match":test.obs_id.astype(str).tolist()==old.obs_id.astype(str).tolist(),"new_prediction_hash":arr_hash(p),"frozen_prediction_hash":arr_hash(old.y_pred.to_numpy(float)),
        "context_hash":ensemble_hash(model),"pretrained_weight_sha256":checkpoint_hash(model),"fit_seconds":fitsec,"predict_seconds":predsec,**runtime_info(torch)}
    pd.DataFrame([rec]).to_csv(outpath,index=False,encoding="utf-8-sig")


def run_full_validation():
    seed_path=OUT/"fixed_hashseed_candidate.json"
    hashseed=int(json.loads(seed_path.read_text()).get("python_hash_seed",1)) if seed_path.exists() else 42
    rows=[]
    for task,fold in FAILED:
        for rep in range(1,REPEATS+1):
            p=OUT/f"full_{task}_{fold}_r{rep}.csv"
            if p.exists():
                r=pd.read_csv(p).iloc[0].to_dict();rows.append(r);continue
            env=os.environ.copy();env["PYTHONHASHSEED"]=str(hashseed)
            cmd=[sys.executable,str(Path(__file__).resolve()),"--full-worker",task,fold,str(rep),str(p)]
            try:
                cp=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=300)
                if cp.returncode==0 and p.exists(): rows.append(pd.read_csv(p).iloc[0].to_dict())
                else: rows.append({"task":task,"fold":fold,"replicate":rep,"status":"PROCESS_FAILED","error":cp.stderr[-2000:]})
            except subprocess.TimeoutExpired:
                rows.append({"task":task,"fold":fold,"replicate":rep,"status":"TIMEOUT_300S"})
            save("03_full_fold_reproduction_validation.csv",rows)
            if rows[-1].get("status") not in ("EXACT_REPRODUCTION","REPRODUCTION_FAILED"):
                break
        save("03_full_fold_reproduction_validation.csv",rows)


def original_ale_coverage(base):
    src=pd.read_csv(V3/"ale_1d/ale_1d_by_fold.csv")
    rep=pd.read_csv(V3/"00_inference_reproduction.csv")
    rows=[];cache={}
    for (task,fold,feat),g in src.groupby(["task","fold","feature"],sort=True):
        if (task,fold) not in cache:cache[(task,fold)]=prepare(base,task,fold)
        features,train,test,imp,xtr,xte,ytr,y0,pos,q,frozen,meta=cache[(task,fold)]
        j=features.index(feat); rawtr=train[feat].to_numpy(float); rawq=test[feat].to_numpy(float)
        rawtr=rawtr[np.isfinite(rawtr)];rawq=rawq[np.isfinite(rawq)]
        counts=g.n_samples.to_numpy(int); ns=g.n_stations.to_numpy(int)
        e=np.r_[g.lower.to_numpy(float),g.upper.to_numpy(float)[-1:]] if len(g) else np.array([])
        frozenstatus=rep.loc[(rep.task==task)&(rep.fold==fold),"status"].iloc[0]
        original_edges=np.quantile(xtr[:,j],np.linspace(0,1,11));dedup_edges=np.unique(original_edges)
        sample_pos=base.sample_positions(test,64,30000+int(fold[-1])+100*list(base.TASKS).index(task))
        train_lo=float(np.min(rawtr)) if len(rawtr) else np.nan;train_hi=float(np.max(rawtr)) if len(rawtr) else np.nan
        query_lo=float(np.min(rawq)) if len(rawq) else np.nan;query_hi=float(np.max(rawq)) if len(rawq) else np.nan
        coverage_status="RAW_QUERY_ALL_MISSING" if not len(rawq) else "AUDITED"
        rows.append({"task":task,"fold":fold,"feature":feat,"reference_label":"station-holdout test features; no target labels used",
            "query_n_recorded":int(g.n_samples.sum()),"query_n_total_used_original":64 if g.reference.astype(str).str.contains("test features").all() else np.nan,
            "query_station_count_original":int(test.iloc[sample_pos].station_id.nunique()),"test_feature_n_available":len(rawq),"test_station_count_all":int(test.station_id.nunique()),
            "train_nonmissing_n":len(rawtr),"query_nonmissing_n":len(rawq),"coverage_status":coverage_status,
            "train_min":train_lo,"train_max":train_hi,"train_q01":float(np.quantile(rawtr,.01)) if len(rawtr) else np.nan,"train_q99":float(np.quantile(rawtr,.99)) if len(rawtr) else np.nan,
            "query_min":query_lo,"query_max":query_hi,"query_q01":float(np.quantile(rawq,.01)) if len(rawq) else np.nan,"query_q99":float(np.quantile(rawq,.99)) if len(rawq) else np.nan,
            "original_bins":len(g),"original_empty_bins":int((counts==0).sum()),"original_min_n":int(counts.min()),"original_max_n":int(counts.max()),
            "original_min_station_n":int(ns.min()),"original_median_n":float(np.median(counts)),"requested_training_quantile_bins":10,"duplicate_raw_quantile_edges":int(len(original_edges)-len(dedup_edges)),
            "original_edges_json":json.dumps([[float(a),float(b)] for a,b in zip(g.lower,g.upper)]),"counts_json":json.dumps(counts.tolist()),"stations_json":json.dumps(ns.tolist()),
            "overlap_train_query_fraction":float(np.mean((rawq>=train_lo)&(rawq<=train_hi))) if len(rawq) and len(rawtr) else 0.0,"shap_sample_ids_used_for_ale":False,
            "initial_full_fold_reproduction":frozenstatus,"original_ALE_inference_status":str(pd.read_csv(V3/"00_ale_inference_reproduction.csv").query("task==@task and fold==@fold").status.iloc[0])})
    save("05_ale_1d_original_coverage.csv",rows)
    return src


def adaptive_edges(q: np.ndarray, train: np.ndarray, target_bins=10, min_n=30, min_stations=5, stations=None):
    # Restrict to actual train-query overlap. Candidate intervals are quantiles of query features only.
    if not np.isfinite(train).any() or not np.isfinite(q).any():
        return None,np.zeros(len(q),dtype=bool),None,"NO_COMMON_SUPPORT"
    lo=max(float(np.nanmin(train)),float(np.nanmin(q)));hi=min(float(np.nanmax(train)),float(np.nanmax(q)))
    keep=np.isfinite(q)&(q>=lo)&(q<=hi)
    vals=q[keep];st=stations[keep] if stations is not None else np.arange(len(vals)).astype(str)
    if hi<=lo or len(vals)<min_n: return None,keep,None,"NO_COMMON_SUPPORT"
    best=None
    for nb in range(target_bins,1,-1):
        edges=np.unique(np.quantile(vals,np.linspace(0,1,nb+1)))
        if len(edges)<3: continue
        # Ensure endpoints exactly bound observed common support; repeated quantiles are removed.
        edges[0]=lo;edges[-1]=hi;edges=np.unique(edges)
        bid=np.clip(np.digitize(vals,edges[1:-1],right=True),0,len(edges)-2)
        n=np.bincount(bid,minlength=len(edges)-1); ns=np.array([len(set(st[bid==i])) for i in range(len(edges)-1)])
        if len(n) and np.all(n>=min_n) and np.all(ns>=min_stations): return edges,keep,(n,ns),"SUPPORTED"
        if len(n) and (best is None or len(n)<len(best[0])-1): best=(edges,(n,ns))
    # 若无任一分箱数满足标准，保留最低分辨率(通常2箱)的真实计数，不返回10箱方案。
    if best is None:return None,keep,None,"INSUFFICIENT_SUPPORT"
    return best[0],keep,best[1],"INSUFFICIENT_SUPPORT"


def rebin_ale_1d(base,retry_unsupported=False):
    import torch
    cfg=pd.read_csv(V3/"config"/"chla_surface_ale_features.csv") if (V3/"config/chla_surface_ale_features.csv").exists() else None
    # Use exactly the 16 task-feature pairs recorded in the original ALE output.
    old=pd.read_csv(V3/"ale_1d/ale_1d_by_fold.csv")
    pairs=old[["task","feature"]].drop_duplicates().sort_values(["task","feature"])
    outpath=OUT/"06_ale_1d_rebinning_comparison.csv"
    sumpath=OUT/"06_ale_1d_rebinning_summary.csv"
    allrows=pd.read_csv(outpath).to_dict("records") if outpath.exists() else []
    summary=pd.read_csv(sumpath).to_dict("records") if sumpath.exists() else []
    completed={(str(r.get("task")),str(r.get("fold")),str(r.get("feature"))) for r in summary if not retry_unsupported or r.get("status")=="SUPPORTED"}
    # 先按task-fold循环，一折只构造一次输入与上下文，再处理该折全部已有变量。
    task_features={t:sorted(g.feature.unique()) for t,g in pairs.groupby("task",sort=True)}
    for task,featlist in task_features.items():
      for fold in [f"fold_{i}" for i in range(1,6)]:
        todo=[feat for feat in featlist if (task,fold,feat) not in completed]
        if not todo: continue
        prepared=prepare(base,task,fold)
        features,train,test,imp,xtr,xte,ytr,y0,pos,qprobe,frozen,meta=prepared
        model=estimator(base,xtr,ytr)
        for feat in todo:
          j=features.index(feat)
          # 支持度按原始非缺失环境观测计算，不将median填补值冒充实际观测。
          trainv=train[feat].to_numpy(float);query=test[feat].to_numpy(float); stations=test.station_id.astype(str).to_numpy()
          edges,keep,cover,status=adaptive_edges(query,trainv,10,30,5,stations)
          if edges is None:
              summary.append({"task":task,"fold":fold,"feature":feat,"status":status,"bins":0,"query_n_total":len(test),"query_n_common":int(keep.sum()) if keep is not None else 0});save("06_ale_1d_rebinning_summary.csv",summary);continue
          n,ns=cover; q=xte[keep];st=stations[keep]; bid=np.clip(np.digitize(query[keep],edges[1:-1],right=True),0,len(edges)-2)
          if status!="SUPPORTED":
              summary.append({"task":task,"fold":fold,"feature":feat,"status":status,"bins":len(edges)-1,"query_n_total":len(test),"query_n_common":len(q),"bin_counts_json":json.dumps(n.tolist()),"bin_stations_json":json.dumps(ns.tolist()),"support_rule":"n>=30 & stations>=5"});save("06_ale_1d_rebinning_summary.csv",summary);continue
          lower=q.copy();upper=q.copy();lower[:,j]=edges[bid];upper[:,j]=edges[bid+1]
          t=time.perf_counter(); p=pred_original(model,np.vstack([lower,upper]),np.nanmin(y0)>=0);elapsed=time.perf_counter()-t
          local=p[len(q):]-p[:len(q)]
          le=np.array([local[bid==b].mean() for b in range(len(edges)-1)])
          ale=np.cumsum(le); ale-=np.average(ale,weights=n)
          for b in range(len(edges)-1):
              allrows.append({"task":task,"fold":fold,"feature":feat,"bin":b+1,"lower":edges[b],"upper":edges[b+1],"center":(edges[b]+edges[b+1])/2,
                  "local_effect":le[b],"ale":ale[b],"n_samples":int(n[b]),"n_stations":int(ns[b]),"status":"SUPPORTED","query_n_total":len(test),"query_n_common":len(q),
                  "edge_source":"quantiles of unlabeled held-out query feature within train-query overlap","min_n_rule":30,"min_station_rule":5,"context_hash":ensemble_hash(model),"elapsed_seconds":elapsed,
                  "initial_reproduction_status":str(pd.read_csv(V3/"00_inference_reproduction.csv").query("task==@task and fold==@fold").status.iloc[0])})
          summary.append({"task":task,"fold":fold,"feature":feat,"status":"SUPPORTED","bins":len(edges)-1,"query_n_total":len(test),"query_n_common":len(q),
              "bin_counts_json":json.dumps(n.tolist()),"bin_stations_json":json.dumps(ns.tolist()),"amplitude":float(ale.max()-ale.min()),"elapsed_seconds":elapsed})
          save("06_ale_1d_rebinning_comparison.csv",allrows)
          save("06_ale_1d_rebinning_summary.csv",summary)
    save("06_ale_1d_rebinning_comparison.csv",allrows)
    save("06_ale_1d_rebinning_summary.csv",summary)


def best_supported_rectangle(mask):
    # Return largest contiguous rectangular all-supported block, if at least 2x2.
    nr,nc=mask.shape;best=None;area=0
    for r0 in range(nr):
      for r1 in range(r0+2,nr+1):
       for c0 in range(nc):
        for c1 in range(c0+2,nc+1):
         a=(r1-r0)*(c1-c0)
         if a>area and mask[r0:r1,c0:c1].all(): best=(r0,r1,c0,c1);area=a
    return best


def finalize_ale_1d_comparison():
    """新旧曲线限定在共同支持范围对照，并输出折间形态/幅度稳定性。"""
    newp=OUT/"06_ale_1d_rebinning_comparison.csv"; sump=OUT/"06_ale_1d_rebinning_summary.csv"
    if not newp.exists() or not sump.exists(): return
    new=pd.read_csv(newp);old=pd.read_csv(V3/"ale_1d/ale_1d_by_fold.csv");summary=pd.read_csv(sump)
    old_rows=[]; fold_curves={}
    for r in summary.itertuples(index=False):
        if r.status!="SUPPORTED": continue
        n=new.query("task==@r.task and fold==@r.fold and feature==@r.feature").sort_values("center")
        o=old.query("task==@r.task and fold==@r.fold and feature==@r.feature").sort_values("center")
        if len(n)<2 or len(o)<2: continue
        lo=max(float(n.center.min()),float(o.center.min()));hi=min(float(n.center.max()),float(o.center.max()))
        m=(n.center>=lo)&(n.center<=hi)
        if hi<=lo or int(m.sum())<2: rho=np.nan;oa=np.nan;na=np.nan;direction="NO_COMMON_CURVE_SUPPORT";major=np.nan
        else:
            xn=n.loc[m,"center"].to_numpy(float);yn=n.loc[m,"ale"].to_numpy(float);yo=np.interp(xn,o.center,o.ale)
            rho=float(pd.Series(yn).corr(pd.Series(yo),method="spearman"))
            oa=float(np.max(yo)-np.min(yo));na=float(np.max(yn)-np.min(yn))
            slope=float(np.polyfit(xn,yn,1)[0]);direction="INCREASING" if slope>0 else ("DECREASING" if slope<0 else "FLAT")
            if len(xn)>1: major=float((xn[np.argmax(np.abs(np.diff(yn)))]+xn[np.argmax(np.abs(np.diff(yn)))+1])/2)
        fold_curves[(r.task,r.feature,r.fold)]=(float(n.center.min()),float(n.center.max()),n.center.to_numpy(float),n.ale.to_numpy(float))
        old_rows.append({"task":r.task,"fold":r.fold,"feature":r.feature,"status":r.status,"new_curve_direction":direction,
            "original_amplitude_common_range":oa,"rebinned_amplitude_common_range":na,"amplitude_change":na-oa if np.isfinite(na) and np.isfinite(oa) else np.nan,
            "common_curve_min":lo,"common_curve_max":hi,"common_curve_points":int(m.sum()) if hi>lo else 0,
            "spearman_new_vs_original_common_range":rho,"largest_new_adjacent_change_midpoint":major})
    comp=pd.DataFrame(old_rows)
    if len(comp):
        cross=[]
        for (task,feature),g in comp.groupby(["task","feature"]):
            folds=[f for f in g.fold if (task,feature,f) in fold_curves]; rhos=[];amps=[]
            for i,f1 in enumerate(folds):
                a=fold_curves[(task,feature,f1)];amps.append(float(g.loc[g.fold.eq(f1),"rebinned_amplitude_common_range"].iloc[0]))
                for f2 in folds[i+1:]:
                    b=fold_curves[(task,feature,f2)];lo=max(a[0],b[0]);hi=min(a[1],b[1])
                    if hi<=lo: continue
                    xx=np.linspace(lo,hi,100);rho=np.corrcoef(np.interp(xx,a[2],a[3]),np.interp(xx,b[2],b[3]))[0,1]
                    if np.isfinite(rho):rhos.append(float(rho))
            cross.append({"task":task,"feature":feature,"supported_folds":len(folds),"median_pairwise_fold_curve_spearman":float(np.median(rhos)) if rhos else np.nan,
                "amplitude_cv_across_supported_folds":float(np.std(amps,ddof=1)/np.mean(amps)) if len(amps)>1 and np.mean(amps)!=0 else np.nan,
                "fold_amplitudes_json":json.dumps(amps)})
        comp=comp.merge(pd.DataFrame(cross),on=["task","feature"],how="left")
    save("06_ale_1d_curve_comparison.csv",comp)


def deduplicate_ale_1d_summary():
    p=OUT/"06_ale_1d_rebinning_summary.csv"
    if p.exists():
        d=pd.read_csv(p);d=d.drop_duplicates(["task","fold","feature"],keep="last")
        d.to_csv(p,index=False,encoding="utf-8-sig")


def ale2d_sensitivity(base,recompute_corrected=False):
    import torch
    pairs=pd.read_csv(CLOSURE/"07_ale_2d_reliability_review.csv")[["task","feature_1","feature_2"]].drop_duplicates()
    outpath=OUT/"07_ale_2d_grid_sensitivity.csv";surfpath=OUT/"07_ale_2d_supported_surfaces.csv"
    rows=pd.read_csv(outpath).to_dict("records") if outpath.exists() else []
    curves=pd.read_csv(surfpath).to_dict("records") if surfpath.exists() else []
    if recompute_corrected:
        # The earlier attempt used mismatched axis weights; retain its files separately,
        # but exclude every computed surface from this corrected rerun.
        rows=[r for r in rows if r.get("surface_status")!="COMPUTED_ON_FULLY_SUPPORTED_RECTANGLE"]
        curves=[]
    completed={(str(r.get("task")),str(r.get("fold")),str(r.get("feature_1")),str(r.get("feature_2")),str(r.get("grid_requested"))) for r in rows}
    current_key=None;prepared=None;model=None
    for pr in pairs.itertuples(index=False):
      for fold in [f"fold_{i}" for i in range(1,6)]:
        if current_key!=(pr.task,fold):
            current_key=(pr.task,fold);prepared=prepare(base,pr.task,fold);model=None
        features,train,test,imp,xtr,xte,ytr,y0,pos,qprobe,frozen,meta=prepared
        i,j=features.index(pr.feature_1),features.index(pr.feature_2)
        for bins in (4,5):
          key=(str(pr.task),str(fold),str(pr.feature_1),str(pr.feature_2),f"{bins}x{bins}")
          if key in completed: continue
          train_a=train[pr.feature_1].to_numpy(float);train_b=train[pr.feature_2].to_numpy(float)
          train_a=train_a[np.isfinite(train_a)];train_b=train_b[np.isfinite(train_b)]
          query_a=test[pr.feature_1].to_numpy(float);query_b=test[pr.feature_2].to_numpy(float)
          if not len(train_a) or not len(train_b):
            rows.append({"task":pr.task,"fold":fold,"feature_1":pr.feature_1,"feature_2":pr.feature_2,"grid_requested":f"{bins}x{bins}","status":"INSUFFICIENT_SUPPORT","reason":"no raw nonmissing training observations"});continue
          ea=np.unique(np.quantile(train_a,np.linspace(0,1,bins+1))); eb=np.unique(np.quantile(train_b,np.linspace(0,1,bins+1)))
          if len(ea)<3 or len(eb)<3:
            rows.append({"task":pr.task,"fold":fold,"feature_1":pr.feature_1,"feature_2":pr.feature_2,"grid_requested":f"{bins}x{bins}","status":"INSUFFICIENT_SUPPORT","reason":"duplicate train quantile edges"});save("07_ale_2d_grid_sensitivity.csv",rows);continue
          # Restrict to overlapping train-query support and score empirical joint grid on all test covariates.
          # 只把真实未缺失的held-out环境取值计入联合支持；模型查询仍用既定median预处理输入。
          mask=np.isfinite(query_a)&np.isfinite(query_b)&(query_a>=ea[0])&(query_a<=ea[-1])&(query_b>=eb[0])&(query_b<=eb[-1])
          qa=query_a[mask];qb=query_b[mask];sts=test.loc[mask,"station_id"].astype(str).to_numpy()
          ca=np.clip(np.digitize(qa,ea[1:-1],right=True),0,len(ea)-2);cb=np.clip(np.digitize(qb,eb[1:-1],right=True),0,len(eb)-2)
          counts=np.zeros((len(ea)-1,len(eb)-1),int);nsta=np.zeros_like(counts)
          for a in range(counts.shape[0]):
            for b in range(counts.shape[1]):
              m=(ca==a)&(cb==b);counts[a,b]=int(m.sum());nsta[a,b]=len(set(sts[m]))
          support=(counts>=30)&(nsta>=5);rect=best_supported_rectangle(support)
          rho=float(pd.Series(qa).corr(pd.Series(qb),method="spearman")) if len(qa)>2 else np.nan
          rec={"task":pr.task,"fold":fold,"feature_1":pr.feature_1,"feature_2":pr.feature_2,"grid_requested":f"{bins}x{bins}",
              "grid_actual":f"{len(ea)-1}x{len(eb)-1}","n_query_common":int(mask.sum()),"cell_n_json":json.dumps(counts.tolist()),"cell_station_n_json":json.dumps(nsta.tolist()),
              "feature_1_train_quantile_edges_json":json.dumps(ea.tolist()),"feature_2_train_quantile_edges_json":json.dumps(eb.tolist()),
              "feature_1_joint_query_range_json":json.dumps([float(np.min(qa)),float(np.max(qa))]) if len(qa) else "[]",
              "feature_2_joint_query_range_json":json.dumps([float(np.min(qb)),float(np.max(qb))]) if len(qb) else "[]",
              "supported_cell_fraction_n30_station5":float(support.mean()),"empty_cell_fraction":float((counts==0).mean()),"sparse_cell_fraction_n_lt30_or_station_lt5":float((~support).mean()),
              "spearman_joint_correlation":rho,"best_full_support_rectangle":str(rect),"status":"INSUFFICIENT_SUPPORT" if rect is None else "PARTIALLY_SUPPORTED","surface_status":"NOT_COMPUTED_YET"}
          # Only compute true second-order ALE in a fully observed contiguous rectangle; no empty-cell filling.
          if rect is not None:
            r0,r1,c0,c1=rect;aa=ea[r0:r1+1];bb=eb[c0:c1+1];mrect=mask & (query_a>=aa[0])&(query_a<=aa[-1])&(query_b>=bb[0])&(query_b<=bb[-1]);xr=xte[mrect];
            sta=test.loc[mrect,"station_id"].astype(str).to_numpy();bi=np.clip(np.digitize(xr[:,i],aa[1:-1],right=True),0,len(aa)-2);bj=np.clip(np.digitize(xr[:,j],bb[1:-1],right=True),0,len(bb)-2)
            variants=[]
            for da,db in [(0,0),(1,0),(0,1),(1,1)]:
              z=xr.copy();z[:,i]=aa[bi+da];z[:,j]=bb[bj+db];variants.append(z)
            if model is None:model=estimator(base,xtr,ytr)
            ps=pred_original(model,np.vstack(variants),np.nanmin(y0)>=0);nq=len(xr)
            sec=ps[3*nq:]-ps[nq:2*nq]-ps[2*nq:3*nq]+ps[:nq]
            local=np.full((r1-r0,c1-c0),np.nan);cnt=np.zeros_like(local,int);stations_grid=np.zeros_like(local,int)
            for a in range(local.shape[0]):
              for b in range(local.shape[1]):
                mm=(bi==a)&(bj==b);cnt[a,b]=int(mm.sum());stations_grid[a,b]=len(set(sta[mm]));local[a,b]=float(sec[mm].mean()) if mm.any() else np.nan
            if np.all(cnt>=30) and np.all(stations_grid>=5):
              surface=np.cumsum(np.cumsum(local,axis=0),axis=1);w=cnt.astype(float)
              # axis=1 averages across columns and therefore needs column weights;
              # axis=0 averages across rows and needs row weights.
              rowm=np.average(surface,axis=1,weights=np.maximum(w.sum(0),1))
              colm=np.average(surface,axis=0,weights=np.maximum(w.sum(1),1))
              surface=surface-rowm[:,None]-colm[None,:]+np.average(surface,weights=w)
              surface-=np.average(surface,weights=w)
              for a in range(surface.shape[0]):
               for b in range(surface.shape[1]):curves.append({"task":pr.task,"fold":fold,"feature_1":pr.feature_1,"feature_2":pr.feature_2,"grid":f"{bins}x{bins}","bin_1":a+1+r0,"bin_2":b+1+c0,"feature_1_center":float((aa[a]+aa[a+1])/2),"feature_2_center":float((bb[b]+bb[b+1])/2),"ale_2d":surface[a,b],"n_samples":cnt[a,b],"n_stations":stations_grid[a,b],"context_hash":ensemble_hash(model),"centered_weighted_mean":float(np.average(surface,weights=w))})
              rec["surface_status"]="COMPUTED_ON_FULLY_SUPPORTED_RECTANGLE";rec["surface_amplitude"]=float(np.nanmax(surface)-np.nanmin(surface));rec["context_hash"]=ensemble_hash(model)
            else:
              rec["surface_status"]="COMPUTATION_ERROR";rec["reason"]="support rectangle changed under exact query mask; refused to fill cells"
          rows.append(rec);completed.add(key);save("07_ale_2d_grid_sensitivity.csv",rows);save("07_ale_2d_supported_surfaces.csv",curves)
    save("07_ale_2d_grid_sensitivity.csv",rows)
    save("07_ale_2d_supported_surfaces.csv",curves)


def finalize_ale_2d_summary():
    p=OUT/"07_ale_2d_grid_sensitivity.csv";sp=OUT/"07_ale_2d_supported_surfaces.csv"
    if not p.exists():return
    d=pd.read_csv(p);sur=pd.read_csv(sp) if sp.exists() else pd.DataFrame();out=[]
    for (task,f1,f2),g in d.groupby(["task","feature_1","feature_2"]):
        for grid,gg in g.groupby("grid_requested"):
            supported=gg[gg.status.ne("INSUFFICIENT_SUPPORT")];computed=gg[gg.surface_status.eq("COMPUTED_ON_FULLY_SUPPORTED_RECTANGLE")] if "surface_status" in gg else gg.iloc[0:0]
            physical=[]
            for col in ("feature_1_joint_query_range_json","feature_2_joint_query_range_json"):
                vals=[json.loads(x) for x in supported[col].dropna() if isinstance(x,str) and x.startswith("[") and json.loads(x)]
                if vals: physical.append([max(v[0] for v in vals),min(v[1] for v in vals)])
                else:physical.append([np.nan,np.nan])
            shape_corr=[]
            if len(sur):
                ss=sur[(sur.task==task)&(sur.feature_1==f1)&(sur.feature_2==f2)&(sur.grid==grid)]
                folds=sorted(ss.fold.unique())
                for ix,fa in enumerate(folds):
                    aa=ss[ss.fold==fa].set_index(["bin_1","bin_2"]).ale_2d
                    for fb in folds[ix+1:]:
                        bb=ss[ss.fold==fb].set_index(["bin_1","bin_2"]).ale_2d;ixs=aa.index.intersection(bb.index)
                        if len(ixs)>=4:
                            rho=float(pd.Series(aa.loc[ixs].to_numpy()).corr(pd.Series(bb.loc[ixs].to_numpy()),method="spearman"))
                            if np.isfinite(rho):shape_corr.append(rho)
            out.append({"task":task,"feature_1":f1,"feature_2":f2,"grid_requested":grid,"folds_audited":int(gg.fold.nunique()),
                "folds_with_any_contiguous_supported_rectangle":int(len(supported)),"folds_with_computed_second_order_surface":int(len(computed)),
                "median_supported_cell_fraction":float(supported.supported_cell_fraction_n30_station5.median()) if len(supported) else 0.0,
                "median_sparse_cell_fraction":float(gg.sparse_cell_fraction_n_lt30_or_station_lt5.median()) if "sparse_cell_fraction_n_lt30_or_station_lt5" in gg else 1.0,
                "common_feature_1_joint_range_json":json.dumps(physical[0]),"common_feature_2_joint_range_json":json.dumps(physical[1]),
                "median_pairwise_percentile_grid_spearman":float(np.median(shape_corr)) if shape_corr else np.nan,"pairwise_correlations_json":json.dumps(shape_corr),
                "status":"INSUFFICIENT_SUPPORT" if not len(supported) else ("PARTIALLY_SUPPORTED" if len(computed)<int(gg.fold.nunique()) else "PARTIALLY_SUPPORTED")})
    pd.DataFrame(out).to_csv(OUT/"07_ale_2d_fold_stability_summary.csv",index=False,encoding="utf-8-sig")
    errors=pd.DataFrame([{"task":"chla_surface","fold":"fold_2","feature_1":"ssrd_sum_3d","feature_2":"wind_speed_x_rain14d","grid_requested":"5x5",
        "initial_status":"COMPUTATION_ERROR","error":"The initial ALE-2D weighted-centering implementation used axis-1 weights with row totals, causing a shape mismatch on a non-square supported rectangle; square rectangles also used the wrong marginal weight orientation.",
        "impact":"Three surfaces saved before the error had incorrect weighting and are not valid final results; the first failing cell was the next non-square rectangle in this pair/fold sequence.",
        "correction":"Corrected axis-1 to use column totals and axis-0 to use row totals; recomputed all previously computed surfaces.",
        "preserved_pre_correction_files":"07_ale_2d_grid_pre_correction.csv;07_ale_2d_surfaces_pre_correction.csv",
        "corrected_weighted_center_max_abs":float(sur.centered_weighted_mean.abs().max()) if len(sur) and "centered_weighted_mean" in sur else np.nan,
        "corrected_run_status":"COMPUTATION_ERROR_CORRECTED"}])
    errors.to_csv(OUT/"07_ale_2d_computation_errors.csv",index=False,encoding="utf-8-sig")


def summarize_shap_sensitivity():
    # A/B from prior closure; original calculations remain untouched.
    d=pd.read_csv(ROOT/"runs/final_all_experiments_closure_v1/03_shap_validation_and_sensitivity.csv")
    rows=[]
    for (task,feat),g in d.groupby(["task","feature"]):
      a=g.loc[g.analysis_set.eq("ALL_20_FOLDS")];b=g.loc[g.analysis_set.eq("STRICT_PASS_FOLDS_ONLY")]
      if len(a) and len(b): rows.append({"task":task,"feature":feat,"rank_all20":int(a.iloc[0]["rank"]),"rank_initial_strict":int(b.iloc[0]["rank"]),"rank_delta":int(b.iloc[0]["rank"]-a.iloc[0]["rank"]),"top10_all20":bool(a.iloc[0]["top10"]),"top10_initial_strict":bool(b.iloc[0]["top10"]),"n_strict_folds":int(b.iloc[0]["n_folds"]),"all_strict_spearman":float(a.iloc[0]["all_vs_strict_rank_spearman"]),"top10_jaccard":float(a.iloc[0]["all_vs_strict_top10_jaccard"]),"six_source_relative_change":float(a.iloc[0]["source_group_relative_contribution"]-b.iloc[0]["source_group_relative_contribution"])})
    # 此轮没有确立可用于解释重算的三次稳定全测试推理状态；显式标注未重算。
    probe=pd.read_csv(OUT/"02_determinism_experiment_results.csv") if (OUT/"02_determinism_experiment_results.csv").exists() else pd.DataFrame()
    task_hashes=probe.groupby("task").context_hash.nunique().to_dict() if len(probe) and "context_hash" in probe else {}
    for r in rows:
        r["observed_context_hash_count_across_diagnostics"]=int(task_hashes.get(r["task"],0))
        r["new_stable_state_shap_recomputed"]=False
        r["reason_not_recomputed"]="No PYTHONHASHSEED candidate completed three independent strict probe runs; no stable reconstruction state or full-test validation was established."
    save("04_shap_reconstruction_sensitivity.csv",rows)


def write_reports():
    probe=pd.read_csv(OUT/"02_determinism_experiment_results.csv") if (OUT/"02_determinism_experiment_results.csv").exists() else pd.DataFrame()
    full=pd.read_csv(OUT/"03_full_fold_reproduction_validation.csv") if (OUT/"03_full_fold_reproduction_validation.csv").exists() else pd.DataFrame()
    cov=pd.read_csv(OUT/"05_ale_1d_original_coverage.csv") if (OUT/"05_ale_1d_original_coverage.csv").exists() else pd.DataFrame()
    rebin=pd.read_csv(OUT/"06_ale_1d_rebinning_summary.csv") if (OUT/"06_ale_1d_rebinning_summary.csv").exists() else pd.DataFrame()
    grid=pd.read_csv(OUT/"07_ale_2d_grid_sensitivity.csv") if (OUT/"07_ale_2d_grid_sensitivity.csv").exists() else pd.DataFrame()
    order=pd.read_csv(OUT/"02_context_order_pairwise_delta.csv") if (OUT/"02_context_order_pairwise_delta.csv").exists() else pd.DataFrame()
    hrows=[]
    for hp in OUT.glob("hashorder_*_seed*.csv"):
        try:hrows.append(pd.read_csv(hp).iloc[0].to_dict())
        except Exception:pass
    hpairs=pd.DataFrame(hrows)
    hpass=int(((hpairs.max_abs_vs_frozen_probe<=STRICT_MAX)&(hpairs.mean_abs_vs_frozen_probe<=STRICT_MEAN)).sum()) if len(hpairs) else 0
    full_run=full[full.get("condition",pd.Series(dtype=str)).ne("full_test_not_run")] if len(full) else full
    if len(full_run)==15 and full_run.status.eq("EXACT_REPRODUCTION").all(): repro="STABLE_REPRODUCTION_CONFIRMED"
    elif len(full_run) and full_run.status.eq("EXACT_REPRODUCTION").any(): repro="REPRODUCTION_VARIABILITY_REDUCED"
    else: repro="REPRODUCTION_UNRESOLVED"
    if len(rebin) and rebin.status.eq("SUPPORTED").all(): ale="ALE_SUPPORT_IMPROVED"
    elif len(rebin) and rebin.status.eq("SUPPORTED").any(): ale="ALE_PARTIALLY_SUPPORTED"
    else: ale="ALE_INSUFFICIENT_SUPPORT"
    # 明确落盘：没有满足三次独立稳定探针的推理状态，故不启动完整测试集复现。
    fullpath=OUT/"03_full_fold_reproduction_validation.csv"
    if not fullpath.exists():
        prior=pd.read_csv(V3/"00_inference_reproduction.csv")
        rows=[]
        for task,fold in FAILED:
            old=prior.query("task==@task and fold==@fold")
            rows.append({"task":task,"fold":fold,"replicate":0,"condition":"full_test_not_run",
                "status":"NOT_RUN_NO_THREE_REPEAT_STABLE_PILOT","initial_reproduction_status":str(old.status.iloc[0]) if len(old) else "UNKNOWN",
                "max_threshold":STRICT_MAX,"mean_threshold":STRICT_MEAN,
                "reason":"Fixed hashseed pilot exactly matched only 2/3 runs; third independent worker exceeded the 240-second diagnostic budget. No stable inference state established; the formal 3-repeat full-test validation was therefore not launched."})
        pd.DataFrame(rows).to_csv(fullpath,index=False,encoding="utf-8-sig")
    report=f"""# TabICLv2根因与剩余限制

## 推理复现

复现状态：`{repro}`。严格阈值保持 max≤1e-6、mean≤1e-7。原始首次失败记录不改写。

- 五个失败折的训练/测试obs_id顺序与集合、77特征顺序、原始/填补训练矩阵、median统计量、训练标签、模型权重SHA-256、目标变换、上下文样本数与顺序、上下文/预测哈希均记录于`01_context_reconstruction_audit.csv`和`02_determinism_experiment_results.csv`。
- A/B/C/A4/D各3次探针已完成；同一已构造上下文与同条件内重复预测的差异为0。Chl-a fold1、fold2的C新实例首个重复分别出现约5.72e-3和8.84e-3最大差异，后续重复在不同状态下收敛到约2.94e-6、5.42e-6（仍超过max阈值）。
- TabICLv2 2.2.0源码`tabicl/_sklearn/preprocessing.py:1129`用`list(set(...))`生成归一化方法顺序。独立进程实测`none,power`与`power,none`产生不同上下文哈希；Chl-a fold1/2在前者下32样本探针分别达到1.78e-15/4.44e-16最大差异，后者分别约2.94e-6/5.42e-6。该结果强烈支持Python哈希派生的集成视图顺序是差异来源之一，但不能证明它解释全部五折、全测试样本差异。
- 固定Python/NumPy/PyTorch/CUDA随机状态，仅改变`PYTHONHASHSEED=1`与`42`的成对试验完成{len(order)}/{len(FAILED)}折；权重哈希相同而归一化视图顺序和上下文哈希不同。32探针成对预测最大误差范围为{float(order.max_pairwise_abs_error.min()) if len(order) else float('nan'):.3g}至{float(order.max_pairwise_abs_error.max()) if len(order) else float('nan'):.3g}，平均误差最高为{float(order.mean_pairwise_abs_error.max()) if len(order) else float('nan'):.3g}。十个固定随机状态的hash-order输出仅{hpass}/10同时满足冻结探针的max/mean阈值；这表明哈希顺序和全局随机状态都会影响预测，且未找到可用于全部五折的合格统一重建设置。明细见`02_context_order_pairwise_delta.csv`与`hashorder_*_seed*.csv`。
- 固定`PYTHONHASHSEED=1`的Chl-a fold1试点只完成2/3次（三次中的第三次超过240秒诊断预算）；已完成两次均在32探针上精确匹配，但不构成预先要求的三次稳定证据。禁用AMP与启用确定性算法的fold1试点仍未达到冻结探针阈值；CPU试点180秒超时。
- 独立全测试集三次复现实际运行数：{len(full_run)}；延期占位行：{int(len(full)-len(full_run)) if len(full) else 0}。因为没有稳定三次探针，不启动全测试集正式验证；不改变阈值，五个首次失败状态继续保留。

## 一维ALE支持度

ALE状态：`{ale}`。原始分箱覆盖及重复值、空箱、训练—查询支持差异见`05_ale_1d_original_coverage.csv`。新分箱敏感性只使用测试环境特征与训练—查询共同范围，不读取测试目标标签；初始门槛为每箱至少30条观测及5个站点。新增曲线在`06_ale_1d_rebinning_comparison.csv`，未通过覆盖筛选者不生成曲线。

- 原始变量—折记录：{len(cov)}。
- 新分箱变量—折通过记录：{int(rebin.status.eq('SUPPORTED').sum()) if len(rebin) else 0}/{len(rebin)}。
- 未支持组合集中在夜光代理：支持区域通常仅有1—2个监测站；Secchi fold4无原始共同非缺失观测。减少至2箱仍不满足每箱≥5站点。
- 65条支持曲线的新旧共同范围Spearman任务中位数约0.80—0.93；曲线折间排名相关性较高，但幅度CV约0.18—0.63，且Secchi部分曲线方向变化，不能据此称影响幅度稳定。详细方向、幅度与共同范围见`06_ale_1d_curve_comparison.csv`。
- 该65条曲线使用本轮当前重建上下文，未通过全测试集严格复现确认；新旧曲线差异只能作为覆盖敏感性结果。

## 二维ALE

七组既有候选以4×4及5×5训练分位网格重审，网格覆盖、站点数、稀疏率及Spearman相关记录于`07_ale_2d_grid_sensitivity.csv`。只在至少2×2的完全有支持连续矩形上计算二阶ALE；其余标记为覆盖不足，不插值、不填空。通过范围的面数据保存于`07_ale_2d_supported_surfaces.csv`。

- 网格审计记录：{len(grid)}。
- 真正重新计算的支持子面：{int(grid.surface_status.eq('COMPUTED_ON_FULLY_SUPPORTED_RECTANGLE').sum()) if len(grid) and 'surface_status' in grid else 0}。
- 新实现的中心化轴权重错误已修正，预修正曲面另存为`07_ale_2d_*_pre_correction.csv`；本轮曲面加权中心最大绝对残差为1.38e-17。`07_ale_2d_fold_stability_summary.csv`以分位网格索引对齐计算形态相关，非物理坐标插值。
- Secchi `ntl_avg_rad_roll3 × ssrd_mean_7d`五折均为`INSUFFICIENT_SUPPORT`；其余有连续支持子矩形的组合仍因支持域仅覆盖联合分布的一部分而标记`PARTIALLY_SUPPORTED`。不同网格分辨率的折间形态可不一致，不作稳定物理耦合结论。

剩余限制：预训练权重哈希、77特征和预处理输入未见差异；差异已定位至新模型/上下文重建阶段，并发现Python集合顺序的可复现关联。CUDA/AMP、内部随机状态及batch执行细节仍不能完全排除。CPU/无AMP/确定性设置仅用于诊断，不替代冻结预测。
"""
    (OUT/"08_root_cause_and_remaining_limitations.md").write_text(report,encoding="utf-8")
    decision=f"""# 最终修复决策

## 状态

- 复现状态：`{repro}`
- ALE状态：`{ale}`

## 验收结论

原始CLOSED_WITH_LIMITATIONS及所有首次失败折记录均保留。全测试集复现仅按事前max绝对误差≤{STRICT_MAX:g}且mean绝对误差≤{STRICT_MEAN:g}判断；需每折3次独立运行连续通过，才可称严格稳定。新分箱初始支持标准为每箱≥30条查询观测且≥5个站点，这是本轮敏感性标准，不替代原始ALE标准。

五个失败折均完成三次A/B/C/A4/D小样本诊断；相同上下文重复稳定，但新模型初始化出现波动。固定所有RNG后仅改变Python哈希种子的成对试验显示，五折均有不同集成视图顺序/上下文哈希，探针最大差值介于{float(order.max_pairwise_abs_error.min()) if len(order) else float('nan'):.3g}与{float(order.max_pairwise_abs_error.max()) if len(order) else float('nan'):.3g}之间。此为重建差异的直接来源证据之一，不代表新推理已通过全测试集验收。

没有重新训练或微调预训练权重，没有改变77特征、正式split或冻结预测/指标；所有替代设备和禁用AMP结果只用于定位数值问题。现有SHAP未在本轮不稳定的重建状态下重算；旧曲线已通过新分箱覆盖敏感性复核，但未确认严格冻结推理状态。二维ALE不支持区域不外推、不填补，且不将统计交互解释为因果机制。

详细数据见01至08 CSV及根因报告。本轮不修改Word，不扩展其他模型实验。
"""
    (OUT/"09_final_repair_decision.md").write_text(decision,encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--probe-fold",nargs=2,metavar=("TASK","FOLD"))
    p.add_argument("--probe-all",action="store_true")
    p.add_argument("--config-diagnostics",action="store_true")
    p.add_argument("--hashseed-candidate",type=int)
    p.add_argument("--hash-order-pairwise",action="store_true")
    p.add_argument("--worker",nargs=5,metavar=("TASK","FOLD","CONDITION","REPLICATE","OUTFILE"))
    p.add_argument("--full-worker",nargs=4,metavar=("TASK","FOLD","REPLICATE","OUTFILE"))
    p.add_argument("--audit-inputs",action="store_true")
    p.add_argument("--full-validate",action="store_true")
    p.add_argument("--ale-audit",action="store_true")
    p.add_argument("--ale-rebin",action="store_true")
    p.add_argument("--ale-rebin-retry-unsupported",action="store_true")
    p.add_argument("--finalize-ale1d",action="store_true")
    p.add_argument("--ale-2d",action="store_true")
    p.add_argument("--ale-2d-recompute-corrected",action="store_true")
    p.add_argument("--finalize-ale2d",action="store_true")
    p.add_argument("--shap-sensitivity",action="store_true")
    p.add_argument("--refresh-manifest",action="store_true")
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.worker:
        task,fold,cond,rep,out=a.worker;probe_worker(task,fold,cond,int(rep),out);return
    if a.full_worker:
        task,fold,rep,out=a.full_worker;full_fold_worker(task,fold,int(rep),out);return
    base=load_base()
    if a.audit_inputs: context_and_input_audit(base)
    if a.probe_all: probe_all_folds()
    if a.probe_fold: probe_fold(base,*a.probe_fold)
    if a.config_diagnostics: run_config_diagnostics()
    if a.hashseed_candidate is not None: run_hashseed_candidate(a.hashseed_candidate)
    if a.hash_order_pairwise: run_hash_order_pairwise()
    if a.full_validate: run_full_validation()
    if a.ale_audit: original_ale_coverage(base)
    if a.ale_rebin: rebin_ale_1d(base);finalize_ale_1d_comparison()
    if a.ale_rebin_retry_unsupported: rebin_ale_1d(base,retry_unsupported=True);deduplicate_ale_1d_summary();finalize_ale_1d_comparison()
    if a.finalize_ale1d: deduplicate_ale_1d_summary();finalize_ale_1d_comparison()
    if a.ale_2d: ale2d_sensitivity(base);finalize_ale_2d_summary()
    if a.ale_2d_recompute_corrected: ale2d_sensitivity(base,recompute_corrected=True);finalize_ale_2d_summary()
    if a.finalize_ale2d: finalize_ale_2d_summary()
    if a.shap_sensitivity: summarize_shap_sensitivity()
    if a.refresh_manifest:
        import torch
        m=json.loads((OUT/"00_run_manifest.json").read_text(encoding="utf-8")) if (OUT/"00_run_manifest.json").exists() else {}
        m["script_sha256_current"]=file_hash(Path(__file__).resolve());m["manifest_refreshed_local_date"]="2026-10-21";m["run_environment_current"]=runtime_info(torch)
        m["fixed_hashseed_candidate"]={"python_hash_seed":1,"probe_runs_completed":2,"probe_runs_required":3,"stable_full_validation_started":False}
        (OUT/"00_run_manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8")
    write_reports()


if __name__=="__main__":
    main()
