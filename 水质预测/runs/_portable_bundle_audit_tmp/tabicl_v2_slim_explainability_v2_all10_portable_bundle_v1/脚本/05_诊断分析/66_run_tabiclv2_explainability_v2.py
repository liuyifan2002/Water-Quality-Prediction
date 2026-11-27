"""TabICLv2 V2：固定batch的关键变量置换重要性与一维ALE。

不训练或微调模型；每个fold首先按正式全测试集调用复现冻结预测，然后才执行
固定查询集合上的事后解释诊断。每个特征单独调用预测，避免V1拼接不同扰动批次。
"""

from __future__ import annotations

import argparse
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
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/tabicl_v2_explainability_v2"
MAIN = ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv"
FEATURE_MANIFEST = ROOT / "runs/tabiclv2_pilot/01_strict_feature_manifest.csv"
TAXONOMY = ROOT / "runs/multisource_ablation_v1/00_feature_source_taxonomy.csv"
ADAPTER = ROOT / "models/tabiclv2_adapter.py"
SEED = 42
MAX_ABS_TOL = 1e-6
MEAN_ABS_TOL = 1e-7
METRIC_TOL = 1e-6
EXPLORATION_N = 48
CONFIRMATION_N = 64
EXPLORATION_REPEATS = 1
CONFIRMATION_REPEATS = 2
ALE_N = 64
ALE_BINS = 10

TASKS = {
    "chla_surface": {"target": "chla_ug_l", "depth": "surface", "manifest": ROOT / "runs/chla_family_v2/03_chla_surface_manifest.csv", "prediction": ROOT / "runs/chla_family_v2/predictions/chla_surface_tabiclv2_{fold}.csv"},
    "secchi": {"target": "secchi_depth_m", "depth": None, "manifest": ROOT / "runs/modern_baseline_gate_v1/01_secchi_manifest.csv", "prediction": ROOT / "runs/modern_baseline_gate_v1/predictions/secchi_tabiclv2_recovered.csv"},
    "tn_surface": {"target": "tn_mg_l", "depth": "surface", "manifest": ROOT / "runs/tn_family_v2/03_tn_surface_manifest.csv", "prediction": ROOT / "runs/tn_family_v2/predictions/tn_surface_tabiclv2_{fold}.csv"},
    "tp_middle": {"target": "tp_mg_l", "depth": "middle", "manifest": ROOT / "runs/tp_family_v2/04_tp_middle_manifest.csv", "prediction": ROOT / "runs/tp_family_v2/predictions/tp_middle_tabiclv2_{fold}.csv"},
}


def now() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


def append_csv(path: Path, frame: pd.DataFrame, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        signature = frame[keys].astype(str).agg("|".join, axis=1)
        old_signature = old[keys].astype(str).agg("|".join, axis=1)
        old = old.loc[~old_signature.isin(set(signature))]
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def hash_ids(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def load_features() -> list[str]:
    m = pd.read_csv(FEATURE_MANIFEST)
    f = m.loc[m.included_in_pilot.astype(bool), "feature"].astype(str).tolist()
    if len(f) != 77 or len(set(f)) != 77:
        raise RuntimeError("strict features不是77个唯一变量")
    return f


def load_fold(task: str, fold: str, features: list[str]):
    spec = TASKS[task]
    cols = ["obs_id", "station", "date", spec["target"], *features]
    if spec["depth"] is not None:
        cols.append("depth")
    data = pd.read_csv(MAIN, usecols=list(dict.fromkeys(cols)), dtype={"obs_id": "string", "station": "string"}, low_memory=False)
    data["obs_id"] = data.obs_id.astype("string").str.strip(); data["station_id"] = data.station.astype("string").str.strip(); data["date"] = pd.to_datetime(data.date)
    if spec["depth"] is not None:
        data = data.loc[data.depth.astype(str).str.lower().eq(spec["depth"])].copy()
    data[spec["target"]] = pd.to_numeric(data[spec["target"]], errors="coerce"); data = data.loc[data[spec["target"]].notna()].copy()
    for f in features:
        data[f] = pd.to_numeric(data[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
    man = pd.read_csv(spec["manifest"], dtype={"obs_id": "string"}); man["obs_id"] = man.obs_id.astype("string").str.strip(); man = man.loc[man.fold.astype(str).eq(fold)].copy()
    role = "role"; train_order = man.loc[man[role].eq("train"), "obs_id"].astype(str).tolist(); test_order = man.loc[man[role].eq("test"), "obs_id"].astype(str).tolist()
    if task == "secchi":
        indexed = data.set_index(data.obs_id.astype(str), drop=False); train = indexed.loc[train_order].reset_index(drop=True); test = indexed.loc[test_order].reset_index(drop=True)
    else:
        train = data.loc[data.obs_id.astype(str).isin(set(train_order))].copy(); test = data.loc[data.obs_id.astype(str).isin(set(test_order))].copy()
    if len(train) != len(train_order) or len(test) != len(test_order):
        raise RuntimeError(f"{task}/{fold}: obs_id匹配失败")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_model(x_train, y_train):
    spec = importlib.util.spec_from_file_location("tabicl_v2_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    model = module.TabICLv2Regressor(); model.fit(x_train, y_train)
    return model


def metric(y, p):
    return {"R2": float(r2_score(y, p)), "RMSE": float(mean_squared_error(y, p) ** .5), "MAE": float(mean_absolute_error(y, p))}


def sample_positions(frame: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); pools = {k: list(v) for k, v in frame.groupby("station_id", sort=True).groups.items()}; selected = []
    while len(selected) < min(n, len(frame)):
        changed = False
        for station in sorted(pools):
            available = [i for i in pools[station] if i not in selected]
            if available:
                selected.append(int(rng.choice(available))); changed = True
                if len(selected) >= min(n, len(frame)): break
        if not changed: break
    return np.asarray(selected, int)


def frozen(task, fold):
    p = Path(str(TASKS[task]["prediction"]).format(fold=fold)); d = pd.read_csv(p, dtype={"obs_id": "string"})
    if task == "secchi": d = d.loc[d.fold.astype(str).eq(fold)].copy()
    d["obs_id"] = d.obs_id.astype("string").str.strip(); return d.reset_index(drop=True)


def load_candidate_file(path: str | None, features: list[str]) -> list[str]:
    if not path: return features
    d = pd.read_csv(path); column = "feature" if "feature" in d.columns else d.columns[0]
    selected = d[column].astype(str).tolist()
    missing = sorted(set(selected) - set(features))
    if missing: raise RuntimeError(f"候选变量不在77特征中: {missing}")
    return selected


def run(task: str, fold: str, mode: str, candidate_file: str | None, ale_file: str | None):
    started = time.perf_counter(); features = load_features(); train, test = load_fold(task, fold, features); target = TASKS[task]["target"]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True); xtr = imputer.fit_transform(train[features]).astype("float32"); xte = imputer.transform(test[features]).astype("float32")
    ytr0 = train[target].to_numpy(float); ytr = np.log1p(ytr0).astype("float32") if np.nanmin(ytr0) >= 0 else ytr0.astype("float32"); inverse = np.expm1 if np.nanmin(ytr0) >= 0 else (lambda z: z)
    torch.cuda.reset_peak_memory_stats(); fit_start = time.perf_counter(); model = load_model(xtr, ytr); fit_seconds = time.perf_counter() - fit_start
    def predict(x): return np.maximum(inverse(model.predict(np.asarray(x, dtype="float32"))), 0)

    # Formal reproduction must be the first predict call after fit.
    t = time.perf_counter(); formal = predict(xte); formal_seconds = time.perf_counter() - t; old = frozen(task, fold)
    aligned = pd.DataFrame({"obs_id": test.obs_id.astype(str), "reproduced": formal}).merge(old[["obs_id", "y_true", "y_pred"]].rename(columns={"y_pred": "frozen"}), on="obs_id", validate="one_to_one")
    delta = np.abs(aligned.reproduced - aligned.frozen); mr = metric(aligned.y_true, aligned.reproduced); mf = metric(aligned.y_true, aligned.frozen)
    exact = bool(test.obs_id.astype(str).tolist() == old.obs_id.astype(str).tolist() and delta.max() <= MAX_ABS_TOL and delta.mean() <= MEAN_ABS_TOL and all(abs(mr[k]-mf[k]) <= METRIC_TOL for k in mr))
    reproduction = pd.DataFrame([{"task":task,"fold":fold,"status":"EXACT_REPRODUCTION" if exact else "REPRODUCTION_FAILED","n_train":len(train),"n_test":len(test),"max_abs_error":delta.max(),"mean_abs_error":delta.mean(),"R2_abs_diff":abs(mr['R2']-mf['R2']),"RMSE_abs_diff":abs(mr['RMSE']-mf['RMSE']),"MAE_abs_diff":abs(mr['MAE']-mf['MAE']),"formal_predict_seconds":formal_seconds,"train_obs_hash":hash_ids(train.obs_id),"test_obs_hash":hash_ids(test.obs_id)}])
    append_csv(RUN/"01_inference_reproduction.csv", reproduction, ["task","fold"])
    if not exact: raise RuntimeError(f"{task}/{fold}正式预测未严格复现")

    candidates = load_candidate_file(candidate_file, features); n = EXPLORATION_N if mode == "exploration" else CONFIRMATION_N; repeats = EXPLORATION_REPEATS if mode == "exploration" else CONFIRMATION_REPEATS
    idx = sample_positions(test, n, SEED + int(fold[-1]) + 100 * list(TASKS).index(task)); xe = xte[idx]; ye = test.iloc[idx][target].to_numpy(float); stations = test.iloc[idx].station_id.astype(str)
    baseline_metrics=[]; baseline_times=[]
    for repeat in range(3):
        t=time.perf_counter(); bp=predict(xe); baseline_times.append(time.perf_counter()-t); baseline_metrics.append(metric(ye,bp))
    base_rmse=float(np.mean([x['RMSE'] for x in baseline_metrics])); base_mae=float(np.mean([x['MAE'] for x in baseline_metrics])); noise_rmse=float(np.std([x['RMSE'] for x in baseline_metrics],ddof=1)); noise_mae=float(np.std([x['MAE'] for x in baseline_metrics],ddof=1))
    tax=pd.read_csv(TAXONOMY).set_index("feature_name"); pi_rows=[]
    (RUN/"01_importance/raw"/task).mkdir(parents=True,exist_ok=True)
    for feature in candidates:
        j=features.index(feature)
        for repeat in range(repeats):
            rng=np.random.default_rng(SEED + int(fold[-1])*1000 + features.index(feature)*10 + repeat); altered=xe.copy(); altered[:,j]=altered[rng.permutation(len(altered)),j]
            t=time.perf_counter(); pp=predict(altered); elapsed=time.perf_counter()-t; mm=metric(ye,pp)
            pi_rows.append({"task":task,"fold":fold,"analysis_role":"EXPLORATORY" if mode=="exploration" else "CONFIRMATORY","feature":feature,"source_group":tax.loc[feature,'source_group'] if feature in tax.index else "UNKNOWN","repeat":repeat+1,"n_query":len(xe),"n_stations":stations.nunique(),"baseline_RMSE":base_rmse,"permuted_RMSE":mm['RMSE'],"delta_RMSE":mm['RMSE']-base_rmse,"baseline_MAE":base_mae,"permuted_MAE":mm['MAE'],"delta_MAE":mm['MAE']-base_mae,"baseline_noise_sd_RMSE":noise_rmse,"baseline_noise_sd_MAE":noise_mae,"elapsed_seconds":elapsed,"posthoc_test_label_diagnostic":True})
            pd.DataFrame(pi_rows[-1:]).to_csv(RUN/"01_importance/raw"/task/f"{fold}_{feature}_repeat{repeat+1}.csv",index=False,encoding="utf-8-sig")
    pi=pd.DataFrame(pi_rows); append_csv(RUN/"02_key_feature_importance_by_fold.csv",pi,["task","fold","feature","repeat"])

    ale_features = load_candidate_file(ale_file, features)[:4] if ale_file else []
    ale_rows=[]
    if ale_features:
        aidx=sample_positions(test,ALE_N,7000+SEED+int(fold[-1])+100*list(TASKS).index(task)); xr=xte[aidx]; st=test.iloc[aidx].station_id.astype(str).to_numpy()
        for feature in ale_features:
            j=features.index(feature); edges=np.unique(np.quantile(xtr[:,j],np.linspace(0,1,ALE_BINS+1)))
            if len(edges)<4: continue
            bid=np.clip(np.digitize(xr[:,j],edges[1:-1],right=True),0,len(edges)-2); lo=xr.copy();hi=xr.copy();lo[:,j]=edges[bid];hi[:,j]=edges[bid+1]
            t=time.perf_counter(); paired=predict(np.vstack([lo,hi])); elapsed=time.perf_counter()-t; local=paired[len(lo):]-paired[:len(lo)]
            le=np.array([np.mean(local[bid==b]) if np.any(bid==b) else np.nan for b in range(len(edges)-1)]); counts=np.array([(bid==b).sum() for b in range(len(edges)-1)]); nstations=np.array([len(set(st[bid==b])) for b in range(len(edges)-1)])
            acc=np.cumsum(np.nan_to_num(le)); acc-=np.average(acc,weights=np.maximum(counts,1))
            for b in range(len(edges)-1):
                ale_rows.append({"task":task,"fold":fold,"feature":feature,"bin":b+1,"lower":edges[b],"upper":edges[b+1],"center":(edges[b]+edges[b+1])/2,"local_effect":le[b],"ale":acc[b],"n_samples":counts[b],"n_stations":nstations[b],"reference_distribution":"station-holdout test features; labels unused","query_batch_n":len(paired),"elapsed_seconds_feature":elapsed})
        if ale_rows: append_csv(RUN/"03_ale_1d_by_fold.csv",pd.DataFrame(ale_rows),["task","fold","feature","bin"])

    runtime=pd.DataFrame([{"task":task,"fold":fold,"mode":mode,"n_candidates":len(candidates),"permutation_repeats":repeats,"n_query":len(xe),"fit_seconds":fit_seconds,"formal_predict_seconds":formal_seconds,"baseline_predict_seconds_mean":np.mean(baseline_times),"importance_seconds":pi.elapsed_seconds.sum(),"ale_features":len(ale_features),"total_elapsed_seconds":time.perf_counter()-started,"peak_gpu_mb":torch.cuda.max_memory_allocated()/1024**2,"kv_cache":False,"batch_policy":"each perturbation separate; fixed query set and size","completed_at":now()}])
    append_csv(RUN/"00_runtime_registry.csv",runtime,["task","fold","mode"])
    print(json.dumps(runtime.iloc[0].to_dict(),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=list(TASKS),required=True);p.add_argument('--fold',choices=[f'fold_{i}' for i in range(1,6)],required=True);p.add_argument('--mode',choices=['exploration','confirmation'],required=True);p.add_argument('--candidate-file');p.add_argument('--ale-file');a=p.parse_args()
    RUN.mkdir(parents=True,exist_ok=True); run(a.task,a.fold,a.mode,a.candidate_file,a.ale_file)


if __name__=='__main__': main()
