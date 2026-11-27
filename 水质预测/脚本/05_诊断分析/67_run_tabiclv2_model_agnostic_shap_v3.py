"""TABICLV2_EXPLAINABILITY_V3：完整77维Permutation SHAP，逐样本检查点。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/tabicl_v2_explainability_v3"
V2_RUNNER = ROOT / "脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py"
SEED = 42
BACKGROUND_N = 16
EXPLANATION_N = 5
FIXED_BATCH_N = 256
MAX_EVALS = 155
ADDITIVITY_TOL = 0.02


def load_base():
    spec = importlib.util.spec_from_file_location("tabicl_v3_base", V2_RUNNER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


BASE = load_base()


def append_csv(path: Path, frame: pd.DataFrame, keys: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        sig = frame[keys].astype(str).agg("|".join, axis=1); old_sig = old[keys].astype(str).agg("|".join, axis=1)
        old = old.loc[~old_sig.isin(set(sig))]; frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def representative_positions(frame: pd.DataFrame, x: np.ndarray, n: int, seed: int) -> np.ndarray:
    """不使用水质标签：先覆盖站点-年份，再以77维标准化空间最大最小距离补足。"""
    meta = frame[["station_id", "date"]].copy(); meta["year"] = pd.to_datetime(meta.date).dt.year
    rng = np.random.default_rng(seed); candidates=[]
    for _, group in meta.groupby(["station_id", "year"], sort=True):
        candidates.append(int(rng.choice(group.index.to_numpy())))
    candidates = list(dict.fromkeys(candidates))
    scale = np.nanstd(x, axis=0); scale[scale == 0] = 1; z = (x - np.nanmean(x, axis=0)) / scale
    if not candidates: candidates = list(range(len(frame)))
    first = candidates[int(rng.integers(len(candidates)))]; selected=[first]
    pool = candidates + [i for i in range(len(frame)) if i not in set(candidates)]
    while len(selected) < min(n, len(frame)):
        remain = np.asarray([i for i in pool if i not in selected], int)
        dist = np.min(((z[remain, None, :] - z[np.asarray(selected)[None, :], :]) ** 2).mean(axis=2), axis=1)
        selected.append(int(remain[np.argmax(dist)]))
    return np.asarray(selected, int)


class FixedBatchPredictor:
    def __init__(self, predict, anchors: np.ndarray, batch_n: int):
        self.predict = predict; self.anchors = np.asarray(anchors, dtype="float32"); self.batch_n = batch_n; self.calls=[]

    def __call__(self, matrix):
        matrix=np.asarray(matrix,dtype="float32"); outputs=[]; started=time.perf_counter()
        for start in range(0,len(matrix),self.batch_n):
            part=matrix[start:start+self.batch_n]; need=self.batch_n-len(part)
            if need:
                reps=int(np.ceil(need/len(self.anchors))); pad=np.tile(self.anchors,(reps,1))[:need]
                query=np.vstack([part,pad])
            else: query=part
            outputs.append(self.predict(query)[:len(part)])
        self.calls.append({"input_rows":len(matrix),"fixed_batches":int(np.ceil(len(matrix)/self.batch_n)),"seconds":time.perf_counter()-started})
        return np.concatenate(outputs)


def run(task: str, fold: str, sample_count: int, shap_seed: int, force: bool):
    features=BASE.load_features(); train,test=BASE.load_fold(task,fold,features); target=BASE.TASKS[task]["target"]
    imputer=SimpleImputer(strategy="median",keep_empty_features=True);xtr=imputer.fit_transform(train[features]).astype("float32");xte=imputer.transform(test[features]).astype("float32")
    y0=train[target].to_numpy(float);y=np.log1p(y0).astype("float32") if np.nanmin(y0)>=0 else y0.astype("float32");inverse=np.expm1 if np.nanmin(y0)>=0 else (lambda z:z)
    torch.cuda.reset_peak_memory_stats();fit_start=time.perf_counter();model=BASE.load_model(xtr,y);fit_seconds=time.perf_counter()-fit_start
    def predict(z): return np.maximum(inverse(model.predict(np.asarray(z,dtype="float32"))),0)
    # exact reproduction, first call after fit
    t=time.perf_counter();formal=predict(xte);formal_seconds=time.perf_counter()-t;old=BASE.frozen(task,fold);delta=np.abs(formal-old.y_pred.to_numpy(float));exact=bool(test.obs_id.astype(str).tolist()==old.obs_id.astype(str).tolist() and delta.max()<=BASE.MAX_ABS_TOL and delta.mean()<=BASE.MEAN_ABS_TOL)
    append_csv(RUN/"00_inference_reproduction.csv",pd.DataFrame([{"task":task,"fold":fold,"status":"EXACT_REPRODUCTION" if exact else "REPRODUCTION_FAILED","max_abs_error":delta.max(),"mean_abs_error":delta.mean(),"n_train":len(train),"n_test":len(test),"formal_predict_seconds":formal_seconds}]),["task","fold"])
    if not exact: raise RuntimeError(f"{task}/{fold}: reproduction failed")

    bg_idx=representative_positions(train,xtr,BACKGROUND_N,9000+int(fold[-1])+100*list(BASE.TASKS).index(task)); ex_idx=representative_positions(test,xte,sample_count,12000+int(fold[-1])+100*list(BASE.TASKS).index(task));background=xtr[bg_idx]
    bg_manifest=train.iloc[bg_idx][["obs_id","station_id","date"]].copy();bg_manifest.insert(0,"task",task);bg_manifest.insert(1,"fold",fold);bg_manifest["background_order"]=range(1,len(bg_manifest)+1);append_csv(RUN/"background/background_manifest.csv",bg_manifest,["task","fold","obs_id"])
    exp_manifest=test.iloc[ex_idx][["obs_id","station_id","date"]].copy();exp_manifest.insert(0,"task",task);exp_manifest.insert(1,"fold",fold);exp_manifest["explanation_order"]=range(1,len(exp_manifest)+1);exp_manifest["selection_used_test_label"]=False;append_csv(RUN/"explanation/explanation_sample_manifest.csv",exp_manifest,["task","fold","obs_id"])

    # 延迟导入，确保正式复现阶段的初始化顺序与原TabICLv2流程一致。
    import shap
    masker=shap.maskers.Independent(background,max_samples=BACKGROUND_N)
    for order,pos in enumerate(ex_idx,1):
        obs_id=str(test.iloc[pos].obs_id); out=RUN/"shap_values"/task/fold/f"{obs_id}_seed{shap_seed}.csv.gz"
        if out.exists() and not force:
            print(f"SKIP {task}/{fold}/{obs_id}/seed{shap_seed}",flush=True);continue
        anchors=np.delete(xte,pos,axis=0)
        anchor_idx=representative_positions(test.drop(index=pos).reset_index(drop=True),anchors,min(FIXED_BATCH_N-1,len(anchors)),15000+pos+shap_seed)
        wrapped=FixedBatchPredictor(predict,anchors[anchor_idx],FIXED_BATCH_N)
        explainer=shap.PermutationExplainer(wrapped,masker,feature_names=features,seed=shap_seed)
        started=time.perf_counter();sv=explainer(xte[[pos]],max_evals=MAX_EVALS,batch_size=FIXED_BATCH_N,silent=True);elapsed=time.perf_counter()-started
        check_pred=float(wrapped(xte[[pos]])[0]);shap_sum=float(np.asarray(sv.base_values).reshape(-1)[0]+np.asarray(sv.values).reshape(1,-1).sum());add_error=abs(check_pred-shap_sum);formal_sample=float(formal[pos]);wrapper_drift=abs(check_pred-formal_sample)
        frame=pd.DataFrame({"task":task,"fold":fold,"obs_id":obs_id,"station_id":str(test.iloc[pos].station_id),"date":pd.to_datetime(test.iloc[pos].date).strftime('%Y-%m-%d'),"shap_seed":shap_seed,"feature_order":range(1,78),"feature":features,"feature_value":xte[pos],"shap_value":np.asarray(sv.values).reshape(-1),"base_value":float(np.asarray(sv.base_values).reshape(-1)[0]),"shap_sum_prediction":shap_sum,"wrapper_prediction_check":check_pred,"formal_full_batch_prediction":formal_sample,"additivity_abs_error":add_error,"wrapper_vs_formal_abs_error":wrapper_drift,"output_scale":"original water-quality scale"})
        out.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(out,index=False,encoding="utf-8-sig",compression="gzip")
        runtime=pd.DataFrame([{"task":task,"fold":fold,"obs_id":obs_id,"shap_seed":shap_seed,"status":"OK" if add_error<=ADDITIVITY_TOL else "ADDITIVITY_FAILED","n_features":77,"background_n":BACKGROUND_N,"max_evals":MAX_EVALS,"fixed_batch_n":FIXED_BATCH_N,"model_input_rows":sum(c['input_rows'] for c in wrapped.calls),"fixed_model_batches":sum(c['fixed_batches'] for c in wrapped.calls),"elapsed_seconds":elapsed,"seconds_per_explanation":elapsed,"additivity_abs_error":add_error,"additivity_tolerance":ADDITIVITY_TOL,"wrapper_vs_formal_abs_error":wrapper_drift,"peak_gpu_mb":torch.cuda.max_memory_allocated()/1024**2,"fit_seconds_shared":fit_seconds,"formal_predict_seconds_shared":formal_seconds,"output_path":str(out)}])
        append_csv(RUN/"01_shap_runtime_benchmark.csv",runtime,["task","fold","obs_id","shap_seed"])
        append_csv(RUN/"02_tabicl_shap_values_manifest.csv",runtime[["task","fold","obs_id","shap_seed","status","n_features","background_n","max_evals","output_path"]],["task","fold","obs_id","shap_seed"])
        print(json.dumps(runtime.iloc[0].to_dict(),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=list(BASE.TASKS),required=True);p.add_argument('--fold',choices=[f'fold_{i}' for i in range(1,6)],required=True);p.add_argument('--sample-count',type=int,default=EXPLANATION_N);p.add_argument('--shap-seed',type=int,default=SEED);p.add_argument('--force',action='store_true');a=p.parse_args();RUN.mkdir(parents=True,exist_ok=True);run(a.task,a.fold,a.sample_count,a.shap_seed,a.force)


if __name__=='__main__':main()
