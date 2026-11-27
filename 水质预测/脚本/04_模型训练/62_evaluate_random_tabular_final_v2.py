"""Evaluate a pre-registered model on frozen external Random Strict test IDs.

Run only after 06_final_selection_plan.json exists; this script never selects
models or weights from external test outcomes.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/random_tabular_model_zoo_v2'
V1=ROOT/'runs/random_10task_v1'
TABLE=ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv'
BASE_PATH=ROOT/'脚本/04_模型训练/57_run_random_tabular_model_zoo_v2.py'
spec=importlib.util.spec_from_file_location('random_zoo_baseline',BASE_PATH);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PORTABLE=ROOT/'结果/扩展实验/表格预训练算法对比_20260908/portable_package/run_experiment.py'
CFG=ROOT/'结果/扩展实验/表格预训练算法对比_20260908/portable_package/config.json'
TASKS=base.TASKS

def portable():
    spec=importlib.util.spec_from_file_location('random_zoo_portable',PORTABLE)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def load(task):
    target,depth=TASKS[task]
    d=pd.read_csv(TABLE,usecols=['obs_id','station','date','depth',target,*base.FEATURES],low_memory=False)
    d=d.loc[d.depth.eq(depth)&d[target].notna()].copy();d.obs_id=d.obs_id.astype(str)
    s=pd.read_csv(V1/'01_random_split_manifest.csv',dtype={'obs_id':str})
    s=s.loc[s.task.eq(task),['obs_id','split']]
    d=d.merge(s,on='obs_id',how='inner',validate='one_to_one')
    if len(d)!=len(s):raise ValueError('outer split coverage mismatch')
    tr=d.loc[d.split.eq('train')].sort_values('obs_id').copy()
    te=d.loc[d.split.eq('test')].sort_values('obs_id').copy()
    return tr,te,target

def predict_model(task,model,tr,te,target):
    ytr=tr[target].to_numpy(float);use_log=np.min(ytr)>=0;yf=np.log1p(ytr) if use_log else ytr
    xtr=tr[base.FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    xte=te[base.FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    start=time.perf_counter();origin='trained_on_full_outer_train'
    if model=='tabicl':
        from models.tabiclv2_adapter import TabICLv2Regressor
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        xa=imp.fit_transform(xtr).astype(np.float32);xb=imp.transform(xte).astype(np.float32)
        estimator=TabICLv2Regressor();estimator.fit(xa,yf);raw=estimator.predict(xb)
    elif model in base.MODELS or model.endswith('_tuned'):
        identity=model.removesuffix('_tuned');params={}
        if model.endswith('_tuned'):
            meta=json.loads((OUT/'03_inner_metrics'/f'{task}_{model}.json').read_text(encoding='utf-8'))
            params=meta['chosen_params']
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        xa=imp.fit_transform(xtr).astype(np.float32);xb=imp.transform(xte).astype(np.float32)
        if identity in base.STANDARD:
            scale=StandardScaler();xa=scale.fit_transform(xa);xb=scale.transform(xb)
        estimator=base.model_for(identity).set_params(**params)
        estimator.fit(xa,yf);raw=estimator.predict(xb)
    elif model in {'tabpfn','tabm'}:
        cfg=json.loads(CFG.read_text(encoding='utf-8'))
        cfg['model_parameters']['tabm'].update({'arch_type':'tabm','k':32})
        cfg['model_parameters']['tabpfn']['model_path']=str((CFG.parent/'models/tabpfn-v2-regressor.ckpt').resolve())
        mod=portable()
        raw=mod.fit_tabpfn(xtr,yf,xte,42,cfg) if model=='tabpfn' else mod.fit_tabm(xtr,yf,xte,42,cfg)
    else:raise ValueError(f'unsupported final model {model}')
    return np.maximum(np.expm1(raw) if use_log else raw,0),origin,time.perf_counter()-start

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--model',required=True);a=p.parse_args()
    plan_path=OUT/'06_final_selection_plan.json'
    if not plan_path.exists():raise FileNotFoundError('inner OOF selection plan must be locked before external test')
    dest=OUT/'07_final_predictions'/f'{a.task}_{a.model}.csv';metric=OUT/'07_final_predictions'/f'{a.task}_{a.model}.json'
    if dest.exists() and metric.exists():print('SKIP',a.task,a.model,flush=True);return
    gpu_measure=a.model in {'tabicl','tabpfn','tabm'}
    if gpu_measure:
        import torch
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    tr,te,target=load(a.task);yp,origin,seconds=predict_model(a.task,a.model,tr,te,target)
    yt=te[target].to_numpy(float)
    if len(yp)!=len(yt) or not np.isfinite(yp).all():raise ValueError('prediction invalid')
    dest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'task':a.task,'model':a.model,'obs_id':te.obs_id,'station_id':te.station,'date':te.date,'y_true':yt,'y_pred':yp,'absolute_error':abs(yp-yt),'squared_error':(yp-yt)**2}).to_csv(dest,index=False,encoding='utf-8-sig')
    result={'task':a.task,'model':a.model,'n_train':len(tr),'n_test':len(te),'R2':float(r2_score(yt,yp)),'RMSE':float(mean_squared_error(yt,yp)**.5),'MAE':float(mean_absolute_error(yt,yp)),'runtime_seconds':seconds,'gpu_peak_allocated_mb':float(torch.cuda.max_memory_allocated()/2**20) if gpu_measure and torch.cuda.is_available() else None,'source':origin}
    metric.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(a.task,a.model,result['R2'],flush=True)

if __name__=='__main__':main()
