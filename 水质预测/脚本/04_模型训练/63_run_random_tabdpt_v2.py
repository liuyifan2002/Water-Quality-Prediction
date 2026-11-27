"""TabDPT 1.3.2 on the frozen inner folds; use the isolated TabDPT environment.

The complete external-training fold is passed to fit. Retrieval is part of
TabDPT's documented prediction mechanism and does not discard training rows.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/random_tabular_model_zoo_v2'
BASE=ROOT/'脚本/04_模型训练/57_run_random_tabular_model_zoo_v2.py'
spec=importlib.util.spec_from_file_location('zoo_base',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def model():
    from tabdpt import TabDPTRegressor
    weights=list((Path.home()/'.cache/huggingface/hub/models--Layer6--TabDPT/snapshots').glob('*/tabdpt1_3.safetensors'))
    if len(weights)!=1:raise FileNotFoundError('TabDPT 1.3 checkpoint not uniquely found')
    return TabDPTRegressor(device='cuda',use_flash=False,compile=False,
        context_reduction='retrieval',model_weight_path=str(weights[0]),verbose=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=base.TASKS,required=True)
    p.add_argument('--smoke-fold',type=int,default=0);a=p.parse_args()
    ids,x,y=base.load_task(a.task)
    output=OUT/'02_inner_oof'/f'{a.task}_tabdpt.csv'
    meta=OUT/'03_inner_metrics'/f'{a.task}_tabdpt.json'
    if a.smoke_fold==0 and output.exists() and meta.exists():print('SKIP',a.task);return
    pred=np.full(len(y),np.nan);folds=[]
    for fold in range(1,6):
        if a.smoke_fold and fold!=a.smoke_fold:continue
        train=ids.inner_fold.to_numpy()!=fold;val=~train
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        xa=imp.fit_transform(x[train]).astype(np.float32);xb=imp.transform(x[val]).astype(np.float32)
        yt=y[train];use_log=np.min(yt)>=0;yf=np.log1p(yt) if use_log else yt
        m=model();start=time.perf_counter();m.fit(xa,yf)
        raw=m.predict(xb,n_ensembles=2,context_size=1024,batch_size=64,seed=42)
        seconds=time.perf_counter()-start
        yp=np.maximum(np.expm1(raw) if use_log else raw,0);pred[val]=yp
        folds.append({'fold':fold,'n_train':int(train.sum()),'n_val':int(val.sum()),
            'R2':float(r2_score(y[val],yp)),'RMSE':float(mean_squared_error(y[val],yp)**.5),
            'MAE':float(mean_absolute_error(y[val],yp)),'runtime_seconds':seconds})
        print(a.task,fold,folds[-1]['R2'],seconds,flush=True)
        del m
        import torch
        torch.cuda.empty_cache()
    if a.smoke_fold:
        path=OUT/'external/tabdpt_smoke.json'
        path.write_text(json.dumps({'task':a.task,'fold':folds[0]},ensure_ascii=False,indent=2),encoding='utf-8')
        return
    if np.isnan(pred).any():raise ValueError('incomplete TabDPT OOF')
    pd.DataFrame({'task':a.task,'model':'tabdpt','obs_id':ids.obs_id,
        'inner_fold':ids.inner_fold,'y_true':y,'y_pred':pred}).to_csv(output,index=False,encoding='utf-8-sig')
    meta.write_text(json.dumps({'task':a.task,'model':'tabdpt','n_samples':len(y),
        'inner_oof_R2':float(r2_score(y,pred)),
        'inner_oof_RMSE':float(mean_squared_error(y,pred)**.5),
        'inner_oof_MAE':float(mean_absolute_error(y,pred)),
        'runtime_seconds':sum(v['runtime_seconds'] for v in folds),
        'folds':folds,'identity':'TabDPT 1.3.2; v1.3 official checkpoint; retrieval context 1024; ensemble 2'},
        ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
