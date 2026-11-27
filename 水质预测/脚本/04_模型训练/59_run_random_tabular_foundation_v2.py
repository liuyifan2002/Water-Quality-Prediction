"""Five-fold inner OOF for TabICLv2, pinned TabPFN V2, and TabM k=32.

Only external training obs_id are loaded. No external test inference occurs here.
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/random_tabular_model_zoo_v2'
TABLE=ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv'
FEATURES=pd.read_csv(ROOT/'runs/tabiclv2_pilot/01_strict_feature_manifest.csv').query('included_in_pilot == True').feature.tolist()
TASKS={
 'chla_surface':('chla_ug_l','surface'),'chla_middle':('chla_ug_l','middle'),'chla_bottom':('chla_ug_l','bottom'),
 'secchi':('secchi_depth_m','surface'),'tn_surface':('tn_mg_l','surface'),'tn_middle':('tn_mg_l','middle'),
 'tn_bottom':('tn_mg_l','bottom'),'tp_surface':('tp_mg_l','surface'),'tp_middle':('tp_mg_l','middle'),
 'tp_bottom':('tp_mg_l','bottom')}
PORTABLE=ROOT/'结果/扩展实验/表格预训练算法对比_20260908/portable_package/run_experiment.py'
CFG=ROOT/'结果/扩展实验/表格预训练算法对比_20260908/portable_package/config.json'

def load(task):
    target,depth=TASKS[task]
    d=pd.read_csv(TABLE,usecols=['obs_id','depth',target,*FEATURES],low_memory=False)
    d=d.loc[d.depth.eq(depth)&d[target].notna()].copy();d.obs_id=d.obs_id.astype(str)
    s=pd.read_csv(OUT/'01_inner_fold_manifest.csv',dtype={'obs_id':str})
    s=s.loc[s.task.eq(task),['obs_id','inner_fold']]
    d=d.merge(s,on='obs_id',how='inner',validate='one_to_one').sort_values('obs_id').reset_index(drop=True)
    if len(d)!=len(s):raise ValueError('training obs_id mismatch')
    x=d[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    return d[['obs_id','inner_fold']],x,d[target].to_numpy(float)

def portable():
    spec=importlib.util.spec_from_file_location('random_zoo_portable',PORTABLE)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def predict(name,xa,ya,xb,base,cfg):
    if name=='tabicl':
        from models.tabiclv2_adapter import TabICLv2Regressor
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        x1=imp.fit_transform(xa).astype(np.float32);x2=imp.transform(xb).astype(np.float32)
        model=TabICLv2Regressor();model.fit(x1,ya);return model.predict(x2)
    if name=='tabpfn':
        return base.fit_tabpfn(xa,ya,xb,42,cfg)
    if name=='tabm':
        return base.fit_tabm(xa,ya,xb,42,cfg)
    raise ValueError(name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--model',required=True,choices=['tabicl','tabpfn','tabm']);a=p.parse_args()
    dest=OUT/'02_inner_oof'/f'{a.task}_{a.model}.csv';metric=OUT/'03_inner_metrics'/f'{a.task}_{a.model}.json'
    if dest.exists() and metric.exists():print('SKIP',a.task,a.model,flush=True);return
    if len(FEATURES)!=77 or 'water_zone_code' in FEATURES:raise ValueError('77 feature manifest invalid')
    ids,x,y=load(a.task);base=portable();cfg=json.loads(CFG.read_text(encoding='utf-8'))
    cfg['model_parameters']['tabm'].update({'arch_type':'tabm','k':32})
    cfg['model_parameters']['tabpfn']['model_path']=str((CFG.parent/'models/tabpfn-v2-regressor.ckpt').resolve())
    pred=np.full(len(y),np.nan);rows=[]
    for fold in range(1,6):
        tr=ids.inner_fold.to_numpy()!=fold;val=~tr
        yt=y[tr];use_log=np.min(yt)>=0;yf=np.log1p(yt) if use_log else yt
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        except ImportError:pass
        t=time.perf_counter();raw=predict(a.model,x.iloc[tr],yf,x.iloc[val],base,cfg);elapsed=time.perf_counter()-t
        yp=np.maximum(np.expm1(raw) if use_log else raw,0)
        pred[val]=yp
        row={'fold':fold,'n_train':int(tr.sum()),'n_val':int(val.sum()),'R2':float(r2_score(y[val],yp)),'RMSE':float(mean_squared_error(y[val],yp)**.5),'MAE':float(mean_absolute_error(y[val],yp)),'runtime_seconds':elapsed,'gpu_peak_allocated_mb':float(torch.cuda.max_memory_allocated()/2**20) if 'torch' in sys.modules and torch.cuda.is_available() else None}
        rows.append(row);print(a.task,a.model,fold,row['R2'],flush=True)
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:pass
        gc.collect()
    if np.isnan(pred).any():raise ValueError('incomplete OOF')
    dest.parent.mkdir(parents=True,exist_ok=True);metric.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'task':a.task,'model':a.model,'obs_id':ids.obs_id,'inner_fold':ids.inner_fold,'y_true':y,'y_pred':pred}).to_csv(dest,index=False,encoding='utf-8-sig')
    m={'task':a.task,'model':a.model,'n_samples':len(y),'inner_oof_R2':float(r2_score(y,pred)),'inner_oof_RMSE':float(mean_squared_error(y,pred)**.5),'inner_oof_MAE':float(mean_absolute_error(y,pred)),'runtime_seconds':sum(z['runtime_seconds'] for z in rows),'folds':rows,'model_identity':{'tabicl':'tabicl==2.2.0 default TabICLv2Regressor','tabpfn':'tabpfn==9.0.0 pinned V2 checkpoint','tabm':'tabm==0.0.3 arch_type=tabm k=32'}[a.model]}
    metric.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');print('DONE',a.task,a.model,m['inner_oof_R2'],flush=True)

if __name__=='__main__':main()
