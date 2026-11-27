"""MODERN_BASELINE_GATE_V1: only TabPFN and TabM on frozen station manifests."""
from __future__ import annotations
import copy, importlib.util, json, time, gc
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs'/'modern_baseline_gate_v1'; RUN.mkdir(parents=True,exist_ok=True)
MAIN=ROOT/'数据'/'处理后数据'/'正式主表'/'model_table_3depth_2014_2024.csv'
STRICT=ROOT/'runs'/'tabiclv2_pilot'/'01_strict_feature_manifest.csv'
SAVED_SPLIT=ROOT/'runs'/'tabiclv2_pilot'/'02_split_manifest.csv'
PORTABLE=ROOT/'结果'/'扩展实验'/'表格预训练算法对比_20260908'/'portable_package'/'run_experiment.py'
CONFIG=ROOT/'结果'/'扩展实验'/'表格预训练算法对比_20260908'/'portable_package'/'config.json'
MODELS=('tabm','tabpfn'); FOLDS=tuple(f'fold_{i}' for i in range(1,6)); SEED=20260921
TASKS={'chla_surface':('chla_ug_l','surface',ROOT/'runs/chla_family_v2/03_chla_surface_manifest.csv'), 'secchi':('secchi_depth_m',None,None), 'tn_surface':('tn_mg_l','surface',ROOT/'runs/tn_family_v2/03_tn_surface_manifest.csv'), 'tp_middle':('tp_mg_l','middle',ROOT/'runs/tp_family_v2/04_tp_middle_manifest.csv')}

def import_portable():
    spec=importlib.util.spec_from_file_location('portable_gate_model',PORTABLE); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
BASE=import_portable()

def features():
    f=pd.read_csv(STRICT); out=f.loc[f['included_in_pilot'].astype(bool),'feature'].dropna().astype(str).tolist()
    if len(out)!=77: raise RuntimeError(f'strict feature count is {len(out)}')
    return out

def load_data():
    cols=pd.read_csv(MAIN,nrows=0).columns.tolist(); fs=features()
    for target,_,_ in TASKS.values():
        if target not in cols: raise RuntimeError(f'missing target {target}')
    return pd.read_csv(MAIN,usecols=list(dict.fromkeys(['obs_id','station','date','depth']+[v[0] for v in TASKS.values()]+fs))),fs

def secchi_manifest(data):
    s=pd.read_csv(SAVED_SPLIT,usecols=['obs_id','station','date','split','task','validation','fold'])
    s=s[(s.task=='secchi')&(s.validation=='station_holdout')].copy(); s=s.rename(columns={'station':'station_id'}); s['depth']='all'; return s

def make_manifests(data):
    out={}
    for task,(target,depth,path) in TASKS.items():
        if task=='secchi': m=secchi_manifest(data)
        else:
            m=pd.read_csv(path,dtype={'obs_id':'string','station_id':'string'}); m=m[['obs_id','station_id','date','fold','role']].copy(); m['depth']=depth
        if 'role' not in m: m['role']=m['split'].map({'test':'test','train':'train'})
        m['task']=task; m.to_csv(RUN/f'01_{task}_manifest.csv',index=False,encoding='utf-8-sig'); out[task]=m
    return out

def preflight(data,fs,manifests):
    rows=[]
    for task,(target,depth,_) in TASKS.items():
        m=manifests[task]; ok=True; err=[]
        for fold in FOLDS:
            test=m[(m.fold==fold)&(m.role=='test')]; train=m[(m.fold==fold)&(m.role=='train')]
            overlap=set(test.station_id.astype(str))&set(train.station_id.astype(str)); obs_overlap=set(test.obs_id.astype(str))&set(train.obs_id.astype(str))
            if overlap or obs_overlap: ok=False; err.append(f'{fold}:overlap')
            if test.obs_id.duplicated().any() or train.obs_id.duplicated().any(): ok=False; err.append(f'{fold}:duplicate')
            rows.append({'task':task,'fold':fold,'train_rows':len(train),'test_rows':len(test),'n_features':len(fs),'station_overlap':len(overlap),'obs_id_overlap':len(obs_overlap),'feature_match':len(fs)==77,'status':'PASS' if not overlap and not obs_overlap else 'FAIL'})
        if not set(m.obs_id.astype(str)).issubset(set(data.obs_id.astype(str))): ok=False; err.append('unknown obs')
        if data.loc[data.obs_id.astype(str).isin(set(m.obs_id.astype(str))),target].isna().any(): ok=False; err.append('missing target')
        if not ok: raise RuntimeError(f'{task} preflight failed: {err}')
    result=pd.DataFrame(rows); result.to_csv(RUN/'02_preflight.csv',index=False,encoding='utf-8-sig'); (RUN/'00_protocol.json').write_text(json.dumps({'protocol':'MODERN_BASELINE_GATE_V1','seed':SEED,'features':77,'models':MODELS,'tasks':list(TASKS),'config':json.loads(CONFIG.read_text(encoding='utf-8'))},ensure_ascii=False,indent=2),encoding='utf-8'); print(result.to_string(index=False)); print('PREFLIGHT=PASS')

def registry():
    p=RUN/'03_run_registry.csv'
    if p.exists(): return pd.read_csv(p)
    rows=[{'task':t,'model':m,'fold':f,'status':'PENDING','start_time':'','end_time':'','runtime':np.nan,'prediction_file':str(RUN/'predictions'/f'{t}_{m}_{f}.csv'),'metrics_file':str(RUN/'fold_metrics'/f'{t}_{m}_{f}.csv'),'R2':np.nan,'RMSE':np.nan,'MAE':np.nan,'error':''} for t in TASKS for m in MODELS for f in FOLDS]
    r=pd.DataFrame(rows); r.to_csv(p,index=False); return r

def run(data,fs,manifests):
    cfg=json.loads(CONFIG.read_text(encoding='utf-8')); reg=registry(); (RUN/'predictions').mkdir(exist_ok=True); (RUN/'fold_metrics').mkdir(exist_ok=True)
    for task,(target,depth,_) in TASKS.items():
        for model in MODELS:
            for fold in FOLDS:
                q=(reg.task==task)&(reg.model==model)&(reg.fold==fold); row=reg.loc[q].iloc[0]
                pred_path=Path(row.prediction_file)
                if row.status=='OK' and pred_path.exists(): print('SKIP',task,model,fold); continue
                m=manifests[task]; tr=m[(m.fold==fold)&(m.role=='train')]; te=m[(m.fold==fold)&(m.role=='test')]; train=data[data.obs_id.astype(str).isin(set(tr.obs_id.astype(str)))]; test=data[data.obs_id.astype(str).isin(set(te.obs_id.astype(str)))]; train=train.set_index(train.obs_id.astype(str)).loc[tr.obs_id.astype(str)].reset_index(drop=True); test=test.set_index(test.obs_id.astype(str)).loc[te.obs_id.astype(str)].reset_index(drop=True)
                y=train[target].to_numpy(float); ytest=test[target].to_numpy(float); yfit,inv=BASE.target_transform(y); xtr=train[fs].copy(); xte=test[fs].copy(); started=time.perf_counter(); reg.loc[q,'status']='RUNNING'; reg.to_csv(RUN/'03_run_registry.csv',index=False)
                try:
                    fn=BASE.fit_tabm if model=='tabm' else BASE.fit_tabpfn; p=inv(fn(xtr,yfit,xte,SEED,cfg)); mm={'R2':float(BASE.r2_score(ytest,p)),'RMSE':float(BASE.mean_squared_error(ytest,p)**.5),'MAE':float(BASE.mean_absolute_error(ytest,p))}; out=pd.DataFrame({'obs_id':test.obs_id.astype(str),'station_id':test.station.astype(str),'date':test.date,'target':ytest,'y_pred':p,'residual':p-ytest,'task':task,'model':model,'fold':fold,'feature_set':'X_strict_77','split_version':'MODERN_BASELINE_GATE_V1'}); out.to_csv(pred_path,index=False,encoding='utf-8-sig'); pd.DataFrame([mm|{'task':task,'model':model,'fold':fold,'n_train':len(train),'n_test':len(test)}]).to_csv(Path(row.metrics_file),index=False,encoding='utf-8-sig'); reg.loc[q,['status','end_time','runtime','R2','RMSE','MAE']]=['OK',datetime.now().isoformat(timespec='seconds'),time.perf_counter()-started,mm['R2'],mm['RMSE'],mm['MAE']]; reg.to_csv(RUN/'03_run_registry.csv',index=False); print('OK',task,model,fold,mm)
                except Exception as e:
                    reg.loc[q,['status','end_time','runtime','error']]=['FAILED',datetime.now().isoformat(timespec='seconds'),time.perf_counter()-started,f'{type(e).__name__}: {e}']; reg.to_csv(RUN/'03_run_registry.csv',index=False); raise
                finally:
                    try:
                        import torch; gc.collect(); torch.cuda.empty_cache()
                    except Exception: pass

def main():
    import argparse; a=argparse.ArgumentParser(); a.add_argument('--preflight-only',action='store_true'); a.add_argument('--run',action='store_true'); x=a.parse_args(); data,fs=load_data(); manifests=make_manifests(data); preflight(data,fs,manifests); 
    if x.run: run(data,fs,manifests)
if __name__=='__main__': main()
