"""Step 0/1 for MULTISOURCE_ABLATION_V1: metric and feature-source audit."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]; RUN=ROOT/'runs'/'multisource_ablation_v1'; RUN.mkdir(parents=True,exist_ok=True)
STRICT=ROOT/'runs'/'tabiclv2_pilot'/'01_strict_feature_manifest.csv'; GATE=ROOT/'runs'/'modern_baseline_gate_v1'; FOLDS=tuple(f'fold_{i}' for i in range(1,6)); TASKS=('chla_surface','secchi','tn_surface','tp_middle')
def metric(y,p): return r2_score(y,p),mean_squared_error(y,p)**.5,mean_absolute_error(y,p)
def feature_taxonomy():
 d=pd.read_csv(STRICT); d=d[d.included_in_pilot.astype(bool)].copy(); rows=[]
 for _,r in d.iterrows():
  f=str(r.feature); g=str(r.group); source=str(r.source); unresolved='';
  if g=='空间与站点': primary='SPATIAL_STATIC'; dynamic='static'; meaning='站点经纬度和空间位置背景'
  elif g=='气象': primary='METEOROLOGY'; dynamic='dynamic'; meaning='气温、云量、风速风向及其合法滞后/交互项'
  elif g=='ERA5' and f.startswith('sst_'): primary='OCEAN_PHYSICS'; dynamic='dynamic'; meaning='ERA5/海面温度窗口统计及物理交互项'
  elif g=='ERA5': primary='METEOROLOGY'; dynamic='dynamic'; meaning='ERA5短期辐射窗口统计及交互项'
  elif g in ('降雨','潮汐'): primary='RAINFALL_HYDRO_TIDE'; dynamic='dynamic'; meaning='降雨或潮汐窗口统计及交互项'
  elif g=='时间': primary='TEMPORAL_PROCESS'; dynamic='dynamic'; meaning='日历、季节和采样间隔过程变量'
  elif g=='夜间灯光': primary='HUMAN_ACTIVITY'; dynamic='static_or_dynamic'; meaning='夜间灯光强度、变化、滞后和覆盖数量'
  else: primary='UNRESOLVED'; dynamic='unknown'; meaning='无法从冻结 manifest 的来源字段确定'; unresolved='manifest group/source 未匹配已定义来源规则'
  rows.append({'feature_name':f,'source_group':primary,'original_source':source,'physical_meaning':meaning,'static_dynamic':dynamic,'notes':unresolved or 'primary group determined from frozen manifest group and source'})
 out=pd.DataFrame(rows); out.to_csv(RUN/'00_feature_source_taxonomy.csv',index=False,encoding='utf-8-sig');
 if len(out)!=77 or out.feature_name.duplicated().any() or out.source_group.eq('UNRESOLVED').any(): raise RuntimeError(f'taxonomy failed rows={len(out)} unresolved={out.source_group.eq("UNRESOLVED").sum()} duplicate={out.feature_name.duplicated().sum()}')
 out.groupby('source_group',sort=True).size().rename('n_features').reset_index().to_csv(RUN/'00_feature_source_group_counts.csv',index=False,encoding='utf-8-sig'); print(out.groupby('source_group').size().to_string()); print('TAXONOMY=PASS'); return out
def load_gate(task,model):
 p=GATE/'predictions'/f'{task}_{model}_fold_1.csv'
 frames=[pd.read_csv(GATE/'predictions'/f'{task}_{model}_{f}.csv',dtype={'obs_id':'string','station_id':'string'}) for f in FOLDS]; return pd.concat(frames,ignore_index=True)
def metrics_step0():
 rows=[]
 for task in TASKS:
  for model in ('tabm','tabpfn'):
   d=load_gate(task,model); y=d.target.to_numpy(float); p=d.y_pred.to_numpy(float); r,rm,ma=metric(y,p); folds=d.groupby('fold').apply(lambda x:r2_score(x.target,x.y_pred),include_groups=False).to_numpy(); rows.append({'task':task,'model':model,'pooled_OOF_R2':r,'pooled_RMSE':rm,'pooled_MAE':ma,'fold_R2_mean':folds.mean(),'fold_R2_sd':folds.std(ddof=1),'n_obs':len(d),'n_station':d.station_id.nunique()})
  # Frozen xgb/tabicl predictions are handled from their family outputs; Secchi TabICL recovery is now part of the Gate directory.
  for model in ('xgboost','tabiclv2'):
   for task in TASKS:
    if task=='secchi': p=GATE/'predictions'/('secchi_tabiclv2_recovered.csv' if model=='tabiclv2' else 'secchi_xgb_frozen_for_gate.csv')
    if task=='secchi' and model=='xgboost' and not p.exists():
     source=pd.read_csv(ROOT/'runs/tabiclv2_pilot/09_secchi_predictions.csv'); source=source[(source.validation=='station')&(source.model.astype(str).str.lower()=='xgb')].copy(); source.to_csv(p,index=False)
    else:
     p=None
    if p is not None and p.exists():
     d=pd.read_csv(p); y=d.y_true.to_numpy(float); q=d.y_pred.to_numpy(float); r,rm,ma=metric(y,q); fs=d.groupby('fold').apply(lambda x:r2_score(x.y_true,x.y_pred),include_groups=False).to_numpy(); rows.append({'task':task,'model':model,'pooled_OOF_R2':r,'pooled_RMSE':rm,'pooled_MAE':ma,'fold_R2_mean':fs.mean(),'fold_R2_sd':fs.std(ddof=1),'n_obs':len(d),'n_station':d.station_id.nunique()})
    elif model in ('xgboost','tabiclv2') and task != 'secchi':
     root={'chla_surface':ROOT/'runs/chla_family_v2','tn_surface':ROOT/'runs/tn_family_v2','tp_middle':ROOT/'runs/tp_family_v2'}[task]; frames=[pd.read_csv(root/'predictions'/f'{task}_{model}_{f}.csv') for f in FOLDS]; d=pd.concat(frames); y=d.y_true.to_numpy(float); q=d.y_pred.to_numpy(float); r,rm,ma=metric(y,q); fs=d.groupby('fold').apply(lambda x:r2_score(x.y_true,x.y_pred),include_groups=False).to_numpy(); rows.append({'task':task,'model':model,'pooled_OOF_R2':r,'pooled_RMSE':rm,'pooled_MAE':ma,'fold_R2_mean':fs.mean(),'fold_R2_sd':fs.std(ddof=1),'n_obs':len(d),'n_station':d.station_id.nunique()})
 out=pd.DataFrame(rows); out.to_csv(RUN/'01_metric_consistent_core_table.csv',index=False,encoding='utf-8-sig'); return out
def metrics_step0():
 rows=[]
 def add(task, model, d, y_col='y_true', pred_col='y_pred'):
  y=d[y_col].to_numpy(float); p=d[pred_col].to_numpy(float); r,rm,ma=metric(y,p)
  fs=d.groupby('fold', sort=True).apply(lambda x:r2_score(x[y_col],x[pred_col]),include_groups=False).to_numpy()
  rows.append({'task':task,'model':model,'pooled_OOF_R2':r,'pooled_RMSE':rm,'pooled_MAE':ma,'fold_R2_mean':fs.mean(),'fold_R2_sd':fs.std(ddof=1),'n_obs':len(d),'n_station':d.station_id.nunique()})
 for task in TASKS:
  for model in ('tabm','tabpfn'): add(task, model, load_gate(task,model), 'target', 'y_pred')
 roots={'chla_surface':ROOT/'runs/chla_family_v2','tn_surface':ROOT/'runs/tn_family_v2','tp_middle':ROOT/'runs/tp_family_v2'}
 for task in TASKS:
  for model in ('xgboost','tabiclv2'):
   if task=='secchi':
    p=GATE/'predictions'/('secchi_tabiclv2_recovered.csv' if model=='tabiclv2' else 'secchi_xgb_frozen_for_gate.csv')
    add(task,model,pd.read_csv(p))
   else:
    d=pd.concat([pd.read_csv(roots[task]/'predictions'/f'{task}_{model}_{f}.csv') for f in FOLDS],ignore_index=True)
    add(task,model,d)
 out=pd.DataFrame(rows); out.to_csv(RUN/'01_metric_consistent_core_table.csv',index=False,encoding='utf-8-sig'); return out

def main():
 tax=feature_taxonomy(); core=metrics_step0(); print(core.to_string(index=False)); print('STEP0_STEP1=PASS')
if __name__=='__main__': main()
