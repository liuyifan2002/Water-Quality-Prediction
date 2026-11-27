"""Run XGBoost source-group leave-one-source-out ablation."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[2]; RUN=ROOT/'runs'/'multisource_ablation_v1'; MAIN=ROOT/'数据'/'处理后数据'/'正式主表'/'model_table_3depth_2014_2024.csv'; STRICT=ROOT/'runs'/'tabiclv2_pilot'/'01_strict_feature_manifest.csv'; CONFIG=ROOT/'结果'/'扩展实验'/'表格预训练算法对比_20260908'/'portable_package'/'config.json'; FOLDS=tuple(f'fold_{i}' for i in range(1,6)); TASKS=('chla_surface','secchi','tn_surface','tp_middle')
def load_features():
 t=pd.read_csv(RUN/'00_feature_source_taxonomy.csv'); return t
def manifest(task):
 if task=='secchi': return pd.read_csv(ROOT/'runs/modern_baseline_gate_v1/01_secchi_manifest.csv',dtype={'obs_id':'string','station_id':'string'})
 root={'chla_surface':ROOT/'runs/chla_family_v2','tn_surface':ROOT/'runs/tn_family_v2','tp_middle':ROOT/'runs/tp_family_v2'}[task]; depth={'chla_surface':'surface','tn_surface':'surface','tp_middle':'middle'}[task]; prefix={'chla_surface':'03_chla','tn_surface':'03_tn','tp_middle':'04_tp'}[task]; return pd.read_csv(root/f'{prefix}_{depth}_manifest.csv',dtype={'obs_id':'string','station_id':'string'})
def load_main():
 tax=load_features(); targets={'chla_surface':'chla_ug_l','secchi':'secchi_depth_m','tn_surface':'tn_mg_l','tp_middle':'tp_mg_l'}; cols=['obs_id','station','date','depth']+list(targets.values())+tax.feature_name.tolist(); d=pd.read_csv(MAIN,usecols=list(dict.fromkeys(cols)),low_memory=False); d.obs_id=d.obs_id.astype('string'); d.station=d.station.astype('string'); return d,targets,tax
def full_pred(task):
 if task=='secchi': return pd.read_csv(GATE/'predictions/secchi_xgb_frozen_for_gate.csv',dtype={'obs_id':'string','station_id':'string'})
 root={'chla_surface':ROOT/'runs/chla_family_v2','tn_surface':ROOT/'runs/tn_family_v2','tp_middle':ROOT/'runs/tp_family_v2'}[task]; return pd.concat([pd.read_csv(root/'predictions'/f'{task}_xgboost_{f}.csv',dtype={'obs_id':'string','station_id':'string'}) for f in FOLDS],ignore_index=True)
GATE=ROOT/'runs'/'modern_baseline_gate_v1'
def run():
 d,targets,tax=load_main(); (RUN/'predictions').mkdir(parents=True,exist_ok=True); cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig')); params=dict(cfg['model_parameters']['xgboost']); full_cache={t:full_pred(t) for t in TASKS}; rows=[]
 for group in sorted(tax.source_group.unique()):
  if group=='REMOTE_SENSING_OCEAN_COLOR': continue
  kept=tax.loc[tax.source_group.ne(group),'feature_name'].tolist(); removed=tax.loc[tax.source_group.eq(group),'feature_name'].tolist();
  for task in TASKS:
   target=targets[task]; depth={'chla_surface':'surface','secchi':None,'tn_surface':'surface','tp_middle':'middle'}[task]; m=manifest(task); td=d[d[target].notna()].copy();
   if depth is not None: td=td[td.depth==depth]
   for fold in FOLDS:
    tr=m[(m.fold==fold)&(m.role=='train')]; te=m[(m.fold==fold)&(m.role=='test')]; train=td[td.obs_id.isin(set(tr.obs_id))].set_index(td[td.obs_id.isin(set(tr.obs_id))].obs_id.astype(str)).loc[tr.obs_id.astype(str)].reset_index(drop=True); test=td[td.obs_id.isin(set(te.obs_id))].set_index(td[td.obs_id.isin(set(te.obs_id))].obs_id.astype(str)).loc[te.obs_id.astype(str)].reset_index(drop=True); imp=SimpleImputer(strategy='median',keep_empty_features=True); xtr=imp.fit_transform(train[kept]).astype('float32'); xte=imp.transform(test[kept]).astype('float32'); y=train[target].to_numpy(float); yt=test[target].to_numpy(float); yf=np.log1p(y) if np.nanmin(y)>=0 else y; start=time.perf_counter(); model=XGBRegressor(**params,random_state=42,objective='reg:squarederror'); model.fit(xtr,yf); p=model.predict(xte); p=np.expm1(p) if np.nanmin(y)>=0 else p; p=np.maximum(p,0); rows.append({'task':task,'source_group_removed':group,'fold':fold,'n_features':len(kept),'n_train':len(train),'n_test':len(test),'obs_id':test.obs_id.astype(str).tolist(),'station_id':test.station.astype(str).tolist(),'y_true':yt.tolist(),'y_pred':p.tolist(),'runtime':time.perf_counter()-start,'removed_features':'|'.join(removed)})
    print('DONE',group,task,flush=True)
 out=[]
 for r in rows:
  base=full_cache[r['task']]; frame=pd.DataFrame({'obs_id':r['obs_id'],'station_id':r['station_id'],'y_true':r['y_true'],'y_pred':r['y_pred']}); bf=base[base.fold.astype(str)==r['fold']][['obs_id','y_pred']].rename(columns={'y_pred':'full_pred'}); frame=frame.merge(bf,on=['obs_id'],validate='one_to_one'); y=frame.y_true.to_numpy(float); a=frame.full_pred.to_numpy(float); b=frame.y_pred.to_numpy(float); out.append({'task':r['task'],'source_group_removed':r['source_group_removed'],'fold':r['fold'],'n_features':r['n_features'],'n_train':r['n_train'],'n_test':r['n_test'],'full_R2':r2_score(y,a),'ablated_R2':r2_score(y,b),'delta_R2':r2_score(y,b)-r2_score(y,a),'full_RMSE':mean_squared_error(y,a)**.5,'ablated_RMSE':mean_squared_error(y,b)**.5,'delta_RMSE':mean_squared_error(y,b)**.5-mean_squared_error(y,a)**.5,'full_MAE':mean_absolute_error(y,a),'ablated_MAE':mean_absolute_error(y,b),'delta_MAE':mean_absolute_error(y,b)-mean_absolute_error(y,a),'runtime':r['runtime'],'removed_features':r['removed_features']}); frame['task']=r['task']; frame['source_group_removed']=r['source_group_removed']; frame['fold']=r['fold']; frame.to_csv(RUN/'predictions'/f"{r['task']}_{r['source_group_removed']}_{r['fold']}.csv",index=False)
 pd.DataFrame(out).to_csv(RUN/'02_ablation_fold_metrics.csv',index=False,encoding='utf-8-sig'); print('ABLATION_COMPLETE',len(out))
if __name__=='__main__':
 from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
 run()
