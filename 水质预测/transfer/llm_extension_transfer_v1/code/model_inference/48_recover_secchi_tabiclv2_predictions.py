"""恢复 Secchi 历史 TabICLv2 station-holdout 逐样本预测。"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]; RUN=ROOT/'runs'/'modern_baseline_gate_v1'; MAIN=ROOT/'数据'/'处理后数据'/'正式主表'/'model_table_3depth_2014_2024.csv'; SPLIT=ROOT/'runs'/'tabiclv2_pilot'/'02_split_manifest.csv'; STRICT=ROOT/'runs'/'tabiclv2_pilot'/'01_strict_feature_manifest.csv'; BASE_PATH=ROOT/'脚本'/'04_模型训练'/'33_run_tabiclv2_chla_smoke.py'; FOLDS=tuple(f'fold_{i}' for i in range(1,6))
def base():
 s=importlib.util.spec_from_file_location('secchi_tabicl_base',BASE_PATH); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m
B=base()
def main():
 f=pd.read_csv(STRICT); features=f.loc[f.included_in_pilot.astype(bool),'feature'].astype(str).tolist(); assert len(features)==77
 data=pd.read_csv(MAIN,usecols=['obs_id','station','date','secchi_depth_m',*features],low_memory=False); data.obs_id=data.obs_id.astype('string'); data.station=data.station.astype('string'); data.date=pd.to_datetime(data.date); data=data[data.secchi_depth_m.notna()].copy();
 split=pd.read_csv(SPLIT,dtype={'obs_id':'string'}); split=split[(split.task=='secchi')&(split.validation=='station_holdout')].copy(); split=split.rename(columns={'station':'station_id'}); all_rows=[]; metrics=[]
 for fold in FOLDS:
  q=split[split.fold==fold]; tr=data[data.obs_id.isin(set(q[q.split=='train'].obs_id))].copy(); te=data[data.obs_id.isin(set(q[q.split=='test'].obs_id))].copy(); tr=tr.set_index(tr.obs_id.astype(str)).loc[q[q.split=='train'].obs_id.astype(str)].reset_index(drop=True); te=te.set_index(te.obs_id.astype(str)).loc[q[q.split=='test'].obs_id.astype(str)].reset_index(drop=True); imp=SimpleImputer(strategy='median',keep_empty_features=True); xtr=imp.fit_transform(tr[features]).astype('float32'); xte=imp.transform(te[features]).astype('float32'); y=tr.secchi_depth_m.to_numpy(float); yt=te.secchi_depth_m.to_numpy(float); yf,inv=B.target_transform(y); start=time.perf_counter(); raw=B.fit_tabicl(xtr,yf,xte); pred=np.maximum(inv(raw),0); m={'fold':fold,'R2':r2_score(yt,pred),'RMSE':mean_squared_error(yt,pred)**.5,'MAE':mean_absolute_error(yt,pred),'runtime':time.perf_counter()-start,'n_train':len(tr),'n_test':len(te)}; metrics.append(m); all_rows.append(pd.DataFrame({'obs_id':te.obs_id.astype(str),'station_id':te.station.astype(str),'date':te.date.dt.strftime('%Y-%m-%d'),'target':'secchi','validation':'station','fold':fold,'y_true':yt,'y_pred':pred,'residual':pred-yt,'model':'tabicl','feature_set':'X_strict_77','split_source':'original_secchi_station_holdout_obs_id'})); print(fold,m,flush=True)
 hist=pd.read_csv(ROOT/'runs/tabiclv2_pilot/08_secchi_fold_results.csv'); hist=hist[hist.validation=='station'].sort_values('fold'); cur=pd.DataFrame(metrics).sort_values('fold'); cur['historical_R2']=hist.TabICLv2_R2.to_numpy(); cur['abs_diff_R2']=(cur.R2-cur.historical_R2).abs(); cur['status']=np.where(cur.abs_diff_R2<=0.03,'PASS','STOP'); cur.to_csv(RUN/'secchi_tabiclv2_recovery_check.csv',index=False,encoding='utf-8-sig');
 if (cur.status=='STOP').any(): raise RuntimeError('Secchi TabICLv2 fold R2 replication exceeded tolerance; historical results not overwritten')
 pd.concat(all_rows,ignore_index=True).to_csv(RUN/'predictions/secchi_tabiclv2_recovered.csv',index=False,encoding='utf-8-sig'); print('REPRODUCTION_PASS pooled_R2=',r2_score(pd.concat(all_rows).y_true,pd.concat(all_rows).y_pred),flush=True)
if __name__=='__main__': main()
