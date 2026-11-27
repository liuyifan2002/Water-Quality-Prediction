"""基于V3完整SHAP候选运行TabICLv2 1D/2D ALE。"""
from __future__ import annotations
import argparse, importlib.util, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/tabicl_v2_explainability_v3';BASE_PATH=ROOT/'脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py';BINS=10;REF_N=64;GRID=5;GRID_REF_N=128
def base():
 s=importlib.util.spec_from_file_location('ale_v3_base',BASE_PATH);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
B=base()
def append(path,frame,keys):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  old=pd.read_csv(path);sig=frame[keys].astype(str).agg('|'.join,axis=1);osig=old[keys].astype(str).agg('|'.join,axis=1);old=old[~osig.isin(set(sig))];frame=pd.concat([old,frame],ignore_index=True)
 frame.to_csv(path,index=False,encoding='utf-8-sig')
def run(task,fold):
 features=B.load_features();train,test=B.load_fold(task,fold,features);target=B.TASKS[task]['target'];imputer=SimpleImputer(strategy='median',keep_empty_features=True);xtr=imputer.fit_transform(train[features]).astype('float32');xte=imputer.transform(test[features]).astype('float32');y0=train[target].to_numpy(float);y=np.log1p(y0).astype('float32') if np.nanmin(y0)>=0 else y0.astype('float32');inv=np.expm1 if np.nanmin(y0)>=0 else (lambda z:z);model=B.load_model(xtr,y)
 def pred(z):return np.maximum(inv(model.predict(np.asarray(z,dtype='float32'))),0)
 formal=pred(xte);old=B.frozen(task,fold);delta=np.abs(formal-old.y_pred.to_numpy(float));exact=bool(test.obs_id.astype(str).tolist()==old.obs_id.astype(str).tolist() and delta.max()<=B.MAX_ABS_TOL and delta.mean()<=B.MEAN_ABS_TOL)
 append(RUN/'00_ale_inference_reproduction.csv',pd.DataFrame([{'task':task,'fold':fold,'status':'EXACT_REPRODUCTION' if exact else 'REPRODUCTION_FAILED','max_abs_error':delta.max(),'mean_abs_error':delta.mean()}]),['task','fold'])
 if not exact:raise RuntimeError(f'{task}/{fold} ALE reproduction failed')
 cfg=pd.read_csv(RUN/'config'/f'{task}_ale_features.csv');selected=cfg.feature.astype(str).tolist();idx=B.sample_positions(test,min(REF_N,len(test)),30000+int(fold[-1])+100*list(B.TASKS).index(task));xr=xte[idx];stations=test.iloc[idx].station_id.astype(str).to_numpy();rows=[]
 for feature in selected:
  j=features.index(feature);edges=np.unique(np.quantile(xtr[:,j],np.linspace(0,1,BINS+1)));bid=np.clip(np.digitize(xr[:,j],edges[1:-1],right=True),0,len(edges)-2);lo=xr.copy();hi=xr.copy();lo[:,j]=edges[bid];hi[:,j]=edges[bid+1];t=time.perf_counter();p=pred(np.vstack([lo,hi]));elapsed=time.perf_counter()-t;local=p[len(lo):]-p[:len(lo)];le=np.array([np.mean(local[bid==b]) if np.any(bid==b) else np.nan for b in range(len(edges)-1)]);count=np.array([(bid==b).sum() for b in range(len(edges)-1)]);nst=np.array([len(set(stations[bid==b])) for b in range(len(edges)-1)]);acc=np.cumsum(np.nan_to_num(le));acc-=np.average(acc,weights=np.maximum(count,1))
  for b in range(len(edges)-1):rows.append({'task':task,'fold':fold,'feature':feature,'bin':b+1,'lower':edges[b],'upper':edges[b+1],'center':(edges[b]+edges[b+1])/2,'local_effect':le[b],'ale':acc[b],'n_samples':count[b],'n_stations':nst[b],'reference':'station-holdout test features; labels unused','elapsed_seconds_feature':elapsed})
 append(RUN/'ale_1d/ale_1d_by_fold.csv',pd.DataFrame(rows),['task','fold','feature','bin'])
 # Full test coverage audit for all candidate pairs.
 coverage=[]
 for a,b in combinations(selected,2):
  ia,ib=features.index(a),features.index(b);ea=np.unique(np.quantile(xtr[:,ia],np.linspace(0,1,GRID+1)));eb=np.unique(np.quantile(xtr[:,ib],np.linspace(0,1,GRID+1)));ca=np.clip(np.digitize(xte[:,ia],ea[1:-1],right=True),0,len(ea)-2);cb=np.clip(np.digitize(xte[:,ib],eb[1:-1],right=True),0,len(eb)-2);supported=0;total=(len(ea)-1)*(len(eb)-1)
  for i in range(len(ea)-1):
   for j in range(len(eb)-1):
    mask=(ca==i)&(cb==j);supported+=int(mask.sum()>=20 and test.loc[mask,'station_id'].nunique()>=3)
  coverage.append({'task':task,'fold':fold,'feature_1':a,'feature_2':b,'grid':f'{len(ea)-1}x{len(eb)-1}','supported_cells':supported,'total_cells':total,'support_fraction':supported/total,'threshold':'n>=20 and stations>=3'})
 cov=pd.DataFrame(coverage);append(RUN/'ale_2d/joint_coverage.csv',cov,['task','fold','feature_1','feature_2']);best=cov.sort_values('support_fraction',ascending=False).iloc[0]
 if best.support_fraction<.60:return
 a,b=best.feature_1,best.feature_2;ia,ib=features.index(a),features.index(b);gidx=B.sample_positions(test,min(GRID_REF_N,len(test)),40000+int(fold[-1])+100*list(B.TASKS).index(task));x=xte[gidx];st=test.iloc[gidx].station_id.astype(str).to_numpy();ea=np.unique(np.quantile(xtr[:,ia],np.linspace(0,1,GRID+1)));eb=np.unique(np.quantile(xtr[:,ib],np.linspace(0,1,GRID+1)));ca=np.clip(np.digitize(x[:,ia],ea[1:-1],right=True),0,len(ea)-2);cb=np.clip(np.digitize(x[:,ib],eb[1:-1],right=True),0,len(eb)-2);variants=[]
 for da,db in [(0,0),(1,0),(0,1),(1,1)]:z=x.copy();z[:,ia]=ea[ca+da];z[:,ib]=eb[cb+db];variants.append(z)
 p=pred(np.vstack(variants));n=len(x);local=p[3*n:]-p[n:2*n]-p[2*n:3*n]+p[:n];grid=np.full((len(ea)-1,len(eb)-1),np.nan);counts=np.zeros_like(grid,int);nst=np.zeros_like(grid,int)
 for i in range(grid.shape[0]):
  for j in range(grid.shape[1]):
   mask=(ca==i)&(cb==j);counts[i,j]=mask.sum();nst[i,j]=len(set(st[mask]));grid[i,j]=local[mask].mean() if mask.any() else np.nan
 acc=np.cumsum(np.cumsum(np.nan_to_num(grid),0),1);w=counts;rw=w.sum(1);cw=w.sum(0);rmean=np.array([np.average(acc[i],weights=np.maximum(cw,1)) for i in range(len(rw))]);cmean=np.array([np.average(acc[:,j],weights=np.maximum(rw,1)) for j in range(len(cw))]);inter=acc-rmean[:,None]-cmean[None,:]+np.sum(acc*w)/max(w.sum(),1);out=[]
 for i in range(grid.shape[0]):
  for j in range(grid.shape[1]):out.append({'task':task,'fold':fold,'feature_1':a,'feature_2':b,'bin_1':i+1,'bin_2':j+1,'ale_2d':inter[i,j],'local_second_difference':grid[i,j],'n_samples':counts[i,j],'n_stations':nst[i,j],'sparse_mask':bool(counts[i,j]<5 or nst[i,j]<2),'precheck_support_fraction':best.support_fraction})
 append(RUN/'ale_2d/ale_2d_by_fold.csv',pd.DataFrame(out),['task','fold','feature_1','feature_2','bin_1','bin_2'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=list(B.TASKS),required=True);p.add_argument('--fold',choices=[f'fold_{i}' for i in range(1,6)],required=True);a=p.parse_args();run(a.task,a.fold)
if __name__=='__main__':main()
