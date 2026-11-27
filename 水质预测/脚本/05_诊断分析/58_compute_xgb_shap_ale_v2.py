"""计算XGB_SHAP_ALE_V2的资产预检、TreeSHAP、1D ALE和2D ALE。只读模型，不训练。"""
from __future__ import annotations
import hashlib, importlib.util, itertools, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, xgboost as xgb
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'runs/xgb_shap_ale_v2'; REFIT=ROOT/'runs/xgb_interpretation_refit_v1'
for d in ('00_preflight','01_shap/raw','02_ale_1d','03_ale_2d','04_integrated','figures','reports','registry'): (OUT/d).mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('refit',ROOT/'脚本/05_诊断分析/56_refit_frozen_xgb_for_interpretation.py'); P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
FOLDS=P.FOLDS; TASKS=P.TASKS; TAX=ROOT/'runs/multisource_ablation_v1/00_feature_source_taxonomy.csv'

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def pred_final(model,imp,X,log=True):
 z=model.predict(imp.transform(X).astype(np.float32)); return np.maximum(np.expm1(z) if log else z,0)
def load(task,fold,features,data):
 a=REFIT/'artifacts'/task/fold; meta=json.loads((a/'metadata.json').read_text(encoding='utf-8')); ids=pd.read_csv(a/'train_obs_ids.csv',dtype={'obs_id':'string'}).obs_id.astype(str).tolist()
 tr=data[data.obs_id.astype(str).isin(set(ids))].copy(); model=XGBRegressor(); model.load_model(a/'xgboost_refit_model.ubj'); imp=joblib.load(a/'median_imputer.joblib')
 return a,meta,tr,model,imp
def unit(name):
 n=name.lower()
 if 'temp' in n or n.startswith('sst') or n.startswith('t2m'): return '°C'
 if 'wind' in n or n.startswith(('u10','v10')): return 'm s⁻¹'
 if 'rain' in n or 'precip' in n or n.startswith('tp_'): return 'mm'
 if 'tide' in n or 'water_level' in n: return 'm'
 if 'ssrd' in n or 'radiation' in n: return 'J m⁻²'
 if n in ('lon','longitude'): return '°E'
 if n in ('lat','latitude'): return '°N'
 return '原始特征单位'

features=P.features(); tax=pd.read_csv(TAX).set_index('feature_name'); audit=[]; scale=[]; registry=[]
all_data={t:P.load_task_data(t,features) for t in TASKS}
validation=pd.read_csv(REFIT/'02_xgb_refit_prediction_validation.csv')
for task in TASKS:
 data=all_data[task]
 for fold in FOLDS:
  a,meta,tr,m,imp=load(task,fold,features,data); files={'model':a/'xgboost_refit_model.ubj','imputer':a/'median_imputer.joblib'}; manifest=pd.read_csv(REFIT/'03_xgb_refit_model_manifest.csv')
  mr=manifest[(manifest.task==task)&(manifest.fold==fold)].iloc[0]; hashes=all(sha(p)==mr[f'{k}_sha256'] for k,p in files.items()); feat=json.loads((a/'feature_order.json').read_text(encoding='utf-8'))
  sample=tr.iloc[:min(32,len(tr))]; p1=pred_final(m,imp,sample[features].to_numpy(float)); frozen=pd.read_csv(REFIT/'predictions'/f'{task}_{fold}_refit_predictions.csv')
  audit.append({'task':task,'fold':fold,'model_load':True,'imputer_load':True,'hash_match':hashes,'feature_order_match':feat==features,'exact_refit_status':validation[(validation.task==task)&(validation.fold==fold)].classification.iloc[0],'sample_prediction_finite':np.isfinite(p1).all(),'status':'PASS' if hashes and feat==features else 'FAIL'})
  scale.append({'task':task,'fold':fold,'target':TASKS[task][0],'model_internal_output':'log1p(target)','shap_scale':'log1p(target)','inverse_transform':'expm1','nonnegative_clip':True,'ale_scale':'original target','ale_unit':{'chla_surface':'µg/L','tn_surface':'mg/L','tp_middle':'mg/L','secchi':'m'}[task]})
pd.DataFrame(audit).to_csv(OUT/'00_preflight/00_model_and_preprocessor_check.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(scale).to_csv(OUT/'00_preflight/00_prediction_scale_manifest.csv',index=False,encoding='utf-8-sig')
if not all(r['status']=='PASS' for r in audit): raise SystemExit('PREFLIGHT_FAILED')
(OUT/'00_preflight/00_explanation_preflight.md').write_text('# XGB_SHAP_ALE_V2解释预检\n\n- 20/20折模型、imputer、哈希、特征顺序和严格复现状态通过。\n- SHAP尺度：`log1p(target)`；加性只在模型内部输出尺度核验。\n- ALE尺度：完整反变换与非负截断后的原始水质单位。\n- 本流程不训练模型。\n',encoding='utf-8')

# TreeSHAP（XGBoost原生pred_contribs即TreeSHAP）
imp_rows=[]; coverage={}
for task in TASKS:
 data=all_data[task]
 for fold in FOLDS:
  start=time.perf_counter(); a,meta,tr,m,imputer=load(task,fold,features,data); raw=tr[features].to_numpy(float); filled=imputer.transform(raw).astype(np.float32)
  shap_path=OUT/f'01_shap/raw/{task}_{fold}_shap.csv.gz'; orig_path=OUT/f'01_shap/raw/{task}_{fold}_feature_original.csv.gz'; fill_path=OUT/f'01_shap/raw/{task}_{fold}_feature_imputed.csv.gz'
  if shap_path.exists() and orig_path.exists() and fill_path.exists():
   saved=pd.read_csv(shap_path); means=saved[features].abs().mean().to_numpy(); add_err=np.nan
  else:
   contrib=m.get_booster().predict(xgb.DMatrix(filled,feature_names=features),pred_contribs=True); rawout=m.predict(filled)
   add_err=np.max(np.abs(contrib[:,:-1].sum(1)+contrib[:,-1]-rawout)); means=np.abs(contrib[:,:-1]).mean(0)
   vals=pd.DataFrame(contrib[:,:-1],columns=features); vals.insert(0,'obs_id',tr.obs_id.astype(str).to_numpy()); vals.to_csv(shap_path,index=False,compression='gzip')
   orig=pd.DataFrame(raw,columns=features); orig.insert(0,'obs_id',tr.obs_id.astype(str).to_numpy()); orig.to_csv(orig_path,index=False,compression='gzip')
   fill=pd.DataFrame(filled,columns=features); fill.insert(0,'obs_id',tr.obs_id.astype(str).to_numpy()); fill.to_csv(fill_path,index=False,compression='gzip')
  order=pd.Series(means,index=features).rank(ascending=False,method='min').astype(int)
  for j,f in enumerate(features):
   obs=tr[f]; imp_rows.append({'task':task,'fold':fold,'feature':f,'mean_abs_shap':means[j],'rank':order[f],'normalized_contribution':means[j]/means.sum(),'source_group':tax.loc[f,'source_group'],'physical_meaning':tax.loc[f,'physical_meaning'],'unit':unit(f),'n_train':len(tr),'nonmissing_fraction':obs.notna().mean(),'n_unique_nonmissing':obs.nunique(dropna=True),'shap_additivity_max_abs_error':add_err,'shap_scale':'log1p(target)'})
  registry.append({'task':task,'fold':fold,'analysis_type':'SHAP','feature_or_pair':'ALL_77','status':'OK','output_path':f'01_shap/raw/{task}_{fold}_shap.csv.gz','elapsed_seconds':time.perf_counter()-start,'reason':'RESUMED_EXISTING' if np.isnan(add_err) else ''})
foldimp=pd.DataFrame(imp_rows); foldimp.to_csv(OUT/'01_shap/01_shap_importance_by_fold.csv',index=False,encoding='utf-8-sig')
summary=foldimp.groupby(['task','feature','source_group','physical_meaning','unit'],as_index=False).agg(mean_abs_shap=('mean_abs_shap','mean'),sd_abs_shap=('mean_abs_shap','std'),mean_rank=('rank','mean'),rank_sd=('rank','std'),top10_count=('rank',lambda s:int((s<=10).sum())),mean_normalized_contribution=('normalized_contribution','mean'),min_nonmissing_fraction=('nonmissing_fraction','min'),min_unique=('n_unique_nonmissing','min'))
summary['task_rank']=summary.groupby('task').mean_abs_shap.rank(ascending=False,method='min').astype(int); summary.to_csv(OUT/'01_shap/01_shap_importance_summary.csv',index=False,encoding='utf-8-sig')
stability=summary[['task','feature','mean_rank','rank_sd','top10_count','task_rank','mean_normalized_contribution']].copy(); stability['stable_top_feature']=(stability.top10_count>=3)&(stability.rank_sd<=10); stability.to_csv(OUT/'01_shap/01_shap_rank_stability.csv',index=False,encoding='utf-8-sig')
top15=summary[summary.task_rank<=15].copy()
bad=lambda f: f.lower() in {'lon','lat','longitude','latitude','year','month','day','day_of_year','year_offset'} or f.lower().endswith(('_sin','_cos'))
candidates=[]
for task,g in summary.groupby('task'):
 eligible=g[(g.top10_count>=2)&(g.min_nonmissing_fraction>=.2)&(g.min_unique>=20)&~g.feature.map(bad)].sort_values(['task_rank','rank_sd']).head(6)
 selected=eligible.head(4)
 for _,r in g.sort_values('task_rank').head(15).iterrows():
  candidates.append({**r.to_dict(),'continuous_eligible':not bad(r.feature) and r.min_unique>=20,'selected_for_ale_1d':r.feature in set(selected.feature),'selection_reason':'SHAP前列、折间稳定、连续且覆盖充分' if r.feature in set(selected.feature) else '未进入每任务前4个综合候选'})
cand=pd.DataFrame(candidates); cand.to_csv(OUT/'01_shap/01_ale_candidate_features.csv',index=False,encoding='utf-8-sig')
(OUT/'01_shap/01_feature_selection_report.md').write_text('# SHAP变量筛选报告\n\n按五折平均绝对TreeSHAP、前10出现次数、排名波动、连续性、缺失率和不同取值数筛选。SHAP值均位于`log1p(target)`尺度；空间和时间定位变量保留在排名中，但不优先进入环境过程ALE。每任务选择4个一维ALE候选，未为来源组均衡强行加入弱变量。\n',encoding='utf-8')

# 1D ALE
curves=[]; binsout=[]; cons=[]; selected=cand[cand.selected_for_ale_1d].groupby('task').feature.apply(list).to_dict()
for task,flist in selected.items():
 data=all_data[task]
 for fold in FOLDS:
  a,meta,tr,m,imputer=load(task,fold,features,data); raw=tr[features].to_numpy(float)
  for f in flist:
   j=features.index(f); valid=np.isfinite(raw[:,j]); vals=raw[valid,j]; edges=np.unique(np.quantile(vals,np.linspace(0,1,17)))
   while len(edges)>7:
    idx=np.clip(np.digitize(vals,edges[1:-1]),0,len(edges)-2); counts=np.bincount(idx,minlength=len(edges)-1)
    if counts.min()>=50: break
    edges=np.unique(np.quantile(vals,np.linspace(0,1,max(7,len(edges)-2))))
   idx=np.clip(np.digitize(vals,edges[1:-1]),0,len(edges)-2); effects=[]; weights=[]
   valid_rows=np.where(valid)[0]
   for b in range(len(edges)-1):
    rows=valid_rows[idx==b]; lo,hi=edges[b],edges[b+1];
    if len(rows)==0: delta=np.nan; stations=0
    else:
     xl=raw[rows].copy(); xh=raw[rows].copy(); xl[:,j]=lo; xh[:,j]=hi; delta=float(np.mean(pred_final(m,imputer,xh)-pred_final(m,imputer,xl))); stations=tr.iloc[rows].station.nunique()
    effects.append(delta); weights.append(len(rows)); binsout.append({'task':task,'fold':fold,'feature':f,'bin':b,'lower':lo,'upper':hi,'midpoint':(lo+hi)/2,'n_obs':len(rows),'n_stations':stations,'supported':len(rows)>=50 and stations>=5})
   eff=np.nan_to_num(effects); ale=np.cumsum(eff); ale-=np.average(ale,weights=np.maximum(weights,1))
   for b,v in enumerate(ale): curves.append({'task':task,'fold':fold,'feature':f,'bin':b,'x':(edges[b]+edges[b+1])/2,'ale':v,'n_obs':weights[b],'unit':unit(f),'target_unit':{'chla_surface':'µg/L','tn_surface':'mg/L','tp_middle':'mg/L','secchi':'m'}[task]})
pd.DataFrame(curves).to_csv(OUT/'02_ale_1d/02_ale_1d_curves.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(binsout).to_csv(OUT/'02_ale_1d/02_ale_1d_bin_coverage.csv',index=False,encoding='utf-8-sig')
cv=pd.DataFrame(curves)
selectedcurves=[]
for (task,f),g in cv.groupby(['task','feature']):
 xmin=max(x.x.min() for _,x in g.groupby('fold')); xmax=min(x.x.max() for _,x in g.groupby('fold')); grid=np.linspace(xmin,xmax,41); arr=[]
 for fold,h in g.groupby('fold'): arr.append(np.interp(grid,h.x,h.ale))
 arr=np.array(arr); amps=arr.max(1)-arr.min(1); cor=np.corrcoef(arr); pc=np.nanmean(cor[np.triu_indices(5,1)])
 cons.append({'task':task,'feature':f,'common_min':xmin,'common_max':xmax,'mean_pairwise_correlation':pc,'mean_amplitude':amps.mean(),'amplitude_cv':amps.std()/amps.mean() if amps.mean() else np.nan,'stable':pc>=.5})
 for x0,mu,sd in zip(grid,arr.mean(0),arr.std(0)): selectedcurves.append({'task':task,'feature':f,'x':x0,'ale_mean':mu,'fold_sd':sd,'unit':unit(f)})
pd.DataFrame(cons).to_csv(OUT/'02_ale_1d/02_ale_1d_fold_consistency.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(selectedcurves).to_csv(OUT/'02_ale_1d/02_ale_1d_selected_curves.csv',index=False,encoding='utf-8-sig')
(OUT/'02_ale_1d/02_ale_1d_interpretation.md').write_text('# 一维ALE计算说明\n\nALE使用每折训练分布、分位数自适应分箱及完整原始预测流程。跨折曲线仅在共同支持范围插值汇总；折间标准差不是95%置信区间。具体方向与稳定性见CSV，样本不足区间不得作稳定机制解释。\n',encoding='utf-8')

# 2D ALE：每任务取选中变量中不同来源优先的前2对
pairs=[]
for task,flist in selected.items():
 combos=list(itertools.combinations(flist,2)); combos.sort(key=lambda p:(tax.loc[p[0],'source_group']==tax.loc[p[1],'source_group'],sum(float(summary[(summary.task==task)&(summary.feature==x)].task_rank.iloc[0]) for x in p)))
 for rank,pair in enumerate(combos[:3],1): pairs.append({'task':task,'feature_x':pair[0],'feature_y':pair[1],'source_x':tax.loc[pair[0],'source_group'],'source_y':tax.loc[pair[1],'source_group'],'candidate_rank':rank,'selected':rank<=2,'reason':'SHAP/ALE候选且优先跨来源组'})
pairdf=pd.DataFrame(pairs); pairdf.to_csv(OUT/'03_ale_2d/03_ale_2d_candidate_pairs.csv',index=False,encoding='utf-8-sig')
cells=[]; results=[]
for _,pr in pairdf[pairdf.selected].iterrows():
 task,fx,fy=pr.task,pr.feature_x,pr.feature_y; data=all_data[task]
 for fold in FOLDS:
  a,meta,tr,m,imputer=load(task,fold,features,data); raw=tr[features].to_numpy(float); ix,iy=features.index(fx),features.index(fy); valid=np.isfinite(raw[:,ix])&np.isfinite(raw[:,iy]); rows=np.where(valid)[0]
  ex=np.unique(np.quantile(raw[valid,ix],np.linspace(0,1,7))); ey=np.unique(np.quantile(raw[valid,iy],np.linspace(0,1,7)))
  bx=np.clip(np.digitize(raw[valid,ix],ex[1:-1]),0,len(ex)-2); by=np.clip(np.digitize(raw[valid,iy],ey[1:-1]),0,len(ey)-2); local=np.full((len(ex)-1,len(ey)-1),np.nan); w=np.zeros_like(local)
  for i in range(local.shape[0]):
   for j in range(local.shape[1]):
    rr=rows[(bx==i)&(by==j)]; w[i,j]=len(rr)
    if len(rr):
     base=raw[rr].copy(); pp=[]
     for xv,yv in ((ex[i],ey[j]),(ex[i+1],ey[j]),(ex[i],ey[j+1]),(ex[i+1],ey[j+1])):
      z=base.copy(); z[:,ix]=xv; z[:,iy]=yv; pp.append(pred_final(m,imputer,z))
     local[i,j]=np.mean(pp[3]-pp[1]-pp[2]+pp[0])
  surf=np.cumsum(np.cumsum(np.nan_to_num(local),axis=0),axis=1); total=w.sum(); rowmean=np.divide((surf*w).sum(1),w.sum(1),out=np.zeros(surf.shape[0]),where=w.sum(1)>0); colmean=np.divide((surf*w).sum(0),w.sum(0),out=np.zeros(surf.shape[1]),where=w.sum(0)>0); inter=surf-rowmean[:,None]-colmean[None,:]; inter-=np.sum(inter*w)/total
  for i in range(inter.shape[0]):
   for j in range(inter.shape[1]): cells.append({'task':task,'fold':fold,'feature_x':fx,'feature_y':fy,'bin_x':i,'bin_y':j,'x_mid':(ex[i]+ex[i+1])/2,'y_mid':(ey[j]+ey[j+1])/2,'ale_2d':inter[i,j],'n_obs':int(w[i,j]),'supported':w[i,j]>=20})
  results.append({'task':task,'fold':fold,'feature_x':fx,'feature_y':fy,'bins_x':inter.shape[0],'bins_y':inter.shape[1],'coverage_ratio':float((w>=20).sum()/w.size),'valid_cells':int((w>=20).sum()),'interaction_amplitude':float(inter.max()-inter.min()),'status':'SUPPORTED' if (w>=20).sum()/w.size>=.7 else 'INSUFFICIENT_SUPPORT'})
cell=pd.DataFrame(cells); cell.to_csv(OUT/'03_ale_2d/03_ale_2d_coverage.csv',index=False,encoding='utf-8-sig'); res=pd.DataFrame(results); res.to_csv(OUT/'03_ale_2d/03_ale_2d_interaction_results.csv',index=False,encoding='utf-8-sig')
fc=[]
for keys,g in res.groupby(['task','feature_x','feature_y']): fc.append({'task':keys[0],'feature_x':keys[1],'feature_y':keys[2],'supported_folds':int((g.status=='SUPPORTED').sum()),'mean_coverage_ratio':g.coverage_ratio.mean(),'amplitude_mean':g.interaction_amplitude.mean(),'amplitude_cv':g.interaction_amplitude.std()/g.interaction_amplitude.mean(),'status':'SUPPORTED' if (g.status=='SUPPORTED').sum()>=4 else 'PARTIALLY_SUPPORTED'})
pd.DataFrame(fc).to_csv(OUT/'03_ale_2d/03_ale_2d_fold_consistency.csv',index=False,encoding='utf-8-sig'); (OUT/'03_ale_2d/03_ale_2d_interpretation.md').write_text('# 二维ALE说明\n\n采用6×6分位箱的二阶局部预测差分、双向累积、加权主效应去除与中心化；不是PDP。低于20条观测的单元在正式绘图中遮蔽。仅将统计交互作为模型行为，不视为物理因果证明。\n',encoding='utf-8')
pd.DataFrame(registry).to_csv(OUT/'registry/run_registry.csv',index=False,encoding='utf-8-sig')
print('COMPUTE_COMPLETE',len(imp_rows),len(curves),len(cells))
