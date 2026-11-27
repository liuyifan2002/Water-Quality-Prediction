"""汇总V3完整77维SHAP并生成ALE候选。"""
from pathlib import Path
import importlib.util, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/tabicl_v2_explainability_v3';TAX=ROOT/'runs/multisource_ablation_v1/00_feature_source_taxonomy.csv';BASE_PATH=ROOT/'脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py'

def base():
 s=importlib.util.spec_from_file_location('v3_summary_base',BASE_PATH);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def main():
 (RUN/'tables').mkdir(parents=True,exist_ok=True)
 files=sorted((RUN/'shap_values').glob('*/*/*.csv.gz'));frames=[pd.read_csv(p) for p in files];raw=pd.concat(frames,ignore_index=True);raw['abs_shap']=raw.shap_value.abs()
 tax=pd.read_csv(TAX)[['feature_name','source_group','physical_meaning']].rename(columns={'feature_name':'feature'})
 byfold=raw.groupby(['task','fold','feature'],as_index=False).agg(mean_abs_shap=('abs_shap','mean'),mean_shap=('shap_value','mean'),sd_abs_shap=('abs_shap','std'),n_explanation_samples=('obs_id','nunique'))
 byfold['fold_rank']=byfold.groupby(['task','fold']).mean_abs_shap.rank(ascending=False,method='min').astype(int);byfold=byfold.merge(tax,on='feature',how='left');byfold.to_csv(RUN/'03_tabicl_shap_importance_by_fold.csv',index=False,encoding='utf-8-sig')
 summary=byfold.groupby(['task','feature','source_group','physical_meaning'],as_index=False).agg(mean_abs_shap=('mean_abs_shap','mean'),sd_fold_abs_shap=('mean_abs_shap','std'),mean_rank=('fold_rank','mean'),rank_sd=('fold_rank','std'),top10_count=('fold_rank',lambda s:int((s<=10).sum())),n_folds=('fold','nunique'),n_samples=('n_explanation_samples','sum'))
 totals=summary.groupby('task').mean_abs_shap.transform('sum');summary['task_relative_contribution']=summary.mean_abs_shap/totals;summary['task_rank']=summary.groupby('task').mean_abs_shap.rank(ascending=False,method='min').astype(int);summary.to_csv(RUN/'04_tabicl_shap_importance_summary.csv',index=False,encoding='utf-8-sig')
 stability=summary[['task','feature','source_group','mean_rank','rank_sd','top10_count','n_folds','task_rank']].copy();stability['stability_class']=np.select([(stability.n_folds>=4)&(stability.top10_count>=stability.n_folds-1),(stability.top10_count>=2)],['STABLE','PARTIALLY_STABLE'],'UNSTABLE');stability.to_csv(RUN/'05_tabicl_shap_stability.csv',index=False,encoding='utf-8-sig')
 source=summary.groupby(['task','source_group'],as_index=False).task_relative_contribution.sum();source.to_csv(RUN/'tables/tabicl_shap_source_group_contribution.csv',index=False,encoding='utf-8-sig')
 # Select four process variables per task; suppress pairwise |r|>=0.95 using available formal rows.
 B=base();features=B.load_features();configs=[]
 for task in B.TASKS:
  q=summary[(summary.task==task)&~summary.source_group.isin(['SPATIAL_STATIC','TEMPORAL_PROCESS'])].sort_values(['mean_abs_shap','mean_rank'],ascending=[False,True])
  train,test=B.load_fold(task,'fold_1',features);data=pd.concat([train,test],ignore_index=True);corr=data[features].corr().abs();selected=[]
  for feature in q.feature:
   if all(pd.isna(corr.loc[feature,s]) or corr.loc[feature,s]<.95 for s in selected): selected.append(feature)
   if len(selected)==4:break
  c=pd.DataFrame({'task':task,'feature':selected,'selection_order':range(1,len(selected)+1),'selection_basis':'完整77维TabICLv2 SHAP；排除纯定位变量；|r|<0.95'})
  path=RUN/'config'/f'{task}_ale_features.csv';path.parent.mkdir(parents=True,exist_ok=True);c.to_csv(path,index=False,encoding='utf-8-sig');configs.append(c)
 pd.concat(configs).to_csv(RUN/'tables/ale_candidate_features.csv',index=False,encoding='utf-8-sig')
 print('files',len(files),'samples',raw.obs_id.nunique(),'rows',len(raw));print(summary.sort_values(['task','task_rank']).groupby('task').head(10)[['task','feature','source_group','task_rank','mean_abs_shap','n_folds']].to_string(index=False))

if __name__=='__main__':main()
