"""Summarize source-group ablation and station bootstrap."""
from __future__ import annotations
from datetime import datetime
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
ROOT=Path(__file__).resolve().parents[2]; RUN=ROOT/'runs'/'multisource_ablation_v1'; TASKS=('chla_surface','secchi','tn_surface','tp_middle'); FOLDS=tuple(f'fold_{i}' for i in range(1,6)); N=1000; SEED=20260921
def full(task):
 if task=='secchi': return pd.read_csv(ROOT/'runs/modern_baseline_gate_v1/predictions/secchi_xgb_frozen_for_gate.csv',dtype={'obs_id':'string','station_id':'string'})
 root={'chla_surface':ROOT/'runs/chla_family_v2','tn_surface':ROOT/'runs/tn_family_v2','tp_middle':ROOT/'runs/tp_family_v2'}[task]; return pd.concat([pd.read_csv(root/'predictions'/f'{task}_xgboost_{f}.csv',dtype={'obs_id':'string','station_id':'string'}) for f in FOLDS],ignore_index=True)
def main():
 fold=pd.read_csv(RUN/'02_ablation_fold_metrics.csv'); tax=pd.read_csv(RUN/'00_feature_source_taxonomy.csv'); groups=sorted(tax.source_group.unique()); pool=[]; boot=[]; summary=[]; rng=np.random.default_rng(SEED)
 for task,group in product(TASKS,groups):
  sub=fold[(fold.task==task)&(fold.source_group_removed==group)]; rows=[]
  for f in FOLDS:
   p=pd.read_csv(RUN/'predictions'/f'{task}_{group}_{f}.csv'); rows.append(p)
  a=pd.concat(rows,ignore_index=True); y=a.y_true.to_numpy(float); fp=a.full_pred.to_numpy(float); ap=a.y_pred.to_numpy(float); fm=(r2_score(y,fp),mean_squared_error(y,fp)**.5,mean_absolute_error(y,fp)); am=(r2_score(y,ap),mean_squared_error(y,ap)**.5,mean_absolute_error(y,ap)); pool.append({'task':task,'source_group':group,'n_features_ablated':int(sub.n_features.iloc[0]),'n_removed_features':int(tax.source_group.eq(group).sum()),'full_R2':fm[0],'ablated_R2':am[0],'delta_R2':am[0]-fm[0],'full_RMSE':fm[1],'ablated_RMSE':am[1],'delta_RMSE':am[1]-fm[1],'full_MAE':fm[2],'ablated_MAE':am[2],'delta_MAE':am[2]-fm[2],'fold_R2_mean':sub.ablated_R2.mean(),'fold_R2_sd':sub.ablated_R2.std(ddof=1)})
  station=[]
  for sid,g in a.groupby('station_id'):
   station.append({'station_id':sid,'delta_RMSE':mean_squared_error(g.y_true,g.y_pred)**.5-mean_squared_error(g.y_true,g.full_pred)**.5,'delta_MAE':mean_absolute_error(g.y_true,g.y_pred)-mean_absolute_error(g.y_true,g.full_pred)})
  sf=pd.DataFrame(station); s_rm=(sf.delta_RMSE>0).mean(); s_ma=(sf.delta_MAE>0).mean(); stations=np.array(sorted(sf.station_id.astype(str))); a_station=a.station_id.astype(str).to_numpy(); station_indices={s:np.flatnonzero(a_station==s) for s in stations}; vals=[]
  for _ in range(N):
   chosen=rng.choice(stations,len(stations),replace=True); ix=np.concatenate([station_indices[s] for s in chosen]); yy=a.y_true.to_numpy(float)[ix]; ff=a.full_pred.to_numpy(float)[ix]; aa=a.y_pred.to_numpy(float)[ix]; vals.append({'delta_RMSE':mean_squared_error(yy,aa)**.5-mean_squared_error(yy,ff)**.5,'delta_MAE':mean_absolute_error(yy,aa)-mean_absolute_error(yy,ff),'delta_R2':r2_score(yy,aa)-r2_score(yy,ff)})
  v=pd.DataFrame(vals)
  for metric in ('delta_R2','delta_RMSE','delta_MAE'): boot.append({'task':task,'source_group':group,'metric':metric,'median':v[metric].median(),'CI_low':v[metric].quantile(.025),'CI_high':v[metric].quantile(.975),'n_bootstrap':N,'seed':SEED})
  r=v[v.columns[0]]; d=pool[-1]; br=boot[-2]; bm=boot[-1]; important=d['delta_RMSE']>0 and d['delta_MAE']>0 and br['CI_low']>0 and bm['CI_low']>0 and s_rm>.6 and s_ma>.6; supportive=(d['delta_RMSE']>0 or d['delta_MAE']>0) and (s_rm>=.5 or s_ma>=.5); cls='IMPORTANT' if important else 'SUPPORTIVE' if supportive else 'NEUTRAL' if d['delta_RMSE']<=0 and d['delta_MAE']<=0 else 'POSSIBLY_REDUNDANT'; summary.append({'task':task,'source_group':group,'n_stations':len(sf),'station_win_rate_RMSE':s_rm,'station_win_rate_MAE':s_ma,'pooled_delta_R2':d['delta_R2'],'pooled_delta_RMSE':d['delta_RMSE'],'pooled_delta_MAE':d['delta_MAE'],'bootstrap_RMSE_CI_low':br['CI_low'],'bootstrap_RMSE_CI_high':br['CI_high'],'bootstrap_MAE_CI_low':bm['CI_low'],'bootstrap_MAE_CI_high':bm['CI_high'],'classification':cls})
 pd.DataFrame(pool).to_csv(RUN/'multisource_ablation_metrics.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(boot).to_csv(RUN/'multisource_ablation_bootstrap.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(summary).to_csv(RUN/'multisource_ablation_source_summary.csv',index=False,encoding='utf-8-sig')
 lines=['# MULTISOURCE_ABLATION_V1 决策报告','',f'- 完成时间：{datetime.now().isoformat(timespec="seconds")}','- 模型：XGBoost','- 任务：Chl-a surface、Secchi、TN surface、TP middle','- 评价：统一 station holdout，来源级 leave-one-source-out，1000 次站点 bootstrap','', '## 来源分组', tax.groupby('source_group').size().to_string(), '', '## 结果', pd.DataFrame(summary).to_string(index=False), '', '## 遥感结论', '当前冻结的 77 个 strict features 中没有真实遥感波段、Ocean Colour 产品或遥感 availability flag，因此本轮不能运行独立 NO_REMOTE_SENSING 消融。不能把 ERA5、SST 或夜间灯光的贡献解释为遥感贡献。', '', '## 总体判断', '本轮仅完成来源级 XGBoost 消融，不开展 TabICLv2 消融、SHAP、ALE 或新模型实验。']
 (RUN/'multisource_ablation_decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); (RUN/'remote_sensing_contribution_decision.md').write_text('# 遥感贡献判断\n\n当前 77 个 strict features 中遥感来源组为 0 个，无法进行有效的遥感删除消融。后续若要回答遥感是否提供增量贡献，必须先把经过质量筛选的遥感特征正式纳入 strict feature set，再设计独立对照。\n',encoding='utf-8'); print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__': main()
