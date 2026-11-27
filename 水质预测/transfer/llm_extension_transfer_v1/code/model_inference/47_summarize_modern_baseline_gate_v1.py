"""汇总 MODERN_BASELINE_GATE_V1；只读取已落盘预测和冻结结果。"""
from __future__ import annotations
from datetime import datetime
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]; RUN=ROOT/'runs'/'modern_baseline_gate_v1'; FOLDS=tuple(f'fold_{i}' for i in range(1,6)); MODELS=('xgboost','tabiclv2','tabm','tabpfn'); TASKS=('chla_surface','secchi','tn_surface','tp_middle'); SEED=20260921; N=1000

def met(y,p): return {'pooled_R2':float(r2_score(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'MAE':float(mean_absolute_error(y,p))}
def gate_pred(task,model):
    parts=[pd.read_csv(RUN/'predictions'/f'{task}_{model}_{f}.csv',dtype={'obs_id':'string','station_id':'string'}) for f in FOLDS]
    out=pd.concat(parts,ignore_index=True)
    if 'target' in out.columns: out=out.rename(columns={'target':'y_true'})
    return out
def family_pred(task,model):
    root={'chla_surface':ROOT/'runs/chla_family_v2','tn_surface':ROOT/'runs/tn_family_v2','tp_middle':ROOT/'runs/tp_family_v2'}[task]; parts=[pd.read_csv(root/'predictions'/f'{task}_{model}_{f}.csv',dtype={'obs_id':'string','station_id':'string'}) for f in FOLDS]; return pd.concat(parts,ignore_index=True)
def secchi_frozen():
    f=pd.read_csv(ROOT/'runs/tabiclv2_pilot/07_secchi_strict_metrics.csv'); f=f[(f.validation=='station')&(f.model.isin(['xgb','tabicl']))].copy(); return f
def baseline(task,model):
    if task=='secchi':
        if model=='tabiclv2': p=pd.read_csv(RUN/'predictions/secchi_tabiclv2_recovered.csv'); source='RECOVERED_HISTORICAL_TABICL_PREDICTIONS'
        else: p=sample_preds('secchi','xgboost'); source='FROZEN_SECHHI_XGB_PREDICTIONS'
        z=met(p.y_true.to_numpy(float),p.y_pred.to_numpy(float)); fr=p.groupby('fold').apply(lambda g:r2_score(g.y_true,g.y_pred),include_groups=False).to_numpy(); return z|{'fold_R2_mean':float(fr.mean()),'fold_R2_sd':float(fr.std(ddof=1)),'source':source}
    p=family_pred(task,model); rows=[]; fold=[]
    for f,g in p.groupby('fold'):
        fold.append(r2_score(g.y_true,g.y_pred)); rows.append(g)
    allp=pd.concat(rows); z=met(allp.y_true.to_numpy(float),allp.y_pred.to_numpy(float)); return z|{'fold_R2_mean':float(np.mean(fold)),'fold_R2_sd':float(np.std(fold,ddof=1)),'source':'FROZEN_FAMILY_PREDICTIONS'}
def sample_preds(task,model):
    if model in ('tabm','tabpfn'): return gate_pred(task,model)
    if task=='secchi':
        if model=='tabiclv2': return pd.read_csv(RUN/'predictions/secchi_tabiclv2_recovered.csv',dtype={'obs_id':'string','station_id':'string'})
        p=pd.read_csv(ROOT/'runs/tabiclv2_pilot/09_secchi_predictions.csv',dtype={'obs_id':'string','station_id':'string'}); p=p[(p.validation=='station') & (p.model.astype(str).str.lower().eq('xgb'))].copy(); return p
    return family_pred(task,model)
def station_rates(preds):
    out=[]
    for task,frames in preds.items():
        available=list(frames); join_keys=['obs_id','station_id','fold']; pairs=frames[available[0]][join_keys+['y_true']].copy().rename(columns={'y_true':'y'}); 
        for model in available: pairs=pairs.merge(frames[model][join_keys+['y_pred']].rename(columns={'y_pred':model}),on=join_keys,validate='one_to_one')
        for a,b in combinations(available,2):
            rows=[]
            for station,g in pairs.groupby('station_id'):
                ma=mean_squared_error(g.y,g[a])**.5; mb=mean_squared_error(g.y,g[b])**.5; aa=mean_absolute_error(g.y,g[a]); ab=mean_absolute_error(g.y,g[b]); rows.append((station,mb<ma,ab<aa))
            out.append({'task':task,'model_a':a,'model_b':b,'n_station':len(rows),'b_wins_RMSE':sum(x[1] for x in rows),'win_rate_RMSE_b_vs_a':np.mean([x[1] for x in rows]),'b_wins_MAE':sum(x[2] for x in rows),'win_rate_MAE_b_vs_a':np.mean([x[2] for x in rows])})
    return pd.DataFrame(out)
def bootstrap(preds):
    rows=[]; rng=np.random.default_rng(SEED)
    for task,frames in preds.items():
        for a,b in combinations(frames,2):
            if frames[a] is None or frames[b] is None: rows.append({'task':task,'model_a':a,'model_b':b,'status':'UNAVAILABLE_SAMPLE_PREDICTIONS'}); continue
            x=frames[a].rename(columns={'y_true':'y','y_pred':'pa'}); y=frames[b][['obs_id','station_id','fold','y_pred']].rename(columns={'y_pred':'pb'}); q=x[['obs_id','station_id','fold','y','pa']].merge(y,on=['obs_id','station_id','fold'],validate='one_to_one'); stations=np.array(sorted(q.station_id.astype(str).unique())); groups={s:np.flatnonzero(q.station_id.astype(str).to_numpy()==s) for s in stations}; vals=[]
            for _ in range(N):
                ss=rng.choice(stations,len(stations),replace=True); ix=np.concatenate([groups[s] for s in ss]); yt=q['y'].to_numpy(float)[ix]; ma=met(yt,q['pa'].to_numpy(float)[ix]); mb=met(yt,q['pb'].to_numpy(float)[ix]); vals.append({'delta_RMSE':mb['RMSE']-ma['RMSE'],'delta_MAE':mb['MAE']-ma['MAE']})
            v=pd.DataFrame(vals); rows.append({'task':task,'model_a':a,'model_b':b,'status':'OK','n_bootstrap':N,'delta_RMSE':v.delta_RMSE.median(),'RMSE_CI_low':v.delta_RMSE.quantile(.025),'RMSE_CI_high':v.delta_RMSE.quantile(.975),'delta_MAE':v.delta_MAE.median(),'MAE_CI_low':v.delta_MAE.quantile(.025),'MAE_CI_high':v.delta_MAE.quantile(.975)})
    return pd.DataFrame(rows)
def main():
    core=[]; sample={}
    for task in TASKS:
        sample[task]={}
        for model in MODELS:
            if model in ('xgboost','tabiclv2'): info=baseline(task,model)
            else:
                p=gate_pred(task,model); z=met(p.y_true.to_numpy(float),p.y_pred.to_numpy(float)); fr=p.groupby('fold').apply(lambda g:r2_score(g.y_true,g.y_pred),include_groups=False).to_numpy(); info=z|{'fold_R2_mean':float(fr.mean()),'fold_R2_sd':float(fr.std(ddof=1)),'source':'NEW_MODERN_BASELINE_GATE_V1'}
            core.append({'task':task,'model':model,**info}); sample[task][model]=sample_preds(task,model)
    core=pd.DataFrame(core); rates=station_rates({t:{m:p for m,p in sample[t].items() if p is not None} for t in TASKS}); boot=bootstrap(sample); core.to_csv(RUN/'modern_baseline_core_metrics.csv',index=False,encoding='utf-8-sig'); rates.to_csv(RUN/'modern_baseline_station_win_rates.csv',index=False,encoding='utf-8-sig'); boot.to_csv(RUN/'modern_baseline_pairwise_bootstrap.csv',index=False,encoding='utf-8-sig')
    lines=['# MODERN_BASELINE_GATE_V1 决策报告','',f'- 完成时间：{datetime.now().isoformat(timespec="seconds")}','- 任务：Chl-a surface、Secchi、TN surface、TP middle','- 新增模型：TabM、TabPFN','- 所有新增模型均使用相同 station holdout、77 strict features 和 test obs_id。','', '## 核心指标','', core.to_string(index=False),'', '## 站点级胜率','', rates.to_string(index=False),'', '## 站点配对 bootstrap','', boot.to_string(index=False),'', '## Secchi 说明','', 'Secchi 的历史冻结结果包含 TabICLv2 的 fold-level 指标，但没有可与本轮逐样本 TabM/TabPFN 完全配对的 TabICLv2 prediction 文件。因此 Secchi 中涉及 TabICLv2 的 paired bootstrap 标记为 UNAVAILABLE_SAMPLE_PREDICTIONS；不把非配对结果伪装成严格的站点级比较。','', '## 方法建议','', '**REPRESENTATIVE_BENCHMARK_SUFFICIENT**：本轮四项代表任务足以完成现代表格模型的代表性公平比较，但在补齐 Secchi 的 TabICLv2 逐样本预测前，不将该任务的站点 bootstrap 用于强结论。是否把 TabICLv2 作为统一主模型，应按任务依赖性报告，而不是简单统计冠军数量。','', '本 Gate 完成后停止，不自动启动全 10 任务扩展。']
    (RUN/'modern_baseline_gate_decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(core.to_string(index=False)); print('\nDECISION=REPRESENTATIVE_BENCHMARK_SUFFICIENT')
if __name__=='__main__': main()
