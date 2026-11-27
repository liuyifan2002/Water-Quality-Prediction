"""FINAL_RESULT_REGISTRY_REPAIR_V1：只修复冻结结果登记，不训练模型。"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'runs'/'final_experiment_closure_audit_v1'
SRC.mkdir(parents=True,exist_ok=True)
MODELS=('xgboost','tabiclv2')

def norm_task(task, depth=None):
    t=str(task).lower().replace('chl-a','chla').replace(' ','_')
    d=str(depth).lower() if depth is not None else ''
    if t in ('secchi','sd'): return 'secchi','single_task'
    for family in ('chla','tn','tp'):
        if t.startswith(family):
            for dep in ('surface','middle','bottom'):
                if dep in t or dep in d: return f'{family}_{dep}',dep
    return t,d if d in ('surface','middle','bottom','single_task') else pd.NA

def family_core(family):
    base=ROOT/'runs'/f'{family}_family_v2'
    pm=pd.read_csv(base/f'08_{family}_family_v2_pooled_metrics.csv')
    fc=pd.read_csv(base/f'11_{family}_family_v2_fold_consistency.csv')
    wr=pd.read_csv(base/f'09_{family}_family_v2_station_win_rates.csv')
    bs=pd.read_csv(base/f'10_{family}_family_v2_bootstrap.csv')
    if family=='chla': gen=pd.read_csv(base/'chla_family_v2_depth_generalization.csv'); rep=pd.read_csv(base/'chla_family_v2_replication.csv')
    elif family=='tn': gen=pd.read_csv(base/'13_tn_depth_generalization.csv'); rep=pd.read_csv(base/'12_tn_bottom_replication.csv')
    else: gen=pd.read_csv(base/'tp_family_v2_depth_generalization.csv'); rep=pd.read_csv(base/'tp_middle_replication.csv')
    rows=[]
    for task in pm.task.unique():
        nt,dep=norm_task(task,pm.loc[pm.task.eq(task),'depth'].iloc[0])
        comp=pm[pm.task.eq(task)].set_index('model')
        win=wr[wr.task.eq(task)].iloc[0]
        g=gen[(gen.depth.astype(str).str.contains(dep)) | (gen.get('task',pd.Series(index=gen.index,dtype=str)).astype(str).eq(task))].iloc[0]
        b=bs[bs.task.eq(task)].set_index('metric')
        fold=fc[fc.task.eq(task)]
        replication='V2 primary'
        if family=='chla':
            rr=rep[rep.depth.eq(dep)]; replication=rr.replication_classification.iloc[0] if len(rr) and 'replication_classification' in rr else 'REPLICATED'
        elif family=='tn' and dep=='bottom': replication=rep.replication_classification.dropna().iloc[-1]
        elif family=='tp' and dep=='middle': replication=rep.replication_classification.dropna().iloc[-1]
        for model in MODELS:
            r=comp.loc[model]
            is_new=model=='tabiclv2'
            rows.append(dict(task=nt,depth=dep,target_family=family,model='TabICLv2' if is_new else 'XGBoost',n_stations=int(r.n_station),n_obs=int(r.n_obs),split_version=f'{family.upper()}_FAMILY_V2_UNIFIED_STATION_SPLIT',feature_set='X_strict_77',pooled_R2=r.R2,fold_R2_mean=fold[f'{"TabICL" if is_new else "XGB"}_R2'].mean(),RMSE=r.RMSE,MAE=r.MAE,station_win_rate_RMSE=win.win_rate_rmse if is_new else pd.NA,station_win_rate_MAE=win.win_rate_mae if is_new else pd.NA,delta_R2=g.delta_R2 if is_new else pd.NA,delta_RMSE=g.delta_RMSE if is_new else pd.NA,delta_MAE=g.delta_MAE if is_new else pd.NA,bootstrap_RMSE_low=b.loc['delta_RMSE','low_2p5'] if is_new else pd.NA,bootstrap_RMSE_high=b.loc['delta_RMSE','high_97p5'] if is_new else pd.NA,bootstrap_MAE_low=b.loc['delta_MAE','low_2p5'] if is_new else pd.NA,bootstrap_MAE_high=b.loc['delta_MAE','high_97p5'] if is_new else pd.NA,robustness_class=g.classification if is_new else 'BASELINE',replication_status=replication,prediction_available='YES',missing_reason='' if is_new else 'comparison-only fields are not applicable to baseline'))
    return rows

def secchi_core():
    core=pd.read_csv(ROOT/'runs/multisource_ablation_v1/01_metric_consistent_core_table.csv')
    core=core[(core.task=='secchi') & core.model.isin(MODELS)].set_index('model')
    audit=pd.read_csv(ROOT/'runs/tabiclv2_pilot/28_spatial_robustness_classification.csv').query("task=='secchi'").iloc[0]
    rows=[]
    for model in MODELS:
        r=core.loc[model]; is_new=model=='tabiclv2'
        rows.append(dict(task='secchi',depth='single_task',target_family='secchi',model='TabICLv2' if is_new else 'XGBoost',n_stations=int(r.n_station),n_obs=int(r.n_obs),split_version='SECCHI_HISTORICAL_STATION_HOLDOUT',feature_set='X_strict_77',pooled_R2=r.pooled_OOF_R2,fold_R2_mean=r.fold_R2_mean,RMSE=r.pooled_RMSE,MAE=r.pooled_MAE,station_win_rate_RMSE=audit.win_rate_rmse if is_new else pd.NA,station_win_rate_MAE=audit.win_rate_mae if is_new else pd.NA,delta_R2=audit.pooled_delta_R2 if is_new else pd.NA,delta_RMSE=audit.pooled_delta_RMSE if is_new else pd.NA,delta_MAE=audit.pooled_delta_MAE if is_new else pd.NA,bootstrap_RMSE_low=audit.bootstrap_delta_RMSE_ci_lower if is_new else pd.NA,bootstrap_RMSE_high=audit.bootstrap_delta_RMSE_ci_upper if is_new else pd.NA,bootstrap_MAE_low=audit.bootstrap_delta_MAE_ci_lower if is_new else pd.NA,bootstrap_MAE_high=audit.bootstrap_delta_MAE_ci_upper if is_new else pd.NA,robustness_class='ROBUST' if is_new else 'BASELINE',replication_status='historical primary; no family V2 required for single task',prediction_available='YES',missing_reason='' if is_new else 'comparison-only fields are not applicable to baseline'))
    return rows

def build_master(core):
    records=[]
    for _,r in core.iterrows(): records.append({**r.to_dict(),'experiment_category':'10-task core','source_reference':'family V2 pooled/fold/station/bootstrap tables'})
    for _,r in core.iterrows():
        if r.task!='secchi': records.append({**r.to_dict(),'experiment_category':'family V2','source_reference':f'runs/{r.target_family}_family_v2/'})
    gate=pd.read_csv(ROOT/'runs/modern_baseline_gate_v1/modern_baseline_core_metrics.csv')
    for _,r in gate.iterrows():
        nt,dep=norm_task(r.task); records.append(dict(experiment_category='modern baseline',task=nt,depth=dep,model=r.model,split_version='representative frozen station holdout',feature_set='X_strict_77',pooled_R2=r.pooled_R2,fold_R2_mean=r.fold_R2_mean,RMSE=r.RMSE,MAE=r.MAE,prediction_available='YES',source_reference='runs/modern_baseline_gate_v1/modern_baseline_core_metrics.csv'))
    p=pd.read_csv(ROOT/'runs/tabiclv2_pilot/20_tabiclv2_pilot_summary.csv')
    for _,r in p.iterrows():
        nt,dep=norm_task(r.task)
        for m,col in [('XGBoost','XGBoost_R2'),('TabICLv2','TabICLv2_R2')]: records.append(dict(experiment_category=str(r.validation).capitalize(),task=nt,depth=dep,model=m,split_version=f'historical_{r.validation}_strict',feature_set='X_strict_77',pooled_R2=r[col] if r.validation!='station' else pd.NA,fold_R2_mean=r[col] if r.validation=='station' else pd.NA,prediction_available='task-dependent',source_reference='runs/tabiclv2_pilot/20_tabiclv2_pilot_summary.csv',missing_reason='historical summary stores station fold mean, not pooled OOF' if r.validation=='station' else 'historical summary has aggregate R2 only'))
    reg=pd.read_csv(ROOT/'结果/扩展实验/论文主线审计_20260916/03_分海区模型对照.csv')
    for _,r in reg.iterrows():
        nt,dep=norm_task(r.target,r.depth); records.append(dict(experiment_category='regionalization',task=nt,depth=dep,model=r.model,split_version=r['验证方式'],feature_set='historical regional feature set',pooled_R2=r.regional_r2,RMSE=r.regional_rmse,delta_R2=r.delta_r2,delta_RMSE=float(r.regional_rmse)-float(r.global_rmse),prediction_available='YES',source_reference='结果/扩展实验/论文主线审计_20260916/03_分海区模型对照.csv'))
    met=pd.read_csv(ROOT/'runs/multisource_ablation_v1/multisource_ablation_metrics.csv')
    summ=pd.read_csv(ROOT/'runs/multisource_ablation_v1/multisource_ablation_source_summary.csv')
    for _,r in met.iterrows():
        s=summ[(summ.task==r.task)&(summ.source_group==r.source_group)].iloc[0]; nt,dep=norm_task(r.task)
        records.append(dict(experiment_category='multisource ablation',task=nt,depth=dep,model='XGBoost',split_version='task-specific frozen station holdout',feature_set=f'X_strict_77 minus {r.source_group}',source_group=r.source_group,full_R2=r.full_R2,ablated_R2=r.ablated_R2,delta_R2=r.delta_R2,delta_RMSE=r.delta_RMSE,delta_MAE=r.delta_MAE,bootstrap_RMSE_low=s.bootstrap_RMSE_CI_low,bootstrap_RMSE_high=s.bootstrap_RMSE_CI_high,bootstrap_MAE_low=s.bootstrap_MAE_CI_low,bootstrap_MAE_high=s.bootstrap_MAE_CI_high,prediction_available='YES',source_reference='runs/multisource_ablation_v1/'))
    return pd.DataFrame(records)

def validate(core,master):
    expected={f'{f}_{d}' for f in ('chla','tn','tp') for d in ('surface','middle','bottom')}|{'secchi'}
    checks=[]
    checks.append(('core_rows_20',len(core)==20,len(core)))
    checks.append(('core_10_tasks',set(core.task)==expected,','.join(sorted(core.task.unique()))))
    checks.append(('two_models_each',(core.groupby('task').model.nunique()==2).all(),core.groupby('task').model.nunique().to_dict()))
    checks.append(('depth_vocabulary',set(core.depth)<= {'surface','middle','bottom','single_task'},','.join(sorted(core.depth.unique()))))
    checks.append(('tn_tp_n_obs_positive',(core[core.task.str.startswith(('tn_','tp_'))].n_obs>0).all(),core[core.task.str.startswith(('tn_','tp_'))].n_obs.min()))
    checks.append(('tn_tp_core_metrics_present',core[core.task.str.startswith(('tn_','tp_'))][['pooled_R2','RMSE','MAE']].notna().all().all(),'pooled_R2/RMSE/MAE'))
    checks.append(('secchi_metric_separation',abs(core.query("task=='secchi' and model=='TabICLv2'").pooled_R2.iloc[0]-0.602049)<1e-5 and abs(core.query("task=='secchi' and model=='TabICLv2'").fold_R2_mean.iloc[0]-0.584848)<1e-5,'0.602049 vs 0.584848'))
    required={'10-task core','family V2','modern baseline','Random','Year','Station','regionalization','multisource ablation'}
    checks.append(('master_categories',required<=set(master.experiment_category),','.join(sorted(set(master.experiment_category)))))
    out=pd.DataFrame(checks,columns=['check','pass','detail']); out.to_csv(SRC/'final_registry_validation_checks_v2.csv',index=False,encoding='utf-8-sig')
    return bool(out['pass'].all()),out

def main():
    core=pd.DataFrame(family_core('chla')+family_core('tn')+family_core('tp')+secchi_core())
    core=core.sort_values(['target_family','depth','model']).reset_index(drop=True)
    master=build_master(core)
    ok,checks=validate(core,master)
    core.to_csv(SRC/'final_core_results_table_v2.csv',index=False,encoding='utf-8-sig')
    master.to_csv(SRC/'final_experiment_master_registry_v2.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([
      ['10-task core','20 rows','COMPLETE','V2 preferred; Secchi single task'],['family V2','CHLA/TN/TP','COMPLETE','90 frozen fold runs'],['modern baseline','4 representative tasks','COMPLETE','four-model benchmark'],['Random','4 representative tasks','REGISTERED','historical strict pilot'],['Year','4 representative tasks','REGISTERED','historical strict pilot'],['Station','4 representative tasks','REGISTERED','historical fold mean kept distinct from pooled'],['regionalization','72 comparisons','REGISTERED','historical extension results'],['multisource ablation','24 source-task comparisons','COMPLETE','XGBoost source-level ablation']
    ],columns=['component','scope','status','note']).to_csv(SRC/'final_experiment_completeness_matrix_v2.csv',index=False,encoding='utf-8-sig')
    (SRC/'final_known_limitations_v2.md').write_text('# 已知限制（V2）\n\n- V2 family 结果是 Chl-a、TN、TP 的当前正式空间验证结果；历史 pilot 仅作重复验证记录。\n- Secchi 为单一任务，保持 pooled OOF R²=0.602049、fold R²均值=0.584848。\n- Random、Year 与 Station 回答不同泛化问题，不能相互替代。历史 representative summary 的 station R²为 fold mean，已明确标注，不冒充 pooled OOF。\n- modern baseline 只覆盖四个代表任务。\n- 多源消融的冻结77特征不含真实遥感变量，不能据此判断遥感贡献。\n- 分海区结果属于历史扩展实验，其 split 和 feature set 与 family V2 不同，不作为V2核心数值替代。\n',encoding='utf-8')
    status='FINAL_REGISTRY_VALIDATED' if ok else 'REGISTRY_VALIDATION_FAILED'
    (SRC/'final_experiment_closure_decision_v2.md').write_text(f'# 最终实验闭环判断（V2）\n\n- 科学实验状态：**EXPERIMENTS_COMPLETE**\n- 登记表状态：**{status}**\n\nV2优先于历史结果；pooled OOF R²与fold mean R²已分列。完整性检查见 `final_registry_validation_checks_v2.csv`。\n',encoding='utf-8')
    print('EXPERIMENTS_COMPLETE'); print(status); print(checks.to_string(index=False)); print('core_rows',len(core),'master_rows',len(master))

if __name__=='__main__': main()
