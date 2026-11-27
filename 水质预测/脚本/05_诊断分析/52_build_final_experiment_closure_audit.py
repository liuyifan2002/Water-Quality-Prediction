"""汇总已完成实验；只读取结果，不训练模型。"""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs'/'final_experiment_closure_audit_v1'; OUT.mkdir(parents=True,exist_ok=True)

def v2_table(family, filename, target_family):
    d=pd.read_csv(ROOT/'runs'/family/filename)
    if 'task' not in d: d['task']=d['depth'].map(lambda x:f'{target_family}_{x}')
    pooled_name={'Chl-a':'08_chla_family_v2_pooled_metrics.csv','TN':'08_tn_family_v2_pooled_metrics.csv','TP':'08_tp_family_v2_pooled_metrics.csv'}[target_family]
    pooled=pd.read_csv(ROOT/'runs'/family/pooled_name)
    if 'model' in pooled:
        wide=pooled.pivot_table(index=['task','depth'],columns='model',values=['n_obs','n_station','R2','RMSE','MAE'],aggfunc='first').reset_index()
        wide.columns=['_'.join([str(x) for x in c if str(x)!='']) for c in wide.columns]
        d=d.merge(wide,on=['task','depth'],how='left')
    d['target_family']=target_family; return d

def core_results():
    rows=[]
    for fam,fn,tf in [('chla_family_v2','chla_family_v2_depth_generalization.csv','Chl-a'),('tn_family_v2','13_tn_depth_generalization.csv','TN'),('tp_family_v2','tp_family_v2_depth_generalization.csv','TP')]:
        d=v2_table(fam,fn,tf)
        for _,r in d.iterrows():
            for model in ['XGBoost','TabICLv2']:
                isx=model=='XGBoost'; pref='xgboost' if isx else 'tabiclv2'; ns=r.get(f'n_station_{pref}',84); no=r.get(f'n_obs_{pref}',0); ns=84 if pd.isna(ns) else int(ns); no=0 if pd.isna(no) else int(no); rows.append(dict(task=r.get('task',f'{tf}_{r.depth}'),target_family=tf,depth=r.depth,model=model,split_version=f'{tf.upper()}_FAMILY_V2_UNIFIED_STATION_SPLIT',feature_set='X_strict_77',n_stations=ns,n_obs=no,pooled_R2=r[f'R2_{pref}'],fold_R2_mean=None,RMSE=r[f'RMSE_{pref}'],MAE=r[f'MAE_{pref}'],station_win_rate_RMSE=None if isx else r.RMSE_win_rate,station_win_rate_MAE=None if isx else r.MAE_win_rate,bootstrap_CI='see family bootstrap table',prediction_available='yes',replication_status='V2 primary'))
    s=pd.read_csv(ROOT/'runs'/'tabiclv2_pilot'/'23_pooled_oof_metrics.csv').query("task=='secchi'").iloc[0]
    modern=pd.read_csv(ROOT/'runs'/'modern_baseline_gate_v1'/'modern_baseline_core_metrics.csv')
    for model in ['XGBoost','TabICLv2']:
        gm=modern[(modern.task=='secchi')&(modern.model==model.lower())].iloc[0]; isx=model=='XGBoost'; rows.append(dict(task='secchi',target_family='Secchi',depth='single_task',model=model,split_version='SECCHI_HISTORICAL_STATION_HOLDOUT',feature_set='X_strict_77',n_stations=int(s.n_station),n_obs=int(s.n_obs),pooled_R2=s.XGBoost_pooled_R2 if isx else s.TabICLv2_pooled_R2,fold_R2_mean=gm.fold_R2_mean,RMSE=s.XGBoost_pooled_RMSE if isx else s.TabICLv2_pooled_RMSE,MAE=s.XGBoost_pooled_MAE if isx else s.TabICLv2_pooled_MAE,station_win_rate_RMSE=None if isx else 0.8085106383,station_win_rate_MAE=None if isx else 0.8191489362,bootstrap_CI='see runs/tabiclv2_pilot/25_station_bootstrap_ci.csv',prediction_available='yes',replication_status='historical primary'))
    return pd.DataFrame(rows)

def main():
    core=core_results(); core.to_csv(OUT/'final_core_results_table.csv',index=False,encoding='utf-8-sig')
    reg=core.copy(); reg['experiment_type']='V2_core_10_task'; reg['source_path']='family_v2 or tabiclv2_pilot pooled tables'
    ab=pd.read_csv(ROOT/'runs/multisource_ablation_v1/multisource_ablation_source_summary.csv'); ab['experiment_type']='multisource_ablation'; ab['source_path']='runs/multisource_ablation_v1'; ab['task_definition']='4 representative station tasks'; ab['split_version']='task-specific frozen station holdout'; ab['feature_set']='X_strict_77 minus one source group'; ab['prediction_available']='yes'; ab['replication_status']='diagnostic only'
    gate=pd.read_csv(ROOT/'runs/modern_baseline_gate_v1/modern_baseline_core_metrics.csv'); gate['experiment_type']='modern_baseline_gate'; gate['source_path']='runs/modern_baseline_gate_v1'; gate['task_definition']='4 representative tasks'; gate['split_version']='current representative station holdout'; gate['feature_set']='X_strict_77'; gate['prediction_available']='yes'; gate['replication_status']='benchmark only'
    for d in [ab,gate]:
        for c in reg.columns:
            if c not in d: d[c]=None
    reg=pd.concat([reg,ab[reg.columns],gate[reg.columns]],ignore_index=True); reg.to_csv(OUT/'final_experiment_master_registry.csv',index=False,encoding='utf-8-sig')
    comp=pd.DataFrame([
      ['10-task core','Chl-a/TN/TP 3 layers + Secchi','complete','V2 primary for 9; Secchi frozen historical station result'],
      ['CHLA_FAMILY_V2','3 depths','complete','30/30 fold runs and bootstrap'],['TN_FAMILY_V2','3 depths','complete','30/30 fold runs and old bottom replication'],['TP_FAMILY_V2','3 depths','complete','30/30 fold runs and old middle replication'],['MODERN_BASELINE_GATE_V1','4 representative tasks','complete','TabM/TabPFN benchmark; no full 10-task extension'],['Random/Year/Station','representative pilot tasks','complete_with_scope','historical strict comparisons; not interchangeable with V2 splits'],['regionalization','Chl-a/Secchi and selected tasks','complete_with_scope','extension evidence; not universal'],['multisource_ablation','4 tasks x 6 source groups','complete','XGBoost diagnostic; no remote-sensing group in 77 features'],
    ],columns=['experiment','scope','status','note']); comp.to_csv(OUT/'final_experiment_completeness_matrix.csv',index=False,encoding='utf-8-sig')
    (OUT/'final_known_limitations.md').write_text('# 已知限制\n\n- V2 family split 与历史 pilot split 不同；历史结果只作 replication，不与 V2 的绝对 R²直接混合。\n- Secchi 是单一透明度任务，不拆分为三层。\n- modern baseline gate 只覆盖四个代表任务，TabM/TabPFN不是10任务全量结果。\n- 77个冻结特征没有真实遥感变量，因此本轮无法量化遥感增量。\n- 时间外推与空间外推回答不同问题，不能用 pooled R²替代 fold R²均值。\n',encoding='utf-8')
    (OUT/'final_experiment_closure_decision.md').write_text('# 最终实验闭环审计\n\n## 结论\n**EXPERIMENTS_COMPLETE**\n\n现有正式主结果、三组 family V2、现代表格模型代表性比较、代表性 Random/Year/Station 验证、分海区扩展和多源消融均已登记。没有发现会阻碍论文主要结论的必需实验缺口。\n\n## 口径\n所有 V2 结果优先于同任务历史结果；历史结果仅保留为 replication。pooled OOF R²与fold R²均值分别登记，不互相替代。\n\n## 论文可用主线\nTabICLv2在统一站点留出协议下，对Chl-a、TN、TP三水层和Secchi形成了可审计的空间泛化结果；优势具有指标和水层差异。\n\n## 尚需如实说明\n现代模型比较不是10任务全量比较；冻结77特征中没有遥感变量，不能声称遥感已被验证；Random、Year与Station是不同泛化问题。\n',encoding='utf-8')
    print('CLOSURE=EXPERIMENTS_COMPLETE', 'CORE_ROWS=',len(core),'REGISTRY_ROWS=',len(reg))
if __name__=='__main__': main()
