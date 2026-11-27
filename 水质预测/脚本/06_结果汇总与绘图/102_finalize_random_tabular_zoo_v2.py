"""Compile locked inner selection, external test predictions, and family summaries."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/random_tabular_model_zoo_v2'
V1=ROOT/'runs/random_10task_v1'
TASKS=['chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']
FAMILY={'xgboost':'traditional','catboost':'traditional','lightgbm':'traditional','random_forest':'traditional','extra_trees':'traditional','hist_gb':'traditional','svr_rbf':'traditional','ridge':'traditional','mlp':'deep','tabm':'deep','tabicl':'foundation','tabpfn':'foundation','tabdpt':'foundation','tabfm':'foundation','mitra_v2':'foundation'}

def metrics(y,p):return {'R2':float(r2_score(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'MAE':float(mean_absolute_error(y,p))}

def load_pred(task,model):
    path=OUT/'07_final_predictions'/f'{task}_{model}.csv'
    if not path.exists():raise FileNotFoundError(path)
    d=pd.read_csv(path,dtype={'obs_id':str})
    if d.obs_id.duplicated().any():raise ValueError(f'{task} {model} duplicate test obs_id')
    return d

def main():
    plans=json.loads((OUT/'06_final_selection_plan.json').read_text(encoding='utf-8'))
    all_rows=[];task_rows=[];family_rows=[]
    inner=pd.read_csv(OUT/'03_inner_cv_metrics.csv')
    for task in TASKS:
        plan=next(p for p in plans if p['task']==task)
        available=[]
        for path in (OUT/'07_final_predictions').glob(f'{task}_*.json'):
            model=path.stem[len(task)+1:]
            pred_path=path.with_suffix('.csv')
            if not pred_path.exists():continue
            meta=json.loads(path.read_text(encoding='utf-8'))
            d=load_pred(task,model)
            # float32 neural predictions can change R2 by ~1e-9 on CSV round-trip.
            if abs(metrics(d.y_true,d.y_pred)['R2']-meta['R2'])>1e-6:raise ValueError(f'{task} {model} metric mismatch')
            available.append((model,d,meta))
            all_rows.append({'task':task,'model':model,'family':FAMILY.get(model.removesuffix('_tuned'),'unknown'),**meta})
        if not available:raise ValueError(f'{task} has no final predictions')
        base=available[0][1][['obs_id','y_true','station_id']].copy()
        for model,d,meta in available:
            if set(d.obs_id)!=set(base.obs_id):raise ValueError(f'{task} {model} external test IDs differ')
            if not np.allclose(d.set_index('obs_id').loc[base.obs_id,'y_true'],base.y_true):raise ValueError('target mismatch')
        for model in plan['selected_models']:
            if model not in [z[0] for z in available]:raise ValueError(f'{task} selected {model} missing final prediction')
        matrix=np.column_stack([load_pred(task,m).set_index('obs_id').loc[base.obs_id,'y_pred'].to_numpy() for m in plan['selected_models']])
        final=matrix@np.asarray(plan['weights'])
        chosen=plan['selected_kind']
        final_name='selected_'+chosen
        final_path=OUT/'07_final_predictions'/f'{task}_{final_name}.csv'
        pd.DataFrame({'task':task,'model':final_name,'obs_id':base.obs_id,'station_id':base.station_id,'y_true':base.y_true,'y_pred':final,'absolute_error':abs(final-base.y_true),'squared_error':(final-base.y_true)**2}).to_csv(final_path,index=False,encoding='utf-8-sig')
        fm=metrics(base.y_true,final)
        all_rows.append({'task':task,'model':final_name,'family':'selected',
                         'n_train':np.nan,'n_test':len(base),**fm,
                         'source':'locked inner OOF selection and nonnegative weights'})
        single=plan['inner_oof_top3'][0]
        single_meta=next(z[2] for z in available if z[0]==single)
        v1=pd.read_csv(V1/'05_random_10task_summary.csv').set_index('task').loc[task]
        paired_xgb=load_pred(task,'xgboost') if (OUT/'07_final_predictions'/f'{task}_xgboost.csv').exists() else None
        if paired_xgb is not None:
            px=paired_xgb.set_index('obs_id').loc[base.obs_id,'y_pred'].to_numpy()
            ex=np.abs(px-base.y_true.to_numpy());ef=np.abs(final-base.y_true.to_numpy())
            paired={'delta_RMSE_vs_xgboost':fm['RMSE']-metrics(base.y_true,px)['RMSE'],
                    'delta_MAE_vs_xgboost':float(ef.mean()-ex.mean()),
                    'paired_sample_absolute_error_win_rate':float((ef<ex).mean())}
        else:paired={'delta_RMSE_vs_xgboost':np.nan,'delta_MAE_vs_xgboost':np.nan,
                     'paired_sample_absolute_error_win_rate':np.nan}
        task_rows.append({'task':task,'selected_kind':chosen,'selected_models':';'.join(plan['selected_models']),'weights':';'.join(f'{w:.6f}' for w in plan['weights']),'inner_meta_cv_R2':plan['meta_cv'][chosen]['R2'],'inner_best_single_model':single,'inner_best_single_test_R2':single_meta['R2'],'final_test_R2':fm['R2'],'final_test_RMSE':fm['RMSE'],'final_test_MAE':fm['MAE'],'test_delta_R2_vs_inner_best_single':fm['R2']-single_meta['R2'],'v1_xgb_test_R2':v1.R2_xgb,'v1_tabicl_test_R2':v1.R2_tabicl,'delta_R2_vs_v1_xgb':fm['R2']-v1.R2_xgb,'delta_R2_vs_v1_tabicl':fm['R2']-v1.R2_tabicl,**paired})
        for family in ['traditional','deep','foundation']:
            subset=inner.loc[inner.task.eq(task)&inner.family.eq(family)].sort_values('R2',ascending=False)
            if subset.empty:family_rows.append({'task':task,'family':family,'status':'BLOCKED'});continue
            model=subset.iloc[0]['model'];sample=next((z for z in available if z[0]==model),None)
            family_rows.append({'task':task,'family':family,'status':'COMPLETE' if sample else 'INNER_ONLY','inner_best_model':model,'inner_oof_R2':subset.iloc[0]['R2'],'test_R2':sample[2]['R2'] if sample else np.nan})
        print(task,chosen,round(fm['R2'],4),flush=True)
    pd.DataFrame(all_rows).to_csv(OUT/'08_all_models_test_comparison.csv',index=False,encoding='utf-8-sig')
    summary=pd.DataFrame(task_rows);summary.to_csv(OUT/'09_task_best_model_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(family_rows).to_csv(OUT/'10_model_family_comparison.csv',index=False,encoding='utf-8-sig')
    station=pd.read_csv(V1/'07_station_v2_comparison.csv')
    ref=summary.merge(station.loc[station.model.eq('tabicl'),['task','split_version','n_obs','pooled_R2']],on='task',how='left')
    ref['note']='Random seed42 single split vs Station V2 pooled OOF; distinct evaluation targets and sample scopes'
    ref.to_csv(OUT/'11_station_v2_reference_comparison.csv',index=False,encoding='utf-8-sig')
    runtime=[];tuned=[]
    required=['xgboost','catboost','lightgbm','random_forest','extra_trees','hist_gb','svr_rbf','ridge',
              'mlp','tabm','tabicl','tabpfn','tabdpt','tabfm','mitra_v2']
    for task in TASKS:
        for model in required:
            path=OUT/'03_inner_metrics'/f'{task}_{model}.json'
            if path.exists():
                m=json.loads(path.read_text(encoding='utf-8'))
                runtime.append({'task':task,'model':model,'status':'COMPLETE',
                                'inner_oof_R2':m['inner_oof_R2'],
                                'inner_runtime_seconds':m.get('runtime_seconds',m.get('fit_seconds',0)+m.get('infer_seconds',0)),
                                'inner_fit_seconds':m.get('fit_seconds',np.nan),
                                'inner_infer_seconds':m.get('infer_seconds',np.nan),
                                'gpu_peak_allocated_mb':max((f.get('gpu_peak_allocated_mb') or 0 for f in m.get('folds',[])),default=0),
                                'reason':''})
            else:
                reasons={'tabdpt':'官方权重与单折完整样本试跑通过；约259秒/折，十任务五折规模运行本轮未完成',
                         'tabfm':'官方实现支持回归，但所需新版PyTorch与现有冻结环境不兼容；独立整规模运行未完成',
                         'mitra_v2':'官方回归权重和专用微调接口已核实；独立整规模运行未完成'}
                reason=reasons.get(model,'内部五折未完成')
                runtime.append({'task':task,'model':model,'status':'BLOCKED','reason':reason})
        for path in (OUT/'03_inner_metrics').glob(f'{task}_*_tuned.json'):
            m=json.loads(path.read_text(encoding='utf-8'))
            base_inner=inner.loc[inner.task.eq(task)&inner.model.eq(m['base_model'])].iloc[0]
            base_test=next((z[2] for z in available if z[0]==m['base_model']),None)
            tuned_test=next((z[2] for z in available if z[0]==m['model']),None)
            tuned.append({'task':task,'model':m['model'],'selected_candidate':m['chosen_candidate'],
                          'selected_parameters':json.dumps(m['chosen_params'],ensure_ascii=False),
                          'base_inner_oof_R2':base_inner.R2,'base_inner_oof_RMSE':base_inner.RMSE,'base_inner_oof_MAE':base_inner.MAE,
                          'inner_oof_R2':m['inner_oof_R2'],'inner_oof_RMSE':m['inner_oof_RMSE'],
                          'inner_oof_MAE':m['inner_oof_MAE'],
                          'inner_gain_R2_vs_base':m['inner_oof_R2']-base_inner.R2,
                          'inner_delta_RMSE_vs_base':m['inner_oof_RMSE']-base_inner.RMSE,
                          'inner_delta_MAE_vs_base':m['inner_oof_MAE']-base_inner.MAE,
                          'base_test_R2':base_test['R2'] if base_test else np.nan,
                          'base_test_RMSE':base_test['RMSE'] if base_test else np.nan,
                          'base_test_MAE':base_test['MAE'] if base_test else np.nan,
                          'tuned_test_R2':tuned_test['R2'] if tuned_test else np.nan,
                          'tuned_test_RMSE':tuned_test['RMSE'] if tuned_test else np.nan,
                          'tuned_test_MAE':tuned_test['MAE'] if tuned_test else np.nan,
                          'search_fit_seconds':m['fit_seconds'],'search_infer_seconds':m['infer_seconds']})
    pd.DataFrame(runtime).to_csv(OUT/'01_model_runtime_status.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(tuned).to_csv(OUT/'05_tuned_model_metrics.csv',index=False,encoding='utf-8-sig')
    mirror=OUT/'04_inner_oof_predictions';mirror.mkdir(exist_ok=True)
    for path in (OUT/'02_inner_oof').glob('*.csv'):
        dest=mirror/path.name
        if not dest.exists():shutil.copy2(path,dest)
    n=int(summary.final_test_R2.ge(.5).sum())
    lines=['# RANDOM_TABULAR_MODEL_ZOO_V2 最终判定','',f'- 内部 OOF 锁定的任务级单模型/非负融合在固定 Random 测试集上达到 R²≥0.5：{n}/10。','- 表中最优配置由训练集内部验证选择，外部测试结果只用于最终报告。','- Random 单次样本划分与 Station V2 五折留站 pooled OOF 不可作为同协议提升解释。','','| 任务 | 锁定模型 | 测试R² | RMSE | MAE |','|---|---|---:|---:|---:|']
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.task} | {row.selected_models} | {row.final_test_R2:.4f} | {row.final_test_RMSE:.4f} | {row.final_test_MAE:.4f} |')
    lines += ['','## 模型家族与融合判读','',
        '- 传统、深度和预训练模型的逐任务内部排名见 `02_model_registry.csv`，逐模型外部表现见 `08_all_models_test_comparison.csv`；不能用外部测试排名事后替换锁定方案。',
        '- 内部 OOF 的调参前后差值及搜索耗时见 `05_tuned_model_metrics.csv`；融合权重与二级交叉验证见 `06_ensemble_weights.csv`。',
        '- 最终测试 R² 与内部最佳单模型测试 R² 的差异只描述已经锁定方案的结果，不是选择融合的依据。',
        '- TabDPT 的官方权重和完整样本单折试跑通过，但十任务五折规模运行本轮未完成；TabFM、Mitra-v2 未完成本机独立整规模运行，均未填入虚构分数。',
        '- 本机实际 GPU 为 RTX 2070 Max-Q 8GB。内部和外部运行耗时、GPU 峰值内存记录分别见 `01_model_runtime_status.csv` 与逐模型预测 JSON；原脚本未拆分计时的模型仅报告总耗时，训练/推理分项为 NA。',
        '- Random 单次样本划分结果不能作为五折留站空间泛化结果。',
        '','| 任务 | 内部最佳单模型 | 单模型测试R² | 锁定方案测试R² | 差值 |',
        '|---|---|---:|---:|---:|']
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.task} | {row.inner_best_single_model} | {row.inner_best_single_test_R2:.4f} | {row.final_test_R2:.4f} | {row.test_delta_R2_vs_inner_best_single:+.4f} |')
    (OUT/'12_final_decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('R2>=0.5',n,'/10')

if __name__=='__main__':main()
