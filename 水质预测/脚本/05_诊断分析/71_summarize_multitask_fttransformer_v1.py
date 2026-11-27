"""汇总冻结留站实验的十任务、种子、水区指标和可比基准。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/multitask_fttransformer_v1'
BASE = ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115/03_正式十任务重算/predictions'
TASKS = ['chla_surface','chla_middle','chla_bottom','secchi','tn_surface',
         'tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']


def score(frame):
    y=frame.y_true.to_numpy(float);p=frame.y_pred.to_numpy(float)
    return {'R2':float(r2_score(y,p)) if len(y)>1 else np.nan,
            'RMSE':float(mean_squared_error(y,p)**.5),
            'MAE':float(mean_absolute_error(y,p)), 'n':len(y)}


def run():
    files=list((OUT/'predictions').glob('*.csv'))
    if not files: raise ValueError('尚无逐折预测文件')
    all_pred=pd.concat((pd.read_csv(f,dtype={'obs_id':str}) for f in files),ignore_index=True)
    if all_pred.duplicated(['variant','seed','task','obs_id']).any():
        raise ValueError('模型-种子-任务预测ID重复')
    fold_count=all_pred.groupby(['variant','seed','task']).fold.nunique()
    if fold_count.min()!=5 or fold_count.size!=90:
        raise ValueError(f'预期3模型×3种子×10任务各5折；目前{fold_count.size}组，最少{fold_count.min()}折')
    metrics=[]; zone=[]
    for (variant,seed,task),sub in all_pred.groupby(['variant','seed','task'],sort=False):
        metrics.append({'variant':variant,'seed':seed,'task':task,'aggregation':'pooled_oof',**score(sub)})
        for fold, part in sub.groupby('fold'):
            metrics.append({'variant':variant,'seed':seed,'task':task,
                            'aggregation':fold,**score(part)})
        for water_zone,part in sub.groupby('water_zone'):
            zone.append({'variant':variant,'seed':seed,'task':task,
                         'water_zone':water_zone,**score(part)})
    table=pd.DataFrame(metrics)
    table.to_csv(OUT/'task_and_fold_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(zone).to_csv(OUT/'water_zone_metrics.csv',index=False,encoding='utf-8-sig')
    pooled=table.loc[table.aggregation.eq('pooled_oof')]
    # RMSE/MAE have different units across water-quality targets and are not macro-averaged.
    macro=pooled.groupby(['variant','seed'],as_index=False).agg(macro_mean_R2=('R2','mean'))
    macro.to_csv(OUT/'macro_metrics.csv',index=False,encoding='utf-8-sig')
    baseline=[]
    for task in TASKS:
        example=all_pred.loc[all_pred.task.eq(task) & all_pred.variant.eq('M0') &
                             all_pred.seed.eq(42)]
        if len(example)==0:raise ValueError(f'{task}: M0缺失')
        expected=example.set_index('obs_id')
        for model in ['xgboost','tabiclv2']:
            paths=sorted(BASE.glob(f'{task}_{model}_fold_*.csv'))
            if len(paths)!=5: continue
            current=pd.concat((pd.read_csv(p,dtype={'obs_id':str}) for p in paths),ignore_index=True)
            if current.obs_id.duplicated().any() or set(current.obs_id)!=set(expected.index):
                continue
            current=current.set_index('obs_id').loc[expected.index]
            if current.dataset_version.nunique()!=1 or current.dataset_version.iloc[0]!='WQ77_REPAIRED_20261115_V1':
                continue
            if not np.allclose(current.y_true,expected.y_true,atol=1e-8,rtol=0):
                continue
            baseline.append({'task':task,'model':model,'comparison_status':'同基础版本/同任务/同测试ID；并集留站训练范围及折内缺失处理不同，仅描述性参考',**score(current)})
    pd.DataFrame(baseline).to_csv(OUT/'current_version_baseline_reference.csv',
                                index=False,encoding='utf-8-sig')
    summary=pooled.groupby(['variant','task'],as_index=False).agg(
        mean_R2=('R2','mean'),sd_R2=('R2','std'),mean_RMSE=('RMSE','mean'),
        mean_MAE=('MAE','mean'),n=('n','first'))
    m0=summary.loc[summary.variant.eq('M0'),['task','mean_R2']].rename(columns={'mean_R2':'M0_R2'})
    m1=summary.loc[summary.variant.eq('M1'),['task','mean_R2']].rename(columns={'mean_R2':'M1_R2'})
    summary=summary.merge(m0,on='task').merge(m1,on='task')
    summary['delta_R2_vs_M0']=summary.mean_R2-summary.M0_R2
    summary['delta_R2_vs_M1']=summary.mean_R2-summary.M1_R2
    summary.to_csv(OUT/'ten_task_summary.csv',index=False,encoding='utf-8-sig')
    by_model=summary.groupby('variant').mean_R2.mean().to_dict()
    m1_gain=summary.loc[summary.variant.eq('M1'),'delta_R2_vs_M0']
    m2_gain=summary.loc[summary.variant.eq('M2'),'delta_R2_vs_M1']
    lines=['# 多任务FT-Transformer留站实验结果','',
      '## 协议与解释边界','',
      '使用修复版WQ77_REPAIRED_20261115_V1的77项strict特征及既有五折留站manifest。'
      'Chl-a/TN/TP的84站折次相互一致，Secchi的94站折次不一致；每折从所有任务训练中排除两套测试站点的并集，'
      '避免共享编码器见到任何当前任务留出站点标签。M0/M1/M2共用这一边界；因此与原XGBoost/TabICLv2基准的训练样本范围和折内缺失处理不同，'
      '基准仅作同测试ID的描述性对照。每折以剩余训练站点划分内部早停验证站点；预处理及目标标准化仅用对应训练站点拟合。', '',
      'M0为10个独立FT-Transformer；M1/M2共享特征Tokenizer和Transformer并使用10个任务头。'
      'M1按所有有效标签计算样本平均掩码MSE，M2先按任务各自有效标签均方误差计算再对活跃任务平均。'
      '目标缺失只通过mask排除，不填补标签。站点和水区不进入输入。'
      '训练中曾出现CUDA驱动异常，进程重启后按原配置续跑，已完成文件不重算。', '',
      '## 十任务结果（三种种子的pooled OOF均值）','',
      '|任务|M0 R²|M1 R²|M2 R²|M1−M0|M2−M1|', '|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        sub=summary.loc[summary.task.eq(task)].set_index('variant')
        lines.append(f'|{task}|{sub.loc["M0","mean_R2"]:.4f}|{sub.loc["M1","mean_R2"]:.4f}|'
            f'{sub.loc["M2","mean_R2"]:.4f}|{sub.loc["M1","delta_R2_vs_M0"]:+.4f}|'
            f'{sub.loc["M2","delta_R2_vs_M1"]:+.4f}|')
    lines+=['',f'十任务宏平均R²：M0 {by_model["M0"]:.4f}、M1 {by_model["M1"]:.4f}、M2 {by_model["M2"]:.4f}。'
        f'M1相对M0改善{int((m1_gain>0).sum())}/10项；M2相对M1改善{int((m2_gain>0).sum())}/10项。'
        '负迁移任务及全部五折/种子结果见CSV，不以宏平均掩盖退化。','',
        '## 对照与后续','',
        '现有XGBoost和TabICLv2只在修复版数据、相同目标及完全相同留站测试obs_id时列为参考；'
        'Random Strict的ADAPTIVE_ENSEMBLE_V1/V2不符合相同测试样本条件，未作数值横比。'
        '分水区结果只用于描述不同海区的误差差异，不把相关性解释为因果机制。'
        '只有共享模型在多任务上呈现一致改善且无严重负迁移，才值得进入任务元数据条件化Hypernetwork阶段；'
        '其预测头可由指标类别和水层元数据生成，Secchi无水层标签。本轮没有实现或训练该模型。','',
        '## 方法参考','',
        '- [香港近岸水质动力的数据驱动建模](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1897038/full)：研究区异质性与时间评价背景；本轮只开展留站验证。',
        '- [采矿区多指标水质FT-Transformer](https://www.sciencedirect.com/science/article/pii/S0013935126015410)：特征Tokenizer与联合预测的设计参考。',
        '- [多目标LSTM水量水质预测](https://hess.copernicus.org/articles/30/7149/2026/)：不完整目标标签与任务均衡损失的研究参考。',
        '- [IJCAI 2024元数据条件化Hypernetwork](https://ijcai.org/proceedings/2024/806)：仅为后续任务头设计参考。']
    (OUT/'experiment_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('summary complete',len(summary),'variant-task rows',flush=True)


if __name__=='__main__':run()
