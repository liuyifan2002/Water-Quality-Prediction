"""Produce Chinese, test-only-once comparison after ten locked evaluations exist."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tabicl_water_quality_adaptation_v1'
ZOO = ROOT / 'runs/random_tabular_model_zoo_v2'
TASKS = ['chla_surface','chla_middle','chla_bottom','secchi','tn_surface',
         'tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']


def main():
    chosen = pd.read_csv(OUT / '10_task_best_tabicl_config.csv')
    baseline = pd.read_csv(OUT / '01_baseline_tabicl_results.csv')
    zoo = pd.read_csv(ZOO / '08_all_models_test_comparison.csv')
    zoo = zoo.loc[~zoo.family.eq('selected')].copy()
    rows = []
    rankings = []
    for task in TASKS:
        path = OUT / '08_final_test_predictions' / f'{task}_tabicl_adapted.json'
        if not path.exists(): raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding='utf-8'))
        b = baseline.loc[baseline.task.eq(task)].iloc[0]
        others = zoo.loc[zoo.task.eq(task)].copy()
        others = others.loc[~others.model.eq('tabicl')]
        ranking = pd.concat([others[['task','model','family','R2','RMSE','MAE']],
                             pd.DataFrame([dict(task=task,model='tabicl_adapted',family='foundation',
                                                R2=result['R2'],RMSE=result['RMSE'],MAE=result['MAE'])])],
                            ignore_index=True)
        ranking = ranking.sort_values(['R2','RMSE','MAE'],ascending=[False,True,True]).reset_index(drop=True)
        ranking.insert(0,'rank',range(1,len(ranking)+1))
        rankings.append(ranking)
        rank = int(ranking.loc[ranking.model.eq('tabicl_adapted'),'rank'].iloc[0])
        competitor = ranking.loc[~ranking.model.eq('tabicl_adapted')].iloc[0]
        rows.append(dict(task=task,config_id=result['config_id'],old_R2=b.old_test_R2,
                         new_R2=result['R2'],delta_R2=result['R2']-b.old_test_R2,
                         old_RMSE=b.old_test_RMSE,new_RMSE=result['RMSE'],
                         delta_RMSE=result['RMSE']-b.old_test_RMSE,
                         old_MAE=b.old_test_MAE,new_MAE=result['MAE'],
                         delta_MAE=result['MAE']-b.old_test_MAE,
                         n_test=result['n_test'],single_model_rank=rank,
                         best_other_model=competitor.model,best_other_R2=competitor.R2,
                         R2_at_least_0_5=result['R2']>=0.5))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / '09_final_10task_comparison.csv',index=False,encoding='utf-8-sig')
    pd.concat(rankings,ignore_index=True).to_csv(OUT / '11_model_ranking.csv',index=False,encoding='utf-8-sig')
    best = int(table.single_model_rank.eq(1).sum())
    half = int(table.R2_at_least_0_5.sum())
    lines = ['# TabICLv2近岸水质领域适配最终决定','',
             '本实验只在固定Random Strict外部训练集内部选配置，固定测试集仅用于锁定后评价。',
             f'优化后TabICLv2单模型在{best}/10项任务排名第一；{half}/10项任务测试R²≥0.5。',
             '目标为研究期望，未达到的任务均如实保留。', '',
             '|任务|内部锁定配置|旧R²|新R²|ΔR²|单模型名次|',
             '|---|---|---:|---:|---:|---:|']
    for row in table.itertuples():
        lines.append(f'|{row.task}|{row.config_id}|{row.old_R2:.4f}|{row.new_R2:.4f}|{row.delta_R2:+.4f}|{row.single_model_rank}|')
    lines += ['', '本结果属于单次Random Strict外部划分，不代表未见站点的Station Holdout泛化。',
              '与Model Zoo竞争模型的排名只是相同测试集上的描述性比较；本实验未据此选择配置。']
    inner = pd.read_csv(OUT / '03_all_inner_candidate_metrics.csv')
    complete = inner.loc[inner.folds.eq(5) & ~inner.config_id.str.startswith('FT')].copy()
    lines += ['', '## 内部验证与适配判断', '',
              '完整五折内部OOF仅用于锁定配置；单折筛选结果不等于十任务验证结论。',
              '默认归一化为none+power，Z1与默认相同故未重算。Z2/Z3仅在部分任务有小幅内部收益；特征排列、估计器数量和特征筛选没有形成跨任务一致优势。',
              '目标处理方面，Chl-a中层及TP底层选择原始目标，其他任务保持log1p；所有指标在原始水质单位计算。',
              '完整五折入选的特征子集仅为TN中层F60，其内部OOF R²为%.4f，原始77特征基线为%.4f；外部测试仍按已锁定配置报告。' % (
                  float(complete.loc[complete.task.eq('tn_middle') & complete.config_id.eq('F60'),'R2'].iloc[0]),
                  float(complete.loc[complete.task.eq('tn_middle') & complete.config_id.eq('Z0'),'R2'].iloc[0])),
              '', '## 官方微调代表任务', '',
              'TN中层采用官方FinetunedTabICLRegressor进行一个内部折试验（FT1_CHUNK2048）：完整训练折保留5184个训练样本，2048仅为微调计算块大小；早停第7轮结束。',
              '微调首折R²=0.6263，低于相同首折零样本基线0.6432；峰值Torch显存分配约9552 MiB。本试验未进入完整五折、配置选择或外部测试，不能宣称领域微调有效。',
              '', '## 结果边界', '',
              '十任务中仅2项取得相同测试集上TabICLv2单模型第一，9项测试R²达到0.5；TP底层仍低于0.5。',
              '外部测试中TN表层、TN中层相对原TabICLv2基线退化，保留其真实结果，不按测试分数回退配置。']
    (OUT / '13_final_decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FIRST',best,'R2_GE_0.5',half)


if __name__=='__main__':main()
