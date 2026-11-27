"""Summarize frozen-test adaptive ensemble predictions and paired site evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/adaptive_ensemble_v1'
SPLIT = ROOT / 'runs/random_10task_v1/01_random_split_manifest.csv'


def deltas(frame):
    y = frame.y_true.to_numpy(float)
    f = frame.fusion.to_numpy(float)
    b = frame.baseline.to_numpy(float)
    return (float(r2_score(y, f)-r2_score(y, b)),
            float(mean_squared_error(y, f)**.5-mean_squared_error(y, b)**.5),
            float(mean_absolute_error(y, f)-mean_absolute_error(y, b)))


def main():
    comparison = pd.read_csv(OUT/'03_ten_task_ensemble_comparison.csv')
    inventory = pd.read_csv(OUT/'00_existing_model_asset_inventory.csv')
    predictions = pd.read_parquet(OUT/'02_ensemble_predictions.parquet')
    station = pd.read_csv(SPLIT, usecols=['task','obs_id','station','split'],
                          dtype={'obs_id': str}).query("split == 'test'")
    wins, boot = [], []
    for result in comparison.itertuples():
        selected = predictions.loc[predictions.task.eq(result.task) &
            predictions.method.eq(result.selected_fusion) &
            predictions.split.eq('external_test'), ['obs_id','y_true','y_pred']].rename(
                columns={'y_pred': 'fusion'})
        if selected.obs_id.duplicated().any():
            raise ValueError(f'{result.task}: duplicate fusion test ID')
        for comparator_kind, model in [
            ('inner_selected_single', result.inner_best_single_model),
            ('external_best_descriptive', result.external_best_single_descriptive)]:
            row = inventory.loc[(inventory.task == result.task) & (inventory.model == model)]
            if len(row) != 1 or row.status.iloc[0] != '可用':
                raise ValueError(f'{result.task}: comparator model unavailable: {model}')
            base = pd.read_csv(ROOT/row.test_prediction_file.iloc[0],
                               usecols=['obs_id','y_true','y_pred'], dtype={'obs_id': str}).rename(
                                   columns={'y_true': 'baseline_truth', 'y_pred': 'baseline'})
            pair = selected.merge(base, on='obs_id', validate='one_to_one').merge(
                station.loc[station.task.eq(result.task), ['obs_id','station']],
                on='obs_id', validate='one_to_one')
            if len(pair) != len(selected) or not np.allclose(pair.y_true, pair.baseline_truth,
                                                               rtol=0, atol=1e-8):
                raise ValueError(f'{result.task}: paired comparator mismatch')
            rmse_win = mae_win = 0
            groups = []
            for _, item in pair.groupby('station'):
                rmse_win += float(mean_squared_error(item.y_true, item.fusion) <
                                  mean_squared_error(item.y_true, item.baseline))
                mae_win += float(mean_absolute_error(item.y_true, item.fusion) <
                                 mean_absolute_error(item.y_true, item.baseline))
                groups.append(item[['y_true','fusion','baseline']].copy())
            n_station = len(groups)
            wins.append({'task': result.task, 'comparator_kind': comparator_kind,
                'comparator_model': model, 'n_test': len(pair), 'n_stations': n_station,
                'station_RMSE_win_rate': rmse_win/n_station,
                'station_MAE_win_rate': mae_win/n_station})
            rng = np.random.default_rng(42)
            draws = np.asarray([deltas(pd.concat([groups[i] for i in rng.integers(
                0, n_station, n_station)], ignore_index=True)) for _ in range(1000)])
            observed = deltas(pair)
            for idx, metric in enumerate(['delta_R2','delta_RMSE','delta_MAE']):
                low, high = np.quantile(draws[:, idx], [0.025, 0.975])
                boot.append({'task': result.task, 'comparator_kind': comparator_kind,
                    'comparator_model': model, 'metric': metric,
                    'point': observed[idx], 'ci95_low': float(low),
                    'ci95_high': float(high), 'crosses_zero': bool(low <= 0 <= high),
                    'n_stations': n_station, 'resamples': 1000})
    pd.DataFrame(wins).to_csv(OUT/'04_station_win_rates.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(boot).to_csv(OUT/'05_paired_bootstrap.csv', index=False, encoding='utf-8-sig')
    details = json.loads((OUT/'method_details.json').read_text(encoding='utf-8'))
    lines = ['# ADAPTIVE_ENSEMBLE_V1 十任务自适应融合决策', '',
        '## 方法与比较口径', '',
        '最终实验闭环登记表仅保存较早四任务 historical_random_strict 的汇总值，缺少十任务完整obs_id与OOF；本轮沿用其后由 Pilot 划分扩展并冻结的 RANDOM_10TASK_V1 manifest，以及与之完全对应的 Model Zoo V2 五折OOF。这是同一Random Strict链条的后续独立扩展，不将Station Holdout预测混入。', '',
        '固定 Random Strict 外部测试与内部五折、77项 strict 特征；直接复用现有基础模型的训练集OOF和外部测试预测，未重新训练基础模型。每任务从完整OOF模型中按训练集OOF R²选前三个配置。M0为等权，M1为非负且和为1的向等权收缩融合，M2为Ridge，M3为77特征经训练折内填补、标准化和六维PCA后的线性Softmax门控。二层模型在五折OOF上交叉拟合并仅依据内层R²选择；外部测试未用于模型或权重选择。', '',
        '二层交叉拟合是在已有基础模型OOF上进行的内部选择依据，不是完全重新训练基础模型的严格嵌套外层性能估计；独立外部测试是本轮主要评价。最强单模型一栏是在固定外部测试上对所有可用单模型作**事后描述性比较**，没有用于选择融合器。站点Bootstrap相对该事后最强模型的区间也须按这一选择条件谨慎解释。', '',
        '## 十任务结果', '',
        '|任务|测试最强单模型|单模型R²|内层选定融合|融合R²|ΔR²|ΔRMSE|ΔMAE|',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for row in comparison.itertuples():
        lines.append(f'|{row.task}|{row.external_best_single_descriptive}|{row.external_best_single_R2:.4f}|{row.selected_fusion}|{row.fusion_R2:.4f}|{row.delta_R2_vs_external_best_single:+.4f}|{row.delta_RMSE_vs_external_best_single:+.4f}|{row.delta_MAE_vs_external_best_single:+.4f}|')
    improved = int((comparison.delta_R2_vs_external_best_single > 0).sum())
    above = int((comparison.fusion_R2 >= .5).sum())
    rmse_improved = int((comparison.delta_RMSE_vs_external_best_single < 0).sum())
    mae_improved = int((comparison.delta_MAE_vs_external_best_single < 0).sum())
    lines += ['', f'融合在 {improved}/10 项任务超过事后最强单模型，平均ΔR²={comparison.delta_R2_vs_external_best_single.mean():+.4f}；{above}/10 项融合R²≥0.5。RMSE改善{rmse_improved}/10项，MAE改善{mae_improved}/10项。预定的“至少7/10项超过最强单模型”目标**未达到**，不能据十任务全过0.5掩盖这一事实。', '',
        '## 融合机制与站点证据', '']
    counts = comparison.selected_fusion.value_counts().to_dict()
    lines.append('内层选择频次：'+'、'.join(f'{name} {counts.get(name,0)}项' for name in ['M0','M1','M2','M3'])+'。M2在多数任务中被选中；M1仅在部分任务被选，M3在部分TN/TP任务中被选，但其外部增益并不稳定。M0保留为基准，不因其简单而省略。')
    tabicl_tasks = [task for task, item in details.items() if any('tabicl' in model for model in item['selected_base_models'])]
    lines.append(f'TabICLv2及其已锁定变体进入前3基础模型的任务为 {len(tabicl_tasks)}/10：'+ '、'.join(tabicl_tasks)+'。未进入的任务，不能声称TabICLv2对该融合预测有直接贡献。进入候选池也不等于已证明单独增益；M1权重与M3平均权重见 `method_details.json`。')
    strongest_wins = pd.DataFrame(wins).query("comparator_kind == 'external_best_descriptive'")
    lines.append(f'相对事后最强单模型，站点RMSE胜率超过50%的任务有 {int((strongest_wins.station_RMSE_win_rate > .5).sum())}/10；MAE胜率超过50%的任务有 {int((strongest_wins.station_MAE_win_rate > .5).sum())}/10。逐任务比例与1000次配对站点Bootstrap区间分别见 `04_station_win_rates.csv`、`05_paired_bootstrap.csv`。')
    lines += ['', '## 实验判断与边界', '',
        '本轮观察到任务级融合对部分Chl-a、Secchi及TN任务有帮助，但TN中层、TP表层、TP中层、TP底层等任务并未超过最强单模型；这些负面结果全部保留。TP底层虽然融合R²达到0.5以上，但其最强单模型仍更好。融合改善是预测统计关联，不能解释为环境因果贡献。Random Strict不代表未见站点泛化；本轮不启动站点修正或三水层多任务学习。']
    (OUT/'06_final_decision.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Summary:', improved, 'of 10 beat descriptive strongest single;', above, 'of 10 R2>=0.5')


if __name__ == '__main__':
    main()
