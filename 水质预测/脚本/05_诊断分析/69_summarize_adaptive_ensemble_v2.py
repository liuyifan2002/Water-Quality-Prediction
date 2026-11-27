"""ADAPTIVE_ENSEMBLE_V2 站点配对统计与中文结果报告。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/adaptive_ensemble_v2'
V1 = ROOT / 'runs/adaptive_ensemble_v1'


def differences(y, selected, reference):
    return (float(r2_score(y, selected)-r2_score(y, reference)),
            float(mean_squared_error(y, selected)**.5-mean_squared_error(y, reference)**.5),
            float(mean_absolute_error(y, selected)-mean_absolute_error(y, reference)))


def run():
    selection = pd.read_csv(OUT/'03_task_selection.csv')
    comparison = pd.read_csv(OUT/'05_v2_test_metrics.csv')
    predicted = pd.read_parquet(OUT/'04_v2_selected_predictions.parquet')
    inventory = pd.read_csv(V1/'00_existing_model_asset_inventory.csv')
    v1_details = pd.read_csv(V1/'03_ten_task_ensemble_comparison.csv')
    wins, intervals = [], []
    for task in selection.task:
        subset = predicted.loc[predicted.task.eq(task)]
        wide = subset.pivot(index='obs_id', columns='variant', values='y_pred')
        meta = subset.loc[subset.variant.eq('v2_selected')].set_index('obs_id')
        if len(meta) != len(wide) or not wide.notna().all().all():
            raise ValueError(f'{task}: 测试预测配对不完整')
        wide = wide.loc[meta.index]
        y = meta.y_true.to_numpy(float)
        p = wide.v2_selected.to_numpy(float)
        stations = meta.station_id.to_numpy(str)
        groups = [np.flatnonzero(stations == station) for station in sorted(set(stations))]
        for kind, reference in [('v1_fusion', 'v1_fusion'),
                                ('v1_inner_selected_single', 'v1_inner_selected_single')]:
            b = wide[reference].to_numpy(float)
            rmse_diffs = [np.mean((y[g]-p[g])**2)-np.mean((y[g]-b[g])**2) for g in groups]
            mae_diffs = [np.mean(abs(y[g]-p[g]))-np.mean(abs(y[g]-b[g])) for g in groups]
            station_rmse_wins = sum(value < -1e-12 for value in rmse_diffs)
            station_mae_wins = sum(value < -1e-12 for value in mae_diffs)
            wins.append({'task': task, 'reference': kind, 'n_stations': len(groups),
                'n_test': len(y), 'station_RMSE_win_rate': station_rmse_wins/len(groups),
                'station_MAE_win_rate': station_mae_wins/len(groups),
                'station_RMSE_tie_rate': sum(abs(value) <= 1e-12 for value in rmse_diffs)/len(groups),
                'station_MAE_tie_rate': sum(abs(value) <= 1e-12 for value in mae_diffs)/len(groups)})
            rng = np.random.default_rng(42)
            draws = np.empty((1000, 3))
            for i in range(1000):
                ids = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
                draws[i] = differences(y[ids], p[ids], b[ids])
            points = differences(y, p, b)
            for column, name in enumerate(['delta_R2', 'delta_RMSE', 'delta_MAE']):
                lower, upper = np.quantile(draws[:, column], [.025, .975])
                intervals.append({'task': task, 'reference': kind, 'metric': name,
                    'point': points[column], 'ci95_low': lower, 'ci95_high': upper,
                    'crosses_zero': bool(lower <= 0 <= upper), 'resamples': 1000,
                    'n_stations': len(groups)})
    pd.DataFrame(wins).to_csv(OUT/'06_station_win_rates.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(intervals).to_csv(OUT/'07_paired_bootstrap.csv', index=False, encoding='utf-8-sig')
    summary = []
    for row in selection.itertuples():
        summary.append({'task': row.task, 'selected_type': row.selected_candidate_type,
            'selected_candidate': row.selected_candidate,
            'selected_family': ('TabICLv2' if 'tabicl' in row.selected_candidate else
                                '其他单模型' if row.selected_candidate_type == 'single' else '融合'),
            'tabicl_selected_as_single': int(row.selected_candidate_type == 'single' and
                                             'tabicl' in row.selected_candidate),
            'tabicl_in_selected_fusion': int(row.selected_candidate_type == 'fusion' and
                'tabicl' in str(v1_details.set_index('task').loc[row.task, 'selected_base_models']))})
    pd.DataFrame(summary).to_csv(OUT/'08_selection_summary.csv', index=False, encoding='utf-8-sig')
    registry = pd.DataFrame([{'stage': 'V1资产读取', 'status': 'COMPLETE', 'tasks': 10},
        {'stage': '内层统一候选评价', 'status': 'COMPLETE', 'tasks': len(selection)},
        {'stage': '外部测试评价', 'status': 'COMPLETE', 'tasks': len(comparison)},
        {'stage': '站点配对Bootstrap', 'status': 'COMPLETE', 'tasks': 10,
         'resamples_per_task_and_reference': 1000}])
    registry.to_csv(OUT/'10_run_registry.csv', index=False, encoding='utf-8-sig')
    lines = ['# ADAPTIVE_ENSEMBLE_V2 十任务联合选择实验', '',
        '## 口径', '',
        '本轮沿用冻结Random Strict外部测试、Model Zoo V2内层五折、77项strict特征，以及V1保存的151份单模型资产和M0—M3交叉拟合预测。'
        '仅使用训练集OOF进行联合选择：融合相对最优单模型的pooled OOF R²优势须严格大于0.005，且至少3折胜出。'
        '选择结果在外部测试评价前写入`03_task_selection.csv`。没有训练基础模型。', '',
        'V1的二层交叉拟合基于已生成的基础模型OOF，不等同于重新训练基础模型的严格嵌套OOF。'
        '外部测试结果是性能判断依据，不能据其重新选择候选。Random Strict不表示未见站点泛化。', '',
        '## 十任务外部测试结果', '',
        '|任务|V2选择|V1 R²|V2 R²|ΔR²|V1 RMSE|V2 RMSE|V1 MAE|V2 MAE|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparison.itertuples():
        lines.append(f'|{row.task}|{row.selected_candidate_type}: {row.selected_candidate}|'
            f'{row.v1_R2:.4f}|{row.v2_R2:.4f}|{row.delta_R2_vs_v1:+.4f}|'
            f'{row.v1_RMSE:.4f}|{row.v2_RMSE:.4f}|{row.v1_MAE:.4f}|{row.v2_MAE:.4f}|')
    selected = selection.selected_candidate_type.value_counts()
    counts = selection.loc[selection.selected_candidate_type.eq('fusion'), 'selected_candidate'].value_counts()
    above = int((comparison.v2_R2 >= .5).sum())
    improved = int((comparison.delta_R2_vs_v1 > 1e-12).sum())
    worsened = int((comparison.delta_R2_vs_v1 < -1e-12).sum())
    tabicl_single = sum((selection.selected_candidate_type.eq('single')) &
                        selection.selected_candidate.str.contains('tabicl'))
    tabicl_fusion = int(pd.DataFrame(summary).tabicl_in_selected_fusion.sum())
    lines += ['', '## 选择与站点证据', '',
        f'按预定规则，单模型入选{int(selected.get("single", 0))}项，融合入选{int(selected.get("fusion", 0))}项；'
        + '、'.join(f'{m} {int(counts.get(m,0))}项' for m in ['M0','M1','M2','M3'])+'。',
        f'TabICLv2及其已有变体作为最终单模型入选{tabicl_single}项，参与最终融合组合{tabicl_fusion}项。'
        '“参与融合”仅表示进入该融合的基础模型组合，不证明其单独增益。',
        f'V2相对V1：R²提升{improved}项、退化{worsened}项、完全相同{10-improved-worsened}项，'
        f'平均ΔR²={comparison.delta_R2_vs_v1.mean():+.4f}；'
        f'V2达到R²≥0.5的任务为{above}/10。',
        '站点RMSE/MAE胜率和1000次站点聚类配对Bootstrap的ΔR²、ΔRMSE、ΔMAE区间分别见'
        '`06_station_win_rates.csv`与`07_paired_bootstrap.csv`。两项比较对象分别是V1融合预测和V1按内层OOF选出的单模型；'
        '同一重抽样中双方使用同一批站点。站点胜率以全部站点为分母，预测完全相同的站点按平局单列；'
        '区间跨零时不宣称明确优势。', '',
        '## 实验判断', '',
        '本轮形成了以训练集OOF为依据的任务自适应单模型—融合模型联合选择流程。'
        f'但测试集平均ΔR²为{comparison.delta_R2_vs_v1.mean():+.4f}，不能宣称V2总体优于V1。'
        'TN中层、TP表层、TP中层转用单模型后有所提升；TN底层和TP底层转用单模型后退化，'
        '说明OOF实用差异阈值只能降低部分融合风险，不能保证外部测试逐任务提升。'
        '不得把测试集事后最强单模型用于选择，'
        '也不得把Random Strict结果外推至Station Holdout。'
        '未开展V3、基础模型训练、SHAP或ALE。', '',
        '最终状态：ADAPTIVE_ENSEMBLE_V2_COMPLETE']
    (OUT/'09_final_decision.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'V2 complete: {above}/10 R2>=0.5; {improved}/10 improved over V1')


if __name__ == '__main__':
    run()
