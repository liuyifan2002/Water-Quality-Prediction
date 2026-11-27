"""在冻结 Random Strict OOF 上联合选择单模型与 V1 融合模型。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / 'runs/adaptive_ensemble_v1'
OUT = ROOT / 'runs/adaptive_ensemble_v2'
SPLIT = ROOT / 'runs/random_10task_v1/01_random_split_manifest.csv'
INNER = ROOT / 'runs/random_tabular_model_zoo_v2/01_inner_fold_manifest.csv'
SPLIT_SHA = 'bb9269039de23f8fb4412eb71bf8268a972ec40ca17c3d0b333a6e2fa791b359'
INNER_SHA = 'c4fea605966709a29baf014ed7eb6c8c557d06122728017ea127ef9ae21213cc'
TASKS = ['chla_surface', 'chla_middle', 'chla_bottom', 'secchi',
         'tn_surface', 'tn_middle', 'tn_bottom', 'tp_surface', 'tp_middle', 'tp_bottom']


def metrics(y, p):
    return {'R2': float(r2_score(y, p)),
            'RMSE': float(mean_squared_error(y, p) ** .5),
            'MAE': float(mean_absolute_error(y, p))}


def parse_prediction(paths, expected, truth=None, folds=None):
    frames = [pd.read_csv(ROOT / p, dtype={'obs_id': str}) for p in paths.split('|')]
    frame = pd.concat(frames, ignore_index=True).set_index('obs_id')
    if not frame.index.is_unique or set(frame.index) != set(expected):
        raise ValueError('obs_id 不匹配或重复')
    frame = frame.loc[expected]
    if truth is not None and not np.allclose(frame.y_true, truth, atol=1e-8, rtol=0):
        raise ValueError('真实值不一致')
    if folds is not None:
        column = 'inner_fold' if 'inner_fold' in frame else 'fold'
        if not np.array_equal(frame[column].to_numpy(int), folds):
            raise ValueError('内层折次不一致')
    if not np.isfinite(frame.y_pred.to_numpy(float)).all():
        raise ValueError('预测中含非有限值')
    return frame


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for path, wanted in [(SPLIT, SPLIT_SHA), (INNER, INNER_SHA)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
            raise ValueError(f'冻结划分哈希不一致: {path}')
    inventory = pd.read_csv(V1 / '00_existing_model_asset_inventory.csv')
    coverage = pd.read_csv(V1 / '01_oof_coverage.csv')
    v1_comparison = pd.read_csv(V1 / '03_ten_task_ensemble_comparison.csv').set_index('task')
    fusion = pd.read_parquet(V1 / '02_ensemble_predictions.parquet')
    split = pd.read_csv(SPLIT, dtype={'obs_id': str})
    inner = pd.read_csv(INNER, dtype={'obs_id': str})
    availability, scores, selections = [], [], []
    # 第一阶段只读取内层OOF，选择结果落盘前不评价外部测试。
    for task in TASKS:
        expected = pd.Index(inner.loc[inner.task.eq(task), 'obs_id'].astype(str))
        fold_map = inner.loc[inner.task.eq(task)].set_index('obs_id').inner_fold
        folds = fold_map.loc[expected].to_numpy(int)
        if len(expected) != int(coverage.set_index('task').loc[task, 'n_train_oof']):
            raise ValueError(f'{task}: OOF 样本数与V1不符')
        task_fusion = fusion.loc[fusion.task.eq(task) & fusion.split.eq('inner_meta_cv')]
        anchor = task_fusion.loc[task_fusion.method.eq('M0')].set_index('obs_id')
        if not anchor.index.is_unique or set(anchor.index) != set(expected):
            raise ValueError(f'{task}: V1融合OOF不完整')
        truth = anchor.loc[expected, 'y_true'].to_numpy(float)
        task_scores = []
        for asset in inventory.loc[inventory.task.eq(task)].itertuples():
            try:
                frame = parse_prediction(asset.oof_prediction_file, expected, truth, folds)
                pred = frame.y_pred.to_numpy(float)
                status, reason = 'AVAILABLE', ''
            except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
                status, reason = 'EXCLUDED', str(error)
                pred = None
            availability.append({'task': task, 'candidate_type': 'single', 'candidate': asset.model,
                                 'status': status, 'reason': reason, 'oof_n': len(expected) if pred is not None else 0})
            if pred is not None:
                task_scores.append((asset.model, 'single', pred))
        for method in ['M0', 'M1', 'M2', 'M3']:
            frame = task_fusion.loc[task_fusion.method.eq(method)].set_index('obs_id')
            if frame.index.is_unique and set(frame.index) == set(expected) and np.array_equal(
                frame.loc[expected, 'inner_fold'].to_numpy(int), folds) and np.allclose(
                    frame.loc[expected, 'y_true'].to_numpy(float), truth, atol=1e-8, rtol=0):
                pred = frame.loc[expected, 'y_pred'].to_numpy(float)
                status, reason = ('AVAILABLE', '') if np.isfinite(pred).all() else ('EXCLUDED', '非有限预测')
            else:
                pred, status, reason = None, 'EXCLUDED', 'V1二层交叉拟合预测不完整'
            availability.append({'task': task, 'candidate_type': 'fusion', 'candidate': method,
                                 'status': status, 'reason': reason, 'oof_n': len(expected) if status == 'AVAILABLE' else 0})
            if status == 'AVAILABLE':
                task_scores.append((method, 'fusion', pred))
        for name, kind, pred in task_scores:
            overall = metrics(truth, pred)
            fold_scores = [metrics(truth[folds == f], pred[folds == f])['R2'] for f in sorted(set(folds))]
            row = {'task': task, 'candidate_type': kind, 'candidate': name,
                   'pooled_oof_R2': overall['R2'], 'pooled_oof_RMSE': overall['RMSE'],
                   'pooled_oof_MAE': overall['MAE'], 'fold_mean_R2': float(np.mean(fold_scores)),
                   'fold_sd_R2': float(np.std(fold_scores, ddof=1))}
            row.update({f'fold_{i}_R2': value for i, value in enumerate(fold_scores, 1)})
            scores.append(row)
        single = max((x for x in task_scores if x[1] == 'single'),
                     key=lambda x: (metrics(truth, x[2])['R2'], x[0]))
        best_fusion = max((x for x in task_scores if x[1] == 'fusion'),
                          key=lambda x: (metrics(truth, x[2])['R2'], x[0]))
        advantage = metrics(truth, best_fusion[2])['R2'] - metrics(truth, single[2])['R2']
        fold_wins = sum(metrics(truth[folds == f], best_fusion[2][folds == f])['R2'] >
                        metrics(truth[folds == f], single[2][folds == f])['R2']
                        for f in sorted(set(folds)))
        choose_fusion = advantage > .005 and fold_wins >= 3
        chosen = best_fusion if choose_fusion else single
        selections.append({'task': task, 'best_single_model': single[0],
            'best_single_OOF_R2': metrics(truth, single[2])['R2'],
            'best_fusion_method': best_fusion[0],
            'best_fusion_OOF_R2': metrics(truth, best_fusion[2])['R2'],
            'fusion_advantage': advantage, 'fusion_fold_win_count': fold_wins,
            'selected_candidate_type': chosen[1], 'selected_candidate': chosen[0],
            'selection_reason': ('融合优势>0.005且至少3折胜出' if choose_fusion else
                                 '单模型优先：融合优势或折次胜出未达到预定阈值')})
    pd.DataFrame(availability).to_csv(OUT/'01_candidate_availability.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(scores).to_csv(OUT/'02_inner_candidate_scores.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(selections).to_csv(OUT/'03_task_selection.csv', index=False, encoding='utf-8-sig')
    # 选择表已固定；第二阶段才读取外部测试预测并计算指标。
    selected_predictions, comparison = [], []
    for choice in selections:
        task = choice['task']
        task_split = split.loc[split.task.eq(task) & split.split.eq('test')]
        expected = pd.Index(task_split.obs_id.astype(str))
        def read_single(name):
            row = inventory.loc[inventory.task.eq(task) & inventory.model.eq(name)]
            if len(row) != 1:
                raise ValueError(f'{task}/{name}: 单模型资产未唯一登记')
            return parse_prediction(row.iloc[0].test_prediction_file, expected)
        def read_fusion(method):
            frame = fusion.loc[fusion.task.eq(task) & fusion.method.eq(method) &
                               fusion.split.eq('external_test')].set_index('obs_id')
            if not frame.index.is_unique or set(frame.index) != set(expected):
                raise ValueError(f'{task}/{method}: V1融合测试预测不完整')
            return frame.loc[expected]
        single = read_single(choice['best_single_model'])
        v1_method = v1_comparison.loc[task, 'selected_fusion']
        v1 = read_fusion(v1_method)
        chosen = (single if choice['selected_candidate_type'] == 'single'
                  else read_fusion(choice['selected_candidate']))
        y = chosen.y_true.to_numpy(float)
        if not np.allclose(single.y_true.to_numpy(float), y, atol=1e-8, rtol=0) or not np.allclose(
            v1.y_true.to_numpy(float), y, atol=1e-8, rtol=0):
            raise ValueError(f'{task}: 测试真实值不一致')
        p = chosen.y_pred.to_numpy(float)
        p1 = v1.y_pred.to_numpy(float)
        p_single = single.y_pred.to_numpy(float)
        result, old, base = metrics(y, p), metrics(y, p1), metrics(y, p_single)
        for label, values in [('v2_selected', p), ('v1_fusion', p1), ('v1_inner_selected_single', p_single)]:
            selected_predictions.append(pd.DataFrame({'task': task, 'variant': label,
                'obs_id': expected, 'station_id': task_split.set_index('obs_id').loc[expected, 'station'].values,
                'y_true': y, 'y_pred': values}))
        comparison.append({'task': task, 'selected_candidate_type': choice['selected_candidate_type'],
            'selected_candidate': choice['selected_candidate'], 'v1_fusion_method': v1_method,
            'v1_R2': old['R2'], 'v1_RMSE': old['RMSE'], 'v1_MAE': old['MAE'],
            'v1_inner_single_R2': base['R2'], 'v1_inner_single_RMSE': base['RMSE'],
            'v1_inner_single_MAE': base['MAE'], 'v2_R2': result['R2'],
            'v2_RMSE': result['RMSE'], 'v2_MAE': result['MAE'],
            'delta_R2_vs_v1': result['R2'] - old['R2'],
            'delta_RMSE_vs_v1': result['RMSE'] - old['RMSE'],
            'delta_MAE_vs_v1': result['MAE'] - old['MAE']})
    pd.concat(selected_predictions, ignore_index=True).to_parquet(
        OUT/'04_v2_selected_predictions.parquet', index=False)
    pd.DataFrame(comparison).to_csv(OUT/'05_v2_test_metrics.csv', index=False, encoding='utf-8-sig')
    (OUT/'00_v2_protocol.md').write_text(
        '# ADAPTIVE_ENSEMBLE_V2 固定协议\n\n'
        '沿用 RANDOM_10TASK_V1 外部划分、Model Zoo V2 五折及 V1 的全部基础预测与四种融合二层交叉拟合预测。'
        '单模型按训练集 pooled OOF R² 选择；融合仅当其 OOF R² 优势严格大于0.005且至少3折胜出时入选。'
        '选择表先写出，再读取外部测试预测。外部测试仅作评价，不参与选择。'
        'V1二层交叉拟合不是基础模型严格嵌套OOF；独立外部测试为主要性能证据。\n\n'
        f'外部划分 SHA256：`{SPLIT_SHA}`；内部划分 SHA256：`{INNER_SHA}`。\n', encoding='utf-8')
    print('V2 selection and external prediction complete:', len(comparison), 'tasks')


if __name__ == '__main__':
    run()
