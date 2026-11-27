"""Summarize isolated context/temporal predictions without rerunning models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tabicl_context_temporal_adaptation_v1'
OLD = ROOT / 'runs/tabicl_water_quality_adaptation_v1'
ZOO = ROOT / 'runs/random_tabular_model_zoo_v2'
TASKS = ['chla_surface','chla_middle','chla_bottom','secchi','tn_surface',
         'tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']


def score(frame: pd.DataFrame) -> dict:
    return dict(R2=float(r2_score(frame.y_true, frame.y_pred)),
                RMSE=float(mean_squared_error(frame.y_true, frame.y_pred) ** .5),
                MAE=float(mean_absolute_error(frame.y_true, frame.y_pred)))


def read_parts(directory: str) -> pd.DataFrame:
    paths = sorted((OUT / directory).glob('*.csv'))
    if not paths:
        return pd.DataFrame(columns=['task','fold','scheme','temporal_set','obs_id','y_true','y_pred'])
    return pd.concat((pd.read_csv(path, dtype={'obs_id': str}) for path in paths), ignore_index=True)


def metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=['task','scheme','temporal_set','folds','n','R2','RMSE','MAE','fold_R2_mean','fold_R2_sd'])
    rows = []
    for (task, scheme, temporal), item in frame.groupby(['task','scheme','temporal_set'], sort=False):
        folds = [score(group)['R2'] for _, group in item.groupby('fold')]
        rows.append(dict(task=task, scheme=scheme, temporal_set=temporal,
                         folds=len(folds), n=len(item), **score(item),
                         fold_R2_mean=float(np.mean(folds)),
                         fold_R2_sd=float(np.std(folds, ddof=1)) if len(folds)>1 else np.nan))
    return pd.DataFrame(rows)


def paired_station_win(candidate: pd.DataFrame, control: pd.DataFrame, stations: pd.DataFrame) -> dict:
    pair = candidate[['obs_id','y_true','y_pred']].merge(
        control[['obs_id','y_pred']].rename(columns={'y_pred':'control_pred'}),
        on='obs_id', validate='one_to_one')
    if len(pair) != len(candidate) or len(pair) != len(control):
        raise ValueError('Candidate and control query IDs differ')
    pair = pair.merge(stations, on='obs_id', validate='many_to_one')
    wins_rmse = wins_mae = 0
    for _, item in pair.groupby('station'):
        diff_a = item.y_pred - item.y_true
        diff_b = item.control_pred - item.y_true
        wins_rmse += float(np.sqrt(np.mean(diff_a**2)) < np.sqrt(np.mean(diff_b**2)))
        wins_mae += float(np.mean(np.abs(diff_a)) < np.mean(np.abs(diff_b)))
    n = pair.station.nunique()
    return dict(n_stations=int(n), station_RMSE_win_rate=wins_rmse/n,
                station_MAE_win_rate=wins_mae/n)


def summarize_inner() -> None:
    frame = read_parts('03_context_oof_parts')
    if frame.empty:
        return
    frame.to_parquet(OUT / '03_context_oof_predictions.parquet', index=False)
    frame.loc[frame.temporal_set.ne('L0')].to_parquet(OUT / '06_temporal_oof_predictions.parquet', index=False)
    table = metric_table(frame)
    stations = pd.read_csv(ROOT / '数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv',
                           usecols=['obs_id','station'], dtype={'obs_id': str})
    wins = []
    for (task, fold, scheme, temporal), candidate in frame.groupby(['task','fold','scheme','temporal_set']):
        control_temporal = 'L0' if scheme == 'C0' else temporal
        control = frame.loc[frame.task.eq(task) & frame.fold.eq(fold) &
                            frame.scheme.eq('C0' if scheme != 'C0_32' else 'C0_32') &
                            frame.temporal_set.eq(control_temporal)]
        if scheme.startswith('C1_'):
            control = frame.loc[frame.task.eq(task) & frame.fold.eq(fold) &
                                frame.scheme.eq('C0_32') & frame.temporal_set.eq(temporal)]
        if control.empty or len(control) != len(candidate): continue
        wins.append(dict(task=task, fold=fold, scheme=scheme, temporal_set=temporal,
                         **paired_station_win(candidate, control, stations)))
    table.to_csv(OUT / '02_context_screen_metrics.csv', index=False, encoding='utf-8-sig')
    table.loc[table.temporal_set.ne('L0')].to_csv(OUT / 'temporal_oof_metrics.csv', index=False, encoding='utf-8-sig')
    table.loc[table.temporal_set.ne('L0') & ~table.scheme.eq('C0')].to_csv(
        OUT / '07_combined_oof_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(wins).to_csv(OUT / 'station_win_rates_by_fold.csv', index=False, encoding='utf-8-sig')
    runtime = []
    for directory in ['03_context_oof_parts','09_final_test_predictions_parts',
                      'external_same_batch_references']:
        for path in sorted((OUT / directory).glob('*.json')):
            runtime.append(json.loads(path.read_text(encoding='utf-8')))
    pd.DataFrame(runtime).to_csv(OUT / '12_runtime_metrics.csv', index=False, encoding='utf-8-sig')
    print(table.sort_values(['task','temporal_set','scheme']).to_string(index=False))


def lock_configs() -> None:
    path = OUT / '08_final_task_configuration.csv'
    if path.exists(): raise FileExistsError('Final configurations already locked')
    table = pd.read_csv(OUT / '02_context_screen_metrics.csv')
    prior = pd.read_csv(OLD / '10_task_best_tabicl_config.csv').set_index('task')
    rows = []
    for task in TASKS:
        candidates = table.loc[table.task.eq(task) & table.folds.eq(5) &
                               ~table.scheme.eq('C0_32') & ~table.scheme.str.startswith('C1_')].copy()
        baseline = candidates.loc[candidates.scheme.eq('C0') & candidates.temporal_set.eq('L0')]
        if baseline.empty:
            if not candidates.empty:
                raise ValueError(f'{task}: fivefold candidate has no same-batch global baseline')
            old = prior.loc[task]
            rows.append(dict(task=task, scheme='C0', temporal_set='L0',
                             inner_oof_R2=old.inner_oof_R2, inner_oof_RMSE=old.inner_oof_RMSE,
                             inner_oof_MAE=old.inner_oof_MAE, n_inner=int(old.n_inner),
                             inner_source='prior_locked_global',
                             selection_rule='no promising new candidate; retain prior inner-OOF lock'))
            continue
        reference_n = int(baseline.n.iloc[0])
        candidates = candidates.loc[candidates.n.eq(reference_n)]
        best = candidates.sort_values(['R2','RMSE','MAE'], ascending=[False,True,True]).iloc[0]
        if best.scheme == 'C0' and best.temporal_set == 'L0':
            old = prior.loc[task]
            rows.append(dict(task=task, scheme='C0', temporal_set='L0',
                             inner_oof_R2=old.inner_oof_R2, inner_oof_RMSE=old.inner_oof_RMSE,
                             inner_oof_MAE=old.inner_oof_MAE, n_inner=int(old.n_inner),
                             inner_source='prior_locked_global',
                             selection_rule='new candidates failed same-batch fivefold; retain prior lock'))
            continue
        rows.append(dict(task=task, scheme=best.scheme, temporal_set=best.temporal_set,
                         inner_oof_R2=best.R2, inner_oof_RMSE=best.RMSE,
                         inner_oof_MAE=best.MAE, n_inner=reference_n,
                         inner_source='current_fixed_batch_fivefold',
                         selection_rule='max same-batch fivefold pooled OOF R2; RMSE/MAE tie break'))
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    (OUT / '08_final_task_configuration.sha256').write_text(
        hashlib.sha256(path.read_bytes()).hexdigest()+'  08_final_task_configuration.csv\n', encoding='utf-8')
    print(pd.DataFrame(rows)[['task','scheme','temporal_set','inner_oof_R2']].to_string(index=False))


def summarize_final() -> None:
    lock = OUT / '08_final_task_configuration.csv'
    expected = (OUT / '08_final_task_configuration.sha256').read_text(encoding='utf-8').split()[0]
    if hashlib.sha256(lock.read_bytes()).hexdigest() != expected:
        raise ValueError('Final configuration SHA256 mismatch')
    decisions = pd.read_csv(lock).set_index('task')
    frame = read_parts('09_final_test_predictions_parts')
    if not frame.empty:
        frame['prediction_source'] = 'current_fixed_batch_run'
    frames = [frame] if not frame.empty else []
    for task in TASKS:
        if task in set(frame.task):
            continue
        decision = decisions.loc[task]
        if decision.inner_source != 'prior_locked_global':
            raise ValueError(f'{task}: selected new scheme has no final test prediction')
        prior_path = OLD / '08_final_test_predictions' / f'{task}_tabicl_adapted.csv'
        old_prediction = pd.read_csv(prior_path, dtype={'obs_id': str})
        frames.append(old_prediction[['task','obs_id','y_true','y_pred']].assign(
            fold=0, scheme='C0', temporal_set='L0', prediction_source='prior_locked_external_test'))
    frame = pd.concat(frames, ignore_index=True)
    if frame.task.nunique() != 10: raise ValueError('Ten final test tasks not completed')
    split = pd.read_csv(ROOT / 'runs/random_10task_v1/01_random_split_manifest.csv',
                        dtype={'obs_id': str})
    expected_ids = split.loc[split.split.eq('test'), ['task','obs_id']]
    actual_ids = frame[['task','obs_id']]
    if len(actual_ids) != len(expected_ids) or actual_ids.duplicated().any() or not (
        actual_ids.merge(expected_ids, on=['task','obs_id'], validate='one_to_one').shape[0] == len(expected_ids)):
        raise ValueError('External test obs_id mismatch against frozen split')
    frame.to_parquet(OUT / '09_final_test_predictions.parquet', index=False)
    old = pd.read_csv(OLD / '09_final_10task_comparison.csv').set_index('task')
    zoo = pd.read_csv(ZOO / '08_all_models_test_comparison.csv')
    zoo = zoo.loc[~zoo.family.eq('selected') & ~zoo.model.eq('tabicl')]
    rows = []
    for task, item in frame.groupby('task'):
        current = score(item)
        previous = old.loc[task]
        reference_path = OUT / 'external_same_batch_references' / f'{task}_C0_L0_fold0.csv'
        if reference_path.exists():
            reference = pd.read_csv(reference_path, dtype={'obs_id': str})
            if len(reference) != len(item) or set(reference.obs_id) != set(item.obs_id):
                raise ValueError(f'{task}: same-batch external reference IDs differ')
            reference_score = score(reference)
        else:
            reference_score = None
        competitors = zoo.loc[zoo.task.eq(task)]
        rank = int((competitors.R2 > current['R2']).sum())+1
        rows.append(dict(task=task, scheme=item.scheme.iloc[0], temporal_set=item.temporal_set.iloc[0],
                         prediction_source=item.prediction_source.iloc[0] if 'prediction_source' in item else 'current_run',
                         old_R2=previous.new_R2, new_R2=current['R2'], delta_R2=current['R2']-previous.new_R2,
                         old_RMSE=previous.new_RMSE, new_RMSE=current['RMSE'],
                         delta_RMSE=current['RMSE']-previous.new_RMSE,
                         old_MAE=previous.new_MAE, new_MAE=current['MAE'],
                         delta_MAE=current['MAE']-previous.new_MAE,
                         same_batch_C0_R2=reference_score['R2'] if reference_score else np.nan,
                         delta_R2_vs_same_batch_C0=current['R2']-reference_score['R2'] if reference_score else np.nan,
                         same_batch_C0_RMSE=reference_score['RMSE'] if reference_score else np.nan,
                         delta_RMSE_vs_same_batch_C0=current['RMSE']-reference_score['RMSE'] if reference_score else np.nan,
                         same_batch_C0_MAE=reference_score['MAE'] if reference_score else np.nan,
                         delta_MAE_vs_same_batch_C0=current['MAE']-reference_score['MAE'] if reference_score else np.nan,
                         prior_single_model_rank=int(previous.single_model_rank),
                         single_model_rank=rank, R2_at_least_0_5=bool(current['R2'] >= .5),
                         n_test=len(item)))
    summary = pd.DataFrame(rows).set_index('task').loc[TASKS].reset_index()
    summary.to_csv(OUT / '10_final_10task_comparison.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))


def bootstrap_selected() -> None:
    """Resample paired inner-OOF predictions by station; no model reruns."""
    decisions = pd.read_csv(OUT / '08_final_task_configuration.csv')
    predictions = read_parts('03_context_oof_parts')
    stations = pd.read_csv(ROOT / '数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv',
                           usecols=['obs_id','station'], dtype={'obs_id': str})
    rows = []
    for decision in decisions.itertuples():
        if decision.inner_source != 'current_fixed_batch_fivefold' or (
            decision.scheme == 'C0' and decision.temporal_set == 'L0'):
            continue
        subset = predictions.loc[predictions.task.eq(decision.task)]
        candidate = subset.loc[subset.scheme.eq(decision.scheme) &
                               subset.temporal_set.eq(decision.temporal_set)]
        control = subset.loc[subset.scheme.eq('C0') & subset.temporal_set.eq('L0')]
        joined = candidate[['obs_id','y_true','y_pred']].merge(
            control[['obs_id','y_pred']].rename(columns={'y_pred':'control_pred'}),
            on='obs_id', validate='one_to_one').merge(stations, on='obs_id', validate='many_to_one')
        if len(joined) != len(candidate) or len(joined) != len(control):
            raise ValueError(f'{decision.task}: paired OOF observations differ')
        groups = [item[['y_true','y_pred','control_pred']].to_numpy(float)
                  for _, item in joined.groupby('station')]
        n_stations = len(groups)
        def differences(array):
            truth, pred, baseline = array.T
            return np.array([r2_score(truth, pred)-r2_score(truth, baseline),
                np.sqrt(np.mean((pred-truth)**2))-np.sqrt(np.mean((baseline-truth)**2)),
                np.mean(np.abs(pred-truth))-np.mean(np.abs(baseline-truth))])
        observed = differences(np.concatenate(groups))
        rng = np.random.default_rng(42)
        draws = np.stack([differences(np.concatenate([groups[i] for i in rng.integers(
            0, n_stations, size=n_stations)])) for _ in range(1000)])
        for index, metric in enumerate(['delta_R2','delta_RMSE','delta_MAE']):
            low, high = np.quantile(draws[:, index], [.025,.975])
            rows.append(dict(task=decision.task, scheme=decision.scheme,
                temporal_set=decision.temporal_set, metric=metric, point=float(observed[index]),
                ci95_low=float(low), ci95_high=float(high), n_stations=n_stations,
                resamples=1000, ci_crosses_zero=bool(low <= 0 <= high)))
    pd.DataFrame(rows, columns=['task','scheme','temporal_set','metric','point','ci95_low',
        'ci95_high','n_stations','resamples','ci_crosses_zero']).to_csv(
        OUT / '11_station_bootstrap.csv', index=False, encoding='utf-8-sig')
    print(f'Bootstrap rows: {len(rows)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['inner','lock','final','bootstrap'], required=True)
    action = parser.parse_args().action
    if action == 'inner': summarize_inner()
    elif action == 'lock': lock_configs()
    elif action == 'final': summarize_final()
    else: bootstrap_selected()


if __name__ == '__main__':
    main()
