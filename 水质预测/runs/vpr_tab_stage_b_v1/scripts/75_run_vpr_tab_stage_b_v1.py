"""VPR-Tab Stage B: nested station-fold residual enhancement; isolated outputs."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / '文档/后续重构计划.md').exists()
            and (p / '结果/扩展实验/候选数据验收与一致性重算_v1_20261115').exists())
BASE = ROOT / '结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
FROZEN = BASE / '02_冻结候选'
FORMAL = BASE / '03_正式十任务重算'
OUT = ROOT / 'runs/vpr_tab_stage_b_v1'
DEPTHS = ('surface', 'middle', 'bottom')
FOLDS = tuple(f'fold_{i}' for i in range(1, 6))
LGB_PARAMS = dict(objective='regression', n_estimators=300, learning_rate=.03,
                  num_leaves=15, max_depth=-1, min_child_samples=30,
                  subsample=.8, subsample_freq=1, colsample_bytree=.8,
                  reg_lambda=2., random_state=42, verbosity=-1, n_jobs=4)


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setup():
    for name in ('scripts', 'inner_predictions', 'fold_predictions'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    formal = import_file('vpr_formal_preprocessing', ROOT / '脚本/04_模型训练/51_recompute_accepted_candidate_v1.py')
    data = pd.read_csv(FROZEN / '基础输入_77特征.csv', parse_dates=['date'], low_memory=False)
    daily = pd.read_csv(FROZEN / 'ERA5逐日气候态来源.csv', parse_dates=['date']).sort_values('date')
    features = json.loads((FROZEN / '特征与时间协议.json').read_text(encoding='utf-8'))['feature_order']
    assert len(features) == 77 and not data.obs_id.duplicated().any()
    config = json.loads((FORMAL / '模型固定配置.json').read_text(encoding='utf-8'))
    assert config['dataset_version'] == 'WQ77_REPAIRED_20261115_V1'
    return formal, data, daily, features, config


def load_outer(indicator: str, model: str):
    blocks = []
    for depth in DEPTHS:
        task = f'{indicator}_{depth}'
        for fold in FOLDS:
            p = FORMAL / 'predictions' / f'{task}_{model}_{fold}.csv'
            z = pd.read_csv(p)
            assert z.dataset_version.eq('WQ77_REPAIRED_20261115_V1').all()
            assert z.task.eq(task).all() and z.fold.eq(fold).all()
            assert np.isfinite(z.y_pred).all()
            z = z[['obs_id', 'station_id', 'fold', 'y_true', 'y_pred']].copy()
            z['sample_id'] = z.obs_id.str.rsplit('_', n=1).str[0]
            z['depth'] = depth
            blocks.append(z)
    x = pd.concat(blocks, ignore_index=True)
    assert not x.duplicated(['obs_id']).any()
    assert x.groupby('sample_id').fold.nunique().max() == 1
    assert x.groupby('station_id').fold.nunique().max() == 1
    return x


def fit_predict(model: str, xtrain: np.ndarray, ytrain: np.ndarray,
                xvalid: np.ndarray, config: dict):
    nonnegative = float(np.min(ytrain)) >= 0
    target = np.log1p(ytrain) if nonnegative else ytrain.copy()
    if model == 'xgboost':
        from xgboost import XGBRegressor
        params = dict(config['model_parameters'], random_state=42, objective='reg:squarederror')
        reg = XGBRegressor(**params)
    elif model == 'tabiclv2':
        import importlib.metadata
        import torch
        assert torch.cuda.is_available() and importlib.metadata.version('tabicl') == '2.2.0'
        module = import_file('vpr_tabicl_adapter', ROOT / 'models/tabiclv2_adapter.py')
        reg = module.TabICLv2Regressor(random_state=42)
    else:
        raise ValueError(model)
    reg.fit(xtrain, target)
    pred = np.asarray(reg.predict(xvalid), dtype=float).reshape(-1)
    pred = np.maximum(np.expm1(pred) if nonnegative else pred, 0)
    assert len(pred) == len(xvalid) and np.isfinite(pred).all()
    del reg
    gc.collect()
    if model == 'tabiclv2':
        import torch
        torch.cuda.empty_cache()
    return pred


def inner_prediction(indicator, model, outer, inner, depth, formal, data, daily, features, config, fold_map):
    path = OUT / 'inner_predictions' / f'{indicator}_{model}_{outer}_{inner}_{depth}.parquet'
    if path.exists():
        z = pd.read_parquet(path)
        assert z.fold.eq(inner).all() and z.outer_fold.eq(outer).all()
        return z
    target = {'tn': 'tn_mg_l', 'tp': 'tp_mg_l'}[indicator]
    work = data[data.depth.eq(depth) & data[target].notna() & data.station.isin(fold_map)].copy()
    work['fold'] = work.station.map(fold_map)
    assert work.fold.notna().all()
    train = work[~work.fold.isin((outer, inner))].copy()
    valid = work[work.fold.eq(inner)].copy()
    assert len(train) and len(valid) and not set(train.station) & set(valid.station)
    xt, xv, _, _, _, _, _ = formal.feature_arrays(data, train, valid, features, daily)
    pred = fit_predict(model, xt, train[target].to_numpy(float), xv, config)
    z = pd.DataFrame({'obs_id': valid.obs_id.to_numpy(str), 'sample_id': valid.sample_id.to_numpy(str),
                      'station_id': valid.station.to_numpy(str), 'fold': inner, 'outer_fold': outer,
                      'depth': depth, 'y_true': valid[target].to_numpy(float), 'y_pred': pred})
    z.to_parquet(path, index=False)
    print('INNER_OK', indicator, model, outer, inner, depth, len(z), flush=True)
    return z


def pivot_complete(long: pd.DataFrame):
    cols = ['sample_id', 'station_id', 'fold', 'y_true', 'y_pred', 'depth']
    assert not long.duplicated(['sample_id', 'depth']).any()
    wide = long[cols].pivot(index='sample_id', columns='depth', values=['y_true', 'y_pred'])
    wide = wide.dropna(subset=[('y_pred', d) for d in DEPTHS] + [('y_true', 'middle')]).copy()
    result = pd.DataFrame(index=wide.index)
    result['y_true'] = wide[('y_true', 'middle')]
    result['p_surface'] = wide[('y_pred', 'surface')]
    result['p_middle'] = wide[('y_pred', 'middle')]
    result['p_bottom'] = wide[('y_pred', 'bottom')]
    result['d_sm'] = result.p_surface - result.p_middle
    result['d_mb'] = result.p_middle - result.p_bottom
    result['curvature'] = result.p_surface - 2 * result.p_middle + result.p_bottom
    middle = long[long.depth.eq('middle')].set_index('sample_id')
    result['obs_id'] = middle.reindex(result.index).obs_id
    result['station_id'] = middle.reindex(result.index).station_id
    result['fold'] = middle.reindex(result.index).fold
    assert result[['obs_id', 'station_id', 'fold']].notna().all().all()
    return result.reset_index()


def residual_features(frame: pd.DataFrame, outer: str, indicator: str,
                      features: list[str], is_train: bool):
    cache = np.load(FORMAL / 'inputs' / f'{indicator}_middle_{outer}.npz')
    ids = cache['train_ids' if is_train else 'test_ids'].astype(str)
    matrix = cache['x_train' if is_train else 'x_test']
    assert matrix.shape[1] == len(features)
    loc = pd.Index(ids).get_indexer(frame.obs_id.astype(str))
    assert (loc >= 0).all(), 'middle observations absent from frozen outer input'
    return pd.DataFrame(matrix[loc], columns=features, index=frame.index)


def score(y, p):
    return dict(R2=float(r2_score(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)))


def run_branch(indicator, model, formal, data, daily, features, config):
    outer_all = load_outer(indicator, model)
    station_fold = outer_all[['station_id', 'fold']].drop_duplicates().set_index('station_id').fold.to_dict()
    assert len(station_fold) == 84
    for outer in FOLDS:
        final = OUT / 'fold_predictions' / f'{indicator}_{model}_{outer}.parquet'
        if final.exists():
            print('OUTER_ALREADY_DONE', indicator, model, outer, flush=True)
            continue
        start = time.perf_counter()
        inner_frames = []
        for inner in FOLDS:
            if inner == outer:
                continue
            for depth in DEPTHS:
                inner_frames.append(inner_prediction(indicator, model, outer, inner, depth,
                                                     formal, data, daily, features, config, station_fold))
        train = pivot_complete(pd.concat(inner_frames, ignore_index=True))
        test = pivot_complete(outer_all[outer_all.fold.eq(outer)])
        assert not set(train.station_id) & set(test.station_id)
        assert train.sample_id.is_unique and test.sample_id.is_unique
        xtr = residual_features(train, outer, indicator, features, True)
        xte = residual_features(test, outer, indicator, features, False)
        b1 = ['p_middle']
        b2 = ['p_middle', 'p_surface', 'p_bottom', 'd_sm', 'd_mb', 'curvature']
        residual = train.y_true.to_numpy() - train.p_middle.to_numpy()
        result = test[['sample_id', 'obs_id', 'station_id', 'fold', 'y_true']].copy()
        result['indicator'] = indicator
        result['base_model'] = model
        result['B0'] = test.p_middle.to_numpy()
        for name, extras in [('B1', b1), ('B2', b2)]:
            train_x = pd.concat([xtr, train[extras].reset_index(drop=True)], axis=1)
            test_x = pd.concat([xte, test[extras].reset_index(drop=True)], axis=1)
            assert train_x.columns.tolist() == test_x.columns.tolist()
            learner = LGBMRegressor(**LGB_PARAMS)
            learner.fit(train_x, residual)
            correction = np.asarray(learner.predict(test_x), dtype=float)
            result[name] = result.B0.to_numpy() + correction
        assert len(result) == len(test) and result[['B0', 'B1', 'B2']].notna().all().all()
        result.to_parquet(final, index=False)
        print('OUTER_OK', indicator, model, outer, 'train', len(train), 'test', len(test),
              'seconds', round(time.perf_counter() - start, 1), flush=True)


def summarize():
    frames = [pd.read_parquet(p) for p in sorted((OUT / 'fold_predictions').glob('*.parquet'))]
    if not frames:
        return
    z = pd.concat(frames, ignore_index=True)
    assert not z.duplicated(['indicator', 'base_model', 'obs_id']).any()
    z.to_parquet(OUT / 'stage_b_oof_predictions.parquet', index=False)
    rows = []
    for (ind, model), g in z.groupby(['indicator', 'base_model']):
        for name in ['B0', 'B1', 'B2']:
            for fold, part in [('pooled', g), *list(g.groupby('fold'))]:
                rows.append(dict(indicator=ind, base_model=model, version=name, fold=fold,
                                 n_events=len(part), n_stations=part.station_id.nunique(),
                                 **score(part.y_true, part[name])))
        if g.fold.nunique() == 5:
            for name in ['B0', 'B1', 'B2']:
                vals=[score(part.y_true,part[name])['R2'] for _,part in g.groupby('fold')]
                rows.append(dict(indicator=ind,base_model=model,version=name,fold='fold_mean',
                                 n_events=len(g),n_stations=g.station_id.nunique(),R2=float(np.mean(vals)),RMSE=np.nan,MAE=np.nan))
    m = pd.DataFrame(rows)
    wide = m.pivot(index=['indicator', 'base_model', 'fold'], columns='version',
                   values=['R2', 'RMSE', 'MAE', 'n_events', 'n_stations'])
    wide.columns = [f'{version}_{metric}' for metric, version in wide.columns]
    wide = wide.reset_index()
    wide['B1_minus_B0_R2'] = wide.B1_R2 - wide.B0_R2
    wide['B2_minus_B1_R2'] = wide.B2_R2 - wide.B1_R2
    wide['B2_minus_B0_R2'] = wide.B2_R2 - wide.B0_R2
    wide.to_csv(OUT / 'stage_b_metrics.csv', index=False, encoding='utf-8-sig')
    return wide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('xgboost', 'tabiclv2', 'all'), default='all')
    parser.add_argument('--indicator', choices=('tn', 'tp', 'all'), default='all')
    args = parser.parse_args()
    formal, data, daily, features, config = setup()
    for ind in (('tn', 'tp') if args.indicator == 'all' else (args.indicator,)):
        for model in (('xgboost', 'tabiclv2') if args.model == 'all' else (args.model,)):
            try:
                run_branch(ind, model, formal, data, daily, features, config)
            finally:
                summarize()


if __name__ == '__main__':
    main()
