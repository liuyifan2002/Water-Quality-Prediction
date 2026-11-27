"""Run one registered TabICLv2 adaptation candidate on frozen Random inner folds.

This script reads external-training rows only. It never reads external-test targets.
All imputers, feature selectors and target transforms fit within each inner fold.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.tabiclv2_adapter import TabICLv2Regressor

ZOO = ROOT / 'runs/random_tabular_model_zoo_v2'
OUT = ROOT / 'runs/tabicl_water_quality_adaptation_v1'
TABLE = ROOT / '数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv'
FEATURE_MANIFEST = ROOT / 'runs/tabiclv2_pilot/01_strict_feature_manifest.csv'
FEATURES = pd.read_csv(FEATURE_MANIFEST).query('included_in_pilot == True').feature.tolist()
TASKS = {
    'chla_surface': ('chla_ug_l', 'surface'), 'chla_middle': ('chla_ug_l', 'middle'),
    'chla_bottom': ('chla_ug_l', 'bottom'), 'secchi': ('secchi_depth_m', 'surface'),
    'tn_surface': ('tn_mg_l', 'surface'), 'tn_middle': ('tn_mg_l', 'middle'),
    'tn_bottom': ('tn_mg_l', 'bottom'), 'tp_surface': ('tp_mg_l', 'surface'),
    'tp_middle': ('tp_mg_l', 'middle'), 'tp_bottom': ('tp_mg_l', 'bottom'),
}
CONFIGS = {
    'Z0': {},
    'Z2': {'norm_methods': ['none', 'power', 'robust']},
    'Z3': {'norm_methods': ['none', 'power', 'quantile']},
    'Z4': {'norm_methods': 'robust'},
    'Z5': {'norm_methods': 'quantile'},
    'P_shift': {'feat_shuffle_method': 'shift'},
    'P_random': {'feat_shuffle_method': 'random'},
    'P_none': {'feat_shuffle_method': 'none'},
    'N16': {'n_estimators': 16},
    'N24': {'n_estimators': 24},
    'O3': {'outlier_threshold': 3.0},
    'O6': {'outlier_threshold': 6.0},
    'MEDIAN': {'output_type': 'median'},
    'RAW': {'target_transform': 'raw'},
    'YEO': {'target_transform': 'yeojohnson'},
    'NOCLIP': {'nonnegative_clip': False},
    'F60': {'top_features': 60},
    'F40': {'top_features': 40},
    'I4': {'interactions': True},
}

INTERACTIONS = [
    ('sst_mean_7d', 'wind_speed'),
    ('sst_anom_7d', 'tidal_range'),
    ('rain_3d', 'tidal_range'),
    ('rain_14d', 'wind_speed'),
]


def load(task: str):
    target, depth = TASKS[task]
    frame = pd.read_csv(TABLE, usecols=['obs_id', 'depth', target, *FEATURES], low_memory=False)
    frame = frame.loc[frame.depth.eq(depth) & frame[target].notna()].copy()
    frame.obs_id = frame.obs_id.astype(str)
    fold = pd.read_csv(ZOO / '01_inner_fold_manifest.csv', dtype={'obs_id': str})
    fold = fold.loc[fold.task.eq(task), ['obs_id', 'inner_fold']]
    frame = frame.merge(fold, on='obs_id', how='inner', validate='one_to_one').sort_values('obs_id').reset_index(drop=True)
    if len(frame) != len(fold) or len(FEATURES) != 77 or 'water_zone_code' in FEATURES:
        raise ValueError('Frozen training IDs or 77-feature manifest mismatch')
    x = frame[FEATURES].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return frame[['obs_id', 'inner_fold']], x, frame[target].to_numpy(float)


def score(y, p):
    return {'R2': float(r2_score(y, p)), 'RMSE': float(mean_squared_error(y, p) ** .5),
            'MAE': float(mean_absolute_error(y, p))}


def run_fold(task: str, config_id: str, fold: int, ids, x, y):
    config = CONFIGS[config_id].copy()
    train = ids.inner_fold.to_numpy() != fold
    val = ~train
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    xa = imputer.fit_transform(x.iloc[train]).astype(np.float32)
    xb = imputer.transform(x.iloc[val]).astype(np.float32)
    feature_names = FEATURES.copy()
    if config.pop('interactions', False):
        xa = np.column_stack([xa, *(xa[:, FEATURES.index(a)] * xa[:, FEATURES.index(b)]
                                    for a, b in INTERACTIONS)]).astype(np.float32)
        xb = np.column_stack([xb, *(xb[:, FEATURES.index(a)] * xb[:, FEATURES.index(b)]
                                    for a, b in INTERACTIONS)]).astype(np.float32)
        feature_names.extend(f'{a}_x_{b}' for a, b in INTERACTIONS)
    if 'top_features' in config:
        k = config.pop('top_features')
        relevance = mutual_info_regression(xa, y[train], random_state=42)
        selected = np.sort(np.argsort(-relevance, kind='stable')[:k])
        xa, xb = xa[:, selected], xb[:, selected]
        feature_names = [FEATURES[i] for i in selected]
    target_transform = config.pop('target_transform', 'log1p')
    if target_transform == 'log1p':
        if np.min(y[train]) < 0:
            raise ValueError('log1p candidate requires nonnegative training targets')
        yt = np.log1p(y[train])
        inverse = np.expm1
    elif target_transform == 'raw':
        yt = y[train]
        inverse = lambda p: p
    elif target_transform == 'yeojohnson':
        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        yt = transformer.fit_transform(y[train].reshape(-1, 1)).ravel()
        inverse = lambda p: transformer.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()
    else:
        raise ValueError(target_transform)
    output_type = config.pop('output_type', 'mean')
    nonnegative_clip = config.pop('nonnegative_clip', True)
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = TabICLv2Regressor(**config)
    start = time.perf_counter()
    model.fit(xa, yt)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    raw = model.model.predict(xb, output_type=output_type)
    predict_seconds = time.perf_counter() - start
    prediction = inverse(np.asarray(raw).reshape(-1))
    if nonnegative_clip:
        prediction = np.maximum(prediction, 0)
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite prediction')
    pred = pd.DataFrame({'task': task, 'config_id': config_id, 'obs_id': ids.loc[val, 'obs_id'].to_numpy(),
                         'fold': fold, 'y_true': y[val], 'y_pred': prediction})
    meta = {'task': task, 'config_id': config_id, 'fold': fold, 'n_train': int(train.sum()),
            'n_val': int(val.sum()), 'n_features': len(feature_names),
            'feature_sha256': hashlib.sha256('\n'.join(feature_names).encode()).hexdigest(),
            'target_transform': target_transform, 'output_type': output_type,
            'nonnegative_clip': nonnegative_clip,
            'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
            'gpu_peak_allocated_mb': float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
            **score(y[val], prediction)}
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--config', choices=CONFIGS, required=True)
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=None)
    args = parser.parse_args()
    ids, x, y = load(args.task)
    directory = OUT / '07_inner_oof_predictions'
    directory.mkdir(parents=True, exist_ok=True)
    meta_dir = OUT / 'inner_fold_metrics'
    meta_dir.mkdir(parents=True, exist_ok=True)
    folds = [args.fold] if args.fold else range(1, 6)
    for fold in folds:
        path = directory / f'{args.task}_{args.config}_fold{fold}.csv'
        metric_path = meta_dir / f'{args.task}_{args.config}_fold{fold}.json'
        if path.exists() and metric_path.exists():
            print('SKIP', args.task, args.config, fold, flush=True)
            continue
        pred, metric = run_fold(args.task, args.config, fold, ids, x, y)
        pred.to_csv(path, index=False, encoding='utf-8-sig')
        metric_path.write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding='utf-8')
        print(args.task, args.config, fold, metric['R2'], metric['fit_seconds'], metric['predict_seconds'], flush=True)


if __name__ == '__main__':
    main()
