"""Fixed-query TabICLv2 context experiments on frozen Random Strict folds.

Only train-fold features/labels determine preprocessors, clusters, context IDs,
target transforms and feature selection. Validation/test targets score outputs only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT / '脚本/04_模型训练/67_run_tabicl_water_quality_inner.py'
spec = importlib.util.spec_from_file_location('tabicl_base_adaptation', BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from models.tabiclv2_adapter import TabICLv2Regressor

OUT = ROOT / 'runs/tabicl_context_temporal_adaptation_v1'
OLD = ROOT / 'runs/tabicl_water_quality_adaptation_v1'
TAXONOMY = ROOT / 'runs/multisource_ablation_v1/00_feature_source_taxonomy.csv'
TEMPORAL = OUT / 'temporal_features_by_obs.csv'
BATCH_SIZE = 64
SCHEMES = ('C0', 'C0_32', 'C1_512', 'C1_1024', 'C2', 'C3', 'C4')
TEMPORAL_SETS = ('L0', 'L1', 'L2', 'L12')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_frozen() -> None:
    expected = {
        ROOT / 'runs/random_10task_v1/01_random_split_manifest.csv': 'bb9269039de23f8fb4412eb71bf8268a972ec40ca17c3d0b333a6e2fa791b359',
        ROOT / 'runs/random_tabular_model_zoo_v2/01_inner_fold_manifest.csv': 'c4fea605966709a29baf014ed7eb6c8c557d06122728017ea127ef9ae21213cc',
    }
    for path, value in expected.items():
        if sha(path) != value:
            raise ValueError(f'Frozen split hash mismatch: {path}')
    selected = OLD / '10_task_best_tabicl_config.csv'
    lock = (OLD / '10_task_best_tabicl_config.sha256').read_text(encoding='utf-8').split()[0]
    if sha(selected) != lock or len(base.FEATURES) != 77 or 'water_zone_code' in base.FEATURES:
        raise ValueError('Previous locked settings or 77-feature manifest changed')


def load_data(task: str, fold: int | None, temporal_set: str):
    if fold is not None:
        ids, x, y = base.load(task)
        ids = ids.copy()
        train = ids.inner_fold.to_numpy() != fold
        query = ~train
    else:
        target, depth = base.TASKS[task]
        frame = pd.read_csv(base.TABLE, usecols=['obs_id', 'depth', target, *base.FEATURES], low_memory=False)
        frame = frame.loc[frame.depth.eq(depth) & frame[target].notna()].copy()
        frame.obs_id = frame.obs_id.astype(str)
        split = pd.read_csv(ROOT / 'runs/random_10task_v1/01_random_split_manifest.csv', dtype={'obs_id': str})
        split = split.loc[split.task.eq(task), ['obs_id', 'split']]
        frame = frame.merge(split, on='obs_id', validate='one_to_one').sort_values('obs_id').reset_index(drop=True)
        if len(frame) != len(split):
            raise ValueError('Outer split coverage mismatch')
        ids = frame[['obs_id', 'split']]
        x = frame[base.FEATURES].apply(pd.to_numeric, errors='coerce')
        y = frame[target].to_numpy(float)
        train = ids.split.eq('train').to_numpy()
        query = ~train
    if temporal_set != 'L0':
        t = pd.read_csv(TEMPORAL, dtype={'obs_id': str})
        t.obs_id = t.obs_id.astype(str)
        names = list(pd.read_csv(OUT / '04_temporal_feature_manifest.csv').feature)
        l1 = [n for n in names if n.startswith(('ssrd_', 'sst_')) and n.endswith('prior2d')]
        l2 = [n for n in names if n.startswith(('ssrd_', 'sst_')) and ('prior30d' in n or 'prior90d' in n)]
        chosen = l1 if temporal_set == 'L1' else l2 if temporal_set == 'L2' else l1 + l2
        joined = ids[['obs_id']].merge(t[['obs_id', *chosen]], on='obs_id', validate='one_to_one')
        if len(joined) != len(ids) or not np.array_equal(joined.obs_id, ids.obs_id):
            raise ValueError('Temporal obs_id alignment mismatch')
        x = pd.concat([x.reset_index(drop=True), joined[chosen].reset_index(drop=True)], axis=1)
    return ids, x, y, train, query


def model_arrays(x: pd.DataFrame, y: np.ndarray, train: np.ndarray, config: str, temporal_set: str):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    xa = imputer.fit_transform(x.iloc[train]).astype(np.float32)
    xb = imputer.transform(x.iloc[~train]).astype(np.float32)
    if config == 'F60':
        scores = mutual_info_regression(xa[:, :77], y[train], random_state=42)
        keep = np.sort(np.argsort(-scores, kind='stable')[:60])
        if temporal_set != 'L0': keep = np.r_[keep, np.arange(77, xa.shape[1])]
        xa, xb = xa[:, keep], xb[:, keep]
    transform = 'raw' if config == 'RAW' else 'log1p'
    if transform == 'raw':
        target, inverse = y[train], lambda a: a
    else:
        if np.min(y[train]) < 0: raise ValueError('Negative target under log1p')
        target, inverse = np.log1p(y[train]), np.expm1
    kwargs = base.CONFIGS[config].copy()
    kwargs.pop('target_transform', None)
    kwargs.pop('top_features', None)
    output_type = kwargs.pop('output_type', 'mean')
    return xa, xb, target, inverse, kwargs, output_type


def distance_arrays(x: pd.DataFrame, train: np.ndarray):
    # Distance always uses the same original 77 environmental inputs. The
    # locked model may use F60 or temporal additions; context IDs remain fair.
    taxonomy = pd.read_csv(TAXONOMY).set_index('feature_name').source_group
    if set(base.FEATURES) - set(taxonomy.index):
        raise ValueError('Frozen source groups do not cover 77 features')
    categories = ['quarter', 'is_wet_season']
    continuous = [f for f in base.FEATURES if f not in categories]
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = RobustScaler(quantile_range=(25, 75))
    a = scaler.fit_transform(imputer.fit_transform(x.loc[train, continuous])).astype(np.float32)
    b = scaler.transform(imputer.transform(x.loc[~train, continuous])).astype(np.float32)
    a = np.clip(a, -5, 5)
    b = np.clip(b, -5, 5)
    group_columns = {g: [i for i, feature in enumerate(continuous)
                         if taxonomy.loc[feature] == g] for g in taxonomy.unique()}
    # One-hot category distance avoids treating arbitrary category codes as
    # ordered magnitudes. Empty categories are not invented.
    for feature in categories:
        values = x[feature].fillna(-999).astype(str)
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ca = encoder.fit_transform(values.loc[train].to_numpy().reshape(-1, 1)).astype(np.float32)
        cb = encoder.transform(values.loc[~train].to_numpy().reshape(-1, 1)).astype(np.float32)
        start = a.shape[1]
        a, b = np.column_stack((a, ca)), np.column_stack((b, cb))
        group_columns[taxonomy.loc[feature]].extend(range(start, a.shape[1]))
    return a, b, group_columns


def context_indices(scheme: str, query: np.ndarray, train_x: np.ndarray, train_clusters: np.ndarray,
                    query_cluster: int, balanced: dict[str, list[int]], global_order: np.ndarray):
    if scheme in ('C0', 'C0_32'): return np.arange(len(train_x))
    if scheme == 'C2':
        distance = cdist(query, train_x, metric='sqeuclidean').mean(axis=0)
        inside = np.flatnonzero(train_clusters == query_cluster)
        order = inside[np.argsort(distance[inside], kind='stable')]
        if len(order) < 1024:
            other = np.flatnonzero(train_clusters != query_cluster)
            order = np.r_[order, other[np.argsort(distance[other], kind='stable')]]
        return order[:1024]
    if scheme == 'C3':
        parts = []
        for columns in balanced.values():
            if columns:
                parts.append(cdist(query[:, columns], train_x[:, columns], metric='sqeuclidean').mean(axis=0) / len(columns))
        distance = np.mean(parts, axis=0)
        return np.argsort(distance, kind='stable')[:1024]
    if scheme == 'C4':
        distance = cdist(query, train_x, metric='sqeuclidean').mean(axis=0)
        local = np.argsort(distance, kind='stable')[:512]
        global_part = global_order[~np.isin(global_order, local)][:512]
        return np.r_[local, global_part]
    if scheme.startswith('C1_') or scheme == 'C0_32':
        count = int(scheme.split('_')[1])
        distance = cdist(query, train_x, metric='sqeuclidean').mean(axis=0)
        return np.argsort(distance, kind='stable')[:count]
    raise ValueError(scheme)


def predict_context(task: str, fold: int | None, scheme: str, temporal_set: str):
    assert_frozen()
    selected = pd.read_csv(OLD / '10_task_best_tabicl_config.csv').set_index('task').loc[task, 'selected_config']
    ids, x, y, train, query = load_data(task, fold, temporal_set)
    xa, xb, yt, inverse, kwargs, output_type = model_arrays(x, y, train, selected, temporal_set)
    da, db, groups = distance_arrays(x.iloc[:, :77], train)
    t0 = time.perf_counter()
    km = KMeans(n_clusters=8, random_state=42, n_init=5, max_iter=100)
    train_clusters = km.fit_predict(da)
    query_clusters = km.predict(db)
    # Round-robin nearest-centroid representatives: deterministic and independent of y.
    representative_lists = []
    for cluster in range(8):
        members = np.flatnonzero(train_clusters == cluster)
        distance = cdist(da[members], km.cluster_centers_[cluster:cluster+1], metric='sqeuclidean').ravel()
        representative_lists.append(members[np.argsort(distance, kind='stable')])
    global_order = []
    for rank in range(max(map(len, representative_lists))):
        global_order.extend(int(a[rank]) for a in representative_lists if rank < len(a))
    global_order = np.asarray(global_order, dtype=int)
    prep_seconds = time.perf_counter() - t0
    query_ids = ids.loc[query, 'obs_id'].to_numpy()
    order = []
    for cluster in range(8):
        members = np.flatnonzero(query_clusters == cluster)
        members = members[np.argsort(query_ids[members], kind='stable')]
        order.extend((cluster, members[i:i+BATCH_SIZE]) for i in range(0, len(members), BATCH_SIZE))
    if scheme.startswith('C1_') or scheme == 'C0_32':
        chosen = np.linspace(0, len(query_ids)-1, min(32, len(query_ids)), dtype=int)
        order = [(int(query_clusters[i]), np.asarray([i])) for i in chosen]
    import torch
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    output = np.full(len(query_ids), np.nan)
    fit_seconds = predict_seconds = 0.0
    model = None
    if scheme in ('C0', 'C0_32'):
        model = TabICLv2Regressor(**kwargs)
        tick = time.perf_counter(); model.fit(xa, yt); fit_seconds += time.perf_counter()-tick
    for cluster, batch in order:
        if scheme not in ('C0', 'C0_32'):
            context = context_indices(scheme, db[batch], da, train_clusters, cluster, groups, global_order)
            model = TabICLv2Regressor(**kwargs)
            tick = time.perf_counter(); model.fit(xa[context], yt[context]); fit_seconds += time.perf_counter()-tick
        tick = time.perf_counter()
        raw = model.model.predict(xb[batch], output_type=output_type)
        predict_seconds += time.perf_counter()-tick
        output[batch] = np.maximum(inverse(np.asarray(raw).reshape(-1)), 0)
        if scheme not in ('C0', 'C0_32'):
            del model
            model = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    if model is not None: del model
    if scheme.startswith('C1_') or scheme == 'C0_32':
        selected_positions = np.flatnonzero(np.isfinite(output))
    else:
        selected_positions = np.arange(len(output))
    if not np.isfinite(output[selected_positions]).all():
        raise ValueError('Missing or invalid query predictions')
    predictions = pd.DataFrame(dict(task=task, fold=fold if fold is not None else 0,
                                    scheme=scheme, temporal_set=temporal_set,
                                    obs_id=query_ids[selected_positions],
                                    y_true=y[query][selected_positions],
                                    y_pred=output[selected_positions]))
    metric = dict(task=task, fold=fold if fold is not None else 0,
                  scheme=scheme, temporal_set=temporal_set, config_id=selected,
                  n_train=int(train.sum()), n_query=len(predictions),
                  n_query_batches=len(order), query_batch_size=BATCH_SIZE,
                  prep_seconds=prep_seconds, fit_seconds=fit_seconds,
                  predict_seconds=predict_seconds,
                  gpu_peak_allocated_mb=float(torch.cuda.max_memory_allocated()/2**20) if torch.cuda.is_available() else None,
                  **base.score(predictions.y_true, predictions.y_pred))
    return predictions, metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=base.TASKS, required=True)
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    parser.add_argument('--scheme', choices=SCHEMES, required=True)
    parser.add_argument('--temporal-set', choices=TEMPORAL_SETS, default='L0')
    parser.add_argument('--reference-only', action='store_true',
                        help='Run same-batch C0/L0 external reference; never enter final selection')
    args = parser.parse_args()
    if args.reference_only and (args.fold is not None or args.scheme != 'C0' or args.temporal_set != 'L0'):
        raise ValueError('Reference-only mode requires external C0/L0')
    if args.fold is None:
        lock = OUT / '08_final_task_configuration.csv'
        if not lock.exists(): raise FileNotFoundError('Final configuration not locked')
        choices = pd.read_csv(lock).set_index('task')
        row = choices.loc[args.task]
        if args.reference_only and row.inner_source != 'current_fixed_batch_fivefold':
            raise ValueError('Reference-only mode is limited to newly selected tasks')
        if not args.reference_only and (args.scheme, args.temporal_set) != (row.scheme, row.temporal_set):
            raise ValueError('Final evaluation may only use locked context/temporal settings')
    predictions, metric = predict_context(args.task, args.fold, args.scheme, args.temporal_set)
    directory = OUT / ('external_same_batch_references' if args.reference_only else
                       '09_final_test_predictions_parts' if args.fold is None else
                       '03_context_oof_parts')
    directory.mkdir(parents=True, exist_ok=True)
    stem = f'{args.task}_{args.scheme}_{args.temporal_set}_fold{args.fold or 0}'
    path = directory / f'{stem}.csv'
    if path.exists(): raise FileExistsError(path)
    predictions.to_csv(path, index=False, encoding='utf-8-sig')
    path.with_suffix('.json').write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metric, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
