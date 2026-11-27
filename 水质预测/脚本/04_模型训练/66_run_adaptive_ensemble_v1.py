"""Frozen Random Strict OOF stacking; no base-model retraining."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
ZOO = ROOT / 'runs/random_tabular_model_zoo_v2'
RANDOM = ROOT / 'runs/random_10task_v1'
ADAPTED = ROOT / 'runs/tabicl_water_quality_adaptation_v1'
CONTEXT = ROOT / 'runs/tabicl_context_temporal_adaptation_v1'
OUT = ROOT / 'runs/adaptive_ensemble_v1'
MASTER = ROOT / '数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv'
FEATURE_MANIFEST = ROOT / 'runs/tabiclv2_pilot/01_strict_feature_manifest.csv'
TASKS = ['chla_surface', 'chla_middle', 'chla_bottom', 'secchi', 'tn_surface',
         'tn_middle', 'tn_bottom', 'tp_surface', 'tp_middle', 'tp_bottom']
SPLIT_SHA = 'bb9269039de23f8fb4412eb71bf8268a972ec40ca17c3d0b333a6e2fa791b359'
INNER_SHA = 'c4fea605966709a29baf014ed7eb6c8c557d06122728017ea127ef9ae21213cc'
M1_GRID = [0.0, 0.01, 0.1, 1.0]
M2_GRID = [1.0, 10.0, 100.0]
M3_GRID = [0.01, 0.1, 1.0]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y, p):
    return {'R2': float(r2_score(y, p)),
            'RMSE': float(mean_squared_error(y, p) ** .5),
            'MAE': float(mean_absolute_error(y, p))}


def fit_m1(p, y, lam):
    m = p.shape[1]
    equal = np.full(m, 1 / m)
    variance = max(float(np.var(y)), 1e-12)
    result = minimize(lambda w: np.mean((p @ w-y)**2)/variance + lam*np.sum((w-equal)**2),
                      equal, method='SLSQP', bounds=[(0, 1)] * m,
                      constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w)-1}],
                      options={'maxiter': 200, 'ftol': 1e-10})
    if not result.success:
        raise RuntimeError(f'M1 optimization failed: {result.message}')
    return result.x


def gate_features(x_train, x_other):
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    a = imp.fit_transform(x_train)
    b = imp.transform(x_other)
    scale = StandardScaler()
    a = np.clip(scale.fit_transform(a), -8, 8)
    b = np.clip(scale.transform(b), -8, 8)
    pca = PCA(n_components=6, svd_solver='randomized', random_state=42)
    a = pca.fit_transform(a)
    b = pca.transform(b)
    component_scale = np.maximum(np.std(a, axis=0), 1e-9)
    a = np.column_stack([np.ones(len(a)), np.clip(a/component_scale, -6, 6)])
    b = np.column_stack([np.ones(len(b)), np.clip(b/component_scale, -6, 6)])
    return a, b


def fit_m3(p, y, z, lam):
    n, m = p.shape
    variance = max(float(np.var(y)), 1e-12)
    def loss_gradient(flat):
        theta = flat.reshape(z.shape[1], m)
        weights = softmax(z @ theta, axis=1)
        pred = np.sum(weights*p, axis=1)
        residual = pred-y
        loss = np.mean(residual**2)/variance + lam*np.sum(theta**2)
        deriv = weights*(p-pred[:, None])
        gradient = 2*z.T @ (residual[:, None]*deriv)/(n*variance) + 2*lam*theta
        return loss, gradient.ravel()
    result = minimize(loss_gradient, np.zeros(z.shape[1]*m), jac=True,
                      method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
    if not result.success and result.nit == 0:
        raise RuntimeError(f'M3 optimization failed: {result.message}')
    return result.x.reshape(z.shape[1], m)


def fit_predict(method, lam, p_train, y_train, p_other, x_train=None, x_other=None):
    if method == 'M0':
        return p_other.mean(axis=1), np.full(p_train.shape[1], 1/p_train.shape[1]).tolist()
    if method == 'M1':
        w = fit_m1(p_train, y_train, lam)
        return p_other @ w, w.tolist()
    if method == 'M2':
        scaler = StandardScaler()
        a = scaler.fit_transform(p_train)
        b = scaler.transform(p_other)
        model = Ridge(alpha=lam)
        model.fit(a, y_train)
        return np.maximum(model.predict(b), 0), {'alpha': lam, 'coef': model.coef_.tolist()}
    if method == 'M3':
        z_train, z_other = gate_features(x_train, x_other)
        theta = fit_m3(p_train, y_train, z_train, lam)
        weights = softmax(z_other @ theta, axis=1)
        return np.maximum(np.sum(weights*p_other, axis=1), 0), {
            'lambda': lam, 'pca_components': 6, 'mean_test_weights': weights.mean(axis=0).tolist()}
    raise ValueError(method)


def load_task(task, split, inner, features, registry):
    tr_ids = split.loc[(split.task == task) & (split.split == 'train'), 'obs_id'].astype(str)
    te_ids = split.loc[(split.task == task) & (split.split == 'test'), 'obs_id'].astype(str)
    fold_map = inner.loc[inner.task == task, ['obs_id', 'inner_fold']].set_index('obs_id')
    if set(tr_ids) != set(fold_map.index) or set(tr_ids) & set(te_ids):
        raise ValueError(f'{task}: outer/inner split mismatch')
    inventory, oof_columns, test_columns = [], {}, {}
    base = None
    test_base = None
    names = sorted(p.stem[len(task)+1:] for p in (ZOO/'04_inner_oof_predictions').glob(f'{task}_*.csv'))
    for name in names:
        op = ZOO/'04_inner_oof_predictions'/f'{task}_{name}.csv'
        tp = ZOO/'07_final_predictions'/f'{task}_{name}.csv'
        row = registry.loc[(registry.task == task) & (registry.model == name)]
        config = str(row.config.iloc[0]) if len(row) else '未登记'
        status = '缺少外部预测' if not tp.exists() else '可用'
        if status == '可用':
            o = pd.read_csv(op, dtype={'obs_id': str}).set_index('obs_id')
            t = pd.read_csv(tp, dtype={'obs_id': str}).set_index('obs_id')
            if (not o.index.is_unique or not t.index.is_unique or set(o.index) != set(tr_ids)
                or set(t.index) != set(te_ids) or
                not o.inner_fold.astype(int).equals(fold_map.loc[o.index].inner_fold.astype(int))):
                status = 'ID或折次不一致'
            elif base is None:
                base = o[['y_true','inner_fold']].sort_index()
                test_base = t[['y_true','station_id']].sort_index()
            elif (not np.allclose(o.loc[base.index, 'y_true'], base.y_true, rtol=0, atol=1e-8)
                  or not np.allclose(t.loc[test_base.index, 'y_true'], test_base.y_true, rtol=0, atol=1e-8)):
                status = '目标值不一致'
            if status == '可用':
                oof_columns[name] = o.loc[base.index, 'y_pred'].to_numpy(float)
                test_columns[name] = t.loc[test_base.index, 'y_pred'].to_numpy(float)
        inventory.append({'task': task, 'model': name, 'config': config,
                          'oof_prediction_file': str(op.relative_to(ROOT)),
                          'test_prediction_file': str(tp.relative_to(ROOT)) if tp.exists() else '',
                          'oof_n': len(tr_ids) if status == '可用' else 0,
                          'test_n': len(te_ids) if status == '可用' else 0,
                          'status': status})
    locked = pd.read_csv(ADAPTED/'10_task_best_tabicl_config.csv').set_index('task')
    selected_config = locked.loc[task, 'selected_config']
    additional = []
    fold_paths = sorted((ADAPTED/'07_inner_oof_predictions').glob(
        f'{task}_{selected_config}_fold*.csv'))
    if len(fold_paths) == 5:
        additional.append(('tabicl_adapted', selected_config, fold_paths,
            ADAPTED/'08_final_test_predictions'/f'{task}_tabicl_adapted.csv'))
    context_lock = pd.read_csv(CONTEXT/'08_final_task_configuration.csv').set_index('task').loc[task]
    if context_lock.inner_source == 'current_fixed_batch_fivefold':
        scheme, temporal = context_lock.scheme, context_lock.temporal_set
        context_folds = sorted((CONTEXT/'03_context_oof_parts').glob(
            f'{task}_{scheme}_{temporal}_fold*.csv'))
        additional.append(('tabicl_context_temporal', f'{scheme}_{temporal}', context_folds,
            CONTEXT/'09_final_test_predictions_parts'/f'{task}_{scheme}_{temporal}_fold0.csv'))
    for name, config, files, tp in additional:
        status = '可用' if len(files) == 5 and tp.exists() else 'OOF或外部预测缺失'
        if status == '可用':
            o = pd.concat((pd.read_csv(path, dtype={'obs_id': str}) for path in files),
                          ignore_index=True).set_index('obs_id')
            t = pd.read_csv(tp, dtype={'obs_id': str}).set_index('obs_id')
            fold_col = 'fold'
            if (not o.index.is_unique or not t.index.is_unique or set(o.index) != set(tr_ids)
                or set(t.index) != set(te_ids)
                or not o[fold_col].astype(int).equals(fold_map.loc[o.index].inner_fold.astype(int))
                or not np.allclose(o.loc[base.index, 'y_true'], base.y_true, rtol=0, atol=1e-8)
                or not np.allclose(t.loc[test_base.index, 'y_true'], test_base.y_true, rtol=0, atol=1e-8)):
                status = 'ID、折次或目标值不一致'
            else:
                oof_columns[name] = o.loc[base.index, 'y_pred'].to_numpy(float)
                test_columns[name] = t.loc[test_base.index, 'y_pred'].to_numpy(float)
        inventory.append({'task': task, 'model': name, 'config': config,
            'oof_prediction_file': '|'.join(str(path.relative_to(ROOT)) for path in files),
            'test_prediction_file': str(tp.relative_to(ROOT)) if tp.exists() else '',
            'oof_n': len(tr_ids) if status == '可用' else 0,
            'test_n': len(te_ids) if status == '可用' else 0, 'status': status})
    if len(oof_columns) < 2:
        raise ValueError(f'{task}: fewer than two valid base models')
    if not np.isfinite(np.column_stack(list(oof_columns.values()))).all():
        raise ValueError(f'{task}: nonfinite OOF base prediction')
    data = pd.read_csv(MASTER, usecols=['obs_id', *features], dtype={'obs_id': str}, low_memory=False)
    if data.obs_id.duplicated().any(): raise ValueError('Master duplicate obs_id')
    data = data.set_index('obs_id')
    x_train = data.loc[base.index, features].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    x_test = data.loc[test_base.index, features].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    return inventory, base, test_base, oof_columns, test_columns, x_train, x_test


def crossfit(method, lam, p, y, folds, x):
    pred = np.full(len(y), np.nan)
    for fold in sorted(set(folds)):
        train = folds != fold
        val = ~train
        pred[val], _ = fit_predict(method, lam, p[train], y[train], p[val], x[train], x[val])
    if not np.isfinite(pred).all(): raise ValueError('Incomplete level-2 crossfit')
    return pred


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split_file = RANDOM/'01_random_split_manifest.csv'
    inner_file = ZOO/'01_inner_fold_manifest.csv'
    if sha(split_file) != SPLIT_SHA or sha(inner_file) != INNER_SHA:
        raise ValueError('Frozen split SHA256 changed')
    split = pd.read_csv(split_file, dtype={'obs_id': str})
    inner = pd.read_csv(inner_file, dtype={'obs_id': str})
    registry = pd.read_csv(ZOO/'02_model_registry.csv')
    features = pd.read_csv(FEATURE_MANIFEST).query('included_in_pilot == True').feature.tolist()
    if len(features) != 77 or 'water_zone_code' in features:
        raise ValueError('Strict feature list changed')
    inventory_rows, coverage_rows, prediction_rows, comparison_rows = [], [], [], []
    method_details = {}
    for task in TASKS:
        try:
            inv, base, test_base, oof, test, x_train, x_test = load_task(
                task, split, inner, features, registry)
        except Exception as error:
            coverage_rows.append({'task': task, 'status': f'BLOCKED: {error}'})
            print(task, 'BLOCKED', error, flush=True)
            continue
        inventory_rows.extend(inv)
        y = base.y_true.to_numpy(float)
        yt = test_base.y_true.to_numpy(float)
        folds = base.inner_fold.to_numpy(int)
        ranked = sorted(oof, key=lambda m: (-score(y, oof[m])['R2'], m))
        selected_models = ranked[:3]
        p = np.column_stack([oof[name] for name in selected_models])
        q = np.column_stack([test[name] for name in selected_models])
        coverage_rows.append({'task': task, 'status': '可用', 'n_train_oof': len(y),
                              'n_test': len(yt), 'n_models': len(oof),
                              'selected_models': '|'.join(selected_models)})
        methods = {'M0': [0.0], 'M1': M1_GRID, 'M2': M2_GRID, 'M3': M3_GRID}
        internal = {}
        for method, grid in methods.items():
            candidates = []
            for lam in grid:
                cv_prediction = crossfit(method, lam, p, y, folds, x_train)
                candidates.append((score(y, cv_prediction)['R2'], lam, cv_prediction))
            _, lam, cv_prediction = max(candidates, key=lambda row: (row[0], -row[1]))
            external_prediction, detail = fit_predict(method, lam, p, y, q, x_train, x_test)
            internal[method] = {'lambda': lam, 'cv_prediction': cv_prediction,
                                'test_prediction': external_prediction, 'detail': detail,
                                'cv_metrics': score(y, cv_prediction),
                                'test_metrics': score(yt, external_prediction)}
            for split_name, ids, truth, predicted, fold_values in [
                ('inner_meta_cv', base.index, y, cv_prediction, folds),
                ('external_test', test_base.index, yt, external_prediction, np.zeros(len(yt), dtype=int))]:
                prediction_rows.append(pd.DataFrame({'task': task, 'method': method,
                    'split': split_name, 'obs_id': ids, 'inner_fold': fold_values,
                    'y_true': truth, 'y_pred': predicted}))
        chosen = max(internal, key=lambda method: internal[method]['cv_metrics']['R2'])
        inner_best = ranked[0]
        external_best = max(test, key=lambda model: score(yt, test[model])['R2'])
        best_test = score(yt, test[external_best])
        chosen_test = internal[chosen]['test_metrics']
        best_inner = score(yt, test[inner_best])
        comparison_rows.append({'task': task, 'selected_base_models': '|'.join(selected_models),
            'inner_best_single_model': inner_best, 'inner_best_single_test_R2': best_inner['R2'],
            'external_best_single_descriptive': external_best,
            'external_best_single_R2': best_test['R2'],
            'external_best_single_RMSE': best_test['RMSE'],
            'external_best_single_MAE': best_test['MAE'],
            'selected_fusion': chosen, 'selected_lambda': internal[chosen]['lambda'],
            'inner_meta_cv_R2': internal[chosen]['cv_metrics']['R2'],
            'inner_meta_cv_RMSE': internal[chosen]['cv_metrics']['RMSE'],
            'inner_meta_cv_MAE': internal[chosen]['cv_metrics']['MAE'],
            'inner_fold_mean_R2': np.mean([score(y[folds==f], internal[chosen]['cv_prediction'][folds==f])['R2'] for f in sorted(set(folds))]),
            'fusion_R2': chosen_test['R2'], 'fusion_RMSE': chosen_test['RMSE'],
            'fusion_MAE': chosen_test['MAE'],
            'delta_R2_vs_external_best_single': chosen_test['R2']-best_test['R2'],
            'delta_RMSE_vs_external_best_single': chosen_test['RMSE']-best_test['RMSE'],
            'delta_MAE_vs_external_best_single': chosen_test['MAE']-best_test['MAE'],
            'delta_R2_vs_inner_best_single': chosen_test['R2']-best_inner['R2']})
        method_details[task] = {'selected_base_models': selected_models,
            'selected_fusion_by_inner_cv': chosen,
            'methods': {method: {'lambda': item['lambda'], 'detail': item['detail'],
                'inner_meta_cv': item['cv_metrics'], 'external_test': item['test_metrics']}
                for method, item in internal.items()}}
        print(task, selected_models, chosen, 'external R2', chosen_test['R2'], flush=True)
    pd.DataFrame(inventory_rows).to_csv(OUT/'00_existing_model_asset_inventory.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(coverage_rows).to_csv(OUT/'01_oof_coverage.csv', index=False, encoding='utf-8-sig')
    pd.concat(prediction_rows, ignore_index=True).to_parquet(OUT/'02_ensemble_predictions.parquet', index=False)
    pd.DataFrame(comparison_rows).to_csv(OUT/'03_ten_task_ensemble_comparison.csv', index=False, encoding='utf-8-sig')
    (OUT/'method_details.json').write_text(json.dumps(method_details, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
