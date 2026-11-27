"""Evaluate locked TabICLv2 configurations once on unchanged Random test IDs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / '脚本/04_模型训练/67_run_tabicl_water_quality_inner.py'
spec = importlib.util.spec_from_file_location('tabicl_adaptation_inner', SOURCE)
inner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inner)
from models.tabiclv2_adapter import TabICLv2Regressor


def outer_data(task):
    target, depth = inner.TASKS[task]
    frame = pd.read_csv(inner.TABLE, usecols=['obs_id','station','date','depth',target,*inner.FEATURES], low_memory=False)
    frame = frame.loc[frame.depth.eq(depth) & frame[target].notna()].copy()
    frame.obs_id = frame.obs_id.astype(str)
    split = pd.read_csv(ROOT / 'runs/random_10task_v1/01_random_split_manifest.csv', dtype={'obs_id': str})
    split = split.loc[split.task.eq(task), ['obs_id','split']]
    frame = frame.merge(split, on='obs_id', how='inner', validate='one_to_one')
    if len(frame) != len(split):
        raise ValueError('Frozen Random split coverage mismatch')
    return (frame.loc[frame.split.eq('train')].sort_values('obs_id').copy(),
            frame.loc[frame.split.eq('test')].sort_values('obs_id').copy(), target)


def predict(config_id, train, test, target):
    config = inner.CONFIGS[config_id].copy()
    xtrain = train[inner.FEATURES].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    xtest = test[inner.FEATURES].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    xa = imputer.fit_transform(xtrain).astype(np.float32)
    xb = imputer.transform(xtest).astype(np.float32)
    y = train[target].to_numpy(float)
    if config.pop('interactions', False):
        xa = np.column_stack([xa, *(xa[:, inner.FEATURES.index(a)] * xa[:, inner.FEATURES.index(b)]
                                    for a,b in inner.INTERACTIONS)]).astype(np.float32)
        xb = np.column_stack([xb, *(xb[:, inner.FEATURES.index(a)] * xb[:, inner.FEATURES.index(b)]
                                    for a,b in inner.INTERACTIONS)]).astype(np.float32)
    if 'top_features' in config:
        k = config.pop('top_features')
        relevance = mutual_info_regression(xa, y, random_state=42)
        selected = np.sort(np.argsort(-relevance, kind='stable')[:k])
        xa, xb = xa[:,selected], xb[:,selected]
    transform = config.pop('target_transform', 'log1p')
    if transform == 'log1p':
        if y.min() < 0: raise ValueError('negative target under log1p')
        yt, inverse = np.log1p(y), np.expm1
    elif transform == 'raw':
        yt, inverse = y, lambda v: v
    elif transform == 'yeojohnson':
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        yt = pt.fit_transform(y.reshape(-1,1)).ravel()
        inverse = lambda v: pt.inverse_transform(np.asarray(v).reshape(-1,1)).ravel()
    else: raise ValueError(transform)
    output_type = config.pop('output_type','mean')
    nonnegative_clip = config.pop('nonnegative_clip', True)
    model = TabICLv2Regressor(**config)
    t0 = time.perf_counter(); model.fit(xa,yt); fit_seconds=time.perf_counter()-t0
    t0 = time.perf_counter(); raw=model.model.predict(xb,output_type=output_type)
    predict_seconds=time.perf_counter()-t0
    prediction = inverse(np.asarray(raw).reshape(-1))
    if nonnegative_clip: prediction = np.maximum(prediction,0)
    return prediction,fit_seconds,predict_seconds


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--task',choices=inner.TASKS,required=True)
    task=parser.parse_args().task
    selection_path=inner.OUT/'10_task_best_tabicl_config.csv'
    lock=(inner.OUT/'10_task_best_tabicl_config.sha256').read_text(encoding='utf-8').split()[0]
    if hashlib.sha256(selection_path.read_bytes()).hexdigest()!=lock:
        raise ValueError('Locked selection hash changed; external test prohibited')
    selected=pd.read_csv(selection_path)
    config_id=selected.loc[selected.task.eq(task),'selected_config'].iloc[0]
    output=inner.OUT/'08_final_test_predictions';output.mkdir(parents=True,exist_ok=True)
    path=output/f'{task}_tabicl_adapted.csv'
    if path.exists():
        print('SKIP',path,flush=True);return
    train,test,target=outer_data(task)
    old=pd.read_csv(ROOT/'runs/random_tabular_model_zoo_v2/07_final_predictions'/f'{task}_tabicl.csv')
    if set(old.obs_id)!=set(test.obs_id): raise ValueError('Baseline test IDs differ')
    if config_id=='Z0':
        prediction=old.set_index('obs_id').loc[test.obs_id,'y_pred'].to_numpy(float)
        fit_seconds=predict_seconds=0.0
        origin='strict_reuse_original_random_test_prediction'
    else:
        import torch
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        prediction,fit_seconds,predict_seconds=predict(config_id,train,test,target)
        origin='new_locked_tabicl_single_model_evaluation'
    y=test[target].to_numpy(float)
    if len(prediction)!=len(y) or not np.isfinite(prediction).all():raise ValueError('invalid prediction')
    frame=pd.DataFrame(dict(task=task,model='tabicl_adapted',config_id=config_id,
                            obs_id=test.obs_id.to_numpy(),station_id=test.station.to_numpy(),
                            date=test.date.to_numpy(),y_true=y,y_pred=prediction,
                            absolute_error=np.abs(prediction-y),squared_error=(prediction-y)**2))
    frame.to_csv(path,index=False,encoding='utf-8-sig')
    peak=None
    if config_id!='Z0': peak=float(torch.cuda.max_memory_allocated()/2**20) if torch.cuda.is_available() else None
    meta=dict(task=task,config_id=config_id,n_train=len(train),n_test=len(test),
              fit_seconds=fit_seconds,predict_seconds=predict_seconds,
              gpu_peak_allocated_mb=peak,source=origin,**inner.score(y,prediction))
    path.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(task,config_id,meta['R2'],flush=True)


if __name__=='__main__':main()
