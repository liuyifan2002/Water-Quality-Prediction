"""Official fold-local TabICLv2 fine-tuning; never accesses the external test set."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '脚本/04_模型训练/67_run_tabicl_water_quality_inner.py'
spec = importlib.util.spec_from_file_location('tabicl_adaptation_inner', SOURCE)
inner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=inner.TASKS, required=True)
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--config', choices=['FT1', 'FT2', 'FT3', 'FT1_CHUNK2048'], default='FT1')
    args = parser.parse_args()
    from tabicl import FinetunedTabICLRegressor
    import torch

    ids, x, y = inner.load(args.task)
    evaluation = ids.inner_fold.to_numpy() == args.fold
    train_indices = np.flatnonzero(~evaluation)
    fit_indices, early_indices = train_test_split(train_indices, test_size=0.1, random_state=42, shuffle=True)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    x_fit = imputer.fit_transform(x.iloc[fit_indices]).astype(np.float32)
    x_early = imputer.transform(x.iloc[early_indices]).astype(np.float32)
    x_eval = imputer.transform(x.iloc[evaluation]).astype(np.float32)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    checkpoint = ROOT / 'runs/tabicl_explanation_portable_bundle_v2/model/tabicl-regressor-v2-20260212.ckpt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = FinetunedTabICLRegressor(
        epochs=30, learning_rate=3e-5 if args.config == 'FT2' else 1e-5,
        weight_decay=.01, grad_clip=1.0, amp=True, n_estimators_finetune=2,
        n_estimators_validation=2, n_estimators_inference=8,
        early_stopping=True, patience=8, eval_metric='mse', time_limit=1200,
        max_data_size=2048 if args.config == 'FT1_CHUNK2048' else 10000,
        freeze_icl=args.config == 'FT3', device='cuda', random_state=42,
        model_path=checkpoint, allow_auto_download=False, verbose=True)
    out = inner.OUT / 'finetuned_models' / f'{args.task}_{args.config}_fold{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model.fit(x_fit, y[fit_indices], X_val=x_early, y_val=y[early_indices], output_dir=out)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = np.asarray(model.predict(x_eval)).reshape(-1)
    predict_seconds = time.perf_counter() - start
    meta = dict(task=args.task, config_id=args.config, fold=args.fold,
                n_fit=len(fit_indices), n_early_stop=len(early_indices), n_eval=int(evaluation.sum()),
                fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                gpu_peak_allocated_mb=float(torch.cuda.max_memory_allocated() / 2**20))
    for suffix, values in [('', predictions), ('_CLIP', np.maximum(predictions, 0))]:
        name = args.config + suffix
        pred = inner.pd.DataFrame({'task': args.task, 'config_id': name,
                                   'obs_id': ids.loc[evaluation, 'obs_id'].to_numpy(),
                                   'fold': args.fold, 'y_true': y[evaluation], 'y_pred': values})
        pred_path = inner.OUT / '07_inner_oof_predictions' / f'{args.task}_{name}_fold{args.fold}.csv'
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(pred_path, index=False, encoding='utf-8-sig')
        result = dict(meta, config_id=name, nonnegative_clip=bool(suffix),
                      **inner.score(y[evaluation], values))
        meta_path = inner.OUT / 'inner_fold_metrics' / f'{args.task}_{name}_fold{args.fold}.json'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
