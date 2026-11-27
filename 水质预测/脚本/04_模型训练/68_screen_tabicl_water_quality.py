"""Run a bounded, resumable TabICLv2 inner-fold candidate screen."""
from __future__ import annotations

import argparse
import importlib.util
import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '脚本/04_模型训练/67_run_tabicl_water_quality_inner.py'
spec = importlib.util.spec_from_file_location('tabicl_adaptation_inner', SOURCE)
inner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', choices=inner.TASKS, default=list(inner.TASKS))
    parser.add_argument('--configs', nargs='+', choices=inner.CONFIGS, required=True)
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6), default=[1])
    args = parser.parse_args()
    pred_dir = inner.OUT / '07_inner_oof_predictions'
    metric_dir = inner.OUT / 'inner_fold_metrics'
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        ids, x, y = inner.load(task)
        for config in args.configs:
            for fold in args.folds:
                stem = f'{task}_{config}_fold{fold}'
                pred_path = pred_dir / f'{stem}.csv'
                metric_path = metric_dir / f'{stem}.json'
                if pred_path.exists() and metric_path.exists():
                    print('SKIP', stem, flush=True)
                    continue
                try:
                    pred, metric = inner.run_fold(task, config, fold, ids, x, y)
                    pred.to_csv(pred_path, index=False, encoding='utf-8-sig')
                    metric_path.write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding='utf-8')
                    print('PASS', stem, f"R2={metric['R2']:.6f}", flush=True)
                except Exception as exc:
                    error_path = metric_dir / f'{stem}.error.txt'
                    error_path.write_text(traceback.format_exc(), encoding='utf-8')
                    print('FAILED', stem, str(exc), flush=True)


if __name__ == '__main__':
    main()
