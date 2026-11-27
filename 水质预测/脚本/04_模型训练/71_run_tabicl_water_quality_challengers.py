"""Finish fivefold OOF for preselected zero-shot challengers, one GPU process per fold."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tabicl_water_quality_adaptation_v1'
WORKER = ROOT / '脚本/04_模型训练/67_run_tabicl_water_quality_inner.py'


def main():
    selected = pd.read_csv(OUT / '02_inference_screen_selection.csv')
    selected = selected.loc[selected.selected_for_full_fivefold & selected.config_id.ne('Z0')]
    if len(selected) != 10 or selected.task.nunique() != 10:
        raise ValueError('Expected one first-fold challenger for each of ten tasks')
    for row in selected.itertuples():
        for fold in range(2,6):
            cmd = [sys.executable, str(WORKER), '--task', row.task,
                   '--config', row.config_id, '--fold', str(fold)]
            try:
                result = subprocess.run(cmd, cwd=ROOT, timeout=240, check=False)
                if result.returncode:
                    print('FAILED', row.task, row.config_id, fold, result.returncode, flush=True)
            except subprocess.TimeoutExpired:
                path = OUT/'inner_fold_metrics'/f'{row.task}_{row.config_id}_fold{fold}.error.txt'
                path.write_text('超过240秒单折预算，候选不作为完整五折结果。\n',encoding='utf-8')
                print('TIMEOUT',row.task,row.config_id,fold,flush=True)


if __name__=='__main__':main()
