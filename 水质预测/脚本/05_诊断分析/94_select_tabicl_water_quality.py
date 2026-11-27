"""Lock TabICLv2 task configs using training-only fivefold OOF metrics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tabicl_water_quality_adaptation_v1'
TASKS = ['chla_surface','chla_middle','chla_bottom','secchi','tn_surface',
         'tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']


def main():
    metrics = pd.read_csv(OUT / '03_all_inner_candidate_metrics.csv')
    rows = []
    for task in TASKS:
        candidates = metrics.loc[(metrics.task == task) & (metrics.folds == 5) &
                                 (~metrics.config_id.str.startswith('FT'))].copy()
        if not candidates.config_id.eq('Z0').any():
            raise ValueError(f'{task}: baseline fivefold missing')
        if len(candidates) < 2:
            raise ValueError(f'{task}: no fivefold challenger; do not select from one fold')
        expected_n = int(candidates.loc[candidates.config_id.eq('Z0'),'n'].iloc[0])
        if not candidates.n.eq(expected_n).all():
            raise ValueError(f'{task}: candidate OOF row count differs from baseline')
        candidates = candidates.sort_values(['R2','RMSE','MAE','config_id'],
                                             ascending=[False,True,True,True], kind='stable')
        best = candidates.iloc[0]
        rows.append(dict(task=task, selected_config=best.config_id,
                         inner_oof_R2=best.R2, inner_oof_RMSE=best.RMSE,
                         inner_oof_MAE=best.MAE, fold_R2_mean=best.fold_R2_mean,
                         fold_R2_sd=best.fold_R2_sd, n_inner=int(best.n),
                         selection_rule='max fivefold pooled OOF R2; RMSE/MAE tie break'))
    destination = OUT / '10_task_best_tabicl_config.csv'
    if destination.exists():
        raise FileExistsError('Selection already locked; never overwrite after external test')
    pd.DataFrame(rows).to_csv(destination, index=False, encoding='utf-8-sig')
    (OUT / '10_task_best_tabicl_config.sha256').write_text(
        hashlib.sha256(destination.read_bytes()).hexdigest() + '  10_task_best_tabicl_config.csv\n',
        encoding='utf-8')
    print(pd.DataFrame(rows)[['task','selected_config','inner_oof_R2']].to_string(index=False))


if __name__ == '__main__':
    main()
