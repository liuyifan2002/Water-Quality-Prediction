"""Pick one normalization challenger per task from the common first inner fold."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tabicl_water_quality_adaptation_v1'
ZOO = ROOT / 'runs/random_tabular_model_zoo_v2'
TASKS = ['chla_surface','chla_middle','chla_bottom','secchi','tn_surface',
         'tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']


def main():
    rows=[]
    for task in TASKS:
        base=pd.read_csv(ZOO/'02_inner_oof'/f'{task}_tabicl.csv')
        base=base.loc[base.inner_fold.eq(1)]
        candidates=[dict(task=task,config_id='Z0',fold1_R2=r2_score(base.y_true,base.y_pred))]
        for config in ['Z2','Z3','Z4','Z5']:
            path=OUT/'07_inner_oof_predictions'/f'{task}_{config}_fold1.csv'
            if not path.exists():
                candidates.append(dict(task=task,config_id=config,fold1_R2=float('nan')))
                continue
            frame=pd.read_csv(path)
            if set(frame.obs_id)!=set(base.obs_id):raise ValueError(f'{task} {config}: differing fold IDs')
            candidates.append(dict(task=task,config_id=config,fold1_R2=r2_score(frame.y_true,frame.y_pred)))
        candidates=sorted(candidates,key=lambda x:(pd.isna(x['fold1_R2']),
                                                    -x['fold1_R2'] if pd.notna(x['fold1_R2']) else 0,
                                                    x['config_id']))
        challenger=next(row for row in candidates if row['config_id']!='Z0' and pd.notna(row['fold1_R2']))
        rows.extend(dict(**row,selected_for_full_fivefold=row['config_id'] in {'Z0',challenger['config_id']})
                    for row in candidates)
    pd.DataFrame(rows).to_csv(OUT/'02_inference_screen_selection.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(rows).loc[lambda d:d.selected_for_full_fivefold][['task','config_id','fold1_R2']].to_string(index=False))


if __name__=='__main__':main()
