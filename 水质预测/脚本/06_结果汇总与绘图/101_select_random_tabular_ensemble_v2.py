"""Select single/convex-blend candidates using shared inner OOF only."""
from __future__ import annotations
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/random_tabular_model_zoo_v2'
FAMILIES={
 'xgboost':'traditional','catboost':'traditional','lightgbm':'traditional','random_forest':'traditional',
 'extra_trees':'traditional','hist_gb':'traditional','svr_rbf':'traditional','ridge':'traditional',
 'mlp':'deep','tabm':'deep','tabicl':'foundation','tabpfn':'foundation','tabdpt':'foundation','tabfm':'foundation','mitra_v2':'foundation'}
VERSIONS={'tabicl':'tabicl==2.2.0','tabpfn':'tabpfn==9.0.0; pinned V2 checkpoint',
          'tabm':'tabm==0.0.3; arch_type=tabm; k=32','tabdpt':'tabdpt==1.3.2; v1.3 checkpoint'}
for identity,package in [('xgboost','xgboost'),('catboost','catboost'),('lightgbm','lightgbm')]:
    VERSIONS[identity]=f'{package}=={version(package)}'
for identity in ['random_forest','extra_trees','hist_gb','svr_rbf','ridge','mlp']:
    VERSIONS[identity]=f'scikit-learn=={version("scikit-learn")}'
TASKS=['chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']

def weights(p,y):
    n=p.shape[1]
    if n==1:return np.ones(1)
    result=minimize(lambda w:np.mean((p@w-y)**2),np.ones(n)/n,method='SLSQP',bounds=[(0,1)]*n,constraints={'type':'eq','fun':lambda w:np.sum(w)-1},options={'maxiter':1000,'ftol':1e-12})
    if not result.success or not np.isfinite(result.x).all():raise ValueError(f'weight fit failed: {result.message}')
    return result.x

def metrics(y,p):return {'R2':float(r2_score(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'MAE':float(mean_absolute_error(y,p))}

def load_task(task):
    files=list((OUT/'02_inner_oof').glob(f'{task}_*.csv'))
    frames=[]
    for path in files:
        model=path.stem[len(task)+1:]
        if not (OUT/'03_inner_metrics'/f'{task}_{model}.json').exists():continue
        d=pd.read_csv(path,dtype={'obs_id':str})
        if d.obs_id.duplicated().any():continue
        frames.append((model,d.set_index('obs_id')))
    if not frames:raise ValueError(f'{task}: no complete inner OOF')
    base=frames[0][1][['y_true']].sort_index().copy()
    for model,frame in frames:
        if set(frame.index)!=set(base.index) or not np.allclose(frame.loc[base.index,'y_true'],base.y_true):raise ValueError(f'{task} {model} OOF IDs/targets differ')
        base[model]=frame.loc[base.index,'y_pred']
    return base

def main():
    plans=[];all_metrics=[];all_weights=[];rankings=[]
    for task in TASKS:
        d=load_task(task);y=d.y_true.to_numpy();models=[c for c in d if c!='y_true']
        if len(models)<3:raise ValueError(f'{task}: fewer than three complete models')
        for m in models:
            all_metrics.append({'task':task,'model':m,'family':FAMILIES.get(m.removesuffix('_tuned'),'unknown'),**metrics(y,d[m].to_numpy())})
        ranked_all=sorted(models,key=lambda m:r2_score(y,d[m]),reverse=True)
        rank=[];seen_base=set()
        for m in ranked_all:
            identity=m.removesuffix('_tuned')
            if identity not in seen_base:rank.append(m);seen_base.add(identity)
        models=rank
        for i,m in enumerate(rank,1):
            record=json.loads((OUT/'03_inner_metrics'/f'{task}_{m}.json').read_text(encoding='utf-8'))
            rankings.append({'task':task,'rank':i,'model':m,
                'family':FAMILIES.get(m.removesuffix('_tuned'),'unknown'),
                'model_identity':VERSIONS.get(m.removesuffix('_tuned'),m.removesuffix('_tuned')),
                'config':json.dumps(record.get('chosen_params',record.get('config',record.get('model_identity',record.get('identity',{})))),ensure_ascii=False,default=str),
                'inner_oof_R2':r2_score(y,d[m])})
        meta={k:np.full(len(d),np.nan) for k in ['single','pair_equal','pair_weighted','triple_weighted']}
        frequencies={m:0 for m in models};fold_notes=[]
        for fold,(a,b) in enumerate(KFold(n_splits=5,shuffle=True,random_state=314159).split(d),1):
            ordered=sorted(models,key=lambda m:r2_score(y[a],d.iloc[a][m]),reverse=True)
            top=ordered[:3]
            for m in top:frequencies[m]+=1
            va=d.iloc[b]
            meta['single'][b]=va[top[0]].to_numpy()
            meta['pair_equal'][b]=va[top[:2]].to_numpy().mean(axis=1)
            w2=weights(d.iloc[a][top[:2]].to_numpy(),y[a]);w3=weights(d.iloc[a][top[:3]].to_numpy(),y[a])
            meta['pair_weighted'][b]=va[top[:2]].to_numpy()@w2
            meta['triple_weighted'][b]=va[top[:3]].to_numpy()@w3
            fold_notes.append({'fold':fold,'top3':top,'pair_weights':w2.tolist(),'triple_weights':w3.tolist()})
        cv={k:metrics(y,p) for k,p in meta.items()}
        choice=max(cv,key=lambda k:cv[k]['R2'])
        if choice!='single' and cv[choice]['R2']<cv['single']['R2']+.002:choice='single'
        final_models=rank[:(1 if choice=='single' else 2 if choice.startswith('pair') else 3)]
        final_weights=[1.] if choice=='single' else [0.5,0.5] if choice=='pair_equal' else weights(d[final_models].to_numpy(),y).tolist()
        plan={'task':task,'selected_kind':choice,'selected_models':final_models,'weights':final_weights,'meta_cv':cv,'inner_oof_top3':rank[:3],'top3_selection_frequency':frequencies,'meta_fold_details':fold_notes}
        plans.append(plan)
        all_weights.append({'task':task,'selection':choice,'models':json.dumps(final_models),'weights':json.dumps(final_weights),'meta_cv_R2':cv[choice]['R2'],'single_meta_cv_R2':cv['single']['R2']})
        print(task,choice,final_models,round(cv[choice]['R2'],4),flush=True)
    pd.DataFrame(all_metrics).to_csv(OUT/'03_inner_cv_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rankings).to_csv(OUT/'02_model_registry.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_weights).to_csv(OUT/'06_ensemble_weights.csv',index=False,encoding='utf-8-sig')
    (OUT/'06_final_selection_plan.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
