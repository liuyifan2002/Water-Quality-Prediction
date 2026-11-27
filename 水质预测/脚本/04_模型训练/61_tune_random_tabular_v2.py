"""Bounded inner-OOF tuning of the two strongest traditional baselines per task."""
from __future__ import annotations
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/random_tabular_model_zoo_v2'
BASE_PATH=ROOT/'脚本/04_模型训练/57_run_random_tabular_model_zoo_v2.py'
spec=importlib.util.spec_from_file_location('random_zoo_baseline',BASE_PATH)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
TRAD=['xgboost','catboost','lightgbm','random_forest','extra_trees','hist_gb','svr_rbf','ridge']
GRID={
 'xgboost':[{'max_depth':d,'n_estimators':n,'learning_rate':lr,'min_child_weight':mc} for d,n,lr,mc in [(3,500,.04,1),(4,500,.04,1),(5,400,.05,1),(3,700,.03,3),(4,650,.035,3),(5,550,.04,3),(3,450,.06,5),(4,450,.06,5)]],
 'catboost':[{'depth':d,'iterations':n,'learning_rate':lr,'l2_leaf_reg':l2} for d,n,lr,l2 in [(4,600,.05,3),(5,500,.05,3),(6,650,.04,3),(5,700,.035,5),(7,450,.05,5),(6,500,.06,7)]],
 'lightgbm':[{'num_leaves':leaves,'n_estimators':n,'learning_rate':lr,'min_child_samples':mc} for leaves,n,lr,mc in [(15,500,.05,20),(31,500,.04,20),(63,400,.05,30),(15,700,.035,40),(31,650,.035,40),(63,500,.04,50)]],
 'random_forest':[{'n_estimators':n,'min_samples_leaf':leaf,'max_features':feat} for n,leaf,feat in [(350,1,.8),(350,2,1.0),(350,4,.8),(500,2,.6)]],
 'extra_trees':[{'n_estimators':n,'min_samples_leaf':leaf,'max_features':feat} for n,leaf,feat in [(350,1,.8),(350,2,1.0),(350,4,.8),(500,2,.6)]],
 'hist_gb':[{'max_iter':n,'max_leaf_nodes':leaves,'learning_rate':lr} for n,leaves,lr in [(400,15,.05),(400,31,.05),(600,31,.035),(500,63,.04)]],
 'svr_rbf':[{'C':c,'epsilon':eps,'gamma':'scale'} for c,eps in [(1,.05),(3,.05),(10,.1),(30,.1)]],
 'ridge':[{'alpha':a} for a in [.1,1,100]],
}

def candidates(task):
    rows=[]
    for model in TRAD:
        p=OUT/'03_inner_metrics'/f'{task}_{model}.json'
        if p.exists():rows.append((model,json.loads(p.read_text(encoding='utf-8'))['inner_oof_R2']))
    if len(rows)<2:raise ValueError('traditional baseline incomplete')
    return [x[0] for x in sorted(rows,key=lambda z:z[1],reverse=True)[:2]]

def evaluate(task,model,params,ids,x,y):
    pred=np.full(len(y),np.nan);fit=0.;infer=0.
    for fold in range(1,6):
        a=ids.inner_fold.to_numpy()!=fold;b=~a
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        xa=imp.fit_transform(x[a]).astype(np.float32);xb=imp.transform(x[b]).astype(np.float32)
        if model in base.STANDARD:
            scale=StandardScaler();xa=scale.fit_transform(xa);xb=scale.transform(xb)
        yt=y[a];log=np.min(yt)>=0;yf=np.log1p(yt) if log else yt
        estimator=base.model_for(model).set_params(**params)
        t=time.perf_counter();estimator.fit(xa,yf);fit+=time.perf_counter()-t
        t=time.perf_counter();v=estimator.predict(xb);infer+=time.perf_counter()-t
        pred[b]=np.maximum(np.expm1(v) if log else v,0)
    return pred,{'R2':float(r2_score(y,pred)),'RMSE':float(mean_squared_error(y,pred)**.5),'MAE':float(mean_absolute_error(y,pred)),'fit_seconds':fit,'infer_seconds':infer}

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=base.TASKS);a=p.parse_args()
    ids,x,y=base.load_task(a.task)
    (OUT/'05_tuned_candidates').mkdir(exist_ok=True)
    for model in candidates(a.task):
        dest=OUT/'02_inner_oof'/f'{a.task}_{model}_tuned.csv';meta=OUT/'03_inner_metrics'/f'{a.task}_{model}_tuned.json'
        if dest.exists() and meta.exists():print('SKIP',a.task,model,flush=True);continue
        results=[];best=None
        for i,params in enumerate(GRID[model]):
            pred,metric=evaluate(a.task,model,params,ids,x,y)
            row={'candidate':i,'params':params,**metric};results.append(row)
            if best is None or metric['R2']>best[0]:best=(metric['R2'],pred,i,params,metric)
            print(a.task,model,i,round(metric['R2'],4),flush=True)
        assert best is not None
        _,pred,i,params,metric=best
        pd.DataFrame({'task':a.task,'model':model+'_tuned','obs_id':ids.obs_id,'inner_fold':ids.inner_fold,'y_true':y,'y_pred':pred}).to_csv(dest,index=False,encoding='utf-8-sig')
        meta.write_text(json.dumps({'task':a.task,'model':model+'_tuned','base_model':model,'chosen_candidate':i,'chosen_params':params,'inner_oof_R2':metric['R2'],'inner_oof_RMSE':metric['RMSE'],'inner_oof_MAE':metric['MAE'],'fit_seconds':metric['fit_seconds'],'infer_seconds':metric['infer_seconds'],'search':results},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
