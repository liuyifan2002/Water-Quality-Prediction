"""Shared-inner-fold baseline for RANDOM_TABULAR_MODEL_ZOO_V2.

Runs one task/model at a time and writes only training-set OOF predictions.
The frozen external test set is never opened by this script.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/random_tabular_model_zoo_v2'
TABLE=ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv'
FEATURES=pd.read_csv(ROOT/'runs/tabiclv2_pilot/01_strict_feature_manifest.csv').query('included_in_pilot == True').feature.tolist()
TASKS={
 'chla_surface':('chla_ug_l','surface'),'chla_middle':('chla_ug_l','middle'),'chla_bottom':('chla_ug_l','bottom'),
 'secchi':('secchi_depth_m','surface'),'tn_surface':('tn_mg_l','surface'),'tn_middle':('tn_mg_l','middle'),
 'tn_bottom':('tn_mg_l','bottom'),'tp_surface':('tp_mg_l','surface'),'tp_middle':('tp_mg_l','middle'),
 'tp_bottom':('tp_mg_l','bottom')}
MODELS=('xgboost','catboost','lightgbm','random_forest','extra_trees','hist_gb','svr_rbf','ridge','mlp')
STANDARD={'svr_rbf','ridge','mlp'}

def model_for(name):
    if name=='xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,n_jobs=4,random_state=42,objective='reg:squarederror')
    if name=='catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=500,depth=6,learning_rate=.05,l2_leaf_reg=3,thread_count=4,random_seed=42,loss_function='RMSE',verbose=False,allow_writing_files=False)
    if name=='lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=400,num_leaves=31,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=42,n_jobs=4,verbosity=-1)
    if name=='random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=250,min_samples_leaf=2,max_features=.8,n_jobs=4,random_state=42)
    if name=='extra_trees':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=250,min_samples_leaf=2,max_features=.8,n_jobs=4,random_state=42)
    if name=='hist_gb':
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=31,l2_regularization=1,random_state=42)
    if name=='svr_rbf':
        from sklearn.svm import SVR
        return SVR(kernel='rbf',C=10,epsilon=.05,gamma='scale',cache_size=1000)
    if name=='ridge':
        from sklearn.linear_model import Ridge
        return Ridge(alpha=10)
    if name=='mlp':
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(128,64),alpha=.001,learning_rate_init=.001,max_iter=200,early_stopping=True,n_iter_no_change=15,random_state=42)
    raise ValueError(name)

def load_task(task):
    target,depth=TASKS[task]
    cols=['obs_id','depth',target,*FEATURES]
    data=pd.read_csv(TABLE,usecols=cols,low_memory=False)
    data=data.loc[data.depth.eq(depth)&data[target].notna()].copy()
    data.obs_id=data.obs_id.astype(str)
    split=pd.read_csv(OUT/'01_inner_fold_manifest.csv',dtype={'obs_id':str})
    split=split.loc[split.task.eq(task),['obs_id','inner_fold']]
    data=data.merge(split,on='obs_id',how='inner',validate='one_to_one')
    if len(data)!=len(split) or data.obs_id.duplicated().any():raise ValueError(f'{task} training IDs mismatch')
    x=data[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(float)
    y=data[target].to_numpy(float)
    if len(FEATURES)!=77 or 'water_zone_code' in FEATURES or not np.isfinite(y).all():raise ValueError('features/target invalid')
    return data[['obs_id','inner_fold']],x,y

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--model',choices=MODELS,required=True);a=p.parse_args()
    path=OUT/'02_inner_oof'/f'{a.task}_{a.model}.csv';met=OUT/'03_inner_metrics'/f'{a.task}_{a.model}.json'
    if path.exists() and met.exists():print('SKIP',a.task,a.model,flush=True);return
    ids,x,y=load_task(a.task);pred=np.full(len(y),np.nan);rows=[]
    for fold in range(1,6):
        train=(ids.inner_fold.to_numpy()!=fold);val=~train
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        xa=imp.fit_transform(x[train]).astype(np.float32);xb=imp.transform(x[val]).astype(np.float32)
        if a.model in STANDARD:
            scale=StandardScaler();xa=scale.fit_transform(xa);xb=scale.transform(xb)
        yt=y[train];use_log=np.min(yt)>=0
        yf=np.log1p(yt) if use_log else yt
        model=model_for(a.model)
        t=time.perf_counter();model.fit(xa,yf);fit_sec=time.perf_counter()-t
        t=time.perf_counter();yp=model.predict(xb);infer_sec=time.perf_counter()-t
        yp=np.maximum(np.expm1(yp) if use_log else yp,0)
        pred[val]=yp
        rows.append({'fold':fold,'n_train':int(train.sum()),'n_val':int(val.sum()),'R2':float(r2_score(y[val],yp)),'RMSE':float(mean_squared_error(y[val],yp)**.5),'MAE':float(mean_absolute_error(y[val],yp)),'fit_seconds':fit_sec,'infer_seconds':infer_sec})
        print(a.task,a.model,'fold',fold,'R2',rows[-1]['R2'],flush=True)
    if np.isnan(pred).any():raise ValueError('incomplete OOF')
    path.parent.mkdir(parents=True,exist_ok=True);met.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'task':a.task,'model':a.model,'obs_id':ids.obs_id,'inner_fold':ids.inner_fold,'y_true':y,'y_pred':pred}).to_csv(path,index=False,encoding='utf-8-sig')
    result={'task':a.task,'model':a.model,'n_samples':len(y),'inner_oof_R2':float(r2_score(y,pred)),'inner_oof_RMSE':float(mean_squared_error(y,pred)**.5),'inner_oof_MAE':float(mean_absolute_error(y,pred)),'fit_seconds':sum(r['fit_seconds'] for r in rows),'infer_seconds':sum(r['infer_seconds'] for r in rows),'folds':rows,'config':model_for(a.model).get_params()}
    met.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('DONE',a.task,a.model,'OOF R2',result['inner_oof_R2'],flush=True)

if __name__=='__main__':main()
