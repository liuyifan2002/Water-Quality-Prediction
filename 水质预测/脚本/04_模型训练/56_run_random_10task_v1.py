"""Run missing Random Strict 77-feature XGBoost/TabICLv2 tasks only."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/random_10task_v1"
TABLE = ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv"
PILOT = ROOT / "runs/tabiclv2_pilot"
FEATURES = pd.read_csv(PILOT / "01_strict_feature_manifest.csv").query("included_in_pilot == True").feature.tolist()
TASKS = {
    "chla_surface": ("chla_ug_l", "surface"), "chla_middle": ("chla_ug_l", "middle"),
    "chla_bottom": ("chla_ug_l", "bottom"), "secchi": ("secchi_depth_m", "surface"),
    "tn_surface": ("tn_mg_l", "surface"), "tn_middle": ("tn_mg_l", "middle"),
    "tn_bottom": ("tn_mg_l", "bottom"), "tp_surface": ("tp_mg_l", "surface"),
    "tp_middle": ("tp_mg_l", "middle"), "tp_bottom": ("tp_mg_l", "bottom"),
}
SAVED = {"secchi": "09_secchi_predictions.csv", "tn_bottom": "14_tn_bottom_predictions.csv", "tp_middle": "17_tp_middle_predictions.csv"}
BASE = json.loads((ROOT / "结果/扩展实验/候选模型稳健性验证_20260914/portable_package/config.json").read_text(encoding="utf-8-sig"))["model_parameters"]["xgboost"]

def scores(y, p):
    return {"R2": float(r2_score(y,p)), "RMSE": float(mean_squared_error(y,p)**0.5), "MAE": float(mean_absolute_error(y,p))}

def load_data(task):
    target, depth = TASKS[task]
    cols = ["obs_id","station","date","depth",target,*FEATURES]
    d = pd.read_csv(TABLE,usecols=cols,low_memory=False)
    d["obs_id"] = d.obs_id.astype(str)
    d = d.loc[d.depth.eq(depth) & d[target].notna()].copy()
    d[target] = pd.to_numeric(d[target],errors="coerce")
    for col in FEATURES:
        d[col] = pd.to_numeric(d[col],errors="coerce").replace([np.inf,-np.inf],np.nan)
    return d

def arrays(task):
    d = load_data(task)
    split = pd.read_csv(OUT / "01_random_split_manifest.csv",dtype={"obs_id":str},usecols=["task","obs_id","split"])
    split = split.loc[split.task.eq(task),["obs_id","split"]]
    d = d.merge(split,on="obs_id",how="inner",validate="one_to_one")
    if len(d)!=len(split) or d.split.isna().any(): raise ValueError(f"{task} split mismatch")
    tr=d.loc[d.split.eq("train")].copy(); te=d.loc[d.split.eq("test")].copy()
    imp=SimpleImputer(strategy="median",keep_empty_features=True)
    xtr=imp.fit_transform(tr[FEATURES].to_numpy(float)).astype(np.float32)
    xte=imp.transform(te[FEATURES].to_numpy(float)).astype(np.float32)
    ytr=tr[TASKS[task][0]].to_numpy(float); yte=te[TASKS[task][0]].to_numpy(float)
    if np.min(ytr)>=0: return tr,te,xtr,xte,np.log1p(ytr),yte,np.expm1
    return tr,te,xtr,xte,ytr,yte,lambda x:x

def xgb_predict(xtr,ytr,xte,do_tune):
    from xgboost import XGBRegressor
    params=dict(BASE);params.update(random_state=42,objective="reg:squarederror")
    candidates=[params]
    if do_tune:
        candidates += [dict(params,n_estimators=650,max_depth=3,learning_rate=0.035),dict(params,n_estimators=550,max_depth=5,learning_rate=0.035,min_child_weight=4,reg_lambda=3)]
    chosen=0;val_scores=[]
    if do_tune:
        a,b=train_test_split(np.arange(len(ytr)),test_size=.2,random_state=42,shuffle=True)
        for p in candidates:
            m=XGBRegressor(**p);m.fit(xtr[a],ytr[a]);val_scores.append(float(r2_score(ytr[b],m.predict(xtr[b]))))
        chosen=int(np.argmax(val_scores))
    m=XGBRegressor(**candidates[chosen]);m.fit(xtr,ytr)
    return np.asarray(m.predict(xte),float),chosen,val_scores

def tabicl_predict(xtr,ytr,xte):
    from models.tabiclv2_adapter import TabICLv2Regressor
    model=TabICLv2Regressor();model.fit(xtr,ytr)
    return model.predict(xte)

def save(task,model,tr,te,yte,p,origin,chosen=None,val_scores=None,runtime=0):
    pred=pd.DataFrame({"obs_id":te.obs_id.to_numpy(),"station_id":te.station.to_numpy(),"date":te.date.to_numpy(),"task":task,"validation":"random_strict","fold":"seed_42","y_true":yte,"y_pred":p,"residual":p-yte,"model":model,"feature_set":"X_strict_77","source":origin})
    path=OUT/"03_predictions"/f"{task}_{model}.csv";path.parent.mkdir(parents=True,exist_ok=True)
    pred.to_csv(path,index=False,encoding="utf-8-sig")
    m={"task":task,"model":model,"n_train":len(tr),"n_test":len(te),**scores(yte,p),"source":origin,"xgb_candidate":chosen,"inner_val_r2":json.dumps(val_scores),"runtime_seconds":runtime,"prediction_file":str(path.relative_to(ROOT))}
    return m

def reuse(task):
    src=pd.read_csv(PILOT/SAVED[task],dtype={"obs_id":str})
    src=src.loc[src.validation.eq("random") & src.feature_set.eq("X_strict_77")].copy()
    expected=pd.read_csv(OUT/"01_random_split_manifest.csv",dtype={"obs_id":str},usecols=["task","obs_id","split"])
    ids=set(expected.loc[expected.task.eq(task)&expected.split.eq("test"),"obs_id"])
    rows=[]
    for model in ["xgb","tabicl"]:
        s=src.loc[src.model.eq(model)]
        if len(s)!=len(ids) or set(s.obs_id)!=ids: raise ValueError(f"{task} {model}: old prediction IDs differ")
        path=OUT/"03_predictions"/f"{task}_{model}.csv";path.parent.mkdir(parents=True,exist_ok=True)
        p=s.rename(columns={"target":"task","split_source":"source"}).copy();p["validation"]="random_strict";p.to_csv(path,index=False,encoding="utf-8-sig")
        ntrain=int((expected.task.eq(task)&expected.split.eq("train")).sum())
        rows.append({"task":task,"model":model,"n_train":ntrain,"n_test":len(s),**scores(s.y_true,s.y_pred),"source":str((PILOT/SAVED[task]).relative_to(ROOT)),"xgb_candidate":None,"inner_val_r2":"","runtime_seconds":0,"prediction_file":str(path.relative_to(ROOT))})
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--task",choices=list(TASKS)+["all"],default="all");args=parser.parse_args()
    if len(FEATURES)!=77 or "water_zone_code" in FEATURES:raise ValueError("77特征清单异常")
    (OUT/"04_metrics").mkdir(parents=True,exist_ok=True)
    for task in ([args.task] if args.task!="all" else TASKS):
        metric_path=OUT/"04_metrics"/f"{task}.csv"
        if metric_path.exists():print(f"SKIP {task}: metrics exist",flush=True);continue
        start=time.time()
        if task in SAVED: rows=reuse(task)
        else:
            tr,te,xtr,xte,ytr,yte,inverse=arrays(task);rows=[]
            for model in ["xgb","tabicl"]:
                t=time.time()
                if model=="xgb":raw,chosen,vals=xgb_predict(xtr,ytr,xte,task!="chla_surface")
                else:raw=tabicl_predict(xtr,ytr,xte);chosen=None;vals=None
                pred=np.maximum(inverse(raw),0)
                rows.append(save(task,model,tr,te,yte,pred,"本轮训练，冻结 Random Strict seed_42",chosen,vals,time.time()-t))
                print(task,model,rows[-1]["R2"],flush=True)
        pd.DataFrame(rows).to_csv(metric_path,index=False,encoding="utf-8-sig")
        print(f"DONE {task} {time.time()-start:.1f}s",flush=True)

if __name__=="__main__":main()
