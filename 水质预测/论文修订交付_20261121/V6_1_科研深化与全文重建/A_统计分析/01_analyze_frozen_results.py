"""只读冻结文件，生成 V6.1 的配对统计和数据覆盖证据。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / '结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
OUT = Path(__file__).resolve().parent
VERSION = 'WQ77_REPAIRED_20261115_V1'
TASKS = ['chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']
N_BOOT, SEED = 1000, 20260921

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def metric(y,p):
    denominator=np.sum((y-y.mean())**2)
    return {'R2':1-np.sum((y-p)**2)/denominator if denominator>0 else np.nan,
            'RMSE':np.sqrt(np.mean((y-p)**2)),'MAE':np.mean(np.abs(y-p))}

def get_paired(task):
    dr=SRC/'03_正式十任务重算/predictions'
    frames=[]; inputs=[]
    for k in range(1,6):
        a=dr/f'{task}_xgboost_fold_{k}.csv'; b=dr/f'{task}_tabiclv2_fold_{k}.csv'
        x=pd.read_csv(a); t=pd.read_csv(b); inputs.extend([a,b])
        for d in (x,t):
            assert len(d)==len(d.obs_id.unique())
            assert set(d.dataset_version)=={VERSION}
        cols=['obs_id','station_id','date','fold','y_true','manifest_sha256']
        z=x[cols+['y_pred']].merge(t[cols+['y_pred']],on='obs_id',suffixes=('_x','_t'),validate='one_to_one')
        assert len(z)==len(x)==len(t)
        for c in ['station_id','date','fold','manifest_sha256']:
            assert (z[c+'_x'].astype(str)==z[c+'_t'].astype(str)).all(),(task,k,c)
        assert np.allclose(z.y_true_x,z.y_true_t,rtol=0,atol=0)
        z=z.rename(columns={'station_id_x':'station_id','fold_x':'fold','y_true_x':'y_true','y_pred_x':'xgb','y_pred_t':'tabicl'})
        frames.append(z[['obs_id','station_id','fold','y_true','xgb','tabicl']])
    p=pd.concat(frames,ignore_index=True)
    assert len(p)==len(p.obs_id.unique())
    return p,inputs

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=[]; results=[]; stations=[]; folds=[]
    for task in TASKS:
        p,files=get_paired(task)
        evidence.extend({'path':str(f.relative_to(ROOT)),'sha256':sha(f)} for f in files)
        y=p.y_true.to_numpy(float); x=p.xgb.to_numpy(float); t=p.tabicl.to_numpy(float)
        mx,mt=metric(y,x),metric(y,t)
        groups={s:g.index.to_numpy() for s,g in p.groupby('station_id',sort=True)}
        labels=np.array(sorted(groups))
        rng=np.random.default_rng(SEED)
        vals=[]
        for _ in range(N_BOOT):
            sampled=rng.choice(labels,size=len(labels),replace=True)
            ix=np.concatenate([groups[s] for s in sampled])
            a,b=metric(y[ix],x[ix]),metric(y[ix],t[ix])
            vals.append((b['RMSE']-a['RMSE'],b['MAE']-a['MAE']))
        v=np.array(vals)
        row={'task':task,'dataset_version':VERSION,'n_obs':len(p),'n_station':len(labels),'n_bootstrap':N_BOOT,'seed':SEED,
             'XGBoost_pooled_R2':mx['R2'],'TabICLv2_pooled_R2':mt['R2'],'delta_pooled_R2':mt['R2']-mx['R2'],
             'XGBoost_pooled_RMSE':mx['RMSE'],'TabICLv2_pooled_RMSE':mt['RMSE'],'delta_pooled_RMSE':mt['RMSE']-mx['RMSE'],
             'XGBoost_pooled_MAE':mx['MAE'],'TabICLv2_pooled_MAE':mt['MAE'],'delta_pooled_MAE':mt['MAE']-mx['MAE']}
        for j,key in enumerate(['RMSE','MAE']):
            lo,hi=np.quantile(v[:,j],[.025,.975])
            row.update({f'delta_{key}_bootstrap_median':np.median(v[:,j]),f'delta_{key}_CI_low':lo,f'delta_{key}_CI_high':hi,
                        f'delta_{key}_CI_cross_zero':bool(lo<=0<=hi)})
        for s,g in p.groupby('station_id',sort=True):
            a,b=metric(g.y_true.to_numpy(float),g.xgb.to_numpy(float)),metric(g.y_true.to_numpy(float),g.tabicl.to_numpy(float))
            stations.append({'task':task,'station_id':s,'n_obs':len(g),'xgb_RMSE':a['RMSE'],'tabicl_RMSE':b['RMSE'],
                             'delta_RMSE':b['RMSE']-a['RMSE'],'xgb_MAE':a['MAE'],'tabicl_MAE':b['MAE'],'delta_MAE':b['MAE']-a['MAE']})
        row['station_RMSE_win_rate']=np.mean([r['delta_RMSE']<0 for r in stations if r['task']==task])
        for fold,g in p.groupby('fold',sort=True):
            a,b=metric(g.y_true.to_numpy(float),g.xgb.to_numpy(float)),metric(g.y_true.to_numpy(float),g.tabicl.to_numpy(float))
            folds.append({'task':task,'fold':fold,'n_obs':len(g),'delta_R2':b['R2']-a['R2'],'delta_RMSE':b['RMSE']-a['RMSE'],'delta_MAE':b['MAE']-a['MAE']})
        results.append(row)
    r=pd.DataFrame(results); r.to_csv(OUT/'十任务Bootstrap结果.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(stations).to_csv(OUT/'逐站点配对误差.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(folds).to_csv(OUT/'逐折配对指标.csv',index=False,encoding='utf-8-sig')
    # 原始冻结表的空值是填补前状态；不可把有效填补矩阵误当成观测覆盖。
    base=SRC/'02_冻结候选/基础输入_77特征.csv'; featurefile=SRC/'02_冻结候选/77项正式输入清单.csv'
    frame=pd.read_csv(base); fd=pd.read_csv(featurefile)
    names=[c for c in fd.columns if 'feature' in c.lower() or '特征' in c]
    assert names,fd.columns.tolist()
    features=fd[names[0]].astype(str).tolist(); assert len(features)==77 and len(set(features))==77
    assert all(c in frame.columns for c in features)
    frame['date']=pd.to_datetime(frame.date)
    cov=[]
    for f in features:
        v=frame[f]; valid=v.notna()
        cov.append({'feature':f,'n_rows':len(v),'n_nonmissing':int(valid.sum()),'coverage_rate':float(valid.mean()),
                    'n_stations_valid':frame.loc[valid,'station'].nunique(),'n_years_valid':frame.loc[valid,'date'].dt.year.nunique(),
                    'first_date':str(frame.loc[valid,'date'].min().date()) if valid.any() else '',
                    'last_date':str(frame.loc[valid,'date'].max().date()) if valid.any() else '',
                    'n_unique_nonmissing':v.nunique(dropna=True)})
    pd.DataFrame(cov).to_csv(OUT/'77特征真实观测覆盖.csv',index=False,encoding='utf-8-sig')
    targetsites=['VT11','SM18','SM19','PT3']
    cs=[]
    for s in targetsites:
        g=frame[frame.station==s]
        for f in features:
            cs.append({'station':s,'feature':f,'n_rows':len(g),'n_nonmissing':g[f].notna().sum(),
                       'coverage_rate':g[f].notna().mean() if len(g) else np.nan,
                       'first_date':str(g.date.min().date()) if len(g) else '',
                       'last_date':str(g.date.max().date()) if len(g) else ''})
    pd.DataFrame(cs).to_csv(OUT/'四站当前特征覆盖.csv',index=False,encoding='utf-8-sig')
    evidence.extend([{'path':str(f.relative_to(ROOT)),'sha256':sha(f)} for f in [base,featurefile,SRC/'08_重算结果核验/新候选_pooled与逐折指标.csv']])
    pd.DataFrame(evidence).to_csv(OUT/'证据路径与SHA256.csv',index=False,encoding='utf-8-sig')
    print(r[['task','delta_pooled_R2','delta_pooled_RMSE','delta_RMSE_CI_low','delta_RMSE_CI_high','station_RMSE_win_rate']].to_string(index=False))
    print('coverage',len(frame),frame.station.nunique(),frame.date.min(),frame.date.max(),len(features))

if __name__=='__main__': main()
