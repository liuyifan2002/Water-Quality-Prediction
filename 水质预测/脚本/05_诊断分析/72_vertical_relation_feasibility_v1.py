"""只读取冻结数据及预测，分析真实采样事件的垂向配对和残差。"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
OUT = ROOT / 'runs/vertical_relation_feasibility_v1'
OUT.mkdir(parents=True, exist_ok=True)
DATA = BASE / '02_冻结候选/基础输入_77特征.csv'
PRED = BASE / '03_正式十任务重算/predictions'
MT = ROOT / 'runs/multitask_fttransformer_v1/predictions'
LEVELS = ['surface','middle','bottom']
TARGETS = {'chla':'chla_ug_l','tn':'tn_mg_l','tp':'tp_mg_l'}
PAIRS = [('surface','middle'),('middle','bottom'),('surface','bottom')]

def corr(a,b):
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3 or np.std(a[ok])==0 or np.std(b[ok])==0:
        return np.nan,np.nan
    return float(pearsonr(a[ok],b[ok]).statistic),float(spearmanr(a[ok],b[ok]).statistic)

def metric(y,p):
    return (float(mean_squared_error(y,p)**.5),float(mean_absolute_error(y,p)))

def main():
    d=pd.read_csv(DATA,usecols=['obs_id','sample_id','station','date','depth','chla_ug_l','tn_mg_l','tp_mg_l'])
    assert not d.duplicated(['sample_id','depth']).any()
    assert d.groupby('sample_id')[['station','date']].nunique().max().max()==1
    d['year']=pd.to_datetime(d.date).dt.year
    # Same station ID/date is not used as a join key; sample_id is the source event ID.
    meta=d[['sample_id','station','year']].drop_duplicates('sample_id')
    station_meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv')
    zone_col=next(c for c in station_meta if 'zone' in c.lower() and station_meta[c].nunique()<=20)
    station_col=next(c for c in station_meta if c.lower() in ('station','station_id'))
    meta=meta.merge(station_meta[[station_col,zone_col]].drop_duplicates(station_col),left_on='station',right_on=station_col,how='left')
    meta=meta.rename(columns={zone_col:'water_zone'})[['sample_id','station','year','water_zone']]

    cov=[];rels=[]
    for q,target in TARGETS.items():
        p=d.pivot(index='sample_id',columns='depth',values=target).reindex(columns=LEVELS).join(meta.set_index('sample_id'))
        masks={k:p[k].notna() for k in LEVELS}
        for scope,sub in [('all',p)]+[(f'year:{v}',g) for v,g in p.groupby('year')]+[(f'zone:{v}',g) for v,g in p.groupby('water_zone')]:
            row={'indicator':q,'scope':scope,'events':len(sub),'stations':sub.station.nunique()}
            for lev in LEVELS: row[f'n_{lev}']=int(sub[lev].notna().sum())
            for a,b in PAIRS:
                valid=sub[[a,b]].notna().all(axis=1)
                row[f'n_{a}_{b}']=int(valid.sum());row[f'stations_{a}_{b}']=sub.loc[valid,'station'].nunique()
            all3=sub[LEVELS].notna().all(axis=1)
            row['n_all3']=int(all3.sum());row['stations_all3']=sub.loc[all3,'station'].nunique()
            row['n_exactly2']=int((sub[LEVELS].notna().sum(axis=1)==2).sum())
            cov.append(row)
        for a,b in PAIRS:
            z=p[[a,b,'station']].dropna();x=z[a].to_numpy();y=z[b].to_numpy();r,s=corr(x,y)
            delta=x-y
            rels.append({'indicator':q,'comparison':f'{a}-{b}','n_events':len(z),'n_stations':z.station.nunique(),
                         'pearson':r,'spearman':s,'difference_mean':delta.mean(),'difference_median':np.median(delta),
                         'difference_sd':delta.std(ddof=1),'difference_q25':np.quantile(delta,.25),'difference_q75':np.quantile(delta,.75)})
        z=p[LEVELS+['station']].dropna();curve=z.surface-2*z.middle+z.bottom
        rels.append({'indicator':q,'comparison':'curvature_S-2M+B','n_events':len(z),'n_stations':z.station.nunique(),
                     'pearson':np.nan,'spearman':np.nan,'difference_mean':curve.mean(),'difference_median':curve.median(),
                     'difference_sd':curve.std(),'difference_q25':curve.quantile(.25),'difference_q75':curve.quantile(.75)})
        for lev in LEVELS:
            x=p[lev].dropna()
            rels.append({'indicator':q,'comparison':f'distribution_{lev}','n_events':len(x),'n_stations':p.loc[x.index,'station'].nunique(),
                         'difference_mean':x.mean(),'difference_median':x.median(),'difference_sd':x.std(),
                         'difference_q25':x.quantile(.25),'difference_q75':x.quantile(.75)})
    pd.DataFrame(cov).to_csv(OUT/'01_vertical_pairing_coverage.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rels).to_csv(OUT/'02_vertical_target_relationships.csv',index=False,encoding='utf-8-sig')

    # M0/M1/M2: one record per variant, seed, task, fold and obs_id.
    mt=pd.concat((pd.read_csv(f,usecols=['obs_id','task','fold','seed','variant','y_true','y_pred','station_id']) for f in MT.glob('*.csv')),ignore_index=True)
    mt=mt[mt.task.str.split('_').str[0].isin(TARGETS)].copy()
    mt['sample_id']=mt.obs_id.str.rsplit('_',n=1).str[0]
    mt['depth']=mt.task.str.rsplit('_',n=1).str[-1]
    mt['indicator']=mt.task.str.split('_').str[0]
    mt['residual']=mt.y_true-mt.y_pred
    assert not mt.duplicated(['variant','seed','task','obs_id']).any()
    residual=[]
    for (q,v,seed,fold),g in mt.groupby(['indicator','variant','seed','fold']):
        pv=g.pivot(index='sample_id',columns='depth',values=['residual','y_pred']).reindex(columns=LEVELS,level=1)
        stations=g[['sample_id','station_id']].drop_duplicates('sample_id').set_index('sample_id')
        for a,b in PAIRS:
            z=pv[[('residual',a),('residual',b),('y_pred',a),('y_pred',b)]].dropna()
            if z.empty:continue
            ra=z[('residual',a)].to_numpy();rb=z[('residual',b)].to_numpy()
            r,s=corr(ra,rb);r_pred,_=corr(ra,z[('y_pred',b)].to_numpy())
            r_delta,_=corr(ra,(z[('y_pred',a)]-z[('y_pred',b)]).to_numpy())
            residual.append({'indicator':q,'variant':v,'seed':seed,'fold':fold,'pair':f'{a}-{b}',
                             'n_events':len(z),'n_stations':stations.loc[z.index,'station_id'].nunique(),
                             'pearson':r,'spearman':s,'mean_residual_a':ra.mean(),'mean_residual_b':rb.mean(),
                             'same_sign_fraction':float(np.mean(np.sign(ra)==np.sign(rb))),
                             'residual_a_vs_other_prediction_pearson':r_pred,
                             'residual_a_vs_predicted_vertical_difference_pearson':r_delta})
        z=pv['residual'][LEVELS].dropna()
        if len(z):
            r=z.to_numpy()
            residual.append({'indicator':q,'variant':v,'seed':seed,'fold':fold,'pair':'all3_same_sign',
                             'n_events':len(z),'n_stations':stations.loc[z.index,'station_id'].nunique(),
                             'same_sign_fraction':float(np.mean((r>0).all(axis=1)|(r<0).all(axis=1)))})
    pd.DataFrame(residual).to_csv(OUT/'03_vertical_residual_relationships.csv',index=False,encoding='utf-8-sig')

    # Station-holdout XGBoost and TabICLv2 are the only verified same-version experts here.
    fs=list(PRED.glob('*.csv'))
    use=[f for f in fs if f.stem.split('_')[0] in TARGETS and ('_xgboost_' in f.stem or '_tabiclv2_' in f.stem)]
    expert=pd.concat((pd.read_csv(f,usecols=['task','obs_id','station_id','fold','model','y_true','y_pred','dataset_version']) for f in use),ignore_index=True)
    expert=expert[expert.dataset_version.eq('WQ77_REPAIRED_20261115_V1')].copy()
    expert['sample_id']=expert.obs_id.str.rsplit('_',n=1).str[0]
    expert['indicator']=expert.task.str.split('_').str[0]
    expert['depth']=expert.task.str.rsplit('_',n=1).str[-1]
    ex=[]
    for task,g in expert.groupby('task'):
        wide=g.pivot(index='obs_id',columns='model',values=['y_true','y_pred']).dropna()
        if len(wide)==0:continue
        assert np.allclose(wide['y_true'].xgboost,wide['y_true'].tabiclv2)
        y=wide['y_true'].xgboost.to_numpy();x=wide['y_pred'].xgboost.to_numpy();t=wide['y_pred'].tabiclv2.to_numpy()
        rx=y-x;rt=y-t;r,s=corr(rx,rt)
        rmse_x,mae_x=metric(y,x);rmse_t,mae_t=metric(y,t)
        ex.append({'scope':'task','indicator':task.rsplit('_',1)[0],'task':task,'n_events':len(wide),
                   'xgb_RMSE':rmse_x,'tabicl_RMSE':rmse_t,'xgb_MAE':mae_x,'tabicl_MAE':mae_t,
                   'expert_residual_pearson':r,'expert_residual_spearman':s,
                   'tabicl_lower_abs_error_fraction':float(np.mean(np.abs(rt)<np.abs(rx))),
                   'oracle_RMSE_DIAGNOSTIC_ONLY':metric(y,np.where(np.abs(rx)<np.abs(rt),x,t))[0]})
        # Zone differences are descriptive only; no gate is fitted and zone is not an input.
        zones=g[['obs_id','sample_id']].drop_duplicates('obs_id').set_index('obs_id').loc[wide.index]
        zones=zones.join(meta.set_index('sample_id')[['water_zone']],on='sample_id')
        for zone,ids in zones.groupby('water_zone').groups.items():
            ix=wide.index.get_indexer(ids)
            if len(ix)<20:continue
            ex.append({'scope':'task_water_zone','indicator':task.rsplit('_',1)[0],'task':task,
                       'water_zone':zone,'n_events':len(ix),
                       'tabicl_lower_abs_error_fraction':float(np.mean(np.abs(rt[ix])<np.abs(rx[ix]))),
                       'expert_residual_pearson':corr(rx[ix],rt[ix])[0]})
    for q,g in expert.groupby('indicator'):
        # A full three-layer win pattern requires all six predictions on the same event.
        w=g.pivot(index='sample_id',columns=['depth','model'],values='y_pred')
        y=g.drop_duplicates(['sample_id','depth']).pivot(index='sample_id',columns='depth',values='y_true')
        if not set((lev,m) for lev in LEVELS for m in ('xgboost','tabiclv2')).issubset(w.columns):continue
        z=w.dropna(subset=[(lev,m) for lev in LEVELS for m in ('xgboost','tabiclv2')])
        z=z.loc[z.index.intersection(y.dropna(subset=LEVELS).index)]
        if z.empty:continue
        winners=[]
        for lev in LEVELS:
            truth=y.loc[z.index,lev]
            winners.append((z[(lev,'tabiclv2')]-truth).abs().to_numpy() < (z[(lev,'xgboost')]-truth).abs().to_numpy())
        a=np.column_stack(winners)
        ex.append({'scope':'three_layer_event','indicator':q,'task':'all3','n_events':len(z),
                   'tabicl_wins_all3_fraction':float(a.all(axis=1).mean()),
                   'xgb_wins_all3_fraction':float((~a).all(axis=1).mean()),
                   'mixed_expert_winners_fraction':float((~a.all(axis=1)&a.any(axis=1)).mean())})
    pd.DataFrame(ex).to_csv(OUT/'04_expert_complementarity.csv',index=False,encoding='utf-8-sig')
    # One compact figure; each cell is a true-event paired Pearson correlation.
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,3,figsize=(11,5.5),layout='constrained')
    names={'chla':'Chl-a','tn':'TN','tp':'TP'}
    order=[f'{a}-{b}' for a,b in PAIRS]
    tr=pd.DataFrame(rels);rr=pd.DataFrame(residual)
    for j,q in enumerate(TARGETS):
        vals=tr[(tr.indicator==q)&tr.comparison.isin(order)].set_index('comparison').reindex(order).pearson.to_numpy()
        rv=rr[(rr.indicator==q)&(rr.variant=='M1')&rr.pair.isin(order)].groupby('pair').pearson.mean().reindex(order).to_numpy()
        for i,(data,title) in enumerate([(vals,'真实观测层间相关'),(rv,'M1残差层间相关')]):
            ax=axes[i,j];ax.imshow(data[None,:],vmin=0,vmax=1,cmap='Blues',aspect='auto')
            ax.set_xticks(range(3),['表—中','中—底','表—底']);ax.set_yticks([])
            for k,v in enumerate(data):ax.text(k,0,f'{v:.2f}',ha='center',va='center',color='white' if v>.55 else '#19324a',fontsize=10)
            ax.set_title(f'{names[q]}：{title}',fontsize=11);ax.spines[:].set_visible(False)
    fig.savefig(OUT/'垂向观测与M1残差相关性.png',dpi=300,facecolor='white')
    plt.close(fig)
    print('coverage',len(cov),'target',len(rels),'residual',len(residual),'expert',len(ex))

if __name__=='__main__':main()
