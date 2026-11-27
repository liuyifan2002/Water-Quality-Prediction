"""Read-only Random Strict training-OOF mechanism screen; no model fit or test scoring."""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
ZOO = ROOT / 'runs/random_tabular_model_zoo_v2'
OUT = ROOT / 'runs/mechanism_screen_v1'
OUT.mkdir(parents=True, exist_ok=True)
REPAIRED = ROOT / '结果/扩展实验/候选数据验收与一致性重算_v1_20261115/02_冻结候选/基础输入_77特征.csv'
OLD = ROOT / '数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv'
SPLIT = ROOT / 'runs/random_10task_v1/01_random_split_manifest.csv'
INNER = ZOO / '01_inner_fold_manifest.csv'
TASKS = ['chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']
TARGETS = {'chla':'chla_ug_l','tn':'tn_mg_l','tp':'tp_mg_l','secchi':'secchi_depth_m'}
ENV = {'rain_14d':'降雨水文潮汐','sst_mean_7d':'海洋物理','wind_speed':'气象',
       'ntl_avg_rad':'人类活动','lat':'空间静态','month_sin':'时间过程'}
MODELS = ['catboost','extra_trees','hist_gb','lightgbm','mlp','random_forest',
          'ridge','svr_rbf','tabicl','tabm','tabpfn','xgboost']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(y, p):
    return float(r2_score(y,p)), float(np.sqrt(mean_squared_error(y,p))), float(mean_absolute_error(y,p))


def safe_corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    return float(spearmanr(a[m],b[m]).statistic) if m.sum()>=20 and np.unique(a[m]).size>2 and np.unique(b[m]).size>2 else np.nan


def load_oof(task, valid_ids, inner):
    z=[]
    for model in MODELS:
        path=ZOO/'02_inner_oof'/f'{task}_{model}.csv'
        if not path.exists():continue
        x=pd.read_csv(path,usecols=['obs_id','inner_fold','y_true','y_pred'])
        if x.obs_id.duplicated().any() or set(x.obs_id)!=valid_ids:continue
        x=x.merge(inner,on='obs_id',validate='one_to_one',suffixes=('','_manifest'))
        if not x.inner_fold.eq(x.inner_fold_manifest).all():continue
        x=x.drop(columns='inner_fold_manifest').set_index('obs_id').sort_index()
        x['model']=model
        z.append(x)
    if len(z)<2:return None
    base=z[0][['y_true','inner_fold']].rename(columns={'inner_fold':'fold'})
    for part in z[1:]:
        assert np.allclose(base.y_true,part.y_true.reindex(base.index))
        assert base.fold.equals(part.inner_fold.reindex(base.index).rename('fold'))
    pred=pd.concat([x.y_pred.rename(x.model.iloc[0]) for x in z],axis=1)
    assert pred.notna().all().all()
    return base.join(pred)


def target_summary(task, d):
    y=d.y_true.to_numpy(float);pos=y[y>0]; near=.01*np.median(pos) if len(pos) else 0
    qs=np.quantile(y,[.01,.05,.25,.5,.75,.95,.99])
    edges=np.linspace(float(y.min()),float(y.max()),11)
    counts=np.histogram(y,bins=edges)[0].tolist()
    return dict(task=task,unit=('μg/L' if task.startswith('chla') else 'm' if task=='secchi' else 'mg/L'),
                n=len(y),mean=y.mean(),std=y.std(ddof=1),median=np.median(y),minimum=y.min(),maximum=y.max(),
                Q01=qs[0],Q05=qs[1],Q25=qs[2],Q50=qs[3],Q75=qs[4],Q95=qs[5],Q99=qs[6],
                skew=float(skew(y)),kurtosis_excess=float(kurtosis(y)),zero_rate=float(np.mean(y==0)),
                near_zero_threshold=near,near_zero_rate=float(np.mean(y<=near)),
                fixed_width_10bin_counts=json.dumps(counts))


def expert_pairs(task, d, station):
    models=[m for m in MODELS if m in d]
    y=d.y_true.to_numpy();scores={m:metric(y,d[m].to_numpy()) for m in models}
    rows=[]
    for a,b in combinations(models,2):
        pa=d[a].to_numpy();pb=d[b].to_numpy();ea=np.abs(y-pa);eb=np.abs(y-pb)
        blend=metric(y,(pa+pb)/2)
        oracle=metric(y,np.where(ea<=eb,pa,pb))
        st=pd.DataFrame({'station':station.reindex(d.index).to_numpy(),'ea':ea,'eb':eb}).groupby('station')[['ea','eb']].apply(lambda q:(q.ea.pow(2).mean()<q.eb.pow(2).mean())).mean()
        fold_wins=[]
        for _,g in d.groupby('fold'):
            fold_wins.append(metric(g.y_true,g[a])[0]>metric(g.y_true,g[b])[0])
        rows.append(dict(task=task,model_a=a,model_b=b,n=len(d),a_R2=scores[a][0],b_R2=scores[b][0],
                         a_RMSE=scores[a][1],b_RMSE=scores[b][1],a_MAE=scores[a][2],b_MAE=scores[b][2],
                         residual_spearman=safe_corr(y-pa,y-pb),absolute_error_spearman=safe_corr(ea,eb),
                         sample_a_win_rate=float(np.mean(ea<eb)),station_a_rmse_win_rate=float(st),
                         equal_blend_R2=blend[0],blend_minus_best_R2=blend[0]-max(scores[a][0],scores[b][0]),
                         oracle_R2_theoretical_only=oracle[0],folds_a_better=sum(fold_wins),
                         evidence_version='historical_main_table_random_strict_OOF'))
    return rows,scores


def env_preferences(task,d,old_features,station,scores):
    best=sorted(scores,key=lambda m:scores[m][0],reverse=True)[:2]
    if len(best)<2:return []
    a,b=best;z=d.join(old_features.reindex(d.index))
    rows=[]
    for feature,group in ENV.items():
        vals=pd.to_numeric(z[feature],errors='coerce');finite=vals.dropna()
        if finite.nunique()<4:continue
        cuts=np.unique(np.quantile(finite,[0,1/3,2/3,1]))
        if len(cuts)!=4:continue
        bins=pd.cut(vals,bins=cuts,labels=['低','中','高'],include_lowest=True,duplicates='drop')
        for label in ['低','中','高']:
            for fold in ['pooled',1,2,3,4,5]:
                mask=bins.eq(label)&(z.fold.eq(fold) if fold!='pooled' else True)
                q=z[mask]
                if len(q)<10:continue
                _,ra,ma=metric(q.y_true,q[a]);_,rb,mb=metric(q.y_true,q[b])
                rows.append(dict(task=task,model_a=a,model_b=b,feature=feature,source_group=group,
                                 bin=label,fold=fold,n=len(q),n_stations=station.reindex(q.index).nunique(),
                                 lower=cuts[['低','中','高'].index(label)],upper=cuts[['低','中','高'].index(label)+1],
                                 a_RMSE=ra,b_RMSE=rb,a_MAE=ma,b_MAE=mb,a_minus_b_RMSE=ra-rb,
                                 a_minus_b_MAE=ma-mb,evidence_version='historical_main_table_random_strict_OOF'))
    return rows


def bin_errors(task,d,scores):
    best=sorted(scores,key=lambda m:scores[m][0],reverse=True)[:3]
    y=d.y_true.to_numpy(float)
    quant=pd.qcut(y,q=5,labels=['Q1','Q2','Q3','Q4','Q5'],duplicates='drop')
    edges=np.linspace(y.min(),y.max(),11)
    fixed=pd.cut(y,bins=edges,labels=[f'W{i}' for i in range(1,11)],include_lowest=True)
    q95=np.quantile(y,.95);q99=np.quantile(y,.99)
    bins=[('equal_frequency',quant),('fixed_width',fixed),('upper_5pct',pd.Series(np.where(y>=q95,'top5','other'),index=d.index)),('upper_1pct',pd.Series(np.where(y>=q99,'top1','other'),index=d.index))]
    rows=[]
    for scheme,labels in bins:
        labels=np.asarray(labels.astype(str))
        for label in sorted(set(labels)-{'nan'}):
            if scheme.startswith('upper') and label=='other':continue
            for fold in ['pooled',1,2,3,4,5]:
                sel=(labels==label)&((d.fold.to_numpy()==fold) if fold!='pooled' else True)
                if not sel.any():continue
                yy=y[sel]
                for model in best:
                    e=d[model].to_numpy()[sel]-yy
                    rows.append(dict(task=task,model=model,scheme=scheme,bin=label,fold=fold,n=len(yy),
                                     y_min=float(yy.min()),y_max=float(yy.max()),MAE=float(np.abs(e).mean()),
                                     RMSE=float(np.sqrt(np.mean(e*e))),mean_bias=float(e.mean()),
                                     under_rate=float(np.mean(e<0)),over_rate=float(np.mean(e>0)),
                                     evidence_version='historical_main_table_random_strict_OOF'))
    return rows


def decide(task, d, scores, pairrows, envrows, binrows):
    ranked=sorted(scores,key=lambda m:scores[m][0],reverse=True)
    a,b=ranked[:2]
    pair=next(x for x in pairrows if set((x['model_a'],x['model_b']))==set((a,b)))
    er=pd.DataFrame([x for x in envrows if x['fold']=='pooled'])
    folds=pd.DataFrame([x for x in envrows if x['fold']!='pooled'])
    crossover=[]
    if len(er):
        for feature,g in er.groupby('feature'):
            # Values of these three fields are unchanged, or nearly unchanged,
            # between the old model input and the repaired candidate.
            if feature not in {'lat','month_sin','wind_speed'}:continue
            if len(g)<3 or len(folds[folds.feature.eq(feature)])<10:continue
            f=folds[folds.feature.eq(feature)]
            signs=f.pivot(index='fold',columns='bin',values='a_minus_b_RMSE')
            stable_pos=[c for c in signs if (signs[c]>0).sum()>=4]
            stable_neg=[c for c in signs if (signs[c]<0).sum()>=4]
            if stable_pos and stable_neg:
                crossover.append((feature,','.join(stable_pos),','.join(stable_neg)))
    td=pd.DataFrame(binrows); top=td[(td.model==a)&(td.scheme=='upper_5pct')&(td.bin=='top5')]
    mid=td[(td.model==a)&(td.scheme=='equal_frequency')&(td.bin=='Q3')]
    pooled_top=top[top.fold.eq('pooled')].iloc[0];pooled_mid=mid[mid.fold.eq('pooled')].iloc[0]
    stable_tail=0
    for f in range(1,6):
        h=top[top.fold.eq(f)];k=mid[mid.fold.eq(f)]
        if len(h) and len(k) and h.iloc[0].MAE>k.iloc[0].MAE and h.iloc[0].mean_bias<0:stable_tail+=1
    # Historical-model mechanism findings cannot be promoted to repaired-version GO.
    dynamic='CONDITIONAL' if pair['oracle_R2_theoretical_only']-max(scores[a][0],scores[b][0])>.03 and crossover else 'NO-GO'
    target='CONDITIONAL' if stable_tail>=4 and pooled_top.MAE>1.5*pooled_mid.MAE else 'NO-GO'
    return dict(task=task,dynamic_expert=dynamic,target_aware=target,top_expert=a,second_expert=b,
                top_R2=scores[a][0],second_R2=scores[b][0],blend_minus_best_R2=pair['blend_minus_best_R2'],
                oracle_gap_theoretical=pair['oracle_R2_theoretical_only']-max(scores[a][0],scores[b][0]),
                environment_crossover=';'.join(f'{n}:A劣于B({pos}),A优于B({neg}),各>=4/5折' for n,pos,neg in crossover),
                top5_MAE_to_Q3_ratio=pooled_top.MAE/pooled_mid.MAE,top5_underbias_folds=stable_tail,
                reason='旧主表模型OOF与修复版输入不一致，最多作为下一阶段候选线索')


def main():
    assert sha(SPLIT)=='bb9269039de23f8fb4412eb71bf8268a972ec40ca17c3d0b333a6e2fa791b359'
    assert sha(INNER)=='c4fea605966709a29baf014ed7eb6c8c557d06122728017ea127ef9ae21213cc'
    split=pd.read_csv(SPLIT,usecols=['task','obs_id','station','split'])
    inner=pd.read_csv(INNER)
    old=pd.read_csv(OLD,usecols=['obs_id',*ENV.keys()]).set_index('obs_id')
    # The repaired target values are accessed only after restricting to external-training IDs.
    trainids=set(split.loc[split.split.eq('train'),'obs_id'])
    rep=[]
    for chunk in pd.read_csv(REPAIRED,usecols=['obs_id',*TARGETS.values()],chunksize=10000):
        rep.append(chunk[chunk.obs_id.isin(trainids)])
    repaired=pd.concat(rep).set_index('obs_id')
    all_pair=[];all_env=[];all_dist=[];all_bin=[];all_dec=[]
    for task in TASKS:
        s=split[split.task.eq(task)]
        valid=set(s.loc[s.split.eq('train'),'obs_id'])
        fold=inner[inner.task.eq(task)][['obs_id','inner_fold']]
        assert set(fold.obs_id)==valid
        d=load_oof(task,valid,fold)
        if d is None:
            all_dec.append(dict(task=task,dynamic_expert='NO-GO',target_aware='NO-GO',reason='缺少可配对训练OOF'))
            continue
        target=TARGETS[task.split('_')[0]]
        assert np.allclose(d.y_true,repaired.loc[d.index,target],equal_nan=False)
        station=s.set_index('obs_id').station
        prs,scores=expert_pairs(task,d,station);all_pair+=prs
        env=env_preferences(task,d,old,station,scores);all_env+=env
        bins=bin_errors(task,d,scores);all_bin+=bins
        all_dist.append(target_summary(task,d));all_dec.append(decide(task,d,scores,prs,env,bins))
        print(task,'n',len(d),'models',len(scores),'best',max(scores,key=lambda m:scores[m][0]),flush=True)
    for filename,rows in [('01_expert_complementarity.csv',all_pair),('02_environment_condition_preference.csv',all_env),
                          ('03_target_distribution.csv',all_dist),('04_target_bin_oof_error.csv',all_bin),
                          ('05_ten_task_mechanism_decision.csv',all_dec)]:
        pd.DataFrame(rows).to_csv(OUT/filename,index=False,encoding='utf-8-sig')
    print('DONE',len(all_pair),len(all_env),len(all_dist),len(all_bin),len(all_dec),flush=True)


if __name__=='__main__':main()
