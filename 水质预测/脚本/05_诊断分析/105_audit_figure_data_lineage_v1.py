"""逐层核对图片数据完整性；只复算连接、特征和支持计数，不导入/调用模型。"""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/图片数据完整性整改_v1_20261115';E=OUT/'证据'
E.mkdir(parents=True,exist_ok=True)
hashes={}
def read(rel,**kwargs):
    p=Path(rel);p=p if p.is_absolute() else ROOT/p
    hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,low_memory=False,**kwargs)
def dump(name,x): (E/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def csv(name,x):pd.DataFrame(x).to_csv(E/name,index=False,encoding='utf-8-sig')
def equal(a,b):return np.isclose(np.asarray(a,float),np.asarray(b,float),atol=1e-9,rtol=1e-9,equal_nan=True)
MAIN='数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv'
main=read(MAIN);main.date=pd.to_datetime(main.date)
raw=read('数据/处理后数据/正式主表/viirs_ntl_station_monthly_2014_2024.csv')
meta=read('数据/处理后数据/正式主表/station_meta.csv').rename(columns={'Station':'station'})
legacy=read('数据/处理后数据/衍生版本/gee_station_months_surface_2014_2024.csv')
strict=read('runs/tabiclv2_pilot/01_strict_feature_manifest.csv')
features=strict.loc[strict.included_in_pilot.astype(str).str.lower().eq('true'),'feature'].tolist()
assert len(features)==len(set(features))==77 and 'water_zone_code' not in features
dump('正式77特征顺序.json',features)
assert not raw.duplicated(['station','year','month']).any()
assert not main.obs_id.duplicated().any()
sample=main.drop_duplicates('sample_id').sort_values(['station','date']).reset_index(drop=True)
joint=sample[['sample_id','station','date','year','month','ntl_avg_rad']].merge(raw,on=['station','year','month'],how='left',validate='many_to_one',suffixes=('_main','_raw'),indicator=True)
assert equal(joint.ntl_avg_rad_main,joint.ntl_avg_rad_raw).all()
assert (sample.date.dt.year==sample.year).all() and (sample.date.dt.month==sample.month).all()
prev=sample.groupby('station').date.shift(2)
expected=sample.groupby('station').ntl_avg_rad.shift(2)
assert equal(expected,sample.ntl_avg_rad_lag2).all()
lags=sample[['sample_id','station','date','ntl_avg_rad_lag2']].copy()
lags['lag2来源日期']=prev;lags['lag2间隔天数']=(sample.date-prev).dt.days;lags['重新构造值']=expected
lags['同站有两次历史采样']=prev.notna();csv('lag2逐采样时点对齐核对.csv',lags)
profiles=[]
for label,d in [('原始8站月表',raw),('主表水层观测',main),('独立站点采样时点',sample)]:
    for f in ['ntl_avg_rad','ntl_avg_rad_lag2']:
        if f not in d:continue
        profiles.append({'层级':label,'变量':f,'记录数':len(d),'站点数':d.station.nunique(),'非缺失数':int(d[f].notna().sum()),'缺失数':int(d[f].isna().sum()),'缺失率':d[f].isna().mean(),'唯一值数':d[f].nunique(),'零值数':int(d[f].eq(0).sum()),'非缺失站点数':d.loc[d[f].notna(),'station'].nunique()})
csv('夜光分层缺失与唯一值.csv',profiles)
by_station=main.groupby('station').agg(观测数=('obs_id','size'),夜光非缺失=('ntl_avg_rad','count'),夜光lag2非缺失=('ntl_avg_rad_lag2','count'),起始日期=('date','min'),结束日期=('date','max'))
by_station['夜光缺失率']=1-by_station.夜光非缺失/by_station.观测数
csv('94站夜光覆盖.csv',by_station.reset_index())
years=main.groupby(main.date.dt.year).agg(观测数=('obs_id','size'),夜光非缺失=('ntl_avg_rad','count'),lag2非缺失=('ntl_avg_rad_lag2','count'))
csv('逐年夜光覆盖.csv',years.reset_index())
existing=set(map(tuple,raw[['station','year','month']].to_numpy()))
requests=sample[['station','year','month']].drop_duplicates();requests['源表有记录']=[tuple(v) in existing for v in requests.to_numpy()]
csv('真实监测月份连接覆盖.csv',requests)
cloud=raw.loc[raw.ntl_cf_cvg.le(0)].copy();csv('无有效晴空观测的原始零值.csv',cloud)
duplicates=sample.duplicated(['station','date']).sum()
assert duplicates==0
lag_stats=lags.lag2间隔天数.dropna().describe(percentiles=[.05,.25,.5,.75,.95]).to_dict()
summary={'主表形状':list(main.shape),'采样时点数':len(sample),'水层复用同一采样ID且环境值一致':int(main.groupby('sample_id').ntl_avg_rad.nunique(dropna=False).max())==1,'原始月表记录':len(raw),'原始覆盖站点':raw.station.unique().tolist(),'原始月表与旧8站请求键完全相同':existing==set(map(tuple,legacy[['station','year','month']].to_numpy())),'真实监测站点月份数':len(requests),'监测月份缺少源表记录':int((~requests.源表有记录).sum()),'原始cf_cvg为0的月记录':len(cloud),'这些零值影响水层记录':int(main.merge(cloud[['station','year','month']],on=['station','year','month']).shape[0]),'主表夜光连接值不一致数':0,'lag2错误数':0,'lag2间隔天数':lag_stats,'lag2定义':'同站点按实际采样日期排序后，向前第2个独立采样时点；不是固定2个月、不是2天，不按水层重复位移','数据源':'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG','单位':'nW cm^-2 sr^-1','空间提取':'1000m缓冲区均值，scale=500m；原生像元463.83m','自然月匹配潜在边界':'当月合成覆盖整个自然月，包括月内采样日之后影像；用于历史同期匹配，不构成实时提前预报可用性证明'}
dump('夜光数据根因摘要.json',summary)

# 重建历史参考选择与分箱计数。只对已有CSV核验，不复算任何预测或ALE。
HIST='runs/tabicl_v2_explainability_v3/ale_1d/ale_1d_by_fold.csv'
hist=read(HIST)
reps=['chla_surface','secchi','tn_surface','tp_middle']
specs={'chla_surface':('chla_ug_l','surface','runs/chla_family_v2/03_chla_surface_manifest.csv'),'secchi':('secchi_depth_m',None,'runs/modern_baseline_gate_v1/01_secchi_manifest.csv'),'tn_surface':('tn_mg_l','surface','runs/tn_family_v2/03_tn_surface_manifest.csv'),'tp_middle':('tp_mg_l','middle','runs/tp_family_v2/04_tp_middle_manifest.csv')}
def select(frame,n,seed):
    rng=np.random.default_rng(seed);pools={k:list(v) for k,v in frame.groupby('station_id',sort=True).groups.items()};selected=[]
    while len(selected)<min(n,len(frame)):
        changed=False
        for station in sorted(pools):
            available=[i for i in pools[station] if i not in selected]
            if available:
                selected.append(int(rng.choice(available)));changed=True
                if len(selected)>=min(n,len(frame)):break
        if not changed:break
    return np.asarray(selected,int)
curve_checks=[];bins=[];queryrows=[];fullfeatures=[]
for task in reps:
    target,depth,mp=specs[task];man=read(mp)
    d=main.loc[main[target].notna() & (main.depth.eq(depth) if depth else True)].copy()
    d['station_id']=d.station
    for fold in [f'fold_{k}' for k in range(1,6)]:
        m=man[man.fold.eq(fold)];trainids=m.loc[m.role.eq('train'),'obs_id'].astype(str).tolist();testids=m.loc[m.role.eq('test'),'obs_id'].astype(str).tolist()
        if task=='secchi':
            ix=d.set_index('obs_id',drop=False);train=ix.loc[trainids].reset_index(drop=True);test=ix.loc[testids].reset_index(drop=True)
        else:
            train=d[d.obs_id.isin(set(trainids))].reset_index(drop=True);test=d[d.obs_id.isin(set(testids))].reset_index(drop=True)
        assert len(train)==len(trainids) and len(test)==len(testids)
        idx=select(test,64,30000+int(fold[-1])+100*reps.index(task));q=test.iloc[idx]
        for feature,g in hist[(hist.task==task)&(hist.fold==fold)].groupby('feature',sort=False):
            g=g.sort_values('bin');tr=train[feature].to_numpy(float);te=test[feature].to_numpy(float);median=float(np.nanmedian(tr)) if np.isfinite(tr).any() else 0.
            xtr=np.where(np.isfinite(tr),tr,median).astype('float32');xte=np.where(np.isfinite(te),te,median).astype('float32');xr=xte[idx]
            edges=np.unique(np.quantile(xtr,np.linspace(0,1,11)));bid=np.clip(np.digitize(xr,edges[1:-1],right=True),0,len(edges)-2)
            assert len(g)==len(edges)-1 and np.allclose(g.lower,edges[:-1]) and np.allclose(g.upper,edges[1:])
            fullbid=np.clip(np.digitize(xte,edges[1:-1],right=True),0,len(edges)-2)
            checks=[]
            for bi,row in enumerate(g.itertuples()):
                mask=bid==bi;valid=mask & np.isfinite(te[idx]);stations=q.station.to_numpy()
                count=int(mask.sum());nst=len(set(stations[mask]));assert count==row.n_samples and nst==row.n_stations
                b={'task':task,'fold':fold,'feature':feature,'bin':bi+1,'lower':row.lower,'upper':row.upper,'冻结样本数':count,'冻结站点数':nst,'其中真实非缺失值样本':int(valid.sum()),'其中真实非缺失值站点':len(set(stations[valid])),'其中填补值样本':int((mask & ~np.isfinite(te[idx])).sum()),'完整测试集该箱记录':int((fullbid==bi).sum()),'完整测试集该箱原始非缺失':int(((fullbid==bi)&np.isfinite(te)).sum()),'local_effect缺失':bool(pd.isna(row.local_effect))}
                bins.append(b);checks.append(count)
            assert sum(checks)==64
            curve_checks.append({'task':task,'fold':fold,'feature':feature,'训练记录':len(train),'测试记录':len(test),'训练原始唯一值':pd.Series(tr).nunique(),'测试原始唯一值':pd.Series(te).nunique(),'训练缺失率':float(np.isnan(tr).mean()),'测试缺失率':float(np.isnan(te).mean()),'训练填补中位数':median,'训练填补后中位数占比':float((xtr==np.float32(median)).mean()),'请求分箱':10,'实际分箱':len(g),'非空箱':int(g.n_samples.gt(0).sum()),'参考样本':len(idx),'参考原始非缺失样本':int(np.isfinite(te[idx]).sum()),'参考原始非缺失站点':q.loc[np.isfinite(te[idx]),'station'].nunique(),'空分箱数':int(g.n_samples.eq(0).sum()),'测试超训练范围':int(((xte<edges[0])|(xte>edges[-1])).sum()),'原始参考值缺失被计入支持':int((~np.isfinite(te[idx])).sum()),'边界及样本/站点计数复现一致':True})
            if feature.startswith('ntl'):
                rawedges=np.unique(np.quantile(tr[np.isfinite(tr)],np.linspace(0,1,11))) if np.isfinite(tr).any() else []
                fullfeatures.append({'task':task,'fold':fold,'feature':feature,'原始非缺失训练值可形成箱数':max(0,len(rawedges)-1),'冻结填补后箱数':len(g),'说明':'只作分箱诊断，未按原始非缺失分箱重算ALE；改变分箱/参考分布需单独登记协议'})
                for j,(_,rr) in enumerate(q.iterrows()):queryrows.append({'task':task,'fold':fold,'feature':feature,'obs_id':rr.obs_id,'station':rr.station,'date':rr.date,'raw':te[idx[j]],'imputed':float(xr[j]),'bin':int(bid[j])+1,'是否填补':not np.isfinite(te[idx[j]])})
csv('历史80条ALE_参考与分箱完整核对.csv',curve_checks);csv('历史590格ALE_真实与填补支持分离.csv',bins);csv('夜光ALE_960条参考取值审计.csv',queryrows);csv('夜光分箱策略信息损失.csv',fullfeatures)

# 正式P1二十曲线：CSV点数与查询清单、支持表相核对，不增加分箱。
p1=ROOT/'runs/tabicl_v2_slim_explainability_v2_all10/ale_1d';p1checks=[]
for p in sorted(p1.glob('*_curve.csv')):
    c=read(p);prefix=p.name[:-len('_curve.csv')];query=read(p1/(prefix+'_query_manifest.csv'));support=read(p1/(prefix+'_bin_support.csv'))
    assert not query.obs_id.duplicated().any()
    for row in c.itertuples():
        z=query[query.bin.eq(row.bin)];assert len(z)==row.n_samples and z.station_id.nunique()==row.n_stations
    assert len(c)==len(support)
    assert np.allclose(c.n_samples,support.selected_n) and np.allclose(c.n_stations,support.selected_stations)
    p1checks.append({'任务':c.task.iloc[0],'变量':c.feature.iloc[0],'曲线点数':len(c),'实际独立查询记录':len(query),'查询独立站点':query.station_id.nunique(),'每箱样本/站点数与实际查询一致':True,'重复obs_id':0,'缺失查询取值':int(query.feature_value.isna().sum()),'实际最少每箱样本':int(c.n_samples.min()),'实际最少每箱站点':int(c.n_stations.min()),'横轴数量级':float(max(abs(c.lower.min()),abs(c.upper.max())))})
assert len(p1checks)==20 and sum(r['曲线点数'] for r in p1checks)==196
csv('正式P1_20曲线196格真实支持核验.csv',p1checks)

# SHAP三核心与种子重复独立核对，覆盖正文和C/F。
samples=read('结果/论文图表/figure_revision_v2/source_inventory/40组文件与30核心样本对应.csv');frozen=read('结果/论文图表/figure_revision_v2/data/30核心样本_77特征完整排名.csv');shapchecks=[]
for task,g in samples.groupby('task',sort=False):
    core=g[g.seed.eq(42)];repeat=g[g.seed.eq(43)];assert len(core)==3 and len(repeat)==1
    frames=[read(p) for p in core.文件];a=pd.concat(frames);assert a.obs_id.nunique()==3 and len(a)==231
    agg=a.assign(v=a.shap_value.abs()).groupby('feature').v.mean();f=frozen[frozen.task.eq(task)].set_index('feature');assert np.allclose(agg.reindex(f.index),f.mean_abs_shap,atol=1e-12)
    assert np.array_equal(agg.rank(ascending=False,method='min').reindex(f.index),f['rank'])
    repsv=read(repeat.文件.iloc[0]);assert repsv.obs_id.iloc[0] in set(a.obs_id)
    shapchecks.append({'任务':task,'独立样本':3,'重复探针':1,'特征数':77,'770项汇总与排名一致':True,'局部加和最大误差':max(abs(z.shap_value.sum()+z.base_value.iloc[0]-z.model_prediction.iloc[0]) for z in frames),'种子重复未作为独立样本':True})
csv('30核心SHAP_770项独立汇总核验.csv',shapchecks)

# 站点空间图使用真实一对一元数据连接；不把N行误当独立站点数。
spatial=[]
for fam,t in [('chla','chla_surface'),('tn','tn_bottom'),('tp','tp_middle'),('secchi','secchi')]:
    if fam=='secchi':d=read('runs/tabiclv2_pilot/24_station_level_comparison.csv');d=d[d.task.eq(t)].rename(columns={'xgb_RMSE':'XGB_RMSE','tabicl_RMSE':'TabICL_RMSE'})
    else:d=read(f'runs/{fam}_family_v2/09_{fam}_family_v2_station_metrics.csv');d=d[d.task.eq(t)]
    z=d.merge(meta,left_on='station_id',right_on='station',how='left',validate='one_to_one');assert len(z)==len(d) and z[['lon','lat']].notna().all().all()
    assert np.allclose(z.TabICL_RMSE-z.XGB_RMSE,z.delta_RMSE)
    spatial.append({'任务':t,'站点数':len(z),'坐标遗漏':0,'一对一连接':True,'ΔRMSE方向与冻结值一致':True})
csv('空间误差图_站点连接核验.csv',spatial)

# 全89图来源普查；深核覆盖范围写清，未核对的模型上游不标通过。
rp=ROOT/'结果/论文最终图件重构_v1/05_数据核验/图件登记.json'
hashes[str(rp)]=hashlib.sha256(rp.read_bytes()).hexdigest();reg=json.loads(rp.read_text(encoding='utf-8'));audit=[]
for num,r in reg.items():
    if num.startswith(('1-','2-')):continue
    sources=[Path(x) if Path(x).is_absolute() else ROOT/x for x in r['数据来源'].split(';') if x]
    missing=[str(p) for p in sources if not p.is_file()]
    for p in sources:
        if p.is_file():hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    if num in ['3-8','3-9','3-10','3-11']:
        focus='已全量复现参考抽样、分箱边界、计数；夜光源覆盖仅8站。修复数据后重算，不删除变量。'
    elif num.startswith('E-') or num in ['3-12','3-13','3-14','3-15','3-16']:
        focus='已逐箱核对20条P1曲线196实际格；六箱是冻结算法结果，不能据此推断原始数据只有六值；数据改变后须重算。'
    elif num.startswith(('C-','F-')) or num in ['3-3','3-4','3-5','3-6','3-7']:
        focus='已核对30独立核心样本、种子重复边界、770项汇总和排名；不是总体分布。'
    elif num=='4-5':focus='四任务84/94/84/84站坐标一对一连接及差值方向核验；输入夜光补齐后模型须更新。'
    elif num in ['D-6','D-7','D-8','D-9','D-10']:
        focus='冻结41点跨折汇总含历史插值，点数不是独立样本量；本轮仅确认来源，未据此宣称原始分箱支持充分。'
    elif num=='5-2':focus='12任务/60任务折历史78特征实验独立登记；上游历史特征需单独追溯后决定重算，不能当正式77特征。'
    elif num=='4-4':focus='TN底层TabPFN/TabM为缺同协议实验结果，非原始环境变量缺失；不能通过补夜光自动填成预测分数。'
    else:focus='来源存在性及结构普查；数值为冻结结果，上游夜光缺失影响待修复后评估，未标记全部科学验证通过。'
    audit.append({'图号':'图'+num,'图名':r['图名'],'数据来源':r['数据来源'],'来源文件数':len(sources),'缺少源文件':';'.join(missing),'检查范围':focus,'当前处置':'保留原图和变量；本轮不覆盖论文；已确诊数据缺陷按补取/重算链处理','是否新绘':'否'})
assert len(audit)==89
csv('89图_数据完整性逐图同步检查.csv',audit)
dump('来源文件保护哈希.json',hashes)
dump('诊断执行摘要.json',{'历史曲线数':len(curve_checks),'历史分箱格':len(bins),'历史空箱曲线':sum(x['空分箱数']>0 for x in curve_checks),'P1曲线':len(p1checks),'P1实际格':196,'SHAP独立样本':30,'SHAP种子重复':10,'全篇审查图数':len(audit),'源文件数':len(hashes),'进行了模型调用':False})
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
print('完成：89图来源普查、80历史曲线、20 P1曲线和30独立SHAP样本核对。',flush=True)
