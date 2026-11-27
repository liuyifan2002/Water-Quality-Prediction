"""验收48项差异与四补充任务，冻结77项基础输入、源哈希和按折气候态协议。"""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
PRIOR=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115'
VERSION='WQ77_REPAIRED_20261115_V1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=RUN/'02_冻结候选';out.mkdir(parents=True,exist_ok=True)
    if (out/'冻结清单.json').exists():raise RuntimeError('冻结版本已存在；禁止覆盖，验收应读取既有清单')
    raw=pd.read_csv(ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv',parse_dates=['date'],low_memory=False)
    previous=pd.read_csv(PRIOR/'修复候选/正式77特征_全环境覆盖修复候选_未用于实验.csv',parse_dates=['date'],low_memory=False)
    manifest=pd.read_csv(ROOT/'runs/tabiclv2_pilot/01_strict_feature_manifest.csv');fs=manifest.loc[manifest.included_in_pilot.astype(str).str.lower().eq('true'),'feature'].tolist()
    assert len(fs)==77 and 'water_zone_code' not in fs
    d=previous.copy();assert d.obs_id.equals(raw.obs_id)
    qub=pd.read_csv(RUN/'01_独立补充任务/QUB/QUB_完整日界支持潮差_2014_2024.csv',parse_dates=['date']).set_index('date')
    d['tidal_range']=d.date.map(qub.tidal_range)
    assert d.tidal_range.notna().all() and d.tidal_range.gt(0).all()
    s=d[['sample_id','station','date','tidal_range']].drop_duplicates('sample_id').sort_values(['station','date']).copy()
    g=s.groupby('station').tidal_range
    s['tidal_range_lag1']=g.shift(1);s['tidal_range_lag2']=g.shift(2);s['tidal_range_roll3']=g.transform(lambda x:x.shift(1).rolling(3,min_periods=1).mean());s['tidal_range_delta1']=s.tidal_range_lag1-s.tidal_range_lag2
    for c in ['tidal_range_lag1','tidal_range_lag2','tidal_range_roll3','tidal_range_delta1']:d[c]=d.sample_id.map(s.set_index('sample_id')[c])
    d['t2m_x_tidal_range']=d.t2m*d.tidal_range;d['rain14d_x_tidal_range']=d.rain_14d*d.tidal_range
    daily=pd.read_csv(PRIOR/'修复候选/ERA5日值_补充2013年前史_未用于实验.csv',parse_dates=['date']).sort_values('date').reset_index(drop=True)
    # 基础表使用事先固定2014—2020参考期供检查；正式拟合时由训练期变换覆盖，不把该参考期硬用于所有折。
    ref=daily[daily.date.between('2014-01-01','2020-12-31')];clim=ref.groupby(ref.date.dt.dayofyear).sst_mean_c.mean()
    anomaly=daily.sst_mean_c-daily.date.dt.dayofyear.map(clim);a=pd.Series(anomaly.values,index=daily.date)
    for n in [3,7,14]:d[f'sst_anom_{n}d']=d.date.map(a.shift(1).rolling(n,min_periods=n).mean())
    d['sst_anom_7d_x_ssrd_mean_7d']=d.sst_anom_7d*d.ssrd_mean_7d
    assert not np.isinf(d[fs].to_numpy(float)).any()
    for c in ['obs_id','sample_id','station','depth','chla_ug_l','secchi_depth_m','tn_mg_l','tp_mg_l']:assert d[c].equals(raw[c]),c
    assert d.date.equals(raw.date)
    rows=[]
    for i,c in enumerate(fs):
        a0=raw[c].to_numpy(float);a1=previous[c].to_numpy(float);a2=d[c].to_numpy(float)
        phase='需要修复→已修复并接受' if ('tidal' in c or c.startswith('sst_anom')) else '接受；保持来源质量缺失并按训练集填补'
        if c.startswith('sst_anom'):reason='原全期气候态改为训练期拟合；基础表参考2014—2020，正式每折覆盖，训练数据边界见协议'
        elif 'tidal' in c:reason='真实HHOT日界/小时+原HLT精确极值；天文潮预报极差，不是实测潮差'
        elif c.startswith('ntl') or c=='viirs_count':reason='真实94站同源月表；无晴空覆盖仍NA；lag是采样步，不补造新样本'
        else:reason='真实日数据补齐511采样日期与2013前史；保持原单位及实际采样滞后定义'
        rows.append({'序号':i+1,'feature':c,'原冻结→前轮候选改变数':int((~np.isclose(a0,a1,equal_nan=True,rtol=1e-7,atol=1e-8)).sum()),'前轮候选→本轮验收改变数':int((~np.isclose(a1,a2,equal_nan=True,rtol=1e-7,atol=1e-8)).sum()),'最终相对原始改变数':int((~np.isclose(a0,a2,equal_nan=True,rtol=1e-7,atol=1e-8)).sum()),'原缺失率':float(np.isnan(a0).mean()),'验收缺失率':float(np.isnan(a2).mean()),'原唯一值':raw[c].nunique(),'验收唯一值':d[c].nunique(),'判定':phase if not np.isclose(a0,a2,equal_nan=True,rtol=1e-7,atol=1e-8).all() else '接受；原始值未改变','依据':reason,'限制':'同期月合成仅用于事后同期泛化，不作实时预测' if c.startswith(('ntl','t2m','tcc','wind')) or c in ['u10','v10','rain_month'] else '见时间协议'})
    acceptance=pd.DataFrame(rows)
    acceptance.loc[acceptance['判定'].eq('接受；原始值未改变'),'依据']='原冻结主表值未改变；原始来源按特征组核验，实际来源与单位见正式77项特征定义与来源总表'
    acceptance.to_csv(RUN/'01_77项逐项验收.csv',index=False,encoding='utf-8-sig')
    changed=acceptance[acceptance['原冻结→前轮候选改变数']>0];assert len(changed)==48
    changed.to_csv(RUN/'02_48项差异核查及验收决定.csv',index=False,encoding='utf-8-sig')
    manifest=manifest[manifest.feature.isin(fs)].copy();manifest['顺序']=manifest.feature.map({f:i+1 for i,f in enumerate(fs)});manifest=manifest.sort_values('顺序');manifest.to_csv(out/'77项正式输入清单.csv',index=False,encoding='utf-8-sig')
    d[['obs_id','sample_id','station','date','depth','chla_ug_l','secchi_depth_m','tn_mg_l','tp_mg_l']+fs].to_csv(out/'基础输入_77特征.csv',index=False,encoding='utf-8-sig')
    daily.to_csv(out/'ERA5逐日气候态来源.csv',index=False,encoding='utf-8-sig')
    original_meta=raw[['obs_id','station','water_zone_en','sample_no','date','depth']];original_meta.to_csv(out/'观测元数据.csv',index=False,encoding='utf-8-sig')
    state=[]
    for name in ['QUB','Landsat','CMEMS海色','CMEMS海陆掩膜']:
        p=RUN/f'01_独立补充任务/{name}/任务状态.json';x=json.loads(p.read_text(encoding='utf-8'));state.append(x)
        if name=='QUB':assert x['自然日']==4018
        if name=='Landsat':assert x['完成月份']==132 and not x['失败']
        if name=='CMEMS海色':assert x['完整产品']==3 and x['旧重叠数值一致']
    pd.DataFrame([{'资产':x['任务'],'状态':x['状态'],'纳入本轮77':'QUB日潮差及派生项' if x['任务']=='QUB' else '不新增字段，独立扩展资产','暂不接受':'未经连通性证明的湿格位移、OLCI别名重复计数、插值光学实测样本'} for x in state]).to_csv(RUN/'03_补充资产纳入与排除决定.csv',index=False,encoding='utf-8-sig')
    policy={'dataset_version':VERSION,'feature_order':fs,'原始划分不变':True,'候选单位保持原值':True,'missing':'真缺失保留NA；SimpleImputer(median,keep_empty_features=True)只在各折训练集拟合','sst_anomaly':'按训练期日均SST拟合DOY气候态，范围2014-01-01至该折训练日期最大值；各折所有模型使用同一处理后的输入；时间留出不得用晚于训练截止日的日值拟合','base_anomaly_reference':'基础CSV固定2014—2020仅供审查，正式训练必须按上条重建','lag':'同站去重sample_id后按日期的采样步；目标值和水层不作为环境滞后步','roll':'环境roll3是前3次采样且先shift(1)；ERA5/新CMEMS日窗是采样前完整自然日','same_month':'保留既有当月气象/夜光/月雨，属于历史同期可得协变量；时间留出只是事后历史年份泛化，不能声称实时提前预测','space':'原94站真实坐标，原位置native nearest；湿格位移不纳入；海色/物理额外列不混入正式77','source_groups':'保持原来源taxonomy，water_zone_code不加入77；历史78特征只能独立版本重算'}
    (out/'特征与时间协议.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=[ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv',ROOT/'runs/tabiclv2_pilot/01_strict_feature_manifest.csv',PRIOR/'修复候选/正式77特征_全环境覆盖修复候选_未用于实验.csv',PRIOR/'修复候选/ERA5日值_补充2013年前史_未用于实验.csv']
    sources+=list((RUN/'01_独立补充任务').rglob('*.csv'))+list((RUN/'01_独立补充任务').rglob('*.nc'))
    pd.DataFrame([{'来源文件':str(p),'SHA256':sha(p),'字节':p.stat().st_size} for p in sources]).to_csv(out/'数据来源及SHA256.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'feature':f,'缺失':int(d[f].isna().sum()),'缺失率':float(d[f].isna().mean()),'唯一值':d[f].nunique(),'站点覆盖':d.loc[d[f].notna(),'station'].nunique()} for f in fs]).to_csv(out/'完整缺失与支持情况.csv',index=False,encoding='utf-8-sig')
    frozen={str(p.relative_to(RUN)):sha(p) for p in out.iterdir() if p.is_file()}
    record={'dataset_version':VERSION,'rows':len(d),'features':77,'input_sha256':sha(out/'基础输入_77特征.csv'),'files':frozen,'四补充任务':state,'状态':'FROZEN_ACCEPTED_FOR_REGISTERED_HINDCAST_RECOMPUTE','训练期气候态必须按折重建':True,'原始目标划分未改':True,'原正式结果未覆盖':True}
    (out/'冻结清单.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'版本说明.md').write_text('# 冻结候选数据版本\n\n版本：'+VERSION+'。原始观测范围、目标、77字段顺序及留站点manifest不变。48项差异已逐项审查，QUB已补齐真实日界。无数据继续NA，禁止伪造或重复样本。基础输入中SST异常使用固定审查参考期；每折正式有效输入按训练期气候态重建并另存哈希，XGBoost与TabICLv2必须复用完全相同的有效输入。\n\n结果限定历史同期环境协变量条件下的泛化，不是实时提前预测。Landsat、CMEMS海色和掩膜为77之外独立补充，不自动扩展正式特征。所有原件保留，后续不得覆盖冻结目录。\n',encoding='utf-8')
    print('冻结完成',VERSION,record['input_sha256'],flush=True)

if __name__=='__main__':main()
