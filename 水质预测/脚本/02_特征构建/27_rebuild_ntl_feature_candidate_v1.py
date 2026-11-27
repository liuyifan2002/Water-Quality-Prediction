"""从真实月表另存夜光质量/覆盖修复候选；保留77特征定义、顺序、观测和冻结划分。"""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/图片数据完整性整改_v1_20261115'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    src=ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv';original=sha(src)
    base=pd.read_csv(src,low_memory=False);base.date=pd.to_datetime(base.date)
    oldpath=ROOT/'数据/处理后数据/正式主表/viirs_ntl_station_monthly_2014_2024.csv';old=pd.read_csv(oldpath)
    backfill=OUT/'数据补充/VIIRS_94站132月_同源原始提取.csv';raw=old.copy();added=0
    keys=['station','year','month']
    if backfill.exists():
        fresh=pd.read_csv(backfill);assert not fresh.duplicated(keys).any()
        existing=set(map(tuple,old[keys].to_numpy()));extra=fresh.loc[[tuple(k) not in existing for k in fresh[keys].to_numpy()]]
        # 全部94站统一使用本次同源提取版本，旧月表只用于前后对照，不混合不同提取时点。
        raw=fresh.copy();added=len(extra)
        overlap=old.merge(fresh,on=keys,how='left',validate='one_to_one',suffixes=('_原','_新'))
        overlap['辐亮度差值']=overlap.ntl_avg_rad_新-overlap.ntl_avg_rad_原
        overlap['晴空覆盖差值']=overlap.ntl_cf_cvg_新-overlap.ntl_cf_cvg_原
        overlap.to_csv(OUT/'数据补充/补取与原8站重叠逐值核对.csv',index=False,encoding='utf-8-sig')
    assert not raw.duplicated(keys).any()
    raw['ntl_avg_rad_raw']=raw.ntl_avg_rad
    invalid=raw.ntl_cf_cvg.isna() | raw.ntl_cf_cvg.le(0) | raw.viirs_count.le(0)
    raw['夜光质量状态']=np.where(invalid,'无有效覆盖，响应值标记缺失','有覆盖；保留原值，未人工填补')
    raw.loc[invalid,'ntl_avg_rad']=np.nan
    raw.to_csv(OUT/'修复候选/夜光月表_质量标记候选.csv',index=False,encoding='utf-8-sig')
    candidate=base.copy()
    fields=['ntl_avg_rad','ntl_cf_cvg','viirs_count','viirs_buffer_m']
    aligned=base[['obs_id',*keys]].merge(raw[[*keys,*fields]],on=keys,how='left',validate='many_to_one')
    assert aligned.obs_id.equals(base.obs_id)
    for f in fields:
        if f in base:candidate[f]=aligned[f].to_numpy()
    sample=candidate[['sample_id','station','date','ntl_avg_rad']].drop_duplicates('sample_id').sort_values(['station','date']).reset_index(drop=True)
    assert not sample.duplicated(['station','date']).any()
    grouped=sample.groupby('station',sort=False).ntl_avg_rad
    envcols=[]
    for step in [1,2,3]:
        name=f'ntl_avg_rad_lag{step}';sample[name]=grouped.shift(step);envcols.append(name)
    for window in [3,5]:
        name=f'ntl_avg_rad_roll{window}';sample[name]=grouped.transform(lambda s,w=window:s.shift(1).rolling(w,min_periods=1).mean());envcols.append(name)
    for suffix,series in [('delta1',sample.ntl_avg_rad_lag1-sample.ntl_avg_rad_lag2),('delta7',sample.ntl_avg_rad_lag1-grouped.shift(7))]:
        name='ntl_avg_rad_'+suffix;sample[name]=series;envcols.append(name)
    sid=sample.set_index('sample_id')
    for f in envcols:candidate[f]=candidate.sample_id.map(sid[f])
    manifest=pd.read_csv(ROOT/'runs/tabiclv2_pilot/01_strict_feature_manifest.csv')
    features=manifest.loc[manifest.included_in_pilot.astype(str).str.lower().eq('true'),'feature'].tolist()
    assert len(features)==77 and 'water_zone_code' not in features
    targets=['chla_ug_l','secchi_depth_m','tn_mg_l','tp_mg_l'];metadata=['obs_id','sample_id','station','date','depth',*targets]
    result=candidate[[*metadata,*features]]
    assert result.obs_id.equals(base.obs_id) and result.station.equals(base.station) and result.depth.equals(base.depth)
    assert result[targets].equals(base[targets]) and result.columns.tolist()[len(metadata):]==features
    changed=[]
    for f in features:
        same=np.isclose(candidate[f].to_numpy(float),base[f].to_numpy(float),rtol=1e-10,atol=1e-10,equal_nan=True)
        n=int((~same).sum())
        if not (f.startswith('ntl_avg_rad') or f=='viirs_count'):assert n==0,(f,n)
        changed.append({'变量':f,'正式特征序号':features.index(f)+1,'改变记录数':n,'修订前缺失率':base[f].isna().mean(),'候选缺失率':candidate[f].isna().mean(),'修订前唯一值':base[f].nunique(),'候选唯一值':candidate[f].nunique(),'修订前非缺失站点':base.loc[base[f].notna(),'station'].nunique(),'候选非缺失站点':candidate.loc[candidate[f].notna(),'station'].nunique()})
    change=pd.DataFrame(changed);change.to_csv(OUT/'证据/77特征候选前后逐项核对.csv',index=False,encoding='utf-8-sig')
    result.to_csv(OUT/'修复候选/正式77特征_夜光修复候选_未用于实验.csv',index=False,encoding='utf-8-sig')
    c=change[change.改变记录数.gt(0)].变量.tolist()
    mask=~np.isclose(candidate[c].to_numpy(float),base[c].to_numpy(float),rtol=1e-10,atol=1e-10,equal_nan=True)
    effects=base.loc[mask.any(axis=1),metadata].copy()
    for f in c:effects[f+'_原值']=base.loc[effects.index,f];effects[f+'_候选']=candidate.loc[effects.index,f]
    effects.to_csv(OUT/'证据/候选改变的观测明细.csv',index=False,encoding='utf-8-sig')
    assert sha(src)==original
    info={'候选记录数':len(result),'正式特征数':len(features),'源主表SHA256未变':True,'源主表SHA256':original,'新增真实月度记录':added,'零覆盖原始月记录数':int(invalid.sum()),'候选改变观测数':len(effects),'改变特征':c,'目标/站点/日期/观测顺序保留':True,'77顺序与定义保留':True,'状态':'仅质量修复候选，94站补取仍未完成；不得与冻结预测混合' if not backfill.exists() else '同源补齐候选；未替换主表，须完整重算受影响实验','进行了插值或复制独立样本':False,'运行了训练/预测/SHAP/ALE':False}
    (OUT/'证据/特征修复候选状态.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
