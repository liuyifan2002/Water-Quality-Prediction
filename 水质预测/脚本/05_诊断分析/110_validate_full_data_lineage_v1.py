"""从原始文件和现有处理函数确定性复核水质、坐标、环境合并及272列。"""
from pathlib import Path
import importlib.util, json, warnings
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115'
E=OUT/'证据'
F=ROOT/'数据/处理后数据/正式主表'

def load(name, rel):
    spec=importlib.util.spec_from_file_location(name,ROOT/rel);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def save(name,d):pd.DataFrame(d).to_csv(E/name,index=False,encoding='utf-8-sig')
def eq(a,b):
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):return np.isclose(a.to_numpy(float,na_value=np.nan),b.to_numpy(float,na_value=np.nan),rtol=1e-7,atol=1e-8,equal_nan=True)
    aa=a.astype('string').str.lower();bb=b.astype('string').str.lower()
    return ((aa==bb)|(aa.isna()&bb.isna())).fillna(False).to_numpy(bool)
def compare(a,b,cols,label,key='obs_id'):
    b=b.set_index(key).loc[a[key]].reset_index();results=[]
    for c in cols:
        if c in b:results.append({'链路':label,'字段':c,'核对行数':len(a),'不一致行数':int((~eq(a[c],b[c])).sum())})
        else:results.append({'链路':label,'字段':c,'核对行数':len(a),'不一致行数':'NA：重建结果缺字段'})
    return results

def main():
    E.mkdir(parents=True,exist_ok=True)
    wq=load('audit_wq','脚本/01_数据准备/02_build_water_quality_master.py')
    st=load('audit_st','脚本/01_数据准备/01_prepare_station_samples.py')
    build=load('audit_build','脚本/02_特征构建/04_build_model_table.py')
    satmod=load('audit_sat','脚本/02_特征构建/03_merge_satellite_features.py')
    daily=load('audit_daily','脚本/02_特征构建/24_download_prepare_era5_ssrd_sst_daily.py')
    rainmod=load('audit_rain','脚本/02_特征构建/03_prepare_hko_rainfall_features.py')
    tidemod=load('audit_tide','脚本/02_特征构建/04_prepare_hko_tide_features.py')
    master=pd.read_csv(F/'model_table_3depth_2014_2024.csv',parse_dates=['date'],low_memory=False)
    water=pd.read_csv(F/'water_quality_master_2014_2024.csv',parse_dates=['date'],low_memory=False)
    original=Path('C:/Users/刘逸凡/Desktop/开题/水质数据')
    raw=wq.load_water_files(original);rewater=wq.build_master_table(raw,wq.load_station_meta(F/'station_meta.csv'))
    comparisons=compare(water,rewater,list(water.columns),'11份原始水质CSV→114列主表')
    save('05_原始水质全部字段溯源核对.csv',comparisons)
    rawmeta=st.load_station_meta(original/'站点位置/download_20251010_1507_converted.shp')
    meta=pd.read_csv(F/'station_meta.csv')
    save('06_坐标原始矢量核对.csv',compare(meta,rawmeta,list(meta.columns),'原始SHP/DBF→94站元数据',key='Station'))
    sat=build.normalize_bool_like_columns(pd.read_csv(F/'station_date_satellite_features_2014_2024.csv',parse_dates=['date']))
    monthly=pd.read_csv(F/'era5_monthly_2014_2024.csv')
    windows=pd.read_csv(ROOT/'数据/处理后数据/衍生版本/era5_ssrd_sst_window_features_2014_2024.csv',parse_dates=['date'])
    rain=pd.read_csv(F/'rainfall_features_2014_2024.csv',parse_dates=['date'])
    tide=pd.read_csv(F/'tide_features_2014_2024.csv',parse_dates=['date'])
    ntl=pd.read_csv(F/'viirs_ntl_station_monthly_2014_2024.csv')
    rebuilt=build.merge_model_table(water,sat,monthly,windows,rain,tide,ntl)
    save('07_272列主表逐项重建核对.csv',compare(master,rebuilt,list(master.columns),'原有全部输入→272列主表'))
    # 以全站实际采样日为分母，避免8站日期表被误当全站支持。
    samples=water[['sample_id','station','date']].drop_duplicates('sample_id');sources=[];missing=[]
    for name,df,keys in [('ERA5月均',monthly,['year','month']),('ERA5窗口',windows,['date']),('降雨',rain,['date']),('潮差',tide,['date']),('夜光',ntl,['station','year','month'])]:
        base=samples.assign(year=samples.date.dt.year,month=samples.date.dt.month)
        merged=base.merge(df[keys].drop_duplicates(),on=keys,how='left',indicator=True,validate='many_to_one')
        bad=merged._merge.eq('left_only')
        sources.append({'来源':name,'源记录':len(df),'采样日期覆盖':df.date.nunique() if 'date' in df else '按年月','需要采样日期':samples.date.nunique(),'未连接独立采样':int(bad.sum()),'未覆盖日期':merged.loc[bad,'date'].nunique(),'未覆盖站点':merged.loc[bad,'station'].nunique()})
        missing.extend({'来源':name,'sample_id':r.sample_id,'站点':r.station,'日期':r.date} for r in merged.loc[bad].itertuples())
    save('08_全站实际采样连接覆盖.csv',sources);save('09_遗漏采样连接明细.csv',missing)
    strict=pd.read_csv(ROOT/'runs/tabiclv2_pilot/01_strict_feature_manifest.csv')
    formal=set(strict.loc[strict.included_in_pilot.astype(str).str.lower().eq('true'),'feature'])
    grouped=[]
    for dimension,group in [('站点',master.station),('年份',master.date.dt.year),('水层',master.depth)]:
        for key,g in master.groupby(group):
            for c in master.columns:
                if pd.api.types.is_numeric_dtype(g[c]):grouped.append({'维度':dimension,'分组':key,'字段':c,'正式77':c in formal,'行数':len(g),'缺失':int(g[c].isna().sum()),'缺失率':float(g[c].isna().mean()),'唯一值':g[c].nunique()})
    save('10_272列分站分年分层覆盖.csv',grouped)
    # 采样粒度与日期长度；证明未把多水层当环境滞后步长。
    ss=samples.sort_values(['station','date']);interval=[]
    for lag in [1,2,3,7]:
        diff=(ss.date-ss.groupby('station').date.shift(lag)).dt.days
        interval.append({'滞后采样步数':lag,'最少天数':diff.min(),'中位天数':diff.median(),'最大天数':diff.max(),'重复同日间隔':int(diff.eq(0).sum())})
    save('11_所有环境滞后时间定义.csv',interval)
    flags=[]
    for c in master:
        if c in formal:continue
        if c.endswith('_filled') or c.endswith('_imputed') or c.endswith('_raw') or c.endswith('_censored') or c.endswith('_detect_limit'):
            continue
        if master[c].dtype.kind in 'fiu':
            flags.append({'字段':c,'进入正式77':False,'缺失率':master[c].isna().mean(),'唯一值':master[c].nunique(),'可用站点':master.loc[master[c].notna(),'station'].nunique(),'风险提示':'目标历史/同期协变量仅适用相应历史/重建协议，不能自动加入新站预测' if any(c.startswith(t+'_') or c==t for t in build.TARGET_HISTORY_COLS) else '来源覆盖及填补需核验，未因排除而跳过'})
    save('12_正式77之外主表字段审查.csv',flags)
    checks={'原水质CSV行数':len(raw),'水质主表记录':len(water),'目标24变量全部原始解析复现':all(r['不一致行数']==0 for r in comparisons),'主表重建列数':len(rebuilt.columns),'实际采样日':samples.date.nunique(),'旧环境日期':len(rain),'未修改任何原件':True,'未调用模型计算':True}
    (E/'正式数据链核验摘要.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,default=int),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2));print(pd.DataFrame(sources).to_string(index=False))

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
    main()
