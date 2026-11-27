"""使用已有真实日数据补齐94站采样日期环境表；独立候选，不训练或推理。"""
from pathlib import Path
import importlib.util,json,warnings
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115'
E=OUT/'证据';C=OUT/'修复候选'
F=ROOT/'数据/处理后数据/正式主表'

def module(name,rel):
    spec=importlib.util.spec_from_file_location(name,ROOT/rel);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    C.mkdir(parents=True,exist_ok=True);E.mkdir(parents=True,exist_ok=True)
    build=module('candidate_build','脚本/02_特征构建/04_build_model_table.py')
    tidemod=module('candidate_tide','脚本/02_特征构建/04_prepare_hko_tide_features.py')
    water=pd.read_csv(F/'water_quality_master_2014_2024.csv',parse_dates=['date'],low_memory=False)
    dates=pd.DatetimeIndex(sorted(water.date.unique()))
    raw=pd.read_csv(ROOT/'数据/原始数据/daily_HKO_RF_ALL.csv',skiprows=2)
    raw=raw.rename(columns={'年/Year':'year','月/Month':'month','日/Day':'day','數值/Value':'rain'})
    raw['date']=pd.to_datetime(raw[['year','month','day']],errors='coerce')
    text=raw.rain.astype(str).str.strip();raw['rain_raw_text']=text;raw['rain']=pd.to_numeric(text,errors='coerce')
    # 非数字只在明确微量T时按原定义记0，并保存痕迹；其他未知值保留NA。
    trace=text.str.upper().isin(['T','TRACE']);raw.loc[trace,'rain']=0.
    daily=raw.loc[raw.date.notna()].set_index('date').rain.sort_index()
    assert daily.index.is_unique
    rss=[];supports=[]
    for date in dates:
        row={'date':date};counts={'date':date}
        for days in [3,14]:
            v=daily.reindex(pd.date_range(date-pd.Timedelta(days=days),date-pd.Timedelta(days=1)))
            row[f'rain_{days}d']=float(v.sum()) if v.notna().sum()==days else np.nan
            counts[f'rain_{days}d_有效天数']=int(v.notna().sum())
        mdays=pd.date_range(date.replace(day=1),date.replace(day=1)+pd.offsets.MonthEnd(0))
        v=daily.reindex(mdays);row['rain_month']=float(v.sum()) if v.notna().all() else np.nan
        counts['rain_month_有效天数']=int(v.notna().sum());rss.append(row);supports.append(counts)
    rain=pd.DataFrame(rss)
    tides=pd.concat([tidemod.parse_tide_file(ROOT/f'数据/原始数据/tide_QUB_{y}.csv',y) for y in range(2014,2025)])
    assert not tides.date.duplicated().any()
    tides['tidal_range_raw_extrema']=tides.tidal_range
    tides['日内极值不足两次']=tides.tide_events.lt(2)
    tides.loc[tides['日内极值不足两次'],'tidal_range']=np.nan
    tides.to_csv(E/'32_QUB潮汐日内极值支持及单极值零占位.csv',index=False,encoding='utf-8-sig')
    tide=pd.DataFrame({'date':dates}).merge(tides[['date','tidal_range']],on='date',how='left',validate='one_to_one')
    prehistory=C/'ERA5日值_补充2013年前史_未用于实验.csv'
    era=pd.read_csv(prehistory if prehistory.exists() else ROOT/'数据/处理后数据/衍生版本/era5_ssrd_sst_daily_2014_2024.csv',parse_dates=['date']).set_index('date')
    assert era.index.is_unique
    windows=[];ws=[]
    for date in dates:
        row={'date':date};sup={'date':date}
        for days in [3,7,14]:
            sub=era.reindex(pd.date_range(date-pd.Timedelta(days=days),date-pd.Timedelta(days=1)))
            for prefix,col,func in [('ssrd_sum','ssrd_sum_j_m2','sum'),('ssrd_mean','ssrd_mean_j_m2','mean'),('sst_mean','sst_mean_c','mean'),('sst_max','sst_max_c','max'),('sst_anom','sst_anom_c','mean')]:
                key=f'{prefix}_{days}d';n=int(sub[col].notna().sum());sup[key+'_有效天数']=n
                row[key]=float(getattr(sub[col],func)()) if n==days else np.nan
        windows.append(row);ws.append(sup)
    windows=pd.DataFrame(windows)
    for name,d in [('94站全部采样日_降雨候选.csv',rain),('94站全部采样日_潮差候选.csv',tide),('94站全部采样日_ERA5窗口候选.csv',windows)]:d.to_csv(C/name,index=False,encoding='utf-8-sig')
    pd.DataFrame(supports).to_csv(E/'新增全采样日_降雨有效天数.csv',index=False,encoding='utf-8-sig');pd.DataFrame(ws).to_csv(E/'新增全采样日_ERA5有效天数.csv',index=False,encoding='utf-8-sig')
    raw.loc[raw.date.between('2013-12-01','2024-12-31')&pd.to_numeric(text,errors='coerce').isna()].to_csv(E/'HKO非数值降雨原始标记.csv',index=False,encoding='utf-8-sig')
    ntl=pd.read_csv(ROOT/'结果/诊断结果/图片数据完整性整改_v1_20261115/修复候选/夜光月表_质量标记候选.csv')
    ntl=ntl[['station','year','month','ntl_avg_rad','ntl_cf_cvg','viirs_count','viirs_buffer_m']]
    sat=build.normalize_bool_like_columns(pd.read_csv(F/'station_date_satellite_features_2014_2024.csv',parse_dates=['date']))
    rebuilt=build.merge_model_table(water,sat,pd.read_csv(F/'era5_monthly_2014_2024.csv'),windows,rain,tide,ntl)
    old=pd.read_csv(F/'model_table_3depth_2014_2024.csv',low_memory=False)
    rebuilt=rebuilt.set_index('obs_id').loc[old.obs_id].reset_index()[old.columns]
    sf=pd.read_csv(ROOT/'runs/tabiclv2_pilot/01_strict_feature_manifest.csv');features=sf.loc[sf.included_in_pilot.astype(str).str.lower().eq('true'),'feature'].tolist()
    changes=[]
    for col in features:
        a=old[col].to_numpy(float);b=rebuilt[col].to_numpy(float)
        changes.append({'特征':col,'改变记录':int((~np.isclose(a,b,rtol=1e-7,atol=1e-8,equal_nan=True)).sum()),'原缺失率':float(np.isnan(a).mean()),'候选缺失率':float(np.isnan(b).mean()),'原唯一值':old[col].nunique(),'候选唯一值':rebuilt[col].nunique(),'原有效站点':old.loc[old[col].notna(),'station'].nunique(),'候选有效站点':rebuilt.loc[rebuilt[col].notna(),'station'].nunique()})
    pd.DataFrame(changes).to_csv(E/'77特征全部环境修复前后.csv',index=False,encoding='utf-8-sig')
    for col in ['obs_id','sample_id','station','depth','chla_ug_l','secchi_depth_m','tn_mg_l','tp_mg_l']:assert old[col].equals(rebuilt[col]),col
    assert pd.to_datetime(old.date).equals(pd.to_datetime(rebuilt.date))
    rebuilt.to_csv(C/'272列全环境覆盖修复候选_未用于实验.csv',index=False,encoding='utf-8-sig')
    rebuilt[['obs_id','sample_id','station','date','depth','chla_ug_l','secchi_depth_m','tn_mg_l','tp_mg_l']+features].to_csv(C/'正式77特征_全环境覆盖修复候选_未用于实验.csv',index=False,encoding='utf-8-sig')
    state={'原采样日期':1024,'全站采样日期':len(dates),'新增真实采样日期':len(dates)-1024,'改变正式特征数':sum(r['改变记录']>0 for r in changes),'原始水质和正式主表已覆盖':False,'重新运行训练预测解释':False,'严格窗口支持':'已补取同源2013年末前史且重叠验证通过；不补造缺失天数' if prehistory.exists() else '天数不足保留NA，需要真实前史补取','潮汐单极值日':int(tides['日内极值不足两次'].sum()),'潮汐单极值采样日':int(tide.tidal_range.isna().sum()),'保留问题':'QUB单极值不支持日潮差，候选标NA；SST异常基于全期气候态，时间外推须独立训练期基准；遥感仍为旧来源，未宣称本候选全数据链已修复'}
    (E/'全环境覆盖修复状态.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(state,ensure_ascii=False,indent=2))

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
    main()
