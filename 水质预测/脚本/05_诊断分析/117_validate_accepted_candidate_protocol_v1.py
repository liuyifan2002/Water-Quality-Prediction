"""独立核验冻结候选的时间、空间、原始支持与输入血缘；不拟合模型。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';OUT=RUN/'07_时间协议与空间核验'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);fr=RUN/'02_冻结候选';f=json.loads((fr/'冻结清单.json').read_text(encoding='utf-8'));pol=json.loads((fr/'特征与时间协议.json').read_text(encoding='utf-8'));fs=pol['feature_order'];checks=[]
    def check(name,ok,detail):
        checks.append({'检查项':name,'结果':'PASS' if bool(ok) else 'FAIL','依据':str(detail)});print(name,checks[-1]['结果'],flush=True)
    check('冻结文件哈希',all(sha(RUN/p)==h for p,h in f['files'].items()),len(f['files']))
    d=pd.read_csv(fr/'基础输入_77特征.csv',parse_dates=['date']);old=pd.read_csv(ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv',parse_dates=['date'],low_memory=False)
    check('30719观测与77顺序',len(d)==30719 and d.columns[9:].tolist()==fs and len(fs)==77 and 'water_zone_code' not in fs,'9元字段+77输入；water_zone_code不属于正式输入')
    for c in ['obs_id','sample_id','station','date','depth','chla_ug_l','secchi_depth_m','tn_mg_l','tp_mg_l']:check('原始字段未变_'+c,d[c].equals(old[c]),len(d))
    check('唯一obs_id',d.obs_id.is_unique,len(d))
    s=d.drop_duplicates('sample_id').sort_values(['station','date']);check('站点采样日期无重号',not s.duplicated(['station','date']).any(),len(s))
    lagrows=[]
    for base in ['t2m','tcc','wind_speed','rain_3d','rain_14d','rain_month','tidal_range','ntl_avg_rad']:
        g=s.groupby('station')[base]
        expected={f'{base}_lag1':g.shift(1),f'{base}_lag2':g.shift(2),f'{base}_roll3':g.transform(lambda x:x.shift(1).rolling(3,min_periods=1).mean()),f'{base}_delta1':g.shift(1)-g.shift(2)}
        for c,x in expected.items():
            bad=int((~np.isclose(s[c],x,rtol=1e-7,atol=1e-8,equal_nan=True)).sum());lagrows.append({'feature':c,'采样层行数':len(s),'不一致数':bad,'定义':'同站采样步，roll3前3步min_periods=1；delta1=lag1-lag2'})
    pd.DataFrame(lagrows).to_csv(OUT/'32项滞后滚动独立重构核对.csv',index=False,encoding='utf-8-sig');check('32项环境滞后滚动',sum(x['不一致数'] for x in lagrows)==0,lagrows)
    lagdays=(s.date-s.groupby('station').date.shift(2)).dropna().dt.days
    pd.DataFrame([{'滞后':2,'单位':'采样步，不是2个月','最短日数':lagdays.min(),'中位日数':lagdays.median(),'最长日数':lagdays.max(),'前史不足的采样次数':int(s.groupby('station').cumcount().lt(2).sum())}]).to_csv(OUT/'lag2实际时间跨度.csv',index=False,encoding='utf-8-sig')
    meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv').rename(columns={'Station':'station'})
    coord=d[['station','lon','lat']].drop_duplicates().merge(meta[['station','lon','lat']],on='station',validate='one_to_one',suffixes=('_candidate','_source'))
    check('94站真实坐标匹配',len(coord)==94 and np.allclose(coord[['lon_candidate','lat_candidate']],coord[['lon_source','lat_source']]),'未进行湿格位移')
    coord.to_csv(OUT/'94站坐标原值核对.csv',index=False,encoding='utf-8-sig')
    ntl=pd.read_csv(ROOT/'结果/诊断结果/图片数据完整性整改_v1_20261115/修复候选/夜光月表_质量标记候选.csv').set_index(['station','year','month'])
    keys=pd.MultiIndex.from_arrays([d.station,d.date.dt.year,d.date.dt.month])
    check('夜光94站月份连接',np.isclose(d.ntl_avg_rad,ntl.ntl_avg_rad.reindex(keys).to_numpy(),equal_nan=True).all() and np.isclose(d.viirs_count,ntl.viirs_count.reindex(keys).to_numpy(),equal_nan=True).all(),'同站同年月；缺清晰覆盖时NA')
    monthly=pd.read_csv(ROOT/'数据/处理后数据/正式主表/era5_monthly_2014_2024.csv').set_index(['year','month']);keys=pd.MultiIndex.from_arrays([d.date.dt.year,d.date.dt.month])
    check('ERA5月值月份连接',all(np.isclose(d[c],monthly[c].reindex(keys).to_numpy(),equal_nan=True).all() for c in ['t2m','tcc','u10','v10']),'原同月定义；不是采样前截止值')
    rad=pd.read_csv(ROOT/'数据/原始数据/daily_HKO_RF_ALL.csv',skiprows=2).rename(columns={'年/Year':'year','月/Month':'month','日/Day':'day','數值/Value':'value'});rad['date']=pd.to_datetime(rad[['year','month','day']],errors='coerce');text=rad.value.astype(str).str.strip();rad['value']=pd.to_numeric(text,errors='coerce');rad.loc[text.str.upper().isin(['T','TRACE']),'value']=0.;rain=rad.loc[rad.date.notna()].set_index('date').value.sort_index();rainchecks=[]
    for date in s.date.unique():
        dt=pd.Timestamp(date)
        for c,n in [('rain_3d',3),('rain_14d',14),('rain_month',None)]:
            dates=pd.date_range(dt-pd.Timedelta(days=n),dt-pd.Timedelta(days=1)) if n else pd.date_range(dt.replace(day=1),dt.replace(day=1)+pd.offsets.MonthEnd(0));v=rain.reindex(dates);expected=v.sum() if v.notna().all() else np.nan;actual=s.loc[s.date.eq(dt),c].to_numpy();rainchecks.append(np.isclose(actual,expected,equal_nan=True).all())
    check('HKO降雨真实日期窗口',all(rainchecks),'所有1535采样日期；前3/14完整日，月量为完整同期月')
    formulas={'wind_speed':np.sqrt(d.u10**2+d.v10**2),'wind_speed_sq':d.u10**2+d.v10**2,'t2m_x_rain14d':d.t2m*d.rain_14d,'t2m_x_tidal_range':d.t2m*d.tidal_range,'wind_speed_x_rain14d':d.wind_speed*d.rain_14d,'rain14d_x_ssrd_mean_7d':d.rain_14d*d.ssrd_mean_7d,'rain14d_x_tidal_range':d.rain_14d*d.tidal_range,'sst_mean_7d_x_ssrd_mean_7d':d.sst_mean_7d*d.ssrd_mean_7d,'sst_anom_7d_x_ssrd_mean_7d':d.sst_anom_7d*d.ssrd_mean_7d}
    check('风速及交互项连接',all(np.isclose(d[c],v,rtol=1e-7,atol=1e-7,equal_nan=True).all() for c,v in formulas.items()),','.join(formulas))
    qub=pd.read_csv(RUN/'01_独立补充任务/QUB/QUB_完整日界支持潮差_2014_2024.csv',parse_dates=['date']);check('QUB小时日界支持',len(qub)==4018 and qub['HHOT日界及小时数'].eq(25).all() and qub.tidal_range.gt(0).all(),len(qub))
    check('QUB日期连接',np.allclose(d.tidal_range,d.date.map(qub.set_index('date').tidal_range)),len(d))
    daily=pd.read_csv(fr/'ERA5逐日气候态来源.csv',parse_dates=['date']).set_index('date').sort_index();check('ERA5每日连续前史',daily.index.is_unique and len(daily)==(daily.index.max()-daily.index.min()).days+1,str(daily.index.min())+'至'+str(daily.index.max()))
    windows=[]
    for n in [3,7,14]:
        for outcol,rawcol,how in [('ssrd_sum','ssrd_sum_j_m2','sum'),('ssrd_mean','ssrd_mean_j_m2','mean'),('sst_mean','sst_mean_c','mean'),('sst_max','sst_max_c','max')]:
            c=f'{outcol}_{n}d';expect=getattr(daily[rawcol].shift(1).rolling(n,min_periods=n),how)().reindex(s.date).to_numpy();bad=int((~np.isclose(s[c],expect,rtol=1e-7,atol=1e-7,equal_nan=True)).sum());windows.append({'feature':c,'不一致数':bad,'协议':'采样日前n个完整自然日，不含采样日；不足n日保留NA'})
    pd.DataFrame(windows).to_csv(OUT/'12项ERA5真实日窗核验.csv',index=False,encoding='utf-8-sig');check('ERA5真实日窗',sum(x['不一致数'] for x in windows)==0,windows)
    for base in ['Landsat','CMEMS海色','CMEMS海陆掩膜','QUB']:check('补充任务状态文件_'+base,(RUN/f'01_独立补充任务/{base}/任务状态.json').exists(),'来源与支持限制详见各任务目录')
    ls=pd.read_csv(RUN/'01_独立补充任务/Landsat/Landsat_94站132月_真实原始统计.csv');check('Landsat94站132月唯一',len(ls)==12408 and ls.station.nunique()==94 and not ls.duplicated(['station','year','month']).any(),len(ls))
    lsprof=[]
    for c in ['blue','green','red','nir','swir1','swir2','ndvi','ndwi','mndwi','ndti']:lsprof.append({'来源':'Landsat','变量':c,'行数':len(ls),'非缺失':ls[c].count(),'缺失率':ls[c].isna().mean(),'有效站点':ls.loc[ls[c].notna(),'station'].nunique()})
    oc=pd.read_csv(RUN/'01_独立补充任务/CMEMS海色/CMEMS海色_真实自然日窗候选.csv',parse_dates=['date']);od=pd.read_csv(RUN/'01_独立补充任务/CMEMS海色/CMEMS海色_真实逐日94站.csv',parse_dates=['date']);oix=oc.set_index(['station','date']);occhecks=[]
    for c in [c for c in od if c.startswith('oc_')]:
        for n in [3,7,14]:
            for station,g in od.groupby('station'):
                g=g.set_index('date').sort_index();q=oix.loc[station];expect=g[c].shift(1).rolling(n,min_periods=n).mean().reindex(q.index);actual=q[f'{c}_previous_{n}d_mean'];occhecks.append(int((~np.isclose(actual,expect,equal_nan=True,rtol=1e-7,atol=1e-9)).sum()))
        lsprof.append({'来源':'CMEMS海色','变量':c,'行数':len(oc),'非缺失':oc[c].count(),'缺失率':oc[c].isna().mean(),'有效站点':oc.loc[oc[c].notna(),'station'].nunique()})
    check('CMEMS真实自然日窗独立重构',sum(occhecks)==0 and len(oc)==10866 and oc.sample_id.is_unique,'8变量×3窗口×94站；不足支持保留NA')
    pd.DataFrame(lsprof).to_csv(OUT/'77之外真实补充覆盖.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(OUT/'协议独立核验结果.csv',index=False,encoding='utf-8-sig')
    (OUT/'协议验收说明.md').write_text('# 特征构造和时间空间协议验收\n\n'+f'独立检查{len(checks)}项，失败{sum(x["结果"]=="FAIL" for x in checks)}项。\n\n'+
      '环境lag2为同站前两次采样，不是两个月；roll3按前3次采样、min_periods=1，有效次数不足3时仍可生成，绝非3日平均。delta1为lag1减lag2。days_since_prev_sample单独按同站同水层的观测间隔，保留原定义。\n\n'+
      '当月夜光、当月气象、整月降雨及日历日潮差均含采样时刻之后可能才完整获得的信息。本版用于历史同期、留站点泛化；不是提前预测。拟开展未来实时预测时，必须另立版本仅取截止时已发布数据，不以本版年份留出冒充提前预报。\n\n'+
      'SST异常在正式有效输入中由各折训练期拟合DOY气候态。此轮实际重算留站点协议；未生成新的年份/随机结果。冻结基础CSV的固定2014—2020异常只用于审查，不直接进入正式拟合。DOY按年内序号，沿用原定义，非同月同日气候态。\n\n'+
      'CMEMS海色为多传感器产品；555/670 nm保留真实波长，不将其改名560/665或称为独立OLCI。海陆有效性掩膜是原生数据支持，不等于精细海岸线；尚无连通性证明的湿格位移不纳入。新Landsat仅保存真实观测与QA，未运行旧的双侧插值填充程序。\n',encoding='utf-8')
    assert all(x['结果']=='PASS' for x in checks),'请检查失败记录，不得发布未通过候选'
    print('VALIDATION_PASS',len(checks),flush=True)
if __name__=='__main__':main()
