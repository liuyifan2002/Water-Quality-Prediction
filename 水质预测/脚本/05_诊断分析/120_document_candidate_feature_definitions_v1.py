"""为已冻结77项值添加单位、可用时间和原始来源元数据；不改冻结表。"""
from pathlib import Path
import json,hashlib,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';FR=RUN/'02_冻结候选'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    frozen=json.loads((FR/'冻结清单.json').read_text(encoding='utf-8'));features=pd.read_csv(FR/'77项正式输入清单.csv');profile=pd.read_csv(FR/'完整缺失与支持情况.csv').set_index('feature')
    monthly=ROOT/'数据/处理后数据/正式主表/era5_monthly_2014_2024.csv';era=FR/'ERA5逐日气候态来源.csv';qub=RUN/'01_独立补充任务/QUB/QUB_完整日界支持潮差_2014_2024.csv';ntl=ROOT/'结果/诊断结果/图片数据完整性整改_v1_20261115/修复候选/夜光月表_质量标记候选.csv';rain=ROOT/'数据/原始数据/daily_HKO_RF_ALL.csv';station=ROOT/'数据/处理后数据/正式主表/station_meta.csv';main=ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv'
    units={'t2m':'°C','tcc':'1（0—1比例）','u10':'m/s','v10':'m/s','wind_speed':'m/s','wind_speed_sq':'m²/s²','wind_dir_sin':'1','wind_dir_cos':'1','rain_3d':'mm','rain_14d':'mm','rain_month':'mm','tidal_range':'m','ntl_avg_rad':'nW/(cm²·sr)','viirs_count':'cf_cvg计数的缓冲区空间平均，允许非整数','lon':'°E','lat':'°N','year_offset':'年','quarter':'季度序号1—4','day_of_year':'年内序号1—366','days_since_prev_sample':'日','month_sin':'1','month_cos':'1','is_wet_season':'0/1'}
    formulas={'wind_speed':'sqrt(u10²+v10²)','wind_speed_sq':'u10²+v10²','wind_dir_sin':'sin(atan2(v10,u10))','wind_dir_cos':'cos(atan2(v10,u10))','ntl_avg_rad':'VIIRS VCMSLCFG同月avg_rad，1000m缓冲/500m尺度；cf_cvg<=0时NA','viirs_count':'沿用旧字段名：VIIRS cf_cvg的空间均值，不是研究样本数','tidal_range':'日历日[00,24]内官方HHOT小时/边界与原HLT精确极值联合max−min；天文潮预报非实测','rain_3d':'采样前完整3自然日雨量总和','rain_14d':'采样前完整14自然日雨量总和','rain_month':'采样当月全部自然日雨量和，包含采样之后日期','month_sin':'sin(2π(month−1)/12)','month_cos':'cos(2π(month−1)/12)','is_wet_season':'month在4—9月为1','year_offset':'year−2014','days_since_prev_sample':'同站同水层，按date/sample_no排序的相邻观测日差','quarter':'采样日期季度','day_of_year':'采样日期年内日序，闰年单独编号','lat':'原站点纬度','lon':'原站点经度','t2m':'原ERA5月均2m气温，K减273.15','tcc':'原ERA5月均总云量','u10':'原ERA5月均10m东西风分量','v10':'原ERA5月均10m南北风分量'}
    # 实际风向构造公式需与原实现逐值一致，不能套用气象来向。
    d=pd.read_csv(FR/'基础输入_77特征.csv');import numpy as np
    assert np.allclose(d.wind_dir_sin,np.sin(np.arctan2(d.v10,d.u10)),equal_nan=True)
    assert np.allclose(d.wind_dir_cos,np.cos(np.arctan2(d.v10,d.u10)),equal_nan=True)
    rows=[]
    for _,r in features.iterrows():
        c=r.feature;base=re.sub(r'_(lag[12]|roll3|delta1)$','',c);unit=units.get(base);source=None;definition=formulas.get(base);time='同期月合成，完整值通常晚于采样时刻；仅历史同期协议'
        if base.startswith('ssrd_'):
            source=era;unit='J/m²';n=re.search(r'_(3|7|14)d$',base).group(1);definition=('前'+n+'完整自然日的ssrd_sum_j_m2之和（先按小时累积求日总、再对原3点求均）') if base.startswith('ssrd_sum') else ('前'+n+'完整自然日ssrd_mean_j_m2之均值（小时累计值日均，再按点均）；非W/m²');time='采样日前完整自然日，不含采样日'
        elif base.startswith('sst_'):
            source=era;unit='°C';definition='原3点日SST温度减273.15后，按变量名求采样前完整自然日均值/最大值；anomaly先减该折训练期DOY气候态';time='采样日前完整自然日；气候态仅用本折训练期拟合'
        elif base in ['lon','lat']:source=station;time='固定站点坐标'
        elif base in ['day_of_year','days_since_prev_sample','is_wet_season','month_cos','month_sin','quarter','year_offset']:source=main;time='采样日期/此前采样日；不含目标测值'
        elif base.startswith('ntl') or base=='viirs_count':source=ntl
        elif base.startswith('rain_'):source=rain;time='同期完整月份' if base=='rain_month' else '采样日前完整自然日'
        elif base=='tidal_range':source=qub;time='官方天文潮预报日历日极差；不是采样时刻潮相'
        elif base in ['t2m','tcc','u10','v10','wind_speed','wind_speed_sq','wind_dir_sin','wind_dir_cos']:source=monthly
        if '_x_' in c:
            terms={'t2m_x_rain14d':('t2m','rain_14d','°C·mm',[monthly,rain]),'t2m_x_tidal_range':('t2m','tidal_range','°C·m',[monthly,qub]),'wind_speed_x_rain14d':('wind_speed','rain_14d','mm·m/s',[monthly,rain]),'rain14d_x_tidal_range':('rain_14d','tidal_range','mm·m',[rain,qub]),'rain14d_x_ssrd_mean_7d':('rain_14d','ssrd_mean_7d','mm·J/m²',[rain,era]),'sst_mean_7d_x_ssrd_mean_7d':('sst_mean_7d','ssrd_mean_7d','°C·J/m²',[era]),'sst_anom_7d_x_ssrd_mean_7d':('sst_anom_7d','ssrd_mean_7d','°C·J/m²',[era])}
            a,b,unit,sources=terms[c];definition=f'{a}×{b}；两项均保留原值与缺失';source=sources;time='由两个组成变量中较晚的可获得时间决定'
        assert unit and definition and source,c
        suffix=c[len(base):]
        if suffix:
            definition={'_lag1':'同站前1次采样的','_lag2':'同站前2次采样的','_roll3':'同站前3次采样（先shift1，min_periods=1）均值：','_delta1':'同站lag1减lag2：'}[suffix]+definition;time='2014—2024正式采样序列内的既往采样步（非固定天/月）；序列起点保留NA，未私自并入早期年度观测；源月合成可用性限制仍适用'
        sources=source if isinstance(source,list) else [source];stat=profile.loc[c]
        rows.append({'顺序':r['顺序'],'feature':c,'数据版本':frozen['dataset_version'],'来源组':r.group,'定义':definition,'单位':unit,'源文件':';'.join(str(p) for p in sources),'源文件SHA256':';'.join(sha(p) for p in sources),'时间与可用性':time,'日界和时区边界':'ERA5原UTC日界；HKO雨量/潮汐为香港日历日；VIIRS/GEE按产品月界；未统一小时截止','缺失数':stat['缺失'],'缺失率':stat['缺失率'],'有效站点':stat['站点覆盖'],'唯一值':stat['唯一值'],'NA处理':'原始NA保留；各模型按原协议只拟合训练样本处理器','旧deployment_safe字段':'只保留原登记，不代表实际发布时序下可提前预测'})
    p=RUN/'正式77项特征定义与来源总表.csv';pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig')
    acceptance=pd.read_csv(RUN/'01_77项逐项验收.csv')
    acceptance.loc[acceptance['判定'].eq('接受；原始值未改变'),'依据']='原冻结主表值未改变；原始来源按特征组核验，实际来源与单位见正式77项特征定义与来源总表'
    acceptance.to_csv(RUN/'01_77项逐项验收.csv',index=False,encoding='utf-8-sig')
    (RUN/'冻结版本补充元数据.json').write_text(json.dumps({'父冻结清单':str(FR/'冻结清单.json'),'父冻结清单SHA256':sha(FR/'冻结清单.json'),'data_version':frozen['dataset_version'],'input_sha256':frozen['input_sha256'],'新元数据文件':str(p),'新元数据SHA256':sha(p),'是否修改冻结数值':False,'说明':'补充真实单位/可用时间/来源定义，不改原manifest或任何模型输入值'},ensure_ascii=False,indent=2),encoding='utf-8');print('DEFINITIONS_READY',len(rows),flush=True)
if __name__=='__main__':main()
