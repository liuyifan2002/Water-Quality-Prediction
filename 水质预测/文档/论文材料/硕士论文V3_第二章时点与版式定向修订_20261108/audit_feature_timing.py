"""只读核对正式主表与源表的时点字段、月度匹配和站点历史项。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(r'E:\研究生\codex\水质预测')
out=Path(__file__).resolve().parent
base=root/'数据/处理后数据/正式主表'
lag_bases=['t2m','tcc','wind_speed','rain_3d','rain_14d','rain_month','tidal_range','ntl_avg_rad']
selected=['sample_id','station','date','depth','year','month','viirs_count','u10','v10']+lag_bases
selected += [f'{c}_{s}' for c in lag_bases for s in ['lag1','lag2','roll3','delta1']]
main=pd.read_csv(base/'model_table_3depth_2014_2024.csv',usecols=selected,parse_dates=['date'],low_memory=False)
sample=main.drop_duplicates('sample_id').sort_values(['station','date']).reset_index(drop=True)
assert len(sample)==10866
assert not sample.duplicated(['station','date']).any()
assert (main.groupby('sample_id')[['station','date']].nunique().max()<=1).all()
report={'main_rows':len(main),'station_date_rows':len(sample),'source_value_mismatches':{},'lag_mismatches':{}}

def mismatch(a,b):
    x=pd.to_numeric(a,errors='coerce').to_numpy(dtype=float)
    y=pd.to_numeric(b,errors='coerce').to_numpy(dtype=float)
    return int((~np.isclose(x,y,rtol=1e-8,atol=1e-8,equal_nan=True)).sum())

def compare_source(filename,keys,cols):
    src=pd.read_csv(base/filename,parse_dates=['date'] if 'date' in keys else None)
    merged=sample[keys+cols].merge(src[keys+cols],on=keys,how='left',validate='many_to_one',suffixes=('_main','_source'))
    for col in cols:
        n=mismatch(merged[col+'_main'],merged[col+'_source'])
        report['source_value_mismatches'][filename+':'+col]=n
compare_source('era5_monthly_2014_2024.csv',['year','month'],['t2m','tcc','u10','v10'])
compare_source('rainfall_features_2014_2024.csv',['date'],['rain_3d','rain_14d','rain_month'])
compare_source('tide_features_2014_2024.csv',['date'],['tidal_range'])
compare_source('viirs_ntl_station_monthly_2014_2024.csv',['station','year','month'],['ntl_avg_rad','viirs_count'])

g=sample.groupby('station',sort=False)
for col in lag_bases:
    s=sample[col]
    expected={
        'lag1':g[col].shift(1),
        'lag2':g[col].shift(2),
        'roll3':g[col].transform(lambda v:v.shift(1).rolling(3,min_periods=1).mean()),
    }
    expected['delta1']=expected['lag1']-expected['lag2']
    for suffix,value in expected.items():
        report['lag_mismatches'][col+'_'+suffix]=mismatch(sample[col+'_'+suffix],value)
month_end=sample.date+pd.offsets.MonthEnd(0)
report['before_month_end']=int((sample.date<month_end).sum())
prior_date=g.date.shift(1)
report['prior_sample_same_calendar_month']=int((prior_date.dt.to_period('M')==sample.date.dt.to_period('M')).sum())
report['prior_sample_same_date']=int((prior_date==sample.date).sum())
report['missing_source_values']={x:int(sample[x].isna().sum()) for x in ['rain_month','t2m','tidal_range','ntl_avg_rad','viirs_count']}
assert max(report['source_value_mismatches'].values())==0,report['source_value_mismatches']
assert max(report['lag_mismatches'].values())==0,report['lag_mismatches']
assert report['prior_sample_same_date']==0
(out/'特征时点源表一致性核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('AUDIT_PASSED',json.dumps(report,ensure_ascii=True))
