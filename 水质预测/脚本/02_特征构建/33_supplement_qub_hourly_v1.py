"""独立QUB补充任务：同源天文潮逐小时预报+原精确极值，恢复日界支持。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,calendar
import numpy as np
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
OUT=RUN/'01_独立补充任务/QUB';OUT.mkdir(parents=True,exist_ok=True)

def fetch(y):
    url=f'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=HHOT&station=QUB&year={y}&rformat=csv'
    p=OUT/f'QUB_HHOT_{y}.csv'
    if not p.exists():
        r=requests.get(url,timeout=60);r.raise_for_status();p.write_bytes(r.content)
    d=pd.read_csv(p);assert {'MM','DD','01','24'}.issubset(d)
    assert len(d)==365+calendar.isleap(y)
    rows=[]
    for r in d.to_dict('records'):
        base=pd.Timestamp(y,int(r['MM']),int(r['DD']))
        for h in range(1,25):rows.append({'datetime':base+pd.Timedelta(hours=h),'height_m':pd.to_numeric(r[f'{h:02}'],errors='coerce')})
    return pd.DataFrame(rows),{'年':y,'URL':url,'路径':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'记录':len(rows),'定义':'官方天文潮预报，HKT；01—24表示当日01时至翌日00时，非实测水位'}

def main():
    frames=[];sources=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(fetch,y) for y in range(2013,2025)]):
            d,s=f.result();frames.append(d);sources.append(s);print('QUB',s['年'],'小时原件完成',flush=True)
    h=pd.concat(frames).sort_values('datetime');assert not h.datetime.duplicated().any();hour=h.set_index('datetime').height_m
    old=pd.read_csv(ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115/证据/32_QUB潮汐日内极值支持及单极值零占位.csv',parse_dates=['date'])
    import csv
    events={}
    for y in range(2014,2025):
        with (ROOT/f'数据/原始数据/tide_QUB_{y}.csv').open(encoding='utf-8-sig') as f:
            reader=csv.reader(f);next(reader)
            for r in reader:events[pd.Timestamp(y,int(r[0]),int(r[1]))]=[float(r[i]) for i in [3,5,7,9] if i<len(r) and r[i].strip()]
    rows=[]
    for r in old.itertuples():
        series=hour.reindex(pd.date_range(r.date,r.date+pd.Timedelta(days=1),freq='h'))
        assert series.notna().sum()==25
        heights=np.r_[series.values,events[r.date]]
        value=float(heights.max()-heights.min())
        assert value>0
        rows.append({'date':r.date,'tidal_range':value,'HHOT日界及小时数':25,'HLT精确极值数':len(events[r.date]),'旧仅HLT极差':r.tidal_range_raw_extrema,'日界前补充变化':value-r.tidal_range_raw_extrema,'同源变量定义':'QUB日历日天文潮预报极差，原精确极值+官方小时与00/24日界；单位m；非实测潮差'})
    pd.DataFrame(rows).to_csv(OUT/'QUB_完整日界支持潮差_2014_2024.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(sources).to_csv(OUT/'原始来源清单.csv',index=False,encoding='utf-8-sig')
    state={'任务':'QUB','状态':'数据补充及日界恢复完成','自然日':len(rows),'单极值59日':'已由真实HHOT及原HLT恢复，不插值','所有日均有25个小时/日界支持':True,'可用于正式77':'是，定义明确为官方天文潮预报日历日极差，保留同一站点与单位'}
    (OUT/'任务状态.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');print(state,flush=True)

if __name__=='__main__':main()
