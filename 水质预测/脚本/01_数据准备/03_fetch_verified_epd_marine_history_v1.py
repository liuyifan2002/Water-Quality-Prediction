"""补取真正的EPD年度海水历史数据，防止近期/月度或泳滩数据误作长历史。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from io import BytesIO
import hashlib
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115'
DEST=OUT/'真实补取/EPD真实海水年度_1986_2024'

def fetch(year):
    url=f'https://cd.epic.epd.gov.hk/marinepsi/en/marinehistorical/marine-historical-{year}-en.csv'
    path=DEST/f'marine-historical-{year}-en.csv';r={'年度':year,'官方URL':url,'保存路径':str(path)}
    try:
        content=path.read_bytes() if path.exists() else requests.get(url,timeout=60).content
        d=pd.read_csv(BytesIO(content))
        required={'Station','Dates','Depth'}
        if not required.issubset(d):raise ValueError('不是含站点日期水层的真实海水历史表')
        dates=pd.to_datetime(d.Dates,errors='coerce')
        assert len(d)>0 and dates.notna().all() and dates.dt.year.eq(year).all()
        if not path.exists():path.write_bytes(content)
        r.update(状态='已补取且年度/海水字段通过',记录数=len(d),站点数=d.Station.nunique(),开始日期=str(dates.min().date()),结束日期=str(dates.max().date()),SHA256=hashlib.sha256(content).hexdigest())
        old=Path(f'C:/Users/刘逸凡/Desktop/开题/水质数据/marine-historical-{year}-en.csv')
        if old.exists():
            previous=pd.read_csv(old);r['既有2014至2024原始文件字节相同']=hashlib.sha256(old.read_bytes()).hexdigest()==r['SHA256']
            r['既有同年全表数值与文本一致']=previous.equals(d)
    except Exception as ex:r.update(状态='未取得合格年度海水数据',错误类型=type(ex).__name__)
    return r

def main():
    DEST.mkdir(parents=True,exist_ok=True);rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch,y) for y in range(1986,2025)]
        for future in as_completed(futures):
            r=future.result();rows.append(r);print(r['年度'],r['状态'],flush=True)
            pd.DataFrame(sorted(rows,key=lambda r:r['年度'])).to_csv(DEST/'真实海水年度下载及内容验收.csv',index=False,encoding='utf-8-sig')
    print('已验收年度',sum(r['状态'].startswith('已补取') for r in rows),'/',len(rows),flush=True)

if __name__=='__main__':main()
