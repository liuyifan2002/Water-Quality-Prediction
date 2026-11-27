"""补取同源ERA5研究期之前的必要窗口前史；不扩大水质训练时段。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.util,json,zipfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115'

def main():
    import cdsapi
    folder=OUT/'真实补取/ERA5_2013年前史';folder.mkdir(parents=True,exist_ok=True)
    config={}
    for p in [ROOT/'待确认_未归类/.cdsapirc',Path.home()/'.cdsapirc']:
        if config.get('url') and config.get('key'):break
        if p.exists():
            try:text=p.read_text(encoding='utf-8-sig')
            except PermissionError:continue
            for line in text.splitlines():
                if ':' in line:
                    k,v=line.split(':',1)
                    if k.strip() in ['url','key']:config.setdefault(k.strip(),v.strip())
    source=ROOT/'脚本/02_特征构建/24_download_prepare_era5_ssrd_sst_daily.py'
    spec=importlib.util.spec_from_file_location('prehistory',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def fetch(point):
        label=point['label'];path=folder/f'{label}_20131215_20140102.zip'
        row={'点位':label,'状态':'开始真实补取','文件':str(path),'源产品':'reanalysis-era5-single-levels-timeseries'}
        try:
            if not path.exists():
                client=cdsapi.Client(**config,timeout=60,quiet=True,progress=False,retry_max=1,sleep_max=5)
                client.retrieve('reanalysis-era5-single-levels-timeseries',{'variable':m.VARIABLES,'location':{'longitude':point['longitude'],'latitude':point['latitude']},'date':['2013-12-15/2014-01-02'],'data_format':'netcdf'},str(path))
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as z:
                    names=[n for n in z.namelist() if n.endswith('.nc')];assert len(names)==1
                    nc=folder/f'{label}_20131215_20140102.nc'
                    if not nc.exists():nc.write_bytes(z.read(names[0]))
            else:nc=path
            row.update(状态='同源前史原件已补取',NetCDF=str(nc))
        except Exception as ex:row.update(状态='未完成，不用插值代替',错误类型=type(ex).__name__)
        print(label,row['状态'],flush=True);return row
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(fetch,p) for p in m.OFFSHORE_POINTS]):
            rows.append(f.result());(folder/'补取执行状态.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    if not all(r['状态'].startswith('同源') for r in rows):return
    fresh=m.aggregate_daily([(r['点位'],Path(r['NetCDF'])) for r in rows])
    old=pd.read_csv(ROOT/'数据/处理后数据/衍生版本/era5_ssrd_sst_daily_2014_2024.csv',parse_dates=['date'])
    common=fresh.merge(old,on='date',suffixes=('_补取','_既有'))
    import numpy as np
    check=[]
    for c in ['ssrd_sum_j_m2','ssrd_mean_j_m2','sst_mean_c','sst_max_c','sst_min_c']:
        check.append({'变量':c,'重叠日数':len(common),'差异':int((~np.isclose(common[c+'_补取'],common[c+'_既有'],equal_nan=True,rtol=1e-6,atol=1e-7)).sum())})
    pd.DataFrame(check).to_csv(OUT/'证据/31_ERA5前史补取重叠验证.csv',index=False,encoding='utf-8-sig')
    if any(r['差异'] for r in check):return
    pre=fresh[fresh.date.lt('2014-01-01')].copy()
    # 保持原研究期气候态定义；其时间外推边界另审，不因追加17天暗改全期异常基准。
    clim=old.groupby(old.date.dt.dayofyear).sst_mean_c.mean()
    pre['sst_anom_c']=pre.sst_mean_c-pre.date.dt.dayofyear.map(clim)
    combined=pd.concat([pre,old],ignore_index=True).sort_values('date')
    combined.to_csv(OUT/'修复候选/ERA5日值_补充2013年前史_未用于实验.csv',index=False,encoding='utf-8-sig')
    print('同源前史完成，保留原研究期日值及气候态定义',flush=True)

if __name__=='__main__':main()
