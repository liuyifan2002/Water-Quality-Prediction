"""独立CMEMS海色补充任务：真实完整日场、自然日窗口、原件重叠核查。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,json,hashlib
import numpy as np
import pandas as pd
import xarray as xr
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';OUT=RUN/'01_独立补充任务/CMEMS海色'
OLD=ROOT/'codex/worktrees/9ff7/水质预测/data/raw/manual/ocean_color/cmems'

def main():
    import copernicusmarine as cm
    OUT.mkdir(parents=True,exist_ok=True);env=dict(os.environ)
    for line in (ROOT/'codex/worktrees/9ff7/水质预测/.env').read_text(encoding='utf-8-sig').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);env.setdefault(k.strip(),v.strip().strip('\"').strip("'"))
    meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv').rename(columns={'Station':'station'})
    def fetch(old):
        id_=old.name.split('_2014-')[0];path=OUT/(id_+'_20131215_20241231.nc')
        with xr.open_dataset(old,engine='h5netcdf') as d:vars=list(d.data_vars)
        if not path.exists():
            cm.subset(dataset_id=id_,dataset_version='202603',variables=vars,minimum_longitude=float(meta.lon.min()-.08),maximum_longitude=float(meta.lon.max()+.08),minimum_latitude=float(meta.lat.min()-.08),maximum_latitude=float(meta.lat.max()+.08),start_datetime='2013-12-15',end_datetime='2024-12-31',coordinates_selection_method='outside',file_format='netcdf',output_filename=path.name,output_directory=str(OUT),username=env.get('COPERNICUSMARINE_USERNAME'),password=env.get('COPERNICUSMARINE_PASSWORD'),disable_progress_bar=True,skip_existing=True)
        return old,path,id_
    done=[];errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(fetch,p) for p in OLD.glob('*.nc')]):
            try:done.append(f.result())
            except Exception as ex:errors.append(type(ex).__name__)
    alias={'CHL':'oc_chl','KD490':'oc_kd490','ZSD':'oc_zsd','RRS412':'oc_rrs_412','RRS443':'oc_rrs_443','RRS490':'oc_rrs_490','RRS555':'oc_rrs_555','RRS670':'oc_rrs_670'}
    dailyframes=[];checks=[];sources=[]
    for old,path,id_ in done:
        with xr.open_dataset(path,engine='h5netcdf') as ds,xr.open_dataset(old,engine='h5netcdf') as prev:
            for var in prev.data_vars:
                # 新扩边如不含旧下载的边缘格，只比较真正共同的原始格点。
                coords={c:np.intersect1d(ds[c].values,prev[c].values) for c in ['time','latitude','longitude']}
                a=prev[var].sel(coords).values;b=ds[var].sel(coords).values
                checks.append({'产品':id_,'变量':var,'重叠元素':int(a.size),'不一致':int((~np.isclose(a,b,equal_nan=True,rtol=1e-5,atol=1e-7)).sum())})
            point=ds.interp(latitude=xr.DataArray(meta.lat.values,dims='site'),longitude=xr.DataArray(meta.lon.values,dims='site'),method='nearest')
            for var in ds.data_vars:
                if var not in alias:continue
                a=point[var].transpose('time','site').values
                for i,s in enumerate(meta.station):dailyframes.append(pd.DataFrame({'station':s,'date':pd.DatetimeIndex(ds.time.values).normalize(),alias[var]:a[:,i]}))
            sources.append({'产品':id_,'文件':str(path),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'变量':','.join(ds.data_vars),'定义':'多传感器海色，非独立Sentinel-3 OLCI；原位置nearest；不替换海陆掩膜'})
    pd.DataFrame(checks).to_csv(OUT/'新旧原始格点重叠核对.csv',index=False,encoding='utf-8-sig');pd.DataFrame(sources).to_csv(OUT/'原始来源清单.csv',index=False,encoding='utf-8-sig')
    if done:
        # 各变量独立读取后在相同station/date聚合，非重复生成独立样本。
        d=pd.concat(dailyframes,ignore_index=True).groupby(['station','date'],as_index=False).first().sort_values(['station','date'])
        d.to_csv(OUT/'CMEMS海色_真实逐日94站.csv',index=False,encoding='utf-8-sig')
        samples=pd.read_csv(ROOT/'数据/处理后数据/正式主表/gee_station_date_samples.csv',parse_dates=['date']).rename(columns={'Station':'station'})
        rows=[]
        for station,g in d.groupby('station'):
            g=g.set_index('date');assert g.index.is_unique
            sd=samples.loc[samples.station.eq(station),['sample_id','station','date']].copy()
            for col in [c for c in g if c.startswith('oc_')]:
                sd[col]=g[col].reindex(sd.date).values
                for n in [3,7,14]:
                    sd[f'{col}_previous_{n}d_mean']=g[col].shift(1).rolling(n,min_periods=n).mean().reindex(sd.date).values
                    sd[f'{col}_previous_{n}d_n_days']=g[col].shift(1).rolling(n,min_periods=1).count().reindex(sd.date).values
            rows.append(sd)
        result=pd.concat(rows).sort_values(['station','date']);assert len(result)==10866
        result.to_csv(OUT/'CMEMS海色_真实自然日窗候选.csv',index=False,encoding='utf-8-sig')
    status={'任务':'CMEMS海色','完整产品':len(done),'错误类型':errors,'状态':'补充与候选完成' if len(done)==3 and not errors else '部分完成','旧重叠数值一致':all(r['不一致']==0 for r in checks),'正式77纳入':'不纳入，作为独立历史海色输入版本','缺失处理':'真实海陆/云掩膜继续NA，不算0；有效天数不足则自然日窗NA'}
    (OUT/'任务状态.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8');print(status,flush=True)

if __name__=='__main__':main()
