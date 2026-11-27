"""验收真实CMEMS扩边原件，另建自然日窗及最深有效模型层代理候选。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115';E=OUT/'证据';C=OUT/'修复候选'
OLD=ROOT/'codex/worktrees/9ff7/水质预测';D=ROOT/'数据/外部数据/试验项目接管_20260521/data_processed_current'

def main():
    C.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv').rename(columns={'Station':'station'})
    samples=pd.read_csv(ROOT/'数据/处理后数据/正式主表/gee_station_date_samples.csv',parse_dates=['date']).rename(columns={'Station':'station'})
    allchecks=[];states=[];support=[]
    for family in ['physics','bgc']:
        p=OUT/f'真实补取/CMEMS全站扩边/{family}_94站扩边_20131215_20241231.nc'
        if not p.exists():states.append({'来源':family,'状态':'补取未完成，没有生成候选'});continue
        oldpath=OLD/f'data/raw/manual/copernicus_{family}/cmems_{family}_2014-01-01_2024-12-31.nc'
        with xr.open_dataset(p,engine='h5netcdf') as ds, xr.open_dataset(oldpath,engine='h5netcdf') as prev:
            for var in prev.data_vars:
                new=ds[var].sel(time=prev.time,latitude=prev.latitude,longitude=prev.longitude)
                if 'depth' in prev[var].dims:new=new.sel(depth=prev.depth)
                a=prev[var].values;b=new.transpose(*prev[var].dims).values
                ok=np.isclose(a,b,equal_nan=True,rtol=1e-6,atol=1e-7)
                allchecks.append({'来源':family,'变量':var,'重叠元素':int(a.size),'差异':int((~ok).sum()),'原件':str(oldpath),'补取':str(p)})
            oldf=pd.read_csv(D/f'copernicus_{family}_features_2014_2024.csv')
            points=ds.interp(latitude=xr.DataArray(meta.lat.to_numpy(),dims='site'),longitude=xr.DataArray(meta.lon.to_numpy(),dims='site'),method='nearest')
            mapping={'cmems_sst':'thetao','cmems_sss':'so','cmems_uo':'uo','cmems_vo':'vo','cmems_ssh':'zos','cmems_mld':'mlotst'} if family=='physics' else {'bgc_chl':'chl','bgc_no3':'no3','bgc_po4':'po4','bgc_si':'si','bgc_o2':'o2','bgc_npp':'nppv'}
            values={}
            for col,var in mapping.items():
                a=points[var]
                if 'depth' in a.dims:a=a.isel(depth=0)
                values[col]=a.transpose('time','site').values
            if family=='physics':
                a=points.thetao.transpose('time','site','depth').values
                idx=np.where(np.isfinite(a),np.arange(a.shape[-1]),-1).max(axis=-1)
                pick=np.maximum(idx,0)
                values['cmems_deepest_valid_layer_temp_proxy']=np.where(idx>=0,np.take_along_axis(a,pick[...,None],axis=-1)[...,0],np.nan)
                values['cmems_deepest_valid_layer_depth_m']=np.where(idx>=0,ds.depth.values[pick],np.nan)
            result=[]
            for i,m in enumerate(meta.itertuples()):
                daily=pd.DataFrame({c:v[:,i] for c,v in values.items()},index=pd.DatetimeIndex(ds.time.values).normalize())
                assert daily.index.is_unique
                if family=='physics':daily['cmems_current_speed']=np.sqrt(daily.cmems_uo**2+daily.cmems_vo**2)
                dates=pd.DatetimeIndex(samples.loc[samples.station.eq(m.station),'date'])
                g=daily.reindex(dates).reset_index(names='date');g['station']=m.station
                bases=['cmems_sss','cmems_mld','cmems_current_speed'] if family=='physics' else ['bgc_no3','bgc_po4','bgc_npp']
                for col in bases:
                    for n in ([3,7,14] if family=='physics' else [7]):
                        counts=daily[col].shift(1).rolling(n,min_periods=1).count()
                        mean=daily[col].shift(1).rolling(n,min_periods=n).mean()
                        g[f'{col}_previous_{n}d_mean']=mean.reindex(dates).to_numpy()
                        g[f'{col}_previous_{n}d_n_days']=counts.reindex(dates).to_numpy()
                result.append(g)
                outside=not(float(ds.latitude.min())<=m.lat<=float(ds.latitude.max()) and float(ds.longitude.min())<=m.lon<=float(ds.longitude.max()))
                col='cmems_sst' if family=='physics' else 'bgc_chl'
                support.append({'来源':family,'站点':m.station,'仍超下载范围':outside,'采样记录':len(g),'旧有效采样':int(oldf.loc[oldf.station.eq(m.station),col].notna().sum()),'扩边后有效采样':int(g[col].notna().sum()),'仍缺失原因':'最近格点海陆掩膜/变量有效支持，不擅自换位置或填补'})
            candidate=pd.concat(result,ignore_index=True).sort_values(['station','date'])
            assert len(candidate)==10866 and not candidate.duplicated(['station','date']).any()
            candidate.to_csv(C/f'CMEMS_{family}_真实日窗及覆盖修复候选_未用于实验.csv',index=False,encoding='utf-8-sig')
            states.append({'来源':family,'状态':'独立候选已生成','记录数':len(candidate),'定义':'原位置nearest；日窗为采样前完整自然日；深层仅为格点最深有效模型层代理','原重叠一致':all(r['差异']==0 for r in allchecks if r['来源']==family),'误称实测底温':False})
    pd.DataFrame(allchecks).to_csv(E/'23_CMEMS扩边重叠原始数值核验.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(support).to_csv(E/'24_CMEMS扩边前后94站实际支持.csv',index=False,encoding='utf-8-sig')
    (E/'CMEMS候选修复状态.json').write_text(json.dumps(states,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(states,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
