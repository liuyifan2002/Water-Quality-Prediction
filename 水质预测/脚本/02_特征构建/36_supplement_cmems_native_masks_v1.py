"""独立海陆掩膜任务：提取原生有效格点及实际匹配距离；不挪站点填满数据。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import xarray as xr
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';OUT=RUN/'01_独立补充任务/CMEMS海陆掩膜'

def hav(lat,lon,lats,lons):
    a=np.sin(np.radians(lats-lat)/2)**2+np.cos(np.radians(lat))*np.cos(np.radians(lats))*np.sin(np.radians(lons-lon)/2)**2
    return 6371.0088*2*np.arcsin(np.minimum(np.sqrt(a),1))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv').rename(columns={'Station':'station'})
    prior=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115/真实补取/CMEMS全站扩边'
    files=[(prior/'physics_94站扩边_20131215_20241231.nc','thetao'),(prior/'bgc_94站扩边_20131215_20241231.nc','chl')]
    for p in (ROOT/'codex/worktrees/9ff7/水质预测/data/raw/manual/ocean_color/cmems').glob('*.nc'):
        with xr.open_dataset(p,engine='h5netcdf') as ds:files.append((p,list(ds.data_vars)[0]))
    grids=[];matches=[];sources=[]
    for p,var in files:
        with xr.open_dataset(p,engine='h5netcdf') as ds:
            a=ds[var]
            if 'depth' in a.dims:a=a.isel(depth=0)
            a=a.transpose('time','latitude','longitude');v=a.values;fraction=np.isfinite(v).mean(axis=0)
            yy,xx=np.meshgrid(ds.latitude.values,ds.longitude.values,indexing='ij')
            stable=fraction>=.95
            for j in range(len(ds.latitude)):
                for k in range(len(ds.longitude)):grids.append({'来源文件':str(p),'变量':var,'原生纬度':float(yy[j,k]),'原生经度':float(xx[j,k]),'有限有效日期比例':float(fraction[j,k]),'稳定有效>=95%':bool(stable[j,k]),'任何日有数据':bool(fraction[j,k]>0)})
            for m in meta.itertuples():
                j=int(np.argmin(abs(ds.latitude.values-m.lat)));k=int(np.argmin(abs(ds.longitude.values-m.lon)))
                distance=hav(m.lat,m.lon,yy,xx);wet=np.where(stable,distance,np.inf);idx=np.unravel_index(np.argmin(wet),wet.shape)
                inside=float(ds.latitude.min())<=m.lat<=float(ds.latitude.max()) and float(ds.longitude.min())<=m.lon<=float(ds.longitude.max())
                matches.append({'来源文件':str(p),'站点':m.station,'站点经度':m.lon,'站点纬度':m.lat,'原生最近格点经度':float(xx[j,k]),'原生最近格点纬度':float(yy[j,k]),'匹配格点距离km':float(distance[j,k]),'坐标范围内':inside,'原位置原生格点有效比例':float(fraction[j,k]),'最近稳定湿格距离km':float(wet[idx]) if np.isfinite(wet[idx]) else np.nan,'最近稳定湿格经度':float(xx[idx]),'最近稳定湿格纬度':float(yy[idx]),'是否采用湿格位移':False,'验收决定':'原位置原生匹配保留' if inside and fraction[j,k]>0 else '原位置缺失继续NA；湿格仅诊断，未证明精细海岸连通，不自动代入'})
            sources.append({'来源文件':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'变量':var,'时间元素':len(ds.time),'原始空间格':int(yy.size),'定义':'供应者原生有效性掩膜，非人为绘制海岸线；有限值比例不等于实测覆盖'})
    pd.DataFrame(grids).to_csv(OUT/'原生格点有效性掩膜.csv',index=False,encoding='utf-8-sig');pd.DataFrame(matches).to_csv(OUT/'94站原生匹配与湿格候选距离.csv',index=False,encoding='utf-8-sig');pd.DataFrame(sources).to_csv(OUT/'原始来源清单.csv',index=False,encoding='utf-8-sig')
    state={'任务':'CMEMS海陆掩膜','状态':'真实原生支持掩膜与94站距离核验完成','站点匹配记录':len(matches),'空间位移是否纳入正式输入':False,'湿格替代':'暂不能纳入；缺精细岸线连通约束，不用很远海点填站点','正式77':'未引入CMEMS新字段，保留原77定义；不因本补充改变站点坐标'}
    (OUT/'任务状态.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');print(state,flush=True)

if __name__=='__main__':main()
