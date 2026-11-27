"""历史候选来源原件核验：CMEMS深度/滚动、NASA逐月提取、水深与海岸距离。"""
from pathlib import Path
import ast,importlib.util,json,math,re
import numpy as np
import pandas as pd
import xarray as xr
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115';E=OUT/'证据'
OLD=ROOT/'codex/worktrees/9ff7/水质预测'
D=ROOT/'数据/外部数据/试验项目接管_20260521/data_processed_current'

def save(name,rows):pd.DataFrame(rows).to_csv(E/name,index=False,encoding='utf-8-sig')
def pure(rel,names,scope=None):
    path=ROOT/rel;tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    env={'np':np,'pd':pd,'xr':xr,'math':math,'re':re,'Path':Path,'Any':object};env.update(scope or {})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env

def main():
    E.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv').rename(columns={'Station':'station'})
    checked=[];sites=[];rolling=[];bottom=[]
    for family,filename in [('physics','copernicus_physics'),('bgc','copernicus_bgc')]:
        original=pd.read_csv(D/f'{filename}_features_2014_2024.csv',parse_dates=['date']).sort_values(['station','date']).reset_index(drop=True)
        path=OLD/f'data/raw/manual/{filename}/cmems_{family}_2014-01-01_2024-12-31.nc'
        with xr.open_dataset(path,engine='h5netcdf') as ds:
            lats=xr.DataArray(meta.lat.to_numpy(),dims='site');lons=xr.DataArray(meta.lon.to_numpy(),dims='site')
            # 与原脚本一致：interp nearest在坐标范围外保留NA，不擅自改为沿岸最近湿格。
            points=ds.interp(latitude=lats,longitude=lons,method='nearest')
            mapping={'cmems_sst':'thetao','cmems_sss':'so','cmems_uo':'uo','cmems_vo':'vo','cmems_ssh':'zos','cmems_mld':'mlotst'} if family=='physics' else {'bgc_chl':'chl','bgc_no3':'no3','bgc_po4':'po4','bgc_si':'si','bgc_o2':'o2','bgc_npp':'nppv'}
            stidx=meta.set_index('station').reset_index().reset_index().set_index('station')['index']
            ixsite=xr.DataArray(original.station.map(stidx).to_numpy(),dims='obs')
            ixtime=xr.DataArray(original.date.to_numpy(),dims='obs')
            for col,var in mapping.items():
                a=points[var]
                if 'depth' in a.dims:a=a.isel(depth=0)
                values=a.sel(time=ixtime,method='nearest').isel(site=ixsite).values
                same=np.isclose(values,original[col].to_numpy(float),rtol=1e-5,atol=1e-6,equal_nan=True)
                checked.append({'来源':filename,'字段':col,'原始文件':str(path),'核对记录':len(original),'差异':int((~same).sum()),'有效值':int(np.isfinite(values).sum())})
            for i,m in enumerate(meta.itertuples()):
                outside=not(float(ds.latitude.min())<=m.lat<=float(ds.latitude.max()) and float(ds.longitude.min())<=m.lon<=float(ds.longitude.max()))
                first='thetao' if family=='physics' else 'chl'
                a=points[first].isel(site=i,depth=0).values
                sites.append({'来源':filename,'站点':m.station,'站点超出原下载经纬范围':outside,'表层有效日':int(np.isfinite(a).sum()),'源总日期':len(ds.time),'下载经度范围':f'{float(ds.longitude.min())}—{float(ds.longitude.max())}','下载纬度范围':f'{float(ds.latitude.min())}—{float(ds.latitude.max())}'})
            if family=='physics':
                a=points.thetao.sel(time=ixtime,method='nearest').isel(site=ixsite).transpose('obs','depth').values
                valid=np.isfinite(a);idx=np.where(valid,np.arange(a.shape[1]),-1).max(axis=1)
                deepest=np.where(idx>=0,a[np.arange(len(a)),np.maximum(idx,0)],np.nan)
                for j,r in enumerate(original.itertuples()):bottom.append({'站点':r.station,'日期':r.date,'旧最末层温度':r.cmems_bottom_temp,'原文件最后深度m':float(ds.depth[-1]),'该格最深有效层温度':deepest[j],'该格最深有效层m':float(ds.depth[idx[j]]) if idx[j]>=0 else np.nan,'意义':'格点最深有效模型层代理，非站点实测底温；仅诊断候选'})
        bases=['cmems_sss','cmems_mld','cmems_current_speed'] if family=='physics' else ['bgc_no3','bgc_po4','bgc_npp']
        for col in bases:
            for n in ([3,7,14] if family=='physics' else [7]):
                field=f'{col}_{n}d';want=original.groupby('station')[col].transform(lambda s:s.rolling(n,min_periods=1).mean())
                rolling.append({'来源':filename,'字段':field,'原值与按记录数滚动不一致':int((~np.isclose(want,original[field],equal_nan=True,rtol=1e-6,atol=1e-7)).sum()),'真实定义':f'最近{n}次不规则监测记录含当前值，非{n}个自然日','应处理':'若保留原值改名为采样记录窗口；若声称日窗则从完整日场另建，须重算实验'})
    save('13_CMEMS原始表层字段复核.csv',checked);save('14_CMEMS真实空间覆盖与越界.csv',sites);save('15_CMEMS历史滚动字段单位复核.csv',rolling);save('16_CMEMS末层与最深有效层诊断.csv',bottom)
    print('CMEMS原件核查完成',flush=True)
    rel='脚本/历史版本/试验项目接管_20260521/data/run_nasa_oceancolor_fullrun.py'
    funcs=pure(rel,{'nearest_index','haversine_km','extract_value_from_dataarray','granule_period','add_monthly_derived_features'})
    wide=pd.read_csv(D/'nasa_oceancolor_features_2014_2024_wide.csv')
    nasa=[];products=[]
    for product,var,outcol in [('VIIRSN_L3m_CHL','chlor_a','viirs_chl'),('VIIRSN_L3m_KD','Kd_490','viirs_kd490')]:
        folder=OLD/f'data/raw/manual/ocean_color/nasa_oceancolor_monthly_9km/{product}'
        paths=sorted(folder.glob('*.nc'));periods=[]
        for path in paths:
            period=funcs['granule_period'](path.name)
            if period is None:continue
            periods.append(str(period))
            with xr.open_dataset(path,engine='h5netcdf') as ds:
                # 只读香港附近足够覆盖5x5回退的切片，保持原像元坐标和提取规则。
                da=ds[var];lat=da.lat.values;lon=da.lon.values
                da=da.sel(lat=slice(23.3,21.4) if lat[0]>lat[-1] else slice(21.4,23.3),lon=slice(113.0,115.5))
                da.load()
                for m in meta.itertuples():
                    r=funcs['extract_value_from_dataarray'](da,m.lat,m.lon)
                    prev=wide[(wide.station_id==m.station)&(wide.year==period.year)&(wide.month==period.month)][outcol]
                    assert len(prev)==1
                    same=np.isclose(r['extracted_value'],prev.iloc[0],rtol=1e-6,atol=1e-7,equal_nan=True)
                    nasa.append({'产品':product,'站点':m.station,'年月':str(period),'原始文件':str(path),'提取值':r['extracted_value'],'冻结值':prev.iloc[0],'一致':bool(same),'原规则':r['extraction_method'],'有效像元':r['valid_pixel_count'],'最近像元距离km':r['nearest_pixel_distance']})
        products.append({'产品':product,'实际文件':len(paths),'有效自然月':len(set(periods)),'重复自然月':len(periods)-len(set(periods)),'缺失自然月':';'.join(sorted(set(str(p) for p in pd.period_range('2014-01','2024-12',freq='M'))-set(periods)))})
        print(product,'逐月原件核验完成',flush=True)
    save('17_NASA两核心产品24816格原始复核.csv',nasa);save('18_NASA核心自然月完整性.csv',products)
    basecols=[c for c in wide if not any(c.endswith(s) for s in ['_lag1','_rolling3_mean','_rolling3_anomaly','_train_climatology','_monthly_anomaly'])]
    recalculated=funcs['add_monthly_derived_features'](wide[basecols])
    derived=[]
    for c in wide:
        if c in recalculated and c not in basecols:derived.append({'字段':c,'核对数':len(wide),'不一致':int((~np.isclose(wide[c],recalculated[c],equal_nan=True,rtol=1e-6,atol=1e-7)).sum()),'协议边界':'月均含当月，lag1按完整自然月；气候态仅<=2020年，不能当所有验证协议统一训练基准'})
    save('19_NASA月滞后与气候态复核.csv',derived)
    # 官方站点水深重新从原报告文本提取，而非根据旧特征猜数。
    from pypdf import PdfReader
    pdf=OLD/'tmp/official_depth_sources/epd_2002_report.pdf'
    text='\n'.join(p.extract_text() or '' for p in PdfReader(str(pdf)).pages)
    dep=pure('脚本/历史版本/试验项目接管_20260521/data/build_epd_station_depth_features.py',{'normalize_text','extract_station_depths'})
    values=dep['extract_station_depths'](dep['normalize_text'](text),meta.station.tolist())
    frozen=pd.read_csv(D/'station_depth_features.csv').set_index('station');rows=[]
    for s in meta.station:rows.append({'站点':s,'原PDF提取水深m':values.get(s,np.nan),'历史表水深m':frozen.loc[s,'station_depth_approx_m'],'一致':bool(np.isclose(values.get(s,np.nan),frozen.loc[s,'station_depth_approx_m'],equal_nan=True)),'来源':str(pdf),'限制':'2002近似站点总水深；0.2/0.6/0.9层深为人为代理不是实测采样深度'})
    save('20_94站水深官方PDF溯源.csv',rows)
    import geopandas as gpd
    from shapely.geometry import Point
    coast=OLD/'data/raw/manual/coastal_context/ne_10m_coastline.zip'
    shore=gpd.read_file(f'zip://{coast}').to_crs('EPSG:2326');geom=shore.geometry.union_all()
    stations=gpd.GeoSeries([Point(r.lon,r.lat) for r in meta.itertuples()],crs='EPSG:4326').to_crs('EPSG:2326')
    olddist=pd.read_csv(D/'coastal_distance_station_features.csv').set_index('station');rows=[]
    for s,dist in zip(meta.station,stations.distance(geom)/1000):rows.append({'站点':s,'原海岸线重建km':dist,'历史km':olddist.loc[s,'distance_to_coast_km'],'一致':bool(np.isclose(dist,olddist.loc[s,'distance_to_coast_km'],rtol=1e-6,atol=1e-7)),'限制':'Natural Earth 1:10m广义海岸线，近岸细节精度不足；不等于真实最近岸线实测距离'})
    save('21_94站近岸距离原始矢量复核.csv',rows)
    print('原始来源专项核查全部完成',flush=True)

if __name__=='__main__':main()
