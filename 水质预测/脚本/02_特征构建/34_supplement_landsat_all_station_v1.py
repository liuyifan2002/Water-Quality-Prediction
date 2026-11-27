"""独立Landsat补充：按已有94站GEE脚本的月尺度与质量规则获取真实值。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';OUT=RUN/'01_独立补充任务/Landsat'
BANDS=['blue','green','red','nir','swir1','swir2','ndvi','ndwi','mndwi','ndti']

def main():
    import ee
    OUT.mkdir(parents=True,exist_ok=True);ee.Initialize(project='798288935020');ee.data.setDeadline(120000)
    meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv').rename(columns={'Station':'station'})
    sites=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon),float(r.lat)]).buffer(150),{'station':r.station}) for r in meta.itertuples()])
    def prepare(image):
        qa=image.select('QA_PIXEL');clear=qa.bitwiseAnd(2).eq(0)
        for bit in [2,3,4,5]:clear=clear.And(qa.bitwiseAnd(1<<bit).eq(0))
        optical=image.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']).multiply(.0000275).add(-.2).rename(BANDS[:6]).toFloat()
        four=optical.select(BANDS[:4]);valid=four.reduce(ee.Reducer.min()).gt(0).And(four.reduce(ee.Reducer.max()).lt(1))
        optical=optical.updateMask(clear).updateMask(image.select('QA_RADSAT').eq(0)).updateMask(valid)
        for pair,name in [(['nir','red'],'ndvi'),(['green','nir'],'ndwi'),(['green','swir1'],'mndwi'),(['red','green'],'ndti')]:optical=optical.addBands(optical.normalizedDifference(pair).rename(name))
        return optical.toFloat().copyProperties(image,['system:time_start'])
    allimages=ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')).filterBounds(sites.geometry())
    def fetch(y,m):
        p=OUT/f'Landsat月度_{y}{m:02}.csv'
        if p.exists():
            d=pd.read_csv(p);assert len(d)==94;return d
        start=ee.Date.fromYMD(y,m,1);raw=allimages.filterDate(start,start.advance(1,'month'))
        col=raw.map(prepare);empty=ee.Image.constant([0]*len(BANDS)).rename(BANDS).toFloat().updateMask(ee.Image.constant(0))
        image=ee.ImageCollection([empty]).merge(col).median()
        result=image.reduceRegions(collection=sites,reducer=ee.Reducer.mean().combine(ee.Reducer.count(),sharedInputs=True),scale=30,tileScale=4)
        result=result.map(lambda f:f.set({'landsat_image_count':raw.filterBounds(f.geometry()).size(),'year':y,'month':m}))
        records=result.getInfo()['features'];d=pd.DataFrame([r['properties'] for r in records])
        for b in BANDS:
            if b+'_mean' not in d:d[b+'_mean']=np.nan
            d[b]=pd.to_numeric(d[b+'_mean'],errors='coerce')
        d['landsat_clear_pixel_count']=pd.to_numeric(d.get('green_count',0),errors='coerce').fillna(0)
        d['landsat_strict_valid_gee']=(d.landsat_clear_pixel_count>=3)&(d.landsat_image_count>0)
        d['landsat_buffer_m']=150;d['landsat_scale_m']=30
        assert len(d)==94 and not d.station.duplicated().any()
        d.to_csv(p,index=False,encoding='utf-8-sig');return d
    pieces=[];errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(fetch,y,m):(y,m) for y in range(2014,2025) for m in range(1,13)}
        for f in as_completed(futures):
            y,m=futures[f]
            try:pieces.append(f.result())
            except Exception as ex:errors.append({'year':y,'month':m,'错误类型':type(ex).__name__})
            if (len(pieces)+len(errors))%6==0:print('Landsat月度',len(pieces),'/132','失败',len(errors),flush=True)
    state={'任务':'Landsat','完成月份':len(pieces),'失败':errors,'协议':'已有94站GEE脚本：LC08/LC09 C2 L2；自然月、150m缓冲、30m；原QA/反射率过滤，不插值','正式77纳入':'不纳入，作为77之外独立补充资产；不可将不同协议patch混入'}
    if len(pieces)==132:
        d=pd.concat(pieces,ignore_index=True).sort_values(['station','year','month']);assert len(d)==12408
        d.to_csv(OUT/'Landsat_94站132月_真实原始统计.csv',index=False,encoding='utf-8-sig');state['状态']='12408站点月补取完成，保留真实质量缺失'
    else:state['状态']='部分完成，可按月缓存续取，不用复制或插值填缺口'
    reference=ROOT/'脚本/03_GEE遥感导出/05_extract_landsat_features_gee.js';state['参考脚本SHA256']=hashlib.sha256(reference.read_bytes()).hexdigest()
    (OUT/'任务状态.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');print(state,flush=True)

if __name__=='__main__':main()
