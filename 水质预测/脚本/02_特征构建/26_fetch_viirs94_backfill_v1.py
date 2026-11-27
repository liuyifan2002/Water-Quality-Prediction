"""同源VIIRS月度补取：94站、原1000m缓冲/500m提取；不覆盖历史表或发起Drive导出。"""
from pathlib import Path
import argparse,json,hashlib,datetime
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/图片数据完整性整改_v1_20261115'
COLLECTION='NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'

def main():
    p=argparse.ArgumentParser();p.add_argument('--project');p.add_argument('--fetch',action='store_true');a=p.parse_args()
    dest=OUT/'数据补充';dest.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv').rename(columns={'Station':'station'})
    assert len(meta)==meta.station.nunique()==94
    old=pd.read_csv(ROOT/'数据/处理后数据/正式主表/viirs_ntl_station_monthly_2014_2024.csv')
    keys=['station','year','month'];seen=set(map(tuple,old[keys].to_numpy()))
    req=pd.DataFrame([{**r,'year':year,'month':month} for r in meta[['station','lon','lat']].to_dict('records') for year in range(2014,2025) for month in range(1,13)])
    req['已有记录']=[tuple(r) in seen for r in req[keys].to_numpy()]
    req['正式水区']=req.station.map(meta.set_index('station').water_zone_en)
    req.to_csv(dest/'94站132月_完整补取请求.csv',index=False,encoding='utf-8-sig')
    status={'源集合':COLLECTION,'请求数':len(req),'已有记录':int(req.已有记录.sum()),'缺少记录':int((~req.已有记录).sum()),'缓冲m':1000,'提取尺度m':500,'数据原生像元m':463.83,'单位':'nW cm^-2 sr^-1','时间频率':'自然月','坐标系':'EPSG:4326','获取日期':datetime.date.today().isoformat(),'新增真实记录':0}
    if not a.fetch:
        status['状态']='仅准备请求，未发起补取'
    else:
        try:
            import ee
            if a.project:ee.Initialize(project=a.project)
            else:ee.Initialize()
            ee.data.setDeadline(60000)
            col=ee.ImageCollection(COLLECTION).select(['avg_rad','cf_cvg'])
            sites=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon),float(r.lat)]).buffer(1000),{'station':r.station}) for r in meta.itertuples()])
            frames=[]
            for year in range(2014,2025):
                pending=[]
                for month in range(1,13):
                    if (dest/f'原始同源月度_{year}{month:02}.csv').exists():continue
                    start=ee.Date.fromYMD(year,month,1);images=col.filterDate(start,start.advance(1,'month'))
                    result=images.mean().reduceRegions(collection=sites,reducer=ee.Reducer.mean(),scale=500,tileScale=4)
                    result=result.map(lambda f,yy=year,mm=month,cc=images.size():f.set({'year':yy,'month':mm,'viirs_count':cc,'viirs_buffer_m':1000}))
                    pending.append(result)
                if pending:
                    raw=ee.FeatureCollection(pending).flatten().getInfo()
                    frame=pd.DataFrame([x['properties'] for x in raw['features']]).rename(columns={'avg_rad':'ntl_avg_rad','cf_cvg':'ntl_cf_cvg'})
                    for month in range(1,13):
                        path=dest/f'原始同源月度_{year}{month:02}.csv'
                        if path.exists():continue
                        part=frame[frame.month.eq(month)][['station','year','month','ntl_avg_rad','ntl_cf_cvg','viirs_count','viirs_buffer_m']]
                        assert len(part)==94 and not part.duplicated(keys).any()
                        part.to_csv(path,index=False,encoding='utf-8-sig')
                for month in range(1,13):
                    path=dest/f'原始同源月度_{year}{month:02}.csv'
                    frame=pd.read_csv(path)
                    frames.append(frame)
                print(f'同源补取完成 {year}年，12月×94站',flush=True)
            all_rows=pd.concat(frames,ignore_index=True)
            assert len(all_rows)==12408 and not all_rows.duplicated(keys).any()
            all_rows.to_csv(dest/'VIIRS_94站132月_同源原始提取.csv',index=False,encoding='utf-8-sig')
            status['状态']='补取完成，尚未替换正式输入'
            status['新增真实记录']=int(sum(tuple(r) not in seen for r in all_rows[keys].to_numpy()))
        except Exception as ex:
            status['状态']='补取未完成'
            # 不记录凭证、认证文件或访问令牌。
            status['阻塞类别']=type(ex).__name__
            status['原因']=str(ex)[:1000]
    (dest/'同源补取执行状态.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
