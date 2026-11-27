"""全量数据文件、字段及获取入口复核；不执行模型计算或改写原件。"""
from pathlib import Path
import hashlib, json, time, zipfile, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115'
E=OUT/'证据'
sys.path.insert(0,str(OUT/'运行依赖'))

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(name, rows):pd.DataFrame(rows).to_csv(E/name,index=False,encoding='utf-8-sig')

def main():
    E.mkdir(parents=True,exist_ok=True)
    roots=[ROOT/'数据',Path('C:/Users/刘逸凡/Desktop/开题/水质数据'),Path('E:/研究生/codex_chla_optimization/水质预测/data'),ROOT/'codex/worktrees/9ff7/水质预测/data',ROOT/'codex/worktrees/9ff7/水质预测/tmp/official_depth_sources']
    supplementary='--supplementary-csv' in sys.argv
    extra=[ROOT/'归档_历史版本',ROOT/'待确认_未归类',ROOT/'transfer/llm_extension_transfer_v1/data']
    files=[]
    for base in (extra if supplementary else roots):
        if base.exists():files.extend(p for p in base.rglob('*.csv' if supplementary else '*') if p.is_file() and '.git' not in p.parts)
    files=sorted(set(files));inventory=[];fields=[];segments=[];binary=[];hashes={}
    if supplementary:
        inventory=pd.read_csv(E/'01_全数据文件清单.csv',low_memory=False).to_dict('records')
        fields=pd.read_csv(E/'02_全字段质量画像.csv',low_memory=False).to_dict('records')
        segments=pd.read_csv(E/'03_数据覆盖分站分年.csv',low_memory=False).to_dict('records')
        binary=pd.read_csv(E/'04_NetCDF变量与真实覆盖.csv',low_memory=False).to_dict('records')
        hashes=json.loads((E/'全数据原件SHA256.json').read_text(encoding='utf-8'))
        seen={r['路径'] for r in inventory};files=[p for p in files if str(p) not in seen]
        roots+=extra
    for i,p in enumerate(files):
        row={'路径':str(p),'字节':p.stat().st_size,'后缀':p.suffix.lower(),'核验层次':'元信息','状态':'已登记'}
        if p.suffix.lower() in ['.csv','.nc','.zip','.xls','.xlsx','.shp','.dbf','.json']:
            hashes[str(p)]=digest(p);row['SHA256']=hashes[str(p)]
        try:
            if p.suffix.lower()=='.csv':
                opts={'low_memory':False}
                if p.name=='daily_HKO_RF_ALL.csv':opts['skiprows']=2
                try:d=pd.read_csv(p,**opts)
                except UnicodeDecodeError:d=pd.read_csv(p,encoding='gb18030',**opts)
                row.update(行数=len(d),列数=len(d.columns),核验层次='全部行全部字段画像',完全重复行=int(d.duplicated().sum()))
                station=next((c for c in ['station','Station','station_id','A_Station'] if c in d),None)
                date=next((c for c in ['date','Dates','datetime','time','valid_time'] if c in d),None)
                dates=pd.to_datetime(d[date],errors='coerce') if date else None
                if station:row['站点数']=int(d[station].nunique())
                if dates is not None:row.update(开始=str(dates.min()),结束=str(dates.max()),独立日期数=int(dates.nunique()),非法日期=int((dates.isna()&d[date].notna()).sum()))
                keys=next((k for k in [['obs_id'],['sample_id'],['station','year','month'],['station','date'],['Station','date'],['year','month'],['date']] if all(c in d for c in k)),None)
                if keys:row.update(候选键='+'.join(keys),候选键重复=int(d.duplicated(keys).sum()),键说明='候选键重复仅为证据，水层/实验长表需按真实粒度解释')
                for c in d:
                    s=d[c];n=int(s.notna().sum());num=pd.to_numeric(s,errors='coerce');finite=np.isfinite(num.to_numpy(dtype=float,na_value=np.nan))
                    f={'路径':str(p),'字段':c,'类型':str(s.dtype),'行数':len(d),'非缺失':n,'缺失率':float(s.isna().mean()) if len(d) else None,'唯一值':int(s.nunique(dropna=True)),'可转数值':int(num.notna().sum()),'无穷数':int((num.notna()&~finite).sum()),'零值':int(num.eq(0).sum()),'负值':int(num.lt(0).sum()),'最小':float(num[finite].min()) if finite.any() else None,'最大':float(num[finite].max()) if finite.any() else None}
                    if station:f['有非缺失站点数']=int(d.loc[s.notna(),station].nunique())
                    fields.append(f)
                # 分站、分年按数值字段整体缺失量定位覆盖异常；关键主表的逐字段分组另由110执行。
                numeric=d.select_dtypes(include='number').columns.tolist()
                for kind,group in ([('站点',d[station])] if station else [])+([('年份',dates.dt.year)] if dates is not None else []):
                    for key,g in d.groupby(group,dropna=False):
                        segments.append({'路径':str(p),'分组维度':kind,'分组值':key,'行数':len(g),'数值字段数':len(numeric),'数值单元格缺失率':float(g[numeric].isna().to_numpy().mean()) if numeric else None})
            elif p.suffix.lower()=='.nc':
                import xarray as xr
                # 部分历史测试包以.nc命名但实际为ZIP；先检查真实格式。
                payload=p
                if zipfile.is_zipfile(p):
                    from io import BytesIO
                    with zipfile.ZipFile(p) as z:
                        names=[n for n in z.namelist() if n.lower().endswith('.nc')]
                        if len(names)!=1:raise ValueError('伪nc容器内不是单一NetCDF')
                        payload=BytesIO(z.read(names[0]));row['实际容器']='ZIP内NetCDF'
                with xr.open_dataset(payload,engine='h5netcdf') as ds:
                    source_dims=dict(ds.sizes)
                    # NASA全球栅格核查全部文件的研究区切片，避免将全球海陆掩膜作为本站缺失率。
                    for coord,lo,hi in [('lat',21.9,22.8),('latitude',21.9,22.8),('lon',113.4,115.1),('longitude',113.4,115.1)]:
                        if coord in ds.dims and ds.sizes[coord]>100:
                            values=ds[coord].values
                            ds=ds.sel({coord:slice(lo,hi) if values[0]<values[-1] else slice(hi,lo)})
                    for v in ds.data_vars:
                        a=ds[v];vals=np.asarray(a.values)
                        numeric=np.issubdtype(vals.dtype,np.number)
                        binary.append({'路径':str(p),'变量':v,'原始维度':str(source_dims),'维度':str(dict(a.sizes)),'核验范围':'全部文件；全球栅格限香港研究区数值切片','单位':a.attrs.get('units','NA'),'元素数':int(vals.size),'缺失数':int((~np.isfinite(vals)).sum()) if numeric else None,'坐标':str({c:str(ds[c].values)[:150] for c in ['latitude','longitude'] if c in ds.coords}),'时间开始':ds.attrs.get('time_coverage_start',''),'时间结束':ds.attrs.get('time_coverage_end',''),'时间粒度':ds.attrs.get('temporal_range',ds.attrs.get('period',''))})
                    row.update(核验层次='全部变量数值缺失/单位/维度',变量数=len(ds.data_vars))
            elif p.suffix.lower()=='.zip':
                with zipfile.ZipFile(p) as z:
                    row.update(核验层次='ZIP目录完整性与CRC检查',ZIP成员数=len(z.infolist()),CRC损坏成员=z.testzip() or '')
            elif p.suffix.lower() in ['.xls','.xlsx']:
                with pd.ExcelFile(p) as book:
                    row['核验层次']='工作表行列与完整读取'
                    row['工作表']=str({s:pd.read_excel(book,sheet_name=s).shape for s in book.sheet_names})
        except Exception as ex:row.update(状态='未完全读取，需专项核查',异常=f'{type(ex).__name__}: {str(ex)[:200]}')
        inventory.append(row)
        if i%40==0:print(f'已读取 {i+1}/{len(files)} 文件',flush=True)
    save('01_全数据文件清单.csv',inventory);save('02_全字段质量画像.csv',fields);save('03_数据覆盖分站分年.csv',segments);save('04_NetCDF变量与真实覆盖.csv',binary)
    (E/'全数据原件SHA256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    (E/'全量盘点摘要.json').write_text(json.dumps({'目录':[str(p) for p in roots],'文件数':len(inventory),'完整字段画像数':len(fields),'读取异常':sum(r['状态']!='已登记' for r in inventory),'数据内容或结果写入原件':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print('盘点完成',len(inventory),'文件',len(fields),'字段',flush=True)

if __name__=='__main__':main()
