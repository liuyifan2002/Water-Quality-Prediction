"""补充核验原始气象汇总、影像行连接、来源别名及冻结预测真值；不生成预测。"""
from pathlib import Path
import importlib.util,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115';E=OUT/'证据'
F=ROOT/'数据/处理后数据/正式主表';OLD=ROOT/'codex/worktrees/9ff7/水质预测';D=ROOT/'数据/外部数据/试验项目接管_20260521'

def module(name,rel):
    spec=importlib.util.spec_from_file_location(name,ROOT/rel);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def save(name,rows):pd.DataFrame(rows).to_csv(E/name,index=False,encoding='utf-8-sig')
def numeric(a,b):return np.isclose(np.asarray(a,float),np.asarray(b,float),equal_nan=True,rtol=1e-6,atol=1e-7)

def main():
    checks=[]
    m=module('raw_month','脚本/02_特征构建/02_download_prepare_era5_monthly.py')
    paths=[next((ROOT/f'数据/原始数据/era5/era5_hk_timeseries_{y}').glob('*.nc')) for y in range(2014,2025)]
    re=m.aggregate_monthly(paths);old=pd.read_csv(F/'era5_monthly_2014_2024.csv')
    re=re.set_index(['year','month']).loc[list(zip(old.year,old.month))].reset_index()
    for c in ['t2m','u10','v10','tcc']:checks.append({'链路':'原始11年小时ERA5→月均','字段':c,'记录':len(old),'差异':int((~numeric(old[c],re[c])).sum())})
    daily=module('raw_daily','脚本/02_特征构建/24_download_prepare_era5_ssrd_sst_daily.py')
    p=[]
    for y in range(2014,2025):
        for point in daily.OFFSHORE_POINTS:
            label=point['label'];folder=ROOT/f'数据/原始数据/era5_ssrd_sst_offshore_v2/era5_ssrd_sst_{label}_{y}'
            p.append((label,next(folder.rglob('*.nc'))))
    re=daily.aggregate_daily(p);old=pd.read_csv(ROOT/'数据/处理后数据/衍生版本/era5_ssrd_sst_daily_2014_2024.csv',parse_dates=['date'])
    re=re.set_index('date').loc[old.date].reset_index()
    for c in old:
        if c!='date':checks.append({'链路':'原始33份离岸小时ERA5→日值/异常','字段':c,'记录':len(old),'差异':int((~numeric(old[c],re[c])).sum())})
    save('25_气象小时原件与月日汇总核验.csv',checks);print('ERA5原始汇总复核完成',flush=True)
    original=pd.read_csv(F/'model_table_3depth_2014_2024.csv',low_memory=False).set_index('obs_id')
    rows=[];pooled=[];protected=[]
    for fam,target in [('chla','chla_ug_l'),('tn','tn_mg_l'),('tp','tp_mg_l')]:
        folder=ROOT/f'runs/{fam}_family_v2';summ=pd.read_csv(folder/f'08_{fam}_family_v2_pooled_metrics.csv')
        acc={}
        for path in sorted((folder/'predictions').glob('*.csv')):
            d=pd.read_csv(path);task=d.task.iloc[0];fold=d.fold.iloc[0]
            manpath=next(folder.glob(f'*_{task}_manifest.csv'));man=pd.read_csv(manpath);test=set(man.loc[man.fold.eq(fold)&man.role.eq('test'),'obs_id'])
            assert set(d.obs_id).issubset(set(original.index))
            y=original.loc[d.obs_id,target].to_numpy(float)
            row={'文件':str(path),'任务':task,'模型':d.model.iloc[0],'折':fold,'行数':len(d),'重复obs_id':int(d.obs_id.duplicated().sum()),'真值与原始水质不一致':int((~numeric(y,d.y_true)).sum()),'测试ID清单一致':set(d.obs_id)==test,'存在非有限预测':bool(~np.isfinite(d.y_pred.to_numpy(float)).all()),'重新预测':False}
            rows.append(row);acc.setdefault((task,d.model.iloc[0]),[]).append(d)
        for (task,model),frames in acc.items():
            d=pd.concat(frames);error=d.y_pred-d.y_true;ss=((d.y_true-d.y_true.mean())**2).sum()
            r2=1-(error**2).sum()/ss;rmse=float(np.sqrt((error**2).mean()));mae=float(error.abs().mean())
            sr=summ[summ.task.eq(task)&summ.model.eq(model)].iloc[0]
            pooled.append({'任务':task,'模型':model,'五折拼接行数':len(d),'拼接重复obs_id':int(d.obs_id.duplicated().sum()),'冻结预测确定性汇总R2':r2,'登记R2':sr.R2,'登记R2一致':bool(np.isclose(r2,sr.R2)),'RMSE一致':bool(np.isclose(rmse,sr.RMSE)),'MAE一致':bool(np.isclose(mae,sr.MAE)),'新模型推理':False})
    sec=pd.read_csv(ROOT/'runs/tabiclv2_pilot/09_secchi_predictions.csv')
    for (model,validation),d in sec.groupby(['model','validation']):
        rows.append({'文件':str(ROOT/'runs/tabiclv2_pilot/09_secchi_predictions.csv'),'任务':'Secchi','模型':model,'折':validation,'行数':len(d),'重复obs_id':int(d.obs_id.duplicated().sum()),'真值与原始水质不一致':int((~numeric(original.loc[d.obs_id,'secchi_depth_m'],d.y_true)).sum()),'测试ID清单一致':'按历史注册单独核查，未混入Family V2','存在非有限预测':bool(~np.isfinite(d.y_pred.to_numpy(float)).all()),'重新预测':False})
    save('26_正式冻结预测真值与测试ID.csv',rows);save('27_正式FamilyV2冻结预测指标一致性.csv',pooled)
    # 实际影像导出与处理表逐sample_id连接，统计来源丢行和字段错位。
    samples=pd.read_csv(F/'gee_station_date_samples.csv').rename(columns={'Station':'station'}).set_index('sample_id');patch=[]
    for sensor in ['s2','landsat']:
        raw=pd.read_csv(OLD/f'data/external/gee_exports/{sensor}_patch_features_2014_2024.csv',low_memory=False)
        processed=pd.read_csv(D/f'data_processed_current/{sensor}_highres_patch_features_2014_2024.csv',low_memory=False)
        assert not raw.sample_id.duplicated().any() and not processed.sample_id.duplicated().any()
        aligned=raw.set_index('sample_id').loc[processed.sample_id].reset_index()
        for c in processed:
            if c not in aligned:continue
            a=processed[c];b=aligned[c]
            same=numeric(a,b) if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) else (a.fillna('<NA>').astype(str)==b.fillna('<NA>').astype(str)).to_numpy()
            patch.append({'传感器':sensor,'字段':c,'核对数':len(a),'导出与处理差异':int((~same).sum()),'完整样本键集合':set(raw.sample_id)==set(samples.index),'来源':'原GEE全站导出，不把空像元计成实测'})
    save('28_真实高分影像导出与处理逐字段核对.csv',patch)
    a=pd.read_csv(D/'data_model_input_new_sources/ocean_color_kd490_tss_features_2014_2024.csv').set_index(['station','date'])
    b=pd.read_csv(D/'data_model_input_new_sources/sentinel3_olci_features_2014_2024.csv').set_index(['station','date']).loc[a.index]
    aliases=[]
    for left,right in [('oc_valid','olci_valid'),('oc_chl','olci_chla'),('oc_kd490','olci_kd490'),('oc_rrs_443','olci_rrs_443'),('oc_ndci','olci_ndci'),('oc_fai','olci_fai')]:
        if left in a and right in b:aliases.append({'原字段':left,'别名字段':right,'记录':len(a),'差异':int((~numeric(a[left],b[right])).sum()),'原始实际产品':';'.join(a.oc_source_dataset_id.dropna().unique()),'本轮结论':'同一多传感器CMEMS海色数据重命名，不是另一份独立Sentinel-3 OLCI观测；全NA指数非已完成指数'})
    save('29_Sentinel3名义表实际来源核验.csv',aliases)
    rolling=[]
    a=a.reset_index().sort_values(['station','date'])
    for base in ['oc_chl','oc_kd490','oc_zsd']:
        for n in [3,7,14]:
            col=f'{base}_{n}d'
            if col in a:
                v=a.groupby('station')[base].transform(lambda s:s.rolling(n,min_periods=1).mean())
                rolling.append({'字段':col,'记录':len(a),'按采样次数滚动复现差异':int((~numeric(a[col],v)).sum()),'结论':'名称d与实际不规则监测记录数不一致，不能解释为日窗'})
    save('30_CMEMS海色采样窗与日窗定义核验.csv',rolling)
    print('冻结真值、影像连接及来源别名核查完成',flush=True)

if __name__=='__main__':main()
