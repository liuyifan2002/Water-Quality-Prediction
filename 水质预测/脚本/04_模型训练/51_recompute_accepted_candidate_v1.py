"""按冻结新候选、原manifest及原参数完成100个正式模型任务折；独立输出。"""
from pathlib import Path
import argparse,importlib.util,json,hashlib,time,gc,sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import joblib
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';FROZEN=RUN/'02_冻结候选';OUT=RUN/'03_正式十任务重算'
CONFIG=ROOT/'结果/扩展实验/候选模型稳健性验证_20260914/portable_package/config.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def import_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def feature_arrays(data,train,test,features,daily):
    # 气候态统计完全由本折训练期决定，外推折不会用到后期日值。
    end=train.date.max();ref=daily[daily.date.between(pd.Timestamp('2014-01-01'),end)]
    clim=ref.groupby(ref.date.dt.dayofyear).sst_mean_c.mean()
    values=pd.Series((daily.sst_mean_c-daily.date.dt.dayofyear.map(clim)).values,index=daily.date)
    tr=train[features].copy();te=test[features].copy()
    for n in [3,7,14]:
        c=f'sst_anom_{n}d';a=values.shift(1).rolling(n,min_periods=n).mean()
        if c in features:tr[c]=train.date.map(a).values;te[c]=test.date.map(a).values
    c='sst_anom_7d_x_ssrd_mean_7d'
    if c in features:tr[c]=tr.sst_anom_7d*tr.ssrd_mean_7d;te[c]=te.sst_anom_7d*te.ssrd_mean_7d
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    xt=imp.fit_transform(tr).astype('float32');xe=imp.transform(te).astype('float32')
    return xt,xe,imp,clim,end,tr,te

def tasks():
    rows=[]
    for fam,target in [('chla','chla_ug_l'),('tn','tn_mg_l'),('tp','tp_mg_l')]:
        for i,depth in enumerate(['surface','middle','bottom'],3):rows.append((f'{fam}_{depth}',target,ROOT/f'runs/{fam}_family_v2/{i:02}_{fam}_{depth}_manifest.csv'))
    rows.append(('secchi','secchi_depth_m',ROOT/'runs/modern_baseline_gate_v1/01_secchi_manifest.csv'))
    return rows

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for s in ['inputs','predictions','fold_metrics','models']:(OUT/s).mkdir(exist_ok=True)
    frozen=json.loads((FROZEN/'冻结清单.json').read_text(encoding='utf-8'))
    for p,h in frozen['files'].items():assert sha(RUN/p)==h,p
    data=pd.read_csv(FROZEN/'基础输入_77特征.csv',parse_dates=['date'],low_memory=False);daily=pd.read_csv(FROZEN/'ERA5逐日气候态来源.csv',parse_dates=['date']).sort_values('date')
    fs=frozen['四补充任务'] # 仅验证元数据已存在，不以补充表增加正式特征。
    policy=json.loads((FROZEN/'特征与时间协议.json').read_text(encoding='utf-8'));features=policy['feature_order'];assert len(features)==77
    rows=[]
    for task,target,mp in tasks():
        man=pd.read_csv(mp);manifesthash=sha(mp)
        for fold in [f'fold_{i}' for i in range(1,6)]:
            s=man[man.fold.eq(fold)];trids=s.loc[s.role.eq('train'),'obs_id'];teids=s.loc[s.role.eq('test'),'obs_id']
            ix=data.set_index('obs_id',drop=False)
            if task=='secchi':train=ix.loc[trids].reset_index(drop=True);test=ix.loc[teids].reset_index(drop=True)
            else:train=data[data.obs_id.isin(set(trids))].copy();test=data[data.obs_id.isin(set(teids))].copy()
            assert len(train)==len(trids) and len(test)==len(teids) and not set(train.station)&set(test.station)
            assert train[target].notna().all() and test[target].notna().all()
            if 'target_value' in s:
                for z in [train,test]:
                    ref=s.set_index('obs_id').loc[z.obs_id,'target_value'].to_numpy(float);assert np.allclose(z[target],ref)
            stem=f'{task}_{fold}';cache=OUT/'inputs'/f'{stem}.npz'
            if not cache.exists():
                xt,xe,imp,clim,end,rt,re=feature_arrays(data,train,test,features,daily)
                assert xt.shape[1]==xe.shape[1]==77 and np.isfinite(xt).all() and np.isfinite(xe).all()
                np.savez_compressed(cache,x_train=xt,x_test=xe,y_train=train[target].to_numpy(float),y_true=test[target].to_numpy(float),train_ids=train.obs_id.to_numpy(str),test_ids=test.obs_id.to_numpy(str),stations=test.station.to_numpy(str),dates=test.date.dt.strftime('%Y-%m-%d').to_numpy(str))
                joblib.dump(imp,OUT/'inputs'/f'{stem}_imputer.joblib');clim.rename('training_period_sst_clim').to_csv(OUT/'inputs'/f'{stem}_气候态.csv',encoding='utf-8-sig')
                rt.assign(obs_id=train.obs_id.values).to_csv(OUT/'inputs'/f'{stem}_训练未填补有效特征.csv',index=False,encoding='utf-8-sig');re.assign(obs_id=test.obs_id.values).to_csv(OUT/'inputs'/f'{stem}_测试未填补有效特征.csv',index=False,encoding='utf-8-sig')
            for model in ['xgboost','tabiclv2']:rows.append({'task':task,'target':target,'model':model,'fold':fold,'status':'PENDING','dataset_version':frozen['dataset_version'],'base_input_sha256':frozen['input_sha256'],'effective_input_sha256':sha(cache),'manifest_path':str(mp),'manifest_sha256':manifesthash,'model_config_sha256':sha(CONFIG),'n_train':len(train),'n_test':len(test),'climatology_training_end':str(train.date.max().date()),'prediction_file':str(OUT/'predictions'/f'{task}_{model}_{fold}.csv'),'metrics_file':str(OUT/'fold_metrics'/f'{task}_{model}_{fold}.csv'),'error':''})
    reg=pd.DataFrame(rows);path=OUT/'运行登记.csv'
    if path.exists():
        old=pd.read_csv(path,keep_default_na=False);assert len(old)==100
        for k in ['task','target','model','fold','dataset_version','base_input_sha256','effective_input_sha256','manifest_sha256','model_config_sha256']:assert reg[k].astype(str).equals(old[k].astype(str)),k
        reg=old
    else:reg.to_csv(path,index=False,encoding='utf-8-sig')
    (OUT/'模型固定配置.json').write_text(json.dumps({'dataset_version':frozen['dataset_version'],'seed':42,'model_parameters':json.loads(CONFIG.read_text(encoding='utf-8-sig'))['model_parameters']['xgboost'],'target_transform':'训练目标非负log1p，预测expm1后截非负，沿用旧协议','TabICLv2':'models/tabiclv2_adapter.py；device=cuda,use_amp=auto,offload_mode=auto,random_state=42；tabicl=2.2.0，torch=2.6.0+cu124','特征':'每折训练期气候态+仅训练中位数；两模型复用同一npz哈希','测试输入批':'完整原测试集单批查询，保持旧正式预测协议，不称P1单查询'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('PREFLIGHT_PASS',len(reg),'任务；50组按折有效输入，特征77',flush=True)
    return reg

def run(reg,which):
    path=OUT/'运行登记.csv';config=json.loads(CONFIG.read_text(encoding='utf-8-sig'));params=dict(config['model_parameters']['xgboost']);params.update(random_state=42,objective='reg:squarederror')
    for i,row in reg.iterrows():
        if which!='all' and row.model!=which:continue
        cache=OUT/'inputs'/f'{row.task}_{row.fold}.npz';assert sha(cache)==row.effective_input_sha256 and sha(row.manifest_path)==row.manifest_sha256
        predpath=Path(row.prediction_file);metricpath=Path(row.metrics_file)
        if row.status=='OK' and predpath.exists() and metricpath.exists():continue
        started=time.perf_counter();reg.loc[i,'status']='RUNNING';reg.to_csv(path,index=False,encoding='utf-8-sig');print('START',row.task,row.model,row.fold,flush=True)
        try:
            a=np.load(cache);xtr=a['x_train'];xte=a['x_test'];y=a['y_train'];yt=a['y_true'];nonnegative=y.min()>=0;yf=np.log1p(y) if nonnegative else y.copy()
            if row.model=='xgboost':
                from xgboost import XGBRegressor
                model=XGBRegressor(**params);model.fit(xtr,yf);p=np.asarray(model.predict(xte),float);model.save_model(OUT/'models'/f'{row.task}_{row.model}_{row.fold}.ubj')
            else:
                import torch,importlib.metadata
                assert torch.cuda.is_available() and importlib.metadata.version('tabicl')=='2.2.0'
                adapter=import_path('repaired_tabicl_adapter',ROOT/'models/tabiclv2_adapter.py');model=adapter.TabICLv2Regressor(random_state=42);model.fit(xtr,yf);p=np.asarray(model.predict(xte),float).reshape(-1)
            p=np.maximum(np.expm1(p) if nonnegative else p,0);assert len(p)==len(yt) and np.isfinite(p).all()
            scores={'R2':float(r2_score(yt,p)),'RMSE':float(mean_squared_error(yt,p)**.5),'MAE':float(mean_absolute_error(yt,p))}
            out=pd.DataFrame({'task':row.task,'obs_id':a['test_ids'],'station_id':a['stations'],'date':a['dates'],'fold':row.fold,'model':row.model,'y_true':yt,'y_pred':p,'residual':p-yt,'dataset_version':row.dataset_version,'base_input_sha256':row.base_input_sha256,'effective_input_sha256':row.effective_input_sha256,'manifest_sha256':row.manifest_sha256})
            out.to_csv(predpath,index=False,encoding='utf-8-sig');metric={**scores,'task':row.task,'model':row.model,'fold':row.fold,'n_train':len(y),'n_test':len(yt),'runtime':time.perf_counter()-started,'dataset_version':row.dataset_version,'effective_input_sha256':row.effective_input_sha256,'status':'OK'}
            pd.DataFrame([metric]).to_csv(metricpath,index=False,encoding='utf-8-sig')
            for k,v in metric.items():reg.loc[i,k]=v
            reg.loc[i,'prediction_sha256']=sha(predpath);reg.loc[i,'error']='';print('OK',row.task,row.model,row.fold,'R2',round(scores['R2'],5),'秒',round(metric['runtime'],1),flush=True)
            del model,a;gc.collect()
            try:
                import torch;torch.cuda.empty_cache()
            except ImportError:pass
        except Exception as ex:
            reg.loc[i,'status']='FAILED';reg.loc[i,'error']=f'{type(ex).__name__}: {ex}';print('FAILED',row.task,row.model,row.fold,reg.loc[i,'error'],flush=True)
        reg.to_csv(path,index=False,encoding='utf-8-sig')
    print(reg.groupby(['model','status']).size().to_string(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--model',default='all',choices=['all','xgboost','tabiclv2']);args=parser.parse_args();reg=prepare()
    if not args.prepare_only:run(reg,args.model)
