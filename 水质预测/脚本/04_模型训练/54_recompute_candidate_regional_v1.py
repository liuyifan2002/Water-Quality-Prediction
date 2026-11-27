"""历史78项分区实验用新候选重算；只复用原站点折与真实目标，不加入正式77项结果。"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('candidate_core_regional',Path(__file__).with_name('51_recompute_accepted_candidate_v1.py'))
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
OLD=ROOT/'transfer/llm_extension_transfer_v1/results/regionalization/分海区五折留站验证_20260906'
OUT=core.RUN/'06_历史78特征分区重算'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in ['inputs','predictions','fold_metrics','decisions','models']:(OUT/p).mkdir(exist_ok=True)
    mod=core.import_path('candidate_regional_helpers',ROOT/'脚本/04_模型训练/30_cross_validate_zone_model_spatial.py')
    args=SimpleNamespace(min_zone_train_rows=300,min_zone_stations=3,n_estimators=400,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=42,n_jobs=4)
    manifest=pd.read_csv(OLD/'五折站点划分清单.csv');assert manifest.station.is_unique
    oldp=pd.read_csv(OLD/'五折留站点验证_预测.csv');ff=pd.read_csv(OLD/'严格验证特征清单.csv')
    fs77=json.loads((core.FROZEN/'特征与时间协议.json').read_text(encoding='utf-8'))['feature_order']
    main=pd.read_csv(ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv',parse_dates=['date'],low_memory=False)
    assert core.sha(ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv')=='3f562fd8709641ddeaa55ffecba86c741eeaeafff7948c3b7a7d1660d2d89ea5'
    data=pd.read_csv(core.FROZEN/'基础输入_77特征.csv',parse_dates=['date'],low_memory=False)
    meta=['obs_id','sample_id','station_name_en','water_zone_en','year','month']
    data=data.merge(main[['obs_id']+[c for c in meta if c!='obs_id' and c not in data]],on='obs_id',how='left',validate='one_to_one')
    # 与历史prepare_dataset完全相同的类别编码，保持10个已知水区集合与排序。
    data['water_zone_code']=pd.Categorical(data.water_zone_en.fillna('unknown')).codes.astype('int16')
    for t,c in [('chla','chla_ug_l'),('secchi','secchi_depth_m'),('tn','tn_mg_l'),('tp','tp_mg_l')]:
        q=oldp[oldp.target.eq(t)];ix=data.set_index('obs_id');assert np.allclose(q.observed,ix.loc[q.obs_id,c])
        fs=json.loads(ff.loc[ff.target.eq(t),'features_json'].iloc[0]);assert len(fs)==78 and set(fs)==set(fs77)|{'water_zone_code'}
    daily=pd.read_csv(core.FROZEN/'ERA5逐日气候态来源.csv',parse_dates=['date']).sort_values('date')
    mfhash=core.sha(OLD/'五折站点划分清单.csv');frozen=json.loads((core.FROZEN/'冻结清单.json').read_text(encoding='utf-8'));rows=[]
    for t,c in [('chla','chla_ug_l'),('secchi','secchi_depth_m'),('tn','tn_mg_l'),('tp','tp_mg_l')]:
        for d in ['surface','middle','bottom']:
            for fold in range(1,6):rows.append({'task':f'{t}_{d}','target':t,'target_col':c,'depth':d,'fold':fold,'status':'PENDING','dataset_version':frozen['dataset_version']+'_HIST78','parent_input_sha256':frozen['input_sha256'],'station_manifest_sha256':mfhash,'feature_manifest_sha256':core.sha(OLD/'严格验证特征清单.csv'),'error':''})
    reg=pd.DataFrame(rows);rp=OUT/'运行登记.csv'
    if rp.exists():
        old=pd.read_csv(rp,keep_default_na=False)
        for c in ['task','fold','dataset_version','parent_input_sha256','station_manifest_sha256','feature_manifest_sha256']:assert old[c].astype(str).equals(reg[c].astype(str))
        reg=old
    else:reg.to_csv(rp,index=False,encoding='utf-8-sig')
    (OUT/'独立历史扩展协议.json').write_text(json.dumps({'feature_count':78,'包含water_zone_code':True,'非正式十任务':True,'Secchi三层':'沿用真实历史表目标，逐obs_id已和冻结预测真值核验，不新增目标','station_manifest_sha256':mfhash,'source_features_sha256':core.sha(OLD/'严格验证特征清单.csv'),'模型参数':vars(args),'变更':'采用新候选输入；每折仅训练期气候态，原回退门限及真实NA不变','地域处理':'保留历史全折训练集类别编码；没有重划站点折'},ensure_ascii=False,indent=2),encoding='utf-8')
    allmetrics=[];allpred=[];alldecisions=[]
    for i,r in reg.iterrows():
        stem=f'{r.task}_fold_{r.fold}';pp=OUT/'predictions'/f'{stem}.csv';mp=OUT/'fold_metrics'/f'{stem}.csv';dp=OUT/'decisions'/f'{stem}.csv'
        if r.status=='OK' and pp.exists():
            assert core.sha(pp)==r.prediction_sha256
            allpred.append(pd.read_csv(pp));allmetrics.append(pd.read_csv(mp));alldecisions.append(pd.read_csv(dp));continue
        start=time.perf_counter();reg.loc[i,'status']='RUNNING';reg.to_csv(rp,index=False,encoding='utf-8-sig')
        try:
            fs=json.loads(ff.loc[ff.target.eq(r.target),'features_json'].iloc[0]);s=data[data.depth.eq(r.depth)&data[r.target_col].notna()].copy();s['fold']=s.station.map(manifest.set_index('station').fold);assert s.fold.notna().all();s=s.sort_values(['fold','date','station']).reset_index(drop=True)
            tr=s[s.fold.ne(r.fold)].drop(columns='fold').reset_index(drop=True);te=s[s.fold.eq(r.fold)].drop(columns='fold').reset_index(drop=True);assert not set(tr.station)&set(te.station)
            q=oldp[oldp.target.eq(r.target)&oldp.depth.eq(r.depth)&oldp.fold.eq(r.fold)];assert set(q.obs_id)==set(te.obs_id) and np.allclose(q.observed,te.set_index('obs_id').loc[q.obs_id,r.target_col])
            _,_,_,clim,_,rawtr,rawte=core.feature_arrays(data,tr,te,fs77,daily)
            for c in ['sst_anom_3d','sst_anom_7d','sst_anom_14d','sst_anom_7d_x_ssrd_mean_7d']:tr[c]=rawtr[c];te[c]=rawte[c]
            trp=OUT/'inputs'/f'{stem}_train.csv';tep=OUT/'inputs'/f'{stem}_test.csv';tr[['obs_id',r.target_col]+fs].to_csv(trp,index=False,encoding='utf-8-sig');te[['obs_id',r.target_col]+fs].to_csv(tep,index=False,encoding='utf-8-sig');clim.to_csv(OUT/'inputs'/f'{stem}_气候态.csv',encoding='utf-8-sig')
            counter=[0]
            def fit_save(train,test,target,features,options):
                y=train[target].to_numpy(float);nonnegative=np.nanmin(y)>=0;m=mod.make_model(options);m.fit(train[features],np.log1p(y) if nonnegative else y);p=m.predict(test[features]);p=np.maximum(np.expm1(p) if nonnegative else p,0);counter[0]+=1;m.save_model(OUT/'models'/f'{stem}_model_{counter[0]:02}.ubj');return p
            mod.fit_predict=fit_save
            mm,p,dec=mod.run_one_task(tr,te,r.target,r.target_col,r.depth,int(r.fold),fs,args)
            p['dataset_version']=r.dataset_version;p['train_input_sha256']=core.sha(trp);p['test_input_sha256']=core.sha(tep);p['station_manifest_sha256']=mfhash
            assert np.isfinite(p[['prediction_global_external','prediction_zone_aware_external']]).all().all()
            p.to_csv(pp,index=False,encoding='utf-8-sig');pd.DataFrame(mm).to_csv(mp,index=False,encoding='utf-8-sig');pd.DataFrame(dec).to_csv(dp,index=False,encoding='utf-8-sig');reg.loc[i,'status']='OK';reg.loc[i,'prediction_sha256']=core.sha(pp);reg.loc[i,'trained_models']=counter[0];reg.loc[i,'error']='';allpred.append(p);allmetrics.append(pd.DataFrame(mm));alldecisions.append(pd.DataFrame(dec));print('REGIONAL_OK',i+1,'/60',r.task,r.fold,flush=True)
        except Exception as ex:reg.loc[i,'status']='FAILED';reg.loc[i,'error']=f'{type(ex).__name__}: {ex}';print('REGIONAL_FAILED',r.task,r.fold,reg.loc[i,'error'],flush=True)
        reg.loc[i,'runtime']=time.perf_counter()-start;reg.to_csv(rp,index=False,encoding='utf-8-sig')
    if allpred:
        pred=pd.concat(allpred,ignore_index=True);metrics=pd.concat(allmetrics,ignore_index=True);dec=pd.concat(alldecisions,ignore_index=True);pred.to_csv(OUT/'五折留站点验证_预测.csv',index=False,encoding='utf-8-sig');metrics.to_csv(OUT/'五折留站点验证_各折指标.csv',index=False,encoding='utf-8-sig');dec.to_csv(OUT/'分区训练决策.csv',index=False,encoding='utf-8-sig');summary,macro=mod.build_summary(metrics,pred);summary.to_csv(OUT/'五折留站点验证_汇总.csv',index=False,encoding='utf-8-sig');macro.to_csv(OUT/'五折总体比较.csv',index=False,encoding='utf-8-sig')
    print('REGIONAL_FINISHED',reg.groupby('status').size().to_dict(),flush=True)

if __name__=='__main__':main()
