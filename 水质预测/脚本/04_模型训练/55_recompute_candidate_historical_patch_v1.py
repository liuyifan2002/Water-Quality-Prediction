"""按原train/test ID回放历史高分随机门控实验；保留目标历史，严禁作为正式77留站点结果。"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,time,joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('candidate_patch_core',Path(__file__).with_name('51_recompute_accepted_candidate_v1.py'));core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
OLD=ROOT/'结果/扩展实验/质量感知遥感门控融合_20260905';OUT=core.RUN/'11_历史高分随机门控重算'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['inputs','models','predictions','metrics','oof']:(OUT/name).mkdir(exist_ok=True)
    mod=core.import_path('candidate_patch_helpers',ROOT/'脚本/04_模型训练/26_run_quality_aware_remote_ab.py')
    args=SimpleNamespace(oof_folds=4,random_state=42,n_estimators=400,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,n_jobs=4)
    formal=mod.load_formal_module();data=formal.prepare_dataset(ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv')
    fixed=pd.read_csv(core.FROZEN/'基础输入_77特征.csv',parse_dates=['date']);fs=json.loads((core.FROZEN/'特征与时间协议.json').read_text(encoding='utf-8'))['feature_order'];assert data.obs_id.equals(fixed.obs_id)
    for c in fs:data[c]=fixed[c].values
    data,_,_=mod.merge_patch_features(data,mod.DEFAULT_LANDSAT_PATCH,mod.DEFAULT_S2_PATCH)
    oldtest=pd.read_csv(OLD/'质量感知遥感融合_预测.csv');oldtrain=pd.read_csv(OLD/'门控训练期OOF记录.csv');feature_manifest=pd.read_csv(OLD/'特征与防泄漏清单.csv');daily=pd.read_csv(core.FROZEN/'ERA5逐日气候态来源.csv',parse_dates=['date']).sort_values('date');frozen=json.loads((core.FROZEN/'冻结清单.json').read_text(encoding='utf-8'));version=frozen['dataset_version']+'_HIST_RANDOM_PATCH'
    jobs=feature_manifest[['target','depth']].copy();jobs['task']=jobs.target+'_'+jobs.depth;jobs['model']='XGBoost_experts+quality_gate';jobs['fold']='original_random_80_20';jobs['status']='PENDING';jobs['dataset_version']=version;jobs['error']='';rp=OUT/'运行登记.csv'
    if rp.exists():
        reg=pd.read_csv(rp,keep_default_na=False)
        for c in ['task','fold','dataset_version']:assert reg[c].equals(jobs[c])
    else:reg=jobs;reg.to_csv(rp,index=False,encoding='utf-8-sig')
    (OUT/'独立历史协议.json').write_text(json.dumps({'parent_input_sha256':frozen['input_sha256'],'dataset_version':version,'outer_split':'直接采用原测试预测及原训练OOF obs_id和原顺序，不重新调用外层train_test_split','inner_split':'原KFold(4,shuffle=True,random_state=42)，按原训练ID顺序确定并在新目录另存','原件哈希':{str(p):core.sha(p) for p in [OLD/'特征与防泄漏清单.csv',OLD/'质量感知遥感融合_预测.csv',OLD/'门控训练期OOF记录.csv',mod.DEFAULT_LANDSAT_PATCH,mod.DEFAULT_S2_PATCH]},'参数':vars(args),'限制':'保留原目标历史变量及双侧影像匹配；只回放历史随机实验，不作为严格留站点/提前预测或独立检验。新月Landsat与本patch不是同一协议，未混入。'},ensure_ascii=False,indent=2),encoding='utf-8')
    allmetrics=[]
    for i,r in reg.iterrows():
        pp=OUT/'predictions'/f'{r.task}.csv';mp=OUT/'metrics'/f'{r.task}.csv'
        if r.status=='OK' and pp.exists():assert core.sha(pp)==r.prediction_sha256;allmetrics.append(pd.read_csv(mp));continue
        start=time.perf_counter();reg.loc[i,'status']='RUNNING';reg.to_csv(rp,index=False,encoding='utf-8-sig')
        try:
            fm=feature_manifest[feature_manifest.target.eq(r.target)&feature_manifest.depth.eq(r.depth)].iloc[0];f1=json.loads(fm.f1_history_features_json);remote=json.loads(fm.remote_expert_features_json);quality=json.loads(fm.gate_features_json);target=formal.TARGET_MAP[r.target]
            otrain=oldtrain[oldtrain.target.eq(r.target)&oldtrain.depth.eq(r.depth)];otest=oldtest[oldtest.target.eq(r.target)&oldtest.depth.eq(r.depth)];ix=data.set_index('obs_id',drop=False);tr=ix.loc[otrain.obs_id].reset_index(drop=True);te=ix.loc[otest.obs_id].reset_index(drop=True)
            assert not set(tr.obs_id)&set(te.obs_id) and np.allclose(tr[target],otrain.observed) and np.allclose(te[target],otest.observed)
            assert set(tr.obs_id)|set(te.obs_id)==set(data.loc[data.depth.eq(r.depth)&data[target].notna(),'obs_id'])
            manifest=pd.concat([pd.DataFrame({'obs_id':tr.obs_id,'role':'train','order':range(len(tr))}),pd.DataFrame({'obs_id':te.obs_id,'role':'test','order':range(len(te))})]);mfp=OUT/'inputs'/f'{r.task}_outer_manifest.csv';manifest.to_csv(mfp,index=False,encoding='utf-8-sig')
            from sklearn.model_selection import KFold
            inner=[]
            for k,(a,b) in enumerate(KFold(n_splits=4,shuffle=True,random_state=42).split(tr),1):
                validation_positions=set(b)
                inner.extend({'obs_id':tr.obs_id.iloc[j],'inner_fold':k,'role':'validation' if j in validation_positions else 'train'} for j in range(len(tr)))
            pd.DataFrame(inner).to_csv(OUT/'inputs'/f'{r.task}_inner_manifest.csv',index=False,encoding='utf-8-sig')
            counter=[0]
            def fit_save(train,test,col,features,options):
                counter[0]+=1;stem=f'{r.task}_fit_{counter[0]:02}';train=train.copy();test=test.copy()
                _,_,_,_,_,rt,re=core.feature_arrays(data,train,test,fs,daily)
                for c in ['sst_anom_3d','sst_anom_7d','sst_anom_14d','sst_anom_7d_x_ssrd_mean_7d']:train[c]=rt[c];test[c]=re[c]
                train[['obs_id',col]+features].to_csv(OUT/'inputs'/f'{stem}_train.csv',index=False,encoding='utf-8-sig');test[['obs_id',col]+features].to_csv(OUT/'inputs'/f'{stem}_test.csv',index=False,encoding='utf-8-sig')
                y,inverse=mod.transform_target(train[col]);m=mod.xgb_model(options);m.fit(train[features],y);pred=np.maximum(inverse(m.predict(test[features])),0);m.save_model(OUT/'models'/f'{stem}.ubj');return pred
            mod.fit_predict=fit_save
            a,b=mod.oof_predictions(tr,target,f1,remote,args);gate,wins=mod.fit_gate_from_oof(tr,target,a,b,quality);p1=mod.fit_predict(tr,te,target,f1,args);p2=mod.fit_predict(tr,te,target,remote,args);weight=mod.gate_weights(gate,te,quality);p3=(1-weight)*p1+weight*p2
            out=te[['obs_id','sample_id','station','date','depth']].copy();out['target']=r.target;out['observed']=te[target];out['prediction_f1_history']=p1;out['prediction_remote_expert']=p2;out['prediction_quality_gate_fusion']=p3;out['gate_remote_weight']=weight;out['quality_bin']=mod.quality_bin(te);out['dataset_version']=version;out['outer_manifest_sha256']=core.sha(mfp);out.to_csv(pp,index=False,encoding='utf-8-sig')
            met=mod.make_metric_rows(out,r.target,r.depth,{'f1_history':p1,'remote_expert':p2,'quality_gate_fusion':p3},{'f1_history':len(f1),'remote_expert':len(remote),'quality_gate_fusion':len(remote)},float(weight.mean()),float(wins.mean()));pd.DataFrame(met).assign(dataset_version=version).to_csv(mp,index=False,encoding='utf-8-sig');allmetrics.append(pd.read_csv(mp))
            oof=tr[['obs_id','station','date','depth']].copy();oof['target']=r.target;oof['observed']=tr[target];oof['prediction_f1_history']=a;oof['prediction_remote_expert']=b;oof['dataset_version']=version;oof.to_csv(OUT/'oof'/f'{r.task}.csv',index=False,encoding='utf-8-sig');joblib.dump(gate,OUT/'models'/f'{r.task}_gate.joblib')
            reg.loc[i,'status']='OK';reg.loc[i,'prediction_file']=str(pp);reg.loc[i,'prediction_sha256']=core.sha(pp);reg.loc[i,'trained_xgb_models']=counter[0];reg.loc[i,'error']='';print('PATCH_OK',i+1,'/12',r.task,flush=True)
        except Exception as ex:reg.loc[i,'status']='FAILED';reg.loc[i,'error']=f'{type(ex).__name__}: {ex}';print('PATCH_FAILED',r.task,reg.loc[i,'error'],flush=True)
        reg.loc[i,'runtime']=time.perf_counter()-start;reg.to_csv(rp,index=False,encoding='utf-8-sig')
    if allmetrics:pd.concat(allmetrics,ignore_index=True).to_csv(OUT/'历史高分随机门控_指标.csv',index=False,encoding='utf-8-sig')
    print('PATCH_FINISHED',reg.groupby('status').size().to_dict(),flush=True)
if __name__=='__main__':main()
