"""核验新候选模型任务、预测血缘、指标和09依赖闭环，不生成模型结果。"""
from pathlib import Path
import json,hashlib,importlib.metadata,sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';OUT=RUN/'08_重算结果核验'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scores(d):return {'pooled_R2':r2_score(d.y_true,d.y_pred),'pooled_RMSE':mean_squared_error(d.y_true,d.y_pred)**.5,'pooled_MAE':mean_absolute_error(d.y_true,d.y_pred)}
def main():
    OUT.mkdir(exist_ok=True);frozen=json.loads((RUN/'02_冻结候选/冻结清单.json').read_text(encoding='utf-8'));checks=[];pooled=[];sites=[];inventory=[];states=[];metric_checks=[]
    def check(name,ok,detail):checks.append({'检查项':name,'结果':'PASS' if bool(ok) else 'FAIL','依据':str(detail)})
    # 每个实际折输入再独立证明：原划分、训练中位数及训练期气候态，而非只信模型登记。
    frozen_data=pd.read_csv(RUN/'02_冻结候选/基础输入_77特征.csv',parse_dates=['date']).set_index('obs_id')
    daily=pd.read_csv(RUN/'02_冻结候选/ERA5逐日气候态来源.csv',parse_dates=['date']).sort_values('date')
    fs=json.loads((RUN/'02_冻结候选/特征与时间协议.json').read_text(encoding='utf-8'))['feature_order']
    core_reg=pd.read_csv(RUN/'03_正式十任务重算/运行登记.csv')
    import joblib,warnings
    for _,r in core_reg.drop_duplicates(['task','fold']).iterrows():
        stem=f'{r.task}_{r.fold}';ip=RUN/'03_正式十任务重算/inputs';a=np.load(ip/f'{stem}.npz');rawtr=pd.read_csv(ip/f'{stem}_训练未填补有效特征.csv');rawte=pd.read_csv(ip/f'{stem}_测试未填补有效特征.csv');tr=frozen_data.loc[a['train_ids']];te=frozen_data.loc[a['test_ids']];man=pd.read_csv(r.manifest_path);q=man[man.fold.eq(r.fold)]
        check(stem+'独立站点观测划分',not set(tr.station)&set(te.station) and not set(tr.index)&set(te.index) and set(q.loc[q.role.eq('train'),'obs_id'])==set(tr.index) and set(q.loc[q.role.eq('test'),'obs_id'])==set(te.index),'未重划或混入测试站')
        check(stem+'原训练目标',np.allclose(a['y_train'],tr[r.target]),len(tr))
        unchanged=[c for c in fs if c not in ['sst_anom_3d','sst_anom_7d','sst_anom_14d','sst_anom_7d_x_ssrd_mean_7d']]
        check(stem+'73项有效输入对应冻结值',np.allclose(rawtr[unchanged],tr[unchanged],rtol=1e-7,atol=1e-8,equal_nan=True) and np.allclose(rawte[unchanged],te[unchanged],rtol=1e-7,atol=1e-8,equal_nan=True) and r.base_input_sha256==frozen['input_sha256'],'其余4项另按训练期气候态独立核验')
        imp=joblib.load(ip/f'{stem}_imputer.joblib')
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(rawtr[fs].to_numpy(float),axis=0)
        med=np.nan_to_num(med,nan=0.0);check(stem+'训练中位数与有效输入',np.allclose(imp.statistics_,med,equal_nan=True) and np.allclose(imp.transform(rawtr[fs]).astype('float32'),a['x_train']) and np.allclose(imp.transform(rawte[fs]).astype('float32'),a['x_test']),'未用测试集统计量')
        ref=daily[daily.date.between('2014-01-01',tr.date.max())];clim=ref.groupby(ref.date.dt.dayofyear).sst_mean_c.mean();anom=pd.Series((daily.sst_mean_c-daily.date.dt.dayofyear.map(clim)).values,index=daily.date)
        good=True
        for n in [3,7,14]:
            expected=anom.shift(1).rolling(n,min_periods=n).mean()
            for raw,meta in [(rawtr,tr),(rawte,te)]:good &= np.isclose(raw[f'sst_anom_{n}d'],meta.date.map(expected),rtol=1e-7,atol=1e-8,equal_nan=True).all()
        check(stem+'独立训练期气候态',good,f'2014-01-01至{tr.date.max()}')
    for label,folder,registry,expected in [('正式十任务','03_正式十任务重算','运行登记.csv',100),('来源消融','04_来源消融重算','消融运行登记.csv',120),('现代模型','05_现代模型重算','运行登记.csv',40),('历史分区','06_历史78特征分区重算','运行登记.csv',60),('历史高分','11_历史高分随机门控重算','运行登记.csv',12)]:
        rp=RUN/folder/registry;reg=pd.read_csv(rp,keep_default_na=False);ok=reg[reg.status.eq('OK')];states.append({'范围':label,'计划任务折':expected,'OK任务折':len(ok),'FAILED任务折':int(reg.status.eq('FAILED').sum()),'其他状态任务折':int((~reg.status.isin(['OK','FAILED'])).sum()),'状态':'完成' if len(ok)==expected else '未完成','登记路径':str(rp)})
        check(label+'任务覆盖',len(ok)==expected and len(reg)==expected,f'{len(ok)}/{expected}')
        if label in ['正式十任务','现代模型','来源消融']:
            preds=[]
            for _,r in ok.iterrows():
                pp=Path(r.prediction_file);d=pd.read_csv(pp);stem=f'{r.task}_{r.fold}';ip=RUN/'03_正式十任务重算/inputs'/f'{stem}.npz';a=np.load(ip)
                check(str(pp.relative_to(RUN))+'真值与ID',d.obs_id.astype(str).tolist()==a['test_ids'].tolist() and np.allclose(d.y_true,a['y_true']) and np.isfinite(d.y_pred).all(),len(d))
                check(str(pp.relative_to(RUN))+'数据版本',d.dataset_version.eq(frozen['dataset_version']).all(),d.dataset_version.unique())
                hashcol='effective_full_input_sha256' if label=='来源消融' else 'effective_input_sha256';check(str(pp.relative_to(RUN))+'有效输入哈希',d[hashcol].eq(sha(ip)).all(),sha(ip))
                if 'prediction_sha256' in reg:check(str(pp.relative_to(RUN))+'预测哈希',sha(pp)==r.prediction_sha256,sha(pp))
                for c,fn in [('R2',lambda:r2_score(d.y_true,d.y_pred)),('RMSE',lambda:mean_squared_error(d.y_true,d.y_pred)**.5),('MAE',lambda:mean_absolute_error(d.y_true,d.y_pred))]:
                    value=fn();tol=1e-7 if label=='现代模型' and r.model=='tabm' else 1e-10
                    # 原TabM返回float32；CSV文本再读为float64有约1e-9的十进制舍入，不改预测或登记值。
                    check(str(pp.relative_to(RUN))+'指标_'+c,np.isclose(float(r[c]),value,rtol=tol,atol=tol),f'登记={r[c]}，CSV复核={value}，rtol=atol={tol}')
                    metric_checks.append({'文件':str(pp),'指标':c,'登记原值':r[c],'CSV复核值':value,'复核减登记':value-float(r[c]),'rtol':tol,'atol':tol,'原因':'TabM原float32预测十进制序列化' if tol==1e-7 else 'float64数值复核'})
                if label=='正式十任务' and r.model=='xgboost':
                    from xgboost import XGBRegressor
                    modelpath=RUN/folder/'models'/f'{r.task}_{r.model}_{r.fold}.ubj';restored=XGBRegressor();restored.load_model(modelpath);p=np.asarray(restored.predict(a['x_test']),float);p=np.maximum(np.expm1(p) if a['y_train'].min()>=0 else p,0)
                    check(str(pp.relative_to(RUN))+'保存模型恢复复核',np.allclose(p,d.y_pred,rtol=1e-10,atol=1e-10),sha(modelpath));del restored
                if label=='来源消融':d['model']='XGBoost_delete_'+r.source_group_removed
                preds.append(d)
                inventory.append({'范围':label,'任务':r.task,'模型':d.model.iloc[0],'折':r.fold,'数据版本':d.dataset_version.iloc[0],'数据输入SHA256':sha(ip),'预测文件':str(pp),'预测SHA256':sha(pp),'核验行数':len(d)})
            if preds:
                allp=pd.concat(preds,ignore_index=True)
                for (task,model),g in allp.groupby(['task','model']):
                    check(label+task+model+'OOF不重号',g.obs_id.is_unique,len(g))
                    fm=[r2_score(q.y_true,q.y_pred) for _,q in g.groupby('fold')];pooled.append({'范围':label,'task':task,'model':model,'n_folds':g.fold.nunique(),'n_test':len(g),'n_stations':g.station_id.nunique(),**scores(g),'fold_R2_mean':np.mean(fm),'fold_R2_std':np.std(fm,ddof=1) if len(fm)>1 else np.nan,'dataset_version':g.dataset_version.iloc[0]})
                    if label=='正式十任务':
                        for station,q in g.groupby('station_id'):sites.append({'task':task,'model':model,'station':station,'n':len(q),'RMSE':mean_squared_error(q.y_true,q.y_pred)**.5,'MAE':mean_absolute_error(q.y_true,q.y_pred),'dataset_version':q.dataset_version.iloc[0]})
        elif label=='历史高分':
            old=ROOT/'结果/扩展实验/质量感知遥感门控融合_20260905';op=pd.read_csv(old/'质量感知遥感融合_预测.csv');ot=pd.read_csv(old/'门控训练期OOF记录.csv')
            for _,r in ok.iterrows():
                pp=Path(r.prediction_file);d=pd.read_csv(pp);q=op[op.target.eq(r.target)&op.depth.eq(r.depth)];t=ot[ot.target.eq(r.target)&ot.depth.eq(r.depth)];mf=RUN/folder/'inputs'/f'{r.task}_outer_manifest.csv';m=pd.read_csv(mf)
                check(str(pp.relative_to(RUN))+'原始训练测试范围',d.obs_id.astype(str).tolist()==q.obs_id.astype(str).tolist() and np.allclose(d.observed,q.observed) and m.loc[m.role.eq('train'),'obs_id'].tolist()==t.obs_id.tolist() and not set(t.obs_id)&set(q.obs_id),len(d))
                check(str(pp.relative_to(RUN))+'预测哈希与独立版本',sha(pp)==r.prediction_sha256 and d.dataset_version.eq(frozen['dataset_version']+'_HIST_RANDOM_PATCH').all() and d.outer_manifest_sha256.eq(sha(mf)).all(),sha(pp))
                mm=pd.read_csv(RUN/folder/'metrics'/f'{r.task}.csv');mm=mm[mm.quality_group.eq('all')]
                for model,column in [('f1_history','prediction_f1_history'),('remote_expert','prediction_remote_expert'),('quality_gate_fusion','prediction_quality_gate_fusion')]:
                    v=mm[mm.model_variant.eq(model)].iloc[0]
                    for metric,value in [('r2',r2_score(d.observed,d[column])),('rmse',mean_squared_error(d.observed,d[column])**.5),('mae',mean_absolute_error(d.observed,d[column]))]:check(str(pp.relative_to(RUN))+model+metric,np.isclose(v[metric],value,rtol=1e-7,atol=1e-7),value)
                check(str(pp.relative_to(RUN))+'真实完整输入与模型',int(r.trained_xgb_models)==10 and all((RUN/folder/'models'/f'{r.task}_fit_{j:02}.ubj').exists() and (RUN/folder/'inputs'/f'{r.task}_fit_{j:02}_train.csv').exists() and (RUN/folder/'inputs'/f'{r.task}_fit_{j:02}_test.csv').exists() for j in range(1,11)),'原4折内OOF两专家+全训练集两专家')
                input_files=sorted((RUN/folder/'inputs').glob(f'{r.task}_fit_??_*.csv'));bundle=[{'file':p.name,'sha256':sha(p)} for p in input_files];bundle_hash=hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest();check(str(pp.relative_to(RUN))+'20份实际专家输入',len(bundle)==20,bundle_hash)
                from xgboost import XGBRegressor
                for j,column in [(9,'prediction_f1_history'),(10,'prediction_remote_expert')]:
                    inp=pd.read_csv(RUN/folder/'inputs'/f'{r.task}_fit_{j:02}_test.csv');m=XGBRegressor();m.load_model(RUN/folder/'models'/f'{r.task}_fit_{j:02}.ubj');pred=np.maximum(np.expm1(m.predict(inp[m.feature_names_in_.tolist()])),0)
                    check(str(pp.relative_to(RUN))+column+'恢复模型与输入',inp.obs_id.tolist()==d.obs_id.tolist() and np.allclose(pred,d[column],rtol=1e-7,atol=1e-7),len(pred));del m
                inventory.append({'范围':label,'任务':r.task,'模型':r.model,'折':r.fold,'数据版本':r.dataset_version,'数据输入SHA256':frozen['input_sha256']+';outer='+sha(mf)+';20input_bundle='+bundle_hash,'预测文件':str(pp),'预测SHA256':sha(pp),'核验行数':len(d)})
        else:
            old=ROOT/'transfer/llm_extension_transfer_v1/results/regionalization/分海区五折留站验证_20260906';op=pd.read_csv(old/'五折留站点验证_预测.csv');mf=pd.read_csv(old/'五折站点划分清单.csv').set_index('station').fold
            decisionfile=RUN/folder/'分区训练决策.csv'
            if decisionfile.exists():
                dec=pd.read_csv(decisionfile);od=pd.read_csv(old/'分区训练决策.csv');joined=dec.merge(od,on=['fold','target','depth','water_zone_en'],validate='one_to_one',suffixes=('_new','_old'))
                check('历史分区原训练回退决策',len(dec)==len(od)==532 and joined.decision_new.eq(joined.decision_old).all() and dec.decision.eq('zone_model').sum()==496 and dec.decision.eq('global_fallback').sum()==36,'496区域+36全域回退，另有60全域对照拟合')
            for _,r in ok.iterrows():
                pp=RUN/folder/'predictions'/f'{r.task}_fold_{r.fold}.csv';d=pd.read_csv(pp);o=op[op.target.eq(r.target)&op.depth.eq(r.depth)&op.fold.eq(int(r.fold))].set_index('obs_id')
                check(str(pp.relative_to(RUN))+'原始目标站点折',set(d.obs_id)==set(o.index) and np.allclose(d.observed,o.loc[d.obs_id,'observed']) and d.station.map(mf).eq(int(r.fold)).all(),len(d))
                check(str(pp.relative_to(RUN))+'预测哈希',sha(pp)==r.prediction_sha256,sha(pp))
                check(str(pp.relative_to(RUN))+'历史独立版本',d.dataset_version.eq(frozen['dataset_version']+'_HIST78').all(),d.dataset_version.unique())
                for role in ['train','test']:
                    ip=RUN/folder/'inputs'/f'{r.task}_fold_{r.fold}_{role}.csv';check(str(pp.relative_to(RUN))+role+'哈希',d[f'{role}_input_sha256'].eq(sha(ip)).all(),sha(ip))
                inventory.append({'范围':label,'任务':r.task,'模型':'global+zone_with_fallback','折':r.fold,'数据版本':d.dataset_version.iloc[0],'数据输入SHA256':d.train_input_sha256.iloc[0]+';'+d.test_input_sha256.iloc[0],'预测文件':str(pp),'预测SHA256':sha(pp),'核验行数':len(d)})
    pooled=pd.DataFrame(pooled)
    pd.DataFrame(states).to_csv(OUT/'重算执行覆盖.csv',index=False,encoding='utf-8-sig');pd.DataFrame(checks).to_csv(OUT/'逐项预测与指标复核.csv',index=False,encoding='utf-8-sig');pooled.to_csv(OUT/'新候选_pooled与逐折指标.csv',index=False,encoding='utf-8-sig');pd.DataFrame(inventory).to_csv(OUT/'模型数据预测版本关联及哈希.csv',index=False,encoding='utf-8-sig')
    oldmetrics=[]
    for family in ['chla','tn','tp']:
        q=pd.read_csv(ROOT/f'runs/{family}_family_v2/08_{family}_family_v2_pooled_metrics.csv');oldmetrics.append(q[['task','model','R2','RMSE','MAE']])
    secchi=pd.read_csv(ROOT/'runs/tabiclv2_pilot/23_pooled_oof_metrics.csv').query("task=='secchi'").iloc[0]
    for m,name in [('xgboost','XGBoost'),('tabiclv2','TabICLv2')]:oldmetrics.append(pd.DataFrame([{'task':'secchi','model':m,'R2':secchi[name+'_pooled_R2'],'RMSE':secchi[name+'_pooled_RMSE'],'MAE':secchi[name+'_pooled_MAE']}]))
    comparison=pooled[pooled['范围'].eq('正式十任务')].merge(pd.concat(oldmetrics,ignore_index=True),on=['task','model'],validate='one_to_one')
    for metric in ['R2','RMSE','MAE']:comparison['new_minus_old_'+metric]=comparison['pooled_'+metric]-comparison[metric]
    comparison.to_csv(OUT/'正式十任务新旧候选指标差异.csv',index=False,encoding='utf-8-sig')
    if sites:
        s=pd.DataFrame(sites);s.to_csv(OUT/'十任务逐站点误差.csv',index=False,encoding='utf-8-sig');wide=s.pivot(index=['task','station'],columns='model',values='RMSE').reset_index();wide['delta_RMSE']=wide.tabiclv2-wide.xgboost;wide['TabICLv2_RMSE降低']=wide.delta_RMSE.lt(0);wide.to_csv(OUT/'十任务逐站点新模型比较.csv',index=False,encoding='utf-8-sig');wins=wide.groupby('task',as_index=False).agg(n_stations=('station','count'),station_RMSE_win_rate=('TabICLv2_RMSE降低','mean'));wins.to_csv(OUT/'十任务站点改善覆盖.csv',index=False,encoding='utf-8-sig')
    protected={ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv':'3f562fd8709641ddeaa55ffecba86c741eeaeafff7948c3b7a7d1660d2d89ea5',ROOT/'结果/论文最终定点排版修订_v1_20261115/水质预测硕士学位论文_最终定点排版修订版.docx':'94ede7bb400ca71854a7f4fccea887973a00d96de045247bb94d5414de8cf668',ROOT/'结果/论文最终定点排版修订_v1_20261115/水质预测硕士学位论文_最终定点排版修订版.pdf':'153ff79fa140c8e66f09c4d13c15b3f51989318c87ad08a7ff97412cca953aa1'}
    pd.DataFrame([{'原件':str(p),'SHA256':sha(p),'原哈希':h,'未修改':sha(p)==h} for p,h in protected.items()]).to_csv(OUT/'原件保护核对.csv',index=False,encoding='utf-8-sig')
    for p,h in protected.items():check('原件未改_'+p.name,sha(p)==h,h)
    pd.DataFrame(checks).to_csv(OUT/'逐项预测与指标复核.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(metric_checks).to_csv(OUT/'指标序列化精度逐项对照.csv',index=False,encoding='utf-8-sig')
    env={'python':sys.version,'executable':sys.executable,'packages':{}}
    for pkg in ['numpy','pandas','scikit-learn','xgboost','torch','tabicl','tabm','tabpfn']:
        try:env['packages'][pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:env['packages'][pkg]='NA：此核验解释器未安装；训练环境见固定配置'
    weights=[ROOT/'models/tabpfn-v2-regressor.ckpt']+list((Path.home()/'.cache/huggingface/hub/models--jingang--TabICL/snapshots').glob('*/tabicl-regressor-v2-20260212.ckpt'))
    env['冻结预训练权重']=[{'文件':str(p),'SHA256':sha(p),'字节':p.stat().st_size,'已微调权重':False} for p in weights if p.exists()]
    scripts=[ROOT/'models/tabiclv2_adapter.py',ROOT/'结果/扩展实验/表格预训练算法对比_20260908/portable_package/run_experiment.py',ROOT/'脚本/04_模型训练/26_run_quality_aware_remote_ab.py',ROOT/'脚本/04_模型训练/30_cross_validate_zone_model_spatial.py',ROOT/'脚本/04_模型训练/05_train_full_station_models.py',ROOT/'脚本/02_特征构建/04_build_model_table.py']
    scripts+=list((ROOT/'脚本/02_特征构建').glob('3[3-6]_*.py'))+list((ROOT/'脚本/04_模型训练').glob('5[1-5]_*.py'))+list((ROOT/'脚本/05_诊断分析').glob('11[6-9]_*.py'))+list((ROOT/'脚本/05_诊断分析').glob('12[01]_*.py'))
    pd.DataFrame([{'脚本':str(p),'SHA256':sha(p),'字节':p.stat().st_size} for p in scripts]).to_csv(OUT/'本轮脚本及模型适配器SHA256.csv',index=False,encoding='utf-8-sig')
    (OUT/'核验环境.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
    failed=[x for x in checks if x['结果']=='FAIL'];(OUT/'验收状态.json').write_text(json.dumps({'核验项':len(checks),'失败':len(failed),'执行覆盖':states,'状态':'PASS' if not failed else 'PARTIAL_OR_FAILED','正式数据版本':frozen['dataset_version']},ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'checks':len(checks),'failed':len(failed),'coverage':states},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
