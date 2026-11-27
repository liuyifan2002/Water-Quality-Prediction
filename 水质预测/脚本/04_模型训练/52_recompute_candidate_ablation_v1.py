"""固定新数据有效输入和原来源taxonomy的四任务六来源消融，不混用旧完整模型。"""
from pathlib import Path
import json,time,hashlib,argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';CORE=RUN/'03_正式十任务重算';OUT=RUN/'04_来源消融重算'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
    frozen=json.loads((RUN/'02_冻结候选/冻结清单.json').read_text(encoding='utf-8'));policy=json.loads((RUN/'02_冻结候选/特征与时间协议.json').read_text(encoding='utf-8'));fs=policy['feature_order']
    taxpath=ROOT/'runs/multisource_ablation_v1/00_feature_source_taxonomy.csv';tax=pd.read_csv(taxpath);assert set(tax.feature_name)==set(fs)
    groups=sorted(g for g in tax.source_group.unique() if g!='REMOTE_SENSING_OCEAN_COLOR');assert len(groups)==6
    cfgpath=ROOT/'结果/扩展实验/表格预训练算法对比_20260908/portable_package/config.json';params=dict(json.loads(cfgpath.read_text(encoding='utf-8-sig'))['model_parameters']['xgboost']);params.update(random_state=42,objective='reg:squarederror')
    previous=pd.read_csv(OUT/'消融运行登记.csv') if (OUT/'消融运行登记.csv').exists() else pd.DataFrame()
    (OUT/'固定配置.json').write_text(json.dumps({'model':'XGBoost','seed':42,'params':params,'config_sha256':sha(cfgpath),'taxonomy_sha256':sha(taxpath),'dataset_version':frozen['dataset_version'],'target_transform':'非负训练目标log1p；expm1后截非负','imputation':'正式有效输入训练中位数；删组不改变保留列的填补统计','delta':'消融减同版本完整模型；不复用旧基线'},ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for task in ['chla_surface','secchi','tn_surface','tp_middle']:
        for k in range(1,6):
            fold=f'fold_{k}';p=CORE/'inputs'/f'{task}_{fold}.npz'
            if not p.exists():raise RuntimeError('需先生成并验证正式按折输入：'+str(p))
            a=np.load(p);y=a['y_train'];yt=a['y_true'];nonneg=y.min()>=0;yf=np.log1p(y) if nonneg else y
            for group in groups:
                kept=[f for f in fs if f not in set(tax.loc[tax.source_group.eq(group),'feature_name'])];ix=[fs.index(f) for f in kept];outp=OUT/'predictions'/f'{task}_{group}_{fold}.csv'
                start=time.perf_counter()
                existed=outp.exists()
                if existed:
                    d=pd.read_csv(outp);assert d.dataset_version.eq(frozen['dataset_version']).all() and d.effective_full_input_sha256.eq(sha(p)).all()
                else:
                    model=XGBRegressor(**params);model.fit(a['x_train'][:,ix],yf);pred=model.predict(a['x_test'][:,ix]);pred=np.maximum(np.expm1(pred) if nonneg else pred,0)
                    d=pd.DataFrame({'obs_id':a['test_ids'],'station_id':a['stations'],'date':a['dates'],'task':task,'fold':fold,'source_group_removed':group,'y_true':yt,'y_pred':pred,'dataset_version':frozen['dataset_version'],'effective_full_input_sha256':sha(p),'taxonomy_sha256':sha(taxpath)})
                    d.to_csv(outp,index=False,encoding='utf-8-sig');del model
                duration=time.perf_counter()-start
                if existed and not previous.empty:
                    old=previous[previous.task.eq(task)&previous.fold.eq(fold)&previous.source_group_removed.eq(group)]
                    if len(old)==1:duration=float(old.runtime.iloc[0])
                rows.append({'task':task,'fold':fold,'source_group_removed':group,'n_features':len(kept),'n_test':len(yt),'R2':r2_score(d.y_true,d.y_pred),'RMSE':mean_squared_error(d.y_true,d.y_pred)**.5,'MAE':mean_absolute_error(d.y_true,d.y_pred),'runtime':duration,'status':'OK','dataset_version':frozen['dataset_version'],'prediction_file':str(outp),'prediction_sha256':sha(outp),'config_sha256':sha(cfgpath),'effective_input_sha256':sha(p)})
                pd.DataFrame(rows).to_csv(OUT/'消融运行登记.csv',index=False,encoding='utf-8-sig');print('ABLATION_OK',len(rows),'/120',task,fold,group,flush=True)
    pooled=[]
    for (task,group),g in pd.DataFrame(rows).groupby(['task','source_group_removed']):
        d=pd.concat([pd.read_csv(p) for p in g.prediction_file]);fullpaths=[CORE/'predictions'/f'{task}_xgboost_fold_{i}.csv' for i in range(1,6)]
        r={'task':task,'source_group_removed':group,'ablated_R2':r2_score(d.y_true,d.y_pred),'ablated_RMSE':mean_squared_error(d.y_true,d.y_pred)**.5,'ablated_MAE':mean_absolute_error(d.y_true,d.y_pred),'dataset_version':frozen['dataset_version']}
        if all(p.exists() for p in fullpaths):
            full=pd.concat([pd.read_csv(p) for p in fullpaths]);assert full.dataset_version.eq(frozen['dataset_version']).all()
            merged=d.merge(full[['obs_id','y_pred','y_true']],on='obs_id',validate='one_to_one',suffixes=('_ablated','_full'));assert np.allclose(merged.y_true_ablated,merged.y_true_full)
            r.update(full_R2=r2_score(full.y_true,full.y_pred),full_RMSE=mean_squared_error(full.y_true,full.y_pred)**.5,full_MAE=mean_absolute_error(full.y_true,full.y_pred));r.update(delta_R2=r['ablated_R2']-r['full_R2'],delta_RMSE=r['ablated_RMSE']-r['full_RMSE'],delta_MAE=r['ablated_MAE']-r['full_MAE'],比较状态='同新版本完整模型对照')
        else:r['比较状态']='完整新模型尚未全部完成；不读取旧基线'
        pooled.append(r)
    pd.DataFrame(pooled).to_csv(OUT/'四任务六来源_新版本完整对照.csv',index=False,encoding='utf-8-sig');print('ABLATION_FINISHED',len(rows),flush=True)

if __name__=='__main__':main()
