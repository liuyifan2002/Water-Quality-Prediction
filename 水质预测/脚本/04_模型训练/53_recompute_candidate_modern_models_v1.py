"""原现代模型注册范围的新输入重算；保留原参数及种子，独立输出。"""
from pathlib import Path
import sys, json, importlib.util, time, gc
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('candidate_core',Path(__file__).with_name('51_recompute_accepted_candidate_v1.py'))
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
OUT=core.RUN/'05_现代模型重算';TASKS=['chla_surface','secchi','tn_surface','tp_middle']
PORTABLE=ROOT/'结果/扩展实验/表格预训练算法对比_20260908/portable_package/run_experiment.py'
CONFIG=PORTABLE.with_name('config.json');SEED=20260921

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for d in ['predictions','fold_metrics']:(OUT/d).mkdir(exist_ok=True)
    cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig'));base=core.import_path('candidate_modern_portable',PORTABLE)
    fs=json.loads((core.FROZEN/'特征与时间协议.json').read_text(encoding='utf-8'))['feature_order']
    cr=pd.read_csv(core.OUT/'运行登记.csv');rows=[]
    for task in TASKS:
        for model in ['tabpfn','tabm']:
            for fold in [f'fold_{i}' for i in range(1,6)]:
                q=cr[(cr.task==task)&(cr.fold==fold)].iloc[0]
                rawtr=core.OUT/'inputs'/f'{task}_{fold}_训练未填补有效特征.csv';rawte=rawtr.with_name(f'{task}_{fold}_测试未填补有效特征.csv')
                rows.append({'task':task,'model':model,'fold':fold,'status':'PENDING','dataset_version':q.dataset_version,'base_input_sha256':q.base_input_sha256,'effective_input_sha256':q.effective_input_sha256,'raw_train_sha256':core.sha(rawtr),'raw_test_sha256':core.sha(rawte),'config_sha256':core.sha(CONFIG),'implementation_sha256':core.sha(PORTABLE),'manifest_sha256':q.manifest_sha256,'prediction_file':str(OUT/'predictions'/f'{task}_{model}_{fold}.csv'),'error':''})
    reg=pd.DataFrame(rows);rp=OUT/'运行登记.csv'
    if rp.exists():
        old=pd.read_csv(rp,keep_default_na=False)
        for c in ['task','model','fold','dataset_version','effective_input_sha256','raw_train_sha256','raw_test_sha256','config_sha256','implementation_sha256','manifest_sha256']:assert reg[c].astype(str).equals(old[c].astype(str)),c
        reg=old
    else:reg.to_csv(rp,index=False,encoding='utf-8-sig')
    (OUT/'固定协议.json').write_text(json.dumps({'seed':SEED,'tasks':TASKS,'config':cfg,'protocol':'原modern_baseline_gate_v1；原77项原始缺失输入，按模型原处理器拟合；仅替换候选值和折训练期SST气候态','code_sha256':core.sha(PORTABLE),'原始预处理':'TabM内部训练/验证中位数与缩放，轮数选择后全部外层训练集重训；TabPFN原生处理缺失','数据版本':reg.dataset_version.iloc[0]},ensure_ascii=False,indent=2),encoding='utf-8')
    # GPU与正式重算串行，不改变任何模型配置以争抢显存。
    while True:
        try:
            current=pd.read_csv(core.OUT/'运行登记.csv')
            if len(current)==100 and current.status.isin(['OK','FAILED']).all():break
        except (pd.errors.EmptyDataError,pd.errors.ParserError,PermissionError):pass
        print('WAIT_FORMAL_GPU',flush=True);time.sleep(30)
    for i,r in reg.iterrows():
        pp=Path(r.prediction_file)
        if r.status=='OK' and pp.exists():
            assert core.sha(pp)==r.prediction_sha256
            continue
        start=time.perf_counter();reg.loc[i,'status']='RUNNING';reg.to_csv(rp,index=False,encoding='utf-8-sig');print('MODERN_START',r.task,r.model,r.fold,flush=True)
        try:
            stem=f'{r.task}_{r.fold}';a=np.load(core.OUT/'inputs'/f'{stem}.npz')
            tr=pd.read_csv(core.OUT/'inputs'/f'{stem}_训练未填补有效特征.csv');te=pd.read_csv(core.OUT/'inputs'/f'{stem}_测试未填补有效特征.csv')
            assert tr.obs_id.astype(str).tolist()==a['train_ids'].tolist() and te.obs_id.astype(str).tolist()==a['test_ids'].tolist()
            y,inv=base.target_transform(a['y_train']);fn=base.fit_tabpfn if r.model=='tabpfn' else base.fit_tabm
            p=inv(fn(tr[fs],y,te[fs],SEED,cfg));assert np.isfinite(p).all() and len(p)==len(te)
            mm={'R2':float(base.r2_score(a['y_true'],p)),'RMSE':float(base.mean_squared_error(a['y_true'],p)**.5),'MAE':float(base.mean_absolute_error(a['y_true'],p))}
            pd.DataFrame({'obs_id':a['test_ids'],'station_id':a['stations'],'date':a['dates'],'task':r.task,'model':r.model,'fold':r.fold,'y_true':a['y_true'],'y_pred':p,'dataset_version':r.dataset_version,'effective_input_sha256':r.effective_input_sha256,'raw_train_sha256':r.raw_train_sha256,'raw_test_sha256':r.raw_test_sha256}).to_csv(pp,index=False,encoding='utf-8-sig')
            pd.DataFrame([mm|{'task':r.task,'model':r.model,'fold':r.fold,'n_train':len(tr),'n_test':len(te),'dataset_version':r.dataset_version}]).to_csv(OUT/'fold_metrics'/pp.name,index=False,encoding='utf-8-sig')
            for k,v in mm.items():reg.loc[i,k]=v
            reg.loc[i,'status']='OK';reg.loc[i,'prediction_sha256']=core.sha(pp);reg.loc[i,'error']=''
            print('MODERN_OK',r.task,r.model,r.fold,mm,flush=True)
        except Exception as ex:
            reg.loc[i,'status']='FAILED';reg.loc[i,'error']=f'{type(ex).__name__}: {ex}';print('MODERN_FAILED',reg.loc[i,'error'],flush=True)
        finally:
            reg.loc[i,'runtime']=time.perf_counter()-start;reg.to_csv(rp,index=False,encoding='utf-8-sig');gc.collect()
            import torch
            torch.cuda.empty_cache()
    print('MODERN_FINISHED',reg.groupby('status').size().to_dict(),flush=True)

if __name__=='__main__':main()
