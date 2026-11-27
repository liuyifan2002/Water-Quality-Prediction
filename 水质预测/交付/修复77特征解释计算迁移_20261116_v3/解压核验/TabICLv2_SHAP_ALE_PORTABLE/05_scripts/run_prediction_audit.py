"""仅迁移机运行：原完整测试批复现+30预登记样本P1重复审查。"""
import argparse,json,time
from portable_context import *
def audit(task):
    import numpy as np,pandas as pd,gc
    cfg=config();stamp=OUT/'audit'/f'{task}.json';signature=file_sha(ROOT/'04_explanation_config/portable_config.json')
    if stamp.exists():
        saved=json.loads(stamp.read_text(encoding='utf-8'))
        if saved.get('configuration_sha256')==signature and saved.get('identity_pass'):return saved
        raise RuntimeError('已有审计与输入版本不符，不覆盖')
    _,fs,tr,te,xt,xe,model,inv,meta=context(task)
    r=next(v for v in cfg['contexts'] if v['task']==task and v['fold']=='fold_1');ref=pd.read_csv(ROOT/r['prediction']).set_index('obs_id').loc[te.obs_id]
    # 固定样本处在原整批中；不能对子集批预测后假称复现原完整测试批。
    pred=np.maximum(inv(model.predict(xe)),0);assert np.isfinite(pred).all()
    delta=np.abs(pred-ref.y_pred.to_numpy());rows=[];indices={v:i for i,v in enumerate(te.obs_id)}
    for s in sample_rows(task).itertuples():
        i=indices[str(s.obs_id)];values=[]
        for repeat in (1,2,3):
            v=float(np.maximum(inv(model.predict(xe[i:i+1])),0)[0]);assert np.isfinite(v);values.append(v)
            rows.append({'task':task,'obs_id':s.obs_id,'repeat':repeat,'p1_prediction':v,'frozen_batch_prediction':float(ref.y_pred.iloc[i]),'p1_minus_frozen_batch':v-float(ref.y_pred.iloc[i]),**meta})
    atomic_csv(OUT/'audit'/f'{task}_fixed_samples.csv',pd.DataFrame(rows))
    fixed=pd.DataFrame(rows);ranges=fixed.groupby('obs_id').p1_prediction.agg(lambda s:s.max()-s.min())
    result={**meta,'identity_pass':True,'batch_reproduction':'PASS' if delta.max()<=1e-6 and delta.mean()<=1e-7 else 'NUMERICAL_REPRODUCTION_FAILED',
       'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),'p1_repeat_max_range':float(ranges.max()),
       'p1_repeat_status':'PASS' if ranges.max()<=1e-6 else 'NUMERICAL_LIMITATION','configuration_sha256':signature,
       'batch_n':len(pred),'fixed_sample_n':len(sample_rows(task)),'scope':'P1与批量差异独立保存，资格不等价不伪装通过；沿用V2计算/质量分离'}
    atomic_json(stamp,result);del model;gc.collect();return result
def main():
    from run_preflight import preflight
    import traceback
    preflight(dependencies=True);p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS);a=p.parse_args()
    failed=[]
    with gpu_lock():
        for t in ([a.task] if a.task else TASKS):
            try:print(json.dumps(audit(t),ensure_ascii=False),flush=True)
            except Exception as e:
                failed.append(t);atomic_json(OUT/'status'/f'audit_{t}.json',{'status':'FAILED','error':str(e),'traceback':traceback.format_exc()})
    if failed:raise SystemExit('审计失败任务（其余已继续）：'+','.join(failed))
if __name__=='__main__':main()
