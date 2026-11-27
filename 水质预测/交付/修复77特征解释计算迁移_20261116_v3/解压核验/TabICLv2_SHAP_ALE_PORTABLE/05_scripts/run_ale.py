"""仅迁移机执行：原20变量/边界/查询ID；数据归属改变时记录阻塞，绝不静默重选。"""
import argparse,json,traceback
from portable_context import *
def prepare(task,feature):
    import numpy as np,pandas as pd
    cfg=config();r=next(v for v in cfg['contexts'] if v['task']==task and v['fold']=='fold_1')
    raw=pd.read_csv(ROOT/r['raw_test']).set_index('obs_id');a=np.load(ROOT/r['npz']);station=dict(zip(a['test_ids'],a['stations']))
    p=ROOT/'04_explanation_config/ale';sup=pd.read_csv(p/f'{task}_{feature}_bin_support.csv').sort_values('bin');q=pd.read_csv(p/f'{task}_{feature}_query_manifest.csv')
    edges=np.array(sup.lower.tolist()+[float(sup.upper.iloc[-1])]);vals=raw.loc[q.obs_id,feature].to_numpy();binidx=np.clip(np.digitize(vals,edges[1:-1],right=True),0,len(sup)-1)+1
    invalid=~np.isfinite(vals)|(vals<edges[0])|(vals>edges[-1])|(binidx!=q.bin.to_numpy())
    if invalid.any():
        atomic_csv(OUT/'ale_1d'/f'{task}_{feature}_data_drift.csv',q.assign(repaired_raw_value=vals,repaired_bin=binidx,invalid=invalid))
        raise RuntimeError(f'REGISTRATION_DATA_DRIFT：{int(invalid.sum())}条原登记查询的分箱归属改变；需核验新版本登记，不自动删样本或补点')
    values=raw[feature].to_numpy();membership=np.clip(np.digitize(values,edges[1:-1],right=True),0,len(sup)-1)+1;valid=np.isfinite(values)&(values>=edges[0])&(values<=edges[-1])
    sts=raw.index.map(station).to_numpy()
    for ix,row in sup.iterrows():
        b=int(row.bin);full=valid&(membership==b);sub=q.bin.eq(b)
        sup.loc[ix,'full_query_n']=int(full.sum());sup.loc[ix,'full_query_stations']=len(set(sts[full]));sup.loc[ix,'selected_n']=int(sub.sum());sup.loc[ix,'selected_stations']=q.loc[sub,'station_id'].nunique()
    sup['full_query_supported']=(sup.full_query_n>=30)&(sup.full_query_stations>=5);sup['selected_support']=(sup.selected_n>=30)&(sup.selected_stations>=5)
    if not sup.full_query_supported.all() or not sup.selected_support.all():raise RuntimeError('INSUFFICIENT_SUPPORT：原分箱在新值中支持不足；保留NA，不生成连续ALE')
    q['feature_value']=vals
    for name,frame in [('bin_support',sup),('query_manifest',q)]:
        out=OUT/'ale_1d'/f'{task}_{feature}_{name}.csv'
        if out.exists():
            old=pd.read_csv(out)
            if old.shape!=frame.shape or old.to_json()!=frame.to_json():raise RuntimeError('已保存ALE登记与当前定义不一致，禁止覆盖')
        else:atomic_csv(out,frame)
def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS);a=p.parse_args();failed=[]
    with gpu_lock():
        for task in ([a.task] if a.task else TASKS):
            for curve in [v for v in config()['ale_curves'] if v['task']==task]:
                feature=curve['feature'];key=f'{task}_{feature}'
                try:
                    prerequisite(task);prepare(task,feature)
                    module=load_module(ROOT/'05_scripts/reused_ale_engine.py','reused_wq77_ale');module.calculate(task,feature)
                except Exception as e:
                    failed.append(key);atomic_json(OUT/'status'/f'ale_{key}.json',{'status':'BLOCKED' if 'DRIFT' in str(e) or 'SUPPORT' in str(e) else 'FAILED','error':str(e),'traceback':traceback.format_exc()})
                else:atomic_json(OUT/'status'/f'ale_{key}.json',{'status':'COMPUTED'})
    if failed:raise SystemExit('未完成曲线（其余已继续）：'+','.join(failed))
if __name__=='__main__':main()
