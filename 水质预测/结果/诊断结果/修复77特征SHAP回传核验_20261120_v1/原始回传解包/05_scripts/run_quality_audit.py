"""不推理：核验新结果完整性、版本血缘、可加性及ALE真实支持，并汇总排名。"""
import json
from portable_context import *
def main():
    import numpy as np,pandas as pd
    from run_preflight import preflight
    preflight(dependencies=False)
    rows=[];importance=[];curves=[];cfg=config();signature=file_sha(ROOT/'04_explanation_config/portable_config.json')
    for task in TASKS:
        for s in sample_rows(task).itertuples():
            p=OUT/'shap'/task/f'{s.obs_id}_seed42.csv.gz';valid=False
            if p.is_file():
                d=pd.read_csv(p);valid=len(d)==77 and d.feature.tolist()==cfg['features'] and d.dataset_version.eq(cfg['dataset_version']).all() and d.configuration_sha256.eq(signature).all() and np.isfinite(d.shap_value).all() and d.additivity_abs_error.le(ADD_TOL).all() and d.status.eq('COMPUTED_PASS').all()
                if valid:importance.append(d.assign(abs_shap=d.shap_value.abs())[['task','feature','abs_shap']])
            rows.append({'task':task,'method':'SHAP','unit':s.obs_id,'status':'PASS' if valid else 'MISSING_OR_FAILED','scope':'独立核心样本；不含种子重复'})
        for c in [v for v in cfg['ale_curves'] if v['task']==task]:
            p=OUT/'ale_1d'/f'{task}_{c["feature"]}_curve.csv';valid=False
            amended=task=='secchi' and c['feature']=='ssrd_sum_3d'
            if p.is_file():
                d=pd.read_csv(p);sup=pd.read_csv(OUT/'ale_1d'/f'{task}_{c["feature"]}_bin_support.csv')
                cr=next(v for v in cfg['contexts'] if v['task']==task and v['fold']=='fold_1');arr=np.load(ROOT/cr['npz']);yy=arr['y_train'];yf=np.log1p(yy) if yy.min()>=0 else yy
                expected=hash_array(arr['x_train'])+':'+hash_array(yf.astype('float32'))
                valid=len(d)==len(sup) and d.protocol.eq(P1).all() and np.isfinite(d.ale).all() and d.n_samples.ge(30).all() and d.n_stations.ge(5).all() and d.context_hash.eq(expected).all() and 'dataset_version' in d and d.dataset_version.eq(cfg['dataset_version']).all() and d.configuration_sha256.eq(signature).all()
                if amended:
                    revision_path=OUT/'ale_1d'/f'{task}_{c["feature"]}_registration_revision.json'
                    source=ROOT/'04_explanation_config/ale'/f'{task}_{c["feature"]}_query_manifest.csv'
                    derived=OUT/'ale_1d'/f'{task}_{c["feature"]}_query_manifest.csv'
                    if not revision_path.is_file() or not derived.is_file():valid=False
                    else:
                        revision=json.loads(revision_path.read_text(encoding='utf-8'))
                        old=pd.read_csv(source);new=pd.read_csv(derived)
                        differences=old.bin.ne(new.bin)
                        valid=valid and revision.get('revision')=='secchi_boundary_v2' and revision.get('source_query_manifest_sha256')==file_sha(source) and old.obs_id.tolist()==new.obs_id.tolist() and set(old.loc[differences,'obs_id'])=={'NT1_20161107_surface','NM6_20161107_surface'} and differences.sum()==2 and old.loc[differences,'bin'].eq(3).all() and new.loc[differences,'bin'].eq(4).all() and np.allclose(new.loc[differences,'feature_value'].to_numpy()-old.loc[differences,'feature_value'].to_numpy(),2,rtol=0,atol=0)
                if valid:curves.append(d.assign(dataset_version=cfg['dataset_version']))
            rows.append({'task':task,'method':'ALE','unit':c['feature'],'status':'PASS' if valid else 'MISSING_OR_FAILED','scope':('Secchi两条跨界样本修订登记v2；原登记保留；' if amended else '')+'实际n/s/局部差值；数值重复状态另列原curve.status'})
    if importance:
        imp=pd.concat(importance).groupby(['task','feature'],sort=False).abs_shap.mean().reset_index(name='mean_abs_shap')
        good={t for t in TASKS if sum(x['status']=='PASS' for x in rows if x['task']==t and x['method']=='SHAP')==3}
        imp=imp[imp.task.isin(good)];imp['rank']=imp.groupby('task').mean_abs_shap.rank(method='min',ascending=False);atomic_csv(OUT/'summary/30核心_77特征排名.csv',imp)
        atomic_csv(OUT/'summary/任务内排名矩阵.csv',imp.pivot(index='feature',columns='task',values='rank').reindex(columns=TASKS).reset_index())
    if curves:atomic_csv(OUT/'summary/ALE真实分箱与支持.csv',pd.concat(curves,ignore_index=True))
    atomic_csv(OUT/'summary/quality_audit.csv',pd.DataFrame(rows));print(json.dumps({'passed':sum(v['status']=='PASS' for v in rows),'planned':50,'注意':'COMPUTED不是科学质量通过；批/P1差异见audit；不足和历史失败不改写'},ensure_ascii=False))
    if any(v['status']!='PASS' for v in rows):raise SystemExit(2)
if __name__=='__main__':main()
