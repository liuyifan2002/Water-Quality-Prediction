"""少量固定P1预测跨进程探针；不重训、不运行SHAP/ALE、不改冻结结果。"""
from pathlib import Path
import json,os,subprocess,sys

HERE=Path(__file__).resolve()
ROOT=HERE.parents[3]
OUT=HERE.parents[1]/'explanation_audit'
OUT.mkdir(exist_ok=True)
PORT=ROOT/'交付/修复77特征解释计算迁移_20261116_v3/TabICLv2_SHAP_ALE_PORTABLE/05_scripts'
assert PORT.is_dir()

def child():
    sys.path.insert(0,str(PORT))
    from portable_context import context,sample_rows
    import numpy as np
    task='chla_surface'
    _,features,tr,te,xt,xe,model,inv,meta=context(task)
    idx={str(v):i for i,v in enumerate(te.obs_id)}
    samples=sample_rows(task)
    result=[]
    for s in samples.itertuples():
        i=idx[str(s.obs_id)]
        value=float(np.maximum(inv(model.predict(xe[i:i+1])),0)[0])
        result.append({'task':task,'obs_id':s.obs_id,'prediction':value})
    out={'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED'),'context_hash':meta['context_hash'],'checkpoint_sha256':meta['checkpoint_sha256'],'feature_order_hash':meta['feature_order_hash'],'fixed_predictions':result}
    print(json.dumps(out,ensure_ascii=False),flush=True)

def main():
    rows=[]
    for seed in ['0','1']:
        env=os.environ.copy();env['PYTHONHASHSEED']=seed;env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONIOENCODING']='utf-8'
        p=subprocess.run([sys.executable,'-B',str(HERE),'--child'],cwd=str(ROOT),env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=2400)
        log=OUT/f'P1_hashseed_{seed}.log';log.write_text(p.stdout+'\nSTDERR\n'+p.stderr,encoding='utf-8')
        if p.returncode:raise RuntimeError(f'PYTHONHASHSEED={seed} failed: {p.returncode}; see {log}')
        data=json.loads(p.stdout.strip().splitlines()[-1]);rows.append(data)
    (OUT/'两种哈希种子P1探针.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__':
    child() if '--child' in sys.argv else main()
