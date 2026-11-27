"""迁移端新版本上下文适配；导入本文件不会加载模型或执行推理。"""
from pathlib import Path
import csv,json,hashlib,sys,importlib.util,os,contextlib,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
P1='TABICLV2_POINTWISE_EXPLANATION_V1'
ADD_TOL=.02
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def config():return json.loads((ROOT/'04_explanation_config/portable_config.json').read_text(encoding='utf-8'))
TASKS=tuple(config()['tasks'])
def hash_array(value):
    import numpy as np
    a=np.ascontiguousarray(np.asarray(value));return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def hash_ids(v):return hashlib.sha256('\n'.join(map(str,v)).encode()).hexdigest()
def atomic_csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(tmp,index=False,encoding='utf-8-sig');tmp.replace(path)
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def load_module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sample_rows(task,phase='CORE_3'):
    import pandas as pd
    d=pd.read_csv(ROOT/'04_explanation_config/shap/core_samples.csv');return d[d.task.eq(task)&d.phase.eq(phase)].copy()
def context(task,fold='fold_1'):
    # 仅迁移机显式运行入口调用；fit只重建官方上下文，不微调模型权重。
    import numpy as np,pandas as pd,joblib
    cfg=config();r=next(v for v in cfg['contexts'] if v['task']==task and v['fold']==fold)
    a=np.load(ROOT/r['npz']);imp=joblib.load(ROOT/r['imputer']);features=cfg['features']
    rawtr=pd.read_csv(ROOT/r['raw_train']);rawte=pd.read_csv(ROOT/r['raw_test'])
    assert np.allclose(imp.transform(rawtr[features]).astype('float32'),a['x_train'])
    assert np.allclose(imp.transform(rawte[features]).astype('float32'),a['x_test'])
    base=pd.read_csv(ROOT/'02_data/repaired_features/base_77.csv').set_index('obs_id',drop=False)
    train=base.loc[a['train_ids']].reset_index(drop=True).rename(columns={'station':'station_id'})
    test=base.loc[a['test_ids']].reset_index(drop=True).rename(columns={'station':'station_id'})
    for c in features:train[c]=rawtr[c].values;test[c]=rawte[c].values
    modelcode=load_module(ROOT/'05_scripts/tabiclv2_adapter.py','wq77_official_adapter')
    ckpt=ROOT/cfg['checkpoint'];assert file_sha(ckpt)==cfg['checkpoint_sha256']
    adapter=modelcode.TabICLv2Regressor(model_path=str(ckpt),allow_auto_download=False)
    y=a['y_train'];use_log=bool(y.min()>=0);yf=np.log1p(y) if use_log else y.copy()
    adapter.fit(a['x_train'],yf)
    inv=np.expm1 if use_log else (lambda x:x)
    meta={'dataset_version':cfg['dataset_version'],'task':task,'fold':fold,'protocol':P1,
          'feature_order_hash':hash_ids(features),'imputer_statistics_hash':hash_array(imp.statistics_),
          'checkpoint_sha256':cfg['checkpoint_sha256'],'context_hash':hash_array(a['x_train'])+':'+hash_array(yf.astype('float32')),
          'effective_input_sha256':file_sha(ROOT/r['npz']),'configuration_sha256':file_sha(ROOT/'04_explanation_config/portable_config.json'),
          'train_obs_order_hash':hash_ids(a['train_ids']),'test_obs_order_hash':hash_ids(a['test_ids']),
          'target_transform':'log1p/expm1' if use_log else 'identity','scope':'P1单查询解释，不保证与完整测试批数值相同'}
    return None,features,train,test,a['x_train'],a['x_test'],adapter,inv,meta
def prerequisite(task):
    from run_preflight import preflight
    preflight(dependencies=True)
    stamp=OUT/'audit'/f'{task}.json'
    if not stamp.exists():raise RuntimeError('先在迁移机运行 run_prediction_audit.py --task '+task)
    d=json.loads(stamp.read_text(encoding='utf-8'))
    if d.get('configuration_sha256')!=file_sha(ROOT/'04_explanation_config/portable_config.json') or not d.get('identity_pass'):
        raise RuntimeError('模型身份/配置预检未通过或审计过期')
    # 数值资格失败是V2质量证据，不假装通过，也不删除原已登记任务。
    return d
@contextlib.contextmanager
def gpu_lock():
    import psutil
    OUT.mkdir(exist_ok=True);p=OUT/'gpu_worker.lock'
    if p.exists():
        v=json.loads(p.read_text());pid=v.get('pid',0)
        if psutil.pid_exists(pid):raise RuntimeError(f'已有GPU队列 pid={pid}，请勿并行启动')
        p.unlink()
    with p.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'started':time.time()},f)
    try:yield
    finally:
        if p.exists():p.unlink()
