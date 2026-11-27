"""Freeze shared five-fold inner CV on the existing Random Strict training IDs."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/random_10task_v1/01_random_split_manifest.csv'
OUT = ROOT / 'runs/random_tabular_model_zoo_v2'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=pd.read_csv(SOURCE,dtype={'obs_id':str},low_memory=False)
    assert set(source.validation)=={'random_strict'} and set(source.fold)=={'seed_42'}
    assert len(source.task.unique())==10
    rows=[]
    for task,frame in source.groupby('task',sort=False):
        train=frame.loc[frame.split.eq('train'),['obs_id']].sort_values('obs_id').reset_index(drop=True)
        test=frame.loc[frame.split.eq('test'),'obs_id']
        assert train.obs_id.is_unique and test.is_unique and not set(train.obs_id)&set(test)
        foldno=[None]*len(train)
        for i,(_,val) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42).split(train),1):
            for j in val:foldno[j]=i
        train['task']=task;train['inner_fold']=foldno;rows.append(train)
    inner=pd.concat(rows,ignore_index=True)
    assert inner.groupby('task').inner_fold.nunique().eq(5).all()
    inner.to_csv(OUT/'01_inner_fold_manifest.csv',index=False,encoding='utf-8-sig')
    protocol={
       'experiment':'RANDOM_TABULAR_MODEL_ZOO_V2','parent_split':str(SOURCE.relative_to(ROOT)),
       'parent_split_sha256':sha(SOURCE),'inner_split':'KFold(n_splits=5, shuffle=True, random_state=42) on sorted external train obs_id',
       'outer_test_policy':'held out for one final evaluation; never used for tuning, fusion weights, or model selection',
       'feature_manifest':'runs/tabiclv2_pilot/01_strict_feature_manifest.csv',
       'feature_manifest_sha256':sha(ROOT/'runs/tabiclv2_pilot/01_strict_feature_manifest.csv')}
    (OUT/'00_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print(inner.groupby(['task','inner_fold']).size().unstack().to_string())

if __name__=='__main__':main()
