"""仅汇总已冻结的WQ77重算结果；不训练或重新预测。"""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
OUT = Path(__file__).resolve().parents[1]
DATA = OUT / 'data'
DATA.mkdir(parents=True, exist_ok=True)
VERSION = 'WQ77_REPAIRED_20261115_V1'
ORDER = ['chla_bottom','chla_middle','chla_surface','secchi','tn_bottom','tn_middle','tn_surface','tp_bottom','tp_middle','tp_surface']
LABEL = {'chla_bottom':'Chl-a底层','chla_middle':'Chl-a中层','chla_surface':'Chl-a表层','secchi':'Secchi','tn_bottom':'TN底层','tn_middle':'TN中层','tn_surface':'TN表层','tp_bottom':'TP底层','tp_middle':'TP中层','tp_surface':'TP表层'}
SELECTED = ['chla_surface','secchi','tn_surface','tp_middle']
GROUP = {'METEOROLOGY':'MET','RAINFALL_HYDRO_TIDE':'RHT','OCEAN_PHYSICS':'OCN','HUMAN_ACTIVITY':'HUM','TEMPORAL_PROCESS':'TMP','SPATIAL_STATIC':'SPA'}

def source(rel):
    p = BASE / rel
    assert p.is_file(), p
    return p

def csv(rel):
    return pd.read_csv(source(rel))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

for sub, name, n in [('03_正式十任务重算','运行登记.csv',100),('04_来源消融重算','消融运行登记.csv',120),('05_现代模型重算','运行登记.csv',40),('06_历史78特征分区重算','运行登记.csv',60)]:
    d=csv(f'{sub}/{name}')
    assert len(d)==n and d.status.eq('OK').all(), (sub,len(d),d.status.value_counts().to_dict())

metrics=csv('08_重算结果核验/新候选_pooled与逐折指标.csv')
formal=metrics[metrics['范围']=='正式十任务'].copy()
assert len(formal)==20 and formal.dataset_version.eq(VERSION).all()
assert set(formal.task)==set(ORDER) and set(formal.model)=={'xgboost','tabiclv2'}
f= formal.pivot(index='task',columns='model',values=['pooled_R2','pooled_RMSE','pooled_MAE','n_test','n_stations'])
station=csv('08_重算结果核验/十任务站点改善覆盖.csv').set_index('task')
rows=[]
for task in ORDER:
    r={'task':task,'任务':LABEL[task]}
    for m,name in [('xgboost','XGBoost'),('tabiclv2','TabICLv2')]:
        for col,new in [('pooled_R2','R2'),('pooled_RMSE','RMSE'),('pooled_MAE','MAE')]:r[f'{name}_{new}']=float(f.loc[task,(col,m)])
    r['delta_R2']=r['TabICLv2_R2']-r['XGBoost_R2']
    r['delta_RMSE']=r['TabICLv2_RMSE']-r['XGBoost_RMSE']
    r['delta_MAE']=r['TabICLv2_MAE']-r['XGBoost_MAE']
    r['n_test']=int(f.loc[task,('n_test','xgboost')]);r['n_stations']=int(station.loc[task,'n_stations'])
    r['station_win_rate']=float(station.loc[task,'station_RMSE_win_rate'])
    rows.append(r)
core=pd.DataFrame(rows);core.to_csv(DATA/'正式十任务指标.csv',index=False,encoding='utf-8-sig')

modern=metrics[metrics['范围']=='现代模型'].copy()
assert len(modern)==8 and set(modern.task)==set(SELECTED)
mr=[]
for task in SELECTED:
    for model in ['xgboost','tabiclv2','tabpfn','tabm']:
        x=metrics[(metrics.task==task)&(metrics.model==model)&(metrics['范围']==('现代模型' if model in ('tabpfn','tabm') else '正式十任务'))]
        assert len(x)==1,(task,model,len(x))
        x=x.iloc[0];mr.append({'task':task,'任务':LABEL[task],'model':model,'pooled_R2':x.pooled_R2,'pooled_RMSE':x.pooled_RMSE,'pooled_MAE':x.pooled_MAE,'n_test':x.n_test})
modern=pd.DataFrame(mr);modern.to_csv(DATA/'四模型同协议指标.csv',index=False,encoding='utf-8-sig')

ab=csv('04_来源消融重算/四任务六来源_新版本完整对照.csv')
assert len(ab)==24 and ab.dataset_version.eq(VERSION).all()
ab['来源']=ab.source_group_removed.map(GROUP);assert ab['来源'].notna().all()
for _,r in ab.iterrows():
    assert abs(r.delta_R2-(r.ablated_R2-r.full_R2))<1e-6
    assert abs(r.delta_RMSE-(r.ablated_RMSE-r.full_RMSE))<1e-6
wins=[]
for row in ab.itertuples():
    raw=[]
    for fold in range(1,6):
        full=pd.read_csv(source(f'03_正式十任务重算/predictions/{row.task}_xgboost_fold_{fold}.csv'))
        less=pd.read_csv(source(f'04_来源消融重算/predictions/{row.task}_{row.source_group_removed}_fold_{fold}.csv'))
        assert set(full.obs_id)==set(less.obs_id)
        joined=full[['obs_id','station_id','y_true','y_pred']].merge(less[['obs_id','station_id','y_true','y_pred']],on='obs_id',suffixes=('_full','_less'),validate='one_to_one')
        assert (joined.station_id_full==joined.station_id_less).all() and (joined.y_true_full-joined.y_true_less).abs().max()<1e-10
        raw.append(joined)
    allrows=pd.concat(raw,ignore_index=True)
    g=allrows.groupby('station_id_full',sort=False)
    a=g.apply(lambda x:((x.y_true_full-x.y_pred_full)**2).mean()**.5,include_groups=False)
    b=g.apply(lambda x:((x.y_true_full-x.y_pred_less)**2).mean()**.5,include_groups=False)
    wins.append((row.task,row.source_group_removed,int(len(a)),float((b<a).mean())))
win=pd.DataFrame(wins,columns=['task','source_group_removed','站点数','删除后RMSE改善站点比例'])
ab=ab.merge(win,on=['task','source_group_removed'],validate='one_to_one')
ab.to_csv(DATA/'四任务六来源消融.csv',index=False,encoding='utf-8-sig')

regional=csv('06_历史78特征分区重算/五折留站点验证_汇总.csv')
folds=csv('06_历史78特征分区重算/五折留站点验证_各折指标.csv')
assert len(regional)==12 and len(folds)==120
regional.to_csv(DATA/'历史78特征分区汇总.csv',index=False,encoding='utf-8-sig')
folds.to_csv(DATA/'历史78特征分区逐折.csv',index=False,encoding='utf-8-sig')

site=csv('08_重算结果核验/十任务逐站点误差.csv')
assert len(site)==1700 and site.dataset_version.eq(VERSION).all()
site.to_csv(DATA/'十任务逐站点误差.csv',index=False,encoding='utf-8-sig')
sitepair=csv('08_重算结果核验/十任务逐站点新模型比较.csv')
sitepair.to_csv(DATA/'十任务逐站点模型差.csv',index=False,encoding='utf-8-sig')

inputs=[source(x) for x in ['08_重算结果核验/新候选_pooled与逐折指标.csv','08_重算结果核验/十任务站点改善覆盖.csv','08_重算结果核验/十任务逐站点误差.csv','04_来源消融重算/四任务六来源_新版本完整对照.csv','06_历史78特征分区重算/五折留站点验证_汇总.csv','06_历史78特征分区重算/五折留站点验证_各折指标.csv']]
(DATA/'输入文件哈希.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in inputs},ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: 20 formal rows, 16 modern rows, 24 ablation rows, 12 regional rows, 1700 site rows')
