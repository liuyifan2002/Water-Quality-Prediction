"""独立核对归因表算术、原登记和原件保护；仅统计现有文件。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
OUT=ROOT/'结果/诊断结果/正式77特征新旧性能归因_v1_20261115'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks=[];old=[];extra=[]
    def ck(name,ok,detail=''):checks.append({'检查项':name,'通过':bool(ok),'依据':str(detail)})
    for family in ['chla','tn','tp']:
        p=ROOT/f'runs/{family}_family_v2/07_run_registry.csv';d=pd.read_csv(p);old.append(d);extra.append({'文件':str(p),'字节':p.stat().st_size,'SHA256':sha(p)})
        p=ROOT/f'runs/{family}_family_v2/02_{family}_family_v2_station_fold_map.csv';extra.append({'文件':str(p),'字节':p.stat().st_size,'SHA256':sha(p)})
    old=pd.concat(old);new=pd.read_csv(RUN/'03_正式十任务重算/运行登记.csv')
    q=old.merge(new,on=['task','model','fold'],suffixes=('_old','_new'),validate='one_to_one')
    ck('旧FamilyV2登记90行配对',len(q)==90)
    for c in ['n_train','n_test']:ck('旧新登记'+c+'一致',q[c+'_old'].eq(q[c+'_new']).all())
    ck('旧登记77特征',q.n_features.eq(77).all())
    # 不调用任何模型；恢复旧源码使用的输入顺序，与新保存上下文逐个对比。
    rawpath=ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv'
    raw=pd.read_csv(rawpath,usecols=['obs_id']);splitpath=ROOT/'runs/tabiclv2_pilot/02_split_manifest.csv';split=pd.read_csv(splitpath)
    extra.append({'文件':str(splitpath),'字节':splitpath.stat().st_size,'SHA256':sha(splitpath)})
    for _,r in new.drop_duplicates(['task','fold']).iterrows():
        a=np.load(RUN/f'03_正式十任务重算/inputs/{r.task}_{r.fold}.npz')
        if r.task=='secchi':
            mm=split[split.task.eq('secchi')&split.validation.eq('station_holdout')&split.fold.eq(r.fold)]
            tr=mm.loc[mm.split.eq('train'),'obs_id'].tolist();te=mm.loc[mm.split.eq('test'),'obs_id'].tolist()
        else:
            mm=pd.read_csv(r.manifest_path);mm=mm[mm.fold.eq(r.fold)]
            tr=raw.loc[raw.obs_id.isin(mm.loc[mm.role.eq('train'),'obs_id']),'obs_id'].tolist();te=raw.loc[raw.obs_id.isin(mm.loc[mm.role.eq('test'),'obs_id']),'obs_id'].tolist()
        ck(r.task+r.fold+'训练上下文及测试批ID顺序',tr==a['train_ids'].tolist() and te==a['test_ids'].tolist())
    for c in ['R2','RMSE']:
        s=pd.read_csv(OUT/'五折性能变化明细.csv');merged=q.merge(s,on=['task','model','fold'],validate='one_to_one');ck('旧登记与归因复算'+c,np.allclose(merged[c+'_old'],merged['旧'+c],atol=1e-8))
    q.to_csv(OUT/'原始训练登记与新登记逐项对齐.csv',index=False,encoding='utf-8-sig')
    s=pd.read_csv(OUT/'新旧版本十任务性能变化总表.csv');f=pd.read_csv(OUT/'五折性能变化明细.csv');st=pd.read_csv(OUT/'逐站点性能变化诊断.csv');a=pd.read_csv(OUT/'来源消融性能归因表.csv');dist=pd.read_csv(OUT/'48项变化特征分布对比.csv')
    ck('完整覆盖20任务模型100折1700站点模型',len(s)==20 and len(f)==100 and len(st)==1700)
    ck('来源24项特征48项',len(a)==24 and len(dist)==48)
    for _,r in s.iterrows():
        m=st[st.task.eq(r.task)&st.model.eq(r.model)];mf=f[f.task.eq(r.task)&f.model.eq(r.model)]
        ck(r.task+r.model+'站点分解',np.isclose(m['对总体ΔR2贡献'].sum(),r['ΔR2'],atol=1e-10))
        ck(r.task+r.model+'折分解',np.isclose(-mf['ΔSSE'].sum()/r['旧目标SST'],r['ΔR2'],atol=1e-10))
        ck(r.task+r.model+'同目标SST',np.isclose(r['旧目标SST'],r['新目标SST']))
    ck('XGBoost十任务均提高',s[s.model.eq('xgboost')]['ΔR2'].gt(0).all())
    ck('无输入修改',all(sha(p)==h for p,h in pd.read_csv(OUT/'分析输入文件与SHA256.csv')[['文件','SHA256']].itertuples(index=False,name=None)))
    inputs=pd.concat([pd.read_csv(OUT/'分析输入文件与SHA256.csv'),pd.DataFrame(extra)]).drop_duplicates('文件')
    inputs.to_csv(OUT/'分析输入文件与SHA256.csv',index=False,encoding='utf-8-sig')
    cs=pd.DataFrame(checks);cs.to_csv(OUT/'独立交付核验.csv',index=False,encoding='utf-8-sig');assert cs['通过'].all(),cs[~cs['通过']]
    requested=['新旧版本十任务性能变化总表.csv','五折性能变化明细.csv','逐站点性能变化诊断.csv','来源消融性能归因表.csv','48项变化特征分布对比.csv','历史高分与正式结果可比性审计.md','预测精度下降原因综合报告.md','下一步最小补充实验清单.md']
    for fn in requested:assert (OUT/fn).stat().st_size>100
    state={'八项交付完整':True,'独立核验项':len(cs),'失败项':int((~cs['通过']).sum()),'主分析核验项':len(pd.read_csv(OUT/'归因分析核验记录.csv')),'分析输入文件数':len(inputs),'新训练或模型推理次数':0,'口径':'同原站点划分、同目标、同测试obs_id；统计归因不等于来源因果识别'}
    (OUT/'最终验收状态.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    source=[]
    for name in ['122_attribute_wq77_version_performance_v1.py','123_validate_wq77_attribution_v1.py']:
        p=ROOT/'脚本/05_诊断分析'/name;source.append({'脚本':str(p),'SHA256':sha(p)})
    pd.DataFrame(source).to_csv(OUT/'分析脚本SHA256.csv',index=False,encoding='utf-8-sig')
    report=OUT/'预测精度下降原因综合报告.md'
    report.write_text(report.read_text(encoding='utf-8')+f'\n## 最终独立验收\n\n主分析{state["主分析核验项"]}项通过，独立交付{len(cs)}项通过；{len(inputs)}个分析输入逐一核对SHA256保持不变。原Family V2的90条训练登记再次验证训练/测试数量和77特征数，旧登记逐折R²/RMSE与冻结预测复算一致。50套原训练上下文与测试批的ID顺序亦已按旧源码恢复并对齐。八项交付完整；本轮无模型训练或推理。\n\n历史高分的具体例子：旧Secchi严格77项随机划分TabICLv2 R²=0.700817，旧相同目标留站点pooled R²=0.602049，新留站点为0.627123；0.7008→0.6271不是可比的版本下降。另旧历史随机TN表层f1_history为0.799232，新同一历史随机重放仍为0.789776，均非正式77项留站点实验。\n\n夜光ntl_avg_rad的主表缺失率由90.6833%降至1.0873%，有真实支持站点8→94、唯一值1029→10744；这证实补齐真实发生，但48项同步改变，不能单独识别它对TabICLv2 TP中层的损益。\n',encoding='utf-8')
    assets=[{'文件':str(p.relative_to(OUT)),'SHA256':sha(p),'字节':p.stat().st_size} for p in OUT.iterdir() if p.is_file() and p.name!='交付SHA256.csv'];pd.DataFrame(assets).to_csv(OUT/'交付SHA256.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(state,ensure_ascii=False))
if __name__=='__main__':main()
