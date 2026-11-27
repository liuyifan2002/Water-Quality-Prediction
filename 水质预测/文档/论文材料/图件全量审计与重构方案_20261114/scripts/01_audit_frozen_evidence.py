"""只读论文、冻结数据与脚本，审计图件可行性；不导入或运行模型代码。"""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
import fitz
from docx import Document

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parents[1]
E=OUT/'evidence'
manifest={};schemas=[];feas=[]
def track(p):
    p=Path(p)
    manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
def read(rel):
    p=ROOT/rel;track(p);d=pd.read_csv(p)
    schemas.append({'文件':rel,'记录数':len(d),'字段':';'.join(d.columns),'SHA256':manifest[str(p.relative_to(ROOT))]})
    return d
def evidence(num,part,rel,d,keys,status,detail):
    feas.append({'候选图号':num,'证据部分':part,'真实数据文件':rel,'字段':';'.join(keys),'记录数量':len(d),'关键字段缺失单元':int(d[keys].isna().sum().sum()),'完整性':detail,'可绘制程度':status})

rows=json.loads((E/'当前图件结构与来源.json').read_text(encoding='utf-8'))
for x in rows:track(ROOT/x['当前文件路径'])
for v in ['figure_revision_v1','figure_revision_v2']:
    for p in (ROOT/'结果/论文图表'/v).rglob('*'):
        if p.is_file():track(p)
docrel='文档/论文材料/硕士论文V4_题名更正_20261109/水质预测硕士学位论文_V4_题名更正'
for base in [docrel,'文档/论文材料/硕士论文V4_结构与地图定向修订_20261108/水质预测硕士学位论文_V4_结构与地图定向修订']:
    for ext in ['.docx','.pdf']:track(ROOT/(base+ext))
doc=Document(ROOT/(docrel+'.docx'));pdf=fitz.open(ROOT/(docrel+'.pdf'))
page_texts=[p.get_text().replace(' ','').replace('\n','') for p in pdf]
for x in rows:
    key=(x['图号']+x['图名']).replace(' ','')
    pages=[i+1 for i,t in enumerate(page_texts) if i>20 and key in t]
    x['V4_PDF物理页']=pages
    # 原Word缩放是现状；改绘资产按其15.5cm画布另列，不能称已嵌入Word。
    x['当前审查说明']='改绘独立资产尚未入Word' if x['当前改绘优先'] else '与Word内嵌SHA256匹配'
    p=ROOT/x['当前文件路径'];sv=p.with_suffix('.svg')
    if sv.exists() and '源PDF最小字号pt' not in x:
        text=sv.read_text(encoding='utf-8');sizes=re.findall(r'font:\s*([\d.]+)px',text)
        x['SVG文字字号可恢复']=bool(sizes)
    # 段落引用只计Normal正文，排除目录、图题与附录自述。
    x['明确正文引用']=[{'索引':i,'文字':p.text} for i,p in enumerate(doc.paragraphs) if p.style.name=='Normal' and 350<=i<550 and '\t' not in p.text and x['图号'] in re.findall(r'图[0-9A-Z]+-\d+',p.text)]
(E/'当前图件结构与来源.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')

coordrel='数据/处理后数据/正式主表/station_meta.csv';coord=read(coordrel)
assert len(coord)==94 and coord.Station.is_unique and not coord[['lat','lon','water_zone_en']].isna().any().any()
evidence('图4-5','真实坐标与海区',coordrel,coord,['Station','lat','lon','water_zone_en'],'可绘制','94个唯一站点，坐标及海区均完整；本轮不改研究区地图')
stationparts=[]
for family in ['chla','tn','tp']:
    rel=f'runs/{family}_family_v2/09_{family}_family_v2_station_metrics.csv';d=read(rel)
    assert len(d)==252 and d.groupby('task').station_id.nunique().eq(84).all()
    z=d.merge(coord,left_on='station_id',right_on='Station',how='left',validate='many_to_one')
    assert not z[['lat','lon','water_zone_en']].isna().any().any()
    stationparts.append(d)
    evidence('图4-5',family+'正式逐站点指标',rel,d,['task','station_id','n_obs','XGB_RMSE','TabICL_RMSE','delta_RMSE'],'可绘制','三水层各84站；坐标海区全匹配；差值=TabICLv2−XGBoost；不重算RMSE')
pilotrel='runs/tabiclv2_pilot/24_station_level_comparison.csv';pilot=read(pilotrel);sec=pilot[pilot.task.eq('secchi')].copy()
sx=read('transfer/llm_extension_transfer_v1/predictions/core_10_tasks/secchi_xgb_frozen_for_gate.csv')
st=read('transfer/llm_extension_transfer_v1/predictions/core_10_tasks/secchi_tabiclv2_recovered.csv')
pp=read('runs/tabiclv2_pilot/09_secchi_predictions.csv')
secverify=[]
for model,frozen in [('xgb',sx),('tabicl',st)]:
    q=pp[(pp.validation=='station')&(pp.model==model)]
    j=frozen.merge(q,on=['obs_id','fold'],suffixes=('_frozen','_pilot'),validate='one_to_one')
    assert len(j)==len(frozen)==len(q)==10866
    same=np.allclose(j.y_pred_frozen,j.y_pred_pilot,rtol=0,atol=1e-12) and np.allclose(j.y_true_frozen,j.y_true_pilot,rtol=0,atol=1e-12)
    assert same
    secverify.append({'model':model,'frozen_rows':len(frozen),'pilot_rows':len(q),'同obs_id及fold预测标签一致':bool(same)})
assert len(sec)==94 and sec.station_id.nunique()==94 and (sec.xgb_n_obs==sec.tabicl_n_obs).all()
evidence('图4-5','Secchi逐站点冻结指标',pilotrel,sec,['station_id','xgb_RMSE','tabicl_RMSE','delta_RMSE','xgb_n_obs'],'可绘制','筛出94站；已将pilot站点预测与正式Secchi冻结预测逐obs_id/fold核验一致，两模型各10866行')
pd.DataFrame(secverify).to_csv(E/'Secchi正式来源一致性.csv',index=False,encoding='utf-8-sig')
allst=pd.concat(stationparts,ignore_index=True)
auditcounts=allst.groupby('task').agg(站点数=('station_id','nunique'),OOF数=('n_obs','sum')).reset_index()
auditcounts=pd.concat([auditcounts,pd.DataFrame([{'task':'secchi','站点数':94,'OOF数':int(sec.xgb_n_obs.sum())}])],ignore_index=True)
auditcounts.to_csv(E/'十任务逐站点覆盖核查.csv',index=False,encoding='utf-8-sig')

base='transfer/llm_extension_transfer_v1/results/regionalization/分海区五折留站验证_20260906/'
reg=read(base+'五折留站点验证_汇总.csv');fm=read(base+'五折留站点验证_各折指标.csv');pred=read(base+'五折留站点验证_预测.csv');spl=read(base+'五折站点划分清单.csv');dec=read(base+'分区训练决策.csv');feat=read(base+'严格验证特征清单.csv')
assert len(reg)==12 and reg.folds.eq(5).all() and len(fm)==120
assert fm.groupby(['target','depth','fold']).size().eq(2).all()
assert fm.groupby(['target','depth','fold']).rows.nunique().eq(1).all()
assert not pred[['observed','prediction_global_external','prediction_zone_aware_external']].isna().any().any()
assert not pred.duplicated(['target','depth','obs_id']).any()
j=pred.merge(spl,on='station',suffixes=('','_split'),validate='many_to_one')
assert (j.fold==j.fold_split).all()
reginfo={'任务数':len(reg),'折次指标行':len(fm),'预测行':len(pred),'站点数':len(spl),'分区训练决策':dec.decision.value_counts().to_dict(),'最小区域模型训练行':int(dec[dec.decision=='zone_model'].train_rows.min()),'最小区域模型训练站点':int(dec[dec.decision=='zone_model'].train_stations.min()),'逐任务特征数':{a.target:len(json.loads(a.features_json)) for a in feat.itertuples()}}
strict=read('runs/tabiclv2_pilot/01_strict_feature_manifest.csv');strictset=set(strict[strict.included_in_pilot].feature)
reginfo['相对77特征新增']=list(set(json.loads(feat.features_json.iloc[0]))-strictset)
for part,rel,d,keys in [('五折任务汇总',base+'五折留站点验证_汇总.csv',reg,['target','depth','folds','positive_folds','delta_r2_pooled']),('成对逐折指标',base+'五折留站点验证_各折指标.csv',fm,['target','depth','fold','model_variant','rows','feature_count','r2']),('配对冻结预测与折次',base+'五折留站点验证_预测.csv',pred,['target','depth','fold','station','obs_id','observed','prediction_global_external','prediction_zone_aware_external']),('区域样本与回退',base+'分区训练决策.csv',dec,['target','depth','fold','water_zone_en','train_rows','test_rows','train_stations','decision']),('相同特征集合',base+'严格验证特征清单.csv',feat,['target','features_json'])]:
    evidence('图5-2',part,rel,d,keys,'可绘制（历史扩展独立展示）','12目标×水层，60任务折、双模型120指标；同一行配对预测和同一站点折次。78特征含water_zone_code，不是正式77特征Family V2实验；Secchi三水层不能冒称正式单任务')
reg.to_csv(E/'分区十二任务现有结果索引.csv',index=False,encoding='utf-8-sig')
(E/'分区可比性核验.json').write_text(json.dumps(reginfo,ensure_ascii=False,indent=2),encoding='utf-8')
case_rel='文档/论文材料/硕士论文V2_学术内容补强_20261101/站点误差与特征覆盖补充核对.csv';cases=read(case_rel)
assert len(cases)==5 and set(cases.station_id)=={'VT11','SM18','SM19','PT3'}
for a in cases.itertuples():
    v=allst[(allst.task==a.task)&(allst.station_id==a.station_id)].iloc[0]
    assert v.n_obs==a.n_oof and np.allclose([v.XGB_RMSE,v.TabICL_RMSE,v.delta_RMSE],[a.XGB_RMSE,a.TabICL_RMSE,a.delta_RMSE],rtol=0,atol=1e-12)
evidence('图5-3','四站五案例冻结诊断表',case_rel,cases,['task','station_id','n_oof','XGB_RMSE','TabICL_RMSE','missing_77_share_in_main_surface','station_date_n_in_main_surface','lat','lon'],'可绘制（诊断线索）','5个站点×任务案例、4个唯一站点；全部误差及n与Family V2站点指标一致。缺失率源自表层主表，不代表SM18中层输入缺失率')
mainrel='数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv';track(ROOT/mainrel)
main=pd.read_csv(ROOT/mainrel,usecols=['station','sample_id','depth','lat','lon']+list(strictset))
surf=main[main.depth=='surface'].drop_duplicates('sample_id')
miss=surf.groupby('station')[list(strictset)].apply(lambda a:float(a.isna().mean().mean()))
assert len(miss)==94
for a in cases.itertuples():assert np.isclose(miss.loc[a.station_id],a.missing_77_share_in_main_surface)
coverage={'表层主表行':len(surf),'站点数':len(miss),'77特征缺失率_站点中位数':float(miss.median()),'VT11缺失率':float(miss.loc['VT11']),'VT11表层记录':int((surf.station=='VT11').sum())}
(E/'缺失率口径核验.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf-8')
evidence('图5-3','原始77特征覆盖核对',mainrel,surf,['station','sample_id','depth'],'可绘制（表层背景覆盖）','94站；未插补缺失率核对通过；不将表层71条主表记录与67条TN/TP OOF误认为同一分母')

ale=read('runs/tabicl_v2_explainability_v3/ale_1d/ale_1d_by_fold.csv')
binrows=[]
for (t,f),g in ale.groupby(['task','feature'],sort=False):
    folds=[z for _,z in g.groupby('fold')]
    same=all(len(z)==len(folds[0]) and np.allclose(z[['lower','upper']],folds[0][['lower','upper']]) for z in folds)
    # 原脚本使用max(count,1)作中心化权重；只核验保存曲线中心，不重新计算ALE。
    binrows.append({'task':t,'feature':f,'rows':len(g),'folds':g.fold.nunique(),'bins_by_fold':';'.join(f'{k}:{len(z)}' for k,z in g.groupby('fold')),'相同分箱边界':same,'零支持分箱':int(g.n_samples.eq(0).sum()),'最小样本':int(g.n_samples.min()),'最小站点':int(g.n_stations.min()),'共同范围下界':g.groupby('fold').lower.min().max(),'共同范围上界':g.groupby('fold').upper.max().min(),'原权重中心偏差_max':max(abs(np.average(z.ale,weights=np.maximum(z.n_samples,1))) for z in folds)})
pd.DataFrame(binrows).to_csv(E/'历史ALE逐特征分箱核查.csv',index=False,encoding='utf-8-sig')
p1rows=[]
for p in sorted((ROOT/'runs/tabicl_v2_slim_explainability_v2_all10/ale_1d').glob('*_curve.csv')):
    track(p);d=pd.read_csv(p);a=d.iloc[0]
    p1rows.append({'task':a.task,'feature':a.feature,'实际分箱':len(d),'总分箱样本记录':int(d.n_samples.sum()),'最小样本':int(d.n_samples.min()),'最小独立站点':int(d.n_stations.min()),'最大独立站点':int(d.n_stations.max()),'末端ALE':float(d.ale.iloc[-1]),'最大重复局部差值':float(d.repeatability_max_local_delta.max()),'源文件':str(p.relative_to(ROOT))})
pd.DataFrame(p1rows).to_csv(E/'P1二十条ALE覆盖核查.csv',index=False,encoding='utf-8-sig')
prows=[]
for p in sorted((ROOT/'runs/tabicl_v2_slim_explainability_v2_all10/gates').glob('*_p1_probes.csv')):
    track(p);d=pd.read_csv(p)
    assert len(d)==9 and d.obs_id.nunique()==3 and d.groupby('obs_id').size().eq(3).all()
    for obs,g in d.groupby('obs_id'):
        prows.append({'task':g.task.iloc[0],'obs_id':obs,'记录数':len(g),'分位':g.prediction_quantile.iloc[0],'同query最小值':g.p1_prediction.min(),'同query最大值':g.p1_prediction.max(),'同query极差':g.p1_prediction.max()-g.p1_prediction.min(),'task混合三query极差_非重复波动':d.p1_prediction.max()-d.p1_prediction.min()})
pd.DataFrame(prows).to_csv(E/'图3-16同查询重复核查.csv',index=False,encoding='utf-8-sig')
for rel in ['runs/tabicl_v2_explainability_closure_v1/01_shap_sample_representativeness.csv','runs/tabicl_v2_explainability_closure_v1/03_shap_importance_by_fold.csv','runs/tabicl_v2_explainability_closure_v1/04_shap_importance_summary.csv','runs/tabicl_v2_explainability_closure_v1/05_shap_rank_stability.csv','runs/tabicl_v2_explainability_closure_v1/06_ale_1d_completeness_and_stability.csv','runs/tabicl_v2_explainability_v3/00_inference_reproduction.csv','runs/tabicl_v2_explainability_v3/00_ale_inference_reproduction.csv','runs/tabiclv2_pilot/20_tabiclv2_pilot_summary.csv','runs/xgb_shap_ale_v2/01_shap/01_shap_importance_summary.csv','runs/xgb_shap_ale_v2/02_ale_1d/02_ale_1d_selected_curves.csv','runs/xgb_shap_ale_v2/02_ale_1d/02_ale_1d_fold_consistency.csv','runs/xgb_shap_ale_v2/03_ale_2d/03_ale_2d_coverage.csv','runs/xgb_shap_ale_v2/04_integrated/04_shap_ale_ablation_comparison.csv','transfer/llm_extension_transfer_v1/results/modern_baseline_gate_v1/modern_baseline_core_metrics.csv','transfer/llm_extension_transfer_v1/results/final_experiment_closure_audit_v1/final_core_results_table_v2.csv']:
    read(rel)
for rel in ['脚本/06_结果汇总与绘图/72_finalize_tabiclv2_explanation_closure_v1.py','脚本/05_诊断分析/69_run_tabiclv2_ale_v3.py','脚本/05_诊断分析/58_compute_xgb_shap_ale_v2.py','脚本/05_诊断分析/59_integrate_plot_xgb_shap_ale_v2.py','脚本/04_模型训练/30_cross_validate_zone_model_spatial.py','文档/论文材料/硕士论文V2_学术内容补强_20261101/refine_figures_v2.py','文档/论文材料/硕士论文V2_学术内容补强_20261101/polish_v2.py','脚本/06_结果汇总与绘图/40_build_report_support_figures.py']:
    track(ROOT/rel)
for p,h in manifest.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
pd.DataFrame([{'路径':p,'SHA256':h} for p,h in manifest.items()]).to_csv(E/'只读来源保护清单.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(schemas).drop_duplicates('文件').to_csv(E/'实际数据字段与行数.csv',index=False,encoding='utf-8-sig')
feas.extend([
    {'候选图号':'图5-2','证据部分':'直接等同正式77特征分区结果','真实数据文件':base+'严格验证特征清单.csv','字段':'target;features_json','记录数量':4,'关键字段缺失单元':0,'完整性':'实际每目标78特征含water_zone_code；没有把它直接当正式77特征分区实验的依据','可绘制程度':'不可绘制为正式Family V2比较；可独立绘历史扩展'},
    {'候选图号':'图5-3','证据部分':'SM18中层/同折输入缺失与误差机制','真实数据文件':case_rel,'字段':'现有missing_77_share_in_main_surface；缺少中层/同折输入缺失字段','记录数量':5,'关键字段缺失单元':'NA：所需字段不存在，非数值缺失','完整性':'现表只支持表层站点背景覆盖，不支持SM18中层/同折输入缺失因果诊断；按最小综合图方案使用已有部分','可绘制程度':'不可绘制机制归因图；可绘表层背景＋误差案例图'}])
pd.DataFrame(feas).to_csv(OUT/'04_新增图件数据可行性审计.csv',index=False,encoding='utf-8-sig')
print(json.dumps({'图数':len(rows),'PDF页数':len(pdf),'实际数据表':len(schemas),'来源保护文件':len(manifest),'分区':reginfo,'缺失率':coverage},ensure_ascii=False))
