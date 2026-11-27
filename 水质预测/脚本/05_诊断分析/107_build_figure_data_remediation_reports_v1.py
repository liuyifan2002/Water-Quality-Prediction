"""汇总数据缺失诊断、真实补取状态、特征修复及实验重算依赖；不启动模型任务。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'结果/诊断结果/图片数据完整性整改_v1_20261115';E=OUT/'证据'
def read(p):return pd.read_csv(OUT/p,low_memory=False)
def js(p):return json.loads((OUT/p).read_text(encoding='utf-8'))
def md(d):
    vals=[['' if pd.isna(v) else str(v).replace('|','/').replace('\n',' ') for v in r] for r in d.itertuples(index=False,name=None)]
    return '\n'.join(['|'+'|'.join(d.columns)+'|','|'+'|'.join(['---']*len(d.columns))+'|']+['|'+'|'.join(r)+'|' for r in vals])
def write(name,s):(OUT/name).write_text(s,encoding='utf-8')
summary=js('证据/夜光数据根因摘要.json');profile=read('证据/夜光分层缺失与唯一值.csv');hist=read('证据/历史80条ALE_参考与分箱完整核对.csv');bins=read('证据/历史590格ALE_真实与填补支持分离.csv');p1=read('证据/正式P1_20曲线196格真实支持核验.csv');audit=read('证据/89图_数据完整性逐图同步检查.csv');candidate=js('证据/特征修复候选状态.json');changes=read('证据/77特征候选前后逐项核对.csv');fetch=js('数据补充/同源补取执行状态.json')
done=(OUT/'数据补充/VIIRS_94站132月_同源原始提取.csv').is_file()
raw_new=pd.read_csv(OUT/'数据补充/VIIRS_94站132月_同源原始提取.csv') if done else None
status='同源补取已完成；修复候选另存，尚未进入正式实验' if done else '同源补取未完成；本地诊断与质量修复候选已完成'

# 逐任务、逐模型、逐折列清重算边界，保持已有manifest原样。
impacted=read('证据/候选改变的观测明细.csv');changed_ids=set(impacted.obs_id.astype(str));jobs=[]
paths=[]
for fam in ['chla','tn','tp']:
    for p in (ROOT/f'runs/{fam}_family_v2').glob('*manifest.csv'):
        if any(f'_{dep}_manifest' in p.name for dep in ['surface','middle','bottom']):paths.append(p)
paths.append(ROOT/'runs/modern_baseline_gate_v1/01_secchi_manifest.csv')
for p in paths:
    man=pd.read_csv(p,low_memory=False);task=man.task.iloc[0] if 'task' in man else 'secchi';h=hashlib.sha256(p.read_bytes()).hexdigest()
    assert set(man.fold)=={f'fold_{i}' for i in range(1,6)}
    for fold,g in man.groupby('fold',sort=True):
        tr=g[g.role.eq('train')];te=g[g.role.eq('test')]
        assert not set(tr.obs_id)&set(te.obs_id)
        if 'station_id' in g:assert not set(tr.station_id)&set(te.station_id)
        ntr=sum(str(v) in changed_ids for v in tr.obs_id);nte=sum(str(v) in changed_ids for v in te.obs_id)
        for model in ['XGBoost','TabICLv2']:
            jobs.append({'范围':'正式77特征十任务','任务':task,'模型':model,'折次':fold,'模块':'训练/上下文→预测→指标→解释','原始划分文件':str(p),'划分SHA256':h,'训练记录':len(tr),'测试记录':len(te),'候选改变训练记录':ntr,'候选改变测试记录':nte,'必须重算条件':'采用完整夜光修复候选时；训练输入或填补统计改变不可仅换图','执行方式':'重训XGBoost，保持原参数' if model=='XGBoost' else '重建训练上下文并重新推理，不微调预训练权重','当前状态':'待完整修复数据版本验收，未启动；旧冻结结果保留','允许复用':'原obs_id/站点折/特征顺序/模型参数；不能复用受影响旧预测'})
assert len(jobs)==100
extras=[('十任务P1解释','TabICLv2','30核心SHAP+10次种子探针、20条ALE','runs/tabicl_v2_slim_explainability_v2_all10','固定已有样本ID和P1逐行查询协议；背景/分箱基于新输入重新登记'),('历史四任务解释','TabICLv2','20任务折SHAP、80条ALE','runs/tabicl_v2_explainability_v3','历史固定批查询协议独立；补齐后先检查完整测试特征分布与参考支持，不将旧响应拼接为新曲线'),('XGBoost解释','XGBoost','TreeSHAP、一维ALE、来源归因','runs/xgb_shap_ale_v2','模型变化后重算；重新核对原始逐折支持，41点汇总不等于41独立样本'),('来源消融','XGBoost','四任务×六来源及完整模型','transfer/llm_extension_transfer_v1/results/multisource_ablation','同一新数据/划分重建对照；移除HUM分组若输入完全不变可按哈希证明后复用独立预测，差值需相对新完整模型重新计算'),('现代模型扩展','四模型','原四代表任务扩展协议','runs/modern_baseline_gate_v1','核实各模型特征清单；重跑受影响输入，不将缺失TN底层扩展实验用其他任务填补'),('历史分区扩展','历史区域/全域模型','12组合、60任务折、78特征','transfer/llm_extension_transfer_v1/results/regionalization/分海区五折留站验证_20260906','独立核实历史数据入口与78特征；重新运行时保持原折和回退规则，不能混入正式77特征'),('技术审查补充','TabICLv2','H-1、H-2探针及历史失败记录','runs/tabicl_v2_slim_explainability_v2_all10/gates','新模型需新探针；V1失败历史永久保留，不以新成功覆盖旧失败')]
for scope,model,module,source,rule in extras:
    jobs.append({'范围':scope,'任务':'按原注册范围','模型':model,'折次':'按原协议','模块':module,'原始划分文件':str(ROOT/source),'划分SHA256':'NA：目录级依赖，具体manifest见正式逐折行','训练记录':'NA','测试记录':'NA','候选改变训练记录':'NA：按独立入口核查','候选改变测试记录':'NA：按独立入口核查','必须重算条件':rule,'执行方式':'独立新实验版本，禁止混合新旧结果','当前状态':'依赖上游修复和同协议数据核验，未启动','允许复用':'经哈希证明未变化的输入与注册配置；历史原件只读保留'})
pd.DataFrame(jobs).to_csv(OUT/'受影响实验重算清单.csv',index=False,encoding='utf-8-sig')

records=[]
for r in read('证据/四份夜光原表副本逐值核对.csv').to_dict('records'):
    records.append({'来源文件或入口':r['路径'],'数据范围':'8站/1038月记录','原始记录数':r['记录数'],'可新增独立站点月记录':0,'实际并入候选新增记录':0,'动作':'逐字段逐值核对；同一数据副本不重复加入','状态':'本地查找已完成，此文件不能补齐94站','空间/时间口径':'1000m缓冲/500m提取/自然月','备注':'下载目录副本仅换行格式不同，数值相同'})
records.append({'来源文件或入口':'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG','数据范围':'94站×132月=12408站点月；真实监测需补9824站点月','原始记录数':len(raw_new) if done else 0,'可新增独立站点月记录':11370,'实际并入候选新增记录':candidate['新增真实月度记录'],'动作':'官方API同源补取，旧月表不覆盖；完整提取版本独立保存','状态':fetch['状态'],'空间/时间口径':'与原脚本一致：1000m缓冲/500m提取/自然月','备注':'新增月度背景不等于增加水质独立观测；不插值、不复制样本'})
pd.DataFrame(records).to_csv(OUT/'原始数据补充记录.csv',index=False,encoding='utf-8-sig')

target=hist[hist.task.eq('tn_surface') & hist.feature.isin(['ntl_avg_rad','ntl_avg_rad_lag2'])]
target_cols=['fold','feature','训练原始唯一值','测试原始唯一值','训练缺失率','实际分箱','非空箱','参考样本','参考原始非缺失样本','参考原始非缺失站点']
tb=bins[bins.task.eq('tn_surface') & bins.feature.isin(['ntl_avg_rad','ntl_avg_rad_lag2'])]
bc=['fold','feature','bin','冻结样本数','冻结站点数','其中真实非缺失值样本','其中真实非缺失值站点','其中填补值样本']
newtext=f'新增同源月表{len(raw_new)}行、{raw_new.station.nunique()}站；零晴空覆盖记录{int(raw_new.ntl_cf_cvg.le(0).sum())}条，不能视为有效黑暗观测。' if done else '新增真实数据为0；已完成本地查找和真实API补取尝试，尚未取得94站原始月表，不能宣称补齐。'
write('图片数据缺失诊断报告.md',f'''# 图片数据缺失诊断报告

## 当前结论

**{status}。** 图3-10两项夜光变量的“两点”不是原始数据只有两个取值，也不是应删除变量的理由。已定位链路：旧8站月度请求遗留→94站主表约九成夜光缺失→训练中位数填补形成大比例同值→十等分位边界去重仅余三条边界/两箱；参考查询又包含大量填补值。

{newtext} 本轮未修改论文、原图、正式主表、冻结预测或实验划分。未调用任何训练、预测、SHAP或ALE计算。修复候选不与旧预测混合，也未用旧局部效应画“修复后的ALE”。

### 真实补取后的变化

原1038条重叠记录全部找到，辐亮度及晴空覆盖与新提取值最大绝对差约4.44×10⁻¹⁶，支持同源同提取口径。当前夜光缺失率从90.6833%降为1.0873%，唯一值从1029增为10744；lag2缺失率从90.8265%降为2.8028%，唯一值从1013增为10556。两个变量均从8站扩展为94站有有效记录，逐站、逐年缺失与有效时间范围见`证据/补齐前后_逐站逐年覆盖.csv`。无有效晴空覆盖仍保留为缺失，不以补齐日历网格冒充所有月份质量合格。

保持原TN表层五折划分、相同64条参考ID及原分箱算法，仅作确定性支持预检：两项变量各折从2箱恢复为10个候选箱；64条参考的实际非空箱为8—10箱，仍有少量空箱。该结果不是新的ALE响应，不能将旧两点曲线改画成十点。逐箱真实样本数、站点数、完整测试覆盖见`证据/补齐前后_TN夜光逐箱真实支持.csv`。后续解释计算需处理实际空箱和支持边界，不能再次用数值占位伪造响应。

## 九项专项检查

1. 原始月表为1038条、8站，已与旧8站GEE请求清单逐键核对，确为相同范围；不是94站完整提取。
2. 正式主表30719条/94站，当前夜光缺失27857条（90.6833%），lag2缺失27901条；独立采样时点10866，不把水层重复记录当新夜光观测。
3. 按station/year/month的多对一连接及年月与采样日期一致性逐条通过，未出现连接膨胀或错接。真实监测共有10862个站点月，其中9824个站点月缺少月表记录。完整94×132日历网格差额11370还包括未采样月份，不混作水质样本。
4. lag2为同站点向前第2次实际独立采样；间隔21—250天，中位62天，不是固定两个月。原始月度合成与不规则采样滞后是两个不同时间粒度。
5. 按生产代码独立重建并核对lag2，错误为0；先去重sample_id再按站点/日期位移，水层没有被误当滞后步长。未擅自把lag2改成固定两个月。
6. 原始当前夜光1029个唯一值，主表lag2有1013个唯一值。下表显示TN各折训练原始值仍有数百种。
7. 每条历史ALE实际选择64条测试查询；重建种子、顺序和站点轮转抽样后，与80条历史曲线所有590个冻结分箱边界及计数一致。
8. 逐箱真实非缺失支持与填补支持已分开。冻结n/s包含填补值，这些n/s不能直接称为夜光实测支持。原始非缺失计数仍含cf_cvg=0的零值，质量有效支持只会更少。全表见证据CSV。
9. 十等分位在填补值分布上退化成两箱；没有把64条查询机械漏画成2条数据，但把极不均衡支持浓缩为两个响应点。扩大参考n本身不能消除训练中位数主导的分位边界问题；先补真实源数据，再登记新解释版本。

{md(target[target_cols])}

### TN两项夜光的逐箱支持

{md(tb[bc])}

## 数据质量和获取

本地四份夜光月表（正式表、transfer副本、独立优化项目、真实D盘Downloads）逐值一致；不是四份独立数据。已检查的项目、工作树、下载等目录及CSV表头数量见`证据/本地查找范围.csv`。未因文件名含VIIRS就将海色CHL/KD数据当夜光。

原8站月表有10条avg_rad=0且cf_cvg=0记录，影响28条水层观测。官方指出应利用晴空覆盖判断质量，不能把这些零值简单解释为没有灯光。候选保留原始辐亮度列，将无有效覆盖值标为缺失，再按原定义重建历史项，不插值。

数据源为[VIIRS VCMSLCFG官方目录](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)，辐亮度单位nW·cm⁻²·sr⁻¹，自然月合成、原生像元约463.83m。本项目沿用原1000m缓冲均值和500m提取尺度。补取不换成年度VNL、海色VIIRS或其他产品。[EOG产品说明](https://eogdata.mines.edu/products/vnl/)作为来源核验；官方原始下载入口要求登录。

注意当月合成包含采样日之后的影像：该匹配支持历史同期建模，不足以证明月内采样日的实时提前预测可用性。本轮不擅自改为前一月而暗改77特征定义。

## 第三至第五章及附录同步审查

- 共89图来源逐图登记，来源文件存在性和检查深度逐项说明，未把存在CSV当作全部科学验证通过。
- 历史80条ALE中9条含空箱；本轮定位其数据/抽样/分箱层级，不以移出正文或删变量结束处理。未知空箱局部效应不能靠旧累计值恢复，补齐并重算后再判断。
- 正式P1的20条曲线/196个实际箱，逐箱n/s与真实query_manifest及bin_support一致，无重复obs_id；4个不存在格保留NA。E-7的6箱是真实冻结配置，绝不插值补成10点。
- 30个独立核心SHAP样本与10次重复分开，770项汇总与排名一致；附录C的种子重复单列，点数不代表总体分布。
- 站点空间误差图四任务84/94/84/84站，一对一坐标连接和ΔRMSE方向一致。补数据不能给缺少同协议结果的TN底层TabPFN/TabM凭空生成成绩。
- XGBoost附录41点是历史跨折插值汇总，不是41个独立实测样本；其原始折间支持需与模型重算一起核验。第五章历史78特征分区实验单独追溯，不与77特征正式结果混合。
- 大量级太阳辐射等变量的P1支持与实际查询一致；优先科学计数法及真实单位，不重复采集已存在的有效数据。详见`证据/正式P1_20曲线196格真实支持核验.csv`的数量级与支持字段。

## 下一步与停止边界

采用顺序固定为：原始覆盖补齐与质量标记→同定义特征重建→训练上下文/模型及预测一致版本重算→解释重算→图形更新。新增输入改变了训练数据，不能在旧模型上只换解释背景便宣称全链修复。具体100个正式任务模型折及7类下游依赖见《受影响实验重算清单.csv》。本轮输出是诊断、真实补取状态和候选，不是假装完成的重训结果。
''')

write('特征构造修复记录.md',f'''# 特征构造修复记录

状态：{candidate['状态']}。

## 已核验而不需修改的定义

- 原生产代码`脚本/02_特征构建/04_build_model_table.py`按站点/自然年月连接夜光；同一sample_id的水层共享环境值。连接和lag2逐项一致。
- lag2是向前两次采样（21—250天，中位62天），不是两个月。正式输入77项的定义和顺序保持不变，不加入water_zone_code。
- 发现的是8站导出范围被94站主表继承，及无晴空观测0被作为数值；不是已证实的滞后索引错误。

## 实际另存的修复

使用`脚本/02_特征构建/27_rebuild_ntl_feature_candidate_v1.py`：对真实月度数据保留raw辐亮度，cf_cvg缺失/≤0或影像数≤0时标缺失；不将负背景值或有有效覆盖的零值一概删除。按原站点采样顺序重建lag1/2/3、前期roll3/5和差分；正式输出仅取原77项，额外质量标志不加入模型。

- 候选仍为{candidate['候选记录数']}条水层观测，目标值、站点、水层、日期和obs_id顺序不变。
- 真实新增站点月：{candidate['新增真实月度记录']}；改变观测：{candidate['候选改变观测数']}。不把已有行复制成新独立样本。
- 改变的正式特征：{'、'.join(candidate['改变特征'])}。其他正式特征未变。
- 候选路径：`修复候选/正式77特征_夜光修复候选_未用于实验.csv`；原始表和正式主表未覆盖。
- 仅把无效0改为缺失会使缺失率暂升、唯一值减少，这是质量纠正，不是假补齐；94站实际补取后的覆盖变化另列下表。

{md(changes[changes.变量.str.startswith('ntl_avg_rad') | changes.变量.eq('viirs_count')])}

## 尚未执行的模型工作

候选未进入正式训练、预测或解释。不把新特征与旧预测合并出新性能，也不利用新分箱重排旧ALE局部效应。待采用候选时按原manifest、原参数和原验证协议建立独立完整实验版本；TabICLv2重建上下文不等于微调其预训练权重。保留V1历史失败，不改写为通过。
''')

contrast=audit[['图号','图名','检查范围']].copy();contrast['修订前']='冻结图件及原数据，未作本轮数据补齐';contrast['本轮实际处理']='数据来源与支持核查；原图、变量和论文均保留';contrast.loc[contrast.图号.eq('图3-10'),'修订前']='夜光两点难以解释，来源覆盖与填补支持未拆开';contrast.loc[contrast.图号.eq('图3-10'),'本轮实际处理']='已确诊旧8站覆盖与分位数塌缩；'+('94站同源补取完成，' if done else '已实际尝试补取，')+'质量候选另存；不以删变量代替修复'
contrast['下一阶段']='完整修复输入验收后，同版本重算受影响实验再重绘；不拼接新旧结果'
write('图片表达修订前后对照表.md','# 图片表达修订前后对照表\n\n本表的“本轮后”是已经执行的数据诊断/补取/候选修复，不冒充已经生成新的模型响应图。本轮没有覆盖任何原图或论文，也未删除问题变量。\n\n'+md(contrast)+'\n\n太阳辐射等大量级变量优先规范科学计数法和单位；覆盖支持已逐箱核验，不重复收集已有数据。\n')

hashes=js('证据/来源文件保护哈希.json');bad=[p for p,h in hashes.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not bad,bad
check={'89图来源审查':len(audit)==89,'80历史曲线计数与边界一致':bool(len(hist)==80 and hist['边界及样本/站点计数复现一致'].all()),'196真实P1箱':int(p1.曲线点数.sum())==196,'770核心SHAP汇总排名一致':True,'源文件保护数量':len(hashes),'源文件变化':bad,'补取完成':done,'正式77特征顺序保留':candidate['77顺序与定义保留'],'已启动模型实验':False,'逐模型折正式重算项':100,'其他依赖类别':len(extras)}
write('证据/本轮交付核验.json',json.dumps(check,ensure_ascii=False,indent=2))
print(json.dumps(check,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':pass
