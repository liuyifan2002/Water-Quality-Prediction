"""按实际文件写验收、补充任务和09执行状态报告；不改变实验结果。"""
from pathlib import Path
from datetime import datetime
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';PRIOR=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    coverage=pd.read_csv(RUN/'08_重算结果核验/重算执行覆盖.csv');acc=pd.read_csv(RUN/'02_48项差异核查及验收决定.csv');status=json.loads((RUN/'08_重算结果核验/验收状态.json').read_text(encoding='utf-8'));version=json.loads((RUN/'02_冻结候选/冻结清单.json').read_text(encoding='utf-8'))
    core=pd.read_csv(RUN/'03_正式十任务重算/运行登记.csv');old09=pd.read_csv(PRIOR/'09_受影响实验重审与重算清单.csv').fillna('NA');records=[]
    aliases={'来源消融':'来源消融','现代模型扩展':'现代模型','历史分区扩展':'历史分区','历史高分影像':'历史高分'}
    for _,r in old09.iterrows():
        row=r.to_dict();row['原清单生成时状态']=row.pop('当前状态');scope=r['范围']
        if scope=='正式77特征十任务':
            q=core[(core.task==r['任务'])&(core.model==str(r['模型']).lower())&(core.fold==r['折次'])];assert len(q)==1;v=q.iloc[0];row.update(本轮训练预测指标状态=v.status,本轮解释状态='未重新生成；旧解释不能代表新候选',新数据版本=v.dataset_version,本轮结果路径=v.prediction_file,本轮登记路径=str(RUN/'03_正式十任务重算/运行登记.csv'))
        elif scope in aliases:
            q=coverage[coverage['范围'].eq(aliases[scope])].iloc[0];row.update(本轮训练预测指标状态=q['状态'],本轮解释状态='本项不生成解释',新数据版本=version['dataset_version']+({'历史分区扩展':'_HIST78','历史高分影像':'_HIST_RANDOM_PATCH'}.get(scope,'')),本轮结果路径=q['登记路径'],本轮登记路径=q['登记路径'])
        elif scope in ['十任务P1解释','历史四任务解释','XGBoost解释','技术审查补充']:
            row.update(本轮训练预测指标状态='新正式模型及预测已另存，解释/探针独立依赖待执行',本轮解释状态='待按原注册协议重算；不得复用旧值',新数据版本=version['dataset_version'],本轮结果路径='NA：本阶段执行训练、预测、指标；未生成新SHAP/ALE/探针',本轮登记路径='NA')
        else:
            row.update(本轮训练预测指标状态='数据已补充或核实；尚不具备原协议等价直接重算条件',本轮解释状态='NA：未生成新实验结果',新数据版本='独立原始补充资产，未纳入正式77',本轮结果路径=str(RUN/'01_独立补充任务') if scope=='历史海色/OLCI' else str(PRIOR),本轮登记路径='NA：须明确历史实验真实入口与特征协议，不能伪造任务')
        records.append(row)
    pd.DataFrame(records).to_csv(RUN/'09_原重算清单逐项执行状态.csv',index=False,encoding='utf-8-sig')
    expanded=[];lineage=pd.read_csv(RUN/'08_重算结果核验/模型数据预测版本关联及哈希.csv')
    for scope,folder,registry in [('正式十任务','03_正式十任务重算','运行登记.csv'),('来源消融','04_来源消融重算','消融运行登记.csv'),('现代模型','05_现代模型重算','运行登记.csv'),('历史78项分区','06_历史78特征分区重算','运行登记.csv'),('历史高分随机','11_历史高分随机门控重算','运行登记.csv')]:
        reg=pd.read_csv(RUN/folder/registry).fillna('NA')
        for _,r in reg.iterrows():
            model=r.get('model','global+zone_with_fallback' if '分区' in scope else 'XGBoost_delete_'+str(r.get('source_group_removed','NA')))
            pred=r.get('prediction_file',str(RUN/folder/'predictions'/f'{r.task}_fold_{r.fold}.csv'))
            input_hash=r.get('effective_input_sha256',r.get('parent_input_sha256','NA'))
            if scope=='历史高分随机':
                q=lineage[lineage['范围'].eq('历史高分')&lineage['任务'].eq(r.task)]
                if len(q)==1:input_hash=q.iloc[0]['数据输入SHA256']
            expanded.append({'范围':scope,'task':r.task,'model':model,'fold':r.fold,'status':r.status,'dataset_version':r.dataset_version,'输入血缘':input_hash,'预测文件':pred,'错误':r.get('error','NA'),'对应登记':str(RUN/folder/registry)})
    pd.DataFrame(expanded).to_csv(RUN/'受影响实验重算清单_展开任务折.csv',index=False,encoding='utf-8-sig')
    supplement=[]
    for name in ['QUB','Landsat','CMEMS海色','CMEMS海陆掩膜']:
        folder=RUN/'01_独立补充任务'/name;state=json.loads((folder/'任务状态.json').read_text(encoding='utf-8'));source=folder/'原始来源清单.csv'
        supplement.append({'独立任务':name,'状态':state['状态'],'原始资产及结果目录':str(folder),'来源清单':str(source) if source.exists() else 'GEE产品/QA见34号脚本及任务状态','数据范围':{'QUB':'2013—2024小时原件；2014—2024共4018日','Landsat':'94站×132月=12408条','CMEMS海色':'3同源产品；2013-12-15至2024-12-31；94站10866采样点','CMEMS海陆掩膜':'94站×5原生产品=470匹配记录'}[name],'是否纳入正式77':'QUB及既有派生字段' if name=='QUB' else '不增加77字段；独立资产','未解决限制':'原生有效性不等于海岸线；湿格位移暂不接受' if name=='CMEMS海陆掩膜' else '原始真实质量缺失保留；不插值不复制'})
    pd.DataFrame(supplement).to_csv(RUN/'原始数据补充记录.csv',index=False,encoding='utf-8-sig')
    (RUN/'特征构造修复记录.md').write_text('# 特征构造修复记录\n\n- 夜光：前轮真实94站补取及cf_cvg质量掩膜继续使用；采样步lag2未改成2个月，32项环境滞后滚动独立验证。\n- QUB：HHOT01—24正确映射至当日01时/翌日00时；联合原HLT极值，使用完整日界支持，不把单极值日的零占位当真值；潮差及7项相关输入逐项登记。\n- ERA5：前3/7/14完整自然日，真实2013前史支撑早期窗口；SST异常4项按各折训练期重建，不用全期或测试期拟合气候态。\n- CMEMS：另存真实自然日均值及有效日数，修复伪日窗；反射率保留555/670 nm，不把多传感器别名当作独立OLCI；不把最深固定5727m空层当观测底温。此类字段未加进本版77项。\n- Landsat：仅获取当前94站脚本协议的真实月统计与QA；不使用历史双侧插值或复制月份。\n- 时序限制：完整当月气象、夜光和月雨保留既有定义，因此仅接受历史同期/空间泛化用途；ERA5与HKO日界不等同小时截止。严格提前预测仍需另立版本，不伪造实时可用性。\n\n冻结输入原值不再修改。具体差异、单位、来源、缺失及哈希见本目录逐项验收、正式77项定义总表和02冻结目录。\n',encoding='utf-8')
    summary=['# 候选数据验收与一致性重算报告','',f'报告生成时间：{datetime.now().astimezone().isoformat(timespec="seconds")}。',f'冻结版本：`{version["dataset_version"]}`。基础输入SHA256：`{version["input_sha256"]}`。','',
      '本轮按用户授权完成数据验收、真实补充、时间空间协议核验，并实际执行受影响的训练/训练上下文、预测及指标。全部输出隔离于扩展实验目录，旧论文、主表、冻结预测和解释原件保留。未将候选数据直接替换到论文或正式结果目录。','',
      '## 1. 48项输入的逐项验收','',
      '48项均有原值/候选值差异数、缺失率、唯一值、接受依据和限制，见`02_48项差异核查及验收决定.csv`。37项在真实补取及质量缺失保留后接受；11项在本轮修复后接受（7项QUB潮差/派生项、4项SST异常/交互）。48项中没有为了绘图删去变量。其余29项未改变，仍逐项登记。','',
      'QUB采用真实官方小时天文潮预报、日界及原精确极值，修复单极值日；不能称为实测水位。SST异常基础表固定审查参考期不直接用于拟合，所有实际模型按折训练期重建气候态。光学无晴空支持、近岸原生海陆掩膜及采样前史不足继续保留NA。','',
      '| 特征 | 原→候选改变行 | 原缺失率 | 候选缺失率 | 原→候选唯一值 | 决定 |','|---|---:|---:|---:|---:|---|']
    for _,r in acc.iterrows():summary.append(f'| {r.feature} | {int(r["最终相对原始改变数"])} | {r["原缺失率"]:.4%} | {r["验收缺失率"]:.4%} | {r["原唯一值"]}→{r["验收唯一值"]} | {r["判定"]} |')
    summary+=['','## 2. 四项独立真实补充任务','',
      '- **QUB**：12份2013—2024年官方HHOT文件，支撑2014—2024的4018日，每日25个小时/日界数值并保留原HLT精确极值。表中HHOT的01—24为当日01时至翌日00时，已按此解析。所有日期应用同一完整支持定义，非仅修改59个原零值日。','- **Landsat**：按项目当前94站GEE脚本，完成94站×132月=12408条真实月记录；LC08/LC09 C2 L2、150m缓冲、30m尺度、原QA及反射率过滤。真实无效像元保留缺失，没有重复采样或插值。该资产在77项之外，不擅自新增模型输入。','- **CMEMS海色**：补取3个原同源产品的2013-12-15至2024-12-31每日场，版本202603；所有旧重叠原始格点数值一致。94站原坐标最近原生格点，真实3/7/14自然日窗及有效日数；保存555/670 nm真实波长，不再以OLCI别名重复计算来源。','- **CMEMS海陆支持**：470条站点×产品匹配检查，保存原生数据有效性掩膜、原位置距离和最近稳定湿格候选；此掩膜不是精细海岸线。缺少岸线连通性证明，湿格位移暂不纳入；没有把远处海点当作缺失站点。','',
      '四任务各自保存原始文件、来源清单、SHA256和状态。补充成功不等于实验提升，也不等于无缺失。独立覆盖画像见`07_时间协议与空间核验/77之外真实补充覆盖.csv`。','',
      '## 3. 特征构造及时间协议','',
      '独立重构核对32项采样滞后/滚动、12项ERA5真实自然日窗、8个海色变量的3类日窗，核对94站真实坐标、观测/目标与冻结文件哈希；见07目录。lag2是前两次真实采样而非两个月；roll3先shift(1)、取前3次采样且min_periods=1；delta1=lag1−lag2。days_since_prev_sample保持同站同水层定义。','',
      '同月气象/夜光、整月降雨和日历日潮汐含同期信息，不能宣称实时提前预测。本版限定历史同期与空间留站点泛化，未制造新的年份外推实验。ERA5沿用UTC日期，HKO潮汐/雨量是香港日历日，观测仅有日期时不能保证统一小时级因果截止。未来提前预测需要单独按时区、采样时刻及数据可发布时间建立严格因果截断输入版本。','',
      '## 4. 实际训练、预测和指标执行','', '| 范围 | 计划任务折 | 已成功 | 失败 | 状态 |','|---|---:|---:|---:|---|']
    for _,r in coverage.iterrows():summary.append(f'| {r["范围"]} | {r["计划任务折"]} | {r["OK任务折"]} | {r["FAILED任务折"]} | {r["状态"]} |')
    current=pd.read_csv(RUN/'08_重算结果核验/新候选_pooled与逐折指标.csv');current=current[current['范围'].eq('正式十任务')].pivot(index='task',columns='model',values='pooled_R2');wins=pd.read_csv(RUN/'08_重算结果核验/十任务站点改善覆盖.csv').set_index('task')
    names={'chla_surface':'Chl-a表层','chla_middle':'Chl-a中层','chla_bottom':'Chl-a底层','secchi':'Secchi','tn_surface':'TN表层','tn_middle':'TN中层','tn_bottom':'TN底层','tp_surface':'TP表层','tp_middle':'TP中层','tp_bottom':'TP底层'}
    summary+=['','### 新候选十任务结果（同原站点折）','','| 任务 | XGBoost pooled R² | TabICLv2 pooled R² | ΔR²（TabICLv2−XGBoost） | RMSE改善站点比例 |','|---|---:|---:|---:|---:|']
    for task,name in names.items():
        if task in current.index:
            q=current.loc[task];summary.append(f'| {name} | {q.xgboost:.4f} | {q.tabiclv2:.4f} | {q.tabiclv2-q.xgboost:+.4f} | {wins.loc[task,"station_RMSE_win_rate"]:.2%} |')
    summary+=['','**结论更新边界**：数据质量补齐不保证每个模型或任务指标提高。若TN中层TabICLv2的pooled R²低于XGBoost，应明确保留，不得继续声称十任务全部领先；站点胜率高也不等于pooled指标必然领先。新旧版本逐项指标差异另存08目录，不改写旧结果。','']
    summary+=['',
      '任务折不是模型拟合次数：一个历史分区任务折含全域模型和多个满足门限的区域模型，其他区域使用全域回退；不能按水区数量虚报模型数。历史分区严格保留78项、water_zone_code、12组合、原站点折及300训练行/3训练站回退门限，单独版本后缀HIST78。历史Secchi三层是原表不同depth行上的同类透明度目标，不能包装成三种独立水层透明度实测或正式三项新任务。现代模型保留原注册Chl-a表层、Secchi、TN表层、TP中层，未用TN底层顶替。','',
      'XGBoost与TabICLv2正式对比消费同一77列、同一有效输入NPZ哈希，填补仅在训练样本拟合；TabICLv2仅重建上下文，不微调预训练权重。TabPFN/TabM按原模型处理器消费同一未填补原始有效特征，TabM按原内部验证选轮数后重训。原参数配置JSON中关于历史随机划分的描述不用于生成本轮划分，所有实际测试均读取原留站点manifest。FamilyV2九分层任务仍为原注册84站，Secchi为94站；原始补齐覆盖94站不等于擅自扩大原模型折次的站点集合。','',
      '折外指标同时保存pooled_R2与fold_R2_mean，二者不互换；不同目标原始RMSE不混比。不以改进与否作为运行覆盖的通过门槛；负R²和退化结果均保留。核心新预测嵌入数据版本、有效输入哈希和原manifest血缘，历史重放以独立协议、输入文件及关联哈希表追溯，未拼入旧冻结预测。','',
      '历史高分门控补充：已找到原完整训练OOF和测试观测ID，直接恢复原外层顺序，另存原4折内部OOF。它包含water_zone_code、4项目标历史和双侧影像匹配，属于随机历史探索，不是正式77项留站点或未来预报。采用原脚本可追溯默认模型参数冻结新执行；未找到旧运行覆盖参数的独立快照，不能将新旧指标差异全部归因为数据修复。每项新专家和门控对照内部使用同一新数据及参数。','',
      f'独立结果核验当前共{status["核验项"]}项，失败/未完成门槛{status["失败"]}项；详见08目录逐项CSV。','',
      '## 5. 09清单边界与后续依赖','',
      '原110行清单包含100个具体模型任务折和10项目录级依赖。已将来源消融、现代模型和历史分区展开为120、40和60个新任务折，另有12项历史高分随机任务；逐行状态见`09_原重算清单逐项执行状态.csv`，展开执行见`受影响实验重算清单_展开任务折.csv`。不把一行目录依赖伪装成一次模型运行。','',
      '本阶段用户要求的训练、预测、指标与解释计算分开验收。新数据改变后，旧30核心SHAP、历史SHAP/ALE、XGBoost解释及重复探针不能代替新模型解释；本轮没有生成这些解释，不宣称09全部科学依赖已关闭。历史V1失败记录永久保留。','',
      '历史海色/OLCI和长水质尚不具备原实验等价直接重算条件：海色别名/伪日窗及错误波长命名修复后，不能仍把这些数据视作真实独立OLCI；39年真实年度水质已补，但旧240条DO/BOD与Chl-a/Secchi/TN/TP并非同一目标，不能自动扩展正式2014—2024训练期。高分随机历史实验已独立重跑，仍未在严格提前预测协议下验证。真实数据完整性与预测增益分别审查，不因过去无效果而删除真实资产。','',
      '## 6. 交付入口','',
      '- 逐项验收：01/02/03号CSV。','- 四补充任务：`01_独立补充任务/`。','- 冻结数据/来源哈希/缺失/协议：`02_冻结候选/`。','- 完整单位、定义、真实来源及可用时间：根目录`正式77项特征定义与来源总表.csv`及`冻结版本补充元数据.json`，未改冻结值。','- 新训练预测与模型/指标：03—06目录，历史高分为11目录。','- 时间空间独立核验：07目录。','- 全部预测哈希、逐项指标复核、逐站点误差：08目录。','- 09原清单逐行执行状态及待办：根目录09/10文件。','',
      '脚本按数据获取/特征、训练、诊断拆分。没有覆盖旧论文、正式主表、原训练结果或原图。原件哈希检查见08目录。']
    (RUN/'00_候选数据验收与一致性重算报告.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    (RUN/'10_待执行与暂不纳入项目.md').write_text('# 待执行与暂不纳入项目\n\n'+
      '## 数据层暂不纳入\n\n- 缺乏精细岸线/连通性约束的湿格位移，不改真实站点位置。\n- 把CMEMS多传感器别名当作独立OLCI来源，以及555/670 nm改名560/665 nm，不接受。\n- 不将光学结构性缺失、无清晰像元或样本前史不足填成真实观测。\n- 旧高分/Landsat双侧插值或未来日期匹配，不直接进入提前预测版本；新月表保留真实原值。\n\n'+
      '## 模型和论文下游\n\n- 新模型对应30核心SHAP、10种子探针和20条正式ALE须按原协议独立重算；旧值仅为旧版本历史结果。\n- 历史五折解释、TreeSHAP/ALE及H类探针需另存新版本，不覆盖V1失败事件。\n- 77之外新增来源的性能检验必须明确新特征清单、可获得时间及原划分；数据补齐本身不是增益结论。\n- 未将新指标或解释替换到论文；需要以本轮结果全面核对原结论后再整合。\n\n'+
      '## 运行失败与覆盖\n\n'+coverage.to_csv(index=False)+'\n详细失败见各运行登记error字段；不得将失败/待执行记为零值或已经完成。\n',encoding='utf-8')
    # 交付资产清单以相对路径与真实哈希定位，不包含凭据。
    assets=[]
    for p in RUN.rglob('*'):
        if p.is_file() and p.name!='交付文件SHA256.csv':assets.append({'相对路径':str(p.relative_to(RUN)),'字节':p.stat().st_size,'SHA256':sha(p)})
    pd.DataFrame(assets).to_csv(RUN/'交付文件SHA256.csv',index=False,encoding='utf-8-sig')
    print('REPORT_READY',len(assets),flush=True)
if __name__=='__main__':main()
