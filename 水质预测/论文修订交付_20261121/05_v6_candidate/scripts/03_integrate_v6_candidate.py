"""以V5.1为底稿生成WQ77候选稿；未核验解释图仅留明确占位。"""
from pathlib import Path
import re
import json
from collections import Counter
import pandas as pd
from docx import Document
from docx.table import Table
from docx.oxml.ns import qn
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.table import WD_ROW_HEIGHT_RULE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
SRC=ROOT/'论文修订交付_20261121/04_v5_1_correction/水质预测硕士学位论文_V5.1.docx'
DEST=OUT/'水质预测硕士学位论文_V6_候选稿_解释待核验.docx'
assert SRC.is_file() and not DEST.exists()
DATA=OUT/'data';FIG=OUT/'figures'
core=pd.read_csv(DATA/'正式十任务指标.csv').set_index('task')
modern=pd.read_csv(DATA/'四模型同协议指标.csv')
ab=pd.read_csv(DATA/'四任务六来源消融.csv')
regional=pd.read_csv(DATA/'历史78特征分区汇总.csv')
meta=pd.read_csv(ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115/正式77项特征定义与来源总表.csv').set_index('feature')
doc=Document(SRC)
assert len(doc.inline_shapes)==94 and len(doc.tables)==13
changes=[]

def note(loc,old,new,source='WQ77_REPAIRED_20261115_V1'):
    changes.append({'位置':loc,'修改前':old,'修改后':new,'依据':source})

def cap(no):
    p=[p for p in doc.paragraphs if p.style.name in ('图题','表题') and re.match(rf'^{re.escape(no)}(?:\s|$)',p.text)]
    assert len(p)==1,(no,len(p));return p[0]

def table(no):
    p=cap(no);n=p._p.getnext();assert n.tag==qn('w:tbl');return Table(n,doc._body)

def figpara(no):
    p=cap(no);n=p._p.getprevious();assert n.tag==qn('w:p') and n.xpath('.//w:drawing'),no
    from docx.text.paragraph import Paragraph
    return Paragraph(n,doc._body)

def picture(no,path,width=15.5):
    p=figpara(no);p.clear();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(path),width=Cm(width));note(no,'V5.1内嵌图',str(path.relative_to(ROOT)))

def put(p,new):
    old=p.text
    if old!=new:p.text=new;note('正文',old[:150],new[:150])

def setrow(row,values):
    assert len(row.cells)==len(values)
    for cell,value in zip(row.cells,values):cell.text=str(value)

def tableformat(t,pt=8):
    for i,row in enumerate(t.rows):
        row._tr.get_or_add_trPr().append(__import__('docx').oxml.OxmlElement('w:cantSplit'))
        for cell in row.cells:
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0)
                for run in p.runs:run.font.size=Pt(pt)

order=list(core.index)
for no,pattern in [('表4-1','baseline'),('表4-2','comparison')]:
    t=table(no);assert len(t.rows)==11
    for i,task in enumerate(order,1):
        r=core.loc[task]
        if no=='表4-1': vals=[r['任务'],f'{r.XGBoost_R2:.3f}',f'{r.XGBoost_RMSE:.4f}',f'{r.XGBoost_MAE:.4f}']
        else: vals=[r['任务'],f'{r.XGBoost_R2:.3f}',f'{r.TabICLv2_R2:.3f}',f'{r.delta_R2:+.3f}',f'{r.XGBoost_RMSE:.4f}',f'{r.TabICLv2_RMSE:.4f}',f'{r.XGBoost_MAE:.4f}',f'{r.TabICLv2_MAE:.4f}',f'{r.station_win_rate*100:.1f}%','待复核']
        setrow(t.rows[i],vals)
    tableformat(t,8 if no=='表4-2' else 9)
    if no=='表4-2':
        for row in t.rows:
            row.height=Cm(1.05);row.height_rule=WD_ROW_HEIGHT_RULE.AT_LEAST
    note(no,'V5.1旧版指标',f'新候选版{len(order)}项pooled OOF结果；稳健性列待新版本bootstrap核验')

t=table('表4-3');assert len(t.rows)==17
for i,r in enumerate(modern.itertuples(),1):
    name={'xgboost':'XGBoost','tabiclv2':'TabICLv2','tabpfn':'TabPFN','tabm':'TabM'}[r.model]
    setrow(t.rows[i],[r.任务,name,f'{r.pooled_R2:.3f}',f'{r.pooled_RMSE:.4f}',f'{r.pooled_MAE:.4f}'])
tableformat(t,9);note('表4-3','V5.1四模型结果','新候选版40/40任务折的四模型指标')

t=table('表5-1');assert len(t.rows)==25
setrow(t.rows[0],['任务','删除来源','ΔR²','ΔRMSE','删组后RMSE改善站点比例','结果方向'])
label=core['任务'].to_dict();groups=['MET','RHT','OCN','HUM','TMP','SPA'];tasks=['chla_surface','secchi','tn_surface','tp_middle'];i=1
for task in tasks:
    for group in groups:
        r=ab[(ab.task==task)&(ab['来源']==group)].iloc[0]
        sign='删组后下降' if r.delta_R2<0 else '删组后提高' if r.delta_R2>0 else '无变化'
        setrow(t.rows[i],[label[task],group,f'{r.delta_R2:+.3f}',f'{r.delta_RMSE:+.4f}',f'{r["删除后RMSE改善站点比例"]*100:.1f}%',sign]);i+=1
tableformat(t,7.7);note('表5-1','V5.1旧版消融','候选版120/120任务折及站点配对误差')

t=table('表2-3');assert len(t.rows)==78
for i in range(1,78):
    row=t.rows[i];feature=row.cells[2].text.strip();assert feature==meta.index[i-1],(i,feature,meta.index[i-1])
    m=meta.loc[feature]
    row.cells[4].text=str(m['来源组']);row.cells[6].text=str(m['单位'])
    if feature=='rain_month':row.cells[5].text='采样所在自然月整月累计；含同期信息'
    elif feature.startswith('ntl_') or feature=='viirs_count':row.cells[5].text=str(m['时间与可用性'])
tableformat(t,7.2);note('表2-3','V5.1来源、单位及rain_month等时间表述','候选版77项正式元数据')

figs={'图4-1':'图4-1_十任务总体提升与站点改善覆盖.png','图4-2':'图4-2_不同指标与水层的模型提升差异.png','图4-4':'图4-4_四项代表任务站点RMSE变化空间分布.png','图4-5':'图4-5_四项代表任务四模型比较.png','图5-1':'图5-1_历史全域与分区建模比较.png','图5-2':'图5-2_四项代表任务的多源环境信息消融比较.png','图5-3':'图5-3_四个典型站点输入覆盖与预测误差诊断.png'}
for no,name in figs.items():picture(no,FIG/name)

# 未通过跨重建数值核验的旧解释图不得进入新模型的正式结果。
plt.rcParams['font.sans-serif']=['Microsoft YaHei','Arial'];plt.rcParams['axes.unicode_minus']=False
placeholder=FIG/'解释图待数值核验_占位.png'
fig=plt.figure(figsize=(15.5/2.54,2.2/2.54));fig.text(.5,.58,'新版本解释结果待数值一致性核验',ha='center',va='center',fontsize=13,color='#555')
fig.text(.5,.26,'原版图像未作为WQ77_REPAIRED模型的解释结果使用',ha='center',va='center',fontsize=8,color='#777');fig.savefig(placeholder,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
pending=[f'图3-{i}' for i in list(range(3,8))+list(range(12,17))]
pending += [f'图{letter}-{i}' for letter,n in [('C',20),('D',12),('E',20),('F',10),('H',2)] for i in range(1,n+1)]
for no in pending:
    picture(no,placeholder)
    p=cap(no);put(p,p.text+'（新版本解释待核验）')
note('解释图','74张旧版模型解释图','74张显式待核验占位；不使用旧版数值',source='2026-11-20回传包独立P1差18/30；10/10批量严格复现失败')

# 结构占位的表3-1不能沿用旧模型SHAP贡献。
t=table('表3-1')
for row in t.rows[1:]:
    row.cells[2].text='待核验';row.cells[3].text='待核验';row.cells[4].text='3';row.cells[5].text='3'
note('表3-1','V5.1旧模型SHAP贡献','新版本解释结果待核验；保留预登记样本数')

def replace_start(prefix,new):
    found=[p for p in doc.paragraphs if p.style.name=='Normal' and p.text.startswith(prefix)]
    assert len(found)==1,(prefix,len(found))
    put(found[0],new)

replace_start('香港近岸水质具有空间、水层与时间差异','本文以2014—2024年香港近岸监测记录构建Chl-a、Secchi、TN与TP十项水质任务，以WQ77_REPAIRED_20261115_V1为唯一正式77特征版本，采用五折留站点预测比较XGBoost和TabICLv2。TabICLv2在九项任务的pooled OOF R²高于XGBoost，在TN中层略低（0.485对0.497）；Chl-a表层和Secchi分别达到0.592和0.627。四项代表任务扩展比较纳入TabPFN、TabM；正式XGBoost来源消融及历史78特征分区实验分别回答来源增量和空间异质性问题。当前预测、消融与分区结果已经冻结，但新模型SHAP/ALE仍未通过跨重建数值一致性核验，故本候选稿暂不据旧版解释图作结论。研究使用的rain_month为采样所在自然月整月累计，并含其他同期输入，结论限于历史同期预测与未见站点空间泛化，不代表实时提前预报。')
replace_start('Coastal water quality varies across locations','This thesis evaluates ten Hong Kong coastal water-quality tasks using the repaired 77-feature version WQ77_REPAIRED_20261115_V1 and five-fold held-out-station validation. TabICLv2 exceeds XGBoost in pooled out-of-fold R² on nine tasks, but not on middle-layer total nitrogen (0.485 versus 0.497). Pooled R² reaches 0.592 for surface chlorophyll-a and 0.627 for Secchi depth. Source ablation with XGBoost and a separate historical 78-feature regionalization experiment provide complementary, protocol-specific evidence. New-model SHAP and ALE results remain pending cross-reconstruction numerical validation and are not used as formal explanations in this candidate manuscript. Same-month and contemporaneous inputs limit the conclusions to retrospective concurrent prediction and spatial generalization to held-out stations, not real-time forecasting.')
replace_start('正式输入依冻结分类','正式WQ77_REPAIRED_20261115_V1输入仍为77项、六来源组：MET气象、RHT降雨水文潮汐、OCN海洋物理、HUM人类活动代理、TMP时间过程、SPA空间静态。表2-3与附录G保留原始字段顺序；来源、物理单位和可用时间以候选版正式特征元数据为准。rain_month为采样所在自然月整月累计，同月气象、夜间灯光和日历日潮汐包含同期信息。各源的UTC日界、香港日历日和卫星产品月界尚未统一为业务预测截止时点，故本研究属于历史同期条件下的未见站点预测，不称严格实时提前预测。夜间灯光仅为人类活动代理，不能等同排放实测；补充数据提高部分字段覆盖，不代表77项特征完全无缺失。')
replace_start('正式预测沿用各Family V2','正式十任务采用WQ77_REPAIRED_20261115_V1候选版冻结五折留站点预测及其模型配置。TabICLv2、XGBoost和四任务TabPFN/TabM的结果分别取自同版注册输出，模型间采用相同测试obs_id、站点折次及原目标单位。新模型解释的预测上下文与数值复现尚待核验，附录D的旧XGBoost解释资产亦不作为候选版模型的正式解释。')
replace_start('表4-1列出十项任务的XGBoost留站基准','表4-1列出新候选版十任务XGBoost基准pooled OOF指标。Chl-a表层、Secchi、TN表层和TP中层R²分别为0.486、0.545、0.304和0.331；各目标误差以本任务原单位计算。TP表层为0.217，仍为相对困难任务，但不再沿用旧版负R²。不同任务的R²可比较模型在各自任务的拟合程度，RMSE/MAE的绝对数值不能跨目标量纲直接比较。')
replace_start('TabICLv2在十项任务的pooled R²均高于XGBoost','新候选版十任务中，TabICLv2的pooled OOF R²在九项任务高于XGBoost；唯一例外为TN中层（0.485对0.497，ΔR²为−0.011）。Chl-a表层、Secchi和TN表层分别从0.486、0.545和0.304提高到0.592、0.627和0.402。TP表层两模型均为正R²（0.217与0.234），旧版负值不再适用。表4-2的站点RMSE胜率来自候选版配对站点误差；“稳健性”一列保留待复核，不沿用旧版bootstrap判定。')
replace_start('Chl-a表层RMSE由4.807降至4.184','Chl-a表层RMSE从4.680降至4.166 μg/L、MAE从2.146降至1.815 μg/L；Secchi RMSE从0.748降至0.677 m、MAE从0.486降至0.431 m。TN表层RMSE从0.352降至0.326 mg/L，TP中层从0.0361降至0.0348 mg/L。TN中层RMSE由0.2287升至0.2313 mg/L，其MAE虽下降但不能据此掩盖R²和RMSE的退化。不同目标的误差绝对值不跨单位比较。')
replace_start('十任务站点RMSE胜率为70.2%—88.1%','候选版十任务站点RMSE改善比例为72.6%—89.3%，表示多数站点的TabICLv2误差降低，但局部退化仍存在。SM18的Chl-a中层RMSE由2.737升至3.003 μg/L，VT11的TN表层由0.411升至0.767 mg/L；SM19的TN表层则由0.992降至0.150 mg/L。PT3的Chl-a表层由4.632略降至4.624 μg/L，不再沿用旧版退化案例判断。')
replace_start('Chl-a三水层、Secchi及TN表层的分级为','图4-2显示九个分层任务的pooled ΔR²和站点RMSE改善比例。TN中层ΔR²为−0.011，而站点RMSE改善比例仍为79.8%，说明pooled指标与站点胜率回答不同问题。当前候选版未完成与旧版完全同口径的配对站点bootstrap，故本稿不沿用“稳健”“中等”的旧版分级。Secchi作为独立任务由图4-1单列。')
replace_start('十项任务上TabICLv2的pooled R²均高于XGBoost','新候选版十任务中TabICLv2的pooled OOF R²在九项任务高于XGBoost，TN中层为0.485对0.497，出现轻微退化。多数任务的站点RMSE胜率高于50%，但逐站点仍有局部退化。四任务扩展比较中TabPFN在TN表层pooled R²为0.422，高于TabICLv2的0.402；模型排序随任务变化。')
replace_start('（1）在统一77项严格特征','（1）WQ77_REPAIRED_20261115_V1的十任务五折留站结果显示，TabICLv2在九项任务的pooled OOF R²高于XGBoost，TN中层为0.485对0.497，不能概括为十项任务全部领先。Chl-a表层与Secchi的TabICLv2 R²分别为0.592和0.627。')
replace_start('（2）总体改善不等于全部站点改善','（2）pooled OOF改善并不等于每个站点改善。逐站点RMSE变化在任务和地理位置间存在差异；站点胜率应与各任务相同站点集合上的配对误差共同解释，不能作为预测准确率。')
replace_start('（3）四项代表任务的扩展比较','（3）四项代表任务中，TabICLv2在Chl-a表层和Secchi具有较高R²，TabPFN在TN表层达到0.422并优于TabICLv2的0.402；TabM的相对表现随任务而变。四任务模型排序不能推广到十任务。')
replace_start('表4-3与图4-5共同比较','表4-3与图4-5比较新候选版Chl-a表层、Secchi、TN表层和TP中层四项任务的XGBoost、TabICLv2、TabPFN和TabM，全部使用相同测试obs_id及五折站点留出。Chl-a表层TabM的pooled R²为0.596，略高于TabICLv2的0.592；Secchi TabICLv2以0.627领先；TN表层TabPFN为0.422，高于TabICLv2的0.402；TP中层TabICLv2为0.378，明显高于TabPFN的0.243。模型排序不能外推到所有十项任务或实时预报。')
replace_start('冻结汇总显示，历史分区模型相对全域模型','候选版历史78特征分区实验中，Chl-a表、中、底层pooled ΔR²分别为+0.071、+0.055、+0.049，改善折数均为4/5；Secchi历史表、中、底层分别为+0.045、+0.039、+0.042，改善折数为5/5、4/5、5/5。Secchi三层仅属于历史78特征扩展协议，正式77特征Secchi仍为单任务。图5-1同时展示12项pooled差值和60个逐折差值。')
replace_start('历史TN表、中、底层pooled ΔR²','历史TN表、中、底层pooled ΔR²分别为−0.019、−0.069、+0.014，改善折数分别为2/5、2/5、4/5；TP表、中、底层分别为−0.205、−0.148、−0.065，改善折数为2/5、0/5、3/5。营养盐任务未出现一致的分区收益，且逐折方向存在差异。该证据限于包含water_zone_code的历史78特征协议。')
replace_start('图5-2与表5-1展示四项代表任务','候选版正式XGBoost删组消融见图5-2和表5-1。删除空间静态组后Chl-a表层与Secchi的pooled R²分别下降0.072和0.113；删除气象组后Chl-a表层、Secchi和TP中层分别下降0.012、0.017和0.019，而TN表层提高0.020。来源效果依任务而异，删组后提高仅说明该验证协议下预测指标改善，不表示该来源对真实水质具有负面因果作用。')
replace_start('图5-3给出VT11','图5-3给出VT11、SM18、SM19和PT3四站的地理位置、原登记表层输入缺失背景及候选版五个站点×任务误差案例。VT11表层主表的旧登记缺失率为62.7%，高于94站中位数8.8%；该覆盖背景并非新候选版修复后缺失率，不能据此解释候选版模型误差。新版本TN表层VT11误差增加，SM19误差明显降低；PT3 Chl-a表层误差轻微降低。站点位置、缺失背景和误差仅作并列诊断，不构成误差成因的因果证据。')
replace_start('正式十任务站点留出比较给出TabICLv2的总体提升','新候选版正式十任务站点留出比较显示九项pooled R²提升及TN中层的轻微退化，XGBoost来源消融揭示不同任务对六来源组的预测依赖，历史78特征分区实验提供独立的空间异质性证据。新TabICLv2模型的SHAP/ALE尚未通过数值一致性核验，本候选稿不将旧版解释并入上述证据链。')
replace_start('（5）正式XGBoost来源消融表明','（5）候选版XGBoost六来源消融表明空间静态组对Chl-a表层与Secchi具有较明显的预测增量，其他来源的删组影响随任务而异。历史78特征分区实验中Chl-a和Secchi多有收益，TN与TP不一致；这不能直接推广为正式77特征分区模型的效果。')
replace_start('（4）TabICLv2历史四任务五折解释','（4）新候选版TabICLv2解释仍处于数值核验阶段。11月20日回传的30核心样本SHAP虽具备77项归因文件，但18/30预测端点与独立P1审查超过1e-6，20条ALE亦不能由内部文件完整性通过替代模型函数一致性。因此本候选稿暂不依据旧版SHAP/ALE解释新模型。')
replace_start('本文的贡献在于对香港近岸多指标','本文对香港近岸多指标、多水层任务开展了统一修复77特征和留站划分下的表格模型比较，并区分正式来源消融、历史78特征分区及逐站点诊断的证据范围。新模型解释结果待数值复现完成后才能纳入最终论文；当前结论仅建立在已冻结并通过核验的预测与对照结果上。')

# 对剩余旧版解释段落和旧性能数字作显式候选状态标记，避免旧结论被当作新结果。
for p in doc.paragraphs:
    if p.style.name.startswith('toc') or p.style.name in ('图题','表题'):continue
    text=p.text
    if text.startswith(('图3-3至图3-6','图3-12至图3-15','图3-16以双面板','十任务P1 SHAP在各自目标尺度内','正式十任务站点留出比较给出TabICLv2的总体提升')):
        put(p,'【V6候选稿待核验】修复77特征模型的SHAP/ALE解释尚未通过跨重建数值一致性审查；本段旧版解释与图像已隔离，待同一预测函数复现后重写。')
    elif text.startswith('图4-4展示四项代表任务的站点误差空间分布'):
        put(p,'')
    elif text.startswith('图5-2与表5-1展示四项代表任务'):
        put(p,'新候选版四任务六来源消融见表5-1和图5-2；ΔR²为删组模型减完整模型。')
    elif text.startswith('注：ΔR²=TabICLv2−XGBoost，正值表示整体R²提高；本图九个分层任务均为正'):
        put(p,'注：ΔR²=TabICLv2−XGBoost；TN中层为负值。站点改善率为RMSE下降站点比例，两面板色标独立。Secchi没有三水层组合，已在图4-1独立展示。')
    elif text.startswith('注：4个唯一站点、5个站点×任务案例。缺失率仅为表层主表77特征背景'):
        put(p,'注：四个唯一站点、五个站点×任务案例。缺失率来自修复前表层主表登记，仅作输入覆盖历史背景；各任务RMSE来自候选版冻结站点误差，TN/TP单位为mg/L，Chl-a为μg/L。两类数据版本不同，不能将旧缺失率作为新误差成因。')

doc.save(DEST)
(OUT/'V5.1_V6_修改记录.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'CREATED {DEST}; modified={len(changes)}; explanation placeholders={len(pending)}')
