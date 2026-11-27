"""基于V6候选稿整合修复版科研结果；旧77特征解释全部退出论文。"""
from pathlib import Path
from copy import deepcopy
import re, json, hashlib
import pandas as pd
from docx import Document
from docx.table import Table
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ROW_HEIGHT_RULE

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
SRC=ROOT/'论文修订交付_20261121/05_v6_candidate/水质预测硕士学位论文_V6_候选稿_解释待核验.docx'
DEST=OUT/'硕士学位论文_V6.1.docx'
FIG=OUT/'正式论文图片'
DATA=ROOT/'论文修订交付_20261121/05_v6_candidate/data'
BOOT=OUT/'A_统计分析/十任务Bootstrap结果.csv'
assert SRC.is_file() and BOOT.is_file() and not DEST.exists()
doc=Document(SRC)
core=pd.read_csv(DATA/'正式十任务指标.csv')
boot=pd.read_csv(BOOT).set_index('task')
ab=pd.read_csv(DATA/'四任务六来源消融.csv')
reg=pd.read_csv(DATA/'历史78特征分区汇总.csv')
changes=[]

def log(where,before,after):changes.append({'位置':where,'修改前':before[:250], '修改后':after[:250]})
def paras():return list(doc.paragraphs)
def one(text):
    xs=[p for p in paras() if not p.style.name.lower().startswith('toc') and p.text.startswith(text)]
    assert len(xs)==1,(text,len(xs));return xs[0]
def cap(no):
    xs=[p for p in paras() if p.style.name in ('图题','表题') and re.match(rf'^{re.escape(no)}(?:\s|$)',p.text)]
    assert len(xs)==1,(no,len(xs));return xs[0]
def text(p,s):
    old=p.text;p.text=s;log('正文',old,s)
def before(anchor,s,style='Normal'):
    p=doc.add_paragraph(style=style);p.add_run(s);anchor._p.addprevious(p._p);return p
def image_before(anchor,path,width=15.5):
    assert Path(path).is_file(),path
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path),width=Cm(width));anchor._p.addprevious(p._p);return p
def remove_element(el):
    parent=el.getparent()
    if parent is not None:parent.remove(el)
def delete_range(start,end):
    a=start._p;b=end._p;node=a.getnext();n=0
    while node is not None and node is not b:
        nxt=node.getnext();remove_element(node);node=nxt;n+=1
    return n
def update_figure(no,name,path,note):
    c=cap(no); old=c.text;text(c,f'{no} {name}')
    prev=c._p.getprevious();assert prev.tag==qn('w:p') and prev.xpath('.//w:drawing'),(no,prev.tag)
    remove_element(prev);image_before(c,FIG/path)
    n=c._p.getnext()
    if n is not None and n.tag==qn('w:p'):
        from docx.text.paragraph import Paragraph
        q=Paragraph(n,doc._body)
        if q.text.startswith('注：'):text(q,'注：'+note)
    log(no,old,c.text)

# 旧版解释章节与图件一并退出：不把占位图留在送审正文。
ch3_start=one('3.2 TabICLv2模型的SHAP解释')
ch4=one('第4章')
removed=delete_range(ch3_start,ch4)
before(ch4,'3.2 当前版本TabICLv2解释的预测函数一致性','Heading 2')
before(ch4,'正式解释对象为WQ77_REPAIRED_20261115_V1的TabICLv2模型。预登记的30个核心样本与20条一维ALE曲线应使用与正式预测完全一致的特征顺序、预处理视图和查询协议。11月20日回传SHAP中18/30个保存预测端点与独立P1审查预测相差超过1×10⁻⁶；在跨进程、跨批次一致性通过以前，不能将其归因数值作为本文正式结果。旧版77特征SHAP、ALE、历史五折解释图及其排名、响应结论均已从本稿移除。')
before(ch4,'3.3 SHAP与ALE的证据边界','Heading 2')
before(ch4,'SHAP用于分解特定预测的特征归因，一维ALE用于描述已观测支持区间内的模型局部响应。二者不能互相替代，也不构成环境因果推断。本研究预设每项任务3个独立核心样本、10个重复探针及每任务2条ALE曲线；这些设计只说明拟开展的解释范围，不能在结果尚未通过预测函数复现时报告变量排名或非线性响应。')
before(ch4,'3.4 本章小结','Heading 2')
before(ch4,'当前版本已建立可追溯的预测模型和解释登记，但正式解释尚未达到数值一致性门槛。因此本章保留解释方法与证据边界，不沿用旧模型解释结论；后续第四、第五章仅依据已通过核验的预测、消融与独立分区实验讨论。')
log('第三章',f'删除旧解释段落及图片相关XML节点{removed}个','保留当前版解释协议和明确未通过状态；旧77解释退出正文')

# 原表2-5是旧77特征XGBoost解释资产，不能沿用为修复版模型证据。
old_asset=cap('表2-5')
asset_tbl=old_asset._p.getnext();assert asset_tbl.tag==qn('w:tbl')
asset_ref=old_asset._p.getprevious()
from docx.text.paragraph import Paragraph
if asset_ref.tag==qn('w:p') and Paragraph(asset_ref,doc._body).text.startswith('表2-5列出'):
    remove_element(asset_ref)
remove_element(asset_tbl);remove_element(old_asset._p)
for q in paras():
    if q.style.name.lower().startswith('toc'):continue
    if '表2-6' in q.text:text(q,q.text.replace('表2-6','表2-5'))
log('第二章','旧版XGBoost解释资产表2-5','删除旧资产表；原评价指标表2-6顺延为表2-5')

# 第四章：更新图与论证，Bootstrap进入表4-2。
tcap=cap('表4-2');t=Table(tcap._p.getnext(),doc._body)
assert len(t.rows)==11 and len(t.columns)==10
t.autofit=False
widths=[2.5,1.8,2.4,1.9,2.0,2.4,1.9,2.4,2.0,4.5]
for j,w in enumerate(widths):
    t.columns[j].width=Cm(w)
    for row in t.rows:row.cells[j].width=Cm(w)
t.rows[0].cells[8].text='站点RMSE胜率'
t.rows[0].cells[9].text='ΔRMSE/ΔMAE 95%CI'
for row,r in zip(t.rows[1:],core.itertuples()):
    b=boot.loc[r.task]
    row.cells[9].text=(f'[{b.delta_RMSE_CI_low:+.4f},{b.delta_RMSE_CI_high:+.4f}]\n'
                       f'[{b.delta_MAE_CI_low:+.4f},{b.delta_MAE_CI_high:+.4f}]')
    row.height=Cm(.90);row.height_rule=WD_ROW_HEIGHT_RULE.AT_LEAST
    for cell in row.cells:
        for p in cell.paragraphs:
            p.paragraph_format.space_before=Pt(0)
            p.paragraph_format.space_after=Pt(0)
            p.paragraph_format.line_spacing=1.0
            for run in p.runs:run.font.size=Pt(8.0)
for cell in t.rows[0].cells:
    for para in cell.paragraphs:
        para.paragraph_format.space_before=Pt(0)
        para.paragraph_format.space_after=Pt(0)
        para.paragraph_format.line_spacing=1.0
        for run in para.runs:run.font.size=Pt(8.0)
log('表4-2','旧稳健性待复核','当前版1000次站点聚类Bootstrap ΔRMSE/ΔMAE 95%CI')
tnote=doc.add_paragraph('注：末列每格第一行为ΔRMSE、第二行为ΔMAE的配对站点聚类Bootstrap 95%区间（1000次）。Δ=TabICLv2−XGBoost；区间跨零时不作明确误差优势判断。')
t._tbl.addnext(tnote._p)

update_figure('图4-1','十项预测任务的模型pooled OOF性能比较','图4-1_十任务模型预测性能比较.png',
 '同一任务、相同留站测试样本下比较XGBoost与TabICLv2。两点间连线只表示配对关系；不同目标的RMSE单位不在本图混用。')
update_figure('图4-2','十项任务预测性能提升与退化','图4-2_十任务预测性能提升与退化.png',
 '左为TabICLv2减XGBoost的pooled ΔR²；右为逐站点RMSE降低比例。TN中层ΔR²为负，但站点胜率大于50%，两种统计量回答的问题不同。')
update_figure('图4-3','十项任务逐站点RMSE比值分布','图4-3_站点层面的预测误差分布.png',
 '横轴为同一站点两模型RMSE比值的log₂，零表示两者相同，负值为TabICLv2误差降低。点为中位数，粗线为四分位范围；全部850个任务—站点记录均纳入，原始RMSE见冻结逐站点表。')
update_figure('图4-4','四项代表任务的站点RMSE变化空间分布','图4-4_四项代表任务的空间误差地图.png',
 '四任务分别为Chl-a表层、Secchi、TN表层和TP中层；负ΔRMSE表示TabICLv2误差降低。每任务采用自身单位的零中心色标，范围取|ΔRMSE|第95百分位，超界站点饱和显示，原始数值未裁改；散点采用真实站点坐标。')
update_figure('图4-5','四项代表任务的四模型pooled OOF比较','图4-5_四种模型的性能比较.png',
 'XGBoost、TabICLv2、TabPFN、TabM在同一任务使用相同测试obs_id；四任务共用R²范围，不代表RMSE单位可跨任务比较。')

text(one('图4-1对照十项任务'), '图4-1并列给出十项任务的XGBoost与TabICLv2 pooled OOF R²，图4-2进一步分离总体ΔR²与站点改善覆盖。九项任务的R²提高，TN中层降低0.011；配对站点Bootstrap的误差区间见表4-2。')
text(one('新候选版十任务中，TabICLv2的pooled OOF R²在九项任务'),
 '修复版十任务中TabICLv2的pooled OOF R²在九项高于XGBoost，TN中层为0.485对0.497（ΔR²=−0.011）。配对站点Bootstrap显示Chl-a三水层及Secchi的ΔRMSE 95%区间均低于零；TN表层、TN中层、TN底层和TP三水层的区间跨零。因此，pooled R²较高并不自动构成站点重抽样下明确的误差优势。表4-2报告两种误差的95%区间，站点改善比例仅反映改善站点所占比例。')
text(one('图4-2显示九个分层任务'),
 '图4-2显示十任务pooled ΔR²与站点改善覆盖。TN中层总体R²略低，但79.8%的站点RMSE降低，提示总体分数可能受到少数大误差站点或目标方差结构影响；这属于数据诊断线索，不应反推为已证实的机制。图4-3展示站点RMSE相对变化分布，较宽的尾部提醒读者同时关注受益与退化站点。')
text(one('图4-2以两组独立色标'), '图4-2保留所有正负ΔR²，图4-3采用同站点RMSE比值的log₂刻度：零表示误差相同，负值表示TabICLv2较低。比值消除了目标单位差异，但仍需在同一任务内解释误差水平。')
text(one('随机划分评价同一监测总体'), '随机划分、年份留出和站点留出分别回答同一监测总体内随机样本预测、未见年份外推和未见站点空间泛化。三者不能据历史异口径结果直接比较数值高低；本文正式结论仅依据统一五折留站实验。')
text(one('站点留出是本文正式主口径'), '统一五折站点留出是本文正式验证口径。十任务均保留同版冻结测试obs_id、pooled OOF指标与站点误差；图4-3展示站点层面的误差分布，不能把随机或年份实验的历史分数与其合并。')
text(one('图4-4显示Chl-a表层'), '图4-4展示Chl-a表层、Secchi、TN表层和TP中层的站点RMSE变化空间分布，分别覆盖84、94、84和84站。不同任务采用各自单位的零中心色标，地图显示改善与退化在空间上并存，不据此确定地理位置是误差成因。')
text(one('本章以正式站点留出为主'), '本章在相同冻结测试样本上比较十任务pooled OOF精度、配对站点Bootstrap误差区间、站点分布和四任务现代模型。TabICLv2在多数任务具有较高R²，但TN中层出现轻微反转，且若干任务的ΔRMSE置信区间跨零；模型优势应限定于具体指标与任务。')

# 原图4-3位于验证协议小节，移到4.2.4末，使编号按首次出现排列。
c=cap('图4-3');prev=c._p.getprevious();note=c._p.getnext();anchor=one('4.3 不同验证口径与证据边界')._p
for el in [prev,c._p,note]:anchor.addprevious(el)

# 第五章重组为：来源消融、来源差异、历史分区、当前覆盖与讨论。
ch5=one('第5章');ch6=one('第6章')
old5=delete_range(ch5,ch6)
def h(s,lv=2):return before(ch6,s,f'Heading {lv}')
def p(s):return before(ch6,s)
def f(no,title,file,note):
    p(f'见{no}。')
    image_before(ch6,FIG/file)
    before(ch6,f'{no} {title}','图题')
    p('注：'+note)

h('5.1 六类环境信息来源的消融效应')
p('正式来源消融以WQ77_REPAIRED_20261115_V1的77项特征和XGBoost为对象，比较四项代表任务的完整模型与六个逐组删除模型。六组来自冻结特征元数据：MET气象、RHT降雨水文潮汐、OCN海洋物理、HUM人类活动代理、TMP时间过程及SPA空间静态。ΔR²定义为删组模型减完整模型；负值表示删除该来源后R²下降，正值表示删组后指标提高。该实验提供预测依赖证据，不代表环境因果效应。')
f('图5-1','六类环境信息来源的消融效应','图5-1_六类环境信息来源的消融效应.png',
 '正式77特征XGBoost、四项代表任务、六来源逐组删除。蓝色负ΔR²表示删组后性能下降；红色正ΔR²表示提高。只有四任务具备当前批次的删组结果，图中不补造其他六任务。')
p('图5-1显示空间静态组在Chl-a表层和Secchi任务具有较大的负向删组ΔR²，分别为−0.072和−0.113，说明该组在当前留站划分下提供了较多可利用的预测信息。气象组删去后，Chl-a表层、Secchi和TP中层的R²分别降低约0.012、0.017和0.019，而TN表层提高约0.020。组间相关与模型重拟合可能改变单组删除效应，因此不能将正ΔR²解释为来源对真实水质有负面作用。')
before(ch6,'表5-1 四项代表任务的六来源消融结果','表题')
tab=doc.add_table(rows=25,cols=5)
tab.style='Table Grid'
ch6._p.addprevious(tab._tbl)
heads=['任务','删除来源','pooled ΔR²','ΔRMSE','删组后RMSE改善站点比例']
for j,v in enumerate(heads):tab.rows[0].cells[j].text=v
task_labels=core.set_index('task')['任务'].to_dict()
ab_order=['chla_surface','secchi','tn_surface','tp_middle']
group_order=['MET','RHT','OCN','HUM','TMP','SPA']
i=1
for task in ab_order:
 for group in group_order:
    r=ab[(ab.task==task)&(ab['来源']==group)].iloc[0]
    vals=[task_labels[task],group,f'{r.delta_R2:+.3f}',f'{r.delta_RMSE:+.4f}',f'{r["删除后RMSE改善站点比例"]*100:.1f}%']
    for j,v in enumerate(vals):tab.rows[i].cells[j].text=v
    i+=1
for row in tab.rows:
    for cell in row.cells:
        for para in cell.paragraphs:
            for run in para.runs:run.font.size=Pt(8)
p('注：完整模型含六来源；删组模型保留其余五来源。ΔR²和ΔRMSE均为删组模型减完整模型。误差单位随任务分别为μg/L、m或mg/L，不跨任务比较绝对值。')
h('5.2 不同水质任务的来源依赖差异')
f('图5-2','不同环境信息来源的任务依赖','图5-2_不同环境信息来源的任务依赖.png',
 '纵轴为删除来源造成的pooled R²损失，即−ΔR²；正值表示该来源被删除后预测下降。四任务是本批次正式来源消融的完整任务范围。')
p('图5-2按相同来源顺序重排四任务删组损失，使不同任务的依赖轮廓更易对照。空间静态信息对Chl-a表层与Secchi的重要性较突出，其他来源方向与幅度随任务变化。对TN、TP的解释应结合相应任务的逐站点误差与数据覆盖，而不能从少数删组分数推断单一水质过程。不同来源可能互为代理，因此逐组消融不能量化所有来源的独立因果份额。')
h('5.3 历史78特征全域与分区建模比较')
p('独立的历史分区扩展实验采用78项特征，包含water_zone_code；共12个目标—水层组合、60个任务折。Secchi的三个水层仅属于该历史协议，正式77特征预测中Secchi仍为单一任务。分区决策记录中496次采用区域模型、36次回退全域模型，不能描述为每个水区都独立训练。图5-3分别展示pooled ΔR²和逐折ΔR²。')
f('图5-3','历史全域与分区模型的任务收益及折次差异','图5-3_历史全域与分区模型比较.png',
 '历史78特征扩展实验，分区−全域；左为合并OOF后的pooled ΔR²和改善折数，右为60个逐折ΔR²。右面板色标在−0.5以下饱和，格内数字保留真实值。两种统计量不能相互代替，也不与正式77特征十任务合并。')
p('Chl-a三水层的历史pooled ΔR²为+0.071、+0.055、+0.049，均有4/5折改善；Secchi历史三水层分别为+0.045、+0.039、+0.042，改善折数为5/5、4/5、5/5。TN表、中层及TP三水层多数未显示一致收益，其中TP表层pooled ΔR²为−0.205。分区效果因此具有明显任务依赖性，且受区域样本规模与全域回退规则约束。')
h('5.4 修复版77项特征的数据覆盖')
p('修复版正式主表覆盖2014—2024年、94个监测站。77项特征的真实观测覆盖须与填补后可用矩阵区分：模型输入完整不意味着原始观测完整。图5-4按来源组展示真实非缺失率；站点和年份覆盖见当前版数据质量附表。')
f('图5-4','修复版77项特征的真实观测覆盖','图5-4_修复版77特征真实观测覆盖.png',
 '按修复版主表和原始缺失掩码统计，不把填补值计为真实观测。各特征的站点、年份和样本覆盖详见数据质量结果表。')
h('5.5 环境信息、空间异质性与泛化边界')
p('正式77特征来源消融、四任务现代模型比较和历史78特征分区结果回答不同问题。来源消融反映XGBoost在当前留站协议下对输入组的预测依赖；历史分区结果显示部分任务可因局地训练而受益，但其特征体系和回退机制不同；站点RMSE图揭示同一正式模型的局地改善并不一致。这些结果共同提示空间异质性，却不足以单独确定误差或水质变化的因果机制。')
p('应用上应优先核对新站点是否落在训练站点的环境覆盖范围，并报告站点层面的误差分布。rain_month使用采样所在自然月的整月累计，同时包含其他同期气象与环境输入，因此本研究结论是历史同期条件下对未见站点的预测泛化，而不是严格提前预报。')
h('5.6 本章小结')
p('修复版四任务六来源消融呈现任务依赖的预测价值；历史78特征分区实验在Chl-a与Secchi组合中较多获益，在TN和TP中不一致。修复版特征覆盖和站点误差应共同约束结果解释；尚未通过数值复现的SHAP/ALE不能参与当前版本的环境变量响应论证。')
log('第五章',f'删除旧第五章节点{old5}个','新建六来源消融→任务依赖→历史分区→当前覆盖→综合讨论，图5-1至5-4')

# 旧77特征附录和技术审查图彻底退出；77字段清单保留并改为附录A。
g=one('附录G 77项严格输入特征清单')
app_end=one('个人简历')
app_start=one('附录A 解释分析覆盖范围')
node=app_start._p
while node is not g._p:
    nxt=node.getnext();remove_element(node);node=nxt
node=g._p.getnext()
while node is not app_end._p:
    nxt=node.getnext()
    # G的77特征清单表格与两段说明保留；H及其图片删除。
    if node.tag==qn('w:p'):
        from docx.text.paragraph import Paragraph
        pp=Paragraph(node,doc._body)
        if pp.text.startswith('附录H '):
            while node is not app_end._p:
                nxt=node.getnext();remove_element(node);node=nxt
            break
    node=nxt
text(g,'附录A 修复版77项正式输入特征清单')
for p0 in paras():
    if p0.text.startswith('正式闭环：runs/final_experiment_closure'):remove_element(p0._p)
log('附录','旧A/B/C/D/E/F/H及解释占位图','仅保留修复版77项正式特征清单，改为附录A')

# 摘要、结论及跨章旧图引用的边界修正。
for pp in paras():
    s=pp.text
    if pp.style.name.startswith('toc') or pp.style.name in ('图题','表题'):continue
    if '附录G' in s:text(pp,s.replace('附录G','附录A'))
    if '附录D的旧XGBoost解释资产亦不作为候选版模型的正式解释' in pp.text:
        text(pp,pp.text.replace('附录D的旧XGBoost解释资产亦不作为候选版模型的正式解释。','旧版解释资产不作为修复版模型的正式解释。'))
    if '附录D的XGBoost TreeSHAP和一维ALE为独立基准解释' in pp.text:
        text(pp,pp.text.replace('附录D的XGBoost TreeSHAP和一维ALE为独立基准解释，不与TabICLv2归因混用。','旧版XGBoost解释已退出本稿；本轮只依据修复版模型的冻结预测结果讨论。'))
    if '不再沿用旧版负R²' in pp.text:
        text(pp,pp.text.replace('，旧版负值不再适用',''))
    if '不再沿用旧版退化案例判断' in pp.text:
        text(pp,pp.text.replace('，不再沿用旧版退化案例判断',''))
    if '新候选版' in pp.text or '候选版' in pp.text:
        text(pp,pp.text.replace('新候选版','修复版').replace('候选版','修复版'))
    if '图H-3' in s:text(pp,pp.text.replace('历史数值对照另见图H-3，','').replace('图H-3','历史异口径技术记录'))
    if '图5-3' in pp.text and pp.text.startswith('图5-3给出VT11'):
        text(pp,'当前版本典型站点误差已纳入第四章站点诊断；旧版输入缺失率与新误差的跨版本拼接已从正文退出。修复版77项特征的真实观测覆盖见图5-4。')
    if pp.text.startswith('本文以2014—2024年香港近岸监测记录构建'):
        text(pp,'本文以2014—2024年香港近岸94站监测记录和WQ77_REPAIRED_20261115_V1的77项正式特征，研究Chl-a、Secchi、TN与TP十项水质任务在未见站点的预测。统一五折留站比较显示，TabICLv2在九项任务的pooled OOF R²高于XGBoost，TN中层为0.485对0.497。站点聚类Bootstrap显示Chl-a三水层与Secchi的ΔRMSE 95%区间低于零，TN及TP任务部分区间跨零。四项代表任务的现代模型比较揭示任务依赖的模型排序；正式XGBoost六来源消融与独立历史78特征分区实验分别说明输入组增量和空间异质性。修复版TabICLv2解释尚未通过预测函数跨重建一致性核验，本稿不采用旧版SHAP、ALE结论。同期环境输入限定研究结论为历史同期预测与留站点空间泛化，不能称为实时提前预报。')
    if pp.text.startswith('This thesis evaluates ten Hong Kong coastal'):
        text(pp,'This thesis evaluates ten coastal water-quality tasks using observations from 94 Hong Kong monitoring stations and the repaired 77-feature dataset WQ77_REPAIRED_20261115_V1. Under unified five-fold held-out-station validation, TabICLv2 improves pooled out-of-fold R² over XGBoost in nine tasks, whereas middle-layer total nitrogen declines from 0.497 to 0.485. A 1,000-replicate paired station-cluster bootstrap places the 95% interval for the RMSE difference below zero for all three chlorophyll-a layers and Secchi depth; several nitrogen and phosphorus intervals cross zero. Four-model comparisons, six-source XGBoost ablation, and a separate historical 78-feature regionalization experiment show task-specific prediction and spatial effects. Current-model SHAP and ALE have not yet passed cross-reconstruction prediction consistency checks; previous-model explanations are excluded. Contemporaneous inputs restrict the conclusions to retrospective concurrent prediction and spatial generalization to unseen stations rather than real-time forecasting.')
    if '（4）新候选版TabICLv2解释仍处于数值核验阶段' in pp.text:
        text(pp,'（4）当前修复版TabICLv2解释尚未通过跨重建预测函数数值一致性核验，旧版SHAP/ALE不进入正式论文结果。')

# 保存并记录直接证据，不改冻结文件。
doc.save(DEST)
(OUT/'论文整合修改记录.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
print('CREATED',DEST,'changes',len(changes),'figures',len(doc.inline_shapes),'tables',len(doc.tables))
