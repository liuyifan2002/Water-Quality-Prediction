"""论文第四轮定向修订：以V3扩充工作副本为唯一母本。"""
from pathlib import Path
import re
import csv
from docx import Document
from docx.shared import Inches
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(r'E:/研究生/codex/水质预测')
SOURCE = ROOT/'文档/论文材料/硕士论文V3_第二章特征与候选数据扩充_20261101/水质预测硕士学位论文_V3_第二章特征与候选数据扩充_工作副本.docx'
OUT = Path(__file__).resolve().parent
TARGET = OUT/'水质预测硕士学位论文_V4_结构与地图定向修订.docx'
MAP = OUT/'图2-1_研究区域与94站点.png'
doc = Document(SOURCE)
body = doc.element.body
changes=[]

def txt(e):
    return ''.join(x.text or '' for x in e.iter(qn('w:t'))).strip()
def find(prefix, style=None):
    ps=[p for p in doc.paragraphs if p.text.strip().startswith(prefix) and not p.style.name.lower().startswith('toc') and (style is None or p.style.name==style)]
    if len(ps)>1 and (prefix.startswith('图') or prefix.startswith('表')):
        ps=[ps[-1]]
    if len(ps)!=1: raise RuntimeError(f'段落匹配不唯一: {prefix}: {len(ps)}')
    return ps[0]
def replace(p,new,why):
    old=p.text
    if old==new:return
    p.text=new
    changes.append((why,old,new))
def new_before(anchor,text,style='Normal'):
    p=doc.add_paragraph(text,style=style)
    anchor._p.addprevious(p._p)
    return p
def remove_element(e):
    e.getparent().remove(e)
def section_children(start,end):
    s=find(start)._p;e=find(end)._p
    a=list(body);return a[a.index(s):a.index(e)]

# 1.2.3：研究进展与方法原理分工。
replace(find('1.2.3 传统机器学习'), '1.2.3 机器学习、深度学习与表格基础模型研究进展','综述标题')
p=find('XGBoost通过逐轮')
replace(p,'传统机器学习为水质预测提供了从线性基准到非线性集成的多层方法。线性回归（LR）便于识别总体线性关系，支持向量回归（SVR）利用核函数处理非线性且对核及参数选择敏感；随机森林（RF）、梯度提升决策树（GBDT）与XGBoost则适于处理多源监测变量的非线性关系。已有珠江流域与近岸海域研究分别比较了SVR、RF、XGBoost等方法，提示模型表现取决于预测指标、特征可用性和验证方式，而非某一算法的普遍优势[15,18,20]。','扩充机器学习研究进展')
anchor=find('1.2.4 水质预测')
new_before(anchor,'深度学习将特征表示与预测函数结合。人工神经网络及多层感知机（ANN/MLP）适合固定长度的多变量输入；卷积网络（CNN）可提取局部空间或时间模式，循环网络（RNN）及其LSTM、GRU变体主要用于连续监测序列中的时序依赖，Transformer则借助注意力机制处理较长序列。在实际水质研究中，CNN-LSTM被用于溶解氧预测，Transformer式方法也用于长序列水质预测[19,21]。这些研究的预测时间步长和输入粒度与本文的站点—日期结构化样本并不相同，不能把其报告精度直接与本文留站结果比较。')
new_before(anchor,'面向表格数据的新型模型进一步探索了预训练与高效神经网络的适用边界。TabPFN和TabICLv2通过预训练形成跨任务的上下文预测能力[2,13]；TabM则是参数共享的表格神经网络集成方法，并非预训练表格基础模型[4]。目前仍需在统一特征、站点隔离和相同评价尺度下检验这些方法在近岸多水层水质任务中的表现；本文据此设置传统树模型基准、表格神经网络及预训练表格模型的对照，而不将未运行的其他算法列为本文实证结果。')

# 图2-1：替换图像，不沿用原图任何站点标注。
oldmap=find('图2-1 研究区地理位置')
image_p=oldmap._p.getprevious()
assert image_p.xpath('.//w:drawing')
from docx.text.paragraph import Paragraph
ip=Paragraph(image_p, oldmap._parent)
ip.clear(); ip.alignment=WD_ALIGN_PARAGRAPH.CENTER
ip.add_run().add_picture(str(MAP),width=Inches(6.2))
changes.append(('图2-1','原图含预设重点站点图例及标签','从冻结94站坐标重新绘制，84+10分层，无单站标签'))
replace(find('注：深蓝点为Chl-a'), '注：深蓝点为Chl-a、TN和TP共同正式任务的84个站点，灰点为原始监测网络中的其余10站；Secchi任务覆盖全部94站。站点坐标来自香港环境保护署监测网络，陆地与道路底图采用Natural Earth 10m数据。','图2-1图注去重点站')
replace(find('研究区位于香港近岸'),find('研究区位于香港近岸').text.replace('主要海区及监测站点分布','地理位置、海岸线和监测站点分布'),'图2-1说明')

# 先迁移3.2的全部现有内容和表格，随后合并六个标题为五个标题。
moving=section_children('3.2 水质预测实验设计','3.3 模型性能')
anchor=find('2.8 样本分布')._p
for e in moving:anchor.addprevious(e)
replace(find('3.2 水质预测实验设计'), '2.8 水质预测实验设计与模型配置','章节迁移')
replace(find('3.2.1 随机划分'), '2.8.1 预测任务与验证方案','实验子节编号')
q=find('所有正式模型采用X_strict_77')
replace(q,'预测任务按2.3节的指标与水层定义，包括Chl-a、TN和TP各三水层及Secchi单任务，共十项。'+q.text,'预测任务说明')
p=find('3.2.2 站点留出')
remove_element(p._p)
replace(find('本文区分随机划分'), '本文区分随机划分、年份留出和站点留出三种泛化问题。随机划分与年份留出仅在协议和数据版本可追溯时作为补充验证方案，不与正式站点留出结果混作同口径配对比较。','验证方案说明')
replace(find('3.2.3 统一五折'), '2.8.2 站点留出与统一五折划分','实验子节编号')
new_before(find('Chl-a、TN和TP使用'), '正式主结果采用冻结五折站点留出：同一站点在任一折只属于训练集或测试集的一侧，使OOF预测面向未见监测站点，而非同站点跨日期插值[14]。')
replace(find('3.2.4 模型配置'), '2.8.3 模型配置与推理环境','实验子节编号')
replace(find('3.2.5 复现性'), '2.8.4 复现性控制与数据泄漏防范','实验子节编号')
replace(find('3.2.6 分海区'), '2.8.5 分海区与来源消融实验设计','实验子节编号')
replace(find('表3-1 XGBoost解释'), '表2-5 XGBoost解释模型资产核验','表号迁移')
replace(find('2.8 样本分布'), '2.9 样本分布及数据特征','二章顺延')
replace(find('2.9 本章小结'), '2.10 本章小结','二章顺延')
replace(find('本章界定了研究对象'),find('本章界定了研究对象').text.replace('统一的观测标识、折内预处理及来源分类','统一的观测标识、冻结站点划分、折内预处理、模型配置及来源分类'),'二章小结')

# 第三章：去实验设计、保留模型原理/评价/解释。
replace(find('第3章 预测模型、实验设计'), '第3章 预测模型与解释方法','第三章标题')
replace(find('3.3 模型性能'), '3.2 模型性能与空间泛化评价指标','三章节号')
replace(find('3.4 TabICLv2'), '3.3 TabICLv2模型的SHAP解释','三章节号')
replace(find('3.5 TabICLv2'), '3.4 TabICLv2模型的一维ALE响应分析','三章节号')
replace(find('3.6 解释结果'), '3.5 解释结果的可靠性分析','三章节号')
replace(find('3.7 本章小结'), '3.6 本章小结','三章节号')
for p in list(doc.paragraphs):
    if p.style.name=='Heading 3' and re.match(r'3\.[456]\.[1-9]',p.text):
        old=p.text;num=int(old[2])-1;replace(p,'3.'+str(num)+old[3:],'三章子节号')
replace(find('表3-2 主要评价指标'), '表3-1 主要评价指标及解释','表号顺延')
replace(find('表3-3 十项水质任务'),find('表3-3 十项水质任务').text.replace('表3-3','表3-2',1),'表号顺延')

# 删除TabICLv2二维支持审查小节及三图。只删除正文段块，不触碰独立XGBoost附录D。
for e in section_children('3.5.3 二维ALE','3.6 本章小结'):
    remove_element(e)
changes.append(('二维ALE正文','原3.6.3及图3-18至图3-20','完整删除，保留附录D的XGBoost独立结果'))
for lead in ('三组二维候选仅完成联合支持审查，未计算二维ALE响应曲面。','Three two-dimensional candidates underwent joint-support assessment only; no two-dimensional ALE response surface was computed.'):
    p=next((p for p in doc.paragraphs if lead in p.text),None)
    if p:replace(p,p.text.replace(lead,''),'摘要去二维ALE')
p=find('本章在统一特征与站点划分下介绍')
new=p.text.replace('二维P1候选仅完成联合支持审查，未计算响应曲面。','')
new=new.replace('并区分十任务正式比较与四任务扩展比较。模型性能将在第四章依据实际结果判定。','并说明评价指标、SHAP与一维ALE的解释协议。模型性能的完整定量结果见第四章。')
replace(p,new,'第三章小结')

# 3.1按模型谱系重写，性能只点出表4-3的关键数值。
oldparts=section_children('3.1 预测模型','3.2 模型性能')
oldtxt='\n'.join(txt(x) for x in oldparts)
assert 'TabICLv2' in oldtxt and 'XGBoost' in oldtxt
for e in oldparts[1:]:remove_element(e)
replace(find('3.1 预测模型'), '3.1 预测模型及基本原理','3.1标题')
anchor=find('3.2 模型性能')._p
def add(t,s='Normal'):new_before(Paragraph(anchor,doc),t,s)
add('本节按传统机器学习、深度学习和新型表格学习三个层次说明模型思想。未在本文正式实验中运行的算法仅用于界定方法发展脉络，不赋予本文实验配置或性能结论。')
add('3.1.1 传统机器学习模型','Heading 3')
add('线性回归（LR）以固定系数表达输入与目标的平均线性关系；支持向量回归（SVR）通过间隔约束与核函数表达非线性；随机森林（RF）利用自助采样与树集成降低单树方差；梯度提升决策树（GBDT）顺序拟合前序模型的残差。XGBoost在提升树目标函数中引入正则化和高效分裂搜索，适合处理异质的结构化环境变量[1,8-9]。水质预测文献已在近岸海域和流域数据上应用这些方法[15,18,20]，但不同研究的目标、样本与划分不能直接横比。本文实际运行的传统机器学习基准为XGBoost；LR、SVR、RF和一般GBDT仅作方法背景。')
add('3.1.2 深度学习模型','Heading 3')
add('ANN/MLP通过多层非线性变换学习固定长度特征表示；CNN适于从局部网格或短窗口提取模式；RNN及LSTM、GRU适于有明确时间顺序的连续监测序列；Transformer利用自注意力建模较长序列关系。已有水质研究在溶解氧、营养盐等时序预测中使用CNN-LSTM与Transformer式方法[19,21]。这些序列模型通常依赖时间窗与时序预测目标，而本文正式任务以站点—日期表格记录和留站测试为核心，故本研究不报告ANN、CNN、RNN、LSTM、GRU或普通Transformer的实测结果。')
add('3.1.3 新型表格学习模型','Heading 3')
add('TabM是基于MLP与参数共享集成思想的表格神经网络，在每项任务内拟合参数；它不是预训练表格基础模型[4]。TabPFN与TabICLv2则利用预训练权重，将当前任务带标签训练样本与待测样本共同构成条件预测上下文[2,13]。本文的TabICLv2推理固定训练上下文、77项特征顺序与预训练权重，并在各项任务的冻结站点划分下评估；非负目标按既定log1p协议处理，预测经expm1恢复原始单位。四模型的具体配置、划分和折内预处理见2.8节，解释分析仅针对所登记预测协议。')
add('3.1.4 本文模型选择与比较依据','Heading 3')
add('四项代表任务的同口径pooled OOF结果见表4-3。TabICLv2在Chl-a表层、Secchi和TP中层的pooled R²分别为0.589、0.602和0.472，位于四模型之首；TN表层则由TabPFN取得0.417，略高于TabICLv2的0.402。TabM在Chl-a表层与Secchi具有竞争力，但在TN表层和TP中层相对较弱。误差指标也提示不能只看R²：例如Secchi的TabM MAE为0.4407 m，低于TabICLv2的0.4486 m；TN表层的TabPFN RMSE/MAE为0.3220/0.1382 mg/L，均低于TabICLv2的0.3261/0.1420 mg/L。')
add('据此，本文以TabICLv2为十项水质任务的重点研究与正文解释对象，使用XGBoost作为传统树模型基准，并将TabPFN和TabM用于四项代表任务的扩展比较。模型优势随指标而异，不将TabICLv2写为所有任务的绝对最优。完整的四模型性能、单位与空间泛化结果均在第四章报告；附录D的XGBoost TreeSHAP/ALE为独立基准解释，不与TabICLv2归因混用。')

# 摘要、目录与图目录更新通过Word域刷新完成；此处只处理普通正文与图题。
# 修正附录D之外少数全文编号引用；所有图3-1至图3-17号不变。
main_start=list(body).index(find('第1章 绪论')._p)
for p in doc.paragraphs:
    if p.style.name.startswith('toc') or p.style.name.startswith('TOC') or p._p.xpath('.//w:fldChar') or p._p.xpath('.//w:instrText'):continue
    if list(body).index(p._p)>main_start and (p.text.startswith('图3-18') or p.text.startswith('图3-19') or p.text.startswith('图3-20')):raise RuntimeError('正文仍含二维图题')

# 参考文献[16][17]原被误放在附录之后；移回参考文献清单，补入四篇经核实的原始文献。
ack=find('致 谢')._p
for num in (16,17):
    p=find(f'[{num}] ')
    ack.addprevious(p._p)
for ref in [
    '[18] WANG X, LI Y, QIAO Q, et al. Water Quality Prediction Based on Machine Learning and Comprehensive Weighting Methods[J]. Entropy, 2023, 25(8): 1186. DOI: 10.3390/e25081186.',
    '[19] TAN W, ZHANG J, WU J, et al. Application of CNN and Long Short-Term Memory Network in Water Quality Predicting[J]. Intelligent Automation & Soft Computing, 2022, 34(3): 1943-1958. DOI: 10.32604/iasc.2022.029660.',
    '[20] DENG T, CHAU K W, DUAN H F. Machine learning based marine water quality prediction for coastal hydro-environment management[J]. Journal of Environmental Management, 2021, 284: 112051. DOI: 10.1016/j.jenvman.2021.112051.',
    '[21] SUN M, XU C, JIA Q, et al. Aquaformer: Multi-source transfer learning model based on Transformer and phase space reconstruction for long sequence water quality forecasting[J]. Journal of Hydrology, 2026, 664: 134372. DOI: 10.1016/j.jhydrol.2025.134372.',
]:new_before(Paragraph(ack,doc),ref)
changes.append(('参考文献','[16][17]误置附录后','移回文末书目，并补入经原始期刊页面核实的[18]-[21]'))

# 学校字体需固定，不依赖主题中文字体回退。
for n in ('Heading 1','Heading 2','Heading 3'):
    style=doc.styles[n]
    rpr=style._element.get_or_add_rPr()
    fonts=rpr.rFonts
    if fonts is None:
        from docx.oxml import OxmlElement
        fonts=OxmlElement('w:rFonts');rpr.insert(0,fonts)
    fonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    fonts.set(qn('w:ascii'),'Times New Roman')
    fonts.set(qn('w:hAnsi'),'Times New Roman')
    for attr in ('eastAsiaTheme','asciiTheme','hAnsiTheme'):
        fonts.attrib.pop(qn('w:'+attr),None)

doc.save(TARGET)
with (OUT/'修改追踪.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['修改类别','原文或原状态','修订后']);w.writerows(changes)
print(TARGET)
print('changes',len(changes),'paragraphs',len(doc.paragraphs),'tables',len(doc.tables))
