from pathlib import Path
from io import BytesIO
from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.enum.section import WD_ORIENT
from docx.shared import Cm
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / '论文修订交付_20261121' / 'V6_1_科研深化与全文重建'
OUT = Path(__file__).resolve().parent
DOC = Document(SRC / '硕士学位论文_V6.1.docx')
LOG = []

def replace(old, new):
    n = 0
    for p in DOC.paragraphs:
        if old in p.text:
            p.text = p.text.replace(old, new)
            n += 1
    LOG.append((old, new, n))

def remove(p):
    p._element.getparent().remove(p._element)

# V6.1 的横向表4-2分节误延续到第四、五章，导致内嵌图片越过页边。
s = DOC.sections[4]
s.orientation = WD_ORIENT.PORTRAIT
s.page_width = Cm(21)
s.page_height = Cm(29.7)
s.left_margin = Cm(2.5)
s.right_margin = Cm(2.0)
s.top_margin = Cm(2.5)
s.bottom_margin = Cm(2.0)

replace('3.2 TabICLv2模型的SHAP解释', '3.2 TabICLv2模型及其解释方法')
for p in list(DOC.paragraphs):
    if p.text == '3.2 当前版本TabICLv2解释的预测函数一致性':
        remove(p)
replace('3.3 SHAP与ALE的证据边界', '3.3 当前版本解释结果的适用边界')
replace('TN表层TabPFN为0.417', 'TN表层TabPFN为0.422')
replace('TP中层两者接近', 'TP中层TabICLv2为0.378，高于TabPFN的0.243')
replace('现有主登记表未提供可与十任务站点留出逐项对齐的正式年份留出结果，故这里保留验证目标与适用边界，而不填入缺乏同口径核验的数字。', '本文以五折留站点结果评价未见站点的空间泛化。年份留出与站点留出分别对应不同的外推对象；本研究不据此比较两种验证方案的数值高低。')
replace('注：原稿原样保留', '注：图示依据对应研究数据和方法绘制。')

for p in DOC.paragraphs:
    t = p.text
    if t.startswith('正式解释对象为WQ77_REPAIRED_20261115_V1的TabICLv2模型。'):
        p.text = '本研究拟解释修复版77项特征下的TabICLv2模型预测。解释样本、特征顺序、预处理和预测调用须与正式站点留出实验保持一致。当前解释结果尚未通过模型预测的一致性核验，因此本章仅介绍SHAP与一维ALE的方法及预定分析范围，不报告特征排名或响应曲线。'
    elif t.startswith('SHAP用于分解特定预测的特征归因，'):
        p.text = 'SHAP用于分析特征对特定模型预测的归因，一维ALE用于描述观测支持区间内的模型局部响应。两者分别回答贡献大小和响应形态问题，均不能用于环境因果推断。正式分析设计覆盖十项任务，每项任务预先确定三个独立解释样本及两条一维ALE曲线；在预测函数一致性确认前，这些设计不构成已经完成的解释结果。'
    elif t.startswith('当前版本已建立可追溯的预测模型和解释登记，'):
        p.text = '本章说明了本文采用的预测模型及解释方法。TabICLv2的正式SHAP与ALE结果仍需通过预测一致性核验，故后续章节仅依据已核验的预测、来源消融和独立分区实验讨论。'
    elif t.startswith('（4）修复版TabICLv2解释仍处于数值核验阶段。'):
        p.text = '（4）修复版TabICLv2的模型解释仍待预测一致性核验；本稿不以尚未核验的SHAP或ALE结果支撑环境变量作用结论。'
    elif t.startswith('修复版十任务中TabICLv2的pooled OOF R²在九项高于XGBoost，TN中层为0.485对0.497'):
        p.text = '修复版十任务中，TabICLv2的pooled OOF R²在九项高于XGBoost，TN中层为0.485对0.497（ΔR²=−0.011）。配对站点Bootstrap的95%区间显示，ΔRMSE有4/10项不跨零，而ΔMAE有9/10项不跨零。RMSE对较大残差更敏感；在TN、TP等任务中，平均绝对误差的改善与少数较大残差对均方根误差的影响可能同时存在，应结合逐站点分布解读。这些区间刻画既定站点样本的重抽样变动，不等同于预先设定的正式假设检验。'
    elif t.startswith('本文对香港近岸多指标、多水层任务开展了统一修复77特征'):
        p.text = '本研究的实际意义体现在三个方面：一是在统一特征与站点留出条件下比较表格基础模型在多指标、多水层近岸水质预测中的适用性；二是通过正式来源消融揭示环境信息预测价值的任务差异；三是利用历史分区实验与逐站点误差分析，明确空间建模收益及未见站点泛化的边界。这些贡献属于应用和评价框架，不构成新算法或环境因果机制。'

for p in list(DOC.paragraphs):
    if p.style.name == 'Heading 1' and not p.text.strip():
        remove(p)
    elif p.text.startswith('本文完成十项正式水质任务的站点留出比较，并以四项代表任务的模型扩展、来源消融和解释结果补充讨论。'):
        remove(p)

for i,p in enumerate(DOC.paragraphs):
    if p.text == '图2-4 叶绿素a、透明度、总氮与总磷的观测分布':
        DOC.paragraphs[i+1].text = '注：Chl-a、TN和TP按三水层展示；Secchi仅作为单一任务展示。'

# 用已核验的原图数据制作不透明RGB PNG，九图均用内嵌方式重新插入。
figs = sorted((SRC / '正式论文图片').glob('图[45]-*.png'))
assert len(figs) == 9, len(figs)
img_out = OUT / '正式论文图片'
img_out.mkdir(parents=True, exist_ok=True)
for src in figs:
    dst = img_out / src.name
    Image.open(src).convert('RGB').save(dst, format='PNG', dpi=(600,600), optimize=True)
    label = src.name.split('_')[0]
    caps = [p for p in DOC.paragraphs if p.text.startswith(label+' ') and p.style.name == '图题']
    assert len(caps) == 1, (label, len(caps))
    cap = caps[0]
    prev = cap._p.getprevious()
    from docx.text.paragraph import Paragraph
    picp = Paragraph(prev, cap._parent)
    assert 'w:drawing' in picp._p.xml, label
    picp.clear()
    picp.alignment = 1
    picp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    picp.paragraph_format.line_spacing = 1.0
    picp.paragraph_format.left_indent = Cm(0)
    picp.paragraph_format.right_indent = Cm(0)
    picp.paragraph_format.first_line_indent = Cm(0)
    picp.paragraph_format.keep_with_next = True
    if label in {'图4-4', '图4-5'}:
        picp.paragraph_format.page_break_before = True
    picp.add_run().add_picture(str(dst), width=Cm(15.5))
    cap.paragraph_format.keep_with_next = False

# 表5-1跨页续表重复完整表头，所有行禁止拆分。
tbl = next(t for t in DOC.tables if len(t.rows)==25 and len(t.columns)==5)
for i,row in enumerate(tbl.rows):
    pr = row._tr.get_or_add_trPr()
    if i == 0:
        el=OxmlElement('w:tblHeader'); el.set(qn('w:val'),'true'); pr.append(el)
    el=OxmlElement('w:cantSplit'); pr.append(el)

DOC.save(OUT / '硕士学位论文_V6.2.docx')
(OUT / '修订操作记录.txt').write_text('\n'.join(f'{n}处: {a} -> {b}' for a,b,n in LOG),encoding='utf-8')
