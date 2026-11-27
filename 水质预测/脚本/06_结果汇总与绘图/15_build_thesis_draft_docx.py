"""汇编论文大纲、已生成图表和结果 CSV 为 Word 论文草稿。

本脚本仅读取已有论文图表和正式结果表，不训练模型，不改写主链结果。
"""

from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "结果" / "论文图表" / "论文图版_随机比例划分_20260905"
TABLE_ROOT = ROOT / "结果" / "论文图表" / "论文表格_随机比例划分_20260905"
OUTPUT = ROOT / "文档" / "论文材料" / "论文草稿_香港近岸多水层水质预测与遥感信息敏感性分析_随机比例划分_20260905.docx"

FIGURES = {
    "ch1": [("图 1-1", "研究技术路线", "第一二章/图1-1_研究技术路线.png"), ("图 1-2", "科学问题与章节关系", "第一二章/图1-2_科学问题与章节关系.png")],
    "ch2": [("图 2-1", "香港近岸水质监测站点与海区分布", "第一二章/图2-1_监测站点与海区分布.png"), ("图 2-2", "三水层观测覆盖情况", "第一二章/图2-2_三水层观测覆盖.png"), ("图 2-3", "四项水质指标在三水层的分布", "第一二章/图2-3_四项水质指标三水层分布.png"), ("图 2-4", "水质观测的时间覆盖", "第一二章/图2-4_水质观测时间覆盖.png"), ("图 2-5", "多源数据接入与样本主键关系", "第一二章/图2-5_多源数据接入与样本主键.png")],
    "ch3": [("图 3-1", "固定随机比例的训练集与测试集划分", "第三四章/图3-1_随机比例划分.png"), ("图 3-2", "十二个任务的随机留出预测性能热图", "第三四章/图3-2_十二个任务预测指标热图.png"), ("图 3-3", "不同特征组的随机留出性能比较", "第三四章/图3-3_特征组总体性能比较.png"), ("图 3-4", "历史特征与遥感特征融合的性能增益", "第三四章/图3-4_历史与遥感融合增益.png"), ("图 3-5", "最佳正式模型的观测值与预测值对比", "第三四章/图3-5_最佳正式模型预测散点图.png")],
    "ch4": [("图 4-1", "正式主表遥感质量覆盖", "第三四章/图4-1_正式主表遥感质量覆盖.png"), ("图 4-2", "融合模型中的遥感特征构成", "第三四章/图4-2_融合模型遥感特征构成.png"), ("图 4-3", "实际可用遥感特征的 SHAP 重要性", "第三四章/图4-3_遥感特征SHAP重要性.png"), ("图 4-4", "实际可用遥感主要特征的 SHAP 依赖关系", "第三四章/图4-4_遥感主要特征SHAP依赖关系.png"), ("图 4-5", "遥感融合的任务级增益判定", "第三四章/图4-5_遥感融合任务级增益判定.png"), ("图 4-6", "不同水层的遥感融合增益分布", "第三四章/图4-6_不同水层遥感增益分布.png")],
    "ch5": [("图 5-1", "Chl-a 表层代表站点的时间序列", "第五章/图5-1_Chla表层代表站点时间序列.png"), ("图 5-2", "Chl-a 表层观测值与预测值散点图", "第五章/图5-2_Chla表层观测值预测值散点图.png"), ("图 5-3", "Chl-a 表层月尺度残差变化", "第五章/图5-3_Chla表层月尺度残差变化.png"), ("图 5-4", "Chl-a 表层主要特征重要性", "第五章/图5-4_Chla表层主要特征重要性.png"), ("图 5-5", "分海区与全站点模型的表现比较", "第五章/图5-5_分海区与全站点模型表现比较.png")],
}
TABLES = {
    "ch2": [("表 2-1", "海区与站点空间覆盖统计", "第一二章/表2-1_海区与站点空间覆盖统计.csv"), ("表 2-2", "多源数据来源与用途", "第一二章/表2-2_多源数据来源与用途.csv"), ("表 2-3", "四项目标变量描述性统计", "第一二章/表2-3_四项目标变量描述性统计.csv"), ("表 2-4", "遥感与短时间窗过程特征覆盖统计", "第一二章/表2-4_遥感与短时间窗过程特征覆盖统计.csv")],
    "ch3": [("表 3-1", "十二个预测任务与样本量", "第三四章/表3-1_十二个预测任务与样本量.csv"), ("表 3-2", "正式模型特征组定义", "第三四章/表3-2_正式模型特征组定义.csv"), ("表 3-3", "十二个任务的正式随机留出结果", "第三四章/表3-3_十二个任务的正式随机留出结果.csv"), ("表 3-4", "各任务最佳正式模型", "第三四章/表3-4_各任务最佳正式模型.csv")],
    "ch4": [("表 4-1", "遥感与短时间窗过程特征质量覆盖", "第三四章/表4-1_遥感与短时间窗过程特征质量覆盖.csv"), ("表 4-2", "融合模型遥感特征构成", "第三四章/表4-2_融合模型遥感特征构成.csv"), ("表 4-3", "遥感融合增益判定", "第三四章/表4-3_遥感融合增益判定.csv")],
    "ch5": [("表 5-1", "Chl-a 表层自相关与峰值波动统计", "第五章/表5-1_Chla表层自相关与峰值波动统计.csv"), ("表 5-2", "Chl-a 表层高值样本误差统计", "第五章/表5-2_Chla表层高值样本误差统计.csv"), ("表 5-3", "分海区 Chl-a 表层最佳模型结果", "第五章/表5-3_分海区Chla表层最佳模型结果.csv")],
    "ch6": [("表 6-1", "主要结论与对应证据", "第六章/表6-1_主要结论与对应证据.csv"), ("表 6-2", "数据缺口与后续改进方向", "第六章/表6-2_数据缺口与后续改进方向.csv")],
}
PENDING_TABLE = ("表 5-4", "连续空间制图待补图表与前提条件", "第五章/第五章待补图表与前提条件.csv")
RESULT_SCOPE_ROWS = [
    ["正式主结果", "全站点三水层；每个任务按固定随机比例 80% 训练、20% 测试；不使用同日其他水质指标", "四项指标的逐水层结果见表 3-4", "论文核心结果，用于评价同分布留出样本的预测表现"],
    ["历史同期重建", "Chl-a 表层；含同日 TN、TP、浊度、盐度、DO、pH 等水质协变量", "Chl-a 表层 0.7166", "只能说明同期多水质信息下的重建上界，不能与正式结果混合"],
    ["站点独立扩展", "表层；按站点分别训练自回归模型；2023-2024 年测试共 192 条", "TN 0.6951；TP 0.4832；Secchi 0.4438", "说明 TN 存在高精度局地预测潜力；缺失填补须经无泄漏复核后才可升级为正式结果"],
    ["表层统一优化", "表层；融合特征 XGBoost；2023-2024 年测试共 192 条", "TN 0.5980", "作为接近 0.60 的扩展证据，不替代全站点三水层主结果"],
]


def set_font(run, size, bold=False, color=None):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill):
    props = cell._tc.get_or_add_tcPr()
    node = OxmlElement("w:shd")
    node.set(qn("w:fill"), fill)
    props.append(node)


def cell_margin(cell):
    props = cell._tc.get_or_add_tcPr()
    margins = OxmlElement("w:tcMar")
    for side, value in (("top", "80"), ("start", "100"), ("bottom", "80"), ("end", "100")):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), value)
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    props.append(margins)


def add_text(doc, text, *, indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(6)
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    set_font(p.add_run(text), 10.5)


def heading(doc, text, level=1):
    p = doc.add_heading(level=level)
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    set_font(p.add_run(text), {1: 16, 2: 14, 3: 12}.get(level, 11), True)


def caption(doc, text, table=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(7)
    set_font(p.add_run(text), 9.5, table)


def add_figure(doc, item):
    label, name, rel = item
    path = FIG_ROOT / rel
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(15.2))
    caption(doc, f"{label} {name}")


def add_table(doc, item):
    label, name, rel = item
    with (TABLE_ROOT / rel).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    caption(doc, f"{label} {name}", table=True)
    if not rows:
        return
    table = doc.add_table(rows=1, cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_props = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_props.append(repeat)
    for index, value in enumerate(rows[0]):
        cell = table.rows[0].cells[index]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade(cell, "1F4E78")
        cell_margin(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(value), 8.2, True, "FFFFFF")
    for row_index, row in enumerate(rows[1:], 1):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cell = cells[index]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index % 2 == 0:
                shade(cell, "EEF4F8")
            cell_margin(cell)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(value) < 18 else WD_ALIGN_PARAGRAPH.LEFT
            set_font(p.add_run(value), 7.8)
    doc.add_paragraph()


def add_manual_table(doc, title, headers, rows):
    caption(doc, title, table=True)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, value in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade(cell, "1F4E78")
        cell_margin(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(value), 8.2, True, "FFFFFF")
    for row_index, row in enumerate(rows, 1):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cell = cells[index]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index % 2 == 0:
                shade(cell, "EEF4F8")
            cell_margin(cell)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            set_font(p.add_run(value), 7.8)
    doc.add_paragraph()


def chapter(doc, title):
    doc.add_page_break()
    heading(doc, title)


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.2)
    sec.left_margin, sec.right_margin = Cm(2.5), Cm(2.5)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)

    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(80)
    p.paragraph_format.space_after = Pt(20)
    set_font(p.add_run("多源数据融合下香港近岸多水层水质预测及遥感信息敏感性分析"), 22, True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(80)
    set_font(p.add_run("论文草稿"), 16)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("依据当前 full_station 正式主链与论文图表生成\n生成日期：2026 年 9 月 5 日"), 11)

    doc.add_page_break()
    heading(doc, "使用说明")
    add_text(doc, "本稿用于承载当前研究的大纲、章节草稿文字、已完成图表和正式结果表，后续可在此基础上补充文献综述、方法细节和规范参考文献。", indent=False)
    add_text(doc, "第三章正式结果采用固定随机比例留出：每个指标—水层任务在全部有效样本中按 80% 训练、20% 测试，随机种子固定为 42。正式主链不使用同日其他水质指标作为协变量，以保持输入变量边界清晰。", indent=False)
    add_text(doc, "项目中存在更高的 R2，但其任务口径不同：早期 Chl-a 高分来自含同日水质协变量的同期重建；TN 的较高分来自表层站点独立扩展实验。两类结果均在本稿中保留，但不与正式全站点三水层预测结果混合。", indent=False)
    add_text(doc, "第五章包含 Chl-a 表层诊断与分海区扩展分析。连续空间制图尚未完成空间外推验证，因此不作为正式预测结论。", indent=False)
    heading(doc, "目录")
    for item in ("摘要", "第一章 绪论", "第二章 研究区、数据与样本构建", "第三章 多水层水质预测模型构建与正式结果", "第四章 遥感信息敏感性与模型可解释性分析", "第五章 Chl-a 短期扰动诊断、分海区建模与空间制图探索", "第六章 结论与展望", "参考文献", "附录"):
        p = doc.add_paragraph()
        set_font(p.add_run(item), 11)

    doc.add_page_break()
    heading(doc, "摘要")
    add_text(doc, "近岸水质受陆源输入、水动力条件、气象过程和光学性质共同影响，且不同水层的主控机制存在显著差异。以香港近岸 94 个水质监测站点 2014-2024 年观测为基础，本研究围绕叶绿素 a（Chl-a）、透明度（Secchi）、总氮（TN）和总磷（TP）四项指标，构建表层、中层和底层共 12 个预测任务。研究融合站点历史水质、Landsat 与 Sentinel-2 遥感、ERA5 气象与短时间窗辐射和海温、降雨、潮汐及夜间灯光等多源信息，形成以站点-采样日期-水层为基本单元的建模数据集。", indent=False)
    add_text(doc, "在固定随机比例留出框架下，研究比较历史过程特征、遥感特征及融合特征组的预测表现，并通过覆盖度统计、任务级消融、SHAP 和残差诊断评估遥感信息的实际贡献。结果表明，历史过程信息是多水层水质预测的基础，遥感信息的贡献受指标、水层和有效观测质量制约；表层 Chl-a 的短期峰值与局地水动力过程仍是当前预测难点。该评价反映同分布留出样本的表现，不直接代表未来年份外推能力。", indent=False)
    add_text(doc, "关键词：近岸水质；多源数据融合；机器学习；遥感信息；随机留出验证；香港", indent=False)

    chapter(doc, "第一章 绪论")
    heading(doc, "1.1 研究背景与意义", 2)
    add_text(doc, "近岸海域连接陆地输入与外海交换，是人类活动影响最强、生态响应最敏感的水体之一。传统站点监测能够提供可靠的水层观测，但在时间连续性和空间覆盖上存在限制。将站点观测与卫星遥感、再分析气象、降雨和潮汐等过程数据融合，有助于识别水质变化的多尺度驱动，并为精细化监测和管理提供技术支撑。研究同时区分同期水质重建与未来时段预测，避免将高相关性的同日协变量误解为可部署的预测信息。")
    heading(doc, "1.2 国内外研究进展", 2)
    add_text(doc, "现有研究已广泛使用经验模型、半分析反演和机器学习方法预测叶绿素、透明度和营养盐等指标。但对于高浑浊、受潮汐和陆源输入强烈影响的近岸复杂水体，遥感观测质量、采样时空错配和水层不可见性仍限制模型泛化。本文以多水层随机留出验证为统一评价口径，讨论遥感特征在多指标预测中的增益边界。")
    heading(doc, "1.3 研究目标、科学问题与技术路线", 2)
    add_text(doc, "本研究回答三个核心问题：多源驱动能否提高四类水质指标在三水层随机留出样本上的预测表现；遥感信息对不同指标和水层的贡献是否稳定且可解释；面对 Chl-a 的短期峰值与局地机制混杂，哪些诊断与扩展路径具有科学价值。")
    for item in FIGURES["ch1"]: add_figure(doc, item)
    heading(doc, "1.4 论文结构", 2)
    add_text(doc, "第二章说明研究区、数据来源与样本构建；第三章给出正式主链的模型和结果；第四章聚焦遥感敏感性与模型解释；第五章开展 Chl-a 诊断和扩展探索；第六章总结主要结论并提出后续工作。")

    chapter(doc, "第二章 研究区、数据与样本构建")
    heading(doc, "2.1 研究区与监测站点", 2)
    add_text(doc, "研究区为香港近岸海域。监测站点覆盖受河口输入、半封闭港湾和外海交换共同影响的不同水动力环境。为表征垂向差异，观测按表层、中层和底层组织。")
    add_figure(doc, FIGURES["ch2"][0]); add_table(doc, TABLES["ch2"][0])
    heading(doc, "2.2 多源数据与变量体系", 2)
    add_text(doc, "原始数据包括水质站点观测、站点空间位置、Landsat 和 Sentinel-2 表面反射率特征、ERA5 气象和短时间窗过程特征、降雨、潮汐以及夜间灯光等外部数据。不同来源通过站点标识和采样日期对齐，并保留来源与覆盖度字段用于质量控制。")
    add_table(doc, TABLES["ch2"][1])
    for item in FIGURES["ch2"][1:]: add_figure(doc, item)
    for item in TABLES["ch2"][2:]: add_table(doc, item)
    heading(doc, "2.3 样本单元、质量控制与主键", 2)
    add_text(doc, "建模样本以站点-采样日期-水层为基本单元，其中 sample_id 表示站点-日期，obs_id 在此基础上区分水层。对原始观测、遥感有效像元和过程特征覆盖度进行检查，避免将无法解释的缺失值或低质量影像直接视为有效信息。")
    heading(doc, "2.4 本章小结", 2)
    add_text(doc, "本章形成了面向 12 个预测任务的多源观测级建模基础，并明确了样本主键、数据覆盖和质量控制原则。")

    chapter(doc, "第三章 多水层水质预测模型构建与正式结果")
    heading(doc, "3.1 结果解释的口径边界", 2)
    add_text(doc, "本研究已形成不同任务口径下的结果。全站点三水层、固定随机比例留出且不使用同日其他水质变量的结果作为论文正式结果，用于比较同分布样本条件下的模型预测表现。较高的同期重建和站点独立结果并非无效，而是分别回答水质信息的可解释上界和局地预测潜力；必须单列，不应替代正式主结果。")
    add_manual_table(doc, "表 3-5 不同结果口径及其论文定位", ["结果层级", "任务与数据边界", "代表性 R2", "论文定位"], RESULT_SCOPE_ROWS)
    heading(doc, "3.2 预测任务与验证设计", 2)
    add_text(doc, "正式主链将 Chl-a、Secchi、TN 和 TP 分别在表层、中层和底层建模，共形成 12 个任务。每个任务在其全部有效样本中按 80%/20% 随机划分训练集和测试集，随机种子固定为 42。该设计用于稳定量化同一总体分布下的留出预测表现；由于训练与测试样本可能来自相同年份，结论不应解释为未来时段预测。")
    add_figure(doc, FIGURES["ch3"][0]); add_table(doc, TABLES["ch3"][0])
    heading(doc, "3.3 特征组、模型与评价指标", 2)
    add_text(doc, "模型比较历史过程特征、遥感特征和融合特征组。历史滞后特征只使用预测时点之前可获得的同指标信息，不使用同日其他水质指标，以避免信息泄漏。模型效果以 R2、RMSE、MAE 和 Bias 等指标评价，并对不同任务分别汇总。")
    add_table(doc, TABLES["ch3"][1])
    for item in FIGURES["ch3"][1:]: add_figure(doc, item)
    for item in TABLES["ch3"][2:]: add_table(doc, item)
    heading(doc, "3.4 正式结果与讨论", 2)
    add_text(doc, "正式结果应以各任务的随机留出测试指标为准。不同指标和水层的可预测性并不相同：部分表层任务表现较好，而 Chl-a 及中底层任务仍受局地过程和短期扰动限制。结果表用于识别每个任务的当前最优特征组，而不是将单一任务的改善外推为所有任务的共同提升。")
    heading(doc, "3.5 本章小结", 2)
    add_text(doc, "本章建立了不依赖同日水质协变量的随机留出正式基线，同时明确保留同期重建上界与站点独立高精度结果作为扩展证据。该分层使论文能够清晰区分统一总体样本预测、同期重建和局地建模三种不同问题。")

    chapter(doc, "第四章 遥感信息敏感性与模型可解释性分析")
    heading(doc, "4.1 遥感特征质量与覆盖", 2)
    add_text(doc, "遥感特征的适用性首先取决于采样日期邻近影像的有效覆盖和有效像元数量。Landsat 与 Sentinel-2 在重访频率、空间分辨率和质量筛选上各有优势，必须将覆盖度统计与模型性能共同解释。")
    for item in FIGURES["ch4"][:2]: add_figure(doc, item)
    for item in TABLES["ch4"][:2]: add_table(doc, item)
    heading(doc, "4.2 遥感融合增益与任务差异", 2)
    add_text(doc, "通过历史过程模型与融合模型的任务级比较，判断遥感信息的增益是否稳定。遥感融合的价值应同时满足数据质量可接受、性能变化具有一致性、且与目标变量的光学或过程机制相符三个条件。")
    for item in FIGURES["ch4"][2:]: add_figure(doc, item)
    add_table(doc, TABLES["ch4"][2])
    heading(doc, "4.3 SHAP 解释与机制讨论", 2)
    add_text(doc, "SHAP 图用于识别模型在特定特征范围内的响应方向和相对贡献，而非替代因果推断。对 Chl-a 和 Secchi 等光学相关指标，可结合蓝绿红近红外波段、NDCI 和 NDWI 等指数讨论光学响应；对 TN 和 TP，应重点审视遥感特征是否只是环境过程的弱代理。")
    heading(doc, "4.4 本章小结", 2)
    add_text(doc, "本章从覆盖、任务级消融和 SHAP 三个层面界定遥感信息的有效边界，为后续优化优先保留高质量 Landsat 特征并严格筛选 Sentinel-2 提供依据。")

    chapter(doc, "第五章 Chl-a 短期扰动诊断、分海区建模与空间制图探索")
    heading(doc, "5.1 Chl-a 表层短期波动与误差结构", 2)
    add_text(doc, "Chl-a 在近岸复杂水体中易受光照、风浪、潮汐、降雨、营养盐突发输入及再悬浮过程扰动。代表站点时间序列、自相关和高值样本误差用于检验历史惯性是否足以解释当前状态，以及模型是否系统性低估突发高值。")
    for item in FIGURES["ch5"][:4]: add_figure(doc, item)
    for item in TABLES["ch5"][:2]: add_table(doc, item)
    heading(doc, "5.2 分海区建模探索", 2)
    add_text(doc, "香港不同海区在河口影响、半封闭程度和外海交换条件上差异显著。分海区结果作为扩展实验，用于判断 pooled 模型是否受到机制混杂影响。该部分需与全站点正式主链分开报告，并以独立样本量和明确的验证口径为前提解释局部改善。")
    add_figure(doc, FIGURES["ch5"][4]); add_table(doc, TABLES["ch5"][2])
    add_table(doc, PENDING_TABLE)
    heading(doc, "5.3 连续空间制图探索的边界", 2)
    add_text(doc, "已准备的网格预测因子仅用于验证连续空间表达的本地流程。由于当前尚未完成独立空间外推验证、网格化驱动数据的全域覆盖验证和不确定性表达，连续空间制图不进入正式结果。后续应先完成留站点空间外推、像元质量控制和不确定性制图，再将其扩展为论文正式图版。")
    heading(doc, "5.4 本章小结", 2)
    add_text(doc, "Chl-a 的误差结构反映了短期过程和局地机制仍未完全表达。后续优化应优先服务于可验证的过程特征和空间外推，而不是将局部正信号直接并入正式主链。")

    chapter(doc, "第六章 结论与展望")
    heading(doc, "6.1 主要结论", 2)
    add_text(doc, "本研究构建了面向香港近岸 94 个站点、四项水质指标和三水层的多源融合预测框架，形成固定随机比例留出下的统一主表、训练、结果汇总、图表和解释链路。研究明确了历史过程信息、遥感信息和多水层差异在不同任务中的作用边界；未来年份外推仍需另行采用时间阻塞验证。")
    add_table(doc, TABLES["ch6"][0])
    heading(doc, "6.2 研究不足与展望", 2)
    add_text(doc, "当前局限包括近岸遥感的有效覆盖与时相匹配、Chl-a 突发峰值的短期机制、营养盐的非光学可观测性及空间外推验证不足。后续将以高质量近同日遥感、日尺度环境窗口、分海区机制、站点留出验证和连续空间不确定性表达为重点。")
    add_table(doc, TABLES["ch6"][1])
    chapter(doc, "参考文献")
    add_text(doc, "本节待结合文献综述和学校格式要求补充。建议围绕近岸水质遥感反演、机器学习预测、多源数据融合、香港海域水质与时空外推验证等主题建立规范参考文献库。", indent=False)
    heading(doc, "附录")
    add_text(doc, "附录可进一步收录完整变量清单、12 个任务的补充指标、模型参数表、数据字典、GEE 导出说明和额外诊断图。", indent=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
