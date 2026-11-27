"""按福州大学研究生学位论文规范汇编当前随机留出正式结果。"""

from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "结果" / "论文图表" / "论文图版_随机比例划分_20260905"
TABLE_ROOT = ROOT / "结果" / "论文图表" / "论文表格_随机比例划分_20260905"
OUTPUT = ROOT / "文档" / "论文材料" / "福州大学格式论文草稿_香港近岸多水层水质预测_随机比例划分_20260905.docx"

BLACK = RGBColor(0, 0, 0)
CHAPTERS = [
    "第1章 绪论",
    "第2章 研究区 数据与样本构建",
    "第3章 方法与随机留出验证设计",
    "第4章 多水层水质预测结果",
    "第5章 遥感敏感性与不确定性讨论",
    "第6章 结论与展望",
]


def set_font(run, size: float, *, chinese: str = "宋体", latin: str = "Times New Roman", bold: bool = False) -> None:
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), chinese)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = BLACK


def set_page_layout(section) -> None:
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.0)
    section.header_distance = Cm(1.2)
    section.footer_distance = Cm(1.2)


def add_field(run, instruction: str) -> None:
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])


def set_page_number_format(section, fmt: str, start: int) -> None:
    sect_pr = section._sectPr
    page_number = sect_pr.find(qn("w:pgNumType"))
    if page_number is None:
        page_number = OxmlElement("w:pgNumType")
        sect_pr.append(page_number)
    page_number.set(qn("w:fmt"), fmt)
    page_number.set(qn("w:start"), str(start))


def configure_header_footer(section, header_text: str, *, page_field: str = "PAGE") -> None:
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    header.paragraph_format.space_after = Pt(0)
    set_font(header.add_run(header_text), 9, chinese="宋体")
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.space_after = Pt(0)
    page_run = footer.add_run()
    set_font(page_run, 9, chinese="宋体")
    add_field(page_run, page_field)


def set_cell_border(cell, color: str = "000000", size: str = "4") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def shade(cell, fill: str = "E7E6E6") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, value: object, *, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = Pt(16)
    set_font(paragraph.add_run(str(value)), 9, bold=bold)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    set_cell_border(cell)


def set_repeat_table_header(row) -> None:
    """让跨页表格在每一页重复首行字段名。"""
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def add_heading(doc: Document, text: str, level: int) -> None:
    style_name = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3"}[level]
    paragraph = doc.add_paragraph(style=style_name)
    paragraph.paragraph_format.line_spacing = Pt(20)
    paragraph.paragraph_format.space_before = Pt(0 if level == 1 else 12)
    paragraph.paragraph_format.space_after = Pt({1: 32, 2: 21, 3: 12}[level])
    if level == 1:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        size = 18
    elif level == 2:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        size = 16
    else:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        size = 12
    set_font(paragraph.add_run(text), size, chinese="黑体", bold=True)


def add_paragraph(doc: Document, text: str, *, indent: bool = True) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.line_spacing = Pt(20)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    if indent:
        paragraph.paragraph_format.first_line_indent = Cm(0.74)
    set_font(paragraph.add_run(text), 12)


def add_bullet(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.line_spacing = Pt(20)
    paragraph.paragraph_format.space_after = Pt(0)
    set_font(paragraph.add_run(text), 12)


def add_caption(doc: Document, label: str, title: str, *, above: bool):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = Pt(16)
    paragraph.paragraph_format.space_before = Pt(6 if above else 0)
    paragraph.paragraph_format.space_after = Pt(6)
    set_font(paragraph.add_run(f"{label} {title}"), 10.5)
    return paragraph


def add_figure(doc: Document, label: str, title: str, relative_path: str) -> None:
    path = FIG_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"缺少图件: {path}")
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.keep_together = True
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Cm(15.1))
    caption = add_caption(doc, label, title, above=False)
    caption.paragraph_format.keep_together = True


def add_table(doc: Document, label: str, title: str, relative_path: str) -> None:
    path = TABLE_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"缺少表格: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return
    add_caption(doc, label, title, above=True)
    table = doc.add_table(rows=1, cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    set_repeat_table_header(table.rows[0])
    for cell, value in zip(table.rows[0].cells, rows[0]):
        shade(cell)
        set_cell_text(cell, value, bold=True)
    for source_row in rows[1:]:
        cells = table.add_row().cells
        for cell, value in zip(cells, source_row):
            set_cell_text(cell, value)


def add_toc(doc: Document) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(24)
    set_font(paragraph.add_run("目 录"), 18, chinese="黑体", bold=True)
    field = doc.add_paragraph()
    field.paragraph_format.line_spacing = Pt(20)
    run = field.add_run()
    set_font(run, 12)
    add_field(run, 'TOC \\o "1-3" \\h \\z \\u')
    add_paragraph(doc, "提示：在 Word 中全选后按 F9 更新目录和页码。", indent=False)


def configure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(12)
    normal.font.color.rgb = BLACK
    for style_name in ("Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[style_name]
        style.font.color.rgb = BLACK
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")


def add_cover(doc: Document) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_font(paragraph.add_run("分类号：__________\n学校代码：10386\n密级：公开\n学号：__________"), 12)
    for _ in range(2):
        doc.add_paragraph()
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(paragraph.add_run("福 州 大 学"), 28, chinese="黑体", bold=True)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(paragraph.add_run("硕 士 学 位 论 文"), 22, chinese="黑体", bold=True)
    for _ in range(2):
        doc.add_paragraph()
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = Pt(28)
    set_font(paragraph.add_run("多源数据融合下香港近岸多水层水质预测\n及遥感信息敏感性分析"), 22, chinese="黑体", bold=True)
    for _ in range(2):
        doc.add_paragraph()
    info = [
        "学位类别：________________",
        "学科专业：________________",
        "研究方向：________________",
        "作者姓名：________________",
        "指导教师：________________",
        "完成日期：2026 年 9 月",
    ]
    for line in info:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(paragraph.add_run(line), 14)


def add_declarations(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, "独创性声明", 1)
    add_paragraph(doc, "本人声明所呈交的学位论文是在导师指导下独立完成的研究成果。除文中已经注明引用的内容外，本论文不包含任何他人已经发表或撰写过的研究成果。正式送审前，请按学校统一模板填写并签署本页。")
    add_paragraph(doc, "作者签名：________________    日期：________________", indent=False)
    add_heading(doc, "学位论文使用授权声明", 1)
    add_paragraph(doc, "本人同意按照福州大学有关规定保存、使用和公开本学位论文。正式送审前，请替换为学校统一授权书并完成签署。")
    add_paragraph(doc, "作者签名：________________    导师签名：________________", indent=False)


def add_symbols(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, "主要符号表", 1)
    rows = [
        ("符号", "含义", "单位"),
        ("Chl-a", "叶绿素 a", "μg/L"),
        ("Secchi", "透明度", "m"),
        ("TN", "总氮", "mg/L"),
        ("TP", "总磷", "mg/L"),
        ("R2", "决定系数", "-"),
        ("RMSE", "均方根误差", "与目标变量相同"),
        ("MAE", "平均绝对误差", "与目标变量相同"),
    ]
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    set_repeat_table_header(table.rows[0])
    for cell, value in zip(table.rows[0].cells, rows[0]):
        shade(cell)
        set_cell_text(cell, value, bold=True)
    for row in rows[1:]:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            set_cell_text(cell, value)


def add_abstracts(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, "摘 要", 1)
    add_paragraph(doc, "香港近岸海域受陆源输入、潮汐交换、气象扰动和人类活动共同影响，水质过程具有显著的多水层差异和时空异质性。以 2014—2024 年香港近岸 94 个监测站点的水质观测为基础，本文针对叶绿素 a（Chl-a）、透明度（Secchi）、总氮（TN）和总磷（TP）构建表层、中层和底层共 12 个预测任务。研究融合站点空间属性、目标变量历史滞后、ERA5 气象及短时间窗辐射和海温、降雨、潮汐、夜间灯光和 Landsat 遥感信息，形成以站点—日期—水层为单元的多源建模样本。")
    add_paragraph(doc, "正式评价采用固定随机比例留出设计：每个指标—水层任务将全部有效样本按 80% 训练、20% 测试随机划分，随机种子为 42；同日其他水质指标不作为输入。XGBoost 结果表明，TN 表层、底层的最佳 R2 分别为 0.798 和 0.602，TP 表层和中层分别为 0.725 和 0.582，Secchi 表层为 0.616；Chl-a 表层、中层和底层分别为 0.465、0.403 和 0.374。历史过程特征是模型性能的主要来源，遥感单独输入表现较弱，而融合遥感在部分任务上呈现小幅增益。")
    add_paragraph(doc, "研究表明，多源数据融合能够支撑香港近岸多指标、多水层水质的统一随机留出预测比较，并揭示不同指标和水层的可预测性差异。由于随机留出样本可能与训练样本来自相同年份，本文结果反映同分布样本预测表现，不应直接解释为未来年份外推能力。")
    add_paragraph(doc, "关键词：近岸水质；多源数据融合；随机留出验证；机器学习；遥感敏感性", indent=False)

    doc.add_page_break()
    add_heading(doc, "Abstract", 1)
    add_paragraph(doc, "Coastal water quality in Hong Kong is jointly influenced by terrestrial inputs, tidal exchange, meteorological forcing, and human activities. Based on observations from 94 monitoring stations during 2014-2024, this study develops twelve tasks for chlorophyll-a, Secchi depth, total nitrogen, and total phosphorus across surface, middle, and bottom layers. The model integrates station attributes, target-specific historical lags, ERA5 process variables, rainfall, tide, nighttime lights, and Landsat features.", indent=False)
    add_paragraph(doc, "The formal evaluation uses a fixed random holdout split, with 80 percent of valid samples used for training and 20 percent for testing under random_state 42. Same-day water-quality variables are excluded. The best R2 values reach 0.798 for surface TN, 0.725 for surface TP, and 0.616 for surface Secchi depth. Historical process features provide the main predictive information, while remote-sensing features offer limited task-specific gains. These results describe prediction performance on holdout samples from the same overall distribution and should not be interpreted as future-year extrapolation performance.", indent=False)
    add_paragraph(doc, "Key words: coastal water quality; multi-source data fusion; random holdout validation; machine learning; remote sensing sensitivity", indent=False)


def add_chapter_1(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, CHAPTERS[0], 1)
    add_heading(doc, "1.1 研究背景与意义", 2)
    add_paragraph(doc, "近岸海域连接陆地输入与外海交换，是水环境变化最敏感的区域之一。站点监测能够提供可靠的垂向水质信息，但难以同时满足高频、连续和全域覆盖要求。融合水质观测、遥感和环境过程数据，有助于比较不同水层和不同指标的预测信息来源。")
    add_heading(doc, "1.2 研究问题与目标", 2)
    add_bullet(doc, "多源过程信息能否提高四项指标在三水层随机留出样本上的预测表现？")
    add_bullet(doc, "遥感特征在不同指标和水层中是否提供稳定且可解释的增益？")
    add_bullet(doc, "Chl-a 高值波动、营养盐差异和水层不可见性如何限制模型表现？")
    add_heading(doc, "1.3 研究内容与技术路线", 2)
    add_paragraph(doc, "论文围绕数据构建、随机留出预测比较、遥感敏感性分析和误差讨论展开。连续空间制图仅作为后续探索，不作为本论文正式精度结论。")
    add_figure(doc, "图1-1", "研究技术路线", "第一二章/图1-1_研究技术路线.png")
    add_figure(doc, "图1-2", "科学问题与章节关系", "第一二章/图1-2_科学问题与章节关系.png")


def add_chapter_2(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, CHAPTERS[1], 1)
    add_heading(doc, "2.1 研究区与监测网络", 2)
    add_paragraph(doc, "研究区覆盖香港近岸 94 个水质监测站点，并按表层、中层和底层建立观测级样本。海区差异、垂向过程和采样覆盖共同构成模型解释的重要背景。")
    add_figure(doc, "图2-1", "香港近岸监测站点与海区分布", "第一二章/图2-1_监测站点与海区分布.png")
    add_table(doc, "表2-1", "海区与站点空间覆盖统计", "第一二章/表2-1_海区与站点空间覆盖统计.csv")
    add_heading(doc, "2.2 数据来源与样本构建", 2)
    add_table(doc, "表2-2", "多源数据来源与用途", "第一二章/表2-2_多源数据来源与用途.csv")
    add_figure(doc, "图2-2", "三水层观测覆盖", "第一二章/图2-2_三水层观测覆盖.png")
    add_figure(doc, "图2-3", "四项水质指标的三水层分布", "第一二章/图2-3_四项水质指标三水层分布.png")
    add_table(doc, "表2-3", "四项目标变量描述性统计", "第一二章/表2-3_四项目标变量描述性统计.csv")
    add_figure(doc, "图2-4", "水质观测时间覆盖", "第一二章/图2-4_水质观测时间覆盖.png")
    add_heading(doc, "2.3 时空匹配与质量控制", 2)
    add_paragraph(doc, "以 sample_id 表示站点—日期，以 obs_id 表示站点—日期—水层。外部数据通过明确的站点和日期键匹配到观测样本；同日其他水质指标不进入正式模型特征。")
    add_figure(doc, "图2-5", "多源数据接入与样本主键关系", "第一二章/图2-5_多源数据接入与样本主键.png")
    add_table(doc, "表2-4", "遥感与短时间窗过程特征覆盖统计", "第一二章/表2-4_遥感与短时间窗过程特征覆盖统计.csv")


def add_chapter_3(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, CHAPTERS[2], 1)
    add_heading(doc, "3.1 预测任务与特征组", 2)
    add_paragraph(doc, "Chl-a、Secchi、TN 和 TP 分别在三水层建模，形成 12 个预测任务。特征组包括基线组、历史与过程组、遥感组和历史—遥感融合组，用于分离不同信息源的作用。")
    add_table(doc, "表3-1", "十二个预测任务与样本量", "第三四章/表3-1_十二个预测任务与样本量.csv")
    add_table(doc, "表3-2", "正式模型特征组定义", "第三四章/表3-2_正式模型特征组定义.csv")
    add_heading(doc, "3.2 固定随机比例留出设计", 2)
    add_paragraph(doc, "每个任务在其全部有效样本中按 80%/20% 随机划分训练集和测试集，随机种子固定为 42。该设计用于量化同分布留出样本的预测表现，不等同于对未来年份的时间外推验证。")
    add_figure(doc, "图3-1", "固定随机比例的训练集与测试集划分", "第三四章/图3-1_随机比例划分.png")
    add_heading(doc, "3.3 模型与评价指标", 2)
    add_paragraph(doc, "各任务采用 XGBoost 回归模型，目标变量在非负条件下进行 log1p 变换。模型效果使用 R2、RMSE、MAE 和 NSE 评价，并按指标、水层和特征组分别报告。")


def add_chapter_4(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, CHAPTERS[3], 1)
    add_heading(doc, "4.1 总体结果与最佳模型", 2)
    add_paragraph(doc, "12 个任务的最佳 R2 呈现显著指标和水层差异。TN 表层达到 0.798，TP 表层达到 0.725，Secchi 表层达到 0.616；Chl-a 三水层结果相对较低，表明其对短期扰动和局地过程更敏感。")
    add_figure(doc, "图4-1", "十二个任务的随机留出预测性能热图", "第三四章/图3-2_十二个任务预测指标热图.png")
    add_table(doc, "表4-1", "十二个任务的正式随机留出结果", "第三四章/表3-3_十二个任务的正式随机留出结果.csv")
    add_table(doc, "表4-2", "各任务最佳正式模型", "第三四章/表3-4_各任务最佳正式模型.csv")
    add_heading(doc, "4.2 特征组比较", 2)
    add_paragraph(doc, "历史与过程组和融合组的平均表现接近，均优于基线组；遥感单独组总体较弱。因此，遥感信息在本研究中应理解为条件性补充，而非独立替代过程信息。")
    add_figure(doc, "图4-2", "不同特征组的总体随机留出性能比较", "第三四章/图3-3_特征组总体性能比较.png")
    add_figure(doc, "图4-3", "最佳正式模型的观测值与预测值对比", "第三四章/图3-5_最佳正式模型预测散点图.png")
    add_heading(doc, "4.3 指标与水层差异", 2)
    add_paragraph(doc, "TN 和 TP 的表层任务在随机留出条件下表现较好；中层和底层结果则更受垂向过程和观测信息不足限制。Chl-a 的预测误差反映其对短期光照、风浪、潮汐和局地水动力扰动的敏感性。")


def add_chapter_5(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, CHAPTERS[4], 1)
    add_heading(doc, "5.1 遥感信息覆盖与融合增益", 2)
    add_paragraph(doc, "遥感质量和字段覆盖决定了其能否进入正式模型。当前正式主表保留 Landsat 变量，但未包含 Sentinel-2 的 s2_ 字段，因此本章仅解释实际进入模型的遥感变量。")
    add_figure(doc, "图5-1", "正式主表遥感质量覆盖", "第三四章/图4-1_正式主表遥感质量覆盖.png")
    add_table(doc, "表5-1", "遥感与短时间窗过程特征质量覆盖", "第三四章/表4-1_遥感与短时间窗过程特征质量覆盖.csv")
    add_figure(doc, "图5-2", "融合模型中的遥感特征构成", "第三四章/图4-2_融合模型遥感特征构成.png")
    add_table(doc, "表5-2", "融合模型遥感特征构成", "第三四章/表4-2_融合模型遥感特征构成.csv")
    add_figure(doc, "图5-3", "遥感融合的任务级增益判定", "第三四章/图4-5_遥感融合任务级增益判定.png")
    add_figure(doc, "图5-4", "不同水层的遥感融合增益分布", "第三四章/图4-6_不同水层遥感增益分布.png")
    add_table(doc, "表5-3", "遥感融合增益判定", "第三四章/表4-3_遥感融合增益判定.csv")
    add_heading(doc, "5.2 遥感特征可解释性", 2)
    add_paragraph(doc, "采用 SHAP 解释融合模型中的实际可用遥感特征。结果用于识别变量在当前样本和当前模型中的相对贡献，不应将其解释为遥感对中层和底层水质的直接观测。")
    add_figure(doc, "图5-5", "实际可用遥感特征的 SHAP 重要性", "第三四章/图4-3_遥感特征SHAP重要性.png")
    add_figure(doc, "图5-6", "实际可用遥感主要特征的 SHAP 依赖关系", "第三四章/图4-4_遥感主要特征SHAP依赖关系.png")
    add_heading(doc, "5.3 Chl-a 误差结构与研究局限", 2)
    add_paragraph(doc, "Chl-a 表层诊断用于说明高值和突变样本的误差来源。该部分为正式随机留出结果的误差分析，不等同于分海区模型或连续空间制图的精度结论。")
    add_figure(doc, "图5-7", "Chl-a 表层代表站点时间序列", "第五章/图5-1_Chla表层代表站点时间序列.png")
    add_figure(doc, "图5-8", "Chl-a 表层观测值与预测值散点图", "第五章/图5-2_Chla表层观测值预测值散点图.png")
    add_figure(doc, "图5-9", "Chl-a 表层月尺度残差变化", "第五章/图5-3_Chla表层月尺度残差变化.png")
    add_figure(doc, "图5-10", "Chl-a 表层主要特征重要性", "第五章/图5-4_Chla表层主要特征重要性.png")
    add_table(doc, "表5-4", "Chl-a 表层自相关与峰值波动统计", "第五章/表5-1_Chla表层自相关与峰值波动统计.csv")
    add_table(doc, "表5-5", "Chl-a 表层高值样本误差统计", "第五章/表5-2_Chla表层高值样本误差统计.csv")


def add_chapter_6(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, CHAPTERS[5], 1)
    add_heading(doc, "6.1 主要结论", 2)
    add_paragraph(doc, "本文建立了覆盖 94 个站点、四项指标和三水层的多源融合预测流程，并在固定随机留出条件下形成统一的模型、汇总、图表和解释链路。TN、TP 和 Secchi 的部分任务达到较高 R2，而 Chl-a 仍是受短期过程影响较强的困难任务。")
    add_table(doc, "表6-1", "主要结论与对应证据", "第六章/表6-1_主要结论与对应证据.csv")
    add_heading(doc, "6.2 局限与展望", 2)
    add_paragraph(doc, "当前结论的适用边界包括：随机留出不能替代未来年份外推验证；中层和底层难以由表面遥感直接表征；Sentinel-2 字段尚未进入当前正式主表。后续应补充采样时刻、河流径流和垂向水动力数据，并开展时间阻塞和留站点空间外推验证。")
    add_table(doc, "表6-2", "数据缺口与后续改进方向", "第六章/表6-2_数据缺口与后续改进方向.csv")
    add_heading(doc, "参考文献", 1)
    add_paragraph(doc, "本节待结合文献综述和学校参考文献著录要求补充。参考文献应仅列出正文直接引用的正式公开文献，并按正文出现顺序连续编号。", indent=False)
    add_heading(doc, "附录", 1)
    add_paragraph(doc, "附录建议收录完整变量清单、12 个任务补充指标、模型参数、数据字典、GEE 导出说明、随机划分复现实验参数及未进入正式主链的扩展实验结果。", indent=False)
    add_heading(doc, "致谢", 1)
    add_paragraph(doc, "致谢内容待作者根据实际情况补充。", indent=False)


def build() -> None:
    doc = Document()
    configure_styles(doc)
    set_page_layout(doc.sections[0])
    doc.sections[0].different_first_page_header_footer = True
    configure_header_footer(doc.sections[0], "福州大学硕士学位论文", page_field="PAGE \\* ROMAN")
    set_page_number_format(doc.sections[0], "upperRoman", 1)

    add_cover(doc)
    add_declarations(doc)
    add_abstracts(doc)
    add_toc(doc)
    add_symbols(doc)

    body = doc.add_section(WD_SECTION.NEW_PAGE)
    set_page_layout(body)
    configure_header_footer(body, "福州大学硕士学位论文")
    set_page_number_format(body, "decimal", 1)
    add_chapter_1(doc)
    add_chapter_2(doc)
    add_chapter_3(doc)
    add_chapter_4(doc)
    add_chapter_5(doc)
    add_chapter_6(doc)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
