"""生成硕士论文扩展审阅版：补足章节、图表目录和图表审阅评论。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import win32com.client
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "文档" / "论文材料" / "福州大学格式论文草稿_香港近岸多水层水质预测_数据架构扩展版_20260906.docx"
OUTPUT = ROOT / "文档" / "论文材料" / "福州大学格式论文草稿_香港近岸多水层水质预测_硕士论文扩展审阅版_20260907.docx"
BLACK = RGBColor(0, 0, 0)

FORMAL_DIR = ROOT / "结果" / "正式结果" / "full_station"
FIG_DIR = ROOT / "结果" / "论文图表"
DIAG_DIR = ROOT / "结果" / "诊断结果"

FIGURES = {
    "图2-6": DIAG_DIR / "水质时空分异与水团背景_20260907" / "图1_四指标分水区季节水层中位数热图.png",
    "图2-7": DIAG_DIR / "水质时空分异与水团背景_20260907" / "图2_表底层配对差异.png",
    "图2-8": DIAG_DIR / "水质时空分异与水团背景_20260907" / "图4_表层观测水质状态聚类.png",
    "图4-6": FIG_DIR / "full_station_figures" / "fig01a_正式模型性能热图_透明度单列.png",
    "图4-7": FIG_DIR / "full_station_figures" / "fig04a_代表性目标观测预测散点图.png",
    "图5-11": DIAG_DIR / "遥感质量与光学类型_20260907" / "图1_遥感质量条件下的R2变化.png",
    "图5-12": DIAG_DIR / "遥感质量与光学类型_20260907" / "图2_严格筛选S2光学观测类型.png",
    "图5-13": DIAG_DIR / "Chl-a高值与误差结构_20260907" / "图1_高值样本观测预测散点图.png",
    "图5-14": DIAG_DIR / "Chl-a高值与误差结构_20260907" / "图2_高值低估与绝对误差.png",
    "图5-15": DIAG_DIR / "Chl-a高值与误差结构_20260907" / "图3_surface_高值低估水区季节热图.png",
    "图5-16": DIAG_DIR / "Chl-a高值与误差结构_20260907" / "图3_middle_高值低估水区季节热图.png",
    "图5-17": DIAG_DIR / "Chl-a高值与误差结构_20260907" / "图3_bottom_高值低估水区季节热图.png",
    "图5-18": DIAG_DIR / "Chl-a高值与误差结构_20260907" / "图4_高值区间覆盖率.png",
    "图5-19": DIAG_DIR / "空间泛化边界_20260907" / "图1_五折留站点任务R2变化.png",
    "图5-20": DIAG_DIR / "空间泛化边界_20260907" / "图3_时间与留站点验证R2变化.png",
}


def set_font(run, size: float = 12, *, chinese: str = "宋体", bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), chinese)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = BLACK


def set_cell_border(cell, color: str = "D9D9D9") -> None:
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
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), color)


def shade(cell, fill: str = "E7E6E6") -> None:
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(element)


def find_paragraph(doc: Document, starts_with: str):
    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith(starts_with):
            return paragraph
    raise KeyError(f"未找到段落：{starts_with}")


def format_paragraph(paragraph, *, heading: bool = False) -> None:
    paragraph.paragraph_format.line_spacing = Pt(20)
    paragraph.paragraph_format.space_after = Pt(3)
    if not heading:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        paragraph.paragraph_format.first_line_indent = Cm(0.74)
        for run in paragraph.runs:
            set_font(run)
    else:
        paragraph.paragraph_format.space_before = Pt(12)
        paragraph.paragraph_format.space_after = Pt(10)
        for run in paragraph.runs:
            set_font(run, 14, chinese="黑体", bold=True)


def insert_heading_before(anchor, text: str, level: int = 2):
    paragraph = anchor.insert_paragraph_before(text, style=f"Heading {level}")
    format_paragraph(paragraph, heading=True)
    return paragraph


def insert_text_before(anchor, text: str):
    paragraph = anchor.insert_paragraph_before(text)
    paragraph.style = "Normal"
    format_paragraph(paragraph)
    return paragraph


def insert_caption_before(anchor, text: str, *, above: bool = False):
    paragraph = anchor.insert_paragraph_before(text)
    paragraph.style = "Normal"
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = Pt(16)
    paragraph.paragraph_format.space_before = Pt(6 if above else 0)
    paragraph.paragraph_format.space_after = Pt(6)
    for run in paragraph.runs:
        set_font(run, 10.5)
    return paragraph


def insert_figure_before(anchor, figure_id: str, caption: str, width_cm: float = 14.8) -> None:
    path = FIGURES[figure_id]
    if not path.exists():
        raise FileNotFoundError(path)
    paragraph = anchor.insert_paragraph_before()
    paragraph.style = "Normal"
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_together = True
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
    insert_caption_before(anchor, f"{figure_id} {caption}")


def add_table_before(doc: Document, anchor, caption: str, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    insert_caption_before(anchor, caption, above=True)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.width = Cm(widths[i])
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(paragraph.add_run(text), 9, bold=True)
        shade(cell)
        set_cell_border(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cell = cells[i]
            cell.width = Cm(widths[i])
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.line_spacing = Pt(13)
            set_font(paragraph.add_run(str(text)), 8.5)
            set_cell_border(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    anchor._p.addprevious(table._tbl)


def replace_paragraph_starting(doc: Document, starts_with: str, replacement: str) -> None:
    paragraph = find_paragraph(doc, starts_with)
    paragraph.text = replacement
    format_paragraph(paragraph)


def remove_redundant_break_before_chapter_one(doc: Document) -> None:
    """原草稿的手工分页与一级标题样式分页叠加，会生成空白页。"""
    paragraphs = list(doc.paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if paragraph.text.strip().startswith("第1章 绪论") and index:
            previous = paragraphs[index - 1]
            for br in previous._p.xpath('.//w:br[@w:type="page"]'):
                br.getparent().remove(br)
            return


def insert_text_after(anchor, text: str):
    """在标题后添加一段简短导读，不改变标题层级和编号。"""
    element = OxmlElement("w:p")
    anchor._p.addnext(element)
    paragraph = Paragraph(element, anchor._parent)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.line_spacing = Pt(18)
    paragraph.paragraph_format.space_after = Pt(5)
    set_font(paragraph.add_run(text), 10.5)
    return paragraph


def add_heading_guides(doc: Document) -> None:
    """为正文的每个章节标题补充简洁的写作导读。"""
    guides = {
        "摘要": "本节概括全文。主要交代研究对象、方法、结果和结论。它帮助读者快速判断本文的重点。",
        "摘 要": "本节概括全文。主要交代研究对象、方法、结果和结论。它帮助读者快速判断本文的重点。",
        "Abstract": "This section gives a brief overview of the study. It states the data, method, main results, and conclusion. It helps readers identify the focus of the thesis.",
        "图表目录": "本节列出全文图表。主要用于核对图表与章节内容是否对应。它方便写作和审阅时查找证据。",
        "主要符号表": "本节说明常用符号。主要列出变量和指标的含义。它减少后文阅读中的歧义。",
        "第1章": "本章说明研究从哪里出发。主要介绍研究问题、目标和技术路线。它为后文的方法和结果定下范围。",
        "1.1": "本节说明为什么要关注香港近岸水质。主要介绍近岸水体的实际问题。它引出本文的研究价值。",
        "1.2": "本节把研究问题说清楚。主要明确预测对象和需要回答的几个问题。它避免后文的分析偏离主题。",
        "1.3": "本节交代本文怎么做。主要概述数据、模型和验证安排。它帮助读者先看懂全文结构。",
        "1.4": "本节概括本文已经完成的工作。主要说明数据整理、模型比较和补充分析。它对应后文各章的内容。",
        "第2章": "本章说明数据从哪里来、样本怎样形成。主要介绍研究区、站点、观测和质量控制。它为建模结果提供数据基础。",
        "2.1": "本节介绍研究区和站点分布。主要说明水区、站点和水层覆盖。它解释后文为什么要讨论空间差异。",
        "2.2": "本节说明使用了哪些数据。主要介绍水质、遥感、气象和辅助数据的来源。它保证样本构建过程清楚可查。",
        "2.3": "本节说明不同数据怎样按站点和日期对应。主要介绍匹配规则和质量控制。它保证输入数据可用于后续比较。",
        "2.4": "本节先看水质本身的变化。主要比较水区、季节和水层的差异。它帮助解释模型结果为什么不相同。",
        "2.5": "本节补充表层水质的局地情况。主要归纳几类常见观测状态。它为后面的遥感和高值讨论提供背景。",
        "2.6": "本节归纳本章的基础发现。主要总结样本结构和水质差异。它承接下一章的特征和模型设计。",
        "第3章": "本章说明模型怎样训练和比较。主要介绍任务、特征、数据划分和评价指标。它保证不同结果可以放在同一口径下比较。",
        "3.1": "本节定义预测任务和输入信息。主要说明四类特征组的内容。它决定后续比较回答什么问题。",
        "3.2": "本节说明正式结果怎样划分训练集和测试集。主要介绍固定随机比例留出。它用于得到可重复的主结果。",
        "3.3": "本节介绍使用的模型和评价指标。主要说明 XGBoost 与 R2、RMSE、MAE 的计算。它给出结果比较的共同尺度。",
        "3.4": "本节介绍分海区的补充检验。主要说明五折留站点的设置。它用来观察模型面对新站点时的表现。",
        "3.5": "本节区分不同验证结果的含义。主要说明随机留出、留站点和留年份各自回答什么。它避免把不同难度的结果混在一起。",
        "3.6": "本节总结建模方法。主要归纳训练、比较和验证的安排。它引出下一章的正式结果。",
        "第4章": "本章报告正式模型结果。主要比较不同指标、水层和特征组。它回答当前数据下哪些任务预测得更好。",
        "4.1": "本节给出总体结果。主要列出各任务的最佳模型和指标。它是全文结果比较的起点。",
        "4.2": "本节比较不同特征组。主要看历史过程信息和遥感信息带来的变化。它回答不同数据源是否有用。",
        "4.3": "本节比较指标和水层的差别。主要解释哪些目标较稳定，哪些目标更难预测。它为第5章的深入分析提供线索。",
        "4.4": "本节展示分海区的留站点结果。主要比较全域模型和分区模型。它补充正式模型的空间表现。",
        "4.5": "本节单列说明透明度结果。主要处理三层记录重复的问题。它保证透明度不会被误写成三项独立证据。",
        "4.6": "本节归纳正式结果。主要总结各指标的表现和主要信息来源。它连接后面的诊断讨论。",
        "第5章": "本章解释模型在哪些条件下有效，在哪些条件下容易出错。主要讨论遥感质量、高值误差和空间外推。它补足单一 R2 看不到的内容。",
        "5.1": "本节检查遥感数据的覆盖情况。主要比较不同传感器和融合特征的作用。它说明遥感信息的实际贡献。",
        "5.2": "本节说明模型主要使用了哪些遥感信息。主要结合特征重要性讨论光学变量。它帮助解释遥感结果。",
        "5.3": "本节分析 Chl-a 的误差。主要看高值和不同条件下的偏差。它说明 Chl-a 为什么比其他指标更难预测。",
        "5.4": "本节梳理已经完成的补充工作。主要说明哪些内容进入正式结果，哪些只作分析。它保持全文结果层次清楚。",
        "5.5": "本节检查遥感质量是否影响模型。主要比较严格筛选前后的结果。它回答遥感在什么条件下更值得使用。",
        "5.6": "本节专门看 Chl-a 高值。主要统计高值低估和漏报情况。它补充总体误差指标看不到的问题。",
        "5.7": "本节检查分海区模型能否推广。主要比较留站点和留年份结果。它给出分区模型的适用范围。",
        "5.8": "本节归纳补充分析。主要总结遥感质量、高值和空间外推的发现。它为最后的结论提供依据。",
        "第6章": "本章收束全文。主要归纳主要发现、适用范围和下一步工作。它回答本文最终得到什么结论。",
        "6.1": "本节列出主要结论。主要对应数据、模型和诊断结果。它把分散证据汇成清晰结论。",
        "6.2": "本节说明还存在什么问题。主要提出数据和验证上的下一步工作。它指出研究可以继续完善的方向。",
        "参考文献": "本节列出本文引用的资料。主要说明研究依据和数据方法来源。它支撑论文的学术规范。",
        "附录": "本节放置正文未展开的补充材料。主要包括完整表格和辅助结果。它保证正文简洁，同时保留可查证信息。",
        "致谢": "本节说明研究得到的帮助。主要感谢指导、数据和协作支持。它完成论文的必要组成部分。",
    }
    exact_titles = {"摘要", "摘 要", "Abstract", "图表目录", "主要符号表", "参考文献", "附录", "致谢"}
    for paragraph in list(doc.paragraphs):
        title = paragraph.text.strip()
        guide = next(
            (
                text
                for prefix, text in guides.items()
                if (title == prefix if prefix in exact_titles else title.startswith(prefix))
            ),
            None,
        )
        if guide:
            insert_text_after(paragraph, guide)


def figure_directory_rows() -> list[list[str]]:
    return [
        ["图1-1", "多源数据融合下水质预测技术路线", "第1章"],
        ["图2-1至图2-5", "研究区、站点、水层覆盖、变量分布、主键关系", "第2章"],
        ["图2-6至图2-8", "水区季节水层背景、垂向配对差异、观测水质状态", "第2章"],
        ["图3-1", "固定随机比例留出示意", "第3章"],
        ["图4-1至图4-5", "正式预测性能、特征组、散点与留站点比较", "第4章"],
        ["图4-6至图4-7", "透明度单列的正式性能与四目标代表性散点", "第4章"],
        ["图5-1至图5-10", "遥感覆盖、融合增益、SHAP 与 Chl-a 基础诊断", "第5章"],
        ["图5-11至图5-12", "遥感质量条件与严格 S2 光学类型", "第5章"],
        ["图5-13至图5-18", "Chl-a 高值误差、水区季节结构与区间覆盖", "第5章"],
        ["图5-19至图5-20", "五折留站点与时间留出空间泛化边界", "第5章"],
    ]


def table_directory_rows() -> list[list[str]]:
    return [
        ["表2-1至表2-4", "水区覆盖、数据来源、目标统计、质量覆盖", "第2章"],
        ["表2-5至表2-6", "同日变量的分析边界与表层观测状态", "第2章"],
        ["表3-1至表3-3", "任务样本量、特征组和验证设计", "第3章"],
        ["表4-1至表4-4", "正式结果、最佳模型、空间扩展和透明度单列汇总", "第4章"],
        ["表5-1至表5-6", "遥感增益、Chl-a 诊断和扩展工作定位", "第5章"],
        ["表5-7至表5-9", "严格 S2 覆盖、高值误差、空间泛化判定", "第5章"],
        ["表6-1", "主要结论与证据来源", "第6章"],
    ]


def insert_front_matter(doc: Document) -> None:
    anchor = find_paragraph(doc, "主要符号表")
    insert_heading_before(anchor, "图表目录", level=1)
    insert_text_before(anchor, "目录按章节排列。定稿时可在 Word 中统一更新编号和页码。")
    add_table_before(doc, anchor, "图表目录之一 图目录", ["编号", "图名或图组", "拟放章节"], figure_directory_rows(), [2.7, 9.2, 3.1])
    add_table_before(doc, anchor, "图表目录之二 表目录", ["编号", "表名或表组", "拟放章节"], table_directory_rows(), [2.7, 9.2, 3.1])


def insert_chapter_two_content(doc: Document) -> None:
    anchor = find_paragraph(doc, "第3章 方法与随机留出验证设计")
    insert_heading_before(anchor, "2.4 香港近岸水质时空分异与垂向背景")
    insert_text_before(anchor, "本节先看原始水质观测的基本差异。使用水温、盐度、浊度、悬浮物、无机氮、正磷酸盐、溶解氧和 pH 说明水区、季节和水层的背景。这些同日水质变量只用于分析，不放入预测模型。")
    insert_figure_before(anchor, "图2-6", "水区 季节与水层条件下的水质中位数分异", 14.2)
    insert_text_before(anchor, "Chl-a、TN 和 TP 的三层分布显示出不同的水区和季节组合特征。透明度在原始三层记录中完全一致，应按站点—日期尺度解释，不用于垂向差异推断。")
    insert_figure_before(anchor, "图2-7", "表层与底层水质的配对差异 不含透明度", 13.8)
    add_table_before(
        doc,
        anchor,
        "表2-5 原始水质变量在论文中的使用边界",
        ["变量类别", "主要字段示例", "论文用途", "进入预测模型"],
        [
            ["预测目标", "Chl-a、TN、TP、Secchi", "正式预测与性能比较", "是"],
            ["同日水质背景", "盐度、浊度、悬浮物、TIN、正磷酸盐、DO、pH", "时空分异和机制背景", "否"],
            ["外部过程", "ERA5、降雨、潮汐、夜间灯光、Landsat", "过程驱动与遥感融合", "是"],
        ],
        [2.5, 5.5, 4.5, 2.0],
    )
    insert_heading_before(anchor, "2.5 表层观测水质状态与局地背景")
    insert_text_before(anchor, "本节用盐度、浊度、悬浮物、TIN、正磷酸盐、Chl-a 和透明度描述表层观测。结果分为两类常见状态。这个分类只用于说明近岸背景，不参与模型训练。")
    insert_figure_before(anchor, "图2-8", "表层观测水质状态的特征与空间分布", 14.0)
    add_table_before(
        doc,
        anchor,
        "表2-6 表层观测水质状态的解释边界",
        ["状态", "主要观测特征", "论文中可使用的解释", "不可使用的解释"],
        [
            ["状态1", "较高盐度、较低浊度与较低营养盐", "近岸背景状态", "稳定物理水团或因果类别"],
            ["状态2", "较低盐度、较高浊度与较高营养盐，在 Deep Bay 较集中", "淡水影响和局地高浊高营养背景", "单一陆源输入的定量归因"],
        ],
        [1.6, 5.2, 4.7, 3.0],
    )
    insert_heading_before(anchor, "2.6 本章小结")
    insert_text_before(anchor, "本章形成研究区、观测级样本和水质背景的统一描述。香港近岸水质存在显著的水区、季节和垂向异质性，说明采用全站点统一模型时必须同时报告其适用边界；同日水质变量则保留为机理分析证据，而非预测协变量。")


def insert_chapter_three_content(doc: Document) -> None:
    anchor = find_paragraph(doc, "第4章 多水层水质预测结果")
    insert_heading_before(anchor, "3.5 预测任务与验证口径的解释边界")
    insert_text_before(anchor, "正式结果采用固定随机种子 42 的 80%/20% 留出。它用于比较不同指标、信息源和水层。留站点和留年份检验另作补充，用来观察模型在新条件下的表现。")
    add_table_before(
        doc,
        anchor,
        "表3-3 正式主结果与扩展验证的口径对照",
        ["验证方式", "训练与测试划分", "历史目标特征", "论文用途"],
        [
            ["正式随机留出", "全样本随机 80%/20%，种子42", "按正式主链设定", "比较主结果"],
            ["五折留站点", "整站点留出，按水区分折", "移除", "空间泛化边界"],
            ["时间留出", "2014—2022 训练，2023—2024 测试", "移除", "跨年份适用边界"],
        ],
        [2.6, 5.0, 2.6, 4.3],
    )
    insert_heading_before(anchor, "3.6 本章小结")
    insert_text_before(anchor, "本章明确了多源特征的构建、四组信息源控制比较和不同验证方式的职责。正式结果回答同分布样本上的可比预测能力，扩展验证回答模型在未知站点和未来年份条件下的适用范围，二者不得互相替代。")


def insert_chapter_four_content(doc: Document) -> None:
    anchor = find_paragraph(doc, "第5章 遥感敏感性与不确定性讨论")
    insert_heading_before(anchor, "4.5 透明度单列口径下的正式结果汇总")
    insert_text_before(anchor, "完整三水层样本中的透明度记录完全一致。本文把它解释为站点—日期塞氏盘观测。本节按单列透明度展示结果，原有三条训练记录只作重复检验。")
    insert_figure_before(anchor, "图4-6", "透明度单列口径下的正式模型性能热图", 14.0)
    insert_figure_before(anchor, "图4-7", "Chl-a TN TP 与站点日期透明度的代表性观测预测散点图", 12.3)
    best = pd.read_csv(FORMAL_DIR / "full_station_best_by_target_depth.csv")
    selected = best.loc[
        ((best["target"] != "secchi") | (best["depth"] == "surface")),
        ["target", "depth", "feature_set", "r2", "rmse", "mae"],
    ].copy()
    selected["target"] = selected["target"].replace({"chla": "Chl-a", "tn": "TN", "tp": "TP", "secchi": "透明度"})
    selected["depth"] = selected["depth"].replace({"surface": "表层", "middle": "中层", "bottom": "底层"})
    selected.loc[selected["target"] == "透明度", "depth"] = "站点—日期"
    selected[["r2", "rmse", "mae"]] = selected[["r2", "rmse", "mae"]].round(3)
    add_table_before(doc, anchor, "表4-4 透明度单列口径下的正式最佳模型汇总", ["目标", "水层或尺度", "最佳特征组", "R2", "RMSE", "MAE"], selected.values.tolist(), [2.0, 2.8, 3.4, 1.6, 2.0, 2.0])
    insert_heading_before(anchor, "4.6 本章小结")
    insert_text_before(anchor, "正式主结果表明，不同水质目标的可预测性差异显著：TN 表层和 TP 表层在固定随机留出下表现较好，Chl-a 三个水层仍受短期扰动和局地过程限制。遥感单独组总体较弱，历史与过程信息是当前稳定性能的主要来源。")


def insert_chapter_five_content(doc: Document) -> None:
    anchor = find_paragraph(doc, "第6章 结论与展望")
    insert_heading_before(anchor, "5.5 遥感质量条件与严格光学样本敏感性")
    insert_text_before(anchor, "严格 Sentinel-2 样本要求日期差不超过 3 天、云量不高于 20%，并且波段完整。符合条件的样本有 1,254 条，占 11.5%。筛选后的平均 R2 只增加 0.0011，说明高质量遥感样本数量仍然不够。")
    insert_figure_before(anchor, "图5-11", "遥感质量条件下的模型 R2 变化", 14.2)
    insert_figure_before(anchor, "图5-12", "严格 Sentinel-2 筛选下的光学观测类型", 14.0)
    add_table_before(
        doc,
        anchor,
        "表5-7 严格 Sentinel-2 质量条件的样本覆盖与论文解释",
        ["条件", "样本数", "覆盖率", "论文解释"],
        [["近同日、低云量、光谱完整", "1,254 / 10,866", "11.5%", "高质量样本数量有限，只用于敏感性和局部机制比较"]],
        [4.0, 3.3, 2.5, 5.0],
    )
    insert_heading_before(anchor, "5.6 Chl-a 高值误差结构")
    insert_text_before(anchor, "本节以各水层训练集 P90 定义高值。三个水层的高值都常被低估，低估比例为 89.7%—91.5%。高值事件漏报比例为 55.4%—57.7%，因此还要单独看高值误差。")
    insert_figure_before(anchor, "图5-13", "Chl-a 高值样本的观测值与预测值关系", 14.2)
    insert_figure_before(anchor, "图5-14", "Chl-a 高值样本的低估比例与绝对误差", 14.0)
    insert_figure_before(anchor, "图5-15", "Chl-a 表层高值低估的水区季节结构", 13.8)
    insert_figure_before(anchor, "图5-16", "Chl-a 中层高值低估的水区季节结构", 13.8)
    insert_figure_before(anchor, "图5-17", "Chl-a 底层高值低估的水区季节结构", 13.8)
    insert_figure_before(anchor, "图5-18", "Chl-a 高值样本的预测区间覆盖率", 13.8)
    high = pd.read_csv(DIAG_DIR / "Chl-a高值与误差结构_20260907" / "表1_各水层高值误差汇总.csv")
    high_rows = []
    for _, row in high.iterrows():
        high_rows.append(
            [
                str(row["depth"]),
                f"{int(row['总样本数'])}",
                f"{int(row['高值样本数'])}",
                f"{row['高值比例'] * 100:.1f}%",
                f"{row['高值平均绝对误差']:.3f}",
                f"{row['高值低估比例'] * 100:.1f}%",
                f"{row['高值事件漏报比例'] * 100:.1f}%",
            ]
        )
    add_table_before(
        doc,
        anchor,
        "表5-8 Chl-a 高值样本的误差汇总",
        ["水层", "总样本数", "高值样本数", "高值比例", "高值 MAE", "低估比例", "事件漏报比例"],
        high_rows,
        [1.6, 1.9, 2.0, 2.0, 2.2, 2.3, 2.5],
    )
    insert_heading_before(anchor, "5.7 分海区模型的空间泛化边界")
    insert_text_before(anchor, "留站点检验中，分海区模型提高了 Chl-a 三个水层的结果。TN 底层也有小幅提高，但 TN 表层、中层和 TP 不稳定。留年份检验中，除 TN 底层外没有提高，因此分海区模型只作为补充分析。")
    insert_figure_before(anchor, "图5-19", "五折留站点条件下分海区模型的任务 R2 变化 透明度仅为站点日期重复检验", 14.0)
    insert_figure_before(anchor, "图5-20", "时间留出与留站点验证下的 R2 变化 透明度仅为站点日期重复检验", 14.0)
    spatial = pd.read_csv(DIAG_DIR / "空间泛化边界_20260907" / "表3_空间泛化边界判定.csv")
    spatial_rows = []
    for _, row in spatial.head(12).iterrows():
        spatial_rows.append(
            [
                str(row["target"]),
                str(row["depth"]),
                f"{row['delta_r2_pooled']:+.3f}",
                f"{int(row['positive_folds'])}/5",
                f"{row['时间外推delta_R2']:+.3f}",
                str(row["论文定位"]),
            ]
        )
    add_table_before(
        doc,
        anchor,
        "表5-9 分海区模型的空间与时间验证边界",
        ["指标", "水层", "五折留站点 R2 变化", "正增益折数", "2023—2024 时间留出 R2 变化", "论文定位"],
        spatial_rows,
        [1.4, 1.5, 2.5, 1.8, 3.1, 4.0],
    )
    insert_heading_before(anchor, "5.8 本章小结")
    insert_text_before(anchor, "本章补充了正式结果的使用条件。遥感是否有效取决于覆盖和质量。Chl-a 高值仍容易低估，分海区模型只在部分空间检验中表现较好。")


def review_note(caption: str) -> str:
    if caption.startswith("图1-1"):
        return "图表说明：概括数据获取、观测级样本、特征组、建模和扩展验证。论文意义：让读者先理解数据与代码的整体关系。正文应强调正式主结果与扩展诊断分层报告。"
    if caption.startswith("图2"):
        return f"图表说明：{caption}。论文意义：提供研究区、样本与水质背景证据，解释后续模型为何需要考虑水区、季节和水层异质性。正文不得把同日水质背景变量写成正式预测输入。"
    if caption.startswith("图3"):
        return f"图表说明：{caption}。论文意义：明确样本划分和验证口径，使读者能够区分随机留出、空间外推和时间外推。"
    if caption.startswith("图4"):
        return f"图表说明：{caption}。论文意义：支撑正式主结果、特征组比较和目标差异解释。透明度仅按站点—日期尺度解释，不能作为三条独立水层观测证据。"
    if caption.startswith("图5"):
        return f"图表说明：{caption}。论文意义：用于界定遥感质量、Chl-a 高值和空间泛化的适用边界。该图不替代正式随机留出主结果，不应夸大为整体精度突破。"
    if caption.startswith("表2"):
        return f"表格说明：{caption}。论文意义：为研究区、数据来源、质量控制和水质背景提供可复核的定量信息。"
    if caption.startswith("表3"):
        return f"表格说明：{caption}。论文意义：说明任务定义、特征组或验证口径，防止模型比较混用不同数据划分。"
    if caption.startswith("表4"):
        return f"表格说明：{caption}。论文意义：汇总正式模型表现；正文须标注其为固定随机留出，不等同于跨年份外推。"
    if caption.startswith("表5"):
        return f"表格说明：{caption}。论文意义：汇总遥感敏感性、Chl-a 高值误差或空间泛化边界；结论应使用条件性表述。"
    if caption.startswith("表6"):
        return f"表格说明：{caption}。论文意义：将最终结论与对应证据相连，避免扩展实验被误写为正式主结果。"
    return f"审阅提示：{caption}。请在正文首次引用处说明图表展示对象、样本或验证口径，并明确其对本节结论的支撑作用。"


def add_word_comments(output: Path) -> tuple[int, list[str]]:
    docx = Document(output)
    captions = [p.text.strip() for p in docx.paragraphs if p.text.strip().startswith(("图", "表"))]
    temp_root = Path(tempfile.mkdtemp(prefix="wq_review_"))
    temp_doc = temp_root / "master_review.docx"
    shutil.copy2(output, temp_doc)
    word = None
    document = None
    missed: list[str] = []
    count = 0
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        word.UserName = "刘逸凡"
        word.UserInitials = "LYF"
        document = word.Documents.Open(str(temp_doc), ReadOnly=False, AddToRecentFiles=False)
        document.TrackRevisions = True
        for caption in captions:
            search_range = document.Content
            finder = search_range.Find
            finder.ClearFormatting()
            finder.Text = caption
            finder.Forward = True
            finder.Wrap = 0
            if finder.Execute():
                document.Comments.Add(search_range, review_note(caption))
                count += 1
            else:
                missed.append(caption)
        document.Save()
    finally:
        if document is not None:
            document.Close(SaveChanges=True)
        if word is not None:
            word.Quit()
    shutil.copy2(temp_doc, output)
    shutil.rmtree(temp_root, ignore_errors=True)
    return count, missed


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, OUTPUT)
    doc = Document(OUTPUT)
    insert_front_matter(doc)
    replace_paragraph_starting(
        doc,
        "香港近岸海域受陆源输入",
        "香港近岸海域受陆源输入、潮汐交换、气象扰动和人类活动共同影响，水质过程具有显著的水层差异和时空异质性。以 2014—2024 年香港近岸 94 个监测站点为基础，本文对 Chl-a、TN 和 TP 构建表层、中层、底层九项水层预测任务，并将 Secchi 透明度作为站点—日期尺度的塞氏盘观测单列解释。研究旨在回答多源过程信息的相对贡献、遥感在何种质量条件下可用，以及模型在高值、未知站点和跨年份条件下的适用边界。",
    )
    replace_paragraph_starting(
        doc,
        "Hong Kong coastal waters are shaped",
        "Hong Kong coastal waters are shaped by terrestrial inputs, tidal exchange, meteorological disturbance, and human activities, producing strong vertical and spatial heterogeneity. Using observations from 94 monitoring stations during 2014-2024, this study develops nine layer-specific tasks for Chl-a, total nitrogen, and total phosphorus, and treats Secchi depth as a station-date indicator because the three layer records are duplicated. It examines the contribution of multi-source process information, the quality conditions under which remote sensing may be useful, and the boundary of model applicability for high values, unseen stations, and held-out years.",
    )
    replace_paragraph_starting(
        doc,
        "第三，形成“统一主模型加扩展验证”",
        "第三，形成“统一主模型加扩展验证”的证据组织方式。正式主链保留 12 条建模记录以维持原始数据结构，但论文解释将 Chl-a、TN 和 TP 作为九项多水层任务，将 Secchi 作为一个站点—日期透明度任务；分海区、按年份留出、五折留站点、Chl-a 短期过程诊断和预测区间作为独立扩展，专门检验空间转移、高值误差和时间外推边界。这样既保留可重复的主结果，也避免用单一 R2 掩盖不同泛化条件下的差异。",
    )
    replace_paragraph_starting(
        doc,
        "Chl-a、Secchi、TN 和 TP 分别在三水层建模",
        "原始建模表将 Chl-a、Secchi、TN 和 TP 分别保留在三层记录中，形成 12 条训练记录。由于 Secchi 在完整三水层样本中完全重复，论文分析将其单列为站点—日期透明度任务；Chl-a、TN 和 TP 则构成九项多水层任务。特征组包括基线组、历史与过程组、遥感组和历史—遥感融合组，用于分离不同信息源的作用。",
    )
    replace_paragraph_starting(doc, "表3-1 十二个预测任务与样本量", "表3-1 原始建模记录与样本量 Secchi 三层记录仅作重复检验")
    replace_paragraph_starting(
        doc,
        "12 个任务的最佳 R2",
        "正式随机留出记录的最佳 R2 呈现显著指标和水层差异。TN 表层达到 0.798，TP 表层达到 0.725，站点—日期透明度达到 0.616；Chl-a 三个水层结果相对较低，表明其对短期扰动和局地过程更敏感。透明度的其余两条重复训练记录不作为独立水层结果解释。",
    )
    replace_paragraph_starting(
        doc,
        "Chl-a 三水层和 Secchi 三水层的合并折外",
        "五折留站点中，分海区模型对 Chl-a 三个水层的合并折外 R2 均高于全域模型；透明度的三条重复训练记录也呈现相近方向，但仅作为站点—日期透明度的重复检验。TN 表层、中层和 TP 三水层在部分折次出现明显退化。因此，分海区模型可作为光学目标的空间外推扩展结果，不宜替代全域模型成为四项指标的统一正式方法。",
    )
    replace_paragraph_starting(
        doc,
        "本文建立了覆盖 94 个站点",
        "本文建立了覆盖 94 个站点的多源融合预测流程：Chl-a、TN 和 TP 构成九项多水层任务，Secchi 作为站点—日期透明度任务单列解释，并在固定随机留出条件下形成统一的模型、汇总、图表和解释链路。TN、TP 和透明度在部分任务上达到较高 R2，而 Chl-a 仍是受短期过程影响较强的困难目标。",
    )
    replace_paragraph_starting(
        doc,
        "五折留站点验证表明",
        "五折留站点验证表明，分海区训练对 Chl-a 的空间转移具有较稳定的改善，透明度重复训练记录的方向相近，但在 TN 和 TP 中不具备一致性。因此，它为空间差异解释提供证据，但不作为四项指标的统一模型替代方案。",
    )
    insert_chapter_two_content(doc)
    insert_chapter_three_content(doc)
    insert_chapter_four_content(doc)
    insert_chapter_five_content(doc)
    add_heading_guides(doc)
    remove_redundant_break_before_chapter_one(doc)
    doc.save(OUTPUT)
    comment_count, missed = add_word_comments(OUTPUT)
    print(f"output={OUTPUT}")
    print(f"comments={comment_count}")
    if missed:
        print("missed_comments=")
        for item in missed:
            print(item)


if __name__ == "__main__":
    main()
