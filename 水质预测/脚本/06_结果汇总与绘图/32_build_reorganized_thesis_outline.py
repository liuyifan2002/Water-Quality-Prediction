"""根据当前正式结果和诊断结果生成论文结构重组版 Word 草稿。"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "文档"
    / "论文材料"
    / "福州大学格式论文草稿_香港近岸多水层水质预测_结果重组版_20260907.docx"
)

BLACK = RGBColor(0, 0, 0)
FIGURES = {
    "图1-1": ROOT / "结果" / "论文图表" / "详细技术路线_20260906" / "图1-1_详细研究技术路线.png",
    "图2-3": ROOT / "结果" / "诊断结果" / "水质时空分异与水团背景_20260907" / "图1_四指标分水区季节水层中位数热图.png",
    "图2-4": ROOT / "结果" / "诊断结果" / "水质时空分异与水团背景_20260907" / "图2_表底层配对差异.png",
    "图2-5": ROOT / "结果" / "诊断结果" / "水质时空分异与水团背景_20260907" / "图4_表层观测水质状态聚类.png",
    "图4-1": ROOT / "结果" / "论文图表" / "full_station_figures" / "fig01a_正式模型性能热图_透明度单列.png",
    "图4-2": ROOT / "结果" / "论文图表" / "full_station_figures" / "fig02_feature_set_summary.png",
    "图4-3": ROOT / "结果" / "论文图表" / "full_station_figures" / "fig04a_代表性目标观测预测散点图.png",
    "图4-4": ROOT / "结果" / "论文图表" / "full_station_shap" / "fig05_remote_shap_importance_grid.png",
    "图5-1": ROOT / "结果" / "诊断结果" / "遥感质量与光学类型_20260907" / "图1_遥感质量条件下的R2变化.png",
    "图5-2": ROOT / "结果" / "诊断结果" / "遥感质量与光学类型_20260907" / "图2_严格筛选S2光学观测类型.png",
    "图5-3": ROOT / "结果" / "诊断结果" / "Chl-a高值与误差结构_20260907" / "图1_高值样本观测预测散点图.png",
    "图5-4": ROOT / "结果" / "诊断结果" / "Chl-a高值与误差结构_20260907" / "图2_高值低估与绝对误差.png",
    "图5-5": ROOT / "结果" / "诊断结果" / "空间泛化边界_20260907" / "图1_五折留站点任务R2变化.png",
    "图5-6": ROOT / "结果" / "诊断结果" / "空间泛化边界_20260907" / "图3_时间与留站点验证R2变化.png",
}


def set_font(run, size: float, *, chinese: str = "宋体", bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), chinese)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = BLACK


def set_layout(section) -> None:
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.0)


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
    tc_pr = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    tc_pr.append(element)


def add_heading(doc: Document, text: str, level: int) -> None:
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    paragraph.paragraph_format.space_after = Pt(10)
    paragraph.paragraph_format.line_spacing = Pt(22 if level == 1 else 20)
    set_font(paragraph.add_run(text), 16 if level == 1 else 14, chinese="黑体", bold=True)


def add_paragraph(doc: Document, text: str, *, indent: bool = True) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.line_spacing = Pt(20)
    paragraph.paragraph_format.space_after = Pt(3)
    if indent:
        paragraph.paragraph_format.first_line_indent = Cm(0.74)
    set_font(paragraph.add_run(text), 12)


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.keep_with_next = True
    set_font(paragraph.add_run(text), 10.5)


def add_figure(doc: Document, key: str, caption: str, width_cm: float = 14.8) -> None:
    path = FIGURES[key]
    if not path.exists():
        raise FileNotFoundError(path)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_together = True
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
    add_caption(doc, f"{key} {caption}")


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.width = Cm(widths[index])
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(paragraph.add_run(header), 9.5, bold=True)
        shade(cell)
        set_cell_border(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cell = cells[index]
            cell.width = Cm(widths[index])
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.line_spacing = Pt(14)
            set_font(paragraph.add_run(value), 9)
            set_cell_border(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    doc.add_paragraph()


def add_figure_plan(doc: Document, rows: list[list[str]]) -> None:
    add_paragraph(doc, "本章图表以能够直接支撑结论的结果为主。完整分任务表、未校正趋势筛查和补充敏感性图移入附录，避免同一结论被相似图件重复表达。", indent=False)
    add_table(doc, ["编号", "图表内容", "结论用途"], rows, [2.0, 7.6, 5.4])


def configure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(12)
    normal.font.color.rgb = BLACK
    for name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style.font.color.rgb = BLACK
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    title_p_pr = doc.styles["Title"].element.find(qn("w:pPr"))
    if title_p_pr is not None:
        borders = title_p_pr.find(qn("w:pBdr"))
        if borders is not None:
            title_p_pr.remove(borders)


def remove_paragraph_border(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is not None:
        p_pr.remove(borders)


def main() -> None:
    doc = Document()
    set_layout(doc.sections[0])
    configure_styles(doc)
    doc.core_properties.title = "香港近岸多水层水质预测论文结构重组版"

    title = doc.add_paragraph(style="Title")
    remove_paragraph_border(title)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(16)
    set_font(title.add_run("香港近岸多水层水质预测论文结构重组版"), 20, chinese="黑体", bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(20)
    set_font(subtitle.add_run("基于正式模型、遥感质量、高值误差与空间泛化结果"), 12)
    add_paragraph(
        doc,
        "本稿用于重组论文的章节逻辑和图表配置。正式主结果仍以全站点固定随机 80%/20% 留出模型为准；遥感质量、Chl-a 高值误差和空间泛化分析用于界定适用边界，不替代正式主模型。",
    )
    add_paragraph(
        doc,
        "写作时须将 Chl-a、TN 和 TP 作为三水层预测变量，将 Secchi 透明度作为站点—日期尺度的塞氏盘观测单列解释。原始表中三层透明度记录完全一致，故不将其作为三个独立水层观测证据。",
    )

    add_heading(doc, "第1章 绪论", 1)
    add_heading(doc, "1.1 研究背景与意义", 2)
    add_paragraph(doc, "近岸海域同时受到陆源输入、潮汐交换、风浪混合和季节变化影响，水质变化具有明显的空间异质性和短期波动。站点观测能够提供可信的现场信息，但难以连续覆盖整个近岸区域；遥感和再分析资料能够补充环境背景，却受到影像质量和近岸复杂光学条件约束。")
    add_heading(doc, "1.2 研究内容与创新", 2)
    add_paragraph(doc, "论文围绕观测级多源数据架构、全站点统一预测、遥感质量敏感性和适用边界展开。创新不表述为单一模型替换，而是建立可追溯的数据接入、特征对照与验证分层链路，回答遥感何时可能有效、Chl-a 高值何时容易漏报，以及分海区训练能否在严格空间和时间验证中保持改善。")
    add_heading(doc, "1.3 技术路线", 2)
    add_figure(doc, "图1-1", "香港近岸多源数据融合下水质预测技术路线", 14.3)
    add_figure_plan(doc, [["表1-1", "代表性研究比较", "从目标、水层、数据源、验证与遥感角色说明本文定位"]])

    doc.add_page_break()
    add_heading(doc, "第2章 研究区 数据与水质背景", 1)
    add_heading(doc, "2.1 研究区、监测网络与数据来源", 2)
    add_paragraph(doc, "研究区覆盖香港近岸 94 个水质监测站点和 10 个水区。2014—2024 年原始水质主表包含 30,719 条观测记录，并保留盐度、温度、浊度、悬浮物、无机氮、正磷酸盐、溶解氧和 pH 等现场或实验室观测变量。除 Chl-a、TN、TP 和透明度外的同日水质变量仅用于本章机理背景分析，不进入预测模型。")
    add_heading(doc, "2.2 观测级样本与质量控制", 2)
    add_paragraph(doc, "以 sample_id 表示站点—日期，以 obs_id 表示站点—日期—水层。ERA5、降雨、潮汐、夜间灯光和卫星数据通过显式站点和日期键接入；日尺度环境过程按采样日前窗口统计。遥感同时保存有效像元、缺失、插补和可用性状态，使模型结果能够回溯至影像质量条件。")
    add_heading(doc, "2.3 水质时空分异与垂向背景", 2)
    add_paragraph(doc, "Chl-a、TN 和 TP 的水区、季节及水层差异用于解释建模任务的背景机制；透明度按站点—日期尺度展示，不进行表中底层比较。表底层配对结果显示 Chl-a、TN、盐度、浊度和悬浮物存在不同程度的垂向差异，为多水层预测的必要性提供观测依据。")
    add_figure(doc, "图2-3", "水区 季节与水层条件下的九项多水层水质和站点日期透明度中位数", 14.5)
    add_figure(doc, "图2-4", "表层与底层配对差异 不含透明度", 13.8)
    add_heading(doc, "2.4 表层观测水质状态", 2)
    add_paragraph(doc, "利用表层完整观测样本开展描述性聚类，识别出以较高盐度、较低浊度为特征的背景状态，以及以较低盐度、较高浊度和较高营养盐为特征的局地状态。后者在 Deep Bay 表层样本中占比较高，提示淡水影响和高浊高营养背景可能是近岸异质性的重要来源；该结果不作因果识别，也不作为模型输入。")
    add_figure(doc, "图2-5", "表层观测水质状态及其特征", 14.2)
    add_figure_plan(doc, [
        ["图2-1", "监测站点与水区分布", "说明空间覆盖和分区依据"],
        ["图2-2", "观测记录和目标变量时间覆盖", "说明样本时段和缺失背景"],
        ["图2-3", "水区 季节 水层中位数", "展示时空分异"],
        ["图2-4", "表层 底层配对差异", "说明垂向异质性"],
        ["图2-5", "表层观测水质状态", "说明淡水影响和局地高浊高营养背景"],
        ["表2-1", "多源数据来源与用途", "列明时空尺度、用途和是否进入模型"],
        ["表2-2", "目标变量统计", "Chl-a TN TP 三层列示 透明度单列"],
        ["表2-3", "表底层配对差异", "报告配对样本量和中位数差异"],
    ])

    doc.add_page_break()
    add_heading(doc, "第3章 多源特征构建与预测方法", 1)
    add_heading(doc, "3.1 预测目标与信息边界", 2)
    add_paragraph(doc, "正式预测由 Chl-a、TN 和 TP 的表层、中层、底层九项任务，以及站点—日期透明度任务构成。为避免同一采样事件的共测变量造成解释混淆，模型不使用同日其他水质指标。历史目标特征的使用仅限于与对应验证设计相容的情形；在留站点和跨年份严格验证中予以移除。")
    add_heading(doc, "3.2 特征构建与质量记录", 2)
    add_paragraph(doc, "模型将变量组织为时空基线、历史与过程、遥感、历史—遥感融合四组。过程信息包括气象、日尺度海表温度和短波辐射窗口、降雨累积、潮汐和夜间灯光；遥感信息包括 Landsat 光谱和指数，以及 Sentinel-2 的近同日质量记录。")
    add_heading(doc, "3.3 正式评价与扩展验证", 2)
    add_paragraph(doc, "正式主结果采用固定随机种子 42 的 80%/20% 留出，并以 R2、RMSE、MAE 和 NSE 评价。扩展验证包括严格 Sentinel-2 质量敏感性、Chl-a 高值误差诊断、五折留站点和 2023—2024 年时间留出分海区对照。扩展结果不回填或替代正式随机留出结果。")
    add_figure_plan(doc, [
        ["图3-1", "样本主键 特征组和模型输入关系", "说明 obs_id sample_id 及多源数据接入"],
        ["图3-2", "正式随机留出与扩展验证关系", "区分同分布比较 空间外推和时间外推"],
        ["表3-1", "预测任务与样本量", "九项多水层任务与透明度单列"],
        ["表3-2", "特征组定义", "说明变量类别和机制 而非罗列全部字段"],
        ["表3-3", "验证设计与用途", "明确各实验是否使用目标历史信息"],
    ])

    doc.add_page_break()
    add_heading(doc, "第4章 全站点正式预测结果", 1)
    add_heading(doc, "4.1 正式随机留出性能", 2)
    add_paragraph(doc, "正式主链在全站点固定随机留出下形成统一对照。TN 表层 R2 为 0.798，TP 表层为 0.725，站点—日期透明度为 0.616；Chl-a 三个水层为 0.374—0.465。该结果用于描述同分布留出样本下的模型可比表现，不等同于中长期跨年份外推精度。")
    add_figure(doc, "图4-1", "全站点正式模型的预测性能热图", 14.4)
    add_heading(doc, "4.2 特征组贡献与遥感总体作用", 2)
    add_paragraph(doc, "历史与过程组以及历史—遥感融合组整体优于基线，而遥感单独组总体较弱。由此，遥感在本研究中应定位为受质量和目标属性约束的条件性补充，而非对过程信息的替代。")
    add_figure(doc, "图4-2", "不同特征组的正式随机留出性能比较", 14.2)
    add_heading(doc, "4.3 观测与预测关系及可解释性", 2)
    add_paragraph(doc, "代表性散点图用于展示不同目标的拟合范围，特征重要性用于解释当前样本和模型中实际进入模型的变量贡献。该类结果不作变量因果效应解释，尤其不将表面遥感解释为对中层和底层水质的直接观测。")
    add_figure(doc, "图4-3", "代表性正式模型的观测值与预测值对比", 12.0)
    add_figure_plan(doc, [
        ["图4-1", "正式模型性能热图", "主结果总览 透明度仅作一个站点日期任务解释"],
        ["图4-2", "四组信息源性能比较", "比较过程信息和遥感信息贡献"],
        ["图4-3", "观测预测散点图", "展示代表性任务拟合范围"],
        ["表4-1", "正式模型完整指标", "R2 RMSE MAE NSE"],
        ["表4-2", "各目标最佳特征组", "记录主结果和主要解释变量"],
    ])

    add_heading(doc, "第5章 遥感适用条件 Chl-a 高值误差与空间泛化边界", 1)
    add_heading(doc, "5.1 遥感质量条件与光学敏感性", 2)
    add_paragraph(doc, "严格 Sentinel-2 条件为采样日差不超过 3 天、云量不高于 20% 且光谱字段完整，符合条件的样本为 1,254/10,866，仅占 11.5%。质量门控融合相对于历史与过程模型的全样本平均 R2 变化仅为 +0.0011，故本节用于界定遥感适用条件，而不将其作为整体精度突破。")
    add_figure(doc, "图5-1", "遥感质量条件下的模型 R2 变化", 14.2)
    add_figure(doc, "图5-2", "严格 Sentinel-2 筛选下的光学观测类型", 14.2)
    add_heading(doc, "5.2 Chl-a 高值误差结构", 2)
    add_paragraph(doc, "以各水层训练集 P90 为高值阈值，三个水层高值样本均呈现明显低估。高值点预测低估比例为 89.7%—91.5%，高值事件漏报比例为 55.4%—57.7%。全局校准区间在高值尾部的覆盖不足，说明现有区间方法尚不能可靠刻画极端高值风险。")
    add_figure(doc, "图5-3", "Chl-a 高值样本的观测值与预测值关系", 14.2)
    add_figure(doc, "图5-4", "Chl-a 高值样本的低估比例和绝对误差", 14.2)
    add_heading(doc, "5.3 分海区模型的空间泛化边界", 2)
    add_paragraph(doc, "五折留站点中，分海区模型对 Chl-a 三个水层有稳定局部改善，对 TN 底层为弱正信号；TN 表中层和 TP 三层不稳定，五折宏平均变化为负。2023—2024 年时间留出中，除 TN 底层外其余任务未获得正增益。因此分海区模型可用于解释空间异质性和局部空间外推边界，但不升级为四项指标的中长期正式主模型。")
    add_figure(doc, "图5-5", "五折留站点条件下分海区模型的任务 R2 变化 透明度仅为站点日期重复检验", 14.2)
    add_figure(doc, "图5-6", "时间留出与留站点验证下的 R2 变化 透明度仅为站点日期重复检验", 14.2)
    add_figure_plan(doc, [
        ["图5-1", "遥感质量条件 R2 变化", "界定质量门控的整体和局部增益"],
        ["图5-2", "严格 S2 光学类型", "解释严格近同日样本覆盖与水体条件"],
        ["图5-3", "Chl-a 高值散点", "说明高值低估和回归均值误差"],
        ["图5-4", "Chl-a 高值误差", "报告低估比例和绝对误差"],
        ["图5-5", "五折留站点 R2 变化", "判断局部空间改善是否稳定"],
        ["图5-6", "时间与留站点 R2 变化", "界定中长期和空间泛化边界"],
        ["表5-1", "严格 S2 覆盖和质量门控比较", "不把低覆盖高质量子样本夸大为全样本突破"],
        ["表5-2", "Chl-a 高值误差汇总", "按水层报告阈值 样本数 MAE 低估和漏报"],
        ["表5-3", "分海区空间和时间验证结论", "明确可保留与不升级的任务"],
    ])

    add_heading(doc, "第6章 结论与展望", 1)
    add_heading(doc, "6.1 主要结论", 2)
    add_paragraph(doc, "本文构建了面向香港近岸水质的观测级多源融合预测链路，形成九项多水层水质任务和一个站点—日期透明度任务的统一比较。正式随机留出结果表明 TN 表层和 TP 表层具有较高同分布预测表现，Chl-a 仍受短期扰动和局地过程限制。遥感的总体增益有限且依赖影像质量；Chl-a 高值存在系统性低估；分海区模型的空间局部改善不能直接推广为跨年份中长期预测结论。")
    add_heading(doc, "6.2 局限与展望", 2)
    add_paragraph(doc, "后续研究应优先补充采样时刻、近岸水动力、河流径流及更稳定的高质量遥感观测，并在时间阻断验证下检验其是否真正提升中长期预测。透明度应继续按站点—日期观测尺度组织；年际趋势分析需在季节调整、多重检验和自相关处理后再用于推断性结论。")
    add_figure_plan(doc, [["表6-1", "核心结论 证据来源与适用范围", "使主结果和扩展诊断可追溯且不混用"]])

    add_heading(doc, "附录图表安排", 1)
    add_paragraph(doc, "附录A放置未做季节调整和多重检验校正的年际趋势筛查；附录B放置正式模型全量指标和特征清单；附录C放置遥感质量与空间泛化的全任务表；附录D放置 Chl-a 高值水区季节误差图及高值区间覆盖率图。")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
