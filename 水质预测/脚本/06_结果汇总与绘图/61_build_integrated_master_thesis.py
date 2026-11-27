"""依据冻结结果生成硕士学位论文整合排版稿及可追溯审计文件。

本脚本只读取既有 DOCX、CSV、Markdown 和图片，不训练模型、不生成预测、
不重算 SHAP/ALE。输出位于文档/论文材料/硕士论文整合排版稿_20261018。
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "文档" / "论文材料" / "硕士论文整合排版稿_20261018"
OUT_DOCX = OUT_DIR / "水质预测硕士学位论文_整合排版稿.docx"
SOURCE_DOCX = ROOT / "文档" / "论文材料" / "福州大学格式论文草稿_香港近岸多水层水质预测_开题内容融合与图件修订版_v2_20260907.docx"

CORE = ROOT / "runs" / "final_experiment_closure_audit_v1" / "final_core_results_table_v2.csv"
REGISTRY = ROOT / "runs" / "final_experiment_closure_audit_v1" / "final_experiment_master_registry_v2.csv"
MODERN = ROOT / "runs" / "modern_baseline_gate_v1" / "modern_baseline_core_metrics.csv"
ABLATION = ROOT / "runs" / "multisource_ablation_v1" / "multisource_ablation_source_summary.csv"
GROUPS = ROOT / "runs" / "multisource_ablation_v1" / "00_feature_source_group_counts.csv"
SHAP_DIR = ROOT / "runs" / "xgb_shap_ale_v2"
FIG_INV = SHAP_DIR / "xgb_shap_ale_v2_figure_inventory.csv"
REFIT_DECISION = ROOT / "runs" / "xgb_interpretation_refit_v1" / "06_xgb_refit_final_decision.md"

OTHER_FIGURES = [
    ("图1-1", "研究技术路线", ROOT / "结果/论文图表/详细技术路线_20260906/图1-1_详细研究技术路线.png", "1.5"),
    ("图2-1", "香港近岸监测站点与海区分布", ROOT / "结果/论文图表/论文图版_20260905/第一二章/图2-1_监测站点与海区分布.png", "2.1"),
    ("图2-2", "水质观测时间覆盖", ROOT / "结果/论文图表/论文图版_20260905/第一二章/图2-4_水质观测时间覆盖.png", "2.2"),
    ("图2-3", "四项水质指标在不同水层的分布", ROOT / "结果/论文图表/论文图版_20260905/第一二章/图2-3_四项水质指标三水层分布.png", "2.7"),
    ("图2-4", "多源数据接入与样本主键关系", ROOT / "结果/论文图表/论文图版_20260905/第一二章/图2-5_多源数据接入与样本主键.png", "2.4"),
    ("图4-1", "十项任务总体提升与站点改善覆盖", ROOT / "结果/论文图表/报告辅助图表_20261011/图1_十任务总体提升与站点改善覆盖.png", "4.2"),
    ("图4-2", "不同指标与水层的模型提升差异", ROOT / "结果/论文图表/报告辅助图表_20261011/图2_不同指标与水层的提升差异.png", "4.2"),
    ("图4-3", "代表性任务四模型比较", ROOT / "结果/论文图表/报告辅助图表_20261011/图3_代表性任务四模型比较.png", "4.4"),
    ("图4-4", "三种验证方式下的模型差异", ROOT / "结果/论文图表/报告辅助图表_20261011/图4_三种验证方式的模型差异.png", "4.3"),
    ("图5-1", "多源环境信息来源级消融比较", ROOT / "结果/论文图表/报告辅助图表_20261011/图5_多源环境信息消融比较.png", "5.3"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_run_font(run, east_asia="宋体", latin="Times New Roman", size=12, bold=False):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.size = Pt(size)
    run.font.bold = bold


def set_cell_text(cell, value, size=9, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.line_spacing = Pt(14)
    set_run_font(p.add_run(str(value)), size=size, bold=bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    cell.margin_top = cell.margin_bottom = Cm(0.05)


def shade(cell, fill="E7E6E6"):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def set_three_line_borders(table):
    for ri, row in enumerate(table.rows):
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            borders = tc_pr.first_child_found_in("w:tcBorders")
            if borders is None:
                borders = OxmlElement("w:tcBorders")
                tc_pr.append(borders)
            for edge in ("left", "right", "insideV"):
                el = OxmlElement(f"w:{edge}")
                el.set(qn("w:val"), "nil")
                borders.append(el)
            for edge in ("top", "bottom"):
                el = OxmlElement(f"w:{edge}")
                if ri == 0 or (edge == "bottom" and ri == len(table.rows) - 1):
                    el.set(qn("w:val"), "single")
                    el.set(qn("w:sz"), "8" if ri == 0 else "6")
                    el.set(qn("w:color"), "000000")
                else:
                    el.set(qn("w:val"), "nil")
                borders.append(el)
            if ri == 0:
                inside = OxmlElement("w:insideH")
                inside.set(qn("w:val"), "single")
                inside.set(qn("w:sz"), "6")
                borders.append(inside)


def add_field(paragraph, instruction):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "更新域"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, separate, text, end):
        run._r.append(el)


def configure_styles(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.top_margin, sec.left_margin, sec.bottom_margin, sec.right_margin = Mm(25), Mm(25), Mm(20), Mm(20)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(12)
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(20)
    pf.first_line_indent = Cm(0.74)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_after = Pt(0)

    for name, size, level, before, after in [
        ("Heading 1", 18, 1, 0, 33),
        ("Heading 2", 16, 2, 18, 21),
        ("Heading 3", 15, 3, 12, 13),
        ("Heading 4", 12, 4, 6, 8),
    ]:
        st = doc.styles[name]
        st.font.name = "黑体"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        st.font.size = Pt(size)
        st.font.bold = True
        st.paragraph_format.line_spacing = Pt(20)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True
        st.paragraph_format.page_break_before = level == 1
        if level == 1:
            st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for name, size, east in [("图题", 10.5, "宋体"), ("表题", 10.5, "宋体"), ("参考文献", 10.5, "宋体")]:
        if name not in doc.styles:
            st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), east)
        st.font.size = Pt(size)
        st.paragraph_format.first_line_indent = Cm(0)
        st.paragraph_format.line_spacing = Pt(17 if name == "参考文献" else 16)
        st.paragraph_format.space_before = Pt(3)
        st.paragraph_format.space_after = Pt(3)
        st.paragraph_format.keep_together = True
        if name != "参考文献":
            st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="Normal")
    if bold_prefix and text.startswith(bold_prefix):
        set_run_font(p.add_run(bold_prefix), bold=True)
        set_run_font(p.add_run(text[len(bold_prefix):]))
    else:
        set_run_font(p.add_run(text))
    return p


def add_heading(doc, text, level):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        set_run_font(r, east_asia="黑体", size={1: 18, 2: 16, 3: 15, 4: 12}[level], bold=True)
    return p


def add_caption(doc, text, table=False):
    p = doc.add_paragraph(style="表题" if table else "图题")
    set_run_font(p.add_run(text), size=10.5)
    return p


def add_figure(doc, path, caption, width_cm=15.5):
    if not path.exists():
        raise FileNotFoundError(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    # Normal样式使用固定20 pt；若不覆盖，Word会把行内图片裁成一条窄带。
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.line_spacing = 1
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    add_caption(doc, caption)


def add_table(doc, caption, headers, rows, widths=None):
    add_caption(doc, caption, table=True)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True)
        shade(table.rows[0].cells[i])
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value, align=WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER)
    for row in table.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    set_three_line_borders(table)
    return table


def add_front_matter(doc):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_run_font(p.add_run("分类号：__________\n学校代码：10386\n密级：__________\n学号：__________"), size=12)
    for _ in range(2): doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("福 州 大 学"), east_asia="黑体", size=24, bold=True)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("硕 士 学 位 论 文"), east_asia="黑体", size=22, bold=True)
    for _ in range(3): doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("基于多源环境数据与表格基础模型的\n近岸水质预测及空间泛化研究"), east_asia="黑体", size=16, bold=True)
    for _ in range(3): doc.add_paragraph()
    for line in ["学位类别：________________", "学科专业：________________", "研究方向：________________", "作者姓名：________________", "指导教师：________________", "完成日期：________________"]:
        p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Cm(0); set_run_font(p.add_run(line), size=14)
    doc.add_page_break()

    for title, lines in [
        ("遵守学术行为规范承诺", ["学校正式承诺文本及本人签名页待按研究生院模板插入。", "本人签名：__________    日期：__________"]),
        ("独创性声明", ["学校正式独创性声明文本待按研究生院模板插入。", "作者签名：__________    导师签名：__________    日期：__________"]),
        ("关于论文使用授权的说明", ["学校正式授权说明文本、保密选项及解密年份待本人确认后插入。", "作者签名：__________    导师签名：__________    日期：__________"]),
    ]:
        add_heading(doc, title, 1)
        for line in lines: add_body(doc, line)
        doc.add_page_break()

    add_heading(doc, "摘 要", 1)
    abstract = (
        "香港近岸海域受珠江径流、潮汐交换、气象扰动及人类活动共同影响，水质过程具有明显的时空异质性和垂向差异。"
        "本文基于2014—2024年香港近岸水质观测与多源环境数据，构建叶绿素a、透明度、总氮和总磷共10项正式预测任务，"
        "以77个严格特征为统一输入，比较XGBoost、TabICLv2、TabPFN和TabM在站点留出条件下的预测表现，并结合随机划分和年份留出结果讨论验证口径差异。"
        "在最终闭环结果中，TabICLv2在10项任务上的汇总决定系数均高于对应XGBoost基准，其中叶绿素a表层、透明度、总氮表层和总磷中层的决定系数分别由0.457、0.535、0.259和0.304提高到0.589、0.602、0.402和0.472。"
        "站点胜率与配对自助法结果表明，叶绿素a三水层和透明度的改善较稳定；总氮、总磷部分任务虽有总体提升，但折间差异和置信区间提示其稳健性相对有限。"
        "进一步以20个数值等价复现的XGBoost折模型开展TreeSHAP和累积局部效应分析。空间坐标在四个代表任务中均具有较高贡献，气温、云量、风场及时间过程变量的重要性随预测指标而异。"
        "来源级消融显示，空间静态信息对叶绿素a表层和透明度尤为重要，气象信息对透明度、总氮表层及总磷中层具有较稳定贡献。"
        "一维ALE揭示若干变量与模型输出之间存在非线性统计关系；二维ALE的部分组合受联合样本覆盖限制，因而仅作条件性解释。"
        "研究结果说明，表格基础模型能够在统一特征和严格留站验证下改善部分近岸水质任务，但性能增益具有指标、水层和空间条件依赖性。SHAP与ALE反映的是模型统计响应而非环境因果机制，相关结论应限定于本研究的数据范围与验证设计。"
    )
    add_body(doc, abstract)
    add_body(doc, "关键词：近岸水质；多源环境数据；表格基础模型；空间泛化；SHAP；累积局部效应", "关键词：")

    add_heading(doc, "Abstract", 1)
    p=doc.add_paragraph(style="Normal"); p.paragraph_format.first_line_indent=Cm(0)
    text=("Hong Kong coastal waters are jointly affected by riverine inputs, tidal exchange, meteorological variability, and human activities, resulting in marked spatial and vertical heterogeneity. Using coastal water-quality observations and multi-source environmental data from 2014 to 2024, this thesis defines ten prediction tasks for chlorophyll-a, Secchi depth, total nitrogen, and total phosphorus. With the same set of 77 strict features, XGBoost, TabICLv2, TabPFN, and TabM are compared under held-station validation. TabICLv2 improves pooled R-squared over XGBoost for all ten registered tasks. For surface chlorophyll-a, Secchi depth, surface total nitrogen, and middle-layer total phosphorus, R-squared increases from 0.457, 0.535, 0.259, and 0.304 to 0.589, 0.602, 0.402, and 0.472, respectively. TreeSHAP and accumulated local effects are then calculated with twenty numerically equivalent refitted XGBoost fold models. Spatial coordinates are important in all four representative tasks, while meteorological and temporal-process variables show task-specific contributions. Source-level ablation broadly supports the importance of spatial-static and meteorological information. The results indicate that tabular foundation models can improve several coastal water-quality tasks under strict spatial validation, but the gains depend on target, depth, and spatial setting. SHAP and ALE describe model associations rather than causal environmental mechanisms.")
    set_run_font(p.add_run(text), latin="Times New Roman", east_asia="宋体", size=12)
    p=doc.add_paragraph(); p.paragraph_format.first_line_indent=Cm(0)
    set_run_font(p.add_run("Key words: "), latin="Arial", east_asia="黑体", size=12, bold=True)
    set_run_font(p.add_run("coastal water quality; multi-source environmental data; tabular foundation model; spatial generalization; SHAP; accumulated local effects"), size=12)

    add_heading(doc, "目 录", 1)
    p=doc.add_paragraph(); p.paragraph_format.first_line_indent=Cm(0); add_field(p, 'TOC \\o "1-3" \\h \\z \\u')
    add_body(doc, "提示：首次打开文档后请在Word中全选并按F9更新目录、图表编号和页码。")

    add_heading(doc, "主要符号表", 1)
    add_table(doc, "表0-1 主要符号与含义", ["符号", "含义", "单位"], [
        ["Chl-a", "叶绿素a", "μg/L"], ["Secchi", "透明度", "m"], ["TN", "总氮", "mg/L"], ["TP", "总磷", "mg/L"],
        ["R²", "决定系数", "—"], ["RMSE", "均方根误差", "与目标变量一致"], ["MAE", "平均绝对误差", "与目标变量一致"],
        ["SHAP", "Shapley加性解释", "模型输出单位"], ["ALE", "累积局部效应", "模型输出单位"],
    ])


def chapter_one(doc):
    add_heading(doc, "第1章 绪论", 1)
    add_heading(doc, "1.1 研究背景与意义", 2)
    for t in [
        "近岸海域连接陆源输入与外海交换，水质同时受径流、潮汐、季风、辐射、海温和城市活动影响。香港海岸线曲折，港湾、半封闭水域与外海水域并存，使相同指标在不同站点和水层呈现不同的变化范围。常规监测能够提供可靠的站点观测，但在连续覆盖和空间外推方面仍存在限制。",
        "水质预测的实际价值不仅在于提高已见样本上的拟合精度，还在于判断模型能否迁移到训练阶段未出现的站点。随机划分常把同一站点的不同时期同时放入训练集和测试集，得到的结果更接近同分布插值；站点留出则要求模型面对新的空间位置，难度更高，也更接近监测网络扩展时的应用情形。",
        "多源环境特征能够补充单次采样难以表达的背景过程。气象变量描述热力与风场条件，降雨和潮汐变量提供陆源输入与水动力代理，海洋物理信息反映海气状态，夜间灯光可作为人类活动强度代理，时间和空间变量则刻画长期趋势、季节周期及地理位置。将这些信息纳入统一验证框架，有助于区分模型提升来自何种信息，而不是仅比较一个汇总分数。",
    ]: add_body(doc,t)
    add_heading(doc, "1.2 国内外研究现状", 2)
    add_heading(doc, "1.2.1 近岸水质预测研究进展", 3)
    add_body(doc, "水质预测经历了从经验统计模型、时间序列模型到集成学习和深度学习的发展。线性与广义加性模型便于解释，但面对变量间非线性和交互时表达能力有限；随机森林、梯度提升树等方法能够处理混合尺度变量、缺失和复杂交互，已成为环境表格数据的重要基准。近岸场景还要求研究者处理站点依赖、空间异质性和极端值样本不足等问题，因此模型评价必须与数据划分方式共同报告。")
    add_heading(doc, "1.2.2 多源环境特征在水质预测中的应用", 3)
    add_body(doc, "现有研究常融合气象、降雨、海温、遥感反射率、地形和土地利用等信息。不同数据源的时空分辨率与缺失机制并不一致，直接拼接可能引入错配或信息泄漏。对于中、底层水质，表面遥感变量的代表性尤其需要谨慎；经纬度和月份即使具有较高预测贡献，也主要反映空间或季节定位，不能直接解释为物理作用。")
    add_heading(doc, "1.2.3 传统机器学习与新型表格基础模型", 3)
    add_body(doc, "XGBoost通过逐轮拟合残差方向的树模型并在目标函数中加入正则项，在中小规模结构化数据上具有稳健性和较高计算效率[1]。TabPFN利用在大量合成任务上预训练的Transformer执行上下文学习，将训练样本作为上下文直接产生预测[2-3]。TabM通过共享参数的多分支表格网络提升神经网络在常规表格基准上的表现[4]。TabICLv2同样利用上下文学习思想处理监督表格任务，其优势是否能够迁移到具有站点结构和环境噪声的近岸水质数据，需要在统一特征和严格空间划分下验证。")
    add_heading(doc, "1.2.4 水质预测的空间泛化与时间外推", 3)
    add_body(doc, "随机划分、年份留出与站点留出回答不同问题。随机划分衡量同一总体分布内的插值能力；年份留出强调时间外推；站点留出强调空间迁移。三者的样本依赖结构、难度和误差来源不同，R²差值不能脱离划分方式直接比较。本文以正式站点留出结果为主，并用其他验证场景说明模型结论的边界。")
    add_heading(doc, "1.2.5 可解释机器学习在环境预测中的应用", 3)
    add_body(doc, "SHAP基于合作博弈中的Shapley值，将单个预测分解为各特征贡献；TreeSHAP为树模型提供高效且一致的计算方法[5-6]。ALE通过比较局部区间内特征变化导致的预测差异并累积中心化，能够在相关特征存在时减少对数据分布之外区域的外推[7]。二者解释的是已训练模型在给定数据分布下的统计响应，不能单独证明环境变量之间的因果关系。")
    add_heading(doc, "1.3 现有研究的不足", 2)
    add_body(doc, "综合来看，已有研究仍可能存在模型间特征和划分条件不一致、随机划分被误作空间泛化证据、不同指标与水层被合并评价、解释结果忽略折间稳定性和样本覆盖等问题。本文不以否定既有工作为前提，而是通过统一特征、冻结划分、逐站点评价、来源级消融及覆盖约束下的SHAP/ALE分析，减少上述因素对比较结论的干扰。")
    add_heading(doc, "1.4 研究目标与研究内容", 2)
    add_body(doc, "本文旨在构建面向香港近岸多指标、多水层任务的可复现预测与解释框架。主要内容包括：建立2014—2024年多源环境特征表；在10项正式任务上比较XGBoost与TabICLv2；在四个代表任务上比较TabPFN和TabM；对照随机、年份和站点留出结果；评估分海区建模与六类来源特征；利用数值等价复现的XGBoost模型开展SHAP和一、二维ALE分析。")
    add_heading(doc, "1.5 技术路线", 2)
    add_body(doc, "研究流程包括数据整理与特征匹配、冻结任务与空间划分、模型比较、稳健性检验、可解释性分析以及分区和消融对照。图1-1仅展示已完成工作，不包含尚未开展的实时部署或因果识别。")
    add_figure(doc, OTHER_FIGURES[0][2], "图1-1 多源环境数据与表格模型的近岸水质预测技术路线", 15.5)
    add_heading(doc, "1.6 研究创新点", 2)
    add_body(doc, "本文的贡献属于应用与验证层面：第一，将TabICLv2引入香港近岸水质预测，在相同77特征和严格留站划分下与XGBoost比较；第二，跨指标、跨水层和跨验证场景考察模型性能，明确总体提升与稳健性并不等价；第三，将来源级消融、空间分区与SHAP/ALE结合，在折间稳定性和样本覆盖约束下讨论模型响应。本文不声称提出TabICLv2算法，也不将解释统计量表述为物理因果机制。")
    add_heading(doc, "1.7 论文结构", 2)
    add_body(doc, "全文共六章。第1章阐述背景、现状与研究问题；第2章介绍研究区、监测数据及77特征构建；第3章说明模型、实验设计和XGBoost解释方法；第4章比较新型表格模型的预测效果及空间泛化；第5章讨论分区建模、来源消融与空间异质性；第6章总结主要结论、局限和后续工作。")
    add_heading(doc, "1.8 本章小结", 2)
    add_body(doc, "本章明确了本文关注的核心不是单一模型分数，而是统一条件下的性能比较、未见站点泛化和解释边界。后续章节据此依次给出数据、方法、结果与讨论。")


def chapter_two(doc, core, groups):
    add_heading(doc, "第2章 研究区、监测数据与多源特征构建", 1)
    add_heading(doc, "2.1 研究区概况", 2)
    add_body(doc, "研究区覆盖香港近岸环境保护署水质监测网络。现有原始主表包含约30719条站点—日期—水层记录，覆盖94个监测站点；进入Chl-a、TN和TP正式Family V2任务的站点为84个，透明度单任务覆盖94个站点。站点分布跨越河口影响区、港湾与外海水域，为评估空间异质性和未见站点迁移提供了基础。")
    add_figure(doc, OTHER_FIGURES[1][2], "图2-1 香港近岸监测站点与海区分布", 15.2)
    add_heading(doc, "2.2 水质监测数据", 2)
    add_body(doc, "水质观测时间范围为2014—2024年，主要研究指标为Chl-a、Secchi、TN和TP。Chl-a、TN与TP按表层、中层和底层建立任务；Secchi为站点—日期尺度的塞氏盘透明度观测，正式闭环中作为单任务处理，避免把重复的水层记录当作三项独立证据。")
    add_figure(doc, OTHER_FIGURES[2][2], "图2-2 水质监测记录的时间覆盖", 15.2)
    add_heading(doc, "2.3 预测指标与任务定义", 2)
    tasks=[]
    for _,r in core[core.model.eq("TabICLv2")].iterrows():
        tasks.append([r.task, r.target_family, r.depth, int(r.n_stations), int(r.n_obs), r.split_version])
    add_table(doc, "表2-1 十项正式预测任务及样本规模", ["任务", "指标", "水层", "站点数", "样本数", "正式划分"], tasks)
    add_body(doc, "最终闭环登记包含10项核心任务，而非早期方案中的12个候选任务。样本数以最终主登记表为准：Chl-a三水层分别为9783、9456和9783；TN与TP表、中、底层分别为9307、9000和9307；Secchi为10866。")
    add_heading(doc, "2.4 多源环境数据及时间空间匹配", 2)
    add_body(doc, "模型输入由气象、降雨—水文—潮汐、海洋物理、人类活动代理、时间过程和空间静态六类信息组成。外部变量依据站点、日期或相应时间窗口与观测匹配；滞后和滚动特征只使用预测时点之前的信息。无法从冻结清单确认的字段来源不在正文中补造。")
    add_figure(doc, OTHER_FIGURES[4][2], "图2-4 多源环境数据接入与样本主键关系", 15.2)
    add_heading(doc, "2.5 数据清洗与质量控制", 2)
    add_body(doc, "数据处理保留观测标识并核对站点、日期、水层和目标变量。模型缺失值处理在每一折内独立完成：仅使用该折训练样本拟合中位数填补器，再应用于训练集和测试集，从而避免测试分布进入预处理。受控复现进一步固定float32输入、原样本顺序、目标变换与非负截断规则。")
    add_heading(doc, "2.6 严格特征集X_strict", 2)
    rows=[[r.source_group,int(r.n_features)] for _,r in groups.iterrows()]
    add_table(doc, "表2-2 严格特征集的来源分组", ["来源组", "特征数"], rows)
    add_body(doc, "六类特征合计77个：气象29个、降雨—水文—潮汐22个、海洋物理11个、时间过程7个、人类活动6个、空间静态2个。所有模型比较使用相同特征名称和顺序。经纬度与时间编码允许参与预测和重要性计算，但其贡献应解释为定位信息或未显式过程的代理。")
    add_heading(doc, "2.7 样本分布及数据特征", 2)
    add_body(doc, "四类水质指标量纲与分布范围不同，Chl-a高值样本具有明显偏态，TN和TP在空间与水层间也存在差异。本文因此分别报告R²、RMSE和MAE，不跨指标直接比较RMSE绝对值，并在图表中保留各目标的实际单位。")
    add_figure(doc, OTHER_FIGURES[3][2], "图2-3 四项水质指标在不同水层的观测分布", 15.2)
    add_heading(doc, "2.8 本章小结", 2)
    add_body(doc, "本章界定了研究对象、10项正式任务及77个严格特征。统一的观测标识、折内预处理和来源分类为后续模型比较、消融与解释提供了可追溯基础。")


def chapter_three(doc, shap_inv):
    add_heading(doc, "第3章 模型构建、实验设计与可解释性分析", 1)
    add_heading(doc, "3.1 总体研究框架", 2)
    add_body(doc, "本章以冻结划分和统一特征为前提，先说明四类模型及评价方案，再给出XGBoost解释模型的受控复现、TreeSHAP和ALE方法。预测性能比较与解释分析相互关联但不混同：TabICLv2用于正式模型比较，SHAP和ALE解释对象明确限定为XGBoost。")
    add_heading(doc, "3.2 预测模型", 2)
    add_heading(doc, "3.2.1 XGBoost基准模型", 3)
    add_body(doc, "XGBoost以加法方式构建回归树。第t轮在既有预测上加入新树，使损失函数的一阶、二阶近似与树复杂度正则项共同最小化[1]。其对非线性、变量交互和不同尺度特征具有较强适应性，且TreeSHAP能够直接计算树模型贡献，因此本文将其作为统一基准和解释对象。")
    add_heading(doc, "3.2.2 TabICLv2表格基础模型", 3)
    add_body(doc, "TabICLv2将带标签训练样本与待预测样本共同组织为上下文，通过预训练模型在推理阶段完成任务适配。本文不对其内部机理作超出公开资料的推断，而把它作为新型表格基础模型，重点检验其在固定77特征和正式站点留出划分下是否能稳定优于XGBoost。")
    add_heading(doc, "3.2.3 TabPFN与TabM", 3)
    add_body(doc, "TabPFN通过合成任务预训练获得面向小中规模表格任务的先验，并以近似贝叶斯推断的方式进行上下文预测[2-3]。TabM采用参数高效的多分支结构，使多个预测分支共享大部分参数并集成输出[4]。二者仅在四个代表任务中作为现代基线，不自动扩展为10任务主结论。")
    add_heading(doc, "3.3 水质预测实验设计与模型配置", 2)
    add_heading(doc, "3.3.1 统一特征及模型输入", 3)
    add_body(doc, "所有正式模型采用X_strict_77。预处理在折内完成，测试标签不参与特征筛选、缺失值统计或解释变量筛选。不同模型的输出均转换回原始水质尺度后评价。")
    add_heading(doc, "3.3.2 验证方案", 3)
    add_body(doc, "本文区分随机划分、年份留出和站点留出。正式主结果采用冻结的五折站点留出：每个站点只属于一个测试折，保证同一站点不会同时出现在该折训练集和测试集。随机与年份留出用于解释不同泛化问题，不能与正式站点留出作简单配对差值。")
    add_heading(doc, "3.3.3 五折留站验证", 3)
    add_body(doc, "Chl-a、TN和TP使用各自Family V2统一站点划分；Secchi沿用冻结的历史站点留出方案。每折产生逐样本预测、折指标与站点指标，随后汇总pooled OOF R²、RMSE、MAE、折均值、站点胜率和配对bootstrap区间。")
    add_heading(doc, "3.3.4 模型配置及可重复性", 3)
    add_body(doc, "正式实验配置、随机种子和运行环境记录在各实验注册表中。为解释分析重新生成的20个XGBoost模型均为受控复现资产，不是找回的原始冻结二进制。它们的逐样本测试预测在事前固定容差下20/20折达到EXACT_REPRODUCTION，因此可用于解释与正式预测数值等价的模型函数。")
    add_table(doc, "表3-1 XGBoost解释模型资产核验", ["核验项目", "结果", "说明"], [
        ["任务数", "4", "Chl-a表层、Secchi、TN表层、TP中层"], ["折数", "20", "每任务5折"], ["严格复现", "20/20", "逐样本预测达到事前阈值"], ["模型与填补器", "完整", "可独立重新加载"], ["解释对象", "XGBoost", "不是TabICLv2内部机理"],
    ])
    add_heading(doc, "3.3.5 分海区及来源消融的实验设计", 3)
    add_body(doc, "分海区实验比较全域模型与区域模型在冻结验证下的差异；来源消融从77特征中逐次移除一个来源组，其余训练流程保持不变。消融读取既有结果，不在论文整合阶段重跑。")
    add_heading(doc, "3.4 模型性能评价指标", 2)
    add_body(doc, "决定系数R²衡量预测对总体变异的解释程度；RMSE对较大误差更敏感；MAE反映平均绝对偏差。pooled OOF指标由所有折测试预测合并后计算，fold mean则先逐折计算再取平均，两者不等价。站点胜率表示TabICLv2在站点级误差上优于XGBoost的比例。")
    add_table(doc, "表3-2 主要评价指标及解释", ["指标", "表达式含义", "使用说明"], [["R²", "1-SSE/SST", "越大越好，可为负"], ["RMSE", "误差平方均值的平方根", "越小越好，对大误差敏感"], ["MAE", "绝对误差均值", "越小越好"], ["站点胜率", "改进站点数/有效站点数", "反映空间改善覆盖"], ["配对bootstrap", "按配对预测重采样", "评价误差差值稳定性"]])

    # 17 figures are grouped under their required sections.
    fig_rows=[]
    mapping={row.figure_id: row for _,row in shap_inv.iterrows()}
    add_heading(doc, "3.5 基于XGBoost的SHAP变量贡献分析", 2)
    add_heading(doc, "3.5.1 受控复现模型及一致性验证", 3)
    add_body(doc, "解释分析仅纳入20个通过数值一致性核验的折模型。TreeSHAP加性检验20/20折通过；每折以训练样本为解释样本或抽样基础，测试集真实水质值不参与变量筛选。")
    add_heading(doc, "3.5.2 TreeSHAP方法及特征贡献", 3)
    add_body(doc, "图3-1给出四任务全局平均绝对SHAP贡献。四个任务的经纬度均位于前列，提示模型显著利用空间定位信息；Chl-a表层还受到2 m气温、月份和云量变量影响，Secchi的重要变量包括年内日序、年份趋势和风向，TN表层强调年份趋势与风场，TP中层则包含年份趋势、气温滚动量和云量。")
    order=["fig01_shap_four_task_importance","fig_shap_chla_surface","fig_shap_secchi","fig_shap_tn_surface","fig_shap_tp_middle"]
    for n,fid in enumerate(order,1):
        row=mapping[fid]; add_figure(doc, ROOT/Path(row.png_path), f"图3-{n} {row.title}", 15.5); fig_rows.append((f"图3-{n}",row))
    add_heading(doc, "3.5.3 变量筛选与稳定性判断", 3)
    add_body(doc, "ALE候选变量综合考虑SHAP贡献、五折排名、连续性、物理含义、有效观测覆盖和站点覆盖。空间坐标虽然重要，但不作为环境过程ALE变量；不同来源变量也不为追求形式均衡而强行纳入。")

    add_heading(doc, "3.6 一维ALE影响曲线", 2)
    add_heading(doc, "3.6.1 一维ALE的基本原理", 3)
    add_body(doc, "一维ALE在训练样本分位区间内计算特征由区间下界变到上界所引起的局部预测差异，并累积、中心化得到效应曲线[7]。每折分箱边界来自该折训练数据，跨折平均仅在共同支持区间内插值，不对无样本区域外推。")
    add_heading(doc, "3.6.2 变量响应曲线", 3)
    add_body(doc, "四任务各筛选4个连续变量，共形成16个一维ALE变量和80条逐折曲线。图3-6至图3-9分别展示任务组合图。曲线反映模型预测随变量变化的局部累计响应；局部上升或下降不等同于环境变量对水质的因果效应。")
    order=["fig_ale1d_chla_surface","fig_ale1d_secchi","fig_ale1d_tn_surface","fig_ale1d_tp_middle"]
    for idx,fid in enumerate(order,6):
        row=mapping[fid]; add_figure(doc, ROOT/Path(row.png_path), f"图3-{idx} {row.title}", 15.5); fig_rows.append((f"图3-{idx}",row))
    add_heading(doc, "3.6.3 折间稳定性及样本覆盖", 3)
    add_body(doc, "正式曲线同时展示折间范围和分箱覆盖。总体上，一维ALE具有较高折间相关性，但端点区间的样本和站点覆盖较少，因而主要变化区间的解释强于极端取值。图3-10汇总稳定性，避免只展示某一条视觉上更平滑的折曲线。")
    row=mapping["fig_ale1d_stability"]; add_figure(doc, ROOT/Path(row.png_path), f"图3-10 {row.title}", 15.5); fig_rows.append(("图3-10",row))

    add_heading(doc, "3.7 二维ALE联合影响分析", 2)
    add_heading(doc, "3.7.1 二维ALE的基本原理", 3)
    add_body(doc, "二维ALE通过矩形分箱内的二阶有限差分计算联合效应，并去除两个变量的一维主效应后中心化。本文使用分位数网格，并以网格样本数、站点数和五折共同支持为约束；稀疏网格不插值填补。")
    add_heading(doc, "3.7.2 二维ALE曲面分析", 3)
    add_body(doc, "八组候选中，Chl-a的气温—辐射和气温—云量组合、Secchi的气温变化—采样间隔和风场—采样间隔组合具有较好支持；TN的v10—u10及TP的两个气温过程变量组合在部分折缺少足够支持，因此降低解释强度。")
    order=["fig_ale2d_chla_surface","fig_ale2d_secchi","fig_ale2d_tn_surface","fig_ale2d_tp_middle"]
    for idx,fid in enumerate(order,11):
        row=mapping[fid]; add_figure(doc, ROOT/Path(row.png_path), f"图3-{idx} {row.title}", 15.5); fig_rows.append((f"图3-{idx}",row))
    add_heading(doc, "3.7.3 联合取值覆盖与解释边界", 3)
    add_body(doc, "图3-15汇总二维网格覆盖。缺少联合观测的区域被遮蔽，避免把模型在稀疏区域的数值表面误作稳定规律。二维ALE只能表明模型中的统计交互，不能据此确认水环境物理耦合机制。")
    row=mapping["fig_ale2d_coverage"]; add_figure(doc, ROOT/Path(row.png_path), f"图3-15 {row.title}", 15.5); fig_rows.append(("图3-15",row))
    add_heading(doc, "3.8 SHAP、ALE与来源消融的综合对照", 2)
    add_body(doc, "个体特征的SHAP贡献与删除整个来源组造成的性能变化回答不同问题。相关特征可相互替代，因此高SHAP不必然对应显著消融损失。图3-16和图3-17将两类证据分开标准化后对照，不把不同单位的绝对值放入同一色标。")
    for idx,fid in [(16,"fig_integrated_shap_ablation"),(17,"fig_integrated_feature_comparison")]:
        row=mapping[fid]; add_figure(doc, ROOT/Path(row.png_path), f"图3-{idx} {row.title}", 15.5); fig_rows.append((f"图3-{idx}",row))
    add_heading(doc, "3.9 本章小结", 2)
    add_body(doc, "本章建立了统一预测与解释框架，并明确SHAP/ALE解释对象为数值等价复现的XGBoost。17张解释图覆盖全局贡献、一维响应、二维交互、覆盖稳定性及来源消融对照，为后续模型结果讨论提供依据。")
    return fig_rows


def chapter_four(doc, core, modern):
    add_heading(doc, "第4章 新型表格模型的预测效果与空间泛化能力", 1)
    add_heading(doc, "4.1 XGBoost基准模型预测结果", 2)
    add_heading(doc, "4.1.1 十项水质任务的基础预测精度", 3)
    rows=[]
    for _,r in core[core.model.eq("XGBoost")].iterrows(): rows.append([r.task,f"{r.pooled_R2:.3f}",f"{r.RMSE:.4f}",f"{r.MAE:.4f}"])
    add_table(doc, "表4-1 十项任务的XGBoost站点留出基准结果", ["任务", "pooled R²", "RMSE", "MAE"], rows)
    add_body(doc, "XGBoost在不同指标和水层上的难度差异明显。Secchi的pooled R²为0.535，Chl-a三水层为0.353—0.457，TN三水层为0.259—0.508，TP中层和底层为0.304和0.383，而TP表层为-0.127，说明该任务在未见站点上尚未超过以总体均值预测的基准。")
    add_heading(doc, "4.1.2 不同水质指标预测难度的差异", 3)
    add_body(doc, "任务差异与目标分布、空间异质性、可用环境代理及极端样本有关。R²低并不意味着RMSE绝对值必然更大，因为各指标量纲不同；因此后续比较均在同一任务内进行。")
    add_heading(doc, "4.2 TabICLv2与XGBoost预测效果比较", 2)
    add_heading(doc, "4.2.1 十项水质任务整体预测精度", 3)
    rows=[]
    for task,g in core.groupby("task",sort=False):
        x=g[g.model.eq("XGBoost")].iloc[0]; t=g[g.model.eq("TabICLv2")].iloc[0]
        rows.append([task,f"{x.pooled_R2:.3f}",f"{t.pooled_R2:.3f}",f"{t.delta_R2:.3f}",f"{t.station_win_rate_RMSE:.1%}",t.robustness_class])
    add_table(doc, "表4-2 TabICLv2相对XGBoost的十任务比较", ["任务", "XGB R²", "TabICLv2 R²", "ΔR²", "RMSE站点胜率", "稳健性"], rows)
    add_body(doc, "TabICLv2在10项任务上的pooled R²均高于XGBoost。Chl-a三水层的ΔR²为0.106—0.136，Secchi为0.067，TN三水层为0.019—0.143，TP三水层为0.124—0.374。TP表层虽由负R²提高到0.247，但其bootstrap RMSE区间跨越零，故应将总体分数改善与稳健性证据分开表述。")
    add_figure(doc, OTHER_FIGURES[5][2], "图4-1 十项任务总体提升与站点改善覆盖", 15.5)
    add_heading(doc, "4.2.2 RMSE和MAE改善情况", 3)
    add_body(doc, "Chl-a表层RMSE由4.807降至4.184，MAE由2.262降至1.836；Secchi分别由0.756降至0.700、由0.496降至0.449；TN表层由0.363降至0.326、由0.170降至0.142；TP中层由0.0368降至0.0321、由0.0161降至0.0132。误差改善在不同量纲任务间不直接比较绝对值。")
    add_figure(doc, OTHER_FIGURES[6][2], "图4-2 不同指标与水层的模型提升差异", 15.5)
    add_heading(doc, "4.2.3 逐站点预测改善情况", 3)
    add_body(doc, "TabICLv2的RMSE站点胜率在10项任务中为70.2%—88.1%，MAE站点胜率为75.0%—92.9%。这说明多数站点受益，但并非每个站点均改善；站点胜率与pooled指标共同报告，可减少少数大样本站点主导总体指标的风险。")
    add_heading(doc, "4.2.4 不同水层及指标的模型差异", 3)
    add_body(doc, "Chl-a三水层均达到ROBUST；Secchi同样为ROBUST。TN表层为ROBUST，中、底层为MODERATE；TP三水层均为MODERATE。结果表明模型优势具有任务依赖性，不能由四个代表任务外推为所有指标和水层的普遍规律。")
    add_heading(doc, "4.3 不同验证场景下的模型表现", 2)
    for h,t in [("4.3.1 随机划分结果","随机划分允许同一站点的不同时期跨越训练集与测试集，更接近同分布插值，通常不能代表新站点表现。"),("4.3.2 年份留出结果","年份留出要求模型迁移到未参与训练的时间段，受长期趋势和分布漂移影响。"),("4.3.3 站点留出结果","站点留出是本文正式主口径，直接检验未见监测站点上的空间泛化。不同验证结果只用于说明难度与边界，不作不匹配的配对显著性比较。")]:
        add_heading(doc,h,3); add_body(doc,t)
    add_figure(doc, OTHER_FIGURES[8][2], "图4-4 随机、年份与站点验证下的模型差异", 15.5)
    add_heading(doc, "4.4 TabM、TabPFN与TabICLv2的代表性对比", 2)
    rows=[]
    for _,r in modern.iterrows(): rows.append([r.task,r.model,f"{r.pooled_R2:.3f}",f"{r.RMSE:.4f}",f"{r.MAE:.4f}"])
    add_table(doc, "表4-3 四个代表任务的现代模型比较", ["任务", "模型", "pooled R²", "RMSE", "MAE"], rows)
    add_body(doc, "在Chl-a表层和Secchi任务上，TabICLv2的pooled R²最高；TN表层中TabPFN为0.417，略高于TabICLv2的0.402；TP中层中TabICLv2为0.472，TabPFN为0.468，二者接近。TabM在Chl-a表层和Secchi具有竞争力，但在TN表层和TP中层的总体表现较弱。四任务结果用于识别适用性，不替代十任务主比较。")
    add_figure(doc, OTHER_FIGURES[7][2], "图4-3 四个代表任务中XGBoost、TabICLv2、TabPFN和TabM的比较", 15.5)
    add_heading(doc, "4.5 本章小结", 2)
    add_body(doc, "TabICLv2在十项正式站点留出任务上均提高pooled R²，多数站点的RMSE和MAE也得到改善，但稳健性随指标和水层变化。现代模型对比进一步表明，TabPFN在TN表层具有竞争力，不能把单一模型描述为所有任务上的绝对最优。")


def chapter_five(doc, ablation):
    add_heading(doc, "第5章 空间分区建模与多源特征贡献分析", 1)
    add_heading(doc, "5.1 分海区建模思路与实验设置", 2)
    add_body(doc, "分海区建模用于检验空间异质性是否足以抵消区域样本量减少带来的代价。全域模型和分区模型使用既有冻结结果与相同评价口径；本章不重新训练，也不根据结果图形选择更有利的区域方案。")
    add_heading(doc, "5.2 全域模型与分区模型比较", 2)
    add_heading(doc, "5.2.1 Chl-a与Secchi的分区效果", 3)
    add_body(doc, "既有分区验证表明，Chl-a与Secchi在部分海区和留站折次中可从局地模型获益，提示两类指标的空间背景差异较明显。但改善并非覆盖所有站点，仍应结合区域样本规模和折间稳定性判断。")
    add_heading(doc, "5.2.2 TN与TP的分区效果", 3)
    add_body(doc, "TN与TP的分区效果不具普遍稳定性。营养盐可能受跨区输运、局地输入和时间过程共同影响，简单按海区拆分会减少训练样本，未必优于保留全域数据的模型。")
    add_heading(doc, "5.2.3 分区建模的适用边界", 3)
    add_body(doc, "分区模型适用于空间过程差异强、区域样本充足且跨折改善一致的任务。对于样本不足或边界站点，区域划分可能放大方差。论文中的分区结果因此作为空间异质性证据，而非默认的生产部署方案。")
    add_heading(doc, "5.3 多源特征来源级消融", 2)
    add_heading(doc, "5.3.1 特征来源分组与消融方法", 3)
    add_body(doc, "来源消融按冻结分类逐次删除METEOROLOGY、RAINFALL_HYDRO_TIDE、OCEAN_PHYSICS、HUMAN_ACTIVITY、TEMPORAL_PROCESS或SPATIAL_STATIC，其余特征保持不变。pooled ΔR²为消融模型减去完整模型；负值表示删除该组后性能下降。")
    add_heading(doc, "5.3.2 不同任务的来源贡献", 3)
    rows=[]
    for _,r in ablation.iterrows():
        rows.append([r.task,r.source_group,f"{r.pooled_delta_R2:.3f}",f"{r.pooled_delta_RMSE:.4f}",f"{r.station_win_rate_RMSE:.1%}",r.classification])
    add_table(doc, "表5-1 四个代表任务的来源级消融结果", ["任务", "删除来源", "ΔR²", "ΔRMSE", "RMSE站点胜率", "判定"], rows)
    add_body(doc, "Chl-a表层删除空间静态信息后R²下降0.087，Secchi下降0.171，两者均被判定为IMPORTANT；气象信息在Secchi、TN表层和TP中层中也表现为IMPORTANT。人类活动代理在Chl-a、Secchi和TP中层为NEUTRAL，说明现有代理变量的额外预测信息有限，但不能据此断言人类活动对真实水质过程无作用。")
    add_figure(doc, OTHER_FIGURES[9][2], "图5-1 四个代表任务的多源环境信息消融比较", 15.5)
    add_heading(doc, "5.3.3 来源贡献的稳定性", 3)
    add_body(doc, "来源判断综合pooled变化、站点胜率和bootstrap区间。部分来源删除后总体指标变化很小，但站点方向不一致，说明信息可能被相关变量替代；因此SHAP高贡献与消融显著下降并非一一对应。")
    add_heading(doc, "5.3.4 遥感输入的实际范围与研究限制", 3)
    add_body(doc, "本研究的77特征包括海洋物理和多源环境代理，但不能把所有环境变量统称为遥感波段。对中、底层任务，表面观测的代表性有限；云覆盖、时间匹配和可用像元也会限制光学信息。正文仅依据冻结分类讨论实际进入模型的字段。")
    add_heading(doc, "5.4 空间异质性与模型适用边界讨论", 2)
    add_body(doc, "四任务SHAP均把经纬度排在前列，来源消融也显示空间静态信息对Chl-a和Secchi重要。这一一致性支持模型显著利用空间结构，但经纬度可能同时代理水团、岸距、污染源和监测网络布局。模型在新站点上的性能取决于新站点是否仍处于训练网络覆盖的环境范围，不能据此推断任意海域的空间外推能力。")
    add_heading(doc, "5.5 工程应用价值与局限性", 2)
    add_body(doc, "研究结果可用于监测网络内的风险筛查、模型选择与信息源优先级判断，但尚不足以替代法定监测或直接部署为实时预警系统。主要限制包括月度观测频率、极端样本稀少、未显式纳入完整水动力过程、分区样本量不均，以及原始Git commit和逐折环境锁文件缺失。解释模型虽与正式预测数值等价，仍不是原始冻结二进制。")
    add_heading(doc, "5.6 本章小结", 2)
    add_body(doc, "分区建模的收益主要集中于部分Chl-a和Secchi场景，TN与TP未表现出普遍优势。来源消融与SHAP共同说明空间静态和气象信息在代表任务中较重要，但相关性、代理变量和样本覆盖限制了物理机制层面的解释。")


def chapter_six(doc):
    add_heading(doc, "第6章 结论与展望", 1)
    add_heading(doc, "6.1 主要研究结论", 2)
    for t in [
        "（1）在统一的77个严格特征和冻结站点留出条件下，TabICLv2在10项正式任务上的pooled R²均高于XGBoost。改善幅度和稳健性具有明显任务差异，Chl-a三水层、Secchi及TN表层证据最稳定。",
        "（2）总体指标改善并不等于所有站点均改善。TabICLv2的RMSE站点胜率为70.2%—88.1%，说明多数站点受益，但仍存在局地退化；bootstrap区间进一步区分ROBUST与MODERATE结果。",
        "（3）四个代表任务的现代模型比较显示，TabICLv2在Chl-a表层和Secchi领先，TabPFN在TN表层略优，在TP中层与TabICLv2接近，表明模型适用性依赖任务。",
        "（4）20个数值等价复现的XGBoost折模型支持可追溯解释。空间坐标在四任务中贡献较高，气温、云量、风场和时间变量呈现任务特异性；一维ALE较稳定，部分二维组合因联合覆盖不足只能作条件性讨论。",
        "（5）来源消融显示空间静态信息对Chl-a表层和Secchi重要，气象信息对Secchi、TN表层和TP中层重要。分区建模对部分Chl-a和Secchi任务有帮助，但不适合作为TN和TP的普遍策略。",
    ]: add_body(doc,t)
    add_heading(doc, "6.2 研究创新与实际意义", 2)
    add_body(doc, "本文在应用层面将表格基础模型引入香港近岸多指标、多水层水质预测，并以严格空间验证而非单一随机划分评价模型；同时将逐站点改善、bootstrap、来源消融和覆盖约束下的SHAP/ALE纳入同一证据链。该框架有助于识别模型何时有效、何时证据不足。")
    add_heading(doc, "6.3 研究不足与展望", 2)
    add_body(doc, "后续工作可从四方面推进：增加河流径流、采样时刻和垂向水动力信息；扩大极端水质和稀疏海区样本；在独立年份和外部海域开展预注册验证；探索能够显式表达空间邻接与不确定性的模型。任何进一步解释都应继续区分模型关联与环境因果，并保留特征覆盖和跨折稳定性检查。")
    add_heading(doc, "结 论", 1)
    add_body(doc, "本文完成了香港近岸10项水质任务的统一空间验证和四个代表任务的现代模型、来源消融与可解释性对照。结果支持表格基础模型在若干任务中的预测优势，也表明这种优势受指标、水层、空间位置与样本覆盖约束。论文的核心结论不是某一模型在所有场景中绝对最优，而是严格划分、逐站点评价和解释边界对于近岸水质预测不可或缺。")


def add_back_matter(doc):
    add_heading(doc, "参考文献", 1)
    refs=[
        "[1] CHEN T, GUESTRIN C. XGBoost: A scalable tree boosting system[C]//Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York: ACM, 2016: 785-794. DOI: 10.1145/2939672.2939785.",
        "[2] HOLLMANN N, MÜLLER S, PURUCKER L, et al. Accurate predictions on small data with a tabular foundation model[J]. Nature, 2025, 637: 319-326. DOI: 10.1038/s41586-024-08328-6.",
        "[3] MÜLLER S, HOLLMMANN N, ARTEAGA A, et al. Transformers can do Bayesian inference[C]//International Conference on Learning Representations. 2022.",
        "[4] GORISHNIY Y, KOTELNIKOV A, BABENKO A. TabM: Advancing tabular deep learning with parameter-efficient ensembling[C]//International Conference on Learning Representations. 2025.",
        "[5] LUNDBERG S M, LEE S I. A unified approach to interpreting model predictions[C]//Advances in Neural Information Processing Systems. 2017, 30: 4765-4774.",
        "[6] LUNDBERG S M, ERION G, CHEN H, et al. From local explanations to global understanding with explainable AI for trees[J]. Nature Machine Intelligence, 2020, 2: 56-67. DOI: 10.1038/s42256-019-0138-9.",
        "[7] APLEY D W, ZHU J. Visualizing the effects of predictor variables in black box supervised learning models[J]. Journal of the Royal Statistical Society: Series B, 2020, 82(4): 1059-1086. DOI: 10.1111/rssb.12377.",
        "[8] BREIMAN L. Random forests[J]. Machine Learning, 2001, 45: 5-32. DOI: 10.1023/A:1010933404324.",
        "[9] FRIEDMAN J H. Greedy function approximation: A gradient boosting machine[J]. The Annals of Statistics, 2001, 29(5): 1189-1232. DOI: 10.1214/aos/1013203451.",
        "[10] MOLNAR C. Interpretable Machine Learning: A Guide for Making Black Box Models Explainable[M]. 2nd ed. 2022.",
        "[11] HONG KONG ENVIRONMENTAL PROTECTION DEPARTMENT. Marine water quality in Hong Kong[EB/OL]. [2026-10-18]. https://www.epd.gov.hk/epd/english/environmentinhk/water/marine_quality/marine_quality.html.",
        "[12] EUROPEAN CENTRE FOR MEDIUM-RANGE WEATHER FORECASTS. ERA5 hourly data on single levels from 1940 to present[DB/OL]. Copernicus Climate Change Service Climate Data Store.",
    ]
    for ref in refs:
        p=doc.add_paragraph(style="参考文献"); set_run_font(p.add_run(ref), size=10.5)
    add_heading(doc, "致 谢", 1)
    add_body(doc, "本部分涉及导师、课题组、数据支持与个人经历，应由作者根据真实情况补充。为避免虚构人名和贡献，本整合稿不自动生成具体致谢内容。")
    add_heading(doc, "附 录", 1)
    add_heading(doc, "附录A 解释分析覆盖范围", 2)
    add_body(doc, "SHAP/ALE解释限于Chl-a表层、Secchi、TN表层和TP中层四个代表任务；其余水层仅列为后续候选。17张解释图均来自XGB_SHAP_ALE_V2冻结输出。")
    add_heading(doc, "附录B 复现与结果路径", 2)
    add_body(doc, "正式闭环：runs/final_experiment_closure_audit_v1；解释模型复现：runs/xgb_interpretation_refit_v1；SHAP/ALE：runs/xgb_shap_ale_v2；现代基线：runs/modern_baseline_gate_v1；来源消融：runs/multisource_ablation_v1。")
    add_heading(doc, "个人简历", 1)
    add_body(doc, "待作者填写真实教育经历与个人信息。")
    add_heading(doc, "在学期间的研究成果及发表的学术论文", 1)
    add_body(doc, "待作者依据已发表、已录用或正式登记的成果填写；本稿不自动生成未核实成果。")


def add_headers_footers(doc):
    for sec in doc.sections:
        sec.different_first_page_header_footer = True
        sec.odd_and_even_pages_header_footer = True
        sec.header_distance = Mm(12)
        sec.footer_distance = Mm(12)
        for header, text in [(sec.header, "基于多源环境数据与表格基础模型的近岸水质预测及空间泛化研究"), (sec.even_page_header, "福州大学硕士学位论文")]:
            p=header.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Cm(0)
            set_run_font(p.add_run(text), size=10.5)
            pPr=p._p.get_or_add_pPr(); borders=OxmlElement("w:pBdr"); bottom=OxmlElement("w:bottom")
            for k,v in [("val","single"),("sz","6"),("space","1"),("color","000000")]: bottom.set(qn(f"w:{k}"),v)
            borders.append(bottom); pPr.append(borders)
        for footer in (sec.footer, sec.even_page_footer):
            p=footer.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Cm(0); add_field(p,"PAGE")


def write_audits(fig_rows, core, shap_inv):
    fig_index=[]
    for no,title,path,section in OTHER_FIGURES:
        fig_index.append({"figure_no":no,"title":title,"path":str(path.relative_to(ROOT)),"section":section,"source":"既有论文图表","embedded":"YES"})
    for no,row in fig_rows:
        fig_index.append({"figure_no":no,"title":row.title,"path":row.png_path,"section":"3.5-3.8","source":row.data_path,"embedded":"YES"})
    pd.DataFrame(fig_index).to_csv(OUT_DIR/"论文图表索引.csv",index=False,encoding="utf-8-sig")
    mapping=[
        ["第2章任务与样本数",str(CORE.relative_to(ROOT)),"task/depth/n_stations/n_obs"],
        ["第4章十任务主结果",str(CORE.relative_to(ROOT)),"pooled_R2/RMSE/MAE/delta/station win rate"],
        ["第4章四模型比较",str(MODERN.relative_to(ROOT)),"pooled_R2/RMSE/MAE"],
        ["第5章来源消融",str(ABLATION.relative_to(ROOT)),"pooled delta/bootstrap/classification"],
        ["第3章解释模型核验",str(REFIT_DECISION.relative_to(ROOT)),"20/20 exact"],
        ["第3章SHAP/ALE",str(FIG_INV.relative_to(ROOT)),"17 distinct figures"],
    ]
    pd.DataFrame(mapping,columns=["正文位置","源文件","使用字段或结论"]).to_csv(OUT_DIR/"论文结果数据来源映射.csv",index=False,encoding="utf-8-sig")
    pending="""# 论文待人工确认事项

1. 封面：分类号、UDC、密级、学号、学位类别、学科专业、研究方向、作者、导师和完成日期。
2. 声明页：必须从学校正式模板插入原文，并由作者和导师亲笔签署；本稿未生成签名。
3. 中文题目最终用语与是否严格控制在30字以内，请由学院确认。
4. 自动目录、页码和图表编号需在Microsoft Word中全选后按F9更新。
5. TabICLv2原始论文的最终出版书目信息尚需作者结合实际采用版本核对；本稿未虚构条目。
6. 参考文献目前为方法与数据源的核心条目，不等同于完整文献综述库；提交前需补充并逐条核对水质预测领域文献。
7. 致谢、个人简历和在学成果必须由作者按真实情况填写。
8. 学校规范PDF未在当前项目中定位到本地副本；页面与样式按任务中列出的2016年7月修订要求实现，声明正式文本仍待模板核对。
"""
    (OUT_DIR/"论文待人工确认事项.md").write_text(pending,encoding="utf-8")
    report=f"""# 论文整合与格式审计报告

## 整合结论

- 基线原稿：`{SOURCE_DOCX.relative_to(ROOT)}`，未覆盖。
- 新稿：`{OUT_DOCX.name}`。
- 正式结果口径：`final_core_results_table_v2.csv`，10项站点留出任务。
- 解释图：使用 `xgb_shap_ale_v2`，17张不同PNG全部嵌入；未误用旧v1目录，也未将PNG/SVG重复计数。
- 未执行模型训练、预测生成、SHAP或ALE重算。

## 内容处理

旧稿的随机80/20划分和早期XGBoost结果与最终闭环口径冲突，因此仅保留其封面信息框架和经核实的数据背景，不沿用旧结果结论。第3—5章依据最终登记、Family V2、现代基线、解释分析和来源消融重建。

## 格式设置

- A4；上/左25 mm，下/右20 mm。
- 章标题黑体小二、一级节黑体三号、二级节黑体小三、三级节黑体小四；正文宋体小四；固定20 pt。
- 图题、表题宋体五号；表格采用无左右边线的三线表式样并禁止行内跨页。
- 已写入自动目录域、PAGE页码域和奇偶页页眉。Word首次打开后仍需F9刷新。
- 参考文献宋体五号、固定17 pt。

## 结构检查

- 正文包含六章、结论、致谢、参考文献、附录、个人简历和在学成果。
- 17张SHAP/ALE图以及10张既有研究/结果图已嵌入，图题与正文位置可追溯。
- 核心数值来自冻结CSV，没有使用测试集标签筛选解释变量。

## 已知限制

- 当前参考文献只完成核心方法和数据源条目的可读排版，领域综述仍需进一步扩充和人工核对。
- 学校正式声明页文本和封面个人信息未提供，均保留待确认项。
- Word域的显示页码依赖桌面Word刷新；脚本不能替代最终人工签字和学院格式审查。
"""
    (OUT_DIR/"论文整合与格式审计报告.md").write_text(report,encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    required=[SOURCE_DOCX,CORE,REGISTRY,MODERN,ABLATION,GROUPS,FIG_INV,REFIT_DECISION]
    missing=[p for p in required if not p.exists()]
    if missing: raise FileNotFoundError("缺少必要文件："+", ".join(map(str,missing)))
    core=pd.read_csv(CORE); modern=pd.read_csv(MODERN); ablation=pd.read_csv(ABLATION); groups=pd.read_csv(GROUPS); shap_inv=pd.read_csv(FIG_INV)
    assert len(core)==20 and core.task.nunique()==10
    assert len(shap_inv)==17 and all((ROOT/Path(p)).exists() for p in shap_inv.png_path)

    doc=Document(); configure_styles(doc); add_front_matter(doc)
    chapter_one(doc); chapter_two(doc,core,groups); fig_rows=chapter_three(doc,shap_inv)
    chapter_four(doc,core,modern); chapter_five(doc,ablation); chapter_six(doc); add_back_matter(doc); add_headers_footers(doc)
    props=doc.core_properties
    props.title="基于多源环境数据与表格基础模型的近岸水质预测及空间泛化研究"
    props.subject="硕士学位论文整合排版稿"
    props.comments="依据冻结实验结果生成；解释对象为XGBoost，不代表TabICLv2内部机理。"
    doc.save(OUT_DOCX)
    with zipfile.ZipFile(OUT_DOCX) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f"DOCX包损坏：{bad}")
    write_audits(fig_rows,core,shap_inv)
    manifest={"docx":str(OUT_DOCX),"sha256":sha256(OUT_DOCX),"paragraphs":len(doc.paragraphs),"tables":len(doc.tables),"inline_shapes":len(doc.inline_shapes),"shap_ale_figures":17,"all_embedded_figures":len(doc.inline_shapes)}
    (OUT_DIR/"构建清单.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
