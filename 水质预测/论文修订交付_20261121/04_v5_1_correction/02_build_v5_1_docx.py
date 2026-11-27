"""以V5为底稿，仅修复指定图表编号、位置、引用与冻结表格字段。"""
from __future__ import annotations

import csv
import json
import re
from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "论文修订交付_20261121/01_manuscript/水质预测硕士学位论文_V5_正文修订.docx"
DEST = OUT / "水质预测硕士学位论文_V5.1.docx"
PLAN = ROOT / "论文修订交付_20261121/02_figure_table_plan"
RANK = ROOT / "结果/论文图表/figure_revision_v2/data/30核心样本_77特征完整排名.csv"
CORE_SAMPLES = ROOT / "结果/论文图表/figure_revision_v2/source_inventory/40组文件与30核心样本对应.csv"
UNITS = ROOT / "结果/论文最终图件重构_v1/05_数据核验/77项正式特征单位与时间窗核对.csv"
XGB_ASSETS = ROOT / "runs/xgb_shap_ale_v2/00_preflight/00_model_and_preprocessor_check.csv"
CORE_RESULTS = ROOT / "runs/final_experiment_closure_audit_v1/final_core_results_table_v2.csv"
MODERN = ROOT / "transfer/llm_extension_transfer_v1/results/modern_baseline_gate_v1/modern_baseline_core_metrics.csv"
assert SRC.is_file() and not DEST.exists()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


doc = Document(SRC)
assert len(doc.tables) == 13 and len(doc.inline_shapes) == 94
original_references = [p.text for p in doc.paragraphs if p.style.name == "参考文献"]
source_table_values = [[row.cells[j].text for j in range(len(row.cells))]
                       for table in doc.tables for row in table.rows]
changes = []


def record(kind, location, before, after, evidence="", status="通过"):
    changes.append({"类型": kind, "修改位置": location, "修改前": before,
                    "修改后": after, "证据": evidence, "验收": status})


def caption(number):
    found = [p for p in doc.paragraphs if p.style.name in ("图题", "表题")
             and re.match(rf"^{re.escape(number)}(?:\s|$)", p.text)]
    assert len(found) == 1, (number, len(found))
    return found[0]


def heading(text):
    found = [p for p in doc.paragraphs if p.style.name.startswith("Heading") and p.text.startswith(text)]
    assert len(found) == 1, (text, len(found))
    return found[0]


def normal_start(text):
    found = [p for p in doc.paragraphs if p.style.name == "Normal" and p.text.startswith(text)]
    assert len(found) == 1, (text, len(found))
    return found[0]


def insert_before(element, text, reason):
    anchor = element._p if isinstance(element, Paragraph) else element
    node = OxmlElement("w:p")
    anchor.addprevious(node)
    p = Paragraph(node, doc._body)
    p.style = "Normal"
    p.add_run(text)
    p.paragraph_format.keep_with_next = True
    record("引用补充", reason, "无前置明确引用", text)
    return p


def image_para(fig_caption):
    node = fig_caption._p.getprevious()
    assert node is not None and node.tag == qn("w:p") and node.xpath(".//w:drawing"), fig_caption.text
    return Paragraph(node, doc._body)


def figure_nodes(fig_caption):
    image = image_para(fig_caption)
    nodes = [image._p, fig_caption._p]
    next_node = fig_caption._p.getnext()
    if next_node is not None and next_node.tag == qn("w:p"):
        next_p = Paragraph(next_node, doc._body)
        if next_p.text.startswith("注："):
            nodes.append(next_node)
    return nodes


def move_nodes_before(nodes, anchor, reason):
    anchor_node = anchor._p if isinstance(anchor, Paragraph) else anchor
    for node in nodes:
        anchor_node.addprevious(node)
    record("位置调整", reason, "原小节", "指定小节", "仅迁移既有OOXML节点与原图片二进制")


def table_nodes(table_caption):
    table_node = table_caption._p.getnext()
    assert table_node.tag == qn("w:tbl"), table_caption.text
    return [table_caption._p, table_node]


def replace_picture(fig_caption, path, reason):
    assert path.is_file()
    p = image_para(fig_caption)
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(15.5))
    record("指定图件重导", reason, "V5内嵌旧图", str(path), "冻结绘图数据；其余92张不重导")


def change_text(p, text, reason):
    old = p.text
    if old != text:
        p.text = text
        record("正文或图注", reason, old, text)


def append_text(p, sentence, reason):
    old = p.text
    assert sentence not in old
    p.add_run(("" if old.endswith(("。", "；")) else "。") + sentence)
    record("引用补充", reason, old[-120:], p.text[-180:])


def swap_references():
    swap = {"图4-4": "图4-5", "图4-5": "图4-4", "图5-1": "图5-2", "图5-2": "图5-1"}
    pattern = re.compile(r"图(?:4-[45]|5-[12])(?!\d)")
    count = Counter()
    for p in doc.paragraphs:
        if p.style.name.startswith("toc"):
            continue
        old = p.text
        if pattern.search(old):
            # Word把“图”和“4-4”拆为两个run；必须按完整段落处理。
            for match in pattern.finditer(old):
                count[match.group(0)] += 1
            p.text = pattern.sub(lambda match: swap[match.group(0)], old)
    record("图号迁移", "全文正文、图题、图注及附录引用", "图4-4↔图4-5；图5-1↔图5-2",
           "按首次章节顺序重新编号", json.dumps(dict(count), ensure_ascii=False))


def lock_titles():
    rows = read_csv(PLAN / "全文图件总表.csv")
    targets = {row["图号"]: row["锁定图名"] for row in rows}
    targets.update({"图4-4": "四项代表任务站点RMSE变化的空间分布",
                    "图4-5": "四项代表任务中XGBoost、TabICLv2、TabPFN和TabM的比较",
                    "图5-1": "历史全域与分区建模的任务收益及折次一致性",
                    "图5-2": "四项代表任务的多源环境信息消融比较"})
    for number, title in targets.items():
        p = caption(number)
        desired = number + " " + title
        if p.text != desired:
            change_text(p, desired, "图题" + number)
    assert len(targets) == 94
    for row in read_csv(PLAN / "全文表格总表.csv"):
        number, title = row["表号"], row["锁定表名"]
        p = caption(number)
        desired = number + " " + title
        if p.text != desired:
            change_text(p, desired, "表题" + number)


def table_for(number):
    from docx.table import Table
    return Table(table_nodes(caption(number))[1], doc._body)


def set_headers(table, headers):
    assert len(table.columns) == len(headers)
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = text


def add_rows_to(table, wanted):
    while len(table.rows) < wanted:
        table.add_row()
    while len(table.rows) > wanted:
        table._tbl.remove(table.rows[-1]._tr)


def set_compact_table(table, widths_cm, font_pt=8.0):
    assert len(widths_cm) == len(table.columns)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_width = table._tbl.tblPr.find(qn("w:tblW"))
    assert table_width is not None
    table_width.set(qn("w:w"), str(sum(Cm(v).twips for v in widths_cm)))
    table_width.set(qn("w:type"), "dxa")
    for column, width in zip(table.columns, widths_cm):
        column.width = Cm(width)
    for row_index, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        if row_index == 0 and not tr_pr.xpath("./w:tblHeader"):
            tr_pr.append(OxmlElement("w:tblHeader"))
        if not tr_pr.xpath("./w:cantSplit"):
            tr_pr.append(OxmlElement("w:cantSplit"))
        for cell, width in zip(row.cells, widths_cm):
            cell.width = Cm(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                for run in p.runs:
                    run.font.size = Pt(font_pt)


def fix_tables():
    # 表2-3：正文77字段序号与单位核对表逐字段同序，严禁按名称排序。
    unit_rows = read_csv(UNITS)
    unit_map = {r["正式字段"]: r["最终单位"] for r in unit_rows}
    assert len(unit_map) == 77 and all(unit_map.values())
    t = table_for("表2-3")
    assert len(t.rows) == 78 and len(t.columns) == 6
    before = [[c.text for c in row.cells] for row in t.rows]
    t.add_column(Cm(1.55))
    set_headers(t, ["序号", "特征类别", "精确英文变量名", "中文名称", "数据来源", "时间尺度或构造方式", "单位"])
    for index, row in enumerate(t.rows[1:], 1):
        name = row.cells[2].text
        assert int(row.cells[0].text) == index and name in unit_map
        row.cells[6].text = unit_map[name]
        assert [c.text for c in row.cells[:6]][1:] == before[index][1:]
    assert len({row.cells[2].text for row in t.rows[1:]}) == 77
    set_compact_table(t, [0.8, 1.3, 3.0, 2.8, 1.5, 4.1, 2.0], 7.8)
    record("表格字段", "表2-3", "6列，无单位", "7列，77字段原序增加单位", str(UNITS))

    # 表2-5：原20折资产核验文件仅支撑受控复现XGBoost的四任务五折。
    assets = read_csv(XGB_ASSETS)
    assert len(assets) == 20 and all(r["status"] == "PASS" and r["feature_order_match"] == "True" for r in assets)
    grouped = defaultdict(list)
    for row in assets:
        grouped[row["task"]].append(row)
    tasks = [("chla_surface", "Chl-a表层"), ("secchi", "Secchi"),
             ("tn_surface", "TN表层"), ("tp_middle", "TP中层")]
    assert all(len(grouped[task]) == 5 for task, _ in tasks)
    t = table_for("表2-5")
    while len(t.columns) < 6:
        t.add_column(Cm(2))
    add_rows_to(t, 5)
    set_headers(t, ["解释资产", "对应模型", "特征版本", "解释类型及协议", "核验状态", "说明"])
    for row, (task, name) in zip(t.rows[1:], tasks):
        values = [f"{name}五折资产", "受控复现XGBoost", "正式77特征",
                  "TreeSHAP、一维ALE；五折", "5/5折通过",
                  "逐样本数值等价；非找回原始二进制"]
        for cell, value in zip(row.cells, values):
            cell.text = value
    set_compact_table(t, [2.3, 2.3, 1.75, 3.1, 1.55, 4.5], 8.0)
    record("表格字段", "表2-5", "核验项目｜结果｜说明", "六列，四任务五折资产与协议", str(XGB_ASSETS))

    # 表3-1：冻结30核心×77的真实绝对SHAP均值；样本清单给出站点数。
    ranks = read_csv(RANK)
    sample_rows = [r for r in read_csv(CORE_SAMPLES) if r["用于图3-5"] == "True"]
    assert len(ranks) == 770 and len(sample_rows) == 30
    by_task = defaultdict(list)
    for row in ranks:
        by_task[row["task"]].append(row)
    sites = defaultdict(set)
    sample_count = Counter()
    for row in sample_rows:
        assert row["seed"] == "42" and row["phase"] == "CORE_3"
        sites[row["task"]].add(row["obs_id"].split("_")[0])
        sample_count[row["task"]] += 1
    task_names = {"Chl-a底层": "chla_bottom", "Chl-a中层": "chla_middle", "Chl-a表层": "chla_surface",
                  "Secchi": "secchi", "TN底层": "tn_bottom", "TN中层": "tn_middle", "TN表层": "tn_surface",
                  "TP底层": "tp_bottom", "TP中层": "tp_middle", "TP表层": "tp_surface"}
    old_table = table_for("表3-1")
    assert len(old_table.rows) == 11 and len(old_table.columns) == 4
    old_order = [row.cells[0].text for row in old_table.rows[1:]]
    t = doc.add_table(rows=11, cols=6)
    t.style = "Table Grid"
    old_table._tbl.addprevious(t._tbl)
    old_table._tbl.getparent().remove(old_table._tbl)
    set_headers(t, ["任务", "目标单位", "主要贡献变量", "平均绝对SHAP贡献", "独立核心样本数", "独立站点数"])
    for row, name in zip(t.rows[1:], old_order):
        task = task_names[name]
        top = sorted(by_task[task], key=lambda x: (float(x["rank"]), x["feature"]))[:3]
        assert len(by_task[task]) == 77 and sample_count[task] == 3
        unit = "μg/L" if task.startswith("chla") else "m" if task == "secchi" else "mg/L"
        values = [name, unit, "\n".join(x["feature"] for x in top),
                  "\n".join(f"{float(x['mean_abs_shap']):.6g}" for x in top),
                  str(sample_count[task]), str(len(sites[task]))]
        for cell, value in zip(row.cells, values):
            cell.text = value
    set_compact_table(t, [2.0, 1.4, 4.3, 3.0, 2.4, 2.4], 8.0)
    record("表格字段", "表3-1", "前3变量、来源组、协议/覆盖", "六列，真实均值与3核心样本/独立站点", str(RANK) + "；" + str(CORE_SAMPLES))

    # 表4-2：不重新算预测；仅读取冻结总体指标与登记的delta_R2。
    results = read_csv(CORE_RESULTS)
    by_pair = {(r["task"], r["model"]): r for r in results}
    old_table = table_for("表4-2")
    assert len(old_table.rows) == 11 and len(old_table.columns) == 6
    old_order = [row.cells[0].text for row in old_table.rows[1:]]
    t = doc.add_table(rows=11, cols=10)
    t.style = "Table Grid"
    old_table._tbl.addprevious(t._tbl)
    old_table._tbl.getparent().remove(old_table._tbl)
    set_headers(t, ["任务", "XGB\nR²", "TabICLv2\nR²", "ΔR²", "XGB\nRMSE", "TabICLv2\nRMSE",
                    "XGB\nMAE", "TabICLv2\nMAE", "站点RMSE\n胜率", "稳健性"])
    for row, name in zip(t.rows[1:], old_order):
        task = task_names[name]
        xgb, tab = by_pair[task, "XGBoost"], by_pair[task, "TabICLv2"]
        assert xgb["feature_set"] == tab["feature_set"] == "X_strict_77"
        values = [name, f"{float(xgb['pooled_R2']):.3f}", f"{float(tab['pooled_R2']):.3f}",
                  f"{float(tab['delta_R2']):.3f}", f"{float(xgb['RMSE']):.4f}", f"{float(tab['RMSE']):.4f}",
                  f"{float(xgb['MAE']):.4f}", f"{float(tab['MAE']):.4f}",
                  f"{100*float(tab['station_win_rate_RMSE']):.1f}%", "稳健" if tab["robustness_class"] == "ROBUST" else "中等"]
        for cell, value in zip(row.cells, values):
            cell.text = value
    set_compact_table(t, [3.2, 2.0, 2.6, 2.4, 2.3, 2.5, 2.1, 2.4, 2.5, 2.5], 8.5)
    put_table_on_landscape_page(caption("表4-2"), t)
    record("表格字段", "表4-2", "6列，未列两模型RMSE/MAE", "10列，按原任务顺序补齐冻结指标", str(CORE_RESULTS))

    # 表4-3保留16个真实可用模型记录，逐单元格与冻结四模型比较文件比对。
    modern = {(r["task"], r["model"]): r for r in read_csv(MODERN)}
    t = table_for("表4-3")
    assert len(t.rows) == 17 and len(modern) == 16
    model_map = {"Chl-a表层": "chla_surface", "Secchi": "secchi", "TN表层": "tn_surface", "TP中层": "tp_middle"}
    for row in t.rows[1:]:
        task, model = model_map[row.cells[0].text], row.cells[1].text
        assert (task, model) in modern
        frozen = modern[task, model]
        for col, key, digits in ((2, "pooled_R2", 3), (3, "RMSE", 4), (4, "MAE", 4)):
            assert abs(float(row.cells[col].text) - float(frozen[key])) <= 0.51 * 10 ** (-digits)
    record("冻结数据核验", "表4-3", "四项任务现有16行", "保留原值；TN表层四模型完整", str(MODERN))
    note = normal_start("注：MET=气象")
    change_text(note, "注：完整模型使用MET、RHT、OCN、HUM、TMP、SPA六个来源组；消融模型删除其中一个来源组、保留其余五组。ΔR²=消融模型R²−完整模型R²，ΔRMSE=消融模型RMSE−完整模型RMSE。负ΔR²表示删除后R²下降；SHAP特征归因不能直接等同于来源级消融贡献。", "表5-1表注")


def put_table_on_landscape_page(table_caption, table):
    """仅表4-2使用独立横向节；前后正文保持原纵向页面。"""
    base = doc._element.body.sectPr
    assert base is not None

    def section_copy(landscape):
        sect = deepcopy(base)
        section_type = sect.find(qn("w:type"))
        if section_type is None:
            section_type = OxmlElement("w:type")
            sect.insert(0, section_type)
        section_type.set(qn("w:val"), "nextPage")
        if landscape:
            page_size = sect.find(qn("w:pgSz"))
            assert page_size is not None
            page_size.set(qn("w:w"), str(Cm(29.7).twips))
            page_size.set(qn("w:h"), str(Cm(21).twips))
            page_size.set(qn("w:orient"), "landscape")
        return sect

    def break_paragraph(sect):
        p = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        ppr.append(sect)
        p.append(ppr)
        return p

    table_caption._p.addprevious(break_paragraph(section_copy(False)))
    table._tbl.addnext(break_paragraph(section_copy(True)))
    record("表格版面", "表4-2", "纵向页十列断字", "独立横向页，前后恢复纵向", "仅变更页面方向；全部指标原值保留")


def amend_reference_and_positions():
    # 移动图4-2至4.2.4；原图片及注释节点保持原样。
    move_nodes_before(figure_nodes(caption("图4-2")), heading("4.3 "), "图4-2至4.2.4")
    # 将原四模型图（此时已重编号为图4-5）及表4-3移入4.5。
    move_nodes_before(table_nodes(caption("表4-3")), heading("4.6 "), "表4-3至4.5")
    move_nodes_before(figure_nodes(caption("图4-5")), heading("4.6 "), "图4-5至4.5")
    # 移去4.4内过时的“表与图任务不同”说明；4.5改为同任务说明。
    obsolete = normal_start("表4-3保留四项代表任务的冻结四模型比较")
    record("正文一致性", "4.4旧四模型段", obsolete.text, "迁至4.5后统一说明")
    obsolete._p.getparent().remove(obsolete._p)
    p = normal_start("TabICLv2的优势适用于本研究冻结的十项站点留出任务")
    change_text(p, "表4-3与图4-5共同比较Chl-a表层、Secchi、TN表层和TP中层四项代表任务的XGBoost、TabICLv2、TabPFN和TabM结果，使用同一冻结比较口径。TabICLv2在Chl-a表层与Secchi的pooled R²较高；TN表层TabPFN为0.417，略高于TabICLv2的0.402；TP中层两者接近。四项代表任务的结果不能推广为全部十项任务的四模型排序。TabICLv2在正式十任务站点留出比较中的优势，也不能直接外推到独立海域、不同年份或实时业务。", "4.5四模型任务及直接引用")
    p = normal_start("四项任务的站点误差空间图见")
    change_text(p, "图4-4展示四项代表任务的站点误差空间分布。各任务使用独立的零中心ΔRMSE色标，不跨水质目标比较误差绝对值。", "4.4空间图引用")
    note = normal_start("注：指定Chl-a表层")
    change_text(note, "注：图示为同一冻结四模型比较中的Chl-a表层、Secchi、TN表层、TP中层，各任务均有四模型的实际pooled R²。不同水质目标的R²可用于各任务内部模型比较，不表示跨目标误差量纲相同。", "新图4-5图注")
    # 图5-1与图5-2按新编号保留既有图片，修订相邻注释中的指向。
    note = normal_start("注：历史扩展实验，78特征")
    assert "78特征" in note.text
    note = normal_start("注：四任务XGBoost六来源冻结消融")
    assert "XGBoost" in note.text

    # 精确补充原报告指出的前置引用；图片引用插在图片前、表引用插在表题前。
    references = [
        ("图2-3", "图2-3说明各来源数据经站点、采样时间和obs_id匹配后形成统一样本主键。", "2.4"),
        ("图2-4", "图2-4按真实任务展示四类水质指标的观测分布，其中Secchi仅作为单一任务。", "2.9"),
        ("图3-1", "图3-1展示历史四项代表任务五折固定批查询下的任务内SHAP相对贡献份额。", "3.2.2"),
        ("图3-8", "图3-8至图3-11分别展示Chl-a表层、Secchi、TN表层和TP中层的历史五折一维ALE响应；空分箱相关限制按本节说明。", "3.3.1"),
        ("图4-1", "图4-1对照十项任务的pooled R²及站点RMSE改善覆盖，二者应分别解读。", "4.2.1"),
        ("表2-2", "六类正式特征来源的数量及含义列于表2-2。", "2.6"),
        ("表2-5", "表2-5列出XGBoost解释模型资产与正式特征版本的核验情况。", "2.8.3"),
        ("表2-6", "表2-6汇总各评价指标的定义和使用范围。", "2.8.6"),
        ("表4-1", "表4-1给出十项任务的XGBoost正式留站基准结果。", "4.1.1"),
        ("表4-2", "表4-2并列给出两模型的总体指标、误差指标及站点改善覆盖。", "4.2.1"),
        ("表5-1", "表5-1列出四项代表任务XGBoost六来源消融的冻结指标。", "5.3.2"),
    ]
    for number, sentence, section in references:
        target = caption(number)
        anchor = image_para(target) if number.startswith("图") else target
        if number == "表4-2":
            # 表4-2前有横向节分页符；前置引用应留在4.2.1纵向正文页。
            anchor = target._p.getprevious()
            assert anchor is not None and anchor.xpath("./w:pPr/w:sectPr")
        insert_before(anchor, sentence, section + " " + number)
    # 图4-2已在4.2.4，紧邻图片补明确引用。
    target = caption("图4-2")
    insert_before(image_para(target), "图4-2以两组独立色标呈现不同水质指标及水层的ΔR²与站点RMSE改善率；Secchi不按水层重复。", "4.2.4 图4-2")
    # 附录E/F首次整体引用必须在首图之前。
    for number, sentence in [
        ("图E-1", "附录E的图E-1至图E-20依次列出十项正式任务的20条P1一维ALE曲线及真实分箱支持。"),
        ("图F-1", "附录F的图F-1至图F-10分别列出十项任务三个独立核心样本的SHAP前10特征。")]:
        insert_before(image_para(caption(number)), sentence, "附录前置引用" + number)


def main():
    swap_references()
    lock_titles()
    replace_picture(caption("图3-7"), OUT / "修订图件/图3-7_仅修正内部表号.png", "图3-7旧表3-2改表3-1")
    replace_picture(caption("图4-5"), OUT / "修订图件/图4-5_四项代表任务四模型比较.png", "图4-5由TN底层改TN表层，使用冻结四模型")
    fix_tables()
    amend_reference_and_positions()
    assert len(doc.inline_shapes) == 94 and len(doc.tables) == 13
    assert [p.text for p in doc.paragraphs if p.style.name == "参考文献"] == original_references
    assert len([p for p in doc.paragraphs if p.style.name == "图题"]) == 94
    assert len([p for p in doc.paragraphs if p.style.name == "表题"]) == 12
    assert "图5-3" in normal_start("图5-3给出").text
    doc.save(DEST)
    (OUT / "修改记录.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {DEST}；修改记录 {len(changes)} 项；94张图、12张编号表")


if __name__ == "__main__":
    main()
