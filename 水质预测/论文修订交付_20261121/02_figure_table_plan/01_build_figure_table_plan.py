"""只读核对 V5 图表与锁定方案，输出中文清单。"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DOC = ROOT / "论文修订交付_20261121/01_manuscript/水质预测硕士学位论文_V5_正文修订.docx"
FIGURES = ROOT / "结果/论文最终图件重构_v1/最终图件清单.csv"
MAPPING = ROOT / "结果/论文最终图件重构_v1/原图与新图编号映射.csv"
PAGES = ROOT / "论文修订交付_20261121/01_manuscript/排版核查/逐图实际页码.csv"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows, fields):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


MAIN_TITLES = {
    "图1-1": "多源环境数据与表格模型的近岸水质预测技术路线",
    "图2-1": "香港近岸研究区域及94站监测网络",
    "图2-2": "水质监测记录的时间覆盖",
    "图2-3": "多源环境数据接入与样本主键关系",
    "图2-4": "叶绿素a、透明度、总氮与总磷的观测分布",
    "图3-1": "历史五折SHAP特征相对贡献份额",
    "图3-2": "历史五折SHAP特征排名跨折差异",
    "图3-3": "Chl-a三水层P1核心样本特征贡献",
    "图3-4": "Secchi P1核心样本特征贡献",
    "图3-5": "TN三水层P1核心样本特征贡献",
    "图3-6": "TP三水层P1核心样本特征贡献",
    "图3-7": "十项任务P1核心样本特征重要性排名矩阵",
    "图3-8": "Chl-a表层历史五折一维ALE响应",
    "图3-9": "Secchi历史五折一维ALE响应",
    "图3-10": "TN表层历史五折一维ALE响应",
    "图3-11": "TP中层历史五折一维ALE响应",
    "图3-12": "Chl-a表层P1一维ALE响应",
    "图3-13": "Secchi P1一维ALE响应",
    "图3-14": "TN表层P1一维ALE响应",
    "图3-15": "TP中层P1一维ALE响应",
    "图3-16": "十项任务P1 ALE样本与独立站点双重支持",
    "图4-1": "十项任务总体提升与站点改善覆盖",
    "图4-2": "不同指标与水层的模型提升差异",
    "图4-3": "三种验证方案的评价对象与适用边界",
    "图4-4": "四项代表任务中XGBoost、TabICLv2、TabPFN和TabM的比较",
    "图4-5": "四项代表任务站点RMSE变化的空间分布",
    "图5-1": "四项代表任务的多源环境信息消融比较",
    "图5-2": "历史全域与分区建模的任务收益及折次一致性",
    "图5-3": "四个典型站点的输入覆盖与预测误差诊断",
    "图D-1": "四任务XGBoost TreeSHAP特征份额",
    "图D-10": "XGBoost一维ALE折间相关性",
    "图D-11": "TreeSHAP来源归因与消融标准化对照",
    "图D-12": "四任务XGBoost主要特征排名矩阵",
    "图H-1": "九任务P1同查询重复预测审查",
    "图H-2": "叶绿素a表层解释尺度与重复波动对照",
    "图H-3": "历史验证划分结果的独立口径对照",
}
for n, task in enumerate(("Chl-a表层", "Secchi", "TN表层", "TP中层"), 2):
    MAIN_TITLES[f"图D-{n}"] = f"{task}TreeSHAP总体解释"
for n, task in enumerate(("Chl-a表层", "Secchi", "TN表层", "TP中层"), 6):
    MAIN_TITLES[f"图D-{n}"] = f"{task}XGBoost一维ALE"

FIG_SECTIONS = {
    "图1-1": "1.5", "图2-1": "2.1", "图2-2": "2.2", "图2-3": "2.4", "图2-4": "2.9",
    "图3-1": "3.2.2", "图3-2": "3.2.3",
    **{f"图3-{n}": "3.2.4" for n in range(3, 8)},
    **{f"图3-{n}": "3.3.1" for n in range(8, 12)},
    **{f"图3-{n}": "3.3.2" for n in range(12, 16)},
    "图3-16": "3.4.1", "图4-1": "4.2.1", "图4-2": "4.2.4",
    "图4-3": "4.3", "图4-4": "4.5", "图4-5": "4.4",
    "图5-1": "5.3.2", "图5-2": "5.2", "图5-3": "5.7",
}

TABLES = {
    "表0-1": ("主要符号与含义", "前置部分", "符号、含义、单位"),
    "表2-1": ("十项正式预测任务及样本规模", "2.3", "任务名称、预测指标、水层、有效样本数、站点数"),
    "表2-2": ("严格特征集的来源分组", "2.6", "六个特征来源组、定义、特征数量"),
    "表2-3": ("77项严格输入特征的名称、来源与构造", "2.6", "字段名称、中文解释、来源类别、构造方式、单位"),
    "表2-4": ("历史候选环境数据的来源、覆盖与筛选结果", "2.7", "候选数据源、时间覆盖、空间覆盖、分辨率、实际筛选状态"),
    "表2-5": ("XGBoost解释模型资产核验", "2.8", "解释资产、模型对应关系、特征版本、解释类型、核验状态"),
    "表2-6": ("主要评价指标及解释", "2.8.6", "R²、RMSE、MAE、站点RMSE胜率、配对站点bootstrap的定义和用途"),
    "表3-1": ("十项水质任务第1折P1单查询SHAP主要变量与证据覆盖", "3.2.4", "任务、目标单位、主要变量、平均绝对SHAP贡献、独立样本支持、站点支持"),
    "表4-1": ("十项任务的XGBoost站点留出基准结果", "4.2", "十任务XGBoost正式留站指标"),
    "表4-2": ("TabICLv2相对XGBoost的十任务比较", "4.2", "XGBoost与TabICLv2的pooled R²、ΔR²、RMSE、MAE、站点改善指标"),
    "表4-3": ("四项代表任务的现代模型比较", "4.5", "四项代表任务四种模型的真实可用指标；缺失项NA"),
    "表5-1": ("四项代表任务的来源级消融结果", "5.3.2", "四项任务六来源消融的ΔR²及RMSE变化；完整/消融模型定义"),
}


def current_section(text, style, section):
    if style.startswith("Heading"):
        match = re.match(r"^(\d+(?:\.\d+){0,2})(?:\s|$)", text)
        if match:
            return match.group(1)
        match = re.match(r"^附录([A-H])", text)
        if match:
            return f"附录{match.group(1)}"
        if text == "主要符号表":
            return "前置部分"
    return section


def split_caption(text):
    match = re.match(r"^([图表](?:[0-5]|[CDEFH])-\d+)\s*(.*)$", text.strip())
    return (match.group(1), match.group(2)) if match else ("", "")


def references(text):
    # 展开“图3-3至图3-5”，其余编号按正文原文识别。
    found = set(re.findall(r"[图表](?:[0-5]|[CDEFH])-\d+", text))
    for kind, letter, a, b in re.findall(r"([图表])([1-5CDEFH])-(\d+)\s*(?:至|—|~)\s*(?:[图表]?[1-5CDEFH]-)?(\d+)", text):
        for value in range(int(a), int(b) + 1):
            found.add(f"{kind}{letter}-{value}")
    return found


def main():
    assert all(p.exists() for p in (DOC, FIGURES, MAPPING, PAGES))
    document = Document(DOC)
    assets = {row["新图号"]: row for row in read_csv(FIGURES)}
    old_mapping = read_csv(MAPPING)
    pages = {row["图号"]: row["PDF页"] for row in read_csv(PAGES)}
    captions, refs = {}, defaultdict(list)
    section = "前置部分"
    paragraphs = document.paragraphs
    for index, p in enumerate(paragraphs):
        text, style = p.text.strip(), p.style.name
        section = current_section(text, style, section)
        if style in ("图题", "表题"):
            number, title = split_caption(text)
            if number:
                captions[number] = {"序号": index, "图名": title, "小节": section}
        if index >= 259 and style not in ("图题", "表题", "toc 1", "toc 2", "toc 3") and not style.startswith("Heading"):
            for number in references(text):
                refs[number].append((index, section, text[:100]))

    expected_figs = (["图1-1"] + [f"图2-{n}" for n in range(1, 5)]
                     + [f"图3-{n}" for n in range(1, 17)]
                     + [f"图4-{n}" for n in range(1, 6)]
                     + [f"图5-{n}" for n in range(1, 4)]
                     + [f"图C-{n}" for n in range(1, 21)]
                     + [f"图D-{n}" for n in range(1, 13)]
                     + [f"图E-{n}" for n in range(1, 21)]
                     + [f"图F-{n}" for n in range(1, 11)]
                     + [f"图H-{n}" for n in range(1, 4)])
    assert len(expected_figs) == 94 and len(set(expected_figs)) == 94
    figure_rows, position_rows = [], []
    for number in expected_figs:
        asset = assets.get(number, {})
        caption = captions.get(number, {})
        actual_title = caption.get("图名", "")
        target_title = MAIN_TITLES.get(number, actual_title)
        target_section = FIG_SECTIONS.get(number, f"附录{number[1]}")
        image_path = asset.get("最终图片路径", "")
        first = refs.get(number, [None])[0]
        figure_rows.append({
            "图号": number, "锁定图名": target_title, "当前V5图名": actual_title,
            "所属部分": "正文" if number[1].isdigit() else f"附录{number[1]}",
            "锁定插入小节": target_section, "当前V5小节": caption.get("小节", ""),
            "当前图件路径": image_path, "数据来源": asset.get("数据来源", ""),
            "绘图脚本": asset.get("绘图脚本", ""), "原图号": asset.get("原图号", ""),
            "PDF实际页": pages.get(number, ""),
            "协议与用途": protocol(number),
            "图名一致": "是" if target_title == actual_title else "否",
            "位置一致": "是" if section_matches(target_section, caption.get("小节", "")) else "否",
            "资产存在": "是" if image_path and Path(image_path).exists() else "否",
        })
        if number[1].isdigit():
            position_rows.append(position_row(number, "图", target_title, target_section, caption, first, pages.get(number, "")))

    table_rows = []
    for number, (title, target_section, fields) in TABLES.items():
        caption = captions.get(number, {})
        first = refs.get(number, [None])[0]
        row = {
            "表号": number, "锁定表名": title, "当前V5表名": caption.get("图名", ""),
            "锁定插入小节": target_section, "当前V5小节": caption.get("小节", ""),
            "锁定字段与内容": fields, "当前表格列标题": table_headers(document, caption.get("序号")),
            "图名一致": "是" if title == caption.get("图名", "") else "否",
            "位置一致": "是" if section_matches(target_section, caption.get("小节", "")) else "否",
            "首次正文引用": first[2] if first else "NA：未找到",
        }
        table_rows.append(row)
        position_rows.append(position_row(number, "表", title, target_section, caption, first, ""))

    write_csv("全文图件总表.csv", figure_rows, list(figure_rows[0]))
    write_csv("全文表格总表.csv", table_rows, list(table_rows[0]))
    write_csv("正文图表插入位置清单.csv", position_rows, list(position_rows[0]))
    migration_rows = []
    for row in old_mapping:
        target = row["新图号"]
        target_ids = re.findall(r"图(?:[1-5]|[CDEFH])-\d+", target)
        migration_rows.append({**row, "锁定新图名": MAIN_TITLES.get(target, captions.get(target, {}).get("图名", "")),
                               "当前V5状态": "存在" if target_ids and all(t in captions for t in target_ids) else "不在V5（含删除项）"})
    write_csv("图件编号与迁移对照.csv", migration_rows, list(migration_rows[0]))

    report = make_report(expected_figs, figure_rows, table_rows, position_rows, migration_rows, captions, refs)
    (OUT / "图表引用核查报告.md").write_text(report, encoding="utf-8")
    print(f"图件 {len(figure_rows)}、编号表格 {len(table_rows)}、迁移记录 {len(migration_rows)}、正文插入项 {len(position_rows)}")
    print(f"图名差异 {sum(r['图名一致']=='否' for r in figure_rows)}；位置差异 {sum(r['位置一致']=='否' for r in figure_rows if r['所属部分']=='正文')}")


def protocol(number):
    if number.startswith("图3-"):
        n = int(number.split("-")[1])
        return ("历史四任务五折固定批查询SHAP" if n <= 2 else
                "正式十任务第1折P1单查询、30独立核心样本SHAP" if n <= 7 else
                "历史五折ALE；空分箱需限制解释" if n <= 11 else
                "正式十任务第1折P1一维ALE" if n <= 15 else
                "20条P1 ALE曲线分箱样本与站点支持，非响应强度")
    if number.startswith("图4-"):
        return {"图4-1": "正式十任务总体预测和站点改善", "图4-2": "正式十任务指标与水层差异",
                "图4-3": "随机、年份和站点留出评价对象示意", "图4-4": "四任务现代表格模型扩展比较；TN底层缺项NA",
                "图4-5": "四任务站点配对ΔRMSE空间分布"}[number]
    if number.startswith("图5-"):
        return {"图5-1": "77特征XGBoost六来源正式消融", "图5-2": "历史78特征、12任务、60任务折分区扩展实验",
                "图5-3": "四站五案例输入覆盖和预测误差诊断"}[number]
    if number.startswith("图C-"):
        return "TabICLv2 P1局部SHAP；3独立seed42核心样本，seed43为重复探针"
    if number.startswith("图D-"):
        return "XGBoost基准解释对照；非TabICLv2解释"
    if number.startswith("图E-"):
        return "正式十任务P1一维ALE；真实bin与支持度，不插值"
    if number.startswith("图F-"):
        return "正式十任务3个独立核心样本平均绝对SHAP Top10"
    if number.startswith("图H-"):
        return "历史/局部技术审查，不支撑正式十任务总体结论"
    return "研究技术路线或冻结观察数据展示"


def section_matches(target, actual):
    return bool(actual) and (target == actual or target == "前置部分" and actual == "前置部分"
                            or target.count(".") == 1 and actual.startswith(target + "."))


def position_row(number, kind, title, target_section, caption, first, page):
    ordinal = caption.get("序号")
    return {
        "类型": kind, "编号": number, "锁定图表名": title, "锁定小节": target_section,
        "当前V5小节": caption.get("小节", "NA"), "图题段落序号": ordinal if ordinal is not None else "NA",
        "首次正文引用段落序号": first[0] if first else "NA",
        "首次正文引用小节": first[1] if first else "NA",
        "首次正文引用原文摘录": first[2] if first else "NA：未找到",
        "PDF实际页": page, "首次引用在图表前": "是" if first and ordinal is not None and first[0] < ordinal else "否/待核",
        "位置符合锁定": "是" if section_matches(target_section, caption.get("小节", "")) else "否",
    }


def table_headers(document, caption_index):
    if caption_index is None:
        return "NA：未找到编号表题"
    p = document.paragraphs[caption_index]._element
    sibling = p.getnext()
    while sibling is not None and sibling.tag.rsplit("}", 1)[-1] != "tbl":
        if sibling.tag.rsplit("}", 1)[-1] == "p":
            break
        sibling = sibling.getnext()
    if sibling is None or sibling.tag.rsplit("}", 1)[-1] != "tbl":
        return "NA：表题后无紧邻表格"
    from docx.table import Table
    table = Table(sibling, document)
    return "｜".join(cell.text.replace("\n", "/") for cell in table.rows[0].cells)


def make_report(expected_figs, figures, tables, positions, mapping, captions, refs):
    title_diff = [r for r in figures if r["图名一致"] == "否"]
    place_diff = [r for r in figures if r["所属部分"] == "正文" and r["位置一致"] == "否"]
    table_place_diff = [r for r in tables if r["位置一致"] == "否"]
    no_ref = [r for r in positions if r["首次正文引用段落序号"] == "NA"]
    late_ref = [r for r in positions if r["首次正文引用段落序号"] != "NA" and r["首次引用在图表前"] != "是"]
    missing_figs = [n for n in expected_figs if n not in captions]
    excess_figs = sorted(n for n in captions if n.startswith("图") and n not in expected_figs)
    deleted_refs = sorted(n for n in refs if re.fullmatch(r"图D-(1[3-7])", n))
    lines = ["# 图表引用核查报告", "", "## 核查依据与边界", "",
             "以 V5 正文 DOCX、逐图实际页码表、最终图件清单和原图迁移表为当前状态；锁定图名、位置及数量来自本次任务书。只读核验，未改动正文、图件、数据或模型。",
             "", f"- 锁定图件 {len(expected_figs)} 张：正文29张、附录65张；当前V5编号图题 {sum(n in captions for n in expected_figs)} 张。",
             f"- 锁定编号表格 {len(TABLES)} 张；当前编号表题 {sum(n in captions for n in TABLES)} 张；附录G另有未编号的77特征清单表，不计入12张。",
             f"- 当前正文缺图：{', '.join(missing_figs) or '无'}；额外编号图：{', '.join(excess_figs) or '无'}。",
             f"- 图名不一致 {len(title_diff)} 张；正文图件位置不一致 {len(place_diff)} 张；表格位置不一致 {len(table_place_diff)} 张。",
             f"- 正文图表未找到前置引用 {len(no_ref)} 项，首次引用晚于图表 {len(late_ref)} 项。",
             "", "## 锁定条件内部顺序冲突", "",
             "1. 图4-4固定于4.5，而图4-5固定于4.4；按章节顺序，图4-5必先于图4-4。",
             "2. 图5-1固定于5.3.2，而图5-2固定于5.2；按章节顺序，图5-2必先于图5-1。",
             "因此，当前图号与指定位置可并存，但不能同时满足正文首次引用严格按编号递增。本清单保持本次锁定图号和位置，未擅自改号。",
             "", "## 图表与章节矩阵", "",
             "| 部分 | 固定图件 | 固定编号表格 | 当前V5图题数 | 当前V5表题数 |",
             "|---|---|---|---:|---:|",
             "| 前置部分 | 无 | 表0-1 | 0 | 1 |",
             "| 第一章 | 图1-1 | 无 | 1 | 0 |",
             "| 第二章 | 图2-1至图2-4 | 表2-1至表2-6 | 4 | 6 |",
             "| 第三章 | 图3-1至图3-16 | 表3-1 | 16 | 1 |",
             "| 第四章 | 图4-1至图4-5 | 表4-1至表4-3 | 5 | 3 |",
             "| 第五章 | 图5-1至图5-3 | 表5-1 | 3 | 1 |",
             "| 附录C/D/E/F/H | 20/12/20/10/3张 | 无 | 65 | 0 |",
             "| 附录G | 无 | 77项清单（未编号） | 0 | 0 |",
             "", "## 当前V5位置差异", ""]
    lines += [f"- {r['图号']}：锁定{r['锁定插入小节']}；当前{r['当前V5小节']}。" for r in place_diff] or ["- 无。"]
    lines += [f"- {r['表号']}：锁定{r['锁定插入小节']}；当前{r['当前V5小节']}。" for r in table_place_diff]
    lines += ["", "## 图名差异（锁定名 ← 当前V5）", ""]
    lines += [f"- {r['图号']}：{r['锁定图名']} ← {r['当前V5图名']}。" for r in title_diff] or ["- 无。"]
    lines += ["", "## 引用缺失与顺序", ""]
    lines += [f"- {r['编号']}：当前小节{r['当前V5小节']}，锁定小节{r['锁定小节']}。" for r in no_ref] or ["- 所有正文编号图表均找到前置正文引用。"]
    lines += [f"- {r['编号']}：首次引用段落{r['首次正文引用段落序号']}，图表题段落{r['图题段落序号']}。" for r in late_ref]
    lines += [f"- 已删除的原D-11至D-15及旧D-16/17引用：{', '.join(deleted_refs) or '未发现'}。",
              "", "## 表格字段核查重点", "",
              "以下只比对当前表头和锁定字段，不补数值，不把历史结果填入NA。"]
    for r in tables:
        lines.append(f"- {r['表号']}：锁定字段：{r['锁定字段与内容']}；当前首行：{r['当前表格列标题']}。")
    lines += ["", "当前表2-3、表3-1、表4-2的首行字段不足以直接满足锁定字段。表4-3当前列示TN表层四模型，而锁定的图4-4使用TN底层且TabPFN/TabM为NA；两者任务范围不同，后续需明确表格取舍与图文对应，不得把TN表层数值移填TN底层。相关单元格及数据证据应在正文修订任务中另行核验，本任务不填造数值。",
              "", "## 冗余与证据边界", "",
              "- 图4-1呈现逐任务总体精度与站点覆盖，图4-2呈现指标和水层结构，属于互补。",
              "- 图3-1/3-2与图3-3至3-7分别对应历史固定批查询和正式P1单查询，不应合并为同一统计口径。",
              "- 附录C/F与正文SHAP图存在主题重叠，但承担样本级案例、各任务独立Top10及正文汇总的不同作用；保留分层定位。",
              "- 图5-2历史78特征分区与图5-1正式77特征XGBoost消融协议不同；图H-3仅保留历史比较边界。",
              "- 图D-1至D-12属于XGBoost解释对照，不能作为TabICLv2内部解释。",
              "", "## 迁移与图目录", "",
              f"原图迁移登记 {len(mapping)} 条，含原D-11至D-15退出论文、原D-16/17迁为D-11/12及新增图件；逐行见迁移对照CSV。",
              "当前V5图目录和图题均沿用当前V5图名；锁定图名与当前不一致的项目见上表，后续正文整合需同步改图题和图目录，不得只改一处。",
              "", "## 本任务处理决定", "",
              "五份清单已固定目标体系并标记实际差异。图号、图名、章节位置及数据边界以本次任务书为锁定目标；上述两处首次引用顺序冲突保留为待定，不在本只读任务中改动V5正文。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
