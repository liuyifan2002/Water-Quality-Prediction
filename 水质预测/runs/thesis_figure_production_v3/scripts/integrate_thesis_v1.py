from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image


ROOT = Path(r"E:\研究生\codex\水质预测")
SRC = next((ROOT / "文档/论文材料/硕士论文TabICLv2解释闭环修订稿_20261019").glob("*.docx"))
FIGROOT = ROOT / "runs/thesis_figure_production_v3"
INV = pd.read_csv(FIGROOT / "figure_inventory_v3.csv").fillna("")
BASE_OUT = ROOT / "文档/论文材料/硕士论文TabICLv2_SHAP_ALE_插图整合初稿_20261031"
OUT = BASE_OUT
if OUT.exists() and any(OUT.iterdir()):
    n = 2
    while (BASE_OUT.parent / f"{BASE_OUT.name}_v{n}").exists(): n += 1
    OUT = BASE_OUT.parent / f"{BASE_OUT.name}_v{n}"
OUT.mkdir(parents=True, exist_ok=True)
DOCX = OUT / "水质预测硕士学位论文_TabICLv2_SHAP_ALE_插图整合初稿.docx"

TASKS = ["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom"]
TASK_CN = {"chla_surface": "Chl-a表层", "chla_middle": "Chl-a中层", "chla_bottom": "Chl-a底层", "secchi": "透明度", "tn_surface": "TN表层", "tn_middle": "TN中层", "tn_bottom": "TN底层", "tp_surface": "TP表层", "tp_middle": "TP中层", "tp_bottom": "TP底层"}
F_CN = {"t2m": "2 m气温", "t2m_lag1": "前1日2 m气温", "t2m_lag2": "前2日2 m气温", "t2m_roll3": "3日平均2 m气温", "t2m_delta1": "2 m气温日变化", "wind_speed": "风速", "wind_speed_lag1": "前1日风速", "wind_speed_sq": "风速平方", "ssrd_sum_3d": "3日太阳辐射总量", "v10": "10 m经向风", "u10": "10 m纬向风", "tcc": "总云量"}


def path_for(row) -> Path:
    return FIGROOT / (str(row.file_stem).replace("\\", "/") + ".png")


def para_after(doc, anchor, text="", style=None):
    p = doc.add_paragraph(style=style)
    if text: p.add_run(text)
    anchor._p.addnext(p._p)
    return p


def add_bookmark(p, name: str, ident: int):
    start = OxmlElement("w:bookmarkStart"); start.set(qn("w:id"), str(ident)); start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd"); end.set(qn("w:id"), str(ident))
    p._p.insert(0, start); p._p.append(end)


def set_keep(p, keep_next=False, keep_together=True):
    p.paragraph_format.keep_with_next = keep_next
    p.paragraph_format.keep_together = keep_together
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)


def insert_pic(doc, anchor, img_path, caption, meta, records, bookmark_id):
    with Image.open(img_path) as im:
        wpx, hpx = im.size
    sec = doc.sections[0]
    max_w = (sec.page_width - sec.left_margin - sec.right_margin) / 914400 - 0.16
    max_h = (sec.page_height - sec.top_margin - sec.bottom_margin) / 914400 - 1.25
    scale = min(max_w / wpx, max_h / hpx)
    width = min(wpx * scale, max_w)
    picp = para_after(doc, anchor)
    picp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picp.add_run().add_picture(str(img_path), width=Inches(width))
    picp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    picp.paragraph_format.line_spacing = 1.0
    set_keep(picp, keep_next=True)
    cap = para_after(doc, picp, caption, style="图题")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_keep(cap, keep_next=True)
    add_bookmark(cap, f"fig_{bookmark_id:03d}", bookmark_id)
    note = para_after(doc, cap, "图注：" + meta.get("图注", ""))
    note.paragraph_format.left_indent = Inches(0.18)
    note.paragraph_format.right_indent = Inches(0.18)
    note.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    note.paragraph_format.line_spacing = 1.0
    note.paragraph_format.keep_together = True
    note.paragraph_format.space_after = Pt(6)
    for r in note.runs: r.font.size = Pt(8)
    records.append({**meta, "原始图片路径": str(img_path), "图题": caption, "插入状态": "已插入", "图片像素": f"{wpx}x{hpx}", "插入宽度英寸": round(width, 3), "书签": f"fig_{bookmark_id:03d}"})
    return note


def text_nodes(doc):
    return list(doc._element.body.iter(qn("w:t")))


def add_pageref(p, bookmark):
    r = p.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = f" PAGEREF {bookmark} \\h "
    sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
    val = OxmlElement("w:t"); val.text = "待更新"
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    for x in (begin, instr, sep, val, end): r._r.append(x)


def main():
    doc = Document(SRC)
    original_nodes = text_nodes(doc)
    caption_node_set = {node for p in doc.paragraphs if p.style.name == "图题" for node in p._p.iter(qn("w:t"))}
    original_images = len(doc.inline_shapes)
    original_tables = len(doc.tables)
    original_sections = len(doc.sections)
    sec = doc.sections[0]
    records = []
    skipped = []
    mark_id = 1000

    def original_anchor(prefix):
        hits = [p for p in doc.paragraphs if p.text.startswith(prefix) and p.style.name.startswith("Heading")]
        if len(hits) != 1: raise RuntimeError(f"anchor not unique: {prefix}, hits={len(hits)}")
        return hits[0]

    def insert_before_heading(prefix, rows, chapter):
        nonlocal mark_id
        heading = original_anchor(prefix)
        anchor = heading._p.getprevious()
        from docx.text.paragraph import Paragraph
        prev = Paragraph(anchor, heading._parent)
        for row, caption_tail, note in rows:
            mark_id += 1
            img = path_for(row)
            if not img.exists():
                skipped.append((row.figure_id, str(img), "图片缺失")); continue
            prev = insert_pic(doc, prev, img, "图3-NEW " + caption_tail, {
                "图件ID": row.figure_id, "插入章节": chapter, "图号": "待统一编号", "正文/附录": "正文", "图注": note,
            }, records, mark_id)

    def select(fid):
        return INV[(INV.figure_id == fid) & (INV.status == "GENERATED")]

    # GLOBAL-01 is only a reference to existing MAIN-02; no duplicate insertion.
    skipped.append(("GLOBAL-01", "existing MAIN-02", "与已有SHAP主图重复；仅保留原图"))
    skipped.append(("GLOBAL-04", "figures/global/global_04_feature_source_group_share.png", "源组公式审查未通过：V3绘图脚本取组内均值后归一化，与预登记的逐特征求和公式不一致；本轮禁止重绘"))
    skipped.append(("DIST-02", "", "研究者指定暂不生成Violin"))

    global_rows = []
    for task in TASKS:
        z = select("GLOBAL-02"); row = z[z.file_stem.str.contains(f"global_02_{task}_")].iloc[0]
        global_rows.append((row, f"{TASK_CN[task]}前10项特征的平均绝对SHAP重要性", "按本任务逐样本已冻结SHAP值汇总；不跨任务比较原始贡献量纲。"))
    row = select("GLOBAL-03").iloc[0]
    global_rows.append((row, "10项任务主要特征的任务内SHAP排名变化", "纵轴仅为任务内排名，不代表贡献大小。"))
    insert_before_heading("3.6 基于一维ALE", global_rows, "3.5 SHAP模型解释")

    ale_rows = []
    for task in TASKS:
        rows = select("ALE-01")
        rows = rows[rows.file_stem.str.contains(f"ale_01_{task}_")].sort_values("file_stem")
        if len(rows) != 2: raise RuntimeError(f"ALE count not 2 for {task}: {len(rows)}")
        for _, row in rows.iterrows():
            feature = str(row.file_stem).replace("\\", "/").split(f"ale_01_{task}_", 1)[1]
            note = "基于冻结ALE结果绘制；零线为无效应参考，数字为各bin样本数。"
            if feature == "ssrd_sum_3d": note += " 本曲线保留6个原始bin，未插值补点。"
            if task == "chla_middle": note += " 本任务含数值敏感的小效应，需结合原始重复性审查谨慎解释。"
            ale_rows.append((row, f"{TASK_CN[task]}的{F_CN.get(feature, feature)}一维ALE响应", note))
    insert_before_heading("3.7 基于二维ALE", ale_rows, "3.6 ALE一维非线性响应")

    reliability_rows = []
    for fid, caption, note in [
        ("SUPPORT-01", "20条ALE曲线的分箱样本数量分布", "仅反映样本支持度，不是ALE模型响应图；不存在的bin留白，不作零值。"),
        ("SUPPORT-02", "20条ALE曲线的分箱独立站点数量分布", "仅反映站点支持度，不是ALE模型响应图；不存在的bin留白，不作零值。"),
        ("STABILITY-01", "9项任务P1重复探针预测范围", "仅9项任务有三次重复探针，缺少Chl-a表层；该范围不是预测置信区间。"),
        ("STABILITY-02", "历史数值抖动与解释效应尺度对比", "引用历史冻结数值稳定性报告；不将既有失败改写为通过。"),
    ]:
        reliability_rows.append((select(fid).iloc[0], caption, note))
    insert_before_heading("3.7 基于二维ALE", reliability_rows, "3.6 解释可靠性与支持度")

    two_rows = []
    for fid, task, supported in [("ALE2D-01", "tn_surface", "13/16"), ("ALE2D-02", "tn_bottom", "7/16"), ("ALE2D-03", "tp_bottom", "13/16")]:
        row = select(fid).iloc[0]
        two_rows.append((row, f"{TASK_CN[task]}的ALE二维候选联合支持审查矩阵", f"{supported}格通过；ALE_2D_BLOCKED_BY_SUPPORT。该V3候选未计算ALE二维响应曲面，不等同于原稿五折二维ALE分析。"))
    insert_before_heading("3.8 XGBoost", two_rows, "3.7 ALE二维适用性审查")

    # Add a short scope note without rewriting historical experimental conclusions.
    h38 = original_anchor("3.8 XGBoost")
    from docx.text.paragraph import Paragraph
    scope_note = doc.add_paragraph("补充说明：上述三组V3候选仅为联合支持审查，其状态为ALE_2D_BLOCKED_BY_SUPPORT，未计算对应ALE二维响应曲面；与前文已完成的五折二维ALE分析口径不同。")
    h38._p.addprevious(scope_note._p)

    # Append new atlas after existing A/B appendices, preserving their order and content.
    heading = doc.add_paragraph("附录C 模型解释补充图件", style="Heading 2")
    heading.paragraph_format.page_break_before = True
    heading.paragraph_format.keep_with_next = True
    appendix_count = 0
    for fid, label, key in [("LOCAL-01", "典型样本SHAP局部贡献", "waterfall"), ("DIST-01", "SHAP分布（3个独立样本及1次种子重复）", "beeswarm")]:
        rows = select(fid)
        for task in TASKS:
            z = rows[rows.file_stem.str.contains(f"{task}.*{key}")]
            if len(z) != 1: raise RuntimeError(f"missing appendix {fid} {task}")
            row = z.iloc[0]; img = path_for(row); mark_id += 1; appendix_count += 1
            note = "预登记CORE_3且target_quantile=0.5；77特征SHAP贡献加和已核验。" if fid == "LOCAL-01" else "本任务仅3个独立样本，另含同一样本种子重复；不代表总体SHAP分布。"
            insert_pic(doc, doc.paragraphs[-1], img, f"图C-{appendix_count} {TASK_CN[task]}{label}", {"图件ID": fid, "插入章节": "附录C", "图号": f"图C-{appendix_count}", "正文/附录": "附录", "图注": note}, records, mark_id)

    # Assign chapter-3 figure numbers in document order. Existing captions retain text except number.
    chapter = False; counter = 0; old_to_new = {}
    for p in doc.paragraphs:
        t = p.text
        if t.startswith("第3章 "): chapter = True
        if t.startswith("第4章 "): chapter = False
        if not chapter or p.style.name != "图题": continue
        if not (re.match(r"^图3-\d+\s", t) or t.startswith("图3-NEW ")): continue
        counter += 1
        match = re.match(r"^图3-(\d+)\s", t)
        if match: old_to_new[int(match.group(1))] = counter
        for run in p.runs:
            if run.text.startswith("图3-NEW "):
                run.text = run.text.replace("图3-NEW ", f"图3-{counter} ", 1)
            elif re.match(r"^图3-\d+\s", run.text):
                run.text = re.sub(r"^图3-\d+\s", f"图3-{counter} ", run.text, count=1)
            elif re.match(r"^3-\d+\s", run.text):
                run.text = re.sub(r"^3-\d+\s", f"3-{counter} ", run.text, count=1)
        for rec in records:
            if rec["书签"] in [x.get(qn("w:name")) for x in p._p.iter(qn("w:bookmarkStart"))]:
                rec["图号"] = f"图3-{counter}"; rec["图题"] = p.text

    # Update original plain-text references only, retaining original runs/field XML.
    for node in original_nodes:
        if node.text and node not in caption_node_set:
            node.text = re.sub(r"(?<!\d)3-(\d{1,2})(?!\d)", lambda m: f"3-{old_to_new.get(int(m.group(1)), int(m.group(1)))}", node.text)

    # Add a simple figure catalogue before the main symbol table; entries point to numbered captions.
    symbol = next(p for p in doc.paragraphs if p.text == "主要符号表" and p.style.name.startswith("Heading"))
    cat = doc.add_paragraph("图目录", style="Heading 1")
    symbol._p.addprevious(cat._p)
    all_captions = [p for p in doc.paragraphs if p.style.name == "图题" and (re.match(r"^图3-\d+\s", p.text) or re.match(r"^图C-\d+\s", p.text))]
    for cp in all_captions:
        names = [x.get(qn("w:name")) for x in cp._p.iter(qn("w:bookmarkStart"))]
        if names:
            bookmark = names[0]
        else:
            mark_id += 1; bookmark = f"fig_{mark_id:03d}"; add_bookmark(cp, bookmark, mark_id)
        p = doc.add_paragraph(cp.text + "\t")
        p.style = "Normal"; p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        for r in p.runs: r.font.size = Pt(9)
        add_pageref(p, bookmark)
        symbol._p.addprevious(p._p)
    # Existing chapter-3 captions are included as well.
    existing_caps = [p.text for p in doc.paragraphs if p.style.name == "图题" and re.match(r"^图3-\d+\s", p.text)]
    if len(existing_caps) != counter: raise RuntimeError(f"caption count mismatch {len(existing_caps)} != {counter}")

    doc.settings.update_fields_on_open = True
    doc.save(DOCX)
    test = Document(DOCX)
    if len(test.inline_shapes) != original_images + len(records): raise RuntimeError("image relationship count mismatch")
    if len(test.tables) != original_tables or len(test.sections) != original_sections: raise RuntimeError("table/section count mismatch")
    nums = [int(m.group(1)) for p in test.paragraphs if p.style.name == "图题" if (m := re.match(r"^图3-(\d+)\s", p.text))]
    if nums != list(range(1, len(nums)+1)): raise RuntimeError("chapter 3 figure numbering not continuous")
    with ZipFile(SRC) as z: old = z.namelist()
    with ZipFile(DOCX) as z: new = z.namelist()
    old_parts = {x for x in old if re.match(r"word/(header|footer)\d+\.xml$", x)}
    new_parts = {x for x in new if re.match(r"word/(header|footer)\d+\.xml$", x)}
    if old_parts != new_parts: raise RuntimeError("header/footer parts differ")

    columns = ["图件ID", "原始图片路径", "插入章节", "图号", "图题", "正文/附录", "插入状态", "图片像素", "插入宽度英寸", "书签", "图注"]
    with (OUT / "figure_insertion_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=columns); wr.writeheader(); wr.writerows([{k: rec.get(k, "") for k in columns} for rec in records])
    by_section = Counter(x["插入章节"] for x in records)
    (OUT / "figure_integration_report.md").write_text(
        "# 插图整合报告\n\n"
        f"- 源稿：`{SRC}`。选择依据：正文439段、表格9、原图{original_images}张，保有完整TabICLv2解释闭环及域代码；并非仅按修改日期选择。\n"
        f"- 新稿：`{DOCX}`。\n"
        f"- 新增插图：{len(records)}张；第三章{len(records)-appendix_count}张，附录C {appendix_count}张。\n"
        + "".join(f"- {k}：{v}张。\n" for k,v in by_section.items())
        + "- 未插入：\n" + "".join(f"  - {a}：{reason}（{path}）。\n" for a,path,reason in skipped)
        + "- 附录A/B原样保留；新增附录C以避免覆盖既有附录编号。\n"
        + "- V3二维支持审查与原稿五折二维ALE属于不同分析口径，已用简短说明区分；原有结论未改写。\n"
        + "- 图目录已新增；正文旧图号引用按章节图号映射更新。\n",
        encoding="utf-8",
    )
    (OUT / "thesis_format_quality_audit.md").write_text(
        "# 论文插图格式质量审计\n\n"
        f"- DOCX重新打开：通过；原图{original_images}张保留，新图{len(records)}张；表格{original_tables}张、分节{original_sections}节不变。\n"
        f"- 第三章图题：{counter}条，编号连续，无重复。\n"
        "- 图片以原始600 dpi PNG按版心等比缩放，居中；图题紧随图片并设置同页保持。\n"
        "- 页眉/页脚XML部件名保留；PDF分页及图目录页码需在导出后复核。\n"
        "- 科学限制：GLOBAL-04未插入，因已生成图件的来源组百分比公式与预登记公式不一致；不能作为通过质量验收的正式图件。\n"
        "- 科学限制：V3 Beeswarm只有3个独立样本及1次种子重复，图注不作总体分布推断。\n"
        "- 科学限制：V3 ALE图内SMALL_EFFECT标签来自生成脚本固定0.05阈值，未找到冻结阈值依据；新增图注不将该标签作为已验证统计判定，正式提交前应核对。\n",
        encoding="utf-8",
    )
    print(f"OUTPUT={DOCX}\nINSERTED={len(records)}\nCHAPTER3={len(records)-appendix_count}\nAPPENDIX={appendix_count}\nCAPTIONS={counter}")


if __name__ == "__main__": main()
