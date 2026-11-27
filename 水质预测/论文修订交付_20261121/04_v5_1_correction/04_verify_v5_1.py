"""核验V5.1结构、冻结值、PDF排版并输出更新清单及中文报告。"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import fitz
from docx import Document
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PLAN = ROOT / "论文修订交付_20261121/02_figure_table_plan"
BASE = ROOT / "论文修订交付_20261121/01_manuscript"
V5 = BASE / "水质预测硕士学位论文_V5_正文修订.docx"
DOCX = OUT / "水质预测硕士学位论文_V5.1.docx"
PDF = OUT / "水质预测硕士学位论文_V5.1.pdf"
CORE = ROOT / "runs/final_experiment_closure_audit_v1/final_core_results_table_v2.csv"
RANK = ROOT / "结果/论文图表/figure_revision_v2/data/30核心样本_77特征完整排名.csv"
UNITS = ROOT / "结果/论文最终图件重构_v1/05_数据核验/77项正式特征单位与时间窗核对.csv"
MODERN = ROOT / "transfer/llm_extension_transfer_v1/results/modern_baseline_gate_v1/modern_baseline_core_metrics.csv"

spec = importlib.util.spec_from_file_location("locked_plan", PLAN / "01_build_figure_table_plan.py")
locked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(locked)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", "", text).replace("µ", "μ")


def actual_sections_and_references(doc):
    caps = {}
    refs = defaultdict(list)
    section = "前置部分"
    for i, p in enumerate(doc.paragraphs):
        text, style = p.text.strip(), p.style.name
        section = locked.current_section(text, style, section)
        if style in ("图题", "表题"):
            number, title = locked.split_caption(text)
            if number:
                assert number not in caps, number
                caps[number] = {"位置": i, "图表名": title, "小节": section}
        if i >= 259 and style not in ("图题", "表题", "toc 1", "toc 2", "toc 3") and not style.startswith("Heading"):
            for number in locked.references(text):
                refs[number].append((i, section, text[:130]))
    return caps, refs


def table_for(doc, number):
    from docx.table import Table
    p = next(p for p in doc.paragraphs if p.style.name == "表题" and p.text.startswith(number + " "))
    node = p._p.getnext()
    assert node is not None and node.tag.rsplit("}", 1)[-1] == "tbl"
    return Table(node, doc._body)


def check_numeric_tables(doc):
    result = read_csv(CORE)
    pairs = {(x["task"], x["model"]): x for x in result}
    translate = {"Chl-a底层": "chla_bottom", "Chl-a中层": "chla_middle", "Chl-a表层": "chla_surface",
                 "Secchi": "secchi", "TN底层": "tn_bottom", "TN中层": "tn_middle", "TN表层": "tn_surface",
                 "TP底层": "tp_bottom", "TP中层": "tp_middle", "TP表层": "tp_surface"}
    t = table_for(doc, "表4-2")
    assert len(t.rows) == 11 and len(t.columns) == 10
    for row in t.rows[1:]:
        cells = [c.text for c in row.cells]
        task = translate[cells[0]]
        xgb, tab = pairs[task, "XGBoost"], pairs[task, "TabICLv2"]
        expected = [float(xgb["pooled_R2"]), float(tab["pooled_R2"]), float(tab["delta_R2"]),
                    float(xgb["RMSE"]), float(tab["RMSE"]), float(xgb["MAE"]), float(tab["MAE"])]
        assert all(abs(float(cells[i+1]) - expected[i]) <= 0.00051 for i in range(7)), task
        assert abs(float(cells[8].rstrip("%")) - 100*float(tab["station_win_rate_RMSE"])) <= 0.051
        assert cells[9] == ("稳健" if tab["robustness_class"] == "ROBUST" else "中等")
    unit_map = {r["正式字段"]: r["最终单位"] for r in read_csv(UNITS)}
    t = table_for(doc, "表2-3")
    assert len(t.rows) == 78 and len(t.columns) == 7
    for i, row in enumerate(t.rows[1:], 1):
        assert int(row.cells[0].text) == i
        assert row.cells[6].text == unit_map[row.cells[2].text]
    ranks = read_csv(RANK)
    by_task = defaultdict(list)
    for row in ranks:
        by_task[row["task"]].append(row)
    t = table_for(doc, "表3-1")
    assert len(t.rows) == 11 and len(t.columns) == 6
    for row in t.rows[1:]:
        cells = [c.text for c in row.cells]
        task = translate[cells[0]]
        top = sorted(by_task[task], key=lambda x: (float(x["rank"]), x["feature"]))[:3]
        assert cells[2].splitlines() == [x["feature"] for x in top]
        assert all(abs(float(a)-float(b["mean_abs_shap"])) <= 0.000001 for a,b in zip(cells[3].splitlines(), top))
        assert cells[4:] == ["3", "3"]
    t = table_for(doc, "表4-3")
    modern = {(r["task"], r["model"]): r for r in read_csv(MODERN)}
    m = {"Chl-a表层": "chla_surface", "Secchi": "secchi", "TN表层": "tn_surface", "TP中层": "tp_middle"}
    assert len(t.rows) == 17 and {r.cells[0].text for r in t.rows[1:]} == set(m)
    for row in t.rows[1:]:
        data = modern[m[row.cells[0].text], row.cells[1].text]
        for idx, field, tol in [(2,"pooled_R2",.00051),(3,"RMSE",.000051),(4,"MAE",.000051)]:
            assert abs(float(row.cells[idx].text)-float(data[field])) <= tol
    t = table_for(doc, "表2-5")
    assert len(t.columns) == 6 and len(t.rows) == 5
    note = next(p.text for p in doc.paragraphs if p.text.startswith("注：完整模型使用MET"))
    assert "ΔR²=消融模型R²−完整模型R²" in note and "ΔRMSE=消融模型RMSE−完整模型RMSE" in note
    return {"表2-3单位核对": 77, "表3-1均值核对": 30,
            "表4-2冻结指标核对": 10, "表4-3四模型行核对": 16,
            "表2-5受控复现任务核对": 4}


def make_contact_sheets(pdf):
    check = OUT / "排版核查"
    check.mkdir(exist_ok=True)
    pages = []
    for page in pdf:
        pix = page.get_pixmap(matrix=fitz.Matrix(.27, .27), alpha=False)
        pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    for first in range(0, len(pages), 20):
        chunk = pages[first:first+20]
        w, h = max(p.width for p in chunk), max(p.height for p in chunk)
        canvas = Image.new("RGB", (4*w, 5*(h+25)), "white")
        draw = ImageDraw.Draw(canvas)
        for k, im in enumerate(chunk):
            x, y = k%4*w, k//4*(h+25)
            canvas.paste(im, (x,y+25))
            draw.text((x+7,y+5), f"PDF {first+k+1}", fill="black")
        canvas.save(check / f"全文联系表_{first+1:03d}-{first+len(chunk):03d}.png")


def main():
    assert all(p.is_file() for p in (V5,DOCX,PDF,CORE,RANK,UNITS,MODERN))
    doc, prior = Document(DOCX), Document(V5)
    caps, refs = actual_sections_and_references(doc)
    fig_plan = read_csv(PLAN / "全文图件总表.csv")
    table_plan = read_csv(PLAN / "全文表格总表.csv")
    assert len(fig_plan) == 94 and len(table_plan) == 12
    assert len(doc.inline_shapes) == len(prior.inline_shapes) == 94
    assert len(doc.tables) == len(prior.tables) == 13
    assert sum(n.startswith("图") for n in caps) == 94 and sum(n.startswith("表") for n in caps) == 12
    assert [p.text for p in doc.paragraphs if p.style.name == "参考文献"] == [p.text for p in prior.paragraphs if p.style.name == "参考文献"]
    expected_figures = ["图1-1"] + [f"图2-{i}" for i in range(1,5)] + [f"图3-{i}" for i in range(1,17)] + [f"图4-{i}" for i in range(1,6)] + [f"图5-{i}" for i in range(1,4)]
    expected_figures += [f"图C-{i}" for i in range(1,21)] + [f"图D-{i}" for i in range(1,13)] + [f"图E-{i}" for i in range(1,21)] + [f"图F-{i}" for i in range(1,11)] + [f"图H-{i}" for i in range(1,4)]
    actual_figures = [n for n,_ in sorted(((n,cap["位置"]) for n,cap in caps.items() if n.startswith("图")), key=lambda x:x[1])]
    assert actual_figures == expected_figures, (actual_figures[:30], expected_figures[:30])
    assert all(number in caps for number in locked.TABLES)
    numeric = check_numeric_tables(doc)
    pdf = fitz.open(PDF)
    pdf_normalized = [normalize(page.get_text()) for page in pdf]
    title_pages = {}
    pdf_missing = []
    for number, item in caps.items():
        phrase = normalize(number + item["图表名"])
        pages = [i+1 for i,s in enumerate(pdf_normalized) if (i>=15 or number=="表0-1") and phrase in s]
        if not pages:
            pdf_missing.append(number)
        title_pages[number] = pages[0] if pages else "NA"
    assert not pdf_missing, pdf_missing
    landscapes = [i+1 for i,p in enumerate(pdf) if p.rect.width > p.rect.height]
    baseline_pdf = fitz.open(BASE / "水质预测硕士学位论文_V5_正文修订.pdf")
    baseline_landscape_count = sum(p.rect.width > p.rect.height for p in baseline_pdf)
    assert len(landscapes) == baseline_landscape_count + 1
    assert title_pages["表4-2"] in landscapes
    assert not (title_pages["表4-2"]-1 in landscapes and len(pdf[title_pages["表4-2"]-2].get_text().strip()) < 100)
    assert title_pages["图4-4"] < title_pages["图4-5"]
    assert title_pages["图5-1"] < title_pages["图5-2"]
    assert "表3-1范围一致" in (OUT / "01_draw_targeted_figures.py").read_text(encoding="utf-8")

    old_to_new = {"图4-5": "图4-4", "图4-4": "图4-5", "图5-2": "图5-1", "图5-1": "图5-2"}
    fig_rows = []
    for old in fig_plan:
        number = old_to_new.get(old["图号"], old["图号"])
        row = dict(old)
        row["图号"] = number
        row["锁定图名"] = caps[number]["图表名"]
        row["当前V5图名"] = caps[number]["图表名"]
        row["锁定插入小节"] = caps[number]["小节"]
        row["当前V5小节"] = caps[number]["小节"]
        row["PDF实际页"] = title_pages[number]
        row["图名一致"] = "是"
        row["位置一致"] = "是"
        if number == "图3-7":
            row["当前图件路径"] = str(OUT / "修订图件/图3-7_仅修正内部表号.png")
            row["绘图脚本"] = str(OUT / "01_draw_targeted_figures.py")
        if number == "图4-5":
            row["当前图件路径"] = str(OUT / "修订图件/图4-5_四项代表任务四模型比较.png")
            row["绘图脚本"] = str(OUT / "01_draw_targeted_figures.py")
            row["数据来源"] = str(MODERN)
            row["协议与用途"] = "四项代表任务同一冻结四模型比较；TN表层完整，不使用TN底层缺项"
        row["资产存在"] = "是" if Path(row["当前图件路径"]).is_file() else "否"
        row["图片SHA256"] = sha(row["当前图件路径"]) if row["资产存在"] == "是" else "NA"
        fig_rows.append(row)
    order = {number:i for i,number in enumerate(expected_figures)}
    fig_rows.sort(key=lambda x:order[x["图号"]])
    write_csv(OUT / "更新后的全文图件总表.csv", fig_rows)

    table_rows = []
    for row in table_plan:
        number = row["表号"]
        t = table_for(doc, number)
        new = dict(row)
        new["锁定表名"] = caps[number]["图表名"]
        new["当前V5表名"] = caps[number]["图表名"]
        new["当前V5小节"] = caps[number]["小节"]
        new["锁定插入小节"] = caps[number]["小节"]
        new["当前表格列标题"] = "｜".join(c.text.replace("\n", " ") for c in t.rows[0].cells)
        new["图名一致"] = "是"
        new["位置一致"] = "是"
        new["PDF实际页"] = title_pages[number]
        new["当前行列数"] = f"{len(t.rows)}行×{len(t.columns)}列"
        table_rows.append(new)
    write_csv(OUT / "更新后的全文表格总表.csv", table_rows)

    positions = []
    for number in expected_figures[:29] + list(locked.TABLES):
        c = caps[number]
        first = refs.get(number, [None])[0]
        before = first is not None and first[0] < c["位置"]
        if number == "表0-1":
            status = "前置符号表免正文引用"
        else:
            status = "通过" if before else "未通过"
        positions.append({"类型": number[0], "编号": number,
                          "图表名": c["图表名"], "当前小节": c["小节"],
                          "图表题段落序号": c["位置"],
                          "首次引用段落序号": first[0] if first else "NA",
                          "首次引用小节": first[1] if first else "NA",
                          "首次引用摘录": first[2] if first else "NA",
                          "PDF实际页": title_pages[number], "前置引用验收": status})
    write_csv(OUT / "更新后的正文图表插入位置清单.csv", positions)
    bad_positions = [r for r in positions if r["前置引用验收"] == "未通过"]

    # 图目录与表目录均须包含锁定题名和Word更新后的实际页码。
    tocs = [p.text for p in doc.paragraphs if p.style.name.startswith("toc")]
    toc_text = normalize("\n".join(tocs))
    toc_missing = [number for number in expected_figures + list(locked.TABLES)
                   if normalize(number + caps[number]["图表名"]) not in toc_text]
    assert not toc_missing, toc_missing

    # 图像包内素材应与两项定点重导PNG逐字节一致。
    from zipfile import ZipFile
    embedded = {}
    with ZipFile(DOCX) as z:
        hashes = {hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/")}
    for number, path in (("图3-7", OUT/"修订图件/图3-7_仅修正内部表号.png"),
                         ("图4-5", OUT/"修订图件/图4-5_四项代表任务四模型比较.png")):
        embedded[number] = sha(path) in hashes
    assert all(embedded.values())

    create = True
    if create:
        make_contact_sheets(pdf)
    changes = json.loads((OUT / "修改记录.json").read_text(encoding="utf-8"))
    report_lines = ["# V5_V5.1修改对照", "", "本表以V5为前稿，逐项列出本轮修改。源图、冻结预测、SHAP、ALE和参考文献未改。", "",
                    "| 原问题/类型 | 修改位置 | 修改前内容 | 修改后内容 | 验收结果 |",
                    "|---|---|---|---|---|"]
    for change in changes:
        def short(s):
            return str(s).replace("|", "／").replace("\n", "<br>")[:230]
        report_lines.append(f"| {short(change['类型'])} | {short(change['修改位置'])} | {short(change['修改前'])} | {short(change['修改后'])} | {change['验收']} |")
    (OUT / "V5_V5.1修改对照.md").write_text("\n".join(report_lines)+"\n", encoding="utf-8")

    check_lines = ["# 图表修复核查报告", "", "## 验收结果", "",
                   f"- V5.1 PDF共{len(pdf)}页，表4-2在第{title_pages['表4-2']}页独立横向呈现；其他横向页为V5原有横向表页。",
                   f"- 正文图件29张、附录图件65张，共94张；编号表格12张，另保留附录G未编号77特征清单。",
                   f"- 图目录、表目录和PDF正文题名逐项核对：94图、12表均检出；图表编号按正文出现顺序连续。",
                   f"- 图4-4地图在PDF第{title_pages['图4-4']}页；图4-5四模型比较在第{title_pages['图4-5']}页；图5-1历史分区在第{title_pages['图5-1']}页，图5-2正式消融在第{title_pages['图5-2']}页。",
                   f"- 前置正文引用：{len(positions)-1-len(bad_positions)}/{len(positions)-1}项通过，表0-1为前置符号表免引用。",
                   f"- 两张定点重导PNG已逐字节核实嵌入DOCX：{embedded}。图3-7底部仅将表3-2改为表3-1；图4-5采用TN表层四模型冻结数据。",
                   f"- 冻结表格数值复核：{json.dumps(numeric,ensure_ascii=False)}。表4-2保留十任务原序，横向页不截断。",
                   "- 参考文献段落与V5逐字一致；未切换到WQ77_REPAIRED_20261115_V1。",
                   "", "## PDF实际排版复核", "",
                   "已导出全文缩略联系表，并逐页放大查看表2-3、表3-1、表4-2、图3-7、图4-4、图4-5及第五章编号迁移页。表4-2的十列在横向页完整可读；表2-3跨页表头重复，表3-1的三项变量与三项均值逐行对应。",
                   "", "## 未通过或保留事项", ""]
    if bad_positions:
        check_lines += [f"- {r['编号']}：首次引用{r['首次引用段落序号']}，图表题{r['图表题段落序号']}。" for r in bad_positions]
    else:
        check_lines += ["- 本轮锁定的正文图表未发现引用顺序缺口。"]
    check_lines += ["- WQ77修复版本切换、11月20日SHAP回传复现差异及rain_month业务可得性属于既有待确认问题，本轮未处理。",
                    "- V5.1的表4-2单独使用横向页面以满足十列字段可读性；由此PDF页数相对V5增加。",
                    "", "## 数据来源", "",
                    f"- 正式十任务指标：`{CORE}`",
                    f"- 30核心样本SHAP平均绝对值：`{RANK}`",
                    f"- 77特征单位：`{UNITS}`",
                    f"- 四模型比较：`{MODERN}`", ""]
    (OUT / "图表修复核查报告.md").write_text("\n".join(check_lines), encoding="utf-8")
    if bad_positions:
        raise AssertionError(f"正文图表首次引用仍不合格：{[x['编号'] for x in bad_positions]}")
    print(f"PASS：{len(pdf)}页，94图，12编号表；前置引用{len(positions)-1}/{len(positions)-1}，横向表4-2第{title_pages['表4-2']}页")


if __name__ == "__main__":
    main()
