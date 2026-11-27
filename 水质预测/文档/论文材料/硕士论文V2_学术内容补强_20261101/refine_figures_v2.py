"""整理V2正文解释图：原始结果只读，旧图迁入附录，正文改为可读的紧凑图。"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt


ROOT = Path(r"E:\研究生\codex\水质预测")
OUT = ROOT / "文档/论文材料/硕士论文V2_学术内容补强_20261101"
INPUT = OUT / "水质预测硕士学位论文_学术内容补强V2.docx"
OUTPUT = OUT / "水质预测硕士学位论文_学术内容补强V2_正式整理稿_v3.docx"
SRC = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
ALE = SRC / "ale_1d"
MAP = pd.read_csv(ROOT / "runs/explanation_figures_v1/metadata/feature_display_name_mapping.csv").set_index("raw_feature_name")
TASKS = ["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom"]
TASKCN = ["叶绿素a表层", "叶绿素a中层", "叶绿素a底层", "透明度", "TN表层", "TN中层", "TN底层", "TP表层", "TP中层", "TP底层"]
TARGET_UNITS = {"chla": "µg/L", "secchi": "m", "tn": "mg/L", "tp": "mg/L"}
PALETTE = ["#3d6b83", "#7a929b", "#ad8068", "#758a71", "#8a79a0"]
plt.rcParams.update({"font.family": ["Arial", "SimSun"], "font.size": 8.7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})


def display(feature: str) -> str:
    return str(MAP.loc[feature, "display_name_en"]) if feature in MAP.index else feature


def unit(task: str) -> str:
    return TARGET_UNITS[task.split("_")[0]]


def find(doc, prefix):
    ps = [p for p in doc.paragraphs if p.text.startswith(prefix) and not p.style.name.startswith("toc") and not p._p.xpath(".//w:fldChar")]
    if re.match(r"^图[0-9A-Z]-\d+ ", prefix):
        ps = [p for p in ps if p.style.name == "图题"]
    if len(ps) != 1:
        raise ValueError((prefix, len(ps)))
    return ps[0]


def save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def shap_summary():
    records = []
    for task in TASKS:
        files = sorted((SRC / "shap" / task).glob("*_seed42.csv.gz"))
        assert len(files) == 3, (task, len(files))
        d = pd.concat([pd.read_csv(f, compression="gzip", usecols=["feature", "shap_value"]) for f in files], ignore_index=True)
        assert d.feature.nunique() == 77
        a = d.assign(abs_shap=d.shap_value.abs()).groupby("feature").abs_shap.mean().sort_values(ascending=False)
        for rank, (feature, value) in enumerate(a.items(), 1):
            records.append({"task": task, "feature": feature, "rank": rank, "mean_abs_shap": value,
                            "source_group": MAP.loc[feature, "source_group"], "n_core_samples": len(files)})
    out = pd.DataFrame(records)
    out.to_csv(OUT / "冻结SHAP汇总核对表.csv", index=False, encoding="utf-8-sig")
    return out


def shap_panels(stats):
    outputs = []
    for panel_id, indices in enumerate([range(0, 5), range(5, 10)], 1):
        fig, axes = plt.subplots(3, 2, figsize=(7.8, 8.1), constrained_layout=True)
        for ax, ti in zip(axes.flat, indices):
            task = TASKS[ti]
            rows = stats[stats.task.eq(task)].head(6).iloc[::-1]
            y = np.arange(len(rows))
            ax.hlines(y, 0, rows.mean_abs_shap, color=PALETTE[ti % len(PALETTE)], lw=1.1)
            ax.scatter(rows.mean_abs_shap, y, s=16, color=PALETTE[ti % len(PALETTE)])
            ax.set_yticks(y, [display(x) for x in rows.feature], fontsize=7)
            ax.set_xlabel(f"Mean |SHAP| ({unit(task)})", fontsize=7.5)
            ax.set_title(f"({chr(97 + ti)}) {TASKCN[ti]}", loc="left", fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", color="#e6ebed", lw=0.4)
        for ax in list(axes.flat)[len(indices):]: ax.axis("off")
        outputs.append(save(fig, OUT / "figures" / f"图3-{panel_id+2}_十任务SHAP多面板_{panel_id}"))
    return outputs


def ale_frame(task, feature):
    p = ALE / f"{task}_{feature}_curve.csv"
    d = pd.read_csv(p)
    assert len(d) in (6, 10) and d.task.eq(task).all() and d.feature.eq(feature).all()
    return d, p


def draw_curve(ax, d, task, feature, color, compact=False):
    x = (d.lower + d.upper) / 2
    ax.plot(x, d.ale, marker="o", lw=1.25, ms=3.2, color=color)
    ax.axhline(0, color="#59656b", lw=0.65, ls="--")
    ax.set_xlabel(f"{display(feature)} ({MAP.loc[feature, 'unit']})", fontsize=7.3 if compact else 8.5)
    ax.set_ylabel(f"Centered ALE ({unit(task)})", fontsize=7.3 if compact else 8.5)
    ax.tick_params(labelsize=7 if compact else 8)
    ax.grid(axis="y", color="#e6ebed", lw=0.4)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    if not compact:
        for xi, yi, n in zip(x, d.ale, d.n_samples):
            ax.annotate(str(int(n)), (xi, yi), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.4, color="#5c6970")


def ale_main_panels():
    specs = [("chla_surface", ["wind_speed", "wind_speed_lag1"]),
             ("secchi", ["ssrd_sum_3d", "t2m_roll3"]),
             ("tn_surface", ["t2m_delta1", "wind_speed_lag1"]),
             ("tp_middle", ["t2m_lag1", "tcc"])]
    outputs = []
    for i, (task, features) in enumerate(specs, 10):
        fig, axs = plt.subplots(1, 2, figsize=(8.1, 3.4), constrained_layout=True)
        for j, (ax, feature) in enumerate(zip(axs, features)):
            d, _ = ale_frame(task, feature)
            draw_curve(ax, d, task, feature, PALETTE[i % len(PALETTE)], compact=True)
            ax.set_title(f"({chr(97+j)}) {display(feature)}", loc="left", fontsize=8.4)
        outputs.append(save(fig, OUT / "figures" / f"图3-{i}_{task}_ALE代表任务"))
    return outputs


def ale_appendix_plots():
    outputs = []
    # 顺序与V1图3-18至图3-37一一相同。
    specs = [
        ("chla_surface", "wind_speed"), ("chla_surface", "wind_speed_lag1"),
        ("chla_middle", "wind_speed"), ("chla_middle", "wind_speed_sq"),
        ("chla_bottom", "t2m"), ("chla_bottom", "t2m_lag1"),
        ("secchi", "ssrd_sum_3d"), ("secchi", "t2m_roll3"),
        ("tn_surface", "t2m_delta1"), ("tn_surface", "wind_speed_lag1"),
        ("tn_middle", "t2m_delta1"), ("tn_middle", "v10"),
        ("tn_bottom", "t2m_lag2"), ("tn_bottom", "t2m_roll3"),
        ("tp_surface", "t2m_lag2"), ("tp_surface", "u10"),
        ("tp_middle", "t2m_lag1"), ("tp_middle", "tcc"),
        ("tp_bottom", "t2m_lag1"), ("tp_bottom", "wind_speed_lag1"),
    ]
    for i, (task, feature) in enumerate(specs, 1):
        d, source = ale_frame(task, feature)
        fig, ax = plt.subplots(figsize=(6.1, 3.9), constrained_layout=True)
        draw_curve(ax, d, task, feature, PALETTE[TASKS.index(task) % len(PALETTE)])
        path = save(fig, OUT / "figures" / f"图E-{i}_{task}_{feature}")
        outputs.append((path, source, len(d), int(d.n_samples.min()), int(d.n_stations.min())))
    return outputs


def replace_image(p, path, width_cm=15.4):
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width_cm))


def insert_figure_before(anchor, path, caption, width_cm=15.4):
    p = anchor.insert_paragraph_before()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    cap = anchor.insert_paragraph_before(caption, style="图题")
    cap.paragraph_format.keep_with_next = False
    p.paragraph_format.keep_with_next = True
    return cap


def transfer_block(doc, old_number, new_number, dest_heading, image_path=None):
    caps = [p for p in doc.paragraphs if p.style.name == "图题" and p.text.startswith(f"图3-{old_number} ")]
    if len(caps) != 1: raise ValueError((old_number, len(caps)))
    cap = caps[0]
    ps = doc.paragraphs
    idx = next(i for i, p in enumerate(ps) if p._p is cap._p)
    img = ps[idx - 1]
    assert img._p.xpath(".//w:drawing"), old_number
    note = ps[idx + 1] if idx + 1 < len(ps) and ps[idx + 1].text.startswith("图注：") else None
    if image_path is not None:
        replace_image(img, image_path)
    cap.text = cap.text.replace(f"图3-{old_number}", f"图{dest_heading}-{new_number}", 1)
    if note is not None:
        note.text = note.text.replace("零线为无效应参考", "虚线为中心化零参考线")
        note.text = re.sub(r"图内程序状态文字仅为历史原图标签。", "", note.text)
    body = doc._element.body
    for p in [img, cap] + ([note] if note is not None else []):
        body.append(p._p)
    return cap


def build():
    assert INPUT.is_file() and not OUTPUT.exists()
    doc = Document(INPUT)
    stats = shap_summary()
    shap_paths = shap_panels(stats)
    ale_paths = ale_main_panels()
    appendix_plots = ale_appendix_plots()

    # 迁移原有逐任务图至附录，正文保留来源不变的紧凑表达。
    doc.add_heading("附录E 十项任务一维ALE完整曲线", level=2)
    for n, item in enumerate(appendix_plots, 1):
        transfer_block(doc, 17+n, n, "E", item[0])
    doc.add_heading("附录F 十项任务SHAP独立特征重要性图", level=2)
    for n in range(1, 11):
        transfer_block(doc, n+2, n, "F")

    # 老图号到新图号：用单次正则替换，避免级联替换。
    def convert(m):
        n = int(m.group(1))
        if 3 <= n <= 12: return f"图F-{n-2}"
        if n == 13: return "图3-5"
        if 14 <= n <= 17: return f"图3-{n-8}"
        if 18 <= n <= 37: return f"图E-{n-17}"
        if 38 <= n <= 44: return f"图3-{n-24}"
        return m.group(0)
    for p in doc.paragraphs:
        if p.style.name.startswith("toc"): continue
        if "图3-" in p.text and not p._p.xpath(".//w:drawing"):
            p.text = re.sub(r"图3-(\d+)", convert, p.text)

    # 原3.5.4说明按正文与附录范围重写；十任务的冻结贡献聚合表使用原始3个CORE样本。
    p = find(doc, "图F-1至图3-5对应")
    p.text = "图3-3与图3-4压缩呈现十项任务第1折P1单查询SHAP贡献，各任务独立排序；图3-5只展示任务内排名变化，不能当作跨任务贡献大小。逐任务完整Top10图见附录F。两组解释产物与历史四任务固定批查询的五折结果分开报告。"
    anchor = find(doc, "图3-5 TabICLv2十项任务")
    anchor_idx = next(i for i, x in enumerate(doc.paragraphs) if x._p is anchor._p)
    anchor_image = doc.paragraphs[anchor_idx - 1]
    for i, path in enumerate(shap_paths, 3):
        insert_figure_before(anchor_image, path, f"图3-{i} TabICLv2十项任务第1折SHAP特征贡献（第{i-2}组）", 15.7)
    # 插入后原图3-5的图像仍与图题相邻。
    t = doc.add_table(rows=1, cols=4)
    t.style = "Table Grid"
    for cell, text in zip(t.rows[0].cells, ["任务", "主要贡献变量（前3）", "来源组", "协议/覆盖"]): cell.text = text
    group_codes = {"METEOROLOGY":"MET", "RAINFALL_HYDRO_TIDE":"RHT", "OCEAN_PHYSICS":"OCN", "HUMAN_ACTIVITY":"HUM", "TEMPORAL_PROCESS":"TMP", "SPATIAL_STATIC":"SPA"}
    for task, label in zip(TASKS, TASKCN):
        rows = stats[stats.task.eq(task)].head(3)
        cells = t.add_row().cells
        for cell, text in zip(cells, [label, " / ".join(rows.feature),
                                      " / ".join(group_codes[x] for x in rows.source_group),
                                      "P1；n=3"]): cell.text = text
        for cell in cells:
            for p2 in cell.paragraphs:
                for run in p2.runs: run.font.size = Pt(8)
    t.autofit = False
    for row in t.rows:
        for cell, width in zip(row.cells, [Cm(2.2), Cm(8.0), Cm(2.8), Cm(2.0)]):
            cell.width = width
    table_caption = find(doc, "3.6 TabICLv2模型的一维ALE响应分析")
    table_caption.insert_paragraph_before("表3-3 十项水质任务第1折P1单查询SHAP主要变量与证据覆盖", style="表题")
    table_caption._p.addprevious(t._element)
    table_caption.insert_paragraph_before("注：主要变量按已冻结3个核心样本的平均绝对SHAP值汇总。MET=气象，RHT=降雨/水文/潮汐，OCN=海洋物理，HUM=人类活动代理，TMP=时间过程，SPA=空间静态；来源组按冻结77特征清单。P1为单查询协议，排名不作跨任务数值比较。")

    # 代表任务每项两条ALE并排；完整20条曲线保留于附录E。
    p = find(doc, "图E-1至图E-20是")
    p.text = "图3-10至图3-13集中展示四项代表任务的第1折P1一维ALE，每项两条冻结曲线；另六项任务与全部20条完整曲线、实际分箱及分箱计数见附录E。透明度三日太阳辐射总量仅有6个实际分箱，未补点。虚线为中心化零参考线，不以某个零交点推断环境过程没有作用。"
    anchor = find(doc, "3.7 TabICLv2解释结果的可靠性")
    analyses = [
        "Chl-a表层：风速约1.05—5.94 m/s范围内，两条ALE曲线的方向并不相同。当前风速曲线由低端−0.434 µg/L升至高端0.185 µg/L，前1日风速则由0.770降至−0.500 µg/L；各实际分箱至少60个样本、17个站点。P1 SHAP中风速—降雨代理有一定贡献，但3日太阳辐射贡献更高，故不能把这两条风速曲线概括为唯一主导关系。",
        "透明度：三日太阳辐射总量的6个实际分箱上ALE幅度较小（−0.018—0.017 m），每箱至少44个样本和11个站点；3日平均气温曲线从低温端0.081 m降至高温端−0.075 m，每箱至少60个样本、19个站点。SHAP全局排名更突出空间位置和季节编码，说明局部气象响应与全局贡献排序回答不同问题。",
        "TN表层：气温日变化与前1日风速的曲线在各自观测区间内总体由负值转正值，首末分箱分别为−0.013/0.017 mg/L和−0.014/0.024 mg/L；每箱至少60个样本和17个站点。SHAP中经度、年份偏移居前，因此气象曲线应解释为已选择变量上的条件响应，而非该任务全部重要性排序。",
        "TP中层：前1日气温与总云量曲线的幅度很小，分别约为−0.0021—0.0037 mg/L与−0.0012—0.0012 mg/L，每箱至少60个样本、17个站点。P1 SHAP排名以时间与空间变量为主；这些小效应尤其需要对照历史数值抖动尺度，不能仅凭曲线形状作强机制判断。",
    ]
    for i, (path, text) in enumerate(zip(ale_paths, analyses), 10):
        insert_figure_before(anchor, path, f"图3-{i} TabICLv2{TASKCN[[0,3,4,8][i-10]]}第1折P1一维ALE响应", 15.6)
        anchor.insert_paragraph_before(text)

    # 可靠性图内程序码不再作为正文图题，原始支持判定不变。
    for p in doc.paragraphs:
        if p.style.name.startswith("toc"): continue
        if p.text.startswith("图注："):
            p.text = p.text.replace("零线为无效应参考", "虚线为中心化零参考线")
            p.text = p.text.replace("图内程序状态文字仅为历史原图标签。", "")
    # 附录A/B与C/D/E/F连续排在一起；简历及成果置于附录之后（正式校样仍需核学校模板）。
    cv = find(doc, "个人简历")
    c = find(doc, "附录C TabICLv2")
    ps = doc.paragraphs
    i0 = next(i for i,x in enumerate(ps) if x._p is cv._p)
    i1 = next(i for i,x in enumerate(ps) if x._p is c._p)
    for p in ps[i0:i1]: doc._element.body.append(p._p)
    doc.save(OUTPUT)
    with (OUT / "图件来源与编号映射.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["V1图号", "V2图号", "处理", "冻结来源"])
        for n in range(3, 45):
            v = convert(re.match(r"图3-(\d+)", f"图3-{n}"))
            w.writerow([f"图3-{n}", v, "原图移至附录" if v.startswith(("图E", "图F")) else "正文重编号", "V1内嵌图或原始ALE曲线"])
    print("FIGURE_REFINED", OUTPUT, "appendix_ALE", len(appendix_plots), "shap", 10)


if __name__ == "__main__": build()
