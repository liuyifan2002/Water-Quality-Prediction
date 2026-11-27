"""仅重排既有DOCX段落与图件；不计算模型或修改原始图像。"""
from __future__ import annotations

import csv
import difflib
import hashlib
import re
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph

ROOT = Path(r"E:\研究生\codex\水质预测")
BASE = ROOT / "文档/论文材料/硕士论文TabICLv2_SHAP_ALE_插图整合初稿_20261031_v4/水质预测硕士学位论文_TabICLv2_SHAP_ALE_插图整合初稿.docx"
AUDIT = ROOT / "runs/thesis_model_explanation_provenance_audit_v1/02_figure_model_mapping.csv"
OUT = ROOT / "文档/论文材料/硕士论文解释章节重构V1_20261101"
TARGET = OUT / "水质预测硕士学位论文_解释章节重构V1.docx"
TOC_START, TOC_END = 158, 239
XGB_NUMBERS = list(range(3, 8)) + list(range(23, 28)) + list(range(52, 57)) + [60, 61]
MAIN_NUMBERS = [1, 2] + list(range(8, 23)) + list(range(28, 52)) + list(range(57, 60))
CN = {"Chl-a": "叶绿素a", "Secchi": "透明度", "surface": "表层", "middle": "中层", "bottom": "底层"}


def cn(s: str) -> str:
    for a, b in CN.items():
        s = s.replace(a, b)
    return s


def set_text(p: Paragraph, text: str) -> None:
    """改已有w:t，保留段落、图片关系、书签和域代码。"""
    ts = p._p.xpath(".//w:t")
    if not ts:
        p.add_run(text)
        return
    ts[0].text = text
    for t in ts[1:]:
        t.text = ""


def set_toc_label(p: Paragraph, label: str) -> None:
    """只改PAGEREF域之前的可见文本，不触碰页码域。"""
    nodes = []
    for node in p._p.iter():
        if node.tag.endswith("}fldChar") and node.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}fldCharType") == "begin":
            break
        if node.tag.endswith("}t"):
            nodes.append(node)
    if not nodes:
        raise RuntimeError("图目录条目缺少文字节点")
    nodes[0].text = label
    for node in nodes[1:]:
        node.text = ""


def insert_after(anchor, nodes) -> None:
    for node in nodes:
        anchor.addnext(node)
        anchor = node


def remove(p: Paragraph) -> None:
    p._p.getparent().remove(p._p)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def title_for(old: int, original: str, new: str) -> str:
    title = cn(original.split(" ", 1)[1])
    if old == 1:
        return "TabICLv2历史五折四项任务SHAP平均绝对贡献"
    if old == 2:
        return "TabICLv2历史五折SHAP特征排名稳定性"
    if 8 <= old <= 17:
        name = title.split("前10项", 1)[0]
        return f"TabICLv2{name}第1折主要特征平均绝对SHAP贡献"
    if old == 18:
        return "TabICLv2十项任务第1折特征重要性排名变化"
    if 19 <= old <= 22:
        task = ["叶绿素a表层", "透明度", "TN表层", "TP中层"][old - 19]
        return f"TabICLv2{task}历史五折一维ALE响应"
    if 28 <= old <= 47:
        return "TabICLv2" + title.replace("的", "第1折", 1) + "（P1单查询）"
    if old == 48:
        return "TabICLv2十项任务一维ALE各分箱样本数"
    if old == 49:
        return "TabICLv2十项任务一维ALE各分箱独立站点数"
    if old == 50:
        return "TabICLv2九项任务P1单查询重复预测范围"
    if old == 51:
        return "TabICLv2历史数值波动与解释效应尺度对照"
    if 57 <= old <= 59:
        task = ["TN表层", "TN底层", "TP底层"][old - 57]
        return f"TabICLv2{task}二维ALE候选联合支持审查"
    xgb_titles = {
        3: "四项任务TreeSHAP主要特征重要性", 4: "叶绿素a表层TreeSHAP总体解释",
        5: "透明度TreeSHAP总体解释", 6: "TN表层TreeSHAP总体解释",
        7: "TP中层TreeSHAP总体解释", 23: "叶绿素a表层一维ALE响应",
        24: "透明度一维ALE响应", 25: "TN表层一维ALE响应",
        26: "TP中层一维ALE响应", 27: "四项任务一维ALE稳定性",
        52: "叶绿素a表层二维ALE交互", 53: "透明度二维ALE交互",
        54: "TN表层二维ALE交互", 55: "TP中层二维ALE交互",
        56: "二维ALE联合覆盖", 60: "TreeSHAP贡献与来源消融对照",
        61: "四项任务主要特征重要性比较",
    }
    return "数值等价受控复现XGBoost模型的" + xgb_titles[old]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {r["图号"]: r for r in csv.DictReader(AUDIT.open(encoding="utf-8-sig", newline=""))}
    src = Document(BASE)
    original = list(src.paragraphs)
    before = [p.text for p in original]
    assert len(original) == 699 and len(src.inline_shapes) == 91
    figures = {}
    for i, p in enumerate(original):
        m = re.match(r"^图(3-(\d+)|C-(\d+))\s", p.text.strip())
        if not m or i < 250:
            continue
        key = "图" + m.group(1)
        image = original[i - 1]
        assert image._p.xpath(".//a:blip"), (key, i)
        nodes = [image._p, p._p]
        if i + 1 < len(original) and original[i + 1].text.strip().startswith("图注："):
            nodes.append(original[i + 1]._p)
        figures[key] = (p, nodes, i)
        image.paragraph_format.keep_with_next = True
        p.paragraph_format.keep_together = True
        p.paragraph_format.keep_with_next = len(nodes) == 3
    assert len(figures) == 81

    mapping = {}
    for new_n, old_n in enumerate(MAIN_NUMBERS, 1):
        mapping[f"图3-{old_n}"] = f"图3-{new_n}"
    for new_n, old_n in enumerate(XGB_NUMBERS, 1):
        mapping[f"图3-{old_n}"] = f"图D-{new_n}"
    for n in range(1, 21):
        mapping[f"图C-{n}"] = f"图C-{n}"
    assert len(mapping) == 81 and len(set(mapping.values())) == 81

    # 对已有书签所附的图题改字；不重新嵌图，故原图字节和关系ID保持。
    original_titles = {}
    for old, new in mapping.items():
        p, _, _ = figures[old]
        original_titles[old] = p.text.strip()
        if old.startswith("图3-"):
            title = title_for(int(old[3:]), p.text, new)
            set_text(p, f"{new} {title}")
        else:
            set_text(p, cn(p.text).replace("图C-", "图C-"))

    # 图目录条目的PAGEREF域/书签原样保留，标签按新图题更新，XGBoost条目排到附录D后。
    figure_toc = {}
    for p in original[TOC_START:TOC_END]:
        m = re.match(r"^(图(?:3-\d+|C-\d+))\s", p.text.strip())
        if m:
            figure_toc[m.group(1)] = p
    assert len(figure_toc) == 81, len(figure_toc)
    for old, p in figure_toc.items():
        set_toc_label(p, figures[old][0].text)
    toc_anchor = original[TOC_END - 1]._p
    for old_n in XGB_NUMBERS:
        old = f"图3-{old_n}"
        if old in figure_toc:
            toc_anchor.addnext(figure_toc[old]._p)
            toc_anchor = figure_toc[old]._p

    # 核心段落：以审计结论为依据，限定模型、协议、折次和尺度。
    set_text(original[299], "3.1 预测模型概述")
    set_text(original[300], "本节概述XGBoost、TabM与TabPFN的基本定位，并将TabICLv2作为本文表格基础模型解释性分析的主线。XGBoost承担统一预测基准与树模型解释对照；TabM和TabPFN仅用于四项代表任务的现代基线比较。")
    set_text(original[301], "本章以统一的站点划分和77项严格特征为基础，重点分析TabICLv2在未见站点预测中的模型解释结果。针对TabICLv2，采用模型无关SHAP识别特征对预测的相对贡献，并利用一维ALE分析观测数据支持范围内的局部响应。历史四项代表任务的五折解释用于补充稳定性分析；后续十项任务解释采用预先定义的P1单查询预测函数，并统一呈现各任务第1折结果。数值等价受控复现XGBoost模型的TreeSHAP和ALE仅作为树模型解释对照，置于附录D，不用于解释TabICLv2的内部预测行为。XGBoost TreeSHAP归因定义于log1p目标尺度，而XGBoost ALE及TabICLv2 SHAP/ALE采用逆变换后的原始目标尺度，不直接比较不同模型的原始SHAP绝对值。所有解释结果描述模型统计响应，不直接等同于环境因果机制。")
    set_text(original[304], "XGBoost以加法方式构建回归树。第t轮在既有预测上加入新树，使损失函数的一阶、二阶近似与树复杂度正则项共同最小化[1]。其能够处理非线性与变量交互，TreeSHAP则用于树模型归因；本文将XGBoost作为统一预测基准及附录D中的树模型解释对照对象，而不是TabICLv2解释结果的来源。")
    set_text(original[333], "3.5 TabICLv2模型的SHAP解释")
    set_text(original[334], "3.5.1 模型无关SHAP方法与两种预测协议")
    set_text(original[335], "TabICLv2历史四任务五折SHAP使用固定批查询条件下的模型无关解释；后续十任务扩展采用P1单查询预测函数。历史20折SHAP已完成计算，但其中部分折未达到预登记的严格逐样本预测复现门槛，不能写作全部稳定通过。SHAP贡献仅反映模型预测中的特征归因，不代表环境因果效应。")
    set_text(original[336], "3.5.2 历史四项代表任务的五折SHAP结果")
    set_text(original[339], "原图展示四项代表任务的历史五折平均绝对SHAP贡献。经纬度在多个任务中排名靠前，说明模型利用了空间定位信息；不同任务的其他重要变量见图。该结果基于历史固定批查询条件，不与后续P1单查询SHAP原始绝对值直接比较。")
    set_text(original[340], "历史模型无关SHAP覆盖四项代表任务全部20折，共175个任务—折—样本文件，每个样本保留77项特征归因。计算覆盖不等于质量门槛全部通过；历史解释最终状态为部分完成，折间排名的支持程度亦因任务而异。")
    set_text(original[341], "3.5.3 历史五折特征排名稳定性")
    set_text(original[355], "3.5.4 十项水质任务第1折P1单查询SHAP解释")
    set_text(original[356], "下列十项任务图件均来自TabICLv2的P1单查询预测函数，分别呈现各任务第1折的SHAP特征重要性；这与前述历史固定批查询解释不是同一条件预测函数。小效应和数值敏感结果须结合独立质量审查谨慎解读，不作因果归因。")
    set_text(original[357], "ALE候选变量综合考虑SHAP贡献、连续性、物理意义、有效观测与站点覆盖；空间坐标虽可用于定位预测，但不直接作为环境过程的因果解释变量。")
    remove(original[344])

    set_text(original[391], "3.6 TabICLv2模型的一维ALE响应分析")
    set_text(original[392], "3.6.1 历史五折响应稳定性分析")
    set_text(original[393], "一维ALE按训练样本分位区间计算局部预测差异并累积、中心化；历史四任务五折曲线按各折训练分布分箱。它们反映历史固定批查询条件下的模型响应，不是十任务P1单查询结果，也不直接揭示环境因果机制。")
    remove(original[394])
    set_text(original[403], "历史四任务各有4项候选连续特征，共形成80条逐折一维ALE曲线。部分折未达到严格逐样本预测复现门槛，故图示只用于呈现历史五折响应差异，不作为十任务第1折P1结果的直接统计依据。")
    set_text(original[404], "3.6.2 十项水质任务的一维ALE响应")
    remove(original[405])
    remove(original[416])
    set_text(original[417], "以下20张曲线分别对应TabICLv2十项水质任务第1折P1单查询一维ALE，每项任务两条；保留原始分箱、零效应参考线及各分箱样本数。透明度三日太阳辐射总量曲线仅有6个实际分箱。数值敏感的小效应不被删去，但应与历史波动尺度对照，谨慎解释。")
    original[404]._p.addnext(original[417]._p)
    for i in range(28, 48):
        old = f"图3-{i}"
        for node in figures[old][1][2:]:
            note = Paragraph(node, original[0]._parent)
            set_text(note, cn(note.text).replace("各bin样本数", "各分箱样本数").replace("6个原始bin", "6个实际分箱"))

    set_text(original[490], "3.7 TabICLv2解释结果的可靠性与二维支持审查")
    set_text(original[491], "3.7.1 ALE样本与站点覆盖")
    set_text(original[492], "本节先审查20条一维ALE曲线各分箱的样本数和独立站点数。覆盖图只反映观测支持程度，不代表模型响应，也不对缺失分箱补零。")
    set_text(original[493], "3.7.2 数值稳定性")
    set_text(original[494], "P1重复预测探针仅覆盖九项任务，缺少叶绿素a表层；三次重复的观测范围不是预测置信区间。历史数值波动仍是解释小效应的限制，不能因后续探针重复而改写为严格稳定通过。")
    set_text(original[495], "3.7.3 二维ALE联合样本支持审查")
    remove(original[496])
    remove(original[505])
    set_text(original[506], "三组TabICLv2第1折P1候选仅完成4×4联合分箱支持审查：TN表层13/16格、TN底层7/16格、TP底层13/16格满足预登记联合支持条件。由于部分单元的样本与独立站点覆盖不足，未进一步计算对应二维ALE响应曲面。")
    set_text(original[518], "上述三张矩阵仅表示联合样本与站点支持，不是二维ALE模型响应图；历史四项任务的五折二维诊断与此处P1候选审查须分开表述。")
    for i, ratio in [(57, "13/16"), (58, "7/16"), (59, "13/16")]:
        note = Paragraph(figures[f"图3-{i}"][1][2], original[0]._parent)
        set_text(note, f"图注：{ratio}个联合分箱满足支持条件；由于部分单元样本或独立站点覆盖不足，未计算对应二维ALE响应曲面。图内程序状态文字仅为历史原图标签。")

    # 支持度和数值稳定性图从旧3.6迁到3.7两个新小节。
    insert_after(original[492]._p, figures["图3-48"][1] + figures["图3-49"][1])
    insert_after(original[494]._p, figures["图3-50"][1] + figures["图3-51"][1])
    for i in [48, 49, 50, 51]:
        notes = figures[f"图3-{i}"][1][2:]
        if notes:
            note = Paragraph(notes[0], original[0]._parent)
            set_text(note, cn(note.text).replace("bin", "分箱"))

    set_text(original[519], "3.8 XGBoost基准模型十项任务预测结果")
    set_text(original[520], "个体特征的TreeSHAP归因与删除整个特征来源组造成的预测性能变化回答不同问题，不能互作因果验证。两类证据在附录D中分别标准化后对照，不比较不同单位的原始数值。")
    # 此段稍后移入附录D；3.8保留最小预测结果指引。
    original[519]._p.addnext(original[519]._p.__class__(original[519]._p)) if False else None
    set_text(original[526], "本节仅对可比的预测评价结果作讨论。TabICLv2模型无关SHAP与XGBoost TreeSHAP的目标输出尺度和预测协议不同；解释对照仅限任务内排名或来源组，不直接比较原始SHAP绝对值。")
    set_text(original[528], "本章以TabICLv2为解释主线：历史四任务五折模型无关SHAP和一维ALE用于补充响应稳定性讨论，十任务第1折P1单查询结果用于主解释图件。二维P1候选仅完成联合支持审查，没有计算对应响应曲面。XGBoost解释结果作为独立树模型对照置于附录D。")
    set_text(original[529], "两类模型的SHAP输出尺度与预测函数条件不同，故不将XGBoost归因当作TabICLv2内部解释，也不把ALE统计响应解释为环境因果机制。")
    set_text(original[631], "本论文的TabICLv2解释包含历史四任务五折与后续十任务第1折P1单查询两类协议；XGBoost受控复现解释结果独立置于附录D。计算覆盖、支持度和数值稳定性分别评价。")
    set_text(original[638], "附录C TabICLv2局部SHAP解释结果")
    for i in range(1, 21):
        key = f"图C-{i}"
        p, nodes, _ = figures[key]
        set_text(p, cn(p.text))
        if len(nodes) == 3:
            note = Paragraph(nodes[2], original[0]._parent)
            if i <= 10:
                set_text(note, "图注：代表样本按预先登记的选择规则确定；77项特征的SHAP贡献加和已经核验。本图解释TabICLv2第1折P1单查询预测函数。")
            else:
                set_text(note, "图注：仅3个独立样本，另有同一样本的一次随机种子重复；不代表测试总体的SHAP分布。本图解释TabICLv2第1折P1单查询预测函数。")

    # 附录D只移动已核实为XGBoost的完整图像/图题块，图像字节不变。
    appendix = src.add_paragraph("附录D XGBoost基准模型解释对照", style="Heading 2")
    appendix.paragraph_format.page_break_before = True
    d_intro = src.add_paragraph("为对照传统树模型的特征归因与响应特征，本附录保留基于数值等价受控复现XGBoost模型的TreeSHAP和ALE结果。该部分仅用于基准模型解释对照，不作为TabICLv2内部预测机制的解释依据。XGBoost TreeSHAP定义于log1p目标尺度，而TabICLv2 SHAP采用原始目标尺度，两类原始SHAP绝对值不直接比较。受控复现模型与冻结测试预测数值一致，但不是原始冻结二进制模型。")
    src.add_paragraph("D.1 TreeSHAP特征归因", style="Heading 3")
    for n in range(3, 8):
        for node in figures[f"图3-{n}"][1]:
            src.element.body.insert(len(src.element.body) - 1, node)
    src.add_paragraph("D.2 一维ALE响应与稳定性", style="Heading 3")
    for n in range(23, 28):
        for node in figures[f"图3-{n}"][1]:
            src.element.body.insert(len(src.element.body) - 1, node)
    src.add_paragraph("D.3 二维ALE交互与联合覆盖", style="Heading 3")
    for n in range(52, 57):
        for node in figures[f"图3-{n}"][1]:
            src.element.body.insert(len(src.element.body) - 1, node)
    src.add_paragraph("D.4 TreeSHAP与来源消融对照", style="Heading 3")
    src.element.body.insert(len(src.element.body) - 1, original[520]._p)
    for n in [60, 61]:
        for node in figures[f"图3-{n}"][1]:
            src.element.body.insert(len(src.element.body) - 1, node)

    # 3.8保留标题而不重复附录D的图件。
    p38 = original[519].insert_paragraph_before("XGBoost预测性能与统一基准的定位见前述实验设计及后续结果章节；其TreeSHAP、ALE和来源消融解释对照集中列于附录D。")
    original[519]._p.addnext(p38._p)

    # 逐图图注与正文图号更新；不改带有PAGEREF/TOC域的目录段落。
    pattern = re.compile(r"图(?:3|C)-(?:\d+)")
    for p in src.paragraphs:
        if p.style.name == "图题":
            continue
        if p in [figure_toc.get(k) for k in figure_toc]:
            continue
        if p._p.xpath(".//w:fldChar"):
            continue
        text = p.text
        if not text or not pattern.search(text):
            continue
        revised = pattern.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
        if revised != text:
            set_text(p, revised)

    for old, p in figure_toc.items():
        set_toc_label(p, figures[old][0].text)

    # 在完成旧号替换后添加指向新号的正文引用，避免新号被再次当作旧号重映射。
    set_text(original[339], "图3-1展示四项代表任务的历史五折平均绝对SHAP贡献。经纬度在多个任务中排名靠前，说明模型利用了空间定位信息；不同任务的其他重要变量见图。该结果基于历史固定批查询条件，不与后续P1单查询SHAP原始绝对值直接比较。")
    rank_ref = src.add_paragraph("图3-2展示历史四项任务五折间的特征排名稳定性；计算覆盖和排名支持程度应分别评价。")
    original[341]._p.addnext(rank_ref._p)
    set_text(original[356], "图3-3至图3-13对应十项任务TabICLv2的P1单查询预测函数，分别呈现各任务第1折的SHAP特征重要性与任务内排名；这与历史固定批查询解释不是同一条件预测函数。小效应和数值敏感结果须结合独立质量审查谨慎解读，不作因果归因。")
    set_text(original[403], "图3-14至图3-17呈现历史四任务的五折一维ALE响应；每项任务有4项候选连续特征，共形成80条逐折曲线。部分折未达到严格逐样本预测复现门槛，故这些图只用于呈现历史五折响应差异，不作为十任务第1折P1结果的直接统计依据。")
    set_text(original[417], "图3-18至图3-37是TabICLv2十项水质任务第1折P1单查询一维ALE，每项任务两条；保留原始分箱、零效应参考线及各分箱样本数。透明度三日太阳辐射总量曲线仅有6个实际分箱。数值敏感的小效应不被删去，但应与历史波动尺度对照，谨慎解释。")
    set_text(original[492], "图3-38和图3-39分别审查20条一维ALE曲线各分箱的样本数与独立站点数。覆盖图只反映观测支持程度，不代表模型响应，也不对缺失分箱补零。")
    set_text(original[494], "图3-40的P1重复预测探针仅覆盖九项任务，缺少叶绿素a表层；三次重复的观测范围不是预测置信区间。图3-41保留历史数值波动的尺度证据，不能因后续探针重复而改写为严格稳定通过。")
    set_text(original[506], "图3-42至图3-44仅为TabICLv2第1折P1候选的4×4联合分箱支持审查：TN表层13/16格、TN底层7/16格、TP底层13/16格满足预登记联合支持条件。由于部分单元的样本与独立站点覆盖不足，未进一步计算对应二维ALE响应曲面。")
    set_text(p38, "XGBoost预测性能与统一基准的定位见前述实验设计及后续结果章节；其TreeSHAP、ALE和来源消融解释对照集中列于附录D图D-1至图D-17。")
    set_text(d_intro, "为对照传统树模型的特征归因与响应特征，图D-1至图D-17保留基于数值等价受控复现XGBoost模型的TreeSHAP和ALE结果。该部分仅用于基准模型解释对照，不作为TabICLv2内部预测机制的解释依据。XGBoost TreeSHAP定义于log1p目标尺度，而TabICLv2 SHAP采用原始目标尺度，两类原始SHAP绝对值不直接比较。受控复现模型与冻结测试预测数值一致，但不是原始冻结二进制模型。")
    c_intro = src.add_paragraph("图C-1至图C-10为按预先登记规则确定的典型样本局部SHAP；图C-11至图C-20为每项任务仅3个独立样本及一次种子重复的分布展示。两类图均解释TabICLv2第1折P1单查询预测函数，不代表全测试集分布。")
    original[638]._p.addnext(c_intro._p)

    # 页码由Word重排后更新；预置更新域属性。
    settings = src.settings.element
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    if not settings.xpath(".//w:updateFields"):
        u = OxmlElement("w:updateFields")
        u.set(qn("w:val"), "true")
        settings.append(u)
    src.save(TARGET)

    records = []
    for old, new in mapping.items():
        old_row = audit[old]
        model = "XGBoost受控复现模型" if new.startswith("图D-") else "TabICLv2"
        section = "附录D" if new.startswith("图D-") else "附录C" if new.startswith("图C-") else "3.5" if int(new[3:]) <= 13 else "3.6" if int(new[3:]) <= 37 else "3.7"
        records.append({"old_figure_id": old, "new_figure_id": new, "model": model, "new_section": section, "status": "已移动并重编号" if new.startswith("图D-") else "已重编号" if old != new else "保留并规范化"})
    write_csv(OUT / "figure_renumber_mapping.csv", records)
    write_csv(OUT / "xgboost_appendix_migration_manifest.csv", [
        {"old_figure_id": r["old_figure_id"], "new_figure_id": r["new_figure_id"], "source_png": audit[r["old_figure_id"]]["图片原始路径"], "source_sha256": audit[r["old_figure_id"]]["图片SHA256"], "status": "附录D已迁入"}
        for r in records if r["new_figure_id"].startswith("图D-")])
    write_csv(OUT / "tabicl_maintext_figure_manifest.csv", [
        {"old_figure_id": r["old_figure_id"], "new_figure_id": r["new_figure_id"], "section": r["new_section"], "source_png": audit[r["old_figure_id"]]["图片原始路径"], "source_sha256": audit[r["old_figure_id"]]["图片SHA256"], "status": "正文保留"}
        for r in records if r["new_figure_id"].startswith("图3-")])
    with (OUT / "chapter3_diff_view.md").open("w", encoding="utf-8") as f:
        f.write("# 第三章修改差异视图\n\n```diff\n")
        f.writelines(line + "\n" for line in difflib.unified_diff(before[299:530], [p.text for p in src.paragraphs if p.text.startswith(("3.", "图3-"))], fromfile="原稿第三章片段", tofile="重构稿第三章标题与图题", lineterm=""))
        f.write("```\n")
    print(f"saved {TARGET} mapping={len(records)} XGB={len(XGB_NUMBERS)} main={len(MAIN_NUMBERS)}")


if __name__ == "__main__":
    main()
