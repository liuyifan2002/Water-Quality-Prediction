"""检查输出DOCX/PDF结构、图号、图片与分页，并生成只用于验收的缩略预览。"""
from __future__ import annotations

import csv
import difflib
import hashlib
import re
import zipfile
from collections import Counter
from pathlib import Path

import fitz
from PIL import Image, ImageDraw
from docx import Document

ROOT = Path(r"E:\研究生\codex\水质预测")
OUT = ROOT / "文档/论文材料/硕士论文解释章节重构V1_20261101"
BASE = ROOT / "文档/论文材料/硕士论文TabICLv2_SHAP_ALE_插图整合初稿_20261031_v4/水质预测硕士学位论文_TabICLv2_SHAP_ALE_插图整合初稿.docx"
DOCX = OUT / "水质预测硕士学位论文_解释章节重构V1.docx"
PDF = OUT / "水质预测硕士学位论文_解释章节重构V1.pdf"


def media_hashes(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        return sorted(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/"))


def main() -> None:
    src, doc = Document(BASE), Document(DOCX)
    assert len(src.tables) == len(doc.tables) == 9
    assert len(src.sections) == len(doc.sections) == 1
    assert media_hashes(BASE) == media_hashes(DOCX)
    captions = [p.text for p in doc.paragraphs if p.style.name == "图题"]
    cap = {k: [x for x in captions if re.match(rf"^图{k}-\d+\s", x)] for k in ("3", "C", "D")}
    assert (len(cap["3"]), len(cap["C"]), len(cap["D"])) == (44, 20, 17)
    for k, n in (("3", 44), ("C", 20), ("D", 17)):
        assert [int(re.match(rf"^图{k}-(\d+)", x).group(1)) for x in cap[k]] == list(range(1, n + 1))
    assert "SHAP和ALE解释对象明确限定为XGBoost" not in "\n".join(p.text for p in doc.paragraphs)
    assert "TreeSHAP加性检验20/20折通过" not in "\n".join(p.text for p in doc.paragraphs)
    assert not any(t in p.text for p in doc.paragraphs for t in ("ALE_2D_BLOCKED_BY_SUPPORT", "CORE_3", "target_quantile=0.5", "fold_1"))
    p35 = next(i for i, p in enumerate(doc.paragraphs) if p.text.startswith("3.5 TabICLv2") and p.style.name == "Heading 2")
    pd = next(i for i, p in enumerate(doc.paragraphs) if p.text.startswith("附录D ") and p.style.name == "Heading 2")
    maintext = "\n".join(p.text for p in doc.paragraphs[p35:pd])
    assert not re.search(r"^图D-\d+", maintext, re.M)
    dtext = "\n".join(p.text for p in doc.paragraphs[pd:])
    assert not re.search(r"^图3-\d+", dtext, re.M)
    toc = [p for p in doc.paragraphs if p.text.startswith(("图3-", "图C-", "图D-")) and p._p.xpath(".//w:fldChar")]
    assert len(toc) == 81
    assert [int(re.match(r"^图D-(\d+)", p.text).group(1)) for p in toc if p.text.startswith("图D-")] == list(range(1, 18))

    pdf = fitz.open(PDF)
    found = {"3": {}, "C": {}, "D": {}}
    suspect_pages = []
    page_texts = [re.sub(r"\s+", "", page.get_text()) for page in pdf]
    captions_by_page = {}
    for kind, lines in cap.items():
        for caption in lines:
            label = re.match(rf"^图{kind}-\d+", caption).group(0)
            needle = re.sub(r"\s+", "", caption)[:min(26, len(re.sub(r"\s+", "", caption)))]
            hits = [i for i, text in enumerate(page_texts) if i >= 13 and needle in text]
            if len(hits) != 1:
                raise AssertionError((label, "caption match pages", [i + 1 for i in hits]))
            found[kind][label] = hits[0] + 1
            captions_by_page.setdefault(hits[0], []).append(label)
    for index, labels in captions_by_page.items():
        if len(pdf[index].get_images(full=True)) < len(labels):
            suspect_pages.append((index + 1, labels, len(pdf[index].get_images(full=True))))
    for k, n in (("3", 44), ("C", 20), ("D", 17)):
        assert set(found[k]) == {f"图{k}-{i}" for i in range(1, n + 1)}, (k, len(found[k]))
    assert not suspect_pages, suspect_pages
    toc_page_mismatches = []
    for entry in toc:
        m = re.match(r"^(图(3|C|D)-\d+)\s.*?(\d+)\s*$", entry.text)
        if not m:
            toc_page_mismatches.append((entry.text[:50], "无法提取页码"))
        elif int(m.group(3)) != found[m.group(2)][m.group(1)]:
            toc_page_mismatches.append((m.group(1), m.group(3), found[m.group(2)][m.group(1)]))
    assert not toc_page_mismatches, toc_page_mismatches

    preview_pages = [found["3"]["图3-1"], found["3"]["图3-18"], found["3"]["图3-38"], found["3"]["图3-42"], found["C"]["图C-1"], found["D"]["图D-1"], found["D"]["图D-17"]]
    thumbs = []
    for n in preview_pages:
        pix = pdf[n - 1].get_pixmap(matrix=fitz.Matrix(0.75, 0.75), alpha=False)
        im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        thumbs.append((n, im))
    cols = 3
    w = max(im.width for _, im in thumbs)
    h = max(im.height for _, im in thumbs) + 27
    contact = Image.new("RGB", (cols * w, ((len(thumbs) + cols - 1) // cols) * h), "#e7e7e7")
    draw = ImageDraw.Draw(contact)
    for i, (n, im) in enumerate(thumbs):
        x, y = (i % cols) * w, (i // cols) * h
        contact.paste(im, (x, y + 27))
        draw.text((x + 8, y + 6), f"PDF p.{n}", fill="black")
    contact.save(OUT / "quality_contact_sheet.png")

    with (OUT / "qa_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in [
            ("pdf_pages", len(pdf)), ("source_media", len(media_hashes(BASE))), ("output_media", len(media_hashes(DOCX))),
            ("maintext_figures", len(cap["3"])), ("appendix_c_figures", len(cap["C"])), ("appendix_d_figures", len(cap["D"])),
            ("figure_toc_entries", len(toc)), ("figure_toc_page_mismatches", len(toc_page_mismatches)),
            ("captions_with_same_page_image", 81), ("suspect_pages", len(suspect_pages)),
        ]:
            writer.writerow([k, v])
    with (OUT / "chapter3_diff_view.md").open("w", encoding="utf-8") as f:
        f.write("# 全文段落差异视图（原稿 → 重构稿）\n\n说明：图件移动会显示为删除与新增；原图文件字节未改变。\n\n```diff\n")
        f.writelines(line + "\n" for line in difflib.unified_diff(
            [p.text for p in src.paragraphs], [p.text for p in doc.paragraphs],
            fromfile="原插图整合稿", tofile="解释章节重构V1", lineterm=""))
        f.write("```\n")
    changes = [
        (301, "本章以统一的站点划分", "纠正统一归属XGBoost的错误", "模型归属修正"),
        (335, "TabICLv2历史四任务五折SHAP", "分离XGBoost TreeSHAP质量记录", "事实修正"),
        (403, "图3-14至图3-17", "区分历史五折与P1单折ALE", "科学边界修正"),
        (494, "图3-40的P1重复预测探针", "保留九任务探针与历史数值波动限制", "科学边界修正"),
        (518, "上述三张矩阵仅表示", "将二维程序状态改为中文支持说明", "术语统一"),
        (529, "两类模型的SHAP输出尺度", "清除互相矛盾的章节小结", "模型归属修正"),
        (638, "附录C TabICLv2", "明确附录C只包含TabICLv2", "标题修正"),
    ]
    with (OUT / "chapter3_review_view.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["原稿段落索引", "新稿段落索引", "原文", "修改后", "修改原因", "修改类型", "是否改变科学归属", "待作者确认"])
        for old_i, prefix, reason, kind in changes:
            hit = next(((i, p.text) for i, p in enumerate(doc.paragraphs) if p.text.startswith(prefix)), None)
            assert hit is not None, prefix
            w.writerow([old_i, hit[0], src.paragraphs[old_i].text, hit[1], reason, kind, "是" if kind in ("模型归属修正", "事实修正") else "否", "否，依据已核实溯源审计"])
    print("QA_OK", "pages", len(pdf), "figures", {k: len(v) for k, v in found.items()}, "preview", OUT / "quality_contact_sheet.png")


if __name__ == "__main__":
    main()
