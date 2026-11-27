"""依据已导出的PDF逐图定位页码，重建V2图目录并保存独立版本。"""
from pathlib import Path
import csv
import re

import fitz
from docx import Document
from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.shared import Cm, Pt

OUT = Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文V2_学术内容补强_20261101")
STEM = "水质预测硕士学位论文_学术内容补强V2_终审稿"
SOURCE = OUT / (STEM + ".docx")
PDF = OUT / (STEM + ".pdf")
TARGET = OUT / "水质预测硕士学位论文_学术内容补强V2_正式版.docx"


def norm(text):
    return re.sub(r"\s+", "", text)


def main():
    assert not TARGET.exists()
    doc = Document(SOURCE)
    pages = [norm(page.get_text()) for page in fitz.open(PDF)]
    captions = [p.text.strip() for p in doc.paragraphs if p.style.name == "图题" and re.match(r"^图(?:\d+|[A-F])-\d+\s", p.text)]
    records = []
    for caption in captions:
        full = norm(caption)
        hits = []
        for length in [min(48, len(full)), min(32, len(full)), min(22, len(full))]:
            hits = [i + 1 for i, page in enumerate(pages) if i >= 15 and full[:length] in page]
            if len(hits) == 1:
                break
        if len(hits) != 1:
            raise AssertionError((caption, hits))
        records.append((caption, hits[0]))
    heads = [i for i, p in enumerate(doc.paragraphs) if p.style.name == "Heading 1" and p.text == "图目录"]
    assert len(heads) == 1
    start = heads[0]
    end = next(i for i in range(start+1, len(doc.paragraphs)) if doc.paragraphs[i].style.name == "Heading 1")
    old = doc.paragraphs[start+1:end]
    anchor = doc.paragraphs[end]
    for paragraph in old:
        paragraph._p.getparent().remove(paragraph._p)
    for caption, page in records:
        p = anchor.insert_paragraph_before()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.tab_stops.add_tab_stop(Cm(15.0), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        run = p.add_run(f"{caption}\t{page}")
        run.font.name = "宋体"
        run.font.size = Pt(9)
    # V2为可阅读校样，不保留要求读者自行F9的旧提示。
    for p in doc.paragraphs:
        if p.text.startswith("提示：首次打开文档后请在Word中全选并按F9"):
            p._p.getparent().remove(p._p)
    doc.save(TARGET)
    with (OUT / "图号与交叉引用核验.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["图号", "图题", "PDF页码", "目录状态"])
        for caption, page in records:
            number = re.match(r"^图(?:\d+|[A-F])-\d+", caption).group(0)
            w.writerow([number, caption, page, "已定位；静态目录与PDF对齐"])
    print("FIGURE_CATALOG_BUILT", len(records), TARGET)


if __name__ == "__main__":
    main()
