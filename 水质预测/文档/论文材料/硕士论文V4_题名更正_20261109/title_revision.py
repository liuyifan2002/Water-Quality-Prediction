from __future__ import annotations

import hashlib
from pathlib import Path

from docx import Document


ROOT = Path(r"E:\研究生\codex\水质预测\文档\论文材料")
SOURCE = ROOT / "硕士论文V4_结构与地图定向修订_20261108" / "水质预测硕士学位论文_V4_结构与地图定向修订.docx"
OUT = ROOT / "硕士论文V4_题名更正_20261109"
TARGET = OUT / "水质预测硕士学位论文_V4_题名更正.docx"
OLD = "基于多源环境数据与表格基础模型的近岸水质预测及空间泛化研究"
NEW = "基于多源环境数据与表格模型的近岸水质预测研究"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = digest(SOURCE)
    doc = Document(SOURCE)
    cover = doc.paragraphs[8]
    assert len(cover.runs) == 3
    assert cover.text.replace("\n", "") == OLD
    cover.runs[0].text = "基于多源环境数据与表格模型的"
    assert cover.runs[1].text == "\n"
    cover.runs[2].text = "近岸水质预测研究"
    changed_headers = 0
    for section in doc.sections:
        for paragraph in section.header.paragraphs:
            if paragraph.text == OLD:
                assert len(paragraph.runs) == 1
                paragraph.runs[0].text = NEW
                changed_headers += 1
    assert changed_headers == len(doc.sections), (changed_headers, len(doc.sections))
    assert doc.core_properties.title == OLD
    doc.core_properties.title = NEW
    doc.save(TARGET)
    assert digest(SOURCE) == source_hash, "原始母本被修改"

    check = Document(TARGET)
    assert check.paragraphs[8].text.replace("\n", "") == NEW
    assert check.core_properties.title == NEW
    assert [p.text for p in check.paragraphs[9:]] == [p.text for p in Document(SOURCE).paragraphs[9:]]
    assert len(check.tables) == len(Document(SOURCE).tables)
    assert len(check.inline_shapes) == len(Document(SOURCE).inline_shapes)
    print("SOURCE_SHA256", source_hash)
    print("TARGET_SHA256", digest(TARGET))
    print("UPDATED_HEADERS", changed_headers)
    print("TARGET", TARGET)


if __name__ == "__main__":
    main()
