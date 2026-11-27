"""恢复附录E/F在OOXML中的真实连续顺序，并将sectPr置于文档末尾。"""
from pathlib import Path
from docx import Document

OUT = Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文V2_学术内容补强_20261101")
SOURCE = OUT / "水质预测硕士学位论文_学术内容补强V2_校样.docx"
TARGET = OUT / "水质预测硕士学位论文_学术内容补强V2_终审稿.docx"

assert SOURCE.is_file() and not TARGET.exists()
doc = Document(SOURCE)
f = next(p for p in doc.paragraphs if p.style.name == "Heading 2" and p.text.startswith("附录F "))
end = next(p for p in doc.paragraphs if p.style.name == "图题" and p.text.startswith("图E-20 "))
ps = doc.paragraphs
idx = next(i for i, p in enumerate(ps) if p._p is end._p)
note = ps[idx+1] if idx+1 < len(ps) and ps[idx+1].text.startswith("图注：") else end
note._p.addnext(f._p)
body = doc._element.body
sect = body.sectPr
if sect is not None:
    body.remove(sect)
    body.append(sect)
doc.save(TARGET)
print("APPENDICES_REORDERED", TARGET)
