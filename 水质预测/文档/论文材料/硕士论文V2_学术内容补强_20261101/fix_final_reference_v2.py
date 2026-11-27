"""修正V2第11条官方网页链接；仅修改V2副本。"""
from pathlib import Path
from docx import Document

root=Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文V2_学术内容补强_20261101")
source=root/"水质预测硕士学位论文_学术内容补强V2_最终交付.docx"
target=root/"水质预测硕士学位论文_学术内容补强V2_核验交付.docx"
assert source.is_file() and not target.exists()
d=Document(source)
old="https://www.epd.gov.hk/epd/english/environmentinhk/water/marine_quality/marine_quality.html"
new="https://www.epd.gov.hk/epd/english/environmentinhk/water/hkwqrc/waterquality/marine.html"
hits=0
for p in d.paragraphs:
    if p.text.startswith("[11] ") and old in p.text:
        for run in p.runs:
            if old in run.text:
                run.text=run.text.replace(old,new)
                hits+=1
assert hits==1,hits
d.save(target)
print(target)
