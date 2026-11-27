"""依据最终PDF正文图题回填图目录物理页码。"""
from pathlib import Path
import re
import fitz
from docx import Document
HERE=Path(__file__).resolve().parent
DOCX=HERE/'水质预测硕士学位论文_V6.3_新增15图整合稿.docx'
PDF=DOCX.with_suffix('.pdf')
doc=Document(DOCX);pdf=fitz.open(PDF);ps=doc.paragraphs
start=max(i for i,p in enumerate(ps) if p.text.strip()=='图目录')
end=next(i for i in range(start+1,len(ps)) if ps[i].text.strip()=='表目录')
rows=[]
for p in ps[start+1:end]:
    m=re.match(r'^图((?:\d+|[A-Z])-\d+)\s+(.+?)(?:\t|$)',p.text.strip())
    if not m:continue
    num,title=m.group(1),m.group(2)
    key=re.sub(r'\s+','',f'图{num} {title}')[:18]
    hits=[]
    for i,page in enumerate(pdf):
        if i<12 or not page.get_images():continue
        lines=[re.sub(r'\s+','',x) for x in page.get_text().splitlines()]
        if any(line.startswith(key) for line in lines):hits.append(i+1)
    assert hits,(num,title)
    p.text=f'图{num} {title}\t{hits[0]}'
    rows.append((num,hits[0]))
assert len(rows)==48,len(rows)
doc.save(DOCX)
print('图目录',len(rows),'entries',rows)
