"""从首次PDF实际排版回填静态图目录页码；不改变图件内容。"""
from pathlib import Path
import re
import fitz
from docx import Document

HERE=Path(__file__).resolve().parent
DOCX=HERE/'硕士学位论文_V6.3_图件优化稿.docx'
PDF=DOCX.with_suffix('.pdf')
doc=Document(DOCX);pdf=fitz.open(PDF)
ps=doc.paragraphs
start=max(i for i,p in enumerate(ps) if p.text.strip()=='图目录')
end=next(i for i in range(start+1,len(ps)) if ps[i].text.strip()=='表目录')
updated=[]
for p in ps[start+1:end]:
 m=re.match(r'^图((?:\d+|[A-Z])-\d+)\b',p.text.strip())
 if not m:continue
 num=m[1];hits=[]
 title_key=re.sub(r'\s+','',p.text.split('\t')[0])[:18]
 for i,page in enumerate(pdf):
  if i<12 or not page.get_images():continue
  if any(re.sub(r'\s+','',line).startswith(title_key) for line in page.get_text().splitlines()):hits.append(i+1)
 assert hits,(num,'figure caption not found on an image page')
 page_no=hits[0]
 p.text=p.text.split('\t')[0]+'\t'+str(page_no)
 updated.append((num,page_no))
assert len(updated)==33,len(updated)
doc.save(DOCX)
print('figure directory updated',len(updated),updated)
