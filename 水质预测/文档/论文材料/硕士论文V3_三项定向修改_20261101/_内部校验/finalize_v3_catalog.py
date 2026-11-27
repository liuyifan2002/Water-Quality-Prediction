"""根据PDF实际页码重建V3图目录，保留正文和原图号。"""
from pathlib import Path
import csv
import re
import sys
import fitz
from docx import Document
from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.shared import Cm, Pt

base=Path(__file__).resolve().parent
stem='水质预测硕士学位论文_V3_三项定向修改'
target=base/(stem+'_核验交付.docx')
refresh='--refresh' in sys.argv
source=target if refresh else base/(stem+'.docx')
pdf=fitz.open(base/(stem+('_核验交付' if refresh else '')+'.pdf'))
doc=Document(source)
norm=lambda s:re.sub(r'\s+','',s)
page_text=[norm(p.get_text()) for p in pdf]
captions=[p.text.strip() for p in doc.paragraphs if p.style.name=='图题' and re.match(r'^图(?:\d+|[A-F])-\d+\s',p.text)]
records=[]
for caption in captions:
    full=norm(caption)
    for limit in (48,32,22):
        hits=[i+1 for i,text in enumerate(page_text) if i>=12 and full[:limit] in text]
        if len(hits)==1:break
    assert len(hits)==1,(caption,hits)
    records.append((caption,hits[0]))
assert len(records)==97,len(records)
start=next(i for i,p in enumerate(doc.paragraphs) if p.style.name=='Heading 1' and p.text=='图目录')
end=next(i for i in range(start+1,len(doc.paragraphs)) if doc.paragraphs[i].style.name=='Heading 1')
anchor=doc.paragraphs[end]
for p in doc.paragraphs[start+1:end]:p._p.getparent().remove(p._p)
for caption,page in records:
    p=anchor.insert_paragraph_before()
    p.paragraph_format.space_after=Pt(0)
    p.paragraph_format.line_spacing=1.0
    p.paragraph_format.first_line_indent=Pt(0)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(15.0),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
    run=p.add_run(f'{caption}\t{page}')
    run.font.name='宋体';run.font.size=Pt(9)
doc.save(target)
with (base/'图目录页码核验.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['图号','图题','PDF实际页码'])
    for caption,page in records:
        writer.writerow([re.match(r'^图(?:\d+|[A-F])-\d+',caption).group(),caption,page])
print('CATALOG_FINALIZED',len(records),target)
