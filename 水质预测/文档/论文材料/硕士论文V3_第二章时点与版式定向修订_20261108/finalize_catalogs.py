"""按PDF实际页面同步静态图、表目录；不改变正文图表与编号。"""
from pathlib import Path
import csv,re
import fitz
from docx import Document
from docx.shared import Pt

base=Path(__file__).resolve().parent
stem='水质预测硕士学位论文_V3_第二章定向修订_20261108'
doc_path=base/(stem+'.docx');pdf=fitz.open(base/(stem+'.pdf'))
doc=Document(doc_path)
norm=lambda s:re.sub(r'\s+','',s)
page_text=[norm(p.get_text()) for p in pdf]
for kind,style,expected in [('图','图题',97),('表','表题',12)]:
    captions=[p.text.strip() for p in doc.paragraphs if p.style.name==style and re.match(rf'^{kind}(?:\d+|[A-F])-\d+\s',p.text)]
    assert len(captions)==expected,(kind,len(captions))
    records=[]
    for c in captions:
        s=norm(c)
        for limit in (54,42,32,22):
            hits=[i+1 for i,t in enumerate(page_text) if i>=13 and s[:limit] in t]
            if len(hits)==1:break
        assert len(hits)==1,(c,hits)
        records.append((c,hits[0]))
    a=doc.paragraphs
    start=next(i for i,p in enumerate(a) if p.text==kind+'目录' and p.style.name=='Heading 1')
    end=next(i for i in range(start+1,len(a)) if a[i].style.name=='Heading 1')
    entries=a[start+1:end]
    assert len(entries)==expected,(kind,len(entries))
    for p,(caption,page) in zip(entries,records):
        p.text=f'{caption}\t{page}'
        p.paragraph_format.space_after=Pt(0)
        p.paragraph_format.line_spacing=1.0
        p.paragraph_format.first_line_indent=Pt(0)
        for r in p.runs:r.font.name='宋体';r.font.size=Pt(9)
    with (base/(kind+'目录页码核验.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow([kind+'号',kind+'题','PDF实际页码'])
        w.writerows((re.match(rf'^{kind}(?:\d+|[A-F])-\d+',c).group(),c,p) for c,p in records)
doc.save(doc_path)
print('CATALOGS_UPDATED',len(pdf),'pages',flush=True)
