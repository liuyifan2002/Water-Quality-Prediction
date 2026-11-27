"""按Word首次排版PDF中的真实页码刷新手工图、表目录。"""
from pathlib import Path
import re
from docx import Document
from docx.oxml.ns import qn
import fitz

OUT=Path(__file__).resolve().parent
DOC=OUT/'水质预测硕士学位论文_V4_结构与地图定向修订.docx'
PDF=OUT/'水质预测硕士学位论文_V4_结构与地图定向修订.pdf'
doc=Document(DOC);pdf=fitz.open(PDF)
pages=[''.join(p.get_text().split()) for p in pdf]
body=list(doc.element.body)
def text(e):return ''.join(t.text or '' for t in e.iter(qn('w:t'))).strip()
def one_position(s):
    a=[i for i,e in enumerate(body) if text(e)==s]
    assert len(a)==1,(s,a)
    return a[0]
first=next(i for i,e in enumerate(body) if text(e).startswith('第1章 绪论') and e.find(qn('w:pPr')) is not None and i>260)
for kind in ('图','表'):
    actual=[text(e) for e in body[first:] if re.match(r'^'+kind+r'(?:[1-6]|[A-Z])-\d+\s',text(e)) and e.tag==qn('w:p')]
    assert len(actual)==(94 if kind=='图' else 11),(kind,len(actual))
    a=one_position('图目录' if kind=='图' else '表目录')
    z=one_position('表目录' if kind=='图' else '主要符号表')
    entries=[e for e in body[a+1:z] if e.tag==qn('w:p') and re.match(r'^'+kind+r'(?:[1-6]|[A-Z])-\d+',text(e))]
    assert len(entries)==(97 if kind=='图' else 11),(kind,len(entries))
    for e in entries[len(actual):]:e.getparent().remove(e)
    for e,title in zip(entries,actual):
        key=''.join(title.split())
        found=[i+1 for i,p in enumerate(pages[14:],start=14) if key in p]
        assert len(found)==1,(title,found)
        ts=list(e.iter(qn('w:t')))
        assert len(ts)==2,(title,[x.text for x in ts])
        ts[0].text=title
        ts[1].text=str(found[0])
    print(kind,len(actual),'directory refreshed')
doc.save(DOC)
