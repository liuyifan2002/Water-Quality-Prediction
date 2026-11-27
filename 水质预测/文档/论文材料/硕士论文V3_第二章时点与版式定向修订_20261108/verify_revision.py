"""核对修订版的冻结内容、编号目录、17类来源和PDF文字。"""
from pathlib import Path
from collections import Counter
import hashlib,re,zipfile
import fitz
import pandas as pd
from docx import Document

root=Path(r'E:\研究生\codex\水质预测')
base=Path(__file__).resolve().parent
src=root/'文档/论文材料/硕士论文V3_第二章特征与候选数据扩充_20261101/水质预测硕士学位论文_V3_第二章特征与候选数据扩充_工作副本.docx'
stem='水质预测硕士学位论文_V3_第二章定向修订_20261108'
target=base/(stem+'.docx');pdf=fitz.open(base/(stem+'.pdf'))
old=Document(src);new=Document(target)
feature=pd.read_csv(root/'runs/tabiclv2_pilot/01_strict_feature_manifest.csv')
feature=feature.loc[feature.included_in_pilot.astype(bool),'feature'].tolist()
assert len(feature)==77
assert len(old.tables)==len(new.tables)==13
assert len([p for p in new.paragraphs if p.style.name=='表题'])==12
assert len([p for p in new.paragraphs if p.style.name=='图题'])==97
assert [r.cells[2].text for r in new.tables[3].rows[1:]]==feature
assert [r.cells[1].text for r in new.tables[12].rows[1:]]==feature
assert Counter(r.cells[1].text for r in new.tables[3].rows[1:])=={'MET':29,'RHT':22,'OCN':11,'HUM':6,'TMP':7,'SPA':2}
for i in [0,1,2,3,*range(5,13)]:
    assert [[c.text for c in r.cells] for r in old.tables[i].rows]==[[c.text for c in r.cells] for r in new.tables[i].rows],i
for n in range(1,18):
    a=old.tables[4].rows[n].cells;b=new.tables[4].rows[n].cells
    for col in (0,1,2,5,7):assert a[col].text==b[col].text,(n,col)
    assert b[7].text==f'E{n:02d}'

def chapter(doc,n):
    pars=doc.paragraphs
    first=next(i for i,p in enumerate(pars) if p.style.name=='Heading 1' and re.match(rf'^第\s*{n}\s*章',p.text))
    last=next(i for i in range(first+1,len(pars)) if pars[i].style.name=='Heading 1' and re.match(rf'^第\s*{n+1}\s*章',pars[i].text))
    return [p.text for p in pars[first:last]]
for n in (3,4,5):assert chapter(old,n)==chapter(new,n),n
def media(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        return Counter(hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist() if name.startswith('word/media/'))
assert media(src)==media(target)
norm=lambda x:re.sub(r'\s+','',x)
page_text=[norm(p.get_text()) for p in pdf]
body=''.join(page_text[22:25])
assert all(norm(f) in body for f in feature)
assert len(pdf)==108
for kind,style,count in [('图','图题',97),('表','表题',12)]:
    cap=[p.text.strip() for p in new.paragraphs if p.style.name==style and re.match(rf'^{kind}(?:\d+|[A-F])-\d+',p.text)]
    a=new.paragraphs
    begin=next(i for i,p in enumerate(a) if p.style.name=='Heading 1' and p.text==kind+'目录')
    end=next(i for i in range(begin+1,len(a)) if a[i].style.name=='Heading 1')
    entries=a[begin+1:end]
    assert len(cap)==len(entries)==count
    for c,p in zip(cap,entries):
        text,page=p.text.rsplit('\t',1)
        assert c==text and page.isdecimal() and norm(c)[:22] in page_text[int(page)-1],(c,page)
assert all(pdf[i].rect.width>pdf[i].rect.height for i in (26,27))
assert all(f'E{n:02d}' in ''.join(page_text[26:28]) for n in range(1,18))
assert '2.7候选环境数据的探索与筛选' in page_text[25]
assert '表2-3' in ''.join(page_text[22:25]) and '表2-4' in page_text[26]
print('VERIFIED',len(pdf),'pages; 77/77 features; 17/17 sources; 13 objects, 12 numbered tables; 97 media',flush=True)
