"""核验正文特征、历史候选来源、冻结章节和PDF排版证据。"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import re
import zipfile
import fitz
import pandas as pd
from docx import Document

root=Path(r'E:\研究生\codex\水质预测')
base=Path(__file__).resolve().parent
source=root/r'文档\论文材料\硕士论文V3_三项定向修改_20261101\水质预测硕士学位论文_V3_三项定向修改_核验交付.docx'
stem='水质预测硕士学位论文_V3_第二章特征与候选数据扩充_工作副本'
target=base/(stem+'.docx');pdf_path=base/(stem+'.pdf')
old=Document(source);new=Document(target)
fm=pd.read_csv(root/r'runs\tabiclv2_pilot\01_strict_feature_manifest.csv')
feature=fm.loc[fm.included_in_pilot.astype(bool),'feature'].tolist()
body=next(t for t in new.tables if len(t.rows)==78 and len(t.columns)==6 and t.rows[0].cells[2].text=='精确英文变量名')
appold=old.tables[-1];appnew=new.tables[-1]
assert len(feature)==77
assert [r.cells[2].text for r in body.rows[1:]]==feature
assert [r.cells[1].text for r in appnew.rows[1:]]==feature
assert [[c.text for c in r.cells] for r in appold.rows]==[[c.text for c in r.cells] for r in appnew.rows]
groups=Counter(r.cells[1].text for r in body.rows[1:])
assert groups=={'MET':29,'RHT':22,'OCN':11,'HUM':6,'TMP':7,'SPA':2},groups
assert len(new.tables)==len(old.tables)+2
for table in old.tables:
    contents=[[c.text for c in r.cells] for r in table.rows]
    assert any([[c.text for c in r.cells] for r in t.rows]==contents for t in new.tables)
def media(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        return Counter(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/'))
assert media(source)==media(target)
def chapter_text(doc,number):
    pars=doc.paragraphs
    begin=next(i for i,p in enumerate(pars) if p.style.name=='Heading 1' and (p.text.strip().startswith(f'第 {number} 章') or p.text.strip().startswith(f'第{number}章')))
    end=next(i for i in range(begin+1,len(pars)) if pars[i].style.name=='Heading 1' and (pars[i].text.strip().startswith(f'第 {number+1} 章') or pars[i].text.strip().startswith(f'第{number+1}章')))
    return [p.text for p in pars[begin:end]]
for ch in (4,5):assert chapter_text(old,ch)==chapter_text(new,ch),ch
with (base/'候选环境数据来源核验表.csv').open(encoding='utf-8-sig',newline='') as f:candidates=list(csv.DictReader(f))
assert len(candidates)==17
for c in candidates:
    for p in c['evidence_files'].split(';'):
        assert (root/p.strip()).is_file(),(c['data_source'],p)
pdf=fitz.open(pdf_path);norm=lambda x:re.sub(r'\s+','',x)
texts=[norm(p.get_text()) for p in pdf]
feature_region=''.join(texts[22:25])
missing=[s for s in feature if norm(s) not in feature_region]
assert not missing,missing
for kind,style,total in [('图','图题',97),('表','表题',12)]:
    caption=[p.text.strip() for p in new.paragraphs if p.style.name==style and re.match(rf'^{kind}(?:\d+|[A-F])-\d+',p.text)]
    heading=next(i for i,p in enumerate(new.paragraphs) if p.text==kind+'目录' and p.style.name=='Heading 1')
    stop=next(i for i in range(heading+1,len(new.paragraphs)) if new.paragraphs[i].style.name=='Heading 1')
    entries=new.paragraphs[heading+1:stop]
    assert len(caption)==len(entries)==total
    for p,c in zip(entries,caption):
        title,page=p.text.rsplit('\t',1)
        assert title==c and page.isdigit()
        assert norm(c)[:22] in texts[int(page)-1],(c,page)
assert len(pdf)==108
assert all(pdf[i].rect.width>pdf[i].rect.height for i in (26,27))
assert all(pdf[i].rect.width<pdf[i].rect.height for i in (22,23,24,25,28))
assert all(len(pdf[i].get_text().strip())>500 for i in range(22,28))
assert '2.7候选环境数据的探索与筛选' in texts[25]
for s in ('表2-377项严格输入特征的名称、来源与构造','表2-4历史候选环境数据的来源、覆盖与筛选结果'):
    assert sum(s in t for t in texts[14:])==1,(s,sum(s in t for t in texts[14:]))
print('VERIFIED',len(pdf),'pages',len(feature),'features',len(candidates),'candidate sources',
      len(new.tables),'tables',sum(media(target).values()),'media items',groups)
