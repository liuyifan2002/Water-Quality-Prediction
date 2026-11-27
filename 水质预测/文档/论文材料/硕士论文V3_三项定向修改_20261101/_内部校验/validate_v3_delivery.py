"""只读验收：V3目录、冻结结果表、图件数量和PDF页码。"""
from pathlib import Path
import csv
import re
import zipfile
import fitz
import pandas as pd
from docx import Document

base=Path(__file__).resolve().parent
root=Path(r'E:\研究生\codex\水质预测')
v2=root/r'文档\论文材料\硕士论文V2_学术内容补强_20261101\水质预测硕士学位论文_学术内容补强V2_核验交付.docx'
stem='水质预测硕士学位论文_V3_三项定向修改_核验交付'
word=base/(stem+'.docx');pdf_path=base/(stem+'.pdf')
assert zipfile.ZipFile(word).testzip() is None
old=Document(v2);new=Document(word);pdf=fitz.open(pdf_path)
assert len(new.inline_shapes)==len(old.inline_shapes)==97
assert len(new.tables)==len(old.tables)+1==11
for index in range(3,10):
    a=[[c.text for c in row.cells] for row in old.tables[index].rows]
    b=[[c.text for c in row.cells] for row in new.tables[index].rows]
    assert a==b,('TABLE_CHANGED',index)
assert len(new.tables[2].rows)==7 and len(new.tables[2].columns)==5
assert sum(int(row.cells[1].text) for row in new.tables[2].rows[1:])==77
manifest=pd.read_csv(root/r'runs\tabiclv2_pilot\01_strict_feature_manifest.csv')
features=manifest.loc[manifest.included_in_pilot.astype(bool),'feature'].tolist()
appendix=[row.cells[1].text for row in new.tables[-1].rows[1:]]
assert appendix==features and len(appendix)==77
headings=[p.text for p in new.paragraphs if p.style.name.startswith('Heading')]
for number in range(1,8):
    assert sum(h.startswith(f'3.{number} ') for h in headings)==1,number
assert not any(h.startswith(('3.8 ','3.9 ','3.10 ')) for h in headings)
start=next(i for i,p in enumerate(new.paragraphs) if p.style.name=='Heading 1' and p.text=='图目录')
end=next(i for i in range(start+1,len(new.paragraphs)) if new.paragraphs[i].style.name=='Heading 1')
directory=[p.text for p in new.paragraphs[start+1:end] if p.text.startswith('图')]
assert len(directory)==97,len(directory)
normalize=lambda s:re.sub(r'\s+','',s)
pages=[normalize(p.get_text()) for p in pdf]
with (base/'图目录页码核验.csv').open(encoding='utf-8-sig',newline='') as f:
    records=list(csv.DictReader(f))
assert len(records)==97
for entry in records:
    caption=entry['图题'];page=int(entry['PDF实际页码'])
    assert normalize(caption)[:min(48,len(normalize(caption)))] in pages[page-1],(caption,page)
    assert any(line.startswith(caption+'\t'+str(page)) for line in directory),caption
assert '图2-1 研究区地理位置与香港近岸水质监测站点分布' in [p.text for p in new.paragraphs]
assert pdf.page_count==102
assert [round(pdf[i].rect.width) for i in (97,98,99)]==[842,842,842]
print('VALIDATED',len(pdf),'pages',len(directory),'figures',len(appendix),'features','results_tables_unchanged=7')
