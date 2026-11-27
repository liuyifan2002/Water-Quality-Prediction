"""Word更新目录导出PDF，并进行图题、表题、旧解释与页码静态检查。"""
from pathlib import Path
import csv, json, re, hashlib
import fitz
from docx import Document
import win32com.client

OUT=Path(__file__).resolve().parents[1]
DOC=OUT/'硕士学位论文_V6.1.docx'
PDF=OUT/'硕士学位论文_V6.1.pdf'
assert DOC.is_file()
word=win32com.client.DispatchEx('Word.Application')
word.Visible=False;word.DisplayAlerts=0
try:
    d=word.Documents.Open(str(DOC),ReadOnly=False)
    d.Fields.Update()
    for item in d.TablesOfContents:item.Update()
    for item in d.TablesOfFigures:item.Update()
    d.Save();d.ExportAsFixedFormat(str(PDF),17)
    d.Close(False)
finally:word.Quit()

doc=Document(DOC);pdf=fitz.open(PDF)
fig=[p.text for p in doc.paragraphs if p.style.name=='图题']
tbl=[p.text for p in doc.paragraphs if p.style.name=='表题']
names=[re.match(r'^图([1-9]-\d+)',s).group(1) for s in fig if re.match(r'^图([1-9]-\d+)',s)]
expected=['1-1']+[f'2-{i}' for i in range(1,5)]+[f'4-{i}' for i in range(1,6)]+[f'5-{i}' for i in range(1,5)]
assert names==expected,(names,expected)
assert len(fig)==14 and len(tbl)==10 and len(doc.inline_shapes)==14 and len(doc.tables)==11
body='\n'.join(p.text for p in doc.paragraphs if not p.style.name.lower().startswith('toc'))
for x in ['新版本解释待核验','图C-','图D-','图E-','图F-','图H-','图3-1 ','图3-8 ']:assert x not in body,x
for x in ['WQ77_REPAIRED_20261115_V1','TN中层','配对站点Bootstrap','图5-4']:assert x in body,x
page_map=[]
for i,p in enumerate(pdf):
    if i<12 or not p.get_images(full=True):continue
    s=p.get_text()
    for no in expected:
        if re.search(rf'图{re.escape(no)}\s+',s):page_map.append({'图号':'图'+no,'PDF页序':i+1})
rows=[]
for x in fig:
    m=re.match(r'^(图\d-\d+)\s+(.+)$',x)
    rows.append({'图号':m.group(1),'图名':m.group(2),'所在章':m.group(1)[1],
      '正文位置':'正文','数据版本':'WQ77_REPAIRED_20261115_V1' if m.group(1)[1] in '45' and m.group(1)!='图5-3' else '历史78特征独立实验' if m.group(1)=='图5-3' else '原研究区或监测数据',
      'PDF页序': next((r['PDF页序'] for r in page_map if r['图号']==m.group(1)),''),
      '插入状态':'是'})
with (OUT/'图表目录与插图位置清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
report={'DOCX_SHA256':hashlib.sha256(DOC.read_bytes()).hexdigest(),'PDF_SHA256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
 'PDF页数':len(pdf),'正文图题数':len(fig),'表题数':len(tbl),'内嵌图片数':len(doc.inline_shapes),'实体表格数':len(doc.tables),
 '图号顺序':names,'表题':tbl}
(OUT/'排版核查/自动核查结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2)[:800])
