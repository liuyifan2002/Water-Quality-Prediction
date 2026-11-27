"""核对终稿对象保护及实际PDF分页，输出联系表、重点页与核验记录。"""
from pathlib import Path
from collections import Counter
import json,re,csv,hashlib,io,zipfile
import fitz
from PIL import Image,ImageDraw,ImageFont,ImageChops
from docx import Document
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
QA=OUT/'02_PDF实排检查';QA.mkdir(exist_ok=True)
P=OUT/'水质预测硕士学位论文_最终定点排版修订版.docx'
S=ROOT/'结果/论文最终图件重构_v1/07_论文整合稿/水质预测硕士学位论文_图件重构与终稿整合稿.docx'
doc=Document(P);src=Document(S);pdf=fitz.open(P.with_suffix('.pdf'))
# 仅清理本次独立目录中脚本生成的预览，防止分页变化后遗留过期页图。
assert QA.resolve().parent==OUT.resolve()
for pattern in ['页_*.jpg','重点检查页_*.png','全文联系表_*.jpg','第*章图件检查_*.jpg','附录*图件检查_*.jpg']:
    for old in QA.glob(pattern):
        assert old.is_file() and old.resolve().parent==QA.resolve()
        old.unlink()
def compact(s):return re.sub(r'\s+','',s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(name,rows):
    with (QA/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def media(p):
    with zipfile.ZipFile(p) as z:return sorted(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/'))
def nums(d):
    return {(ti,ri,ci):c.text for ti,t in enumerate(d.tables) for ri,r in enumerate(t.rows) for ci,c in enumerate(r.cells) if re.fullmatch(r'[+−-]?\d+(?:\.\d+)?%?',c.text.strip())}
caps=[p.text for p in doc.paragraphs if p.style.name=='图题']
capids=[re.match(r'^图([A-Z0-9]+-\d+)',p)[1] for p in caps]
oldcaps=[re.match(r'^图([A-Z0-9]+-\d+)',p.text)[1] for p in src.paragraphs if p.style.name=='图题']
assert capids==oldcaps and len(set(capids))==94
assert len(doc.tables)==len(src.tables)==13
assert nums(doc)==nums(src)
def paragraph_numbers(d):
    return Counter(tuple(re.findall(r'\d+(?:\.\d+)?',p.text)) for p in d.paragraphs if p.text.strip() and not p.style.name.lower().startswith('toc'))
assert paragraph_numbers(doc)==paragraph_numbers(src)
assert media(P)==media(S)
assert [p.text for p in doc.paragraphs if p.style.name=='参考文献']==[p.text for p in src.paragraphs if p.style.name=='参考文献']
for ti in [3,12]:assert [[c.text for c in r.cells] for r in doc.tables[ti].rows]==[[c.text for c in r.cells] for r in src.tables[ti].rows]
assert [r.cells[1].text for r in doc.tables[4].rows]==[r.cells[1].text for r in src.tables[4].rows]
protected=json.loads((OUT/'00_记录/原件保护哈希.json').read_text(encoding='utf-8'))
assert all(sha(p)==h for p,h in protected.items())

for t in doc.tables:
    assert t.rows[0]._tr.trPr.find(qn('w:tblHeader')) is not None
    assert all(r._tr.trPr.find(qn('w:cantSplit')) is not None for r in t.rows)
word=json.loads((OUT/'00_记录/Word实排核查.json').read_text(encoding='utf-8'))
row_splits=[r for r in word['表格行布局'] if r['行跨页']]
writecsv('全部表格逐行实排页码.csv',word['表格行布局'])

# 仅精确字段清单保留程序字段名；正文任务名和展示性结果表不保留内部任务标识。
internal=re.compile(r'\b(?:chla|tn|tp)_(?:surface|middle|bottom)\b|\b(?:ROBUST|MODERATE)\b|Secchi(?:P1|TreeSHAP|XGBoost)|四(?:个)?代表任务')
text='\n'.join(p.text for p in doc.paragraphs)
bad=internal.findall(text)
for ti,t in enumerate(doc.tables):
    if ti in [3,12]:continue
    for r in t.rows:
        for c in r.cells:bad+=internal.findall(c.text)
assert not bad,bad
pdftext=[p.get_text() for p in pdf]
assert not re.search(r'未定义书签|未找到引用源|Error!|Reference source not found','\n'.join(pdftext))

# 所有原生图片逐点核对，实际框与图题在同页且相邻。
shapes=list(doc.inline_shapes);images=[];actual={};idx=0
for page in pdf:
    for im in sorted(page.get_image_info(xrefs=True),key=lambda x:(x['bbox'][1],x['bbox'][0])):
        shape=shapes[idx];rid=shape._inline.xpath('.//a:blip')[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
        blob=doc.part.related_parts[rid].blob
        with Image.open(io.BytesIO(blob)) as a0,Image.open(io.BytesIO(pdf.extract_image(im['xref'])['image'])) as b0:
            a=a0.convert('RGB');b=b0.convert('RGB');same=a.size==b.size and ImageChops.difference(a,b).getbbox() is None
            dims=a.size;a.close();b.close()
        rect=fitz.Rect(im['bbox']);num=capids[idx]
        assert same,('图像像素不同',num)
        assert rect.x0>=0 and rect.y0>=0 and rect.x1<=page.rect.width+.1 and rect.y1<=page.rect.height+.1
        assert compact(caps[idx]) in compact(page.get_text()),('图题不在同页',num,page.number+1)
        lines=[]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                st=''.join(s['text'] for s in line['spans']).strip()
                if re.match(r'^图'+re.escape(num)+r'(?:\s|$)',st):lines.append(fitz.Rect(line['bbox']))
        assert any(0<=c.y0-rect.y1<35 for c in lines),('图题与图不相邻',num)
        images.append({'图号':'图'+num,'实际PDF页':page.number+1,'原图像素':str(dims),'Word与PDF像素逐点一致':same,'实际宽cm':rect.width/72*2.54,'实际高cm':rect.height/72*2.54,'有效DPI':dims[0]/rect.width*72,'图框在页内':True,'图题同页相邻':True})
        actual[num]=page.number+1;idx+=1
assert idx==94
writecsv('94图原生像素及实际排版核对.csv',images)
notes=[]
for pi,p in enumerate(doc.paragraphs):
    if p.style.name!='图题':continue
    note=doc.paragraphs[pi+1].text
    if not note.startswith('注'):continue
    num=re.match(r'^图([A-Z0-9]+-\d+)',p.text)[1];pn=actual[num]
    notes.append({'图题':p.text,'实际页':pn,'图注同页完整':compact(note) in compact(pdftext[pn-1]),'图注':note})
assert len(notes)==91 and all(r['图注同页完整'] for r in notes)
(OUT/'00_记录/全部图注同页核对.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
print('94幅图像、数值表格、参考文献及原件保护核验通过',flush=True)

# 从Word更新后的图目录逐项比对实际图页；表目录检查表题与表格首行的真实页。
catalogs=[]
for p in doc.paragraphs:
    if not p.style.name.lower().startswith('toc'):continue
    m=re.fullmatch(r'(图([A-Z0-9]+-\d+)\s+.+)\t(\d+)',p.text)
    if m:
        num=m[2];catalogs.append({'类型':'图','编号':num,'目录页码':int(m[3]),'实际页码':actual[num],'图名一致':compact(m[1])==compact(caps[capids.index(num)]),'页码一致':int(m[3])==actual[num]})
assert len(catalogs)==94 and all(x['页码一致'] and x['图名一致'] for x in catalogs)
tablecaps=[p.text for p in doc.paragraphs if p.style.name=='表题']
for p in doc.paragraphs:
    if not p.style.name.lower().startswith('toc'):continue
    m=re.fullmatch(r'(表([A-Z0-9]+-\d+)\s+.+)\t(\d+)',p.text)
    if m:
        cap=m[1];ti=next(i for i,s in enumerate(tablecaps) if re.match(r'^表'+re.escape(m[2])+r'\s',s))
        page=next(r['起始物理页'] for r in word['表格行布局'] if r['表格序号']==ti+1 and r['行序号']==1)
        catalogs.append({'类型':'表','编号':m[2],'目录页码':int(m[3]),'实际页码':page,'图名一致':compact(cap)==compact(tablecaps[ti]),'页码一致':int(m[3])==page})
assert len(catalogs)==106 and all(x['页码一致'] and x['图名一致'] for x in catalogs)
writecsv('图目录表目录逐项页码核对.csv',catalogs)
main_toc=[]
for p in doc.paragraphs:
    if not p.style.name.lower().startswith('toc') or re.match(r'^[图表][A-Z0-9]+-\d+',p.text):continue
    m=re.fullmatch(r'(.+)\t(\d+)',p.text)
    if m:
        pn=int(m[2]);main_toc.append({'目录标题':m[1],'页码':pn,'目标页存在标题':0<pn<=len(pdf) and compact(m[1]) in compact(pdftext[pn-1])})
writecsv('全文目录逐项页码核对.csv',main_toc)
assert main_toc and all(r['目标页存在标题'] for r in main_toc),[r for r in main_toc if not r['目标页存在标题']]

headers=[]
for ti,t in enumerate(doc.tables):
    rs=[r for r in word['表格行布局'] if r['表格序号']==ti+1]
    start=min(r['起始物理页'] for r in rs);end=max(r['结束物理页'] for r in rs)
    for pn in range(start,end+1):
        txt=compact(pdftext[pn-1]);cells=[compact(c.text) for c in t.rows[0].cells]
        missing=[s for s in cells if s not in txt]
        headers.append({'表格序号':ti+1,'物理页':pn,'跨页表格':start!=end,'表头文字完整出现':not missing,'缺少表头':';'.join(missing)})
writecsv('各页完整表头核对.csv',headers)

# 目录与正文的图表引用合法性，忽略概括性连续图号表达。
references=[]
validfig=set(capids);validtable={re.match(r'^表([A-Z0-9]+-\d+)',c)[1] for c in tablecaps}
for pi,p in enumerate(doc.paragraphs):
    if p.style.name in ['图题','表题','参考文献'] or p.style.name.lower().startswith('toc'):continue
    for typ,num in re.findall(r'(图|表)([A-Z0-9]+-\d+)',p.text):
        exists=num in (validfig if typ=='图' else validtable)
        references.append({'段落':pi,'引用':typ+num,'编号存在':exists,'正文':p.text})
writecsv('正文图表编号引用核对.csv',references)

font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
for i,page in enumerate(pdf):
    pix=page.get_pixmap(matrix=fitz.Matrix(.5,.5),alpha=False)
    im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((310,450))
    canvas=Image.new('RGB',(320,480),'#edf0f2');canvas.paste(im,((320-im.width)//2,0))
    ImageDraw.Draw(canvas).text((10,456),f'PDF {i+1}',font=font,fill='#333333')
    canvas.save(QA/f'页_{i+1:03}.jpg',quality=90)
def contacts(name,indices,cols=4):
    for off in range(0,len(indices),cols*5):
        take=indices[off:off+cols*5]
        canvas=Image.new('RGB',(320*cols,480*((len(take)+cols-1)//cols)),'#cfd7de')
        for j,pn in enumerate(take):
            with Image.open(QA/f'页_{pn:03}.jpg') as im:canvas.paste(im,((j%cols)*320,(j//cols)*480))
        canvas.save(QA/f'{name}_{off//(cols*5)+1:02}.jpg',quality=92)
contacts('全文联系表',list(range(1,len(pdf)+1)))
for ch in ['3','4','5']:
    contacts('第'+ch+'章图件检查',sorted({pn for n,pn in actual.items() if n.startswith(ch+'-')}),3)
for app in ['C','D','E','F','H']:
    contacts('附录'+app+'图件检查',sorted({pn for n,pn in actual.items() if n.startswith(app+'-')}),3)
table42=sorted({r['起始物理页'] for r in word['表格行布局'] if r['表格序号']==10})
focus=set([55,56]+table42)
focus.update(int(c['目录页码']) for c in catalogs if c['类型']=='表')
focus.update(p+1 for p,t in enumerate(pdftext) if p<16 and ('图目录' in t or '表目录' in t))
for pn in sorted(focus):pdf[pn-1].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(str(QA/f'重点检查页_{pn:03}.png'))
result={'PDF页数':len(pdf),'表4-2物理页':table42,'图件数':94,'表格数':13,'图目录条目':94,'表目录条目':12,'数值单元格一致':True,'数值单元格数量':len(nums(doc)),'94图媒体哈希及PDF逐点像素一致':True,'图题同图同页相邻':True,'91条图注同页完整':True,'图表目录标题页码逐项一致':True,'所有表格首行重复设置':True,'所有行禁止拆分设置':True,'实际被拆分行':row_splits,'表头实排未完整出现记录':[r for r in headers if not r['表头文字完整出现']],'无效正文图表编号':[r for r in references if not r['编号存在']],'正式特征及候选字段标识保留':True,'参考文献未改':True,'原稿四份保护哈希一致':True,'遗留内部任务名或粘连标题':bad,'DOCX_SHA256':sha(P),'PDF_SHA256':sha(P.with_suffix('.pdf'))}
(OUT/'00_记录/最终格式自动核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
