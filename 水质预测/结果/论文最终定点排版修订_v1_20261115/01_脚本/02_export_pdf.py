"""更新独立修订稿的目录、分页与PDF；嵌入同源图像保持原生像素。"""
from pathlib import Path
import json, io, hashlib
import win32com.client
import fitz
from PIL import Image
from docx import Document

OUT=Path(__file__).resolve().parents[1]
P=OUT/'水质预测硕士学位论文_最终定点排版修订版.docx'
word=win32com.client.DispatchEx('Word.Application')
word.Visible=False;word.DisplayAlerts=0
d=None
try:
    d=word.Documents.Open(str(P),ReadOnly=False,AddToRecentFiles=False)
    d.Repaginate()
    for _ in range(2):
        d.Fields.Update()
        for toc in d.TablesOfContents:toc.Update()
        d.Repaginate()
    d.Save()
    rows=[]
    for ti in range(1,d.Tables.Count+1):
        table=d.Tables(ti)
        for ri in range(1,table.Rows.Count+1):
            row=table.Rows(ri);r=row.Range.Duplicate
            a=r.Start;b=r.End
            r.SetRange(a,a+1);sp=r.Information(3)
            r.SetRange(b-2,b-1);ep=r.Information(3)
            rows.append({'表格序号':ti,'行序号':ri,'起始物理页':sp,'结束物理页':ep,'首行重复':bool(row.HeadingFormat),'行跨页':sp!=ep})
    info={'Word版本':word.Version,'页数':d.ComputeStatistics(2),'图片数':d.InlineShapes.Count,'目录域数':d.TablesOfContents.Count,'表格数':d.Tables.Count,'表格行布局':rows}
    (OUT/'00_记录/Word实排核查.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    d.ExportAsFixedFormat2(OutputFileName=str(P.with_suffix('.pdf')),ExportFormat=17,OptimizeFor=0,OptimizeForImageQuality=True)
    print({k:v for k,v in info.items() if k!='表格行布局'},flush=True)
finally:
    if d is not None:d.Close(False)
    word.Quit()

source=Document(P);raw=[]
for shape in source.inline_shapes:
    rid=shape._inline.xpath('.//a:blip')[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    raw.append(source.part.related_parts[rid].blob)
pdf=fitz.open(P.with_suffix('.pdf'));before=[pg.get_text() for pg in pdf];frames=[];idx=0;seen={}
for page in pdf:
    for im in sorted(page.get_image_info(xrefs=True),key=lambda im:(im['bbox'][1],im['bbox'][0])):
        blob=raw[idx];idx+=1
        with Image.open(io.BytesIO(blob)) as pix:
            size=(pix.width,pix.height)
        rect=fitz.Rect(im['bbox']);assert abs(size[0]/size[1]-rect.width/rect.height)<.01
        h=hashlib.sha256(blob).hexdigest()
        if im['xref'] in seen:assert seen[im['xref']]==h
        else:page.replace_image(im['xref'],stream=blob);seen[im['xref']]=h
        frames.append({'页':page.number+1,'原生像素':size,'实际框':list(rect),'原生DPI':size[0]/rect.width*72,'图像SHA256':h})
assert idx==len(raw)==94
tmp=OUT/'PDF同源图像保真处理中.pdf'
pdf.save(tmp,garbage=3,deflate=True);pdf.close()
check=fitz.open(tmp);assert [pg.get_text() for pg in check]==before;check.close()
tmp.replace(P.with_suffix('.pdf'))
(OUT/'00_记录/PDF同源图像保真记录.json').write_text(json.dumps({'文字分页保持一致':True,'图片数':idx,'图像对象':frames},ensure_ascii=False,indent=2),encoding='utf-8')
print('PDF导出及94幅同源图像嵌入完成',flush=True)
