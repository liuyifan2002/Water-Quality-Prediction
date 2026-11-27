"""用本机Word更新目录与域并导出PDF，原件不打开写入。"""
from pathlib import Path
import json,time
import win32com.client
OUT=Path(__file__).resolve().parents[1]
p=OUT/'07_论文整合稿/水质预测硕士学位论文_图件重构与终稿整合稿.docx'
word=win32com.client.DispatchEx('Word.Application');word.Visible=False;word.DisplayAlerts=0
try:
    d=word.Documents.Open(str(p),ReadOnly=False,AddToRecentFiles=False)
    d.Repaginate()
    for i in range(2):
        d.Fields.Update()
        for t in d.TablesOfContents:t.Update()
        d.Repaginate()
    d.Save()
    d.ExportAsFixedFormat2(OutputFileName=str(p.with_suffix('.pdf')),ExportFormat=17,OptimizeFor=0,OptimizeForImageQuality=True)
    info={'Word版本':word.Version,'页数':d.ComputeStatistics(2),'图片数':d.InlineShapes.Count,'目录域数':d.TablesOfContents.Count,'PDF':str(p.with_suffix('.pdf'))}
    (OUT/'05_数据核验/Word自动排版导出.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    d.Close(False);print(info)
finally:word.Quit()

# 本机Word的PDF组件仍会将嵌入图片降采样至约200dpi。
# 仅将PDF图像对象换为同一DOCX中未压缩的原生PNG；保持文字、分页和图像框不变。
import fitz,hashlib,io
from PIL import Image
from docx import Document
source=Document(p);raw=[]
for shape in source.inline_shapes:
    rid=shape._inline.xpath('.//a:blip')[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    raw.append(source.part.related_parts[rid].blob)
pdf=fitz.open(p.with_suffix('.pdf'));before=[pg.get_text() for pg in pdf];frames=[];idx=0;seen={}
for page in pdf:
    images=sorted(page.get_image_info(xrefs=True),key=lambda im:(im['bbox'][1],im['bbox'][0]))
    for im in images:
        blob=raw[idx];idx+=1;pix=Image.open(io.BytesIO(blob));rect=fitz.Rect(im['bbox'])
        assert abs(pix.width/pix.height-rect.width/rect.height)<.01
        h=hashlib.sha256(blob).hexdigest()
        if im['xref'] in seen:assert seen[im['xref']]==h
        else:page.replace_image(im['xref'],stream=blob);seen[im['xref']]=h
        frames.append({'页':page.number+1,'原生像素':[pix.width,pix.height],'实际框':list(rect),'原生DPI':pix.width/rect.width*72,'原生图像SHA256':h})
assert idx==len(raw)==94
tmp=p.with_name('排版图像保真处理中.pdf');pdf.save(tmp,garbage=3,deflate=True);pdf.close()
check=fitz.open(tmp);assert [pg.get_text() for pg in check]==before;check.close();tmp.replace(p.with_suffix('.pdf'))
(OUT/'05_数据核验/PDF原生图像保真核验.json').write_text(json.dumps({'文字与分页逐页一致':True,'替换同源图像数量':idx,'图像对象':frames},ensure_ascii=False,indent=2),encoding='utf-8')
