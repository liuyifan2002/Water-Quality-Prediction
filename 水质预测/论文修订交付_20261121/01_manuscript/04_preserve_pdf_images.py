"""把Word导出PDF中的图像对象替换为同一V5 DOCX的原生图像字节。"""
from pathlib import Path
from hashlib import sha256
from io import BytesIO
import json
import fitz
from PIL import Image
from docx import Document

OUT=Path(__file__).resolve().parent
docx=OUT/'水质预测硕士学位论文_V5_正文修订.docx'
pdfpath=docx.with_suffix('.pdf')
tmp=OUT/'V5_PDF图像保真处理中.pdf'
assert docx.is_file() and pdfpath.is_file() and not tmp.exists()
d=Document(docx)
raw=[]
for shape in d.inline_shapes:
    blip=shape._inline.xpath('.//a:blip')[0]
    rid=blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    raw.append(d.part.related_parts[rid].blob)
pdf=fitz.open(pdfpath)
before=[page.get_text() for page in pdf]
records=[];i=0;seen={}
for page in pdf:
    ims=sorted(page.get_image_info(xrefs=True),key=lambda im:(im['bbox'][1],im['bbox'][0]))
    for im in ims:
        assert i<len(raw)
        blob=raw[i];i+=1
        pix=Image.open(BytesIO(blob));rect=fitz.Rect(im['bbox'])
        assert abs(pix.width/pix.height-rect.width/rect.height)<0.01,(page.number,i)
        h=sha256(blob).hexdigest()
        if im['xref'] in seen:assert seen[im['xref']]==h
        else:page.replace_image(im['xref'],stream=blob);seen[im['xref']]=h
        records.append({'图像序号':i,'PDF页':page.number+1,'像素宽':pix.width,
                        '像素高':pix.height,'排版宽pt':rect.width,
                        '有效DPI':round(pix.width/rect.width*72,1),'SHA256':h})
assert i==len(raw)==94
pdf.save(tmp,garbage=3,deflate=True);pdf.close()
verify=fitz.open(tmp)
assert [p.get_text() for p in verify]==before
assert sum(len(p.get_image_info(xrefs=True)) for p in verify)==94
verify.close();tmp.replace(pdfpath)
(OUT/'PDF图像保真记录.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'图像数':len(records),'最低有效DPI':min(x['有效DPI'] for x in records),
                  '文字分页不变':True},ensure_ascii=False))
