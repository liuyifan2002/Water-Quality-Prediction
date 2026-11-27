"""逐图比较PDF图像与DOCX原生图像像素，防止导出缓存或错误对象替换。"""
from common import *
from PIL import Image,ImageChops
import fitz,io
p=OUT/'07_论文整合稿/水质预测硕士学位论文_图件重构与终稿整合稿.docx';doc=Document(p);pdf=fitz.open(p.with_suffix('.pdf'))
caps=[x.text.split()[0] for x in doc.paragraphs if x.style.name=='图题'];shapes=list(doc.inline_shapes);idx=0;rows=[]
for page in pdf:
    for info in sorted(page.get_image_info(xrefs=True),key=lambda x:(x['bbox'][1],x['bbox'][0])):
        s=shapes[idx];rid=s._inline.xpath('.//a:blip')[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed');blob=doc.part.related_parts[rid].blob
        a=Image.open(io.BytesIO(blob)).convert('RGB');b=Image.open(io.BytesIO(pdf.extract_image(info['xref'])['image'])).convert('RGB');same=a.size==b.size;box=ImageChops.difference(a,b).getbbox() if same else '尺寸不同'
        rows.append({'图号':caps[idx],'PDF页':page.number+1,'Word原图像素':str(a.size),'PDF图像像素':str(b.size),'像素逐点一致':same and box is None,'差异范围':str(box)});idx+=1;a.close();b.close()
assert idx==len(shapes)==len(caps)==94
pd.DataFrame(rows).to_csv(OUT/'06_图片质量检查/94图Word与PDF原生像素逐点核验.csv',index=False,encoding='utf-8-sig')
bad=[x for x in rows if not x['像素逐点一致']];assert not bad,bad
print('94张PDF图像与Word原生图像逐点一致',flush=True)
