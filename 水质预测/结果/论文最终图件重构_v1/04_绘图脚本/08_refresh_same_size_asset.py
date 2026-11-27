"""已排版图件的同尺寸更新；不改Word任何XML、分页或目录。参数为图号。"""
from common import *
import zipfile,fitz,sys
num=sys.argv[1] if len(sys.argv)>1 else '3-2';dp=OUT/'07_论文整合稿/水质预测硕士学位论文_图件重构与终稿整合稿.docx'
m=json.loads((OUT/'05_数据核验/图件登记.json').read_text(encoding='utf-8'));r=m[num];d=Document(dp)
for i,p in enumerate(d.paragraphs):
    if p.style.name=='图题' and p.text.startswith('图'+num+' '):
        rid=d.paragraphs[i-1]._p.xpath('.//a:blip')[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed');part=str(d.part.related_parts[rid].partname).lstrip('/');break
else:raise AssertionError('未定位目标图媒体')
blob=Path(r['最终图片路径']).read_bytes();tmp=dp.with_name('同尺寸图像替换处理中.docx')
with zipfile.ZipFile(dp) as src,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as dest:
    for z in src.infolist():dest.writestr(z,blob if z.filename==part else src.read(z.filename))
with zipfile.ZipFile(dp) as a,zipfile.ZipFile(tmp) as b:
    unchanged=all(a.read(n)==b.read(n) for n in a.namelist() if n!=part);assert unchanged
tmp.replace(dp)
manifest=pd.read_csv(OUT/'最终图件清单.csv');page_no=int(manifest[manifest['新图号']=='图'+num]['PDF实际页'].iloc[0]);pp=dp.with_suffix('.pdf');pdf=fitz.open(pp);before=[p.get_text() for p in pdf];page=pdf[page_no-1];images=page.get_image_info(xrefs=True);assert len(images)==1
page.replace_image(images[0]['xref'],stream=blob);tmp=pp.with_name('同尺寸图像替换处理中.pdf');pdf.save(tmp,garbage=3,deflate=True);pdf.close();check=fitz.open(tmp);assert [p.get_text() for p in check]==before;check.close();tmp.replace(pp)
r['是否插入Word']=True;r['PDF实际页']=page_no;(OUT/'05_数据核验/图件登记.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
f=OUT/'05_数据核验/PDF原生图像保真核验.json'
if f.exists():
    meta=json.loads(f.read_text(encoding='utf-8'))
    for item in meta['图像对象']:
        if item['页']==page_no:item['原生图像SHA256']=sha(r['最终图片路径'])
    f.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'00_任务记录'/('同尺寸图像更新记录_图'+num+'.json')).write_text(json.dumps({'图号':'图'+num,'唯一替换DOCX媒体':part,'其他全部ZIP成员逐字节一致':True,'PDF文字与分页逐页一致':True,'图像框未变':True,'图片SHA256':sha(r['最终图片路径']),'新DOCX_SHA256':sha(dp),'新PDF_SHA256':sha(pp)},ensure_ascii=False,indent=2),encoding='utf-8')
print('图'+num+'同尺寸原生图像更新完成')
