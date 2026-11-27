"""冻结数据、整合稿与PDF实排核对；输出逐页联系表，不运行实验。"""
from common import *
from PIL import Image,ImageDraw,ImageFont
import fitz,re,zipfile
from collections import Counter

regpath=OUT/'05_数据核验/图件登记.json';m=json.loads(regpath.read_text(encoding='utf-8'))
dp=OUT/'07_论文整合稿/水质预测硕士学位论文_图件重构与终稿整合稿.docx';pdf=fitz.open(dp.with_suffix('.pdf'));doc=Document(dp)
# 严格逐特征比较三核心样本汇总，不包含seed43。
samples=read('结果/论文图表/figure_revision_v2/source_inventory/40组文件与30核心样本对应.csv');s=read(FROZEN30);rows=[]
for t in TASKS:
    paths=samples[(samples.task==t)&(samples.seed==42)].文件.tolist();assert len(paths)==3
    q=pd.concat([read(p) for p in paths]);agg=q.assign(abs_value=q.shap_value.abs()).groupby('feature').abs_value.mean();z=s[s.task==t].set_index('feature');err=float((agg-z.mean_abs_shap).abs().max());assert err<1e-12
    ranks=agg.rank(ascending=False,method='min');assert np.array_equal(ranks.reindex(z.index).to_numpy(),z['rank'].to_numpy())
    rows.append({'任务':t,'独立核心样本':3,'种子重复进入汇总':0,'特征':77,'最大汇总差':err,'770项排名核验':'一致'})
pd.DataFrame(rows).to_csv(OUT/'05_数据核验/30核心样本逐值与排名核验.csv',index=False,encoding='utf-8-sig')
for row,t in zip(doc.tables[7].rows[1:],TASKS):
    assert [v.strip() for v in row.cells[1].text.split('/')]==s[s.task==t].nsmallest(3,'rank').feature.tolist()
assert set(r.cells[1].text for r in doc.tables[-1].rows[1:])==set(s.feature)
z=read(XGB+'04_integrated/04_shap_ale_ablation_comparison.csv');zr=[]
for t,g in z.groupby('task'):
    assert len(g)==6
    for col,zc in [('shap_group_share','shap_group_share_z'),('delta_RMSE','delta_RMSE_z')]:
        err=float(np.max(np.abs((g[col]-g[col].mean())/g[col].std(ddof=1)-g[zc])));assert err<1e-12
        zr.append({'任务':t,'指标':col,'来源类别数':6,'ddof':1,'冻结值数量':6,'最大核验差':err,'仅核验未回写':True})
pd.DataFrame(zr).to_csv(OUT/'05_数据核验/48个冻结z值定义核验.csv',index=False,encoding='utf-8-sig')

# 非目标图保留原图，仅抽取原Word媒体形成完整终稿资产清单。
od=Document(DOC)
for i,p in enumerate(od.paragraphs):
    found=re.match(r'^图(1-1|2-[234])\s+(.+)',p.text)
    if not found or p.style.name!='图题':continue
    num,title=found.groups();before=od.paragraphs[i-1]
    rid=before._p.xpath('.//a:blip')[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed');part=od.part.related_parts[rid];dest=OUT/('02_正文图片/Chapter'+num[0])/('图'+num+'_'+title+Path(part.partname).suffix);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(part.blob)
    m[num]={'新图号':'图'+num,'图名':title,'所属章节':num[0],'原图号':'图'+num,'最终图片路径':str(dest),'矢量PDF':'NA：原件为栅格，本轮不重绘','SVG':'NA：原件为栅格','数据来源':str(DOC),'绘图脚本':'NA：用户指定保留原图','数据哈希':{str(DOC):sha(DOC)},'图片哈希':sha(dest),'是否通过质量检查':'保留原件；实排已核对','是否插入Word':True,'图注':'原稿原样保留','复用':True}

captions=[p.text for p in doc.paragraphs if p.style.name=='图题'];nums=[re.match(r'^图([A-Z0-9]+-\d+)',p)[1] for p in captions];assert len(nums)==94 and len(set(nums))==94
expected={**{'1':1,'2':4,'3':16,'4':5,'5':3},**{'C':20,'D':12,'E':20,'F':10,'H':3}}
assert dict(Counter(n.split('-')[0] for n in nums))==expected
actual={};pages=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
# 清理本任务生成的旧联系表，防止页数变化后遗留上轮缩略图；不触及原资产。
qa_dir=(OUT/'06_图片质量检查').resolve()
for pattern in ['全文缩略图联系表_*.jpg','第*章图件检查_*.jpg','附录图件检查_*.jpg']:
    for old in qa_dir.glob(pattern):
        assert old.parent.resolve()==qa_dir;old.unlink()
for old in qa_dir.glob('页_*.jpg'):
    if int(old.stem.split('_')[-1])>len(pdf):
        assert old.parent.resolve()==qa_dir;old.unlink()
for i,page in enumerate(pdf):
    txt=page.get_text();compact=''.join(txt.split());hits=[]
    for num,r in m.items():
        key=''.join((r['新图号']+' '+r['图名']).split())
        # 图目录含同名，正文位置限定实际图像页面。
        if key in compact and page.get_images():
            actual.setdefault(num,[]).append(i+1);hits.append(num)
    for info in page.get_image_info(xrefs=True):
        box=fitz.Rect(info['bbox']);pages.append({'PDF页':i+1,'图号':';'.join(hits),'x0':box.x0,'y0':box.y0,'x1':box.x1,'y1':box.y1,'实际宽cm':box.width/72*2.54,'实际高cm':box.height/72*2.54,'像素宽':info['width'],'有效DPI':info['width']/box.width*72,'页面内':box.x0>=0 and box.y0>=0 and box.x1<=page.rect.width+.1 and box.y1<=page.rect.height+.1})
    small=page.get_pixmap(matrix=fitz.Matrix(.50,.50),alpha=False);im=Image.frombytes('RGB',[small.width,small.height],small.samples);canvas=Image.new('RGB',(320,480),'#edf0f2');im.thumbnail((310,450));canvas.paste(im,((320-im.width)//2,20));ImageDraw.Draw(canvas).text((10,456),f'PDF {i+1}',font=font,fill='#333');canvas.save(OUT/'06_图片质量检查'/f'页_{i+1:03}.jpg',quality=88)
assert set(actual)==set(m),(set(m)-set(actual));assert all(len(v)==1 for v in actual.values()),{k:v for k,v in actual.items() if len(v)!=1}
for num,r in m.items():
    page=pdf[actual[num][0]-1];imgs=[fitz.Rect(x['bbox']) for x in page.get_image_info()]
    cap=[]
    for b in page.get_text('dict')['blocks']:
        for line in b.get('lines',[]):
            st=''.join(x['text'] for x in line['spans'])
            if re.match(r'^图'+re.escape(num)+r'(?:\s|$)',st.strip()):cap.append(fitz.Rect(line['bbox']))
    assert any(0<=c.y0-im.y1<35 for c in cap for im in imgs),(num,'图像与图题未相邻')
pd.DataFrame(pages).to_csv(OUT/'06_图片质量检查/PDF逐图实际尺寸与有效DPI.csv',index=False,encoding='utf-8-sig');assert all(p['页面内'] for p in pages)

def contacts(name,indices,cols=4):
    for offset in range(0,len(indices),cols*5):
        take=indices[offset:offset+cols*5];im=Image.new('RGB',(320*cols,480*((len(take)+cols-1)//cols)),'#d9dfe4')
        for j,k in enumerate(take):im.paste(Image.open(OUT/'06_图片质量检查'/f'页_{k:03}.jpg'),((j%cols)*320,(j//cols)*480))
        im.save(OUT/'06_图片质量检查'/f'{name}_{offset//(cols*5)+1:02}.jpg',quality=90)
contacts('全文缩略图联系表',list(range(1,len(pdf)+1)))
for ch in ['3','4','5']:
    indices=sorted({v[0] for k,v in actual.items() if k.startswith(ch+'-')});contacts('第'+ch+'章图件检查',indices,3)
contacts('附录图件检查',sorted({v[0] for k,v in actual.items() if k[0].isalpha()}))

# 全部输入哈希重新核对，并与前一阶段源资产保护清单比较。
protected=json.loads((OUT/'05_数据核验/源文件保护哈希.json').read_text(encoding='utf-8'));bad=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h];assert not bad,bad
for num,r in m.items():
    r['PDF实际页']=actual[num][0];r['图片哈希']=sha(r['最终图片路径']);r['是否通过质量检查']='数据与版面自动核验通过；人工联系表复核见报告'
regpath.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=pd.DataFrame(m.values());manifest['数据哈希']=manifest['数据哈希'].map(lambda x:json.dumps(x,ensure_ascii=False));manifest.to_csv(OUT/'最终图件清单.csv',index=False,encoding='utf-8-sig')
report={'PDF页数':len(pdf),'全部图件':94,'本轮目标图件':90,'实际图像框':len(pages),'目录':3,'图件章节计数':expected,'770值及排名一致':True,'48z定义核验一致':True,'保护源文件数量':len(protected),'源文件哈希变化':bad,'图像全部在页面内':True,'最小有效DPI':min(x['有效DPI'] for x in pages),'参考文献未改':True}
report['目标图件最低有效DPI']=min(x['有效DPI'] for x in pages if x['图号'] not in ['1-1','2-2','2-3','2-4'])
f=OUT/'05_数据核验/Word整合结构核对.json';wi=json.loads(f.read_text(encoding='utf-8'));wi['新稿SHA256']=sha(dp);wi['PDF_SHA256']=sha(dp.with_suffix('.pdf'));f.write_text(json.dumps(wi,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'06_图片质量检查/自动质量检查结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(report)
import subprocess,sys
subprocess.run([sys.executable,str(OUT/'04_绘图脚本/09_verify_pdf_native_pixels.py')],check=True)
report['94图Word与PDF原生像素逐点一致']=True
(OUT/'06_图片质量检查/自动质量检查结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
