"""核对V6候选稿的冻结数值、图像嵌入、PDF页与未完成解释状态。"""
from pathlib import Path
import csv,hashlib,re,json
from collections import Counter
import fitz
import pandas as pd
from docx import Document
from docx.table import Table
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
SRC=ROOT/'论文修订交付_20261121/04_v5_1_correction/水质预测硕士学位论文_V5.1.docx'
NEW=OUT/'水质预测硕士学位论文_V6_候选稿_解释待核验.docx'
PDF=OUT/'水质预测硕士学位论文_V6_候选稿_解释待核验.pdf'
assert SRC.is_file() and NEW.is_file() and PDF.is_file()
old=Document(SRC);new=Document(NEW);pdf=fitz.open(PDF)
assert len(new.inline_shapes)==94 and len(new.tables)==13 and len(pdf)>80
core=pd.read_csv(OUT/'data/正式十任务指标.csv').set_index('task')
modern=pd.read_csv(OUT/'data/四模型同协议指标.csv')
ab=pd.read_csv(OUT/'data/四任务六来源消融.csv')
def capt(d,no):
    v=[p for p in d.paragraphs if p.style.name in ('图题','表题') and re.match(rf'^{re.escape(no)}(?:\s|$)',p.text)]
    assert len(v)==1,(no,len(v));return v[0]
def table(d,no):
    n=capt(d,no)._p.getnext();assert n.tag==qn('w:tbl');return Table(n,d._body)
for no in ['表4-1','表4-2','表4-3','表5-1']:
    assert len(table(old,no).rows)==len(table(new,no).rows)
assert len([p for p in new.paragraphs if p.style.name=='图题'])==94
assert len([p for p in new.paragraphs if p.style.name=='表题'])==12
assert sum('（新版本解释待核验）' in p.text for p in new.paragraphs if p.style.name=='图题')==74
for i,task in enumerate(core.index,1):
    r=core.loc[task];a=table(new,'表4-2').rows[i].cells
    assert a[0].text==r['任务'] and abs(float(a[1].text)-r.XGBoost_R2)<.0006 and abs(float(a[2].text)-r.TabICLv2_R2)<.0006
    assert abs(float(a[3].text)-r.delta_R2)<.0006 and a[9].text=='待复核'
    assert abs(float(a[4].text)-r.XGBoost_RMSE)<.00006 and abs(float(a[5].text)-r.TabICLv2_RMSE)<.00006
for i,r in enumerate(modern.itertuples(),1):
    a=table(new,'表4-3').rows[i].cells
    assert a[0].text==r.任务 and a[1].text=={'xgboost':'XGBoost','tabiclv2':'TabICLv2','tabpfn':'TabPFN','tabm':'TabM'}[r.model]
    assert abs(float(a[2].text)-r.pooled_R2)<.0006

def sha(b):return hashlib.sha256(b).hexdigest()
embedded={sha(part.blob) for part in new.part.related_parts.values() if getattr(part,'content_type','').startswith('image/')}
figure_files=sorted((OUT/'figures').glob('图*.png'))
assert len(figure_files)==7 and all(sha(p.read_bytes()) in embedded for p in figure_files)
placeholder=OUT/'figures/解释图待数值核验_占位.png';assert sha(placeholder.read_bytes()) in embedded

body='\n'.join(p.text for p in new.paragraphs if not p.style.name.startswith('toc'))
for obsolete in ['十项任务的pooled R²均高于XGBoost','0.589、0.602','从−0.127升至0.247','本图九个分层任务均为正','0.098、+0.083、+0.097']:
    assert obsolete not in body,obsolete
assert 'WQ77_REPAIRED_20261115_V1' in body and 'TN中层' in body
flat=[re.sub(r'\s+','',p.get_text()) for p in pdf]
for x in ['表4-2TabICLv2相对XGBoost的十任务比较','图4-2不同指标与水层的模型提升差异','图5-3四个典型站点的输入覆盖与预测误差诊断']:
    assert any(x in y for y in flat),x
land=[i+1 for i,p in enumerate(pdf) if p.rect.width>p.rect.height]
assert any('表4-2' in flat[i-1] for i in land),land

# V5.1旧表与新表逐字段对照；新值为WQ77候选版冻结汇总。
oldrows=table(old,'表4-2').rows
cols=[('XGBoost_R2',1,'pooled R²'),('TabICLv2_R2',2,'pooled R²'),('delta_R2',3,'ΔR²'),('XGBoost_RMSE',4,'RMSE'),('TabICLv2_RMSE',5,'RMSE'),('XGBoost_MAE',6,'MAE'),('TabICLv2_MAE',7,'MAE'),('station_win_rate',8,'站点RMSE胜率')]
records=[]
for i,task in enumerate(core.index,1):
    r=core.loc[task]
    for col,j,metric in cols:
        oldv=oldrows[i].cells[j].text
        newv=f'{r[col]*100:.1f}%' if col=='station_win_rate' else f'{r[col]:.6f}'
        records.append({'任务':r['任务'],'字段':col,'指标':metric,'旧版V5.1':oldv,'新候选V6':newv,'来源文件':str((OUT/'data/正式十任务指标.csv').relative_to(ROOT)),'论文位置':'表4-2；第4章'} )
pd.DataFrame(records).to_csv(OUT/'新旧实验结果替换对照表.csv',index=False,encoding='utf-8-sig')

# 同一份对照表继续列出现代四模型、来源消融及历史78特征分区的逐项替换。
oldmodern=table(old,'表4-3')
for i,r in enumerate(modern.itertuples(),1):
    before=oldmodern.rows[i].cells
    for field,j in [('pooled_R2',2),('pooled_RMSE',3),('pooled_MAE',4)]:
        records.append({'任务':r.任务+' · '+{'xgboost':'XGBoost','tabiclv2':'TabICLv2','tabpfn':'TabPFN','tabm':'TabM'}[r.model],'字段':field,'指标':field,'旧版V5.1':before[j].text,'新候选V6':f'{getattr(r,field):.6f}','来源文件':str((OUT/'data/四模型同协议指标.csv').relative_to(ROOT)),'论文位置':'表4-3；图4-5；第4章'})
oldab=table(old,'表5-1')
newab=table(new,'表5-1')
for i in range(1,25):
    for field,j in [('delta_R2',2),('delta_RMSE',3),('站点RMSE改善比例',4)]:
        records.append({'任务':newab.rows[i].cells[0].text+' · 删除'+newab.rows[i].cells[1].text,'字段':field,'指标':field,'旧版V5.1':oldab.rows[i].cells[j].text,'新候选V6':newab.rows[i].cells[j].text,'来源文件':str((OUT/'data/四任务六来源消融.csv').relative_to(ROOT)),'论文位置':'表5-1；图5-2；第5章'})
oldreg=ROOT/'transfer/llm_extension_transfer_v1/results/regionalization/分海区五折留站验证_20260906/五折留站点验证_汇总.csv'
newreg=OUT/'data/历史78特征分区汇总.csv'
if oldreg.is_file():
    a=pd.read_csv(oldreg).set_index(['target','depth']);b=pd.read_csv(newreg).set_index(['target','depth'])
    for key in b.index:
        records.append({'任务':str(key),'字段':'delta_r2_pooled','指标':'历史78特征分区pooled ΔR²','旧版V5.1':f'{a.loc[key,"delta_r2_pooled"]:.6f}' if key in a.index else 'NA','新候选V6':f'{b.loc[key,"delta_r2_pooled"]:.6f}','来源文件':str(newreg.relative_to(ROOT)),'论文位置':'图5-1；第5章5.2'})
pd.DataFrame(records).to_csv(OUT/'新旧实验结果替换对照表.csv',index=False,encoding='utf-8-sig')

pending={f'图3-{i}' for i in list(range(3,8))+list(range(12,17))}
pending|={f'图{letter}-{i}' for letter,n in [('C',20),('D',12),('E',20),('F',10),('H',2)] for i in range(1,n+1)}
updates=[]
for p in new.paragraphs:
    if p.style.name=='图题':
        no=p.text.split()[0];kind='新候选版重绘' if no in ['图4-1','图4-2','图4-4','图4-5','图5-1','图5-2','图5-3'] else '解释待核验占位' if no in pending else '保留历史或概念图'
        updates.append({'图表号':no,'图表名':p.text,'处理':kind,'论文状态':'候选，不作为最终解释' if no in pending else '候选版可核验','数据版本':'WQ77_REPAIRED_20261115_V1' if kind=='新候选版重绘' else '旧版隔离' if no in pending else '原登记历史/概念'})
for p in new.paragraphs:
    if p.style.name=='表题':
        no=p.text.split()[0];kind='新候选版更新' if no in ('表4-1','表4-2','表4-3','表5-1') else '候选版元数据更新' if no=='表2-3' else '解释待核验占位' if no=='表3-1' else '保留'
        updates.append({'图表号':no,'图表名':p.text,'处理':kind,'论文状态':'候选，解释待核验' if no=='表3-1' else '候选版','数据版本':'WQ77_REPAIRED_20261115_V1' if kind!='保留' else '原登记'})
pd.DataFrame(updates).to_csv(OUT/'图表更新清单.csv',index=False,encoding='utf-8-sig')

qa=OUT/'排版核查';qa.mkdir(exist_ok=True)
for n in [25,37,52,54,56,57,60,62,64]:
    if n<=len(pdf):pdf[n-1].get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(qa/f'终稿第{n}页.png')
for start in range(0,len(pdf),18):
    pages=range(start,min(start+18,len(pdf)))
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(900,((len(pages)+2)//3)*420),'white');draw=ImageDraw.Draw(sheet)
    for j,i in enumerate(pages):
        pix=pdf[i].get_pixmap(matrix=fitz.Matrix(.35,.35),alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        im.thumbnail((275,375));x=(j%3)*300+(300-im.width)//2;y=(j//3)*420+20
        sheet.paste(im,(x,y));draw.text((x,y-15),str(i+1),fill='black')
    sheet.save(qa/f'全文联系表_{start+1:03d}-{min(start+18,len(pdf)):03d}.png')

status={'PDF页数':len(pdf),'内嵌图片':len(new.inline_shapes),'图题':94,'表题':12,'待解释核验占位图':74,'新候选版结果图':7,'横向页':land,'冻结表格匹配':'通过','新图嵌入哈希':'通过','PDF关键题名':'通过'}
(OUT/'核验状态.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(status,ensure_ascii=False))
