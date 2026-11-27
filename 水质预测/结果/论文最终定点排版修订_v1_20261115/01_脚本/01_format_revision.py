"""论文定点格式修订：仅文字标识及表格排版，不触及图像或科研数据。"""
from pathlib import Path
from copy import deepcopy
import re,json,hashlib,zipfile
from collections import Counter
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT,WD_ROW_HEIGHT_RULE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'结果/论文最终图件重构_v1/07_论文整合稿/水质预测硕士学位论文_图件重构与终稿整合稿.docx'
TARGET=OUT/'水质预测硕士学位论文_最终定点排版修订版.docx'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=Document(SOURCE);doc=Document(SOURCE)
baseline={str(p):sha(p) for p in [SOURCE,SOURCE.with_suffix('.pdf'),ROOT/'文档/论文材料/硕士论文V4_题名更正_20261109/水质预测硕士学位论文_V4_题名更正.docx',ROOT/'文档/论文材料/硕士论文V4_题名更正_20261109/水质预测硕士学位论文_V4_题名更正.pdf']}
(OUT/'00_记录/原件保护哈希.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding='utf-8')
tasks={f'{a}_{b}':x+y for a,x in [('chla','Chl-a'),('tn','TN'),('tp','TP')] for b,y in [('surface','表层'),('middle','中层'),('bottom','底层')]};tasks['secchi']='Secchi'
models={'xgboost':'XGBoost','tabiclv2':'TabICLv2','tabpfn':'TabPFN','tabm':'TabM'}
groups={'METEOROLOGY':'气象','RAINFALL_HYDRO_TIDE':'降雨、水文与潮汐','OCEAN_PHYSICS':'海洋物理','HUMAN_ACTIVITY':'人类活动代理','TEMPORAL_PROCESS':'时间过程','SPATIAL_STATIC':'空间静态'}
shortgroups={'MET':'气象','RHT':'降雨、水文与潮汐','OCN':'海洋物理','HUM':'人类活动代理','TMP':'时间过程','SPA':'空间静态'}
features={r.cells[1].text:r.cells[2].text for r in source.tables[-1].rows[1:]};assert len(features)==77
log=[]
def replace_text(p,new,where):
    old=p.text
    if old==new:return
    assert not p._p.xpath('.//w:drawing')
    # 保持首个文字run的字体和段落格式。引用文献段落在调用端排除。
    if p.runs:
        p.runs[0].text=new
        for r in p.runs[1:]:r.text=''
    else:p.add_run(new)
    log.append({'位置':where,'原文':old,'修订':new})

def translate(s,feature_names=False):
    for mapping in [tasks,models,groups,{'ROBUST':'稳健','MODERATE':'中等','IMPORTANT':'重要','NEUTRAL':'中性','station_meta':'站点元数据表','single_task':'单一任务'}]:
        for a,b in mapping.items():s=re.sub(r'(?<![A-Za-z0-9_])'+re.escape(a)+r'(?![A-Za-z0-9_])',lambda m:b,s)
    s=s.replace('四代表任务','四项代表任务').replace('四个代表任务','四项代表任务')
    s=re.sub(r'Secchi(?=(?:P1|TreeSHAP|XGBoost))','Secchi ',s)
    s=s.replace('Secchi P1一维ALE','Secchi P1 一维ALE')
    if feature_names:
        pat=r'(?<![A-Za-z0-9_])(?:'+'|'.join(re.escape(k) for k in sorted(features,key=len,reverse=True))+r')(?![A-Za-z0-9_])'
        s=re.sub(pat,lambda m:features[m[0]],s)
    return s

for i,p in enumerate(doc.paragraphs):
    if p.style.name=='参考文献' or p.style.name.lower().startswith('toc') or p._p.xpath('.//w:instrText'):continue
    s=translate(p.text)
    s=s.replace('任务后缀S/M/B分别表示表/中/底层；','任务名称按指标与水层统一列示；')
    replace_text(p,s,f'正文段落{i}')

for k,t in enumerate(doc.tables):
    for ri,row in enumerate(t.rows):
        for ci,c in enumerate(row.cells):
            # 2-3及G精确字段清单完整保留；历史候选表代表字段列也保留原标识。
            if k in [3,12] or (k==4 and ci==1):continue
            for p in c.paragraphs:
                s=translate(p.text,feature_names=(k in [2,7] or k==4 and ci!=1))
                if k==1:
                    s={'chla':'Chl-a','tn':'TN','tp':'TP','surface':'表层','middle':'中层','bottom':'底层'}.get(s,s)
                if k==7 and ci==0:
                    z=re.fullmatch(r'(Chl-a|TN|TP)-(S|M|B)',s)
                    if z:s=z[1]+{'S':'表层','M':'中层','B':'底层'}[z[2]]
                if k==11 and ci==1:s=shortgroups.get(s,s)
                s=s.replace('X_strict_77','正式77特征')
                replace_text(p,s,f'表格{k+1} 行{ri+1} 列{ci+1}')

def setting(parent,tag,value='1'):
    e=parent.find(qn('w:'+tag))
    if e is None:e=OxmlElement('w:'+tag);parent.append(e)
    e.set(qn('w:val'),value);return e

# 全部跨页表格重复完整第一行；任何数据行均禁止拆到两个页面。
for k,t in enumerate(doc.tables):
    for i,row in enumerate(t.rows):
        pr=row._tr.get_or_add_trPr();setting(pr,'cantSplit')
        h=pr.find(qn('w:tblHeader'))
        if i==0:setting(pr,'tblHeader')
        elif h is not None:pr.remove(h)
        if row.height_rule==WD_ROW_HEIGHT_RULE.EXACTLY:row.height_rule=WD_ROW_HEIGHT_RULE.AT_LEAST
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_together=True
                p.paragraph_format.keep_with_next=(i==0)

# 结果表按有效宽15.5cm重分配列宽，维持9pt或原8.5pt，不用缩字解决挤断。
widths={7:[2.3,8.4,2.0,2.8],8:[3.5,4,4,4],9:[3.1,2.4,2.7,1.7,3.5,2.1],10:[3.1,3.2,3.1,3.05,3.05],11:[2.3,3.8,1.5,2.2,3.6,2.1]}
for k,ws in widths.items():
    t=doc.tables[k];assert abs(sum(ws)-15.5)<1e-8;t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
    tw=t._tbl.tblPr.find(qn('w:tblW'));tw.set(qn('w:type'),'dxa');tw.set(qn('w:w'),str(round(15.5/2.54*1440)))
    for col,w in zip(t.columns,ws):col.width=Cm(w)
    for ri,row in enumerate(t.rows):
        row.height=Cm(.45);row.height_rule=WD_ROW_HEIGHT_RULE.AT_LEAST
        for ci,(cell,w) in enumerate(zip(row.cells,ws)):
            cell.width=Cm(w);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                pf=p.paragraph_format;pf.space_before=Pt(1);pf.space_after=Pt(1);pf.line_spacing=1.05;pf.first_line_indent=Pt(0)
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if k==7 and ci==1 else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size=Pt(8.5 if k==11 else 9)
                    if ri==0:r.bold=True
    # 第一列简洁化后，表4-2不再需要英文等级的强制断字。

replace_text(doc.tables[9].cell(0,1).paragraphs[0],'XGBoost R²','表4-2表头')
# 在不缩字的前提下收紧两张相邻结果表的行内留白，避免末行孤悬。
for k in [8,9]:
    for row in doc.tables[k].rows:
        row.height=Cm(.42)
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_before=Pt(0)
                p.paragraph_format.space_after=Pt(0)
# 清除无文字、无图片、无分节信息的空一级标题，避免继承标题分页属性。
for p in list(doc.paragraphs):
    if p.style.name=='Heading 1' and not p.text.strip() and not p._p.xpath('.//w:drawing | .//w:sectPr'):
        log.append({'位置':'空一级标题','原文':'空段落继承一级标题分页属性','修订':'移除空段落；前后正文不变'})
        p._p.getparent().remove(p._p)
# 原整合稿的4.5正文误落在4.6小结之后；整段归位，不改写任何字句。
h45=next(p for p in doc.paragraphs if p.style.name=='Heading 2' and p.text.startswith('4.5 '))
h46=next(p for p in doc.paragraphs if p.style.name=='Heading 2' and p.text.startswith('4.6 '))
limits=next(p for p in doc.paragraphs if p.text.startswith('TabICLv2在十项正式站点留出任务上均提高pooled R²'))
assert h45._p.getnext()==h46._p
h46._p.addprevious(limits._p)
log.append({'位置':'4.5正文归位','原文':limits.text,'修订':'同一段落移至4.5标题后，文字、数值及结论不变'})
for p in [h45,h46]:
    p.paragraph_format.space_before=Pt(8)
    p.paragraph_format.space_after=Pt(6)
# 仅收紧结尾相邻标题的段间距，防止完整结语独占一页；不缩小正文或图片。
for p in doc.paragraphs:
    if p.style.name=='Heading 2' and re.match(r'^6\.[123] ',p.text):
        p.paragraph_format.space_before=Pt(9)
        p.paragraph_format.space_after=Pt(8)
for p in doc.paragraphs:
    if p.style.name in ['图题','表题']:p.paragraph_format.keep_with_next=True;p.paragraph_format.keep_together=True

def numeric_cells(d):
    return {(ti,ri,ci):c.text for ti,t in enumerate(d.tables) for ri,r in enumerate(t.rows) for ci,c in enumerate(r.cells) if re.fullmatch(r'[+−-]?\d+(?:\.\d+)?%?',c.text.strip())}
assert numeric_cells(source)==numeric_cells(doc)
assert [p.text for p in source.paragraphs if p.style.name=='参考文献']==[p.text for p in doc.paragraphs if p.style.name=='参考文献']
source_nums=[re.match(r'^图([A-Z0-9]+-\d+)',p.text)[1] for p in source.paragraphs if p.style.name=='图题']
new_nums=[re.match(r'^图([A-Z0-9]+-\d+)',p.text)[1] for p in doc.paragraphs if p.style.name=='图题'];assert new_nums==source_nums and len(new_nums)==94
assert len(doc.tables)==len(source.tables)==13
assert [[r.cells[1].text for r in doc.tables[k].rows[1:]] for k in [3,12]]==[[r.cells[1].text for r in source.tables[k].rows[1:]] for k in [3,12]]
doc.save(TARGET)
def media(p):
    with zipfile.ZipFile(p) as z:return sorted(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/'))
assert media(SOURCE)==media(TARGET)
(OUT/'00_记录/逐项格式修改.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'00_记录/修订前后内容保护核对.json').write_text(json.dumps({'数值单元格逐项一致':True,'所有图像媒体SHA256一致':True,'全部94图编号及顺序一致':True,'13张表格完整保留':True,'正式字段及候选字段清单未改':True,'参考文献逐段一致':True,'文字替换记录数':len(log),'原件保护':baseline},ensure_ascii=False,indent=2),encoding='utf-8')
print(TARGET,flush=True)
