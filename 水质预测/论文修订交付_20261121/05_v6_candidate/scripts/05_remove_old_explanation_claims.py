"""清除V6候选稿中与旧版解释图相连的剩余正文断言。"""
from pathlib import Path
import re
from docx import Document
import pandas as pd
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT

OUT=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parents[3]
path=OUT/'水质预测硕士学位论文_V6_候选稿_解释待核验.docx'
temp=OUT/'水质预测硕士学位论文_V6_候选稿_文字校正.docx'
assert path.is_file() and not temp.exists()
d=Document(path)
changed=[]
meta=pd.read_csv(ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115/正式77项特征定义与来源总表.csv')
assert len(meta)==77
pending='【V6候选稿待核验】修复77特征模型的解释预测函数尚未通过跨重建数值一致性审查；旧版结果已隔离，本节待重算后填写。'
note='注：本图为新版本解释结果的图号占位，不提供科学数值；旧版图像未用于WQ77_REPAIRED模型解释。'

def replace(p,new):
    if p.text!=new:changed.append((p.text[:80],new[:80]));p.text=new

def heading_pos(s):
    hits=[i for i,p in enumerate(d.paragraphs) if p.style.name.startswith('Heading') and p.text.startswith(s)]
    assert len(hits)==1,(s,hits)
    return hits[0]

for start,end in [('3.2.4 ','3.3 '),('3.3.2 ','3.4 '),('3.4.1 ','3.4.2 ')]:
    a,b=heading_pos(start),heading_pos(end)
    first=True
    for p in d.paragraphs[a+1:b]:
        if p.style.name!='Normal' or not p.text.strip():continue
        if p.text.startswith('注：'):replace(p,note)
        elif first:replace(p,pending);first=False
        else:replace(p,'')

for prefix in ['3.4.2 ','3.5 ']:
    a=heading_pos(prefix)
    b=heading_pos('3.5 ') if prefix=='3.4.2 ' else heading_pos('4.1 ')
    first=True
    for p in d.paragraphs[a+1:b]:
        if p.style.name=='Normal' and p.text.strip():
            replace(p,pending if first else '');first=False

for p in d.paragraphs:
    if p.style.name=='Normal' and p.text.startswith('注：前3变量按每任务三个预登记核心样本'):
        replace(p,'注：表3-1仅保留预登记独立样本数；主要变量及平均绝对SHAP贡献等待新版本数值核验。')
    if p.style.name=='Normal' and p.text.startswith('图C-1至图C-10'):
        replace(p,'附录C原局部SHAP图已隔离。新候选模型的30个核心样本和seed43重复探针待预测函数一致性核验后重算；现有图号仅为结构占位。')
    if p.style.name=='Normal' and p.text.startswith('图D-1至图D-12'):
        replace(p,'附录D原XGBoost解释图仅对应旧版模型，已从本候选稿科学结果中隔离；新候选版XGBoost解释待核验，不得与TabICLv2解释混同。')
    if p.style.name=='Normal' and p.text.startswith('图D-11的SHAP来源份额'):
        replace(p,'图D-11与图D-12的新候选版解释待核验；旧标准化值和排名不作为新模型结果。')
    if p.style.name=='Normal' and p.text.startswith('附录E的图E-1至图E-20'):
        replace(p,'附录E保留20条正式一维ALE曲线图号；新候选版的局部效应及实际分箱待模型预测函数一致性核验后填入。')
    if p.style.name=='Normal' and p.text.startswith('附录F的图F-1至图F-10'):
        replace(p,'附录F保留十任务SHAP重要性图号；新候选版30核心样本归因待数值核验，不沿用旧版Top10。')
    if p.style.name=='Normal' and p.text.startswith('图H-1记录九任务固定查询重复范围'):
        replace(p,'图H-1与图H-2原审查图属于旧版预测函数，已隔离；图H-3仅保留历史验证协议对照，不替代新候选版正式十任务结果。')

# 74张占位图之后的旧图注不可被误读为WQ77新结果。
pending_re=re.compile(r'^图(?:3-(?:[3-7]|1[2-6])|[CDEF]-\d+|H-[12])\s')
for i,p in enumerate(d.paragraphs):
    if p.style.name=='图题' and pending_re.match(p.text):
        if i+1<len(d.paragraphs):
            nxt=d.paragraphs[i+1]
            if nxt.style.name=='Normal' and nxt.text.startswith('注：'):replace(nxt,note)
    if p.style.name=='表题' and p.text.startswith('表3-1 '):
        for nxt in d.paragraphs[i+1:i+5]:
            if nxt.style.name=='Normal' and nxt.text.startswith('注：'):
                replace(nxt,'注：新候选版SHAP主要变量及贡献数值待核验；每任务仍保留三个预登记独立核心样本，不将seed43重复探针计为独立样本。')
                break

# 附录G的77项完整定义与候选正式元数据同步，保留原字段顺序和类别。
g=[t for t in d.tables if len(t.rows)==78 and t.rows[0].cells[0].text=='编号']
assert len(g)==1
for i,m in enumerate(meta.itertuples(),1):
    row=g[0].rows[i]
    assert row.cells[1].text.strip()==m.feature,(i,row.cells[1].text,m.feature)
    row.cells[4].text=str(m.单位)
    row.cells[5].text=str(m.来源组)
    row.cells[6].text=str(m.定义)
widths=[1.0,3.1,3.5,1.1,2.3,2.8,10.8]
g[0].autofit=False
for col,w in zip(g[0].columns,widths):col.width=Cm(w)
for row in g[0].rows:
    for cell,w in zip(row.cells,widths):
        cell.width=Cm(w);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in cell.paragraphs:
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.02
            for run in p.runs:run.font.size=Pt(8.3)

d.save(temp)
print('corrected',len(changed),'paragraphs',temp)
