"""为表3-3设置跨页重复表头与禁止拆行；不更改表内科学数值。"""
from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

OUT=Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文V2_学术内容补强_20261101")
SOURCE=OUT/"水质预测硕士学位论文_学术内容补强V2_交付版.docx"
TARGET=OUT/"水质预测硕士学位论文_学术内容补强V2_排版精修版.docx"
assert SOURCE.is_file() and not TARGET.exists()
d=Document(SOURCE)
t=next(t for t in d.tables if len(t.rows)>0 and "主要贡献变量" in t.rows[0].cells[1].text)
task_short={"叶绿素a表层":"Chl-a表层","叶绿素a中层":"Chl-a中层","叶绿素a底层":"Chl-a底层"}
codes={"MET":"M","RHT":"R","OCN":"O","HUM":"H","TMP":"T","SPA":"S"}
t.autofit=False
widths=[2.5,8.3,1.8,2.4]
for col,width in zip(t.columns,widths):col.width=Cm(width)
for i,row in enumerate(t.rows):
    trPr=row._tr.get_or_add_trPr()
    if trPr.find(qn("w:cantSplit")) is None: trPr.append(OxmlElement("w:cantSplit"))
    if i==0 and trPr.find(qn("w:tblHeader")) is None: trPr.append(OxmlElement("w:tblHeader"))
    if i>0:
        row.cells[0].text=task_short.get(row.cells[0].text.strip(),row.cells[0].text.strip())
        row.cells[2].text="/".join(codes.get(x.strip(),x.strip()) for x in row.cells[2].text.split("/"))
    for c in row.cells:
        for p in c.paragraphs:
            p.paragraph_format.space_before=0
            p.paragraph_format.space_after=0
            p.paragraph_format.line_spacing=1.0
            for run in p.runs:run.font.size=Pt(9)
    for c,width in zip(row.cells,widths):c.width=Cm(width)
for p in d.paragraphs:
    if p.text.startswith("注：主要变量按已冻结3个核心样本"):
        p.text="注：前3变量按冻结核心样本的平均绝对SHAP值排序。来源组代码：M=气象，R=降雨/水文/潮汐，O=海洋物理，H=人类活动代理，T=时间过程，S=空间静态。P1为第1折单查询协议，每任务3个预登记核心样本；不同任务的原始贡献数值不直接比较。"
        break
d.save(TARGET)
print("TABLE_LAYOUT_FIXED",TARGET)
