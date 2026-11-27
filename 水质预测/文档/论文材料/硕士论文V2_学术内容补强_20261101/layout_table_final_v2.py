"""仅修复新增表3-3的断行显示，不修改任何科学数值。"""
from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文V2_学术内容补强_20261101")
source = ROOT / "水质预测硕士学位论文_学术内容补强V2_排版精修版.docx"
target = ROOT / "水质预测硕士学位论文_学术内容补强V2_最终交付.docx"
assert source.is_file() and not target.exists()
d = Document(source)
t = next(t for t in d.tables if len(t.rows) == 11 and "主要贡献变量" in t.rows[0].cells[1].text)
assert len(t.columns) == 4
widths = [Cm(3.0), Cm(7.0), Cm(1.9), Cm(3.1)]
names = {
    "Chl-a表层": "Chl-a-S", "Chl-a中层": "Chl-a-M", "Chl-a底层": "Chl-a-B",
    "TN表层": "TN-S", "TN中层": "TN-M", "TN底层": "TN-B",
    "TP表层": "TP-S", "TP中层": "TP-M", "TP底层": "TP-B",
    "透明度": "Secchi",
}
t.autofit = False
for col, width in zip(t.columns, widths):
    col.width = width
for i, row in enumerate(t.rows):
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))
    if i == 0 and tr_pr.find(qn("w:tblHeader")) is None:
        tr_pr.append(OxmlElement("w:tblHeader"))
    if i:
        row.cells[0].text = names.get(row.cells[0].text.strip(), row.cells[0].text.strip())
        row.cells[3].text = "P1·3例"
    for cell, width in zip(row.cells, widths):
        cell.width = width
        for p in cell.paragraphs:
            p.paragraph_format.space_before = 0
            p.paragraph_format.space_after = 0
            p.paragraph_format.line_spacing = 1.0
            for run in p.runs:
                run.font.size = Pt(9)
for p in d.paragraphs:
    if p.text.startswith("注：前3变量按冻结核心样本"):
        p.text = ("注：任务后缀S/M/B分别表示表/中/底层；前3变量按冻结核心样本的平均绝对SHAP值排序。"
                  "来源组：M=气象，R=降雨/水文/潮汐，O=海洋物理，H=人类活动代理，T=时间过程，S=空间静态。"
                  "P1为第1折单查询协议，每任务3个预登记核心样本；不同任务的原始贡献数值不直接比较。")
        break
d.save(target)
print(target)
