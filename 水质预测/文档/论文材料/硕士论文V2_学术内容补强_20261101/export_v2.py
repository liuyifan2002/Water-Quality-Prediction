"""更新V2目录域并导出PDF；不接触V1母本。"""
from pathlib import Path
import win32com.client

OUT = Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文V2_学术内容补强_20261101")
DOCX = OUT / "水质预测硕士学位论文_学术内容补强V2_核验交付.docx"
PDF = OUT / "水质预测硕士学位论文_学术内容补强V2_核验交付.pdf"

word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
doc = None
try:
    doc = word.Documents.Open(str(DOCX), ReadOnly=False, AddToRecentFiles=False, ConfirmConversions=False)
    doc.Repaginate()
    for i in range(1, doc.TablesOfContents.Count + 1):
        doc.TablesOfContents(i).Update()
    for i in range(1, doc.TablesOfFigures.Count + 1):
        doc.TablesOfFigures(i).Update()
    doc.Fields.Update()
    doc.Repaginate()
    doc.Save()
    doc.ExportAsFixedFormat(str(PDF), 17, OpenAfterExport=False, OptimizeFor=0,
                            Item=0, IncludeDocProps=True, KeepIRM=True, CreateBookmarks=1)
    print("V2_EXPORT_OK", "pages", doc.ComputeStatistics(2), "toc", doc.TablesOfContents.Count,
          "figures", doc.TablesOfFigures.Count, PDF)
finally:
    if doc is not None:
        doc.Close(SaveChanges=False)
    word.Quit()
