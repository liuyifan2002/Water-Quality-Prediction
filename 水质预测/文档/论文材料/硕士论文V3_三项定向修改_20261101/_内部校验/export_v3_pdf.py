"""使用本机 Word 更新目录、交叉引用域并导出 V3 PDF。"""
from pathlib import Path
import sys
import time
import win32com.client

base = Path(__file__).resolve().parent
stem = "水质预测硕士学位论文_V3_三项定向修改"
suffix = sys.argv[1] if len(sys.argv) > 1 else ""
source = base / f"{stem}{suffix}.docx"
pdf = base / f"{stem}{suffix}.pdf"
app = win32com.client.DispatchEx("Word.Application")
app.Visible = False
app.DisplayAlerts = 0
doc = None
try:
    doc = app.Documents.Open(str(source), ReadOnly=False, AddToRecentFiles=False)
    for toc in doc.TablesOfContents:
        toc.Update()
    for tof in doc.TablesOfFigures:
        tof.Update()
    doc.Fields.Update()
    for section in doc.Sections:
        section.Headers(1).Range.Fields.Update()
        section.Footers(1).Range.Fields.Update()
    doc.Repaginate()
    pages = doc.ComputeStatistics(2)
    doc.Save()
    doc.ExportAsFixedFormat(str(pdf), 17, OpenAfterExport=False)
    print("EXPORTED", source, pdf, "PAGES", pages, flush=True)
finally:
    if doc is not None:
        doc.Close(SaveChanges=False)
    app.Quit()
