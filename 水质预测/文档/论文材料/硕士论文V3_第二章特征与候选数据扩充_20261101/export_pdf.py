"""使用本机 Word 更新域、重分页并导出当前论文工作副本。"""
from pathlib import Path
import win32com.client

base = Path(__file__).resolve().parent
source = base / '水质预测硕士学位论文_V3_第二章特征与候选数据扩充_工作副本.docx'
pdf = source.with_suffix('.pdf')
app = win32com.client.DispatchEx('Word.Application')
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
    print('EXPORTED', pdf, 'PAGES', pages, flush=True)
finally:
    if doc is not None:
        doc.Close(SaveChanges=False)
    app.Quit()
