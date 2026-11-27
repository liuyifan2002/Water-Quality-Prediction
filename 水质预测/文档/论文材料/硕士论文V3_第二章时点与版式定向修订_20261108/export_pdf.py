"""用本机Word重开DOCX、更新域、重新分页并导出对应PDF。"""
from pathlib import Path
import sys
import win32com.client

base=Path(__file__).resolve().parent
docx=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else base/'水质预测硕士学位论文_V3_第二章定向修订_20261108.docx'
pdf=docx.with_suffix('.pdf')
app=win32com.client.DispatchEx('Word.Application')
app.Visible=False;app.DisplayAlerts=0
doc=None
try:
    doc=app.Documents.Open(str(docx),ReadOnly=False,AddToRecentFiles=False)
    for toc in doc.TablesOfContents:toc.Update()
    for tof in doc.TablesOfFigures:tof.Update()
    doc.Fields.Update()
    for section in doc.Sections:
        section.Headers(1).Range.Fields.Update()
        section.Footers(1).Range.Fields.Update()
    doc.Repaginate()
    pages=doc.ComputeStatistics(2)
    doc.Save()
    doc.ExportAsFixedFormat(str(pdf),17,OpenAfterExport=False)
    print('EXPORTED',str(pdf),'pages',pages,flush=True)
finally:
    if doc is not None:doc.Close(SaveChanges=False)
    app.Quit()
