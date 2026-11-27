"""由Microsoft Word实际打开、重分页、刷新目录并导出PDF。"""
from pathlib import Path
import pythoncom
import win32com.client as win32

OUT=Path(__file__).resolve().parent
docx=OUT/'水质预测硕士学位论文_V4_结构与地图定向修订.docx'
pdf=OUT/'水质预测硕士学位论文_V4_结构与地图定向修订.pdf'
pythoncom.CoInitialize()
app=win32.DispatchEx('Word.Application')
app.Visible=False
app.DisplayAlerts=0
try:
    d=app.Documents.Open(str(docx),ReadOnly=False,AddToRecentFiles=False,ConfirmConversions=False)
    print('opened',d.Name,flush=True)
    d.Repaginate()
    for i in range(1,d.TablesOfContents.Count+1):
        d.TablesOfContents(i).Update()
    print('toc',d.TablesOfContents.Count,flush=True)
    for i in range(1,d.TablesOfFigures.Count+1):
        d.TablesOfFigures(i).Update()
    print('figtables',d.TablesOfFigures.Count,flush=True)
    d.Fields.Update()
    d.Repaginate()
    d.Save()
    d.ExportAsFixedFormat(OutputFileName=str(pdf),ExportFormat=17,OpenAfterExport=False,OptimizeFor=0,CreateBookmarks=1)
    print('saved_pdf',pdf,flush=True)
    d.Close(SaveChanges=False)
finally:
    app.Quit()
    pythoncom.CoUninitialize()
