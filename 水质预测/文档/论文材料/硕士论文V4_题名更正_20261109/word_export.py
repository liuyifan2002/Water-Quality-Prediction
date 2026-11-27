"""用 Microsoft Word 更新目录与分页，并导出题名修正版 PDF。"""

from pathlib import Path

import pythoncom
import win32com.client as win32


OUT = Path(__file__).resolve().parent
DOCX = OUT / "水质预测硕士学位论文_V4_题名更正.docx"
PDF = OUT / "水质预测硕士学位论文_V4_题名更正.pdf"

pythoncom.CoInitialize()
app = win32.DispatchEx("Word.Application")
app.Visible = False
app.DisplayAlerts = 0
try:
    document = app.Documents.Open(
        str(DOCX), ReadOnly=False, AddToRecentFiles=False, ConfirmConversions=False
    )
    document.Repaginate()
    for i in range(1, document.TablesOfContents.Count + 1):
        document.TablesOfContents(i).Update()
    for i in range(1, document.TablesOfFigures.Count + 1):
        document.TablesOfFigures(i).Update()
    document.Fields.Update()
    document.Repaginate()
    pages = document.ComputeStatistics(2)
    document.Save()
    document.ExportAsFixedFormat(
        OutputFileName=str(PDF),
        ExportFormat=17,
        OpenAfterExport=False,
        OptimizeFor=0,
        CreateBookmarks=1,
    )
    print("WORD_PAGES", pages)
    print("TOC_COUNT", document.TablesOfContents.Count)
    print("PDF", PDF)
    document.Close(SaveChanges=False)
finally:
    app.Quit()
    pythoncom.CoUninitialize()
