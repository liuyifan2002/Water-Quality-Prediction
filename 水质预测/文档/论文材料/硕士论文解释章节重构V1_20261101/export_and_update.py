"""用独立、隐藏的Microsoft Word实例更新目录/页码域并导出PDF。"""
from pathlib import Path

import win32com.client


OUT = Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文解释章节重构V1_20261101")
DOCX = OUT / "水质预测硕士学位论文_解释章节重构V1.docx"
PDF = OUT / "水质预测硕士学位论文_解释章节重构V1.pdf"


def main() -> None:
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = None
    try:
        doc = word.Documents.Open(str(DOCX), ReadOnly=False, AddToRecentFiles=False, ConfirmConversions=False)
        doc.Repaginate()
        for i in range(1, doc.TablesOfContents.Count + 1):
            doc.TablesOfContents(i).Update()
        doc.Repaginate()
        doc.Fields.Update()
        doc.Repaginate()
        doc.Save()
        doc.ExportAsFixedFormat(str(PDF), 17, OpenAfterExport=False, OptimizeFor=0, Item=0, IncludeDocProps=True, KeepIRM=True, CreateBookmarks=1)
        print(f"WORD_EXPORT_OK toc={doc.TablesOfContents.Count} pages={doc.ComputeStatistics(2)} pdf={PDF}")
    finally:
        if doc is not None:
            doc.Close(SaveChanges=False)
        word.Quit()


if __name__ == "__main__":
    main()
