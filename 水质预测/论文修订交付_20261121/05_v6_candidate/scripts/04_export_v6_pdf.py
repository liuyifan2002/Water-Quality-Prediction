"""使用本机Word更新目录和分页，导出V6未完成候选稿PDF。"""
from pathlib import Path
import json
import win32com.client

OUT=Path(__file__).resolve().parents[1]
docx=OUT/'水质预测硕士学位论文_V6_候选稿_解释待核验.docx'
pdf=OUT/'水质预测硕士学位论文_V6_候选稿_解释待核验.pdf'
assert docx.is_file() and not pdf.exists()
word=win32com.client.DispatchEx('Word.Application')
word.Visible=False;word.DisplayAlerts=0
try:
    d=word.Documents.Open(str(docx),ReadOnly=False,AddToRecentFiles=False)
    try:
        d.Repaginate()
        for _ in range(2):
            d.Fields.Update()
            for toc in d.TablesOfContents:toc.Update()
            d.Repaginate()
        d.Save()
        d.ExportAsFixedFormat2(OutputFileName=str(pdf),ExportFormat=17,OptimizeFor=0,OptimizeForImageQuality=True)
        info={'Word版本':word.Version,'页数':d.ComputeStatistics(2),'内嵌图片数':d.InlineShapes.Count,'目录数':d.TablesOfContents.Count}
        (OUT/'Word导出记录.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(info,ensure_ascii=False))
    finally:d.Close(False)
finally:word.Quit()
