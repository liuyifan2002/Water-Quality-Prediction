"""只生成内部试排副本，测试2.7节自然接续是否优于强制换页。"""
from pathlib import Path
from docx import Document
base=Path(__file__).resolve().parent
src=base/'水质预测硕士学位论文_V3_第二章定向修订_20261108.docx'
dst=base/'_内部校验/版式试排_自然接续.docx'
doc=Document(src)
p=next(p for p in doc.paragraphs if p.text=='2.7 候选环境数据的探索与筛选' and p.style.name=='Heading 2')
assert p.paragraph_format.page_break_before is True
p.paragraph_format.page_break_before=False
doc.save(dst)
print('TRIAL',dst)
