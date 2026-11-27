"""只读定位 V3 关键页面并导出低分辨率检查图。"""
from pathlib import Path
import fitz

base=Path(__file__).resolve().parent
pdf=base/'水质预测硕士学位论文_V3_三项定向修改.pdf'
d=fitz.open(pdf)
targets={
    'fig21':'图2-1 研究区地理位置',
    'sec26':'2.6 严格特征集的组成',
    'chapter3':'第3章 预测模型、实验设计与解释方法',
    'sec31':'3.1 预测模型及基本原理',
    'appendixg':'附录G 77项严格输入特征清单',
}
for key,needle in targets.items():
    pages=[i+1 for i,p in enumerate(d) if needle.replace(' ','') in p.get_text().replace(' ','').replace('\n','')]
    print(key,pages)
    if pages:
        page=pages[-1]-1
        d[page].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(base/f'QA_{key}.png')
print('page_count',len(d))
