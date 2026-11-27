"""结构、冻结结果、目录页码、PDF完整性核验；不运行任何科研模型。"""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
import re
import csv
import fitz
from docx import Document
from docx.oxml.ns import qn

ROOT=Path(r'E:/研究生/codex/水质预测')
SRC=ROOT/'文档/论文材料/硕士论文V3_第二章特征与候选数据扩充_20261101/水质预测硕士学位论文_V3_第二章特征与候选数据扩充_工作副本.docx'
OUT=Path(__file__).resolve().parent
DOC=OUT/'水质预测硕士学位论文_V4_结构与地图定向修订.docx'
PDF=OUT/'水质预测硕士学位论文_V4_结构与地图定向修订.pdf'
source=Document(SRC);final=Document(DOC);pdf=fitz.open(PDF)
def text(e):return ''.join(x.text or '' for x in e.iter(qn('w:t'))).strip()
def block(doc,a,z):
    b=list(doc.element.body)
    start_index=next(i for i,e in enumerate(b) if text(e).startswith(a) and i>260)
    end_index=next(i for i,e in enumerate(b) if text(e).startswith(z) and i>start_index)
    return b[start_index:end_index]
def blob_after(doc,title):
    p=next(p for p in doc.paragraphs if p.text==title)
    q=p._p.getprevious(); blips=q.xpath('.//a:blip')
    assert len(blips)==1,title
    rid=blips[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    return doc.part.related_parts[rid].blob
def sha(b):return sha256(b).hexdigest()
def check(ok,msg):
    rows.append((msg,'通过' if ok else '失败'))
    if not ok: problems.append(msg)
rows=[];problems=[]
with ZipFile(DOC) as z:check(z.testzip() is None,'DOCX ZIP结构完整')
check(len(pdf)>90,'PDF可读且页数合理')
old45='\n'.join(text(e) for e in block(source,'第4章 ','第6章 '))
new45='\n'.join(text(e) for e in block(final,'第4章 ','第6章 '))
check(old45==new45,'第四、五章全文及表格文字逐字一致')
check(sha(blob_after(source,'图4-4 四个代表任务中XGBoost、TabICLv2、TabPFN和TabM的比较'))==sha(blob_after(final,'图4-4 四个代表任务中XGBoost、TabICLv2、TabPFN和TabM的比较')),'图4-4嵌入图像字节不变')
for title in [p.text for p in source.paragraphs if re.match(r'^图3-(?:[1-9]|1[0-7])\s',p.text) and '\t' not in p.text]:
    try: same=sha(blob_after(source,title))==sha(blob_after(final,title))
    except Exception:same=False
    check(same,'原一维ALE/SHAP/稳定性图像未变：'+title[:24])
body=list(final.element.body)
start=next(i for i,e in enumerate(body) if text(e)=='第1章 绪论')
end=next(i for i,e in enumerate(body) if text(e)=='参考文献')
main='\n'.join(text(e) for e in body[start:end])
check(not any(k in main for k in ['二维ALE','二维 ALE','图3-18','图3-19','图3-20','重点站点','重点研究站']),'正文无TabICLv2二维ALE/重点站点残留')
first1=next(i for i,p in enumerate(final.paragraphs) if p.style.name=='Heading 1' and p.text.startswith('第1章'))
abstracts='\n'.join(p.text for p in final.paragraphs[:first1] if 'Hong Kong' in p.text or '香港近岸水质具有' in p.text)
check('二维ALE' not in abstracts and '二维 ALE' not in abstracts and 'two-dimensional' not in abstracts,'中英文摘要不含二维ALE')
heads=[p.text for p in final.paragraphs if p.style.name.startswith('Heading')]
for h in ['2.8 水质预测实验设计与模型配置','2.8.1 预测任务与验证方案','2.8.2 站点留出与统一五折划分','2.9 样本分布及数据特征','2.10 本章小结','第3章 预测模型与解释方法','3.1.4 本文模型选择与比较依据','3.2 模型性能与空间泛化评价指标','3.5 解释结果的可靠性分析','3.6 本章小结']:
    check(h in heads,'结构标题：'+h)
check(not any(h.startswith('3.5.3') or h.startswith('3.6.3') for h in heads),'二维ALE小节已删除')
check(sum(p.text.startswith('表2-5 XGBoost') for p in final.paragraphs[first1:])==1,'原表3-1迁至表2-5')
check(sum(p.text.startswith('表3-1 主要评价') for p in final.paragraphs[first1:])==1,'原表3-2改为表3-1')
check(sum(p.text.startswith('表3-2 十项水质') for p in final.paragraphs[first1:])==1,'原表3-3改为表3-2')
# 表2-3中的77个精确英文名称应与母本逐行一致。
def after_table(doc,prefix):
    p=next(p for p in doc.paragraphs if p.text.startswith(prefix) and p.style.name not in ('Normal',))
    e=p._p.getnext();assert e.tag==qn('w:tbl');return [text(x) for x in e.findall(qn('w:tr'))]
check(after_table(source,'表2-3 77项')==after_table(final,'表2-3 77项'),'表2-3的77项字段逐行不变')
check(len(after_table(final,'表2-3 77项'))==78,'表2-3含表头+77项')
check(len(after_table(final,'表2-4 历史候选'))==18,'表2-4含表头+17类候选')
refnums={int(m.group(1)) for p in final.paragraphs for m in [re.match(r'^\[(\d+)\]',p.text)] if m}
check(refnums>=set(range(1,22)),'参考文献[1]-[21]存在')
pages=[''.join(p.get_text().split()) for p in pdf]
for kind in ('图','表'):
    a=next(i for i,e in enumerate(body) if text(e)==kind+'目录')
    z=next(i for i,e in enumerate(body) if text(e)==('表目录' if kind=='图' else '主要符号表'))
    entries=[e for e in body[a+1:z] if re.match(r'^'+kind+r'(?:[1-6]|[A-Z])-\d+',text(e))]
    caps=[text(e) for e in body[start:] if e.tag==qn('w:p') and re.match(r'^'+kind+r'(?:[1-6]|[A-Z])-\d+\s',text(e))]
    check(len(entries)==len(caps),'图表目录条目数与图题一致：'+kind)
    for entry,cap in zip(entries,caps):
        ts=list(entry.iter(qn('w:t')))
        actual=[i+1 for i,s in enumerate(pages[14:],start=14) if ''.join(cap.split()) in s]
        if len(ts)!=2 or ts[0].text!=cap or len(actual)!=1 or ts[1].text!=str(actual[0]):
            problems.append('目录页码或图题不一致：'+cap[:40])
blank=[i+1 for i,p in enumerate(pdf) if i>14 and not p.get_text().strip() and not p.get_images()]
check(not blank,'正文和附录无真正空白页')
print('DOCX',DOC,'PDF',PDF,'PAGES',len(pdf),'CHECKS',len(rows),'PROBLEMS',len(problems))
for x in problems:print('FAIL',x)
with (OUT/'结构与结果核验.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['核验项','状态']);w.writerows(rows)
with (OUT/'核验摘要.txt').open('w',encoding='utf-8') as f:
    f.write('源DOCX SHA256：'+sha(SRC.read_bytes())+'\n输出DOCX SHA256：'+sha(DOC.read_bytes())+'\n输出PDF SHA256：'+sha(PDF.read_bytes())+'\nPDF页数：'+str(len(pdf))+'\n问题：'+str(problems)+'\n')
if problems:raise SystemExit(1)
