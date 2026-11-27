"""核对V4保护、V5章图表结构、冻结指标和实际PDF排版。"""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import csv,json,re
import fitz
import pandas as pd
from docx import Document

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
V4=ROOT/'文档/论文材料/硕士论文V4_结构与地图定向修订_20261108/水质预测硕士学位论文_V4_结构与地图定向修订.docx'
V5=OUT/'水质预测硕士学位论文_V5_正文修订.docx'
PDF=V5.with_suffix('.pdf')
CORE=ROOT/'transfer/llm_extension_transfer_v1/results/final_experiment_closure_audit_v1/final_core_results_table_v2.csv'
REG=ROOT/'transfer/llm_extension_transfer_v1/results/regionalization/分海区五折留站验证_20260906/五折留站点验证_汇总.csv'
ASSETS=ROOT/'结果/论文最终图件重构_v1/最终图件清单.csv'
assert all(p.is_file() for p in [V4,V5,PDF,CORE,REG,ASSETS])
a,b=Document(V4),Document(V5)
pdf=fitz.open(PDF)
alltext='\n'.join(p.text for p in b.paragraphs)
imgs=json.loads((OUT/'PDF图像保真记录.json').read_text(encoding='utf-8'))
figs=[re.match(r'图([1-6CDEFH]-\d+)',p.text).group(1) for p in b.paragraphs if p.style.name=='图题']
assert len(figs)==len(set(figs))==len(b.inline_shapes)==len(imgs)==94
assert len(pdf)==120 and sum(len(pg.get_image_info(xrefs=True)) for pg in pdf)==94
assert min(x['有效DPI'] for x in imgs)>=300
assert [p.text for p in a.paragraphs if p.style.name=='参考文献']==[p.text for p in b.paragraphs if p.style.name=='参考文献']
headings=[p.text for p in b.paragraphs if p.style.name.startswith('Heading')]
assert '1.7 本章小结' in headings and '1.7 论文结构' not in headings
assert '2.8.6 模型性能与空间泛化评价指标' in headings
assert '3.2 TabICLv2模型的SHAP解释' in headings
assert '5.4 历史分区扩展结果与空间异质性' in headings
assert 'D.3 TreeSHAP与来源消融对照' in headings
assert not any(x in alltext for x in ['图D-16','图D-17','图3-17','二维ALE交互','ROBUST','MODERATE','主模型'])
assert len(b.tables)==13
numbers=lambda d:Counter(c.text.strip() for t in d.tables for row in t.rows for c in row.cells if re.fullmatch(r'[+−-]?\d+(?:\.\d+)?%?',c.text.strip()))
assert numbers(a)==numbers(b)
core=pd.read_csv(CORE);regional=pd.read_csv(REG)
task_map={'chla_surface':'Chl-a表层','chla_middle':'Chl-a中层','chla_bottom':'Chl-a底层',
          'tn_surface':'TN表层','tn_middle':'TN中层','tn_bottom':'TN底层',
          'tp_surface':'TP表层','tp_middle':'TP中层','tp_bottom':'TP底层','secchi':'Secchi'}
for tabidx,model in [(8,'XGBoost'),(9,'TabICLv2')]:
    t=b.tables[tabidx]
    for row in t.rows[1:]:
        name=row.cells[0].text.replace(' ','')
        task=[k for k,v in task_map.items() if v==name]
        assert len(task)==1,name
        rec=core[(core.task==task[0])&(core.model==model)]
        assert len(rec)==1
        rr=rec.iloc[0]
        col=1 if tabidx==8 else 2
        assert abs(float(row.cells[col].text)-float(rr.pooled_R2))<.00051,(task,model)
        if tabidx==9:
            assert abs(float(row.cells[3].text)-float(rr.delta_R2))<.00051
            assert abs(float(row.cells[4].text.strip('%'))/100-float(rr.station_win_rate_RMSE))<.00051
            assert row.cells[5].text in ('稳健','中等')
for row in b.tables[1].rows[1:]:
    task=row.cells[0].text.replace(' ','')
    ids=[k for k,v in task_map.items() if v==task]
    assert len(ids)==1
    rr=core[core.task==ids[0]]
    assert len(rr)==2 and int(row.cells[3].text)==int(rr.iloc[0].n_stations) and int(row.cells[4].text)==int(rr.iloc[0].n_obs)
assert len(regional)==12 and (regional.folds==5).all()
assert len(b.tables[-1].rows)==78 and 'water_zone_code' not in [r.cells[1].text for r in b.tables[-1].rows]

# Word图题依序与PDF中的图像框同页；正文/附录每幅图都有真实框。
image_pages=[]
for i,(id,record) in enumerate(zip(figs,imgs),start=1):
    page=pdf[record['PDF页']-1]
    assert re.search(r'图\s*'+re.escape(id),page.get_text()),(i,id,record['PDF页'])
    image_pages.append({'图号':'图'+id,'PDF页':record['PDF页'],'有效DPI':record['有效DPI']})
for page in pdf:
    for im in page.get_image_info(xrefs=True):
        x0,y0,x1,y1=im['bbox']
        assert 0<=x0<x1<=page.rect.width and 0<=y0<y1<=page.rect.height
    for line in page.get_text('dict')['blocks']:
        if 'lines' not in line:continue
        for l in line['lines']:
            for s in l['spans']:
                x0,y0,x1,y1=s['bbox']
                assert x0>=-1 and x1<=page.rect.width+1,(page.number,s['text'])
pdftext='\n'.join(p.get_text() for p in pdf)
assert not any(x in pdftext for x in ['未定义书签','更新目录后显示页码','错误!'])
table_page=56
table_text=pdf[table_page-1].get_text()
assert '表4-2' in table_text and '表4-2' not in pdf[table_page-2].get_text()
assert all(x in table_text for x in ['0.354','0.136','0.374','75.0%','稳健','中等'])

with (OUT/'章节证据对应表.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['章节','论点或作用','正文图号','正文表号','数据或证据来源','科学口径及限制'])
    w.writeheader()
    ps=b.paragraphs
    for i,p in enumerate(ps):
        if p.style.name not in ('Heading 2','Heading 3') or not re.match(r'^[1-6]\.\d',p.text):continue
        depth=int(p.style.name[-1]);collected=[]
        for q in ps[i+1:]:
            if q.style.name.startswith('Heading') and int(q.style.name[-1])<=depth:break
            collected.append(q.text)
        txt=' '.join(collected)
        found_fig=sorted(set(re.findall(r'图(?:[1-6]|[CDEFH])-\d+',txt)))
        found_tab=sorted(set(re.findall(r'表[1-6]-\d+',txt)))
        sources=[]
        with ASSETS.open(encoding='utf-8-sig',newline='') as rf:
            reg={r['新图号']:r for r in csv.DictReader(rf)}
        for fig in found_fig:
            if fig in reg and reg[fig]['数据来源']:sources.append(reg[fig]['数据来源'])
        ch=p.text[0]
        if ch=='2':sources.append(str(CORE.relative_to(ROOT)) if p.text.startswith('2.2') or p.text.startswith('2.3') else 'V4正文表2-2、表2-3及附录G')
        elif ch=='4':sources.append(str(CORE.relative_to(ROOT)))
        elif ch=='5':sources.append(str(REG.relative_to(ROOT)) if p.text.startswith(('5.1','5.2','5.4')) else str(ASSETS.relative_to(ROOT)))
        elif ch=='3':sources.append('历史冻结解释结果；P1核心样本文件；图件清单逐图来源')
        else:sources.append('V4正文与原参考文献')
        boundary=('历史78特征分区扩展，不与正式77特征混合' if ch=='5' and p.text.startswith(('5.1','5.2','5.4'))
                  else '十任务正式站点留出；pooled OOF与逐折分开' if ch=='4'
                  else '历史固定批与第1折P1单查询分开' if ch=='3'
                  else '正式77特征；Secchi单任务' if ch=='2'
                  else '研究背景或研究目标，非新增实验结果')
        w.writerow({'章节':p.text,'论点或作用':re.sub(r'\s+',' ',txt[:120]),'正文图号':'；'.join(found_fig),
                    '正文表号':'；'.join(found_tab),'数据或证据来源':'；'.join(dict.fromkeys(sources)),
                    '科学口径及限制':boundary})
with (OUT/'排版核查/逐图实际页码.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['图号','PDF页','有效DPI']);w.writeheader();w.writerows(image_pages)
state={'V4_SHA256':sha256(V4.read_bytes()).hexdigest(),'V5_DOCX_SHA256':sha256(V5.read_bytes()).hexdigest(),
       'V5_PDF_SHA256':sha256(PDF.read_bytes()).hexdigest(),'PDF页数':len(pdf),'图题和图像数':94,
       '正文表格数':len(b.tables),'数值单元格逐项一致':True,'参考文献逐段一致':True,
       '最低图像有效DPI':min(x['有效DPI'] for x in imgs),'表4-2完整页':table_page,
       '图题图像同页':True,'章节证据行数':sum(1 for _ in open(OUT/'章节证据对应表.csv',encoding='utf-8-sig'))-1}
(OUT/'排版核查/最终结构与排版核验.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(state,ensure_ascii=False))
