"""按附件确定的文本在V2副本上做三项定向修改；不碰冻结结果。"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import csv
import re

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

ROOT=Path(r"E:\研究生\codex\水质预测")
OUT=ROOT/r"文档\论文材料\硕士论文V3_三项定向修改_20261101"
SOURCE=ROOT/r"文档\论文材料\硕士论文V2_学术内容补强_20261101\水质预测硕士学位论文_学术内容补强V2_核验交付.docx"
TARGET=OUT/"水质预测硕士学位论文_V3_三项定向修改.docx"
ATTACH=Path(r"D:\Users\刘逸凡\Downloads\%E6%B0%B4%E8%B4%A8%E8%AE%BA%E6%96%87V3_%E4%B8%89%E9%A1%B9%E5%AE%9A%E5%90%91%E4%BF%AE%E6%94%B9_Codex%E6%89%A7%E8%A1%8C%E7%A8%BF.md")
MANIFEST=ROOT/r"runs\tabiclv2_pilot\01_strict_feature_manifest.csv"
MAPPING=ROOT/r"runs\explanation_figures_v2\metadata\feature_display_name_mapping.csv"
MAIN_TABLE=ROOT/r"数据\处理后数据\正式主表\model_table_3depth_2014_2024.csv"
MAP=OUT/"Figure_2_1.png"
assert all(p.is_file() for p in (SOURCE,ATTACH,MANIFEST,MAPPING,MAIN_TABLE,MAP))
spec=ATTACH.read_text(encoding="utf-8")
d=Document(SOURCE)

def find(text):
    hits=[p for p in d.paragraphs if p.text.strip()==text]
    assert len(hits)==1,(text,len(hits))
    return hits[0]
def remove(p):p._p.getparent().remove(p._p)
def quote_after(marker):
    section=spec.split(marker,1)[1]
    m=re.search(r"^> (.+)$",section,re.M)
    assert m,marker
    return m.group(1).strip()
def clean(s):
    s=s.replace("$X_{\\mathrm{strict}}$","X_strict")
    return s.replace("`","")

# A：现有图2-1原位替换；新图是ArcGIS可编辑工程导出，而非参考截图。
heading=find("2.1 研究区概况")
heading._p.getnext()
p21=next(p for p in d.paragraphs if p.text.startswith("研究区覆盖香港近岸环境保护署水质监测网络。"))
p21.text=quote_after("2.1 建议增补的正文")
cap=find("图2-1 香港近岸监测站点与海区分布")
cap.text="图2-1 研究区地理位置与香港近岸水质监测站点分布"
imgp=Paragraph(cap._p.getprevious(),d)
assert imgp._p.xpath('.//w:drawing'),"图2-1上一段不是原图"
imgp.clear()
imgp.alignment=WD_ALIGN_PARAGRAPH.CENTER
imgp.add_run().add_picture(str(MAP),width=Cm(15.0))
find("2.2 水质监测数据").insert_paragraph_before(
    "注：深蓝点为Chl-a、TN和TP共同正式任务的84个站点；灰点为原始网络中其余10站，透明度任务覆盖94站。"
    "SM18、SM19、PT3和VT11为正文涉及的重点站点。站点坐标来自香港环境保护署监测网络；陆地底图采用Natural Earth，"
    "主图为WGS84/UTM 50N米制投影，比例尺随地图框动态生成。",style="Normal")

# B：冻结manifest 78行包含明确排除的water_zone_code；入模77项与显示映射顺序完全一致。
fm=pd.read_csv(MANIFEST)
fm=fm.loc[fm.included_in_pilot.astype(bool)].reset_index(drop=True)
mp=pd.read_csv(MAPPING)
assert len(fm)==len(mp)==77
assert fm.feature.tolist()==mp.raw_feature_name.tolist()
main_cols=set(pd.read_csv(MAIN_TABLE,nrows=0).columns)
assert set(fm.feature)<=main_cols
assert not any(x.lower() in ('chla','chl_a','secchi','tn','tp','obs_id','station_id','water_zone_code') for x in fm.feature)
counts=Counter(mp.source_group)
expected={'METEOROLOGY':29,'RAINFALL_HYDRO_TIDE':22,'OCEAN_PHYSICS':11,
          'HUMAN_ACTIVITY':6,'TEMPORAL_PROCESS':7,'SPATIAL_STATIC':2}
assert counts==expected,counts
assert 'water_zone_code' not in fm.feature.values
strict=find("2.6 严格特征集X_strict")
strict.text="2.6 严格特征集的组成与构建"
section=spec.split("正文按以下文字替换",1)[1].split("**表 2-2",1)[0]
quotes=re.findall(r"^> (.+)$",section,re.M)
assert len(quotes)==3,len(quotes)
para1=clean(quotes[0]);para2=clean(quotes[1]);para3=clean(quotes[2])
para2=para2.replace("前一日风速","同站上一采样时次风速")
assert "前一日风速" not in para2
para3=para3.replace(
    "带有 lag、3d 或 14d 等标识的历史特征应按照冻结特征生成程序中的时间窗口构建，只使用相应预测时点之前或截至该时点允许获得的信息。",
    "带有 lag、3d 或 14d 等标识的历史特征按冻结生成程序构建：3d和14d窗使用采样日前的自然日，lag表示同站前序采样记录，并非前一自然日。rain_month为采样所在日历月的整月累计降雨量，其观测当日可得性仍须核查。")
title=find("表2-2 严格特征集的来源分组")
title.insert_paragraph_before(para1,style="Normal")
title.insert_paragraph_before(para2,style="Normal")
old_body=next(p for p in d.paragraphs if p.text.startswith("六类特征合计77个："))
old_body.text=para3
t_old=d.tables[2]
assert len(t_old.rows)==7 and len(t_old.columns)==2
rows=[
    ["特征来源组","数量","代表特征","信息含义","构造说明"],
    ["气象（MET）","29","ssrd_sum_3d、wind_speed_lag1、wind_speed_x_rain14d","辐射、风和气温等变化","原始量、历史滚动/滞后及组合"],
    ["降雨—水文—潮汐（RHT）","22","rain_3d、rain_14d、tidal_range","降雨和潮汐背景","按字段定义的时间尺度匹配"],
    ["海洋物理（OCN）","11","sst_mean_3d","海温等海洋环境代理","与观测日匹配的统计量"],
    ["人类活动（HUM）","6","ntl_avg_rad","夜间灯光等活动代理","站点位置与对应数据匹配"],
    ["时间过程（TMP）","7","year_offset、month_sin","年际趋势和季节周期","根据观测时间编码"],
    ["空间静态（SPA）","2","lon、lat","监测站地理位置","站点级固定属性"],
]
widths=[Cm(3.0),Cm(1.1),Cm(5.0),Cm(3.3),Cm(3.0)]
t=d.add_table(rows=7,cols=5)
t.style='Table Grid'
t_old._tbl.addprevious(t._tbl)
t_old._tbl.getparent().remove(t_old._tbl)
t.autofit=False
for col,w in zip(t.columns,widths):col.width=w
for i,row in enumerate(t.rows):
    for j,(cell,value) in enumerate(zip(row.cells,rows[i])):
        cell.text=value;cell.width=widths[j]
        for p in cell.paragraphs:
            p.paragraph_format.space_before=0;p.paragraph_format.space_after=0
            p.paragraph_format.line_spacing=1.0
            p.paragraph_format.first_line_indent=Pt(0)
            p.paragraph_format.left_indent=Pt(0)
            p.paragraph_format.right_indent=Pt(0)
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER if i==0 or j==1 else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:r.font.size=Pt(8.0)
    trpr=row._tr.get_or_add_trPr()
    if trpr.find(qn('w:cantSplit')) is None:trpr.append(OxmlElement('w:cantSplit'))
    if i==0 and trpr.find(qn('w:tblHeader')) is None:trpr.append(OxmlElement('w:tblHeader'))
old_body.insert_paragraph_before("注：组别缩写 MET、RHT、OCN、HUM、TMP、SPA 与附录 G 一致。",style="Normal")

# C：旧3.1/3.2整块由附件指定的中立比较段替换；其余方法结果保留。
start=find("3.1 预测模型概述")
stop=find("3.3 水质预测实验设计与模型配置")
nodes=[];cur=start._p
while cur is not stop._p:
    nxt=cur.getnext();nodes.append(cur);cur=nxt
assert len(nodes)>=13
for node in nodes:node.getparent().remove(node)
new_sections=[
    ("3.1 预测模型及基本原理","Heading 2"),
    (clean(quote_after("### 3.1 预测模型及基本原理")),"Normal"),
    ("3.1.1 XGBoost","Heading 3"),(clean(quote_after("### 3.1.1 XGBoost")),"Normal"),
    ("3.1.2 TabPFN","Heading 3"),(clean(quote_after("### 3.1.2 TabPFN")),"Normal"),
    ("3.1.3 TabM","Heading 3"),(clean(quote_after("### 3.1.3 TabM")),"Normal"),
    ("3.1.4 TabICLv2","Heading 3"),(clean(quote_after("### 3.1.4 TabICLv2")),"Normal"),
    (clean(quote_after("**新 3.1 的末段（直接使用）：**")),"Normal"),
]
for text,style in new_sections:stop.insert_paragraph_before(text,style=style)
find("第3章 模型构建、实验设计与可解释性分析").text="第3章 预测模型、实验设计与解释方法"
for p in d.paragraphs:
    if p.style.name.startswith('Heading'):
        m=re.match(r'^(3)\.(\d+)(\.\d+)?( .*)$',p.text)
        if not m:continue
        section_no=int(m.group(2))
        if 3<=section_no<=7:
            p.text=f"3.{section_no-1}{m.group(3) or ''}{m.group(4)}"
find("3.2 水质预测实验设计与模型配置")
find("3.3 模型性能与空间泛化评价指标")
find("3.4 TabICLv2模型的SHAP解释")
find("3.5 TabICLv2模型的一维ALE响应分析")
find("3.6 TabICLv2解释结果的可靠性与二维支持审查").text="3.6 解释结果的可靠性与二维支持审查"
for old_title in ("3.8 XGBoost基准模型十项任务预测结果","3.9 TabICLv2与XGBoost预测及解释结果比较"):
    p=find(old_title);following=Paragraph(p._p.getnext(),d)
    assert following.style.name=="Normal"
    remove(following);remove(p)
find("3.10 本章小结").text="3.7 本章小结"
summary=next(p for p in d.paragraphs if p.text.startswith("本章以TabICLv2为解释主线："))
summary.text=("本章在统一特征与站点划分下介绍XGBoost、TabPFN、TabM和TabICLv2，并区分十任务正式比较与四任务扩展比较。"
              "模型性能将在第四章依据实际结果判断。解释分析以TabICLv2为对象，历史四任务五折与十任务第1折P1单查询结果分层报告；"
              "XGBoost的TreeSHAP/ALE作为独立基准模型解释置于附录D。二维P1候选仅完成联合支持审查，未计算响应曲面。")
overview=next(p for p in d.paragraphs if p.text.startswith("全文共六章。"))
overview.text=("全文共六章。第1章提出近岸水质预测的空间泛化问题；第2章介绍研究区、监测数据和77项严格特征；"
               "第3章介绍四种候选模型的基本原理、统一实验设计、评价指标及模型解释方法；第4章分析十任务预测精度与站点异质性；"
               "第5章讨论区域建模、来源消融与任务环境依赖；第6章总结主要结论、限制和后续工作。")
duplicate=next(p for p in d.paragraphs if p.text.startswith("分海区实验比较全域模型与区域模型在冻结验证下的差异；"))
remove(duplicate)

# 附录G：77行按正式入模顺序，确证不了的单位明确写待核，不补造。
def meaning(name,raw):
    s=str(raw).replace("mean","均值").replace("sum","总量").replace("max","最大值")
    s=s.replace(" Δ1"," 前两次采样差值").replace(" 滞后1"," 前1次采样").replace(" 滞后2"," 前2次采样")
    s=s.replace(" 滚动3"," 前3次采样均值")
    return s
def unit(name,group):
    if '_x_' in name:return '组合量（待核）'
    if name.startswith('ssrd_'):return 'J/m²'
    if name.startswith('sst_') or name.startswith('t2m'):return '°C'
    if name.startswith('wind_speed_sq'):return 'm²/s²'
    if name.startswith(('u10','v10','wind_speed')):return 'm/s'
    if name.startswith('rain_'):return 'mm'
    if name.startswith('tidal_range'):return 'm'
    if name=='viirs_count':return '次'
    if name.startswith('ntl_'):return '待核'
    if name in ('lat','lon'):return '°'
    if name in ('day_of_year','days_since_prev_sample'):return 'd'
    if name=='year_offset':return '年'
    if name in ('month_cos','month_sin','is_wet_season'):return '无量纲'
    if name=='quarter':return '季度序号'
    if name.startswith(('wind_dir_','tcc')):return '无量纲'
    return '待核'
def source_name(name,group):
    if '_x_' in name:
        if 'rain' in name and ('ssrd' in name or 'wind_speed' in name or 't2m' in name):return 'ERA5+香港天文台'
        if 'rain' in name and 'tidal' in name:return '香港天文台'
        if 'tidal' in name and 't2m' in name:return 'ERA5+香港天文台'
        return 'ERA5'
    return {'METEOROLOGY':'ERA5','OCEAN_PHYSICS':'ERA5','RAINFALL_HYDRO_TIDE':'香港天文台',
            'HUMAN_ACTIVITY':'VIIRS夜间灯光','TEMPORAL_PROCESS':'监测日期','SPATIAL_STATIC':'环保署站点'}[group]
def processing(name):
    if '_x_' in name:return '两列直接乘积；沿用分量时间窗'
    if name.endswith('_delta1'):return '同站前1次减前2次采样值'
    if name.endswith('_lag1'):return '同站前1次采样值'
    if name.endswith('_lag2'):return '同站前2次采样值'
    if name.endswith('_roll3'):return '同站前3次采样均值'
    if name.startswith('rain_month'):return '采样所在日历月整月累计；时点可得性待核'
    if name.startswith('rain_3d'):return '采样日前3个自然日累计'
    if name.startswith('rain_14d'):return '采样日前14个自然日累计'
    m=re.search(r'_(3|7|14)d$',name)
    if m:return '采样日前'+m.group(1)+'个自然日统计'
    if name=='year_offset':return '年份减主表最小年份'
    if name in ('month_sin','month_cos'):return '月份周期编码'
    if name=='is_wet_season':return '4—9月为湿季'
    if name=='days_since_prev_sample':return '同站同水层距上次采样天数'
    if name in ('lat','lon'):return '站点级固定坐标'
    if name=='tidal_range':return '采样日高潮位减低潮位'
    if name.startswith('ntl_') or name=='viirs_count':return '站点—月份匹配'
    if name in ('u10','v10','t2m','tcc'):return 'ERA5月度值匹配'
    if name=='wind_speed':return 'sqrt(u10²+v10²)'
    if name=='wind_speed_sq':return 'wind_speed²'
    if name in ('wind_dir_sin','wind_dir_cos'):return 'u10/v10除以风速'
    if name=='quarter':return '采样日期季度'
    if name=='day_of_year':return '采样日期年内日序'
    return '按冻结构建脚本生成；细节待核'

cv=find("个人简历")
portrait=deepcopy(d.sections[-1]._sectPr)
landscape=deepcopy(portrait)
sz=landscape.find(qn('w:pgSz'))
sz.set(qn('w:w'),'16838')
sz.set(qn('w:h'),'11906')
sz.set(qn('w:orient'),'landscape')
margin=landscape.find(qn('w:pgMar'))
margin.set(qn('w:left'),'1134')
margin.set(qn('w:right'),'1134')
for prop in (portrait,landscape):
    section_type=prop.find(qn('w:type'))
    if section_type is None:section_type=OxmlElement('w:type');prop.insert(0,section_type)
    section_type.set(qn('w:val'),'nextPage')
start_break=cv.insert_paragraph_before("")
start_break._p.get_or_add_pPr().append(portrait)
cv.insert_paragraph_before("附录G 77项严格输入特征清单",style="Heading 2")
cv.insert_paragraph_before("下表按正式入模顺序列出77项字段。来源组代码：MET=气象，RHT=降雨—水文—潮汐，OCN=海洋物理，HUM=人类活动代理，TMP=时间过程，SPA=空间静态。待核项目不补造单位。",style="Normal")
columns=['编号','精确字段名','中文含义','类别','单位','原始数据来源','时间窗口或加工方式']
tab=d.add_table(rows=1,cols=7)
tab.style='Table Grid';tab.autofit=False
colwidths=[Cm(1.2),Cm(5.0),Cm(3.4),Cm(1.6),Cm(2.2),Cm(2.8),Cm(8.1)]
for i,(cell,value) in enumerate(zip(tab.rows[0].cells,columns)):
    cell.text=value;cell.width=colwidths[i]
for i,(manifest_row,map_row) in enumerate(zip(fm.itertuples(index=False),mp.itertuples(index=False)),1):
    name=manifest_row.feature;group=map_row.source_group
    code={'METEOROLOGY':'MET','RAINFALL_HYDRO_TIDE':'RHT','OCEAN_PHYSICS':'OCN',
          'HUMAN_ACTIVITY':'HUM','TEMPORAL_PROCESS':'TMP','SPATIAL_STATIC':'SPA'}[group]
    values=[str(i),name,meaning(name,map_row.display_name_cn),code,unit(name,group),source_name(name,group),processing(name)]
    cells=tab.add_row().cells
    for j,(cell,value) in enumerate(zip(cells,values)):
        cell.text=value;cell.width=colwidths[j]
for row_i,row in enumerate(tab.rows):
    trpr=row._tr.get_or_add_trPr()
    if trpr.find(qn('w:cantSplit')) is None:trpr.append(OxmlElement('w:cantSplit'))
    if row_i==0 and trpr.find(qn('w:tblHeader')) is None:trpr.append(OxmlElement('w:tblHeader'))
    for col_i,cell in enumerate(row.cells):
        for p in cell.paragraphs:
            p.paragraph_format.space_before=0;p.paragraph_format.space_after=0
            p.paragraph_format.line_spacing=1.0
            p.paragraph_format.first_line_indent=Pt(0)
            p.paragraph_format.left_indent=Pt(0)
            p.paragraph_format.right_indent=Pt(0)
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER if row_i==0 or col_i in (0,3,4) else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:run.font.size=Pt(8.0)
for col,w in zip(tab.columns,colwidths):col.width=w
cv._p.addprevious(tab._tbl)
cv.insert_paragraph_before("注：ssrd与sst日窗均截至采样日前一日；lag/roll基于前序采样记录而非固定自然日。rain_month整月累计及部分VIIRS变量单位仍需依原始数据字典核查。",style="Normal")
end_break=cv.insert_paragraph_before("")
end_break._p.get_or_add_pPr().append(landscape)
assert len(tab.rows)==78

# 静态图目录仅改图名，页码在PDF首轮导出后另行逐图更新。
for p in d.paragraphs:
    if p.style.name.startswith('toc') and p.text.startswith('图2-1 香港近岸监测站点与海区分布'):
        p.text=p.text.replace('图2-1 香港近岸监测站点与海区分布','图2-1 研究区地理位置与香港近岸水质监测站点分布')

d.save(TARGET)
print('V3_DOCX_CREATED',TARGET,'FEATURE_ROWS',len(tab.rows)-1,'TABLES',len(d.tables),'IMAGES',len(d.inline_shapes))
