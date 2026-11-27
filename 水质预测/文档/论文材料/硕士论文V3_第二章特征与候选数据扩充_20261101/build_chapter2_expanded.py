"""在V3工作副本中扩充2.6/2.7；原V3与冻结结果保持只读。"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import csv
import re

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT=Path(r'E:\研究生\codex\水质预测')
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/r'文档\论文材料\硕士论文V3_三项定向修改_20261101\水质预测硕士学位论文_V3_三项定向修改_核验交付.docx'
TARGET=OUT/'水质预测硕士学位论文_V3_第二章特征与候选数据扩充_工作副本.docx'
MANIFEST=ROOT/r'runs\tabiclv2_pilot\01_strict_feature_manifest.csv'
MAPPING=ROOT/r'runs\explanation_figures_v2\metadata\feature_display_name_mapping.csv'
MAIN=ROOT/r'数据\处理后数据\正式主表\model_table_3depth_2014_2024.csv'
CANDIDATES=OUT/'候选环境数据来源核验表.csv'
assert all(x.is_file() for x in (SOURCE,MANIFEST,MAPPING,MAIN,CANDIDATES))
d=Document(SOURCE)
fm=pd.read_csv(MANIFEST)
fm=fm.loc[fm.included_in_pilot.astype(bool)].reset_index(drop=True)
mp=pd.read_csv(MAPPING)
assert len(fm)==len(mp)==77 and fm.feature.tolist()==mp.raw_feature_name.tolist()
assert set(fm.feature)<=set(pd.read_csv(MAIN,nrows=0).columns)
assert Counter(mp.source_group)=={'METEOROLOGY':29,'RAINFALL_HYDRO_TIDE':22,'OCEAN_PHYSICS':11,
                                   'HUMAN_ACTIVITY':6,'TEMPORAL_PROCESS':7,'SPATIAL_STATIC':2}
appendix=d.tables[-1]
assert len(appendix.rows)==78 and len(appendix.columns)==7
assert [row.cells[1].text for row in appendix.rows[1:]]==fm.feature.tolist()
with CANDIDATES.open(encoding='utf-8-sig',newline='') as f:candidates=list(csv.DictReader(f))
assert len(candidates)==17

def one(text):
    hits=[p for p in d.paragraphs if p.text.strip()==text]
    assert len(hits)==1,(text,len(hits))
    return hits[0]
def ready_table(table,widths,font_pt=8.0):
    table.style='Table Grid';table.autofit=False
    for col,width in zip(table.columns,widths):col.width=Cm(width)
    for i,row in enumerate(table.rows):
        trpr=row._tr.get_or_add_trPr()
        if trpr.find(qn('w:cantSplit')) is None:trpr.append(OxmlElement('w:cantSplit'))
        if i==0 and trpr.find(qn('w:tblHeader')) is None:trpr.append(OxmlElement('w:tblHeader'))
        for j,cell in enumerate(row.cells):
            cell.width=Cm(widths[j])
            for p in cell.paragraphs:
                f=p.paragraph_format
                f.space_before=Pt(0);f.space_after=Pt(0);f.line_spacing=1.0
                f.first_line_indent=Pt(0);f.left_indent=Pt(0);f.right_indent=Pt(0)
                p.alignment=WD_ALIGN_PARAGRAPH.CENTER if i==0 or j==0 else WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:run.font.size=Pt(font_pt)

# 2.6：从现有附录G逐行复制定义；不得生成新字段、重排或重算。
old27=one('2.7 样本分布及数据特征')
old28=one('2.8 本章小结')
old27.text='2.8 样本分布及数据特征'
old28.text='2.9 本章小结'
old28_body=next(p for p in d.paragraphs if p.text.startswith('本章界定了研究对象、10项正式任务及77个严格特征。'))
old28_body.text=('本章界定了研究对象、10项正式任务和77项严格输入，完整列示各字段及其构造依据；'
                 '同时梳理了历史环境数据扩充的来源、覆盖和验证边界。统一的观测标识、折内预处理及来源分类'
                 '为后续模型比较、消融与解释提供可追溯基础，候选数据的探索性结果不替代正式站点留出结论。')
old27.insert_paragraph_before(
    '正式77项变量的精确字段名、中文含义、数据来源和时间尺度见表2-3。为保持训练输入的冻结顺序，'
    '下表按manifest逐项排列，以类别列标明六个来源组，而不按组别重新排序；序号即模型输入列序号。'
    '附录G在相同顺序下另保留单位和更完整的构造说明。',style='Normal')
old27.insert_paragraph_before('表2-3 77项严格输入特征的名称、来源与构造',style='表题')
feature_rows=[['序号','类别','精确英文变量名','中文名称','数据来源','时间尺度或计算方式']]
codes={'METEOROLOGY':'MET','RAINFALL_HYDRO_TIDE':'RHT','OCEAN_PHYSICS':'OCN',
       'HUMAN_ACTIVITY':'HUM','TEMPORAL_PROCESS':'TMP','SPATIAL_STATIC':'SPA'}
for idx,row in enumerate(appendix.rows[1:],1):
    c=row.cells
    feature_rows.append([str(idx),c[3].text,c[1].text,c[2].text,c[5].text,c[6].text])
assert [r[2] for r in feature_rows[1:]]==fm.feature.tolist()
assert Counter(r[1] for r in feature_rows[1:])==Counter(codes[x] for x in mp.source_group)
tab=d.add_table(rows=len(feature_rows),cols=6)
for i,values in enumerate(feature_rows):
    for j,value in enumerate(values):tab.rows[i].cells[j].text=value
ready_table(tab,[0.8,1.3,3.65,3.0,2.1,5.6],8.0)
old27._p.addprevious(tab._tbl)
old27.insert_paragraph_before(
    '注：MET为气象，RHT为降雨—水文—潮汐，OCN为海洋物理，HUM为人类活动代理，TMP为时间过程，SPA为空间静态。'
    '表内“前1次采样”等滞后量基于同站前序采样记录；rain_month为整月累计，潮差的业务时点可得性仍需核查。',style='Normal')

# 2.7：先用论文语言说明证据层级，再呈现完整候选来源表。
new27=old27.insert_paragraph_before('2.7 候选环境数据的探索与筛选',style='Heading 2')
new27.paragraph_format.page_break_before=True
old27.insert_paragraph_before(
    '在确定统一的严格特征集之前，研究曾探索海洋再分析与生物地球化学场、光学卫星观测、'
    '近岸空间结构及额外水文气象代理。扩充的目的在于检验不同来源能否为近岸水质任务提供'
    '超出已有气象、降雨、海温、夜间灯光与站点位置的信息。候选数据首先接受站点—日期匹配和缺失审查；'
    '只有完成预测对照的来源才讨论性能变化。空间格网或月产品与站点观测存在尺度差异，'
    '近岸光学产品亦可能受有效像元与复杂水色条件限制，这些是解释负迁移的可能因素而非本研究已证实的原因[16-17]。',
    style='Normal')
old27.insert_paragraph_before(
    '历史年份留出A/B与本文正式五折留站评价的样本和验证口径不同，表2-4中的历史指标仅用于说明'
    '当时的候选筛选过程，不能与第四章正式预测结果直接并列。以Chl-a中层为例，CMEMS BGC候选'
    '在历史测试集上使R²由0.3646升至0.3928、RMSE由3.8751降至3.7879、MAE由2.4261降至2.2885'
    '（训练5,988、测试1,788），但验证R²由0.2467降至0.2161；CMEMS Physics候选的测试R²由'
    '0.3646升至0.3772、RMSE由3.8751降至3.8363、MAE由2.4261降至2.3739，验证R²则由'
    '0.2467降至0.2156。两者具有局部正信号，却不构成跨任务稳定增益。',style='Normal')
old27.insert_paragraph_before(
    '与此相对，NASA OceanColor月尺度候选在Chl-a表层历史测试集（训练6,882、测试2,052）中的'
    'R²由0.3298降至0.2232，RMSE由5.1970升至5.5952，MAE由3.0433升至3.3046；'
    'Sentinel-2 patch同任务测试R²由0.3298降至0.2821。岸线距离在TN表层测试集使R²由'
    '0.5201小幅升至0.5245，RMSE由0.3411降至0.3396，MAE由0.2473降至0.2440'
    '（训练6,882、测试2,052），但这一局部变化不足以支持统一入模。以上仅描述原历史A/B的结果，'
    '不将月尺度光学或静态地理代理概括为对所有任务“无效”。',style='Normal')
old27.insert_paragraph_before(
    '另有数据源只达到获取、字段构造或覆盖验证阶段。例如，Sentinel-3 OLCI候选表可定位部分光学字段，'
    '但本次未找到可独立归属的正式预测A/B；GEBCO未形成稳定的研究区数值特征；GPM IMERG完成'
    '产品发现而样本下载受账户条款阻断；CHIRPS和ERA5-Land主要停留在可行性调查。'
    '这些来源不应写成“预测无效”。正式模型最终仍采用预先冻结的77项统一输入，'
    '其中VIIRS夜间灯光六项相关字段明确保留。对于未进入正式输入的候选，仅在表2-4与核验报告中保留其探索证据。',
    style='Normal')

# 横向表页：保留每个来源的验证口径，详细原始路径见独立CSV。
portrait=deepcopy(d.sections[0]._sectPr)
landscape=deepcopy(portrait)
size=landscape.find(qn('w:pgSz'))
size.set(qn('w:w'),'16838');size.set(qn('w:h'),'11906');size.set(qn('w:orient'),'landscape')
mar=landscape.find(qn('w:pgMar'))
mar.set(qn('w:left'),'950');mar.set(qn('w:right'),'950')
for sect in (portrait,landscape):
    typ=sect.find(qn('w:type'))
    if typ is None:typ=OxmlElement('w:type');sect.insert(0,typ)
    typ.set(qn('w:val'),'nextPage')
break1=old27.insert_paragraph_before('')
break1._p.get_or_add_pPr().append(portrait)
old27.insert_paragraph_before('表2-4 历史候选环境数据的来源、覆盖与筛选结果',style='表题')
heads=['数据来源','代表字段','分辨率／有效覆盖','阶段与验证口径','入模情况','性能变化','最终处理','证据编号']
matrix=[heads]
for i,c in enumerate(candidates,1):
    matrix.append([c['data_source'],c['representative_fields'].replace('; ','、'),
                   c['spatiotemporal_resolution']+'；'+c['effective_coverage'],
                   c['experiment_stage']+'；'+c['validation_protocol'],c['entered_model'],
                   c['performance_change'],c['final_treatment'],f'E{i:02d}'])
ct=d.add_table(rows=len(matrix),cols=8)
for i,values in enumerate(matrix):
    for j,value in enumerate(values):ct.rows[i].cells[j].text=value
ready_table(ct,[2.55,3.0,3.5,3.9,2.1,4.3,5.1,1.25],8.5)
old27._p.addprevious(ct._tbl)
old27.insert_paragraph_before(
    '注：表内历史A/B均非第四章正式站点五折结果；E01—E17的完整文件路径、覆盖与状态见随文核验表。'
    '“未定位到A/B”表示证据不足，不能据此判断预测无效；VIIRS为正式77项之一。',style='Normal')
break2=old27.insert_paragraph_before('')
break2._p.get_or_add_pPr().append(landscape)

# 引用仅支撑近岸光学观测的背景限制，不拿文献替代本地A/B证据。
appendix_a=one('附录A 解释分析覆盖范围')
appendix_a.insert_paragraph_before(
    '[16] LANG S E, LUIS K M A, DONEY S C, et al. Modeling Coastal Water Clarity Using Landsat-8 and Sentinel-2[J]. '
    'Earth and Space Science, 2023, 10(7): e2022EA002579. DOI: 10.1029/2022EA002579.',style='Normal')
appendix_a.insert_paragraph_before(
    '[17] WOŹNIAK S B, MELER J. Modelling Water Colour Characteristics in an Optically Complex Nearshore Environment '
    'in the Baltic Sea; Quantitative Interpretation of the Forel-Ule Scale and Algorithms for the Remote Estimation '
    'of Seawater Composition[J]. Remote Sensing, 2020, 12(17): 2852. DOI: 10.3390/rs12172852.',style='Normal')

# V3原本没有表目录；增设独立目录，页码在最终PDF定位后填入。
figure_dir=one('图目录')
heads1=[p for p in d.paragraphs if p.style.name=='Heading 1']
next_head=heads1[next(i for i,p in enumerate(heads1) if p._p is figure_dir._p)+1]
next_head.insert_paragraph_before('表目录',style='Heading 1')
for caption in [p.text for p in d.paragraphs if p.style.name=='表题']:
    p=next_head.insert_paragraph_before(caption+'\t待核')
    p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.0
    p.paragraph_format.first_line_indent=Pt(0)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(15.0),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
    for run in p.runs:run.font.size=Pt(9)

d.save(TARGET)
print('WORD_WORKING_COPY_UPDATED',TARGET,'features',len(tab.rows)-1,'candidates',len(ct.rows)-1,
      'tables',len(d.tables),'sections',len(d.sections))
