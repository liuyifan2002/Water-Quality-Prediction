"""把已绘制的V6.3新增图插入V6.2副本；不改变冻结实验表或既有图号。"""
from pathlib import Path
from shutil import copy2
from docx import Document
from docx.shared import Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
FIG=HERE/'正式论文图片'
OLD=ROOT/'论文修订交付_20261121/V6_2_图片体系优化_20261122'
OLD1=ROOT/'论文修订交付_20261121/V6_1_科研深化与全文重建'
SOURCE=OLD/'硕士学位论文_V6.2.docx'
TARGET=HERE/'硕士学位论文_V6.3_图件优化稿.docx'
for base in [OLD1/'正式论文图片',OLD/'正式论文图片']:
 for p in base.glob('*.png'):
  dest=FIG/p.name
  if not dest.exists():copy2(p,dest)
DOC=Document(SOURCE)
from PIL import Image
# 现有2-2至2-4按原图位序替换内嵌媒体，维持原图号并保留可编辑DOCX。
for idx,name in {2:'图2-2_水质监测时间覆盖_修订.png',3:'图2-3_数据接入与样本主键_修订.png',4:'图2-4_四指标观测分布_修订.png'}.items():
 shape=list(DOC.inline_shapes)[idx]
 rid=shape._inline.graphic.graphicData.pic.blipFill.blip.embed
 data=(FIG/name).read_bytes()
 DOC.part.related_parts[rid]._blob=data
 with Image.open(FIG/name) as im:
  shape.width=Cm(15.5)
  shape.height=Cm(15.5*im.height/im.width)

SPECS=[
 ('2-5','2.10 ','77项严格输入特征的来源组成与信息维度','77项输入特征来源组成','图2-5按正式特征清单汇总六类输入：MET 29项、RHT 22项、OCN 11项、HUM 6项、TMP 7项、SPA 2项。图中“信息表征”描述预测变量类别，不能据此推断环境因果机制。','数据：WQ77_REPAIRED_20261115_V1正式77项特征清单；夜光属于人类活动代理。'),
 ('2-6','2.10 ','多源特征时间窗口构造示意','多源特征时间窗口构造','图2-6区分自然日统计窗口、同站前序采样滞后与自然月尺度变量。整月降雨和月尺度夜光属于历史同期匹配，采样时刻未必已获得整月信息，因此本文结果不能表述为严格实时提前预测。','窗口位置为示意；自然月起点与采样日的相对距离随日期而变。'),
 ('3-1','3.1.4 ','四类预测模型的原理与预测流程','四类预测模型原理与流程','图3-1概括四种模型在统一77项输入条件下的预测流程。XGBoost与TabICLv2覆盖十项正式任务，TabPFN与TabM仅参与四项代表任务扩展比较。','TabM在当前任务数据上训练；图中不推断各模型未公开的内部结构。'),
 ('3-2','3.2 ','五折站点留出验证流程','五折站点留出验证流程','图3-2展示站点留出划分。Chl-a、TN、TP共用84站的五折划分；Secchi使用自身94站划分。每折测试站点与训练站点互斥，五折测试预测合并构成OOF集合。','Secchi各折测试站点数直接由冻结预测文件统计。'),
 ('3-3','3.2 ','Pooled OOF与Fold Mean R²计算差异','pooled与fold均值R2口径','图3-3说明pooled OOF R²与五折R²算术平均的计算顺序。本文正式十任务的总体性能以合并全部OOF测试预测后统一计算的pooled指标为准。','两种指标的样本权重和目标方差处理不同，数值无需相同。'),
 ('4-7','4.6 ','十项任务×十水区的空间泛化表现','十任务水区空间泛化热图','图4-7将逐站点RMSE比值按真实水区归属汇总。负的log₂比值代表TabICLv2误差较低，正值代表误差较高；格内数字为有效站点数。水区汇总提供十任务空间差异视角，与图4-4的四任务位置分布互补。','每格为有效站点log₂(RMSE_TabICLv2/RMSE_XGBoost)的中位数；空格为无有效站点。'),
 ('4-8','4.6 ','Chl-a、TN、TP垂向预测性能剖面','三指标垂向预测性能剖面','图4-8以同一pooled OOF R²横轴比较Chl-a、TN、TP在表、中、底层的模型表现。不同水层的预测难度和模型差异并不一致；Secchi是单一任务，不纳入垂向剖面。','全部点值来自修复版77特征正式十任务指标。'),
 ('4-9','4.6 ','十项任务实测值与预测值经验累积分布','十任务实测与预测经验累积分布','图4-9比较十项任务实测值与两模型OOF预测值的经验累积分布，补充观察各指标高值尾部的再现情况。分布接近不等于逐样本预测准确。','两模型按同一obs_id配对；每项任务采用自身水质单位与横轴范围。'),
 ('4-10','4.6 ','十项任务五折性能稳定性','十任务五折性能稳定性','图4-10呈现各任务自身五折的TabICLv2与XGBoost R²差值。总体pooled结果不能代替逐折观察；图中保留负值和异常折次。','不同任务的fold编号不一定对应同一组测试站点，格内为冻结折次R²直接相减。'),
 ('4-11','4.6 ','十项任务预测误差改善及置信区间','十任务误差变化及置信区间','图4-11将已有1000次配对站点Bootstrap的ΔRMSE和ΔMAE区间换算为相对XGBoost基准误差的百分比。负值表示TabICLv2误差较低；区间跨零时不作明确优势判断。','相对变化=100%×(TabICLv2−XGBoost)/XGBoost；使用已有绝对差值区间除以同任务冻结基准误差，无新增重抽样。'),
 ('4-12','4.6 ','十项任务实测值与预测值对比','十任务实测预测对比','图4-12逐任务展示两模型的全部配对OOF预测与实测值。1:1线便于观察不同浓度区间的偏离，高浓度样本的分散程度应结合图4-13的残差进一步判断。','每个面板使用自身目标单位与独立坐标范围；全部点来自同obs_id冻结OOF预测。'),
 ('4-13','4.6 ','十项任务预测残差分布','十任务残差分布','图4-13以预测值减实测值为残差，展示十任务在不同观测浓度范围的误差。残差形态反映预测表现，不能直接作为环境过程的因果证据。','两模型按相同obs_id配对；零线表示无残差，各面板使用任务自身单位。'),
 ('5-7','5.6 ','77项特征逐年真实观测覆盖','77特征逐年真实观测覆盖','图5-7按正式清单顺序展示77项特征在2014—2024年各年的填补前非缺失率，补充图5-4的来源级覆盖概览。热图的空缺不表示模型输入矩阵为空，而表示原始记录在填补前缺少真实值。','依据正式任务五折测试样本的布尔缺失掩码合并；同一obs_id跨任务掩码一致，重复记录只计一次。')]

def target_heading(prefix):
 matches=[p for p in DOC.paragraphs if p.text.strip().startswith(prefix) and '\t' not in p.text and len(p.text)<100]
 # 前面的目录条目带页码并通常带制表符，保留正文最后一项。
 assert matches,(prefix,'heading missing')
 return matches[-1]

def insert(spec):
 num,heading,title,stem,intro,note=spec
 pth=next(FIG.glob(f'图{num}_{stem}.png'),None) if num!='5-7' else FIG/'图5-7(a)_77特征逐年覆盖_a.png'
 assert pth and pth.stat().st_size>10000,(num,pth)
 anchor=target_heading(heading)
 intro_p=anchor.insert_paragraph_before(intro,style='Normal')
 intro_p.paragraph_format.keep_with_next=False
 pic=anchor.insert_paragraph_before(style='Normal');pic.alignment=WD_ALIGN_PARAGRAPH.CENTER
 pic.paragraph_format.line_spacing_rule=WD_LINE_SPACING.SINGLE
 pic.paragraph_format.line_spacing=1.0
 pic.paragraph_format.keep_with_next=True;pic.paragraph_format.page_break_before=num in {'4-9','4-12','4-13','5-7'}
 pic.add_run().add_picture(str(pth),width=Cm(15.5))
 cap=anchor.insert_paragraph_before(f'图{num} {title}',style='图题');cap.paragraph_format.keep_with_next=True
 anchor.insert_paragraph_before('注：'+note,style='Normal')
 if num=='5-7':
  second=FIG/'图5-7(b)_77特征逐年覆盖_b.png';assert second.exists()
  pic2=anchor.insert_paragraph_before(style='Normal');pic2.alignment=WD_ALIGN_PARAGRAPH.CENTER
  pic2.paragraph_format.line_spacing_rule=WD_LINE_SPACING.SINGLE;pic2.paragraph_format.line_spacing=1.0
  pic2.paragraph_format.page_break_before=True;pic2.paragraph_format.keep_with_next=True
  pic2.add_run().add_picture(str(second),width=Cm(15.5))
  anchor.insert_paragraph_before('图5-7（续） 77项特征逐年真实观测覆盖（后38项）',style='图题')
for spec in SPECS:insert(spec)

# 静态图目录按编号插入；页码在PDF导出后回填。
import re
def order(num):
 m=re.match(r'(\d+|[A-Z])-(\d+)',num)
 return (int(m[1]) if m[1].isdigit() else 10+ord(m[1])-65,int(m[2]))
def figure_directory_entry(num):
 ps=DOC.paragraphs;start=max(i for i,p in enumerate(ps) if p.text.strip()=='图目录')
 end=next((i for i in range(start+1,len(ps)) if ps[i].text.strip()=='表目录'),len(ps))
 for p in ps[start+1:end]:
  m=re.match(r'^图(\d+-\d+|[A-Z]-\d+)\b',p.text.strip())
  if m and order(m[1])>order(num):return p
 return ps[end]
for num,_,title,*_ in SPECS:
 d=figure_directory_entry(num).insert_paragraph_before(f'图{num} {title}\t待更新',style='Normal')
 d.paragraph_format.keep_with_next=False
DOC.save(TARGET)
print(TARGET,len(SPECS),'new figures')
