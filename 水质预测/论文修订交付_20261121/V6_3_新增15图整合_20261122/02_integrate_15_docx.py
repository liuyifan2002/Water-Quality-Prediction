"""在V6.3副本追加15张图；源DOCX保持不变。"""
from pathlib import Path
import re
from docx import Document
from docx.shared import Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'论文修订交付_20261121/V6_3_图件扩展_20261122/硕士学位论文_V6.3_图件优化稿.docx'
TARGET=HERE/'水质预测硕士学位论文_V6.3_新增15图整合稿.docx'
FIG=HERE/'figures'
assert SOURCE.exists(),SOURCE
DOC=Document(SOURCE)
SPECS=[
('2-7','2.10 ','十项预测任务在十个水区的样本数量与站点覆盖','十任务水区样本站点覆盖','图2-7同时呈现十项任务在各水区的有效样本数和独立站点数。部分水区的样本支持明显少于其他水区，后续站点泛化结果应结合这一分布理解。','格内上行为有效样本数、下行为独立站点数；样本来自正式十任务冻结测试记录。'),
('2-8','2.10 ','十项预测任务年度有效观测样本量变化','十任务年度有效观测','图2-8展示2014—2024年各任务有效观测数量的变化。各面板采用相同年份范围，便于识别不同指标和水层的监测覆盖差异。','柱高为对应任务与采样年份的有效观测数；未用填补后的特征矩阵代替观测。'),
('2-9','2.10 ','十项预测任务逐站点有效观测频次分布','逐站点有效观测频次','图2-9以站点为单位展示各任务的有效观测频次。箱体表示站点间分布，散点保留每个站点的实际记录数。','每个点为一个站点；箱体表示四分位范围，中线为中位数。'),
('2-10','2.10 ','四类水质指标月尺度观测变化特征','四指标月尺度观测变化','图2-10显示十项任务按采样月份分组后的实测值变化。月度中位数与四分位范围提供描述性比较，图中同步给出各月有效样本量。','曲线为月度中位数，阴影为四分位范围；仅展示实际有观测的月份，不作插值。'),
('3-4','3.2 ','随机划分、年份留出与站点留出验证方案对比','三种验证方案对比','图3-4区分随机样本、未参与训练的时间段和未见站点三种测试对象。本文正式十任务结果主要依据五折站点留出，因此评价重点是未见站点的空间泛化。','流程为验证对象示意，未设定未经登记的划分比例；站点留出按冻结五折轮换。'),
('3-5','3.2 ','逐样本预测结果的多层级评价与统计汇总流程','逐样本多层级评价流程','图3-5概括逐样本OOF预测到总体、折次及站点评价的汇总路径。配对站点Bootstrap基于两模型相同测试对象的误差差值，不改变正式预测结果。','pooled指标在合并五折OOF预测后计算；折次、站点和Bootstrap分别回答不同层级的问题。'),
('4-14','4.6 ','十项预测任务的分箱预测—观测一致性曲线','分箱预测观测一致性','图4-14将两模型的OOF预测分别按预测值等频分箱，比较箱内平均预测值和平均实测值。各面板的1:1线用于观察不同预测范围内的系统偏离。','每模型按预测值独立等频10箱，遇重复值自动减少箱数；该图不是独立概率校准实验。'),
('4-15','4.6 ','十项预测任务逐年预测误差变化','十任务逐年预测误差','图4-15按真实采样年份汇总两模型的留站测试MAE。年度曲线揭示同一空间验证协议下不同年份的误差波动，不能当作年份留出实验结果。','每年样本量标在横轴下方；纵轴为任务自身目标单位，缺失年份不插值。'),
('4-16','4.6 ','十项预测任务不同实测浓度区间的绝对误差比较','浓度分位区间绝对误差','图4-16按实测值将每项任务的共同测试样本分为五个分位区间，比较两模型MAE。高浓度区间的误差可与图4-13的残差形态结合判断。','两模型共用实测值分箱；横轴下方为每箱样本量，纵轴使用该任务原始单位。'),
('4-17','4.6 ','站点观测数量与模型预测误差改善幅度的关系','站点样本数与误差变化','图4-17将各站点测试样本数与TabICLv2相对XGBoost的RMSE变化率对应展示。零线以下表示TabICLv2的站点误差较低，散点关系仅作描述。','相对变化率=100%×(RMSE_TabICLv2−RMSE_XGBoost)/RMSE_XGBoost；每点为一个有效站点。'),
('4-18','4.6 ','十项预测任务站点级相对误差变化的跨任务相关性','站点相对误差变化跨任务相关性','图4-18检验站点级模型改善与退化是否在不同任务之间具有一致性。相关系数仅由实际共同站点计算，并在下方面板列出每对任务的配对站点数。','站点效应为log₂(RMSE_TabICLv2/RMSE_XGBoost)；上方面板为Spearman相关，下方面板为共同有效站点数。'),
('5-8','5.6 ','四项代表任务六类环境信息来源消融效应的五折分布','六来源消融五折效应','图5-8保留六来源消融在四项代表任务中每一折的实际ΔR²。折间变化说明pooled结果之外仍需关注留站划分下的稳定程度。','ΔR²=删组模型R²−完整模型R²；负值表示删除来源后预测性能下降。'),
('5-9','5.6 ','六类环境信息来源消融的双指标效应象限图','六来源双指标消融效应','图5-9同时展示删组后的pooled ΔR²与相对ΔRMSE。两指标方向一致的来源更容易辨认，方向不一致处应保留为不同指标的真实差异。','纵轴=100%×(删组RMSE−完整RMSE)/完整RMSE；仅反映XGBoost模型的来源级预测变化，非环境因果效应。'),
('5-10','5.6 ','历史分区模型的局地模型采用与全域回退决策分布','历史分区局地模型与全域回退','图5-10依据历史分区决策原始记录，展示12个任务组合在各水区采用区域模型或回退全域模型的实际次数。532条记录中，496次采用区域模型，36次全域回退。','历史78特征扩展协议，含water_zone_code；格内为回退次数／决策总数，不与正式77特征十任务结果合并。'),
('5-11','5.6 ','六类环境信息来源在不同水区的真实观测覆盖率','六来源各水区真实观测覆盖','图5-11将填补前真实非缺失率按来源组与水区汇总，显示多源输入的空间覆盖差异。热图反映数据支持，不表示预测效应或环境响应。','每格依次为组内特征真实非缺失率中位数及最低值；依据冻结缺失掩码，不计填补值。'),
]
def heading(prefix):
    matches=[p for p in DOC.paragraphs if p.text.strip().startswith(prefix) and '\t' not in p.text and len(p.text)<100]
    assert matches,prefix
    return matches[-1]
for num,section,title,stem,intro,note in SPECS:
    path=FIG/f'图{num}_{stem}.png';assert path.exists(),path
    anchor=heading(section)
    p=anchor.insert_paragraph_before(intro,style='Normal');p.paragraph_format.page_break_before=True;p.paragraph_format.keep_with_next=True
    pic=anchor.insert_paragraph_before(style='Normal');pic.alignment=WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.line_spacing_rule=WD_LINE_SPACING.SINGLE;pic.paragraph_format.line_spacing=1.0;pic.paragraph_format.keep_with_next=True
    with Image.open(path) as im:
        w=min(15.3,19.2*im.width/im.height)
    pic.add_run().add_picture(str(path),width=Cm(w))
    cap=anchor.insert_paragraph_before(f'图{num} {title}',style='图题');cap.paragraph_format.keep_with_next=True
    anchor.insert_paragraph_before('注：'+note,style='Normal')

# 在现有静态图目录中按编号插入条目；PDF导出后回填页码。
def order(num):
    m=re.match(r'(\d+|[A-Z])-(\d+)',num)
    return (int(m[1]) if m[1].isdigit() else 10+ord(m[1])-65,int(m[2]))
def next_entry(num):
    ps=DOC.paragraphs;start=max(i for i,p in enumerate(ps) if p.text.strip()=='图目录')
    end=next(i for i in range(start+1,len(ps)) if ps[i].text.strip()=='表目录')
    for p in ps[start+1:end]:
        m=re.match(r'^图(\d+-\d+|[A-Z]-\d+)\b',p.text.strip())
        if m and order(m[1])>order(num):return p
    return ps[end]
for num,_,title,*_ in SPECS:
    next_entry(num).insert_paragraph_before(f'图{num} {title}\t待更新',style='Normal')
DOC.save(TARGET)
print(TARGET,len(SPECS),'inserted')
