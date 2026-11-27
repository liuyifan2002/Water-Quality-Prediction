"""以V6.1科研内容和V6.2已修复排版为底稿，插入冻结结果补充图。"""
from pathlib import Path
from docx import Document
from docx.shared import Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_LINE_SPACING

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'V6_2_图表修复与全文一致性修订'/'硕士学位论文_V6.2.docx'
DOC=Document(BASE)

def insert_before_heading(heading,intro,img_name,caption,note):
    matches=[p for p in DOC.paragraphs if p.text.strip()==heading]
    assert len(matches)==1,(heading,len(matches))
    anchor=matches[0]
    p=anchor.insert_paragraph_before(intro,style='Normal')
    p.paragraph_format.keep_with_next=False
    pic=anchor.insert_paragraph_before(style='Normal')
    pic.alignment=WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.line_spacing_rule=WD_LINE_SPACING.SINGLE
    pic.paragraph_format.line_spacing=1.0
    pic.paragraph_format.page_break_before=True
    pic.paragraph_format.keep_with_next=True
    pic.add_run().add_picture(str(HERE/'正式论文图片'/img_name),width=Cm(15.5))
    cap=anchor.insert_paragraph_before(caption,style='图题')
    cap.paragraph_format.keep_with_next=True
    anchor.insert_paragraph_before(note,style='Normal')

insert_before_heading(
 '4.6 本章小结',
 '为了区分总体R²变化与误差指标变化，图4-6进一步给出十任务RMSE、MAE相对XGBoost的变化。两项指标在TN中层呈现不同方向：RMSE略增而MAE下降，说明少数较大残差对RMSE的影响不能被平均绝对误差概括。其余任务也应将相对变化与表4-2的原始单位数值及站点重抽样区间结合解读。',
 '图4-6_十任务RMSE与MAE相对变化.png',
 '图4-6 十项任务RMSE与MAE相对XGBoost的变化',
 '注：相对变化=100%×（TabICLv2误差−XGBoost误差）/XGBoost误差；负值表示TabICLv2误差降低。两面板均由同一版本的pooled OOF误差确定性换算，不代表Bootstrap区间。')

insert_before_heading(
 '5.6 本章小结',
 '图5-5按任务自身单位展开六来源删组后的RMSE变化。Chl-a表层和Secchi删除空间静态信息后RMSE增加，与图5-1中的R²下降方向一致；TN表层及TP中层的来源响应则更分散。由于来源组间可相互提供代理信息，单组消融度量的是当前XGBoost配置下的增量预测价值，不能据此确定单一环境过程的因果作用。',
 '图5-5_四任务六来源删组RMSE变化.png',
 '图5-5 四项代表任务六来源删组后的RMSE变化',
 '注：ΔRMSE=删组模型RMSE−完整模型RMSE；正值表示删组后误差增加。四任务分别使用自身目标单位和独立坐标范围，来源顺序一致。该图与图5-1分别显示误差变化和pooled R²变化，均为正式77特征XGBoost消融。')

# 附录B完整展示十任务站点误差，不选择性删除退化点。
anchor=next(p for p in DOC.paragraphs if p.text.strip()=='个人简历')
h=anchor.insert_paragraph_before('附录B 十项任务逐站点预测误差补充',style='Heading 2')
h.paragraph_format.page_break_before=True
anchor.insert_paragraph_before('图B-1至图B-4按指标展示十项正式任务全部有效站点的配对RMSE差值。站点经度只表示东西向位置；各面板中的点为真实站点，不是样本重抽样或预测置信区间。',style='Normal')
for i,name in enumerate(['Chl-a','Secchi','TN','TP'],1):
    pic=anchor.insert_paragraph_before(style='Normal')
    pic.alignment=WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.line_spacing_rule=WD_LINE_SPACING.SINGLE
    pic.paragraph_format.line_spacing=1.0
    pic.paragraph_format.keep_with_next=True
    if i>1:pic.paragraph_format.page_break_before=True
    pic.add_run().add_picture(str(HERE/'正式论文图片'/f'图B-{i}_{name}逐站点RMSE变化.png'),width=Cm(15.5))
    cap=anchor.insert_paragraph_before(f'图B-{i} {name}任务逐站点RMSE变化',style='图题')
    cap.paragraph_format.keep_with_next=True
    anchor.insert_paragraph_before('注：ΔRMSE=TabICLv2站点RMSE−XGBoost站点RMSE；负值表示TabICLv2误差降低。使用正式77特征统一站点留出结果，未对站点进行筛选。',style='Normal')

for p in DOC.paragraphs:
    if p.text.startswith('图4-4展示Chl-a表层、Secchi、TN表层和TP中层的站点RMSE变化空间分布'):
        p.add_run(' 十任务所有有效站点的误差变化详见附录B图B-1至图B-4。')

DOC.save(HERE/'硕士学位论文_V6.2.docx')
