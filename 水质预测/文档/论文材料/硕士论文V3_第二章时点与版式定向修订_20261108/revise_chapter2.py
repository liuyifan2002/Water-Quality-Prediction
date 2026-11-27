"""基于现有V3工作副本，仅定向修改第二章及表2-4；不改冻结数据。"""
from pathlib import Path
import sys
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

root=Path(r'E:\研究生\codex\水质预测')
src=root/'文档/论文材料/硕士论文V3_第二章特征与候选数据扩充_20261101/水质预测硕士学位论文_V3_第二章特征与候选数据扩充_工作副本.docx'
out=Path(__file__).resolve().parent/'水质预测硕士学位论文_V3_第二章定向修订_20261108.docx'
assert src.is_file() and (not out.exists() or '--rebuild' in sys.argv), '目标文件已存在，停止以免覆盖'
d=Document(src)
def paragraph(fragment):
    hits=[p for p in d.paragraphs if fragment in p.text and p.style.name=='Normal']
    assert len(hits)==1,(fragment,len(hits))
    return hits[0]

# 第23页仅替换读者指引；表2-3的77个实际字段及附录G保持不变。
p=paragraph('表 2-2 对六类变量的数量')
old='表 2-2 对六类变量的数量、代表特征与信息作用进行归纳，完整 77 项字段及其定义列于附录 G。'
new='表2-2归纳了六类特征的组成及其信息作用，77项正式输入变量的精确字段名、中文含义、数据来源和时间尺度见表2-3，附录G进一步列出了各变量的单位及详细构造说明。'
assert old in p.text
p.text=p.text.replace(old,new)

# 将原先“待核”的整月雨量时点问题改为已由原始日雨量汇总代码证实的事实。
p=paragraph('rain_month为采样所在日历月的整月累计降雨量')
old_rain='rain_month为采样所在日历月的整月累计降雨量，其观测当日可得性仍须核查。'
new_rain='rain_month为采样所在日历月的整月累计降雨量；对于月末之前的采样日，该变量包含采样日之后的降雨。'
assert old_rain in p.text
p.text=p.text.replace(old_rain,new_rain)

# 避免紧随表2-2后的新旧两句重复罗列“77项+附录G”。
p=paragraph('正式77项变量的精确字段名')
p.text='为保持冻结的训练输入顺序，表2-3依序列示各字段，以类别列标明六个来源组；序号对应模型输入列序号。'

# 方法定位由脚本、源表和官方产品定义支持；不写成实时预报。
anchor=paragraph('为保持冻结的训练输入顺序')
timing=anchor.insert_paragraph_before(
    '时点核查表明，rain_3d与rain_14d仅使用采样日前的自然日；rain_month、ERA5的t2m、tcc、u10、v10以及VIIRS夜间灯光则按采样所在完整自然月匹配，月末以前的记录因而含有采样日期之后的环境信息。本文据此将正式任务界定为利用回溯性环境资料的同日水质估计及未见站点泛化评价，不将其表述为采样时点的实时预报。tidal_range取全年天文潮汐预报表中的当日高、低潮预报高度极差，并非采样时刻的实测潮位；环境滞后项按同站较早的站点—日期记录构建，各水层共享站点环境值。',
    style='Normal')
timing.paragraph_format.keep_with_next=False

note=paragraph('表内“前1次采样”')
note.text=note.text.replace('潮差的业务时点可得性仍需核查。','潮差来自当日天文潮汐预报表，历史归档的实际发布时间尚待核实。')

# 表2-4只压缩重复措辞；性能数值、覆盖率、17条来源和证据编号全部保留。
table=next(t for t in d.tables if len(t.rows)==18 and len(t.columns)==8 and t.cell(0,0).text=='数据来源')
for rowno,row in enumerate(table.rows[1:],1):
    cells=row.cells
    stage=cells[3].text
    for a,b in [('已提取并完成历史A/B；','历史A/B；'),('已构造并完成历史A/B；','历史A/B；'),('已构建并完成历史A/B；','历史A/B；'),('历史年份留出；',''),('；非正式站点五折','')]:
        stage=stage.replace(a,b)
    cells[3].text=stage
    if rowno==17:
        cells[4].text='是（6项）'
    else:
        cells[4].text='否'
    assert cells[7].text==f'E{rowno:02d}'

decisions={
1:'局部改善、验证退化；未统一纳入',
2:'局部改善、验证退化；未统一纳入',
3:'月尺度单源未见稳定增益；未纳入',
4:'保留光学候选；预测效应待核',
5:'覆盖有限；未统一纳入',
6:'局部正值不足以统一纳入',
7:'来源与覆盖已核；预测待核',
8:'局部改善；未统一纳入',
9:'近似水深；未统一纳入',
10:'保留获取缺口；区别于站点近似水深',
11:'保留历史探索；并非瞬时潮位',
12:'待获取；不替代HKO降雨',
13:'备选来源',
14:'备选来源；未作正式预测',
15:'未取得；不写预测结果',
16:'未取得；不写预测结果',
17:'保留于正式77项输入',
}
for rowno in range(1,18):table.rows[rowno].cells[6].text=decisions[rowno]
table.rows[0].cells[4].text='正式入模'
widths=[2.5,3.2,3.7,3.7,1.6,4.5,5.0,1.5]
assert abs(sum(widths)-25.7)<1e-8
table.autofit=False
for col,width in zip(table.columns,widths):col.width=Cm(width)
for i,row in enumerate(table.rows):
    trpr=row._tr.get_or_add_trPr()
    if trpr.find(qn('w:cantSplit')) is None:trpr.append(OxmlElement('w:cantSplit'))
    if i==0 and trpr.find(qn('w:tblHeader')) is None:trpr.append(OxmlElement('w:tblHeader'))
    for j,cell in enumerate(row.cells):
        cell.width=Cm(widths[j])
        tcpr=cell._tc.get_or_add_tcPr()
        mar=tcpr.find(qn('w:tcMar'))
        if mar is None:mar=OxmlElement('w:tcMar');tcpr.append(mar)
        for edge in ('left','right'):
            el=mar.find(qn('w:'+edge))
            if el is None:el=OxmlElement('w:'+edge);mar.append(el)
            el.set(qn('w:w'),'75');el.set(qn('w:type'),'dxa')
        for p in cell.paragraphs:
            f=p.paragraph_format
            f.space_before=Pt(0);f.space_after=Pt(0);f.line_spacing=1.0
            f.first_line_indent=Pt(0);f.left_indent=Pt(0);f.right_indent=Pt(0)
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER if i==0 or j in (4,7) else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:run.font.size=Pt(9)

p=paragraph('E01—E17的完整文件路径')
p.text=('注：表内历史A/B采用当时的年份留出探索口径，均非第四章正式站点五折结果；“正式入模”指是否属于X_strict_77。'
        'E01—E17的完整文件路径、覆盖与状态见随文核验表；未定位到A/B不表示预测无效。'
        'VIIRS夜间灯光为正式输入；增强潮汐与ERA5-Land候选未入模，但原tidal_range和ERA5变量仍保留。')

# 原表2-3之后的分页保留：现有自然接续试排导致末段孤立于横向表前一页。
heading=next(p for p in d.paragraphs if p.text=='2.7 候选环境数据的探索与筛选' and p.style.name=='Heading 2')
assert heading.paragraph_format.page_break_before is True
d.save(out)
print('REVISED',out,'tables',len(d.tables),'formal captions',sum(p.style.name=='表题' for p in d.paragraphs))
