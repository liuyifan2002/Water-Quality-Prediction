"""基于原V4建立独立整合稿；不改原件，保留原文献和非目标图件。"""
from common import *
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
import re,copy

doc=Document(DOC);original=list(doc.paragraphs);body=doc._element.body
registry=json.loads((OUT/'05_数据核验/图件登记.json').read_text(encoding='utf-8'))
registry={k:v for k,v in registry.items() if k not in ['1-1','2-2','2-3','2-4']}
assert len(registry)==90,len(registry)
protect(DOC)
refs=[p.text for p in original if p.style.name=='参考文献']

def text(p,s):
    p.clear();p.add_run(s)

def remove(p):
    e=p._p
    if e.getparent() is not None:e.getparent().remove(e)

def after(anchor,s='',style='Normal'):
    e=OxmlElement('w:p');anchor._p.addnext(e);p=Paragraph(e,doc._body);p.style=style;p.add_run(s);return p

def figure(anchor,num):
    r=registry[num];p=after(anchor);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    pf=p.paragraph_format;pf.space_before=Pt(3);pf.space_after=Pt(2);pf.keep_with_next=True;pf.line_spacing=1
    p.add_run().add_picture(r['最终图片路径'],width=Cm(15.5))
    cap=after(p,r['新图号']+' '+r['图名'],'图题');cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next=True;cap.paragraph_format.space_after=Pt(2)
    for run in cap.runs:run.font.size=Pt(10)
    note=after(cap,'注：'+r['图注']);note.paragraph_format.line_spacing=1.05;note.paragraph_format.space_after=Pt(6);note.paragraph_format.first_line_indent=Pt(0);note.paragraph_format.keep_together=True
    for run in note.runs:run.font.size=Pt(9)
    r['是否插入Word']=True
    return note

# 清除旧的静态目录，后续由Word从真实标题和图题重新生成。
for lo,hi in [(38,149),(151,244),(246,257)]:
    for p in original[lo:hi+1]:remove(p)
def toc(anchor,code):
    p=after(anchor);r=p.add_run();b=OxmlElement('w:fldChar');b.set(qn('w:fldCharType'),'begin');r._r.append(b)
    ins=OxmlElement('w:instrText');ins.set(qn('xml:space'),'preserve');ins.text=code;r._r.append(ins)
    sep=OxmlElement('w:fldChar');sep.set(qn('w:fldCharType'),'separate');r._r.append(sep);p.add_run('更新目录后显示页码')
    end=OxmlElement('w:fldChar');end.set(qn('w:fldCharType'),'end');p.add_run()._r.append(end)
toc(original[37],' TOC \\o "1-3" \\h \\z \\u ')
toc(original[150],' TOC \\h \\z \\t "图题,1" ')
toc(original[245],' TOC \\h \\z \\t "表题,1" ')

# 评价方法连同原指标表移至第二章，避免在结果解释章重复配置。
start=original[365]._p;stop=original[370]._p;nodes=[];node=start
while node is not stop:nodes.append(node);node=node.getnext()
target=original[346]._p
for node in nodes:target.addprevious(node)
text(original[365],'2.8.6 模型性能与空间泛化评价指标');original[365].style='Heading 3'
text(original[369],'表2-6 主要评价指标及解释')
text(original[344],'2.8.5 历史分区与正式来源消融的协议区分')
text(original[345],'历史分海区扩展实验采用78项输入（含water_zone_code），覆盖12个目标×水层组合、60个任务折，区域样本不足时按冻结决策回退全域模型；它不属于正式77特征十任务主实验。来源消融采用正式77特征六来源分类，每次删除一个来源组，评价口径及模型配置沿用各自冻结实验；XGBoost消融不表述为TabICLv2内部贡献。')

mapping={'3-1':['3-1'],'3-2':['3-2'],'3-3':['3-3','3-4'],'3-4':['3-5','3-6'],'3-5':['3-7'],**{f'3-{i}':[f'3-{i+2}'] for i in range(6,14)},'3-14':['3-16'],'3-15':[],'3-16':[],'3-17':[],**{f'4-{i}':[f'4-{i}'] for i in range(1,5)},'5-1':['5-1'],'2-1':['2-1']}
for ch,n in [('C',20),('E',20),('F',10)]:mapping.update({f'{ch}-{i}':[f'{ch}-{i}'] for i in range(1,n+1)})
mapping.update({f'D-{i}':[f'D-{i}'] for i in range(1,11)});mapping.update({f'D-{i}':[] for i in range(11,16)});mapping.update({'D-16':['D-11'],'D-17':['D-12']})
for i,p in enumerate(original):
    if p.style.name!='图题':continue
    m=re.match(r'图([A-Z0-9]+-\d+)\s',p.text)
    if not m or m[1] not in mapping:continue
    old=m[1];anchor=p
    for num in mapping[old]:anchor=figure(anchor,num)
    if i and original[i-1]._p.xpath('.//w:drawing'):remove(original[i-1])
    remove(p)
    if i+1<len(original) and re.match(r'^(?:图注[：:]|注[：:])',original[i+1].text):remove(original[i+1])

# 更新正文引用采用一次正则替换，防止连续替换造成编号串改。
refmap={k:'、'.join('图'+v for v in vs) for k,vs in mapping.items() if vs}
refmap.update({'3-15':'图3-16','3-16':'图H-1','3-17':'图H-2'})
for p in original[260:]:
    if p._p.getparent() is None or p.style.name in ['图题','参考文献']:continue
    s=p.text
    s=re.sub(r'图([A-Z0-9]+-\d+)',lambda m:refmap.get(m[1],m[0]),s)
    s=s.replace('表3-1','表2-6').replace('表3-2','表3-1')
    if re.match(r'^3\.[3-6](?:\.|\s)',s):s=re.sub(r'^3\.([3-6])',lambda m:'3.'+str(int(m[1])-1),s)
    if s!=p.text:text(p,s)

updates={
376:'图3-1展示四项代表任务的历史五折特征相对贡献份额。先等权汇总各折平均绝对SHAP，再在每任务全部77特征内归一化；它不是原始平均绝对SHAP。经纬度在多个任务居前，表明模型利用空间定位信息。该历史固定批查询结果不能与P1原始SHAP数值直接比较。',
378:'3.2.3 历史五折特征排名跨折差异',
379:'图3-2分别展示各任务Top10特征在五折全部77特征中的实际名次、Top10出现次数及折间排名相关统计；计算完成度与排名一致性分开评价，不将解释任务完成称为稳定性通过。',
383:'图3-3至图3-6按Chl-a、Secchi、TN和TP分组展示第1折P1 SHAP，分层任务各自Top6、Secchi Top10，各任务独立原始尺度。每任务仅使用3个预登记核心样本seed42，共30个；10次seed43重复不进入汇总。对每特征计算三个核心样本绝对SHAP的算术均值，在77特征内降序排名；固定10个展示特征依据冻结的平均名次选择文件确定，图3-7显示其真实任务内名次。表3-1与附录F使用相同30样本口径，不作总体代表性或跨目标绝对贡献推断。',
387:'Chl-a三水层的P1解释见图3-3，Secchi见图3-4。相对贡献及排序以三个预登记核心样本的冻结汇总为准；图中每项任务采用自身单位，不在Chl-a与营养盐任务间比较原始SHAP绝对数值。',
390:'TN与TP三水层分别见图3-5和图3-6。排名只表达任务内部归因，目标量纲不同会影响贡献的数值量级。三个核心样本覆盖有限，这些第1折P1结果不能替代历史五折的折间差异审查。',
397:'3.3.1 历史五折响应差异及空分箱审查',
407:'图3-8至图3-11把四任务80条历史曲线按折×特征分别排列。核对冻结局部效应与分箱计数后，发现9条曲线含空bin；原过程以nan_to_num及max(count,1)作占位，未知局部效应同时影响后续累计值和中心化基准，不能仅断开空点便展示余下累计响应。因此9个面板仅保留审查说明，另外71条曲线保留原始分箱与原值，不跨折平均、平滑或补点。部分历史折另有严格逐样本预测复现限制，均属探索性记录。',
409:'图3-12至图3-15展示四项代表任务的第1折P1一维ALE，两条曲线分别使用自身坐标和实际分箱，底部支持轨逐箱列出样本数n及独立站点数s。全部20条曲线见附录E；透明度三日太阳辐射总量仅6个真实分箱。零线表示中心化参考，不代表环境过程无作用。',
424:'图3-16将20条P1 ALE曲线的样本数与独立站点数合并为左右面板，共196个实际分箱和4个NA格。两种计数各用独立色标；分箱索引不表示跨变量可比的物理数值，颜色不表示响应强度。',
432:'同查询重复预测审查移至附录H：图H-1仅含九项任务，27个输入各重复3次，必须在同一query内求极差，不能将不同输入的预测差异当作重复波动；其范围不是预测置信区间。图H-2只涉及Chl-a表层的两个SHAP样本及两条ALE曲线；V1历史失败与V2探针零波动分别保留，后者不能否定前者，也不能外推为十任务总体解释稳定。',
465:'随机留出评价同一总体内随机样本预测，允许同站点不同时段记录分别进入训练与测试；不直接等同于特征空间插值。它与冻结站点留出的评价对象不同。图4-3区分三类评价对象；历史数值仅作独立补充见图H-3，不与正式十任务数据配对比较。',
473:'4.4 现代表格模型比较与站点空间表现',
474:'表4-3保留原四任务、四模型完整冻结结果，其中TN为表层。图4-4按本轮指定的Chl-a表层、Secchi、TN底层和TP中层展示；TN底层仅有同协议XGBoost与TabICLv2，TabPFN及TabM以NA标明。已有TN底层的历史80/84特征结果与正式77特征配置不一致，不能拼接成四模型同口径比较；TN表层TabPFN的实际优势仍按表4-3保留。',
478:'',
484:'5.1 历史分海区扩展实验与正式消融的区分',
485:'本章先讨论历史78特征（包含water_zone_code）的分海区扩展实验，再分析正式77特征六来源消融，最后结合站点诊断讨论空间差异。前者包括12个目标×水层组合、60个任务折，Secchi历史上也分三水层；后者与正式十任务特征范围一致。两类实验不得合并为同一注册协议。',
486:'5.2 历史全域模型与分区模型比较',
492:'历史分区结果包含496条区域训练决策和36条全域回退决策；不能描述为每个海区均独立训练区域模型。12项任务的pooled差值与五折差异在5.4节图5-2完整展示，positive_folds为折内ΔR²大于零的次数。该扩展结果不能替代正式77特征十任务主比较。',
506:'5.4 历史分区扩展结果与空间异质性',
513:'图5-3展示四个唯一站点、五个站点×任务案例。VT11表层71条主表记录的77项特征平均缺失率为62.6852%，94站中位数为8.7882%；TN、TP表层各有67条OOF记录，不能混为相同样本范围。SM18的覆盖背景只核实到表层，不推断其中层特征缺失率。五案例分别使用任务自身RMSE单位及尺度，缺失率、位置与误差仅作为诊断线索，不构成因果解释。',
524:'（4）TabICLv2历史四任务五折解释与十任务第1折P1解释分层报告。历史ALE有9条曲线受空分箱占位影响，未作为连续响应展示；历史部分折亦未通过严格逐样本复现门槛。P1 SHAP汇总仅含30个预登记核心样本，重复预测探针仅覆盖九任务，不宣称总体解释稳定性通过。',
624:'图D-1至图D-12展示数值等价受控复现XGBoost的TreeSHAP、一维ALE及来源对照，解释对象不是TabICLv2。TreeSHAP基于log1p目标尺度，不与P1原始目标尺度SHAP直接比较。本附录不包含二维ALE响应结果。',
658:'D.3 TreeSHAP与来源消融对照',
659:'图D-11直接使用48个冻结z值。SHAP来源份额与删组ΔRMSE分别在每任务六来源内计算z=(x−均值)/样本标准差，ddof=1。z正负不等于原始ΔRMSE正负，z=0为参考中心而非没有性能变化；不同任务z值不能当作原始效应量比较，归因与消融不构成因果验证。图D-12保留四任务主要特征的实际排名，空格不补零。',
759:'注：ssrd与sst日窗截止于采样前一日；lag/roll表示前次采样记录而非固定自然日。rain_month为对应自然月累计。夜间灯光采用VIIRS月合成平均辐亮度单位nW·cm⁻²·sr⁻¹，其差分、滞后及滑动均值保留该单位；交互特征按实际数值乘积列单位，不作环境因果解释。',
}
for idx,s in updates.items():
    if original[idx]._p.getparent() is not None and s:text(original[idx],s)
# 删除二维小节标题（所有五个图及原图像已按映射移除）。
remove(original[647])
# 纠正前次采样滞后与自然日的混称，保留实际数值。
for p in doc.paragraphs:
    if p.style.name=='参考文献':continue
    s=p.text
    if p._p.getparent() is not None and not p._p.xpath('.//w:drawing'):
        s=s.replace('前1日风速','前1次采样风速').replace('前1日气温','前1次采样气温').replace('气温日变化','相邻采样气温变化').replace('3日平均气温','前3次采样平均气温')
        if '二维' in s and p.style.name!='图题':
            s=re.sub(r'二维候选[^。]*。','',s)
        if s!=p.text:text(p,s)

# 追加新结果，按图号出现顺序安排，历史分区图单列5.4而不塞入消融节。
a=original[479].insert_paragraph_before('图4-5给出Chl-a表层、Secchi、TN底层及TP中层的站点空间误差差异，分别包含84、94、84及84个有冻结指标的站点。ΔRMSE=TabICLv2−XGBoost，小于零表示误差降低；任务色标各自独立。总体改善与局地退化并存，不能以站点胜率代替所有站点均改善。')
figure(a,'4-5')
figure(original[507],'5-2');figure(original[513],'5-3')
# G完整特征表保留，并校正可由冻结构造公式确定的单位。
gtable=doc.tables[-1];features=[r.cells[1].text for r in gtable.rows[1:]];assert len(features)==77 and len(set(features))==77 and 'water_zone_code' not in features
for table in doc.tables:
    for row in table.rows:
        vals=[c.text for c in row.cells]
        if len(vals)>4 and vals[1] in FM and row.cells[4].text!=FM[vals[1]]['unit']:
            cp=row.cells[4].paragraphs[0]
            if cp.runs:
                cp.runs[0].text=FM[vals[1]]['unit']
                for run in cp.runs[1:]:run.text=''
            else:cp.add_run(FM[vals[1]]['unit'])

# 新附录位于G之后、个人简历之前。
a=original[761].insert_paragraph_before('附录H 预测与解释协议技术审查补充',style='Heading 2')
a=after(a,'本附录对应正文3.4.2和4.3的可靠性边界说明，不将少量探针或历史异口径结果替代正式主实验。')
for num in ['H-1','H-2','H-3']:a=figure(a,num)

# 尽量保持图—题—注同页；原第一、二章非目标图片与文献不变。
setting=OxmlElement('w:doNotAutoCompressPictures');setting.set(qn('w:val'),'1');doc.settings.element.append(setting)
for p in doc.paragraphs:
    if p.style.name=='图题':p.paragraph_format.keep_with_next=True
assert [p.text for p in doc.paragraphs if p.style.name=='参考文献']==refs
assert all(r['是否插入Word'] for r in registry.values())
name=OUT/'07_论文整合稿/水质预测硕士学位论文_图件重构与终稿整合稿.docx';doc.save(name)
(OUT/'05_数据核验/图件登记.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'05_数据核验/Word整合结构核对.json').write_text(json.dumps({'原参考文献条数':len(refs),'参考文献逐段一致':True,'正式特征数':77,'正式特征不含water_zone_code':True,'整合图件数':len(registry),'原稿SHA256':sha(DOC),'新稿SHA256':sha(name)},ensure_ascii=False,indent=2),encoding='utf-8')
print(name)
