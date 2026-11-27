"""汇集当前版统计、解释核验与图表证据，生成中文交付报告。"""
from pathlib import Path
import csv, hashlib, json, shutil, re, zipfile
import fitz
from PIL import Image,ImageDraw
from docx import Document
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
A=OUT/'A_统计分析';B=OUT/'B_解释复现';FIG=OUT/'正式论文图片'
V6=ROOT/'论文修订交付_20261121/05_v6_candidate'
CAND=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
for src,dest in [(A/'十任务Bootstrap结果.csv',OUT/'十任务Bootstrap结果.csv'),
 (A/'当前版本实验结果综合分析.md',OUT/'当前版本实验结果综合分析.md'),
 (B/'SHAP_ALE正式解释结果报告.md',OUT/'SHAP_ALE正式解释结果报告.md')]:
    assert src.is_file();shutil.copy2(src,dest)

doc=Document(OUT/'硕士学位论文_V6.1.docx')
# 第1、2章复用原图，从新稿内嵌媒体抽取，确保交付目录包含所有正文图件。
for no in ['图1-1','图2-1','图2-2','图2-3','图2-4']:
    caps=[p for p in doc.paragraphs if p.style.name=='图题' and p.text.startswith(no+' ')]
    assert len(caps)==1,no
    cap=caps[0];prev=cap._p.getprevious();assert prev.xpath('.//a:blip'),no
    rid=prev.xpath('.//a:blip')[0].get(qn('r:embed'))
    part=cap.part.related_parts[rid]
    ext={'image/png':'.png','image/jpeg':'.jpg','image/svg+xml':'.svg'}.get(part.content_type)
    assert ext,part.content_type
    path=FIG/(no+'_沿用原图'+ext)
    if not path.exists():path.write_bytes(part.blob)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources=[
 ('正式十任务pooled指标',CAND/'08_重算结果核验/新候选_pooled与逐折指标.csv','第四章；十任务'),
 ('正式逐站点误差',CAND/'08_重算结果核验/十任务逐站点误差.csv','第四章；空间诊断'),
 ('当前版六来源消融',CAND/'04_来源消融重算/四任务六来源_新版本完整对照.csv','第五章5.1—5.2'),
 ('当前批次历史78特征分区',CAND/'06_历史78特征分区重算/五折留站点验证_汇总.csv','第五章5.3'),
 ('当前批次历史78特征逐折',CAND/'06_历史78特征分区重算/五折留站点验证_各折指标.csv','第五章5.3'),
 ('正式77特征元数据',CAND/'正式77项特征定义与来源总表.csv','第二章、附录A'),
 ('修复版77特征基础输入',CAND/'02_冻结候选/基础输入_77特征.csv','第五章5.4'),
 ('真实站点元数据',ROOT/'数据/处理后数据/正式主表/station_meta.csv','图4-4'),
 ('海岸线底图',OUT/'绘图源代码/底图数据/ne_10m_land.shp','图4-4'),
 ('配对站点Bootstrap',OUT/'十任务Bootstrap结果.csv','表4-2'),
 ('当前版77特征覆盖',A/'77特征真实观测覆盖.csv','第五章5.4'),
 ('正式解释资格审查',B/'SHAP_ALE正式解释结果报告.md','第三章证据边界')]
rows=[]
for label,path,loc in sources:
    if path.is_file():rows.append({'证据':label,'类型':'冻结数据或只读派生','版本':'WQ77_REPAIRED_20261115_V1' if '历史78' not in label else '当前候选批次历史78特征独立协议',
      '绝对路径':str(path.resolve()),'SHA256':sha(path),'论文位置':loc,'限制':'历史78不与正式77合并' if '历史78' in label else ''})
for path in sorted(FIG.glob('图*.png')):
    rows.append({'证据':path.stem,'类型':'论文图','版本':'WQ77_REPAIRED_20261115_V1' if path.name.startswith(('图4','图5')) and not path.name.startswith('图5-3') else '独立历史78或沿用研究区/监测图',
      '绝对路径':str(path.resolve()),'SHA256':sha(path),'论文位置':path.name.split('_')[0],'限制':'图5-3为历史78特征' if path.name.startswith('图5-3') else ''})
with (OUT/'当前版本科研证据登记表.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

pdf=fitz.open(OUT/'硕士学位论文_V6.1.pdf')
# 正文图、表统一列出；页序从实际PDF标题位置读取，不取目录页。
contents=[];section='前置部分'
for para in doc.paragraphs:
    if para.style.name.startswith('Heading') and para.text.strip():section=para.text.strip()
    if para.style.name not in ('图题','表题'):continue
    m=re.match(r'^((?:图|表)[0-9]-[0-9]+)\s+(.+)$',para.text)
    if not m:continue
    no,title=m.groups();page_no=''
    for k,page in enumerate(pdf):
        if k<(11 if no=='表0-1' else 12):continue
        tx=page.get_text()
        if re.search(rf'(?m)^\s*{re.escape(no)}\s+',tx):page_no=k+1;break
    image_path=next((str(x.resolve()) for x in FIG.glob(no+'_*.png')), '') if no.startswith('图') else ''
    contents.append({'类型':'图' if no.startswith('图') else '表','编号':no,'名称':title,'所属位置':section,
                     'PDF页序':page_no,'图片资产路径':image_path,'当前版状态':'沿用研究区/监测资产' if no.startswith(('图1','图2')) else '历史78特征独立协议' if no=='图5-3' else '修复版冻结结果'})
with (OUT/'图表目录与插图位置清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=contents[0].keys());w.writeheader();w.writerows(contents)
thumbs=[]
for i,page in enumerate(pdf):
    pix=page.get_pixmap(matrix=fitz.Matrix(.32,.32),alpha=False)
    im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    c=Image.new('RGB',(230,310),'white');c.paste(im,((230-im.width)//2,15))
    ImageDraw.Draw(c).text((8,2),str(i+1),fill='black');thumbs.append(c)
cols=6;rows_n=(len(thumbs)+cols-1)//cols
sheet=Image.new('RGB',(cols*230,rows_n*310),'#ddd')
for i,im in enumerate(thumbs):sheet.paste(im,((i%cols)*230,(i//cols)*310))
sheet.save(OUT/'排版核查/全文PDF缩略联系表.png')

issues='''# 尚未完成事项

1. **正式TabICLv2解释未通过预测函数数值一致性门槛。** 11月20日回传SHAP有18/30个端点与独立P1审查不一致；本轮两样本跨进程探针虽彼此相同，其中SM13与冻结整批预测仍差2.98548×10⁻⁶，大于1×10⁻⁶门槛。10个seed43探针未回传。正式SHAP/ALE未启动，旧版解释全部从V6.1正文和附录退出；第三章尚无可发表的当前版解释结果。处理路径见《SHAP_ALE正式解释结果报告.md》。
2. **论文尚未达到完整送审稿要求。** 第三章目前只保留方法、协议和未通过状态；当前版本SHAP特征贡献、ALE响应、支持度图和相关环境过程讨论均待数值一致性通过后补入。
3. **预测时间协议有同期输入。** rain_month为采样所在自然月整月累计，部分气象、夜光、潮汐也为同期信息。当前结论限定为历史同期条件下的留站点空间泛化，不是实时提前预报。
4. **个人与行政材料沿用底稿待作者填写。** 封面、声明、授权、个人简历等不由本轮编造。
5. **原有第一、二章图件沿用。** 本轮未按新修复版重新构造研究区底图和观测分布；其基础观测与站点坐标未被当前数据版本变更证明失效，仍需最终人工核对学校版式。
'''
(OUT/'尚未完成事项.md').write_text(issues,encoding='utf-8')

qa=json.loads((OUT/'排版核查/自动核查结果.json').read_text(encoding='utf-8'))
report=f'''# V6.1科研深化与全文整合报告

本轮以V6候选DOCX为底稿、`WQ77_REPAIRED_20261115_V1`为唯一正式77特征版本，另存V6.1 DOCX和PDF。**目前不是完整送审稿**：正式TabICLv2解释仍未通过数值一致性门槛，故第三章没有当前版SHAP/ALE结果。

## 已完成

- 十任务正式预测、四模型、四任务六来源消融及当前批次历史78特征分区，均读取既有冻结产物，没有重新训练或预测。
- 对十任务执行1000次配对站点聚类Bootstrap。重算pooled指标与冻结汇总最大差3.05×10⁻¹⁶。Chl-a三水层和Secchi的ΔRMSE 95%区间全低于零；TN三水层和TP三水层的区间跨零。表4-2纳入ΔRMSE与ΔMAE区间并在单一横向页排版。
- 绘制当前版第四章五张结果图、第五章四张结果图，均有600 dpi PNG及矢量PDF/SVG（图5-4有PNG/PDF）。图5-1与图5-2只展示已完成的四任务消融，不补造其余任务；图5-3单独标明78特征历史分区。
- 第三章旧77特征SHAP/ALE及其附录占位图全部退出。原附录G的77项正式特征清单改为附录A；旧XGBoost解释资产表撤下。第四、第五章及中英文摘要、结论中的预测口径相应更新。
- 当前版77项输入的填补前非缺失率、年份与站点覆盖以真实基础输入表统计；图5-4没有把填补值当真实观测。

## 图表和排版核验

Word更新目录后导出PDF，共{qa['PDF页数']}页；正文图题{qa['正文图题数']}、内嵌图{qa['内嵌图片数']}、表题{qa['表题数']}，图号按1、2、4、5章首次出现顺序排列。已渲染并检查表4-2、第四和第五章结果页；表4-2在单一横向页，未见裁切。全文缩略联系表位于`排版核查/`。DOCX SHA-256 `{qa['DOCX_SHA256']}`；PDF SHA-256 `{qa['PDF_SHA256']}`。

## 未通过的科学门槛

解释复现支线判定`BLOCKED_NUMERICAL_IDENTITY`。跨进程视图顺序差异确实存在，但不能单独解释当前P1与冻结预测的差异。回传ALE的内部算术完整性不等于模型函数一致性。正式解释未启动，旧版结果没有移入附录或混入当前图表。具体证据见《SHAP_ALE正式解释结果报告.md》；因此V6.1不能宣称预测与解释已经一致，也不能作为完整送审稿。
'''
(OUT/'V6.1科研深化与全文整合报告.md').write_text(report,encoding='utf-8')
print('FINALIZED',len(pdf),'pages',len(rows),'evidence entries')
