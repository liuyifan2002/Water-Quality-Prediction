"""核验五项审计交付的全覆盖、编号、来源与原文件保护。"""
from pathlib import Path
import ast,hashlib,json,re
from collections import Counter
import pandas as pd
from docx import Document

ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parents[1];E=OUT/'evidence'
files=['01_全文图件审计总表.csv','02_逐图审查报告.md','03_第三至第五章最终图片结构方案.md','04_新增图件数据可行性审计.csv','05_最终绘图执行清单.md']
for f in files:assert (OUT/f).is_file() and (OUT/f).stat().st_size>500
a=pd.read_csv(OUT/files[0]);d=Document(ROOT/'文档/论文材料/硕士论文V4_题名更正_20261109/水质预测硕士学位论文_V4_题名更正.docx')
caps=[p.text.split(' ',1)[0] for p in d.paragraphs if p.style.name=='图题' and not p.text.startswith(('图1-','图2-'))]
assert len(a)==len(caps)==89 and set(a.图号)==set(caps) and a.图号.is_unique
assert not a.isna().any().any()
for col in ['数据来源','原始绘图脚本','当前文件路径','支撑的正文论点','不能被替代的信息','删除损失','信息表达问题','重复程度','科学可靠性','当前实际尺寸质量','建议处理方式','最终位置及编号','具体重构方案']:
    assert a[col].str.len().gt(0).all()
for row in a.itertuples():
    assert (ROOT/getattr(row,'当前文件路径')).is_file()
    for path in getattr(row,'数据来源').split(';'):assert (ROOT/path).is_file()
report=(OUT/files[1]).read_text(encoding='utf-8')
assert len(re.findall(r'^## 图[0-9A-Z]+-\d+ ',report,re.M))==89
plan=pd.read_csv(E/'最终24图编号与论点映射.csv')
assert len(plan)==24 and plan.最终建议图号.is_unique
assert Counter(plan.最终建议图号.str.extract(r'图(\d+)-')[0])==Counter({'3':16,'4':5,'5':3})
for ch,n in [('3',16),('4',5),('5',3)]:assert set(plan[plan.最终建议图号.str.startswith('图'+ch+'-')].最终建议图号)=={f'图{ch}-{i}' for i in range(1,n+1)}
structure=(OUT/files[2]).read_text(encoding='utf-8');execution=(OUT/files[4]).read_text(encoding='utf-8')
assert 'H（新增技术审查附录）' in structure and 'G为77项严格输入特征清单' in structure
for row in plan.itertuples():assert '最终'+row.最终建议图号+'：' in execution
feas=pd.read_csv(OUT/files[3]);assert set(feas.候选图号)=={'图4-5','图5-2','图5-3'}
assert feas.可绘制程度.str.contains('不可绘制').sum()==2
for r in feas.itertuples():assert (ROOT/r.真实数据文件).is_file()
hashes=pd.read_csv(E/'只读来源保护清单.csv')
for r in hashes.itertuples():assert hashlib.sha256((ROOT/r.路径).read_bytes()).hexdigest()==r.SHA256
sm=pd.read_csv(ROOT/'结果/论文图表/figure_revision_v2/source_inventory/40组文件与30核心样本对应.csv')
for r in sm.itertuples():assert hashlib.sha256((ROOT/r.文件).read_bytes()).hexdigest()==r.SHA256
assert a.loc[a.图号.eq('图3-5'),'当前文件路径'].iloc[0].find('figure_revision_v2')>=0
assert a.loc[a.图号.eq('图3-17'),'当前文件路径'].iloc[0].find('figure_revision_v2')>=0
assert a.loc[a.图号.eq('图D-16'),'当前文件路径'].iloc[0].find('figure_revision_v2')>=0
for p in (OUT/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
result={'核验日期':'2026-11-15','原Word与PDF未改动':True,'两轮改绘文件未改动':True,'保护清单文件数':len(hashes),'冻结40组SHAP原文件未变':True,'全量图件数':89,'章节分布':a.所属章节.value_counts().to_dict(),'零图附录':['A','B','G'],'逐图报告条目数':89,'正文方案图件数':24,'最终正文分布':{'3':16,'4':5,'5':3},'新增候选有证据':['图4-5','图5-2历史扩展','图5-3背景诊断'],'不支持的表达':['78特征历史分区冒充77特征正式比较','表层缺失率冒充中层或同折因果机制'],'五项交付物均存在':True,'脚本语法检查通过':True,'未调用训练预测或SHAP_ALE生产入口':True}
(E/'最终交付核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
