"""仅替换V6.3中7张指定图及其失效说明，另存V6.4。"""
from pathlib import Path
from shutil import copy2
from docx import Document
from docx.shared import Cm
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'论文修订交付_20261121/V6_3_新增15图整合_20261122/水质预测硕士学位论文_V6.3_新增15图整合稿.docx'
TARGET=HERE/'水质预测硕士学位论文_V6.4_图件修订.docx'
FIG=HERE/'figures'
NAMES={'2-10':'四指标月尺度观测变化','4-14':'分箱预测观测一致性','4-15':'十任务逐年预测误差','4-16':'浓度分位区间绝对误差','4-17':'站点样本数与误差变化','5-10':'历史分区局地模型与全域回退','5-11':'六来源各水区真实观测覆盖'}
assert SOURCE.exists() and not TARGET.exists()
copy2(SOURCE,TARGET)
doc=Document(TARGET);paragraphs=doc.paragraphs
used=[]
for num,stem in NAMES.items():
    caps=[i for i,p in enumerate(paragraphs) if p.text.startswith(f'图{num} ') and i>200]
    assert len(caps)==1,(num,caps)
    idx=caps[0];pic_p=paragraphs[idx-1]
    inlines=pic_p._p.xpath('.//wp:inline');assert len(inlines)==1,num
    blips=pic_p._p.xpath('.//a:blip');assert len(blips)==1,num
    rid=blips[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    assert rid not in used,(num,'reused image relation');used.append(rid)
    path=FIG/f'图{num}_{stem}.png';assert path.exists() and path.stat().st_size>10000,path
    doc.part.related_parts[rid]._blob=path.read_bytes()
    shape=next(s for s in doc.inline_shapes if s._inline is inlines[0])
    with Image.open(path) as im:
        shape.width=Cm(15.3)
        shape.height=Cm(15.3*im.height/im.width)

    if num=='2-10':
        p=paragraphs[idx-2]
        assert '图中同步给出各月有效样本量' in p.text
        p.text=p.text.replace('，图中同步给出各月有效样本量','')
    elif num=='4-15':
        p=paragraphs[idx+1]
        assert '每年样本量标在横轴下方；' in p.text
        p.text=p.text.replace('每年样本量标在横轴下方；','')
    elif num=='4-16':
        p=paragraphs[idx+1]
        assert '横轴下方为每箱样本量，' in p.text
        p.text=p.text.replace('横轴下方为每箱样本量，','')
    elif num=='5-10':
        p=paragraphs[idx+1]
        p.add_run(' Secchi的表、中、底层仅为历史78特征扩展实验任务组合，不代表透明度按三个水层分层观测。')
    elif num=='5-11':
        p=paragraphs[idx+1]
        p.add_run(' 色标显示范围为95%—100%；格内统计值未截断。')
doc.save(TARGET)
print('V6.4 saved:',TARGET,'replaced',len(used),'image relations')
