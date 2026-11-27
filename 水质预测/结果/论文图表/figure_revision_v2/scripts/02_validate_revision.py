"""核验第二轮冻结文件、实际尺寸和矢量导出，并从PDF生成审阅预览。"""
from pathlib import Path
import ast, hashlib, json
import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
items=[]
for _,r in pd.read_csv(OUT/'source_inventory/源文件SHA256清单.csv').iterrows():
    assert sha(ROOT/r['原始路径'])==r['SHA256']
for _,r in pd.read_csv(OUT/'source_inventory/三张冻结图直接复用.csv').iterrows():
    assert sha(ROOT/r['来源'])==sha(OUT/r['副本'])==r['SHA256']
assert pd.read_csv(OUT/'reports/数据与版式核验.csv')['通过'].all()
sm=pd.read_csv(OUT/'source_inventory/40组文件与30核心样本对应.csv')
assert len(sm)==40 and sm['用于图3-5'].sum()==30
source=ROOT/'runs/xgb_shap_ale_v2/04_integrated/04_shap_ale_ablation_comparison.csv'
copy=OUT/'data'/source.name
assert sha(source)==sha(copy)
h=pd.read_csv(OUT/'data/jitter_vs_explanation_scale.csv')
assert len(h)==216 and h.prediction_jitter_range.eq(0).sum()>0
cards=[]
for num in ['3-5','3-17','D-16']:
    p=next((OUT/'final_figures').glob('图'+num+'_*.pdf'))
    d=fitz.open(p);page=d[0]
    assert len(d)==1 and not page.get_images()
    assert all(d.extract_font(f[0])[3] for f in page.get_fonts())
    assert abs(page.rect.width/72*2.54-15.5)<.01 and page.rect.height/72*2.54<=20
    png=Image.open(p.with_suffix('.png'))
    assert abs(png.width-600*15.5/2.54)<2
    assert min(float(s['size']) for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'])>=7.49
    pix=page.get_pixmap(matrix=fitz.Matrix(2,2));preview=OUT/'preview'/('图'+num+'_实际尺寸PDF预览.png');pix.save(preview)
    card=Image.new('RGB',(1050,1430),'white');im=Image.open(preview).convert('RGB');im.thumbnail((1030,1350));card.paste(im,((1050-im.width)//2,55));ImageDraw.Draw(card).text((20,12),'图'+num,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27),fill='black');cards.append(card)
    items.append({'图号':'图'+num,'宽cm':round(page.rect.width/72*2.54,2),'高cm':round(page.rect.height/72*2.54,2),'PNG像素':list(png.size),'PDF纯矢量且字体嵌入':True,'PNG_SHA256':sha(p.with_suffix('.png')),'PDF_SHA256':sha(p)})
overview=Image.new('RGB',(3150,1430),'white')
for i,card in enumerate(cards): overview.paste(card,(i*1050,0))
overview.save(OUT/'preview/三张修订图_总览.jpg',quality=95)
for p in (OUT/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
(OUT/'reports/最终导出核验.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
print('通过：源文件及复用三图SHA256；30独立核心样本；冻结z值未变；216行波动数据；3张600dpi PNG及嵌字纯矢量PDF；实际宽15.5cm、最小字号7.5pt；脚本语法。')
