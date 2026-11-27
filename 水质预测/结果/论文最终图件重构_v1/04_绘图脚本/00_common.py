"""论文终稿绘图公共样式与冻结来源登记；不含模型调用。"""
from pathlib import Path
import hashlib,json,shutil,warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from docx import Document

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
DOC=ROOT/'文档/论文材料/硕士论文V4_题名更正_20261109/水质预测硕士学位论文_V4_题名更正.docx'
TASKS=['chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']
LAB=dict(zip(TASKS,['Chl-a 表层','Chl-a 中层','Chl-a 底层','Secchi','TN 表层','TN 中层','TN 底层','TP 表层','TP 中层','TP 底层']))
UNITS={'chla':'µg/L','secchi':'m','tn':'mg/L','tp':'mg/L'}
REPS=['chla_surface','secchi','tn_surface','tp_middle']
MAPREPS=['chla_surface','secchi','tn_bottom','tp_middle']
GROUPS=['METEOROLOGY','RAINFALL_HYDRO_TIDE','OCEAN_PHYSICS','HUMAN_ACTIVITY','TEMPORAL_PROCESS','SPATIAL_STATIC']
GL=['气象','降雨、水文与潮汐','海洋物理','人类活动代理','时间过程','空间静态']
BLUE,ORANGE,GREEN,PURPLE,GRAY='#4477AA','#EE9944','#228833','#AA3377','#777777'
FOLDC=[BLUE,ORANGE,GREEN,PURPLE,GRAY]
DIV=LinearSegmentedColormap.from_list('delta',[BLUE,'#ffffff',ORANGE])
RANKC=LinearSegmentedColormap.from_list('rank',[BLUE,'#d7e5ef','#f8fafb'])
plt.rcParams.update({'font.family':['Arial','Microsoft YaHei'],'font.size':8.5,'axes.labelsize':9,'axes.titlesize':9.5,'xtick.labelsize':8,'ytick.labelsize':8.5,'legend.fontsize':8,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
_doc=Document(DOC)
FM={r.cells[1].text:{'cn':r.cells[2].text,'unit':r.cells[4].text,'source':r.cells[5].text,'window':r.cells[6].text} for r in _doc.tables[-1].rows[1:]}
assert len(FM)==77 and 'water_zone_code' not in FM
FM['wind_speed_sq']['unit']='m²/s²'
FM['wind_speed_x_rain14d']['unit']='(m/s)·mm'
for f,u in {'sst_anom_7d_x_ssrd_mean_7d':'℃·J/m²','sst_mean_7d_x_ssrd_mean_7d':'℃·J/m²','t2m_x_rain14d':'℃·mm','t2m_x_tidal_range':'℃·m','rain14d_x_ssrd_mean_7d':'mm·J/m²','rain14d_x_tidal_range':'mm·m'}.items():FM[f]['unit']=u
for f in FM:
    if f.startswith('ntl_avg_rad'):FM[f]['unit']='nW·cm⁻²·sr⁻¹'
def cn(f):return FM[f]['cn'] if f in FM else f
def unit(f):return FM[f]['unit'] if f in FM else 'NA（源记录未定义）'
def targetunit(t):return UNITS[t.split('_')[0]]
def xlabel(f):return cn(f)+'（'+unit(f)+'）'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protect(path):
    path=Path(path).resolve();mfile=OUT/'05_数据核验/源文件保护哈希.json';m=json.loads(mfile.read_text(encoding='utf-8')) if mfile.exists() else {}
    key=str(path);h=sha(path)
    if key in m:assert m[key]==h,key
    m[key]=h;mfile.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');return h
def read(rel,**kw):
    p=Path(rel);p=p if p.is_absolute() else ROOT/p;protect(p);return pd.read_csv(p,**kw)
def plotfolder(num):
    ch=num.split('-')[0]
    return OUT/('02_正文图片/Chapter'+ch if ch.isdigit() else '03_附录图片/Appendix_'+ch)
def grid(ax,axis='x'):
    ax.grid(axis=axis,color='#e6e9ec',lw=.45);ax.set_axisbelow(True)
def record(num,title,paths,sources,script,old='',caption='',reused=False):
    f=OUT/'05_数据核验/图件登记.json';m=json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    sh={str(p):protect(ROOT/p if not Path(p).is_absolute() else p) for p in sources}
    m[num]={'新图号':'图'+num,'图名':title,'所属章节':num.split('-')[0],'原图号':old,'最终图片路径':str(paths[0]),'矢量PDF':str(paths[1]),'SVG':str(paths[2]),'数据来源':';'.join(sources),'绘图脚本':str(OUT/'04_绘图脚本'/script),'数据哈希':sh,'图片哈希':sha(paths[0]),'是否通过质量检查':'待PDF实排检查','是否插入Word':False,'图注':caption,'复用':reused}
    f.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
def save(fig,num,title,sources,script,old='',caption=''):
    stem=plotfolder(num)/(f'图{num}_{title}');stem.parent.mkdir(exist_ok=True,parents=True)
    assert abs(fig.get_figwidth()*2.54-15.5)<.02,(num,fig.get_figwidth())
    paths=[stem.with_suffix('.'+x) for x in ['png','pdf','svg']]
    for p in paths:fig.savefig(p,dpi=600,facecolor='white')
    fig.savefig(OUT/'06_图片质量检查'/f'图{num}_预览.png',dpi=150)
    record(num,title,paths,sources,script,old,caption)
    plt.close(fig);print('完成 图'+num,flush=True)
def reuse(num,title,source,sources,script,old='',caption=''):
    source=ROOT/source;paths=[]
    for ext in ['png','pdf','svg']:
        p=source.with_suffix('.'+ext);protect(p);dest=plotfolder(num)/(f'图{num}_{title}.'+ext);shutil.copy2(p,dest);assert sha(p)==sha(dest);paths.append(dest)
    record(num,title,paths,sources,script,old,caption,True);print('复用 图'+num,flush=True)
def figcm(h):return plt.figure(figsize=(15.5/2.54,h/2.54))
P1='runs/tabicl_v2_slim_explainability_v2_all10/'
HIST='runs/tabicl_v2_explainability_closure_v1/'
XGB='runs/xgb_shap_ale_v2/'
CORE='transfer/llm_extension_transfer_v1/results/final_experiment_closure_audit_v1/final_core_results_table_v2.csv'
MODERN='transfer/llm_extension_transfer_v1/results/modern_baseline_gate_v1/modern_baseline_core_metrics.csv'
FROZEN30='结果/论文图表/figure_revision_v2/data/30核心样本_77特征完整排名.csv'
REG='transfer/llm_extension_transfer_v1/results/regionalization/分海区五折留站验证_20260906/'
COORD='数据/处理后数据/正式主表/station_meta.csv'
for p in [DOC,DOC.with_suffix('.pdf')]:protect(p)
