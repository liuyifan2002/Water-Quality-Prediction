"""只读冻结结果，重绘六张论文图；不导入模型、SHAP或ALE计算模块。"""
from pathlib import Path
import hashlib, json, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import fitz
from docx import Document
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
for x in ['source_inventory','data','preview','final_figures','reports']:
    (OUT/x).mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE, GRAY = '#4477AA','#EE9944','#228833','#AA3377','#777777'
plt.rcParams.update({'font.family':['Arial','Microsoft YaHei'], 'font.size':8.5,
    'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8.5,
    'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','mathtext.fontset':'dejavusans',
    'axes.unicode_minus':False,'figure.facecolor':'white','axes.facecolor':'white'})
TASKS=['chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']
LABELS=['Chl-a 表层','Chl-a 中层','Chl-a 底层','Secchi','TN 表层','TN 中层','TN 底层','TP 表层','TP 中层','TP 底层']
LAB=dict(zip(TASKS,LABELS))
GROUPS=['METEOROLOGY','RAINFALL_HYDRO_TIDE','OCEAN_PHYSICS','HUMAN_ACTIVITY','TEMPORAL_PROCESS','SPATIAL_STATIC']
GL=['气象','降雨、水文与潮汐','海洋物理','人类活动代理','时间过程','空间静态']
REPS=['chla_surface','secchi','tn_surface','tp_middle']
manifest=[]; checks=[]; figures=[]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p,kind):
    p=Path(p); manifest.append({'类型':kind,'原始路径':str(p.relative_to(ROOT)), '字节数':p.stat().st_size,'SHA256':sha(p)})
def read(rel,name=None):
    p=ROOT/rel; record(p,'冻结数据'); target=OUT/'data'/(name or p.name)
    shutil.copy2(p,target)
    assert sha(p)==sha(target)
    return pd.read_csv(target)
def check(name,ok,detail=''):
    checks.append({'核验项':name,'通过':bool(ok),'说明':detail})
    assert ok,(name,detail)
def save(fig,num,title,note):
    # 固定画布宽15.5 cm，不使用tight裁切改变打印字号。
    stem=OUT/'final_figures'/f'图{num}_{title}'
    fig.canvas.draw()
    outside=[]
    renderer=fig.canvas.get_renderer(); bbox=fig.bbox
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text(): continue
        bb=t.get_window_extent(renderer)
        if bb.x0 < -2 or bb.y0 < -2 or bb.x1>bbox.x1+2 or bb.y1>bbox.y1+2: outside.append(t.get_text())
    check(f'图{num}文字未超画布',not outside,str(outside))
    for ext in ['png','pdf','svg']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=600,facecolor='white')
    fig.savefig(OUT/'preview'/f'图{num}.png',dpi=180)
    check(f'图{num}页面尺寸',fig.get_figwidth()*2.54<=15.51 and fig.get_figheight()*2.54<=20.01)
    figures.append({'图号':'图'+num,'标题':title,'宽cm':round(fig.get_figwidth()*2.54,2),'高cm':round(fig.get_figheight()*2.54,2),'图注':note,'文件':stem.name})
    plt.close(fig)
def grid(ax):
    ax.grid(axis='x',color='#e5e8eb',lw=.45); ax.set_axisbelow(True)

docrel='文档/论文材料/硕士论文V4_结构与地图定向修订_20261108/水质预测硕士学位论文_V4_结构与地图定向修订'
latest='文档/论文材料/硕士论文V4_题名更正_20261109/水质预测硕士学位论文_V4_题名更正'
for base in [docrel,latest]:
    for ext in ['.docx','.pdf']: record(ROOT/(base+ext),'论文只读基准')
doc=Document(ROOT/(docrel+'.docx'))
captions={}; embedded={}
for i,p in enumerate(doc.paragraphs):
    if p.style.name=='图题' and any(p.text.startswith('图'+n+' ') for n in ['3-5','3-17','4-1','4-3','5-1','D-16']):
        n=p.text.split()[0]; captions[n]=p.text
        for q in reversed(doc.paragraphs[max(0,i-3):i]):
            blips=q._p.xpath('.//a:blip')
            if blips:
                rid=blips[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                blob=doc.part.related_parts[rid].blob; (OUT/'source_inventory'/(n+'_原图.png')).write_bytes(blob)
                embedded[n]=hashlib.sha256(blob).hexdigest(); break
originals={
 '图3-5':'runs/thesis_figure_production_v3/figures/global/global_03_10_task_feature_rank_change.png',
 '图3-17':'runs/thesis_figure_production_v3/figures/stability/stability_02_jitter_vs_explanation_scale.png',
 '图4-1':'结果/论文图表/报告辅助图表_20261011/图1_十任务总体提升与站点改善覆盖.png',
 '图4-3':'结果/论文图表/报告辅助图表_20261011/图4_三种验证方式的模型差异.png',
 '图5-1':'结果/论文图表/报告辅助图表_20261011/图5_多源环境信息消融比较.png',
 '图D-16':'runs/xgb_shap_ale_v2/figures/fig_integrated_shap_ablation.png'}
for n,rel in originals.items(): record(ROOT/rel,'原始图件'); check(n+'内嵌原图匹配',sha(ROOT/rel)==embedded[n])
scripts=['runs/thesis_figure_production_v3/scripts/produce_v3_figures.py','脚本/06_结果汇总与绘图/40_build_report_support_figures.py','脚本/05_诊断分析/59_integrate_plot_xgb_shap_ale_v2.py']
for rel in scripts: record(ROOT/rel,'原始绘图脚本')
pdf=fitz.open(ROOT/(docrel+'.pdf')); pdfpages={}
for i,page in enumerate(pdf):
    if i<20: continue
    text=page.get_text().replace(' ','')
    for n in originals:
        if n+captions[n].split(' ',1)[1] in text:
            pdfpages[n]=i+1
            page.get_pixmap(matrix=fitz.Matrix(1.1,1.1)).save(OUT/'source_inventory'/(n+'_原PDF页面.png'))

# 图3-5：严格复现原图40组冻结SHAP汇总的排名与10个入图特征。
mapping=read('runs/explanation_figures_v1/metadata/feature_display_name_mapping.csv').set_index('raw_feature_name')
paths=sorted((ROOT/'runs/tabicl_v2_slim_explainability_v2_all10/shap').glob('**/*.csv.gz'))
check('图3-5沿用原图40组',len(paths)==40)
parts=[]
for p in paths:
    record(p,'图3-5冻结SHAP'); dest=OUT/'data'/'shap'/p.parent.name/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    parts.append(pd.read_csv(dest))
shap=pd.concat(parts,ignore_index=True)
stats=shap.groupby(['task','feature'],as_index=False).agg(mean_abs_shap=('shap_value',lambda x:float(np.mean(np.abs(x)))))
stats['rank']=stats.groupby('task').mean_abs_shap.rank(ascending=False,method='min')
features=stats.groupby('feature')['rank'].mean().nsmallest(10).index.tolist()
mat=stats[stats.feature.isin(features)].pivot(index='feature',columns='task',values='rank').reindex(index=features,columns=TASKS)
mat.to_csv(OUT/'data/图3-5_原图排名矩阵.csv',encoding='utf-8-sig')
cn={'lat':'纬度','lon':'经度','year_offset':'年份偏移','ssrd_sum_3d':'3日累计太阳辐射','ssrd_mean_3d':'3日平均太阳辐射','ssrd_mean_7d':'7日平均太阳辐射','sst_mean_3d':'3日平均海表温度','wind_speed_x_rain14d':'风速 × 14日降雨','wind_speed_lag1':'风速一期滞后','month_sin':'月份正弦编码','month_cos':'月份余弦编码','ntl_avg_rad':'夜间灯光','wind_speed_roll3':'风速3期滚动均值','t2m_roll3':'气温3期滚动均值','ntl_avg_rad_lag2':'夜间灯光2期滞后'}
def pretty(f):
    s=cn.get(f,str(mapping.loc[f,'display_name_cn']))
    for a,b in [('mean','均值'),('sum','累计'),('max','最大值')]: s=s.replace(a,b)
    return s
fig,ax=plt.subplots(figsize=(15.5/2.54,12.8/2.54));fig.subplots_adjust(left=.31,right=.90,bottom=.22,top=.88)
cmap=LinearSegmentedColormap.from_list('rank',['#4477AA','#d7e5ef','#f5f7f9']);cmap.set_bad('white')
im=ax.pcolormesh(np.arange(11)-.5,np.arange(11)-.5,mat.to_numpy(),cmap=cmap,vmin=1,vmax=77,shading='flat');ax.set_ylim(9.5,-.5);ax.set_xlim(-.5,9.5)
for i in range(len(features)):
    for j in range(10):
        v=mat.iloc[i,j]
        if pd.notna(v): ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=8.5,color='white' if v<23 else '#253746')
ax.set_yticks(range(len(features)),[pretty(f) for f in features]);ax.set_xticks(range(10),[x.replace(' ','\n') for x in LABELS],rotation=0,fontsize=8)
ax.tick_params(length=0);ax.set_xticks(np.arange(-.5,10,1),minor=True);ax.set_yticks(np.arange(-.5,10,1),minor=True);ax.grid(which='minor',color='white',lw=.7);ax.tick_params(which='minor',length=0)
for v in [2.5,3.5,6.5]: ax.axvline(v,color='white',lw=2)
cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025);cb.solids.set_rasterized(False);cb.set_ticks([1,20,40,60,77]);cb.set_label('任务内排名（1最重要）')
fig.text(.31,.945,'十项任务的特征排名',fontsize=10)
fig.text(.03,.05,'仅比较任务内相对排名，不代表跨任务贡献大小。',fontsize=8)
save(fig,'3-5','特征排名矩阵','沿用原图全部10个特征及100个排名，40组冻结SHAP；不重新计算SHAP。原图含种子重复，与正文3个核心样本汇总的口径差异另列待确认。')

# 图3-17：全部尺度定义保留；不把零替换为极小正数。
h=read('runs/tabicl_explanation_numerical_stability_v2/jitter_vs_explanation_scale.csv')
record(ROOT/'runs/tabicl_explanation_numerical_stability_v2/numerical_stability_decision.md','尺度定义与失败记录')
types=['median_abs_shap','p75_abs_shap','p90_abs_shap','top10_feature_abs_shap','ale_curve_range','median_abs_local_effect','p90_abs_local_effect','median_abs_bin_to_bin_change','p90_abs_bin_to_bin_change','bin_to_bin_abs_change']
tl=['SHAP |贡献|中位数','SHAP |贡献|第75百分位','SHAP |贡献|第90百分位','SHAP 前10特征|贡献|','ALE 曲线极差','ALE |局部效应|中位数','ALE |局部效应|第90百分位','ALE 相邻分箱|变化|中位数','ALE 相邻分箱|变化|第90百分位','ALE 各相邻分箱|变化|']
unique=h.drop_duplicates(['method','effect_sample_id','effect_scale_type','feature','effect_scale'])
check('历史尺度为正',unique.effect_scale.gt(0).all());check('零波动原样保留',h.prediction_jitter_range.eq(0).any())
hist=float(h.loc[h.probe_id.str.startswith('V1_'),'prediction_jitter_range'].iloc[0])
fig=plt.figure(figsize=(15.5/2.54,18.3/2.54));gs=fig.add_gridspec(2,1,height_ratios=[3.1,1.1],left=.43,right=.95,bottom=.15,top=.91,hspace=.58)
ax=fig.add_subplot(gs[0]);
for i,t in enumerate(types):
    z=unique[unique.effect_scale_type.eq(t)];c=BLUE if i<4 else ORANGE
    ax.hlines(i,z.effect_scale.min(),z.effect_scale.max(),color=c,lw=1.2)
    for sample,g in z.groupby('effect_sample_id'):
        marker={'MM5_20201119_surface':'o','WM2_20211208_surface':'s','chla_surface_fold_1':'D'}[sample]
        ax.scatter(g.effect_scale,np.repeat(i,len(g)),color=c,marker=marker,s=18,alpha=.8,zorder=3)
ax.set_xscale('log');ax.set_xlim(.001,2);ax.set_xticks([.001,.01,.1,1],['0.001','0.01','0.1','1']);ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax.set_yticks(range(10),tl,fontsize=8);ax.invert_yaxis();ax.axvline(hist,color=PURPLE,ls='--',lw=1)
ax.set_xlabel('解释效应尺度（Chl-a，µg/L）');grid(ax);ax.set_title('(a) 冻结解释尺度与历史波动参照',loc='left',pad=10)
fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',ls='',label='SHAP样本1'),Line2D([],[],color=BLUE,marker='s',ls='',label='SHAP样本2'),Line2D([],[],color=ORANGE,marker='D',ls='',label='ALE曲线')],frameon=False,fontsize=7.5,ncol=3,loc='upper center',bbox_to_anchor=(.65,.995),columnspacing=.8)
ax2=fig.add_subplot(gs[1]);probes=h[['probe_id','prediction_jitter_range']].drop_duplicates();pl={'V1_historical_ssrd_repeatability_failure':'V1 历史失败事件','chla_fold1_failed_ssrd_lower':'V2 原失败点即时重复','chla_surface_fold1_frozen_pred_low':'V2 低预测值探针','chla_surface_fold1_frozen_pred_high':'V2 高预测值探针'}
probes=probes.set_index('probe_id').reindex(pl).reset_index()
ax2.scatter(probes.prediction_jitter_range,range(4),c=[PURPLE,GRAY,GRAY,GRAY],s=25,zorder=3);ax2.set_yticks(range(4),[pl[x] for x in probes.probe_id],fontsize=8);ax2.invert_yaxis();ax2.set_xlim(-.00015,.0022);ax2.set_xticks([0,.001,.002]);ax2.set_xlabel('重复预测极差（最大值 − 最小值，µg/L）');grid(ax2);ax2.set_title('(b) 波动记录分开呈现（线性坐标）',loc='left',pad=10)
for i,v in enumerate(probes.prediction_jitter_range): ax2.annotate('0' if v==0 else f'{v:.7f}',(v,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
fig.text(.03,.06,'虚线：V1 历史波动 0.0014984 µg/L。V2 零值不否定历史失败。\nSHAP仅两条样本，ALE仅两条曲线；连线表示尺度范围，不是置信区间。',fontsize=8,linespacing=1.7)
save(fig,'3-17','历史波动与解释尺度','全部216行输入保留副本；尺度去除探针交叉重复用于显示，不删除独立解释值。零值采用独立线性坐标。此来源仅Chl-a解释尺度，非十任务。')

core=read('transfer/llm_extension_transfer_v1/results/final_experiment_closure_audit_v1/final_core_results_table_v2.csv')
table=next(t for t in doc.tables if [c.text for c in t.rows[0].cells][:2]==['任务','XGB R²'])
order=[row.cells[0].text for row in table.rows[1:]]
tab=core[core.model.eq('TabICLv2')].set_index('task').loc[order];xgb=core[core.model.eq('XGBoost')].set_index('task').loc[order]
for row in table.rows[1:]:
    task=row.cells[0].text; expected=[f'{xgb.loc[task,"pooled_R2"]:.3f}',f'{tab.loc[task,"pooled_R2"]:.3f}',f'{tab.loc[task,"delta_R2"]:.3f}',f'{tab.loc[task,"station_win_rate_RMSE"]*100:.1f}%']
    check('表4-2 '+task,expected==[c.text for c in row.cells[1:5]])
fig,axes=plt.subplots(1,2,figsize=(15.5/2.54,14/2.54),gridspec_kw={'width_ratios':[1.65,1]});fig.subplots_adjust(left=.20,right=.97,bottom=.17,top=.86,wspace=.57)
y=np.arange(10);ax=axes[0]
ax.hlines(y,xgb.pooled_R2,tab.pooled_R2,color='#b8c1c8',lw=1.8)
ax.scatter(xgb.pooled_R2,y,color=BLUE,s=26,label='XGBoost',zorder=3);ax.scatter(tab.pooled_R2,y,color=ORANGE,s=26,label='TabICLv2',zorder=3)
ax.set_yticks(y,[LAB[t] for t in order]);ax.set_xlim(-.18,.73);ax.set_xticks([-.1,.1,.3,.5,.7]);ax.set_ylim(9.7,-.7);ax.axvline(0,color=GRAY,lw=.6);ax.set_xlabel('pooled R²');ax.set_title('(a) 总体预测精度',loc='left',pad=18);grid(ax)
for i,v in enumerate(tab.delta_R2): ax.annotate(f'+{v:.3f}',(tab.pooled_R2.iloc[i],i),xytext=(5,0),textcoords='offset points',fontsize=7.5,va='center')
ax.legend(frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(-.05,1.035),fontsize=8,handletextpad=.3,columnspacing=.8)
ax=axes[1];win=tab.station_win_rate_RMSE*100;ax.barh(y,win,color=GREEN,height=.5);ax.axvline(50,color=GRAY,lw=.8,ls='--');ax.set_yticks(y,[]);ax.set_ylim(9.7,-.7);ax.set_xlim(0,112);ax.set_xticks([0,50,100]);ax.set_xlabel('RMSE站点胜率（%）');ax.set_title('(b) 站点改善覆盖',loc='left',pad=18);grid(ax)
for i,v in enumerate(win): ax.text(v+2,i,f'{v:.1f}',va='center',fontsize=8)
fig.text(.20,.94,'十项任务：总体提升与站点改善覆盖',fontsize=10)
fig.text(.03,.025,'左：标注为 ΔR²（TabICLv2 − XGBoost）；右：虚线为50%。\n按表4-2排列；站点胜率不表示所有站点均有改善。',fontsize=8,linespacing=1.5)
save(fig,'4-1','总体精度与站点覆盖','数值按表4-2逐项核对，pooled R²不是fold R²均值；负R²完整保留。')

pilot=read('runs/tabiclv2_pilot/20_tabiclv2_pilot_summary.csv')
fig,axes=plt.subplots(2,2,figsize=(15.5/2.54,14.7/2.54));fig.subplots_adjust(left=.14,right=.97,bottom=.21,top=.90,wspace=.40,hspace=.55)
for k,(ax,task) in enumerate(zip(axes.flat,['chla_surface','secchi','tn_bottom','tp_middle'])):
    z=pilot[pilot.task.eq(task)].set_index('validation').loc[['random','year','station']];v=z.delta_R2.to_numpy()
    ax.bar(range(3),v,color=[BLUE,ORANGE,GREEN],width=.5);ax.axhline(0,color=GRAY,lw=.9);ax.set_xticks(range(3),['随机划分','年份留出','站点留出'],fontsize=8);ax.set_title(f'({chr(97+k)}) {LAB[task]}',loc='left');ax.set_ylim(-.94,.43)
    ax.set_ylabel('ΔR²');ax.grid(axis='y',color='#e5e8eb',lw=.45);ax.set_axisbelow(True)
    for i,a in enumerate(v):ax.annotate(f'{a:+.3f}',(i,a),xytext=(0,5 if a>=0 else -5),textcoords='offset points',ha='center',va='bottom' if a>=0 else 'top',fontsize=8)
fig.text(.03,.085,'ΔR² = TabICLv2 − XGBoost；历史代表任务的不同划分实验。\n协议与十任务 Family V2 及 Secchi 正式站点留出不完全一致。\nTN沿用原图及原始数据的底层，未替换为缺少本组数据的表层。',fontsize=8,linespacing=1.7)
save(fig,'4-3','历史验证口径比较','保留原始TN底层，原始数值不变；待确认任务要求中的TN表层冲突。')

abl=read('transfer/llm_extension_transfer_v1/results/multisource_ablation/multisource_ablation_source_summary.csv')
table5=next(t for t in doc.tables if [c.text for c in t.rows[0].cells][:2]==['任务','删除来源'])
short=dict(zip(GROUPS,['MET','RHT','OCN','HUM','TMP','SPA'])); t5lab={'Chl-a表层':'chla_surface','Secchi':'secchi','TN表层':'tn_surface','TP中层':'tp_middle'}
for row in table5.rows[1:]:
    task=t5lab[row.cells[0].text];group=next(g for g in GROUPS if short[g]==row.cells[1].text);v=abl[(abl.task==task)&(abl.source_group==group)].pooled_delta_R2.iloc[0]
    check('表5-1 '+task+' '+group,f'{v:.3f}'==row.cells[2].text)
fig,axes=plt.subplots(2,2,figsize=(15.5/2.54,17/2.54));fig.subplots_adjust(left=.26,right=.97,bottom=.15,top=.90,wspace=.16,hspace=.45)
for k,(ax,task) in enumerate(zip(axes.flat,REPS)):
    z=abl[abl.task.eq(task)].set_index('source_group').loc[GROUPS];v=z.pooled_delta_R2.to_numpy()
    ax.barh(range(6),v,color=[BLUE if a<0 else ORANGE for a in v],height=.55);ax.axvline(0,color='#444444',lw=1);ax.set_ylim(5.6,-.6);ax.set_xlim(-.26,.15);ax.set_xticks([-.2,-.1,0,.1]);ax.set_yticks(range(6),GL if k%2==0 else []);ax.set_xlabel('ΔR²（消融 − 完整）');ax.set_title(f'({chr(97+k)}) {LAB[task]}',loc='left');grid(ax)
    for i,a in enumerate(v): ax.annotate(f'{a:+.3f}',(a,i),xytext=(-4 if a<0 else 4,0),textcoords='offset points',ha='right' if a<0 else 'left',va='center',fontsize=7.5)
fig.legend(handles=[Line2D([],[],color=BLUE,lw=6,label='删除后性能下降'),Line2D([],[],color=ORANGE,lw=6,label='删除后性能提高')],loc='upper center',bbox_to_anchor=(.61,.985),ncol=2,frameon=False,fontsize=8)
fig.text(.03,.035,'负值：删组后 R²下降；正值：删组后 R²提高。\n正值不直接代表该来源具有负面物理作用；数值与表5-1一致。',fontsize=8,linespacing=1.7)
save(fig,'5-1','来源消融性能变化','冻结pooled_delta_R2逐项与表5-1核对；六类来源统一顺序、横轴范围和正负配色。')

comp=read('runs/xgb_shap_ale_v2/04_integrated/04_shap_ale_ablation_comparison.csv')
fig=plt.figure(figsize=(15.5/2.54,19.7/2.54));gs=fig.add_gridspec(4,2,left=.27,right=.97,top=.92,bottom=.13,wspace=.30,hspace=.68)
for k,task in enumerate(REPS):
    z=comp[comp.task.eq(task)].set_index('source_group').loc[GROUPS]
    for j,col in enumerate(['shap_group_share_z','delta_RMSE_z']):
        ax=fig.add_subplot(gs[k,j]);v=z[col].to_numpy();ax.barh(range(6),v,color=BLUE if j==0 else ORANGE,height=.52);ax.axvline(0,color=GRAY,lw=.75);ax.set_yticks(range(6),GL if j==0 else [],fontsize=8);ax.set_ylim(5.6,-.6);ax.set_xlim(-2.4,2.7);ax.set_xticks([-2,-1,0,1,2]);grid(ax)
        ax.set_title((f'({chr(97+k)}) {LAB[task]} · SHAP' if j==0 else '删组 RMSE变化'),loc='left',fontsize=9,pad=5)
        for i,a in enumerate(v): ax.annotate(f'{a:+.2f}',(a,i),xytext=(3 if a>=0 else -3,0),textcoords='offset points',ha='left' if a>=0 else 'right',va='center',fontsize=7.5)
        if k==3: ax.set_xlabel('来源份额 z 分数' if j==0 else 'ΔRMSE z 分数',fontsize=8.5)
fig.text(.03,.97,'TreeSHAP 来源归因与删组性能变化：两类独立证据',fontsize=10)
fig.text(.03,.025,'两列均保留原图的任务内 z 标准化，各用独立坐标轴。\n右列是 ΔRMSE 的 z 分数，不是 ΔR²，也不表示原始 ΔRMSE的正负。\nTreeSHAP源于 log1p 目标尺度；两类证据不构成因果验证。',fontsize=8,linespacing=1.6)
save(fig,'D-16','来源归因与消融分区对照','保留原始shap_group_share_z、delta_RMSE_z，无重新标准化。右轴原始ΔRMSE的零与z=0不同，已明确说明。改为ΔR²将改变原图指标与标准化，需用户确认。')

# 来源清单、不可变性验证和可打印预览。
for item in manifest: check('源文件未改写 '+item['原始路径'],sha(ROOT/item['原始路径'])==item['SHA256'])
pd.DataFrame(manifest).to_csv(OUT/'source_inventory/源文件清单与SHA256.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(checks).to_csv(OUT/'reports/数据与版式核验.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(figures).to_csv(OUT/'reports/图件尺寸与建议图注.csv',index=False,encoding='utf-8-sig')
designs=['排名矩阵热图','效应尺度分组与独立波动面板','R²哑铃图＋站点胜率条形图','2×2历史验证小倍数图','2×2零中心发散条形图','四任务×两区域独立指标图']
sources=['40组冻结SHAP CSV.GZ','jitter_vs_explanation_scale.csv','final_core_results_table_v2.csv','20_tabiclv2_pilot_summary.csv','multisource_ablation_source_summary.csv','04_shap_ale_ablation_comparison.csv']
rows=[]
for (n,orig),design,source in zip(originals.items(),designs,sources):
    rows.append({'图号':n,'V4 PDF页':pdfpages.get(n),'原始图片':orig,'数据文件':source,'绘图脚本':scripts[0] if n in ['图3-5','图3-17'] else scripts[2] if n=='图D-16' else scripts[1],'当前问题':{'图3-5':'折线交叉，排名易误读为贡献','图3-17':'分组聚合掩盖零波动与历史事件差别','图4-1':'柱形拥挤，站点与总体比较逻辑不清','图4-3':'字号与标签小，历史协议易误读','图5-1':'原图RMSE与表5-1的ΔR²主展示不一致','图D-16':'两个标准化指标混在同一柱图'}[n],'新设计':design})
pd.DataFrame(rows).to_csv(OUT/'source_inventory/六图来源与改绘清单.csv',index=False,encoding='utf-8-sig')
cards=[]
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28)
for f in figures:
    im=Image.open(OUT/'preview'/(f['图号']+'.png')).convert('RGB');im.thumbnail((1050,1270))
    card=Image.new('RGB',(1110,1350),'white');card.paste(im,((1110-im.width)//2,65));ImageDraw.Draw(card).text((25,15),f['图号']+'  '+f['标题'],font=font,fill='#253746');cards.append(card)
overview=Image.new('RGB',(2220,4050),'#e8ecef')
for i,card in enumerate(cards):overview.paste(card,((i%2)*1110,(i//2)*1350))
overview.save(OUT/'preview/六张核心图_预览总览.jpg',quality=94)
review=fitz.open()
for f in figures:
    p=fitz.open(OUT/'final_figures'/(f['文件']+'.pdf'));review.insert_pdf(p)
review.save(OUT/'preview/六张核心图_实际尺寸审阅.pdf')
report='''# 六张核心图第一阶段改绘报告

## 范围与保护

仅论文图件输出；未运行模型、预测、SHAP或ALE。源文件、论文与原图均只读，按SHA256复核。输出为15.5 cm固定宽度、600 dpi PNG、嵌字矢量PDF及路径化SVG；最小图内标注7.5 pt。中文微软雅黑，英文数字Arial。未替换或保存任何论文DOCX。

PDF内容与排版依据：20261108结构与地图定向修订版；20261109题名更正版也记录SHA256，作为后续整合候选，不覆盖。六张原图与DOCX内嵌图片逐字节匹配。来源路径、脚本、数据及PDF页详见清单。

## 六图设计与数据

'''
for row,f in zip(rows,figures):
    report+=f"### {row['图号']}：{row['新设计']}\n\n原问题：{row['当前问题']}。数据：`{row['数据文件']}`。PDF物理页：{row['V4 PDF页']}。\n\n{f['图注']}\n\n"
report+='''## 科学表达待确认

1. 图4-3：原图及历史冻结汇总为TN底层，任务要求写TN表层。本阶段保留TN底层，未伪造或替用不同协议的数据。确认后才能改变任务组成。
2. 图D-16：原图为任务内z标准化SHAP来源份额与ΔRMSE，正文D.4也说明分别标准化。本阶段保留两列原始z值；若需右侧改成ΔR²，必须确认改指标，并明确是否对ΔR²另作任务内标准化。不能将标准化ΔRMSE直接改名ΔR²。
3. 图3-5：原图脚本读取40组SHAP（包括每任务3个独立样本及1次种子重复），正文表3-2按30个核心样本汇总。为遵守保留原图排名，本图复现40组的10特征、100排名，不改为30样本重新排序。建议确认是否统一为核心样本口径；当前图不声称10特征来自30个样本。
4. 图3-17：源表中解释尺度只涉及Chl-a表层，两条SHAP样本与两条ALE曲线；波动列包含3组V2零值及1条V1非零历史事件。此表不覆盖TN解释尺度，不能绘成十任务对照。历史波动=重复预测max−min，SHAP尺度为|贡献|的中位数/分位数/前10项，ALE为曲线极差、|局部效应|及|相邻分箱差|；均为逆变换目标的µg/L，含零波动因此采用分组尺度图而非双对数散点。原V1失败保留。全部216行保留，显示时仅去除同一尺度随不同探针重复的副本，不改数值。

## 核验与交付

表4-2逐任务核对两模型pooled R²、ΔR²及站点胜率；图4-1顺序严格沿用表4-2。表5-1全部24个ΔR²逐项核对（论文显示精度）。图4-3保留12个原值；D-16保留48个原始z值。全部数据副本、排名矩阵及原始图片可审计。无无法追溯数据而放弃重绘的图；上述三处内容口径差异保留为明确待确认事项。

图3-5没有虚填排名；本次原图10特征在10任务均有冻结排名，因此无缺失格。图3-17零值未替为极小正数。图4-1保留负R²。图5-1统一六来源、正负方向与横轴。未处理D-11～D-15，也未引入取消的分析。

阶段停在六图审阅；待用户确认风格及口径后，再开展其余图件及Word/PDF整合。
'''
(OUT/'reports/六张核心图改绘与科学边界说明.md').write_text(report,encoding='utf-8')
print(json.dumps({'输出':str(OUT),'核验项数':len(checks),'图数':len(figures),'PDF页':pdfpages},ensure_ascii=False))
