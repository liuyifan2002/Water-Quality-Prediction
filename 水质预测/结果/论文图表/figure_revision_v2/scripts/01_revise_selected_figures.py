"""第二轮只修订三图；复用另外三图，不导入模型、SHAP或ALE计算模块。"""
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


V1=ROOT/'结果/论文图表/figure_revision_v1'
# 将第一轮全部文件及论文、数据登记为保护对象，最终复核；不调用第一轮脚本。
protected={str(p.relative_to(ROOT)):sha(p) for p in V1.rglob('*') if p.is_file()}
for folder in ['硕士论文V4_结构与地图定向修订_20261108','硕士论文V4_题名更正_20261109']:
    for p in (ROOT/'文档/论文材料'/folder).glob('*'):
        if p.suffix in ['.pdf','.docx']: protected[str(p.relative_to(ROOT))]=sha(p)
pre=read('runs/tabicl_v2_slim_explainability_v2_all10/01_preregistered_shap_samples.csv')
core=pre[pre.phase.eq('CORE_3')].copy()
check('预登记30核心样本',len(core)==30 and core.groupby('task').size().eq(3).all())
check('预登记独立且事前',not core.duplicated(['task','obs_id']).any() and core.pre_registered_before_explanation.all() and not core.selection_uses_test_label.any())
mapping=read('runs/explanation_figures_v1/metadata/feature_display_name_mapping.csv').set_index('raw_feature_name')
paths=sorted((ROOT/'runs/tabicl_v2_slim_explainability_v2_all10/shap').glob('**/*.csv.gz'))
check('原40组冻结文件',len(paths)==40)
parts=[]; sample_rows=[]
for p in paths:
    record(p,'冻结SHAP'); dest=OUT/'data/shap'/p.parent.name/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    d=pd.read_csv(dest); a=d.iloc[0]
    check('SHAP文件77维及元数据 '+p.name,len(d)==77 and d.feature.nunique()==77 and set(d.feature)==set(mapping.index) and all(d[c].nunique()==1 for c in ['task','obs_id','shap_seed','selection_phase','fold']))
    registered=core[(core.task==a.task)&(core.obs_id==a.obs_id)]
    check('预登记样本对应 '+p.name,len(registered)==1 and a.fold=='fold_1')
    expected_phase='CORE_3' if a.shap_seed==42 else 'SENSITIVITY_1'
    check('种子与阶段对应 '+p.name,a.shap_seed in [42,43] and a.selection_phase==expected_phase and p.name==f'{a.obs_id}_seed{a.shap_seed}.csv.gz')
    if a.shap_seed==43: check('重复为中位样本 '+a.task,registered.target_quantile.iloc[0]==.5)
    selected=a.shap_seed==42
    sample_rows.append({'task':a.task,'obs_id':a.obs_id,'seed':a.shap_seed,'phase':a.selection_phase,'预登记分位':registered.target_quantile.iloc[0],'用于图3-5':selected,'文件':str(p.relative_to(ROOT)),'对应核心文件':f'{a.obs_id}_seed42.csv.gz','SHA256':sha(p)})
    if selected: parts.append(d)
sm=pd.DataFrame(sample_rows);sm.to_csv(OUT/'source_inventory/40组文件与30核心样本对应.csv',index=False,encoding='utf-8-sig')
check('选中30组排除10重复',len(parts)==30 and (~sm['用于图3-5']).sum()==10)
shap=pd.concat(parts,ignore_index=True)
check('独立样本每任务3个',shap.groupby('task').obs_id.nunique().eq(3).all())
# 每样本77个特征恰有一行；先取绝对值，再对三个独立样本等权平均。
stats=shap.groupby(['task','feature'],as_index=False).agg(mean_abs_shap=('shap_value',lambda x:float(np.mean(np.abs(x)))))
stats['rank']=stats.groupby('task').mean_abs_shap.rank(ascending=False,method='min')
check('77特征×10任务完整排名',len(stats)==770 and stats.groupby('task').feature.nunique().eq(77).all())
select=stats.groupby('feature')['rank'].mean().reset_index(name='十任务平均排名').sort_values(['十任务平均排名','feature'],kind='stable')
features=select.head(10).feature.tolist()
select['展示']=select.feature.isin(features);select.to_csv(OUT/'data/77特征展示选择依据.csv',index=False,encoding='utf-8-sig')
stats.to_csv(OUT/'data/30核心样本_77特征完整排名.csv',index=False,encoding='utf-8-sig')
mat=stats[stats.feature.isin(features)].pivot(index='feature',columns='task',values='rank').reindex(index=features,columns=TASKS)
mat.to_csv(OUT/'data/图3-5_30核心样本排名矩阵.csv',encoding='utf-8-sig')
# 与表3-2及其生成时冻结汇总逐项核验。
frozen=read('文档/论文材料/硕士论文V2_学术内容补强_20261101/冻结SHAP汇总核对表.csv')
v=stats.merge(frozen,on=['task','feature'],suffixes=('_new','_old'),validate='one_to_one')
check('770特征均值与表3-2冻结汇总一致',len(v)==770 and np.allclose(v.mean_abs_shap_new,v.mean_abs_shap_old,rtol=1e-12,atol=1e-14))
check('770排名与表3-2冻结汇总一致',np.array_equal(v['rank_new'],v['rank_old']))
docpath=ROOT/'文档/论文材料/硕士论文V4_题名更正_20261109/水质预测硕士学位论文_V4_题名更正.docx'
doc=Document(docpath)
table=next(t for t in doc.tables if t.rows[0].cells[0].text=='任务' and '主要贡献变量' in t.rows[0].cells[1].text)
for task,row in zip(TASKS,table.rows[1:]):
    top=stats[stats.task==task].sort_values(['rank','feature']).head(3).feature.tolist()
    check('表3-2前3变量 '+task,top==[x.strip() for x in row.cells[1].text.split('/')])
old=read('结果/论文图表/figure_revision_v1/data/图3-5_原图排名矩阵.csv','第一轮排名矩阵.csv').set_index('feature');old.columns.name='task'
comparison=mat.stack().rename('新排名').reset_index().merge(old.stack().rename('原排名').reset_index(),on=['feature','task'],how='outer')
comparison.to_csv(OUT/'reports/图3-5两轮排名变化.csv',index=False,encoding='utf-8-sig')
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
fig.text(.03,.035,'每任务3个预登记独立核心样本；与表3-2范围一致。\n仅比较任务内相对排名，不代表跨任务贡献大小。',fontsize=8,linespacing=1.6)
save(fig,'3-5','30核心样本特征排名矩阵','三个独立核心样本的绝对SHAP等权平均；77特征降序竞争排名，十任务平均排名前10展示。')
# 图3-17：全部尺度定义保留；不把零替换为极小正数。
h=read('runs/tabicl_explanation_numerical_stability_v2/jitter_vs_explanation_scale.csv')
record(ROOT/'runs/tabicl_explanation_numerical_stability_v2/numerical_stability_decision.md','尺度定义与失败记录')
types=['median_abs_shap','p75_abs_shap','p90_abs_shap','top10_feature_abs_shap','ale_curve_range','median_abs_local_effect','p90_abs_local_effect','median_abs_bin_to_bin_change','p90_abs_bin_to_bin_change','bin_to_bin_abs_change']
tl=['SHAP |贡献|中位数','SHAP |贡献|第75百分位','SHAP |贡献|第90百分位','SHAP 前10特征|贡献|','ALE 曲线极差','ALE |局部效应|中位数','ALE |局部效应|第90百分位','ALE 相邻分箱|变化|中位数','ALE 相邻分箱|变化|第90百分位','ALE 各相邻分箱|变化|']
unique=h.drop_duplicates(['method','effect_sample_id','effect_scale_type','feature','effect_scale'])
check('历史尺度为正',unique.effect_scale.gt(0).all());check('零波动原样保留',h.prediction_jitter_range.eq(0).any())
hist=float(h.loc[h.probe_id.str.startswith('V1_'),'prediction_jitter_range'].iloc[0])
fig=plt.figure(figsize=(15.5/2.54,18.3/2.54));gs=fig.add_gridspec(2,1,height_ratios=[3.1,1.1],left=.43,right=.95,bottom=.15,top=.87,hspace=.58)
ax=fig.add_subplot(gs[0]);
for i,t in enumerate(types):
    z=unique[unique.effect_scale_type.eq(t)];c=BLUE if i<4 else ORANGE
    if t!='top10_feature_abs_shap': ax.hlines(i,z.effect_scale.min(),z.effect_scale.max(),color=c,lw=1.2)
    for sample,g in z.groupby('effect_sample_id'):
        marker={'MM5_20201119_surface':'o','WM2_20211208_surface':'s','chla_surface_fold_1':'D'}[sample]
        ax.scatter(g.effect_scale,np.repeat(i,len(g)),color=c,marker=marker,s=18,alpha=.8,zorder=3)
ax.set_xscale('log');ax.set_xlim(.001,2);ax.set_xticks([.001,.01,.1,1],['0.001','0.01','0.1','1']);ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax.set_yticks(range(10),tl,fontsize=8);ax.invert_yaxis();ax.axvline(hist,color=PURPLE,ls='--',lw=1)
ax.set_xlabel('解释效应尺度（Chl-a，µg/L）');grid(ax);ax.set_title('(a) 冻结解释尺度与历史波动参照',loc='left',pad=10)
fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',ls='',label='SHAP样本1'),Line2D([],[],color=BLUE,marker='s',ls='',label='SHAP样本2'),Line2D([],[],color=ORANGE,marker='D',ls='',label='ALE曲线')],frameon=False,fontsize=7.5,ncol=3,loc='upper center',bbox_to_anchor=(.65,.945),columnspacing=.8)
ax2=fig.add_subplot(gs[1]);probes=h[['probe_id','prediction_jitter_range']].drop_duplicates();pl={'V1_historical_ssrd_repeatability_failure':'V1 历史失败事件','chla_fold1_failed_ssrd_lower':'V2 原失败点即时重复','chla_surface_fold1_frozen_pred_low':'V2 低预测值探针','chla_surface_fold1_frozen_pred_high':'V2 高预测值探针'}
probes=probes.set_index('probe_id').reindex(pl).reset_index()
ax2.scatter(probes.prediction_jitter_range,range(4),c=[PURPLE,GRAY,GRAY,GRAY],s=25,zorder=3);ax2.set_yticks(range(4),[pl[x] for x in probes.probe_id],fontsize=8);ax2.invert_yaxis();ax2.set_xlim(-.00015,.0022);ax2.set_xticks([0,.001,.002]);ax2.set_xlabel('重复预测极差（最大值 − 最小值，µg/L）');grid(ax2);ax2.set_title('(b) 波动记录分开呈现（线性坐标）',loc='left',pad=10)
for i,v in enumerate(probes.prediction_jitter_range): ax2.annotate('0' if v==0 else f'{v:.7f}',(v,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
fig.text(.03,.975,'TabICLv2叶绿素a表层解释效应尺度与预测重复波动对照',fontsize=9.3)
fig.text(.03,.06,'虚线：V1 历史波动 0.0014984 µg/L。V2 零值不否定历史失败。\nSHAP仅两个样本，ALE仅两条曲线；不外推十任务稳定性。',fontsize=8,linespacing=1.7)
save(fig,'3-17','TabICLv2叶绿素a表层解释效应尺度与预测重复波动对照','全部216行输入保留副本；尺度去除探针交叉重复用于显示，不删除独立解释值。零值采用独立线性坐标。此来源仅Chl-a解释尺度，非十任务。')

comp=read('runs/xgb_shap_ale_v2/04_integrated/04_shap_ale_ablation_comparison.csv')

for rel in ['脚本/05_诊断分析/58_compute_xgb_shap_ale_v2.py','脚本/05_诊断分析/59_integrate_plot_xgb_shap_ale_v2.py']:
    record(ROOT/rel,'标准化定义原脚本')
summary=read('runs/xgb_shap_ale_v2/01_shap/01_shap_importance_summary.csv')
ref=[]
for task,g in comp.groupby('task'):
    check('D16六来源 '+task,set(g.source_group)==set(GROUPS) and len(g)==6)
    shares=summary[summary.task==task].groupby('source_group').mean_normalized_contribution.sum()
    check('SHAP来源份额定义 '+task,np.allclose(g.set_index('source_group').shap_group_share.sort_index(),shares.sort_index(),rtol=1e-12))
    for raw,zcol in [('shap_group_share','shap_group_share_z'),('delta_RMSE','delta_RMSE_z')]:
        mean=g[raw].mean();sd=g[raw].std(ddof=1)
        check('只核验不替换z定义 '+task+' '+raw,np.allclose((g[raw]-mean)/sd,g[zcol],rtol=1e-12,atol=1e-12))
        ref.append({'task':task,'原始指标':raw,'来源数':6,'参考均值':mean,'样本标准差_ddof1':sd,'冻结z列':zcol})
pd.DataFrame(ref).to_csv(OUT/'reports/D16标准化参考中心核查.csv',index=False,encoding='utf-8-sig')

fig=plt.figure(figsize=(15.5/2.54,19.7/2.54));gs=fig.add_gridspec(4,2,left=.27,right=.97,top=.92,bottom=.19,wspace=.30,hspace=.68)
for k,task in enumerate(REPS):
    z=comp[comp.task.eq(task)].set_index('source_group').loc[GROUPS]
    for j,col in enumerate(['shap_group_share_z','delta_RMSE_z']):
        ax=fig.add_subplot(gs[k,j]);v=z[col].to_numpy();ax.barh(range(6),v,color=BLUE if j==0 else ORANGE,height=.52);ax.axvline(0,color=GRAY,lw=.75);ax.set_yticks(range(6),GL if j==0 else [],fontsize=8);ax.set_ylim(5.6,-.6);ax.set_xlim(-2.4,2.7);ax.set_xticks([-2,-1,0,1,2]);grid(ax)
        ax.set_title((f'({chr(97+k)}) {LAB[task]} · SHAP' if j==0 else '删组ΔRMSE\n（任务内z分数）'),loc='left',fontsize=9,pad=5)
        for i,a in enumerate(v): ax.annotate(f'{a:+.2f}',(a,i),xytext=(3 if a>=0 else -3,0),textcoords='offset points',ha='left' if a>=0 else 'right',va='center',fontsize=7.5)
        if k==3: ax.set_xlabel('来源份额 z 分数' if j==0 else 'ΔRMSE z 分数',fontsize=8.5)
fig.text(.03,.97,'TreeSHAP 来源归因与删组性能变化：两类独立证据',fontsize=10)
fig.text(.03,.025,'各任务独立在六类来源内标准化：z = (x − 均值) / 样本标准差（ddof=1）。\n左列 x 为SHAP来源份额；右列 x 为原始删组ΔRMSE。\nz正负仅表示高于/低于任务内来源均值，不等同于原始ΔRMSE正负。\nz=0是参考中心，不代表无性能变化；不用于跨任务原始效应大小比较。\n两类证据不构成因果验证；直接读取冻结z值，未重新标准化。',fontsize=7.5,linespacing=1.7)
save(fig,'D-16','来源归因与消融分区对照','两指标各在任务内六类来源中以均值和样本标准差（ddof=1）标准化；直接使用冻结z值。z正负表示相对来源均值的位置，不等于原始ΔRMSE正负；z=0不是无性能变化，不跨任务比较原始效应大小，不构成因果验证。')


# 已通过三图只复制，SHA256必须一致；不重新绘制。
reused=[]
for num in ['4-1','4-3','5-1']:
    for p in (V1/'final_figures').glob('图'+num+'_*'):
        target=OUT/'reused_figures'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
        check('冻结图直接复用 '+p.name,sha(p)==sha(target))
        reused.append({'图号':'图'+num,'来源':str(p.relative_to(ROOT)),'副本':str(target.relative_to(OUT)),'SHA256':sha(p)})
pd.DataFrame(reused).to_csv(OUT/'source_inventory/三张冻结图直接复用.csv',index=False,encoding='utf-8-sig')
for path,digest in protected.items(): check('保护文件未改写 '+path,sha(ROOT/path)==digest)
for item in manifest: check('冻结输入未改写 '+item['原始路径'],sha(ROOT/item['原始路径'])==item['SHA256'])
pd.DataFrame(manifest).to_csv(OUT/'source_inventory/源文件SHA256清单.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(checks).to_csv(OUT/'reports/数据与版式核验.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(figures).to_csv(OUT/'reports/图件尺寸与图注.csv',index=False,encoding='utf-8-sig')
review=fitz.open()
for f in figures:
    p=fitz.open(OUT/'final_figures'/(f['文件']+'.pdf'));review.insert_pdf(p)
review.save(OUT/'preview/三张修订图_实际尺寸审阅.pdf')
print(json.dumps({'输出':str(OUT),'图数':len(figures),'核验数':len(checks),'核心样本':len(parts),'重复排除':10,'展示特征':features},ensure_ascii=False))
