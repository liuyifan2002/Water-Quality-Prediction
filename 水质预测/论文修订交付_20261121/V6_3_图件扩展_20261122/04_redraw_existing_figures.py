"""定点改绘V6.2图2-2至2-4；读取冻结观测元数据及正式OOF真值。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'正式论文图片';OUT.mkdir(exist_ok=True)
EXP=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
META=pd.read_csv(EXP/'02_冻结候选/观测元数据.csv')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],'font.size':8.5,'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,stem):
 for ext in ['png','pdf','svg']:
  fig.savefig(OUT/f'{stem}.{ext}',dpi=600 if ext=='png' else None,bbox_inches='tight',pad_inches=.07)
 plt.close(fig)

# 2-2：站点×采样日期只计一次，不把三水层重复计为三个站点-日期。
dd=META[['station','date']].drop_duplicates();dd['date']=pd.to_datetime(dd.date);dd['year']=dd.date.dt.year;dd['month']=dd.date.dt.month
g=dd.groupby(['year','month']).size().unstack().reindex(index=range(2014,2025),columns=range(1,13))
fig,ax=plt.subplots(figsize=(15.5/2.54,8/2.54));fig.subplots_adjust(left=.12,right=.89,top=.93,bottom=.14)
im=ax.imshow(g.to_numpy(),cmap='YlGnBu',vmin=0,vmax=np.nanmax(g.to_numpy()),aspect='auto')
for i in range(11):
 for j in range(12):
  v=g.iloc[i,j]
  if pd.notna(v) and v>0:ax.text(j,i,f'{int(v)}',ha='center',va='center',fontsize=7.6,color='white' if v>64 else '#26343d')
ax.set_xticks(range(12),range(1,13));ax.set_yticks(range(11),range(2014,2025));ax.set_xlabel('采样月份');ax.set_ylabel('年份')
fig.colorbar(im,ax=ax,fraction=.03,pad=.025,label='有记录的站点—日期数')
save(fig,'图2-2_水质监测时间覆盖_修订')

# 2-3：左侧六来源，中间匹配规则，右侧主键与建模任务，减少装饰和无意义留白。
fig,ax=plt.subplots(figsize=(15.5/2.54,8.4/2.54));fig.subplots_adjust(left=.03,right=.97,top=.94,bottom=.10);ax.axis('off')
groups=[('MET 29','气象'),('RHT 22','降雨·水文·潮汐'),('OCN 11','海洋物理'),('HUM 6','人类活动代理'),('TMP 7','时间过程'),('SPA 2','空间静态')]
for i,(key,name) in enumerate(groups):
 y=.82-i*.115
 box=FancyBboxPatch((.03,y-.045),.26,.085,boxstyle='round,pad=0.01,rounding_size=.015',facecolor='#e8f0f5',edgecolor='#4477AA',lw=.8,transform=ax.transAxes)
 ax.add_patch(box);ax.text(.16,y,f'{key}  {name}',ha='center',va='center',fontsize=8,transform=ax.transAxes)
ax.annotate('',xy=(.42,.51),xytext=(.30,.51),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#777','lw':1.4})
box=FancyBboxPatch((.43,.30),.25,.42,boxstyle='round,pad=0.01,rounding_size=.015',facecolor='#edf5ee',edgecolor='#228833',lw=1,transform=ax.transAxes);ax.add_patch(box)
for y,t in zip([.63,.55,.47,.39],['站点空间匹配','采样时间匹配','历史窗口与滞后特征','缺失值及质量控制']):ax.text(.555,y,t,ha='center',va='center',fontsize=8,transform=ax.transAxes)
ax.annotate('',xy=(.78,.51),xytext=(.69,.51),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#777','lw':1.4})
box=FancyBboxPatch((.79,.32),.19,.38,boxstyle='round,pad=0.01,rounding_size=.015',facecolor='#f9f0e6',edgecolor='#EE9944',lw=1,transform=ax.transAxes);ax.add_patch(box)
for y,t in zip([.63,.53,.43,.35],['obs_id','站点×日期×水层','77项模型输入','十项水质任务']):ax.text(.885,y,t,ha='center',va='center',fontsize=8 if y!=.53 else 7.5,transform=ax.transAxes)
ax.text(.03,.10,'原始数据 → 匹配与构造 → 单条水层观测；HUM为人类活动代理信息，并非污染排放实测。',fontsize=7.8,transform=ax.transAxes)
save(fig,'图2-3_数据接入与样本主键_修订')

# 2-4：每项正式任务的冻结OOF真值；Secchi仅一次，所有真实极值都保留在箱线范围内。
PRED=EXP/'03_正式十任务重算/predictions'
def values(task):
 d=pd.concat([pd.read_csv(PRED/f'{task}_xgboost_fold_{i}.csv',usecols=['obs_id','y_true']) for i in range(1,6)])
 assert d.obs_id.is_unique
 return d.y_true.dropna().to_numpy()
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,11.2/2.54));fig.subplots_adjust(left=.10,right=.98,top=.94,bottom=.12,wspace=.30,hspace=.45)
for ax,prefix,title,unit in zip(axs.flat,['chla','secchi','tn','tp'],['Chl-a','Secchi','TN','TP'],['μg/L','m','mg/L','mg/L']):
 tasks=['secchi'] if prefix=='secchi' else [f'{prefix}_{x}' for x in ['surface','middle','bottom']]
 arrays=[values(t) for t in tasks];assert all((a>0).all() for a in arrays)
 boxes=ax.boxplot(arrays,whis=(0,100),patch_artist=True,widths=.55,showfliers=False,medianprops={'color':'#AA3377','lw':1.2},boxprops={'edgecolor':'#4477AA','lw':1.1},whiskerprops={'color':'#4477AA','lw':.9},capprops={'color':'#4477AA','lw':.9})
 for box in boxes['boxes']:box.set_facecolor('#dbe8f0')
 labels=['透明度'] if prefix=='secchi' else ['表层','中层','底层'];ax.set_xticks(range(1,len(tasks)+1),[f'{lab}\nn={len(a):,}' for lab,a in zip(labels,arrays)])
 ax.set_yscale('log');ax.set_ylabel(f'{title}（{unit}）');ax.set_title(title,loc='left');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#e0e4e7',lw=.5)
save(fig,'图2-4_四指标观测分布_修订')
print('DONE 图2-2、图2-3、图2-4')
