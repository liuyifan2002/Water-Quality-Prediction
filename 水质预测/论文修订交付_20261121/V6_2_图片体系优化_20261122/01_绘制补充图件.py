"""仅从已冻结的V6候选汇总表绘制两张补充论文图。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'05_v6_candidate'/'data'
OUT=HERE/'正式论文图片';OUT.mkdir(parents=True,exist_ok=True)
blue,orange,purple,grey='#4477AA','#EE9944','#AA3377','#777777'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
 'font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,'figure.facecolor':'white'})
core=pd.read_csv(SRC/'正式十任务指标.csv')
ab=pd.read_csv(SRC/'四任务六来源消融.csv')
assert len(core)==10 and len(ab)==24
order=['chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']
core=core.set_index('task').loc[order].reset_index()

def save(fig,name):
    for ext in ('png','pdf','svg'):
        kw={'dpi':600} if ext=='png' else {}
        fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',pad_inches=.06,**kw)
    plt.close(fig)

# 无量纲相对变化允许十任务同图比较；负值为TabICLv2误差下降。
fig,axs=plt.subplots(1,2,figsize=(15.5/2.54,11.5/2.54),sharey=True)
fig.subplots_adjust(left=.20,right=.96,bottom=.12,top=.94,wspace=.20)
y=np.arange(10)
for ax,metric in zip(axs,('RMSE','MAE')):
    v=100*(core[f'TabICLv2_{metric}']-core[f'XGBoost_{metric}'])/core[f'XGBoost_{metric}']
    color=np.where(v<=0,blue,purple)
    ax.barh(y,v,color=color,height=.62)
    ax.axvline(0,color='#333333',lw=.9)
    ax.set_xlim(min(-25,float(v.min())-3),max(10,float(v.max())+3))
    ax.set_yticks(y,core['任务'])
    if metric=='MAE': ax.tick_params(axis='y',labelleft=False)
    if metric=='RMSE': ax.invert_yaxis()
    ax.set_xlabel(f'{metric}相对变化（%）')
    ax.set_title(f'({"a" if metric=="RMSE" else "b"}) {metric}',loc='left')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',color='#d9e1e6',lw=.5);ax.set_axisbelow(True)
    for i,z in enumerate(v): ax.annotate(f'{z:+.1f}',(z,i),xytext=(3 if z>=0 else -3,0),
             textcoords='offset points',ha='left' if z>=0 else 'right',va='center',fontsize=7.6)
save(fig,'图4-6_十任务RMSE与MAE相对变化')

# 消融后RMSE变化采用各任务独立尺度和单位，避免跨目标量纲混读。
groups=['MET','RHT','OCN','HUM','TMP','SPA']
names={'MET':'气象','RHT':'降雨、水文与潮汐','OCN':'海洋物理','HUM':'人类活动代理','TMP':'时间过程','SPA':'空间静态'}
tasks=['chla_surface','secchi','tn_surface','tp_middle']
units={'chla_surface':'μg/L','secchi':'m','tn_surface':'mg/L','tp_middle':'mg/L'}
fig,axs=plt.subplots(4,1,figsize=(15.5/2.54,18.0/2.54))
fig.subplots_adjust(left=.25,right=.95,bottom=.07,top=.97,hspace=.63)
for ax,t in zip(axs,tasks):
    d=ab[ab.task==t].set_index('来源').loc[groups]
    v=d.delta_RMSE.to_numpy(float); pos=np.arange(6)
    ax.barh(pos,v,color=np.where(v>=0,purple,blue),height=.64)
    ax.axvline(0,color='#333333',lw=.9)
    ax.set_yticks(pos,[names[g] for g in groups]);ax.invert_yaxis()
    ax.set_title(core.set_index('task').loc[t,'任务'],loc='left')
    ax.set_xlabel(f'删组−完整模型 ΔRMSE（{units[t]}）')
    span=max(abs(v.min()),abs(v.max()));ax.set_xlim(-1.6*span,1.6*span)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',color='#d9e1e6',lw=.5);ax.set_axisbelow(True)
    for i,z in enumerate(v): ax.annotate(f'{z:+.5f}' if t in ('tn_surface','tp_middle') else f'{z:+.4f}',(z,i),xytext=(2 if z>=0 else -2,0),
              textcoords='offset points',ha='left' if z>=0 else 'right',va='center',fontsize=7.5)
save(fig,'图5-5_四任务六来源删组RMSE变化')

# 附录B：十任务的全部850条站点误差，按同指标分面，不抽选好坏站点。
site=pd.read_csv(SRC/'十任务逐站点模型差.csv')
meta=pd.read_csv(HERE.parents[1]/'数据/处理后数据/正式主表/station_meta.csv')
site=site.merge(meta[['Station','lon']],left_on='station',right_on='Station',validate='many_to_one')
assert len(site)==850 and site.lon.notna().all()
panels=[('Chl-a',['chla_surface','chla_middle','chla_bottom'],'μg/L'),
        ('Secchi',['secchi'],'m'),('TN',['tn_surface','tn_middle','tn_bottom'],'mg/L'),
        ('TP',['tp_surface','tp_middle','tp_bottom'],'mg/L')]
for number,(indicator,subtasks,unit) in enumerate(panels,1):
    fig,axs=plt.subplots(len(subtasks),1,figsize=(15.5/2.54,(5.5 if len(subtasks)==1 else 13.8)/2.54),squeeze=False)
    fig.subplots_adjust(left=.16,right=.96,bottom=.12,top=.92,hspace=.42)
    for ax,task in zip(axs.flat,subtasks):
        x=site[site.task==task].sort_values('lon')
        ax.scatter(x.lon,x.delta_RMSE,c=np.where(x.delta_RMSE<0,blue,purple),s=12,alpha=.82,linewidths=0)
        ax.axhline(0,color='#333333',lw=.8);ax.set_xlim(113.8,114.55)
        ax.set_title(f'{core.set_index("task").loc[task,"任务"]}（{len(x)}站）',loc='left')
        ax.set_ylabel(f'ΔRMSE（{unit}）');ax.set_xlabel('站点经度（°E）')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#d9e1e6',lw=.5);ax.set_axisbelow(True)
    save(fig,f'图B-{number}_{indicator}逐站点RMSE变化')
