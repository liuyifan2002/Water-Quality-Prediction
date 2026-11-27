"""修复版77特征的论文结果图。只读取冻结指标和逐站点误差。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
SRC = ROOT / '论文修订交付_20261121/05_v6_candidate/data'
FIG = OUT / '正式论文图片'
FIG.mkdir(exist_ok=True)
core = pd.read_csv(SRC / '正式十任务指标.csv')
modern = pd.read_csv(SRC / '四模型同协议指标.csv')
ab = pd.read_csv(SRC / '四任务六来源消融.csv')
regional = pd.read_csv(SRC / '历史78特征分区汇总.csv')
fold = pd.read_csv(SRC / '历史78特征分区逐折.csv')
site = pd.read_csv(SRC / '十任务逐站点模型差.csv')
assert len(core) == 10 and len(modern) == 16 and len(ab) == 24 and len(regional) == 12
assert set(core.task) == set(site.task)
blue, orange, green, purple, grey = '#4477AA', '#EE9944', '#228833', '#AA3377', '#777777'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
 'font.size':8.5,'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,'figure.facecolor':'white'})

def save(fig, stem):
    fig.savefig(FIG / (stem+'.png'), dpi=600, bbox_inches='tight', pad_inches=.07)
    fig.savefig(FIG / (stem+'.pdf'), bbox_inches='tight', pad_inches=.07)
    fig.savefig(FIG / (stem+'.svg'), bbox_inches='tight', pad_inches=.07)
    plt.close(fig)

def tidy(ax, axis='x'):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis=axis, color='#d9e1e6', lw=.5)
    ax.set_axisbelow(True)

# 4-1：配对模型的任务内pooled R²；负结果原样呈现。
fig, ax = plt.subplots(figsize=(15.5/2.54, 10.5/2.54))
fig.subplots_adjust(left=.20,right=.96,bottom=.12,top=.93)
y = np.arange(len(core))
ax.hlines(y, core.XGBoost_R2, core.TabICLv2_R2, color='#bdc7ce', lw=1.4)
ax.scatter(core.XGBoost_R2,y,c=blue,s=30,label='XGBoost',zorder=3)
ax.scatter(core.TabICLv2_R2,y,c=orange,s=30,label='TabICLv2',zorder=3)
ax.axvline(0,color=grey,lw=.8)
ax.set_yticks(y,core['任务']); ax.invert_yaxis(); ax.set_xlabel('pooled OOF R²')
ax.legend(frameon=False,ncol=2,loc='lower right'); tidy(ax)
save(fig,'图4-1_十任务模型预测性能比较')

# 4-2：零中心的模型差；站点胜率作为独立一列，二者不共用量纲。
fig,(a,b)=plt.subplots(1,2,figsize=(15.5/2.54,10.5/2.54),gridspec_kw={'width_ratios':[1.25,.8]})
fig.subplots_adjust(left=.20,right=.96,bottom=.13,top=.89,wspace=.25)
colors=np.where(core.delta_R2>=0,blue,purple)
a.barh(y,core.delta_R2,color=colors,height=.62); a.axvline(0,color='#333',lw=1)
a.set_yticks(y,core['任务']); a.invert_yaxis(); a.set_xlabel('TabICLv2 − XGBoost：pooled ΔR²')
a.set_title('(a) 总体精度变化',loc='left'); tidy(a)
for i,v in enumerate(core.delta_R2): a.text(v+(.003 if v>=0 else -.003),i,f'{v:+.3f}',ha='left' if v>=0 else 'right',va='center',fontsize=7.5)
b.scatter(core.station_win_rate*100,y,color=blue,s=27,zorder=3);b.hlines(y,50,core.station_win_rate*100,color='#b9c7d0',lw=1.2)
b.axvline(50,color=grey,ls='--',lw=.8);b.set_xlim(45,105);b.set_yticks(y,[]);b.invert_yaxis()
b.set_xlabel('RMSE改善站点比例（%）');b.set_title('(b) 站点覆盖',loc='left');tidy(b)
for i,v in enumerate(core.station_win_rate*100): b.annotate(f'{v:.1f}',(v,i),xytext=(3,0),textcoords='offset points',va='center',fontsize=7.5)
save(fig,'图4-2_十任务预测性能提升与退化')

# 4-3：同一任务内按站点比较RMSE；log2比值不受目标量纲影响，保留全部站点。
ss=site.merge(core[['task','任务']],on='task',validate='many_to_one')
assert (ss.tabiclv2>0).all() and (ss.xgboost>0).all()
ss['log2_RMSE_ratio']=np.log2(ss.tabiclv2/ss.xgboost)
fig,ax=plt.subplots(figsize=(15.5/2.54,10.7/2.54));fig.subplots_adjust(left=.20,right=.96,bottom=.14,top=.92)
for i,r in core.iterrows():
    v=ss.loc[ss.task==r.task,'log2_RMSE_ratio'].to_numpy()
    if len(v):
        vp=ax.violinplot(v,[i],vert=False,widths=.68,showmeans=False,showmedians=False,showextrema=False)
        for body in vp['bodies']:body.set_facecolor(blue);body.set_edgecolor('none');body.set_alpha(.23)
        ax.scatter(np.median(v),i,c=blue,s=20,zorder=4)
        ax.plot(np.percentile(v,[25,75]),[i,i],color=blue,lw=2.3,zorder=3)
ax.axvline(0,color='#333',lw=.9);ax.set_yticks(y,core['任务']);ax.invert_yaxis();ax.set_xlabel(r'逐站点 $\log_{2}(\mathrm{RMSE}_{TabICLv2}/\mathrm{RMSE}_{XGBoost})$')
ax.set_xlim(ss.log2_RMSE_ratio.min()-.15,ss.log2_RMSE_ratio.max()+.15)
ax.text(.99,.98,'点：中位数；粗线：四分位范围；淡色：站点分布',transform=ax.transAxes,ha='right',va='top',fontsize=7.6)
tidy(ax);save(fig,'图4-3_站点层面的预测误差分布')

# 4-4：真实坐标、同一地理范围；每任务以自身单位对称色标。
meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv')
assert len(meta)==94 and meta.Station.is_unique
import geopandas as gpd
land_path=Path(__file__).resolve().parent/'底图数据/ne_10m_land.shp'
assert land_path.is_file()
land=gpd.read_file(land_path,bbox=(113.82,22.08,114.52,22.65))
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,14.5/2.54));fig.subplots_adjust(left=.12,right=.95,bottom=.10,top=.95,wspace=.32,hspace=.68)
for ax,task,unit in zip(axs.flat,['chla_surface','secchi','tn_surface','tp_middle'],['μg/L','m','mg/L','mg/L']):
    d=ss[ss.task==task].merge(meta,left_on='station',right_on='Station',validate='one_to_one')
    lim=max(float(np.quantile(np.abs(d.delta_RMSE),.95)),1e-9)
    land.plot(ax=ax,color='#e8ecea',edgecolor='#a5b2ad',linewidth=.35,zorder=1)
    sc=ax.scatter(d.lon,d.lat,c=d.delta_RMSE,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),s=18,edgecolors='white',linewidth=.3)
    ax.set_xlim(113.82,114.52);ax.set_ylim(22.08,22.65);ax.set_aspect(1/np.cos(np.deg2rad(22.35)))
    ax.set_title(f'{core.set_index("task").loc[task,"任务"]}（{len(d)}站）',loc='left');ax.set_ylabel('纬度（°N）')
    cb=fig.colorbar(sc,ax=ax,orientation='horizontal',fraction=.05,pad=.15,extend='both')
    cb.set_ticks([-lim,0,lim]);cb.set_label(f'ΔRMSE（{unit}）',fontsize=8);cb.ax.tick_params(labelsize=7)
save(fig,'图4-4_四项代表任务的空间误差地图')

# 4-5：四模型按同任务并列，四面板采用共同R²范围。
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,10.7/2.54));fig.subplots_adjust(left=.18,right=.96,bottom=.12,top=.93,wspace=.32,hspace=.45)
models=['xgboost','tabiclv2','tabpfn','tabm'];names=['XGBoost','TabICLv2','TabPFN','TabM'];cols=[blue,orange,green,purple]
lo=min(0,modern.pooled_R2.min()-.04);hi=max(.7,modern.pooled_R2.max()+.04)
for ax,task in zip(axs.flat,['chla_surface','secchi','tn_surface','tp_middle']):
    d=modern[modern.task==task].set_index('model').loc[models]
    ax.scatter(d.pooled_R2,range(4),c=cols,s=30,zorder=3);ax.set_yticks(range(4),names);ax.invert_yaxis();ax.set_xlim(lo,hi)
    ax.set_xlabel('pooled OOF R²');ax.set_title(core.set_index('task').loc[task,'任务'],loc='left');tidy(ax)
    for i,v in enumerate(d.pooled_R2):ax.annotate(f'{v:.3f}',(v,i),xytext=(4,0),textcoords='offset points',va='center',fontsize=7.5)
save(fig,'图4-5_四种模型的性能比较')

# 5-1：实际仅四项任务有六来源消融；空缺任务不扩充。
tasks=['chla_surface','secchi','tn_surface','tp_middle'];groups=['MET','RHT','OCN','HUM','TMP','SPA']
m=ab.pivot(index='来源',columns='task',values='delta_R2').loc[groups,tasks]
fig,ax=plt.subplots(figsize=(15.5/2.54,7.5/2.54));fig.subplots_adjust(left=.16,right=.84,bottom=.18,top=.91)
lim=max(abs(m.min().min()),abs(m.max().max()));im=ax.imshow(m,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
ax.set_xticks(range(4),[core.set_index('task').loc[t,'任务'] for t in tasks]);ax.set_yticks(range(6),groups)
for i in range(6):
 for j in range(4):ax.text(j,i,f'{m.iloc[i,j]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(m.iloc[i,j])>.65*lim else '#222')
cb=fig.colorbar(im,ax=ax,fraction=.05,pad=.035);cb.set_label('删组模型 − 完整模型：pooled ΔR²')
save(fig,'图5-1_六类环境信息来源的消融效应')

# 5-2：同一消融结果按指标汇总，仅描述四个代表任务；数值不当作因果贡献。
fig,ax=plt.subplots(figsize=(15.5/2.54,8.3/2.54));fig.subplots_adjust(left=.14,right=.97,bottom=.18,top=.88)
for j,t in enumerate(tasks):
    d=ab[ab.task==t].set_index('来源').loc[groups]
    x=np.arange(6)+(j-1.5)*.16
    ax.scatter(x,-d.delta_R2,s=30,color=[blue,orange,green,purple][j],label=core.set_index('task').loc[t,'任务'],zorder=3)
ax.axhline(0,color='#333',lw=.8);ax.set_xticks(range(6),groups);ax.set_ylabel('删除来源造成的 pooled R²损失（−ΔR²）')
ax.legend(frameon=False,ncol=4,loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=7.6);tidy(ax,'y')
save(fig,'图5-2_不同环境信息来源的任务依赖')

# 5-3：历史78特征分区，与正式77特征严格隔离。
rt=regional.copy();rt['task']=rt.target+'_'+rt.depth
rt['名称']=rt.target.map({'chla':'Chl-a','secchi':'Secchi','tn':'TN','tp':'TP'})+' '+rt.depth.map({'surface':'表层','middle':'中层','bottom':'底层'})
f=fold[fold.model_variant.isin(['global_external','zone_aware_external'])].pivot_table(index=['target','depth','fold'],columns='model_variant',values='r2')
f['delta']=f.zone_aware_external-f.global_external
fm=f['delta'].unstack('fold').reindex(pd.MultiIndex.from_frame(rt[['target','depth']]))
fig,(a,b)=plt.subplots(1,2,figsize=(15.5/2.54,11.5/2.54),gridspec_kw={'width_ratios':[.9,1.1]});fig.subplots_adjust(left=.16,right=.94,bottom=.14,top=.91,wspace=.21)
yy=np.arange(12);v=rt.delta_r2_pooled.to_numpy();a.hlines(yy,0,v,color=np.where(v>=0,blue,purple),lw=2);a.scatter(v,yy,c=np.where(v>=0,blue,purple),s=24)
a.axvline(0,color='#333',lw=.8);a.set_yticks(yy,rt['名称']);a.invert_yaxis();a.set_xlabel('分区 − 全域：pooled ΔR²');a.set_xlim(-.23,.12);tidy(a)
for i,r in rt.iterrows():a.text(.115,i,f'{r.positive_folds}/5',ha='right',va='center',fontsize=7)
mm=fm.to_numpy();lim=.5;im=b.imshow(mm,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
b.set_xticks(range(5),[f'折{i}' for i in range(1,6)]);b.set_yticks(yy,[]);b.set_xlabel('逐折 ΔR²')
for i in range(12):
 for j in range(5):b.text(j,i,f'{mm[i,j]:+.2f}',ha='center',va='center',fontsize=6.5,color='white' if abs(mm[i,j])>.7*lim else '#222')
fig.colorbar(im,ax=b,fraction=.05,pad=.035,extend='min').set_label('逐折 ΔR²；低于−0.5饱和显示')
save(fig,'图5-3_历史全域与分区模型比较')
print('CREATED 8 current-version figure sets')
