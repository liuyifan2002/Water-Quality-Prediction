"""WQ77候选版正文结果图；只使用已冻结指标和站点误差。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
DATA=OUT/'data'; FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
core=pd.read_csv(DATA/'正式十任务指标.csv')
modern=pd.read_csv(DATA/'四模型同协议指标.csv')
ab=pd.read_csv(DATA/'四任务六来源消融.csv')
reg=pd.read_csv(DATA/'历史78特征分区汇总.csv')
fold=pd.read_csv(DATA/'历史78特征分区逐折.csv')
site=pd.read_csv(DATA/'十任务逐站点模型差.csv')
meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv')
assert len(meta)==94 and meta.Station.is_unique
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],'font.size':8.5,'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'savefig.facecolor':'white'})
BLUE='#4477AA';ORANGE='#EE9944';GREEN='#228833';PURPLE='#AA3377';GRAY='#777777';LIGHT='#eef2f5'

def finish(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=600,bbox_inches='tight',pad_inches=.06)
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

def simple(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',color='#dbe2e6',lw=.5,zorder=0)
    ax.set_axisbelow(True)

# 图4-1：固定正文表4-2顺序，保留不利结果。
fig,(a,b)=plt.subplots(1,2,figsize=(15.5/2.54,10.0/2.54),gridspec_kw={'width_ratios':[1.25,1]})
fig.subplots_adjust(left=.14,right=.98,bottom=.12,top=.88,wspace=.38)
y=np.arange(10);a.hlines(y,core.XGBoost_R2,core.TabICLv2_R2,color='#aeb9c2',lw=1.1)
a.scatter(core.XGBoost_R2,y,s=20,c=BLUE,zorder=4,label='XGBoost');a.scatter(core.TabICLv2_R2,y,s=20,c=ORANGE,zorder=4,label='TabICLv2')
a.axvline(0,color=GRAY,lw=.7);a.set_yticks(y,core['任务']);a.invert_yaxis();a.set_xlabel('pooled OOF R²');a.set_title('(a) 十项任务总体精度',loc='left');a.legend(loc='lower right',frameon=False,ncol=2,fontsize=7.6);simple(a)
b.barh(y,core.station_win_rate*100,color=BLUE,height=.62);b.axvline(50,color=GRAY,ls='--',lw=.8);b.set_yticks(y,[]);b.invert_yaxis();b.set_xlim(0,105);b.set_xlabel('RMSE改善站点比例（%）');b.set_title('(b) 站点改善覆盖',loc='left');simple(b)
for i,v in enumerate(core.station_win_rate*100):b.text(v+1.3,i,f'{v:.1f}%',va='center',fontsize=7.5)
finish(fig,'图4-1_十任务总体提升与站点改善覆盖')

# 图4-2：两个色标独立，右侧留足边界。
tasks=[f'{q}_{z}' for q in ['chla','tn','tp'] for z in ['surface','middle','bottom']]
mat=np.array([[core.set_index('task').loc[f'{q}_{z}','delta_R2'] for z in ['surface','middle','bottom']] for q in ['chla','tn','tp']])
win=np.array([[core.set_index('task').loc[f'{q}_{z}','station_win_rate']*100 for z in ['surface','middle','bottom']] for q in ['chla','tn','tp']])
fig,axs=plt.subplots(1,2,figsize=(15.5/2.54,8.2/2.54));fig.subplots_adjust(left=.09,right=.93,bottom=.22,top=.84,wspace=.48)
for ax,arr,title,cmap,norm,fmt in [(axs[0],mat,'(a) pooled ΔR²','RdBu_r',TwoSlopeNorm(vmin=min(-.15,float(mat.min())*1.1),vcenter=0,vmax=max(.15,float(mat.max())*1.1)),'{:+.3f}'),(axs[1],win,'(b) RMSE改善站点比例','Blues',None,'{:.1f}%')]:
    im=ax.imshow(arr,cmap=cmap,norm=norm,vmin=0 if norm is None else None,vmax=100 if norm is None else None)
    ax.set_xticks(range(3),['表层','中层','底层']);ax.set_yticks(range(3),['Chl-a','TN','TP']);ax.set_title(title,loc='left')
    for i in range(3):
        for j in range(3):ax.text(j,i,fmt.format(arr[i,j]),ha='center',va='center',color='white' if (arr[i,j]>70 if norm is None else abs(arr[i,j])>.65*max(abs(norm.vmin),norm.vmax)) else '#222',fontsize=8)
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.065);cb.ax.tick_params(labelsize=7.5)
fig.text(.09,.06,'九个分层任务；Secchi为独立任务，见图4-1。两面板使用独立色标。',fontsize=8)
finish(fig,'图4-2_不同指标与水层的模型提升差异')

# 图4-4：同站点冻结RMSE差值；不跨目标共用色标。
land=None
try:
    import geopandas as gpd
    shp=Path('E:/研究生/gis/natural_earth_vector/10m_physical/ne_10m_land.shp')
    if shp.is_file():land=gpd.read_file(shp,bbox=(113.8,22.05,114.55,22.7))
except Exception:pass
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,12.5/2.54));fig.subplots_adjust(left=.11,right=.94,bottom=.12,top=.94,wspace=.30,hspace=.50)
for ax,task,unit in zip(axs.flat,['chla_surface','secchi','tn_bottom','tp_middle'],['μg/L','m','mg/L','mg/L']):
    q=site[site.task==task].merge(meta,left_on='station',right_on='Station',validate='one_to_one')
    assert len(q)==(94 if task=='secchi' else 84)
    if land is not None:land.plot(ax=ax,color='#e5e8e5',edgecolor='#aab2af',linewidth=.3)
    lim=max(abs(q.delta_RMSE.min()),abs(q.delta_RMSE.max()));lim=max(lim,1e-9)
    sc=ax.scatter(q.lon,q.lat,c=q.delta_RMSE,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),s=16,edgecolor='white',linewidth=.35,zorder=3)
    ax.set_xlim(113.82,114.52);ax.set_ylim(22.08,22.65);ax.set_aspect(1/np.cos(np.deg2rad(22.35)));ax.set_title(f'{core.set_index("task").loc[task,"任务"]}（{len(q)}站）',loc='left')
    ax.set_xlabel('经度（°E）');ax.set_ylabel('纬度（°N）');ax.tick_params(labelsize=7)
    cb=fig.colorbar(sc,ax=ax,orientation='horizontal',fraction=.05,pad=.11);cb.set_label(f'ΔRMSE（{unit}）',fontsize=8);cb.ax.tick_params(labelsize=7)
finish(fig,'图4-4_四项代表任务站点RMSE变化空间分布')

# 图4-5：四模型、同一WQ77候选任务和留站折。
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,11/2.54));fig.subplots_adjust(left=.16,right=.97,bottom=.10,top=.94,wspace=.32,hspace=.45)
models=['xgboost','tabiclv2','tabpfn','tabm'];labels=['XGBoost','TabICLv2','TabPFN','TabM'];colors=[BLUE,ORANGE,PURPLE,GRAY]
for ax,task in zip(axs.flat,['chla_surface','secchi','tn_surface','tp_middle']):
    d=modern[modern.task==task].set_index('model').loc[models]
    x=d.pooled_R2.to_numpy();ax.scatter(x,range(4),c=colors,s=30,zorder=3);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(min(0,float(x.min())-.04),min(1,max(.7,float(x.max())+.08)));ax.set_xlabel('pooled OOF R²');ax.set_title(core.set_index('task').loc[task,'任务'],loc='left');simple(ax)
    for i,v in enumerate(x):ax.annotate(f'{v:.3f}',(v,i),xytext=(4,0),textcoords='offset points',va='center',fontsize=7.7)
finish(fig,'图4-5_四项代表任务四模型比较')

# 图5-1：删除一个来源后的R²差；XGBoost，四项任务共用尺度。
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,11/2.54));fig.subplots_adjust(left=.15,right=.97,bottom=.12,top=.94,wspace=.34,hspace=.44)
groups=['MET','RHT','OCN','HUM','TMP','SPA']; lim=max(abs(ab.delta_R2.min()),abs(ab.delta_R2.max()))*1.25
for ax,task in zip(axs.flat,['chla_surface','secchi','tn_surface','tp_middle']):
    d=ab[ab.task==task].set_index('来源').loc[groups]; v=d.delta_R2.to_numpy();y=np.arange(6)
    ax.barh(y,v,color=[BLUE if x<0 else ORANGE for x in v],height=.6);ax.axvline(0,c=GRAY,lw=.8);ax.set_yticks(y,groups);ax.invert_yaxis();ax.set_xlim(-lim,lim);ax.set_xlabel('删除来源后的pooled ΔR²');ax.set_title(core.set_index('task').loc[task,'任务'],loc='left');simple(ax)
    for i,x in enumerate(v):ax.annotate(f'{x:+.3f}',(x,i),xytext=(3 if x>=0 else -3,0),textcoords='offset points',ha='left' if x>=0 else 'right',va='center',fontsize=7.4)
finish(fig,'图5-2_四项代表任务的多源环境信息消融比较')

# 图5-1：历史78特征分区结果与正式77特征隔离。
order=[(q,z) for q in ['chla','secchi','tn','tp'] for z in ['surface','middle','bottom']]
reg=reg.set_index(['target','depth']).loc[order]; pv=fold.pivot(index=['target','depth','fold'],columns='model_variant',values='r2')
matrix=np.array([[pv.loc[(q,z,k),'zone_aware_external']-pv.loc[(q,z,k),'global_external'] for k in range(1,6)] for q,z in order]);names=[('Chl-a' if q=='chla' else q.upper() if q!='secchi' else 'Secchi')+{'surface':'表层','middle':'中层','bottom':'底层'}[z] for q,z in order]
fig,(a,b)=plt.subplots(1,2,figsize=(15.5/2.54,15/2.54),gridspec_kw={'width_ratios':[1,1.12]});fig.subplots_adjust(left=.22,right=.91,bottom=.13,top=.93,wspace=.43)
v=reg.delta_r2_pooled.to_numpy();y=np.arange(12);a.hlines(y,0,v,color=[ORANGE if x>=0 else BLUE for x in v]);a.scatter(v,y,c=[ORANGE if x>=0 else BLUE for x in v],s=18);a.axvline(0,c=GRAY,lw=.7);a.set_yticks(y,names);a.set_ylim(11.5,-.5);a.set_xlabel('pooled ΔR²');a.set_title('(a) 合并OOF差值',loc='left');simple(a)
for i,x in enumerate(v):a.annotate(f'{x:+.3f}',(x,i),xytext=(3 if x>=0 else -3,0),textcoords='offset points',ha='left' if x>=0 else 'right',va='center',fontsize=7)
lim=np.abs(matrix).max();im=b.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto');b.set_yticks(range(12),[]);b.set_xticks(range(5),range(1,6));b.set_xlabel('留站折次');b.set_title('(b) 逐折ΔR²',loc='left')
for i in range(12):
    for j in range(5):b.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontsize=7,color='white' if abs(matrix[i,j])>.7*lim else '#222')
cb=fig.colorbar(im,ax=b,orientation='horizontal',fraction=.04,pad=.08);cb.ax.tick_params(labelsize=7)
fig.text(.03,.04,'历史78特征（含water_zone_code）；12组合、60任务折；区域训练可回退全域模型。',fontsize=7.5)
finish(fig,'图5-1_历史全域与分区建模比较')

# 图5-3：只更新已有五个案例的正式误差；缺失率为历史表层覆盖背景。
case=pd.read_csv(ROOT/'文档/论文材料/硕士论文V2_学术内容补强_20261101/站点误差与特征覆盖补充核对.csv')
assert len(case)==5 and case.station_id.nunique()==4
pair=site.set_index(['task','station'])
site_counts=pd.read_csv(DATA/'十任务逐站点误差.csv')
fig=plt.figure(figsize=(15.5/2.54,19.5/2.54));gs=fig.add_gridspec(5,2,left=.17,right=.95,bottom=.10,top=.94,hspace=1.28,wspace=.42)
a=fig.add_subplot(gs[:2,0]);
if land is not None:land.plot(ax=a,color='#e5e8e5',edgecolor='#aab2af',linewidth=.3)
for sid in sorted(case.station_id.unique()):
    x=meta.set_index('Station').loc[sid];a.scatter(x.lon,x.lat,c=BLUE,s=22);a.annotate(sid,(x.lon,x.lat),xytext=(3,3),textcoords='offset points',fontsize=7.5)
a.set_xlim(113.82,114.52);a.set_ylim(22.08,22.65);a.set_title('(a) 典型站点位置',loc='left')
b=fig.add_subplot(gs[:2,1]);q=case.drop_duplicates('station_id');b.barh(range(4),q.missing_77_share_in_main_surface*100,color=GRAY,height=.55);b.set_yticks(range(4),q.station_id);b.invert_yaxis();b.set_xlabel('表层77特征缺失率（%）');b.set_title('(b) 原登记表层输入覆盖',loc='left');simple(b)
for i,r in enumerate(case.itertuples()):
    ax=fig.add_subplot(gs[2+i//2,i%2]) if i<4 else fig.add_subplot(gs[4,:]);s=pair.loc[(r.task,r.station_id)];vals=[float(s.xgboost),float(s.tabiclv2)];n=int(site_counts[(site_counts.task==r.task)&(site_counts.station==r.station_id)&(site_counts.model=='xgboost')]['n'].iloc[0]);unit='μg/L' if r.task.startswith('chla') else 'mg/L' if r.task.startswith(('tn','tp')) else 'm'
    ax.plot(vals,[0,0],c='#b7c1c9',lw=2);ax.scatter(vals,[0,0],c=[BLUE,ORANGE],s=25);ax.set_xlim(0,max(vals)*1.45);ax.set_ylim(-.55,.55);ax.set_yticks([]);ax.set_title(f'{r.station_id} · {core.set_index("task").loc[r.task,"任务"]}（n={n}，RMSE：{unit}）',loc='left',fontsize=8,pad=8);simple(ax)
    for j,v in enumerate(vals):ax.annotate(f'{v:.3f}',(v,0),xytext=(0,-13 if j==0 else 8),textcoords='offset points',ha='center',va='top' if j==0 else 'bottom',fontsize=7)
fig.legend(handles=[Line2D([],[],marker='o',color=BLUE,ls='',label='XGBoost'),Line2D([],[],marker='o',color=ORANGE,ls='',label='TabICLv2')],loc='lower right',ncol=2,frameon=False,fontsize=8)
finish(fig,'图5-3_四个典型站点输入覆盖与预测误差诊断')
print('PASS: 7 candidate PNG/PDF figures')
