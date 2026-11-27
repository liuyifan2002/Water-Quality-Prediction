"""V6.3 附加图件：仅读取 WQ77_REPAIRED_20261115_V1 冻结资产。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
FIG=HERE/'正式论文图片'; FIG.mkdir(exist_ok=True)
EXP=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
DATA=ROOT/'论文修订交付_20261121/05_v6_candidate/data'
CORE=pd.read_csv(DATA/'正式十任务指标.csv')
STATION=pd.read_csv(DATA/'十任务逐站点模型差.csv')
BOOT=pd.read_csv(ROOT/'论文修订交付_20261121/V6_1_科研深化与全文重建/十任务Bootstrap结果.csv')
TASKS=CORE.task.tolist(); NAMES=dict(zip(CORE.task,CORE['任务']))
UNITS={t:('μg/L' if t.startswith('chla') else 'm' if t=='secchi' else 'mg/L') for t in TASKS}
BLUE='#4477AA'; ORANGE='#EE9944'; GREEN='#228833'; PURPLE='#AA3377'; GRAY='#777777'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],'font.size':8.4,'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,num,short):
 stem=f'图{num}_{short}'
 for ext in ['png','pdf','svg']:
  fig.savefig(FIG/f'{stem}.{ext}',dpi=600 if ext=='png' else None,bbox_inches='tight',pad_inches=.09)
 plt.close(fig);print('DONE',stem)
def tidy(ax,axis='x'):
 ax.spines[['top','right']].set_visible(False);ax.grid(axis=axis,color='#dbe2e6',lw=.5);ax.set_axisbelow(True)
def panel(ax,title):
 ax.set_title(title,loc='left',fontweight='bold');tidy(ax)

# 2-5：来源类别数量仅按冻结清单及其特征定义统计。
feat_dir=EXP/'02_冻结候选'
feat_path=next(p for p in feat_dir.glob('*77*csv') if len(pd.read_csv(p,nrows=1).columns)==10)
features=pd.read_csv(feat_path)
assert len(features)==77 and features.feature.is_unique
def source_group(r):
 g=r['group']; f=r['feature']
 if g=='ERA5':return 'MET' if f.startswith('ssrd_') else 'OCN'
 return {'气象':'MET','降雨':'RHT','潮汐':'RHT','夜间灯光':'HUM','时间':'TMP','空间与站点':'SPA'}[g]
features['source_group']=features.apply(source_group,axis=1)
counts=features.source_group.value_counts().reindex(['MET','RHT','OCN','HUM','TMP','SPA'])
assert counts.to_dict()=={'MET':29,'RHT':22,'OCN':11,'HUM':6,'TMP':7,'SPA':2},counts
fig,ax=plt.subplots(figsize=(15.5/2.54,7.8/2.54));fig.subplots_adjust(left=.04,right=.97,top=.91,bottom=.08);ax.axis('off')
labels=['气象条件','降雨、水文与潮汐','海洋物理','人类活动代理','时间过程','空间位置']
cols=[BLUE,'#6699BB',GREEN,ORANGE,PURPLE,GRAY]
for i,(key,label,c,n) in enumerate(zip(counts.index,labels,cols,counts)):
 x=.025+i*.16;ax.add_patch(Rectangle((x,.40),.145,.31,facecolor=c,alpha=.13,edgecolor=c,lw=1.1,transform=ax.transAxes))
 ax.text(x+.0725,.60,f'{key} · {n}项',ha='center',va='center',fontweight='bold',color=c,transform=ax.transAxes)
 ax.text(x+.0725,.48,label,ha='center',va='center',fontsize=8,transform=ax.transAxes)
 ax.plot([x+.0725,x+.0725],[.40,.28],color=c,lw=1.1,transform=ax.transAxes)
ax.text(.5,.83,'真实环境数据 → 时间/空间匹配与特征构造',ha='center',fontweight='bold',transform=ax.transAxes)
ax.add_patch(Rectangle((.28,.11),.44,.15,facecolor='#eef2f4',edgecolor='#657982',transform=ax.transAxes));ax.text(.5,.185,'77项正式模型输入（按固定特征顺序）',ha='center',va='center',fontweight='bold',transform=ax.transAxes)
save(fig,'2-5','77项输入特征来源组成')

# 2-6：时间窗是特征协议示意，整月同期量单独显示。
fig,ax=plt.subplots(figsize=(15.5/2.54,8/2.54));fig.subplots_adjust(left=.35,right=.96,top=.90,bottom=.18)
ax.set_xlim(-33,5);ax.set_ylim(-.6,6.2);ax.axvline(0,color='#222',lw=1.4);ax.text(0,6.0,'采样日 t',ha='center')
rows=[('自然日 3 d',-3,0,BLUE,5),('自然日 7 d',-7,0,BLUE,4),('自然日 14 d',-14,0,BLUE,3),('同站前序采样 lag',-29,-10,ORANGE,2),('自然月累计降雨／夜光月合成',-24,4,PURPLE,1),('日尺度潮汐匹配',-.5,.5,GREEN,0)]
for label,x0,x1,c,y in rows:
 ax.broken_barh([(x0,x1-x0)],(y-.27,.54),facecolors=c,alpha=.55)
ax.set_xticks([-30,-24,-14,-7,-3,0],['前30日','月内起点*','−14 d','−7 d','−3 d','采样日']);ax.set_yticks([r[4] for r in rows],[r[0] for r in rows]);ax.tick_params(axis='y',length=0,pad=8);ax.set_xlabel('相对采样日期（示意）')
ax.text(.01,-.23,'* 月内起点随采样日而变。前序采样 lag 按同站记录顺序；整月降雨和月夜光可能含采样日之后的信息。',transform=ax.transAxes,fontsize=7.8)
ax.spines[['left','top','right']].set_visible(False)
save(fig,'2-6','多源特征时间窗口构造')

# 3-1：方法流程示意，不推断模型内部结构。
fig,axs=plt.subplots(1,4,figsize=(15.5/2.54,6.5/2.54));fig.subplots_adjust(left=.04,right=.98,top=.86,bottom=.12,wspace=.15)
spec=[('XGBoost','梯度提升树','当前训练样本训练','十项正式任务',BLUE),('TabICLv2','预训练表格模型','当前任务训练上下文','十项正式任务',ORANGE),('TabPFN','预训练表格模型','任务上下文预测','四项代表任务',GREEN),('TabM','表格神经网络','当前任务数据训练','四项代表任务',PURPLE)]
for ax,(name,kind,step,scope,c) in zip(axs,spec):
 ax.axis('off');ax.add_patch(Rectangle((.03,.06),.94,.88,transform=ax.transAxes,facecolor=c,alpha=.12,edgecolor=c,lw=1.2))
 for y,text0,size in [(.81,name,10),(.62,kind,8),(.43,'77项输入特征 → '+step,7.5),(.23,'水质预测输出',8),(.12,scope,7.5)]:ax.text(.5,y,text0,ha='center',va='center',fontsize=size,fontweight='bold' if y==.81 else 'normal',transform=ax.transAxes)
save(fig,'3-1','四类预测模型原理与流程')

# 3-2：站点划分数由冻结预测/manifest核对，不推算Secchi。
reg=pd.read_csv(EXP/'03_正式十任务重算/运行登记.csv')
reg=reg[(reg.status=='OK')&(reg.model=='xgboost')]
assert len(reg)==50 and reg.dataset_version.nunique()==1 and reg.dataset_version.iloc[0]=='WQ77_REPAIRED_20261115_V1'
common=reg[reg.task=='chla_surface'].sort_values('fold').n_test.tolist()
site_counts=[]
for task in ['chla_surface','secchi']:
 vals=[]
 for i in range(1,6):
  p=EXP/f'03_正式十任务重算/predictions/{task}_xgboost_fold_{i}.csv'
  vals.append(pd.read_csv(p,usecols=['station_id']).station_id.nunique())
 site_counts.append(vals)
assert site_counts[0]==[17,17,17,17,16]
fig,axs=plt.subplots(2,1,figsize=(15.5/2.54,6.9/2.54));fig.subplots_adjust(left=.19,right=.96,top=.91,bottom=.16,hspace=.55)
for ax,task,counts0,total in zip(axs,['Chl-a／TN／TP共同站点','Secchi独立站点'],site_counts,[84,94]):
 ax.barh(range(5),[total-v for v in counts0],color='#c7d5df',label='训练站点');ax.barh(range(5),counts0,left=[total-v for v in counts0],color=ORANGE,label='测试站点')
 ax.set_yticks(range(5),[f'fold {i}' for i in range(1,6)]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_title(task,loc='left');ax.set_xlabel('站点数');tidy(ax)
 for i,v in enumerate(counts0):ax.text(total-v/2,i,str(v),ha='center',va='center',fontsize=7.8)
axs[0].legend(frameon=False,ncol=2,loc='upper right',bbox_to_anchor=(1,1.36))
save(fig,'3-2','五折站点留出验证流程')

# 3-3：公式口径图，不附会具体任务分数。
fig,axs=plt.subplots(1,2,figsize=(15.5/2.54,6.1/2.54));fig.subplots_adjust(left=.07,right=.97,top=.82,bottom=.18,wspace=.18)
for ax,title,c in zip(axs,['(a) pooled OOF R²：正式主口径','(b) fold mean R²：折次描述'],[BLUE,ORANGE]):
 ax.axis('off');ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,facecolor=c,alpha=.09,edgecolor=c,lw=1.2));ax.text(.5,.82,title,ha='center',fontweight='bold',transform=ax.transAxes)
for ax,lines in zip(axs,[['五折测试样本预测合并','↓','完整 OOF 集合','↓','统一计算一次 R²'],['每折独立计算 R²','↓','第1至第5折各得一个R²','↓','五个数算术平均']]):
 for i,s in enumerate(lines):ax.text(.5,.65-i*.12,s,ha='center',va='center',transform=ax.transAxes,fontsize=8.5)
fig.text(.5,.07,'两种口径对样本加权与目标方差的处理不同，数值无需相等。',ha='center',fontsize=8)
save(fig,'3-3','pooled与fold均值R2口径')

# 4-7：按真实水区元数据汇总站点比值。
meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv')
assert len(meta)==94 and meta.Station.is_unique
ss=STATION.merge(meta[['Station','water_zone_en']],left_on='station',right_on='Station',validate='many_to_one')
ss=ss[(ss.tabiclv2>0)&(ss.xgboost>0)].copy();ss['value']=np.log2(ss.tabiclv2/ss.xgboost)
zones=meta.water_zone_en.drop_duplicates().tolist();mat=np.full((10,len(zones)),np.nan);ns=np.zeros_like(mat,dtype=int)
for i,t in enumerate(TASKS):
 for j,z in enumerate(zones):
  v=ss.loc[(ss.task==t)&(ss.water_zone_en==z),'value'];ns[i,j]=len(v)
  if len(v):mat[i,j]=v.median()
lim=max(abs(np.nanmin(mat)),abs(np.nanmax(mat)));fig,ax=plt.subplots(figsize=(15.5/2.54,9.5/2.54));fig.subplots_adjust(left=.19,right=.86,top=.93,bottom=.23)
im=ax.imshow(mat,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
ax.set_xticks(range(len(zones)),zones,rotation=50,ha='right');ax.set_yticks(range(10),[NAMES[t] for t in TASKS]);ax.set_xlabel('正式站点所属水区')
for i in range(10):
 for j in range(len(zones)):
  if ns[i,j]:ax.text(j,i,str(ns[i,j]),ha='center',va='center',fontsize=7,color='white' if abs(mat[i,j])>.6*lim else '#222')
fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label=r'站点中位 $\log_2$(RMSE比值)')
save(fig,'4-7','十任务水区空间泛化热图')

# 4-8：统一横轴，按三层显示真实pooled R²。
fig,axs=plt.subplots(1,3,figsize=(15.5/2.54,7.4/2.54),sharex=True);fig.subplots_adjust(left=.11,right=.97,top=.87,bottom=.19,wspace=.20)
lo=min(CORE[['XGBoost_R2','TabICLv2_R2']].min())-.05;hi=max(CORE[['XGBoost_R2','TabICLv2_R2']].max())+.09
for ax,prefix in zip(axs,['chla','tn','tp']):
 d=CORE.set_index('task').loc[[f'{prefix}_{x}' for x in ['surface','middle','bottom']]]
 for model,c,off in [('XGBoost',BLUE,-.08),('TabICLv2',ORANGE,.08)]:
  vals=d[f'{model}_R2'].to_numpy();ax.plot(vals,np.arange(3)+off,'-o',color=c,label=model,markersize=5)
  for i,v in enumerate(vals):ax.annotate(f'{v:.3f}',(v,i+off),xytext=(0,-10 if off<0 else 7),textcoords='offset points',ha='center',fontsize=7)
 ax.set_yticks(range(3),['表层','中层','底层']);ax.invert_yaxis();ax.set_xlim(lo,hi);ax.set_xlabel('pooled OOF R²');ax.set_title({'chla':'Chl-a','tn':'TN','tp':'TP'}[prefix]);tidy(ax)
axs[0].legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.24),fontsize=7,ncol=2)
save(fig,'4-8','三指标垂向预测性能剖面')

# 配对OOF读取与一致性检查。后续4-9、4-12、4-13共用。
PRED=EXP/'03_正式十任务重算/predictions'
def paired(task):
 a=pd.concat([pd.read_csv(PRED/f'{task}_xgboost_fold_{i}.csv',usecols=['obs_id','station_id','y_true','y_pred']) for i in range(1,6)],ignore_index=True)
 b=pd.concat([pd.read_csv(PRED/f'{task}_tabiclv2_fold_{i}.csv',usecols=['obs_id','station_id','y_true','y_pred']) for i in range(1,6)],ignore_index=True)
 assert a.obs_id.is_unique and b.obs_id.is_unique
 d=a.merge(b,on='obs_id',suffixes=('_x','_t'),validate='one_to_one')
 assert len(d)==len(a)==len(b) and (d.station_id_x==d.station_id_t).all() and np.allclose(d.y_true_x,d.y_true_t,rtol=0,atol=1e-10)
 return d
PAIRS={t:paired(t) for t in TASKS}
fig,axs=plt.subplots(5,2,figsize=(15.5/2.54,20/2.54));fig.subplots_adjust(left=.11,right=.97,top=.97,bottom=.07,wspace=.27,hspace=.82)
for ax,t in zip(axs.flat,TASKS):
 d=PAIRS[t];parts=[('实测',d.y_true_x,GRAY),('XGBoost',d.y_pred_x,BLUE),('TabICLv2',d.y_pred_t,ORANGE)]
 for label,v,c in parts:
  x=np.sort(v.to_numpy());ax.plot(x,np.arange(1,len(x)+1)/len(x),label=label,color=c,lw=1.2)
 ax.set_title(NAMES[t],loc='left');ax.set_xlabel(f'水质值（{UNITS[t]}）');ax.set_ylabel('经验累积概率');ax.set_ylim(0,1);tidy(ax)
axs[0,0].legend(frameon=False,ncol=3,fontsize=7)
save(fig,'4-9','十任务实测与预测经验累积分布')

# 4-10：每个任务自身fold，值来自冻结运行登记R²。
mat=np.zeros((10,5))
for i,t in enumerate(TASKS):
 for j in range(1,6):
  sub=reg[(reg.task==t)&(reg.fold==f'fold_{j}')]
  x=pd.read_csv(EXP/f'03_正式十任务重算/fold_metrics/{t}_xgboost_fold_{j}.csv').R2.iloc[0]
  y=pd.read_csv(EXP/f'03_正式十任务重算/fold_metrics/{t}_tabiclv2_fold_{j}.csv').R2.iloc[0]
  mat[i,j-1]=y-x
lim=max(abs(mat.min()),abs(mat.max()));fig,ax=plt.subplots(figsize=(15.5/2.54,9.5/2.54));fig.subplots_adjust(left=.19,right=.86,top=.93,bottom=.12)
im=ax.imshow(mat,cmap='RdBu',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto');ax.set_xticks(range(5),[f'fold {i}' for i in range(1,6)]);ax.set_yticks(range(10),[NAMES[t] for t in TASKS])
for i in range(10):
 for j in range(5):ax.text(j,i,f'{mat[i,j]:+.3f}',ha='center',va='center',fontsize=7.5,color='white' if abs(mat[i,j])>.7*lim else '#222')
fig.colorbar(im,ax=ax,fraction=.04,pad=.025,label='TabICLv2 − XGBoost：折次R²差')
save(fig,'4-10','十任务五折性能稳定性')

# 4-11：将已有配对站点Bootstrap的绝对差值区间除以冻结基准误差。
bb=BOOT.set_index('task').loc[TASKS];fig,axs=plt.subplots(1,2,figsize=(15.5/2.54,10.5/2.54),sharey=True);fig.subplots_adjust(left=.18,right=.96,top=.90,bottom=.13,wspace=.22)
for ax,met in zip(axs,['RMSE','MAE']):
 base=bb[f'XGBoost_pooled_{met}'];point=100*bb[f'delta_pooled_{met}']/base;low=100*bb[f'delta_{met}_CI_low']/base;high=100*bb[f'delta_{met}_CI_high']/base
 for i,(p,l,h) in enumerate(zip(point,low,high)):
  ax.plot([l,h],[i,i],color=BLUE if h<0 else PURPLE,lw=1.5);ax.scatter(p,i,c=BLUE if p<0 else PURPLE,s=20,zorder=3)
 ax.axvline(0,color='#333',lw=.8);ax.set_yticks(range(10),[NAMES[t] for t in TASKS]);ax.invert_yaxis();ax.set_xlabel(f'{met}相对变化（%）');panel(ax,f'({"a" if met=="RMSE" else "b"}) {met}');
save(fig,'4-11','十任务误差变化及置信区间')

# 4-12、4-13：全部配对OOF点，按任务独立原始量纲。
for num,kind in [('4-12','fit'),('4-13','resid')]:
 fig,axs=plt.subplots(5,2,figsize=(15.5/2.54,20.5/2.54));fig.subplots_adjust(left=.12,right=.97,top=.97,bottom=.07,wspace=.27,hspace=.85)
 for ax,t in zip(axs.flat,TASKS):
  d=PAIRS[t];x=d.y_true_x.to_numpy();v1=d.y_pred_x.to_numpy();v2=d.y_pred_t.to_numpy()
  ax.scatter(x,v1 if kind=='fit' else v1-x,s=2,color=BLUE,alpha=.13,rasterized=True,label='XGBoost')
  ax.scatter(x,v2 if kind=='fit' else v2-x,s=2,color=ORANGE,alpha=.13,rasterized=True,label='TabICLv2')
  if kind=='fit':
   lo=min(x.min(),v1.min(),v2.min());hi=max(x.max(),v1.max(),v2.max());ax.plot([lo,hi],[lo,hi],color=GRAY,ls='--',lw=.8);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_ylabel(f'预测值（{UNITS[t]}）')
  else:ax.axhline(0,color=GRAY,ls='--',lw=.8);ax.set_ylabel(f'预测−实测（{UNITS[t]}）')
  ax.set_xlabel(f'实测值（{UNITS[t]}）');ax.set_title(f'{NAMES[t]}（n={len(d):,}）',loc='left');tidy(ax)
 axs[0,0].legend(frameon=False,markerscale=3,fontsize=7)
 save(fig,num,'十任务实测预测对比' if kind=='fit' else '十任务残差分布')

# 5-7：用正式任务五折test的填补前布尔缺失掩码还原每个obs_id。
MASK=ROOT/'交付/修复77特征解释计算迁移_20261116_v3/解压核验/TabICLv2_SHAP_ALE_PORTABLE/02_data/missing_masks'
meta_obs=pd.read_csv(feat_dir/'观测元数据.csv',usecols=['obs_id','date'])
chunks=[]
for t in TASKS:
 for i in range(1,6):
  p=MASK/f'{t}_fold_{i}_test.csv.gz';assert p.exists(),p
  chunks.append(pd.read_csv(p))
mm=pd.concat(chunks,ignore_index=True);assert set(features.feature).issubset(mm.columns)
check=mm.groupby('obs_id')[features.feature.tolist()].nunique(dropna=False).to_numpy()
assert check.max()==1,'同一obs_id的跨任务缺失掩码不一致'
mm=mm.drop_duplicates('obs_id').merge(meta_obs,on='obs_id',validate='one_to_one')
mm['year']=pd.to_datetime(mm.date).dt.year
coverage=np.full((77,11),np.nan)
for j,year in enumerate(range(2014,2025)):
 sub=mm[mm.year==year]
 if len(sub):coverage[:,j]=1-sub[features.feature].mean().to_numpy()
for part,start,stop in [('a',0,39),('b',39,77)]:
 fig,ax=plt.subplots(figsize=(15.5/2.54,17.5/2.54));fig.subplots_adjust(left=.38,right=.89,top=.97,bottom=.09)
 im=ax.imshow(coverage[start:stop],cmap='viridis',vmin=0,vmax=1,aspect='auto')
 ax.set_yticks(range(stop-start),features.feature.iloc[start:stop],fontsize=7.2)
 ax.set_xticks(range(11),range(2014,2025),rotation=45,ha='right',fontsize=8)
 sg=features.source_group.iloc[start:stop].tolist();boundary=[i for i in range(1,len(sg)) if sg[i]!=sg[i-1]]
 for i in boundary:ax.axhline(i-.5,color='white',lw=1.2)
 fig.colorbar(im,ax=ax,fraction=.032,pad=.025,label='填补前真实非缺失率')
 save(fig,f'5-7({part})',f'77特征逐年覆盖_{part}')

# 图5-6需要预先冻结的来源消融配对站点Bootstrap区间；本脚本绝不启动新抽样。
print('SKIPPED 图5-6：未找到冻结的来源消融站点Bootstrap区间。')
