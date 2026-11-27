"""从冻结资产生成 V6.4 定点修订7图；仅作确定性汇总和绘图。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
EXP=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
META=pd.read_csv(EXP/'02_冻结候选/观测元数据.csv',parse_dates=['date'])
CORE=pd.read_csv(ROOT/'论文修订交付_20261121/05_v6_candidate/data/正式十任务指标.csv')
TASKS=CORE.task.tolist();NAMES=dict(zip(CORE.task,CORE['任务']))
UNITS={t:('μg/L' if t.startswith('chla') else 'm' if t=='secchi' else 'mg/L') for t in TASKS}
BLUE='#4477AA';ORANGE='#EE9944';GREEN='#228833';PURPLE='#AA3377';GRAY='#777777'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})
SAVED={}
def save(fig,num,stem,source):
    path=FIG/f'图{num}_{stem}.png'
    for suffix in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+suffix),dpi=600 if suffix=='png' else None,bbox_inches='tight',pad_inches=.10)
    plt.close(fig);SAVED[num]=(path,source);print('DONE',num,path.stat().st_size,flush=True)
def tidy(ax,axis='y'):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis=axis,color='#dce3e7',lw=.5);ax.set_axisbelow(True)
def task_axes():
    fig,axs=plt.subplots(5,2,figsize=(15.5/2.54,19.2/2.54));fig.subplots_adjust(left=.10,right=.97,top=.96,bottom=.075,wspace=.29,hspace=.85)
    return fig,axs.flatten()
PRED=EXP/'03_正式十任务重算/predictions'
def paired(task):
    out=[]
    for model in ['xgboost','tabiclv2']:
        dd=pd.concat([pd.read_csv(PRED/f'{task}_{model}_fold_{i}.csv',usecols=['obs_id','station_id','date','y_true','y_pred','dataset_version']) for i in range(1,6)],ignore_index=True)
        assert dd.obs_id.is_unique and set(dd.dataset_version)=={'WQ77_REPAIRED_20261115_V1'}
        out.append(dd.drop(columns='dataset_version').rename(columns={'y_pred':model}))
    p=out[0].merge(out[1],on='obs_id',suffixes=('','_b'),validate='one_to_one')
    assert len(p)==len(out[0])==len(out[1]) and (p.station_id==p.station_id_b).all() and np.allclose(p.y_true,p.y_true_b)
    p=p.drop(columns=['station_id_b','date_b','y_true_b']);p['date']=pd.to_datetime(p.date)
    return p
PAIRS={t:paired(t) for t in TASKS}
ZONES=list(pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv').water_zone_en.drop_duplicates())


fig,axs=task_axes()
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    d=PAIRS[t];g=d.groupby(d.date.dt.month).y_true
    med=g.median().reindex(range(1,13));lo=g.quantile(.25).reindex(range(1,13));hi=g.quantile(.75).reindex(range(1,13))
    x=np.arange(1,13);ax.fill_between(x,lo.to_numpy(float),hi.to_numpy(float),color=BLUE,alpha=.18);ax.plot(x,med,color=BLUE,marker='o',ms=2,lw=1)
    ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_xticks(x);ax.set_ylabel(f'实测值（{UNITS[t]}）');ax.set_xlabel('采样月份' if k>=8 else '')
    tidy(ax)
save(fig,'2-10','四指标月尺度观测变化','正式十任务冻结OOF样本真实目标值及采样月份')

fig,axs=task_axes();fig.subplots_adjust(hspace=1.05)
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    assert UNITS[t]==('μg/L' if t.startswith('chla') else 'm' if t=='secchi' else 'mg/L')
    d=PAIRS[t];lim=np.nanpercentile(np.r_[d.y_true,d.xgboost,d.tabiclv2],[.5,99.5]);lo=min(0,lim[0]);hi=lim[1]
    for model,col in [('xgboost',BLUE),('tabiclv2',ORANGE)]:
        q=d.copy();q['bin']=pd.qcut(q[model],10,duplicates='drop');g=q.groupby('bin',observed=True).agg(pred=(model,'mean'),obs=('y_true','mean'))
        ax.plot(g.pred,g.obs,marker='o',ms=2.5,lw=1.1,color=col,label='XGBoost' if model=='xgboost' else 'TabICLv2')
    ax.plot([lo,hi],[lo,hi],ls='--',lw=.7,color=GRAY);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_xlabel(f'分箱平均预测值（{UNITS[t]}）');ax.set_ylabel(f'分箱平均实测值（{UNITS[t]}）');tidy(ax)
axs[0].legend(frameon=False,fontsize=7)
save(fig,'4-14','分箱预测观测一致性','两模型相同obs_id冻结OOF预测；每模型预测值独立等频10箱')

fig,axs=task_axes()
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    d=PAIRS[t];year=d.date.dt.year
    for model,col,name in [('xgboost',BLUE,'XGBoost'),('tabiclv2',ORANGE,'TabICLv2')]:
        v=(d[model]-d.y_true).abs().groupby(year).mean().reindex(range(2014,2025));ax.plot(v.index,v.values,marker='o',ms=2.3,lw=1,color=col,label=name)
    ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_xticks(range(2014,2025,2));ax.set_ylabel(f'年度MAE（{UNITS[t]}）');ax.set_xlabel('采样年份' if k>=8 else '')
    tidy(ax)
axs[0].legend(frameon=False,fontsize=7)
save(fig,'4-15','十任务逐年预测误差','两模型相同obs_id冻结留站OOF预测按真实采样年份分组；非年份留出实验')

fig,axs=task_axes()
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    d=PAIRS[t].copy();d['bin']=pd.qcut(d.y_true.rank(method='first'),5,labels=False);g=d.groupby('bin');x=np.arange(5)
    for model,col,off,name in [('xgboost',BLUE,-.16,'XGBoost'),('tabiclv2',ORANGE,.16,'TabICLv2')]:
        v=(d[model]-d.y_true).abs().groupby(d['bin']).mean();ax.bar(x+off,v,width=.31,color=col,label=name)
    ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_xticks(x,[f'Q{i+1}' for i in x]);ax.set_xlabel('');ax.set_ylabel(f'MAE（{UNITS[t]}）');tidy(ax)
axs[0].legend(frameon=False,fontsize=7);fig.supxlabel('实测浓度五等分（Q1—Q5）',y=.025,fontsize=8.5)
save(fig,'4-16','浓度分位区间绝对误差','两模型相同obs_id冻结OOF预测；按实测值排名五等分')

fig,axs=task_axes();station_rows=[]
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    d=PAIRS[t];g=d.groupby('station_id');rows=[]
    for sid,q in g:
        x=np.sqrt(np.mean((q.xgboost-q.y_true)**2));y=np.sqrt(np.mean((q.tabiclv2-q.y_true)**2))
        if x>0:rows.append((sid,len(q),100*(y-x)/x,np.log2(y/x) if y>0 else np.nan))
    s=pd.DataFrame(rows,columns=['station','n','relative','log2ratio']);s['task']=t;station_rows.append(s)
    ax.scatter(s.n,s.relative,s=10,color=BLUE,alpha=.65);ax.axhline(0,color=GRAY,lw=.8);ax.set_title(f'({chr(97+k)}) {NAMES[t]}（{len(s)}站）',loc='left');ax.set_xlabel('站点测试样本数' if k>=8 else '');ax.set_ylabel('相对RMSE变化（%）');tidy(ax)
save(fig,'4-17','站点样本数与误差变化','两模型相同obs_id冻结OOF预测，站点RMSE由现有预测确定性汇总')

decision=pd.read_csv(EXP/'06_历史78特征分区重算/分区训练决策.csv')
assert decision.decision.value_counts().to_dict()=={'zone_model':496,'global_fallback':36}
decision['task']=decision.target+'_'+decision.depth;tasks78=list(decision.task.drop_duplicates());zones78=list(decision.water_zone_en.drop_duplicates())
TASK_LABELS={'chla_surface':'Chl-a表层','chla_middle':'Chl-a中层','chla_bottom':'Chl-a底层','secchi_surface':'Secchi历史表层组合','secchi_middle':'Secchi历史中层组合','secchi_bottom':'Secchi历史底层组合','tn_surface':'TN表层','tn_middle':'TN中层','tn_bottom':'TN底层','tp_surface':'TP表层','tp_middle':'TP中层','tp_bottom':'TP底层'}
assert set(tasks78)==set(TASK_LABELS)
tab=decision.assign(fallback=decision.decision.eq('global_fallback').astype(int)).pivot_table(index='task',columns='water_zone_en',values='fallback',aggfunc='sum').reindex(index=tasks78,columns=zones78)
total=decision.pivot_table(index='task',columns='water_zone_en',values='decision',aggfunc='size').reindex(index=tasks78,columns=zones78)
fig,ax=plt.subplots(figsize=(15.5/2.54,11/2.54));fig.subplots_adjust(left=.16,right=.97,top=.94,bottom=.20)
im=ax.imshow(tab.fillna(0),cmap='Oranges',aspect='auto',vmin=0,vmax=max(1,np.nanmax(tab.values)));ax.set_xticks(range(len(zones78)),zones78,rotation=48,ha='right');ax.set_yticks(range(len(tasks78)),[TASK_LABELS[t] for t in tasks78]);ax.set_xlabel('水区（格内：全域回退次数／决策总数）')
for i in range(len(tasks78)):
    for j in range(len(zones78)):
        if pd.notna(total.iloc[i,j]):ax.text(j,i,f'{int(tab.iloc[i,j])}/{int(total.iloc[i,j])}',ha='center',va='center',fontsize=6.8)
fig.colorbar(im,ax=ax,shrink=.7,label='全域回退次数')
save(fig,'5-10','历史分区局地模型与全域回退','历史78特征扩展实验分区训练决策532条，496区域模型、36全域回退')

features=pd.read_csv(EXP/'02_冻结候选/77项正式输入清单.csv')
def source_group(r):
    g=r['group'];f=r['feature']
    if g=='ERA5':return 'MET' if f.startswith('ssrd_') else 'OCN'
    return {'气象':'MET','降雨':'RHT','潮汐':'RHT','夜间灯光':'HUM','时间':'TMP','空间与站点':'SPA'}[g]
features['source_group']=features.apply(source_group,axis=1);assert len(features)==77
maskdir=ROOT/'交付/修复77特征解释计算迁移_20261116_v3/解压核验/TabICLv2_SHAP_ALE_PORTABLE/02_data/missing_masks'
maskfiles=list(maskdir.glob('*_fold_*_test.csv.gz'));assert len(maskfiles)==50
masks=pd.concat([pd.read_csv(p) for p in maskfiles],ignore_index=True).drop_duplicates('obs_id')
assert set(features.feature).issubset(masks.columns)
masks=masks.merge(META[['obs_id','water_zone_en']],on='obs_id',validate='one_to_one')
cover=np.zeros((6,len(ZONES)));cover_min=np.zeros_like(cover);groups=['MET','RHT','OCN','HUM','TMP','SPA']
for i,g in enumerate(groups):
    cols=features.loc[features.source_group==g,'feature'].tolist()
    for j,z in enumerate(ZONES):
        q=masks.loc[masks.water_zone_en==z,cols]
        vals=1-q.astype(float).mean(axis=0) if len(q) else pd.Series(dtype=float)
        cover[i,j]=vals.median() if len(vals) else np.nan
        cover_min[i,j]=vals.min() if len(vals) else np.nan
fig,ax=plt.subplots(figsize=(15.5/2.54,7.7/2.54));fig.subplots_adjust(left=.10,right=.98,top=.93,bottom=.25)
im=ax.imshow(cover,cmap='Blues',vmin=.95,vmax=1.00,aspect='auto');ax.set_xticks(range(len(ZONES)),ZONES,rotation=48,ha='right');ax.set_yticks(range(6),groups);ax.set_xlabel('格内：组内中位数／最低特征覆盖率')
for i in range(6):
    for j in range(len(ZONES)):ax.text(j,i,f'{cover[i,j]:.0%}\n/{cover_min[i,j]:.0%}' if np.isfinite(cover[i,j]) else 'NA',ha='center',va='center',fontsize=6.6,color='white' if cover[i,j]>.73 else '#263843')
cb=fig.colorbar(im,ax=ax,shrink=.72,label='真实非缺失率中位数');cb.set_ticks(np.arange(.95,1.001,.01));cb.set_ticklabels([f'{v:.0%}' for v in np.arange(.95,1.001,.01)])
save(fig,'5-11','六来源各水区真实观测覆盖','修复版77特征正式清单、五折测试样本填补前缺失掩码与真实水区')

assert set(SAVED)=={'2-10','4-14','4-15','4-16','4-17','5-10','5-11'}
