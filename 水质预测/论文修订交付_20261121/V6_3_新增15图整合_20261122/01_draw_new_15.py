"""从冻结资产生成 V6.3 新增15图；仅作确定性汇总和绘图。"""
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
    fig,axs=plt.subplots(5,2,figsize=(15.5/2.54,18.5/2.54));fig.subplots_adjust(left=.10,right=.97,top=.96,bottom=.065,wspace=.27,hspace=.50)
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

# 第二章：正式任务测试样本即冻结有效观测集合；无填补目标值。
fig,ax=plt.subplots(figsize=(15.5/2.54,9.5/2.54));fig.subplots_adjust(left=.17,right=.98,top=.94,bottom=.20)
counts=np.zeros((10,10),int);sites=np.zeros((10,10),int)
for i,t in enumerate(TASKS):
    d=PAIRS[t][['obs_id','station_id']].merge(META[['obs_id','water_zone_en']],on='obs_id',validate='one_to_one')
    for j,z in enumerate(ZONES):
        q=d[d.water_zone_en==z];counts[i,j]=len(q);sites[i,j]=q.station_id.nunique()
im=ax.imshow(counts,cmap='Blues',aspect='auto');ax.set_xticks(range(10),ZONES,rotation=48,ha='right');ax.set_yticks(range(10),[NAMES[t] for t in TASKS]);ax.set_xlabel('水区（格内：样本数／站点数）')
for i in range(10):
    for j in range(10):ax.text(j,i,f'{counts[i,j]}\n/{sites[i,j]}',ha='center',va='center',fontsize=6.1,color='white' if counts[i,j]>counts.max()*.53 else '#243746')
fig.colorbar(im,ax=ax,shrink=.75,label='有效观测数')
save(fig,'2-7','十任务水区样本站点覆盖','正式十任务冻结OOF样本；观测元数据与站点水区')

fig,axs=task_axes()
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    n=PAIRS[t].groupby(PAIRS[t].date.dt.year).size().reindex(range(2014,2025),fill_value=0)
    ax.bar(n.index,n.values,color=BLUE,width=.8);ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_xticks(range(2014,2025,2));ax.set_ylabel('有效样本数');tidy(ax)
save(fig,'2-8','十任务年度有效观测','正式十任务冻结OOF样本与实际采样日期')

fig,ax=plt.subplots(figsize=(15.5/2.54,9.8/2.54));fig.subplots_adjust(left=.11,right=.97,top=.95,bottom=.22)
series=[PAIRS[t].groupby('station_id').size().values for t in TASKS]
ax.boxplot(series,positions=np.arange(10),widths=.5,showfliers=False,patch_artist=True,boxprops={'facecolor':'#c8d9e9','edgecolor':BLUE},medianprops={'color':ORANGE,'linewidth':1.4},whiskerprops={'color':BLUE},capprops={'color':BLUE})
rng=np.random.default_rng(42)
for i,v in enumerate(series):ax.scatter(rng.normal(i,.08,len(v)),v,s=7,color=BLUE,alpha=.45,zorder=2)
ax.set_xticks(range(10),[NAMES[t] for t in TASKS],rotation=40,ha='right');ax.set_ylabel('站点有效观测数');tidy(ax)
save(fig,'2-9','逐站点有效观测频次','正式十任务冻结OOF样本与站点ID')

fig,axs=task_axes()
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    d=PAIRS[t];g=d.groupby(d.date.dt.month).y_true
    med=g.median().reindex(range(1,13));lo=g.quantile(.25).reindex(range(1,13));hi=g.quantile(.75).reindex(range(1,13));n=g.size().reindex(range(1,13),fill_value=0)
    x=np.arange(1,13);ax.fill_between(x,lo.to_numpy(float),hi.to_numpy(float),color=BLUE,alpha=.18);ax.plot(x,med,color=BLUE,marker='o',ms=2,lw=1)
    ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_xticks(x);ax.set_ylabel(f'实测值（{UNITS[t]}）');ax.set_xlabel('采样月份；下行为样本数')
    for m,v in zip(x,n):ax.text(m,-.24,str(v),ha='center',va='top',fontsize=5.8,transform=ax.get_xaxis_transform())
    tidy(ax)
save(fig,'2-10','四指标月尺度观测变化','正式十任务冻结OOF样本真实目标值及采样月份')

# 第三章：仅画验证/评价流程，不加入未经确认的随机或时间留出比例。
fig,axs=plt.subplots(1,3,figsize=(15.5/2.54,7/2.54));fig.subplots_adjust(left=.035,right=.985,top=.88,bottom=.12,wspace=.15)
spec=[('A 随机划分','同一监测总体样本','随机选取测试样本','评价同总体内随机样本',BLUE),('B 年份留出','按采样年份分组','未参与训练的时间段','评价时间外推',ORANGE),('C 站点留出','按监测站点分组','测试站点不进入训练','五折合并为OOF；空间泛化',GREEN)]
for ax,(title,a,b,c,col) in zip(axs,spec):
    ax.axis('off');ax.add_patch(Rectangle((.02,.04),.96,.92,transform=ax.transAxes,facecolor='white',edgecolor=col,lw=1.2))
    ax.text(.5,.84,title,ha='center',fontweight='bold',color=col,transform=ax.transAxes)
    for y,s in [(.65,a),(.43,b),(.20,c)]:ax.text(.5,y,s,ha='center',va='center',fontsize=7.8,transform=ax.transAxes)
    ax.annotate('',(.5,.30),(.5,.55),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':GRAY})
save(fig,'3-4','三种验证方案对比','论文已登记的随机、年份及五折站点留出协议示意')

fig,ax=plt.subplots(figsize=(15.5/2.54,7.8/2.54));ax.axis('off');fig.subplots_adjust(left=.03,right=.97,top=.9,bottom=.06)
boxes=[(.05,.52,.23,.23,'冻结测试样本\n逐折OOF预测',BLUE),(.37,.52,.23,.23,'合并五折\n相同obs_id配对',BLUE),(.69,.52,.24,.23,'总体评价\npooled R²／RMSE／MAE',BLUE),(.06,.10,.25,.22,'折次评价\n逐折R²与误差差异',ORANGE),(.39,.10,.25,.22,'站点评价\nRMSE／MAE与胜率',GREEN),(.71,.10,.24,.22,'配对站点Bootstrap\n误差差值区间',PURPLE)]
for x,y,w,h,s,c in boxes:
    ax.add_patch(Rectangle((x,y),w,h,transform=ax.transAxes,facecolor=c,alpha=.12,edgecolor=c,lw=1.2));ax.text(x+w/2,y+h/2,s,ha='center',va='center',transform=ax.transAxes,fontsize=8)
for a,b in [((.28,.635),(.37,.635)),((.60,.635),(.69,.635)),((.48,.52),(.185,.32)),((.48,.52),(.515,.32)),((.515,.10),(.83,.10))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=10,color=GRAY,transform=ax.transAxes))
save(fig,'3-5','逐样本多层级评价流程','冻结OOF预测及已登记的总体、折次、站点与Bootstrap评价口径')

# 第四章：全部两模型图均按相同obs_id的冻结测试预测成对绘制。
fig,axs=task_axes()
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    d=PAIRS[t];lim=np.nanpercentile(np.r_[d.y_true,d.xgboost,d.tabiclv2],[.5,99.5]);lo=min(0,lim[0]);hi=lim[1]
    for model,col in [('xgboost',BLUE),('tabiclv2',ORANGE)]:
        q=d.copy();q['bin']=pd.qcut(q[model],10,duplicates='drop');g=q.groupby('bin',observed=True).agg(pred=(model,'mean'),obs=('y_true','mean'))
        ax.plot(g.pred,g.obs,marker='o',ms=2.5,lw=1.1,color=col,label='XGBoost' if model=='xgboost' else 'TabICLv2')
    ax.plot([lo,hi],[lo,hi],ls='--',lw=.7,color=GRAY);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_xlabel(f'分箱平均预测值（{UNITS[t]}）');ax.set_ylabel(f'分箱平均实测值（{UNITS[t]}）');tidy(ax)
axs[0].legend(frameon=False,fontsize=7)
save(fig,'4-14','分箱预测观测一致性','两模型相同obs_id冻结OOF预测；每模型预测值独立等频10箱')

fig,axs=task_axes()
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    d=PAIRS[t];year=d.date.dt.year;n=d.groupby(year).size().reindex(range(2014,2025),fill_value=0)
    for model,col,name in [('xgboost',BLUE,'XGBoost'),('tabiclv2',ORANGE,'TabICLv2')]:
        v=(d[model]-d.y_true).abs().groupby(year).mean().reindex(range(2014,2025));ax.plot(v.index,v.values,marker='o',ms=2.3,lw=1,color=col,label=name)
    ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_xticks(range(2014,2025,2));ax.set_ylabel(f'年度MAE（{UNITS[t]}）');ax.set_xlabel('采样年份；下行为样本数')
    for y,v in n.items():ax.text(y,-.24,str(v),ha='center',va='top',fontsize=5.6,transform=ax.get_xaxis_transform())
    tidy(ax)
axs[0].legend(frameon=False,fontsize=7)
save(fig,'4-15','十任务逐年预测误差','两模型相同obs_id冻结留站OOF预测按真实采样年份分组；非年份留出实验')

fig,axs=task_axes()
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    d=PAIRS[t].copy();d['bin']=pd.qcut(d.y_true.rank(method='first'),5,labels=False);g=d.groupby('bin');x=np.arange(5);n=g.size()
    for model,col,off,name in [('xgboost',BLUE,-.16,'XGBoost'),('tabiclv2',ORANGE,.16,'TabICLv2')]:
        v=(d[model]-d.y_true).abs().groupby(d['bin']).mean();ax.bar(x+off,v,width=.31,color=col,label=name)
    ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_xticks(x,[f'Q{i+1}\n{n.loc[i]}' for i in x]);ax.set_xlabel('实测浓度五等分；下行为样本数');ax.set_ylabel(f'MAE（{UNITS[t]}）');tidy(ax)
axs[0].legend(frameon=False,fontsize=7)
save(fig,'4-16','浓度分位区间绝对误差','两模型相同obs_id冻结OOF预测；按实测值排名五等分')

fig,axs=task_axes();station_rows=[]
for k,(t,ax) in enumerate(zip(TASKS,axs)):
    d=PAIRS[t];g=d.groupby('station_id');rows=[]
    for sid,q in g:
        x=np.sqrt(np.mean((q.xgboost-q.y_true)**2));y=np.sqrt(np.mean((q.tabiclv2-q.y_true)**2))
        if x>0:rows.append((sid,len(q),100*(y-x)/x,np.log2(y/x) if y>0 else np.nan))
    s=pd.DataFrame(rows,columns=['station','n','relative','log2ratio']);s['task']=t;station_rows.append(s)
    ax.scatter(s.n,s.relative,s=10,color=BLUE,alpha=.65);ax.axhline(0,color=GRAY,lw=.8);ax.set_title(f'({chr(97+k)}) {NAMES[t]}（{len(s)}站）',loc='left');ax.set_xlabel('站点测试样本数');ax.set_ylabel('相对RMSE变化（%）');tidy(ax)
save(fig,'4-17','站点样本数与误差变化','两模型相同obs_id冻结OOF预测，站点RMSE由现有预测确定性汇总')

st=pd.concat(station_rows,ignore_index=True).pivot(index='station',columns='task',values='log2ratio');rho=np.full((10,10),np.nan);pairn=np.zeros((10,10),int)
for i,a in enumerate(TASKS):
    for j,b in enumerate(TASKS):
        q=st[[a,b]].dropna();pairn[i,j]=len(q);rho[i,j]=1 if i==j else spearmanr(q[a],q[b]).statistic if len(q)>=3 else np.nan
fig,ax=plt.subplots(figsize=(15.5/2.54,11.8/2.54));fig.subplots_adjust(left=.17,right=.90,top=.95,bottom=.19)
im=ax.imshow(rho,cmap='RdBu_r',vmin=-1,vmax=1);ax.set_xticks(range(10),[NAMES[t] for t in TASKS],rotation=45,ha='right');ax.set_yticks(range(10),[NAMES[t] for t in TASKS]);ax.set_xlabel('格内上行为Spearman ρ，下行为共同站点数')
for i in range(10):
    for j in range(10):
        label=(f'{rho[i,j]:.2f}' if np.isfinite(rho[i,j]) else 'NA')+f'\n{pairn[i,j]}'
        ax.text(j,i,label,ha='center',va='center',fontsize=6.5,color='white' if abs(rho[i,j])>.6 else '#25333d')
fig.colorbar(im,ax=ax,shrink=.75,label='Spearman ρ')
save(fig,'4-18','站点相对误差变化跨任务相关性','两模型同obs_id冻结OOF预测；仅实际共同站点参与Spearman计算')

# 第五章：正式77项来源消融与独立的历史78项分区实验分开绘制。
AB=EXP/'04_来源消融重算/predictions'
SOURCE={'MET':'METEOROLOGY','RHT':'RAINFALL_HYDRO_TIDE','OCN':'OCEAN_PHYSICS','HUM':'HUMAN_ACTIVITY','TMP':'TEMPORAL_PROCESS','SPA':'SPATIAL_STATIC'}
REP=['chla_surface','secchi','tn_surface','tp_middle'];FOLD=[]
for t in REP:
    for short,long in SOURCE.items():
        for i in range(1,6):
            a=pd.read_csv(AB/f'{t}_{long}_fold_{i}.csv',usecols=['obs_id','y_true','y_pred','dataset_version'])
            b=pd.read_csv(PRED/f'{t}_xgboost_fold_{i}.csv',usecols=['obs_id','y_true','y_pred'])
            assert set(a.dataset_version)=={'WQ77_REPAIRED_20261115_V1'}
            q=a.merge(b,on='obs_id',suffixes=('_ab','_full'),validate='one_to_one');assert len(q)==len(a)==len(b) and np.allclose(q.y_true_ab,q.y_true_full)
            y=q.y_true_ab.to_numpy();sse1=((y-q.y_pred_ab)**2).sum();sse0=((y-q.y_pred_full)**2).sum();sst=((y-y.mean())**2).sum()
            FOLD.append((t,short,i,(sse0-sse1)/sst))
fold=pd.DataFrame(FOLD,columns=['task','source','fold','delta_R2'])
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,13/2.54));fig.subplots_adjust(left=.10,right=.97,top=.94,bottom=.10,wspace=.25,hspace=.38)
for k,(t,ax) in enumerate(zip(REP,axs.flatten())):
    q=fold[fold.task==t]
    for i,s in enumerate(SOURCE):
        v=q[q.source==s].sort_values('fold');ax.scatter(np.full(5,i)+np.linspace(-.10,.10,5),v.delta_R2,s=17,color=BLUE,alpha=.8)
    ax.axhline(0,color=GRAY,lw=.8);ax.set_xticks(range(6),list(SOURCE));ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_ylabel('删组−完整：逐折ΔR²');tidy(ax)
save(fig,'5-8','六来源消融五折效应','WQ77修复版四任务正式XGBoost来源消融120折与完整模型冻结OOF预测')

summary=pd.read_csv(EXP/'04_来源消融重算/四任务六来源_新版本完整对照.csv');rev={v:k for k,v in SOURCE.items()};summary['source']=summary.source_group_removed.map(rev)
assert len(summary)==24 and summary.source.notna().all()
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,12/2.54));fig.subplots_adjust(left=.12,right=.96,top=.94,bottom=.10,wspace=.30,hspace=.36)
cols=dict(zip(SOURCE,[BLUE,ORANGE,GREEN,PURPLE,'#66AABB',GRAY]))
for k,(t,ax) in enumerate(zip(REP,axs.flatten())):
    q=summary[summary.task==t]
    for _,r in q.iterrows():
        y=100*r.delta_RMSE/r.full_RMSE;ax.scatter(r.delta_R2,y,s=31,color=cols[r.source]);ax.annotate(r.source,(r.delta_R2,y),xytext=(3,3),textcoords='offset points',fontsize=6.6)
    ax.axhline(0,color=GRAY,lw=.7);ax.axvline(0,color=GRAY,lw=.7);ax.set_title(f'({chr(97+k)}) {NAMES[t]}',loc='left');ax.set_xlabel('删组−完整：pooled ΔR²');ax.set_ylabel('相对ΔRMSE（%）');tidy(ax)
save(fig,'5-9','六来源双指标消融效应','WQ77修复版四任务XGBoost来源消融pooled汇总')

decision=pd.read_csv(EXP/'06_历史78特征分区重算/分区训练决策.csv')
assert decision.decision.value_counts().to_dict()=={'zone_model':496,'global_fallback':36}
decision['task']=decision.target+'_'+decision.depth;tasks78=list(decision.task.drop_duplicates());zones78=list(decision.water_zone_en.drop_duplicates())
tab=decision.assign(fallback=decision.decision.eq('global_fallback').astype(int)).pivot_table(index='task',columns='water_zone_en',values='fallback',aggfunc='sum').reindex(index=tasks78,columns=zones78)
total=decision.pivot_table(index='task',columns='water_zone_en',values='decision',aggfunc='size').reindex(index=tasks78,columns=zones78)
fig,ax=plt.subplots(figsize=(15.5/2.54,11/2.54));fig.subplots_adjust(left=.16,right=.97,top=.94,bottom=.20)
im=ax.imshow(tab.fillna(0),cmap='Oranges',aspect='auto',vmin=0,vmax=max(1,np.nanmax(tab.values)));ax.set_xticks(range(len(zones78)),zones78,rotation=48,ha='right');ax.set_yticks(range(len(tasks78)),tasks78);ax.set_xlabel('水区（格内：全域回退次数／决策总数）')
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
cover=np.zeros((6,len(ZONES)));cover_min=np.zeros_like(cover);groups=list(SOURCE)
for i,g in enumerate(groups):
    cols=features.loc[features.source_group==g,'feature'].tolist()
    for j,z in enumerate(ZONES):
        q=masks.loc[masks.water_zone_en==z,cols]
        vals=1-q.astype(float).mean(axis=0) if len(q) else pd.Series(dtype=float)
        cover[i,j]=vals.median() if len(vals) else np.nan
        cover_min[i,j]=vals.min() if len(vals) else np.nan
fig,ax=plt.subplots(figsize=(15.5/2.54,7.7/2.54));fig.subplots_adjust(left=.10,right=.98,top=.93,bottom=.25)
im=ax.imshow(cover,cmap='Blues',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(ZONES)),ZONES,rotation=48,ha='right');ax.set_yticks(range(6),groups);ax.set_xlabel('格内：组内中位数／最低特征覆盖率')
for i in range(6):
    for j in range(len(ZONES)):ax.text(j,i,f'{cover[i,j]:.0%}\n/{cover_min[i,j]:.0%}' if np.isfinite(cover[i,j]) else 'NA',ha='center',va='center',fontsize=6.6,color='white' if cover[i,j]>.73 else '#263843')
fig.colorbar(im,ax=ax,shrink=.72,label='组内特征真实非缺失率中位数')
save(fig,'5-11','六来源各水区真实观测覆盖','修复版77特征正式清单、五折测试样本填补前缺失掩码与真实水区')

assert len(SAVED)==15,len(SAVED)
pd.DataFrame([{'图号':'图'+k,'图片文件':str(v[0]),'数据来源':v[1]} for k,v in SAVED.items()]).to_csv(HERE/'新增图件数据清单.csv',index=False,encoding='utf-8-sig')
