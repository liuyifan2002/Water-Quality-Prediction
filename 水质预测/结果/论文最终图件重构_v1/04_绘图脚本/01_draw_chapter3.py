"""从冻结解释输出重构第三章；历史空bin不恢复未经识别的累计曲线。"""
from common import *
SCRIPT=Path(__file__).name
stats=read(FROZEN30);assert len(stats)==770
sm=read(HIST+'04_shap_importance_summary.csv');bf=read(HIST+'03_shap_importance_by_fold.csv');stab=read(HIST+'05_shap_rank_stability.csv')
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,18/2.54));fig.subplots_adjust(left=.31,right=.98,top=.94,bottom=.08,wspace=.70,hspace=.35)
for k,(ax,t) in enumerate(zip(axs.flat,REPS)):
    q=sm[sm.task==t].nsmallest(10,'task_rank').sort_values('task_relative_contribution')
    y=np.arange(10);ax.hlines(y,0,q.task_relative_contribution,color=BLUE);ax.scatter(q.task_relative_contribution,y,s=18,c=BLUE);ax.set_yticks(y,[cn(f) for f in q.feature],fontsize=8);ax.set_xlabel('相对平均|SHAP|份额');ax.set_title(f'({chr(97+k)}) {LAB[t]}',loc='left');grid(ax)
save(fig,'3-1','历史五折SHAP特征相对贡献份额',[HIST+'04_shap_importance_summary.csv'],SCRIPT,'图3-1','TabICLv2历史四任务固定批查询五折解释；先对各折平均绝对SHAP等权汇总，再除以该任务全部77特征汇总值之和。展示任务内Top10，相对份额不是原始SHAP绝对值，不能与P1数值直接比较。')
fig=figcm(17);gs=fig.add_gridspec(2,2,left=.26,right=.92,top=.93,bottom=.19,hspace=.39,wspace=1.6)
for k,t in enumerate(REPS):
    ax=fig.add_subplot(gs[k//2,k%2]);feats=sm[sm.task==t].nsmallest(10,'task_rank').feature.tolist();z=bf[(bf.task==t)&bf.feature.isin(feats)].pivot(index='feature',columns='fold',values='fold_rank').reindex(feats)
    im=ax.imshow(z,cmap=RANKC,vmin=1,vmax=77,aspect='auto');ax.set_yticks(range(10),[cn(f) for f in feats],fontsize=8);ax.set_xticks(range(5),['1','2','3','4','5']);ax.set_xlabel('折次');ax.set_title(f'({chr(97+k)}) {LAB[t]}',loc='left',pad=18)
    for i in range(10):
        for j in range(5):ax.text(j,i,int(z.iloc[i,j]),ha='center',va='center',fontsize=7.5,color='white' if z.iloc[i,j]<23 else '#253746')
        ax.text(4.85,i,f'{int((z.iloc[i]<=10).sum())}/5',ha='left',va='center',fontsize=7.5,clip_on=False)
    ax.text(1.04,1.02,'次数/5',transform=ax.transAxes,fontsize=7.5)
    r=stab[stab.task==t].iloc[0];ax.set_title(f'({chr(97+k)}) {LAB[t]}\nρ中位数={r.median_pairwise_rank_spearman:.3f}',loc='left',pad=7,fontsize=9)
cax=fig.add_axes([.36,.06,.43,.014]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',label='折内排名（1最重要）');cb.set_ticks([1,20,40,60,77]);cb.ax.tick_params(labelsize=7.5)
save(fig,'3-2','历史五折SHAP特征排名跨折差异',[HIST+'03_shap_importance_by_fold.csv',HIST+'05_shap_rank_stability.csv'],SCRIPT,'图3-2','颜色和数字为各折77特征内排名；右列为进入Top10的折次计数。完成20个任务折不等于代表性或稳定性检验全部通过。历史固定批查询与P1单查询分开解释。')
for num,tasks in [('3-3',TASKS[:3]),('3-4',['secchi']),('3-5',TASKS[4:7]),('3-6',TASKS[7:])]:
    h=19.5 if len(tasks)==3 else 10.5;fig=figcm(h);axes=fig.subplots(len(tasks),1,squeeze=False).flat;fig.subplots_adjust(left=.38,right=.97,bottom=.085,top=.95,hspace=.65)
    for ax,t in zip(axes,tasks):
        q=stats[stats.task==t].nsmallest(6 if len(tasks)==3 else 10,'rank').sort_values('mean_abs_shap');y=np.arange(len(q));ax.hlines(y,0,q.mean_abs_shap,color=BLUE,lw=1.1);ax.scatter(q.mean_abs_shap,y,s=23,c=BLUE);ax.set_yticks(y,[cn(f) for f in q.feature]);ax.set_title(LAB[t],loc='left');ax.set_xlabel('平均|SHAP|（'+targetunit(t)+'）');grid(ax)
    save(fig,num,('Secchi' if len(tasks)==1 else tasks[0].split('_')[0].replace('chla','Chl-a').upper() if tasks[0].split('_')[0]!='chla' else 'Chl-a')+'核心样本P1 SHAP贡献',[FROZEN30],SCRIPT,'图3-3、图3-4','每任务3个预登记独立核心样本，seed42；排除10次种子重复。按每特征三样本平均绝对SHAP排序；各任务原始量纲与横轴范围独立，不作跨目标绝对贡献比较。')
reuse('3-7','十任务P1核心样本特征重要性排名矩阵','结果/论文图表/figure_revision_v2/final_figures/图3-5_30核心样本特征排名矩阵.png',[FROZEN30,'结果/论文图表/figure_revision_v2/data/77特征展示选择依据.csv'],SCRIPT,'图3-5','复用已确认30核心样本版本。按77特征内实际名次显示；十任务平均排名仅用于固定10个展示特征，任务顺序不变，不代表跨任务贡献大小。')

# 空bin未知局部效应u对第i个中心化ALE的影响系数为I(i>=k)-Σ_{j>=k}w_j/Σw_j。
# w=max(count,1)。有空bin时该未知量影响全曲线基准，不能仅把空点断开后继续当可信累计响应。
a=read('runs/tabicl_v2_explainability_v3/ale_1d/ale_1d_by_fold.csv');checks=[]
for (t,f,fold),g in a.groupby(['task','feature','fold'],sort=False):
    g=g.sort_values('bin');w=np.maximum(g.n_samples.to_numpy(),1);acc=np.cumsum(np.nan_to_num(g.local_effect.to_numpy()));acc-=np.average(acc,weights=w)
    assert np.allclose(acc,g.ale,atol=1e-10), (t,f,fold)
    empty=np.flatnonzero(g.n_samples.to_numpy()==0);coeff=[]
    for k in empty:coeff.append((np.arange(len(g))>=k).astype(float)-w[k:].sum()/w.sum())
    checks.append({'task':t,'feature':f,'fold':fold,'空bin':','.join(str(int(g.bin.iloc[k])) for k in empty),'空bin数':len(empty),'真实支持bin数':int((g.n_samples>0).sum()),'全累计曲线可识别':len(empty)==0,'未知局部效应对各bin的系数':json.dumps([x.tolist() for x in coeff]),'处理':'原值逐点绘制' if len(empty)==0 else '整条累计曲线不纳入响应图，保留原始资产；不按零填充恢复'})
pd.DataFrame(checks).to_csv(OUT/'05_数据核验/历史ALE空bin累计影响审查.csv',index=False,encoding='utf-8-sig')
for num,t in zip(['3-8','3-9','3-10','3-11'],REPS):
    q=a[a.task==t];feats=q.feature.drop_duplicates().tolist();fig,axs=plt.subplots(5,4,figsize=(15.5/2.54,19.5/2.54));fig.subplots_adjust(left=.13,right=.98,top=.90,bottom=.12,wspace=.53,hspace=.78)
    for j,f in enumerate(feats):
        allq=q[q.feature==f];xmin,xmax=allq.center.min(),allq.center.max();ymin,ymax=allq.ale.min(),allq.ale.max();ypad=max((ymax-ymin)*.12,1e-8)
        for i,fold in enumerate([f'fold_{k}' for k in range(1,6)]):
            ax=axs[i,j];g=allq[allq.fold==fold].sort_values('bin');ax.axhline(0,color='#999',lw=.5);ax.set_xlim(xmin,xmax);ax.set_ylim(ymin-ypad,ymax+ypad);ax.locator_params(axis='both',nbins=2);ax.tick_params(labelsize=8,length=2)
            if (g.n_samples==0).any():
                ax.set_facecolor('#f6f6f6');ax.text(.5,.5,'存在空分箱\n累计值不可识别',transform=ax.transAxes,ha='center',va='center',fontsize=7.5,color=GRAY)
            else:ax.plot(g.center,g.ale,'o-',color=FOLDC[i],lw=.85,ms=2.6)
            if i==0:ax.set_title(cn(f)+'\n'+unit(f),fontsize=8,pad=10)
            if j==0:ax.set_ylabel(f'折{i+1}\nALE ({targetunit(t)})',fontsize=8)
    fig.text(.13,.965,LAB[t]+'：历史固定批查询五折一维ALE',fontsize=10)
    fig.text(.03,.025,'每格一折原始分箱；同特征共用坐标，不跨折平均或插值。\n灰格仅保留历史审查位置，不展示空bin占位产生的累计响应。',fontsize=7.5,linespacing=1.5)
    save(fig,num,LAB[t].replace(' ','')+'历史五折ALE分面',['runs/tabicl_v2_explainability_v3/ale_1d/ale_1d_by_fold.csv'],SCRIPT,'图3-'+str(int(num.split('-')[1])-2),'历史固定批查询探索性结果。空bin的未知局部效应会影响后续累计值及中心化基准，不能可靠恢复的9条累计曲线不作为响应展示；其余71条保留原分箱和原值。灰格不是零响应；不重新运行ALE。')

def draw_curve(ax,d,t,f,counts=True):
    x=(d.lower+d.upper)/2;ax.plot(x,d.ale,'o-',c=BLUE,lw=1.1,ms=3.5);ax.axhline(0,c=GRAY,ls='--',lw=.65);ax.set_xlabel(xlabel(f));ax.set_ylabel('ALE（'+targetunit(t)+'）');grid(ax,'y');ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4))
    if counts:
        for xi,yi,nn,ss in zip(x,d.ale,d.n_samples,d.n_stations):ax.annotate(f'{int(nn)}/{int(ss)}',(xi,yi),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7.5)
for num,t,feats in [('3-12','chla_surface',['wind_speed','wind_speed_lag1']),('3-13','secchi',['ssrd_sum_3d','t2m_roll3']),('3-14','tn_surface',['t2m_delta1','wind_speed_lag1']),('3-15','tp_middle',['t2m_lag1','tcc'])]:
    fig,axs=plt.subplots(1,2,figsize=(15.5/2.54,11.5/2.54));fig.subplots_adjust(left=.12,right=.98,bottom=.26,top=.88,wspace=.55);src=[]
    for ax,f in zip(axs,feats):
        rel=P1+'ale_1d/'+t+'_'+f+'_curve.csv';d=read(rel);src.append(rel);draw_curve(ax,d,t,f,False);x=(d.lower+d.upper)/2
        ax.text(-.10,-.27,'n',transform=ax.transAxes,fontsize=7.5);ax.text(-.10,-.35,'s',transform=ax.transAxes,fontsize=7.5)
        for pos,n,ss in zip(np.linspace(.04,.96,len(d)),d.n_samples,d.n_stations):
            ax.text(pos,-.27,str(int(n)),transform=ax.transAxes,ha='center',fontsize=7.5);ax.text(pos,-.35,str(int(ss)),transform=ax.transAxes,ha='center',fontsize=7.5)
        ax.set_title(cn(f),loc='left')
    fig.text(.12,.96,LAB[t]+'：P1局部响应与分箱支持',fontsize=10)
    save(fig,num,LAB[t].replace(' ','')+'P1一维ALE',src,SCRIPT,'图3-'+str(int(num.split('-')[1])-2),'第1折P1单查询，两条冻结曲线；各自横纵坐标独立，支持轨n/s按从左到右实际分箱列示样本数/独立站点数；不插值或平滑。仅描述当前预测函数局部响应，不作环境因果解释。')
spec=[]
for t in TASKS:
    for p in sorted((ROOT/(P1+'ale_1d')).glob(t+'_*_curve.csv')):
        d=read(p);spec.append((t,d.feature.iloc[0],d,str(p.relative_to(ROOT))))
assert len(spec)==20 and sum(len(x[2]) for x in spec)==196
fig,axs=plt.subplots(1,2,figsize=(15.5/2.54,19/2.54));fig.subplots_adjust(left=.34,right=.94,top=.93,bottom=.15,wspace=.25)
for ax,col,title in zip(axs,['n_samples','n_stations'],['(a) 分箱样本数 n','(b) 独立站点数 s']):
    mat=np.array([x[2][col].tolist()+[np.nan]*(10-len(x[2])) for x in spec]);im=ax.imshow(mat,aspect='auto',cmap='Blues');ax.set_yticks(range(20),[LAB[t]+'·'+cn(f) for t,f,_,_ in spec] if col=='n_samples' else [],fontsize=7.5);ax.set_xticks(range(10),range(1,11));ax.set_xlabel('分箱位置索引');ax.set_title(title,loc='left')
    for i in range(20):
        for j in range(10):ax.text(j,i,'NA' if np.isnan(mat[i,j]) else str(int(mat[i,j])),ha='center',va='center',fontsize=7.5,color='white' if np.isfinite(mat[i,j]) and mat[i,j]>(np.nanmin(mat)+.55*(np.nanmax(mat)-np.nanmin(mat))) else '#333')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.10);cb.solids.set_rasterized(False)
save(fig,'3-16','十任务P1 ALE样本与独立站点双重支持',[x[3] for x in spec],SCRIPT,'图3-14、图3-15','20条曲线含196个实际分箱和4个不存在格，NA不是0。样本数和独立站点数使用独立计数色标；横轴仅为bin索引，不能跨变量比较物理值；颜色不表示ALE响应强度。')
