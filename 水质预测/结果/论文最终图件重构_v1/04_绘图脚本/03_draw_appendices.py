"""附录C/D/E/F/H冻结图件统一；直接读取SHAP/ALE输出，不运行解释器。"""
from common import *
SCRIPT=Path(__file__).name
samples=read('结果/论文图表/figure_revision_v2/source_inventory/40组文件与30核心样本对应.csv');stats=read(FROZEN30)
for i,t in enumerate(TASKS,1):
    p=samples[(samples.task==t)&(samples.seed==42)&(samples['预登记分位']==.5)].文件.iloc[0];d=read(p);row=d.iloc[0];assert abs(d.shap_value.sum()+row.base_value-row.model_prediction)<1e-8
    q=d.assign(abs_value=d.shap_value.abs()).nlargest(10,'abs_value');other=d.loc[~d.feature.isin(q.feature),'shap_value'].sum();vals=q.shap_value.tolist()+[other];labels=[cn(f) for f in q.feature]+['其余67特征合计'];y=np.arange(11)
    fig,ax=plt.subplots(figsize=(15.5/2.54,9/2.54));fig.subplots_adjust(left=.38,right=.96,bottom=.20,top=.90);ax.barh(y,vals,color=[ORANGE if v>=0 else BLUE for v in vals],height=.60);ax.axvline(0,c=GRAY,lw=.7);ax.set_yticks(y,labels,fontsize=8.5);ax.invert_yaxis();ax.set_xlabel('SHAP贡献（'+targetunit(t)+'）');ax.set_title(LAB[t]+' · 固定核心样本',loc='left');grid(ax)
    fig.text(.05,.055,f'基准值={row.base_value:.6g}；预测值={row.model_prediction:.6g}；样本：{row.obs_id}',fontsize=8)
    save(fig,f'C-{i}',LAB[t].replace(' ','')+'固定核心样本局部SHAP',[p],SCRIPT,f'图C-{i}',f'{row.obs_id}，seed42，第1折P1。固定预登记中位核心样本；Top10按绝对贡献排序，正负为推高/压低该样本预测，其余67特征合计单列。保留基准值与预测值，不代表总体分布。')
    paths=samples[samples.task==t].文件.tolist();allv=pd.concat([read(p) for p in paths],ignore_index=True);top=stats[stats.task==t].nsmallest(10,'rank').feature.tolist();fig,ax=plt.subplots(figsize=(15.5/2.54,9.5/2.54));fig.subplots_adjust(left=.38,right=.89,bottom=.24,top=.88)
    for yi,f in enumerate(top):
        z=allv[allv.feature==f];fv=z.feature_value.to_numpy();norm=(fv-fv.min())/(fv.max()-fv.min()+1e-12)
        for k,(_,r) in enumerate(z.iterrows()):ax.scatter(r.shap_value,yi+(k-1.5)*.10,c=[norm[k]],cmap='coolwarm',vmin=0,vmax=1,marker='D' if r.shap_seed==43 else 'o',s=28,edgecolors='#333' if r.shap_seed==43 else 'none',lw=.7)
    ax.set_yticks(range(10),[cn(f) for f in top]);ax.invert_yaxis();ax.axvline(0,c=GRAY,lw=.7);ax.set_xlabel('SHAP贡献（'+targetunit(t)+'）');ax.set_title(LAB[t]+' · 三核心样本与一次种子重复',loc='left');grid(ax)
    sm=plt.cm.ScalarMappable(cmap='coolwarm',norm=plt.Normalize(0,1));cb=fig.colorbar(sm,ax=ax,fraction=.04,pad=.04);cb.set_ticks([0,1],labels=['低','高']);cb.set_label('特征值相对高低');cb.solids.set_rasterized(False)
    fig.legend(handles=[Line2D([],[],marker='o',c=GRAY,ls='',label='独立核心样本（3）'),Line2D([],[],marker='D',c=GRAY,ls='',label='同样本seed43重复（1）')],loc='lower center',bbox_to_anchor=(.58,.005),ncol=2,frameon=False,fontsize=7.5)
    save(fig,f'C-{i+10}',LAB[t].replace(' ','')+'稀疏局部SHAP对照',paths+[FROZEN30],SCRIPT,f'图C-{i+10}','3个预登记独立核心样本为圆点，中位样本的一次seed43重复为菱形，不作为第四个独立样本。特征排序仅按三个核心样本平均|SHAP|确定；颜色保留特征值相对高低，散点数量不足以作总体分布推断。')
    q=stats[stats.task==t].nsmallest(10,'rank').sort_values('mean_abs_shap');fig,ax=plt.subplots(figsize=(15.5/2.54,10/2.54));fig.subplots_adjust(left=.39,right=.97,bottom=.17,top=.90);y=np.arange(10);ax.hlines(y,0,q.mean_abs_shap,color=BLUE,lw=1.2);ax.scatter(q.mean_abs_shap,y,s=24,c=BLUE);ax.set_yticks(y,[cn(f) for f in q.feature]);ax.set_xlabel('平均|SHAP|（'+targetunit(t)+'）');ax.set_title(LAB[t]+' · 核心样本Top10',loc='left');grid(ax)
    save(fig,f'F-{i}',LAB[t].replace(' ','')+'核心样本SHAP前10特征',[FROZEN30],SCRIPT,f'图F-{i}','第1折P1，仅三个预登记独立核心样本seed42的平均绝对SHAP；排除种子重复。按全部77特征内原排名选Top10，独立原目标尺度，不跨任务比较原始贡献绝对值。')

specs=[('chla_surface','wind_speed'),('chla_surface','wind_speed_lag1'),('chla_middle','wind_speed'),('chla_middle','wind_speed_sq'),('chla_bottom','t2m'),('chla_bottom','t2m_lag1'),('secchi','ssrd_sum_3d'),('secchi','t2m_roll3'),('tn_surface','t2m_delta1'),('tn_surface','wind_speed_lag1'),('tn_middle','t2m_delta1'),('tn_middle','v10'),('tn_bottom','t2m_lag2'),('tn_bottom','t2m_roll3'),('tp_surface','t2m_lag2'),('tp_surface','u10'),('tp_middle','t2m_lag1'),('tp_middle','tcc'),('tp_bottom','t2m_lag1'),('tp_bottom','wind_speed_lag1')]
for i,(t,f) in enumerate(specs,1):
    rel=P1+'ale_1d/'+t+'_'+f+'_curve.csv';d=read(rel);assert len(d)==(6 if i==7 else 10);x=(d.lower+d.upper)/2
    fig,ax=plt.subplots(figsize=(15.5/2.54,10/2.54));fig.subplots_adjust(left=.14,right=.96,bottom=.24,top=.87);ax.plot(x,d.ale,'o-',c=BLUE,lw=1.15,ms=3.6);ax.axhline(0,c=GRAY,lw=.65,ls='--');ax.set_xlabel(xlabel(f));ax.set_ylabel('ALE（'+targetunit(t)+'）');ax.set_title(LAB[t]+' · '+cn(f),loc='left');grid(ax,'y');ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4));ax.margins(y=.22)
    for k,(xi,yi,n,s) in enumerate(zip(x,d.ale,d.n_samples,d.n_stations)):
        ax.annotate(f'{int(n)}/{int(s)}',(xi,yi),xytext=(0,8 if k%2==0 else -13),textcoords='offset points',ha='center',fontsize=7.5)
    fig.text(.14,.065,'点旁数字：n / s = 分箱样本数 / 独立站点数',fontsize=8)
    save(fig,f'E-{i}',LAB[t].replace(' ','')+cn(f)+'P1 ALE',[rel],SCRIPT,f'图E-{i}',f'第1折P1单查询，{len(d)}个真实冻结分箱；点旁保留每bin样本数n及独立站点数s。横轴为原bin中心、纵轴原目标单位，不插值、不平滑、不删除末端突变；该局部模型响应不作环境因果解释。')

s=read(XGB+'01_shap/01_shap_importance_summary.csv')
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,18/2.54));fig.subplots_adjust(left=.31,right=.97,top=.94,bottom=.08,wspace=.75,hspace=.37)
for ax,t in zip(axs.flat,REPS):
    q=s[s.task==t].nsmallest(10,'task_rank').sort_values('mean_normalized_contribution');y=np.arange(10);ax.hlines(y,0,q.mean_normalized_contribution,color=BLUE);ax.scatter(q.mean_normalized_contribution,y,c=BLUE,s=20);ax.set_yticks(y,[cn(f) for f in q.feature],fontsize=8);ax.set_title(LAB[t],loc='left');ax.set_xlabel('归一化TreeSHAP份额',fontsize=8);grid(ax)
save(fig,'D-1','四任务XGBoost TreeSHAP特征份额',[XGB+'01_shap/01_shap_importance_summary.csv'],SCRIPT,'图D-1','数值等价受控复现XGBoost，五折归一化特征份额均值；TreeSHAP基于log1p目标尺度。各任务独立横轴，不与TabICLv2原始SHAP值混同。')
for i,t in enumerate(REPS,2):
    top=s[s.task==t].nsmallest(10,'task_rank').feature.tolist();chunks=[];paths=[]
    for fold in range(1,6):
        sp=XGB+f'01_shap/raw/{t}_fold_{fold}_shap.csv.gz';op=XGB+f'01_shap/raw/{t}_fold_{fold}_feature_original.csv.gz';sv=read(sp,usecols=['obs_id',*top]);ov=read(op,usecols=['obs_id',*top]);assert sv.obs_id.equals(ov.obs_id);paths +=[sp,op];take=np.linspace(0,len(sv)-1,min(300,len(sv))).astype(int)
        for rank,f in enumerate(top):chunks.append(pd.DataFrame({'shap':sv.iloc[take][f].to_numpy(),'rank':rank,'value_q':ov.iloc[take][f].rank(pct=True).fillna(.5).to_numpy()}))
    q=pd.concat(chunks,ignore_index=True);fig,ax=plt.subplots(figsize=(15.5/2.54,10.5/2.54));fig.subplots_adjust(left=.35,right=.88,bottom=.16,top=.92);sc=ax.scatter(q.shap,q['rank']+np.random.default_rng(42).normal(0,.08,len(q)),c=q.value_q,cmap='coolwarm',s=5,alpha=.45,lw=0);ax.set_yticks(range(10),[cn(f) for f in top]);ax.invert_yaxis();ax.axvline(0,c=GRAY,lw=.65);ax.set_xlabel('TreeSHAP（log1p目标尺度）');ax.set_title(LAB[t]+' · XGBoost总体解释',loc='left');cb=fig.colorbar(sc,ax=ax,fraction=.04,pad=.04);cb.set_label('特征值分位数');cb.solids.set_rasterized(False)
    save(fig,f'D-{i}',LAB[t].replace(' ','')+'TreeSHAP总体解释',paths,SCRIPT,f'图D-{i}','保留原脚本逐折等距抽取至多300个样本/特征及seed42散点抖动；原始TreeSHAP与特征值未改动。颜色是特征值分位数，不是贡献正负；解释对象是XGBoost，非TabICLv2。')
curves=read(XGB+'02_ale_1d/02_ale_1d_selected_curves.csv')
for i,t in enumerate(REPS,6):
    g=curves[curves.task==t];fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,15/2.54));fig.subplots_adjust(left=.14,right=.97,bottom=.13,top=.92,wspace=.5,hspace=.50)
    for ax,f in zip(axs.flat,g.feature.unique()):
        q=g[g.feature==f];ax.plot(q.x,q.ale_mean,c=BLUE,lw=1.2);ax.fill_between(q.x,q.ale_mean-q.fold_sd,q.ale_mean+q.fold_sd,color=BLUE,alpha=.16);ax.axhline(0,c=GRAY,lw=.65);ax.set_title(cn(f),loc='left',fontsize=9);ax.set_xlabel(xlabel(f),fontsize=8);ax.set_ylabel('ALE（'+targetunit(t)+'）',fontsize=8);grid(ax,'y');ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4))
    save(fig,f'D-{i}',LAB[t].replace(' ','')+'XGBoost一维ALE',[XGB+'02_ale_1d/02_ale_1d_selected_curves.csv'],SCRIPT,f'图D-{i}','受控复现XGBoost的历史冻结汇总。原始生产脚本在共同支持域线性插值形成41点均值，阴影为折间±1标准差（ddof=0），不是置信区间；本轮直接读取该冻结汇总，未新增插值或重新计算ALE。')
cons=read(XGB+'02_ale_1d/02_ale_1d_fold_consistency.csv');mat=cons.pivot(index='feature',columns='task',values='mean_pairwise_correlation').reindex(columns=REPS)
fig,ax=plt.subplots(figsize=(15.5/2.54,13.5/2.54));fig.subplots_adjust(left=.40,right=.89,bottom=.15,top=.93);im=ax.imshow(mat,cmap=DIV,vmin=-1,vmax=1,aspect='auto');ax.set_yticks(range(len(mat)),[cn(f) for f in mat.index]);ax.set_xticks(range(4),[LAB[t].replace(' ','\n') for t in REPS]);
for i in range(len(mat)):
    for j in range(4):ax.text(j,i,'NA' if pd.isna(mat.iloc[i,j]) else f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7.5)
cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.04);cb.set_label('五折平均两两相关');cb.solids.set_rasterized(False)
save(fig,'D-10','XGBoost一维ALE折间相关性',[XGB+'02_ale_1d/02_ale_1d_fold_consistency.csv'],SCRIPT,'图D-10','颜色为共同支持范围冻结汇总的五折平均两两相关系数，不是样本覆盖或响应强度。色标−1至1，NA为未定义任务—特征组合，不补零。')
reuse('D-11','TreeSHAP来源归因与消融标准化对照','结果/论文图表/figure_revision_v2/final_figures/图D-16_来源归因与消融分区对照.png',[XGB+'04_integrated/04_shap_ale_ablation_comparison.csv'],SCRIPT,'图D-16','两列各自在每任务六来源内z=(x−均值)/样本标准差（ddof=1）标准化，直接使用48个冻结z值。左x为SHAP来源份额，右x为删组ΔRMSE。z正负不等于原始ΔRMSE正负，z=0不表示无性能变化，不能跨任务比较原始效应大小，两类证据不构成因果验证。')
top=s[s.task_rank<=12].pivot(index='feature',columns='task',values='task_rank').reindex(columns=REPS);fig,ax=plt.subplots(figsize=(15.5/2.54,19/2.54));fig.subplots_adjust(left=.43,right=.90,bottom=.08,top=.95);im=ax.imshow(top,cmap=RANKC,vmin=1,vmax=12,aspect='auto');ax.set_yticks(range(len(top)),[cn(f) for f in top.index],fontsize=7.5);ax.set_xticks(range(4),[LAB[t].replace(' ','\n') for t in REPS]);
for i in range(len(top)):
    for j in range(4):
        if pd.notna(top.iloc[i,j]):ax.text(j,i,int(top.iloc[i,j]),ha='center',va='center',fontsize=7.5,color='white' if top.iloc[i,j]<5 else '#333')
cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.04);cb.set_label('任务内排名');cb.solids.set_rasterized(False)
save(fig,'D-12','四任务XGBoost主要特征排名矩阵',[XGB+'01_shap/01_shap_importance_summary.csv'],SCRIPT,'图D-17','各任务前12特征并集。数字为真实任务内排名，不再使用13−rank分值；未入某任务Top12的格留白，不代表贡献为零。')

fig,axs=plt.subplots(3,3,figsize=(15.5/2.54,19.5/2.54));fig.subplots_adjust(left=.12,right=.98,bottom=.17,top=.94,wspace=.55,hspace=.70);paths=[]
for ax,t in zip(axs.flat,TASKS[1:]):
    rel=P1+'gates/'+t+'_p1_probes.csv';d=read(rel);paths.append(rel);ext=[]
    for j,(obs,g) in enumerate(d.groupby('obs_id',sort=False)):
        assert len(g)==3;ax.plot([j,j],[g.p1_prediction.min(),g.p1_prediction.max()],c=GRAY,lw=1.6)
        for k,v in enumerate(g.p1_prediction):ax.scatter(j+(k-1)*.08,v,s=17,c=FOLDC[k],marker=['o','s','^'][k])
        ext.append(g.p1_prediction.max()-g.p1_prediction.min())
    ax.set_xticks(range(3),[f'{lab}\n极差={v:.2g}' for lab,v in zip(['低','中','高'],ext)],fontsize=7.5);ax.set_title(LAB[t],loc='left');ax.set_ylabel('预测（'+targetunit(t)+'）',fontsize=8);ax.margins(y=.25)
fig.text(.03,.025,'每任务3个不同query，各重复3次；极差按同query分组。\n缺Chl-a表层；三次重复范围不是预测置信区间，不宣称普遍稳定通过。',fontsize=8,linespacing=1.5)
save(fig,'H-1','九任务P1同查询重复预测审查',paths,SCRIPT,'图3-16','81行冻结记录，27个不同输入query各重复3次。不同query的高低预测差异不是重复波动；标注各query真实极差，包括0。任务独立原始单位，缺少Chl-a表层不补造，重复范围不是预测置信区间。')
reuse('H-2','叶绿素a表层解释尺度与重复波动对照','结果/论文图表/figure_revision_v2/final_figures/图3-17_TabICLv2叶绿素a表层解释效应尺度与预测重复波动对照.png',['runs/tabicl_explanation_numerical_stability_v2/jitter_vs_explanation_scale.csv'],SCRIPT,'图3-17','复用第二轮修订。仅Chl-a表层两个SHAP样本、两条ALE曲线；V1为历史失败事件，V2为原失败点及低/高预测值探针，V2零波动不能否定历史失败，不能外推十任务总体稳定性。')
reuse('H-3','历史验证划分结果的独立口径对照','结果/论文图表/figure_revision_v1/final_figures/图4-3_历史验证口径比较.png',['runs/tabiclv2_pilot/20_tabiclv2_pilot_summary.csv'],SCRIPT,'图4-3','历史Chl-a表层、Secchi、TN底层、TP中层。随机及年份为相应单次结果，站点差值取五折R²差值均值，不是三套统一pooled注册实验；不与Family V2正式十任务直接横比，不作时间/空间因素的因果推断。正文4.3保留明确引用。')
