"""真实地理底图与冻结站点指标、区域汇总绘图，不计算预测。"""
from common import *
import geopandas as gpd
from shapely.geometry import box
from matplotlib.patches import Rectangle
SCRIPT=Path(__file__).name
stations=read(COORD);assert len(stations)==94 and stations.Station.is_unique and stations.water_zone_en.nunique()==10
LAND=Path('E:/研究生/gis/natural_earth_vector/10m_physical/ne_10m_land.shp')
COUNTRY=Path('E:/研究生/gis/natural_earth_vector/110m_physical/ne_110m_land.shp')
for path in [LAND,COUNTRY]:
    for p in path.parent.glob(path.stem+'.*'):protect(p)
extent=(113.82,114.52,22.08,22.65)
land=gpd.read_file(LAND,bbox=extent)
def basemap(ax):
    ax.set_facecolor('#eef4f7');land.plot(ax=ax,color='#e4e5df',edgecolor='#a3aaa9',linewidth=.4)
    ax.set_xlim(extent[0],extent[1]);ax.set_ylim(extent[2],extent[3]);ax.set_aspect(1/np.cos(np.deg2rad(22.35)));ax.set_xticks([113.9,114.1,114.3,114.5]);ax.set_yticks([22.1,22.3,22.5]);ax.set_xticklabels(['113.9°E','114.1°E','114.3°E','114.5°E']);ax.set_yticklabels(['22.1°N','22.3°N','22.5°N']);ax.tick_params(labelsize=8);ax.spines[['top','right']].set_visible(True)
    ax.plot([113.89,113.89+10/(111.32*np.cos(np.deg2rad(22.13)))],[22.13,22.13],color='#444',lw=1.4);ax.text(113.89,22.145,'10 km',fontsize=7.5)
    ax.annotate('N',xy=(114.47,22.59),xytext=(114.47,22.52),ha='center',arrowprops={'arrowstyle':'-|>','lw':.8},fontsize=9)
ZCN={'Deep Bay':'后海湾','Eastern Buffer':'东部缓冲区','Junk Bay':'将军澳','Mirs Bay':'大鹏湾','North Western':'西北部水区','Port Shelter':'牛尾海','Southern':'南部水区','Southern Supplementary':'南部补充水区','Tolo Harbour and Channel':'吐露港及赤门海峡','Victoria Harbour':'维多利亚港','Western Buffer':'西部缓冲区'}
zonecolors=['#4477AA','#EE9944','#228833','#AA3377','#66AACC','#CCBB44','#997700','#77AADD','#44AA99','#BB6677']
fig=figcm(16.5);ax=fig.add_axes([.11,.29,.85,.66]);basemap(ax)
handles=[]
for (zone,g),c in zip(stations.groupby('water_zone_en',sort=True),zonecolors):
    ax.scatter(g.lon,g.lat,s=21,c=c,edgecolor='white',lw=.4,zorder=4);handles.append(Line2D([],[],marker='o',color=c,ls='',label=ZCN.get(zone,zone)))
ax.set_title('香港近岸监测网络与水区空间分布',loc='left',pad=9)
ax.text(114.13,22.45,'香港',fontsize=10,c='#555');ax.text(114.35,22.59,'深圳',fontsize=9,c='#555')
ia=fig.add_axes([.03,.065,.23,.19]);world=gpd.read_file(COUNTRY,bbox=(100,15,125,40));world.plot(ax=ia,color='#d9dfdc',edgecolor='#a4adaa',lw=.3);ia.set_xlim(105,123);ia.set_ylim(17,35);ia.scatter([114.17],[22.3],c=ORANGE,s=20,zorder=3);ia.text(114.8,21.9,'香港',fontsize=8);ia.set_xticks([]);ia.set_yticks([]);ia.spines[['top','right']].set_visible(True)
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.65,.06),ncol=2,frameon=False,fontsize=8.5)
fig.text(.06,.015,'94个监测站；颜色表示所属水区。底图：Natural Earth 1:10m。',fontsize=7.5)
save(fig,'2-1','香港近岸研究区域及94站监测网络',[COORD,str(LAND),str(COUNTRY)],SCRIPT,'图2-1','站点坐标及十个水区归属来自冻结station_meta；陆地海岸线来自Natural Earth 10m。所有站点使用同形符号，颜色仅区分水区，不表示筛选或等级。没有经核验的水区边界矢量，故以真实站点归属展示空间关系，不绘制推测边界。')

reuse('4-1','十项任务总体提升与站点改善覆盖','结果/论文图表/figure_revision_v1/final_figures/图4-1_总体精度与站点覆盖.png',[CORE],SCRIPT,'图4-1','正式五折pooled OOF R²及站点RMSE改善覆盖，保持表4-2任务顺序。负R²保留；站点改善率不是预测准确率，也不表示所有站点均受益。')
core=read(CORE);tab=core[core.model=='TabICLv2'];fig,axs=plt.subplots(1,2,figsize=(15.5/2.54,10/2.54));fig.subplots_adjust(left=.12,right=.97,bottom=.22,top=.87,wspace=.55)
for ax,field,title in zip(axs,['delta_R2','station_win_rate_RMSE'],['(a) pooled ΔR²','(b) 站点RMSE改善率（%）']):
    mat=np.array([[tab[tab.task==f+'_'+dep][field].iloc[0] for dep in ['surface','middle','bottom']] for f in ['chla','tn','tp']]);mat=mat*(100 if field.startswith('station') else 1)
    im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=100 if field.startswith('station') else max(.4,float(mat.max())));ax.set_xticks(range(3),['表层','中层','底层']);ax.set_yticks(range(3),['Chl-a','TN','TP']);ax.set_title(title,loc='left')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{mat[i,j]:.1f}%' if field.startswith('station') else f'{mat[i,j]:+.3f}',ha='center',va='center',fontsize=9,color='white' if mat[i,j]>(70 if field.startswith('station') else .28) else '#333')
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.04);cb.solids.set_rasterized(False)
fig.text(.06,.05,'本图为九个分层任务；Secchi为独立任务，完整结果已列图4-1。',fontsize=8)
save(fig,'4-2','不同指标与水层的预测提升差异',[CORE],SCRIPT,'图4-2','ΔR²=TabICLv2−XGBoost，正值表示整体R²提高；本图九个分层任务均为正。站点改善率为RMSE下降站点比例，两面板色标独立。Secchi没有三水层组合，已在图4-1独立展示。')
fig=figcm(8.5);axs=fig.subplots(1,3);fig.subplots_adjust(left=.04,right=.98,bottom=.20,top=.85,wspace=.18)
for ax,c,title,lines in zip(axs,[BLUE,ORANGE,GREEN],['随机留出','年份留出','站点留出'],[['同一总体内','随机样本预测','训练/测试随机分配'],['未参与训练年份','的时间外推','训练时段与测试时段分开'],['未参与训练站点','的新站点泛化','站点内观测整体留出']]):
    ax.axis('off');ax.add_patch(Rectangle((.03,.08),.94,.84,transform=ax.transAxes,facecolor='#f7f9fa',edgecolor=c,lw=1));ax.text(.5,.76,title,ha='center',fontsize=10,c=c);ax.text(.5,.43,'\n'.join(lines),ha='center',va='center',fontsize=8.5,linespacing=1.8)
fig.text(.05,.055,'评价对象不同，不以历史性能数字作统一注册实验的定量横比。',fontsize=8)
save(fig,'4-3','三种验证方案的评价对象与适用边界',[],SCRIPT,'图4-3','随机留出评价同一总体内随机样本预测，并不直接等同特征空间插值。年份与站点留出分别评价时间外推和新站点空间泛化；历史数值对照及不同统计口径详见附录H-3。')
modern=read(MODERN);records=[];fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,13.5/2.54));fig.subplots_adjust(left=.20,right=.97,bottom=.17,top=.93,wspace=.48,hspace=.43)
models=['xgboost','tabiclv2','tabpfn','tabm'];ml=['XGBoost','TabICLv2','TabPFN','TabM']
for ax,t in zip(axs.flat,MAPREPS):
    for i,(m,c) in enumerate(zip(models,[BLUE,ORANGE,PURPLE,GRAY])):
        q=modern[(modern.task==t)&(modern.model==m)];val=None;source=MODERN
        if len(q):val=q.pooled_R2.iloc[0]
        elif m in ['xgboost','tabiclv2']:
            val=core[(core.task==t)&(core.model==('XGBoost' if m=='xgboost' else 'TabICLv2'))].pooled_R2.iloc[0];source=CORE
        if val is not None:ax.scatter(val,i,s=34,c=c);ax.annotate(f'{val:.3f}',(val,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
        else:ax.text(.23,i,'NA：无同口径冻结结果',fontsize=7.5,va='center')
        records.append({'task':t,'model':m,'pooled_R2':val,'source':source if val is not None else 'NA：已查历史80/84特征结果不与正式77特征拼接'})
    ax.set_yticks(range(4),ml);ax.set_ylim(3.6,-.6);ax.set_xlim(.2,.72);ax.set_xlabel('pooled R²');ax.set_title(LAB[t],loc='left');grid(ax)
fig.text(.04,.025,'TN底层仅有两基准模型的正式结果；缺少TabPFN、TabM同口径结果，未补造。',fontsize=7.5)
pd.DataFrame(records).to_csv(OUT/'05_数据核验/图4-4冻结值与缺口.csv',index=False,encoding='utf-8-sig')
save(fig,'4-4','四项指定代表任务的表格模型比较',[MODERN,CORE],SCRIPT,'图4-4','指定Chl-a表层、Secchi、TN底层、TP中层。TN底层未定位到与正式77特征同口径的TabPFN、TabM结果，故标NA，不用历史80/84特征结果填补。表4-3保留完整TN表层四模型记录及TabPFN优势，两者任务范围不同。')

parts=[];metricsources=[]
for f in ['chla','tn','tp']:
    p=f'runs/{f}_family_v2/09_{f}_family_v2_station_metrics.csv';parts.append(read(p));metricsources.append(p)
allm=pd.concat(parts,ignore_index=True);sec=read('runs/tabiclv2_pilot/24_station_level_comparison.csv');sec=sec[sec.task=='secchi'].rename(columns={'xgb_RMSE':'XGB_RMSE','tabicl_RMSE':'TabICL_RMSE','xgb_n_obs':'n_obs'});allm=pd.concat([allm,sec],ignore_index=True)
fig,axs=plt.subplots(2,2,figsize=(15.5/2.54,16.5/2.54));fig.subplots_adjust(left=.10,right=.94,bottom=.07,top=.94,wspace=.42,hspace=.32)
for ax,t in zip(axs.flat,MAPREPS):
    d=allm[allm.task==t].merge(stations,left_on='station_id',right_on='Station',validate='one_to_one');assert len(d)==(94 if t=='secchi' else 84)
    v=d.TabICL_RMSE-d.XGB_RMSE;assert np.allclose(v,d.delta_RMSE);lim=max(abs(v.min()),abs(v.max()));basemap(ax);sc=ax.scatter(d.lon,d.lat,c=v,cmap=DIV,vmin=-lim,vmax=lim,s=24,edgecolors='#666',lw=.25,zorder=4);ax.set_title(LAB[t]+f'（{len(d)}站）',loc='left');cb=fig.colorbar(sc,ax=ax,orientation='horizontal',fraction=.045,pad=.10);cb.set_label('ΔRMSE（'+targetunit(t)+'）',fontsize=8);cb.ax.tick_params(labelsize=8)
save(fig,'4-5','四代表任务站点RMSE变化空间分布',[COORD,*metricsources,'runs/tabiclv2_pilot/24_station_level_comparison.csv',str(LAND)],SCRIPT,'新增','ΔRMSE=TabICLv2−XGBoost；负值（蓝）表示误差降低，正值（橙）表示误差增加。Chl-a表层、Secchi、TN底层、TP中层分别84/94/84/84站，独立零中心色标；不跨目标比较RMSE绝对幅度。Secchi站点指标的预测来源与正式冻结预测已核对一致。')
reuse('5-1','四代表任务多源环境信息消融比较','结果/论文图表/figure_revision_v1/final_figures/图5-1_来源消融性能变化.png',['transfer/llm_extension_transfer_v1/results/multisource_ablation/multisource_ablation_source_summary.csv'],SCRIPT,'图5-1','四任务XGBoost六来源冻结消融。ΔR²=消融模型−完整模型；负值为删除后性能下降，正值为提高，不解释为来源具有负面物理作用。六来源顺序统一，ΔR²共同尺度；RMSE详情见表5-1。')
sm=read(REG+'五折留站点验证_汇总.csv');folds=read(REG+'五折留站点验证_各折指标.csv');dec=read(REG+'分区训练决策.csv');feat=read(REG+'严格验证特征清单.csv');assert len(sm)==12 and len(folds)==120
order=[(f,z) for f in ['chla','secchi','tn','tp'] for z in ['surface','middle','bottom']];sm=sm.set_index(['target','depth']).loc[order];names=[('Chl-a' if f=='chla' else 'Secchi' if f=='secchi' else f.upper())+' '+{'surface':'表层','middle':'中层','bottom':'底层'}[z] for f,z in order]
fig,axs=plt.subplots(1,2,figsize=(15.5/2.54,16.5/2.54),gridspec_kw={'width_ratios':[1,1.4]});fig.subplots_adjust(left=.21,right=.90,bottom=.14,top=.90,wspace=.38)
v=sm.delta_r2_pooled.to_numpy();y=np.arange(12);ax=axs[0];ax.hlines(y,0,v,color=[ORANGE if x>0 else BLUE for x in v]);ax.scatter(v,y,c=[ORANGE if x>0 else BLUE for x in v],s=22);ax.axvline(0,c=GRAY,lw=.7);ax.set_yticks(y,names);ax.set_ylim(11.6,-.6);ax.set_xlim(-.39,.17);ax.set_xlabel('pooled ΔR²');ax.set_title('(a) 合并OOF结果',loc='left');grid(ax)
for i,x in enumerate(v):ax.annotate(f'{x:+.3f}',(x,i),xytext=(4 if x>0 else -4,0),textcoords='offset points',ha='left' if x>0 else 'right',va='center',fontsize=7.5)
pv=folds.pivot(index=['target','depth','fold'],columns='model_variant',values='r2');pv['delta']=pv.zone_aware_external-pv.global_external;mat=np.array([[pv.loc[(f,z,k),'delta'] for k in range(1,6)] for f,z in order]);lim=np.abs(mat).max();ax=axs[1];im=ax.imshow(mat,cmap=DIV,vmin=-lim,vmax=lim,aspect='auto');ax.set_yticks(y,[]);ax.set_xticks(range(5),range(1,6));ax.set_xlabel('留站折次');ax.set_title('(b) 逐折ΔR²',loc='left')
for i in range(12):
    for j in range(5):ax.text(j,i,f'{mat[i,j]:+.2f}',ha='center',va='center',fontsize=7.5,color='white' if abs(mat[i,j])>.65*lim else '#333')
    ax.text(4.65,i,f'{int(sm.positive_folds.iloc[i])}/5',va='center',fontsize=7.5)
ax.text(1.02,1.02,'改善折数',transform=ax.transAxes,fontsize=7.5);cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.07);cb.solids.set_rasterized(False)
fig.text(.03,.035,'历史78特征扩展（含water_zone_code），不是正式77特征十任务主比较。\n12个目标×水层；区域策略有496条区域模型与36条全域回退决策。',fontsize=7.5,linespacing=1.6)
save(fig,'5-2','历史78特征全域与分区建模比较',[REG+'五折留站点验证_汇总.csv',REG+'五折留站点验证_各折指标.csv',REG+'严格验证特征清单.csv',REG+'分区训练决策.csv'],SCRIPT,'新增','历史扩展实验，78特征包含water_zone_code，Secchi历史实验亦有三个水层；共12组合、60任务折。A读取冻结pooled ΔR²，B为冻结逐折区域R²减全域R²。532条决策中496为区域模型、36为全域回退，不表示每个水区均独立训练；不与正式77特征结果混合。')
casefile='文档/论文材料/硕士论文V2_学术内容补强_20261101/站点误差与特征覆盖补充核对.csv';cases=read(casefile);fig=figcm(19.5);gs=fig.add_gridspec(7,2,left=.18,right=.97,bottom=.08,top=.95,hspace=.75,wspace=.55)
ax=fig.add_subplot(gs[:3,0]);basemap(ax)
for sid,g in cases.groupby('station_id'):
    r=g.iloc[0];ax.scatter(r.lon,r.lat,c=BLUE,s=26,zorder=5);ax.annotate(sid,(r.lon,r.lat),xytext=(4,4 if sid!='SM19' else -11),textcoords='offset points',fontsize=8)
ax.set_title('(a) 四个站点位置',loc='left')
ax=fig.add_subplot(gs[:3,1]);q=cases.drop_duplicates('station_id');ax.barh(range(4),q.missing_77_share_in_main_surface*100,color=GRAY,height=.55);ax.set_yticks(range(4),q.station_id);ax.invert_yaxis();ax.axvline(8.788211788211789,ls='--',c=BLUE,lw=.8);ax.set_xlabel('表层77特征缺失率（%）');ax.set_title('(b) 表层输入覆盖背景',loc='left')
for i,v in enumerate(q.missing_77_share_in_main_surface*100):ax.text(v+1,i,f'{v:.1f}',va='center',fontsize=8)
# 五个案例使用嵌套GridSpec的独立横轴，避免不同单位共轴。
sub=gs[3:, :].subgridspec(5,1,hspace=1.0)
for i,r in enumerate(cases.itertuples()):
    ax=fig.add_subplot(sub[i]);ax.plot([r.XGB_RMSE,r.TabICL_RMSE],[0,0],c='#bbc4cb',lw=2);ax.scatter([r.XGB_RMSE,r.TabICL_RMSE],[0,0],c=[BLUE,ORANGE],s=25);ax.set_yticks([]);ax.set_xlim(0,max(r.XGB_RMSE,r.TabICL_RMSE)*1.45);ax.locator_params(axis='x',nbins=4);ax.set_title(f'{r.station_id} · {LAB[r.task]}（n={r.n_oof}）',loc='left',fontsize=8);ax.set_xlabel('RMSE（'+targetunit(r.task)+'）',fontsize=8)
    for v in [r.XGB_RMSE,r.TabICL_RMSE]:ax.annotate(f'{v:.3f}',(v,0),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7.5)
fig.legend(handles=[Line2D([],[],marker='o',c=BLUE,ls='',label='XGBoost'),Line2D([],[],marker='o',c=ORANGE,ls='',label='TabICLv2')],loc='lower right',bbox_to_anchor=(.97,.005),ncol=2,frameon=False,fontsize=8)
save(fig,'5-3','典型站点输入覆盖与预测误差诊断',[casefile,COORD,str(LAND)],SCRIPT,'新增','4个唯一站点、5个站点×任务案例。缺失率仅为表层主表77特征背景，虚线为94站中位数8.7882%；VT11为62.6852%。VT11表层主表71条与TN/TP OOF各67条分母不同。SM18中层不据此推断中层缺失率。各案例RMSE独立坐标；位置和缺失仅为诊断线索，不证明误差成因。')
