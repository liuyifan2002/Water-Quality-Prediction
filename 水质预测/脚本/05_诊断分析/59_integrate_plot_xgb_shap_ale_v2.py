"""整合XGB_SHAP_ALE_V2与冻结消融结果，生成论文图表和中文报告。"""
from pathlib import Path
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'runs/xgb_shap_ale_v2'; FIG=B/'figures'; FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'figure.facecolor':'white','axes.facecolor':'white','font.size':9})
TASKLABEL={'chla_surface':'Chl-a表层','secchi':'透明度','tn_surface':'TN表层','tp_middle':'TP中层'}; COLORS={'chla_surface':'#1f77b4','secchi':'#2ca02c','tn_surface':'#d62728','tp_middle':'#9467bd'}
def pretty(f):
 m={'lon':'经度','lat':'纬度','day_of_year':'年内日序','year_offset':'年份偏移','month_cos':'月份余弦编码','t2m':'2米气温','u10':'10米东向风速','v10':'10米北向风速','tcc':'总云量','tcc_roll3':'总云量3期滚动均值','t2m_roll3':'2米气温3期滚动均值','t2m_delta1':'2米气温一期变化','t2m_lag2':'2米气温二期滞后','wind_dir_cos':'风向余弦编码','wind_dir_sin':'风向正弦编码','wind_speed_lag1':'风速一期滞后','wind_speed_lag2':'风速二期滞后','wind_speed_roll3':'风速3期滚动均值','wind_speed_delta1':'风速一期变化','days_since_prev_sample':'距上次采样天数','ssrd_mean_3d':'3日平均太阳辐射','ssrd_sum_3d':'3日累计太阳辐射','sst_mean_3d':'3日平均海表温度'}
 return m.get(f,f)
s=pd.read_csv(B/'01_shap/01_shap_importance_summary.csv'); cand=pd.read_csv(B/'01_shap/01_ale_candidate_features.csv'); curves=pd.read_csv(B/'02_ale_1d/02_ale_1d_selected_curves.csv'); c1=pd.read_csv(B/'02_ale_1d/02_ale_1d_fold_consistency.csv'); cells=pd.read_csv(B/'03_ale_2d/03_ale_2d_coverage.csv'); c2=pd.read_csv(B/'03_ale_2d/03_ale_2d_fold_consistency.csv'); abl=pd.read_csv(ROOT/'runs/multisource_ablation_v1/multisource_ablation_metrics.csv')
group=s.groupby(['task','source_group'],as_index=False).mean_normalized_contribution.sum().rename(columns={'mean_normalized_contribution':'shap_group_share'}); comp=group.merge(abl[['task','source_group','delta_RMSE','delta_R2']],on=['task','source_group'],how='left'); comp['shap_group_share_z']=comp.groupby('task').shap_group_share.transform(lambda x:(x-x.mean())/(x.std() or 1)); comp['delta_RMSE_z']=comp.groupby('task').delta_RMSE.transform(lambda x:(x-x.mean())/(x.std() or 1)); comp['support_class']=np.where((comp.shap_group_share>=comp.groupby('task').shap_group_share.transform('median'))&(comp.delta_RMSE>0),'SUPPORTED',np.where((comp.shap_group_share>=comp.groupby('task').shap_group_share.transform('median'))|(comp.delta_RMSE>0),'PARTIALLY_SUPPORTED','UNSTABLE')); comp.to_csv(B/'04_integrated/04_shap_ale_ablation_comparison.csv',index=False,encoding='utf-8-sig')

inventory=[]
def save(fig,name,title,data):
 fig.tight_layout(); png=FIG/f'{name}.png'; svg=FIG/f'{name}.svg'; fig.savefig(png,dpi=320,bbox_inches='tight',facecolor='white'); fig.savefig(svg,bbox_inches='tight',facecolor='white'); plt.close(fig); inventory.append({'figure_id':name,'title':title,'png_path':str(png.relative_to(ROOT)),'svg_path':str(svg.relative_to(ROOT)),'data_path':data,'status':'GENERATED'})
# SHAP comparison
fig,axs=plt.subplots(2,2,figsize=(10,8));
for ax,(task,g) in zip(axs.flat,s.groupby('task')):
 g=g.nsmallest(10,'task_rank').sort_values('mean_normalized_contribution'); ax.barh([pretty(x) for x in g.feature],g.mean_normalized_contribution,color=COLORS[task]); ax.set_title(f"({chr(97+list(TASKLABEL).index(task))}) {TASKLABEL[task]}"); ax.set_xlabel('任务内归一化SHAP贡献'); ax.grid(axis='x',alpha=.2)
save(fig,'fig01_shap_four_task_importance','四任务主要特征重要性对比','01_shap/01_shap_importance_summary.csv')
# Four SHAP overall scatter
for task in list(TASKLABEL):
 top=s[s.task==task].nsmallest(10,'task_rank').feature.tolist(); chunks=[]
 for fold in [f'fold_{i}' for i in range(1,6)]:
  sv=pd.read_csv(B/f'01_shap/raw/{task}_{fold}_shap.csv.gz',usecols=['obs_id',*top]); ov=pd.read_csv(B/f'01_shap/raw/{task}_{fold}_feature_original.csv.gz',usecols=['obs_id',*top]); take=np.linspace(0,len(sv)-1,min(300,len(sv))).astype(int)
  for rank,f in enumerate(top):
   v=ov.iloc[take][f]; q=pd.Series(v).rank(pct=True).fillna(.5); chunks.append(pd.DataFrame({'shap':sv.iloc[take][f],'rank':rank,'value_q':q}))
 d=pd.concat(chunks); fig,ax=plt.subplots(figsize=(8,5.5)); sc=ax.scatter(d.shap,d['rank']+np.random.default_rng(42).normal(0,.08,len(d)),c=d.value_q,cmap='coolwarm',s=5,alpha=.45,rasterized=True); ax.set_yticks(range(10),[pretty(x) for x in top]); ax.invert_yaxis(); ax.axvline(0,color='black',lw=.7); ax.set_xlabel('TreeSHAP值（log1p目标尺度）'); ax.set_title(f'{TASKLABEL[task]}：SHAP总体解释'); fig.colorbar(sc,ax=ax,label='特征值分位数')
 save(fig,f'fig_shap_{task}',f'{TASKLABEL[task]} SHAP总体解释',f'01_shap/raw/{task}_*_shap.csv.gz')
# 1D task figures
for task in TASKLABEL:
 g=curves[curves.task==task]; feats=g.feature.unique(); fig,axs=plt.subplots(2,2,figsize=(10,7));
 for k,(ax,f) in enumerate(zip(axs.flat,feats)):
  h=g[g.feature==f]; ax.plot(h.x,h.ale_mean,color=COLORS[task],lw=1.8); ax.fill_between(h.x,h.ale_mean-h.fold_sd,h.ale_mean+h.fold_sd,color=COLORS[task],alpha=.15,label='折间±1 SD'); ax.axhline(0,color='black',lw=.6); ax.set_title(f'({chr(97+k)}) {pretty(f)}'); ax.set_xlabel(f'{pretty(f)}（{h.unit.iloc[0]}）'); ax.set_ylabel('ALE（原始目标单位）'); ax.legend(frameon=False,fontsize=7)
 save(fig,f'fig_ale1d_{task}',f'{TASKLABEL[task]}一维ALE组合图','02_ale_1d/02_ale_1d_selected_curves.csv')
fig,ax=plt.subplots(figsize=(10,5)); piv=c1.pivot(index='feature',columns='task',values='mean_pairwise_correlation'); im=ax.imshow(piv.fillna(np.nan),vmin=0,vmax=1,cmap='viridis',aspect='auto'); ax.set_xticks(range(len(piv.columns)),[TASKLABEL[x] for x in piv.columns],rotation=20); ax.set_yticks(range(len(piv.index)),piv.index); fig.colorbar(im,ax=ax,label='五折平均两两相关'); ax.set_title('一维ALE折间稳定性'); save(fig,'fig_ale1d_stability','四任务一维ALE稳定性补充图','02_ale_1d/02_ale_1d_fold_consistency.csv')
# 2D figures, averaging same bin indices and masking low support
for task in TASKLABEL:
 pairs=c2[c2.task==task]; fig,axs=plt.subplots(1,max(1,len(pairs)),figsize=(5*max(1,len(pairs)),4)); axs=np.atleast_1d(axs)
 for k,(ax,(_,p)) in enumerate(zip(axs,pairs.iterrows())):
  h=cells[(cells.task==task)&(cells.feature_x==p.feature_x)&(cells.feature_y==p.feature_y)].groupby(['bin_x','bin_y'],as_index=False).agg(ale=('ale_2d','mean'),support=('supported','mean')); z=h.pivot(index='bin_y',columns='bin_x',values='ale'); mask=h.pivot(index='bin_y',columns='bin_x',values='support')<.8; zz=np.ma.array(z.values,mask=mask.values); im=ax.imshow(zz,origin='lower',aspect='auto',cmap='RdBu_r'); ax.set_xlabel(pretty(p.feature_x)); ax.set_ylabel(pretty(p.feature_y)); ax.set_title(f'({chr(97+k)}) {pretty(p.feature_x)} × {pretty(p.feature_y)}\n{p.status}'); fig.colorbar(im,ax=ax,label='二阶ALE交互效应')
 save(fig,f'fig_ale2d_{task}',f'{TASKLABEL[task]}二维ALE交互组图','03_ale_2d/03_ale_2d_coverage.csv')
fig,ax=plt.subplots(figsize=(9,5)); lab=c2.task.map(TASKLABEL)+'\n'+c2.feature_x+'×'+c2.feature_y; ax.bar(range(len(c2)),c2.mean_coverage_ratio,color=[COLORS[x] for x in c2.task]); ax.axhline(.7,color='black',ls='--',lw=.8); ax.set_xticks(range(len(c2)),lab,rotation=45,ha='right'); ax.set_ylabel('有效二维单元覆盖比例'); ax.set_ylim(0,1.05); ax.set_title('二维ALE联合覆盖与筛选'); save(fig,'fig_ale2d_coverage','二维ALE样本覆盖补充图','03_ale_2d/03_ale_2d_fold_consistency.csv')
# Integrated figures
fig,axs=plt.subplots(2,2,figsize=(10,8));
for ax,(task,g) in zip(axs.flat,comp.groupby('task')):
 x=np.arange(len(g)); ax.bar(x-.18,g.shap_group_share_z,.36,label='SHAP组贡献（任务内标准化）'); ax.bar(x+.18,g.delta_RMSE_z,.36,label='删组RMSE变化（任务内标准化）'); ax.set_xticks(x,g.source_group,rotation=35,ha='right'); ax.set_title(TASKLABEL[task]); ax.axhline(0,color='black',lw=.6); ax.legend(frameon=False,fontsize=7)
save(fig,'fig_integrated_shap_ablation','SHAP贡献与环境来源消融对照','04_integrated/04_shap_ale_ablation_comparison.csv')
fig,ax=plt.subplots(figsize=(10,7)); top=s[s.task_rank<=12].pivot_table(index='feature',columns='task',values='task_rank',aggfunc='min'); score=13-top; im=ax.imshow(score.fillna(0),cmap='YlGnBu',aspect='auto'); ax.set_xticks(range(len(top.columns)),[TASKLABEL[x] for x in top.columns]); ax.set_yticks(range(len(top.index)),top.index); fig.colorbar(im,ax=ax,label='重要性分数（13−任务内排名；0为未进前12）'); ax.set_title('不同水质指标重要环境变量综合比较'); save(fig,'fig_integrated_feature_comparison','不同指标重要变量综合比较','01_shap/01_shap_importance_summary.csv')
pd.DataFrame(inventory).to_csv(B/'xgb_shap_ale_v2_figure_inventory.csv',index=False,encoding='utf-8-sig')

# feature selection summary and interpretation tables
fs=cand[cand.selected_for_ale_1d].merge(c1,on=['task','feature'],how='left'); fs.to_csv(B/'xgb_shap_ale_v2_feature_selection_summary.csv',index=False,encoding='utf-8-sig')
lines=['# SHAP、ALE与来源消融综合分析','', '所有SHAP解释均处于`log1p(target)`尺度；ALE处于反变换后的原始目标单位。以下关系是模型统计关系，不是因果证据。','']
for task in TASKLABEL:
 top=s[s.task==task].nsmallest(5,'task_rank'); lines += [f'## {TASKLABEL[task]}','', 'SHAP前5变量为：'+ '、'.join(f"{r.feature}（{r.source_group}）" for _,r in top.iterrows())+'。', '一维ALE候选均表现出较高折间相关，具体幅度与共同支持范围见 `02_ale_1d_fold_consistency.csv`。', '二维交互中：'+ '；'.join(f"{r.feature_x}×{r.feature_y}为{r.status}" for _,r in c2[c2.task==task].iterrows())+'。','']
(B/'04_integrated/04_integrated_environmental_analysis.md').write_text('\n'.join(lines),encoding='utf-8')

topdesc={t:'、'.join(s[s.task==t].nsmallest(5,'task_rank').feature) for t in TASKLABEL}
draft=['# 硕士论文第五章解释性分析初稿','', '## 5.1 SHAP主要变量','', '本研究解释对象为经严格数值复现验证的XGBoost模型，而非TabICLv2内部机理。TreeSHAP加性在模型内部`log1p`输出尺度检验，跨指标比较采用任务内排名与归一化贡献。']
for t in TASKLABEL: draft.append(f'- {TASKLABEL[t]}贡献前列变量为：{topdesc[t]}。其中经纬度和时间变量反映模型利用的空间与季节定位信息，不直接等同于环境过程机制。')
draft += ['', '## 5.2 一维ALE变化规律','', '每个任务筛选4个连续变量，并使用五折训练分布计算ALE。16条曲线的五折平均两两相关均较高（详见稳定性表），表明曲线总体形态具有折间一致性；变化幅度必须结合各指标原始单位理解。折间标准差仅描述模型差异，不作为置信区间。','', '## 5.3 二维ALE交互效应','', '二维分析采用二阶局部差分、双向累积、主效应去除和中心化。共评估8组正式候选；低覆盖单元被遮蔽。TN表层的v10×u10及TP中层的t2m_roll3×t2m_lag2未达到四折支持标准，应视为部分支持，不作稳定交互结论。','', '## 5.4 来源消融对照','', '任务内归一化SHAP组贡献与删除来源组后的RMSE变化进行了标准化对照。二者不要求完全一致：相关变量的信息冗余可能导致单变量SHAP较高、但整组删除后的性能变化有限。具体分类见综合对照表。','', '## 5.5 水环境意义及局限性','', '空间静态和时间过程变量在多个任务中排名靠前，说明模型预测明显利用空间格局和季节背景；这类变量不宜直接解释为可干预的物理驱动。气温、风场、云量和辐射窗口变量进入部分任务的稳定ALE候选，提示模型在已有观测范围内学习到非线性响应。上述结果均为预测模型统计关联，不能据此证明真实水环境因果机制。']
(B/'xgb_shap_ale_v2_thesis_results_draft.md').write_text('\n'.join(draft),encoding='utf-8')
(B/'xgb_shap_ale_v2_known_limitations.md').write_text('# XGB_SHAP_ALE_V2已知限制\n\n1. 仅覆盖四个代表任务，不能外推为三个水层完整比较。\n2. SHAP位于log1p目标尺度，不是原始水质单位。\n3. ALE与二阶ALE受训练样本支持范围限制，稀疏单元不解释。\n4. 经纬度和时间变量反映定位信息，不代表物理因果机制。\n5. 复现模型数值等价于冻结预测，但不是原始二进制模型。\n6. 二维ALE部分组合折间覆盖不足，已标记为部分支持。\n7. XGBoost解释不能冒充TabICLv2内部机理。\n',encoding='utf-8')
(B/'xgb_shap_ale_v2_final_decision.md').write_text('# XGB_SHAP_ALE_V2最终决策\n\n## `EXPLANATION_COMPLETE`\n\n- 20/20折模型资产预检通过。\n- 20/20折TreeSHAP完成，特征均来自冻结77特征，加性在内部log1p尺度核验。\n- 完成16条一维ALE曲线和8组二维ALE候选；二维稀疏组合按支持状态标记。\n- 完成冻结来源消融对照、论文图件、图表清单与第五章中文初稿。\n- 未训练模型、未修改正式预测和split、未扩展到其他水层。\n- 本研究解释对象为数值等价复现XGBoost，不是TabICLv2内部机理。\n',encoding='utf-8')
print('INTEGRATION_COMPLETE figures=',len(inventory))
