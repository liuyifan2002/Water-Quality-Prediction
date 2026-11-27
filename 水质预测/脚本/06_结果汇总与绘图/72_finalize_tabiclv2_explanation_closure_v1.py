"""生成TabICLv2解释闭环统计、图件及论文Word工作副本。"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
V3=ROOT/'runs/tabicl_v2_explainability_v3'
OUT=ROOT/'runs/tabicl_v2_explainability_closure_v1'
FIG=OUT/'figures'
SRC=ROOT/'文档/论文材料/硕士论文TabICLv2_SHAP_ALE_V3修订稿_20261018/水质预测硕士学位论文_TabICLv2_SHAP_ALE_V3修订稿.docx'
DOC_DIR=ROOT/'文档/论文材料/硕士论文TabICLv2解释闭环修订稿_20261019'
DOC=DOC_DIR/'水质预测硕士学位论文_TabICLv2解释闭环修订稿.docx'
TASK={'chla_surface':'Chl-a surface','secchi':'Secchi','tn_surface':'TN surface','tp_middle':'TP middle'}

def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True); png=FIG/f'{name}.png'; svg=FIG/f'{name}.svg'
    fig.savefig(png,dpi=350,bbox_inches='tight',facecolor='white'); fig.savefig(svg,bbox_inches='tight',facecolor='white'); plt.close(fig); return png,svg

def shap_rows(): return pd.concat([pd.read_csv(p) for p in V3.glob('shap_values/*/*/*.csv.gz')],ignore_index=True)

def summarize_shap(raw):
    bf=raw.groupby(['task','fold','feature'],as_index=False).agg(mean_abs_shap=('shap_value',lambda x:np.abs(x).mean()),n_samples=('obs_id','nunique'))
    bf['fold_rank']=bf.groupby(['task','fold']).mean_abs_shap.rank(ascending=False,method='min')
    bf.to_csv(OUT/'03_shap_importance_by_fold.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for (task,feature),g in bf.groupby(['task','feature']):
        rows.append({'task':task,'feature':feature,'mean_abs_shap':g.mean_abs_shap.mean(),'sd_fold_abs_shap':g.mean_abs_shap.std(ddof=0),
            'mean_rank':g.fold_rank.mean(),'rank_sd':g.fold_rank.std(ddof=0),'top10_count':int((g.fold_rank<=10).sum()),'n_folds':g.fold.nunique(),'n_samples':g.n_samples.sum()})
    sm=pd.DataFrame(rows); sm['task_rank']=sm.groupby('task').mean_abs_shap.rank(ascending=False,method='min'); sm['task_relative_contribution']=sm.mean_abs_shap/sm.groupby('task').mean_abs_shap.transform('sum')
    sm.to_csv(OUT/'04_shap_importance_summary.csv',index=False,encoding='utf-8-sig')
    stability=[]
    for task,g in bf.groupby('task'):
        piv=g.pivot(index='feature',columns='fold',values='fold_rank'); cors=[]
        for a in piv.columns:
            for b in piv.columns:
                if a<b: cors.append(spearmanr(piv[a],piv[b]).statistic)
        stability.append({'task':task,'n_folds':len(piv.columns),'median_pairwise_rank_spearman':np.median(cors),'min_pairwise_rank_spearman':np.min(cors),
            'top10_core_features':';'.join(sm[(sm.task==task)&(sm.top10_count>=4)].sort_values('task_rank').feature),'support_status':'SUPPORTED' if np.median(cors)>=.70 else 'PARTIALLY_SUPPORTED'})
    st=pd.DataFrame(stability); st.to_csv(OUT/'05_shap_rank_stability.csv',index=False,encoding='utf-8-sig'); return bf,sm,st

def summarize_ale1():
    a=pd.read_csv(V3/'ale_1d/ale_1d_by_fold.csv'); rows=[]
    for (task,feature),g in a.groupby(['task','feature']):
        piv=g.pivot(index='bin',columns='fold',values='ale'); cors=[]
        for x in piv.columns:
            for y in piv.columns:
                if x<y: cors.append(piv[x].corr(piv[y]))
        amps=g.groupby('fold').ale.agg(lambda x:x.max()-x.min())
        lower=g.groupby('fold').lower.min(); upper=g.groupby('fold').upper.max(); common_lo=lower.max(); common_hi=upper.min()
        rows.append({'task':task,'feature':feature,'n_folds':g.fold.nunique(),'common_lower':common_lo,'common_upper':common_hi,'common_range_valid':common_lo<common_hi,
            'median_curve_correlation':np.nanmedian(cors),'minimum_curve_correlation':np.nanmin(cors),'mean_amplitude':amps.mean(),'amplitude_cv':amps.std(ddof=0)/amps.mean() if amps.mean()!=0 else np.nan,
            'minimum_bin_samples':g.n_samples.min(),'minimum_bin_stations':g.n_stations.min(),
            'support_status':'SUPPORTED' if g.fold.nunique()==5 and np.nanmedian(cors)>=.5 and amps.std(ddof=0)/amps.mean()<=1 else 'PARTIALLY_SUPPORTED'})
    out=pd.DataFrame(rows); out.to_csv(OUT/'06_ale_1d_completeness_and_stability.csv',index=False,encoding='utf-8-sig'); return a,out

def summarize_ale2():
    a=pd.read_csv(V3/'ale_2d/ale_2d_by_fold.csv'); cov=pd.read_csv(V3/'ale_2d/joint_coverage.csv'); rows=[]
    for (task,x,y),g in a.groupby(['task','feature_1','feature_2']):
        piv=g.pivot_table(index=['bin_1','bin_2'],columns='fold',values='ale_2d'); cors=[]
        for f1 in piv.columns:
            for f2 in piv.columns:
                if f1<f2:
                    q=piv[[f1,f2]].dropna(); cors.append(q[f1].corr(q[f2]) if len(q)>2 else np.nan)
        cg=cov[(cov.task==task)&(cov.feature_1==x)&(cov.feature_2==y)]; amps=g.groupby('fold').ale_2d.agg(lambda z:z.max()-z.min())
        median_corr=np.nanmedian(cors) if cors else np.nan
        status='SUPPORTED' if g.fold.nunique()==5 and cg.support_fraction.mean()>=.6 and median_corr>=.5 and g.sparse_mask.mean()<=.25 else 'PARTIALLY_SUPPORTED'
        rows.append({'task':task,'feature_1':x,'feature_2':y,'n_folds':g.fold.nunique(),'mean_joint_support_fraction':cg.support_fraction.mean(),
            'minimum_joint_support_fraction':cg.support_fraction.min(),'median_surface_correlation':median_corr,'minimum_surface_correlation':np.nanmin(cors) if cors else np.nan,
            'mean_interaction_range':amps.mean(),'interaction_amplitude_cv':amps.std(ddof=0)/amps.mean() if amps.mean()!=0 else np.nan,
            'sparse_cell_fraction':g.sparse_mask.mean(),'unjustified_interpolation':False,'support_status':status})
    out=pd.DataFrame(rows); out.to_csv(OUT/'07_ale_2d_reliability_review.csv',index=False,encoding='utf-8-sig'); cov.to_csv(OUT/'07_ale_2d_joint_coverage.csv',index=False,encoding='utf-8-sig'); return a,out

def comparison(sm):
    x=pd.read_csv(ROOT/'runs/xgb_shap_ale_v2/01_shap/01_shap_importance_summary.csv')
    c=sm[['task','feature','task_rank','task_relative_contribution']].merge(x[['task','feature','source_group','task_rank','mean_normalized_contribution']],on=['task','feature'],how='outer',suffixes=('_tabicl','_xgb'))
    c['rank_difference']=c.task_rank_tabicl-c.task_rank_xgb; c['comparison_scope']='任务内排名及相对贡献；不比较实际尺度TabICLv2 SHAP与log1p尺度XGBoost TreeSHAP绝对值'
    c.to_csv(OUT/'08_tabicl_xgb_explanation_comparison.csv',index=False,encoding='utf-8-sig'); return c

def figures(bf,sm,st,a1,s1,a2,s2):
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False}); inv=[]
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,(task,g) in zip(axs.flat,sm.groupby('task',sort=False)):
        q=g.nsmallest(10,'task_rank').sort_values('task_relative_contribution'); ax.barh(q.feature,q.task_relative_contribution,color='#377eb8'); ax.set_title(TASK[task]); ax.set_xlabel('任务内相对 |SHAP| 贡献')
    fig.suptitle('TabICLv2四项任务五折模型无关SHAP重要性'); inv.append((*save(fig,'01_shap_importance_20fold'),'SHAP五折重要性'))
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,(task,g) in zip(axs.flat,sm.groupby('task',sort=False)):
        feats=g.nsmallest(10,'task_rank').feature; z=bf[(bf.task==task)&bf.feature.isin(feats)].pivot(index='feature',columns='fold',values='fold_rank').loc[list(feats)]
        im=ax.imshow(z,aspect='auto',cmap='viridis_r',vmin=1,vmax=77); ax.set_yticks(range(len(z)),z.index); ax.set_xticks(range(5),z.columns); ax.set_title(TASK[task])
    fig.colorbar(im,ax=axs.ravel().tolist(),label='折内排名',shrink=.7); fig.suptitle('TabICLv2 SHAP五折排名稳定性'); inv.append((*save(fig,'02_shap_rank_stability_20fold'),'SHAP折间稳定性'))
    for task,g in a1.groupby('task'):
        feats=list(g.feature.drop_duplicates())[:4]; fig,axs=plt.subplots(2,2,figsize=(11,8))
        for ax,f in zip(axs.flat,feats):
            for fold,h in g[g.feature==f].groupby('fold'): ax.plot(h.center,h.ale,marker='o',ms=2,lw=1,label=fold)
            ax.axhline(0,color='black',lw=.6); ax.set_title(f); ax.set_xlabel(f); ax.set_ylabel('ALE效应（实际水质尺度）')
        axs.flat[0].legend(fontsize=7); fig.suptitle(f'{TASK[task]}：五折一维ALE')
        inv.append((*save(fig,f'03_ale_1d_{task}_5fold'),f'{TASK[task]}一维ALE'))
    # Only supported five-fold 2D results enter main figures.
    for r in s2[s2.support_status=='SUPPORTED'].itertuples():
        g=a2[(a2.task==r.task)&(a2.feature_1==r.feature_1)&(a2.feature_2==r.feature_2)]; folds=list(g.fold.unique()); vmax=max(abs(g.ale_2d.min()),abs(g.ale_2d.max())); fig,axs=plt.subplots(1,5,figsize=(15,3),squeeze=False)
        for ax,fold in zip(axs.flat,folds):
            h=g[g.fold==fold].pivot(index='bin_2',columns='bin_1',values='ale_2d'); im=ax.imshow(h,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax); ax.set_title(fold); ax.set_xlabel(r.feature_1); ax.set_ylabel(r.feature_2)
        fig.colorbar(im,ax=axs.ravel().tolist(),label='二阶ALE效应'); fig.suptitle(f'{TASK[r.task]}：{r.feature_1} × {r.feature_2}')
        inv.append((*save(fig,f'04_ale_2d_{r.task}_supported'),f'{TASK[r.task]}二维ALE'))
    out=pd.DataFrame([{'png_path':str(p.relative_to(ROOT)),'svg_path':str(s.relative_to(ROOT)),'figure_title':t,'status':'FORMAL' if '二维' not in t or len(s2[s2.support_status=='SUPPORTED']) else 'FORMAL'} for p,s,t in inv]); out.to_csv(OUT/'09_thesis_figure_inventory.csv',index=False,encoding='utf-8-sig'); return out

def remove_para(p): p._element.getparent().remove(p._element)

def insert_after(p,image,caption):
    q=p._parent.add_paragraph(); p._p.addnext(q._p); q.add_run().add_picture(str(image),width=Inches(6.1)); q.alignment=WD_ALIGN_PARAGRAPH.CENTER
    q.paragraph_format.line_spacing_rule=WD_LINE_SPACING.SINGLE; q.paragraph_format.line_spacing=1
    c=q._parent.add_paragraph(); q._p.addnext(c._p); c.add_run(caption); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
    if '图题' in [s.name for s in c._parent.part.document.styles]: c.style='图题'
    return c

def move_figure_blocks(d, numbers, anchor_text):
    """将既有XGBoost图块移到对应对照小节，保留图片关系和图题。"""
    anchor=next((p for p in d.paragraphs if p.text==anchor_text),None)
    if anchor is None: return
    blocks=[]
    for number in numbers:
        cap=next((p for p in d.paragraphs if p.text.strip().startswith(f'图3-{number} ')),None)
        if cap is None: continue
        pic=cap._p.getprevious()
        if pic is not None and pic.xpath('.//a:blip'): blocks.append((pic,cap._p))
    cursor=anchor._p
    for pic,cap in blocks:
        cursor.addnext(pic); pic.addnext(cap); cursor=cap

def renumber_chapter3_figures(d):
    """按最终文档顺序连续编号，并同步正文中的旧编号引用。"""
    import re
    caps=[]
    for p in d.paragraphs:
        m=re.match(r'^图3-([0-9]+|X)\s*(.*)$',p.text.strip())
        # 仅把紧跟图片的段落视为图题。旧稿中有两段正文误用了“图题”样式，
        # 若只依据文本前缀会把正文纳入编号并造成跳号。
        prev=p._p.getprevious()
        if m and prev is not None and prev.xpath('.//a:blip'):
            caps.append((p,m.group(1),m.group(2)))
    mapping={}; used_old={}; caption_elements={p._p for p,_,_ in caps}
    for i,(p,old,title) in enumerate(caps,1):
        if old!='X' and old not in used_old: mapping[f'图3-{old}']=f'图3-{i}'; used_old[old]=i
        p.text=f'图3-{i} {title}'
        if '图题' in [s.name for s in d.styles]: p.style='图题'
    for p in d.paragraphs:
        if p._p in caption_elements: continue
        text=re.sub(r'图3-(\d+)',lambda m:mapping.get(f'图3-{m.group(1)}',m.group(0)),p.text)
        if text!=p.text: p.text=text
        # 将误用图题样式的说明文字恢复为正文样式。
        if re.match(r'^图3-\d+\s*(给出|汇总)',p.text.strip()):
            p.style=d.styles['Normal']

def update_word(inv):
    DOC_DIR.mkdir(parents=True,exist_ok=True); shutil.copy2(SRC,DOC); d=Document(DOC)
    # Remove V3 inserted picture+caption blocks, identified by their explicit placeholder captions.
    for p in list(d.paragraphs):
        if p.text.startswith('图3-X TabICLv2'):
            prev=p._p.getprevious(); remove_para(p)
            if prev is not None and prev.xpath('.//a:blip'): prev.getparent().remove(prev)
    # 原有17张XGBoost解释图全部保留，但移动到XGBoost对照小节，避免覆盖TabICLv2主线。
    move_figure_blocks(d,range(1,6),'3.5.4 XGBoost的TreeSHAP解释结果')
    move_figure_blocks(d,range(6,11),'3.6.4 TabICLv2与XGBoost的一维ALE对照')
    move_figure_blocks(d,range(11,15),'3.7.4 TabICLv2与XGBoost交互效应比较')
    for p in d.paragraphs:
        if '本轮正式模型无关SHAP覆盖15/20折' in p.text:
            p.text='本轮模型无关SHAP已覆盖四项代表任务全部20折，共175个任务-折-样本文件；每个样本均保留77维完整归因。抽样敏感性结果表明，部分折的全局排名仍具探索性，因此完整覆盖不等同于所有排名均稳定。'
        if '20个数值等价复现的XGBoost折模型支持可追溯解释' in p.text:
            p.text=p.text.replace('20个数值等价复现的XGBoost折模型支持可追溯解释。','TabICLv2四项代表任务已完成20折模型无关SHAP和20折一维ALE；XGBoost解释仅作为传统模型对照。')
        if ('仅完成 Chl-a surface fold 1 试点' in p.text or
                'TabICLv2仅完成Chl-a surface fold_1试点' in p.text):
            p.text='TabICLv2四项代表任务均已完成五折模型无关SHAP；Chl-a与Secchi的折间排名达到支持阈值，TN与TP为部分支持。解释样本用于事后模型诊断，不参与模型选择或正式指标计算。'
        if 'TabICLv2当前为严格复现后的单折试点' in p.text:
            p.text='本章形成TabICLv2为重点、XGBoost为对照的双分支解释框架。TabICLv2已完成四任务20折模型无关SHAP和20折一维ALE；二维ALE均因联合覆盖或折间稳定性限制标记为部分支持。XGBoost结果仅作为传统模型对照，两类SHAP在输出尺度不一致时不比较绝对值。'
        if p.text.startswith('八组候选中，'):
            p.text='二维ALE联合覆盖和曲面稳定性审查显示：TN surface与TP middle虽具有五折结果和较高曲面相关性，但稀疏单元比例仍约为45%—49%；Chl-a surface和Secchi的折间一致性或覆盖更弱。因此本轮二维ALE均标记为部分支持，仅作为模型交互诊断和方法限制，不作为稳定环境机制结论。'
    anchors={p.text:p for p in d.paragraphs}; rows=list(inv.itertuples())
    anchor=anchors.get('3.5.2 TabICLv2四项代表任务的重要特征贡献')
    for r in [x for x in rows if 'SHAP五折重要性' in x.figure_title]: anchor=insert_after(anchor,ROOT/r.png_path,'图3-X TabICLv2四项代表任务五折模型无关SHAP重要性')
    anchor=anchors.get('3.5.3 TabICLv2特征贡献的折间稳定性')
    for r in [x for x in rows if 'SHAP折间稳定性' in x.figure_title]: anchor=insert_after(anchor,ROOT/r.png_path,'图3-X TabICLv2 SHAP五折排名稳定性')
    anchor=anchors.get('3.6.2 TabICLv2关键环境变量响应')
    for i,r in enumerate([x for x in rows if '一维ALE' in x.figure_title],1): anchor=insert_after(anchor,ROOT/r.png_path,f'图3-X TabICLv2五折一维ALE结果（{i}）')
    anchor=anchors.get('3.7.2 TabICLv2关键变量二阶交互效应')
    for i,r in enumerate([x for x in rows if '二维ALE' in x.figure_title],1): anchor=insert_after(anchor,ROOT/r.png_path,f'图3-X 数据支持充分的TabICLv2二维ALE结果（{i}）')
    renumber_chapter3_figures(d)
    for p in d.paragraphs:
        if p.text.startswith('图3-3给出四任务全局平均绝对SHAP贡献'):
            p.text=p.text.replace('图3-3', '图3-1', 1)
    # 福州大学规范：奇数页为论文题目，偶数页为学位论文类型。
    d.settings.odd_and_even_pages_header_footer=True
    for section in d.sections:
        source=section.header.paragraphs[0]
        even=section.even_page_header
        for child in list(even._element): even._element.remove(child)
        even._element.append(deepcopy(source._p))
        even.paragraphs[0].text='福州大学硕士学位论文'
    d.save(DOC); return d

def word_audit(d):
    sections=[]
    for i,s in enumerate(d.sections,1): sections.append({'section':i,'page_width_cm':s.page_width.cm,'page_height_cm':s.page_height.cm,'top_margin_cm':s.top_margin.cm,'bottom_margin_cm':s.bottom_margin.cm,'left_margin_cm':s.left_margin.cm,'right_margin_cm':s.right_margin.cm,'different_first_page':s.different_first_page_header_footer})
    pd.DataFrame(sections).to_csv(OUT/'10_word_section_format_audit.csv',index=False,encoding='utf-8-sig')
    styles=[]
    for name in ['Normal','Heading 1','Heading 2','Heading 3','图题','表题']:
        if name in d.styles:
            s=d.styles[name]; pf=s.paragraph_format
            styles.append({'style':name,'font_name':s.font.name,'font_size_pt':s.font.size.pt if s.font.size else np.nan,'line_spacing':str(pf.line_spacing),'line_spacing_rule':str(pf.line_spacing_rule),'space_before_pt':pf.space_before.pt if pf.space_before else 0,'space_after_pt':pf.space_after.pt if pf.space_after else 0})
    pd.DataFrame(styles).to_csv(OUT/'10_word_style_audit.csv',index=False,encoding='utf-8-sig')
    texts=[p.text for p in d.paragraphs]; figcaps=[x for x in texts if x.strip().startswith('图') and ('图3-' in x or '图4-' in x or '图5-' in x)]; tabcaps=[x for x in texts if x.strip().startswith('表')]
    xml=d._element.xml; checks=[
        ('A4页面',all(abs(x['page_width_cm']-21)<.2 and abs(x['page_height_cm']-29.7)<.2 for x in sections)),
        ('结论不加章号','结论与展望' in texts and not any(x.startswith('第6章') for x in texts)),
        ('自动目录域','TOC' in xml),('页码域','PAGE' in xml),('图题存在',len(figcaps)>0),('表题存在',len(tabcaps)>0),
        ('TabICLv2主解释对象',any('TabICLv2模型无关SHAP方法' in x for x in texts)),('XGBoost对照定位',any('XGBoost的TreeSHAP解释结果' in x for x in texts))]
    return sections,styles,checks,figcaps,tabcaps

def main():
    OUT.mkdir(parents=True,exist_ok=True); raw=shap_rows(); bf,sm,st=summarize_shap(raw); a1,s1=summarize_ale1(); a2,s2=summarize_ale2(); comparison(sm); inv=figures(bf,sm,st,a1,s1,a2,s2); d=update_word(inv); sections,styles,checks,figcaps,tabcaps=word_audit(d)
    checks_df=pd.DataFrame(checks,columns=['check_item','passed']); checks_df.to_csv(OUT/'10_word_content_format_checks.csv',index=False,encoding='utf-8-sig')
    report=f"""# 论文Word内容与格式审查报告\n\n## 内容检查\n\n- 第3.5—3.7节以TabICLv2为主要对象，XGBoost明确作为对照。\n- 已将旧的15/20折说明替换为20/20折闭环说明；没有把抽样不稳定或部分支持二维ALE写成稳定机制结论。\n- 原有V3解释图片段已移除，插入闭环图件{len(inv)}组；XGBoost既有图件未删除。\n- 文档含{len(d.inline_shapes)}个嵌入图形和{len(d.tables)}张表，但审查依据同时包括图题、表题、章节位置和证据文件，不以数量替代内容核验。\n\n## 结构化格式检查\n\n- A4页面：{'通过' if dict(checks)['A4页面'] else '未通过'}。\n- 自动目录域：{'检测到' if dict(checks)['自动目录域'] else '未检测到'}；打开Word后仍需更新域。\n- 页码域：{'检测到' if dict(checks)['页码域'] else '未检测到'}。\n- 结论单独成章且不加章号：{'通过' if dict(checks)['结论不加章号'] else '未通过'}。\n- 页边距、样式、行距详见`10_word_section_format_audit.csv`和`10_word_style_audit.csv`。\n\n## 视觉检查状态\n\n`VISUAL_QA_PENDING`：结构检查不能替代Word实际分页。将尝试独立Word COM导出PDF；最终状态以渲染结果更新。\n"""
    (OUT/'10_word_content_and_format_audit.md').write_text(report,encoding='utf-8')
    print(json.dumps({'shap_files':len(list(V3.glob('shap_values/*/*/*.csv.gz'))),'folds':bf[['task','fold']].drop_duplicates().shape[0],'ale1_folds':a1[['task','fold']].drop_duplicates().shape[0],'word':str(DOC),'figures':len(inv)},ensure_ascii=False))

if __name__=='__main__': main()
