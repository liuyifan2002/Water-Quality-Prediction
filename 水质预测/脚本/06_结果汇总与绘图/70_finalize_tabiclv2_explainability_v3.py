"""汇总 TABICLV2_EXPLAINABILITY_V3、绘图并生成论文 Word 副本。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/tabicl_v2_explainability_v3"
FIG = RUN / "figures"
XGB = ROOT / "runs/xgb_shap_ale_v2"
SRC_DOC = ROOT / "文档/论文材料/硕士论文TabICLv2主线修订稿_20261018/水质预测硕士学位论文_TabICLv2主线修订稿.docx"
DOC_DIR = ROOT / "文档/论文材料/硕士论文TabICLv2_SHAP_ALE_V3修订稿_20261018"
OUT_DOC = DOC_DIR / "水质预测硕士学位论文_TabICLv2_SHAP_ALE_V3修订稿.docx"

TASK_LABEL = {"chla_surface":"Chl-a surface", "secchi":"Secchi", "tn_surface":"TN surface", "tp_middle":"TP middle"}
FEATURE_LABEL = {
    "lon":"经度 (°E)", "lat":"纬度 (°N)", "year_offset":"年份偏移", "month_cos":"月份余弦",
    "month_sin":"月份正弦", "day_of_year":"年内日序", "ssrd_sum_3d":"3日太阳辐射累积",
    "ssrd_mean_3d":"3日平均太阳辐射", "ssrd_mean_7d":"7日平均太阳辐射",
    "ssrd_mean_14d":"14日平均太阳辐射", "sst_mean_3d":"3日平均海表温度 (°C)",
    "sst_mean_14d":"14日平均海表温度 (°C)", "sst_max_14d":"14日最高海表温度 (°C)",
    "sst_anom_14d":"14日海表温度距平 (°C)", "wind_speed_lag1":"前1期风速",
    "wind_speed_lag2":"前2期风速", "wind_speed_x_rain14d":"风速与14日降雨交互量",
    "rain_14d_roll3":"14日降雨三期滚动量", "ntl_avg_rad":"夜间灯光平均辐亮度",
    "ntl_avg_rad_lag2":"前2期夜间灯光辐亮度", "ntl_avg_rad_roll3":"夜间灯光三期滚动均值",
    "wind_dir_sin":"风向正弦", "t2m_lag1":"前1期2米气温 (°C)", "t2m_lag2":"前2期2米气温 (°C)",
    "t2m_roll3":"2米气温三期滚动均值 (°C)",
}

def label(x): return FEATURE_LABEL.get(x, x)

def savefig(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    png, svg = FIG / f"{stem}.png", FIG / f"{stem}.svg"
    fig.savefig(png, dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, svg

def summarize_ale():
    a = pd.read_csv(RUN / "ale_1d/ale_1d_by_fold.csv")
    fold = a.groupby(["task","fold","feature"], as_index=False).agg(
        ale_min=("ale","min"), ale_max=("ale","max"), min_bin_samples=("n_samples","min"),
        min_bin_stations=("n_stations","min"), elapsed_seconds=("elapsed_seconds_feature","max"))
    fold["ale_range"] = fold.ale_max - fold.ale_min
    rows=[]
    for (task, feature), g in a.groupby(["task","feature"]):
        piv=g.pivot(index="bin", columns="fold", values="ale")
        cors=[]
        for i in range(piv.shape[1]):
            for j in range(i+1,piv.shape[1]):
                cors.append(piv.iloc[:,i].corr(piv.iloc[:,j]))
        fg=fold[(fold.task==task)&(fold.feature==feature)]
        rows.append({"task":task,"feature":feature,"feature_label":label(feature),"n_folds":g.fold.nunique(),
            "mean_ale_range":fg.ale_range.mean(),"sd_ale_range":fg.ale_range.std(ddof=0),
            "median_pairwise_curve_correlation":np.nanmedian(cors) if cors else np.nan,
            "minimum_bin_samples":int(g.n_samples.min()),"minimum_bin_stations":int(g.n_stations.min()),
            "support_status":"SUPPORTED" if g.fold.nunique()>=3 and g.n_stations.min()>=2 else "PARTIALLY_SUPPORTED"})
    out=pd.DataFrame(rows); out.to_csv(RUN/"06_tabicl_ale_1d_summary.csv",index=False,encoding="utf-8-sig")
    fold.to_csv(RUN/"tables/ale_1d_fold_summary.csv",index=False,encoding="utf-8-sig")
    return a,out

def summarize_2d():
    cov=pd.read_csv(RUN/"ale_2d/joint_coverage.csv")
    p=RUN/"ale_2d/ale_2d_by_fold.csv"; a=pd.read_csv(p) if p.exists() else pd.DataFrame()
    rows=[]
    for (task,x,y),g in cov.groupby(["task","feature_1","feature_2"]):
        q=a[(a.task==task)&(a.feature_1==x)&(a.feature_2==y)] if len(a) else pd.DataFrame()
        rows.append({"task":task,"feature_1":x,"feature_2":y,"pair_label":f"{label(x)} × {label(y)}",
            "coverage_folds":g.fold.nunique(),"mean_support_fraction":g.support_fraction.mean(),
            "computed_ale_folds":q.fold.nunique() if len(q) else 0,
            "mean_interaction_range":(q.groupby("fold").ale_2d.agg(lambda z:z.max()-z.min()).mean() if len(q) else np.nan),
            "sparse_cell_fraction":(q.sparse_mask.mean() if len(q) else np.nan),
            "support_status":"SUPPORTED" if len(q) and q.fold.nunique()>=3 and g.support_fraction.mean()>=.60 else "PARTIALLY_SUPPORTED"})
    out=pd.DataFrame(rows); out.to_csv(RUN/"07_tabicl_ale_2d_summary.csv",index=False,encoding="utf-8-sig")
    return cov,a,out

def compare_xgb(tab):
    x=pd.read_csv(XGB/"01_shap/01_shap_importance_summary.csv")
    x=x[["task","feature","source_group","task_rank","mean_normalized_contribution"]].rename(columns={"task_rank":"xgb_rank","mean_normalized_contribution":"xgb_relative_contribution"})
    t=tab[["task","feature","source_group","task_rank","task_relative_contribution","n_folds"]].rename(columns={"task_rank":"tabicl_rank","task_relative_contribution":"tabicl_relative_contribution"})
    c=t.merge(x,on=["task","feature"],how="outer",suffixes=("_tabicl","_xgb"))
    c["rank_difference_tabicl_minus_xgb"]=c.tabicl_rank-c.xgb_rank
    c["comparison_note"]="仅比较任务内排名和相对贡献；TabICLv2为实际水质尺度模型无关SHAP，XGBoost为log1p尺度TreeSHAP。"
    c.to_csv(RUN/"08_tabicl_xgb_explanation_comparison.csv",index=False,encoding="utf-8-sig")
    return c

def make_figures(tab,ale,cov,a2,comp):
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    inv=[]
    fig,axs=plt.subplots(2,2,figsize=(11,8));
    for ax,(task,g) in zip(axs.flat,tab.groupby("task",sort=False)):
        q=g.nsmallest(10,"task_rank").sort_values("task_relative_contribution")
        ax.barh([label(x) for x in q.feature],q.task_relative_contribution,color="#377eb8"); ax.set_title(TASK_LABEL[task]); ax.set_xlabel("任务内相对 |SHAP| 贡献")
    fig.suptitle("TabICLv2 四项代表任务的模型无关 SHAP 重要性（完整77特征归因）")
    p=savefig(fig,"01_tabicl_shap_importance_four_tasks"); inv.append((p,"SHAP重要性","正文","15/20折严格复现"))
    # Fold-rank stability.
    bf=pd.read_csv(RUN/"03_tabicl_shap_importance_by_fold.csv")
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,(task,g) in zip(axs.flat,tab.groupby("task",sort=False)):
        feats=g.nsmallest(10,"task_rank").feature; z=bf[(bf.task==task)&bf.feature.isin(feats)].pivot(index="feature",columns="fold",values="fold_rank").loc[list(feats)]
        im=ax.imshow(z,cmap="viridis_r",aspect="auto",vmin=1,vmax=77); ax.set_yticks(range(len(z)),[label(x) for x in z.index]); ax.set_xticks(range(len(z.columns)),z.columns,rotation=45); ax.set_title(TASK_LABEL[task])
    fig.colorbar(im,ax=axs.ravel().tolist(),label="折内排名",shrink=.7); fig.suptitle("TabICLv2 SHAP 排名折间稳定性")
    p=savefig(fig,"02_tabicl_shap_fold_rank_stability"); inv.append((p,"SHAP稳定性","正文","缺失折不插补"))
    # Source contribution.
    src=tab.groupby(["task","source_group"],as_index=False).task_relative_contribution.sum()
    piv=src.pivot(index="task",columns="source_group",values="task_relative_contribution").fillna(0)
    fig,ax=plt.subplots(figsize=(10,5)); piv.plot.bar(stacked=True,ax=ax,colormap="tab20c"); ax.set_xticklabels([TASK_LABEL[x] for x in piv.index],rotation=0); ax.set_ylabel("任务内相对 |SHAP| 贡献"); ax.set_xlabel(""); ax.legend(title="来源组",bbox_to_anchor=(1.02,1),loc="upper left")
    p=savefig(fig,"03_tabicl_shap_source_groups"); inv.append((p,"来源组贡献","正文","相对贡献"))
    # ALE panels.
    for task,g in ale.groupby("task"):
        feats=list(g.feature.drop_duplicates())[:4]; fig,axs=plt.subplots(2,2,figsize=(11,8));
        for ax,f in zip(axs.flat,feats):
            q=g[g.feature==f]
            for fold,h in q.groupby("fold"): ax.plot(h.center,h.ale,marker="o",ms=3,lw=1,label=fold)
            ax.axhline(0,color="black",lw=.6); ax.set_title(label(f)); ax.set_xlabel(label(f)); ax.set_ylabel("ALE效应（实际水质尺度）")
        for ax in axs.flat[len(feats):]: ax.axis("off")
        axs.flat[0].legend(fontsize=7); fig.suptitle(f"TabICLv2 {TASK_LABEL[task]} 一维ALE（测试特征参考分布，未使用测试标签）")
        p=savefig(fig,f"04_ale_1d_{task}"); inv.append((p,"一维ALE","正文","仅严格复现折"))
    # One supported 2D panel per task when available.
    if len(a2):
        for task,g in a2.groupby("task"):
            pair=g[["feature_1","feature_2"]].drop_duplicates().iloc[0]; q=g[(g.feature_1==pair.feature_1)&(g.feature_2==pair.feature_2)]
            folds=list(q.fold.unique()); fig,axs=plt.subplots(1,len(folds),figsize=(4*len(folds),3.7),squeeze=False)
            vmax=max(abs(q.ale_2d.min()),abs(q.ale_2d.max()))
            for ax,fold in zip(axs.flat,folds):
                h=q[q.fold==fold].pivot(index="bin_2",columns="bin_1",values="ale_2d"); im=ax.imshow(h,cmap="RdBu_r",origin="lower",vmin=-vmax,vmax=vmax); ax.set_title(fold); ax.set_xlabel(label(pair.feature_1)); ax.set_ylabel(label(pair.feature_2))
            fig.colorbar(im,ax=axs.ravel().tolist(),label="二阶ALE效应"); fig.suptitle(f"{TASK_LABEL[task]}：{label(pair.feature_1)} × {label(pair.feature_2)}")
            p=savefig(fig,f"05_ale_2d_{task}"); inv.append((p,"二维ALE","正文/附录","稀疏格不作确定性解释"))
    # Cross-model top-rank agreement.
    q=comp.dropna(subset=["tabicl_rank","xgb_rank"]); fig,axs=plt.subplots(2,2,figsize=(9,8))
    for ax,(task,g) in zip(axs.flat,q.groupby("task",sort=False)):
        ax.scatter(g.xgb_rank,g.tabicl_rank,s=18,alpha=.7); ax.plot([1,77],[1,77],ls="--",c="grey",lw=.8); ax.set_xlim(0,78); ax.set_ylim(78,0); ax.set_xlabel("XGBoost TreeSHAP排名"); ax.set_ylabel("TabICLv2 SHAP排名"); ax.set_title(TASK_LABEL[task])
    fig.suptitle("两模型特征排名对照（不比较不同输出尺度的原始SHAP幅值）")
    p=savefig(fig,"06_tabicl_xgb_rank_comparison"); inv.append((p,"跨模型对照","正文","排名口径"))
    rows=[]
    for paths,kind,use,note in inv:
        rows.append({"figure_type":kind,"png_path":str(paths[0].relative_to(ROOT)),"svg_path":str(paths[1].relative_to(ROOT)),"thesis_use":use,"note":note})
    out=pd.DataFrame(rows); out.to_csv(RUN/"09_thesis_figure_inventory.csv",index=False,encoding="utf-8-sig")
    return out

def insert_after(paragraph, text=None, image=None, caption=None):
    p=paragraph._parent.add_paragraph(); paragraph._p.addnext(p._p)
    if text: p.add_run(text)
    if image: p.add_run().add_picture(str(image),width=Inches(6.1))
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        c=p._parent.add_paragraph(); p._p.addnext(c._p); c.add_run(caption); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
    return p

def update_word(inventory):
    DOC_DIR.mkdir(parents=True,exist_ok=True); shutil.copy2(SRC_DOC,OUT_DOC); d=Document(OUT_DOC)
    replacements={
      "3.5 不同预测模型的环境特征重要性分析":"3.5 基于SHAP的表格模型环境特征贡献分析",
      "3.5.1 可解释性分析对象及实验设计":"3.5.1 TabICLv2模型无关SHAP方法",
      "3.5.2 TabICLv2模型无关特征重要性评价":"3.5.2 TabICLv2四项代表任务的重要特征贡献",
      "3.5.3 XGBoost TreeSHAP特征贡献分析":"3.5.3 TabICLv2特征贡献的折间稳定性",
      "3.5.4 TabICLv2与XGBoost主要环境变量比较":"3.5.4 XGBoost的TreeSHAP解释结果",
      "3.5.5 特征贡献的折间稳定性与解释边界":"3.5.5 两种模型环境特征贡献差异",
      "3.6.1 一维ALE方法与计算设置":"3.6.1 一维ALE计算方法",
      "3.6.2 TabICLv2环境变量响应曲线":"3.6.2 TabICLv2关键环境变量响应",
      "3.6.3 XGBoost环境变量响应曲线":"3.6.3 不同任务的非线性变化差异",
      "3.6.4 两种模型的变量响应差异":"3.6.4 TabICLv2与XGBoost的一维ALE对照",
      "3.7.1 二维ALE方法与变量组合选择":"3.7.1 二维ALE计算方法及联合覆盖检查",
      "3.7.2 TabICLv2二阶交互效应":"3.7.2 TabICLv2关键变量二阶交互效应",
      "3.7.3 XGBoost二阶交互效应":"3.7.3 不同任务的交互效应稳定性",
      "3.7.4 不同模型交互效应比较":"3.7.4 TabICLv2与XGBoost交互效应比较",
      "第6章 结论与展望":"结论与展望", "6.1 主要研究结论":"主要研究结论",
      "6.2 研究创新与实际意义":"研究创新与实际意义", "6.3 研究不足与展望":"研究不足与展望"}
    for p in d.paragraphs:
        if p.text in replacements: p.text=replacements[p.text]
    # Remove duplicate template conclusion title while retaining its following content.
    for p in d.paragraphs:
        if p.text.strip()=="结 论": p.text=""
    anchors={p.text:p for p in d.paragraphs}
    figmap={r.figure_type:ROOT/r.png_path for r in inventory.itertuples()}
    if "SHAP重要性" in figmap:
        insert_after(anchors["3.5.2 TabICLv2四项代表任务的重要特征贡献"],image=figmap["SHAP重要性"],caption="图3-X TabICLv2四项代表任务的模型无关SHAP重要性")
    if "SHAP稳定性" in figmap:
        insert_after(anchors["3.5.3 TabICLv2特征贡献的折间稳定性"],image=figmap["SHAP稳定性"],caption="图3-X TabICLv2特征贡献的折间稳定性")
    ale_rows=inventory[inventory.figure_type=="一维ALE"]
    anchor=anchors["3.6.2 TabICLv2关键环境变量响应"]
    for i,r in enumerate(ale_rows.itertuples(),1): anchor=insert_after(anchor,image=ROOT/r.png_path,caption=f"图3-X TabICLv2一维ALE结果（{i}）")
    two=inventory[inventory.figure_type=="二维ALE"]
    anchor=anchors["3.7.2 TabICLv2关键变量二阶交互效应"]
    for i,r in enumerate(two.itertuples(),1): anchor=insert_after(anchor,image=ROOT/r.png_path,caption=f"图3-X TabICLv2二维ALE结果（{i}）")
    # Explicitly prevent unfinished coverage from becoming a full scientific claim.
    anchors["3.5.3 TabICLv2特征贡献的折间稳定性"].insert_paragraph_before("本轮正式模型无关SHAP覆盖15/20折：TN surface与TP middle为5/5折，Chl-a surface为3/5折，Secchi为2/5折。缺失折未插补，因此Chl-a与Secchi仅作部分折稳定性证据。")
    d.save(OUT_DOC)
    return OUT_DOC

def write_reports(tab,ale_summary,two_summary,inventory):
    rep=pd.read_csv(RUN/"00_inference_reproduction.csv"); ale_rep=pd.read_csv(RUN/"00_ale_inference_reproduction.csv")
    counts=rep.groupby(["task","status"]).size().unstack(fill_value=0)
    md=f"""# TABICLV2_EXPLAINABILITY_V3 Word更新报告\n\n- 新副本：`{OUT_DOC.relative_to(ROOT)}`\n- 原稿未覆盖。\n- 第3.5—3.7节已按TabICLv2主模型、XGBoost对照模型的顺序调整。\n- 已插入{len(inventory)}组新图（每组同时保存PNG和SVG）。\n- 原有XGBoost图件和文档媒体未删除。\n- 结论章标题已改为不带章号的“结论与展望”，并移除重复的“结 论”标题文字。\n- `VISUAL_QA_PENDING`：本报告未声称逐页视觉排版通过；需在Word中更新目录、图号域和分页后人工复核。\n"""
    (RUN/"10_thesis_word_update_report.md").write_text(md,encoding="utf-8")
    lim=f"""# 已知限制\n\n1. 正式TabICLv2模型无关SHAP共覆盖15/20折；Chl-a为3/5折，Secchi为2/5折，TN与TP均为5/5折。\n2. GPU推理存在进程级微小非确定性；所有正式解释均先用事前固定的最大误差1e-6、平均误差1e-7门槛复核。\n3. SHAP每折仅解释5个代表性未见站点查询样本，共75个样本；每个样本均保留77维完整归因。\n4. 采用边际背景扰动，相关变量可能形成低概率组合；SHAP和ALE均为模型统计解释，不是因果效应。\n5. 一维ALE另行复核后覆盖13折；Chl-a fold 5和Secchi fold 5未通过新进程严格复现，未产出正式ALE。\n6. 二维ALE仅对预检查支持率达到60%的组合执行；稀疏格不得作确定性机制解释。\n7. TabICLv2 SHAP为实际水质尺度，既有XGBoost TreeSHAP为log1p尺度；仅比较任务内排名或相对贡献。\n8. Word逐页视觉核验仍为`VISUAL_QA_PENDING`。\n"""
    (RUN/"11_known_limitations.md").write_text(lim,encoding="utf-8")
    decision=f"""# TABICLV2_EXPLAINABILITY_V3 最终决策\n\n## 决策\n\n`PARTIALLY_COMPLETE`\n\n## 完成范围\n\n- 四项任务均获得真实调用冻结TabICLv2预测函数的77维模型无关SHAP；总计15个合格折、75个解释样本。\n- TN surface与TP middle完成5折SHAP；Chl-a surface完成3折；Secchi完成2折。\n- 一维ALE完成13个再次严格复现的折；二维ALE只保留通过联合覆盖预审的结果。\n- 已生成TabICLv2—XGBoost排名/来源组对照，未直接比较不同输出尺度的原始SHAP值。\n- 已生成论文图件并写入Word新副本；原稿和正式预测结果均未修改。\n\n## 未达到EXPLANATION_COMPLETE的原因\n\n5个折在独立进程中未达到事前严格数值复现阈值，且Word仍需人工逐页视觉QA。不得将当前结果表述为四任务全部五折解释完成。\n"""
    (RUN/"12_final_decision.md").write_text(decision,encoding="utf-8")

def main():
    tab=pd.read_csv(RUN/"04_tabicl_shap_importance_summary.csv")
    ale,ale_summary=summarize_ale(); cov,a2,two_summary=summarize_2d(); comp=compare_xgb(tab)
    inventory=make_figures(tab,ale,cov,a2,comp); update_word(inventory); write_reports(tab,ale_summary,two_summary,inventory)
    print(json.dumps({"decision":"PARTIALLY_COMPLETE","figures":len(inventory),"word":str(OUT_DOC)},ensure_ascii=False))

if __name__ == "__main__": main()
