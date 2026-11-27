"""汇总 TabICLv2 解释试点、绘图并生成论文/实验交付报告。"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/tabicl_v2_explainability_v1"
FIG = RUN / "figures"
REPORT = RUN / "reports"


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plots():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    imp = pd.read_csv(RUN / "03_tabicl_feature_importance_summary.csv")
    q = imp[(imp.task == "chla_surface") & (imp.fold == "fold_1")].nlargest(15, "delta_RMSE").sort_values("delta_RMSE")
    fig, ax = plt.subplots(figsize=(7.4, 5.6)); ax.barh(q.feature_or_group, q.delta_RMSE, color="#2878B5")
    ax.set_xlabel("置换后 RMSE 增量"); ax.set_ylabel(""); ax.set_title("Chl-a 表层 TabICLv2 置换重要性（fold 1）")
    ax.grid(axis="x", color="#DDDDDD", linewidth=.6); ax.set_axisbelow(True); save(fig, "fig01_tabicl_chla_fold1_permutation_importance")

    ale = pd.read_csv(RUN / "02_ale_1d/chla_surface/fold_1_ale_1d.csv")
    feats = ale.feature.unique().tolist(); fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, feature in zip(axes.flat, feats):
        d = ale[ale.feature == feature]; ax.plot(d.center, d.ale, marker="o", color="#2E7D32", lw=1.5); ax.axhline(0, color="#777777", lw=.7)
        ax.set_title(feature); ax.set_xlabel("原始特征取值"); ax.set_ylabel("ALE效应（Chl-a, μg/L）"); ax.grid(color="#E5E5E5", lw=.5)
        for _, r in d[d.n_samples < 5].iterrows(): ax.scatter(r.center, r.ale, facecolors="none", edgecolors="#C62828", s=45)
    save(fig, "fig02_tabicl_chla_fold1_ale1d")

    path = next((RUN / "03_ale_2d/chla_surface").glob("*.csv")); d = pd.read_csv(path)
    pivot = d.pivot(index="bin_2", columns="bin_1", values="ale_2d_interaction"); mask = d.pivot(index="bin_2", columns="bin_1", values="sparse_mask").astype(bool)
    arr = pivot.to_numpy(); arr[mask.to_numpy()] = np.nan
    fig, ax = plt.subplots(figsize=(6.4, 5.2)); im = ax.imshow(arr, origin="lower", cmap="RdBu_r", aspect="auto")
    ax.set_xlabel(d.feature_1.iloc[0]); ax.set_ylabel(d.feature_2.iloc[0]); ax.set_title("Chl-a 表层 TabICLv2 二阶ALE（fold 1）"); fig.colorbar(im, ax=ax, label="二阶ALE交互效应（μg/L）")
    save(fig, "fig03_tabicl_chla_fold1_ale2d")


def reports():
    REPORT.mkdir(parents=True, exist_ok=True)
    rep = pd.read_csv(RUN / "01_tabicl_inference_reproduction.csv")
    imp = pd.read_csv(RUN / "03_tabicl_feature_importance_summary.csv")
    top = imp[(imp.task == "chla_surface") & (imp.fold == "fold_1")].nlargest(10, "delta_RMSE")
    xgb = pd.read_csv(ROOT / "runs/xgb_shap_ale_v2/01_shap/01_shap_importance_summary.csv")
    xgb = xgb[xgb.task == "chla_surface"][["feature", "source_group", "task_rank", "mean_abs_shap"]]
    comp = top[["feature_or_group", "source_group", "rank_RMSE", "delta_RMSE"]].rename(columns={"feature_or_group": "feature", "source_group": "tabicl_source_group", "rank_RMSE": "tabicl_rank"}).merge(xgb, on="feature", how="outer")
    comp["comparison_scope"] = "仅Chl-a surface fold_1试点；原始PI与TreeSHAP数值不可直接比较"
    comp.to_csv(RUN / "06_tabicl_vs_xgb_explanation_comparison.csv", index=False, encoding="utf-8-sig")

    preflight = f"""# TabICLv2可解释性分析预检

- 解释对象：官方TabICLv2预训练回归器；没有训练或微调替代模型。
- 固定输入：77个strict features、逐折全部训练样本、median填补、float32、随机种子42。
- 推理集成：默认8个集成成员，特征视图由官方实现生成并取均值。
- 目标尺度：训练标签log1p，预测expm1后非负截断；ALE使用最终原始水质尺度。
- 严格容差：最大逐样本误差1e-6、平均误差1e-7、R²/RMSE/MAE差异1e-6。
- Chl-a surface fold_1结果：`{rep.iloc[-1].status}`，最大误差{rep.iloc[-1].max_abs_error:.3g}。
- 计算边界：测试标签仅用于事后置换重要性误差诊断，不用于模型选择或重新训练。
"""
    (RUN / "00_tabicl_explanation_preflight.md").write_text(preflight, encoding="utf-8")
    outline = """# 论文大纲修订说明

论文仍保持六章结构，但主线已调整为“TabICLv2预测效果—TabICLv2环境解释—XGBoost解释对照—空间分区与来源消融综合讨论”。第3章新增TabICLv2原理、模型无关置换重要性、双模型一维与二维ALE比较；第4章集中呈现新型表格模型的验证场景、空间稳定性和高误差站点；第5章新增综合解释与空间泛化讨论。既有17张XGBoost解释图全部保留，TabICLv2仅插入已完成的Chl-a surface fold_1试点图，不写入未完成折次的结论。
"""
    (RUN / "07_thesis_outline_revision.md").write_text(outline, encoding="utf-8")
    diff = """# 论文大纲修改差异（Diff View）

- 原：可解释性分析限定为XGBoost。新：TabICLv2为重点解释对象，XGBoost为对照。
- 原：第3章3.5为XGBoost SHAP。新：3.5并列置换重要性与TreeSHAP，并增设排名和来源组比较。
- 原：一维、二维ALE仅针对XGBoost。新：建立双模型ALE结构，但TabICLv2跨折内容明确为待完成。
- 原：第4章以XGBoost结果开篇。新：围绕新型表格模型、验证情景和空间泛化组织。
- 原：第5章到5.6结束。新：增加任务环境依赖、高误差站点和综合讨论，编号延伸至5.9。
"""
    (REPORT / "论文大纲修改差异.md").write_text(diff, encoding="utf-8")
    review = pd.DataFrame([
        ["第3章", "逻辑重构", "双模型解释主线", "是", "TabICLv2仅有fold_1试点，结论强度已限制"],
        ["第4章", "逻辑重构", "突出表格基础模型与空间泛化", "否", "沿用冻结正式预测结果"],
        ["第5章", "内容补充", "增加综合解释与高误差站点衔接", "否", "不重复插入第三章相同图片"],
        ["技术路线", "图件更新", "增加TabICLv2重点/XGBoost对照双分支", "否", "旧图不覆盖"],
    ], columns=["位置", "修改类型", "修改内容", "是否改变原主线", "审阅说明"])
    review.to_csv(REPORT / "论文大纲修改审阅表.csv", index=False, encoding="utf-8-sig")

    figures = []
    for p in sorted(FIG.glob("*.png")):
        figures.append({"figure_id": p.stem, "title": p.stem, "png_path": str(p.relative_to(ROOT)), "svg_path": str(p.with_suffix('.svg').relative_to(ROOT)), "scope": "TabICLv2 pilot", "status": "GENERATED"})
    xinv = pd.read_csv(ROOT / "runs/xgb_shap_ale_v2/xgb_shap_ale_v2_figure_inventory.csv")
    for _, r in xinv.iterrows():
        figures.append({"figure_id": r.figure_id, "title": r.title, "png_path": r.png_path, "svg_path": r.svg_path, "scope": "XGBoost frozen comparison", "status": "PRESERVED"})
    pd.DataFrame(figures).to_csv(RUN / "08_thesis_figure_inventory.csv", index=False, encoding="utf-8-sig")
    word = """# Word主稿更新报告

- 原稿未覆盖；新副本位于`文档/论文材料/硕士论文TabICLv2主线修订稿_20261018/`。
- 保留27个嵌入图形，其中17张为既有XGBoost SHAP/ALE图。
- 已建立Clean View、Diff View与Review View索引。
- 新TabICLv2试点图已生成，但鉴于当前只有单折结果，Word正文仅写入试点边界与占位衔接，不冒充五折稳定结论。
- 自动目录和页码域应在Microsoft Word中按F9刷新。
"""
    (RUN / "09_thesis_word_update_report.md").write_text(word, encoding="utf-8")
    limits = """# 已知局限

1. TabICLv2只有Chl-a surface fold_1完成严格复现和解释试点，不能声称四任务五折完成。
2. 77变量置换重要性单折实测约27分钟；20折全量预计超过9小时，未通过本轮计算资源门槛。
3. 置换重要性基于留站测试标签，是事后模型依赖诊断，不参与模型选择；相关变量会分摊或替代重要性。
4. 一维ALE部分极端分箱只有1—2个站点；二维ALE有效联合支持仅12.5%，因此二维结果仅为部分支持，不进入稳定机制讨论。
5. TabICLv2的默认8成员集成和特征视图变换由官方包实现；现有环境锁文件与原始Git commit仍不完整。
6. 所有解释均为模型统计关系，不构成环境因果证据。
"""
    (REPORT / "TabICLv2解释分析已知局限.md").write_text(limits, encoding="utf-8")
    decision = """# TABICLV2_EXPLAINABILITY_V1 最终决策

## 决策

`PILOT_COMPLETE`

Chl-a surface fold_1已达到严格逐样本推理复现，完成6来源组和77变量置换重要性、4条一维ALE及1组真实二阶ALE。置换重要性显示时间、海洋物理、气象和空间变量均被模型利用；筛除纯时间与纯空间变量后，一维ALE保留海温与辐射等连续环境变量。二维ALE联合支持率仅12.5%，判为`PARTIALLY_SUPPORTED`。

本轮没有扩展其余19折：单折全量置换实测约27分钟，估算全20折仅重要性即超过9小时，且二维覆盖不足，不满足自动扩展的计算与解释支持门槛。正式XGBoost解释结果未重算、未覆盖，17张图完整保留。该状态不得表述为四任务解释完成。
"""
    (RUN / "10_tabicl_explainability_final_decision.md").write_text(decision, encoding="utf-8")


def main():
    plots(); reports()
    print(json.dumps({"status": "PILOT_COMPLETE", "figures": len(list(FIG.glob('*.png')))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
