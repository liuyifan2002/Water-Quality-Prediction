"""在既有整合稿副本上更新 TabICLv2 主线大纲并生成审阅材料。"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "文档/论文材料/硕士论文整合排版稿_20261018/水质预测硕士学位论文_整合排版稿.docx"
OUT_DIR = ROOT / "文档/论文材料/硕士论文TabICLv2主线修订稿_20261018"
OUT = OUT_DIR / "水质预测硕士学位论文_TabICLv2主线修订稿.docx"
RUN = ROOT / "runs/tabicl_v2_explainability_v1"


RENAME = {
    "3.1 总体研究框架": "3.1 预测模型概述",
    "3.2 预测模型": "3.2 TabICLv2模型原理",
    "3.2.1 XGBoost基准模型": "3.2.1 上下文学习与预训练模型",
    "3.2.2 TabICLv2表格基础模型": "3.2.2 训练上下文构造与任务适配",
    "3.2.3 TabPFN与TabM": "3.2.3 特征编码与集成推理",
    "3.3.1 统一特征及模型输入": "3.3.1 随机划分与年份留出方案",
    "3.3.2 验证方案": "3.3.2 站点留出方案",
    "3.3.3 五折留站验证": "3.3.3 统一五折站点划分",
    "3.3.4 模型配置及可重复性": "3.3.4 模型配置与推理环境",
    "3.3.5 分海区及来源消融的实验设计": "3.3.5 复现性控制与数据泄漏防范",
    "3.4 模型性能评价指标": "3.4 模型性能与空间泛化评价指标",
    "3.5 基于XGBoost的SHAP变量贡献分析": "3.5 不同预测模型的环境特征重要性分析",
    "3.5.1 受控复现模型及一致性验证": "3.5.1 可解释性分析对象及实验设计",
    "3.5.2 TreeSHAP方法及特征贡献": "3.5.2 TabICLv2模型无关特征重要性评价",
    "3.5.3 变量筛选与稳定性判断": "3.5.3 XGBoost TreeSHAP特征贡献分析",
    "3.6 一维ALE影响曲线": "3.6 基于一维ALE的环境变量响应分析",
    "3.6.1 一维ALE的基本原理": "3.6.1 一维ALE方法与计算设置",
    "3.6.2 变量响应曲线": "3.6.2 TabICLv2环境变量响应曲线",
    "3.6.3 折间稳定性及样本覆盖": "3.6.3 XGBoost环境变量响应曲线",
    "3.7 二维ALE联合影响分析": "3.7 基于二维ALE的环境变量交互分析",
    "3.7.1 二维ALE的基本原理": "3.7.1 二维ALE方法与变量组合选择",
    "3.7.2 二维ALE曲面分析": "3.7.2 TabICLv2二阶交互效应",
    "3.7.3 联合取值覆盖与解释边界": "3.7.3 XGBoost二阶交互效应",
    "3.8 SHAP、ALE与来源消融的综合对照": "3.8 XGBoost基准模型十项任务预测结果",
    "3.9 本章小结": "3.9 TabICLv2与XGBoost预测及解释结果比较",
    "4.1 XGBoost基准模型预测结果": "4.1 新型表格模型与基准模型的比较框架",
    "4.2 TabICLv2与XGBoost预测效果比较": "4.2 TabICLv2在十项水质任务中的预测表现",
    "4.3 不同验证场景下的模型表现": "4.3 随机、年份与站点留出验证比较",
    "4.4 TabM、TabPFN与TabICLv2的代表性对比": "4.4 空间泛化稳定性与高误差站点分析",
    "4.5 本章小结": "4.5 新型表格模型的适用范围与局限",
    "5.6 本章小结": "5.8 环境解释、空间分区与泛化表现的综合讨论",
}


def insert_before(paragraph: Paragraph, text: str, style: str) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addprevious(new_p)
    new = Paragraph(new_p, paragraph._parent)
    new.style = style
    new.add_run(text)
    return new


def insert_after(paragraph: Paragraph, text: str, style: str) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new = Paragraph(new_p, paragraph._parent)
    new.style = style
    new.add_run(text)
    return new


def find(doc: Document, exact: str) -> Paragraph:
    for p in doc.paragraphs:
        if p.text.strip() == exact:
            return p
    raise KeyError(exact)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, OUT)
    doc = Document(OUT)
    before = {"paragraphs": len(doc.paragraphs), "tables": len(doc.tables), "figures": len(doc.inline_shapes)}
    for p in doc.paragraphs:
        key = p.text.strip()
        if key in RENAME:
            p.text = RENAME[key]

    insert_after(find(doc, "3.1 预测模型概述"), "本节概述XGBoost、TabM与TabPFN的基本定位，并将TabICLv2作为本文新型表格基础模型的核心研究对象。XGBoost承担统一基准与树模型解释对照，TabM和TabPFN仅用于四个代表任务的现代基线比较。", "Normal")
    p = find(doc, "3.2.3 特征编码与集成推理")
    p = insert_after(p, "3.2.4 预训练权重、随机性与固定上下文", "Heading 3")
    insert_after(p, "本文固定每折训练样本、训练标签、77特征顺序、逐折median填补器、随机种子42和预训练权重。默认8成员集成的多视图预测取均值；解释计算只扰动查询特征，不重新抽取上下文。", "Normal")
    p = find(doc, "3.3.5 复现性控制与数据泄漏防范")
    p = insert_after(p, "3.3.6 分海区与来源消融实验设计", "Heading 3")
    insert_after(p, "分海区与来源级消融均读取既有冻结实验结果，不在论文整合阶段重新训练。", "Normal")
    p = find(doc, "3.4 模型性能与空间泛化评价指标")
    for title in ["3.4.1 决定系数、RMSE与MAE", "3.4.2 Nash-Sutcliffe效率系数", "3.4.3 pooled OOF与折间统计", "3.4.4 逐站点误差与改善比例", "3.4.5 站点重复抽样与稳定性评价"]:
        p = insert_after(p, title, "Heading 3")
    p = find(doc, "3.5.3 XGBoost TreeSHAP特征贡献分析")
    for title in ["3.5.4 TabICLv2与XGBoost主要环境变量比较", "3.5.5 特征贡献的折间稳定性与解释边界"]:
        p = insert_after(p, title, "Heading 3")
    insert_after(p, "TabICLv2仅完成Chl-a surface fold_1试点：正式预测达到严格逐样本复现，77变量置换重要性可计算；测试标签仅用于事后模型依赖诊断，不参与模型选择。其余折次因单折计算代价和二维联合覆盖不足暂未扩展，故本稿不写入跨折结论。", "Normal")
    p = find(doc, "3.6.3 XGBoost环境变量响应曲线")
    for title in ["3.6.4 两种模型的变量响应差异", "3.6.5 一维ALE样本覆盖与稳定性"]:
        p = insert_after(p, title, "Heading 3")
    p = find(doc, "3.7.3 XGBoost二阶交互效应")
    for title in ["3.7.4 不同模型交互效应比较", "3.7.5 二维ALE联合覆盖及解释局限"]:
        p = insert_after(p, title, "Heading 3")
    p = find(doc, "3.9 TabICLv2与XGBoost预测及解释结果比较")
    p = insert_after(p, "本节仅比较已具备共同变量和共同支持范围的结果。置换重要性与TreeSHAP量纲不同，仅比较排名和来源组，不直接比较原始数值。", "Normal")
    p = insert_after(p, "3.10 本章小结", "Heading 2")
    insert_after(p, "本章形成TabICLv2为重点、XGBoost为对照的双分支解释框架。XGBoost解释已完成四任务五折闭环；TabICLv2当前为严格复现后的单折试点，后续扩展必须继续遵守固定上下文和覆盖约束。", "Normal")
    p = find(doc, "4.5 新型表格模型的适用范围与局限")
    p = insert_after(p, "4.6 本章小结", "Heading 2")
    insert_after(p, "本章围绕新型表格模型的十任务预测效果、不同验证场景和未见站点误差展开，不将四个代表任务外推为全部任务的统一结论。", "Normal")
    p = find(doc, "5.8 环境解释、空间分区与泛化表现的综合讨论")
    insert_before(p, "5.6 不同任务环境依赖差异", "Heading 2")
    insert_before(p, "5.7 解释结果与高误差站点的对应关系", "Heading 2")
    p = insert_after(p, "5.9 本章小结", "Heading 2")
    insert_after(p, "本章综合来源消融、空间分区和第三章解释结果，并严格区分模型统计依赖与真实环境因果机制。", "Normal")

    doc.core_properties.comments = "TabICLv2为解释重点、XGBoost为对照；保留既有17张XGBoost解释图。"
    doc.save(OUT)
    after = {"paragraphs": len(doc.paragraphs), "tables": len(doc.tables), "figures": len(doc.inline_shapes)}
    if after["figures"] != 27:
        raise RuntimeError(f"图件数量变化，期望27，实际{after['figures']}")
    report = {"source": str(SOURCE), "output": str(OUT), "before": before, "after": after, "xgb_explanation_figures_preserved": 17}
    (OUT_DIR / "Word结构核验.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"view": "Clean View", "path": str(OUT)}, {"view": "Diff View", "path": str(RUN / "reports/论文大纲修改差异.md")}, {"view": "Review View", "path": str(RUN / "reports/论文大纲修改审阅表.csv")}]).to_csv(OUT_DIR / "修改追踪索引.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
