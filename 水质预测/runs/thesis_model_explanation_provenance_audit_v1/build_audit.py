"""只读溯源：用 DOCX 图片字节哈希和冻结图件目录建立逐图映射。"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from docx import Document

ROOT = Path(r"E:\研究生\codex\水质预测")
OUT = ROOT / "runs/thesis_model_explanation_provenance_audit_v1"
DOCX = ROOT / "文档/论文材料/硕士论文TabICLv2_SHAP_ALE_插图整合初稿_20261031_v4/水质预测硕士学位论文_TabICLv2_SHAP_ALE_插图整合初稿.docx"
XGB = ROOT / "runs/xgb_shap_ale_v2"
CLOSURE = ROOT / "runs/tabicl_v2_explainability_closure_v1"
V3 = ROOT / "runs/thesis_figure_production_v3"
P1 = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
TASKS = ["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def textpath(path: Path) -> str:
    return str(path)


def task_from_name(name: str) -> str:
    for task in sorted(TASKS, key=len, reverse=True):
        if task in name:
            return task
    return "跨任务汇总（四项或十项，见图题）"


def source_data(path: Path, number: str) -> str:
    name = path.stem
    task = task_from_name(name)
    if XGB in path.parents:
        rows = list(csv.DictReader((XGB / "xgb_shap_ale_v2_figure_inventory.csv").open(encoding="utf-8-sig", newline="")))
        hit = next((r for r in rows if Path(r["png_path"]).name == path.name), None)
        return str(XGB / hit["data_path"]) if hit else "原始数据路径待核"
    if CLOSURE in path.parents:
        if "shap_importance" in name:
            return textpath(CLOSURE / "04_shap_importance_summary.csv")
        if "shap_rank" in name:
            return textpath(CLOSURE / "05_shap_rank_stability.csv")
        return textpath(ROOT / "runs/tabicl_v2_explainability_v3/ale_1d/ale_1d_by_fold.csv")
    if V3 in path.parents:
        if "global_02" in name or "dist_01" in name or "local_01" in name:
            return textpath(P1 / "shap" / task / "*.csv.gz")
        if "global_03" in name or "global_04" in name:
            return textpath(P1 / "shap/**/*.csv.gz")
        if "ale_01" in name:
            return textpath(P1 / "ale_1d" / (name.removeprefix("ale_01_") + "_curve.csv"))
        if "ale2d_" in name:
            joint = next((P1 / "ale_2d").glob(task + "_*_joint_support.csv"), None)
            return textpath(P1 / "ale_2d/candidate_pair_decisions.csv") + "；" + (textpath(joint) if joint else "联合支持文件缺失")
        if "support_" in name:
            return textpath(P1 / "ale_1d/*_curve.csv")
        if "stability_01" in name:
            return textpath(P1 / "gates/*_p1_probes.csv")
        if "stability_02" in name:
            return textpath(ROOT / "runs/tabicl_explanation_numerical_stability_v2/jitter_vs_explanation_scale.csv")
    return "未定位"


def classify(path: Path, number: str) -> tuple[str, str, str, str, str]:
    name = path.stem
    task = task_from_name(name)
    if XGB in path.parents:
        kind = "XGBoost受控复现模型；图3-60另含多源消融结果" if number == "图3-60" else "XGBoost受控复现模型"
        protocol = "XGBoost原生TreeSHAP；log1p目标尺度" if "shap" in name or "integrated" in name else "XGBoost受控复现；ALE原始目标尺度"
        if "ale2d_coverage" in name:
            protocol = "XGBoost二维ALE联合支持审查；非单独响应曲面"
        return kind, task, "四任务各fold_1–fold_5", protocol, "A：DOCX图片SHA256一致＋脚本与图件清单"
    if CLOSURE in path.parents:
        protocol = "TabICLv2模型无关Permutation SHAP；固定256批查询包装；原始目标尺度" if "shap" in name else "TabICLv2历史批查询ALE；原始目标尺度；部分折严格复现未通过"
        return "TabICLv2", task, "四任务各fold_1–fold_5", protocol, "A：DOCX图片SHA256一致＋原始脚本与闭环报告"
    if V3 in path.parents:
        if "ale2d_" in name:
            protocol = "P1单查询；仅4×4联合支持审查，未计算二维响应曲面"
            kind = "TabICLv2候选预测函数（支持审查，非解释响应）"
        elif "support_" in name:
            protocol = "P1单查询ALE分箱支持度，非模型响应图"
            kind = "TabICLv2"
        elif "stability_01" in name:
            protocol = "P1三次重复探针；九任务；非预测置信区间"
            kind = "TabICLv2"
        elif "stability_02" in name:
            protocol = "历史TabICLv2数值抖动与解释效应尺度；非新增模型计算"
            kind = "TabICLv2"
        elif "ale_01" in name:
            protocol = "P1单查询ALE；fold_1；原始目标尺度"
            kind = "TabICLv2"
        else:
            protocol = "P1单查询Permutation SHAP；fold_1；原始目标尺度"
            kind = "TabICLv2"
        return kind, task, "fold_1（稳定性图见协议）", protocol, "A：DOCX图片SHA256一致＋绘图脚本及原始结果"
    return "未确认", task, "未确认", "未确认", "C：需要核查"


def build_mapping() -> list[dict]:
    doc = Document(DOCX)
    index: dict[str, Path] = {}
    for base in (XGB / "figures", CLOSURE / "figures", V3 / "figures"):
        for png in base.rglob("*.png"):
            index[sha(png)] = png
    rows = []
    for i, par in enumerate(doc.paragraphs):
        cap = par.text.strip()
        match = re.match(r"^(图(?:3-\d+|C-\d+))\s+", cap)
        if not match or i < 250:
            continue
        image_part = None
        for previous in reversed(doc.paragraphs[max(0, i - 5):i]):
            blips = previous._p.xpath(".//a:blip")
            if blips:
                image_part = doc.part.related_parts[blips[-1].get(REL)]
                break
        if image_part is None:
            raise RuntimeError(f"图题没有邻近图片：{cap}")
        im_hash = hashlib.sha256(image_part.blob).hexdigest()
        src = index.get(im_hash)
        if src is None:
            raise RuntimeError(f"图片未在冻结图件目录匹配：{cap} {im_hash}")
        number = match.group(1)
        model, task, fold, protocol, grade = classify(src, number)
        rows.append({"图号": number, "图件内容": cap[len(number):].strip(), "实际解释模型": model,
                     "任务": task, "折次": fold, "对应结果文件": source_data(src, number), "证据等级": grade,
                     "图片原始路径": textpath(src), "DOCX媒体路径": str(image_part.partname),
                     "图片SHA256": im_hash, "预测/解释协议与尺度": protocol})
    with (OUT / "02_figure_model_mapping.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def build_inventory(rows: list[dict]) -> None:
    paths = {DOCX}
    fixed = [
        "脚本/05_诊断分析/58_compute_xgb_shap_ale_v2.py",
        "脚本/05_诊断分析/59_integrate_plot_xgb_shap_ale_v2.py",
        "脚本/05_诊断分析/51_summarize_multisource_ablation_v1.py",
        "脚本/05_诊断分析/55_audit_xgb_interpretation_refit_v1.py",
        "脚本/05_诊断分析/56_refit_frozen_xgb_for_interpretation.py",
        "脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py",
        "脚本/05_诊断分析/67_run_tabiclv2_model_agnostic_shap_v3.py",
        "脚本/05_诊断分析/69_run_tabiclv2_ale_v3.py",
        "脚本/05_诊断分析/88_run_tabicl_slim_explainability_v1.py",
        "脚本/05_诊断分析/99_compute_tabicl_all10_shap_v2.py",
        "脚本/05_诊断分析/100_compute_tabicl_all10_ale_1d_v2.py",
        "脚本/06_结果汇总与绘图/72_finalize_tabiclv2_explanation_closure_v1.py",
        "models/tabiclv2_adapter.py",
        "runs/thesis_figure_production_v3/scripts/produce_v3_figures.py",
        "runs/xgb_shap_ale_v1/xgb_shap_ale_v1_decision.md",
        "runs/xgb_interpretation_refit_v1/06_xgb_refit_final_decision.md",
        "runs/xgb_shap_ale_v2/xgb_shap_ale_v2_final_decision.md",
        "runs/xgb_shap_ale_v2/xgb_shap_ale_v2_figure_inventory.csv",
        "runs/xgb_interpretation_refit_v1/01_xgb_refit_run_registry.csv",
        "runs/xgb_interpretation_refit_v1/03_xgb_refit_model_manifest.csv",
        "runs/multisource_ablation_v1/multisource_ablation_metrics.csv",
        "runs/tabicl_v2_explainability_closure_v1/12_final_decision.md",
        "runs/tabicl_v2_explainability_closure_v1/09_thesis_figure_inventory.csv",
        "runs/tabicl_v2_explainability_v3/02_tabicl_shap_values_manifest.csv",
        "runs/tabicl_v2_explanation_stability_v1/08_final_explanation_stability_decision.md",
        "runs/tabicl_explanation_numerical_stability_v2/numerical_stability_decision.md",
        "runs/tabicl_v2_slim_explainability_v2_all10/00_实验协议说明.md",
        "runs/tabicl_v2_slim_explainability_v2_all10/run_registry.csv",
        "runs/tabicl_v2_slim_explainability_v2_all10/ale_2d/candidate_pair_decisions.csv",
        "transfer/model_runtime_patch_v1/checkpoint_inventory.csv",
        "transfer/model_runtime_patch_v1/checkpoints/tabicl-regressor-v2-20260212.ckpt",
        "runs/tabicl_v2_explainability_closure_v1/04_shap_importance_summary.csv",
        "runs/tabicl_v2_explainability_closure_v1/05_shap_rank_stability.csv",
        "runs/tabicl_v2_explainability_closure_v1/07_ale_2d_reliability_review.csv",
        "runs/tabicl_v2_explainability_v3/ale_1d/ale_1d_by_fold.csv",
        "runs/tabicl_v2_explainability_v3/ale_2d/ale_2d_by_fold.csv",
        "runs/tabicl_v2_explainability_v3/ale_2d/joint_coverage.csv",
        "runs/tabicl_explanation_numerical_stability_v2/jitter_vs_explanation_scale.csv",
    ]
    paths.update(ROOT / x for x in fixed)
    for row in rows:
        paths.add(Path(row["图片原始路径"]))
        for value in row["对应结果文件"].split("；"):
            p = Path(value)
            if "*" in value:
                paths.update(p.parent.glob(p.name))
            else:
                paths.add(p)
    paths.update((ROOT / "runs/xgb_interpretation_refit_v1/artifacts").rglob("xgboost_refit_model.ubj"))
    paths.update((ROOT / "runs/xgb_interpretation_refit_v1/artifacts").rglob("median_imputer.joblib"))
    paths.update((XGB / "01_shap/raw").glob("*.csv.gz"))
    paths.update((XGB / "registry").glob("*"))
    paths.update((P1 / "shap").rglob("*.csv.gz"))
    paths.update((ROOT / "runs/tabicl_v2_explainability_v3/shap_values").rglob("*.csv.gz"))
    paths.update((P1 / "gates").glob("*_p1_decision.csv"))
    paths.update((P1 / "logs").glob("*.log"))
    paths.update((ROOT / "runs/xgb_interpretation_refit_v1/logs").rglob("*.log"))
    records = []
    for path in sorted(paths, key=str):
        if not path.is_file():
            records.append({"资产路径": str(path), "资产类别": "缺失/需核", "字节数": "", "SHA256": "", "备注": "路径不存在，未以猜测替代"})
            continue
        kind = "图件" if path.suffix == ".png" else "模型资产" if path.suffix in (".ubj", ".ckpt", ".joblib") else "脚本" if path.suffix == ".py" else "论文" if path.suffix == ".docx" else "记录/结果"
        records.append({"资产路径": str(path), "资产类别": kind, "字节数": path.stat().st_size, "SHA256": sha(path), "备注": "只读计算校验值"})
    with (OUT / "03_artifact_evidence_inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(f"figures={len(rows)} inventory={len(records)} missing={sum(r['资产类别']=='缺失/需核' for r in records)}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_inventory(build_mapping())
