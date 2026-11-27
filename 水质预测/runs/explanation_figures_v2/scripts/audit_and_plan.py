from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(r"E:\研究生\codex\水质预测")
SRC = PROJECT / "runs" / "tabicl_v2_slim_explainability_v2_all10"
OUT = PROJECT / "runs" / "explanation_figures_v2"
META = OUT / "metadata"
SHAP = SRC / "shap"
ALE = SRC / "ale_1d"
GATES = SRC / "gates"
ALE2D = SRC / "ale_2d"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_status(path: Path, required: list[str]) -> tuple[bool, str, list[str]]:
    try:
        df = pd.read_csv(path)
        missing = [c for c in required if c not in df.columns]
        return not missing, f"rows={len(df)},cols={len(df.columns)}", missing
    except Exception as exc:
        return False, f"read_error={type(exc).__name__}:{exc}", required


def main() -> None:
    META.mkdir(parents=True, exist_ok=True)
    shap_files = sorted(SHAP.glob("**/*.csv.gz"))
    ale_curves = sorted(ALE.glob("*_curve.csv"))
    ale_support = sorted(ALE.glob("*_bin_support.csv"))
    ale_queries = sorted(ALE.glob("*_query_manifest.csv"))
    ale_repeats = sorted(ALE.glob("*_repeatability.csv"))
    p1_probes = sorted(GATES.glob("*_p1_probes.csv"))
    p1_decisions = sorted(GATES.glob("*_p1_decision.csv"))

    shap_required = ["task", "fold", "obs_id", "station_id", "feature_order", "feature", "feature_value", "shap_value", "base_value", "model_prediction", "status"]
    shap_rows = []
    shap_task_counts = {}
    shap_bad = []
    for path in shap_files:
        ok, detail, missing = csv_status(path, shap_required)
        row_count = None
        if ok:
            df = pd.read_csv(path, compression="gzip")
            row_count = len(df)
            task = str(df["task"].iloc[0])
            shap_task_counts[task] = shap_task_counts.get(task, 0) + 1
            if row_count != 77 or df["feature_order"].tolist() != list(range(1, 78)):
                ok = False
                detail += "; feature_order_not_77_ordered"
            if df["obs_id"].nunique() != 1 or df["status"].nunique() != 1:
                ok = False
                detail += "; sample_or_status_not_unique"
            if df[["feature_value", "shap_value"]].isna().any().any():
                ok = False
                detail += "; feature_or_shap_nan"
        else:
            task = path.parent.name
            shap_bad.append(str(path))
        shap_rows.append({
            "asset_type": "SHAP_GROUP",
            "relative_path": path.relative_to(PROJECT).as_posix(),
            "task": task,
            "rows": row_count,
            "schema_ok": ok,
            "detail": detail,
            "missing_fields": ";".join(missing),
            "sha256": sha256(path),
        })

    ale_required = ["task", "fold", "feature", "bin", "lower", "upper", "n_samples", "n_stations", "local_effect", "ale", "status", "repeatability_max_local_delta"]
    ale_rows = []
    ale_bad = []
    ale_bin_counts = {}
    ale_sensitive = []
    for path in ale_curves:
        ok, detail, missing = csv_status(path, ale_required)
        task = path.stem.removesuffix("_curve")
        n_bins = None
        max_delta = None
        status = ""
        if ok:
            df = pd.read_csv(path)
            n_bins = len(df)
            ale_bin_counts[task] = n_bins
            max_delta = float(df["repeatability_max_local_delta"].max())
            status = ";".join(sorted(df["status"].astype(str).unique()))
            if df[["lower", "upper", "n_samples", "n_stations", "local_effect", "ale"]].isna().any().any():
                ok = False
                detail += "; numeric_support_fields_nan"
            if max_delta > 1e-6:
                ale_sensitive.append(task)
        else:
            ale_bad.append(str(path))
        ale_rows.append({
            "asset_type": "ALE1D_CURVE",
            "relative_path": path.relative_to(PROJECT).as_posix(),
            "task_feature": task,
            "rows_bins": n_bins,
            "schema_ok": ok,
            "detail": detail,
            "missing_fields": ";".join(missing),
            "status": status,
            "max_repeatability_local_delta": max_delta,
            "sha256": sha256(path),
        })

    shap_df = pd.DataFrame(shap_rows)
    ale_df = pd.DataFrame(ale_rows)
    shap_df.to_csv(META / "asset_audit_shap.csv", index=False, encoding="utf-8-sig")
    ale_df.to_csv(META / "asset_audit_ale_1d.csv", index=False, encoding="utf-8-sig")

    mapping_src = PROJECT / "runs" / "explanation_figures_v1" / "metadata" / "feature_display_name_mapping.csv"
    mapping_dst = META / "feature_display_name_mapping.csv"
    if mapping_src.exists():
        shutil.copy2(mapping_src, mapping_dst)
        mapping_state = f"存在，77行，SHA256={sha256(mapping_dst)}"
    else:
        mapping_state = "缺失；不能在本轮伪造publication display names"

    p1_required = ["task", "obs_id", "repeat", "p1_prediction", "frozen_batch_prediction", "absolute_difference"]
    p1_ok = []
    p1_max_range = []
    for path in p1_probes:
        ok, detail, missing = csv_status(path, p1_required)
        if ok:
            df = pd.read_csv(path)
            p1_max_range.append(float(df.groupby("obs_id")["p1_prediction"].agg(lambda x: x.max() - x.min()).max()))
        p1_ok.append({"path": path.relative_to(PROJECT).as_posix(), "schema_ok": ok, "detail": detail, "missing_fields": ";".join(missing)})
    p1_decision_rows = []
    for path in p1_decisions:
        df = pd.read_csv(path)
        p1_decision_rows.extend(df.to_dict("records"))
    p1_decision_df = pd.DataFrame(p1_decision_rows)
    pd.DataFrame(p1_ok).to_csv(META / "asset_audit_p1_probes.csv", index=False, encoding="utf-8-sig")
    if not p1_decision_df.empty:
        cols = [c for c in ["task", "status", "n_queries", "actual_model_calls", "max_p1_vs_frozen_abs", "max_p1_repeat_range", "reason"] if c in p1_decision_df.columns]
        p1_decision_df[cols].to_csv(META / "asset_audit_p1_decisions.csv", index=False, encoding="utf-8-sig")

    ale2d_rows = []
    decision_path = ALE2D / "candidate_pair_decisions.csv"
    if decision_path.exists():
        decision = pd.read_csv(decision_path)
        for _, r in decision.iterrows():
            support_path = ALE2D / f"{r['task']}_{r['feature_1']}_{r['feature_2']}_joint_support.csv"
            support_ok, support_detail, support_missing = csv_status(support_path, ["cell_1", "cell_2", "n_obs", "n_stations", "supported"])
            ale2d_rows.append({
                "task": r["task"], "feature_1": r["feature_1"], "feature_2": r["feature_2"],
                "status": r["status"], "supported_cells": r["supported_cells"], "total_cells": r["total_cells"],
                "joint_observations": r["joint_observations"], "support_file": support_path.relative_to(PROJECT).as_posix(),
                "support_schema_ok": support_ok, "support_detail": support_detail,
                "missing_fields": ";".join(support_missing), "response_surface_computed": False,
            })
    pd.DataFrame(ale2d_rows).to_csv(META / "asset_audit_ale_2d.csv", index=False, encoding="utf-8-sig")

    # Exact planned waterfall rule: one CORE_3 median prediction per task.
    sample_manifest = SRC / "01_preregistered_shap_samples.csv"
    if sample_manifest.exists():
        sm = pd.read_csv(sample_manifest)
        waterfall = sm[(sm["phase"] == "CORE_3") & (sm["target_quantile"] == 0.5)]
        waterfall_rule = f"CORE_3 且 target_quantile=0.5；{len(waterfall)}个任务样本；不按图形美观人工挑选"
        waterfall_ids = ";".join(waterfall["obs_id"].astype(str))
    else:
        waterfall_rule = "预登记样本清单缺失"
        waterfall_ids = ""

    # Audit/plan rows. This stage intentionally writes no figures.
    rows = [
        ("GLOBAL-01", "跨任务SHAP重要性热图", "SHAP全局解释", "heatmap", "40组SHAP；feature；shap_value；feature_display_name_mapping", "具备", "", "是", 1, "第二批", "COMPUTED_RESULT_FIGURE", "Top 20；任务/水质family分区"),
        ("GLOBAL-02", "10任务特征排名图", "SHAP全局解释", "rank_plot", "40组SHAP；task；feature；shap_value；feature display names", "具备", "", "是", 10, "第二批", "COMPUTED_RESULT_FIGURE", "每个任务独立一图，避免过度压缩"),
        ("GLOBAL-03", "特征排名变化图", "SHAP全局解释", "rank_change", "40组SHAP；task；feature；rank", "具备，可由已存SHAP聚合", "预先固定聚合定义需写入脚本", "是", 1, "第二批", "COMPUTED_RESULT_FIGURE", "不重新计算SHAP"),
        ("GLOBAL-04", "特征来源组贡献比较图", "SHAP全局解释", "source_group_comparison", "40组SHAP；feature；shap_value；source_group", "具备", "", "是", 1, "第二批", "COMPUTED_RESULT_FIGURE", "仅作来源组聚合比较"),
        ("DIST-01", "SHAP Beeswarm", "SHAP分布解释", "beeswarm", "逐样本feature_value与shap_value；task；feature", "具备，40组原始逐样本结果", "同一任务内需固定样本聚合规则", "是", 10, "第二批", "COMPUTED_RESULT_FIGURE", "每个任务单独一图"),
        ("DIST-02", "SHAP Violin", "SHAP分布解释", "violin", "逐样本shap_value；task；feature", "具备，40组原始逐样本结果", "同一任务内需固定样本聚合规则", "是", 10, "第二批", "COMPUTED_RESULT_FIGURE", "与Beeswarm回答不同的分布问题"),
        ("LOCAL-01", "10任务典型样本SHAP Waterfall", "SHAP局部解释", "waterfall", "40组SHAP；obs_id；feature_value；shap_value；model_prediction；预登记样本", "具备", "", "是", 10, "第三批", "COMPUTED_RESULT_FIGURE", waterfall_rule),
        ("ALE-01", "10任务ALE 1D响应图", "ALE非线性响应", "ale_1d", "20条ALE curve；bin边界；ale；n_samples；n_stations", "具备", "secchi_ssrd_sum_3d为6 bins，其余19条为10 bins", "是", 10, "第四批", "COMPUTED_RESULT_FIGURE", "每张展示该任务已完成的2个特征；不补算缺失曲线"),
        ("SUPPORT-01", "ALE各bin样本数热图", "ALE支持度", "support_heatmap_n_samples", "20条ALE curve；bin；n_samples", "具备", "曲线bin数不完全一致", "是", 1, "第五批", "SUPPORT_AUDIT_FIGURE", "空白格不代表零样本，表示该曲线没有该bin"),
        ("SUPPORT-02", "ALE各bin独立站点数热图", "ALE支持度", "support_heatmap_n_stations", "20条ALE curve；bin；n_stations", "具备", "曲线bin数不完全一致", "是", 1, "第五批", "SUPPORT_AUDIT_FIGURE", "支持审查，不是模型效应图"),
        ("STABILITY-01", "P1重复预测点区间图", "稳定性与质量控制", "prediction_interval", "10个*_p1_probes.csv；repeat；p1_prediction", "具备部分字段，需明确区间定义", "当前为3次重复点探针，不是全样本预测区间；必须在图注中限定含义", "条件允许", 1, "第五批", "NUMERICAL_STABILITY_FIGURE", "不把单次frozen prediction误称为区间"),
        ("STABILITY-02", "数值抖动与解释效应尺度对比图", "稳定性与质量控制", "jitter_vs_effect", "ALE repeatability_max_local_delta；8组SHAP overlap comparison；P1 absolute_difference", "具备部分字段", "跨设备SHAP比较仅8组；SMALL_EFFECT阈值尚未冻结", "条件允许", 1, "第五批", "NUMERICAL_STABILITY_FIGURE", "明确partial coverage；敏感曲线保留标记"),
        ("ALE2D-01", "TN surface联合支持矩阵", "ALE 2D支持审查", "4x4_support_matrix", "candidate_pair_decisions；联合支持矩阵文件", "具备支持审查数据", "response surface未计算", "是", 1, "第五批", "SUPPORT_AUDIT_FIGURE", "13/16格通过；ALE_2D_BLOCKED_BY_SUPPORT"),
        ("ALE2D-02", "TN bottom联合支持矩阵", "ALE 2D支持审查", "4x4_support_matrix", "candidate_pair_decisions；联合支持矩阵文件", "具备支持审查数据", "response surface未计算", "是", 1, "第五批", "SUPPORT_AUDIT_FIGURE", "7/16格通过；ALE_2D_BLOCKED_BY_SUPPORT"),
        ("ALE2D-03", "TP bottom联合支持矩阵", "ALE 2D支持审查", "4x4_support_matrix", "candidate_pair_decisions；联合支持矩阵文件", "具备支持审查数据", "response surface未计算", "是", 1, "第五批", "SUPPORT_AUDIT_FIGURE", "13/16格通过；ALE_2D_BLOCKED_BY_SUPPORT"),
    ]
    audit = pd.DataFrame(rows, columns=["figure_id", "figure_title", "category", "figure_type", "required_data_fields", "current_available", "missing_or_limitations", "allowed_to_generate", "expected_count", "planned_batch", "classification", "notes"])
    audit.to_csv(META / "figure_data_availability_audit.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(META / "figure_inventory.csv", index=False, encoding="utf-8-sig")

    p1_status = f"具备{len(p1_probes)}个任务的3次重复探针和P1决策字段；缺少chla_surface；图件只能表达探针重复范围，不代表全样本预测置信区间。"
    sensitive = ", ".join(ale_sensitive) if ale_sensitive else "无"
    summary = [
        "# 多样化学位论文图件 V2：数据资产审计与图件规划",
        "",
        "> 本轮仅完成资产审计和规划，不生成图件，不运行模型，不重新计算SHAP/ALE。已有 MAIN-01～03 V1/V2 未被覆盖。",
        "",
        "## 审计摘要",
        f"- SHAP：{len(shap_files)}组，10个任务各4组；每组应为77行、逐特征值和逐特征SHAP值。合格文件：{int(shap_df['schema_ok'].sum())}/{len(shap_df)}。",
        f"- ALE 1D：{len(ale_curves)}条曲线；{sum(v == 10 for v in ale_bin_counts.values())}条为10 bins，{sum(v != 10 for v in ale_bin_counts.values())}条存在自适应bin数。数值敏感候选：{sensitive}。",
        f"- P1：{len(p1_probes)}个任务探针文件、{len(p1_decisions)}个决策文件。{p1_status}",
        "- ALE 2D：3个候选均只有联合支持审查，无response surface；分别为TN surface 13/16、TN bottom 7/16、TP bottom 13/16，状态为ALE_2D_BLOCKED_BY_SUPPORT。",
        f"- publication feature mapping：{mapping_state}。",
        "",
        "## 图件分类边界",
        "- COMPUTED_RESULT_FIGURE：由已完成的SHAP/ALE数值结果直接组织，不改变模型结果。",
        "- SUPPORT_AUDIT_FIGURE：只表达样本数、独立站点数和预登记支持状态，不称为模型解释结果。",
        "- NUMERICAL_STABILITY_FIGURE：只表达P1重复探针、ALE重复性或已有跨运行比较；coverage不完整时必须标注partial。",
        "",
        "## 可复现样本规则",
        f"- Waterfall固定选择：{waterfall_rule}。样本ID已在审计脚本运行时读取，未按图形美观人工选择。",
        "- 10任务图件按任务单独保存；不为压缩数量而把不同科学问题强行合图。",
        "",
        "## 分批执行计划",
        "1. 第一批：保留已有MAIN-01～03 V1/V2，不覆盖。",
        "2. 第二批：SHAP全局与分布解释（热图、任务独立排名、排名变化、来源组、Beeswarm、Violin）。",
        "3. 第三批：10个任务的预登记典型样本Waterfall。",
        "4. 第四批：10个任务的ALE 1D图，每张展示该任务已完成的两个特征。",
        "5. 第五批：ALE支持度、数值稳定性和3个ALE 2D联合支持审查图。",
        "",
        "## 当前阻塞/需在出图前固定的事项",
        "- SMALL_EFFECT_INTERPRET_WITH_CAUTION 的数值阈值尚未在本轮重新规定；生成前应读取历史协议或明确登记阈值，不能临时从图形中推断。",
        "- P1图件的语义限定为重复探针范围，不得写成预测置信区间。",
        "- ALE 2D只能画支持矩阵，禁止生成或暗示响应曲面。",
    ]
    (OUT / "figure_planning.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (OUT / "figure_generation_report.md").write_text("# 图件生成报告\n\n本轮状态：PLANNING_ONLY。已完成数据资产审计和图件规划；尚未生成任何新增PNG/PDF，也未运行模型或重新计算SHAP/ALE。\n\n已有 MAIN-01～03 V1/V2 保持原位置和原文件不变。\n", encoding="utf-8")
    (OUT / "figure_quality_audit.md").write_text("# 图件质量审计\n\n本轮状态：待批量出图。\n\n已通过：原始SHAP/ALE字段存在性、文件数量、ALE支持字段、P1探针字段和ALE 2D候选状态的静态审计。\n\n待出图后检查：PNG 600 dpi、PDF矢量、字体、单位、变量显示名、图题/图注、支持审查与模型结果的分类边界。\n", encoding="utf-8")
    (OUT / "figure_caption_catalog.md").write_text("# 图题与图注目录（规划版）\n\n本轮先登记图题主题，正式图注将在对应批次生成时写入，且必须列出数据文件、任务范围和支持/稳定性限制。\n\n" + "\n".join(f"- {r[0]}：{r[1]}（{r[10]}）" for r in rows) + "\n", encoding="utf-8")

    inventory_lines = ["relative_path,sha256,role"]
    for path in sorted((META).glob("*")):
        if path.is_file():
            inventory_lines.append(f"{path.relative_to(PROJECT).as_posix()},{sha256(path)},audit_or_plan")
    (META / "audit_plan_sha256_manifest.csv").write_text("\n".join(inventory_lines) + "\n", encoding="utf-8")

    print(f"SHAP_FILES={len(shap_files)} SHAP_OK={int(shap_df['schema_ok'].sum())}")
    print(f"ALE_CURVES={len(ale_curves)} ALE_SUPPORT={len(ale_support)} ALE_QUERY={len(ale_queries)} ALE_REPEAT={len(ale_repeats)}")
    print(f"P1_PROBES={len(p1_probes)} P1_DECISIONS={len(p1_decisions)}")
    print(f"FIGURE_PLANS={len(audit)} WATERFALL_RULE={waterfall_rule}")


if __name__ == "__main__":
    main()
