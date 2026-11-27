from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "文档" / "主线接管_20260521"
OUTDIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def fmt(x: str) -> str:
    return f"{float(x):.3f}"


def build_reuse_inventory() -> None:
    fieldnames = [
        "item_name",
        "item_type",
        "source_project_path",
        "mainline_target_path",
        "category",
        "validated",
        "ab_tested",
        "status",
        "can_reuse_directly",
        "needs_adaptation",
        "recommended_action",
        "related_target",
        "related_layer",
        "expected_mainline_value",
        "risk",
        "note",
    ]
    rows = [
        {
            "item_name": "current_best_scorecard",
            "item_type": "scorecard",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/current_best_scorecard.csv",
            "mainline_target_path": "文档/主线接管_20260521/主线接管报告.md",
            "category": "current_conclusion",
            "validated": "True",
            "ab_tested": "True",
            "status": "verified_useful",
            "can_reuse_directly": "True",
            "needs_adaptation": "False",
            "recommended_action": "作为 Chl-a/Secchi 历史对照证据保留，不覆盖主线 current best",
            "related_target": "chla/secchi",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "高",
            "risk": "低",
            "note": "源项目对 Chl-a 三层更强，但不能直接替换主线 current best",
        },
        {
            "item_name": "four_target_bootstrap_scorecard",
            "item_type": "scorecard",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/four_target_bootstrap_scorecard.csv",
            "mainline_target_path": "文档/主线接管_20260521/主线接管报告.md",
            "category": "current_conclusion",
            "validated": "True",
            "ab_tested": "True",
            "status": "verified_useful",
            "can_reuse_directly": "True",
            "needs_adaptation": "False",
            "recommended_action": "作为 12 任务统一比较基准保留",
            "related_target": "all",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "高",
            "risk": "低",
            "note": "可直接用于判断哪些试验成果值得并入主线",
        },
        {
            "item_name": "tn_tp_baseline_bootstrap_scorecard",
            "item_type": "scorecard",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/tn_tp_baseline_bootstrap_scorecard.csv",
            "mainline_target_path": "文档/主线接管_20260521/主线下一轮实验计划.md",
            "category": "ab_result",
            "validated": "True",
            "ab_tested": "True",
            "status": "verified_useful",
            "can_reuse_directly": "True",
            "needs_adaptation": "False",
            "recommended_action": "用于 TN/TP 模型家族与特征组复核，不直接替换主线结果",
            "related_target": "tn/tp",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "中",
            "risk": "低",
            "note": "当前主线除 TN surface 外普遍不弱于该分数卡",
        },
        {
            "item_name": "current_candidate_registry",
            "item_type": "registry",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/experiments_outputs_current/current_candidate_registry.csv",
            "mainline_target_path": "文档/主线接管_20260521/主线禁止重复清单.md",
            "category": "guardrail",
            "validated": "True",
            "ab_tested": "True",
            "status": "verified_useful",
            "can_reuse_directly": "True",
            "needs_adaptation": "False",
            "recommended_action": "作为主线候选准入黑白名单依据",
            "related_target": "all",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "高",
            "risk": "低",
            "note": "其中 eligible_mainline=False 的路线不要直接推进为主线",
        },
        {
            "item_name": "station_depth_features",
            "item_type": "feature_table",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/station_depth_features.csv",
            "mainline_target_path": "数据/处理后数据/衍生版本/station_depth_features.csv",
            "category": "static_spatial_feature",
            "validated": "True",
            "ab_tested": "True",
            "status": "local_positive_signal",
            "can_reuse_directly": "True",
            "needs_adaptation": "True",
            "recommended_action": "作为 station-level 静态协变量候选，单独做 controlled A/B",
            "related_target": "tn/chla",
            "related_layer": "bottom>middle>surface",
            "expected_mainline_value": "中",
            "risk": "中",
            "note": "源项目显示对 TN bottom 与 Chl-a bottom 有局部帮助，但非统一增益",
        },
        {
            "item_name": "coastal_distance_features",
            "item_type": "feature_table",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/coastal_distance_features_2014_2024.csv",
            "mainline_target_path": "数据/处理后数据/衍生版本/coastal_distance_features_2014_2024.csv",
            "category": "static_spatial_feature",
            "validated": "True",
            "ab_tested": "True",
            "status": "local_positive_signal",
            "can_reuse_directly": "True",
            "needs_adaptation": "True",
            "recommended_action": "保留为 observation-level 候选输入，避免直接并入正式主链",
            "related_target": "chla/tn",
            "related_layer": "bottom/surface",
            "expected_mainline_value": "中",
            "risk": "中",
            "note": "A/B 整体无 promote，但对 Chl-a bottom 与 TN surface 有微弱正信号",
        },
        {
            "item_name": "coastal_distance_station_features",
            "item_type": "feature_table",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/coastal_distance_station_features.csv",
            "mainline_target_path": "数据/处理后数据/衍生版本/coastal_distance_station_features.csv",
            "category": "static_spatial_feature",
            "validated": "True",
            "ab_tested": "False",
            "status": "verified_useful",
            "can_reuse_directly": "True",
            "needs_adaptation": "True",
            "recommended_action": "用于 station-level merge 与机制解释，不单独宣称有效",
            "related_target": "all",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "中",
            "risk": "低",
            "note": "来源清楚、无泄漏、可解释性强",
        },
        {
            "item_name": "distance_to_mouth_validation",
            "item_type": "source_validation",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/distance_to_mouth_station_validation.csv",
            "mainline_target_path": "文档/主线接管_20260521/主线接管优先级.md",
            "category": "source_validation",
            "validated": "True",
            "ab_tested": "True",
            "status": "verified_useful",
            "can_reuse_directly": "True",
            "needs_adaptation": "True",
            "recommended_action": "先复用 validation 结论，再按主线键构造正式 observation/station 特征表",
            "related_target": "tn/secchi/chla",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "高",
            "risk": "低",
            "note": "94/94 站点通过验证，但现阶段主要是验证表而非正式主线特征表",
        },
        {
            "item_name": "zone_refinement_validation",
            "item_type": "source_validation",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/zone_refinement_station_validation.csv",
            "mainline_target_path": "文档/主线接管_20260521/主线接管优先级.md",
            "category": "source_validation",
            "validated": "True",
            "ab_tested": "True",
            "status": "verified_useful",
            "can_reuse_directly": "True",
            "needs_adaptation": "True",
            "recommended_action": "复用 scheme4 结论，谨慎并入主线 controlled A/B",
            "related_target": "tn/tp/secchi",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "中",
            "risk": "中",
            "note": "source validation 很扎实，但统一 A/B 提升有限",
        },
        {
            "item_name": "enhanced_tide_proxy_features",
            "item_type": "feature_table",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/enhanced_tide_proxy_features_2014_2024.csv",
            "mainline_target_path": "数据/处理后数据/衍生版本/enhanced_tide_proxy_features_2014_2024.csv",
            "category": "process_feature",
            "validated": "True",
            "ab_tested": "True",
            "status": "local_positive_signal",
            "can_reuse_directly": "True",
            "needs_adaptation": "True",
            "recommended_action": "优先用于 Chl-a bottom 与 Secchi middle 的 controlled combination",
            "related_target": "chla/secchi",
            "related_layer": "bottom/middle",
            "expected_mainline_value": "中",
            "risk": "中",
            "note": "源项目曾促成 Chl-a bottom 与 Secchi middle 的 current best 更新",
        },
        {
            "item_name": "copernicus_physics_features",
            "item_type": "feature_table",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/copernicus_physics_features_2014_2024.csv",
            "mainline_target_path": "数据/处理后数据/衍生版本/copernicus_physics_features_2014_2024.csv",
            "category": "new_remote_source",
            "validated": "True",
            "ab_tested": "True",
            "status": "local_positive_signal",
            "can_reuse_directly": "True",
            "needs_adaptation": "True",
            "recommended_action": "作为主线下一轮最优先单源 A/B",
            "related_target": "chla/secchi",
            "related_layer": "middle>bottom>surface",
            "expected_mainline_value": "高",
            "risk": "中",
            "note": "对 Chl-a middle 与 Secchi middle 有最清晰的局部正信号",
        },
        {
            "item_name": "copernicus_bgc_features",
            "item_type": "feature_table",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/copernicus_bgc_features_2014_2024.csv",
            "mainline_target_path": "数据/处理后数据/衍生版本/copernicus_bgc_features_2014_2024.csv",
            "category": "new_remote_source",
            "validated": "True",
            "ab_tested": "True",
            "status": "deferred",
            "can_reuse_directly": "False",
            "needs_adaptation": "True",
            "recommended_action": "仅保留为 Chl-a/Secchi 补充候选，不作为第一批主线接入",
            "related_target": "chla/secchi",
            "related_layer": "middle/bottom",
            "expected_mainline_value": "低到中",
            "risk": "中",
            "note": "源项目结论是不适合盲目作为统一主线强推",
        },
        {
            "item_name": "nasa_oceancolor_features",
            "item_type": "feature_table",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/nasa_oceancolor_features_2014_2024.csv",
            "mainline_target_path": "数据/处理后数据/衍生版本/nasa_oceancolor_features_2014_2024.csv",
            "category": "new_remote_source",
            "validated": "True",
            "ab_tested": "True",
            "status": "neutral",
            "can_reuse_directly": "True",
            "needs_adaptation": "True",
            "recommended_action": "保留作光学指标参考源，不直接进入第一轮正式主线",
            "related_target": "chla/secchi",
            "related_layer": "surface/middle",
            "expected_mainline_value": "中",
            "risk": "中",
            "note": "覆盖率很好，但统一 A/B 未带来 promote",
        },
        {
            "item_name": "gpm_imerg_validation_bundle",
            "item_type": "source_validation",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/gpm_imerg_source_validation.json",
            "mainline_target_path": "文档/主线接管_20260521/主线数据接入计划.md",
            "category": "source_validation",
            "validated": "True",
            "ab_tested": "False",
            "status": "deferred",
            "can_reuse_directly": "False",
            "needs_adaptation": "True",
            "recommended_action": "保留验证结论，不进入当前主线，待 EULA 与下载链打通后再说",
            "related_target": "tn/tp/chla",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "高但当前不可部署",
            "risk": "高",
            "note": "源项目已明确 Earthdata/EULA 是当前阻塞点",
        },
        {
            "item_name": "high_quality_patch_qc_refined_features",
            "item_type": "feature_table",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/high_quality_patch_qc_refined_features_2014_2024.csv",
            "mainline_target_path": "数据/处理后数据/衍生版本/high_quality_patch_qc_refined_features_2014_2024.csv",
            "category": "patch_remote_feature",
            "validated": "True",
            "ab_tested": "True",
            "status": "not_recommended",
            "can_reuse_directly": "False",
            "needs_adaptation": "True",
            "recommended_action": "不作为统一主线输入，只保留为局部参考",
            "related_target": "chla/secchi",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "低",
            "risk": "高",
            "note": "patch 路线整体高缺失、高噪声，且多次 A/B 无法 promote",
        },
        {
            "item_name": "strong_mechanism_input_features",
            "item_type": "feature_table",
            "source_project_path": "数据/外部数据/试验项目接管_20260521/data_processed_current/strong_mechanism_input_features_2014_2024.csv",
            "mainline_target_path": "数据/处理后数据/衍生版本/strong_mechanism_input_features_2014_2024.csv",
            "category": "mechanism_bundle",
            "validated": "True",
            "ab_tested": "True",
            "status": "local_positive_signal",
            "can_reuse_directly": "True",
            "needs_adaptation": "True",
            "recommended_action": "拆成小规模 controlled combination，不整包并入",
            "related_target": "chla/secchi",
            "related_layer": "middle/bottom",
            "expected_mainline_value": "中",
            "risk": "高",
            "note": "局部任务有效，但整体稳定性不足",
        },
        {
            "item_name": "do_not_repeat_experiments",
            "item_type": "report",
            "source_project_path": "文档/主线接管_20260521/源项目_禁止重复实验清单.md",
            "mainline_target_path": "文档/主线接管_20260521/主线禁止重复清单.md",
            "category": "current_conclusion",
            "validated": "True",
            "ab_tested": "True",
            "status": "verified_useful",
            "can_reuse_directly": "True",
            "needs_adaptation": "False",
            "recommended_action": "直接纳入主线禁止重复清单",
            "related_target": "all",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "高",
            "risk": "低",
            "note": "这是最值得复用的负结果资产之一",
        },
        {
            "item_name": "prediction_ceiling_analysis",
            "item_type": "report",
            "source_project_path": "文档/主线接管_20260521/英文原始报告_归档/源项目_reports_current/prediction_ceiling_analysis.md",
            "mainline_target_path": "文档/主线接管_20260521/主线接管报告.md",
            "category": "current_conclusion",
            "validated": "True",
            "ab_tested": "True",
            "status": "verified_useful",
            "can_reuse_directly": "True",
            "needs_adaptation": "False",
            "recommended_action": "作为“为什么暂时到不了 0.60”的主证据之一",
            "related_target": "all",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "高",
            "risk": "低",
            "note": "清楚指出当前最关键缺口是 exact datetime 与真实底层/径流机制数据",
        },
        {
            "item_name": "trial_script_bundle",
            "item_type": "script_bundle",
            "source_project_path": "脚本/历史版本/试验项目接管_20260521",
            "mainline_target_path": "脚本/历史版本/试验项目接管_20260521",
            "category": "script_reuse",
            "validated": "True",
            "ab_tested": "True",
            "status": "verified_useful",
            "can_reuse_directly": "True",
            "needs_adaptation": "True",
            "recommended_action": "按模块挑选复用，不整包照搬",
            "related_target": "all",
            "related_layer": "surface/middle/bottom",
            "expected_mainline_value": "高",
            "risk": "中",
            "note": "已全部接管到当前项目，可在不依赖外部 worktree 的前提下继续整理",
        },
    ]
    write_csv(OUTDIR / "mainline_reuse_inventory.csv", rows, fieldnames)


def build_script_inventory() -> None:
    fieldnames = [
        "script_name",
        "source_path",
        "purpose",
        "can_reuse_directly",
        "needs_path_change",
        "needs_schema_change",
        "needs_dependency_change",
        "recommended_for_mainline",
        "note",
    ]
    rows = [
        {
            "script_name": "build_coastal_distance_features.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/data/build_coastal_distance_features.py",
            "purpose": "构建 coastal distance observation/station 特征",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "适合改造成主线 station-level/observation-level merge 脚本",
        },
        {
            "script_name": "build_epd_station_depth_features.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/data/build_epd_station_depth_features.py",
            "purpose": "构建站点水深/深度背景特征",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "station-level 静态特征，机制意义明确",
        },
        {
            "script_name": "refine_distance_to_mouth_systems.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/data/refine_distance_to_mouth_systems.py",
            "purpose": "细化 mouth system / zone 体系",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "适合生成主线可解释的 freshwater influence proxy",
        },
        {
            "script_name": "validate_distance_to_river_mouth.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/data/validate_distance_to_river_mouth.py",
            "purpose": "river mouth source validation",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "False",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "最适合直接复用的 source validation 脚本之一",
        },
        {
            "script_name": "validate_zone_refinement.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/data/validate_zone_refinement.py",
            "purpose": "zone refinement source validation",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "False",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "可直接复用 scheme4 结论",
        },
        {
            "script_name": "validate_gpm_imerg_source.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/data/validate_gpm_imerg_source.py",
            "purpose": "GPM IMERG 下载链/source validation",
            "can_reuse_directly": "False",
            "needs_path_change": "True",
            "needs_schema_change": "False",
            "needs_dependency_change": "True",
            "recommended_for_mainline": "False",
            "note": "Earthdata/EULA 阻塞未解，暂不建议迁入正式流程",
        },
        {
            "script_name": "validate_gpm_imerg_observation_alignment.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/data/validate_gpm_imerg_observation_alignment.py",
            "purpose": "GPM IMERG 对齐验证",
            "can_reuse_directly": "False",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "True",
            "recommended_for_mainline": "False",
            "note": "只有在下载链打通后才值得继续",
        },
        {
            "script_name": "run_nasa_oceancolor_fullrun.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/data/run_nasa_oceancolor_fullrun.py",
            "purpose": "OceanColor 全流程下载与构表",
            "can_reuse_directly": "False",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "True",
            "recommended_for_mainline": "False",
            "note": "现有特征表已接管，不需要马上重跑下载链",
        },
        {
            "script_name": "run_coastal_distance_four_target_ab.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/training/run_coastal_distance_four_target_ab.py",
            "purpose": "coastal distance 四指标 A/B",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "适合改造成统一主线 controlled A/B 模板",
        },
        {
            "script_name": "run_depth_source_four_target_ab.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/training/run_depth_source_four_target_ab.py",
            "purpose": "station depth/source 四指标 A/B",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "适合作为 station-level 静态输入 A/B 模板，但不要默认 promote",
        },
        {
            "script_name": "run_distance_to_mouth_four_target_ab.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/training/run_distance_to_mouth_four_target_ab.py",
            "purpose": "distance-to-mouth 四指标 A/B",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "推荐作为 freshwater influence 单源 A/B 入口模板",
        },
        {
            "script_name": "run_zone_refinement_four_target_ab.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/training/run_zone_refinement_four_target_ab.py",
            "purpose": "zone refinement 四指标 A/B",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "适合与 distance-to-mouth 组成 controlled combination",
        },
        {
            "script_name": "run_tn_tp_baseline_bootstrap.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/training/run_tn_tp_baseline_bootstrap.py",
            "purpose": "TN/TP baseline bootstrap",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "可复用其模型家族比较逻辑，但不能直接搬结果",
        },
        {
            "script_name": "run_precision_maximize_multitarget.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/training/run_precision_maximize_multitarget.py",
            "purpose": "多路候选精度极限探索",
            "can_reuse_directly": "False",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "False",
            "note": "探索性过强，建议只参考思路，不直接迁入",
        },
        {
            "script_name": "run_current_highres_patch_ab.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/training/run_current_highres_patch_ab.py",
            "purpose": "高分 patch A/B",
            "can_reuse_directly": "False",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "False",
            "note": "patch 路线已多次证明不适合做统一主线",
        },
        {
            "script_name": "experiment_train_utils.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/experiment_train_utils.py",
            "purpose": "试验训练通用工具",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "False",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "适合作为 future refactor 时的工具函数参考",
        },
        {
            "script_name": "new_data_sources_utils.py",
            "source_path": "脚本/历史版本/试验项目接管_20260521/new_data_sources_utils.py",
            "purpose": "新数据源对齐与下载辅助",
            "can_reuse_directly": "True",
            "needs_path_change": "True",
            "needs_schema_change": "True",
            "needs_dependency_change": "False",
            "recommended_for_mainline": "True",
            "note": "适合后续数据接入时抽象公共 schema 适配逻辑",
        },
    ]
    write_csv(OUTDIR / "mainline_script_reuse_inventory.csv", rows, fieldnames)


def build_markdown_artifacts() -> None:
    current_rows = read_csv(ROOT / "结果" / "正式结果" / "full_station" / "full_station_best_by_target_depth.csv")
    source_rows = read_csv(ROOT / "数据" / "外部数据" / "试验项目接管_20260521" / "data_processed_current" / "four_target_bootstrap_scorecard.csv")
    current_map = {(r["target"], r["depth"]): r for r in current_rows}
    source_map = {(r["target"], r["depth_layer"]): r for r in source_rows}

    comparison_rows: list[tuple[str, ...]] = []
    for target in ["chla", "secchi", "tn", "tp"]:
        for depth in ["surface", "middle", "bottom"]:
            cr = float(current_map[(target, depth)]["r2"])
            sr = float(source_map[(target, depth)]["test_r2"])
            winner = "当前主线" if cr >= sr else "试验项目"
            note = "试验项目只保留证据，不直接替换 current best" if winner == "试验项目" else "保持主线 current best 不动"
            comparison_rows.append((target, depth, fmt(str(cr)), fmt(str(sr)), winner, note))

    comparison_md = md_table(["指标", "水层", "当前主线R²", "试验项目R²", "当前更强者", "说明"], comparison_rows)

    handover_md = f"""# 主线研究接管报告

## 1. 接管范围

本次接管对象是“预测精度提升试验项目”已经完成的数据快照、source validation 结果、A/B 结果、脚本与结论性文档。为避免后续删除外部 worktree 后信息丢失，所有需要继续复用的内容已经复制进当前主线项目。

- 数据快照位置：`数据/外部数据/试验项目接管_20260521/`
- 脚本快照位置：`脚本/历史版本/试验项目接管_20260521/`
- 源项目文档快照位置：`文档/主线接管_20260521/`

## 2. 试验项目已经完成了哪些工作

试验项目已经完成并可直接被主线继承的工作，主要包括：

1. `station depth`、`coastal distance`、`distance-to-river-mouth`、`zone refinement`、`NASA OceanColor`、`Copernicus physics`、`GPM IMERG`、`tide proxy`、`highres patch`、`strong mechanism input` 等 source validation。
2. `coastal distance`、`depth source`、`distance_to_mouth`、`distance_to_mouth + zone`、`zone refinement`、`NASA OceanColor`、`strong input`、`strong mechanism`、`precision improvement`、`TN/TP baseline bootstrap` 等四指标 A/B。
3. 已下载并整理成表的 observation-level / station-level 特征文件，包括 OceanColor、Copernicus、patch、coastal distance、station depth、tide proxy、strong mechanism 等。
4. current best / no promote / do not repeat / prediction ceiling / gap analysis 这一整套“负结果资产”。

## 3. 当前主线与试验项目的 current best 对比

下面这张表只用于判断“哪些成果值得接管”，**不表示直接用试验项目覆盖主线 current best**。

{comparison_md}

### 对比结论

- 试验项目**明显更强的部分集中在 Chl-a 三个水层**。
- 当前主线在 `Secchi` 三层、`TN` 三层、`TP` 三层上整体已经不弱于试验项目，很多任务反而更强。
- 因此，试验项目最有价值的资产不是“整套结果替换主线”，而是：
  - 已验证的数据源结论；
  - 已完成的 A/B 证据；
  - 已被证伪的路线；
  - 对 Chl-a / Secchi / TN / TP 各自最值得继续的候选特征方向。

## 4. 主线可以直接继承哪些成果

### 4.1 可以直接继承且不需要重复下载

1. `distance_to_mouth` 与 `zone refinement` 的 source validation 结论。
2. `station_depth_features.csv`、`coastal_distance_features_2014_2024.csv`、`coastal_distance_station_features.csv`、`enhanced_tide_proxy_features_2014_2024.csv`、`copernicus_physics_features_2014_2024.csv`、`nasa_oceancolor_features_2014_2024.csv` 等已生成特征表。
3. `current_best_scorecard.csv`、`four_target_bootstrap_scorecard.csv`、`tn_tp_baseline_bootstrap_scorecard.csv`、`current_candidate_registry.csv` 等结论文件。
4. `do_not_repeat_experiments.md` 与 `prediction_ceiling_analysis.md` 等负结果与天花板文档。

### 4.2 可以直接成为主线候选输入的 source validation

- `distance-to-river-mouth`：验证通过，机制解释明确，适合进入主线 controlled A/B。
- `zone refinement`：验证通过，`scheme4` 已经被源项目筛出来，主线无需重做 scheme 选择。
- `station depth`：静态背景变量清楚，适合按 station-level merge。
- `coastal distance`：来源明确、无泄漏，但只适合作为谨慎候选，不适合被夸大成主突破。
- `Copernicus physics`：已经形成 observation-level 特征表，并在 `Chl-a middle / Secchi middle` 有清晰局部正信号。

## 5. 哪些路线已被证伪，主线不要重复

1. **把 OceanColor 当作统一主线 promote 路线反复重跑**：覆盖度不错，但统一 A/B 并未带来稳定 promote。
2. **继续无止境堆 patch 路线**：如果还是高缺失、低覆盖、靠 manifest 假装 patch，有害无益。
3. **弱 tide proxy、弱 coastal distance、弱 zone 反复单独重跑**：源项目已经证明这类单源通常只能带来很小增量。
4. **GEBCO/GLOBathy 类 bathymetry 盲目重来**：机制上听起来合理，但现阶段并没有形成稳定、可部署的正式输入。
5. **GPM IMERG 在下载链未打通前硬并入主线**：源项目已经明确 Earthdata/EULA 是阻塞点。

## 6. 哪些路线只有局部正信号，主线应谨慎保留

### 6.1 值得保留但必须 controlled A/B

- `Copernicus physics`：最值得先做单源 A/B，尤其对 `Chl-a middle`、`Secchi middle`。
- `enhanced_tide_proxy`：对 `Chl-a bottom`、`Secchi middle` 有局部作用。
- `station depth`：对 `TN bottom`、`Chl-a bottom` 可能有帮助，但不适合作为统一灵药。
- `distance_to_mouth + zone refinement`：机制解释强，但源项目统一结果混合，有些任务提升，有些任务明显拖累。

### 6.2 只建议保留为参考线

- `NASA OceanColor`：适合做光学指标参考源，不建议作为第一优先主线源。
- `Copernicus BGC`：保留为补充候选，不建议第一批就强推进主线。
- `strong_mechanism_input_features`：拆小组合测试，不整包并入。

## 7. 哪些脚本可以直接复用

### 7.1 可直接复用的 validation / 数据构建脚本

- `build_coastal_distance_features.py`
- `build_epd_station_depth_features.py`
- `refine_distance_to_mouth_systems.py`
- `validate_distance_to_river_mouth.py`
- `validate_zone_refinement.py`

### 7.2 可改造成主线 unified A/B 的脚本

- `run_coastal_distance_four_target_ab.py`
- `run_depth_source_four_target_ab.py`
- `run_distance_to_mouth_four_target_ab.py`
- `run_zone_refinement_four_target_ab.py`
- `run_tn_tp_baseline_bootstrap.py`

### 7.3 只能参考、不建议直接迁移的脚本

- `run_current_highres_patch_ab.py`
- `run_precision_maximize_multitarget.py`
- `run_nasa_oceancolor_fullrun.py`
- `validate_gpm_imerg_source.py`
- `validate_gpm_imerg_observation_alignment.py`

## 8. 如果主线要继续冲击 R² >= 0.60，最该继承哪些成果

### 8.1 立刻值得继承

1. `distance_to_mouth` / `zone refinement` / `station depth` / `coastal distance` 这些机制清楚、无需重下的静态或半静态输入。
2. `Copernicus physics` observation-level 特征表。
3. `enhanced_tide_proxy` 的局部正信号结论。
4. `tn_tp_baseline_bootstrap` 对 TN/TP 可用模型家族与特征组的筛选经验。

### 8.2 如果只选一个“下一轮最有价值候选路线”

**首选：`Copernicus physics` 单源 A/B。**

原因：

- 已经下载完成；
- observation-level 表已存在；
- 字段可解释；
- 无额外认证阻塞；
- 局部正信号最清晰；
- 最适合先用来验证“源项目成果能否稳定迁入主线”。

### 8.3 哪个组合最可能帮助 TN / TP / Secchi 整体提升

**首选组合：`distance_to_mouth + zone_refinement + station_depth + enhanced_tide_proxy`。**

原因：

- 这一组都具有机制意义，且没有明显的未来信息泄漏风险；
- 对营养盐与透明度，比单纯再堆光学遥感更合理；
- 与主线当前 `history/process` 框架天然兼容；
- 不需要再次下载即可先做 controlled A/B。

## 9. 当前最接近 R² = 0.60 的突破口与关键数据缺口

### 当前最接近 0.60 的任务

- 当前主线最接近 `R² = 0.60` 的仍是 **TN surface**，当前约为 **0.521**。
- 这也和源项目的判断一致：真正最有希望率先逼近 0.60 的不是 Chl-a，而是 TN surface 这类过程/径流更强的任务。

### 如果还达不到 0.60，最关键的数据缺口是什么

源项目的 ceiling 结论非常一致：

1. `exact_sampling_datetime`
2. `current/discharge/water-level`
3. `bottom temperature / vertical profile`
4. 更真实的 `freshwater influence / plume` 量化输入

换句话说，**主线下一步不该继续盲目重复弱 proxy，而应优先把“更接近真实机制的数据”并入**。

## 10. 对核心问题的直接回答

### 试验项目中哪些成果主线现在就能直接用？

- validation 结论；
- 已下载的特征表；
- current scorecard；
- do-not-repeat 与 prediction ceiling 文档；
- 统一 A/B 训练脚本模板。

### 主线不需要重复下载哪些数据？

- `Copernicus physics/BGC`
- `NASA OceanColor`
- `station depth`
- `coastal distance`
- `enhanced tide proxy`
- `high-quality patch / highres patch` 的结果快照（尽管不建议直接推进主线）

### 哪些结果虽然没 promote，但值得主线继续？

- `Copernicus physics`
- `distance_to_mouth`
- `zone refinement`
- `station depth`
- `enhanced_tide_proxy`

### 哪些路线主线绝对不要重复？

- 弱 proxy 的无限复试；
- 只靠 manifest 的 patch 假特征；
- 下载链没打通的 GPM IMERG 正式建模；
- 负迁移已明确的 bathymetry / weak patch 盲目重跑。

### 当前最值得作为主线下一轮优先实验的数据源是什么？

- `Copernicus physics`。

### 哪个组合最可能帮助 TN / TP / Secchi 整体提升？

- `distance_to_mouth + zone_refinement + station_depth + enhanced_tide_proxy`。

### 当前最接近 R² = 0.60 的主线突破口在哪里？

- `TN surface`。

### 如果还达不到 0.60，最关键的数据缺口是什么？

- `exact_sampling_datetime + discharge/water-level + bottom temperature/profile + 更真实的 freshwater influence proxy`。
"""
    (OUTDIR / "主线接管报告.md").write_text(handover_md, encoding="utf-8")

    do_not_repeat_md = """# 主线禁止重复清单

本清单用于约束主线后续实验，避免把试验项目已经验证过的低收益、负迁移或不可部署路线再次大量重复。

## 1. 已证明整体无效或不适合统一主线的路线

1. **OceanColor 直接 promote 路线**
   - source validation 虽然通过，但统一 A/B 没有稳定带来四指标 12 任务整体提升。
   - 允许保留为光学指标参考源，不允许当作主线突破口反复重跑。

2. **高缺失高噪声 patch 路线**
   - `s2_patch`、`landsat_patch`、`highres_patch`、`high_quality_patch_qc_refined` 这几条线已经反复证明：
     - 局部偶有正信号；
     - 但整体不稳定；
     - `eligible_mainline=False` 的情况很多。
   - 不允许继续把 patch 当作统一主线增强包。

3. **弱 tide proxy / weak proxy 无限复试**
   - 当整体增益仍停留在小幅波动时，不再继续靠重复调参或堆同类弱 proxy 追突破。

4. **bathymetry / GEBCO / GLOBathy 盲目重跑**
   - 如果还没有稳定、可解释、可部署的 marine subset 与 observation-level 对齐表，不允许重新把它包装成主线优先路线。

## 2. 已证明负迁移明显的组合

1. **distance_to_mouth + zone 盲目全任务推广**
   - 局部任务有正信号，但也拖累了部分 `TN` / `TP` 任务。
   - 不允许不分任务、直接全任务并入正式主链。

2. **depth source 无条件并入**
   - 对 `TN bottom`、`Chl-a bottom` 有局部作用，但不是统一增益。
   - 若未来重试，必须在 controlled A/B 中明确限定任务与层次。

3. **强机制整包输入无脑并入**
   - `strong_mechanism_input_features` 只显示局部收益。
   - 不允许整包一次性塞入正式主链模型。

## 3. 已通过 source validation 但 A/B 失败的路线

1. `NASA OceanColor`
2. `coastal distance` 作为统一主增益路线
3. `zone refinement` 作为单独统一增益路线
4. `distance_to_mouth` 作为单独统一增益路线

这些路线的正确用法是：

- 保留 validation 结论；
- 在主线里作为机制性候选输入；
- 做严格的 controlled A/B；
- 不把“验证通过”误当成“已经证明有效”。

## 4. 仅局部有效但不适合作为统一主线的路线

1. 只对个别站点、个别层、个别指标有收益的 patch 路线。
2. 只对 `Chl-a middle` 或 `Chl-a bottom` 有小增益，但整体 improved station ratio 不够高的组合。
3. 依赖缺失填补过多、样本覆盖变窄、或部署成本过高的遥感源。

## 5. 以下路线未来只有满足前提才值得重试

### 5.1 GPM IMERG / runoff proxy

只有在以下前提满足后才值得重试：

- Earthdata/EULA 已完成；
- 下载链可稳定复现；
- observation-level 对齐表已经生成；
- 能与主线 sample/obs 键稳定 join。

### 5.2 highres patch

只有在以下前提满足后才值得重试：

- 拥有真实 patch 统计，而不是 manifest；
- 覆盖率、有效像元率、有效样本比例达到可部署水平；
- 至少在一个统一设置下显示出明显高于 0.005 的稳定增益。

### 5.3 强机制整包路线

只有在以下前提满足后才值得重试：

- 先拆成小组合；
- 逐个确认无泄漏、无高缺失污染；
- 至少在一个目标组上有可重复收益。

## 6. 主线后续必须特别避免的重复

1. 低收益 patch 重复堆叠。
2. 弱 proxy 无限复试。
3. 已失败的 bathymetry / zone / coastal 组合盲目重跑。
4. 只对单个任务有帮助但拖累其他任务的组合，未经拆分就直接并入正式主链。
5. 只因为“源项目做过很多次”就默认它值得继续。

## 7. 主线的正确姿势

- 先接管证据，再接管数据，再接管脚本。
- 先用 source validation 缩小候选集，再做主线 controlled A/B。
- 以四指标 12 任务的整体价值排序，而不是继续追某一个局部高分。
"""
    (OUTDIR / "主线禁止重复清单.md").write_text(do_not_repeat_md, encoding="utf-8")

    adoption_md = """# 主线接管优先级

本文件把试验项目成果按“立即接管 / 轻微改造后接管 / 暂不接管”三层拆开，避免主线一上来就把所有东西混进正式管道。

## 第一层：立即接管

这些成果已经验证清楚，主线现在就能用，不需要重新下载，也不需要额外认证。

1. `current_best_scorecard.csv`
2. `four_target_bootstrap_scorecard.csv`
3. `tn_tp_baseline_bootstrap_scorecard.csv`
4. `current_candidate_registry.csv`
5. `distance_to_mouth` source validation
6. `zone_refinement` source validation
7. `station_depth_feature_validation.json`
8. `coastal_distance_validation_report.json`
9. `do_not_repeat_experiments.md`
10. `prediction_ceiling_analysis.md`

### 立即接管的意义

- 这些文件直接决定“哪些路线值得继续、哪些不该重跑”。
- 它们本身不改变主线 current best，但会显著减少无效探索。

## 第二层：接管但需轻微改造

这些成果已经形成数据或脚本，但要改路径、字段或 merge 键，才能进入主线。

### 数据

1. `station_depth_features.csv`
2. `coastal_distance_features_2014_2024.csv`
3. `coastal_distance_station_features.csv`
4. `enhanced_tide_proxy_features_2014_2024.csv`
5. `copernicus_physics_features_2014_2024.csv`
6. `nasa_oceancolor_features_2014_2024.csv`
7. `strong_mechanism_input_features_2014_2024.csv`（仅限拆小组合）

### 脚本

1. `build_coastal_distance_features.py`
2. `build_epd_station_depth_features.py`
3. `refine_distance_to_mouth_systems.py`
4. `run_coastal_distance_four_target_ab.py`
5. `run_depth_source_four_target_ab.py`
6. `run_distance_to_mouth_four_target_ab.py`
7. `run_zone_refinement_four_target_ab.py`
8. `run_tn_tp_baseline_bootstrap.py`

### 轻微改造内容

- 统一路径到当前主线目录结构；
- 统一字段到 `station / date / sample_id / obs_id / depth` 这套主线键；
- 统一输出到 `结果/扩展实验/实验名称_日期/`；
- 在不改模型逻辑的前提下接成主线 controlled A/B。

## 第三层：暂不接管

这些成果目前风险高、整体负迁移明显，或下载链没有真正打通。

1. `GPM IMERG` 正式数据接入
2. `Copernicus BGC` 作为第一批主线源
3. `highres_patch_features_2014_2024.csv`
4. `high_quality_patch_qc_refined_features_2014_2024.csv`
5. `run_current_highres_patch_ab.py`
6. `run_precision_maximize_multitarget.py`
7. `run_nasa_oceancolor_fullrun.py`（当前主线不需要重复下载）
8. `GEBCO/GLOBathy` 相关 bathymetry 推进

## 结论

- **第一层解决“别再瞎试什么”。**
- **第二层解决“下一轮主线该先把什么接进来”。**
- **第三层明确“现在不要碰什么”。**
"""
    (OUTDIR / "主线接管优先级.md").write_text(adoption_md, encoding="utf-8")

    data_plan_md = """# 主线数据接入计划

## 1. 接入原则

主线接入试验项目成果时，按以下顺序判断：

1. 文件是否已被接管到当前项目。
2. 字段是否可解释。
3. 时间范围是否覆盖 2014-2024。
4. 是否能按主线 observation-level / station-level 稳定 merge。
5. 是否存在未来信息或隐性泄漏风险。
6. 是否已经完成 source validation 或正式 A/B。

## 2. 哪些数据文件应直接接入主线共享输入层

这些文件已经被复制到当前项目，建议保留在共享输入层，不再重复复制或重复下载：

- `数据/外部数据/试验项目接管_20260521/data_processed_current/station_depth_features.csv`
- `数据/外部数据/试验项目接管_20260521/data_processed_current/coastal_distance_features_2014_2024.csv`
- `数据/外部数据/试验项目接管_20260521/data_processed_current/coastal_distance_station_features.csv`
- `数据/外部数据/试验项目接管_20260521/data_processed_current/enhanced_tide_proxy_features_2014_2024.csv`
- `数据/外部数据/试验项目接管_20260521/data_processed_current/copernicus_physics_features_2014_2024.csv`
- `数据/外部数据/试验项目接管_20260521/data_processed_current/nasa_oceancolor_features_2014_2024.csv`

建议做法：

- **短期内不再复制第二份。**
- 主线脚本先从该共享层引用。
- 只有当某个源通过主线正式 A/B 后，再提升为 `数据/处理后数据/衍生版本/` 的正式衍生表。

## 3. 哪些特征适合 station-level merge

1. `station_depth_features.csv`
2. `coastal_distance_station_features.csv`
3. `distance_to_mouth_station_validation.csv` 生成的 station-level mouth 特征
4. `zone_refinement_station_validation.csv` 生成的 refined zone 特征

适用原则：

- 静态空间背景优先按 `station` merge；
- 不要把 station-level 变量错误当成 date-level 变量重复展开；
- 进入主线前先确认 94 站覆盖完整。

## 4. 哪些特征适合 observation-level merge

1. `coastal_distance_features_2014_2024.csv`
2. `enhanced_tide_proxy_features_2014_2024.csv`
3. `copernicus_physics_features_2014_2024.csv`
4. `nasa_oceancolor_features_2014_2024.csv`
5. `strong_mechanism_input_features_2014_2024.csv`（仅拆小组合）

适用原则：

- 必须能按 `station + date` 或 `sample_id / obs_id` 对齐；
- 若只有 date-level 含义，文档中要显式标注其机制代理性质；
- 不允许用删除样本的方式“抬高”分数。

## 5. 哪些数据只应保留为参考，不进入正式主线模型

1. `current_best_scorecard.csv`
2. `four_target_bootstrap_scorecard.csv`
3. `tn_tp_baseline_bootstrap_scorecard.csv`
4. `current_candidate_registry.csv`
5. `*_validation_report.json`
6. `*_comparison_to_current_best.csv`
7. `do_not_repeat_experiments.md`
8. `prediction_ceiling_analysis.md`

这些文件的用途是：

- 定义接管优先级；
- 定义禁止重复边界；
- 为主线实验立项提供证据；
- 不直接作为模型输入。

## 6. 哪些数据仍需补 source validation 或下载链验证

1. `GPM IMERG`
   - 仍受 Earthdata/EULA 阻塞。
2. 更完整的 `runoff/discharge` 代理
   - 当前仍缺稳定自动链路。
3. `exact_sampling_datetime`
   - 目前主线和试验项目都还没有真正补齐。
4. `bottom temperature / vertical profile / current / water-level`
   - 这是当前冲击 R²=0.60 的关键机制缺口。

## 7. 主线 observation-level / station-level 接入建议

### 第一批 observation-level

- `copernicus_physics_features_2014_2024.csv`
- `enhanced_tide_proxy_features_2014_2024.csv`

### 第一批 station-level

- `station_depth_features.csv`
- `coastal_distance_station_features.csv`
- `distance_to_mouth` / `zone refinement` 生成后的 refined station 表

### 暂缓接入

- `highres_patch_features_2014_2024.csv`
- `high_quality_patch_qc_refined_features_2014_2024.csv`
- `Copernicus BGC` 整包
- `GPM IMERG`
"""
    (OUTDIR / "主线数据接入计划.md").write_text(data_plan_md, encoding="utf-8")

    next_exp_md = """# 主线下一轮实验计划

## 1. 目标

在不破坏当前主线 `full_station` current best、不引入泄漏、不重复无效路线的前提下，启动一轮基于“试验项目已验证成果”的正式主线 controlled A/B 准备。

## 2. 主线下一轮最优先的数据源

### 第一优先数据源

**`Copernicus physics` observation-level 特征表**

文件：

- `数据/外部数据/试验项目接管_20260521/data_processed_current/copernicus_physics_features_2014_2024.csv`

优先原因：

1. 已下载完成；
2. 已形成特征表；
3. 无额外认证阻塞；
4. 局部正信号最清晰；
5. 最适合作为“试验项目成果是否能稳定迁入主线”的第一条单源 A/B。

## 3. 主线下一轮最优先的单源 A/B

### 单源 A/B 1：`copernicus_physics_only`

目标：

- 全部四指标 12 任务统一评估；
- 重点观察 `Chl-a middle`、`Secchi middle`，同时检查是否拖累 TN/TP。

评价口径：

- 沿用主线 current best 对比；
- 至少输出 `R² / RMSE / MAE`；
- 同时检查 improved-task 比例；
- 不允许通过删样本制造“提升”。

## 4. 是否先复用已有 source validation 成果

**是。**

下一轮实验不应重新从零做 source validation，而应直接复用：

- `distance_to_mouth` validation
- `zone_refinement` validation
- `station_depth` validation
- `coastal_distance` validation

这些 validation 已经足够支持主线构造第一批 station-level / observation-level 候选表。

## 5. 是否需要先补 observation-level 特征表

**需要，但只补主线下一轮最值得测的那一小批。**

优先补：

1. `distance_to_mouth` 正式主线 observation/station 特征表
2. `zone_refinement scheme4` 主线特征表
3. 必要时把 `station_depth` 和 `coastal_distance_station_features` 标准化成主线 schema

不优先补：

- patch 相关重型特征
- GPM IMERG
- 任何尚未打通下载链的源

## 6. 是否要先统一四指标 12 任务的训练入口

**要。**

建议在下一轮正式实验前，先保证主线只有一个统一训练入口处理“接管后 A/B”。

### 建议入口

- 训练入口：`脚本/04_模型训练/05_train_full_station_models.py` 的受控扩展版，或在扩展实验目录新增专门入口脚本。
- 输出目录：`结果/扩展实验/mainline_reuse_ab_20260521/`

### 当前阶段不直接创建新脚本的原因

- 现在先完成接管、清单和计划；
- 等用户确认下一轮优先路线后，再决定是复用旧入口还是新建扩展实验入口。

## 7. 主线下一轮最值得测试的组合

### 组合 A：`distance_to_mouth + zone_refinement + station_depth + enhanced_tide_proxy`

最值得服务的任务：

- `TN`
- `TP`
- `Secchi`

原因：

- 这组输入具有统一机制意义；
- 对营养盐与透明度比继续堆光学遥感更合理；
- 试验项目已经给出了足够多的验证和负结果边界。

### 组合 B：`history/process + copernicus_physics + enhanced_tide_proxy`

最值得服务的任务：

- `Chl-a middle`
- `Chl-a bottom`
- `Secchi middle`

原因：

- 是源项目中最有局部正信号的一组遥感/过程代理组合；
- 同时不需要重新下载。

## 8. 哪条路线最有机会推动各目标

### TN

- 最可能受益于：`distance_to_mouth + zone_refinement + station_depth + future runoff/discharge`
- 当前最接近 0.60 的任务也是 `TN surface`

### TP

- 最可能受益于：`freshwater influence proxy + zone + station depth + tide interaction`
- 不建议继续只靠光学遥感硬推

### Secchi

- 最可能受益于：`Copernicus physics + enhanced_tide_proxy + freshwater influence proxy`
- 比单独 OceanColor 更值得先测

### Chl-a

- 最可能受益于：`Copernicus physics + enhanced_tide_proxy + 更真实的 short-term mechanism`
- 若没有 exact sampling datetime 和真实底层/径流机制数据，想直接冲到 0.60 很难

## 9. 下一轮实验的建议输入、输出与评价口径

### 建议输入

- 主线正式主表：`数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv`
- 接管共享输入层：`数据/外部数据/试验项目接管_20260521/data_processed_current/`
- 关键候选表：`copernicus_physics_features_2014_2024.csv`、`enhanced_tide_proxy_features_2014_2024.csv`、`station_depth_features.csv`、`coastal_distance_station_features.csv`

### 预期输出

- `结果/扩展实验/mainline_reuse_ab_20260521/ab_metrics.csv`
- `结果/扩展实验/mainline_reuse_ab_20260521/comparison_to_current_best.csv`
- `结果/扩展实验/mainline_reuse_ab_20260521/no_promote_reason.md`
- `结果/扩展实验/mainline_reuse_ab_20260521/feature_manifest.csv`

### 评价口径

1. 与当前主线 current best 逐任务对比；
2. 同时报告 `R² / RMSE / MAE`；
3. 检查是否存在样本删除、覆盖率骤降或潜在泄漏；
4. 不把局部单任务小增益夸大成“主线突破”。

## 10. 结论

主线下一轮不该从头探索，而应优先做三件事：

1. 复用已有 validation，先把 station-level / observation-level 机制候选表接进共享输入层；
2. 先测 `Copernicus physics` 单源 A/B；
3. 再测 `distance_to_mouth + zone_refinement + station_depth + enhanced_tide_proxy` 的统一组合 A/B。

如果这样仍无法把更多任务推近或超过 `R² = 0.60`，那最关键瓶颈就不是“还缺一个弱代理”，而是**真实机制数据缺口**。
"""
    (OUTDIR / "主线下一轮实验计划.md").write_text(next_exp_md, encoding="utf-8")


def update_migration_record() -> None:
    migration_path = ROOT / "文档" / "文件迁移记录.md"
    append_text = """

## 2026-05-21 试验项目成果主线接管

### 目标

- 将“预测精度提升试验项目”中已经完成的数据、脚本、报告和结论接管到当前主线项目。
- 不修改主线 current best，不直接覆盖正式结果。
- 为下一轮四指标 12 任务正式 A/B 提供可复用清单、禁止重复清单与接入计划。

### 已接管到当前项目的内容

- 数据快照：`数据/外部数据/试验项目接管_20260521/`
- 脚本快照：`脚本/历史版本/试验项目接管_20260521/`
- 文档快照：`文档/主线接管_20260521/`

### 本轮新增正式产物

- `文档/主线接管_20260521/mainline_reuse_inventory.csv`
- `文档/主线接管_20260521/主线接管报告.md`
- `文档/主线接管_20260521/主线禁止重复清单.md`
- `文档/主线接管_20260521/主线接管优先级.md`
- `文档/主线接管_20260521/主线数据接入计划.md`
- `文档/主线接管_20260521/mainline_script_reuse_inventory.csv`
- `文档/主线接管_20260521/主线下一轮实验计划.md`

### 影响范围

- 本次仅新增接管清单、接管报告和实验计划。
- 不修改正式主链模型逻辑、训练参数、特征构造逻辑和绘图风格。
- 不改写 `结果/正式结果/full_station/` 中任何 current best 结果。
"""
    if migration_path.exists():
        old = migration_path.read_text(encoding="utf-8", errors="ignore")
        migration_path.write_text(old + append_text, encoding="utf-8")
    else:
        migration_path.write_text("# 文件迁移记录\n" + append_text, encoding="utf-8")


def main() -> None:
    build_reuse_inventory()
    build_script_inventory()
    build_markdown_artifacts()
    update_migration_record()
    print("HANDOVER_ARTIFACTS_OK")


if __name__ == "__main__":
    main()
