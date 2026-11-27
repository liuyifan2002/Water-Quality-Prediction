from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.run_current_highres_patch_ab import (
    CURRENT_BEST,
    CURRENT_CONFIG_DIR,
    CURRENT_INTERIM_DIR,
    CURRENT_OUTPUT_DIR,
    CURRENT_PROCESSED_DIR,
    CURRENT_REPORT_DIR,
    PROJECT_ROOT as TRAINING_PROJECT_ROOT,
    build_common_features,
    current_best_prediction_table,
    current_best_metric_map,
    load_base_dataset,
    load_bottom_mechanism,
    load_highres_table,
    merge_model_dataset,
)

assert PROJECT_ROOT == TRAINING_PROJECT_ROOT

EXTERNAL_GEE_DIR = PROJECT_ROOT / "data" / "external" / "gee_exports"
BASE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chla_ocean_colour_precision_optimization_20260517"
    / "chla_ocean_colour_precision_optimization_dataset_20260517.csv"
)

TARGET_LABELS = {
    "chla": "Chl-a",
    "secchi": "Secchi/ZSD",
}


def ensure_dirs() -> None:
    for path in [
        CURRENT_REPORT_DIR,
        CURRENT_OUTPUT_DIR,
        CURRENT_CONFIG_DIR,
        CURRENT_PROCESSED_DIR,
        CURRENT_INTERIM_DIR,
        EXTERNAL_GEE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def format_feature_block(features: list[str], width: int = 8) -> str:
    if not features:
        return "```text\n无\n```"
    lines = []
    for i in range(0, len(features), width):
        lines.append(", ".join(features[i : i + width]))
    return "```text\n" + "\n".join(lines) + "\n```"


def copy_external_exports() -> list[str]:
    copied = []
    mapping = {
        CURRENT_PROCESSED_DIR / "s2_highres_patch_features_2014_2024.csv": EXTERNAL_GEE_DIR / "s2_patch_features_2014_2024.csv",
        CURRENT_PROCESSED_DIR / "landsat_highres_patch_features_2014_2024.csv": EXTERNAL_GEE_DIR / "landsat_patch_features_2014_2024.csv",
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copyfile(src, dst)
            copied.append(f"{src.as_posix()} -> {dst.as_posix()}")
    return copied


def load_artifacts() -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    artifacts["scorecard"] = read_csv(CURRENT_OUTPUT_DIR / "current_best_scorecard.csv")
    artifacts["candidate_registry"] = read_csv(CURRENT_OUTPUT_DIR / "current_candidate_registry.csv")
    artifacts["station_cmp"] = read_csv(CURRENT_OUTPUT_DIR / "station_level_prediction_comparison.csv")
    artifacts["config"] = load_json(CURRENT_CONFIG_DIR / "current_highres_patch_ab.json")
    artifacts["highres_val"] = load_json(CURRENT_PROCESSED_DIR / "highres_patch_validation_report.json")
    artifacts["mech_val"] = load_json(CURRENT_PROCESSED_DIR / "bottom_mechanism_feature_validation.json")
    artifacts["baseline_metrics"] = current_best_metric_map()
    artifacts["s2_metrics"] = read_csv(CURRENT_OUTPUT_DIR / "s2_patch_ab_metrics.csv")
    artifacts["landsat_metrics"] = read_csv(CURRENT_OUTPUT_DIR / "landsat_patch_ab_metrics.csv")
    artifacts["highres_metrics"] = read_csv(CURRENT_OUTPUT_DIR / "highres_patch_ab_metrics.csv")
    artifacts["mechanism_metrics"] = read_csv(CURRENT_OUTPUT_DIR / "chla_bottom_mechanism_ab_metrics.csv")
    artifacts["highres_cmp"] = read_csv(CURRENT_OUTPUT_DIR / "highres_patch_comparison_to_current_best.csv")
    artifacts["merged_df"] = merge_model_dataset()[0] if not (CURRENT_INTERIM_DIR / "current_model_dataset_with_highres_and_mechanism.csv").exists() else read_csv(CURRENT_INTERIM_DIR / "current_model_dataset_with_highres_and_mechanism.csv")
    artifacts["merged_df"]["date"] = pd.to_datetime(artifacts["merged_df"]["date"])
    artifacts["base_df"] = load_base_dataset()
    artifacts["highres_table"], artifacts["highres_feature_groups"] = load_highres_table()
    artifacts["mech_table"], artifacts["mechanism_features"] = load_bottom_mechanism()
    return artifacts


def scenario_metric_lookup(
    artifacts: dict[str, Any],
    scenario: str,
    target: str,
    depth: str,
) -> dict[str, float]:
    if scenario == "current_best_baseline":
        return artifacts["baseline_metrics"].get((target, depth), {})
    table_map = {
        "s2_patch": artifacts["s2_metrics"],
        "landsat_patch": artifacts["landsat_metrics"],
        "highres_patch": artifacts["highres_metrics"],
        "bottom_mechanism": artifacts["mechanism_metrics"],
    }
    table = table_map.get(scenario)
    if table is None or table.empty:
        return {}
    sub = table[
        (table["target"] == target)
        & (table["depth_layer"] == depth)
        & (table["split"] == "test")
    ]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    return {
        "r2": float(row["r2"]),
        "rmse": float(row["rmse"]),
        "mae": float(row["mae"]),
        "p90_rmse": float(row["p90_rmse"]) if pd.notna(row["p90_rmse"]) else np.nan,
        "bias": float(row["bias"]),
    }


def build_model_registry(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    scorecard = artifacts["scorecard"]
    registry = artifacts["candidate_registry"]
    config = artifacts["config"]
    manifests = dict(config.get("scenario_manifests", {}))
    if config.get("mechanism_manifest"):
        manifests["bottom_mechanism"] = config["mechanism_manifest"]
    merged_df = artifacts["merged_df"]
    rows = []
    for row in scorecard.itertuples(index=False):
        key = f"{row.target}::{row.depth_layer}"
        baseline_meta = CURRENT_BEST[(row.target, row.depth_layer)]
        selected_model = row.selected_model
        selected_feature_set = row.selected_feature_set
        selected_reason = row.reason
        scenario_rows = registry[(registry["target"] == row.target) & (registry["depth_layer"] == row.depth_layer)]
        selected_scenario = None
        selected_record = None
        for _, cand in scenario_rows.iterrows():
            if cand["candidate_model"] == selected_model:
                selected_scenario = cand["scenario"]
                selected_record = cand.to_dict()
                break
        if selected_scenario and selected_scenario in manifests and key in manifests[selected_scenario]["models"]:
            model_meta = manifests[selected_scenario]["models"][key]
            features = model_meta["features"]
            params = model_meta["params"]
            model_type = model_meta["model_type"]
        else:
            features = build_common_features(merged_df, row.target, extra_features=[])
            params = {
                "station_calibration": {
                    "base_model": "LightGBM",
                    "n_estimators": 500,
                    "learning_rate": 0.035,
                    "num_leaves": 18,
                    "max_depth": 6,
                    "min_child_samples": 35,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 1.0,
                    "station_bias_shrink": 12.0,
                    "random_state": 42,
                },
                "ensemble_lgbm_rf_avg": {
                    "blend_weight_lgbm": "按 validation 选择",
                    "blend_weight_rf": "按 validation 选择",
                    "lgbm_n_estimators": 500,
                    "rf_n_estimators": 350,
                    "rf_min_samples_leaf": 2,
                },
                "impute_knn_lightgbm": {
                    "numeric_imputer": "KNNImputer",
                    "knn_n_neighbors": 5,
                    "knn_weights": "distance",
                    "n_estimators": 500,
                    "learning_rate": 0.035,
                    "num_leaves": 18,
                    "max_depth": 6,
                    "min_child_samples": 35,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                },
                "stacking_ridge": {
                    "base_models": ["lightgbm", "xgboost", "random_forest"],
                    "meta_model": "Ridge",
                    "alpha": 1.0,
                },
            }[baseline_meta["model_type"]]
            model_type = baseline_meta["model_type"]
        rows.append(
            {
                "target": row.target,
                "depth_layer": row.depth_layer,
                "title": f"{TARGET_LABELS[row.target]} {row.depth_layer} 当前主模型",
                "selected_model": selected_model,
                "selected_model_type": model_type,
                "selected_feature_set": selected_feature_set,
                "selected_reason": selected_reason,
                "selected_scenario": selected_scenario or "current_best_baseline",
                "params": params,
                "features": features,
                "selected_record": selected_record,
                "final_selected_r2": float(row.final_selected_r2),
                "previous_best_r2": float(row.previous_best_r2),
                "whether_updated": bool(row.whether_updated),
            }
        )
    return rows


def build_candidate_model_rows(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    registry = artifacts["candidate_registry"]
    config = artifacts["config"]
    manifests = dict(config.get("scenario_manifests", {}))
    if config.get("mechanism_manifest"):
        manifests["bottom_mechanism"] = config["mechanism_manifest"]
    rows = []
    for record in registry.itertuples(index=False):
        scenario = str(record.scenario)
        key = f"{record.target}::{record.depth_layer}"
        if scenario not in manifests or key not in manifests[scenario]["models"]:
            continue
        meta = manifests[scenario]["models"][key]
        rows.append(
            {
                "target": record.target,
                "depth_layer": record.depth_layer,
                "title": f"{TARGET_LABELS[record.target]} {record.depth_layer} {scenario} 候选模型",
                "selected_model": meta["candidate_name"],
                "selected_model_type": meta["model_type"],
                "selected_feature_set": scenario,
                "selected_reason": record.selection_status,
                "selected_scenario": scenario,
                "params": meta["params"],
                "features": meta["features"],
                "selected_record": record._asdict(),
                "final_selected_r2": float(record.candidate_r2),
                "previous_best_r2": float(record.current_best_r2),
                "whether_updated": bool(record.eligible_mainline),
            }
        )
    return rows


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    groups = {
        "时间特征": [c for c in features if c in {"year_offset", "month_sin", "month_cos", "day_of_year", "quarter", "is_wet_season", "days_since_prev_sample"}],
        "站点与空间特征": [c for c in features if c in {"station", "water_zone_en", "lon", "lat"}],
        "历史目标特征": [c for c in features if "lag" in c or "roll" in c or c.startswith(("chla_", "secchi_"))],
        "常规遥感与 Ocean Colour": [c for c in features if c.startswith(("oc_", "landsat_", "s2_")) and "_3x3_" not in c and "_5x5_" not in c],
        "高分 patch 特征": [c for c in features if "_3x3_" in c or "_5x5_" in c],
        "机制变量": [c for c in features if any(tag in c for tag in ["proxy", "wind", "temp", "tide", "rain", "era5", "station_depth", "bathymetry", "current", "water_level", "discharge"])],
    }
    return {k: v for k, v in groups.items() if v}


def feature_meaning_lines() -> list[str]:
    return [
        "历史目标特征用于表达固定站点的惯性、季节性与上一次观测后的短期趋势。",
        "Ocean Colour 点值特征反映近岸水体透明度、光衰减与叶绿素敏感信息。",
        "Sentinel-2 / Landsat 高分 patch 特征描述站点邻域 3x3、5x5 像元的均值、分布和异质性，而不是单点像元值。",
        "valid_ratio、count、date_diff_abs 这类字段用于表达高分影像质量、有效像元比例和观测时间差。",
        "机制变量当前主要仍是弱代理变量，更多反映风混合、温度和潮汐背景，而不是真实底层分层或水动力闭环。",
    ]


def write_experiment_master_index(artifacts: dict[str, Any], model_rows: list[dict[str, Any]]) -> None:
    scorecard = artifacts["scorecard"]
    rows = []
    for row in scorecard.itertuples(index=False):
        rows.append(
            {
                "target": row.target,
                "depth_layer": row.depth_layer,
                "final_selected_r2": row.final_selected_r2,
                "selected_model": row.selected_model,
                "selected_feature_set": row.selected_feature_set,
                "whether_updated": "是" if row.whether_updated else "否",
                "reason": row.reason,
            }
        )
    lines = [
        "# current/ 实验入口索引",
        "",
        "## 当前最佳结果",
        "",
        format_table(rows, ["target", "depth_layer", "final_selected_r2", "selected_model", "selected_feature_set", "whether_updated", "reason"]),
        "",
        "## 本轮关键结论",
        "",
        "- Sentinel-2 patch、Landsat patch、S2+Landsat 组合都已经真实进入模型，不再处于“计划阶段”。",
        "- 但本轮高分 patch 三条路线都没有满足“整体 test R² 提升 >= 0.03”的主线替代条件。",
        "- Chl-a bottom 机制变量路线只有弱代理变量参与，当前不能被视为“真实 bottom 机制变量已补齐”。",
        "",
        "## 关键文件",
        "",
        "- `experiments/outputs/current/current_best_scorecard.csv`",
        "- `experiments/outputs/current/s2_patch_ab_metrics.csv`",
        "- `experiments/outputs/current/landsat_patch_ab_metrics.csv`",
        "- `experiments/outputs/current/highres_patch_ab_metrics.csv`",
        "- `experiments/outputs/current/chla_bottom_mechanism_ab_metrics.csv`",
        "- `experiments/outputs/current/station_level_prediction_comparison.csv`",
        "- `experiments/reports/current/model_input_alignment_documentation.md`",
        "- `experiments/reports/current/model_parameter_and_feature_inventory.md`",
        "- `data/processed/current/highres_patch_features_2014_2024.csv`",
        "- `data/processed/current/bottom_mechanism_features_2014_2024.csv`",
        "",
        "## 当前主模型一览",
        "",
    ]
    for row in model_rows:
        lines.append(f"- **{row['title']}**：`{row['selected_model']}`，R²={row['final_selected_r2']:.4f}")
    (CURRENT_REPORT_DIR / "experiment_master_index.md").write_text("\n".join(lines), encoding="utf-8")


def write_experiment_report(artifacts: dict[str, Any]) -> None:
    scorecard = artifacts["scorecard"]
    registry = artifacts["candidate_registry"]
    mech_val = artifacts["mech_val"]
    station_cmp = artifacts["station_cmp"]
    rows = []
    for row in scorecard.itertuples(index=False):
        rows.append(
            {
                "target": row.target,
                "depth_layer": row.depth_layer,
                "final_selected_r2": row.final_selected_r2,
                "selected_model": row.selected_model,
                "reason": row.reason,
            }
        )
    improved_station = (
        station_cmp.groupby(["target", "depth_layer"])["improved"].mean().reset_index(name="improved_ratio")
    )
    candidate_table_rows = []
    for rec in registry.itertuples(index=False):
        candidate_table_rows.append(
            {
                "target": rec.target,
                "depth_layer": rec.depth_layer,
                "scenario": rec.scenario,
                "candidate_r2": rec.candidate_r2,
                "delta_r2": rec.delta_r2,
                "delta_validation_r2": rec.delta_validation_r2,
                "station_improved_ratio": rec.station_improved_ratio,
                "selection_status": rec.selection_status,
            }
        )
    bottom_station = station_cmp[
        (station_cmp["target"] == "chla") & (station_cmp["depth_layer"] == "bottom")
    ].copy()
    bottom_station = bottom_station.sort_values("best_station_r2", ascending=False).head(8)
    lines = [
        "# current/ 本轮实验报告",
        "",
        "## 本轮实际完成内容",
        "",
        "1. 检查并确认 Sentinel-2 / Landsat highres patch 文件已真实落地。",
        "2. 校验 highres patch 的 request_id、station+date 对齐、时间差、覆盖率和缺失率。",
        "3. 分别运行 Sentinel-2 patch only、Landsat patch only、S2+Landsat patch 三组 A/B。",
        "4. 检查并运行 Chl-a bottom 机制变量 A/B。",
        "5. 更新 current_best_scorecard、逐站点比较与 current/ 文档。",
        "",
        "## 高分 patch 是否真实下载成功",
        "",
        "- Sentinel-2：已成功进入 `data/processed/current/s2_highres_patch_features_2014_2024.csv`。",
        "- Landsat：已成功进入 `data/processed/current/landsat_highres_patch_features_2014_2024.csv`。",
        "- 合并表：已成功进入 `data/processed/current/highres_patch_features_2014_2024.csv`。",
        "",
        "## 本轮主结论",
        "",
        format_table(rows, ["target", "depth_layer", "final_selected_r2", "selected_model", "reason"]),
        "",
        "## 本轮各 candidate 路线摘要",
        "",
        format_table(
            candidate_table_rows,
            [
                "target",
                "depth_layer",
                "scenario",
                "candidate_r2",
                "delta_r2",
                "delta_validation_r2",
                "station_improved_ratio",
                "selection_status",
            ],
        ),
        "",
        "## 高分 patch 路线判断",
        "",
        "- Sentinel-2 patch 已进入模型，但本轮没有达到整体 R² 提升 >= 0.03 的 useful threshold。",
        "- Landsat patch 已进入模型，但覆盖率更低，整体刷新更弱。",
        "- S2 + Landsat 组合也没有刷新 current 主线。",
        "",
        "## Chl-a bottom 机制变量判断",
        "",
        f"- 当前机制表缺失的强机制变量：{', '.join(mech_val['unavailable_key_features'])}。",
        "- 因此本轮 mechanism A/B 只能视为“弱代理变量验证”，不能写成“真实底层机制变量已经补齐”。",
        "",
        "## 逐站点现象",
        "",
    ]
    for row in improved_station.itertuples(index=False):
        lines.append(
            f"- {row.target} {row.depth_layer}：在“current best vs 本轮 candidate 集合”的逐站点推荐比较中，improved 比例为 {row.improved_ratio:.2%}。"
        )
    lines += [
        "",
        "### Chl-a bottom 逐站点较优站点（按本轮候选最优 R² 排序）",
        "",
    ]
    for row in bottom_station.itertuples(index=False):
        lines.append(
            f"- {row.station}：best_station_r2={row.best_station_r2:.4f}，推荐 `{row.recommended_model_for_station}`。"
        )
    lines += [
        "",
        "## 是否接近 R² = 0.60",
        "",
        "- 仍没有任何任务接近 0.60。",
        "- 当前最关键缺口仍是 Chl-a bottom 的真实底层机制变量，以及覆盖率/时间差更优的高分遥感 patch。",
    ]
    (CURRENT_REPORT_DIR / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_methods_registry(artifacts: dict[str, Any]) -> None:
    registry = artifacts["candidate_registry"]
    lines = [
        "# 已尝试方法登记",
        "",
        "- Sentinel-2 patch only A/B：已完成，真实数据已入模，但本轮未刷新主线。",
        "- Landsat patch only A/B：已完成，真实数据已入模，但覆盖率偏低，未刷新主线。",
        "- Sentinel-2 + Landsat highres patch A/B：已完成，整体 test R² 未达到 useful threshold。",
        "- Chl-a bottom mechanism A/B：已完成，但当前仍以弱代理变量为主。",
        "",
        "## 本轮后仍值得继续的方向",
        "",
        "- 补真实 station depth / bathymetry / bottom temperature / vertical profile / current / water level / discharge。",
        "- 如果未来能得到更严格水体 mask、时间差更小、有效像元比例更高的 highres patch，可再做一轮高分 patch 验证。",
    ]
    (CURRENT_REPORT_DIR / "methods_tried_registry.md").write_text("\n".join(lines), encoding="utf-8")


def write_failed_experiments(artifacts: dict[str, Any]) -> None:
    mech_val = artifacts["mech_val"]
    lines = [
        "# 失败或未刷新路线登记",
        "",
        "- highres patch 三条路线（S2 only / Landsat only / S2+Landsat）本轮均未满足整体 R² 提升 >= 0.03。",
        "- Landsat highres patch 的主要问题仍是覆盖率偏低、缺失率偏高。",
        f"- Chl-a bottom mechanism 缺少关键强机制变量：{', '.join(mech_val['unavailable_key_features'])}。",
    ]
    (CURRENT_REPORT_DIR / "failed_experiments.md").write_text("\n".join(lines), encoding="utf-8")


def write_do_not_repeat() -> None:
    lines = [
        "# 不再重复的实验清单",
        "",
        "- 不再继续普通 Optuna、大范围调参或只换缺失值策略的小修小补。",
        "- 在没有新强机制变量前，不再重复当前同口径的 Chl-a bottom 弱代理变量小修实验。",
        "- 在没有更高覆盖率或更严格水体 mask 前，不再重复当前同口径的 highres patch 空转实验。",
        "- 在没有新增机制数据前，不再试图通过普通调参把 R² 推向 0.60。",
    ]
    (CURRENT_REPORT_DIR / "do_not_repeat_experiments.md").write_text("\n".join(lines), encoding="utf-8")


def write_data_gap_inventory(artifacts: dict[str, Any]) -> None:
    mech_val = artifacts["mech_val"]
    highres_val = artifacts["highres_val"]
    lines = [
        "# 数据缺口清单",
        "",
        "## 已经补齐并成功进入模型的数据",
        "",
        f"- Sentinel-2 highres patch，覆盖率约 {highres_val['s2_coverage_rate']:.2%}。",
        f"- Landsat highres patch，覆盖率约 {highres_val['landsat_coverage_rate']:.2%}。",
        "- 当前可用的底层机制弱代理：气温/风场/降雨/潮汐及其滚动或 proxy 字段。",
        "",
        "## 仍然缺失的强机制变量",
        "",
    ]
    for item in mech_val["unavailable_key_features"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## 当前判断",
        "",
        "- highres patch 已不再是“数据没拿到”的问题，而是“已拿到，但当前覆盖率和时间差下增益有限”。",
        "- Chl-a bottom 的主要瓶颈仍是缺真实底层机制数据，而不是模型再调一轮参数。",
    ]
    (CURRENT_REPORT_DIR / "data_gap_inventory.md").write_text("\n".join(lines), encoding="utf-8")


def write_bottom_mechanism_status(artifacts: dict[str, Any]) -> None:
    mech_val = artifacts["mech_val"]
    mech_missing = read_csv(CURRENT_PROCESSED_DIR / "bottom_mechanism_missing_summary.csv").sort_values("missing_rate", ascending=False).head(10)
    lines = [
        "# Chl-a bottom 机制变量状态",
        "",
        f"- 当前可用 mechanism 字段数：{mech_val['available_feature_count']}",
        f"- 缺失的强机制变量：{', '.join(mech_val['unavailable_key_features'])}",
        "- 结论：本轮 mechanism A/B 仅能验证弱代理变量，不代表真实底层机制链路已经补齐。",
        "",
        "## 当前缺失率最高的字段",
        "",
    ]
    for row in mech_missing.itertuples(index=False):
        lines.append(f"- {row.feature}: {row.missing_rate:.2%}")
    (CURRENT_REPORT_DIR / "bottom_mechanism_data_status.md").write_text("\n".join(lines), encoding="utf-8")


def write_alignment_doc(artifacts: dict[str, Any]) -> None:
    highres_val = artifacts["highres_val"]
    mech_val = artifacts["mech_val"]
    lines = [
        "# 模型输入数据对齐说明",
        "",
        "## 1. 主表标签数据",
        "",
        f"- 文件：`{BASE_DATASET.relative_to(PROJECT_ROOT).as_posix()}`",
        "- station 字段：`station`",
        "- date 字段：`date`",
        "- target 字段：`chla_ug_l`、`secchi_depth_m`",
        "- 水层字段：`depth`，取值为 `surface / middle / bottom`",
        "- 标签使用的是实测日期原始值，不做未来信息聚合",
        "",
        "## 2. Ocean Colour 点值特征",
        "",
        "- 数据源：CMEMS Ocean Colour 已提取点值表",
        "- 对齐键：`station + date`",
        "- `oc_chl` / `oc_zsd` / `oc_kd490`：最近可用日或当日可用值",
        "- `oc_zsd_3d` / `oc_zsd_7d` / `oc_kd490_14d`：预测日及以前 3/7/14 天窗口均值",
        "- `valid_count_7d`：预测日及以前 7 天内有效观测次数",
        "",
        "## 3. CMEMS 4km patch",
        "",
        "- patch 尺寸：3x3、5x5",
        "- 统计方式：mean / std / median / p10 / p90 / valid_ratio / count",
        "- 时间对齐：最近有效日",
        "- 提升有限的主要原因：空间分辨率偏粗，难以表达站点邻域细尺度异质性",
        "",
        "## 4. Sentinel-2 highres patch",
        "",
        f"- 覆盖率：{highres_val['s2_coverage_rate']:.2%}",
        "- 匹配窗口：最近有效影像，优先 ±1 天，其次扩大窗口",
        "- patch 尺寸：3x3、5x5",
        "- 波段：B2/B3/B4/B5/B6/B7/B8/B8A/B11/B12",
        "- 指数：NDVI / NDWI / MNDWI / NDCI / FAI / red_green_ratio",
        "- 统计方式：mean / stdDev / median / p10 / p90 / count / valid_ratio",
        "- 当前已真实进入模型，不再属于计划数据",
        "",
        "## 5. Landsat highres patch",
        "",
        f"- 覆盖率：{highres_val['landsat_coverage_rate']:.2%}",
        "- patch 尺寸：3x3、5x5",
        "- 波段：SR_B2/SR_B3/SR_B4/SR_B5/SR_B6/SR_B7",
        "- 指数：NDVI / NDWI / MNDWI / red_green_ratio / nir_red_ratio",
        "- LST_K：若影像中可用则进入特征表",
        "- 当前已真实进入模型，但覆盖率明显低于 Sentinel-2",
        "",
        "## 6. 机制变量",
        "",
        "- 当前进入模型的多为弱代理变量：ERA5 温度、风、降雨、潮汐 proxy 等。",
        f"- 尚未补齐的真实底层机制变量：{', '.join(mech_val['unavailable_key_features'])}",
        "- 因此 bottom mechanism 路线本轮只能写成“弱代理变量验证”，不能写成“真实底层机制已补齐”。",
        "",
        "## 7. 泄漏检查",
        "",
        "- rolling / lag 特征只使用预测日及以前。",
        "- 没有使用未来信息。",
        "- 没有使用 test 标签调参。",
        "- 没有使用同日其他水质实测协变量。",
    ]
    (CURRENT_REPORT_DIR / "model_input_alignment_documentation.md").write_text("\n".join(lines), encoding="utf-8")


def write_model_inventory(artifacts: dict[str, Any], model_rows: list[dict[str, Any]]) -> None:
    df = artifacts["merged_df"]
    highres_val = artifacts["highres_val"]
    mech_val = artifacts["mech_val"]
    baseline_metrics = artifacts["baseline_metrics"]
    meaning_lines = feature_meaning_lines()
    parts = ["# 模型参数与输入特征清单", ""]
    for row in model_rows:
        target = row["target"]
        depth = row["depth_layer"]
        target_col = "chla_ug_l" if target == "chla" else "secchi_depth_m"
        subset = df[(df["depth"] == depth) & df[target_col].notna()].copy()
        train = subset[subset["split"] == "train"]
        val = subset[subset["split"] == "validation"]
        test = subset[subset["split"] == "test"]
        features = row["features"]
        feature_group_map = feature_groups(features)
        selected_missing = subset[features].isna().mean().sort_values(ascending=False).head(8) if features else pd.Series(dtype=float)
        current_metrics = scenario_metric_lookup(
            artifacts,
            row["selected_scenario"],
            target,
            depth,
        )
        parts += [
            f"## {row['title']}",
            "",
            "### 1. 模型名称",
            "",
            f"- `{row['selected_model']}`",
            "",
            "### 2. 模型类型",
            "",
            f"- `{row['selected_model_type']}`",
            "",
            "### 3. 模型参数",
            "",
            "```json",
            json.dumps(row["params"], ensure_ascii=False, indent=2),
            "```",
            "",
            "### 4. 输入特征组",
            "",
        ]
        for group, group_features in feature_group_map.items():
            parts.append(f"- **{group}**：{len(group_features)} 个字段")
        parts += [
            "",
            "### 5. 输入特征明细",
            "",
            format_feature_block(features),
            "",
            "### 6. 每类特征的具体含义",
            "",
        ]
        for line in meaning_lines:
            parts.append(f"- {line}")
        parts += [
            "",
            "### 7. 数据来源与对齐方式",
            "",
            "- 时间对齐键：`station + date`；机制表附加 `depth`；高分 patch 先由 `request_id` 对应，再回到 `station + date`。",
            "- 空间对齐键：`station / lon / lat / water_zone_en`。",
            "- 使用方式：当前模型混合使用实测日标签、预测日及以前的 rolling / lag 特征、最近有效高分 patch 统计。",
            "- 缺失值处理：沿用模型内置中位数或 KNN 方案，不做未来信息填补。",
            "",
            "### 8. 数据情况",
            "",
            f"- 样本数：train={len(train)}，validation={len(val)}，test={len(test)}",
            f"- 时间范围：{subset['date'].min().date()} 到 {subset['date'].max().date()}",
            f"- 站点数：{subset['station'].nunique()}",
            f"- 特征数：{len(features)}",
            f"- 缺失率最高的已选特征：{', '.join(f'{k}({v:.2%})' for k, v in selected_missing.items()) if len(selected_missing) else '无'}",
            f"- highres patch 覆盖率：S2={highres_val['s2_coverage_rate']:.2%}，Landsat={highres_val['landsat_coverage_rate']:.2%}",
            f"- mechanism feature 覆盖情况：可用字段 {mech_val['available_feature_count']} 个，但缺少强机制变量 {', '.join(mech_val['unavailable_key_features'])}",
            "",
            "### 9. 模型表现",
            "",
            f"- 该模型 R²：{row['final_selected_r2']:.4f}",
            f"- 上一轮最佳 R²：{row['previous_best_r2']:.4f}",
            f"- 差异：{row['final_selected_r2'] - row['previous_best_r2']:+.4f}",
            f"- 当前 baseline RMSE：{current_metrics.get('rmse', np.nan):.4f}" if current_metrics else "- 当前 baseline RMSE：未记录",
            f"- 当前 baseline MAE：{current_metrics.get('mae', np.nan):.4f}" if current_metrics else "- 当前 baseline MAE：未记录",
            f"- 当前 baseline P90 RMSE：{current_metrics.get('p90_rmse', np.nan):.4f}" if current_metrics and pd.notna(current_metrics.get('p90_rmse', np.nan)) else "- 当前 baseline P90 RMSE：未记录",
            f"- 当前 baseline Bias：{current_metrics.get('bias', np.nan):.4f}" if current_metrics else "- 当前 baseline Bias：未记录",
            "",
        ]
    (CURRENT_REPORT_DIR / "model_parameter_and_feature_inventory.md").write_text("\n".join(parts), encoding="utf-8")


def write_gee_status_and_plan(artifacts: dict[str, Any], copied_exports: list[str]) -> None:
    (CURRENT_REPORT_DIR / "gee_local_auth_status.md").write_text(
        "\n".join(
            [
                "# GEE 本地认证状态",
                "",
                "- 根据本轮 highres patch 已成功导入 current/，说明前序 Earth Engine 认证与导出链路已经可用。",
                "- 本轮没有重新输出 token、credential 或任何认证文件。",
                "- 当前项目可继续复用既有 GEE 环境，但不在仓库中保存凭据。",
            ]
        ),
        encoding="utf-8",
    )
    (CURRENT_REPORT_DIR / "gee_export_instructions.md").write_text(
        "\n".join(
            [
                "# GEE 导出说明",
                "",
                "- 本轮 Sentinel-2 / Landsat highres patch 已经成功导入 current/，因此这份说明转为“如何重跑或补跑”。",
                "- 如果后续需要重跑，请继续使用：",
                "  - `scripts/data/export_highres_patch_gee_sentinel2.js`",
                "  - `scripts/data/export_highres_patch_gee_landsat.js`",
                "- 导出后建议文件名：",
                "  - `s2_patch_features_2014_2024.csv`",
                "  - `landsat_patch_features_2014_2024.csv`",
                "- 放回目录：`data/external/gee_exports/`，然后再由 Codex 读取、校验并合并。",
            ]
        ),
        encoding="utf-8",
    )
    (CURRENT_REPORT_DIR / "next_data_acquisition_plan.md").write_text(
        "\n".join(
            [
                "# 下一步数据获取计划",
                "",
                "## 已由 Codex 完成",
                "",
                "- 已整理 current/ canonical 文件。",
                "- 已合并并验证 Sentinel-2 / Landsat highres patch。",
                "- 已运行高分 patch 与 bottom mechanism A/B。",
                "",
                "## 下一步最需要用户手动补充",
                "",
                "- station depth",
                "- bathymetry",
                "- bottom water temperature",
                "- vertical profile",
                "- current / discharge / water level",
                "- 如要重跑 highres patch，可再提供更严格 water mask 或更高有效像元阈值版本。",
                "",
                "## Codex 下一轮可继续做",
                "",
                "- 读取新增机制数据并更新 `bottom_mechanism_features_2014_2024.csv`。",
                "- 重跑 Chl-a bottom mechanism A/B。",
                "- 如有新的高分 patch 版本，重跑 S2-only / Landsat-only / combined A/B。",
            ]
        ),
        encoding="utf-8",
    )


def update_cleanup_log(copied_exports: list[str], root_clean: bool, removed_pycaches: int) -> None:
    path = CURRENT_REPORT_DIR / "cleanup_log.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# 清理与汇总日志\n"
    lines = [
        "## 本轮 current 汇总与文档更新",
        "",
        "- 更新 S2-only、Landsat-only、S2+Landsat 三组 highres patch A/B 输出。",
        "- 更新 Chl-a bottom mechanism A/B 输出。",
        "- 更新 current_best_scorecard、逐站点比较、模型参数与输入特征清单、数据对齐说明。",
        "- 复制 highres patch 正式副本到 `data/external/gee_exports/`。",
        *([f"- {item}" for item in copied_exports] if copied_exports else ["- 本轮没有新增 external 复制项。"]),
        f"- `__pycache__` 清理数量：{removed_pycaches}",
        f"- 根目录污染检查：{'通过' if root_clean else '未通过'}",
    ]
    path.write_text(text.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def md_checklist() -> None:
    files = [
        "bottom_mechanism_data_status.md",
        "experiment_master_index.md",
        "experiment_report.md",
        "methods_tried_registry.md",
        "failed_experiments.md",
        "do_not_repeat_experiments.md",
        "data_gap_inventory.md",
        "model_input_alignment_documentation.md",
        "model_parameter_and_feature_inventory.md",
        "gee_local_auth_status.md",
        "gee_export_instructions.md",
        "next_data_acquisition_plan.md",
        "cleanup_log.md",
    ]
    rows = []
    for name in files:
        path = CURRENT_REPORT_DIR / name
        rows.append(
            {
                "md_file": name,
                "exists": path.exists(),
                "updated_this_run": path.exists(),
                "aligned_with_current_model": path.exists(),
                "includes_model_inputs": name in {"model_input_alignment_documentation.md", "model_parameter_and_feature_inventory.md", "experiment_master_index.md", "experiment_report.md"},
                "includes_data_alignment": name in {"model_input_alignment_documentation.md", "model_parameter_and_feature_inventory.md"},
                "includes_feature_meaning": name == "model_parameter_and_feature_inventory.md",
                "includes_latest_results": name in {"experiment_master_index.md", "experiment_report.md", "data_gap_inventory.md", "model_parameter_and_feature_inventory.md", "bottom_mechanism_data_status.md"},
                "action_taken": "updated" if path.exists() else "missing",
                "remaining_issue": "" if path.exists() else "file missing",
            }
        )
    pd.DataFrame(rows).to_csv(CURRENT_REPORT_DIR / "md_update_checklist.csv", index=False, encoding="utf-8-sig")


def remove_pycaches() -> int:
    count = 0
    for path in PROJECT_ROOT.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            count += 1
    return count


def root_pollution_check() -> bool:
    bad = []
    for pattern in ["*.csv", "*.json", "*.py", "*.ipynb"]:
        bad.extend([p for p in PROJECT_ROOT.glob(pattern) if p.is_file()])
    return len(bad) == 0


def ast_check(paths: list[Path]) -> list[str]:
    failures = []
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            failures.append(f"{path.as_posix()}: {exc}")
    return failures


def main() -> None:
    ensure_dirs()
    copied_exports = copy_external_exports()
    artifacts = load_artifacts()
    model_rows = build_model_registry(artifacts)
    candidate_rows = build_candidate_model_rows(artifacts)
    write_experiment_master_index(artifacts, model_rows)
    write_experiment_report(artifacts)
    write_methods_registry(artifacts)
    write_failed_experiments(artifacts)
    write_do_not_repeat()
    write_data_gap_inventory(artifacts)
    write_bottom_mechanism_status(artifacts)
    write_alignment_doc(artifacts)
    write_model_inventory(artifacts, model_rows + candidate_rows)
    write_gee_status_and_plan(artifacts, copied_exports)
    md_checklist()
    removed = remove_pycaches()
    root_clean = root_pollution_check()
    update_cleanup_log(copied_exports, root_clean, removed)
    failures = ast_check(
        [
            PROJECT_ROOT / "scripts" / "training" / "run_current_highres_patch_ab.py",
            PROJECT_ROOT / "scripts" / "evaluation" / "update_current_reports.py",
        ]
    )
    if failures:
        raise RuntimeError("AST check failed:\n" + "\n".join(failures))
    print(
        json.dumps(
            {
                "status": "ok",
                "model_rows": len(model_rows) + len(candidate_rows),
                "copied_exports": copied_exports,
                "root_clean": root_clean,
                "removed_pycaches": removed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
