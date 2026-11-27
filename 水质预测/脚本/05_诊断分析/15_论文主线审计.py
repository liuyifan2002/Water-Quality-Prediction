"""Audit existing evidence for the thesis storyline without changing prior experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
RESULTS = ROOT / "结果" / "扩展实验"
DEFAULT_OUTPUT = RESULTS / "论文主线审计_20260916"
CANDIDATE_ROOT = RESULTS / "候选模型稳健性验证_20260914" / "回传结果_20260914"
ZONE_RANDOM_ROOT = RESULTS / "分海区建模_20260906"
ZONE_STRICT_ROOT = RESULTS / "分海区严格验证_20260906"
ZONE_CV_ROOT = RESULTS / "分海区五折留站验证_20260906"
REPRO_CANDIDATE_ROOT = DEFAULT_OUTPUT / "复现检查_XGBoost_候选模型"
REPRO_SAVED_SPLIT_ROOT = DEFAULT_OUTPUT / "复现检查_XGBoost_保存划分"
REPRO_TP_ROOT = DEFAULT_OUTPUT / "复现检查_XGBoost_TP中层"

TARGETS = {
    "chla": "chla_ug_l",
    "secchi": "secchi_depth_m",
    "tn": "tn_mg_l",
    "tp": "tp_mg_l",
}
DEPTHS = ["surface", "middle", "bottom"]
MODEL_ORDER = ["xgboost", "tabm", "tabpfn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit current paper-storyline evidence.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def task_label(target: str, depth: str) -> str:
    return f"{target}_{depth}"


def markdown_table(data: pd.DataFrame) -> str:
    if data.empty:
        return "无记录。"
    columns = list(data.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in data.fillna("").astype(str).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join([header, divider, *rows])


def inventory_dataset(data: pd.DataFrame, source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    targets: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for key, column in TARGETS.items():
        for depth in DEPTHS:
            subset = data.loc[data["depth"].eq(depth) & data[column].notna()].copy()
            targets.append(
                {
                    "任务": task_label(key, depth),
                    "指标": key,
                    "水层": depth,
                    "目标列": column,
                    "样本数": len(subset),
                    "站点数": subset["station"].nunique(),
                    "最早日期": subset["date"].min().date().isoformat() if not subset.empty else "",
                    "最晚日期": subset["date"].max().date().isoformat() if not subset.empty else "",
                    "缺失率": float(data.loc[data["depth"].eq(depth), column].isna().mean()),
                    "最小值": float(subset[column].min()) if not subset.empty else np.nan,
                    "最大值": float(subset[column].max()) if not subset.empty else np.nan,
                }
            )
            for station, group in subset.groupby("station", sort=True):
                before = group.loc[group["year"] <= 2022]
                future = group.loc[group["year"].between(2023, 2024)]
                coverage.append(
                    {
                        "任务": task_label(key, depth),
                        "指标": key,
                        "水层": depth,
                        "station": station,
                        "water_zone_en": group["water_zone_en"].iloc[0],
                        "适配前可用历史数_2014_2022": len(before),
                        "测试数_2023_2024": len(future),
                        "最早历史日期": before["date"].min().date().isoformat() if not before.empty else "",
                        "最晚测试日期": future["date"].max().date().isoformat() if not future.empty else "",
                        "可做24条适配且测试至少12条": bool(len(before) >= 24 and len(future) >= 12),
                    }
                )

    secchi = data[["station", "date", "depth", "secchi_depth_m"]].dropna()
    secchi_wide = secchi.pivot_table(
        index=["station", "date"], columns="depth", values="secchi_depth_m", aggfunc="first"
    )
    complete = secchi_wide.dropna(subset=DEPTHS)
    secchi_duplicate_rate = float(
        ((complete["surface"] == complete["middle"]) & (complete["surface"] == complete["bottom"])).mean()
    ) if not complete.empty else np.nan
    overview = {
        "主表路径": str(source),
        "行数": len(data),
        "列数": len(data.columns),
        "站点数": int(data["station"].nunique()),
        "水区数": int(data["water_zone_en"].nunique()),
        "最早日期": data["date"].min().date().isoformat(),
        "最晚日期": data["date"].max().date().isoformat(),
        "obs_id重复数": int(data["obs_id"].duplicated().sum()),
        "站点日期水层重复数": int(data.duplicated(["station", "date", "depth"]).sum()),
        "透明度三层同时存在的站点日期数": int(len(complete)),
        "透明度三层完全一致比例": secchi_duplicate_rate,
    }
    return pd.DataFrame(targets), pd.DataFrame(coverage), overview


def candidate_registry() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(CANDIDATE_ROOT.rglob("model_metrics.csv")):
        data = csv(path)
        if not {"evaluation_mode", "target", "depth", "model", "r2"}.issubset(data.columns):
            continue
        keep = [
            "evaluation_mode", "split_id", "fold", "seed", "target", "depth", "model", "r2", "rmse", "mae",
            "target_history_features", "same_day_water_quality_covariates", "status",
        ]
        current = data[[column for column in keep if column in data.columns]].copy()
        current["source_file"] = str(path)
        current["result_family"] = "候选模型稳健性验证"
        current["reproducibility_status"] = "待本机复现XGBoost"
        rows.append(current)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_candidate_evidence(registry: pd.DataFrame) -> pd.DataFrame:
    expected = pd.MultiIndex.from_product(
        [list(TARGETS), DEPTHS, MODEL_ORDER], names=["target", "depth", "model"]
    ).to_frame(index=False)
    if registry.empty:
        return expected

    source = registry.loc[registry["status"].eq("ok")].copy()
    wide_parts: list[pd.DataFrame] = []
    mode_mapping = {
        "random_seeds": "随机",
        "time_holdout": "年份留出",
        "station_5fold": "五折留站",
    }
    for raw_mode, display_mode in mode_mapping.items():
        subset = source.loc[source["evaluation_mode"].eq(raw_mode)]
        if subset.empty:
            continue
        grouped = subset.groupby(["target", "depth", "model"], as_index=False).agg(
            **{
                f"R2_{display_mode}": ("r2", "mean"),
                f"RMSE_{display_mode}": ("rmse", "mean"),
                f"MAE_{display_mode}": ("mae", "mean"),
                f"{display_mode}重复次数": ("r2", "count"),
                f"{display_mode}目标历史": ("target_history_features", lambda x: ";".join(sorted(set(x.astype(str))))),
                f"{display_mode}来源": ("source_file", lambda x: ";".join(sorted(set(x)))),
            }
        )
        wide_parts.append(grouped)

    evidence = expected.copy()
    for part in wide_parts:
        evidence = evidence.merge(part, on=["target", "depth", "model"], how="left")

    for score, rank_column, best_column in [
        ("R2_随机", "随机排名", "随机最佳模型"),
        ("R2_年份留出", "年份留出排名", "年份留出最佳模型"),
        ("R2_五折留站", "五折留站排名", "五折留站最佳模型"),
    ]:
        if score not in evidence:
            evidence[rank_column] = np.nan
            evidence[best_column] = "无结果"
            continue
        evidence[rank_column] = evidence.groupby(["target", "depth"])[score].rank(ascending=False, method="min")
        best = evidence.loc[evidence[rank_column].eq(1), ["target", "depth", "model"]].rename(columns={"model": best_column})
        evidence = evidence.merge(best, on=["target", "depth"], how="left")
        evidence[best_column] = evidence[best_column].fillna("无结果")

    strict_comparable = (
        evidence.get("随机目标历史", pd.Series(index=evidence.index, dtype=object)).eq("excluded")
        & evidence.get("年份留出目标历史", pd.Series(index=evidence.index, dtype=object)).eq("excluded")
        & evidence.get("五折留站目标历史", pd.Series(index=evidence.index, dtype=object)).eq("excluded")
    )
    evidence["时间泛化差距"] = np.where(strict_comparable, evidence["R2_随机"] - evidence["R2_年份留出"], np.nan)
    evidence["空间泛化差距"] = np.where(strict_comparable, evidence["R2_随机"] - evidence["R2_五折留站"], np.nan)
    evidence["差距可比"] = np.where(
        strict_comparable,
        "是",
        "否：随机结果含目标自身历史项，严格验证已剔除，不能把差值解释为纯分布变化损失",
    )
    evidence["结果来源"] = evidence.apply(
        lambda row: " | ".join(
            str(row.get(column, "")) for column in ["随机来源", "年份留出来源", "五折留站来源"] if pd.notna(row.get(column, np.nan))
        ),
        axis=1,
    )
    evidence["可重复性"] = "待复现"
    evidence["备注"] = np.where(
        evidence["target"].eq("secchi"),
        "透明度三层原始值完全重复；只能作为同一透明度目标的重复计算检查，不能解释为三层独立观测。",
        "",
    )
    return evidence


def regionalization_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def append_paired_metrics(path: Path, validation: str, strict: bool) -> None:
        data = csv(path)
        data = data.loc[data["water_zone_en"].eq("all")].copy()
        if "evaluation_mode" in data.columns:
            mode_to_name = {
                "time_holdout_2023_2024": "年份留出",
                "held_station": "留站点",
            }
            groups = data.groupby("evaluation_mode", dropna=False)
        else:
            groups = [(validation, data)]
        for raw_mode, group in groups:
            label = mode_to_name.get(raw_mode, raw_mode) if "evaluation_mode" in data.columns else validation
            for (target, depth), task in group.groupby(["target", "depth"], sort=True):
                global_row = task.loc[task["model_variant"].str.startswith("global_")]
                zone_row = task.loc[task["model_variant"].str.startswith("zone_aware_")]
                if len(global_row) != 1 or len(zone_row) != 1:
                    continue
                global_row = global_row.iloc[0]
                zone_row = zone_row.iloc[0]
                records.append(
                    {
                        "验证方式": label,
                        "target": target,
                        "depth": depth,
                        "model": "xgboost",
                        "global_r2": global_row["r2"],
                        "regional_r2": zone_row["r2"],
                        "delta_r2": zone_row["r2"] - global_row["r2"],
                        "global_rmse": global_row["rmse"],
                        "regional_rmse": zone_row["rmse"],
                        "delta_rmse": zone_row["rmse"] - global_row["rmse"],
                        "global_mae": global_row["mae"],
                        "regional_mae": zone_row["mae"],
                        "样本数": global_row.get("rows", np.nan),
                        "特征数": global_row.get("feature_count", np.nan),
                        "来源文件": str(path),
                        "可比性": "是：同一划分与同一特征组" if strict else "随机结果：同一划分，但含目标历史项",
                    }
                )

    for path in sorted(ZONE_RANDOM_ROOT.rglob("分海区建模_指标.csv")):
        append_paired_metrics(path, "随机80_20", strict=False)

    strict_path = ZONE_STRICT_ROOT / "分海区严格验证_指标.csv"
    if strict_path.exists():
        append_paired_metrics(strict_path, "严格验证", strict=True)

    cv_path = ZONE_CV_ROOT / "五折留站点验证_汇总.csv"
    if cv_path.exists():
        for row in csv(cv_path).itertuples(index=False):
            records.append(
                {
                    "验证方式": "五折留站",
                    "target": row.target,
                    "depth": row.depth,
                    "model": "xgboost",
                    "global_r2": row.global_r2_pooled,
                    "regional_r2": row.zone_r2_pooled,
                    "delta_r2": row.delta_r2_pooled,
                    "global_rmse": row.global_rmse_pooled,
                    "regional_rmse": row.zone_rmse_pooled,
                    "delta_rmse": row.zone_rmse_pooled - row.global_rmse_pooled,
                    "global_mae": np.nan,
                    "regional_mae": np.nan,
                    "样本数": np.nan,
                    "特征数": np.nan,
                    "来源文件": str(cv_path),
                    "可比性": "是：同一折、同一严格外部特征组",
                }
            )
    return pd.DataFrame(records)


def reproducibility_status(delta: float) -> str:
    absolute = abs(delta)
    if absolute <= 0.01:
        return "通过"
    if absolute <= 0.03:
        return "可接受：环境版本差异"
    if absolute <= 0.05:
        return "需解释"
    return "失败"


def reproducibility_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    exact_saved_split = REPRO_SAVED_SPLIT_ROOT / "保存划分复现结果.csv"
    if exact_saved_split.exists():
        exact = csv(exact_saved_split)
        for item in exact.itertuples(index=False):
            rows.append(
                {
                    "来源实验": "候选模型XGBoost（保存划分）",
                    "验证方式": item.evaluation_mode,
                    "任务": task_label(item.target, item.depth),
                    "模型": item.model,
                    "划分": item.split_id,
                    "保存R2": item.saved_r2,
                    "复现R2": item.reproduced_r2,
                    "R2绝对差": item.abs_delta_r2,
                    "结论": item.status,
                    "说明": "直接读取保存的 obs_id 划分；未重新生成随机或留站折。",
                }
            )
    else:
        # This fallback regenerates split assignments and is only a diagnostic
        # of environment sensitivity. It is not treated as exact reproduction.
        saved_candidate = {
            path.parent.name.removeprefix("outputs_"): csv(path)
            for path in CANDIDATE_ROOT.rglob("model_metrics.csv")
        }
        reproduced_candidate = REPRO_CANDIDATE_ROOT / "model_metrics.csv"
        if reproduced_candidate.exists():
            reproduced = csv(reproduced_candidate)
            for mode, old in saved_candidate.items():
                old = old.loc[old["model"].eq("xgboost")]
                new = reproduced.loc[reproduced["evaluation_mode"].eq(mode) & reproduced["model"].eq("xgboost")]
                keys = ["evaluation_mode", "split_id", "target", "depth", "model"]
                joined = old.merge(new, on=keys, suffixes=("_保存", "_复现"))
                for item in joined.itertuples(index=False):
                    delta = float(item.r2_复现 - item.r2_保存)
                    rows.append(
                        {
                            "来源实验": "候选模型XGBoost（重建划分，仅诊断）",
                            "验证方式": item.evaluation_mode,
                            "任务": task_label(item.target, item.depth),
                            "模型": "xgboost",
                            "划分": item.split_id,
                            "保存R2": item.r2_保存,
                            "复现R2": item.r2_复现,
                            "R2绝对差": abs(delta),
                            "结论": "不可作为精确复现",
                            "说明": "重建 GroupKFold 与保存划分不一致，仅用于提示版本或数据排序风险。",
                        }
                    )

    saved_tp = ZONE_STRICT_ROOT / "分海区严格验证_指标.csv"
    reproduced_tp = REPRO_TP_ROOT / "分海区严格验证_指标.csv"
    if saved_tp.exists() and reproduced_tp.exists():
        old = csv(saved_tp)
        new = csv(reproduced_tp)
        old = old.loc[(old["target"].eq("tp")) & (old["depth"].eq("middle")) & old["water_zone_en"].eq("all")]
        new = new.loc[(new["target"].eq("tp")) & (new["depth"].eq("middle")) & new["water_zone_en"].eq("all")]
        keys = ["evaluation_mode", "target", "depth", "model_variant", "water_zone_en"]
        joined = old.merge(new, on=keys, suffixes=("_保存", "_复现"))
        for item in joined.itertuples(index=False):
            delta = float(item.r2_复现 - item.r2_保存)
            rows.append(
                {
                    "来源实验": "严格分海区XGBoost",
                    "验证方式": item.evaluation_mode,
                    "任务": task_label(item.target, item.depth),
                    "模型": item.model_variant,
                    "划分": "固定年份留出或固定留站点",
                    "保存R2": item.r2_保存,
                    "复现R2": item.r2_复现,
                    "R2绝对差": abs(delta),
                    "结论": reproducibility_status(delta),
                    "说明": "同一脚本、同一主表、同一参数。",
                }
            )
    return pd.DataFrame(rows)


def write_inventory_markdown(
    output: Path,
    overview: dict[str, object],
    target_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    registry: pd.DataFrame,
    regional: pd.DataFrame,
    reproducibility: pd.DataFrame,
) -> None:
    zone_files = [
        "脚本/04_模型训练/28_run_zone_specific_full_station_ab.py",
        "脚本/04_模型训练/29_validate_zone_model_time_spatial.py",
        "脚本/04_模型训练/30_cross_validate_zone_model_spatial.py",
    ]
    candidate_files = [
        "脚本/04_模型训练/31_run_tabular_pretrained_ab.py",
        "脚本/04_模型训练/32_validate_candidate_tabular_models.py",
    ]
    text = ["# 项目资产清单", "", "## 主表", ""]
    text.extend([f"- {key}：{value}" for key, value in overview.items()])
    text.extend(["", "## 目标任务", "", markdown_table(target_summary), ""])
    text.extend([
        "## 已识别的关键脚本",
        "",
        "- 正式训练：`脚本/04_模型训练/05_train_full_station_models.py`",
        *[f"- 分海区：`{path}`" for path in zone_files],
        *[f"- 候选模型：`{path}`" for path in candidate_files],
        "- 解释分析：`脚本/05_诊断分析/14_plot_chla_surface_effect_curves.py`",
        "",
        "## 已识别的结果",
        "",
        f"- 候选模型指标记录：{len(registry)} 条。",
        f"- 分海区可比对照：{len(regional)} 条。",
        "- 候选模型有保存的随机三种子、年份留出和五折留站划分清单。",
        f"- 已完成的 XGBoost 复现记录：{len(reproducibility)} 条。",
        "",
        "## 已确认的方法风险",
        "",
        "1. 透明度三层同时存在的记录完全相同，不是三个独立的水层观测任务。",
        "2. 候选模型随机验证保留了目标自身历史项，年份留出和留站验证剔除了该项。因此现有随机 R2 与严格验证 R2 不能直接相减并称为泛化差距。",
        "3. 分海区随机结果也含目标历史项；年份留出和留站结果使用严格外部特征，需分开报告。",
        "4. 主训练脚本声明的部分高质量 S2/Landsat 特征当前主表不存在，实际训练可用特征必须以保存的特征清单为准。",
        "",
        "## 少量本地观测适配可行性",
        "",
        markdown_table(coverage.groupby(["指标", "水层"], as_index=False).agg(
            站点数=("station", "nunique"),
            可试验站点数=("可做24条适配且测试至少12条", "sum"),
            历史样本中位数=("适配前可用历史数_2014_2022", "median"),
            测试样本中位数=("测试数_2023_2024", "median"),
        )),
        "",
    ])
    (output / "01_项目资产清单.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = csv(args.dataset)
    data["date"] = pd.to_datetime(data["date"])
    target_summary, coverage, overview = inventory_dataset(data, args.dataset)
    registry = candidate_registry()
    evidence = aggregate_candidate_evidence(registry)
    regional = regionalization_rows()
    reproducibility = reproducibility_rows()

    target_summary.to_csv(args.output_dir / "目标变量与样本清单.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(args.output_dir / "站点时间覆盖清单.csv", index=False, encoding="utf-8-sig")
    registry.to_csv(args.output_dir / "结果登记.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(args.output_dir / "02_现有证据矩阵.csv", index=False, encoding="utf-8-sig")
    regional.to_csv(args.output_dir / "03_分海区模型对照.csv", index=False, encoding="utf-8-sig")
    reproducibility.to_csv(args.output_dir / "04_复现检查结果.csv", index=False, encoding="utf-8-sig")
    write_inventory_markdown(args.output_dir, overview, target_summary, coverage, registry, regional, reproducibility)

    print(f"审计输出目录：{args.output_dir}")
    print(f"候选模型结果记录：{len(registry)}")
    print(f"证据矩阵行数：{len(evidence)}")
    print(f"分海区对照行数：{len(regional)}")


if __name__ == "__main__":
    main()
