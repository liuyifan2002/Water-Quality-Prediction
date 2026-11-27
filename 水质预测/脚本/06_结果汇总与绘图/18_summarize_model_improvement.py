from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("结果/扩展实验/station_specific")
REPORT_PATH = Path("文档/论文材料/模型改进与科学问题说明.md")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def pct_change(new: float, old: float) -> float:
    if old == 0 or pd.isna(old):
        return np.nan
    return (old - new) / old * 100


def model_rank() -> pd.DataFrame:
    base = read_csv(OUT_DIR / "station_specific_model_comparison.csv")
    ar = read_csv(OUT_DIR / "station_specific_autoregressive_pooled_eval.csv")
    ar["family"] = "Station_specific_autoregressive"
    combined = pd.concat([base, ar], ignore_index=True, sort=False)
    combined = combined.sort_values(["target", "rmse"]).reset_index(drop=True)
    combined.to_csv(OUT_DIR / "station_specific_model_comparison_updated.csv", index=False, encoding="utf-8-sig")

    best = combined.sort_values(["target", "rmse"]).groupby("target", as_index=False).first()
    best.to_csv(OUT_DIR / "station_specific_best_model_overall_updated.csv", index=False, encoding="utf-8-sig")
    return combined


def improvement_table(combined: pd.DataFrame) -> pd.DataFrame:
    old_best = read_csv(OUT_DIR / "station_specific_best_model_overall.csv")
    new_best = read_csv(OUT_DIR / "station_specific_best_model_overall_updated.csv")
    rows = []
    for target in ["Chl_a", "SD", "TN", "TP"]:
        old = old_best[old_best["target"] == target].iloc[0]
        new = new_best[new_best["target"] == target].iloc[0]
        rows.append(
            {
                "target": target,
                "old_best_model": old["model"],
                "old_rmse": old["rmse"],
                "old_mae": old["mae"],
                "old_r2": old["r2"],
                "new_best_model": new["model"],
                "new_rmse": new["rmse"],
                "new_mae": new["mae"],
                "new_r2": new["r2"],
                "rmse_reduction_pct": pct_change(new["rmse"], old["rmse"]),
                "mae_reduction_pct": pct_change(new["mae"], old["mae"]),
                "r2_gain": new["r2"] - old["r2"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "station_specific_model_improvement.csv", index=False, encoding="utf-8-sig")
    return out


def plot_improvement(improvement: pd.DataFrame) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(improvement))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=160)
    ax.bar(x - width / 2, improvement["old_rmse"], width, label="Previous best", color="#7a869a")
    ax.bar(x + width / 2, improvement["new_rmse"], width, label="Updated best", color="#2b8cbe")
    ax.set_xticks(x)
    ax.set_xticklabels(improvement["target"])
    ax.set_ylabel("RMSE")
    ax.set_title("Station-specific model improvement")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for idx, row in improvement.iterrows():
        ax.text(idx + width / 2, row["new_rmse"], f"R2={row['new_r2']:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "station_specific_model_improvement_rmse.png")
    plt.close(fig)


def feature_group(name: str) -> str:
    lower = name.lower()
    if any(key in lower for key in ["chl_a_lag", "sd_lag", "tn_lag", "tp_lag", "roll3", "delta1"]):
        if lower.startswith(("chl_a_", "sd_", "tn_", "tp_")):
            return "target_history"
    if any(key in lower for key in ["blue", "green", "red", "nir", "swir", "ndvi", "ndwi", "mndwi", "ndti", "valid_count", "landsat"]):
        return "landsat"
    if any(key in lower for key in ["rain"]):
        return "rainfall"
    if any(key in lower for key in ["t2m", "u10", "v10", "tcc", "wind"]):
        return "meteorology"
    if any(key in lower for key in ["tidal"]):
        return "tide"
    if any(key in lower for key in ["ntl"]):
        return "nighttime_light"
    if any(key in lower for key in ["month", "season"]):
        return "season"
    return "other"


def summarize_importance() -> pd.DataFrame:
    imp = read_csv(OUT_DIR / "station_specific_autoregressive_top_features.csv")
    imp["feature_group"] = imp["feature"].map(feature_group)
    group = (
        imp.groupby(["target", "feature_group"], as_index=False)["importance"]
        .sum()
        .sort_values(["target", "importance"], ascending=[True, False])
    )
    group.to_csv(OUT_DIR / "station_specific_autoregressive_feature_group_importance.csv", index=False, encoding="utf-8-sig")
    return group


def plot_feature_groups(group_imp: pd.DataFrame) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pivot = group_imp.pivot_table(index="target", columns="feature_group", values="importance", aggfunc="sum").fillna(0)
    preferred = [
        "target_history",
        "landsat",
        "meteorology",
        "rainfall",
        "tide",
        "nighttime_light",
        "season",
        "other",
    ]
    cols = [c for c in preferred if c in pivot.columns]
    pivot = pivot.reindex(["Chl_a", "SD", "TN", "TP"])[cols]
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)

    colors = {
        "target_history": "#d95f0e",
        "landsat": "#2b8cbe",
        "meteorology": "#31a354",
        "rainfall": "#756bb1",
        "tide": "#636363",
        "nighttime_light": "#e6ab02",
        "season": "#a6761d",
        "other": "#bdbdbd",
    }
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=160)
    bottom = np.zeros(len(pivot_norm))
    x = np.arange(len(pivot_norm))
    for col in pivot_norm.columns:
        ax.bar(x, pivot_norm[col], bottom=bottom, label=col, color=colors.get(col, "#bdbdbd"))
        bottom += pivot_norm[col].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_norm.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of top-feature importance")
    ax.set_title("Main signal groups in autoregressive station models")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "station_specific_autoregressive_feature_groups.png", bbox_inches="tight")
    plt.close(fig)


def write_report(improvement: pd.DataFrame, group_imp: pd.DataFrame) -> None:
    final_models = improvement.copy()
    lines = [
        "# 模型改进与科学问题说明",
        "",
        "## 一、这一步做了什么改进",
        "",
        "在用户指出“不同站点不应混入同一个模型”后，本轮继续沿用站点独立建模框架，并新增自回归增强模型。具体做法是：每个站点、每个目标变量分别建模，在原有 F3 融合特征（月节律、气象、降水、潮汐、夜光、Landsat）基础上，加入目标变量自身的历史状态特征：上一期、上两期、前三期均值和上一期变化量。",
        "",
        "这个改进的含义是：原模型只回答“外源环境和遥感信息能解释多少水质变化”，新模型进一步回答“如果已知该站点最近一次或几次实测状态，能否显著提高短期预测”。因此它更接近监测业务中的滚动预测场景。",
        "",
        "## 二、模型改进结果",
        "",
        "| 目标变量 | 原最优模型 | 原 RMSE | 原 R2 | 更新后推荐模型 | 新 RMSE | 新 R2 | RMSE 降低 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in final_models.iterrows():
        lines.append(
            f"| {row['target']} | {row['old_best_model']} | {row['old_rmse']:.4f} | {row['old_r2']:.3f} | "
            f"{row['new_best_model']} | {row['new_rmse']:.4f} | {row['new_r2']:.3f} | {row['rmse_reduction_pct']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "主要结论：TN 和 TP 在加入历史水质状态后提升明显，说明营养盐浓度具有较强的站点记忆效应和连续性；Chl-a 与 SD 没有被自回归模型进一步改善，说明它们更容易受到短期藻类生长、悬浮泥沙、云/水面遥感质量和局地过程影响。",
            "",
            "## 三、目前推荐的模型使用策略",
            "",
            "- Chl-a：继续使用站点独立、调参后的 F3 融合 XGBoost。自回归增强没有明显改善，后续应重点加强遥感水色质量控制、短期天气窗口和异常藻华样本识别。",
            "- SD：继续使用站点独立 RandomForest。透明度与遥感波段/指数关系较直接，但也受浑浊水体和云筛选影响，树模型在小样本下更稳。",
            "- TN：使用站点独立自回归增强模型。TN 的 R2 从约 0.553 提升到约 0.695，是本轮最明显改进。",
            "- TP：使用站点独立自回归增强模型。TP 的 R2 从约 0.381 提升到约 0.483，说明上期营养盐状态对短期预测有实际贡献。",
            "",
            "## 四、本研究的研究意义",
            "",
            "1. 建立了一个面向香港近岸海域的最小可运行水质预测流程，把 EPD 实测水质、ERA5 气象、HKO 降水/潮汐、VIIRS 夜光和 Landsat 遥感特征合并到同一时空尺度。",
            "2. 从“全站点混合模型”修正为“站点独立模型”，更符合近岸水质受局地水动力、污染输入和海湾形态控制的事实，避免把空间差异错误地当成时间预测能力。",
            "3. 通过 F1 过程模型、F2 遥感模型、F3 融合模型和自回归增强模型，可以区分环境过程、遥感观测和历史水质状态各自的贡献。",
            "4. 结果表明不同水质指标的可预测性不同：营养盐 TN/TP 更受历史状态和陆源输入控制，透明度较容易被遥感解释，而 Chl-a 的短期波动更复杂，是后续研究的关键难点。",
            "",
            "## 五、能够回答的科学问题",
            "",
            "1. 在香港近岸不同代表站点，气象、降水、潮汐、夜光和 Landsat 遥感能否共同解释 2014-2024 年表层水质变化？",
            "2. 遥感特征相对于纯过程驱动变量，是否能提高 Chl-a、透明度、TN 和 TP 的预测能力？",
            "3. 不同水质变量的主要控制机制是否一致？例如 Chl-a 是否比 TN/TP 更难由月尺度外源变量稳定预测？",
            "4. 站点之间是否存在显著差异，使得站点独立模型比全站点混合模型更合理？",
            "5. 加入上期水质状态后，营养盐预测是否显著改善，从而证明水质系统存在可利用的短期记忆效应？",
            "",
            "## 六、对下一步工作的影响",
            "",
            "- 对 Chl-a：优先增加短期过程特征，如采样日前 7-14 天太阳辐射、风速极值、连续降水后滞后窗口，并检查异常藻华月份。",
            "- 对 SD：继续优化 Landsat 缺失补齐和水体像元质量筛选，必要时加入 Sentinel-2 作为 2015-2024 增强实验。",
            "- 对 TN/TP：保留自回归增强模型，并在报告中明确其适用前提是能够获得该站点近期实测水质。",
            "- 对成果表达：主报告应把站点独立模型作为正式结果，原全站点模型只作为早期基线或方法探索过程。",
            "",
        ]
    )

    lines.extend(["## 七、自回归增强模型的主要特征组信号", ""])
    for target in ["Chl_a", "SD", "TN", "TP"]:
        sub = group_imp[group_imp["target"] == target].head(5)
        text = "；".join([f"{r.feature_group}: {r.importance:.3f}" for r in sub.itertuples()])
        lines.append(f"- {target}：{text}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    combined = model_rank()
    improvement = improvement_table(combined)
    plot_improvement(improvement)
    group_imp = summarize_importance()
    plot_feature_groups(group_imp)
    write_report(improvement, group_imp)
    print(f"写出 {OUT_DIR / 'station_specific_model_comparison_updated.csv'}")
    print(f"写出 {OUT_DIR / 'station_specific_best_model_overall_updated.csv'}")
    print(f"写出 {OUT_DIR / 'station_specific_model_improvement.csv'}")
    print(f"写出 {OUT_DIR / 'figures' / 'station_specific_model_improvement_rmse.png'}")
    print(f"写出 {OUT_DIR / 'station_specific_autoregressive_feature_group_importance.csv'}")
    print(f"写出 {OUT_DIR / 'figures' / 'station_specific_autoregressive_feature_groups.png'}")
    print(f"写出 {REPORT_PATH}")


if __name__ == "__main__":
    main()
