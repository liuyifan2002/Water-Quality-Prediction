"""生成报告用论文辅助图表。

本脚本只读取已保存的正式汇总结果、现代模型对照结果、验证方式对照结果
和多源特征消融结果，不重新训练模型，也不改写正式主链结果。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
TRANSFER_ROOT = ROOT / "transfer" / "llm_extension_transfer_v1" / "results"
FINAL_DIR = TRANSFER_ROOT / "final_experiment_closure_audit_v1"
MODERN_DIR = TRANSFER_ROOT / "modern_baseline_gate_v1"
ABLATION_DIR = TRANSFER_ROOT / "multisource_ablation"
PILOT_SUMMARY = ROOT / "runs" / "tabiclv2_pilot" / "20_tabiclv2_pilot_summary.csv"
OUTPUT_DIR = ROOT / "结果" / "论文图表" / "报告辅助图表_20261011"


TASK_ORDER = [
    "chla_surface",
    "chla_middle",
    "chla_bottom",
    "tn_surface",
    "tn_middle",
    "tn_bottom",
    "tp_surface",
    "tp_middle",
    "tp_bottom",
    "secchi",
]

TASK_LABELS = {
    "chla_surface": "Chl-a 表层",
    "chla_middle": "Chl-a 中层",
    "chla_bottom": "Chl-a 底层",
    "tn_surface": "TN 表层",
    "tn_middle": "TN 中层",
    "tn_bottom": "TN 底层",
    "tp_surface": "TP 表层",
    "tp_middle": "TP 中层",
    "tp_bottom": "TP 底层",
    "secchi": "透明度",
}

DEPTH_LABELS = {"surface": "表层", "middle": "中层", "bottom": "底层"}
INDICATOR_LABELS = {"chla": "Chl-a", "tn": "TN", "tp": "TP"}
SOURCE_LABELS = {
    "HUMAN_ACTIVITY": "人类活动",
    "METEOROLOGY": "气象",
    "OCEAN_PHYSICS": "海洋物理",
    "RAINFALL_HYDRO_TIDE": "降雨/水文/潮汐",
    "SPATIAL_STATIC": "空间静态",
    "TEMPORAL_PROCESS": "时间过程",
}


def configure_matplotlib() -> None:
    """设置论文图表的字体、线宽和输出风格。"""

    mpl.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    """以 300 dpi PNG 保存图表。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_core_results() -> pd.DataFrame:
    path = FINAL_DIR / "final_core_results_table_v2.csv"
    data = pd.read_csv(path)
    data = data.loc[data["task"].isin(TASK_ORDER)].copy()
    data["task_order"] = pd.Categorical(data["task"], TASK_ORDER, ordered=True)
    return data.sort_values(["task_order", "model"])


def plot_overall_improvement(core: pd.DataFrame) -> None:
    """图 1：10 个任务的总体 R² 和站点 RMSE 改善比例。"""

    tabiclv2 = core.loc[core["model"].eq("TabICLv2")].set_index("task").loc[TASK_ORDER]
    xgboost = core.loc[core["model"].eq("XGBoost")].set_index("task").loc[TASK_ORDER]
    labels = [TASK_LABELS[item] for item in TASK_ORDER]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.2), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    offset = 0.18
    ax.barh(y + offset, xgboost["pooled_R2"], height=0.32, color="#9AA7B4", label="XGBoost")
    ax.barh(y - offset, tabiclv2["pooled_R2"], height=0.32, color="#2A9D8F", label="TabICLv2")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("整体 R²")
    ax.set_title("10 个预测任务的整体 R²")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    for index, task in enumerate(TASK_ORDER):
        delta = float(tabiclv2.loc[task, "delta_R2"])
        value = float(tabiclv2.loc[task, "pooled_R2"])
        ax.text(value + (0.012 if value >= 0 else -0.012), index - offset, f"Δ={delta:+.3f}", va="center", fontsize=8)

    ax = axes[1]
    win_rates = tabiclv2["station_win_rate_RMSE"].to_numpy(dtype=float) * 100
    colors = ["#2A9D8F" if value >= 80 else "#E9C46A" for value in win_rates]
    ax.barh(y, win_rates, color=colors, height=0.55)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("RMSE 得到改善的站点比例（%）")
    ax.set_title("站点层面的 RMSE 改善覆盖")
    ax.grid(axis="x", alpha=0.22)
    for index, value in enumerate(win_rates):
        ax.text(value + 1.2, index, f"{value:.1f}%", va="center", fontsize=8)

    fig.suptitle("TabICLv2 相对 XGBoost 的站点留出验证表现", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "注：R²为五折测试预测合并计算的整体 R²；站点比例来自正式站点层面的 RMSE 比较。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "图1_十任务总体提升与站点改善覆盖.png")


def plot_depth_consistency(core: pd.DataFrame) -> None:
    """图 2：Chl-a、TN、TP 在不同水层的提升差异。"""

    tabiclv2 = core.loc[core["model"].eq("TabICLv2")].copy()
    tabiclv2["indicator"] = tabiclv2["target_family"].map(INDICATOR_LABELS)
    tabiclv2["depth_label"] = tabiclv2["depth"].map(DEPTH_LABELS)
    pivot_delta = tabiclv2.pivot(index="indicator", columns="depth_label", values="delta_R2").reindex(["Chl-a", "TN", "TP"])
    pivot_win = tabiclv2.pivot(index="indicator", columns="depth_label", values="station_win_rate_RMSE").reindex(["Chl-a", "TN", "TP"])
    depth_order = ["表层", "中层", "底层"]
    pivot_delta = pivot_delta[depth_order]
    pivot_win = pivot_win[depth_order] * 100

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8), gridspec_kw={"width_ratios": [1, 1]})
    delta_min = float(pivot_delta.to_numpy().min())
    delta_max = float(pivot_delta.to_numpy().max())
    norm = TwoSlopeNorm(vmin=min(delta_min, -0.01), vcenter=0, vmax=max(delta_max, 0.01))
    image = axes[0].imshow(pivot_delta.to_numpy(), cmap="RdYlGn", norm=norm, aspect="auto")
    axes[0].set_xticks(range(3), depth_order)
    axes[0].set_yticks(range(3), pivot_delta.index)
    axes[0].set_title("整体 R² 提升值 ΔR²")
    for row in range(3):
        for col in range(3):
            axes[0].text(col, row, f"{pivot_delta.iloc[row, col]:+.3f}", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=axes[0], shrink=0.82, label="ΔR²")

    image = axes[1].imshow(pivot_win.to_numpy(), cmap="YlGn", vmin=65, vmax=95, aspect="auto")
    axes[1].set_xticks(range(3), depth_order)
    axes[1].set_yticks(range(3), pivot_win.index)
    axes[1].set_title("RMSE 改善站点比例")
    for row in range(3):
        for col in range(3):
            axes[1].text(col, row, f"{pivot_win.iloc[row, col]:.1f}%", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=axes[1], shrink=0.82, label="站点比例（%）")

    fig.suptitle("不同水质指标与水层的模型提升差异", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "注：同一指标的三个水层采用统一的站点分组；颜色仅表示提升大小，不代表因果关系。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, "图2_不同指标与水层的提升差异.png")


def plot_modern_model_benchmark() -> None:
    """图 3：四个代表任务的四模型 R² 对照。"""

    data = pd.read_csv(MODERN_DIR / "modern_baseline_core_metrics.csv")
    task_order = ["chla_surface", "secchi", "tn_surface", "tp_middle"]
    model_order = ["xgboost", "tabm", "tabpfn", "tabiclv2"]
    model_labels = {"xgboost": "XGBoost", "tabm": "TabM", "tabpfn": "TabPFN", "tabiclv2": "TabICLv2"}
    colors = {"xgboost": "#9AA7B4", "tabm": "#F4A261", "tabpfn": "#457B9D", "tabiclv2": "#2A9D8F"}
    data["task_order"] = pd.Categorical(data["task"], task_order, ordered=True)
    data["model_order"] = pd.Categorical(data["model"], model_order, ordered=True)
    data = data.sort_values(["task_order", "model_order"])

    x = np.arange(len(task_order))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for index, model in enumerate(model_order):
        values = data.loc[data["model"].eq(model)].set_index("task").loc[task_order, "pooled_R2"]
        bars = ax.bar(x + (index - 1.5) * width, values, width=width, color=colors[model], label=model_labels[model])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

    best = data.loc[data.groupby("task", observed=False)["pooled_R2"].idxmax()]
    for index, task in enumerate(task_order):
        row = best.loc[best["task"].eq(task)].iloc[0]
        model_index = model_order.index(row["model"])
        ax.text(index + (model_index - 1.5) * width, float(row["pooled_R2"]) + 0.045, "★", ha="center", va="bottom", fontsize=12, color="#D62828")

    ax.set_xticks(x, [TASK_LABELS[item] for item in task_order])
    ax.set_ylabel("整体 R²")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.text(0.5, 0.01, "注：★表示该任务的最高整体 R²；透明度任务为单一透明度预测任务。", ha="center", fontsize=9)
    fig.suptitle("代表性任务的多模型性能比较", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.11, 1, 0.91))
    save_figure(fig, "图3_代表性任务四模型比较.png")


def plot_validation_strategies() -> None:
    """图 4：随机、年份和站点三种验证方式的 R² 变化。"""

    data = pd.read_csv(PILOT_SUMMARY)
    task_order = ["chla_surface", "secchi", "tn_bottom", "tp_middle"]
    validation_order = ["random", "year", "station"]
    validation_labels = {"random": "随机划分", "year": "年份留出", "station": "站点留出"}
    colors = {"random": "#457B9D", "year": "#E76F51", "station": "#2A9D8F"}
    task_labels = {"chla_surface": "Chl-a 表层", "secchi": "透明度", "tn_bottom": "TN 底层", "tp_middle": "TP 中层"}

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharey=True)
    for axis, task in zip(axes.flat, task_order):
        subset = data.loc[data["task"].eq(task)].set_index("validation").loc[validation_order].copy()
        values = subset["delta_R2"].to_numpy(dtype=float)
        bars = axis.bar([validation_labels[item] for item in validation_order], values, color=[colors[item] for item in validation_order], width=0.62)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_title(task_labels[task])
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + (0.025 if value >= 0 else -0.06), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)

    axes[0, 0].set_ylabel("TabICLv2 相对 XGBoost 的 ΔR²")
    axes[1, 0].set_ylabel("TabICLv2 相对 XGBoost 的 ΔR²")
    fig.suptitle("验证方式改变后，模型优势并不保持一致", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "注：本图使用历史严格验证对照摘要；不同验证实验的划分与部分特征处理不完全相同，不宜直接解释为单一时间或空间效应。", ha="center", fontsize=8.8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(fig, "图4_三种验证方式的模型差异.png")


def plot_feature_source_ablation() -> None:
    """图 5：四个代表任务的来源级特征消融结果。"""

    data = pd.read_csv(ABLATION_DIR / "multisource_ablation_source_summary.csv")
    task_order = ["chla_surface", "secchi", "tn_surface", "tp_middle"]
    source_order = list(SOURCE_LABELS)
    task_labels = {"chla_surface": "Chl-a 表层", "secchi": "透明度", "tn_surface": "TN 表层", "tp_middle": "TP 中层"}
    class_colors = {"IMPORTANT": "#E76F51", "SUPPORTIVE": "#2A9D8F", "NEUTRAL": "#9AA7B4"}

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.5), sharex=False)
    for axis, task in zip(axes.flat, task_order):
        subset = data.loc[data["task"].eq(task)].set_index("source_group").reindex(source_order).reset_index()
        values = subset["pooled_delta_RMSE"].to_numpy(dtype=float)
        colors = [class_colors.get(item, "#9AA7B4") for item in subset["classification"]]
        positions = np.arange(len(source_order))
        axis.barh(positions, values, color=colors, height=0.58)
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_yticks(positions, [SOURCE_LABELS[item] for item in source_order])
        axis.invert_yaxis()
        axis.set_title(task_labels[task])
        axis.grid(axis="x", alpha=0.2)
        span = max(float(np.max(np.abs(values))), 0.001)
        for position, value in zip(positions, values):
            if abs(float(value)) < 0.005:
                axis.text(0.04 * span, position, f"{value:+.4f}", va="center", ha="left", fontsize=8, color="#333333")
            else:
                offset = 0.018 * span
                axis.text(value + offset, position, f"{value:+.4f}", va="center", ha="left", fontsize=8)

    axes[0, 0].set_xlabel("移除特征组后的 RMSE 变化")
    axes[0, 1].set_xlabel("移除特征组后的 RMSE 变化")
    axes[1, 0].set_xlabel("移除特征组后的 RMSE 变化")
    axes[1, 1].set_xlabel("移除特征组后的 RMSE 变化")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in class_colors.values()]
    fig.legend(handles, ["重要", "支持性", "中性"], frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("不同环境信息来源对预测性能的贡献差异", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "注：正值表示去掉该特征组后 RMSE 增大；本图反映预测模型中的增量贡献，不等同于因果效应。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, "图5_多源环境信息消融比较.png")


def write_manifest() -> None:
    """写入中文图表说明与证据清单。"""

    manifest = pd.DataFrame(
        [
            ["图1", "十任务总体提升与站点改善覆盖", "正式主链", str(FINAL_DIR / "final_core_results_table_v2.csv"), "核心结论：10 个任务整体 R² 均提高，站点 RMSE 改善比例整体较高。"],
            ["图2", "不同指标与水层的提升差异", "正式主链", str(FINAL_DIR / "final_core_results_table_v2.csv"), "核心结论：Chl-a 三层较稳定，TN 和 TP 的层间提升幅度不同。"],
            ["图3", "代表性任务四模型比较", "扩展实验/现代基准", str(MODERN_DIR / "modern_baseline_core_metrics.csv"), "核心结论：不存在所有任务都最优的单一模型。"],
            ["图4", "三种验证方式的模型差异", "扩展实验/历史严格验证", str(PILOT_SUMMARY), "核心结论：站点留出、年份留出和随机划分回答的是不同问题。"],
            ["图5", "多源环境信息消融比较", "扩展实验/来源级消融", str(ABLATION_DIR / "multisource_ablation_source_summary.csv"), "核心结论：空间、气象和其他来源在不同任务中的增量贡献不同。"],
        ],
        columns=["图号", "图名", "证据类别", "数据来源", "可支持的报告表述"],
    )
    manifest.to_csv(OUTPUT_DIR / "图表证据清单_20261011.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "图表使用说明_20261011.md").write_text(
        (
            "# 报告辅助图表使用说明\n\n"
            "本目录图表均由已保存的实验结果生成，不包含重新训练或人工调整的指标。\n\n"
            "- 图1和图2使用正式十任务汇总表，适合放在报告的主要结果部分。\n"
            "- 图3用于说明不同表格模型的任务适用差异，适合放在模型对比部分。\n"
            "- 图4使用历史严格验证摘要，适合放在验证方式讨论部分；不要把三种验证的绝对 R² 直接当作同一实验的可交换结果。\n"
            "- 图5使用来源级 XGBoost 消融结果，适合用于解释环境信息来源的模型贡献；不能据此作出因果结论。\n\n"
            "所有图表均未修改 `结果/正式结果/full_station/`，也未改变模型算法、训练参数、特征构造逻辑或已有结果。\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    core = read_core_results()
    plot_overall_improvement(core)
    plot_depth_consistency(core)
    plot_modern_model_benchmark()
    plot_validation_strategies()
    plot_feature_source_ablation()
    write_manifest()
    print(f"已生成论文辅助图表：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
