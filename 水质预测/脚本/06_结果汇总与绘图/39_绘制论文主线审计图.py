"""Render evidence figures for the paper-storyline audit from saved outputs only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "结果" / "扩展实验" / "论文主线审计_20260916"
FIGURES = AUDIT / "图版"


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wrap_cell(value: object, width: int) -> str:
    text = str(value)
    return "\n".join(text[index : index + width] for index in range(0, len(text), width))


def candidate_scores() -> None:
    data = pd.read_csv(AUDIT / "02_现有证据矩阵.csv")
    data = data.loc[data["R2_年份留出"].notna()].copy()
    tasks = ["chla_surface", "chla_middle", "chla_bottom", "secchi_surface", "secchi_middle", "secchi_bottom", "tn_bottom"]
    data["任务"] = data["target"] + "_" + data["depth"]
    data["任务"] = pd.Categorical(data["任务"], tasks, ordered=True)
    data = data.sort_values("任务")
    labels = ["Chl-a 表层", "Chl-a 中层", "Chl-a 底层", "透明度 表层", "透明度 中层", "透明度 底层", "TN 底层"]
    models = [("xgboost", "XGBoost", "#4C78A8"), ("tabm", "TabM", "#F58518"), ("tabpfn", "TabPFN", "#54A24B")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, column, title in zip(axes, ["R2_年份留出", "R2_五折留站"], ["年份留出", "五折留站"]):
        x = np.arange(len(tasks))
        width = 0.24
        for idx, (code, label, color) in enumerate(models):
            values = data.loc[data["model"].eq(code), column].to_numpy()
            axis.bar(x + (idx - 1) * width, values, width=width, label=label, color=color)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.set_ylabel("R2")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("图1 候选模型在严格验证下的任务差异", fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "注：仅展示已有三种验证结果的 7 项候选任务。透明度三水层的原始观测值重复，不能视为三个独立目标。",
        ha="center",
        fontsize=9,
    )
    save(fig, "图1_候选模型严格验证得分.png")


def regionalization_gain() -> None:
    data = pd.read_csv(AUDIT / "03_分海区模型对照.csv")
    data = data.loc[data["验证方式"].eq("五折留站")].copy()
    data["任务"] = data["target"] + "_" + data["depth"]
    data = data.sort_values("delta_r2")
    labels = data["任务"].str.replace("_", " ", regex=False)
    colors = np.where(data["delta_r2"] >= 0, "#2E8B57", "#C44E52")
    fig, axis = plt.subplots(figsize=(9, 6.5))
    axis.barh(labels, data["delta_r2"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("分海区模型相对全域模型的 R2 增益")
    axis.set_title("图2 分海区建模在五折留站验证中的增益", fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
    for y, value in enumerate(data["delta_r2"]):
        axis.text(value + (0.006 if value >= 0 else -0.006), y, f"{value:.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    fig.text(0.5, -0.02, "绿色表示分海区模型提高，红色表示降低。该图检验空间外推，不代表年份外推表现。", ha="center", fontsize=9)
    save(fig, "图2_分海区空间验证增益.png")


def fewshot_curve() -> None:
    data = pd.read_csv(AUDIT / "少量本地适配试验" / "少量本地适配试验结果.csv")
    data["adaptation_level"] = data["adaptation_level"].astype(str)
    zero = data.loc[data["adaptation_level"].eq("0"), ["task", "heldout_station", "r2"]].rename(columns={"r2": "zero_r2"})
    data = data.merge(zero, on=["task", "heldout_station"], how="left")
    data["r2_gain"] = data["r2"] - data["zero_r2"]
    order = ["0", "1", "3", "6", "12", "24", "all_pretest"]
    task_labels = {"chla_surface": "Chl-a 表层", "secchi_surface": "透明度 表层", "tn_bottom": "TN 底层"}
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    for task, group in data.groupby("task"):
        values = group.groupby("adaptation_level")["r2_gain"].median().reindex(order)
        axis.plot(range(len(order)), values, marker="o", linewidth=2, label=task_labels[task])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(order)), ["全部历史" if value == "all_pretest" else value for value in order])
    axis.set_xlabel("测试期前可用的本地观测数")
    axis.set_ylabel("相对零样本的站点 R2 中位数增益")
    axis.set_title("图3 未见站点的少量本地观测适配试验", fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.text(
        0.5,
        -0.02,
        "注：20 个固定站点，外部训练期为 2014-2022，测试期为 2023-2024；本地观测均早于测试期。all_pretest 为该站点全部可用历史观测。",
        ha="center",
        fontsize=9,
    )
    save(fig, "图3_少量本地适配增益曲线.png")


def transferability_matrix() -> None:
    rows = [
        ["任务特异模型选择", "部分支持", "仅 7 项候选任务；严格年份与留站表现存在任务差异", "不能把随机与严格 R2 直接相减"],
        ["分海区建模", "条件支持", "五折留站下 Chl-a 与透明度提高；TP 降低", "不宜作为四指标统一主模型"],
        ["未见站点少量适配", "谨慎保留", "全部历史本地观测时多数站点有小幅提高", "1-24 条观测增益不稳定，尚不足以作核心创新"],
        ["透明度三水层", "不成立", "三水层同日数值完全重复", "应合并为一个透明度任务"],
    ]
    frame = pd.DataFrame(rows, columns=["研究命题", "判定", "证据", "论文处理"])
    for column, width in {"研究命题": 12, "判定": 8, "证据": 14, "论文处理": 14}.items():
        frame[column] = frame[column].map(lambda value: wrap_cell(value, width))
    fig, axis = plt.subplots(figsize=(13, 7.2))
    axis.axis("off")
    table = axis.table(cellText=frame.values, colLabels=frame.columns, cellLoc="left", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 4.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_text_props(color="black")
        if row == 0:
            cell.set_facecolor("#D9EAF7")
            cell.set_text_props(weight="bold")
        elif col == 1:
            cell.set_facecolor("#F6F6F6")
    axis.set_title("图4 论文主线的证据边界", fontweight="bold", pad=16)
    save(fig, "图4_论文主线证据边界.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    candidate_scores()
    regionalization_gain()
    fewshot_curve()
    transferability_matrix()
    print(f"图版输出目录：{FIGURES}")


if __name__ == "__main__":
    main()
