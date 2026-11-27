"""重绘透明度单列口径下的正式结果热图和代表性散点图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "结果" / "正式结果" / "full_station"
OUTPUT_DIR = ROOT / "结果" / "论文图表" / "full_station_figures"

TARGET_LABELS = {"chla": "Chl-a", "tn": "TN", "tp": "TP", "secchi": "透明度"}
DEPTH_LABELS = {"surface": "表层", "middle": "中层", "bottom": "底层"}


def configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False


def save_metric_heatmap(best: pd.DataFrame) -> None:
    tasks = [("chla", "surface"), ("chla", "middle"), ("chla", "bottom"), ("tn", "surface"), ("tn", "middle"), ("tn", "bottom"), ("tp", "surface"), ("tp", "middle"), ("tp", "bottom"), ("secchi", "surface")]
    labels = [f"{TARGET_LABELS[target]}\n{DEPTH_LABELS[depth]}" if target != "secchi" else "透明度\n站点—日期" for target, depth in tasks]
    selected = best.set_index(["target", "depth"]).loc[tasks].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    metrics = [("r2", "R2", "YlGnBu", ".3f"), ("rmse", "RMSE", "YlOrRd", ".3f")]
    for ax, (column, title, cmap, fmt) in zip(axes, metrics):
        values = selected[column].to_numpy(dtype=float)[None, :]
        image = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_yticks([0], ["正式随机留出"])
        ax.set_xticks(range(len(labels)), labels, fontsize=9)
        ax.tick_params(axis="x", length=0)
        for index, value in enumerate(values[0]):
            ax.text(index, 0, format(value, fmt), ha="center", va="center", fontsize=9, color="black")
        fig.colorbar(image, ax=ax, shrink=0.78)
    fig.savefig(OUTPUT_DIR / "fig01a_正式模型性能热图_透明度单列.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_representative_scatter(predictions: pd.DataFrame) -> None:
    representative = [
        ("chla", "surface", "f3_history_remote", "Chl-a 表层"),
        ("tn", "surface", "f3_history_remote", "TN 表层"),
        ("tp", "surface", "baseline", "TP 表层"),
        ("secchi", "surface", "f3_history_remote", "透明度 站点—日期"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.2), constrained_layout=True)
    for ax, (target, depth, feature_set, label) in zip(axes.flat, representative):
        data = predictions.loc[
            (predictions["target"] == target)
            & (predictions["depth"] == depth)
            & (predictions["feature_set"] == feature_set),
            ["observed", "predicted"],
        ].dropna()
        upper = max(float(data["observed"].max()), float(data["predicted"].max())) * 1.03
        ax.scatter(data["observed"], data["predicted"], s=9, alpha=0.38, color="#2c7fb8", edgecolors="none")
        ax.plot([0, upper], [0, upper], color="#c73e1d", linestyle="--", linewidth=1.1)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_title(f"{label}  R2={r2_score(data['observed'], data['predicted']):.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("观测值")
        ax.set_ylabel("预测值")
        ax.grid(alpha=0.22)
    fig.savefig(OUTPUT_DIR / "fig04a_代表性目标观测预测散点图.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best = pd.read_csv(RESULT_DIR / "full_station_best_by_target_depth.csv")
    predictions = pd.read_csv(RESULT_DIR / "full_station_predictions.csv")
    save_metric_heatmap(best)
    save_representative_scatter(predictions)


if __name__ == "__main__":
    main()
