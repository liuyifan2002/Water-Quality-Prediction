"""Plot ten-task delta R2 for the EPD vertical-feature A/B (single figure only)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/epd_vertical_feature_ab_v2_20261126"
SRC = OUT / "05_配对Bootstrap.csv"
DST = OUT / "07_十任务delta_R2_对比图.png"

TASKS = ["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface",
         "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom"]
LABELS = {"chla_surface": "Chl-a 表层", "chla_middle": "Chl-a 中层", "chla_bottom": "Chl-a 底层",
          "secchi": "Secchi", "tn_surface": "TN 表层", "tn_middle": "TN 中层", "tn_bottom": "TN 底层",
          "tp_surface": "TP 表层", "tp_middle": "TP 中层", "tp_bottom": "TP 底层"}
CONTRASTS = [("M1-M0", "M1-M0（仅 4 项垂向差值）", "#7fb3d5"),
             ("M2-M0", "M2-M0（12 项三层原始值）", "#2e86c1"),
             ("M3-M0", "M3-M0（12 项原始值 + 4 项差值）", "#1b4f72")]


def main():
    boot = pd.read_csv(SRC)
    data = boot[(boot.cohort == "complete_SMB")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=True)
    y = np.arange(len(TASKS))[::-1]
    height = 0.26
    for ax, model in zip(axes, ["TabICLv2", "XGBoost"]):
        sub = data[data.model == model]
        for k, (contrast, label, color) in enumerate(CONTRASTS):
            d = sub[sub.contrast == contrast].set_index("task").reindex(TASKS)
            offset = (k - 1) * height
            ax.barh(y + offset, d.delta_R2.to_numpy(float), height=height,
                    color=color, edgecolor="white", linewidth=0.4, label=f"{label}，宏平均 {d.delta_R2.mean():+.4f}")
        ax.axvline(0, color="#444444", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[t] for t in TASKS])
        ax.set_xlabel("ΔR2（相对同样本集重跑的 M0；正值表示改善）")
        ax.set_title(f"{model}｜complete_SMB 三层完整样本集")
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        ax.legend(loc="lower right", fontsize=8, frameon=True)
        ax.set_axisbelow(True)
    fig.suptitle("EPD 原位垂向特征：十任务 ΔR2（冻结 Random Strict，seed=42，1000 次站点 Bootstrap 点估计）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(DST, dpi=200)
    print("SAVED", DST)


if __name__ == "__main__":
    main()
