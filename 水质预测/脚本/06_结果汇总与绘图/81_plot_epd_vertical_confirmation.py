"""EPD-Vertical V3 figure: multi-seed Random Strict and station-holdout delta R2 (single panel pair)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/epd_vertical_confirmation_v1_20261127"
DST = OUT / "06_confirmation_comparison.png"

TASKS = ["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface",
         "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom"]
LABEL = {"chla_surface": "Chl-a surf.", "chla_middle": "Chl-a mid.", "chla_bottom": "Chl-a bot.",
         "secchi": "Secchi", "tn_surface": "TN surf.", "tn_middle": "TN mid.", "tn_bottom": "TN bot.",
         "tp_surface": "TP surf.", "tp_middle": "TP mid.", "tp_bottom": "TP bot."}


def main():
    ms = pd.read_csv(OUT / "01_random_multiseed_metrics.csv")
    ms = ms[ms.feature_set == "M2"].set_index(["task", "seed"])
    bo = pd.read_csv(OUT / "04_paired_bootstrap.csv")
    sh = bo[bo.protocol == "station_holdout"].set_index("task")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=False,
                             gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.12})
    y = np.arange(len(TASKS))[::-1]

    ax = axes[0]
    colors = {42: "#9ecae1", 2026: "#4292c6", 3407: "#08519c"}
    h = 0.24
    for k, seed in enumerate([42, 2026, 3407]):
        v = np.array([ms.loc[(t, seed), "delta_R2"] for t in TASKS])
        ax.barh(y + (1 - k) * h, v, height=h, color=colors[seed], edgecolor="white", linewidth=0.4,
                label=f"model seed {seed}")
    mean = np.array([ms.loc[(t, s), "delta_R2"] for t in TASKS for s in (42, 2026, 3407)]).reshape(10, 3).mean(axis=1)
    ax.plot(mean, y, "D", color="#cb181d", markersize=4, markeredgecolor="white", markeredgewidth=0.5,
            label="mean of 3 seeds", zorder=5)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[t] for t in TASKS])
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel(r"$\Delta R^2$ (M2 - M0)")
    ax.set_title("Random Strict hold-out (3 model seeds)", fontsize=9)

    ax = axes[1]
    v = np.array([sh.loc[t, "delta_R2"] for t in TASKS])
    lo = np.array([sh.loc[t, "delta_R2_ci_low"] for t in TASKS])
    hi = np.array([sh.loc[t, "delta_R2_ci_high"] for t in TASKS])
    ax.barh(y, v, height=0.55, color="#74c476", edgecolor="white", linewidth=0.4)
    ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor="0.25", elinewidth=1.0, capsize=2.5)
    ax.set_yticks([])
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel(r"$\Delta R^2$ (M2 - M0)")
    ax.set_title("5-fold station hold-out (pooled OOF, 95% CI)", fontsize=9)

    for ax in axes:
        ax.grid(axis="x", linestyle=":", linewidth=0.5, color="0.75")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.30, top=0.86)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.02), handlelength=1.6, columnspacing=1.6)
    fig.savefig(DST, dpi=300)
    print("SAVED", DST)


if __name__ == "__main__":
    main()
