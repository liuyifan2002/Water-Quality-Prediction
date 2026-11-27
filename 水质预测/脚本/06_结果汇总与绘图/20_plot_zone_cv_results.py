"""Create paper figures for the five-fold held-station zone-model validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "\u7ed3\u679c" / "\u6269\u5c55\u5b9e\u9a8c" / "\u5206\u6d77\u533a\u4e94\u6298\u7559\u7ad9\u9a8c\u8bc1_20260906"
DEFAULT_OUTPUT_DIR = ROOT / "\u7ed3\u679c" / "\u8bba\u6587\u56fe\u8868" / "\u5206\u6d77\u533a\u4e94\u6298\u7559\u7ad9\u9a8c\u8bc1_20260906"

TARGET_LABELS = {"chla": "Chl-a", "secchi": "Secchi", "tn": "TN", "tp": "TP"}
DEPTH_LABELS = {"surface": "表层", "middle": "中层", "bottom": "底层"}
ORDER = [(target, depth) for target in ("chla", "secchi", "tn", "tp") for depth in ("surface", "middle", "bottom")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot five-fold held-station validation results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_fonts() -> None:
    font_candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["font.sans-serif"] = font_candidates
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"


def plot_pooled_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.set_index(["target", "depth"]).loc[ORDER].reset_index()
    labels = [f"{TARGET_LABELS[target]}\n{DEPTH_LABELS[depth]}" for target, depth in ORDER]
    x = np.arange(len(ordered))
    width = 0.37
    fig, ax = plt.subplots(figsize=(13.2, 6.2), dpi=220)
    ax.bar(x - width / 2, ordered["global_r2_pooled"], width, label="全域模型", color="#2A9D8F")
    ax.bar(x + width / 2, ordered["zone_r2_pooled"], width, label="分海区模型", color="#C8553D")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("合并折外 R2")
    ax.set_xticks(x, labels)
    ax.set_title("五折留站点验证中的合并折外 R2 对比", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    for boundary in (2.5, 5.5, 8.5):
        ax.axvline(boundary, color="#BDBDBD", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_fold_stability(macro: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=220)
    ax.plot(macro["fold"], macro["global_r2_mean"], marker="o", linewidth=2.2, color="#2A9D8F", label="全域模型")
    ax.plot(macro["fold"], macro["zone_r2_mean"], marker="o", linewidth=2.2, color="#C8553D", label="分海区模型")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(macro["fold"])
    ax.set_xlabel("留站点折次")
    ax.set_ylabel("12 个任务的平均 R2")
    ax.set_title("分海区模型在五个留站点折次中的稳定性", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_table(summary: pd.DataFrame, output_path: Path) -> None:
    table = summary.copy()
    table["指标"] = table["target"].map(TARGET_LABELS)
    table["水层"] = table["depth"].map(DEPTH_LABELS)
    table["全域模型合并折外R2"] = table["global_r2_pooled"].round(3)
    table["分海区模型合并折外R2"] = table["zone_r2_pooled"].round(3)
    table["R2变化"] = table["delta_r2_pooled"].round(3)
    table["提高折数"] = table["positive_folds"].astype(str) + "/" + table["folds"].astype(str)
    table["稳定判定"] = table["stable_better"].map({"yes": "稳定提高", "no": "不稳定"})
    table = table.set_index(["target", "depth"]).loc[ORDER].reset_index()
    table[["指标", "水层", "全域模型合并折外R2", "分海区模型合并折外R2", "R2变化", "提高折数", "稳定判定"]].to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    summary = pd.read_csv(args.input_dir / "\u4e94\u6298\u7559\u7ad9\u70b9\u9a8c\u8bc1_\u6c47\u603b.csv")
    macro = pd.read_csv(args.input_dir / "\u4e94\u6298\u603b\u4f53\u6bd4\u8f83.csv").sort_values("fold")
    plot_pooled_comparison(summary, args.output_dir / "\u56fe4-4_\u4e94\u6298\u7559\u7ad9\u70b9\u5408\u5e76\u6298\u5916R2\u5bf9\u6bd4.png")
    plot_fold_stability(macro, args.output_dir / "\u56fe4-5_\u4e94\u6298\u7559\u7ad9\u70b9\u603b\u4f53R2\u6298\u6b21\u5bf9\u6bd4.png")
    export_table(summary, args.output_dir / "\u88684-3_\u5206\u6d77\u533a\u4e94\u6298\u7559\u9a8c\u8bc1\u6c47\u603b.csv")


if __name__ == "__main__":
    main()
