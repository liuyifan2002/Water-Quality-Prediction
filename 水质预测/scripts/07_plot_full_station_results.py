from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_REPORT_DIR = Path("reports/full_station")
DEFAULT_OUT_DIR = DEFAULT_REPORT_DIR / "figures"

FEATURE_SET_LABELS = {
    "baseline": "Baseline",
    "f1_process": "F1 Process",
    "f2_remote": "F2 Remote",
    "f3_fusion": "F3 Fusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 full_station 结果自动绘制论文图。")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少结果文件: {path}")
    return pd.read_csv(path, low_memory=False)


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})


def make_metric_heatmaps(metrics: pd.DataFrame, out_dir: Path) -> None:
    work = metrics.copy()
    work["row_label"] = work["target"].str.upper() + " | " + work["depth"].str.capitalize()
    work["feature_set_label"] = work["feature_set"].map(FEATURE_SET_LABELS).fillna(work["feature_set"])

    r2_pivot = work.pivot(index="row_label", columns="feature_set_label", values="r2")
    rmse_pivot = work.pivot(index="row_label", columns="feature_set_label", values="rmse")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    sns.heatmap(r2_pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0], cbar_kws={"label": "R2"})
    axes[0].set_title("Test R2 by Target, Depth, and Feature Set")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(rmse_pivot, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=axes[1], cbar_kws={"label": "RMSE"})
    axes[1].set_title("Test RMSE by Target, Depth, and Feature Set")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.savefig(out_dir / "fig01_metric_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_feature_set_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    work = summary.copy()
    work["feature_set_label"] = work["feature_set"].map(FEATURE_SET_LABELS).fillna(work["feature_set"])
    order = work.sort_values("mean_r2", ascending=False)["feature_set_label"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sns.barplot(data=work, x="feature_set_label", y="mean_r2", order=order, ax=axes[0], color="#2a9d8f")
    axes[0].set_title("Mean Test R2")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("R2")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(data=work, x="feature_set_label", y="mean_rmse", order=order, ax=axes[1], color="#e76f51")
    axes[1].set_title("Mean Test RMSE")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=15)

    fig.savefig(out_dir / "fig02_feature_set_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_fusion_gain_plot(fusion_gain: pd.DataFrame, out_dir: Path) -> None:
    work = fusion_gain.copy()
    work["row_label"] = work["target"].str.upper() + " | " + work["depth"].str.capitalize()
    work = work.sort_values(["target", "depth"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    y = range(len(work))
    ax.barh([i + 0.18 for i in y], work["fusion_minus_baseline_r2"], height=0.35, label="F3 - Baseline", color="#457b9d")
    ax.barh([i - 0.18 for i in y], work["fusion_minus_process_r2"], height=0.35, label="F3 - F1", color="#f4a261")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(work["row_label"])
    ax.set_xlabel("Delta R2")
    ax.set_ylabel("")
    ax.set_title("Fusion Gain in Test R2")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "fig03_fusion_gain_r2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_best_model_scatter(best: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path) -> None:
    panels = best.sort_values(["target", "depth"]).reset_index(drop=True)
    fig, axes = plt.subplots(4, 3, figsize=(14, 16), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (_, row) in zip(axes_flat, panels.iterrows()):
        panel = predictions[
            (predictions["target"] == row["target"])
            & (predictions["depth"] == row["depth"])
            & (predictions["feature_set"] == row["feature_set"])
        ].copy()

        if panel.empty:
            ax.set_axis_off()
            continue

        x = panel["observed"]
        y = panel["predicted"]
        ax.scatter(x, y, s=12, alpha=0.55, color="#457b9d", edgecolors="none")
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="#d62828")
        ax.set_title(f"{row['target'].upper()} | {row['depth']} | {FEATURE_SET_LABELS.get(row['feature_set'], row['feature_set'])}")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.text(
            0.04,
            0.96,
            f"R2={row['r2']:.3f}\nRMSE={row['rmse']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    fig.savefig(out_dir / "fig04_best_model_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    metrics = load_csv(args.report_dir / "full_station_training_metrics.csv")
    metrics = metrics[metrics["status"] == "ok"].copy()
    feature_summary = load_csv(args.report_dir / "full_station_feature_set_summary.csv")
    fusion_gain = load_csv(args.report_dir / "full_station_fusion_gain.csv")
    best = load_csv(args.report_dir / "full_station_best_by_target_depth.csv")
    predictions = load_csv(args.report_dir / "full_station_predictions.csv")

    make_metric_heatmaps(metrics, args.out_dir)
    make_feature_set_summary(feature_summary, args.out_dir)
    make_fusion_gain_plot(fusion_gain, args.out_dir)
    make_best_model_scatter(best, predictions, args.out_dir)

    print(f"已生成论文图目录: {args.out_dir}")


if __name__ == "__main__":
    main()
