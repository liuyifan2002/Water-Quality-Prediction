from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PRED = Path("结果/扩展实验/station_specific/xgboost_station_specific_predictions.csv")
DEFAULT_METRICS = Path("结果/扩展实验/station_specific/xgboost_station_specific_best_by_station.csv")
DEFAULT_OUT_DIR = Path("结果/扩展实验/station_specific/figures")
TARGETS = ["Chl_a", "SD", "TN", "TP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制站点独立模型诊断图。")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def target_label(target: str) -> str:
    return {"Chl_a": "Chl-a", "SD": "透明度 SD", "TN": "总氮 TN", "TP": "总磷 TP"}[target]


def plot_target(pred: pd.DataFrame, metrics: pd.DataFrame, target: str, out_dir: Path) -> None:
    best = metrics[metrics["target"].eq(target)].copy()
    stations = best["station"].tolist()
    pred_col = f"{target}_pred"

    fig, axes = plt.subplots(len(stations), 2, figsize=(14, 2.3 * len(stations)), constrained_layout=True)
    if len(stations) == 1:
        axes = np.array([axes])

    for i, station in enumerate(stations):
        model = best.loc[best["station"].eq(station), "model"].iloc[0]
        sub = pred[
            pred["station"].eq(station)
            & pred["model"].eq(model)
            & pred[pred_col].notna()
        ].sort_values("date")
        row = best[best["station"].eq(station)].iloc[0]

        ax_ts = axes[i, 0]
        ax_sc = axes[i, 1]
        ax_ts.plot(sub["date"], sub[target], marker="o", linewidth=1.6, label="实测", color="#14324A")
        ax_ts.plot(sub["date"], sub[pred_col], marker="s", linewidth=1.4, label="预测", color="#00A6A6")
        ax_ts.set_title(f"{station} | {model} | RMSE={row['rmse']:.3f}, R²={row['r2']:.3f}", fontsize=10)
        ax_ts.grid(alpha=0.25)
        if i == 0:
            ax_ts.legend(loc="upper right", fontsize=8)

        y = sub[target].to_numpy()
        p = sub[pred_col].to_numpy()
        ax_sc.scatter(y, p, s=28, color="#2F80ED", alpha=0.75, edgecolors="white", linewidth=0.5)
        vmin = np.nanmin([y.min(), p.min()])
        vmax = np.nanmax([y.max(), p.max()])
        pad = (vmax - vmin) * 0.08 if vmax > vmin else 1
        ax_sc.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], "--", color="#D95C5C", linewidth=1)
        ax_sc.set_xlim(vmin - pad, vmax + pad)
        ax_sc.set_ylim(vmin - pad, vmax + pad)
        ax_sc.set_xlabel("实测")
        ax_sc.set_ylabel("预测")
        ax_sc.grid(alpha=0.25)

    fig.suptitle(f"{target_label(target)}：站点独立模型诊断图（每站点采用其最优 F1/F2/F3）", fontsize=16, fontweight="bold")
    out_path = out_dir / f"station_specific_{target}_diagnostics.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"写出 {out_path}")


def main() -> None:
    args = parse_args()
    setup_font()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.predictions, parse_dates=["date"])
    metrics = pd.read_csv(args.metrics)
    for target in TARGETS:
        plot_target(pred, metrics, target, args.out_dir)


if __name__ == "__main__":
    main()
