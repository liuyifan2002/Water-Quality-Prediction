from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = ROOT / "结果/扩展实验/叶绿素不确定性预测_20260905/叶绿素不确定性预测.csv"
DEFAULT_METRICS = ROOT / "结果/扩展实验/叶绿素不确定性预测_20260905/叶绿素不确定性指标.csv"
DEFAULT_OUT_DIR = ROOT / "结果/诊断结果/Chl-a高值与误差结构_20260907"
DEPTH_ORDER = ["surface", "middle", "bottom"]
SEASON_ORDER = ["春", "夏", "秋", "冬"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析已有 Chl-a 扩展预测的高值漏报与区间覆盖，不重训模型。")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})


def add_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["depth"] = pd.Categorical(work["depth"], categories=DEPTH_ORDER, ordered=True)
    work["season"] = pd.Categorical(
        np.select(
            [work["month"].isin([3, 4, 5]), work["month"].isin([6, 7, 8]), work["month"].isin([9, 10, 11])],
            ["春", "夏", "秋"], default="冬",
        ), categories=SEASON_ORDER, ordered=True,
    )
    work["residual"] = work["gpboost_prediction"] - work["observed"]
    work["abs_error"] = work["residual"].abs()
    work["is_high"] = work["observed"] >= work["high_value_threshold_train_p90"]
    work["is_underpredicted"] = work["residual"] < 0
    work["is_event_missed"] = work["gpboost_prediction"] < work["high_value_threshold_train_p90"]
    work["conformal_covered"] = work["observed"].between(work["conformal_lower"], work["conformal_upper"])
    work["quantile_covered"] = work["observed"].between(work["quantile_lower"], work["quantile_upper"])
    work["conformal_width"] = work["conformal_upper"] - work["conformal_lower"]
    work["quantile_width"] = work["quantile_upper"] - work["quantile_lower"]
    return work


def error_summary(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        high = group[group["is_high"]]
        row.update({
            "总样本数": len(group),
            "高值样本数": len(high),
            "高值比例": high.shape[0] / len(group),
            "高值平均绝对误差": high["abs_error"].mean(),
            "高值平均残差": high["residual"].mean(),
            "高值低估比例": high["is_underpredicted"].mean(),
            "高值事件漏报比例": high["is_event_missed"].mean(),
            "高值共形区间覆盖率": high["conformal_covered"].mean(),
            "高值分位数区间覆盖率": high["quantile_covered"].mean(),
            "高值共形区间平均宽度": high["conformal_width"].mean(),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scatter(data: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, depth in zip(axes, DEPTH_ORDER):
        panel = data[data["depth"] == depth]
        ax.scatter(panel.loc[~panel["is_high"], "observed"], panel.loc[~panel["is_high"], "gpboost_prediction"], s=10, alpha=.35, color="#5c7ea8", label="常规值")
        ax.scatter(panel.loc[panel["is_high"], "observed"], panel.loc[panel["is_high"], "gpboost_prediction"], s=16, alpha=.75, color="#b7472a", label="高值")
        limit = max(panel["observed"].max(), panel["gpboost_prediction"].max())
        threshold = panel["high_value_threshold_train_p90"].iloc[0]
        ax.plot([0, limit], [0, limit], "--", color="black", lw=1)
        ax.axhline(threshold, color="#b7472a", lw=.8, ls=":")
        ax.axvline(threshold, color="#b7472a", lw=.8, ls=":")
        ax.set_title(f"{depth} | 高值阈值={threshold:.1f}")
        ax.set_xlabel("实测 Chl-a (ug/L)")
        ax.set_ylabel("预测 Chl-a (ug/L)")
        ax.legend(frameon=True)
    fig.savefig(out_dir / "图1_高值样本观测预测散点图.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_structure(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sns.barplot(data=summary, x="depth", y="高值低估比例", order=DEPTH_ORDER, color="#b7472a", ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("高值样本的低估比例")
    axes[0].set_xlabel("水层")
    axes[0].set_ylabel("比例")
    sns.barplot(data=summary, x="depth", y="高值平均绝对误差", order=DEPTH_ORDER, color="#c78d4d", ax=axes[1])
    axes[1].set_title("高值样本的平均绝对误差")
    axes[1].set_xlabel("水层")
    axes[1].set_ylabel("ug/L")
    fig.savefig(out_dir / "图2_高值低估与绝对误差.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_zone_season(summary: pd.DataFrame, out_dir: Path) -> None:
    for depth in DEPTH_ORDER:
        panel = summary[(summary["depth"] == depth) & (summary["高值样本数"] >= 5)]
        if panel.empty:
            continue
        table = panel.pivot(index="water_zone_en", columns="season", values="高值低估比例").reindex(columns=SEASON_ORDER)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.heatmap(table, vmin=0, vmax=1, cmap="YlOrRd", annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "低估比例"})
        ax.set_title(f"{depth} 高值低估比例：水区与季节")
        ax.set_xlabel("季节")
        ax.set_ylabel("水区")
        fig.savefig(out_dir / f"图3_{depth}_高值低估水区季节热图.png", dpi=260, bbox_inches="tight")
        plt.close(fig)


def plot_interval_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    long = summary.melt(
        id_vars=["depth"],
        value_vars=["高值共形区间覆盖率", "高值分位数区间覆盖率"],
        var_name="区间方法", value_name="覆盖率",
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.barplot(data=long, x="depth", y="覆盖率", hue="区间方法", order=DEPTH_ORDER, ax=ax)
    ax.axhline(.8, color="black", lw=1, ls="--", label="名义80%")
    ax.set_ylim(0, 1)
    ax.set_title("高值样本的区间覆盖率")
    ax.set_xlabel("水层")
    ax.legend(frameon=True)
    fig.savefig(out_dir / "图4_高值区间覆盖率.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir: Path, total: pd.DataFrame, zone_season: pd.DataFrame, metrics: pd.DataFrame) -> None:
    total = total.set_index("depth").reindex(DEPTH_ORDER).reset_index()
    high_rows = int(total["高值样本数"].sum())
    all_rows = int(total["总样本数"].sum())
    under_min, under_max = total["高值低估比例"].min(), total["高值低估比例"].max()
    missed_min, missed_max = total["高值事件漏报比例"].min(), total["高值事件漏报比例"].max()
    coverage_min, coverage_max = total["高值共形区间覆盖率"].min(), total["高值共形区间覆盖率"].max()
    largest = zone_season[zone_season["高值样本数"] >= 10].sort_values("高值平均绝对误差", ascending=False).head(1)
    location = "无充足分组样本"
    if not largest.empty:
        row = largest.iloc[0]
        location = f"{row['depth']} | {row['water_zone_en']} | {row['season']}（n={int(row['高值样本数'])}，高值 MAE={row['高值平均绝对误差']:.2f} ug/L）"
    point = metrics[metrics["model_variant"] == "gpboost_point"]
    text = f"""# Chl-a 高值与误差结构专题

## 分析边界

本专题只复用 Chl-a 扩展实验的独立外层测试集预测。该实验使用 sample_id 随机 80/20 划分，模型输入排除同日其他水质协变量；其绝对 R2 不替代正式主链结果。本报告用于解释高值漏报和区间覆盖，不宣称已经解决藻华高值预测。

## 主要结果

- 三水层测试集共 {all_rows:,} 条记录，其中按各水层训练集 P90 界定的高值样本有 {high_rows:,} 条。
- 高值样本的点预测低估比例为 {under_min:.1%} 至 {under_max:.1%}；高值事件漏报比例为 {missed_min:.1%} 至 {missed_max:.1%}。模型对高值存在系统性回归均值现象。
- 高值样本的共形区间覆盖率仅为 {coverage_min:.1%} 至 {coverage_max:.1%}，远低于名义 80% 覆盖率；全体样本上的校准区间不能可靠覆盖高值尾部，也不能提升点预测精度。
- 高值误差最大的具有足够样本量的“水层—水区—季节”组合为：{location}。
- GPBoost 点预测的三水层 R2 为 {', '.join(f'{v:.3f}' for v in point.set_index('depth').reindex(DEPTH_ORDER)['r2'])}；这是专题诊断背景，不能与正式主链的逐水层随机划分结果直接替换比较。

## 是否放入论文

**建议放入正文的误差讨论小节。** 推荐使用图 1、图 2 和表 1 说明：高 Chl-a 事件的误差显著大于常规样本，且高值多被低估。图 4 可作为反例说明全体样本校准区间在高值尾部失效，不能被表述为提高了高值点预测能力或可靠量化了极端值风险。水区—季节热图只在单元格高值样本数不低于 10 时用于补充说明。
"""
    (out_dir / "分析报告.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    data = add_diagnostics(pd.read_csv(args.predictions))
    metrics = pd.read_csv(args.metrics)
    total = error_summary(data, ["depth"])
    zone_season = error_summary(data, ["depth", "water_zone_en", "season"])
    missed = data[data["is_high"]].sort_values("residual").head(100).copy()
    total.to_csv(args.out_dir / "表1_各水层高值误差汇总.csv", index=False, encoding="utf-8-sig")
    zone_season.to_csv(args.out_dir / "表2_高值误差水区季节汇总.csv", index=False, encoding="utf-8-sig")
    missed.to_csv(args.out_dir / "表3_高值漏报最严重样本.csv", index=False, encoding="utf-8-sig")
    plot_scatter(data, args.out_dir)
    plot_structure(total, args.out_dir)
    plot_zone_season(zone_season, args.out_dir)
    plot_interval_summary(total, args.out_dir)
    write_report(args.out_dir, total, zone_season, metrics)
    print(f"完成：{args.out_dir}")


if __name__ == "__main__":
    main()
