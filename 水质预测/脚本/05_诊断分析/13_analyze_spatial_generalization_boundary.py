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
DEFAULT_CV_SUMMARY = ROOT / "结果/扩展实验/分海区五折留站验证_20260906/五折留站点验证_汇总.csv"
DEFAULT_CV_MACRO = ROOT / "结果/扩展实验/分海区五折留站验证_20260906/五折总体比较.csv"
DEFAULT_STRICT = ROOT / "结果/扩展实验/分海区严格验证_20260906/分海区严格验证_指标.csv"
DEFAULT_OUT_DIR = ROOT / "结果/诊断结果/空间泛化边界_20260907"
DEPTH_ORDER = ["surface", "middle", "bottom"]
TARGET_ORDER = ["chla", "secchi", "tn", "tp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总分海区模型的空间与时间外推边界，不重训模型。")
    parser.add_argument("--cv-summary", type=Path, default=DEFAULT_CV_SUMMARY)
    parser.add_argument("--cv-macro", type=Path, default=DEFAULT_CV_MACRO)
    parser.add_argument("--strict", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})


def strict_summary(strict: pd.DataFrame) -> pd.DataFrame:
    work = strict[strict["water_zone_en"].eq("all")].copy()
    wide = work.pivot_table(
        index=["evaluation_mode", "target", "depth", "rows"], columns="model_variant", values="r2", aggfunc="first"
    ).reset_index()
    wide["delta_r2"] = wide["zone_aware_external"] - wide["global_external"]
    return wide


def decision_table(cv: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    time = strict[strict["evaluation_mode"].eq("time_holdout_2023_2024")][["target", "depth", "delta_r2"]].rename(columns={"delta_r2": "时间外推delta_R2"})
    station = strict[strict["evaluation_mode"].eq("held_station")][["target", "depth", "delta_r2"]].rename(columns={"delta_r2": "单次留站点delta_R2"})
    out = cv.merge(time, on=["target", "depth"], how="left").merge(station, on=["target", "depth"], how="left")
    out["论文定位"] = np.select(
        [
            out["stable_better"].eq("yes") & out["时间外推delta_R2"].lt(0),
            out["stable_better"].eq("yes") & out["时间外推delta_R2"].ge(0),
        ],
        ["空间外推有效，时间外推不支持", "空间与时间外推均有正信号"],
        default="空间外推不稳定，不作推广",
    )
    return out.sort_values(["target", "depth"])


def plot_cv_tasks(decisions: pd.DataFrame, out_dir: Path) -> None:
    work = decisions.copy()
    work["任务"] = work["target"].str.upper() + " | " + work["depth"]
    work = work.sort_values(["target", "depth"])
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = np.where(work["stable_better"].eq("yes"), "#187f7c", "#b7472a")
    ax.barh(work["任务"], work["delta_r2_pooled"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("五折留站点合并预测：分海区 - 全域模型 R2")
    ax.set_title("分海区模型的空间外推增益边界")
    fig.savefig(out_dir / "图1_五折留站点任务R2变化.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_cv_macro(macro: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = np.where(macro["delta_r2_mean"] > 0, "#187f7c", "#b7472a")
    ax.bar(macro["fold"].astype(str), macro["delta_r2_mean"], color=colors)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("留站点折次")
    ax.set_ylabel("12任务平均 R2 变化")
    ax.set_title("五折留站点总体比较")
    fig.savefig(out_dir / "图2_五折留站点总体R2变化.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_validation_modes(strict: pd.DataFrame, out_dir: Path) -> None:
    work = strict.copy()
    work["任务"] = work["target"].str.upper() + " | " + work["depth"]
    table = work.pivot(index="任务", columns="evaluation_mode", values="delta_r2")
    ordered = [f"{target.upper()} | {depth}" for target in TARGET_ORDER for depth in DEPTH_ORDER]
    table = table.reindex([name for name in ordered if name in table.index])
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    sns.heatmap(table, center=0, cmap="RdBu_r", annot=True, fmt=".3f", ax=ax, cbar_kws={"label": "分海区 - 全域 R2"})
    ax.set_title("不同泛化检验下分海区模型的R2变化")
    ax.set_xlabel("验证方式")
    ax.set_ylabel("任务")
    fig.savefig(out_dir / "图3_时间与留站点验证R2变化.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir: Path, decisions: pd.DataFrame, macro: pd.DataFrame) -> None:
    stable = decisions[decisions["stable_better"].eq("yes")]
    unstable = decisions[~decisions["stable_better"].eq("yes")]
    names = "、".join(stable["target"].str.upper() + "-" + stable["depth"])
    bad_names = "、".join(unstable["target"].str.upper() + "-" + unstable["depth"])
    macro_mean = macro["delta_r2_mean"].mean()
    positive = int((macro["delta_r2_mean"] > 0).sum())
    time_positive = int((decisions["时间外推delta_R2"] > 0).sum())
    text = f"""# 空间泛化边界分析

## 分析边界

本专题复用分海区模型的五折留站点验证、单次留站点验证和 2023—2024 年时间留出验证。所有严格验证均排除目标自身历史水质特征和同期其他水质协变量；结果用于判断分区建模的适用边界，不覆盖正式主链。

## 五折留站点结果

- 五折总体平均 R2 变化为 {macro_mean:+.4f}，仅 {positive}/5 折为正，说明分海区模型不具备对全部任务的统一稳定优势。
- 通过“至少 4/5 折提高且合并折外 R2 提高”判定的任务为：{names}。
- 未通过稳定判定的任务为：{bad_names}。

## 与时间外推的对照

- 在 12 个任务中，只有 {time_positive} 个任务的 2023—2024 年时间留出 R2 变化为正；因此空间外推优势不能被解释为中长期时间预测优势。
- 结论应区分两件事：分区可缓解部分光学指标的空间机制混杂，但不能替代跨年份过程信息，也不能作为 TN、TP 的统一解法。
- 透明度为站点—日期塞氏盘观测，在三层记录中重复；其三条“水层”结果不应被视为三个独立的垂向证据。

## 是否放入论文

**建议作为“空间泛化边界”扩展验证放入正文或结果讨论。** 推荐使用图 1 与图 3、表 1；叙述重点是 Chl-a 的三水层和透明度的空间外推出现稳定局部改善，而 TN 仅底层有弱正信号、TP 不稳定。不得把单次留站点的大幅提升或空间验证结果写成对中长期预测的普遍提升。
"""
    (out_dir / "分析报告.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    cv = pd.read_csv(args.cv_summary)
    macro = pd.read_csv(args.cv_macro)
    strict = strict_summary(pd.read_csv(args.strict))
    decisions = decision_table(cv, strict)
    cv.to_csv(args.out_dir / "表1_五折留站点任务汇总.csv", index=False, encoding="utf-8-sig")
    strict.to_csv(args.out_dir / "表2_严格验证方式比较.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(args.out_dir / "表3_空间泛化边界判定.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(args.out_dir / "表4_五折总体比较.csv", index=False, encoding="utf-8-sig")
    plot_cv_tasks(decisions, args.out_dir)
    plot_cv_macro(macro, args.out_dir)
    plot_validation_modes(strict, args.out_dir)
    write_report(args.out_dir, decisions, macro)
    print(f"完成：{args.out_dir}")


if __name__ == "__main__":
    main()
