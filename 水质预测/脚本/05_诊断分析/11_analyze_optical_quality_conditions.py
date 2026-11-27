from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = ROOT / "结果/扩展实验/质量感知遥感门控融合_20260905/质量感知遥感融合_指标.csv"
DEFAULT_PREDICTIONS = ROOT / "结果/扩展实验/质量感知遥感门控融合_20260905/质量感知遥感融合_预测.csv"
DEFAULT_S2 = ROOT / "数据/外部数据/试验项目接管_20260521/data_external_gee_exports/s2_patch_features_2014_2024.csv"
DEFAULT_OUT_DIR = ROOT / "结果/诊断结果/遥感质量与光学类型_20260907"
TARGETS = ["chla", "secchi"]
DEPTHS = ["surface", "middle", "bottom"]
MODEL_COLUMNS = {
    "f1_history": "prediction_f1_history",
    "remote_expert": "prediction_remote_expert",
    "quality_gate_fusion": "prediction_quality_gate_fusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析遥感质量条件和光学观测类型，不重训正式主模型。")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--s2", type=Path, default=DEFAULT_S2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})


def quality_ab(metrics: pd.DataFrame) -> pd.DataFrame:
    subset = metrics[metrics["target"].isin(TARGETS)].copy()
    index = ["target", "depth", "quality_group", "rows"]
    wide = subset.pivot_table(index=index, columns="model_variant", values="r2", aggfunc="first").reset_index()
    for variant in MODEL_COLUMNS:
        if variant not in wide:
            wide[variant] = np.nan
    wide["遥感专家减历史模型_R2"] = wide["remote_expert"] - wide["f1_history"]
    wide["门控融合减历史模型_R2"] = wide["quality_gate_fusion"] - wide["f1_history"]
    wide["证据等级"] = np.select(
        [
            wide["rows"] < 100,
            (wide["rows"] >= 100) & (wide["门控融合减历史模型_R2"] >= .02),
            (wide["rows"] >= 100) & (wide["门控融合减历史模型_R2"] <= -.02),
        ],
        ["样本不足", "局部正信号", "局部负信号"],
        default="效应很小",
    )
    return wide.sort_values(["target", "depth", "quality_group"])


def strict_s2_optical_regimes(s2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    fields = [
        "s2_3x3_B2_mean", "s2_3x3_B3_mean", "s2_3x3_B4_mean", "s2_3x3_B8_mean",
        "s2_3x3_NDCI_mean", "s2_3x3_NDWI_mean", "s2_3x3_FAI_mean",
    ]
    work = s2[["sample_id", "station", "s2_image_count_in_window", "s2_date_diff_abs", "s2_cloudy_pixel_percentage", *fields]].copy()
    for col in fields + ["s2_image_count_in_window", "s2_date_diff_abs", "s2_cloudy_pixel_percentage"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    mask = (
        work["s2_image_count_in_window"].ge(1)
        & work["s2_date_diff_abs"].le(3)
        & work["s2_cloudy_pixel_percentage"].le(20)
    )
    strict = work.loc[mask].dropna(subset=fields).copy()
    # Remove only the outer 1% of each optical descriptor after strict image screening.
    for col in fields:
        lo, hi = strict[col].quantile([.01, .99])
        strict = strict[strict[col].between(lo, hi)]
    x = StandardScaler().fit_transform(strict[fields])
    candidates = []
    for n_clusters in range(2, 5):
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=30).fit_predict(x)
        candidates.append({"类别数": n_clusters, "轮廓系数": silhouette_score(x, labels)})
    scores = pd.DataFrame(candidates)
    n = int(scores.sort_values(["轮廓系数", "类别数"], ascending=[False, True]).iloc[0]["类别数"])
    strict["光学观测类型"] = [f"类型{i + 1}" for i in KMeans(n_clusters=n, random_state=42, n_init=50).fit_predict(x)]
    profile = strict.groupby("光学观测类型", observed=True)[fields].median().reset_index()
    profile.insert(1, "样本数", strict.groupby("光学观测类型", observed=True).size().values)
    coverage = {
        "S2总样本数": int(len(s2)),
        "严格筛选样本数": int(len(strict)),
        "严格筛选占比": float(len(strict) / len(s2)),
        "选择类别数": n,
    }
    return strict, scores, coverage | {"特征数": len(fields)}, profile


def evaluate_by_regime(predictions: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    data = predictions[predictions["target"].isin(TARGETS)].merge(regimes[["sample_id", "光学观测类型"]], on="sample_id", how="inner")
    rows = []
    for (target, depth, regime), group in data.groupby(["target", "depth", "光学观测类型"], observed=True):
        if len(group) < 30:
            continue
        record = {"target": target, "depth": depth, "光学观测类型": regime, "样本数": len(group)}
        for label, column in MODEL_COLUMNS.items():
            record[f"{label}_R2"] = r2_score(group["observed"], group[column])
            record[f"{label}_RMSE"] = mean_squared_error(group["observed"], group[column]) ** .5
        record["遥感专家减历史模型_R2"] = record["remote_expert_R2"] - record["f1_history_R2"]
        record["门控融合减历史模型_R2"] = record["quality_gate_fusion_R2"] - record["f1_history_R2"]
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["target", "depth", "光学观测类型"])


def plot_quality_ab(quality: pd.DataFrame, out_dir: Path) -> None:
    work = quality[quality["quality_group"] != "no_valid_remote"].copy()
    work["任务"] = work["target"].str.upper() + " | " + work["depth"]
    long = work.melt(
        id_vars=["任务", "quality_group", "rows"],
        value_vars=["遥感专家减历史模型_R2", "门控融合减历史模型_R2"],
        var_name="比较", value_name="R2变化",
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True, constrained_layout=True)
    for ax, comparison in zip(axes, ["遥感专家减历史模型_R2", "门控融合减历史模型_R2"]):
        panel = long[long["比较"] == comparison]
        sns.barplot(data=panel, x="任务", y="R2变化", hue="quality_group", errorbar=None, ax=ax)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(comparison.replace("_R2", ""))
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="影像条件", frameon=True)
    fig.suptitle("不同遥感质量条件下的模型R2变化（相对历史过程模型）")
    fig.savefig(out_dir / "图1_遥感质量条件下的R2变化.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_optical_regimes(regimes: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sns.scatterplot(
        data=regimes.sample(min(4000, len(regimes)), random_state=42),
        x="s2_3x3_B3_mean", y="s2_3x3_B4_mean", hue="光学观测类型", alpha=.65, s=18, linewidth=0, palette="Set2", ax=ax,
    )
    ax.set_title("严格质量筛选后的 Sentinel-2 光学观测类型")
    ax.set_xlabel("B3 反射率")
    ax.set_ylabel("B4 反射率")
    fig.savefig(out_dir / "图2_严格筛选S2光学观测类型.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_regime_results(results: pd.DataFrame, out_dir: Path) -> None:
    if results.empty:
        return
    work = results.copy()
    work["任务"] = work["target"].str.upper() + " | " + work["depth"]
    long = work.melt(
        id_vars=["任务", "光学观测类型", "样本数"],
        value_vars=["遥感专家减历史模型_R2", "门控融合减历史模型_R2"],
        var_name="比较", value_name="R2变化",
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True, constrained_layout=True)
    for ax, comparison in zip(axes, ["遥感专家减历史模型_R2", "门控融合减历史模型_R2"]):
        panel = long[long["比较"] == comparison]
        sns.barplot(data=panel, x="任务", y="R2变化", hue="光学观测类型", errorbar=None, ax=ax)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(comparison.replace("_R2", ""))
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="光学类型", frameon=True)
    fig.suptitle("不同光学观测类型下的遥感增益")
    fig.savefig(out_dir / "图3_光学观测类型下的遥感增益.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir: Path, quality: pd.DataFrame, coverage: dict[str, float], regime_results: pd.DataFrame, s2: pd.DataFrame) -> None:
    all_rows = quality[quality["quality_group"] == "all"]
    gate_delta = all_rows["门控融合减历史模型_R2"].mean()
    positive = int((regime_results["门控融合减历史模型_R2"] >= .02).sum()) if not regime_results.empty else 0
    total = int(len(regime_results))
    invalid_ratio = float((pd.to_numeric(s2["s2_3x3_valid_ratio"], errors="coerce") > 1).mean()) if "s2_3x3_valid_ratio" in s2 else np.nan
    text = f"""# 遥感质量与光学观测类型分析

## 分析定位

本分析复用已完成的“质量感知遥感门控融合”固定随机 80/20 测试预测，未重新训练正式主模型。所有比较仍明确排除同日其他水质协变量。光学类型仅在 Sentinel-2 近同日（绝对日期差不超过 3 天）、云量不高于 20%、且光谱字段完整的样本中描述性识别。

## 覆盖与质量

- Sentinel-2 总样本数：{coverage['S2总样本数']:,}；严格质量筛选后：{coverage['严格筛选样本数']:,}（{coverage['严格筛选占比']:.1%}）。
- 严格样本依据 {coverage['特征数']} 个光学变量形成 {coverage['选择类别数']} 类观测类型。该类别描述影像中的水体光学状态，不构成严格水团分类。
- 原始 `s2_3x3_valid_ratio` 中有 {invalid_ratio:.1%} 的值大于 1，因此本分析没有把该字段解释为比例或作为筛选阈值；该 GEE 字段应在后续导出中复核。

## 模型结论

- 全样本上，质量门控融合相对历史过程模型的平均 R2 变化为 {gate_delta:+.4f}，不达到推广为正式统一模型的阈值。
- 严格光学类型子样本中，门控融合出现 R2 增益不低于 0.02 的任务—类型组合为 {positive}/{total}。这类结果只代表局部条件信号，须同时报告样本量。

## 是否放入论文

**建议以“遥感适用条件敏感性分析”放入结果与讨论，不作为正式主模型替换依据。** 正文应报告覆盖率、质量分层和增益不稳定性，结论可表述为：遥感信息对部分光学条件和任务存在局部辅助价值，但受可用影像覆盖与质量制约，尚不形成对全体四指标三水层任务的稳定增益。
"""
    (out_dir / "分析报告.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    metrics = pd.read_csv(args.metrics)
    predictions = pd.read_csv(args.predictions)
    s2 = pd.read_csv(args.s2, low_memory=False)
    quality = quality_ab(metrics)
    regimes, scores, coverage, profiles = strict_s2_optical_regimes(s2)
    regime_results = evaluate_by_regime(predictions, regimes)
    quality.to_csv(args.out_dir / "表1_遥感质量分层模型比较.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(args.out_dir / "表2_光学类型聚类质量.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([coverage]).to_csv(args.out_dir / "表3_严格S2质量覆盖率.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(args.out_dir / "表4_光学观测类型特征概况.csv", index=False, encoding="utf-8-sig")
    regime_results.to_csv(args.out_dir / "表5_光学观测类型下模型比较.csv", index=False, encoding="utf-8-sig")
    plot_quality_ab(quality, args.out_dir)
    plot_optical_regimes(regimes, args.out_dir)
    plot_regime_results(regime_results, args.out_dir)
    write_report(args.out_dir, quality, coverage, regime_results, s2)
    print(f"完成：{args.out_dir}")


if __name__ == "__main__":
    main()
