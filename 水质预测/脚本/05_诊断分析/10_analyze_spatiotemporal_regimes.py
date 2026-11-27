from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, theilslopes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "数据/处理后数据/正式主表/water_quality_master_2014_2024.csv"
DEFAULT_OUT_DIR = ROOT / "结果/诊断结果/水质时空分异与水团背景_20260907"

VARIABLES = {
    "chla_ug_l": "Chl-a (ug/L)",
    "tn_mg_l": "TN (mg/L)",
    "tp_mg_l": "TP (mg/L)",
    "secchi_depth_m": "透明度 (m)",
    "salinity_psu": "盐度 (PSU)",
    "temperature_c": "水温 (C)",
    "turbidity_ntu": "浊度 (NTU)",
    "suspended_solids_mg_l": "悬浮固体 (mg/L)",
    "tin_mg_l": "无机氮 (mg/L)",
    "orthophosphate_p_mg_l": "正磷酸盐-P (mg/L)",
    "do_mg_l": "溶解氧 (mg/L)",
    "ph": "pH",
}
PCA_VARIABLES = [
    "salinity_psu",
    "temperature_c",
    "turbidity_ntu",
    "suspended_solids_mg_l",
    "tin_mg_l",
    "orthophosphate_p_mg_l",
    "chla_ug_l",
    "secchi_depth_m",
]
VERTICAL_VARIABLES = [field for field in VARIABLES if field != "secchi_depth_m"]
SEASON_ORDER = ["春", "夏", "秋", "冬"]
DEPTH_ORDER = ["surface", "middle", "bottom"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析水质时空分异和观测状态，不向预测模型写入同日水质变量。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})


def season_from_month(month: pd.Series) -> pd.Categorical:
    labels = pd.Series(np.select(
        [month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["春", "夏", "秋"],
        default="冬",
    ), index=month.index)
    return pd.Categorical(labels, categories=SEASON_ORDER, ordered=True)


def load_data(path: Path) -> pd.DataFrame:
    usecols = [
        "sample_id", "station", "station_name_en", "water_zone_en", "lon", "lat", "date", "year", "month", "depth",
        *VARIABLES.keys(),
    ]
    df = pd.read_csv(path, usecols=lambda col: col in usecols, parse_dates=["date"], low_memory=False)
    for col in VARIABLES:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["season"] = season_from_month(pd.to_numeric(df["month"], errors="coerce"))
    df["depth"] = pd.Categorical(df["depth"], categories=DEPTH_ORDER, ordered=True)
    return df


def variable_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in VARIABLES.items():
        series = df[col].dropna()
        rows.append({
            "变量": label,
            "字段": col,
            "有效样本数": int(series.size),
            "缺失率": float(df[col].isna().mean()),
            "最小值": series.min(),
            "P25": series.quantile(0.25),
            "中位数": series.median(),
            "均值": series.mean(),
            "P75": series.quantile(0.75),
            "P95": series.quantile(0.95),
            "最大值": series.max(),
        })
    return pd.DataFrame(rows)


def grouped_summary(df: pd.DataFrame) -> pd.DataFrame:
    long = df.melt(
        id_vars=["water_zone_en", "season", "depth"],
        value_vars=list(VARIABLES),
        var_name="字段",
        value_name="值",
    ).dropna(subset=["值"])
    out = long.groupby(["water_zone_en", "season", "depth", "字段"], observed=True)["值"].agg(
        样本数="count", 中位数="median", 均值="mean", P25=lambda x: x.quantile(0.25), P75=lambda x: x.quantile(0.75)
    ).reset_index()
    out.insert(4, "变量", out["字段"].map(VARIABLES))
    return out


def paired_vertical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in VERTICAL_VARIABLES:
        label = VARIABLES[col]
        work = df[["sample_id", "station", "water_zone_en", "season", "depth", col]].dropna().copy()
        wide = work.pivot_table(
            index=["sample_id", "station", "water_zone_en", "season"],
            columns="depth", values=col, aggfunc="first", observed=True,
        )
        if not {"surface", "bottom"}.issubset(wide.columns):
            continue
        delta = wide["surface"] - wide["bottom"]
        rows.append({
            "变量": label,
            "字段": col,
            "配对样本数": int(delta.notna().sum()),
            "表底层中位差": delta.median(),
            "表底层均值差": delta.mean(),
            "P25": delta.quantile(0.25),
            "P75": delta.quantile(0.75),
            "表层更高比例": (delta > 0).mean(),
        })
    return pd.DataFrame(rows)


def secchi_layer_consistency(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["sample_id", "depth", "secchi_depth_m"]].dropna().copy()
    wide = work.pivot_table(index="sample_id", columns="depth", values="secchi_depth_m", aggfunc="first", observed=True)
    all_three = wide.dropna(subset=["surface", "middle", "bottom"])
    exact_same = (
        np.isclose(all_three["surface"], all_three["middle"])
        & np.isclose(all_three["surface"], all_three["bottom"])
    )
    return pd.DataFrame([{
        "完整三层样本数": len(all_three),
        "三层透明度完全一致样本数": int(exact_same.sum()),
        "完全一致比例": float(exact_same.mean()) if len(all_three) else np.nan,
        "解释": "透明度为站点-日期塞氏盘观测，不应解释为三层独立垂向变量",
    }])


def trend_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (zone, depth, field), group in df.melt(
        id_vars=["water_zone_en", "depth", "year"], value_vars=list(VARIABLES), var_name="字段", value_name="值"
    ).dropna().groupby(["water_zone_en", "depth", "字段"], observed=True):
        annual = group.groupby("year", as_index=False)["值"].median().sort_values("year")
        if annual.shape[0] < 6:
            continue
        slope, _, _, _ = theilslopes(annual["值"], annual["year"])
        tau, pvalue = kendalltau(annual["year"], annual["值"])
        rows.append({
            "水区": zone,
            "水层": depth,
            "字段": field,
            "变量": VARIABLES[field],
            "年度数": annual.shape[0],
            "Sen斜率_每年": slope,
            "Kendall_tau": tau,
            "p值": pvalue,
            "趋势判断": "显著上升" if pvalue < 0.05 and slope > 0 else "显著下降" if pvalue < 0.05 and slope < 0 else "不显著",
        })
    return pd.DataFrame(rows)


def fit_observed_regimes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df[df["depth"] == "surface"].copy()
    complete = work.dropna(subset=PCA_VARIABLES).copy()
    scaler = StandardScaler()
    x = scaler.fit_transform(complete[PCA_VARIABLES])
    # Cluster-number selection is evaluated on a fixed subset, then the selected
    # model assigns the complete surface record. This avoids repeated all-record fits.
    sample_index = complete.sample(min(1000, len(complete)), random_state=42).index
    x_sample = x[complete.index.get_indexer(sample_index)]
    candidates = []
    for n_clusters in range(2, 6):
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels = model.fit_predict(x_sample)
        candidates.append({"类别数": n_clusters, "轮廓系数": silhouette_score(x_sample, labels)})
    score = pd.DataFrame(candidates)
    selected = int(score.sort_values(["轮廓系数", "类别数"], ascending=[False, True]).iloc[0]["类别数"])
    final_model = KMeans(n_clusters=selected, random_state=42, n_init=10).fit(x_sample)
    cluster = final_model.predict(x)
    pca = PCA(n_components=2, random_state=42, svd_solver="randomized")
    pcs = pca.fit_transform(x)
    complete["观测状态"] = [f"状态{value + 1}" for value in cluster]
    complete["PC1"] = pcs[:, 0]
    complete["PC2"] = pcs[:, 1]
    profiles = complete.groupby("观测状态", observed=True)[PCA_VARIABLES].median().reset_index()
    profiles.insert(1, "样本数", complete.groupby("观测状态", observed=True).size().values)
    profiles = profiles.rename(columns=VARIABLES)
    loadings = pd.DataFrame({
        "变量": [VARIABLES[col] for col in PCA_VARIABLES],
        "PC1载荷": pca.components_[0],
        "PC2载荷": pca.components_[1],
    })
    return complete, score, profiles, loadings


def regime_distribution(regimes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in ["water_zone_en", "season"]:
        counts = regimes.groupby([dimension, "观测状态"], observed=True).size().rename("样本数").reset_index()
        counts["比例"] = counts["样本数"] / counts.groupby(dimension, observed=True)["样本数"].transform("sum")
        counts.insert(0, "分组维度", dimension)
        counts = counts.rename(columns={dimension: "分组"})
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def plot_heatmaps(grouped: pd.DataFrame, out_dir: Path) -> None:
    targets = ["chla_ug_l", "tn_mg_l", "tp_mg_l", "secchi_depth_m"]
    fig, axes = plt.subplots(4, 3, figsize=(14, 15), constrained_layout=True)
    for row, field in enumerate(targets):
        for col, depth in enumerate(DEPTH_ORDER):
            ax = axes[row, col]
            if field == "secchi_depth_m" and depth != "surface":
                ax.set_axis_off()
                ax.text(.5, .5, "透明度为站点-日期\n塞氏盘观测\n不作水层比较", ha="center", va="center", fontsize=11)
                continue
            data = grouped[(grouped["字段"] == field) & (grouped["depth"] == depth)]
            table = data.pivot(index="water_zone_en", columns="season", values="中位数").reindex(columns=SEASON_ORDER)
            sns.heatmap(table, cmap="YlGnBu", ax=ax, cbar=row == 0 or col == 2, linewidths=.25)
            ax.set_title(f"{VARIABLES[field]} | {depth}")
            ax.set_xlabel("")
            ax.set_ylabel("水区" if col == 0 else "")
    fig.savefig(out_dir / "图1_四指标分水区季节水层中位数热图.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_vertical(vertical: pd.DataFrame, out_dir: Path) -> None:
    work = vertical.sort_values("表底层中位差").copy()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.errorbar(
        work["表底层中位差"], range(len(work)),
        xerr=[work["表底层中位差"] - work["P25"], work["P75"] - work["表底层中位差"]],
        fmt="o", color="#167c80", ecolor="#6f9b9b", capsize=3,
    )
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(work)))
    ax.set_yticklabels(work["变量"])
    ax.set_xlabel("表层 - 底层（配对观测中位差，括号为四分位距）")
    ax.set_title("表底层水质差异")
    fig.savefig(out_dir / "图2_表底层配对差异.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_trends(trends: pd.DataFrame, out_dir: Path) -> None:
    fields = ["chla_ug_l", "tn_mg_l", "tp_mg_l", "secchi_depth_m"]
    work = trends[(trends["水层"] == "surface") & trends["字段"].isin(fields)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, field in zip(axes.flat, fields):
        panel = work[work["字段"] == field]
        table = panel.pivot(index="水区", columns="变量", values="Sen斜率_每年")
        sns.heatmap(table, center=0, cmap="RdBu_r", annot=True, fmt=".3g", ax=ax, cbar_kws={"label": "Sen slope/year"})
        ax.set_title(f"表层 {VARIABLES[field]} 年际趋势")
        ax.set_xlabel("")
        ax.set_ylabel("水区")
    fig.savefig(out_dir / "图3_表层四指标分水区年际趋势.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_regimes(regimes: pd.DataFrame, out_dir: Path) -> None:
    sample = regimes.sample(min(6000, len(regimes)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    sns.scatterplot(data=sample, x="PC1", y="PC2", hue="观测状态", alpha=.55, s=16, linewidth=0, palette="Set2", ax=ax)
    ax.set_title("表层观测水质状态的主成分分布")
    ax.legend(title="状态", frameon=True)
    fig.savefig(out_dir / "图4_表层观测水质状态聚类.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir: Path, df: pd.DataFrame, vertical: pd.DataFrame, trends: pd.DataFrame, scores: pd.DataFrame, profiles: pd.DataFrame, distribution: pd.DataFrame, secchi_check: pd.DataFrame) -> None:
    selected = int(scores.sort_values(["轮廓系数", "类别数"], ascending=[False, True]).iloc[0]["类别数"])
    significant = int((trends["趋势判断"] != "不显著").sum())
    low_salt_state = profiles.sort_values("盐度 (PSU)").iloc[0]["观测状态"]
    zone_share = distribution[(distribution["分组维度"] == "water_zone_en") & (distribution["观测状态"] == low_salt_state)]
    top_zone = zone_share.sort_values("比例", ascending=False).iloc[0]
    text = f"""# 水质时空分异与水团背景分析

## 分析定位

本分析使用 {len(df):,} 条原始水质观测、{df['station'].nunique()} 个站点、{df['water_zone_en'].nunique()} 个水区和三水层记录。盐度、温度、浊度、悬浮固体、无机氮、正磷酸盐、DO、pH 等同日水质变量**仅用于描述水质状态和机理背景**，没有合并到正式中长期预测模型的输入特征中。

## 结果摘要

- 已生成四指标在“水区—季节—水层”尺度上的中位数分布、表底层配对差异及表层年际趋势。
- 表底层配对分析覆盖 {int(vertical['配对样本数'].min()):,} 至 {int(vertical['配对样本数'].max()):,} 对同期样本；具体变量差异见 `表3_表底层配对差异.csv`。
- 透明度在 {int(secchi_check.iloc[0]['完整三层样本数']):,} 个完整三层样本中有 {secchi_check.iloc[0]['完全一致比例']:.1%} 的三层记录完全一致。因此透明度作为站点—日期塞氏盘观测保留在水平和季节比较中，但不进入垂向差异讨论，也不应作为三层独立观测解释。
- 趋势筛查中共有 {significant} 个“水区—水层—变量”组合达到 Mann-Kendall p<0.05。由于该筛查未对季节性、重复检验和自相关作正式校正，图 3 仅保留为探索性附录，不作为正文核心趋势结论或因果归因依据。
- 在完整表层观测中，以 {len(PCA_VARIABLES)} 个变量构建的无监督状态分析选择 {selected} 类（最大轮廓系数见 `表5_观测状态聚类质量.csv`）。这些类别代表观测水质状态，不等同于严格物理水团，也不应作为预测标签。
- 低盐观测状态 `{low_salt_state}` 在 `{top_zone['分组']}` 的表层观测中占 {top_zone['比例']:.1%}；其余水区的分布见 `表9_观测状态水区季节分布.csv`。该空间聚集支持将其作为淡水影响背景的描述性线索，但不作为因果识别。

## 是否放入论文

**建议放入正文。** 建议在数据与研究区章节增加“水质时空分异与观测状态背景”小节，正文使用图 1、图 2 和图 4；图 3 作为探索性补充材料，待完成季节调整与多重检验校正后再决定是否升级。不得把同日水质相关性写成预测模型的贡献或因果证明。
"""
    (out_dir / "分析报告.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    df = load_data(args.input)
    print("已读取数据", flush=True)
    summary = variable_summary(df)
    print("已完成描述统计", flush=True)
    grouped = grouped_summary(df)
    print("已完成分组统计", flush=True)
    vertical = paired_vertical_summary(df)
    secchi_check = secchi_layer_consistency(df)
    print("已完成垂向配对", flush=True)
    trends = trend_summary(df)
    print("已完成趋势检验", flush=True)
    regimes, scores, profiles, loadings = fit_observed_regimes(df)
    distribution = regime_distribution(regimes)
    print("已完成观测状态分析", flush=True)
    summary.to_csv(args.out_dir / "表1_原始水质变量描述统计.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(args.out_dir / "表2_水区季节水层统计.csv", index=False, encoding="utf-8-sig")
    vertical.to_csv(args.out_dir / "表3_表底层配对差异.csv", index=False, encoding="utf-8-sig")
    secchi_check.to_csv(args.out_dir / "表3a_透明度水层一致性检查.csv", index=False, encoding="utf-8-sig")
    trends.to_csv(args.out_dir / "表4_年际趋势检验.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(args.out_dir / "表5_观测状态聚类质量.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(args.out_dir / "表6_观测状态特征概况.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(args.out_dir / "表7_观测状态主成分载荷.csv", index=False, encoding="utf-8-sig")
    regimes.to_csv(args.out_dir / "表8_表层观测状态样本.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(args.out_dir / "表9_观测状态水区季节分布.csv", index=False, encoding="utf-8-sig")
    plot_heatmaps(grouped, args.out_dir)
    print("已绘制热图", flush=True)
    plot_vertical(vertical, args.out_dir)
    print("已绘制垂向图", flush=True)
    plot_trends(trends, args.out_dir)
    print("已绘制趋势图", flush=True)
    plot_regimes(regimes, args.out_dir)
    print("已绘制状态图", flush=True)
    write_report(args.out_dir, df, vertical, trends, scores, profiles, distribution, secchi_check)
    print(f"完成：{args.out_dir}")


if __name__ == "__main__":
    main()
