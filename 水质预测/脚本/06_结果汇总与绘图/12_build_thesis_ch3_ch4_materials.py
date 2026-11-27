from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_REPORT_DIR = Path("结果/正式结果/full_station")
DEFAULT_DIAG_DIR = Path("结果/诊断结果/chla_surface_diagnostics")
DEFAULT_SHAP_DIR = Path("结果/论文图表/full_station_shap")
DEFAULT_FIG_DIR = Path("结果/论文图表/论文图版_随机比例划分_20260905/第三四章")
DEFAULT_TABLE_DIR = Path("结果/论文图表/论文表格_随机比例划分_20260905/第三四章")

DEPTH_LABELS = {"surface": "表层", "middle": "中层", "bottom": "底层"}
FEATURE_LABELS = {
    "baseline": "基线组",
    "f1_history": "历史与过程组",
    "f2_remote": "遥感组",
    "f3_history_remote": "历史—遥感融合组",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成论文第三、四章的正式模型结果和遥感敏感性图表。")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--diag-dir", type=Path, default=DEFAULT_DIAG_DIR)
    parser.add_argument("--shap-dir", type=Path, default=DEFAULT_SHAP_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少输入文件: {path}")
    return pd.read_csv(path, low_memory=False)


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def copy_figure(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"缺少已有图件: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def make_random_split(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.barh(0.64, 0.8, left=0.1, height=0.22, color="#2a9d8f")
    ax.barh(0.32, 0.2, left=0.1, height=0.22, color="#e76f51")
    ax.text(0.5, 0.64, "训练集：80% 随机抽样样本", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.2, 0.32, "测试集：20% 随机留出样本", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.92, "正式模型固定随机比例划分（random_state = 42）", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.06, "每个指标—水层任务在全部有效样本中随机划分；该评价反映同分布留出样本表现，不等同于未来年份外推。", ha="center", va="center", fontsize=10)
    fig.savefig(fig_dir / "图3-1_随机比例划分.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_remote_coverage(coverage: pd.DataFrame, fig_dir: Path) -> None:
    values = dict(zip(coverage["metric"], coverage["value"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    shares = pd.DataFrame(
        {
            "指标": ["Landsat 可用样本比例", "S2 特征字段进入主表"],
            "比例": [values.get("landsat_available_share", float("nan")), values.get("s2_feature_fields_present", float("nan"))],
        }
    )
    sns.barplot(data=shares, x="指标", y="比例", hue="指标", legend=False, palette=["#457b9d", "#f4a261"], ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("正式主表遥感字段覆盖")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("比例")
    for index, value in enumerate(shares["比例"]):
        axes[0].text(index, min(float(value) + 0.03, 0.97), f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    pixels = pd.DataFrame({"传感器": ["Landsat"], "有效影像数中位数": [values.get("landsat_valid_count_median", float("nan"))]})
    sns.barplot(data=pixels, x="传感器", y="有效影像数中位数", hue="传感器", legend=False, palette=["#457b9d"], ax=axes[1])
    axes[1].set_title("Landsat 有效影像数中位数")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("影像数")
    upper = max(float(pixels["有效影像数中位数"].max()), 1.0)
    for index, value in enumerate(pixels["有效影像数中位数"]):
        axes[1].text(index, float(value) + upper * 0.02, f"{value:.0f}", ha="center", va="bottom", fontsize=10)
    fig.savefig(fig_dir / "图4-1_正式主表遥感质量覆盖.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_remote_feature_composition(manifest: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in manifest[manifest["feature_set"] == "f3_history_remote"].iterrows():
        features = json.loads(row["features_json"])
        landsat_count = sum("landsat" in feature.lower() for feature in features)
        s2_count = sum(feature.lower().startswith("s2_") for feature in features)
        rows.append(
            {
                "指标": str(row["target"]).upper(),
                "水层": DEPTH_LABELS.get(row["depth"], row["depth"]),
                "Landsat 特征数": landsat_count,
                "Sentinel-2 特征数": s2_count,
                "遥感特征总数": landsat_count + s2_count,
            }
        )
    summary = pd.DataFrame(rows)
    plot = summary.melt(id_vars=["指标", "水层"], value_vars=["Landsat 特征数", "Sentinel-2 特征数"], var_name="特征来源", value_name="特征数")
    plot["任务"] = plot["指标"] + " | " + plot["水层"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.barplot(data=plot, y="任务", x="特征数", hue="特征来源", palette=["#457b9d", "#f4a261"], ax=ax)
    ax.set_title("融合模型中 Landsat 与 Sentinel-2 遥感特征构成")
    ax.set_xlabel("特征数")
    ax.set_ylabel("")
    ax.legend(title="")
    fig.savefig(fig_dir / "图4-2_融合模型遥感特征构成.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def make_remote_gain_sensitivity(fusion: pd.DataFrame, fig_dir: Path) -> None:
    work = fusion.copy()
    work["任务"] = work["target"].str.upper() + " | " + work["depth"].map(DEPTH_LABELS)
    values = work.set_index("任务")["history_remote_minus_history_r2"].sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(
        values.to_frame("融合组相对历史组的 R2 变化"),
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "R2 变化"},
        ax=ax,
    )
    ax.set_title("遥感并入历史—过程模型后的任务级增益判定")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(fig_dir / "图4-5_遥感融合任务级增益判定.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_remote_gain_by_depth(fusion: pd.DataFrame, fig_dir: Path) -> None:
    work = fusion.copy()
    work["水层"] = work["depth"].map(DEPTH_LABELS)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.boxplot(data=work, x="水层", y="history_remote_minus_history_r2", order=["表层", "中层", "底层"], color="#a8dadc", showfliers=False, ax=ax)
    sns.stripplot(data=work, x="水层", y="history_remote_minus_history_r2", order=["表层", "中层", "底层"], color="#1d3557", size=6, alpha=0.75, ax=ax)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("不同水层中遥感信息对融合模型的增益分布")
    ax.set_xlabel("")
    ax.set_ylabel("融合组相对历史组的 R2 变化")
    fig.savefig(fig_dir / "图4-6_不同水层遥感增益分布.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_ch3_tables(metrics: pd.DataFrame, best: pd.DataFrame, manifest: pd.DataFrame, table_dir: Path) -> None:
    task_table = (
        metrics[metrics["status"] == "ok"]
        .groupby(["target", "target_column", "depth"], as_index=False)
        .agg(训练样本数=("train_rows", "first"), 测试样本数=("test_rows", "first"))
    )
    task_table["指标"] = task_table["target"].str.upper()
    task_table["水层"] = task_table["depth"].map(DEPTH_LABELS)
    task_table["目标变换"] = task_table["target"].map({"chla": "log1p", "tn": "log1p", "tp": "log1p", "secchi": "log1p"}).fillna("以训练脚本为准")
    write_csv(task_table[["指标", "target_column", "水层", "训练样本数", "测试样本数", "目标变换"]], table_dir / "表3-1_十二个预测任务与样本量.csv")

    feature_rows: list[dict[str, object]] = []
    for feature_set, group in manifest.groupby("feature_set"):
        feature_rows.append(
            {
                "特征组": FEATURE_LABELS.get(feature_set, feature_set),
                "特征组代码": feature_set,
                "典型特征数": int(group["feature_count"].median()),
                "说明": {
                    "baseline": "时间、空间位置、海区类别等基线信息",
                    "f1_history": "基线信息、目标变量历史滞后、环境过程及其时间窗口",
                    "f2_remote": "高质量遥感可用性及遥感特征",
                    "f3_history_remote": "历史与过程特征，并入遥感信息的融合组",
                }.get(feature_set, "以特征清单为准"),
                "是否使用同期其他水质指标": "否",
            }
        )
    write_csv(pd.DataFrame(feature_rows), table_dir / "表3-2_正式模型特征组定义.csv")

    result_table = metrics[metrics["status"] == "ok"].copy()
    result_table["指标"] = result_table["target"].str.upper()
    result_table["水层"] = result_table["depth"].map(DEPTH_LABELS)
    result_table["特征组"] = result_table["feature_set"].map(FEATURE_LABELS)
    result_table = result_table.rename(
        columns={
            "train_rows": "训练样本数",
            "test_rows": "测试样本数",
            "r2": "R2",
            "rmse": "RMSE",
            "mae": "MAE",
            "nse": "NSE",
        }
    )
    write_csv(result_table[["指标", "水层", "特征组", "训练样本数", "测试样本数", "R2", "RMSE", "MAE", "NSE"]], table_dir / "表3-3_十二个任务的正式随机留出结果.csv")

    best_table = best.copy()
    best_table["指标"] = best_table["target"].str.upper()
    best_table["水层"] = best_table["depth"].map(DEPTH_LABELS)
    best_table["最优特征组"] = best_table["feature_set"].map(FEATURE_LABELS)
    best_table = best_table.rename(columns={"r2": "R2", "rmse": "RMSE", "mae": "MAE", "nse": "NSE"})
    write_csv(best_table[["指标", "水层", "最优特征组", "R2", "RMSE", "MAE", "NSE"]], table_dir / "表3-4_各任务最佳正式模型.csv")


def build_ch4_tables(coverage: pd.DataFrame, fusion: pd.DataFrame, remote_composition: pd.DataFrame, table_dir: Path) -> None:
    labels = {
        "landsat_available_share": "Landsat 可用样本比例",
        "landsat_valid_count_median": "Landsat 有效影像数中位数",
        "s2_feature_fields_present": "Sentinel-2 特征字段是否进入正式主表",
        "ssrd_7d_missing_share": "SSRD 7 日窗口缺失率",
        "sst_7d_missing_share": "SST 7 日窗口缺失率",
    }
    quality = coverage.copy()
    quality["指标"] = quality["metric"].map(labels).fillna(quality["metric"])
    quality["数值"] = quality["value"].round(4)
    write_csv(quality[["指标", "数值"]], table_dir / "表4-1_遥感与短时间窗过程特征质量覆盖.csv")
    write_csv(remote_composition, table_dir / "表4-2_融合模型遥感特征构成.csv")

    gains = fusion.copy()
    gains["指标"] = gains["target"].str.upper()
    gains["水层"] = gains["depth"].map(DEPTH_LABELS)
    keep = [
        "指标",
        "水层",
        "history_remote_minus_history_r2",
        "history_remote_minus_baseline_r2",
        "history_remote_minus_history_rmse",
        "history_remote_minus_baseline_rmse",
    ]
    gains = gains[[column for column in keep if column in gains.columns]].rename(
        columns={
            "history_remote_minus_history_r2": "融合组相对历史组_R2变化",
            "history_remote_minus_baseline_r2": "融合组相对基线组_R2变化",
            "history_remote_minus_history_rmse": "融合组相对历史组_RMSE变化",
            "history_remote_minus_baseline_rmse": "融合组相对基线组_RMSE变化",
        }
    )
    write_csv(gains, table_dir / "表4-3_遥感融合增益判定.csv")


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    metrics = read_csv(args.report_dir / "full_station_training_metrics.csv")
    best = read_csv(args.report_dir / "full_station_best_by_target_depth.csv")
    manifest = read_csv(args.report_dir / "full_station_feature_manifest.csv")
    fusion = read_csv(args.report_dir / "full_station_fusion_gain.csv")
    coverage = read_csv(args.diag_dir / "coverage_summary.csv")

    make_random_split(args.fig_dir)
    copy_figure(args.report_dir.parent.parent / "论文图表/full_station_figures/fig01_metric_heatmaps.png", args.fig_dir / "图3-2_十二个任务预测指标热图.png")
    copy_figure(args.report_dir.parent.parent / "论文图表/full_station_figures/fig02_feature_set_summary.png", args.fig_dir / "图3-3_特征组总体性能比较.png")
    copy_figure(args.report_dir.parent.parent / "论文图表/full_station_figures/fig03_fusion_gain_r2.png", args.fig_dir / "图3-4_历史与遥感融合增益.png")
    copy_figure(args.report_dir.parent.parent / "论文图表/full_station_figures/fig04_best_model_scatter.png", args.fig_dir / "图3-5_最佳正式模型预测散点图.png")
    make_remote_coverage(coverage, args.fig_dir)
    remote_composition = make_remote_feature_composition(manifest, args.fig_dir)
    copy_figure(args.shap_dir / "fig05_remote_shap_importance_grid.png", args.fig_dir / "图4-3_遥感特征SHAP重要性.png")
    copy_figure(args.shap_dir / "fig06_remote_top_feature_dependence_grid.png", args.fig_dir / "图4-4_遥感主要特征SHAP依赖关系.png")
    make_remote_gain_sensitivity(fusion, args.fig_dir)
    make_remote_gain_by_depth(fusion, args.fig_dir)
    build_ch3_tables(metrics, best, manifest, args.table_dir)
    build_ch4_tables(coverage, fusion, remote_composition, args.table_dir)
    print(f"已生成第三、四章图表: {args.fig_dir} / {args.table_dir}")


if __name__ == "__main__":
    main()
