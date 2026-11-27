from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import seaborn as sns


DEFAULT_DATASET = Path("数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv")
DEFAULT_COVERAGE = Path("结果/诊断结果/chla_surface_diagnostics/coverage_summary.csv")
DEFAULT_FIG_DIR = Path("结果/论文图表/论文图版_随机比例划分_20260905/第一二章")
DEFAULT_TABLE_DIR = Path("结果/论文图表/论文表格_随机比例划分_20260905/第一二章")

TARGETS = {
    "chla_ug_l": ("Chl-a", "ug/L"),
    "secchi_depth_m": ("Secchi", "m"),
    "tn_mg_l": ("TN", "mg/L"),
    "tp_mg_l": ("TP", "mg/L"),
}
DEPTH_LABELS = {"surface": "表层", "middle": "中层", "bottom": "底层"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成论文第一、二章的研究框架、数据概况图和基础表。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})


def load_dataset(path: Path) -> pd.DataFrame:
    columns = [
        "obs_id",
        "sample_id",
        "station",
        "station_name_en",
        "water_zone_en",
        "lon",
        "lat",
        "date",
        "year",
        "month",
        "depth",
        *TARGETS.keys(),
    ]
    if not path.exists():
        raise FileNotFoundError(f"缺少正式建模主表: {path}")
    data = pd.read_csv(path, usecols=columns, parse_dates=["date"], low_memory=False)
    for target in TARGETS:
        data[target] = pd.to_numeric(data[target], errors="coerce")
    return data


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax: plt.Axes, x: float, y: float, label: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x - 0.13, y - 0.08),
        0.26,
        0.16,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#264653",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=10, wrap=True)


def make_technical_route(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        (0.10, 0.73, "水质观测\n站点与海区", "#e9f5db"),
        (0.10, 0.30, "遥感、气象\n降雨、潮汐、夜光", "#e9f5db"),
        (0.35, 0.52, "站点—日期—水层\n样本与质量控制", "#d8f3dc"),
        (0.58, 0.52, "12 个任务\n随机比例留出", "#d0ebff"),
        (0.80, 0.73, "特征组消融\n预测性能比较", "#ffe8cc"),
        (0.80, 0.30, "遥感敏感性、SHAP\n残差与分海区诊断", "#ffe8cc"),
    ]
    for x, y, label, color in nodes:
        draw_box(ax, x, y, label, color)
    arrows = [
        ((0.23, 0.73), (0.22, 0.56)),
        ((0.23, 0.30), (0.22, 0.48)),
        ((0.48, 0.52), (0.45, 0.52)),
        ((0.71, 0.58), (0.67, 0.69)),
        ((0.71, 0.46), (0.67, 0.34)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#264653"})
    ax.text(0.80, 0.06, "结论：预测能力、遥感适用边界与近岸水环境管理启示", ha="center", fontsize=11, weight="bold")
    save_figure(fig, fig_dir / "图1-1_研究技术路线.png")


def make_question_framework(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    questions = [
        (0.18, "问题一\n多源信息能否提升\n三水层水质预测表现？", "第 3 章\n12 任务随机留出评估"),
        (0.50, "问题二\n遥感对不同指标和水层\n是否有稳定增益？", "第 4 章\n质量与敏感性分析"),
        (0.82, "问题三\n为何 Chl-a 高值和不同海区\n预测困难？", "第 5 章\n残差与局地诊断"),
    ]
    for x, question, chapter in questions:
        draw_box(ax, x, 0.68, question, "#dbeafe")
        draw_box(ax, x, 0.25, chapter, "#fef3c7")
        ax.annotate("", xy=(x, 0.36), xytext=(x, 0.56), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#264653"})
    save_figure(fig, fig_dir / "图1-2_科学问题与章节关系.png")


def make_station_zone_map(data: pd.DataFrame, fig_dir: Path) -> None:
    stations = data[["station", "station_name_en", "water_zone_en", "lon", "lat"]].drop_duplicates("station").dropna()
    zones = sorted(stations["water_zone_en"].unique())
    palette = sns.color_palette("tab20", n_colors=len(zones))
    color_map = dict(zip(zones, palette))
    fig, ax = plt.subplots(figsize=(9, 8))
    for zone in zones:
        group = stations[stations["water_zone_en"] == zone]
        ax.scatter(group["lon"], group["lat"], s=38, color=color_map[zone], edgecolors="white", linewidths=0.5, label=zone, alpha=0.9)
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.set_title("香港近岸水质监测站点及海区类别分布")
    ax.legend(title="海区", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, fig_dir / "图2-1_监测站点与海区分布.png")


def make_depth_coverage(data: pd.DataFrame, fig_dir: Path) -> None:
    coverage = (
        data.groupby("depth")
        .agg(观测样本数=("obs_id", "nunique"), 站点数=("station", "nunique"), 站点日期数=("sample_id", "nunique"))
        .reindex(["surface", "middle", "bottom"])
        .reset_index()
    )
    coverage["水层"] = coverage["depth"].map(DEPTH_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    sns.barplot(data=coverage, x="水层", y="观测样本数", hue="水层", legend=False, palette="Blues_d", ax=axes[0])
    axes[0].set_title("三水层观测样本数")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("样本数")
    sns.barplot(data=coverage, x="水层", y="站点数", hue="水层", legend=False, palette="Greens_d", ax=axes[1])
    axes[1].set_title("三水层站点覆盖数")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("站点数")
    save_figure(fig, fig_dir / "图2-2_三水层观测覆盖.png")


def make_target_distributions(data: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (column, (label, unit)) in zip(axes.flat, TARGETS.items()):
        panel = data[["depth", column]].dropna().copy()
        panel["水层"] = panel["depth"].map(DEPTH_LABELS)
        sns.boxplot(data=panel, x="水层", y=column, order=["表层", "中层", "底层"], color="#74c69d", showfliers=False, ax=ax)
        if (panel[column] > 0).all():
            ax.set_yscale("log")
        ax.set_title(f"{label} 三水层分布")
        ax.set_xlabel("")
        ax.set_ylabel(f"{label} ({unit})")
    save_figure(fig, fig_dir / "图2-3_四项水质指标三水层分布.png")


def make_time_coverage(data: pd.DataFrame, fig_dir: Path) -> None:
    samples = data[["sample_id", "year", "month"]].drop_duplicates()
    heatmap = samples.groupby(["year", "month"]).size().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.heatmap(heatmap, cmap="YlGnBu", annot=True, fmt="d", cbar_kws={"label": "站点—日期样本数"}, ax=ax)
    ax.set_title("2014-2024 年水质观测样本时间覆盖")
    ax.set_xlabel("月份")
    ax.set_ylabel("年份")
    save_figure(fig, fig_dir / "图2-4_水质观测时间覆盖.png")


def make_sample_key_flow(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        (0.10, "站点元数据\nstation, lon, lat, 海区"),
        (0.34, "水质采样\n站点 × 日期"),
        (0.58, "遥感与过程驱动\n按站点与日期匹配"),
        (0.84, "建模样本\nobs_id = 站点 × 日期 × 水层"),
    ]
    for x, label in nodes:
        draw_box(ax, x, 0.55, label, "#e0f2fe")
    for x in (0.22, 0.46, 0.70):
        ax.annotate("", xy=(x + 0.08, 0.55), xytext=(x, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#264653"})
    ax.text(0.34, 0.20, "sample_id = 站点 × 日期", ha="center", fontsize=11, weight="bold")
    ax.text(0.84, 0.20, "每个样本对应一次采样中的一个水层观测", ha="center", fontsize=10)
    save_figure(fig, fig_dir / "图2-5_多源数据接入与样本主键.png")


def build_tables(data: pd.DataFrame, coverage_path: Path, table_dir: Path) -> None:
    station_table = (
        data[["station", "water_zone_en", "lon", "lat"]]
        .drop_duplicates("station")
        .groupby("water_zone_en")
        .agg(站点数=("station", "nunique"), 最小经度=("lon", "min"), 最大经度=("lon", "max"), 最小纬度=("lat", "min"), 最大纬度=("lat", "max"))
        .reset_index()
        .rename(columns={"water_zone_en": "海区"})
        .sort_values("海区")
    )
    write_csv(station_table, table_dir / "表2-1_海区与站点空间覆盖统计.csv")

    source_table = pd.DataFrame(
        [
            ["水质监测", "Chl-a、Secchi、TN、TP", "目标变量；站点—日期—水层观测", "2014-2024"],
            ["Landsat", "反射率波段与水体指数", "高质量遥感特征及质量标记", "与采样日期匹配"],
            ["Sentinel-2", "反射率波段与水体指数", "高空间分辨率遥感候选特征及质量标记", "与采样日期匹配"],
            ["ERA5", "气温、云量、风、辐射、海表温度", "气象与短时间窗海气过程特征", "2014-2024"],
            ["降雨与潮汐", "累计降雨、潮差等", "陆源输入与水动力过程特征", "2014-2024"],
            ["VIIRS 夜间灯光", "夜间灯光辐亮度", "人类活动强度代理", "2014-2024"],
        ],
        columns=["数据类别", "主要变量", "本文用途", "时间范围或匹配方式"],
    )
    write_csv(source_table, table_dir / "表2-2_多源数据来源与用途.csv")

    rows: list[dict[str, object]] = []
    for column, (label, unit) in TARGETS.items():
        for depth in ["surface", "middle", "bottom"]:
            values = data.loc[data["depth"] == depth, column]
            valid = values.dropna()
            rows.append(
                {
                    "指标": label,
                    "水层": DEPTH_LABELS[depth],
                    "单位": unit,
                    "样本总数": int(values.shape[0]),
                    "有效样本数": int(valid.shape[0]),
                    "缺失率(%)": round(float(values.isna().mean() * 100), 2),
                    "站点数": int(data.loc[(data["depth"] == depth) & values.notna(), "station"].nunique()),
                    "最小值": round(float(valid.min()), 4),
                    "中位数": round(float(valid.median()), 4),
                    "最大值": round(float(valid.max()), 4),
                    "偏度": round(float(valid.skew()), 4),
                }
            )
    write_csv(pd.DataFrame(rows), table_dir / "表2-3_四项目标变量描述性统计.csv")

    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        labels = {
            "landsat_available_share": "Landsat 可用样本比例",
            "landsat_valid_count_median": "Landsat 有效影像数中位数",
            "s2_feature_fields_present": "Sentinel-2 特征字段是否进入正式主表",
            "ssrd_7d_missing_share": "SSRD 7 日窗口缺失率",
            "sst_7d_missing_share": "SST 7 日窗口缺失率",
        }
        coverage["指标"] = coverage["metric"].map(labels).fillna(coverage["metric"])
        coverage["数值"] = coverage["value"].round(4)
        write_csv(coverage[["指标", "数值"]], table_dir / "表2-4_遥感与短时间窗过程特征覆盖统计.csv")


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    data = load_dataset(args.dataset)
    make_technical_route(args.fig_dir)
    make_question_framework(args.fig_dir)
    make_station_zone_map(data, args.fig_dir)
    make_depth_coverage(data, args.fig_dir)
    make_target_distributions(data, args.fig_dir)
    make_time_coverage(data, args.fig_dir)
    make_sample_key_flow(args.fig_dir)
    build_tables(data, args.coverage, args.table_dir)
    print(f"已生成第一、二章图表: {args.fig_dir} / {args.table_dir}")


if __name__ == "__main__":
    main()
