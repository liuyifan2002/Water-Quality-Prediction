"""从既有冻结绘图数据仅重导图3-7文字与四模型图4-5。"""
from pathlib import Path
import csv
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "修订图件"
OUT.mkdir(parents=True, exist_ok=True)

MATRIX = ROOT / "结果/论文图表/figure_revision_v2/data/图3-5_30核心样本排名矩阵.csv"
MODERN = ROOT / "transfer/llm_extension_transfer_v1/results/modern_baseline_gate_v1/modern_baseline_core_metrics.csv"

plt.rcParams.update({"font.family": ["Arial", "Microsoft YaHei"], "font.size": 8.5,
                     "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8.5,
                     "axes.linewidth": .65, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.unicode_minus": False,
                     "pdf.fonttype": 42, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, stem):
    png = OUT / (stem + ".png")
    pdf = OUT / (stem + ".pdf")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def draw_rank_matrix():
    rows = read_csv(MATRIX)
    tasks = ["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface",
             "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom"]
    labels = ["Chl-a 表层", "Chl-a 中层", "Chl-a 底层", "Secchi", "TN 表层",
              "TN 中层", "TN 底层", "TP 表层", "TP 中层", "TP 底层"]
    assert len(rows) == 10 and all(set(row) == {"feature", *tasks} for row in rows)
    data = np.array([[float(row[t]) for t in tasks] for row in rows])
    assert data.shape == (10, 10) and np.all((1 <= data) & (data <= 77))
    names = {"lon": "经度", "lat": "纬度", "year_offset": "年份偏移",
             "sst_mean_3d": "3日平均海表温度", "day_of_year": "年内日序",
             "ssrd_sum_3d": "3日累计太阳辐射", "month_sin": "月份正弦编码",
             "ntl_avg_rad_lag2": "夜间灯光2期滞后", "ssrd_mean_14d": "太阳辐射均值（14天）",
             "rain_month": "月降雨"}
    assert all(row["feature"] in names for row in rows)
    fig, ax = plt.subplots(figsize=(15.5 / 2.54, 12.8 / 2.54))
    fig.subplots_adjust(left=.31, right=.90, bottom=.22, top=.88)
    cmap = LinearSegmentedColormap.from_list("rank", ["#4477AA", "#d7e5ef", "#f5f7f9"])
    cmap.set_bad("white")
    im = ax.pcolormesh(np.arange(11)-.5, np.arange(11)-.5, data,
                       cmap=cmap, vmin=1, vmax=77, shading="flat")
    ax.set_ylim(9.5, -.5)
    ax.set_xlim(-.5, 9.5)
    for i in range(10):
        for j in range(10):
            value = data[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if value < 23 else "#253746")
    ax.set_yticks(range(10), [names[row["feature"]] for row in rows])
    ax.set_xticks(range(10), [label.replace(" ", "\n") for label in labels], fontsize=8)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    ax.grid(which="minor", color="white", lw=.7)
    ax.tick_params(which="minor", length=0)
    for boundary in (2.5, 3.5, 6.5):
        ax.axvline(boundary, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    cb.solids.set_rasterized(False)
    cb.set_ticks([1, 20, 40, 60, 77])
    cb.set_label("任务内排名（1最重要）")
    fig.text(.31, .945, "十项任务的特征排名", fontsize=10)
    fig.text(.03, .035,
             "每任务3个预登记独立核心样本；与表3-1范围一致。\n仅比较任务内相对排名，不代表跨任务贡献大小。",
             fontsize=8, linespacing=1.6)
    return save(fig, "图3-7_仅修正内部表号")


def draw_modern():
    rows = read_csv(MODERN)
    tasks = ["chla_surface", "secchi", "tn_surface", "tp_middle"]
    names = ["Chl-a表层", "Secchi", "TN表层", "TP中层"]
    models = ["xgboost", "tabiclv2", "tabpfn", "tabm"]
    model_names = ["XGBoost", "TabICLv2", "TabPFN", "TabM"]
    colors = ["#4477AA", "#EE9944", "#AA3377", "#777777"]
    values = {(row["task"], row["model"]): float(row["pooled_R2"]) for row in rows}
    assert len(values) == 16 and all((task, model) in values for task in tasks for model in models)
    fig, axes = plt.subplots(2, 2, figsize=(15.5/2.54, 10.5/2.54))
    fig.subplots_adjust(left=.20, right=.97, bottom=.12, top=.94, wspace=.48, hspace=.43)
    for ax, task, name in zip(axes.flat, tasks, names):
        for index, (model, color) in enumerate(zip(models, colors)):
            value = values[task, model]
            ax.scatter(value, index, s=34, c=color)
            ax.annotate(f"{value:.3f}", (value, index), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(range(4), model_names)
        ax.set_ylim(3.6, -.6)
        ax.set_xlim(.2, .72)
        ax.set_xlabel("pooled R²")
        ax.set_title(name, loc="left")
        ax.grid(axis="x", color="#e2e6e8", linewidth=.5)
    png, pdf = save(fig, "图4-5_四项代表任务四模型比较")
    with (OUT / "图4-5_冻结绘图值.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["任务", "模型", "pooled R²", "来源文件", "来源SHA256"])
        for task in tasks:
            for model in models:
                writer.writerow([task, model, f"{values[task, model]:.17g}", str(MODERN), digest(MODERN)])
    return png, pdf


if __name__ == "__main__":
    print("图3-7", draw_rank_matrix())
    print("图4-5", draw_modern())
