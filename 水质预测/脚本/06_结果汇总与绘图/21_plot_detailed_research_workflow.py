"""Render a publication-style workflow diagram for the formal full-station study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "\u7ed3\u679c" / "\u8bba\u6587\u56fe\u8868" / "\u8be6\u7ec6\u6280\u672f\u8def\u7ebf_20260906"

INK = "#202020"
MUTED = "#6A6A6A"
GRID = "#B8B8B8"
TEAL = "#2E6F62"
OCHRE = "#B5832E"
BRICK = "#A65445"
PALE_TEAL = "#E9F1EF"
PALE_OCHRE = "#F5EEDF"
PALE_BRICK = "#F5E7E4"
PALE_GREY = "#F5F5F5"


def configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["text.color"] = INK


def section_header(ax, y: float, panel: str, title: str) -> None:
    ax.add_patch(Rectangle((0.045, y - 0.022), 0.042, 0.036, facecolor=INK, edgecolor=INK))
    ax.text(0.066, y - 0.004, panel, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.text(0.105, y - 0.004, title, ha="left", va="center", fontsize=9.3, fontweight="bold")
    ax.plot([0.105, 0.955], [y - 0.022, y - 0.022], color=GRID, lw=0.7)


def node(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    accent: str = TEAL,
    fill: str = "white",
    dashed: bool = False,
    title_size: float = 7.5,
    body_size: float = 6.35,
) -> None:
    """Draw a compact flat scientific-figure node with an accent rule."""
    ax.add_patch(
        Rectangle(
            (x, y), width, height, facecolor=fill, edgecolor=GRID,
            linewidth=0.8, linestyle="--" if dashed else "-",
        )
    )
    header_height = 0.025
    ax.add_patch(Rectangle((x, y + height - header_height), width, header_height, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.011, y + height - header_height / 2, title, ha="left", va="center", fontsize=title_size, fontweight="bold", color="white")
    ax.text(x + 0.012, y + 0.014, body, ha="left", va="bottom", fontsize=body_size, color=INK, linespacing=1.14)


def arrow(ax, start: tuple[float, float], end: tuple[float, float], *, dashed: bool = False) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 0.9,
            "color": MUTED,
            "linestyle": "--" if dashed else "-",
            "shrinkA": 1,
            "shrinkB": 2,
        },
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_fonts()

    # A4 portrait-compatible figure. The Word caption supplies the full figure title.
    fig, ax = plt.subplots(figsize=(6.4, 8.3), dpi=360)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    section_header(ax, 0.955, "A", "多源数据获取")
    node(ax, 0.055, 0.812, 0.275, 0.095, "水质与站点观测", "94 个站点，2014-2024\n26 项水质记录与质控标记\n目标：Chl-a、Secchi、TN、TP", accent=TEAL, fill=PALE_TEAL)
    node(ax, 0.362, 0.812, 0.275, 0.095, "陆海过程数据", "ERA5 气象与 SSRD/SST\nHKO 降雨、潮汐\nVIIRS 夜间灯光", accent=OCHRE, fill=PALE_OCHRE)
    node(ax, 0.669, 0.812, 0.275, 0.095, "卫星遥感数据", "GEE：Landsat 与 Sentinel-2\n波段、指数与影像质量\n正式主表使用 Landsat", accent=BRICK, fill=PALE_BRICK)

    section_header(ax, 0.745, "B", "观测级样本构建与质量控制")
    node(ax, 0.055, 0.602, 0.275, 0.094, "标准化与垂向拆分", "单位、检出限和删失信息\n按深度划分表层、中层、底层", fill="white")
    node(ax, 0.362, 0.602, 0.275, 0.094, "显式键值匹配", "sample_id：站点-日期\nobs_id：站点-日期-水层\n30,719 条观测", fill="white")
    node(ax, 0.669, 0.602, 0.275, 0.094, "质量状态保留", "valid_count / missing / imputed\n日尺度窗口与月尺度匹配\n不使用同日水质协变量", fill="white")
    arrow(ax, (0.193, 0.812), (0.193, 0.696))
    arrow(ax, (0.500, 0.812), (0.500, 0.696))
    arrow(ax, (0.807, 0.812), (0.807, 0.696))

    section_header(ax, 0.535, "C", "特征工程与信息源控制对照")
    node(ax, 0.055, 0.382, 0.275, 0.098, "F0 时空基线", "位置、水域、深度、季节\n采样间隔与时间循环编码", fill="white")
    node(ax, 0.362, 0.382, 0.275, 0.098, "F1 历史与过程", "历史 lag / roll / delta\nSSRD/SST 窗口、降雨、潮汐\n风场、夜光及物理交互", fill="white")
    node(ax, 0.669, 0.382, 0.275, 0.098, "F2 遥感与 F3 融合", "Landsat 波段与水体指数\n遥感可用性和质量特征\nF3 = F1 + 遥感", accent=BRICK, fill="white")
    ax.text(0.50, 0.352, "预设特征清单；相同划分和参数下实施控制 A/B", ha="center", va="center", fontsize=6.65, color=MUTED)
    arrow(ax, (0.500, 0.602), (0.500, 0.480))

    section_header(ax, 0.300, "D", "统一建模、评价与论文输出")
    node(
        ax,
        0.055,
        0.158,
        0.275,
        0.094,
        "多水层任务与透明度",
        "Chl-a、TN、TP × 3 水层\nSecchi：站点—日期尺度\n透明度三层记录不作独立证据",
        fill="white",
        body_size=7.0,
    )
    node(ax, 0.362, 0.158, 0.275, 0.094, "模型与评价", "XGBoost；非负目标 log1p\n固定随机 80/20 留出\nR²、RMSE、MAE、Bias", fill="white")
    node(ax, 0.669, 0.158, 0.275, 0.094, "结果解释与归档", "预测值、热图、散点图\n融合增益与 SHAP\n正式结果由 full_station 生成", fill="white")
    arrow(ax, (0.500, 0.382), (0.500, 0.252))

    ax.add_patch(Rectangle((0.055, 0.045), 0.890, 0.060, facecolor=PALE_GREY, edgecolor=GRID, linewidth=0.8, linestyle="--"))
    ax.text(0.070, 0.083, "补充验证（不覆盖正式主结果）", ha="left", va="center", fontsize=7.0, fontweight="bold", color=MUTED)
    ax.text(0.070, 0.060, "遥感质量与 Chl-a 高值诊断  |  按年份、留站点和五折空间验证  |  patch、OceanColor、Copernicus、GPM 来源筛选", ha="left", va="center", fontsize=5.9, color=MUTED)
    arrow(ax, (0.500, 0.158), (0.500, 0.105), dashed=True)

    fig.savefig(OUTPUT_DIR / "\u56fe1-1_\u8be6\u7ec6\u7814\u7a76\u6280\u672f\u8def\u7ebf.png", bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
