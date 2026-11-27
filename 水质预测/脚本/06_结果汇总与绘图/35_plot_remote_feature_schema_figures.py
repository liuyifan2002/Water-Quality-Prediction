"""生成论文第 5 章的遥感字段核对图，不训练或修改任何模型。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
MODEL_TABLE = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
TRAIN_SCRIPT = ROOT / "脚本" / "04_模型训练" / "05_train_full_station_models.py"
OUT_DIR = ROOT / "结果" / "论文图表" / "遥感字段核对_20260907"


def load_train_module():
    spec = importlib.util.spec_from_file_location("full_station_train", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11


def draw_schema_table(columns: set[str], expected: list[str]) -> None:
    raw_landsat = [
        col
        for col in [
            "landsat_blue",
            "landsat_green",
            "landsat_red",
            "landsat_nir",
            "landsat_swir1",
            "landsat_swir2",
            "landsat_ndvi",
            "landsat_ndwi",
            "landsat_mndwi",
            "landsat_ndti",
        ]
        if col in columns
    ]
    available = [col for col in expected if col in columns]
    missing = [col for col in expected if col not in columns]

    rows = [
        ["正式主表已有字段", "Landsat 波段：blue、green、red、nir、swir1、swir2\n指数：NDVI、NDWI、MNDWI、NDTI", f"{len(raw_landsat)} 个 Landsat 波段或指数"],
        ["训练脚本需要的高质量字段", "landsat_hq_* 与 s2_hq_*\n以及相应质量统计", f"{len(missing)} 个字段未写入当前主表"],
        ["当前实际进入模型的字段", "landsat_available\nhq_satellite_available_count" if available else "无", f"{len(available)} 个字段"],
        ["本节解释范围", "覆盖、可用性和特征组比较", "不能给出波段级 SHAP 结论"],
    ]

    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["检查项目", "字段情况", "论文中的含义"],
        colWidths=[0.25, 0.46, 0.29],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#9A9A9A")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#E6E6E6")
            cell.set_text_props(weight="bold", color="black", ha="center")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color="black")
    ax.set_title("当前正式主模型的遥感字段进入情况", fontsize=14, pad=18, color="black")
    fig.savefig(OUT_DIR / "图5-5_遥感字段进入正式模型的情况.png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_interpretation_scope(columns: set[str], expected: list[str]) -> None:
    available = [col for col in expected if col in columns]
    fig, ax = plt.subplots(figsize=(13.5, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.05, 0.30, 0.24, 0.40, "当前正式主表\n保存 Landsat 原始波段、指数\n及可用性标记"),
        (0.38, 0.30, 0.24, 0.40, "当前训练字段\nlandsat_available\nhq_satellite_available_count\n未含高质量波段统计"),
        (0.71, 0.30, 0.24, 0.40, "论文可解释的内容\n遥感覆盖、质量条件\n和特征组比较"),
    ]
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.015", facecolor="#F2F2F2", edgecolor="black", linewidth=1.1
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, color="black", linespacing=1.55)
    for start, end in [(0.29, 0.38), (0.62, 0.71)]:
        ax.annotate("", xy=(end, 0.75), xytext=(start, 0.75), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "black"})
    ax.text(
        0.5,
        0.12,
        "高质量波段统计未完整写入当前正式训练表，因此不采用波段级 SHAP 重要性和依赖关系图。",
        ha="center",
        va="center",
        fontsize=11,
        color="black",
    )
    ax.set_title("当前遥感特征解释的范围", fontsize=14, pad=16, color="black")
    fig.savefig(OUT_DIR / "图5-6_当前遥感特征解释的范围.png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    columns = set(pd.read_csv(MODEL_TABLE, nrows=1).columns)
    train_module = load_train_module()
    expected = list(train_module.REMOTE_FEATURES)
    draw_schema_table(columns, expected)
    draw_interpretation_scope(columns, expected)
    print(f"输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
