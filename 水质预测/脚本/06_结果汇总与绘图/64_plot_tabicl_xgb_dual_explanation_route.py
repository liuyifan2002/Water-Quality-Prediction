"""生成 TabICLv2重点、XGBoost对照的双分支技术路线图。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_explainability_v1/figures"


def box(ax, x, y, w, h, text, color):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", facecolor="white", edgecolor=color, linewidth=1.5)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color="#222222")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor="white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, .34, .84, .32, .10, "多源环境数据 + 77个strict features\n冻结五折站点留出", "#37474F")
    box(ax, .10, .64, .34, .11, "TabICLv2（重点）\n固定训练上下文与预训练权重", "#1565C0")
    box(ax, .56, .64, .34, .11, "XGBoost（对照）\n数值等价受控复现模型", "#C62828")
    box(ax, .08, .43, .18, .11, "模型无关\n置换重要性", "#1565C0")
    box(ax, .28, .43, .18, .11, "1D/2D ALE\n固定上下文", "#1565C0")
    box(ax, .54, .43, .18, .11, "TreeSHAP\n特征贡献", "#C62828")
    box(ax, .74, .43, .18, .11, "1D/2D ALE\n五折稳定性", "#C62828")
    box(ax, .15, .20, .70, .12, "配对比较：变量排名与来源组 + 共同支持范围内ALE\n结合来源消融、空间分区、未见站点误差与高误差站点", "#2E7D32")
    box(ax, .26, .04, .48, .09, "论文结论：模型统计依赖与空间泛化证据\n不表述为环境因果机制", "#6A1B9A")
    arrows=[((.50,.84),(.27,.75)),((.50,.84),(.73,.75)),((.27,.64),(.17,.54)),((.27,.64),(.37,.54)),((.73,.64),(.63,.54)),((.73,.64),(.83,.54)),((.17,.43),(.38,.32)),((.37,.43),(.45,.32)),((.63,.43),(.55,.32)),((.83,.43),(.62,.32)),((.50,.20),(.50,.13))]
    for start,end in arrows:
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#616161",lw=1.2))
    ax.text(.5,.975,"近岸水质预测与可解释性分析技术路线（修订版）",ha="center",va="top",fontsize=14,fontweight="bold")
    ax.text(.5,.005,"注：TabICLv2当前仅完成Chl-a surface fold_1解释试点；XGBoost四任务五折解释已完成。",ha="center",va="bottom",fontsize=8,color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "图1-1_TabICLv2与XGBoost双分支技术路线.png", dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "图1-1_TabICLv2与XGBoost双分支技术路线.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
