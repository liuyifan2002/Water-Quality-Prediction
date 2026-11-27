"""仅重绘三张冻结联合支持矩阵的文字版式；不计算二维ALE曲面。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm

ROOT = Path(r"E:\研究生\codex\水质预测")
OUT = ROOT / "文档/论文材料/硕士论文V2_学术内容补强_20261101"
SOURCE = OUT / "水质预测硕士学位论文_学术内容补强V2_正式版.docx"
TARGET = OUT / "水质预测硕士学位论文_学术内容补强V2_最终版.docx"
ALE2D = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10/ale_2d"
SPECS = [
    (18, "TN表层", "tn_surface_wind_speed_lag1_t2m_delta1_joint_support.csv", 13),
    (19, "TN底层", "tn_bottom_t2m_roll3_t2m_lag2_joint_support.csv", 7),
    (20, "TP底层", "tp_bottom_wind_speed_lag1_t2m_lag1_joint_support.csv", 13),
]
plt.rcParams.update({"font.family": ["Arial", "SimSun"], "font.size": 9, "pdf.fonttype": 42})


def draw(number, label, filename, expected):
    source = ALE2D / filename
    d = pd.read_csv(source)
    assert len(d) == 16 and int(d.supported.sum()) == expected
    fig, ax = plt.subplots(figsize=(5.3, 5.3), constrained_layout=True)
    for row in d.itertuples():
        x, y = int(row.cell_2)-1, int(row.cell_1)-1
        ok = bool(row.supported)
        color = "#eaf2f5" if ok else "#f5ebe7"
        edge = "#799aad" if ok else "#bd8878"
        ax.add_patch(plt.Rectangle((x-.5,y-.5),1,1,facecolor=color,edgecolor=edge,lw=.8))
        ax.text(x,y,f"n={int(row.n_obs)}\n站点={int(row.n_stations)}\n{'通过' if ok else '不足'}",ha="center",va="center",fontsize=8.5,color="#274b5c" if ok else "#8c5148")
    ax.set_xlim(-.5,3.5);ax.set_ylim(3.5,-.5)
    ax.set_xticks(range(4),range(1,5));ax.set_yticks(range(4),range(1,5))
    ax.set_xlabel(f"{d.feature_2.iloc[0]} 分箱")
    ax.set_ylabel(f"{d.feature_1.iloc[0]} 分箱")
    ax.set_aspect("equal")
    ax.spines[:].set_visible(False)
    stem=OUT/"figures"/f"图3-{number}_{label}_联合支持审查_无程序码"
    fig.savefig(stem.with_suffix(".png"),dpi=600,facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"),facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


assert SOURCE.is_file() and not TARGET.exists()
doc=Document(SOURCE)
for number,label,filename,expected in SPECS:
    path=draw(number,label,filename,expected)
    cap=next(p for p in doc.paragraphs if p.style.name=="图题" and p.text.startswith(f"图3-{number} "))
    ps=doc.paragraphs
    idx=next(i for i,p in enumerate(ps) if p._p is cap._p)
    img=ps[idx-1]
    assert img._p.xpath(".//w:drawing")
    img.clear();img.alignment=WD_ALIGN_PARAGRAPH.CENTER
    img.paragraph_format.line_spacing=1.0
    img.add_run().add_picture(str(path),width=Cm(14.5))
    note=ps[idx+1]
    if note.text.startswith("图注："):
        note.text=f"图注：{expected}/16个联合分箱达到预登记样本及独立站点覆盖条件；仅为支持审查，未计算二维ALE响应曲面。"
doc.save(TARGET)
print("SUPPORT_FIGURES_CLEANED", TARGET)
