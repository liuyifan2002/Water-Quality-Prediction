"""生成论文图件系统的初始清单与前三张正文核心图。

本脚本只读取已经落盘的SHAP/ALE/支持审查文件，不导入TabICLv2模型，不调用推理。
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\研究生\codex\水质预测")
RUN = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
OUT = ROOT / "runs/explanation_figures_v1"
MAIN = OUT / "main_text"
APP = OUT / "appendix"
META = OUT / "metadata"
INVENTORY = ROOT / "runs/explanation_figure_inventory_v1.csv"

TASKS = [
    "chla_surface", "chla_middle", "chla_bottom", "secchi",
    "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom",
]
LABELS = {
    "chla_surface": "Chl-a surface", "chla_middle": "Chl-a middle", "chla_bottom": "Chl-a bottom",
    "secchi": "Secchi", "tn_surface": "TN surface", "tn_middle": "TN middle", "tn_bottom": "TN bottom",
    "tp_surface": "TP surface", "tp_middle": "TP middle", "tp_bottom": "TP bottom",
}
REPRESENTATIVE = ["chla_surface", "secchi", "tn_surface", "tp_middle"]
FONT = "DejaVu Sans"
BLUE = "#1F4E79"
TEAL = "#2F7F8F"
ORANGE = "#C77C29"
RED = "#9E3D3D"
GREEN = "#5A7D5A"
GREY = "#6B7280"


def setup() -> None:
    for path in (MAIN, APP, META):
        path.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, stem: str, folder: Path = MAIN) -> list[str]:
    outputs = []
    for suffix in ("png", "pdf"):
        path = folder / f"{stem}.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def shap_files() -> list[Path]:
    files = sorted((RUN / "shap").glob("*/**/*.csv.gz"))
    if len(files) != 40:
        raise RuntimeError(f"预期40组SHAP，实际{len(files)}组")
    return files


def load_shap(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
            frame = pd.read_csv(stream)
        if len(frame) != 77 or frame.status.astype(str).nunique() != 1 or frame.status.iloc[0] != "COMPUTED_PASS":
            raise RuntimeError(f"SHAP文件未通过只读完整性检查：{path}")
        frame["source_file"] = path.name
        frame["task"] = path.parent.name
        frame["seed"] = int(re.search(r"_seed(\d+)\.csv\.gz$", path.name).group(1))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_ale() -> tuple[list[Path], pd.DataFrame]:
    files = sorted((RUN / "ale_1d").glob("*_curve.csv"))
    if len(files) != 20:
        raise RuntimeError(f"预期20条ALE 1D曲线，实际{len(files)}条")
    rows = []
    for path in files:
        frame = pd.read_csv(path)
        bins = sorted(pd.to_numeric(frame.bin, errors="coerce").dropna().astype(int).unique().tolist())
        if len(frame) < 6 or bins != list(range(1, len(bins) + 1)) or frame.status.astype(str).nunique() != 1:
            raise RuntimeError(f"ALE曲线箱支持不完整：{path}")
        rows.append({"task": str(frame.task.iloc[0]), "feature": str(frame.feature.iloc[0]), "status": str(frame.status.iloc[0]), "bins": len(frame), "path": str(path)})
    return files, pd.DataFrame(rows)


def load_2d() -> pd.DataFrame:
    path = RUN / "ale_2d/candidate_pair_decisions.csv"
    frame = pd.read_csv(path)
    expected = {"tn_surface", "tn_bottom", "tp_bottom"}
    if set(frame.task) != expected:
        raise RuntimeError(f"ALE 2D支持审查任务异常：{set(frame.task)}")
    if frame.status.eq("SUPPORTED_FOR_COMPUTATION").any():
        raise RuntimeError("发现可计算ALE 2D组合；本轮不应生成曲面，但清单状态需人工复核")
    return frame


def make_flowchart(n_shap: int, n_ale: int, n_ale2d: int) -> list[str]:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    nodes = [
        (0.5, 4.15, 2.0, 1.15, "Frozen protocol\n77 strict features\nfold_1 / preregistered", BLUE),
        (3.0, 4.15, 2.0, 1.15, "TabICLv2 pointwise\nmodel predictions\nread-only inputs", TEAL),
        (5.5, 4.15, 2.0, 1.15, f"SHAP explanations\n{n_shap} groups\n30 core + 10 sensitivity", ORANGE),
        (8.0, 4.15, 2.0, 1.15, f"ALE 1D curves\n{n_ale} completed\n10 tasks × 2 features", GREEN),
        (10.5, 4.15, 1.2, 1.15, "Thesis\nfigures", RED),
        (3.0, 1.4, 2.0, 1.15, "Feature support\nscreening", GREY),
        (5.5, 1.4, 2.0, 1.15, f"ALE 2D audit\n{n_ale2d} eligible\nno surface computed", RED),
        (8.0, 1.4, 2.0, 1.15, "Interpretation\nflags and caveats", GREY),
    ]
    for x, y, w, h, text, color in nodes:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=1.2, edgecolor=color, facecolor="white")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#1F2937", linespacing=1.35)
    arrows = [((2.5, 4.72), (3.0, 4.72)), ((5.0, 4.72), (5.5, 4.72)), ((7.5, 4.72), (8.0, 4.72)), ((10.0, 4.72), (10.5, 4.72)), ((4.0, 4.15), (4.0, 2.55)), ((5.0, 1.98), (5.5, 1.98)), ((7.5, 1.98), (8.0, 1.98))]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.2})
    ax.text(0.5, 0.35, "All panels use completed, frozen, read-only explanation outputs; no model training or inference is performed during figure generation.", fontsize=8, color=GREY)
    return save_figure(fig, "main_01_explanation_workflow")


def make_shap_heatmap(shap: pd.DataFrame) -> list[str]:
    table = shap.groupby(["task", "feature"], sort=False).shap_value.apply(lambda x: np.mean(np.abs(x))).unstack(fill_value=0).reindex(TASKS)
    normalized = table.div(table.max(axis=1).replace(0, np.nan), axis=0).fillna(0)
    top_features = normalized.mean(axis=0).sort_values(ascending=False).head(20).index.tolist()
    plot = normalized[top_features]
    fig, ax = plt.subplots(figsize=(13, 6.8))
    im = ax.imshow(plot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(top_features)))
    ax.set_xticklabels(top_features, rotation=65, ha="right")
    ax.set_yticks(np.arange(len(TASKS)))
    ax.set_yticklabels([LABELS[t] for t in TASKS])
    ax.set_xlabel("Feature (top 20 by mean within-task normalized |SHAP|)")
    ax.set_ylabel("Task")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Relative mean |SHAP| within task")
    ax.set_title("")
    fig.subplots_adjust(left=0.18, bottom=0.31, right=0.90, top=0.96)
    return save_figure(fig, "main_02_shap_global_overview_heatmap")


def make_top_feature_panel(shap: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), sharex=False)
    colors = [BLUE, TEAL, ORANGE, RED]
    for ax, task, color in zip(axes.ravel(), REPRESENTATIVE, colors):
        table = shap[shap.task == task].groupby("feature").shap_value.apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=True).tail(10)
        ax.barh(table.index, table.values, color=color, alpha=0.88, height=0.72)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_ylabel("")
        ax.text(0.02, 1.03, LABELS[task], transform=ax.transAxes, fontweight="bold", color="#1F2937")
        ax.grid(axis="x", color="#D1D5DB", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.10, top=0.94, wspace=0.35, hspace=0.33)
    return save_figure(fig, "main_03_representative_shap_top_features_2x2")


def planned_inventory(ale_summary: pd.DataFrame, two_d: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    def add(fid: str, title: str, ftype: str, source: str, scope: str, placement: str, priority: str, target: str, status: str, notes: str) -> None:
        rows.append({"figure_id": fid, "figure_title": title, "figure_type": ftype, "data_source": source, "scope": scope, "main_text_or_appendix": placement, "priority": priority, "target_output_path": target, "status": status, "notes": notes})
    add("MAIN-01", "Explanation workflow and evidence boundaries", "flowchart", "frozen protocol; V2 coverage; ALE 2D support audit", "10 tasks; 77 features; explanation-only stage", "main_text", "P0", str(MAIN / "main_01_explanation_workflow.png"), "GENERATED", "PNG and PDF generated; no model calls.")
    add("MAIN-02", "Global SHAP overview across ten tasks", "heatmap", str(RUN / "shap"), "40 SHAP groups; 10 tasks; 77 features", "main_text", "P0", str(MAIN / "main_02_shap_global_overview_heatmap.png"), "GENERATED", "Row-normalized mean absolute SHAP; top 20 features shown for readability.")
    add("MAIN-03", "Representative-task SHAP top-feature panels", "bar_panel_2x2", str(RUN / "shap"), "Chl-a surface; Secchi; TN surface; TP middle", "main_text", "P0", str(MAIN / "main_03_representative_shap_top_features_2x2.png"), "GENERATED", "Four tasks selected to span target families and water-quality responses.")
    add("MAIN-04", "Representative-sample SHAP waterfall panels", "waterfall_panel", str(RUN / "shap"), "4–6 preregistered samples", "main_text", "P0", str(MAIN / "main_04_representative_shap_waterfall_panel.png"), "PLANNED", "Use existing SHAP values only.")
    add("MAIN-05", "SHAP stability and rank consistency", "stability_plot", str(ROOT / "runs/tabicl_v2_shap_merge_20261029/summary/overlap_comparison.csv"), "Cross-seed and cross-GPU comparison", "main_text", "P0", str(MAIN / "main_05_shap_stability_rank_consistency.png"), "PLANNED", "Retain numeric-sensitivity caveats.")
    add("MAIN-06", "Chl-a family ALE 1D panels", "ale_panel", str(RUN / "ale_1d"), "Chl-a surface/middle/bottom; 6 curves", "main_text", "P0", str(MAIN / "main_06_chla_family_ale_1d.png"), "PLANNED", "Plot completed curves only; mark limitation status.")
    add("MAIN-07", "TN family ALE 1D panels", "ale_panel", str(RUN / "ale_1d"), "TN surface/middle/bottom; 6 curves", "main_text", "P0", str(MAIN / "main_07_tn_family_ale_1d.png"), "PLANNED", "Plot completed curves only.")
    add("MAIN-08", "TP family ALE 1D panels", "ale_panel", str(RUN / "ale_1d"), "TP surface/middle/bottom; 6 curves", "main_text", "P0", str(MAIN / "main_08_tp_family_ale_1d.png"), "PLANNED", "Plot completed curves only.")
    add("MAIN-09", "Secchi ALE 1D panel", "ale_panel", str(RUN / "ale_1d"), "Secchi; 2 curves", "main_text", "P1", str(MAIN / "main_09_secchi_ale_1d.png"), "PLANNED", "Plot completed curves only.")
    add("MAIN-10", "ALE support and bin coverage", "support_plot", str(RUN / "ale_1d"), "20 curves; bin-level n and station support", "main_text", "P0", str(MAIN / "main_10_ale_support.png"), "PLANNED", "Show support thresholds explicitly.")
    add("MAIN-11", "ALE 2D support blocking", "support_blocking", str(RUN / "ale_2d/candidate_pair_decisions.csv"), "TN surface; TN bottom; TP bottom; no surface computed", "main_text", "P0", str(MAIN / "main_11_ale_2d_support_blocking.png"), "PLANNED", "State ALE_2D_BLOCKED_BY_SUPPORT; no 2D surface is plotted.")
    for seed, label, count in [(42, "core", 30), (43, "sensitivity", 10)]:
        for path in sorted((RUN / "shap").glob("*/**/*_seed%d.csv.gz" % seed)):
            match = re.search(r"([^/\\]+)_seed%d\.csv\.gz$" % seed, path.name)
            obs = match.group(1) if match else path.stem
            fid = f"APP-SHAP-{label.upper()}-{obs}"
            add(fid, f"{label.title()} SHAP waterfall: {obs}", "waterfall", str(path), path.parent.name, "appendix", "P1", str(APP / "shap_waterfalls" / f"{path.stem}.png"), "PLANNED", "One figure per completed SHAP group.")
    for path, row in zip(sorted((RUN / "ale_1d").glob("*_curve.csv")), ale_summary.to_dict("records")):
        add(f"APP-ALE-{row['task']}-{row['feature']}", f"ALE 1D curve: {row['task']} / {row['feature']}", "ale_curve", str(path), row["task"], "appendix", "P1", str(APP / "ale_1d_curves" / path.name.replace(".csv", ".png")), "PLANNED", row["status"])
    add("APP-01", "Numerical stability summary", "stability_summary", str(RUN / "ale_1d"), "20 ALE curves and SHAP seed comparisons", "appendix", "P1", str(APP / "numeric_stability_summary.png"), "PLANNED", "Mark SMALL_EFFECT_INTERPRET_WITH_CAUTION and NUMERICALLY_SENSITIVE_SMALL_EFFECT.")
    add("APP-02", "ALE 2D support blocking detail", "support_blocking_detail", str(RUN / "ale_2d"), "All audited 4×4 candidate pairs", "appendix", "P1", str(APP / "ale_2d_support_blocking_detail.png"), "PLANNED", "Include TN surface 13/16, TN bottom 7/16, TP bottom 13/16.")
    return rows


def main() -> None:
    setup()
    files = shap_files()
    shap = load_shap(files)
    ale_files, ale_summary = load_ale()
    two_d = load_2d()
    generated = {}
    generated["MAIN-01"] = make_flowchart(len(files), len(ale_files), len(two_d[two_d.selected.astype(bool)]))
    generated["MAIN-02"] = make_shap_heatmap(shap)
    generated["MAIN-03"] = make_top_feature_panel(shap)
    rows = planned_inventory(ale_summary, two_d)
    for row in rows:
        if row["figure_id"] in generated:
            row["status"] = "GENERATED"
            row["target_output_path"] = ";".join(generated[row["figure_id"]])
    with INVENTORY.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["figure_id", "figure_title", "figure_type", "data_source", "scope", "main_text_or_appendix", "priority", "target_output_path", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run": str(RUN), "shap_groups": len(files), "ale_1d_curves": len(ale_files),
        "ale_2d_audited_pairs": len(two_d), "ale_2d_computed_surfaces": 0,
        "generated_figures": sorted(generated), "model_calls": 0,
        "notes": "第一阶段只生成前三张正文核心图；其余图件仅进入总清单。",
    }
    (META / "figure_generation_state.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (META / "source_snapshot.csv").write_text("source,expected_count,observed_count,status\nSHAP,40,%d,PASS\nALE_1D,20,%d,PASS\nALE_2D_SURFACE,0,0,PASS\n" % (len(files), len(ale_files)), encoding="utf-8-sig")
    captions = ["# Figure caption draft (initial stage)", "", "## Main text", "", "- **MAIN-01** Explanation workflow and evidence boundaries. The figure summarizes the frozen 77-feature protocol, read-only TabICLv2 explanation outputs, completed SHAP/ALE 1D coverage, and the ALE 2D support audit. No model training or inference was performed during figure generation.", "- **MAIN-02** Global SHAP overview across ten water-quality tasks. Cell intensity denotes the task-normalized mean absolute SHAP value across the four completed explanation groups per task; the displayed features are the top 20 by average normalized contribution.", "- **MAIN-03** Representative-task SHAP top-feature panels. Bars show mean absolute SHAP values across the completed SHAP groups for Chl-a surface, Secchi, TN surface, and TP middle tasks.", "", "The remaining planned captions will be completed when their corresponding figures are generated."]
    (OUT / "figure_caption_draft.md").write_text("\n".join(captions) + "\n", encoding="utf-8")
    report = ["# Explanation figure generation report (initial stage)", "", f"- SHAP source groups verified: **{len(files)}**.", f"- ALE 1D curves verified: **{len(ale_files)}**.", f"- ALE 2D surfaces generated: **0**; only support-audit files were read.", "- Model training/inference during plotting: **0 calls**.", "- Generated now: MAIN-01 workflow, MAIN-02 SHAP overview heatmap, MAIN-03 representative-task top-feature 2×2 panel, each as PNG and PDF at 600 dpi for PNG.", "- All other requested figures are listed as `PLANNED` in the inventory and were not generated in this stage.", "- SHAP/ALE inputs, split files, prediction files, and model outputs were not modified.", "", "## Numeric interpretation flags", "", "The later figure stages must preserve `SMALL_EFFECT_INTERPRET_WITH_CAUTION` and `NUMERICALLY_SENSITIVE_SMALL_EFFECT` flags in metadata and captions. No completed result is removed because of a small effect or numerical sensitivity."]
    (OUT / "figure_generation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "INITIAL_FIGURE_STAGE_COMPLETE", "inventory": str(INVENTORY), "generated": generated, "model_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
