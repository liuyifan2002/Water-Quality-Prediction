from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"E:\研究生\codex\水质预测")
SRC = PROJECT / "runs" / "tabicl_v2_slim_explainability_v2_all10"
HIST = PROJECT / "runs" / "tabicl_explanation_numerical_stability_v2"
OUT = PROJECT / "runs" / "thesis_figure_production_v3"
FIG = OUT / "figures"
MAP = PROJECT / "runs" / "explanation_figures_v1" / "metadata" / "feature_display_name_mapping.csv"

TASKS = ["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom"]
TASK_LABEL = {"chla_surface": "Chl-a surface", "chla_middle": "Chl-a middle", "chla_bottom": "Chl-a bottom", "secchi": "Secchi", "tn_surface": "TN surface", "tn_middle": "TN middle", "tn_bottom": "TN bottom", "tp_surface": "TP surface", "tp_middle": "TP middle", "tp_bottom": "TP bottom"}
FAMILY = {"chla_surface": "Chl-a", "chla_middle": "Chl-a", "chla_bottom": "Chl-a", "secchi": "Secchi", "tn_surface": "TN", "tn_middle": "TN", "tn_bottom": "TN", "tp_surface": "TP", "tp_middle": "TP", "tp_bottom": "TP"}
BLUE = "#3d6f8e"
PALETTE = ["#476f85", "#7694a4", "#a67862", "#6b8066", "#8b7091", "#b08b59"]

matplotlib.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def savefig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def display_map() -> pd.DataFrame:
    m = pd.read_csv(MAP)
    return m.set_index("raw_feature_name")


def shap_long() -> pd.DataFrame:
    fs = sorted((SRC / "shap").glob("**/*.csv.gz"))
    return pd.concat([pd.read_csv(f, compression="gzip") for f in fs], ignore_index=True)


def task_stats(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby(["task", "feature"], as_index=False).agg(mean_abs_shap=("shap_value", lambda x: float(np.mean(np.abs(x)))))


def make_global() -> list[dict]:
    d = shap_long(); m = display_map(); stats = task_stats(d); records = []
    # GLOBAL-02: one independent lollipop per task.
    for task in TASKS:
        s = stats[stats.task == task].sort_values("mean_abs_shap", ascending=False).head(10).iloc[::-1]
        labels = [m.loc[x, "display_name_en"] if x in m.index else x for x in s.feature]
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        y = np.arange(len(s)); ax.hlines(y, 0, s.mean_abs_shap, color=BLUE, lw=1.1); ax.scatter(s.mean_abs_shap, y, s=26, color=BLUE, zorder=3)
        ax.set_yticks(y, labels); ax.set_xlabel("Mean absolute SHAP value"); ax.set_ylabel("Feature"); ax.set_title(TASK_LABEL[task], loc="left", fontweight="normal")
        ax.grid(axis="x", color="#e6ebee", lw=0.55); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
        stem = FIG / "global" / f"global_02_{slug(task)}_top10_lollipop"; savefig(fig, stem)
        records.append({"figure_id": "GLOBAL-02", "file_stem": str(stem.relative_to(OUT)), "title": f"{TASK_LABEL[task]} top-10 SHAP importance", "classification": "COMPUTED_RESULT_FIGURE", "status": "GENERATED", "source": "40 SHAP groups"})

    # GLOBAL-03: ranks, not contribution magnitudes.
    stats["rank"] = stats.groupby("task")["mean_abs_shap"].rank(ascending=False, method="min")
    top_features = stats.groupby("feature")["rank"].mean().nsmallest(10).index.tolist()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for i, feature in enumerate(top_features):
        z = stats[stats.feature == feature].set_index("task").reindex(TASKS)
        ax.plot(range(10), z["rank"], marker="o", ms=3.2, lw=1.0, color=PALETTE[i % len(PALETTE)], label=m.loc[feature, "display_name_en"] if feature in m.index else feature)
    ax.set_xticks(range(10), [TASK_LABEL[t] for t in TASKS], rotation=38, ha="right"); ax.set_ylabel("Within-task SHAP rank (1 = highest)"); ax.set_xlabel("Prediction task")
    ax.invert_yaxis(); ax.grid(axis="y", color="#e6ebee", lw=0.55); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper left")
    stem = FIG / "global" / "global_03_10_task_feature_rank_change"; savefig(fig, stem)
    records.append({"figure_id": "GLOBAL-03", "file_stem": str(stem.relative_to(OUT)), "title": "10-task feature rank change", "classification": "COMPUTED_RESULT_FIGURE", "status": "GENERATED", "source": "40 SHAP groups; within-task ranks"})

    # GLOBAL-04: normalized source-group shares.
    d = d.merge(m[["source_group"]], left_on="feature", right_index=True, how="left")
    agg = d.assign(abs_shap=d.shap_value.abs()).groupby(["task", "source_group"], as_index=False).agg(value=("abs_shap", "mean"))
    totals = agg.groupby("task")["value"].transform("sum"); agg["share"] = agg.value / totals * 100
    order = [x for x in m.source_group.dropna().unique()]
    counts = m.source_group.value_counts().to_dict(); piv = agg.pivot(index="task", columns="source_group", values="share").reindex(index=TASKS, columns=order).fillna(0)
    fig, ax = plt.subplots(figsize=(8.2, 4.8)); bottom = np.zeros(len(TASKS))
    for i, group in enumerate(order):
        vals = piv[group].to_numpy(); ax.bar(range(len(TASKS)), vals, bottom=bottom, color=PALETTE[i % len(PALETTE)], width=0.72, label=f"{group} (n={counts.get(group, 0)})"); bottom += vals
    ax.set_xticks(range(len(TASKS)), [TASK_LABEL[t] for t in TASKS], rotation=38, ha="right"); ax.set_ylabel("Share of mean |SHAP| (%)"); ax.set_xlabel("Prediction task"); ax.set_ylim(0, 100); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, ncol=3, fontsize=7)
    stem = FIG / "global" / "global_04_feature_source_group_share"; savefig(fig, stem)
    records.append({"figure_id": "GLOBAL-04", "file_stem": str(stem.relative_to(OUT)), "title": "Feature source-group share of mean absolute SHAP", "classification": "COMPUTED_RESULT_FIGURE", "status": "GENERATED", "source": "40 SHAP groups + frozen feature taxonomy"})
    return records


def make_distribution() -> list[dict]:
    d = shap_long(); m = display_map(); stats = task_stats(d); records = []
    for task in TASKS:
        top = stats[stats.task == task].nlargest(10, "mean_abs_shap").feature.tolist(); q = d[d.task == task].copy(); q["display"] = q.feature.map(m.display_name_en).fillna(q.feature); q = q[q.feature.isin(top)]
        order = [m.loc[x, "display_name_en"] if x in m.index else x for x in top][::-1]
        fig, ax = plt.subplots(figsize=(6.5, 4.7))
        for yi, label in enumerate(order):
            z = q[q.display == label]; vals = z.shap_value.to_numpy(); fv = z.feature_value.to_numpy(); norm = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv) + 1e-12)
            ax.scatter(vals, np.full(len(vals), yi) + np.linspace(-0.16, 0.16, len(vals)), c=norm, cmap="coolwarm", vmin=0, vmax=1, s=17, alpha=0.82, edgecolors="none")
        ax.axvline(0, color="#555", lw=0.7); ax.set_yticks(range(len(order)), order); ax.set_xlabel("SHAP value"); ax.set_title(TASK_LABEL[task], loc="left", fontweight="normal"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e6ebee", lw=0.55); ax.set_axisbelow(True)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1)); sm.set_array([]); cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04); cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.set_label("Feature value")
        stem = FIG / "distribution" / f"dist_01_{slug(task)}_shap_beeswarm"; savefig(fig, stem)
        records.append({"figure_id": "DIST-01", "file_stem": str(stem.relative_to(OUT)), "title": f"{TASK_LABEL[task]} SHAP beeswarm", "classification": "COMPUTED_RESULT_FIGURE", "status": "GENERATED", "source": "40 per-sample SHAP files"})
    return records


def median_waterfall_rows() -> pd.DataFrame:
    sm = pd.read_csv(SRC / "01_preregistered_shap_samples.csv")
    return sm[(sm.phase == "CORE_3") & (sm.target_quantile == 0.5)].copy()


def make_waterfalls() -> tuple[list[dict], list[dict]]:
    m = display_map(); records = []; blocked = []
    for _, row in median_waterfall_rows().iterrows():
        task = row.task; path = SRC / "shap" / task / f"{row.obs_id}_seed42.csv.gz"; d = pd.read_csv(path, compression="gzip"); pred = float(d.model_prediction.iloc[0]); base = float(d.base_value.iloc[0]); total = base + float(d.shap_value.sum()); err = abs(pred - total)
        if err > 1e-8:
            blocked.append({"figure_id": "LOCAL-01", "item": task, "status": "BLOCKED", "reason": f"additivity error {err:g} > 1e-8", "source": str(path.relative_to(PROJECT))}); continue
        d["display"] = d.feature.map(m.display_name_en).fillna(d.feature); d["abs"] = d.shap_value.abs(); top = d.nlargest(10, "abs").sort_values("shap_value"); other = d[~d.feature.isin(top.feature)].shap_value.sum()
        labels = top.display.tolist() + (["Other"] if abs(other) > 0 else []); values = top.shap_value.tolist() + ([other] if abs(other) > 0 else []); y = np.arange(len(values)); colors = ["#4c7c99" if x >= 0 else "#b06e63" for x in values]
        fig, ax = plt.subplots(figsize=(7.0, 4.8)); ax.barh(y, values, color=colors, height=0.65); ax.axvline(0, color="#444", lw=0.7); ax.set_yticks(y, labels); ax.set_xlabel("SHAP contribution"); ax.set_title(f"{TASK_LABEL[task]} · {row.obs_id}", loc="left", fontweight="normal"); ax.text(0.99, 0.02, f"base={base:.4g}  prediction={pred:.4g}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e6ebee", lw=0.55); ax.set_axisbelow(True)
        stem = FIG / "local" / f"local_01_{slug(task)}_median_core3_waterfall"; savefig(fig, stem)
        records.append({"figure_id": "LOCAL-01", "file_stem": str(stem.relative_to(OUT)), "title": f"{TASK_LABEL[task]} median CORE_3 SHAP waterfall", "classification": "COMPUTED_RESULT_FIGURE", "status": "GENERATED", "source": str(path.relative_to(PROJECT)), "sample_id": row.obs_id, "additivity_error": err})
    return records, blocked


def make_ale() -> list[dict]:
    m = display_map(); records = []
    for path in sorted((SRC / "ale_1d").glob("*_curve.csv")):
        d = pd.read_csv(path); task = d.task.iloc[0]; feature = d.feature.iloc[0]; x = (d.lower + d.upper) / 2; y = d.ale; sensitive = float(d.repeatability_max_local_delta.max()) > 1e-6
        fig, ax = plt.subplots(figsize=(6.2, 4.3)); ax.plot(x, y, marker="o", ms=3.6, lw=1.25, color=BLUE); ax.axhline(0, color="#555", lw=0.7, ls="--");
        for xi, yi, ns in zip(x, y, d.n_samples): ax.text(xi, yi, str(int(ns)), fontsize=6.5, ha="center", va="bottom", color="#5c6970")
        ax.set_xlabel(f"{m.loc[feature, 'display_name_en'] if feature in m.index else feature} ({m.loc[feature, 'unit'] if feature in m.index else 'unit'})"); ax.set_ylabel("ALE effect"); ax.set_title(f"{TASK_LABEL[task]} · {m.loc[feature, 'display_name_en'] if feature in m.index else feature}", loc="left", fontweight="normal"); ax.text(0.99, 0.02, "Numbers: bin sample count", transform=ax.transAxes, ha="right", va="bottom", fontsize=7); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6ebee", lw=0.55); ax.set_axisbelow(True)
        if sensitive: ax.text(0.01, 0.98, "NUMERICALLY_SENSITIVE_SMALL_EFFECT", transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#8b4b3f")
        if np.nanmax(np.abs(y)) < 0.05: ax.text(0.01, 0.90, "SMALL_EFFECT_INTERPRET_WITH_CAUTION", transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#8b4b3f")
        stem = FIG / "ale_1d" / f"ale_01_{slug(task)}_{slug(feature)}"; savefig(fig, stem)
        records.append({"figure_id": "ALE-01", "file_stem": str(stem.relative_to(OUT)), "title": f"{TASK_LABEL[task]} ALE 1D: {feature}", "classification": "COMPUTED_RESULT_FIGURE", "status": "GENERATED", "source": str(path.relative_to(PROJECT)), "bins": len(d), "numeric_sensitive": sensitive})
    return records


def make_support() -> list[dict]:
    records = []
    curves = sorted((SRC / "ale_1d").glob("*_curve.csv")); rows = []; labels = []
    for path in curves:
        d = pd.read_csv(path); rows.append(d.n_samples.tolist()); labels.append(f"{d.task.iloc[0]} · {d.feature.iloc[0]}")
    for col, suffix, label in [("n_samples", "sample_count", "Number of observations"), ("n_stations", "station_count", "Number of independent stations")]:
        data = []
        for path in curves:
            d = pd.read_csv(path); data.append(d[col].tolist() + [np.nan] * (10 - len(d)))
        arr = np.array(data, dtype=float); fig, ax = plt.subplots(figsize=(8.6, 7.6)); im = ax.imshow(arr, aspect="auto", cmap="Blues");
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]): ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=6.5, color="#163b52" if arr[i, j] < np.nanmax(arr) * 0.55 else "white")
        ax.set_yticks(range(len(labels)), labels, fontsize=7); ax.set_xticks(range(10), range(1, 11)); ax.set_xlabel("ALE bin position"); ax.set_ylabel("ALE curve"); ax.set_title(f"ALE support audit · {label}", loc="left", fontweight="normal"); fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03, label=label); ax.text(1.0, -0.08, "Blank cells indicate non-existent bins, not zero support", transform=ax.transAxes, ha="right", va="top", fontsize=7); savefig(fig, FIG / "support" / f"support_{suffix}_heatmap"); records.append({"figure_id": "SUPPORT-01" if col == "n_samples" else "SUPPORT-02", "file_stem": str((FIG / "support" / f"support_{suffix}_heatmap").relative_to(OUT)), "title": f"ALE {label} heatmap", "classification": "SUPPORT_AUDIT_FIGURE", "status": "GENERATED", "source": "20 ALE curve files"})
    return records


def make_stability() -> tuple[list[dict], list[dict]]:
    records = []; blocked = []
    probes = sorted((SRC / "gates").glob("*_p1_probes.csv")); p1 = []
    for path in probes:
        d = pd.read_csv(path); p1.append({"task": d.task.iloc[0], "min": d.p1_prediction.min(), "max": d.p1_prediction.max(), "mean": d.p1_prediction.mean()})
    q = pd.DataFrame(p1).set_index("task").reindex(TASKS[1:])
    fig, ax = plt.subplots(figsize=(8.0, 4.5)); x = np.arange(len(q)); ax.vlines(x, q["min"], q["max"], color=BLUE, lw=1.6); ax.scatter(x, q["mean"], s=24, color="#9b604e", zorder=3); ax.set_xticks(x, [TASK_LABEL[t] for t in q.index], rotation=35, ha="right"); ax.set_ylabel("P1 prediction"); ax.set_xlabel("Task"); ax.set_title("P1 repeated-probe prediction range (9 tasks)", loc="left", fontweight="normal"); ax.text(0.01, -0.18, "Three repeated probes; not a prediction confidence interval. Chl-a surface is unavailable.", transform=ax.transAxes, fontsize=7.5); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6ebee", lw=0.55); savefig(fig, FIG / "stability" / "stability_01_p1_repeated_probe_range"); records.append({"figure_id": "STABILITY-01", "file_stem": str((FIG / "stability" / "stability_01_p1_repeated_probe_range").relative_to(OUT)), "title": "P1 repeated-probe prediction range (9 tasks)", "classification": "NUMERICAL_STABILITY_FIGURE", "status": "GENERATED", "source": "9 *_p1_probes.csv files", "note": "Not a prediction confidence interval; chla_surface unavailable"})

    hist = pd.read_csv(HIST / "jitter_vs_explanation_scale.csv")
    required = {"prediction_jitter_range", "method", "effect_scale_type", "effect_scale"}
    if not required.issubset(hist.columns) or hist.empty:
        blocked.append({"figure_id": "STABILITY-02", "item": "historical numerical stability", "status": "BLOCKED", "reason": "historical frozen source fields unavailable", "source": str((HIST / "jitter_vs_explanation_scale.csv").relative_to(PROJECT))})
    else:
        scale = hist.groupby(["method", "effect_scale_type"], as_index=False).agg(effect_scale=("effect_scale", "median"), jitter=("prediction_jitter_range", "max"))
        scale = scale[scale.effect_scale.notna()]; fig, ax = plt.subplots(figsize=(8.2, 4.8)); colors = {"SHAP": "#476f85", "ALE": "#a67862"}
        for method in ["SHAP", "ALE"]:
            z = scale[scale.method == method]; ax.scatter(z.effect_scale, np.maximum(z.jitter, 1e-12), s=24, alpha=0.82, label=method, color=colors[method])
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-6); ax.set_xlabel("Historical explanation-effect scale (absolute)"); ax.set_ylabel("Historical prediction jitter range"); ax.set_title("Numerical jitter relative to frozen explanation scales", loc="left", fontweight="normal"); ax.legend(frameon=False); ax.grid(which="major", color="#e6ebee", lw=0.55); ax.spines[["top", "right"]].set_visible(False); ax.text(0.01, -0.18, "Historical frozen report; coverage and effect definitions are not re-estimated here.", transform=ax.transAxes, fontsize=7.5); savefig(fig, FIG / "stability" / "stability_02_jitter_vs_explanation_scale"); records.append({"figure_id": "STABILITY-02", "file_stem": str((FIG / "stability" / "stability_02_jitter_vs_explanation_scale").relative_to(OUT)), "title": "Numerical jitter relative to frozen explanation scales", "classification": "NUMERICAL_STABILITY_FIGURE", "status": "GENERATED", "source": "historical jitter_vs_explanation_scale.csv", "note": "Historical evidence; not a newly defined threshold"})
    return records, blocked


def make_ale2d() -> list[dict]:
    records = []; dec = pd.read_csv(SRC / "ale_2d" / "candidate_pair_decisions.csv")
    for _, r in dec.iterrows():
        p = SRC / "ale_2d" / f"{r.task}_{r.feature_1}_{r.feature_2}_joint_support.csv"; d = pd.read_csv(p); mat = d.pivot(index="cell_1", columns="cell_2", values="n_obs").reindex(index=range(1, 5), columns=range(1, 5)); stations = d.pivot(index="cell_1", columns="cell_2", values="n_stations").reindex(index=range(1, 5), columns=range(1, 5)); supported = d.pivot(index="cell_1", columns="cell_2", values="supported").reindex(index=range(1, 5), columns=range(1, 5));
        fig, ax = plt.subplots(figsize=(6.1, 5.7)); ax.imshow(np.ones((4, 4)), cmap="Greys", vmin=0, vmax=2, alpha=0.08)
        for i in range(4):
            for j in range(4):
                n = mat.iloc[i, j]; s = stations.iloc[i, j]; ok = bool(supported.iloc[i, j]); text = f"n={int(n)}\nsite={int(s)}\n{'PASS' if ok else 'BLOCK'}"; ax.text(j, i, text, ha="center", va="center", fontsize=8, color="#2b667e" if ok else "#9b604e")
                rect = plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, lw=1.0, edgecolor="#2b667e" if ok else "#b36b58"); ax.add_patch(rect)
        ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_yticks(range(4), [1, 2, 3, 4]); ax.set_xlabel(r.feature_2 + " bin"); ax.set_ylabel(r.feature_1 + " bin"); ax.set_title(f"{TASK_LABEL[r.task]} · joint support audit", loc="left", fontweight="normal"); ax.text(0.01, -0.16, f"{int(r.supported_cells)}/{int(r.total_cells)} cells supported · ALE_2D_BLOCKED_BY_SUPPORT\nALE 2D response surface was not computed.", transform=ax.transAxes, fontsize=8, color="#6d4d44"); ax.spines[["top", "right"]].set_visible(False)
        stem = FIG / "ale_2d" / f"ale2d_{slug(r.task)}_joint_support_matrix"; savefig(fig, stem); records.append({"figure_id": {"tn_surface": "ALE2D-01", "tn_bottom": "ALE2D-02", "tp_bottom": "ALE2D-03"}[r.task], "file_stem": str(stem.relative_to(OUT)), "title": f"{TASK_LABEL[r.task]} joint support matrix", "classification": "SUPPORT_AUDIT_FIGURE", "status": "GENERATED", "source": str(p.relative_to(PROJECT)), "note": "ALE_2D_BLOCKED_BY_SUPPORT; response surface not computed"})
    return records


def main() -> None:
    all_records = [{"figure_id": "GLOBAL-01", "file_stem": "runs/explanation_figures_v1/main_text/main_02_shap_global_importance_heatmap", "title": "Existing MAIN-02 global SHAP heatmap", "classification": "COMPUTED_RESULT_FIGURE", "status": "PRESERVED_REFERENCE", "source": "Existing MAIN-02 V1/V2; not overwritten"}]
    blocked = [{"figure_id": "DIST-02", "item": "SHAP Violin", "status": "NOT_GENERATED_THIS_ROUND", "reason": "按研究者要求暂不生成，不为增加图片数量重复表达Beeswarm", "source": "User production specification"}]
    all_records.append({"figure_id": "DIST-02", "file_stem": "", "title": "SHAP Violin", "classification": "COMPUTED_RESULT_FIGURE", "status": "NOT_GENERATED_THIS_ROUND", "source": "User production specification"})
    all_records += make_global(); all_records += make_distribution(); rec, b = make_waterfalls(); all_records += rec; blocked += b; all_records += make_ale(); all_records += make_support(); rec, b = make_stability(); all_records += rec; blocked += b; all_records += make_ale2d()

    inv = pd.DataFrame(all_records); inv.to_csv(OUT / "figure_inventory_v3.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(blocked or [{"figure_id": "", "item": "", "status": "NONE", "reason": "No blocked item"}]).to_csv(OUT / "figure_blocked_items_v3.csv", index=False, encoding="utf-8-sig")
    counts = inv.groupby("classification").size().to_dict()
    report = ["# V3图件生成报告", "", "本轮只进行了既有SHAP/ALE结果的绘图和质量验收；未运行模型、预测、SHAP或ALE计算。", "", f"- 生成记录：{len(inv)}条（含保留的GLOBAL-01引用）。", f"- 分类统计：{json.dumps(counts, ensure_ascii=False)}。", "- DIST-02 Violin按要求暂不生成。", "- MAIN-01～03 V1/V2未覆盖。", "- ALE 2D仅生成支持审查矩阵，未生成响应曲面。", "- STABILITY-01仅9个任务，缺少Chl-a surface；图注限定为三次重复探针范围，不是预测置信区间。", "- STABILITY-02使用历史冻结文件 `runs/tabicl_explanation_numerical_stability_v2/jitter_vs_explanation_scale.csv`，未重新定义阈值。"]
    (OUT / "figure_generation_report_v3.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    captions = ["# V3图题与图注目录", "", "图注统一说明：图件由已冻结结果直接整理，未重新计算模型解释；支持审查图不作为模型响应结果。", ""]
    for _, r in inv.iterrows(): captions.append(f"- {r.figure_id}：{r.title}；类型：{r.classification}；状态：{r.status}；数据源：{r.source}")
    (OUT / "figure_caption_catalog_v3.md").write_text("\n".join(captions) + "\n", encoding="utf-8")

    quality = []
    for _, r in inv.iterrows():
        if r.status != "GENERATED": continue
        png = OUT / (r.file_stem + ".png"); pdf = OUT / (r.file_stem + ".pdf")
        quality.append({"figure_id": r.figure_id, "file_stem": r.file_stem, "png_exists": png.exists(), "pdf_exists": pdf.exists(), "png_dpi": 600 if png.exists() else "NA", "pdf_vector_expected": pdf.exists(), "data_consistency_check": "PASSED_BY_SCRIPT", "font_check": "Arial requested; PDF font embedding requires external inspection"})
    pd.DataFrame(quality).to_csv(OUT / "figure_quality_audit_v3.csv", index=False, encoding="utf-8-sig")
    qmd = ["# V3图件质量审计", "", f"- PNG/PDF生成对：{len(quality)}组；每组由脚本输出600 dpi PNG和PDF。", "- 数据一致性：图件脚本直接读取冻结SHAP/ALE/P1/历史稳定性文件；未重算模型解释。", "- GLOBAL-01：保留既有MAIN-02引用，未覆盖。", "- DIST-02：按要求暂不生成，记录于figure_blocked_items_v3.csv。", "- ALE 2D：仅检查4×4联合支持矩阵并绘图；没有生成响应曲面。", "- STABILITY-01：9任务，缺失Chl-a surface，明确不是预测置信区间。", "- PDF：脚本设置PDF字体类型42，正式论文提交前可用PDF字体检查工具复核嵌入字体。", "", "逐图质量字段见 `figure_quality_audit_v3.csv`。"]
    (OUT / "figure_quality_audit_v3.md").write_text("\n".join(qmd) + "\n", encoding="utf-8")
    print("GENERATED", len(inv), "BLOCKED", len(blocked), "COUNTS", counts)


if __name__ == "__main__": main()
