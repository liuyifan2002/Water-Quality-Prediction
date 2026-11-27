"""按硕士学位论文的一图一主题原则重画MAIN-01～03。

只读取已完成解释结果和特征元数据；不调用模型、不生成新SHAP/ALE。
"""

from __future__ import annotations

import csv
import gzip
import re
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
META = OUT / "metadata"
INVENTORY = ROOT / "runs/explanation_figure_inventory_v1.csv"
FEATURE_MANIFEST = ROOT / "transfer/llm_extension_transfer_v1/data/feature_manifest/01_strict_feature_manifest.csv"
TAXONOMY = ROOT / "runs/multisource_ablation_v1/00_feature_source_taxonomy.csv"

TASKS = ["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom"]
LABELS = {"chla_surface": "Chl-a surface", "chla_middle": "Chl-a middle", "chla_bottom": "Chl-a bottom", "secchi": "Secchi", "tn_surface": "TN surface", "tn_middle": "TN middle", "tn_bottom": "TN bottom", "tp_surface": "TP surface", "tp_middle": "TP middle", "tp_bottom": "TP bottom"}
REPRESENTATIVE = ["chla_surface", "secchi", "tn_surface", "tp_middle"]
GROUP_ORDER = ["METEOROLOGY", "OCEAN_PHYSICS", "RAINFALL_HYDRO_TIDE", "HUMAN_ACTIVITY", "TEMPORAL_PROCESS", "SPATIAL_STATIC"]
GROUP_SHORT = {"METEOROLOGY": "Meteorology", "OCEAN_PHYSICS": "Ocean", "RAINFALL_HYDRO_TIDE": "Rain / hydro", "HUMAN_ACTIVITY": "Human activity", "TEMPORAL_PROCESS": "Temporal", "SPATIAL_STATIC": "Spatial"}
GROUP_COLORS = {"METEOROLOGY": "#5B7FA3", "OCEAN_PHYSICS": "#6F9E9B", "RAINFALL_HYDRO_TIDE": "#8B9E79", "HUMAN_ACTIVITY": "#A68A64", "TEMPORAL_PROCESS": "#8C8C8C", "SPATIAL_STATIC": "#6D6E71"}
PANEL_COLORS = ["#547A9D", "#5B969B", "#B58A4A", "#9E6262"]


def configure() -> None:
    MAIN.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
        "lines.linewidth": 1.1, "figure.dpi": 150, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = MAIN / f"{stem}.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def display_name(raw: str) -> tuple[str, str, str]:
    # Display labels are intentionally concise; raw feature names remain unchanged in source files.
    patterns = [
        (r"^ssrd_(mean|sum)_(\d+)d$", lambda m: (f"Solar rad. {m.group(1)} ({m.group(2)} d)", f"太阳辐射{m.group(1)}（{m.group(2)}天）", "ERA5-derived scale")),
        (r"^sst_anom_(\d+)d$", lambda m: (f"SST anomaly ({m.group(1)} d)", f"海温距平（{m.group(1)}天）", "°C-derived")),
        (r"^sst_(mean|max)_(\d+)d$", lambda m: (f"SST {m.group(1)} ({m.group(2)} d)", f"海温{m.group(1)}（{m.group(2)}天）", "°C-derived")),
        (r"^ntl_avg_rad(_delta1|_lag1|_lag2|_roll3)?$", lambda m: (f"Night lights{({'_delta1': ' Δ1', '_lag1': ' lag1', '_lag2': ' lag2', '_roll3': ' roll3'}.get(m.group(1), ''))}", f"夜间灯光{({'_delta1': ' Δ1', '_lag1': ' 滞后1', '_lag2': ' 滞后2', '_roll3': ' 滚动3'}.get(m.group(1), ''))}", "dataset index")),
        (r"^t2m(_delta1|_lag1|_lag2|_roll3)?$", lambda m: (f"Air temp.{({'_delta1': ' Δ1', '_lag1': ' lag1', '_lag2': ' lag2', '_roll3': ' roll3'}.get(m.group(1), ''))}", f"气温{({'_delta1': ' Δ1', '_lag1': ' 滞后1', '_lag2': ' 滞后2', '_roll3': ' 滚动3'}.get(m.group(1), ''))}", "°C-derived")),
        (r"^tcc(_delta1|_lag1|_lag2|_roll3)?$", lambda m: (f"Cloud cover{({'_delta1': ' Δ1', '_lag1': ' lag1', '_lag2': ' lag2', '_roll3': ' roll3'}.get(m.group(1), ''))}", f"云量{({'_delta1': ' Δ1', '_lag1': ' 滞后1', '_lag2': ' 滞后2', '_roll3': ' 滚动3'}.get(m.group(1), ''))}", "fraction-derived")),
        (r"^wind_speed(_delta1|_lag1|_lag2|_roll3|_sq)?$", lambda m: (f"Wind speed{({'_delta1': ' Δ1', '_lag1': ' lag1', '_lag2': ' lag2', '_roll3': ' roll3', '_sq': ' squared'}.get(m.group(1), ''))}", f"风速{({'_delta1': ' Δ1', '_lag1': ' 滞后1', '_lag2': ' 滞后2', '_roll3': ' 滚动3', '_sq': ' 平方'}.get(m.group(1), ''))}", "m s⁻¹-derived")),
        (r"^rain_(\d+)d(_delta1|_lag1|_lag2|_roll3)?$", lambda m: (f"Rainfall {m.group(1)} d{({'_delta1': ' Δ1', '_lag1': ' lag1', '_lag2': ' lag2', '_roll3': ' roll3'}.get(m.group(2), ''))}", f"降雨{m.group(1)}天{({'_delta1': ' Δ1', '_lag1': ' 滞后1', '_lag2': ' 滞后2', '_roll3': ' 滚动3'}.get(m.group(2), ''))}", "mm-derived")),
        (r"^tidal_range(_delta1|_lag1|_lag2|_roll3)?$", lambda m: (f"Tidal range{({'_delta1': ' Δ1', '_lag1': ' lag1', '_lag2': ' lag2', '_roll3': ' roll3'}.get(m.group(1), ''))}", f"潮差{({'_delta1': ' Δ1', '_lag1': ' 滞后1', '_lag2': ' 滞后2', '_roll3': ' 滚动3'}.get(m.group(1), ''))}", "m-derived")),
    ]
    for pattern, fn in patterns:
        match = re.match(pattern, raw)
        if match:
            return fn(match)
    fixed = {
        "viirs_count": ("VIIRS observation count", "VIIRS观测次数", "count"), "day_of_year": ("Day of year", "年内日序", "day index"),
        "days_since_prev_sample": ("Days since prior sample", "距上次采样天数", "days"), "is_wet_season": ("Wet-season indicator", "湿季指示变量", "binary"),
        "month_cos": ("Month cosine", "月份余弦", "dimensionless"), "month_sin": ("Month sine", "月份正弦", "dimensionless"),
        "quarter": ("Quarter", "季度", "index"), "year_offset": ("Year offset", "年份偏移", "year index"),
        "tcc": ("Cloud cover", "云量", "fraction-derived"), "u10": ("10-m zonal wind", "10米纬向风", "m s⁻¹-derived"),
        "v10": ("10-m meridional wind", "10米经向风", "m s⁻¹-derived"), "wind_dir_cos": ("Wind direction cosine", "风向余弦", "dimensionless"),
        "wind_dir_sin": ("Wind direction sine", "风向正弦", "dimensionless"), "wind_speed": ("Wind speed", "风速", "m s⁻¹-derived"),
        "wind_speed_x_rain14d": ("Wind speed × rain (14 d)", "风速×14天降雨", "derived product scale"),
        "t2m_x_rain14d": ("Air temp. × rain (14 d)", "气温×14天降雨", "derived product scale"),
        "t2m_x_tidal_range": ("Air temp. × tidal range", "气温×潮差", "derived product scale"),
        "rain_month": ("Monthly rainfall", "月降雨", "mm-derived"), "rain_month_delta1": ("Monthly rain Δ1", "月降雨 Δ1", "mm-derived"),
        "rain_month_lag1": ("Monthly rain lag1", "月降雨滞后1", "mm-derived"), "rain_month_lag2": ("Monthly rain lag2", "月降雨滞后2", "mm-derived"),
        "rain_month_roll3": ("Monthly rain roll3", "月降雨滚动3", "mm-derived"), "lat": ("Latitude", "纬度", "degree"), "lon": ("Longitude", "经度", "degree"),
        "rain14d_x_ssrd_mean_7d": ("Rain (14 d) × solar rad. (7 d)", "14天降雨×7天太阳辐射", "derived product scale"),
        "rain14d_x_tidal_range": ("Rain (14 d) × tidal range", "14天降雨×潮差", "derived product scale"),
        "sst_anom_7d_x_ssrd_mean_7d": ("SST anomaly (7 d) × solar rad. mean (7 d)", "7天海温距平×7天平均太阳辐射", "derived product scale"),
        "sst_mean_7d_x_ssrd_mean_7d": ("SST mean (7 d) × solar rad. mean (7 d)", "7天平均海温×7天平均太阳辐射", "derived product scale"),
    }
    return fixed.get(raw, (raw.replace("_", " ").title(), raw, "model feature scale"))


def build_mapping() -> pd.DataFrame:
    manifest = pd.read_csv(FEATURE_MANIFEST).query("included_in_pilot == True")
    taxonomy = pd.read_csv(TAXONOMY)[["feature_name", "source_group"]].rename(columns={"feature_name": "raw_feature_name"})
    frame = manifest[["feature"]].rename(columns={"feature": "raw_feature_name"}).merge(taxonomy, on="raw_feature_name", how="left")
    rows = []
    for row in frame.itertuples(index=False):
        en, cn, unit = display_name(row.raw_feature_name)
        rows.append({"raw_feature_name": row.raw_feature_name, "display_name_en": en, "display_name_cn": cn, "source_group": row.source_group, "unit": unit})
    result = pd.DataFrame(rows)
    if len(result) != 77 or result.source_group.isna().any():
        raise RuntimeError("publication feature mapping未覆盖77个冻结特征")
    result.to_csv(META / "feature_display_name_mapping.csv", index=False, encoding="utf-8-sig")
    return result


def load_shap(mapping: pd.DataFrame) -> pd.DataFrame:
    files = sorted((RUN / "shap").glob("*/**/*.csv.gz"))
    if len(files) != 40:
        raise RuntimeError(f"SHAP数量异常：{len(files)}")
    frames = []
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
            data = pd.read_csv(stream)
        if len(data) != 77 or data.status.iloc[0] != "COMPUTED_PASS":
            raise RuntimeError(f"SHAP文件未通过只读检查：{path}")
        data["task"] = path.parent.name
        data["seed"] = int(re.search(r"_seed(\d+)", path.name).group(1))
        frames.append(data)
    result = pd.concat(frames, ignore_index=True).merge(mapping, left_on="feature", right_on="raw_feature_name", how="left")
    if result.display_name_en.isna().any():
        raise RuntimeError("SHAP中存在未映射特征")
    return result


def redraw_flowchart() -> list[str]:
    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    # 三种主要颜色：蓝=冻结/计算，青=解释分支，红=质量控制。
    nodes = [
        (0.6, 4.65, 2.45, 1.05, "Frozen protocol\n77 strict features\nfold_1 / pre-registered", "#3E6388"),
        (4.0, 4.65, 2.45, 1.05, "P1 pointwise prediction\nfold_1 / read-only\ncompleted model outputs", "#3E6388"),
        (7.4, 5.25, 2.15, 1.05, "SHAP\n40 explanation groups", "#4C8C8C"),
        (7.4, 3.15, 2.15, 1.05, "ALE 1D\n20 completed curves", "#4C8C8C"),
        (4.0, 1.10, 2.45, 1.05, "Numerical stability QC\nrepeatability and sensitivity", "#9A5A55"),
        (0.6, 1.10, 2.45, 1.05, "Feature support\n10-bin / station support", "#9A5A55"),
        (7.4, 1.10, 2.15, 1.05, "ALE 2D support audit\nblocked before model calculation", "#9A5A55"),
    ]
    for x, y, w, h, text, color in nodes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor="white", edgecolor=color, linewidth=1.35))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#263238", linespacing=1.35)
    arrows = [((3.05, 5.18), (4.0, 5.18)), ((6.45, 5.18), (7.4, 5.70)), ((6.45, 5.18), (7.4, 3.68)), ((4.0, 4.80), (3.05, 2.15)), ((3.05, 2.15), (4.0, 1.62)), ((6.45, 1.62), (7.4, 1.62)), ((8.48, 3.15), (8.48, 2.15))]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 1.15, "shrinkA": 5, "shrinkB": 5})
    return save(fig, "main_01_tabicl_v2_explanation_qc_workflow")


def redraw_heatmap(shap: pd.DataFrame, mapping: pd.DataFrame) -> list[str]:
    table = shap.groupby(["task", "feature", "display_name_en", "source_group"], sort=False).shap_value.apply(lambda x: np.mean(np.abs(x))).reset_index(name="mean_abs_shap")
    normalized = table.pivot(index="task", columns="feature", values="mean_abs_shap").reindex(TASKS).fillna(0)
    norm = normalized.div(normalized.max(axis=1).replace(0, np.nan), axis=0).fillna(0)
    top = norm.mean(axis=0).sort_values(ascending=False).head(20).index.tolist()
    order = mapping.set_index("raw_feature_name").loc[top].assign(group_order=lambda x: x.source_group.map({g: i for i, g in enumerate(GROUP_ORDER)})).sort_values(["group_order", "source_group"], kind="stable").index.tolist()
    display = mapping.set_index("raw_feature_name").loc[order]
    plot = norm[order]
    fig = plt.figure(figsize=(13.5, 7.2))
    ax = fig.add_axes([0.18, 0.19, 0.69, 0.67])
    im = ax.imshow(plot.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(display.display_name_en, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(TASKS))); ax.set_yticklabels([LABELS[t] for t in TASKS])
    ax.set_xlabel("Feature (publication display name; top 20 by task-averaged normalized importance)", labelpad=12)
    ax.set_ylabel("Water-quality task")
    # Family separators: Chl-a / Secchi / TN / TP.
    for y in [2.5, 3.5, 6.5]:
        ax.axhline(y, color="#A7ADB4", linewidth=0.75)
    # Category annotation above columns.
    top_ax = fig.add_axes([0.18, 0.87, 0.69, 0.065], sharex=ax)
    top_ax.set_xlim(-0.5, len(order) - 0.5); top_ax.set_ylim(0, 1); top_ax.axis("off")
    groups = display.source_group.tolist()
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            group = groups[start]
            left, right = start - 0.5, i - 0.5
            top_ax.add_patch(patches.Rectangle((left, 0), right - left, 0.52, facecolor=GROUP_COLORS.get(group, "#8C8C8C"), edgecolor="white", linewidth=0.5))
            top_ax.text((start + i - 1) / 2, 0.70, GROUP_SHORT.get(group, group), ha="center", va="bottom", fontsize=7, color="#344054")
            start = i
    cbar_ax = fig.add_axes([0.90, 0.19, 0.018, 0.67])
    cbar = fig.colorbar(im, cax=cbar_ax); cbar.set_label("Within-task normalized mean |SHAP|", labelpad=8); cbar.ax.tick_params(labelsize=8)
    return save(fig, "main_02_shap_global_importance_heatmap")


def redraw_lollipop(shap: pd.DataFrame, mapping: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), sharex=False)
    for ax, task, color, panel in zip(axes.ravel(), REPRESENTATIVE, PANEL_COLORS, "abcd"):
        data = shap[shap.task == task].groupby(["feature", "display_name_en"]).shap_value.apply(lambda x: np.mean(np.abs(x))).reset_index(name="mean_abs_shap").sort_values("mean_abs_shap").tail(10)
        y = np.arange(len(data))
        ax.hlines(y, 0, data.mean_abs_shap, color=color, linewidth=1.2, alpha=0.75)
        ax.scatter(data.mean_abs_shap, y, s=30, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels(data.display_name_en)
        ax.set_xlabel("Mean |SHAP value|")
        ax.text(-0.16, 1.04, f"({panel})", transform=ax.transAxes, fontsize=10, color="#263238")
        ax.text(0.0, 1.04, LABELS[task], transform=ax.transAxes, fontsize=9, color="#263238")
        ax.grid(False); ax.spines["left"].set_color("#4B5563"); ax.spines["bottom"].set_color("#4B5563")
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.10, top=0.93, wspace=0.36, hspace=0.34)
    return save(fig, "main_03_representative_shap_feature_importance_lollipop")


def update_inventory(paths: dict[str, list[str]]) -> None:
    frame = pd.read_csv(INVENTORY)
    replacements = {"MAIN-01": paths["MAIN-01"], "MAIN-02": paths["MAIN-02"], "MAIN-03": paths["MAIN-03"]}
    titles = {"MAIN-01": "TabICLv2解释性分析与质量控制流程", "MAIN-02": "10任务SHAP全局重要性热图", "MAIN-03": "代表性水质任务的SHAP特征重要性"}
    for fid, outputs in replacements.items():
        idx = frame.figure_id.eq(fid)
        frame.loc[idx, "target_output_path"] = ";".join(outputs)
        frame.loc[idx, "status"] = "GENERATED"
        frame.loc[idx, "figure_title"] = titles[fid]
        frame.loc[idx, "notes"] = frame.loc[idx, "notes"].astype(str).str.replace("PNG and PDF generated; no model calls.", "Redesigned for master's thesis; V1 retained under v1_comparison/; no model calls.", regex=False)
    frame.to_csv(INVENTORY, index=False, encoding="utf-8-sig")


def main() -> None:
    configure()
    mapping = build_mapping()
    shap = load_shap(mapping)
    paths = {"MAIN-01": redraw_flowchart(), "MAIN-02": redraw_heatmap(shap, mapping), "MAIN-03": redraw_lollipop(shap, mapping)}
    update_inventory(paths)
    (META / "main_redesign_state.csv").write_text("figure_id,status,model_calls,source_mutation\nMAIN-01,REDESIGNED,0,none\nMAIN-02,REDESIGNED,0,none\nMAIN-03,REDESIGNED,0,none\n", encoding="utf-8-sig")
    (META / "figure_generation_state.json").write_text(
        '{\n  "stage": "MAIN_01_TO_03_REDESIGNED",\n  "shap_groups": 40,\n  "ale_1d_curves": 20,\n  "model_calls": 0,\n  "v1_preserved": true,\n  "canonical_outputs": ["main_01_tabicl_v2_explanation_qc_workflow", "main_02_shap_global_importance_heatmap", "main_03_representative_shap_feature_importance_lollipop"]\n}\n', encoding="utf-8")
    (OUT / "figure_caption_draft.md").write_text(
        "# 学位论文图题草稿（MAIN-01～03重设计版）\n\n"
        "- **MAIN-01 TabICLv2解释性分析与质量控制流程。** 图示冻结的77个严格特征协议、P1逐样本预测、SHAP解释、ALE 1D、数值稳定性质量控制、特征支持审查及ALE 2D联合支持审查之间的科学关系。\n"
        "- **MAIN-02 10任务SHAP全局重要性热图。** 颜色表示各任务内归一化平均绝对SHAP值；列为按source group排序的全局Top 20特征，行按Chl-a、Secchi、TN和TP family分区。\n"
        "- **MAIN-03 代表性水质任务的SHAP特征重要性。** 以lollipop形式展示Chl-a表层、Secchi、TN表层和TP中层任务中Top 10特征的平均绝对SHAP值。\n\n"
        "图件生成仅使用已完成解释结果；未进行模型训练、推理或SHAP/ALE重算。\n", encoding="utf-8")
    (OUT / "figure_generation_report.md").write_text(
        "# 图件生成报告（MAIN-01～03重设计版）\n\n"
        "- 目标：硕士学位论文；采用一图一主题、充分展开的图件组织原则。\n"
        "- MAIN-01：独立科学流程图，最多三种主要颜色；已删除`Thesis figures`节点和底部长句。\n"
        "- MAIN-02：独立10任务SHAP热图；采用白色—浅蓝—深蓝色板，Top 20按source group排序，顶部为合并source-group色带，family之间加入细灰分隔线。\n"
        "- MAIN-03：独立SHAP lollipop 2×2面板；不加入ALE，使用四个指定代表任务、低饱和度单色和(a)–(d)标签。\n"
        "- V1图件已复制到`v1_comparison/main_text/`保留对照。\n"
        "- 本次未运行任何模型，模型调用数为0；未修改split、原始SHAP、ALE或预测结果。\n", encoding="utf-8")
    print("REDESIGN_COMPLETE")
    for fid, outputs in paths.items():
        print(fid, *outputs, sep="\t")


if __name__ == "__main__":
    main()
