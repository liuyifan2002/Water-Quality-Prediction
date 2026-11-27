"""汇总精简解释实验的真实结果，站点bootstrap与论文级图件独立生成。"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v1"
RUNNER = ROOT / "脚本/05_诊断分析/88_run_tabicl_slim_explainability_v1.py"
XGB = ROOT / "runs/xgb_shap_ale_v2/01_shap/01_shap_importance_summary.csv"
XGB_ALE = ROOT / "runs/xgb_shap_ale_v2/02_ale_1d/02_ale_1d_curves.csv"
TAXONOMY = ROOT / "runs/multisource_ablation_v1/00_feature_source_taxonomy.csv"
MODEL_TARGET_UNIT = {"chla": "µg/L", "tn": "mg/L", "tp": "mg/L", "secchi": "m"}


def load_runner():
    spec = importlib.util.spec_from_file_location("tabicl_slim_runner_summary", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_shap(task: str, expected_features: list[str]) -> pd.DataFrame:
    folder = OUT / "shap" / task
    if not folder.is_dir():
        return pd.DataFrame()
    frames = []
    for file in sorted(folder.glob("*_seed42.csv.gz")):
        frame = pd.read_csv(file)
        if len(frame) == 77 and frame.feature.astype(str).tolist() == expected_features \
                and frame.status.iloc[0] == "COMPUTED_PASS":
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def shap_summary(task: str, frame: pd.DataFrame, runner) -> pd.DataFrame:
    tax = pd.read_csv(TAXONOMY).set_index("feature_name")
    result = frame.groupby("feature", as_index=False).shap_value.agg(
        mean_abs_shap=lambda s: float(np.mean(np.abs(s))),
        median_abs_shap=lambda s: float(np.median(np.abs(s))),
    )
    result["rank"] = result.mean_abs_shap.rank(method="first", ascending=False).astype(int)
    result["source_group"] = result.feature.map(tax.source_group)
    result["physical_meaning"] = result.feature.map(tax.physical_meaning)
    result["task"] = task
    result["fold"] = "fold_1"
    result["n_explained_samples"] = frame.obs_id.nunique()
    result["scope"] = "selected_samples_only; not global test-set importance"
    result = result.sort_values("rank")
    runner.atomic_csv(OUT / "shap" / task / "selected_sample_importance.csv", result)
    return result


def bootstrap_ale(task: str, feature: str, runner) -> pd.DataFrame:
    curve_path = OUT / "ale_1d" / f"{task}_{feature}_curve.csv"
    if not curve_path.exists():
        return pd.DataFrame()
    curve = pd.read_csv(curve_path).sort_values("bin")
    with sqlite3.connect(OUT / "checkpoints/ale_1d_local_effects.sqlite") as conn:
        local = pd.read_sql_query("SELECT station_id, bin, local_effect FROM local_effect WHERE task=? AND fold=? AND feature=?",
            conn, params=(task, "fold_1", feature))
    if len(local) != int(curve.n_samples.sum()):
        raise RuntimeError("bootstrap输入与已保存曲线的局部差值数量不一致")
    station_ids = sorted(local.station_id.astype(str).unique())
    n_bins = len(curve)
    # 使用特征哈希的稳定十六进制整数，避免Python随机哈希种子影响bootstrap。
    stable_seed = 42 + runner.TASKS.index(task) * 100 + int(runner.hash_ids([feature])[:8], 16)
    rng = np.random.default_rng(stable_seed)
    by_station = {station: local.loc[local.station_id.astype(str).eq(station)] for station in station_ids}
    curves = []
    for _ in range(500):
        draw = rng.choice(station_ids, size=len(station_ids), replace=True)
        sample = pd.concat([by_station[station] for station in draw], ignore_index=True)
        counts = sample.groupby("bin").size().reindex(curve.bin, fill_value=0).to_numpy(int)
        if np.any(counts == 0):
            continue
        effects = sample.groupby("bin").local_effect.mean().reindex(curve.bin).to_numpy(float)
        cumulative = np.cumsum(effects)
        curves.append(cumulative - np.average(cumulative, weights=counts))
    if not curves:
        raise RuntimeError("500次站点bootstrap没有完整分箱的有效重复")
    values = np.asarray(curves)
    result = curve[["task", "fold", "feature", "bin", "lower", "upper", "ale"]].copy()
    result["bootstrap_low"] = np.quantile(values, .025, axis=0)
    result["bootstrap_high"] = np.quantile(values, .975, axis=0)
    result["bootstrap_valid_repeats"] = len(values)
    result["bootstrap_requested_repeats"] = 500
    result["bootstrap_unit"] = "station"
    result["bootstrap_seed"] = stable_seed
    runner.atomic_csv(OUT / "ale_1d" / f"{task}_{feature}_bootstrap.csv", result)
    return result


def draw_shap(task: str, data: pd.DataFrame) -> list[str]:
    if data.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = data.head(15).iloc[::-1]
    fig, axis = plt.subplots(figsize=(7.2, 5.6), dpi=320)
    axis.barh(selected.feature, selected.mean_abs_shap, color="#28739B")
    unit = MODEL_TARGET_UNIT[task.split("_")[0]]
    axis.set_xlabel(f"Mean |SHAP| over selected queries ({unit})")
    axis.set_title(f"TabICLv2, {task}, fold 1 (n={int(data.n_explained_samples.iloc[0])})")
    axis.grid(axis="x", color="#dddddd", linewidth=.5)
    fig.tight_layout()
    folder = OUT / "figures" / "shap"
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg"):
        path = folder / f"{task}_top15.{suffix}"
        fig.savefig(path, dpi=320, facecolor="white")
        outputs.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return outputs


def draw_ale(task: str, feature: str, values: pd.DataFrame) -> list[str]:
    if values.empty:
        return []
    known_units = pd.read_csv(XGB_ALE, usecols=["feature", "unit"]).drop_duplicates()
    unit_rows = known_units.loc[known_units.feature.eq(feature), "unit"].dropna()
    if unit_rows.empty or len(set(unit_rows.astype(str))) != 1:
        # 不用臆测单位制作论文图；数值曲线与绘图数据仍然保留。
        return []
    unit = str(unit_rows.iloc[0])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = (values.lower.to_numpy(float) + values.upper.to_numpy(float)) / 2
    fig, axis = plt.subplots(figsize=(6.4, 4.3), dpi=320)
    axis.plot(centers, values.ale, color="#28739B", linewidth=1.8)
    axis.fill_between(centers, values.bootstrap_low, values.bootstrap_high,
        color="#28739B", alpha=.12, label="Station bootstrap, 95% interval")
    axis.axhline(0, color="#888888", linewidth=.7)
    axis.set_xlabel(f"{feature} ({unit})")
    axis.set_ylabel(f"ALE ({MODEL_TARGET_UNIT[task.split('_')[0]]})")
    axis.set_title(f"TabICLv2, {task}, fold 1")
    axis.legend(frameon=False)
    axis.grid(color="#dddddd", linewidth=.5)
    fig.tight_layout()
    folder = OUT / "figures" / "ale_1d"
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg"):
        path = folder / f"{task}_{feature}.{suffix}"
        fig.savefig(path, dpi=320, facecolor="white")
        outputs.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return outputs


def draw_ale_2d(task: str, first: str, second: str, values: pd.DataFrame) -> list[str]:
    known = pd.read_csv(XGB_ALE, usecols=["feature", "unit"]).drop_duplicates()
    units = {}
    for feature in (first, second):
        rows = known.loc[known.feature.eq(feature), "unit"].dropna().astype(str).unique()
        if len(rows) != 1:
            return []
        units[feature] = rows[0]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = values.sort_values(["cell_1", "cell_2"])
    if len(values) != 16 or not values.status.eq("SUPPORTED").all():
        return []
    surface = values.ale_2d_interaction.to_numpy(float).reshape(4, 4)
    edge1 = values.sort_values("cell_1").drop_duplicates("cell_1")
    edge2 = values.sort_values("cell_2").drop_duplicates("cell_2")
    x = edge1.edge_1_lower.to_list() + [float(edge1.edge_1_upper.iloc[-1])]
    y = edge2.edge_2_lower.to_list() + [float(edge2.edge_2_upper.iloc[-1])]
    fig, axis = plt.subplots(figsize=(6.4, 4.8), dpi=320)
    mesh = axis.pcolormesh(x, y, surface.T, cmap="RdBu_r", shading="flat",
        vmin=-max(abs(surface.min()), abs(surface.max())),
        vmax=max(abs(surface.min()), abs(surface.max())))
    fig.colorbar(mesh, ax=axis, label=f"Second-order ALE ({MODEL_TARGET_UNIT[task.split('_')[0]]})")
    axis.set_xlabel(f"{first} ({units[first]})")
    axis.set_ylabel(f"{second} ({units[second]})")
    axis.set_title(f"TabICLv2, {task}, fold 1; observed cells only")
    fig.tight_layout()
    folder = OUT / "figures" / "ale_2d"
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg"):
        path = folder / f"{task}_{first}_{second}.{suffix}"
        fig.savefig(path, dpi=320, facecolor="white")
        outputs.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return outputs


def main() -> None:
    runner = load_runner()
    feature_names = runner.load_base().load_features()
    audit = pd.read_csv(OUT / "00_frozen_protocol_task_registry.csv")
    xgb = pd.read_csv(XGB)
    statuses = []
    comparisons = []
    figure_rows = []
    call_rows = []
    unit_rows = []
    sensitivity_rows = []
    for item in audit.itertuples():
        task = str(item.task)
        gate_path = OUT / "gates" / f"{task}_p1_decision.csv"
        gate = pd.read_csv(gate_path).iloc[0] if gate_path.exists() else None
        gate_pass = gate is not None and gate.status == "P1_GATE_PASS"
        batch_path = OUT / "gates" / f"{task}_batch_reproduction.csv"
        batch_state = pd.read_csv(batch_path).status.iloc[0] if batch_path.exists() else str(item.historical_batch_reproduction)
        unit_rows.append({"task": task, "fold": "fold_1", "method": "BATCH_REPRODUCTION_GATE",
            "unit": task, "status": batch_state, "output_path": str(batch_path) if batch_path.exists() else "historical V3 record"})
        unit_rows.append({"task": task, "fold": "fold_1", "method": "P1_GATE",
            "unit": task, "status": gate.status if gate is not None else "PENDING",
            "output_path": str(gate_path) if gate_path.exists() else ""})
        preregistered = pd.read_csv(OUT / "01_preregistered_shap_samples.csv")
        budget_file = OUT / "shap" / task / "core_three_budget_decision.csv"
        optional_needed = bool(pd.read_csv(budget_file).append_optional_two.iloc[0]) if budget_file.exists() else None
        for sample in preregistered.loc[preregistered.task.eq(task)].itertuples():
            path = OUT / "shap" / task / f"{sample.obs_id}_seed42.csv.gz"
            sample_status = "COMPUTED" if path.exists() else (
                "BLOCKED" if item.status == "BLOCKED" else (
                    "SKIPPED_NOT_REQUIRED" if sample.phase == "OPTIONAL_2" and optional_needed is False else "PENDING"))
            unit_rows.append({"task": task, "fold": "fold_1", "method": "SHAP_77",
                "unit": str(sample.obs_id), "status": sample_status,
                "output_path": str(path) if path.exists() else ""})
        shap = read_shap(task, feature_names)
        n_samples = shap.obs_id.nunique() if len(shap) else 0
        candidates = runner.sample_rows(task)
        n_core = len(set(shap.obs_id.astype(str)) & set(candidates.obs_id.astype(str))) if len(shap) else 0
        result = shap_summary(task, shap, runner) if len(shap) else pd.DataFrame()
        if len(result):
            for path in draw_shap(task, result):
                figure_rows.append({"task": task, "type": "SHAP_SELECTED_SAMPLE", "feature": "top15", "path": path})
        comparison = xgb.loc[xgb.task.eq(task)]
        if len(result) and len(comparison):
            xtop = set(comparison.nsmallest(10, "task_rank").feature.astype(str))
            ttop = set(result.head(10).feature.astype(str))
            comparisons.append({"task": task, "fold": "fold_1", "tabicl_samples": n_samples,
                "tabicl_top10_xgb_top10_overlap": len(ttop & xtop),
                "common_top10_features": ";".join(sorted(ttop & xtop)),
                "comparison_definition": "rank overlap only; XGB log1p TreeSHAP and TabICL original-scale SHAP not numerically compared"})
        curves = []
        screening = OUT / "ale_1d" / f"{task}_feature_screening.csv"
        if screening.is_file():
            for feature in pd.read_csv(screening).loc[lambda frame: frame.proposed.astype(bool)].feature.astype(str):
                curve_file = OUT / "ale_1d" / f"{task}_{feature}_curve.csv"
                unit_rows.append({"task": task, "fold": "fold_1", "method": "ALE_1D",
                    "unit": feature, "status": "COMPUTED" if curve_file.exists() else "PENDING",
                    "output_path": str(curve_file) if curve_file.exists() else ""})
                if curve_file.is_file():
                    curve_data = pd.read_csv(curve_file)
                    curves.append(curve_data)
                    jitter = float(curve_data.repeatability_max_local_delta.iloc[0])
                    amplitude = float(curve_data.ale.max() - curve_data.ale.min())
                    typical = float(curve_data.local_effect.abs().median())
                    sensitivity_rows.append({"task": task, "fold": "fold_1", "feature": feature,
                        "observed_repeatability_delta": jitter,
                        "ale_curve_range": amplitude,
                        "median_abs_bin_local_effect": typical,
                        "delta_over_curve_range": jitter / amplitude if amplitude > 0 else np.nan,
                        "delta_over_median_local_effect": jitter / typical if typical > 0 else np.nan,
                        "interpretation": "经验数值尺度比较；不是物理因果检验或全域噪声上界"})
                    bootstrap = bootstrap_ale(task, feature, runner)
                    for path in draw_ale(task, feature, bootstrap):
                        figure_rows.append({"task": task, "type": "ALE_1D", "feature": feature, "path": path})
        pair_decisions = OUT / "ale_2d" / "candidate_pair_decisions.csv"
        if pair_decisions.is_file():
            pairs = pd.read_csv(pair_decisions)
            for pair in pairs.loc[pairs.task.eq(task) & pairs.selected.astype(bool)].itertuples():
                surface = OUT / "ale_2d" / f"{task}_{pair.feature_1}_{pair.feature_2}_surface.csv"
                if surface.exists():
                    for path in draw_ale_2d(task, pair.feature_1, pair.feature_2, pd.read_csv(surface)):
                        figure_rows.append({"task": task, "type": "ALE_2D", "feature": f"{pair.feature_1} × {pair.feature_2}", "path": path})
        n_ale = len(curves)
        shap_files = sorted((OUT / "shap" / task).glob("*_seed*.csv.gz")) if (OUT / "shap" / task).is_dir() else []
        shap_calls = 0
        cache_hits = 0
        for file in shap_files:
            one = pd.read_csv(file, nrows=1)
            shap_calls += int(one.actual_model_calls.iloc[0])
            cache_hits += int(one.exact_input_cache_hits.iloc[0]) if "exact_input_cache_hits" in one else 0
        ale_calls = 0
        ale_db = OUT / "checkpoints" / "ale_1d_local_effects.sqlite"
        if ale_db.exists():
            with sqlite3.connect(ale_db) as conn:
                ale_calls = int(conn.execute("SELECT COALESCE(SUM(actual_model_calls),0) FROM local_effect WHERE task=?",
                    (task,)).fetchone()[0])
        ale_repeat_calls = 0
        for repeat_file in (OUT / "ale_1d").glob(f"{task}_*_repeatability.csv") if (OUT / "ale_1d").exists() else []:
            ale_repeat_calls += int(pd.read_csv(repeat_file).actual_model_calls.sum())
        call_rows.append({"task": task, "fold": "fold_1", "shap_files": len(shap_files),
            "shap_actual_model_calls": shap_calls, "shap_exact_input_cache_hits": cache_hits,
            "ale_1d_local_actual_model_calls": ale_calls,
            "ale_1d_repeatability_actual_model_calls": ale_repeat_calls,
            "reuse_note": "历史合格结果及新输出均按哈希与状态检查；已存在单元不再调用模型"})
        if item.status == "BLOCKED" or (gate is not None and gate.status == "P1_GATE_FAILED"):
            support = "BLOCKED"
        elif not gate_pass or n_core < 3:
            support = "INSUFFICIENT_SUPPORT"
        elif n_ale < 2 or any(curve.status.iloc[0] != "COMPUTED_PASS" for curve in curves):
            support = "PARTIAL_SUPPORT"
        else:
            # 三个代表样本、单折解释没有折间与全测试集稳定性证据。
            support = "PARTIAL_SUPPORT"
        statuses.append({"task": task, "fold": "fold_1", "asset_status": item.status,
            "p1_gate": gate.status if gate is not None else "PENDING",
            "shap_completed_samples": n_samples, "core_shap_completed": n_core,
            "ale_1d_completed_features": n_ale, "decision": support,
            "scope_note": "single fold; selected query SHAP; no fold stability claim"})
    runner.atomic_csv(OUT / "04_task_decisions.csv", pd.DataFrame(statuses))
    runner.atomic_csv(OUT / "05_xgb_rank_reference.csv", pd.DataFrame(comparisons,
        columns=["task", "fold", "tabicl_samples", "tabicl_top10_xgb_top10_overlap",
                 "common_top10_features", "comparison_definition"]))
    runner.atomic_csv(OUT / "06_figure_inventory.csv", pd.DataFrame(figure_rows,
        columns=["task", "type", "feature", "path"]))
    runner.atomic_csv(OUT / "08_actual_calls_and_reuse.csv", pd.DataFrame(call_rows))
    runner.atomic_csv(OUT / "09_run_registry.csv", pd.DataFrame(unit_rows))
    runner.atomic_csv(OUT / "10_numerical_sensitivity.csv", pd.DataFrame(sensitivity_rows,
        columns=["task", "fold", "feature", "observed_repeatability_delta", "ale_curve_range",
            "median_abs_bin_local_effect", "delta_over_curve_range", "delta_over_median_local_effect",
            "interpretation"]))
    counts = pd.DataFrame(statuses).decision.value_counts().to_dict()
    report = ["# TabICLv2十任务单折精简解释：阶段性决策", "",
        f"任务状态计数：`{json.dumps(counts, ensure_ascii=False)}`。", "",
        "| 任务 | 旧批量复现 | P1门槛 | 核心SHAP | 一维ALE | 决策 |",
        "|---|---|---|---:|---:|---|",
        *[f"| {row['task']} | {next(item for item in unit_rows if item['task'] == row['task'] and item['method'] == 'BATCH_REPRODUCTION_GATE')['status']} | {row['p1_gate']} | {row['core_shap_completed']} | {row['ale_1d_completed_features']} | {row['decision']} |" for row in statuses],
        "",
        "审计、P1资格门槛、77维SHAP与ALE分别按保存的数值文件判定；图件不是通过资格的替代证据。",
        "Chl-a表层fold 1的原历史严格复现失败及ALE重复性失败保持不变。",
        "单折与少量查询样本只提供探索性局部解释，不形成十任务全局或五折稳定性结论。",
        "XGBoost对照只使用已有相同任务的前10排名重合，未比较不同输出尺度的原始SHAP值。", ""]
    (OUT / "07_最终实验决策.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"decision_counts": counts, "figures": len(figure_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
