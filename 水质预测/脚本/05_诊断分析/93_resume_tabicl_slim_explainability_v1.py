"""十任务单折精简解释的串行可恢复调度器；仅运行资格通过的缺失单元。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v1"
SCRIPTS = ROOT / "脚本/05_诊断分析"
PATCH = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
TASKS = (
    "chla_surface", "chla_middle", "chla_bottom", "secchi",
    "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom",
)


def run_script(script: str, args: list[str], unit: str) -> bool:
    output = OUT / "logs" / f"{unit}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env["PYTHONPATH"] = str(PATCH) + os.pathsep + env.get("PYTHONPATH", "")
    with output.open("a", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(json.dumps({"unit": unit, "exit_code": result.returncode, "log": str(output)},
        ensure_ascii=False), flush=True)
    return result.returncode == 0


def verified_shap_ids(task: str, phase: str) -> list[str]:
    manifest = pd.read_csv(OUT / "01_preregistered_shap_samples.csv")
    ids = manifest.loc[manifest.task.eq(task) & manifest.phase.eq(phase), "obs_id"].astype(str).tolist()
    complete = []
    for obs in ids:
        file = OUT / "shap" / task / f"{obs}_seed42.csv.gz"
        if file.exists():
            frame = pd.read_csv(file)
            if len(frame) == 77 and frame.status.iloc[0] == "COMPUTED_PASS":
                complete.append(obs)
    return complete


def need_optional_two(task: str) -> bool:
    manifest = pd.read_csv(OUT / "01_preregistered_shap_samples.csv")
    ids = manifest.loc[manifest.task.eq(task) & manifest.phase.eq("CORE_3"), "obs_id"].astype(str).tolist()
    if len(ids) != 3 or len(verified_shap_ids(task, "CORE_3")) != 3:
        return False
    frames = [pd.read_csv(OUT / "shap" / task / f"{obs}_seed42.csv.gz") for obs in ids]
    first = pd.concat(frames[:2]).groupby("feature").shap_value.apply(lambda s: np.mean(np.abs(s)))
    last = pd.concat(frames[1:]).groupby("feature").shap_value.apply(lambda s: np.mean(np.abs(s)))
    top_a, top_b = set(first.nlargest(10).index), set(last.nlargest(10).index)
    overlap = len(top_a & top_b)
    median_id = ids[1]
    repeat_file = OUT / "shap" / task / f"{median_id}_seed43.csv.gz"
    alternate_overlap = np.nan
    alternate_rho = np.nan
    if repeat_file.exists():
        repeat = pd.read_csv(repeat_file)
        original = frames[1].set_index("feature").shap_value.abs()
        alternative = repeat.set_index("feature").shap_value.abs()
        alternate_overlap = len(set(original.nlargest(10).index) & set(alternative.nlargest(10).index))
        alternate_rho = float(original.rank().corr(alternative.rank(), method="spearman"))
    add_two = overlap < 5 or (np.isfinite(alternate_overlap) and alternate_overlap < 5)
    result = {"task": task, "fold": "fold_1", "core_n": 3,
        "top10_overlap_between_2_of_3_subsets": overlap,
        "permutation_seed42_seed43_top10_overlap": alternate_overlap,
        "permutation_seed42_seed43_spearman_abs_rank": alternate_rho,
        "decision_rule": "append_optional_two_if_sample_or_permutation_top10_overlap_lt_5",
        "append_optional_two": add_two,
        "note": "仅为是否增加样本的事前预算规则，不证明排名稳定"}
    pd.DataFrame([result]).to_csv(OUT / "shap" / task / "core_three_budget_decision.csv",
        index=False, encoding="utf-8-sig")
    return add_two


def resume() -> None:
    audit = pd.read_csv(OUT / "00_frozen_protocol_task_registry.csv")
    for row in audit.itertuples():
        task = str(row.task)
        if row.status == "BLOCKED":
            continue
        gate = OUT / "gates" / f"{task}_p1_decision.csv"
        if not gate.exists() or pd.read_csv(gate).status.iloc[0] != "P1_GATE_PASS":
            print(json.dumps({"task": task, "status": "SKIPPED_GATE_NOT_PASS"}, ensure_ascii=False), flush=True)
            continue
        core_ids = pd.read_csv(OUT / "01_preregistered_shap_samples.csv")
        expected = core_ids.loc[core_ids.task.eq(task) & core_ids.phase.eq("CORE_3")].shape[0]
        if len(verified_shap_ids(task, "CORE_3")) < expected:
            if not run_script("89_compute_tabicl_slim_shap_v1.py", ["--task", task, "--phase", "CORE_3"],
                              f"shap_core_{task}"):
                continue
        median_obs = core_ids.loc[core_ids.task.eq(task) & core_ids.target_quantile.eq(0.5), "obs_id"].astype(str)
        if len(median_obs) == 1 and not (OUT / "shap" / task / f"{median_obs.iloc[0]}_seed43.csv.gz").exists():
            if not run_script("89_compute_tabicl_slim_shap_v1.py", ["--task", task, "--phase", "SENSITIVITY_1"],
                              f"shap_sensitivity_{task}"):
                continue
        if need_optional_two(task) and len(verified_shap_ids(task, "OPTIONAL_2")) < 2:
            if not run_script("89_compute_tabicl_slim_shap_v1.py", ["--task", task, "--phase", "OPTIONAL_2"],
                              f"shap_optional_{task}"):
                continue
        screening = OUT / "ale_1d" / f"{task}_feature_screening.csv"
        if not screening.exists():
            if not run_script("90_compute_tabicl_slim_ale_1d_v1.py", ["--task", task, "--screen"],
                              f"ale_screen_{task}"):
                continue
        candidates = pd.read_csv(screening)
        for feature in candidates.loc[candidates.proposed.astype(bool), "feature"].astype(str):
            curve = OUT / "ale_1d" / f"{task}_{feature}_curve.csv"
            if curve.exists():
                saved = pd.read_csv(curve)
                if len(saved) and saved.status.iloc[0] in ("COMPUTED_PASS", "COMPUTED_WITH_NUMERICAL_LIMITATION"):
                    continue
            run_script("90_compute_tabicl_slim_ale_1d_v1.py",
                ["--task", task, "--calculate", "--feature", feature], f"ale_1d_{task}_{feature}")
        run_script("91_summarize_tabicl_slim_explainability_v1.py", [], f"summary_{task}")
    if run_script("92_compute_tabicl_slim_ale_2d_v1.py", ["--screen-all"], "ale_2d_joint_screen"):
        pair_file = OUT / "ale_2d" / "candidate_pair_decisions.csv"
        pairs = pd.read_csv(pair_file)
        for item in pairs.loc[pairs.selected.astype(bool)].itertuples():
            surface = OUT / "ale_2d" / f"{item.task}_{item.feature_1}_{item.feature_2}_surface.csv"
            if not surface.exists():
                run_script("92_compute_tabicl_slim_ale_2d_v1.py", ["--calculate-task", str(item.task)],
                    f"ale_2d_{item.task}")
    run_script("91_summarize_tabicl_slim_explainability_v1.py", [], "summary_final")


if __name__ == "__main__":
    resume()
