"""最多两组有联合支持的TabICLv2二阶ALE；先CPU审查，后P1计算。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v1"
RUNNER = ROOT / "脚本/05_诊断分析/88_run_tabicl_slim_explainability_v1.py"
GRID = 4
MIN_CELL_N = 30
MIN_CELL_STATIONS = 5
MAX_SELECTED_PAIRS = 2


def load_runner():
    spec = importlib.util.spec_from_file_location("tabicl_slim_runner_ale2", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def grid_definition(runner, task: str, first: str, second: str, train, test, x_train, features):
    j1, j2 = features.index(first), features.index(second)
    e1 = np.unique(np.quantile(x_train[:, j1], np.linspace(0, 1, GRID + 1))).astype(np.float32)
    e2 = np.unique(np.quantile(x_train[:, j2], np.linspace(0, 1, GRID + 1))).astype(np.float32)
    if len(e1) != GRID + 1 or len(e2) != GRID + 1:
        return None
    v1 = pd.to_numeric(test[first], errors="coerce").to_numpy(float)
    v2 = pd.to_numeric(test[second], errors="coerce").to_numpy(float)
    valid = np.isfinite(v1) & np.isfinite(v2) & (v1 >= e1[0]) & (v1 <= e1[-1]) \
        & (v2 >= e2[0]) & (v2 <= e2[-1])
    b1 = np.clip(np.digitize(v1, e1[1:-1], right=True), 0, GRID - 1)
    b2 = np.clip(np.digitize(v2, e2[1:-1], right=True), 0, GRID - 1)
    stations = test.station_id.astype(str).to_numpy()
    counts = np.zeros((GRID, GRID), int)
    sites = np.zeros((GRID, GRID), int)
    cells = []
    for i in range(GRID):
        for j in range(GRID):
            indices = np.flatnonzero(valid & (b1 == i) & (b2 == j))
            counts[i, j] = len(indices)
            sites[i, j] = len(set(stations[indices]))
            cells.append({"task": task, "fold": "fold_1", "feature_1": first,
                "feature_2": second, "cell_1": i + 1, "cell_2": j + 1,
                "n_obs": len(indices), "n_stations": sites[i, j],
                "supported": bool(len(indices) >= MIN_CELL_N and sites[i, j] >= MIN_CELL_STATIONS),
                "edge_1_lower": float(e1[i]), "edge_1_upper": float(e1[i + 1]),
                "edge_2_lower": float(e2[j]), "edge_2_upper": float(e2[j + 1])})
    return e1, e2, valid, b1, b2, counts, sites, pd.DataFrame(cells)


def screen_all() -> None:
    runner = load_runner()
    base = runner.load_base()
    features = base.load_features()
    all_pairs = []
    output_dir = OUT / "ale_2d"
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in runner.TASKS:
        gate_file = OUT / "gates" / f"{task}_p1_decision.csv"
        screening_file = OUT / "ale_1d" / f"{task}_feature_screening.csv"
        if not gate_file.exists() or not screening_file.exists():
            continue
        if pd.read_csv(gate_file).status.iloc[0] != "P1_GATE_PASS":
            continue
        selected = pd.read_csv(screening_file).loc[lambda frame: frame.proposed.astype(bool)]
        if len(selected) != 2:
            continue
        first, second = selected.feature.astype(str).tolist()
        train, test = base.load_fold(task, "fold_1", features)
        x_train = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
            train[features]).astype(np.float32)
        definition = grid_definition(runner, task, first, second, train, test, x_train, features)
        if definition is None:
            all_pairs.append({"task": task, "feature_1": first, "feature_2": second,
                "status": "INSUFFICIENT_SUPPORT", "reason": "4x4训练分位边界有重复值",
                "supported_cells": 0, "total_cells": 16, "selected": False})
            continue
        e1, e2, valid, b1, b2, counts, sites, cells = definition
        runner.atomic_csv(output_dir / f"{task}_{first}_{second}_joint_support.csv", cells)
        full_support = bool(cells.supported.all())
        all_pairs.append({"task": task, "feature_1": first, "feature_2": second,
            "status": "SUPPORTED_FOR_COMPUTATION" if full_support else "INSUFFICIENT_SUPPORT",
            "reason": "全部16格达到n>=30和独立站点>=5" if full_support else "至少一格缺乏联合观测支持",
            "supported_cells": int(cells.supported.sum()), "total_cells": 16,
            "min_cell_n": int(counts.min()), "min_cell_stations": int(sites.min()),
            "joint_observations": int(valid.sum()), "edge_1_hash": runner.hash_array(e1),
            "edge_2_hash": runner.hash_array(e2), "selected": False})
    if all_pairs:
        frame = pd.DataFrame(all_pairs)
        qualified = frame.loc[frame.status.eq("SUPPORTED_FOR_COMPUTATION")].sort_values(
            ["min_cell_n", "joint_observations", "task"], ascending=[False, False, True]).head(MAX_SELECTED_PAIRS)
        frame.loc[qualified.index, "selected"] = True
    else:
        frame = pd.DataFrame(columns=["task", "feature_1", "feature_2", "status", "reason",
            "supported_cells", "total_cells", "selected"])
    runner.atomic_csv(output_dir / "candidate_pair_decisions.csv", frame)
    print(json.dumps({"pairs_screened": len(frame),
        "pairs_selected": int(frame.selected.sum()) if len(frame) else 0}, ensure_ascii=False))


def selected_indices(valid, b1, b2, stations, task_index: int):
    rng = np.random.default_rng(42000 + task_index)
    chosen = []
    for i in range(GRID):
        for j in range(GRID):
            ids = np.flatnonzero(valid & (b1 == i) & (b2 == j))
            groups = {str(station): list(rng.permutation(ids[stations[ids] == station]))
                      for station in sorted(set(stations[ids]))}
            picked = []
            while len(picked) < min(40, len(ids)):
                progress = False
                for station in sorted(groups):
                    if groups[station] and len(picked) < min(40, len(ids)):
                        picked.append(int(groups[station].pop()))
                        progress = True
                if not progress:
                    break
            chosen.extend(picked)
    return np.asarray(chosen, dtype=int)


def calculate(task: str) -> None:
    runner = load_runner()
    decisions = pd.read_csv(OUT / "ale_2d" / "candidate_pair_decisions.csv")
    selected = decisions.loc[decisions.task.eq(task) & decisions.selected.astype(bool)]
    if len(selected) != 1:
        raise RuntimeError(f"{task}: 没有预先通过联合覆盖审查的组合")
    first, second = selected.iloc[0][["feature_1", "feature_2"]]
    base, features, train, test, x_train, x_test, adapter, inv, meta = runner.context(task)
    definition = grid_definition(runner, task, first, second, train, test, x_train, features)
    if definition is None:
        raise RuntimeError("4x4分位边界无效")
    e1, e2, valid, b1, b2, counts, sites, cells = definition
    if not cells.supported.all():
        raise RuntimeError("联合覆盖检查没有通过")
    if runner.hash_array(e1) != selected.edge_1_hash.iloc[0] \
            or runner.hash_array(e2) != selected.edge_2_hash.iloc[0]:
        raise RuntimeError("二维分箱与CPU审查的边界哈希不一致")
    station = test.station_id.astype(str).to_numpy()
    indices = selected_indices(valid, b1, b2, station, runner.TASKS.index(task))
    manifest = test.iloc[indices][["obs_id", "station_id", "date"]].copy()
    manifest["cell_1"] = b1[indices] + 1
    manifest["cell_2"] = b2[indices] + 1
    manifest["task"] = task
    manifest["feature_1"] = first
    manifest["feature_2"] = second
    manifest["edge_1_hash"] = runner.hash_array(e1)
    manifest["edge_2_hash"] = runner.hash_array(e2)
    manifest_path = OUT / "ale_2d" / f"{task}_{first}_{second}_query_manifest.csv"
    if manifest_path.exists():
        prior = pd.read_csv(manifest_path)
        if prior.obs_id.astype(str).tolist() != manifest.obs_id.astype(str).tolist():
            raise RuntimeError("已有二维ALE查询清单不同，禁止覆盖")
    else:
        runner.atomic_csv(manifest_path, manifest)
    db_path = OUT / "checkpoints/ale_2d_local_effects.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS local_second_difference (
        task TEXT, fold TEXT, feature_1 TEXT, feature_2 TEXT, obs_id TEXT,
        cell_1 INTEGER, cell_2 INTEGER, second_difference REAL,
        context_hash TEXT, edge_1_hash TEXT, edge_2_hash TEXT,
        PRIMARY KEY(task,fold,feature_1,feature_2,obs_id))""")
    j1, j2 = features.index(first), features.index(second)
    calls = 0
    for index in indices:
        obs_id = str(test.obs_id.iloc[index])
        exists = conn.execute("SELECT context_hash,edge_1_hash,edge_2_hash FROM local_second_difference WHERE task=? AND fold=? AND feature_1=? AND feature_2=? AND obs_id=?",
            (task, "fold_1", first, second, obs_id)).fetchone()
        expected = (meta["context_hash"], runner.hash_array(e1), runner.hash_array(e2))
        if exists:
            if exists != expected:
                raise RuntimeError("二维ALE旧检查点的模型上下文或分箱不同")
            continue
        i, j = int(b1[index]), int(b2[index])
        corner_predictions = []
        for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
            row = x_test[index].copy()
            row[j1] = e1[i + di]
            row[j2] = e2[j + dj]
            output = np.asarray(adapter.predict(row[None, :]), float).reshape(-1)
            calls += 1
            if output.size != 1 or not np.isfinite(output[0]):
                raise RuntimeError("二维ALE角点查询返回无效值")
            corner_predictions.append(float(max(float(inv(output)[0]), 0.0)))
        d2 = corner_predictions[3] - corner_predictions[1] - corner_predictions[2] + corner_predictions[0]
        conn.execute("INSERT INTO local_second_difference VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (task, "fold_1", first, second, obs_id, i + 1, j + 1, d2, *expected))
        conn.commit()
    local = pd.read_sql_query("SELECT * FROM local_second_difference WHERE task=? AND fold=? AND feature_1=? AND feature_2=?",
        conn, params=(task, "fold_1", first, second))
    if len(local) != len(manifest):
        conn.close()
        raise RuntimeError("二维ALE实际局部效应数与预登记样本数不一致")
    L = np.zeros((GRID, GRID), float)
    W = np.zeros((GRID, GRID), float)
    for i in range(GRID):
        for j in range(GRID):
            subset = local.loc[local.cell_1.eq(i + 1) & local.cell_2.eq(j + 1)]
            L[i, j] = subset.second_difference.mean()
            W[i, j] = len(subset)
    if not np.isfinite(L).all() or np.any(W < MIN_CELL_N):
        conn.close()
        raise RuntimeError("二阶局部效应存在空格或样本覆盖不足")
    surface = np.cumsum(np.cumsum(L, axis=0), axis=1)
    for _ in range(200):
        for i in range(GRID):
            surface[i, :] -= np.average(surface[i, :], weights=W[i, :])
        for j in range(GRID):
            surface[:, j] -= np.average(surface[:, j], weights=W[:, j])
    margins = max(np.max(np.abs((surface * W).sum(axis=0) / W.sum(axis=0))),
                  np.max(np.abs((surface * W).sum(axis=1) / W.sum(axis=1))))
    result = cells.copy()
    result["selected_n"] = [W[i, j] for i in range(GRID) for j in range(GRID)]
    result["local_second_difference"] = [L[i, j] for i in range(GRID) for j in range(GRID)]
    result["ale_2d_interaction"] = [surface[i, j] for i in range(GRID) for j in range(GRID)]
    result["weighted_margin_residual"] = float(margins)
    result["context_hash"] = meta["context_hash"]
    result["ale_2d_script_sha256"] = runner.file_sha(Path(__file__))
    result["protocol"] = runner.P1
    result["status"] = "SUPPORTED" if margins <= 1e-6 else "COMPUTATION_ERROR"
    runner.atomic_csv(OUT / "ale_2d" / f"{task}_{first}_{second}_surface.csv", result)
    conn.close()
    print(json.dumps({"task": task, "feature_1": first, "feature_2": second,
        "status": result.status.iloc[0], "query_rows": len(local),
        "new_actual_model_calls": calls, "centering_residual": float(margins)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-all", action="store_true")
    parser.add_argument("--calculate-task")
    args = parser.parse_args()
    if args.screen_all:
        screen_all()
    elif args.calculate_task:
        calculate(args.calculate_task)
    else:
        parser.error("使用--screen-all或--calculate-task")


if __name__ == "__main__":
    main()
