"""TabICLv2 数值复现与模型无关解释分析（独立扩展实验）。

本脚本只读取冻结划分、正式 77 特征和既有预测；不修改正式模型结果。
先通过 ``--stage reproduce`` 核验逐样本预测，再按批准范围运行解释阶段。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tabicl_v2_explainability_v1"
MAIN = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
FEATURE_MANIFEST = ROOT / "runs" / "tabiclv2_pilot" / "01_strict_feature_manifest.csv"
TAXONOMY = ROOT / "runs" / "multisource_ablation_v1" / "00_feature_source_taxonomy.csv"
ADAPTER = ROOT / "models" / "tabiclv2_adapter.py"
SEED = 42

# 运行前固定；禁止根据结果放宽。
MAX_ABS_TOL = 1e-6
MEAN_ABS_TOL = 1e-7
METRIC_ABS_TOL = 1e-6

TASKS = {
    "chla_surface": {
        "target": "chla_ug_l", "depth": "surface",
        "manifest": ROOT / "runs/chla_family_v2/03_chla_surface_manifest.csv",
        "prediction": ROOT / "runs/chla_family_v2/predictions/chla_surface_tabiclv2_{fold}.csv",
        "role": "role", "split_version": "CHLA_FAMILY_V2_UNIFIED_STATION_SPLIT",
    },
    "secchi": {
        "target": "secchi_depth_m", "depth": None,
        "manifest": ROOT / "runs/modern_baseline_gate_v1/01_secchi_manifest.csv",
        "prediction": ROOT / "runs/modern_baseline_gate_v1/predictions/secchi_tabiclv2_recovered.csv",
        "role": "role", "split_version": "original_secchi_station_holdout_obs_id",
    },
    "tn_surface": {
        "target": "tn_mg_l", "depth": "surface",
        "manifest": ROOT / "runs/tn_family_v2/03_tn_surface_manifest.csv",
        "prediction": ROOT / "runs/tn_family_v2/predictions/tn_surface_tabiclv2_{fold}.csv",
        "role": "role", "split_version": "TN_FAMILY_V2_UNIFIED_STATION_SPLIT",
    },
    "tp_middle": {
        "target": "tp_mg_l", "depth": "middle",
        "manifest": ROOT / "runs/tp_family_v2/04_tp_middle_manifest.csv",
        "prediction": ROOT / "runs/tp_family_v2/predictions/tp_middle_tabiclv2_{fold}.csv",
        "role": "role", "split_version": "TP_FAMILY_V2_UNIFIED_STATION_SPLIT",
    },
}


def sha256_text(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def append_row(path: Path, row: dict[str, object], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        for column in keys:
            old[column] = old[column].astype(str)
            new[column] = new[column].astype(str)
        mask = np.ones(len(old), dtype=bool)
        for column in keys:
            mask &= old[column].eq(str(new.iloc[0][column])).to_numpy()
        old = old.loc[~mask]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")


def load_adapter():
    spec = importlib.util.spec_from_file_location("tabicl_explanation_adapter", ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载适配器: {ADAPTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_features() -> list[str]:
    manifest = pd.read_csv(FEATURE_MANIFEST)
    features = manifest.loc[manifest["included_in_pilot"].astype(bool), "feature"].astype(str).tolist()
    if len(features) != 77 or len(set(features)) != 77:
        raise RuntimeError(f"冻结 strict features 数量/唯一性错误: {len(features)}")
    return features


def load_fold(task: str, fold: str, features: list[str]):
    spec = TASKS[task]
    columns = ["obs_id", "station", "date", spec["target"], *features]
    if spec["depth"] is not None:
        columns.append("depth")
    data = pd.read_csv(MAIN, usecols=list(dict.fromkeys(columns)), dtype={"obs_id": "string", "station": "string"}, low_memory=False)
    data["obs_id"] = data["obs_id"].astype("string").str.strip()
    data["station_id"] = data["station"].astype("string").str.strip()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if spec["depth"] is not None:
        data = data.loc[data["depth"].astype(str).str.lower().eq(spec["depth"])].copy()
    data[spec["target"]] = pd.to_numeric(data[spec["target"]], errors="coerce")
    data = data.loc[data[spec["target"]].notna()].copy()
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)

    manifest = pd.read_csv(spec["manifest"], dtype={"obs_id": "string"})
    manifest["obs_id"] = manifest["obs_id"].astype("string").str.strip()
    manifest = manifest.loc[manifest["fold"].astype(str).eq(fold)].copy()
    role_column = spec["role"]
    train_order = manifest.loc[manifest[role_column].astype(str).eq("train"), "obs_id"].astype(str).tolist()
    test_order = manifest.loc[manifest[role_column].astype(str).eq("test"), "obs_id"].astype(str).tolist()
    if task == "secchi":
        # Secchi 恢复脚本显式按 manifest 排序；Family V2 原脚本保持正式主表顺序。
        index = data.set_index(data["obs_id"].astype(str), drop=False)
        train = index.loc[train_order].reset_index(drop=True)
        test = index.loc[test_order].reset_index(drop=True)
    else:
        train_ids, test_ids = set(train_order), set(test_order)
        train = data.loc[data["obs_id"].astype(str).isin(train_ids)].copy()
        test = data.loc[data["obs_id"].astype(str).isin(test_ids)].copy()
    if len(train) != len(train_order) or len(test) != len(test_order):
        raise RuntimeError(f"{task}/{fold}: obs_id 无法完整匹配正式主表")
    return train, test


def frozen_prediction(task: str, fold: str) -> pd.DataFrame:
    path = Path(str(TASKS[task]["prediction"]).format(fold=fold))
    frame = pd.read_csv(path, dtype={"obs_id": "string"})
    if task == "secchi":
        frame = frame.loc[frame["fold"].astype(str).eq(fold)].copy()
    frame["obs_id"] = frame["obs_id"].astype("string").str.strip()
    return frame.reset_index(drop=True)


def prepare(task: str, fold: str):
    features = load_features()
    train, test = load_fold(task, fold, features)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(train[features].to_numpy(float)).astype(np.float32)
    x_test = imputer.transform(test[features].to_numpy(float)).astype(np.float32)
    target = TASKS[task]["target"]
    y_train_original = train[target].to_numpy(float)
    y_train = np.log1p(y_train_original) if np.nanmin(y_train_original) >= 0 else y_train_original.copy()
    inverse = np.expm1 if np.nanmin(y_train_original) >= 0 else (lambda x: x)
    model = load_adapter().TabICLv2Regressor()
    model.fit(x_train, y_train)

    def predict_final(x: np.ndarray) -> np.ndarray:
        return np.maximum(inverse(model.predict(np.asarray(x, dtype=np.float32))), 0)

    return features, train, test, imputer, x_train, x_test, predict_final


def metric_values(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"R2": float(r2_score(y, pred)), "RMSE": float(mean_squared_error(y, pred) ** 0.5), "MAE": float(mean_absolute_error(y, pred))}


def reproduce(task: str, fold: str) -> str:
    started = time.perf_counter()
    features, train, test, _, _, x_test, predict = prepare(task, fold)
    pred = predict(x_test)
    frozen = frozen_prediction(task, fold)
    obs_current = test["obs_id"].astype(str).tolist()
    obs_frozen = frozen["obs_id"].astype(str).tolist()
    set_match = set(obs_current) == set(obs_frozen)
    order_match = obs_current == obs_frozen
    aligned = pd.DataFrame({"obs_id": obs_current, "reproduced_y_pred": pred}).merge(
        frozen[["obs_id", "y_true", "y_pred"]].rename(columns={"y_pred": "frozen_y_pred"}), on="obs_id", how="outer", validate="one_to_one"
    )
    if aligned[["reproduced_y_pred", "frozen_y_pred"]].isna().any().any():
        status = "BLOCKED_INFERENCE_REPRODUCTION"
        max_abs = mean_abs = float("inf")
    else:
        delta = (aligned["reproduced_y_pred"] - aligned["frozen_y_pred"]).abs()
        max_abs, mean_abs = float(delta.max()), float(delta.mean())
        current_metrics = metric_values(aligned["y_true"].to_numpy(float), aligned["reproduced_y_pred"].to_numpy(float))
        frozen_metrics = metric_values(aligned["y_true"].to_numpy(float), aligned["frozen_y_pred"].to_numpy(float))
        metric_ok = all(abs(current_metrics[k] - frozen_metrics[k]) <= METRIC_ABS_TOL for k in current_metrics)
        status = "EXACT_REPRODUCTION" if set_match and order_match and max_abs <= MAX_ABS_TOL and mean_abs <= MEAN_ABS_TOL and metric_ok else "BLOCKED_INFERENCE_REPRODUCTION"
    current_metrics = metric_values(aligned["y_true"].to_numpy(float), aligned["reproduced_y_pred"].to_numpy(float)) if np.isfinite(max_abs) else {k: np.nan for k in ("R2", "RMSE", "MAE")}
    frozen_metrics = metric_values(aligned["y_true"].to_numpy(float), aligned["frozen_y_pred"].to_numpy(float)) if np.isfinite(max_abs) else {k: np.nan for k in ("R2", "RMSE", "MAE")}
    aligned["abs_error"] = (aligned["reproduced_y_pred"] - aligned["frozen_y_pred"]).abs()
    out = RUN / "00_preflight" / task / fold
    out.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(out / "prediction_reproduction.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature_order": range(1, 78), "feature": features}).to_csv(out / "feature_order.csv", index=False, encoding="utf-8-sig")
    row = {
        "task": task, "fold": fold, "status": status, "n_train": len(train), "n_test": len(test),
        "obs_id_set_match": set_match, "obs_id_order_match": order_match,
        "train_obs_hash": sha256_text(train["obs_id"].astype(str).tolist()), "test_obs_hash": sha256_text(obs_current),
        "max_abs_error": max_abs, "mean_abs_error": mean_abs,
        "R2_reproduced": current_metrics["R2"], "R2_frozen": frozen_metrics["R2"], "R2_abs_diff": abs(current_metrics["R2"] - frozen_metrics["R2"]),
        "RMSE_reproduced": current_metrics["RMSE"], "RMSE_frozen": frozen_metrics["RMSE"], "RMSE_abs_diff": abs(current_metrics["RMSE"] - frozen_metrics["RMSE"]),
        "MAE_reproduced": current_metrics["MAE"], "MAE_frozen": frozen_metrics["MAE"], "MAE_abs_diff": abs(current_metrics["MAE"] - frozen_metrics["MAE"]),
        "max_abs_tolerance": MAX_ABS_TOL, "mean_abs_tolerance": MEAN_ABS_TOL, "metric_abs_tolerance": METRIC_ABS_TOL,
        "target_transform": "log1p/expm1", "nonnegative_clip": True, "input_dtype": "float32", "seed": SEED,
        "context_rule": "全部逐折训练样本；无外部上下文抽样", "elapsed_seconds": time.perf_counter() - started,
    }
    append_row(RUN / "01_tabicl_inference_reproduction.csv", row, ["task", "fold"])
    append_row(RUN / "registry/run_registry.csv", {"task": task, "fold": fold, "analysis_type": "reproduction", "feature_or_pair": "ALL", "status": status, "output_path": str(out), "elapsed_seconds": row["elapsed_seconds"], "reason": "predefined strict tolerances"}, ["task", "fold", "analysis_type", "feature_or_pair"])
    return status


def predict_chunked(predict, matrix: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    return np.concatenate([predict(matrix[i:i + chunk_size]) for i in range(0, len(matrix), chunk_size)])


def stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """按站点近似均衡抽样，返回位置索引。"""
    rng = np.random.default_rng(seed)
    groups = {str(k): v.to_numpy() for k, v in frame.groupby("station_id", sort=True).groups.items()}
    chosen: list[int] = []
    while len(chosen) < min(n, len(frame)):
        progressed = False
        for station in sorted(groups):
            remaining = np.setdiff1d(groups[station], np.asarray(chosen, dtype=int), assume_unique=False)
            if len(remaining):
                chosen.append(int(rng.choice(remaining)))
                progressed = True
                if len(chosen) >= min(n, len(frame)):
                    break
        if not progressed:
            break
    return np.asarray(chosen, dtype=int)


def permutation_importance(task: str, fold: str, sample_n: int = 64) -> tuple[list[str], dict[str, object]]:
    started = time.perf_counter()
    features, train, test, _, x_train, x_test, predict = prepare(task, fold)
    taxonomy = pd.read_csv(TAXONOMY).set_index("feature_name")
    idx = stratified_sample(test.reset_index(drop=True), sample_n, SEED + int(fold[-1]))
    x_eval = x_test[idx]
    y_eval = test.iloc[idx][TASKS[task]["target"]].to_numpy(float)
    baseline_started = time.perf_counter()
    baseline = predict(x_eval)
    baseline_seconds = time.perf_counter() - baseline_started
    baseline_rmse = mean_squared_error(y_eval, baseline) ** 0.5
    baseline_mae = mean_absolute_error(y_eval, baseline)
    rng = np.random.default_rng(SEED + int(fold[-1]))
    variants: list[np.ndarray] = []
    labels: list[tuple[str, str, list[int]]] = []
    source_groups = [g for g in taxonomy["source_group"].dropna().unique().tolist()]
    for group in source_groups:
        cols = [features.index(f) for f in features if f in taxonomy.index and taxonomy.loc[f, "source_group"] == group]
        if not cols:
            continue
        altered = x_eval.copy(); order = rng.permutation(len(altered)); altered[:, cols] = altered[order][:, cols]
        variants.append(altered); labels.append(("source_group", group, cols))
    for j, feature in enumerate(features):
        altered = x_eval.copy(); altered[:, j] = altered[rng.permutation(len(altered)), j]
        variants.append(altered); labels.append(("feature", feature, [j]))
    query = np.vstack(variants)
    query_started = time.perf_counter()
    predicted = predict_chunked(predict, query)
    query_seconds = time.perf_counter() - query_started
    rows = []
    for k, (level, name, cols) in enumerate(labels):
        current = predicted[k * len(x_eval):(k + 1) * len(x_eval)]
        rmse = mean_squared_error(y_eval, current) ** 0.5
        mae = mean_absolute_error(y_eval, current)
        group = name if level == "source_group" else (taxonomy.loc[name, "source_group"] if name in taxonomy.index else "UNKNOWN")
        rows.append({"task": task, "fold": fold, "importance_level": level, "feature_or_group": name, "source_group": group,
                     "n_columns": len(cols), "n_eval": len(x_eval), "n_stations": int(test.iloc[idx]["station_id"].nunique()),
                     "baseline_RMSE": baseline_rmse, "permuted_RMSE": rmse, "delta_RMSE": rmse - baseline_rmse,
                     "baseline_MAE": baseline_mae, "permuted_MAE": mae, "delta_MAE": mae - baseline_mae,
                     "seed": SEED + int(fold[-1]), "interpretation": "留站测试标签上的事后模型依赖诊断；不用于模型选择"})
    out = pd.DataFrame(rows)
    path = RUN / "01_importance" / task / f"{fold}_permutation_importance.csv"
    path.parent.mkdir(parents=True, exist_ok=True); out.to_csv(path, index=False, encoding="utf-8-sig")
    feature_rows = out.loc[out["importance_level"].eq("feature")].copy()
    feature_rows["rank_RMSE"] = feature_rows["delta_RMSE"].rank(method="min", ascending=False).astype(int)
    feature_rows["rank_MAE"] = feature_rows["delta_MAE"].rank(method="min", ascending=False).astype(int)
    existing = RUN / "03_tabicl_feature_importance_summary.csv"
    if existing.exists():
        old = pd.read_csv(existing); old = old.loc[~(old["task"].eq(task) & old["fold"].eq(fold))]; feature_rows = pd.concat([old, feature_rows], ignore_index=True)
    feature_rows.to_csv(existing, index=False, encoding="utf-8-sig")
    eligible = out.loc[out["importance_level"].eq("feature") & ~out["source_group"].isin(["SPATIAL_STATIC", "TEMPORAL_PROCESS"])].sort_values(["delta_RMSE", "delta_MAE"], ascending=False)
    selected = eligible.head(4)["feature_or_group"].tolist()
    benchmark = {"task": task, "fold": fold, "analysis": "permutation_importance", "n_eval": len(x_eval), "n_model_calls_equivalent": len(labels),
                 "baseline_seconds": baseline_seconds, "perturbation_seconds": query_seconds, "elapsed_seconds": time.perf_counter() - started,
                 "query_rows": len(query), "chunk_size": 2048, "status": "OK"}
    append_row(RUN / "02_tabicl_explainability_runtime_benchmark.csv", benchmark, ["task", "fold", "analysis"])
    append_row(RUN / "registry/run_registry.csv", {"task": task, "fold": fold, "analysis_type": "permutation_importance", "feature_or_pair": "77_FEATURES_AND_6_GROUPS", "status": "OK", "output_path": str(path), "elapsed_seconds": benchmark["elapsed_seconds"], "reason": "post-hoc station-holdout diagnostic"}, ["task", "fold", "analysis_type", "feature_or_pair"])
    return selected, benchmark


def ale_1d(task: str, fold: str, selected: list[str], sample_n: int = 96, bins: int = 10) -> None:
    features, train, _, _, x_train, _, predict = prepare(task, fold)
    idx = stratified_sample(train.reset_index(drop=True), sample_n, 1000 + SEED + int(fold[-1]))
    x_ref = x_train[idx]; station = train.iloc[idx]["station_id"].astype(str).to_numpy()
    all_rows = []
    for feature in selected:
        started = time.perf_counter(); j = features.index(feature); values = x_train[:, j]
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if len(edges) < 4:
            continue
        bin_id = np.clip(np.digitize(x_ref[:, j], edges[1:-1], right=True), 0, len(edges) - 2)
        lo = x_ref.copy(); hi = x_ref.copy(); lo[:, j] = edges[bin_id]; hi[:, j] = edges[bin_id + 1]
        pred = predict_chunked(predict, np.vstack([lo, hi])); effect = pred[len(lo):] - pred[:len(lo)]
        local = np.array([np.mean(effect[bin_id == b]) if np.any(bin_id == b) else np.nan for b in range(len(edges) - 1)])
        counts = np.array([(bin_id == b).sum() for b in range(len(edges) - 1)])
        stations = np.array([len(set(station[bin_id == b])) for b in range(len(edges) - 1)])
        accumulated = np.cumsum(np.nan_to_num(local)); center = np.average(accumulated, weights=np.maximum(counts, 1)); accumulated -= center
        for b in range(len(local)):
            all_rows.append({"task": task, "fold": fold, "feature": feature, "bin": b + 1, "lower": edges[b], "upper": edges[b + 1],
                             "center": (edges[b] + edges[b + 1]) / 2, "local_effect": local[b], "ale": accumulated[b],
                             "n_samples": counts[b], "n_stations": stations[b], "reference": "fold training distribution", "prediction_scale": "final original target scale"})
        append_row(RUN / "registry/run_registry.csv", {"task": task, "fold": fold, "analysis_type": "ale_1d", "feature_or_pair": feature, "status": "OK", "output_path": str(RUN / "02_ale_1d" / task), "elapsed_seconds": time.perf_counter() - started, "reason": f"{len(edges)-1} quantile bins"}, ["task", "fold", "analysis_type", "feature_or_pair"])
    frame = pd.DataFrame(all_rows); path = RUN / "02_ale_1d" / task / f"{fold}_ale_1d.csv"; path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False, encoding="utf-8-sig")
    summary = frame.groupby(["task", "fold", "feature"], as_index=False).agg(ale_min=("ale", "min"), ale_max=("ale", "max"), n_bins=("bin", "count"), min_bin_samples=("n_samples", "min"), min_bin_stations=("n_stations", "min"))
    summary["ale_range"] = summary["ale_max"] - summary["ale_min"]
    existing = RUN / "04_tabicl_1d_ale_summary.csv"
    if existing.exists():
        old = pd.read_csv(existing); old = old.loc[~(old["task"].eq(task) & old["fold"].eq(fold))]; summary = pd.concat([old, summary], ignore_index=True)
    summary.to_csv(existing, index=False, encoding="utf-8-sig")


def ale_2d(task: str, fold: str, pair: tuple[str, str], sample_n: int = 64, bins: int = 5) -> None:
    features, train, _, _, x_train, _, predict = prepare(task, fold)
    idx = stratified_sample(train.reset_index(drop=True), sample_n, 2000 + SEED + int(fold[-1])); x = x_train[idx]
    a, b = pair; ia, ib = features.index(a), features.index(b)
    ea = np.unique(np.quantile(x_train[:, ia], np.linspace(0, 1, bins + 1))); eb = np.unique(np.quantile(x_train[:, ib], np.linspace(0, 1, bins + 1)))
    ca = np.clip(np.digitize(x[:, ia], ea[1:-1], right=True), 0, len(ea)-2); cb = np.clip(np.digitize(x[:, ib], eb[1:-1], right=True), 0, len(eb)-2)
    variants=[]
    for va, vb in [(0,0),(1,0),(0,1),(1,1)]:
        z=x.copy(); z[:,ia]=ea[ca+va]; z[:,ib]=eb[cb+vb]; variants.append(z)
    pred=predict_chunked(predict,np.vstack(variants)); n=len(x); local=pred[3*n:]-pred[n:2*n]-pred[2*n:3*n]+pred[:n]
    grid=np.full((len(ea)-1,len(eb)-1),np.nan); counts=np.zeros_like(grid,int); station_counts=np.zeros_like(grid,int); stations=train.iloc[idx]["station_id"].astype(str).to_numpy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            mask=(ca==i)&(cb==j); counts[i,j]=mask.sum(); station_counts[i,j]=len(set(stations[mask]));
            if mask.any(): grid[i,j]=local[mask].mean()
    accumulated=np.cumsum(np.cumsum(np.nan_to_num(grid),axis=0),axis=1)
    weights=np.maximum(counts,0); total=max(weights.sum(),1)
    row_w=weights.sum(axis=1); col_w=weights.sum(axis=0)
    row_mean=np.array([np.average(accumulated[i,:],weights=np.maximum(col_w,1)) for i in range(accumulated.shape[0])])
    col_mean=np.array([np.average(accumulated[:,j],weights=np.maximum(row_w,1)) for j in range(accumulated.shape[1])])
    grand=np.sum(accumulated*weights)/total; interaction=accumulated-row_mean[:,None]-col_mean[None,:]+grand
    rows=[]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]): rows.append({"task":task,"fold":fold,"feature_1":a,"feature_2":b,"bin_1":i+1,"bin_2":j+1,"lower_1":ea[i],"upper_1":ea[i+1],"lower_2":eb[j],"upper_2":eb[j+1],"local_second_difference":grid[i,j],"ale_2d_interaction":interaction[i,j],"n_samples":counts[i,j],"n_stations":station_counts[i,j],"sparse_mask":bool(counts[i,j]<5 or station_counts[i,j]<2)})
    frame=pd.DataFrame(rows); path=RUN/"03_ale_2d"/task/f"{fold}_{a}__{b}.csv"; path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False,encoding="utf-8-sig")
    summary={"task":task,"fold":fold,"feature_1":a,"feature_2":b,"grid":f"{grid.shape[0]}x{grid.shape[1]}","interaction_min":np.nanmin(interaction),"interaction_max":np.nanmax(interaction),"interaction_range":np.nanmax(interaction)-np.nanmin(interaction),"supported_cells":int((~frame.sparse_mask).sum()),"total_cells":len(frame),"support_fraction":float((~frame.sparse_mask).mean()),"status":"SUPPORTED" if (~frame.sparse_mask).mean()>=0.6 else "PARTIALLY_SUPPORTED"}
    append_row(RUN/"05_tabicl_2d_ale_summary.csv",summary,["task","fold","feature_1","feature_2"])
    append_row(RUN / "registry/run_registry.csv", {"task": task, "fold": fold, "analysis_type": "ale_2d", "feature_or_pair": f"{a}__{b}", "status": summary["status"], "output_path": str(path), "elapsed_seconds": np.nan, "reason": "true second-order finite differences with main-effect centering"}, ["task", "fold", "analysis_type", "feature_or_pair"])


def explain(task: str, fold: str) -> None:
    validation = pd.read_csv(RUN / "01_tabicl_inference_reproduction.csv")
    match = validation.loc[validation["task"].eq(task) & validation["fold"].eq(fold)]
    if match.empty or match.iloc[-1]["status"] != "EXACT_REPRODUCTION":
        raise RuntimeError(f"{task}/{fold} 未通过严格推理复现")
    selected, _ = permutation_importance(task, fold)
    selected_frame = pd.DataFrame({"task": task, "fold": fold, "feature": selected, "selection_rule": "TabICLv2 delta_RMSE 前4，排除纯空间和纯时间变量"})
    path = RUN / "01_importance" / task / f"{fold}_selected_ale_features.csv"; path.parent.mkdir(parents=True,exist_ok=True); selected_frame.to_csv(path,index=False,encoding="utf-8-sig")
    ale_1d(task, fold, selected)
    if len(selected) >= 2:
        ale_2d(task, fold, (selected[0], selected[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["reproduce", "explain"], required=True)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--fold", choices=[f"fold_{i}" for i in range(1, 6)], required=True)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": sys.version, "platform": platform.platform(), "executable": sys.executable,
        "seed": SEED, "pid": os.getpid(), "tolerances": {"max_abs": MAX_ABS_TOL, "mean_abs": MEAN_ABS_TOL, "metric_abs": METRIC_ABS_TOL},
    }
    (RUN / "00_preflight").mkdir(parents=True, exist_ok=True)
    (RUN / "00_preflight/environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.stage == "reproduce":
        status = reproduce(args.task, args.fold)
        print(f"{args.task}/{args.fold}: {status}", flush=True)
        return 0 if status == "EXACT_REPRODUCTION" else 2
    explain(args.task, args.fold)
    print(f"{args.task}/{args.fold}: EXPLANATION_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
