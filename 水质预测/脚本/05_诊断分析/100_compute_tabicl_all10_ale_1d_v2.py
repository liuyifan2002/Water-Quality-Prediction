"""十任务V2 P1一维ALE：保留原分箱/查询定义，门槛仅记录不阻断计算。"""

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


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
RUNNER = ROOT / "脚本/05_诊断分析/88_run_tabicl_slim_explainability_v1.py"
TAXONOMY = ROOT / "runs/multisource_ablation_v1/00_feature_source_taxonomy.csv"
EXCLUDED_GROUPS = {"TEMPORAL_PROCESS", "SPATIAL_STATIC"}


def load_runner():
    spec = importlib.util.spec_from_file_location("tabicl_slim_runner_ale", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_core_shap(task: str, features: list[str]) -> pd.DataFrame:
    runner = load_runner()
    selected = runner.sample_rows(task)
    frames = []
    for sample in selected.itertuples():
        path = OUT / "shap" / task / f"{sample.obs_id}_seed42.csv.gz"
        if not path.is_file():
            raise RuntimeError(f"{task}: 三个预登记核心样本的77维SHAP尚未齐备")
        frame = pd.read_csv(path)
        if len(frame) != 77 or frame.feature.astype(str).tolist() != features \
                or frame.status.iloc[0] != "COMPUTED_PASS":
            raise RuntimeError(f"{task}: 核心SHAP结果不合格: {path}")
        frames.append(frame[["feature", "shap_value", "obs_id"]])
    return pd.concat(frames, ignore_index=True)


def support_for_bins(train_values, raw_test_values, stations, bins: int) -> tuple[pd.DataFrame, np.ndarray]:
    edges = np.unique(np.quantile(train_values, np.linspace(0, 1, bins + 1)))
    if len(edges) != bins + 1:
        return pd.DataFrame(), edges
    valid = np.isfinite(raw_test_values) & (raw_test_values >= edges[0]) & (raw_test_values <= edges[-1])
    membership = np.clip(np.digitize(raw_test_values, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = valid & (membership == b)
        rows.append({"bin": b + 1, "lower": float(edges[b]), "upper": float(edges[b + 1]),
            "full_query_n": int(mask.sum()), "full_query_stations": int(len(set(stations[mask]))),
            "full_query_supported": bool(mask.sum() >= 30 and len(set(stations[mask])) >= 5)})
    return pd.DataFrame(rows), edges


def stratified_indices(membership: np.ndarray, eligible: np.ndarray,
                       stations: np.ndarray, n_bins: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for b in range(n_bins):
        indices = np.flatnonzero(eligible & (membership == b))
        target_n = min(60, len(indices))
        by_station = {str(station): list(rng.permutation(indices[stations[indices] == station]))
                      for station in sorted(set(stations[indices]))}
        picked = []
        while len(picked) < target_n:
            changed = False
            for station in sorted(by_station):
                if by_station[station] and len(picked) < target_n:
                    picked.append(int(by_station[station].pop()))
                    changed = True
            if not changed:
                break
        selected.extend(picked)
    return np.asarray(selected, dtype=int)


def screen(task: str) -> list[str]:
    runner = load_runner()
    base = runner.load_base()
    features = base.load_features()
    train, test = base.load_fold(task, "fold_1", features)
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imp.fit_transform(train[features]).astype(np.float32)
    x_test = imp.transform(test[features]).astype(np.float32)
    if runner.hash_array(imp.statistics_) != runner.audit_row(task).imputer_statistics_hash:
        raise RuntimeError("审查用的imputer和冻结注册表不一致")
    shap_values = read_core_shap(task, features)
    importance = shap_values.groupby("feature").shap_value.apply(
        lambda values: float(np.mean(np.abs(values)))).sort_values(ascending=False)
    taxonomy = pd.read_csv(TAXONOMY).set_index("feature_name")
    stations = test.station_id.astype(str).to_numpy()
    rows = []
    for feature, contribution in importance.items():
        group = str(taxonomy.loc[feature, "source_group"]) if feature in taxonomy.index else "UNKNOWN"
        meaning = str(taxonomy.loc[feature, "physical_meaning"]) if feature in taxonomy.index else ""
        j = features.index(feature)
        raw = pd.to_numeric(test[feature], errors="coerce").to_numpy(float)
        values = x_train[:, j]
        reason = ""
        selected_bins = None
        coverage = None
        edges = None
        if group in EXCLUDED_GROUPS:
            reason = "空间/时间定位变量，不进入环境过程ALE候选"
        elif len(np.unique(values)) < 11 or float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5)) <= 0:
            reason = "连续取值变化不足"
        else:
            for bins in (10, 8, 6):
                cov, ed = support_for_bins(values, raw, stations, bins)
                if len(cov) and cov.full_query_supported.all():
                    selected_bins, coverage, edges = bins, cov, ed
                    break
            if selected_bins is None:
                reason = "10、8、6箱的完整留站查询特征均有n<30或独立站点<5的箱"
        rows.append({"task": task, "fold": "fold_1", "feature": feature,
            "mean_abs_shap_on_3_core_samples": contribution, "shap_rank": len(rows) + 1,
            "source_group": group, "physical_meaning": meaning,
            "continuous": len(np.unique(values)) >= 11,
            "selected_bins": selected_bins, "all_bins_supported": selected_bins is not None,
            "reason": reason, "proposed": False})
    frame = pd.DataFrame(rows)
    candidates = frame.loc[frame.all_bins_supported & ~frame.source_group.isin(EXCLUDED_GROUPS)].head(2)
    frame.loc[frame.feature.isin(candidates.feature), "proposed"] = True
    runner.atomic_csv(OUT / "ale_1d" / f"{task}_feature_screening.csv", frame)
    for feature in candidates.feature:
        j = features.index(feature)
        raw = pd.to_numeric(test[feature], errors="coerce").to_numpy(float)
        selected_bins = int(frame.loc[frame.feature.eq(feature), "selected_bins"].iloc[0])
        coverage, edges = support_for_bins(x_train[:, j], raw, stations, selected_bins)
        eligible = np.isfinite(raw) & (raw >= edges[0]) & (raw <= edges[-1])
        membership = np.clip(np.digitize(raw, edges[1:-1], right=True), 0, selected_bins - 1)
        seed = 42000 + features.index(feature) + 100 * runner.TASKS.index(task)
        selected = stratified_indices(membership, eligible, stations, selected_bins, seed)
        coverage["selected_n"] = [int(np.sum(membership[selected] == b)) for b in range(selected_bins)]
        coverage["selected_stations"] = [int(len(set(stations[selected[membership[selected] == b]])))
                                         for b in range(selected_bins)]
        coverage["selected_support"] = (coverage.selected_n >= 30) & (coverage.selected_stations >= 5)
        coverage["task"] = task
        coverage["feature"] = feature
        coverage["seed"] = seed
        runner.atomic_csv(OUT / "ale_1d" / f"{task}_{feature}_bin_support.csv", coverage)
        if not coverage.selected_support.all():
            continue
        sample_manifest = test.iloc[selected][["obs_id", "station_id", "date"]].copy()
        sample_manifest["bin"] = membership[selected] + 1
        sample_manifest["feature"] = feature
        sample_manifest["task"] = task
        sample_manifest["feature_value"] = raw[selected]
        sample_manifest["edge_hash"] = runner.hash_array(edges)
        runner.atomic_csv(OUT / "ale_1d" / f"{task}_{feature}_query_manifest.csv", sample_manifest)
    print(json.dumps({"task": task, "proposed": candidates.feature.tolist(),
        "status": "SUPPORT_SCREEN_DONE"}, ensure_ascii=False), flush=True)
    return candidates.feature.astype(str).tolist()


def calculate(task: str, feature: str) -> None:
    runner = load_runner()
    support_path = OUT / "ale_1d" / f"{task}_{feature}_bin_support.csv"
    sample_path = OUT / "ale_1d" / f"{task}_{feature}_query_manifest.csv"
    if not support_path.is_file() or not sample_path.is_file():
        raise RuntimeError("ALE分箱和分层样本未通过CPU审查")
    # 保留筛选阶段写入 CSV 的 float64 分箱边界原值，避免默认解析舍入造成哈希误报。
    support = pd.read_csv(support_path, float_precision="round_trip").sort_values("bin")
    sample = pd.read_csv(sample_path, dtype={"obs_id": "string"})
    if not support.selected_support.astype(bool).all():
        raise RuntimeError("选取样本后仍有分箱n<30或站点<5")
    base, features, train, test, x_train, x_test, adapter, inv, meta = runner.context(task)
    if feature not in features:
        raise RuntimeError("ALE变量不在77个冻结特征中")
    column = features.index(feature)
    edges = np.asarray(support.lower.astype(float).tolist() + [float(support.upper.iloc[-1])], dtype=np.float64)
    if runner.hash_array(edges) != str(sample.edge_hash.iloc[0]):
        raise RuntimeError("ALE分箱边界哈希与选样清单不一致")
    index = {str(obs): i for i, obs in enumerate(test.obs_id.astype(str))}
    database = OUT / "checkpoints" / "ale_1d_local_effects.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database)
    conn.execute("""CREATE TABLE IF NOT EXISTS local_effect (
        task TEXT, fold TEXT, feature TEXT, obs_id TEXT, station_id TEXT,
        bin INTEGER, lower REAL, upper REAL, prediction_lower REAL,
        prediction_upper REAL, local_effect REAL, query_hash TEXT,
        context_hash TEXT, edge_hash TEXT, actual_model_calls INTEGER,
        elapsed_seconds REAL, PRIMARY KEY(task,fold,feature,obs_id))""")
    real_calls = 0
    seconds = 0.0
    cache: dict[str, float] = {}

    def predict_one(row: np.ndarray, use_cache: bool = True) -> float:
        nonlocal real_calls, seconds
        key = runner.hash_array(row)
        if use_cache and key in cache:
            return cache[key]
        started = time.perf_counter()
        result = np.asarray(adapter.predict(row[None, :]), float).reshape(-1)
        seconds += time.perf_counter() - started
        real_calls += 1
        if result.size != 1 or not np.isfinite(result[0]):
            raise RuntimeError("ALE P1输出不是一个有限标量")
        value = float(max(float(inv(result)[0]), 0.0))
        if use_cache:
            cache[key] = value
        return value

    for item in sample.itertuples():
        exists = conn.execute("SELECT context_hash,edge_hash FROM local_effect WHERE task=? AND fold=? AND feature=? AND obs_id=?",
            (task, "fold_1", feature, str(item.obs_id))).fetchone()
        if exists:
            if exists != (meta["context_hash"], str(item.edge_hash)):
                raise RuntimeError("已落盘局部差值的上下文或分箱哈希不同，禁止覆盖")
            continue
        row = x_test[index[str(item.obs_id)]].copy()
        lower, upper = row.copy(), row.copy()
        b = int(item.bin) - 1
        lower[column], upper[column] = edges[b], edges[b + 1]
        started_calls, started_seconds = real_calls, seconds
        p_lower = predict_one(lower)
        p_upper = predict_one(upper)
        conn.execute("INSERT INTO local_effect VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (task, "fold_1", feature, str(item.obs_id), str(item.station_id), int(item.bin),
             float(edges[b]), float(edges[b + 1]), p_lower, p_upper, p_upper - p_lower,
             runner.hash_array(row), meta["context_hash"], str(item.edge_hash),
             real_calls - started_calls, seconds - started_seconds))
        conn.commit()
    local = pd.read_sql_query("SELECT * FROM local_effect WHERE task=? AND fold=? AND feature=?",
        conn, params=(task, "fold_1", feature))
    if len(local) != len(sample):
        conn.close()
        raise RuntimeError("ALE局部差值检查点未覆盖所有预登记查询样本")
    repeat_path = OUT / "ale_1d" / f"{task}_{feature}_repeatability.csv"
    if not repeat_path.exists():
        first = sample.iloc[0]
        row = x_test[index[str(first.obs_id)]].copy()
        b = int(first.bin) - 1
        lower, upper = row.copy(), row.copy()
        lower[column], upper[column] = edges[b], edges[b + 1]
        repeats = []
        for number in (1, 2, 3):
            p0 = predict_one(lower, use_cache=False)
            p1 = predict_one(upper, use_cache=False)
            repeats.append({**meta, "task": task, "feature": feature,
                "obs_id": str(first.obs_id), "repeat": number,
                "lower_hash": runner.hash_array(lower), "upper_hash": runner.hash_array(upper),
                "prediction_lower": p0, "prediction_upper": p1,
                "local_effect": p1 - p0, "actual_model_calls": 2})
        runner.atomic_csv(repeat_path, pd.DataFrame(repeats))
    repeat = pd.read_csv(repeat_path)
    repeat_range = float(repeat.local_effect.max() - repeat.local_effect.min())
    bins = []
    for record in support.itertuples():
        values = local.loc[local.bin.eq(record.bin)]
        bins.append({"task": task, "fold": "fold_1", "feature": feature,
            "bin": int(record.bin), "lower": record.lower, "upper": record.upper,
            "n_samples": len(values), "n_stations": values.station_id.nunique(),
            "local_effect": float(values.local_effect.mean())})
    curve = pd.DataFrame(bins)
    cumulative = curve.local_effect.cumsum()
    curve["ale"] = cumulative - np.average(cumulative, weights=curve.n_samples)
    curve["context_hash"] = meta["context_hash"]
    curve["ale_script_sha256"] = runner.file_sha(Path(__file__))
    curve["protocol"] = runner.P1
    curve["repeatability_max_local_delta"] = repeat_range
    curve["status"] = "COMPUTED_PASS" if repeat_range <= 1e-6 else "COMPUTED_WITH_NUMERICAL_LIMITATION"
    runner.atomic_csv(OUT / "ale_1d" / f"{task}_{feature}_curve.csv", curve)
    conn.close()
    print(json.dumps({"task": task, "feature": feature, "status": curve.status.iloc[0],
        "query_rows": len(local), "new_actual_model_calls": real_calls,
        "new_model_seconds": seconds, "repeat_range": repeat_range}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--calculate", action="store_true")
    parser.add_argument("--task", required=True)
    parser.add_argument("--feature")
    args = parser.parse_args()
    if args.screen:
        screen(args.task)
    elif args.calculate and args.feature:
        calculate(args.task, args.feature)
    else:
        parser.error("使用--screen或--calculate --feature")


if __name__ == "__main__":
    main()
