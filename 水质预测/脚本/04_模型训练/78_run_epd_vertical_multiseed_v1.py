"""EPD-Vertical V3 stage A: multi-seed TabICLv2 confirmation on the frozen Random Strict split.

Reuses the V2 sample construction and feature sets; only the model random_state changes.
seed=42 is NOT re-run here (V2 predictions are reused by the summarizer).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUNNER = SourceFileLoader("v2_runner_78", str(ROOT / "脚本/04_模型训练/74_run_epd_vertical_feature_ab_v2.py")).load_module()

OUT = ROOT / "runs/epd_vertical_confirmation_v1_20261127"
IND = OUT / "random_multiseed_predictions"
V2 = ROOT / "runs/epd_vertical_feature_ab_v2_20261126/individual_predictions"
SEEDS = (2026, 3407)
SETS = ("M0", "M2")
COHORT = "complete_SMB"


def select(task, feature_set, d, s):
    """Identical row selection to V2.run_one for the complete_SMB cohort."""
    target, depth = RUNNER.TASKS[task]
    q = d.loc[d.depth.eq(depth) & d[target].notna()].copy()
    q = q.loc[q[RUNNER.VERTICAL + RUNNER.DIFF].notna().all(axis=1)]
    q = q.merge(s.loc[s.task.eq(task), ["obs_id", "split"]], on="obs_id", how="left", validate="one_to_one")
    if q.split.isna().any():
        raise ValueError(f"{task}: missing frozen split")
    q.sort_values("obs_id", inplace=True)
    q.attrs["target"] = target
    return q


def run_seeded(task, feature_set, seed, d, s):
    q = select(task, feature_set, d, s)
    target = q.attrs["target"]
    tr = q.split.eq("train").to_numpy()
    te = ~tr
    feats = RUNNER.SETS[feature_set]
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    a = imp.fit_transform(q.loc[tr, feats]).astype(np.float32)
    b = imp.transform(q.loc[te, feats]).astype(np.float32)
    ytr = q.loc[tr, target].to_numpy(float)
    yte = q.loc[te, target].to_numpy(float)
    log_target = bool(np.min(ytr) >= 0)
    ztr = np.log1p(ytr) if log_target else ytr
    start = time.time()
    from models.tabiclv2_adapter import TabICLv2Regressor
    model = TabICLv2Regressor(random_state=seed)
    model.fit(a, ztr)
    raw = np.asarray(model.predict(b), dtype=float)
    pred = np.maximum(np.expm1(raw) if log_target else raw, 0)
    if not np.isfinite(pred).all():
        raise ValueError(f"{task}/{feature_set}/seed{seed}: non-finite prediction")
    result = q.loc[te, ["obs_id", "sample_id", "station"]].rename(columns={"station": "station_id"}).copy()
    result["task"] = task
    result["cohort"] = COHORT
    result["feature_set"] = feature_set
    result["model"] = "TabICLv2"
    result["model_seed"] = seed
    result["fold"] = "seed_42_split"
    result["y_true"] = yte
    result["y_pred"] = pred
    metric = {"task": task, "cohort": COHORT, "feature_set": feature_set, "model": "TabICLv2",
              "model_seed": seed, "n_features": len(feats), "n_train": int(tr.sum()), "n_test": int(te.sum()),
              "n_stations": int(q.station.nunique()), "n_test_stations": int(q.loc[te, "station"].nunique()),
              "R2": r2_score(yte, pred), "RMSE": mean_squared_error(yte, pred) ** .5,
              "MAE": mean_absolute_error(yte, pred), "log_target": log_target,
              "runtime_s": time.time() - start}
    return result, metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all")
    p.add_argument("--seed", default="all")
    a = p.parse_args()
    IND.mkdir(parents=True, exist_ok=True)
    d, s = RUNNER.prepare()
    tasks = list(RUNNER.TASKS) if a.task == "all" else [a.task]
    seeds = list(SEEDS) if a.seed == "all" else [int(a.seed)]
    for task in tasks:
        for feature_set in SETS:
            q = select(task, feature_set, d, s)
            v2 = pd.read_parquet(V2 / f"{task}__{COHORT}__{feature_set}__TabICLv2.parquet").set_index("obs_id")
            if set(v2.index) != set(q.loc[q.split.eq("test"), "obs_id"]):
                raise ValueError(f"{task}/{feature_set}: test obs_id differs from V2")
            for seed in seeds:
                stem = f"{task}__{feature_set}__TabICLv2__seed{seed}"
                file = IND / f"{stem}.parquet"
                meta = IND / f"{stem}.json"
                if file.exists() and meta.exists():
                    print("SKIP", stem, flush=True)
                    continue
                pred, metric = run_seeded(task, feature_set, seed, d, s)
                pred.to_parquet(file, index=False)
                meta.write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding="utf-8")
                print("DONE", stem, round(metric["R2"], 4), round(metric["runtime_s"], 1), flush=True)


if __name__ == "__main__":
    main()
