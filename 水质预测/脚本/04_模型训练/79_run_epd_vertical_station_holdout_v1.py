"""EPD-Vertical V3 stage B: five-fold station-holdout confirmation (M0 vs M2, TabICLv2).

Fold assignment comes from frozen, already-approved assets only:
  chla/tn/tp -> runs/<family>_family_v2/0[3-5]_<task>_manifest.csv (role=test per fold)
  secchi     -> 结果/扩展实验/分海区五折留站验证_20260906/五折站点划分清单.csv (station -> fold)
No fold is generated here. Sample base is the V2 complete_SMB cohort.
"""
from __future__ import annotations

import argparse
import glob
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
RUNNER = SourceFileLoader("v2_runner_79", str(ROOT / "脚本/04_模型训练/74_run_epd_vertical_feature_ab_v2.py")).load_module()

OUT = ROOT / "runs/epd_vertical_confirmation_v1_20261127"
IND = OUT / "station_holdout_predictions"
ZONE_MAP = ROOT / "结果/扩展实验/分海区五折留站验证_20260906/五折站点划分清单.csv"
FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4", "fold_5")
SETS = ("M0", "M2")
MODEL_SEED = 42


def complete_cohort(task, d):
    target, depth = RUNNER.TASKS[task]
    q = d.loc[d.depth.eq(depth) & d[target].notna()].copy()
    q = q.loc[q[RUNNER.VERTICAL + RUNNER.DIFF].notna().all(axis=1)]
    q = q.sort_values("obs_id").reset_index(drop=True)
    if q.obs_id.duplicated().any():
        raise ValueError(f"{task}: duplicated obs_id in complete cohort")
    q.attrs["target"] = target
    return q


def fold_assignment(task, q):
    """obs_id -> fold, from frozen assets; hard error if coverage is not exact."""
    if task == "secchi":
        zone = pd.read_csv(ZONE_MAP)
        station_fold = {str(s).strip(): f"fold_{int(f)}" for s, f in zip(zone.station, zone.fold)}
        missing = sorted(set(q.station) - set(station_fold))
        if missing:
            raise ValueError(f"secchi: stations absent from zone fold map: {missing[:5]}")
        obs_fold = pd.Series([station_fold[str(s).strip()] for s in q.station], index=q.obs_id)
    else:
        fam = task.split("_")[0]
        files = sorted(glob.glob(str(ROOT / f"runs/{fam}_family_v2/0[3-5]_{task}_manifest.csv")))
        if not files:
            raise ValueError(f"{task}: no frozen station-holdout manifest found")
        man = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
        test = man[man.role.eq("test")]
        if test.obs_id.duplicated().any():
            raise ValueError(f"{task}: obs_id assigned to more than one test fold")
        obs_fold = test.set_index("obs_id").fold.reindex(q.obs_id)
        if obs_fold.isna().any():
            raise ValueError(f"{task}: {int(obs_fold.isna().sum())} complete-cohort obs_id missing from frozen manifest")
    obs_fold = obs_fold.astype(str)
    if set(obs_fold) - set(FOLDS):
        raise ValueError(f"{task}: unexpected fold labels {sorted(set(obs_fold) - set(FOLDS))}")
    return obs_fold


def run_fold(task, feature_set, fold, q, obs_fold):
    target = q.attrs["target"]
    tr = (obs_fold != fold).to_numpy()
    te = ~tr
    tr_st = set(q.loc[tr, "station"]); te_st = set(q.loc[te, "station"])
    if tr_st & te_st:
        raise ValueError(f"{task}/{fold}: station overlap {sorted(tr_st & te_st)[:5]}")
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
    model = TabICLv2Regressor(random_state=MODEL_SEED)
    model.fit(a, ztr)
    raw = np.asarray(model.predict(b), dtype=float)
    pred = np.maximum(np.expm1(raw) if log_target else raw, 0)
    if not np.isfinite(pred).all():
        raise ValueError(f"{task}/{feature_set}/{fold}: non-finite prediction")
    out = q.loc[te, ["obs_id", "sample_id", "station"]].rename(columns={"station": "station_id"}).copy()
    out["task"] = task
    out["cohort"] = "complete_SMB"
    out["feature_set"] = feature_set
    out["model"] = "TabICLv2"
    out["model_seed"] = MODEL_SEED
    out["fold"] = fold
    out["y_true"] = yte
    out["y_pred"] = pred
    metric = {"task": task, "cohort": "complete_SMB", "feature_set": feature_set, "model": "TabICLv2",
              "model_seed": MODEL_SEED, "fold": fold, "n_features": len(feats),
              "n_train": int(tr.sum()), "n_test": int(te.sum()),
              "n_train_stations": len(tr_st), "n_test_stations": len(te_st),
              "R2": r2_score(yte, pred), "RMSE": mean_squared_error(yte, pred) ** .5,
              "MAE": mean_absolute_error(yte, pred), "log_target": log_target,
              "runtime_s": time.time() - start}
    return out, metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all")
    p.add_argument("--features", default="all")
    p.add_argument("--fold", default="all")
    a = p.parse_args()
    IND.mkdir(parents=True, exist_ok=True)
    d, _ = RUNNER.prepare()
    tasks = list(RUNNER.TASKS) if a.task == "all" else [a.task]
    sets = list(SETS) if a.features == "all" else [a.features]
    folds = list(FOLDS) if a.fold == "all" else [a.fold]
    for task in tasks:
        q = complete_cohort(task, d)
        obs_fold = fold_assignment(task, q)
        print("COHORT", task, len(q), "stations", q.station.nunique(), flush=True)
        for feature_set in sets:
            for fold in folds:
                stem = f"{task}__{feature_set}__{fold}"
                file = IND / f"{stem}.parquet"
                meta = IND / f"{stem}.json"
                if file.exists() and meta.exists():
                    print("SKIP", stem, flush=True)
                    continue
                pred, metric = run_fold(task, feature_set, fold, q, obs_fold)
                pred.to_parquet(file, index=False)
                meta.write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding="utf-8")
                print("DONE", stem, round(metric["R2"], 4), round(metric["runtime_s"], 1), flush=True)


if __name__ == "__main__":
    main()
