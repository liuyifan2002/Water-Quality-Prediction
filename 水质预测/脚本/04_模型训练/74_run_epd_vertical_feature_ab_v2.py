"""EPD vertical feature ablation on frozen Random Strict splits."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/epd_vertical_feature_ab_v2_20261126"
TABLE = ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv"
CAND = ROOT / "数据/处理后数据/衍生版本/EPD原位垂向候选_v1_20261126/epd_vertical_candidate_features_2014_2024.csv"
SPLIT = ROOT / "runs/random_10task_v1/01_random_split_manifest.csv"
STRICT = ROOT / "runs/tabiclv2_pilot/01_strict_feature_manifest.csv"
PARAMS = ROOT / "结果/扩展实验/候选模型稳健性验证_20260914/portable_package/config.json"
TASKS = {
    "chla_surface": ("chla_ug_l", "surface"), "chla_middle": ("chla_ug_l", "middle"),
    "chla_bottom": ("chla_ug_l", "bottom"), "secchi": ("secchi_depth_m", "surface"),
    "tn_surface": ("tn_mg_l", "surface"), "tn_middle": ("tn_mg_l", "middle"),
    "tn_bottom": ("tn_mg_l", "bottom"), "tp_surface": ("tp_mg_l", "surface"),
    "tp_middle": ("tp_mg_l", "middle"), "tp_bottom": ("tp_mg_l", "bottom"),
}
BASE = pd.read_csv(STRICT).query("included_in_pilot == True").feature.tolist()
VERTICAL = [f"epd_{v}_{d}" for v in ("temperature_c", "salinity_psu", "do_mg_l", "turbidity_ntu")
            for d in ("surface", "middle", "bottom")]
DIFF = ["epd_surface_minus_bottom_temperature_c", "epd_surface_minus_bottom_salinity_psu",
        "epd_surface_minus_bottom_do_mg_l", "epd_density0_surface_minus_bottom_kg_m3"]
SETS = {"M0": BASE, "M1": BASE + DIFF, "M2": BASE + VERTICAL, "M3": BASE + VERTICAL + DIFF}
COHORTS = {"paired_SB": ["M0", "M1"], "complete_SMB": ["M0", "M1", "M2", "M3"]}


def prepare():
    if len(BASE) != 77 or "water_zone_code" in BASE:
        raise ValueError("77项冻结特征异常")
    c = pd.read_csv(CAND, low_memory=False)
    s = pd.read_csv(SPLIT, usecols=["task", "obs_id", "split"], dtype={"obs_id": str})
    t = pd.read_csv(TABLE, usecols=["obs_id", "sample_id", "station", "depth", *set(v[0] for v in TASKS.values()), *BASE], low_memory=False)
    if c.obs_id.duplicated().any() or t.obs_id.duplicated().any() or s.duplicated(["task", "obs_id"]).any():
        raise ValueError("obs_id或划分重复")
    d = t.merge(c[["obs_id", "sample_id", "station", "depth", *VERTICAL, *DIFF]], on="obs_id",
                how="left", validate="one_to_one", suffixes=("", "_candidate"))
    if len(d) != len(t) or d[["sample_id_candidate", "station_candidate", "depth_candidate"]].isna().any().any():
        raise ValueError("候选表连接不完整")
    for col in ("sample_id", "station", "depth"):
        if not d[col].astype(str).eq(d[col + "_candidate"].astype(str)).all():
            raise ValueError(f"候选表身份冲突: {col}")
    d.drop(columns=["sample_id_candidate", "station_candidate", "depth_candidate"], inplace=True)
    for col in set(BASE + VERTICAL + DIFF + [v[0] for v in TASKS.values()]):
        d[col] = pd.to_numeric(d[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return d, s


def run_one(task, cohort, feature_set, model_name, d, s):
    target, depth = TASKS[task]
    q = d.loc[d.depth.eq(depth) & d[target].notna()].copy()
    if cohort == "paired_SB":
        q = q.loc[q[DIFF].notna().all(axis=1)]
    else:
        q = q.loc[q[VERTICAL + DIFF].notna().all(axis=1)]
    q = q.merge(s.loc[s.task.eq(task), ["obs_id", "split"]], on="obs_id", how="left", validate="one_to_one")
    if q.split.isna().any() or not q.split.isin(["train", "test"]).all():
        raise ValueError(f"{task}: 冻结划分缺失")
    q.sort_values("obs_id", inplace=True)
    tr = q.split.eq("train").to_numpy(); te = ~tr
    features = SETS[feature_set]
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    a = imp.fit_transform(q.loc[tr, features]).astype(np.float32)
    b = imp.transform(q.loc[te, features]).astype(np.float32)
    ytr = q.loc[tr, target].to_numpy(float)
    yte = q.loc[te, target].to_numpy(float)
    log_target = bool(np.min(ytr) >= 0)
    ztr = np.log1p(ytr) if log_target else ytr
    start = time.time()
    if model_name == "XGBoost":
        params = json.loads(PARAMS.read_text(encoding="utf-8-sig"))["model_parameters"]["xgboost"]
        model = XGBRegressor(**params, random_state=42, objective="reg:squarederror")
        model.fit(a, ztr)
        raw = np.asarray(model.predict(b), dtype=float)
    else:
        from models.tabiclv2_adapter import TabICLv2Regressor
        model = TabICLv2Regressor()
        model.fit(a, ztr)
        raw = np.asarray(model.predict(b), dtype=float)
    pred = np.maximum(np.expm1(raw) if log_target else raw, 0)
    if not np.isfinite(pred).all():
        raise ValueError(f"{task}/{cohort}/{feature_set}/{model_name}: 非有限预测")
    result = q.loc[te, ["obs_id", "sample_id", "station"]].rename(columns={"station": "station_id"}).copy()
    result["task"] = task
    result["cohort"] = cohort
    result["feature_set"] = feature_set
    result["model"] = model_name
    result["fold"] = "seed_42"
    result["y_true"] = yte
    result["y_pred"] = pred
    metric = {"task": task, "cohort": cohort, "feature_set": feature_set, "model": model_name,
              "n_features": len(features), "n_train": int(tr.sum()), "n_test": int(te.sum()),
              "n_stations": int(q.station.nunique()), "n_test_stations": int(q.loc[te, "station"].nunique()),
              "R2": r2_score(yte, pred), "RMSE": mean_squared_error(yte, pred) ** .5,
              "MAE": mean_absolute_error(yte, pred), "runtime_s": time.time() - start}
    return result, metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["XGBoost", "TabICLv2", "all"], default="all")
    p.add_argument("--task", choices=[*TASKS, "all"], default="all")
    a = p.parse_args()
    out = OUT / "individual_predictions"
    out.mkdir(parents=True, exist_ok=True)
    d, s = prepare()
    models = ["XGBoost", "TabICLv2"] if a.model == "all" else [a.model]
    tasks = TASKS if a.task == "all" else [a.task]
    for task in tasks:
        for cohort, feature_sets in COHORTS.items():
            for feature_set in feature_sets:
                for model_name in models:
                    stem = f"{task}__{cohort}__{feature_set}__{model_name}"
                    file = out / f"{stem}.parquet"
                    meta = out / f"{stem}.json"
                    if file.exists() and meta.exists():
                        print("SKIP", stem, flush=True)
                        continue
                    pred, metric = run_one(task, cohort, feature_set, model_name, d, s)
                    pred.to_parquet(file, index=False)
                    meta.write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding="utf-8")
                    print("DONE", stem, round(metric["R2"], 4), round(metric["runtime_s"], 1), flush=True)


if __name__ == "__main__":
    main()
