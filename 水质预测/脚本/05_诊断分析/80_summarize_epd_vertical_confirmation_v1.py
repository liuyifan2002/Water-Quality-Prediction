"""EPD-Vertical V3 stage C: aggregate multi-seed and station-holdout results, paired bootstrap, gate check."""
from __future__ import annotations

import glob
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
B = SourceFileLoader("boot75", str(ROOT / "脚本/05_诊断分析/75_summarize_epd_vertical_feature_ab_v2.py")).load_module()

OUT = ROOT / "runs/epd_vertical_confirmation_v1_20261127"
V2 = ROOT / "runs/epd_vertical_feature_ab_v2_20261126/individual_predictions"
MS = OUT / "random_multiseed_predictions"
SH = OUT / "station_holdout_predictions"
SEEDS = (42, 2026, 3407)
SETS = ("M0", "M2")
FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4", "fold_5")
TASKS = ["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface",
         "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom"]


def v2_pred(task, feature_set):
    df = pd.read_parquet(V2 / f"{task}__complete_SMB__{feature_set}__TabICLv2.parquet")
    return df[["obs_id", "station_id", "y_true", "y_pred"]].copy()


def ms_pred(task, feature_set, seed):
    df = pd.read_parquet(MS / f"{task}__{feature_set}__TabICLv2__seed{seed}.parquet")
    return df[["obs_id", "station_id", "y_true", "y_pred"]].copy()


def random_multiseed():
    rows = []
    for task in TASKS:
        for seed in SEEDS:
            preds = {fs: (v2_pred(task, fs) if seed == 42 else ms_pred(task, fs, seed)) for fs in SETS}
            base_obs = set(preds["M0"].obs_id)
            for fs in SETS:
                if set(preds[fs].obs_id) != base_obs:
                    raise ValueError(f"{task}/seed{seed}/{fs}: test obs_id set differs")
            for fs in SETS:
                p = preds[fs]
                meta = json.loads((V2 / f"{task}__complete_SMB__{fs}__TabICLv2.json").read_text(encoding="utf-8")) if seed == 42 \
                    else json.loads((MS / f"{task}__{fs}__TabICLv2__seed{seed}.json").read_text(encoding="utf-8"))
                rows.append(dict(seed=seed, task=task, model="TabICLv2", feature_set=fs,
                                 n_features=meta["n_features"], n_train=meta["n_train"], n_test=meta["n_test"],
                                 n_stations=meta["n_stations"], n_test_stations=meta["n_test_stations"],
                                 R2=r2_score(p.y_true, p.y_pred),
                                 RMSE=mean_squared_error(p.y_true, p.y_pred) ** .5,
                                 MAE=mean_absolute_error(p.y_true, p.y_pred)))
    m = pd.DataFrame(rows)
    d = m[m.feature_set == "M2"].merge(m[m.feature_set == "M0"], on=["seed", "task", "model"], suffixes=("", "_M0"))
    d["delta_R2"] = d.R2 - d.R2_M0
    d["delta_RMSE"] = d.RMSE - d.RMSE_M0
    d["delta_MAE"] = d.MAE - d.MAE_M0
    m = m.merge(d[["seed", "task", "model", "delta_R2", "delta_RMSE", "delta_MAE"]], on=["seed", "task", "model"], how="left")
    m.loc[m.feature_set == "M0", ["delta_R2", "delta_RMSE", "delta_MAE"]] = 0.0
    m.to_csv(OUT / "01_random_multiseed_metrics.csv", index=False, encoding="utf-8-sig")

    s = (d.groupby("task").delta_R2.agg(["mean", "std", "min", "max"]).reset_index()
         .rename(columns={"mean": "mean_delta_R2", "std": "std_delta_R2", "min": "min_delta_R2", "max": "max_delta_R2"}))
    s["seeds_positive"] = d.groupby("task").delta_R2.apply(lambda x: int((x > 0).sum())).values
    s["seeds_ge_0p02"] = d.groupby("task").delta_R2.apply(lambda x: int((x >= 0.02).sum())).values
    s = s.set_index("task").loc[TASKS].reset_index()
    macro = dict(task="MACRO", mean_delta_R2=d.delta_R2.mean(),
                 std_delta_R2=d.groupby("seed").delta_R2.mean().std(),
                 min_delta_R2=d.groupby("seed").delta_R2.mean().min(),
                 max_delta_R2=d.groupby("seed").delta_R2.mean().max(),
                 seeds_positive=int((d.groupby("seed").delta_R2.mean() > 0).sum()),
                 seeds_ge_0p02="")
    s = pd.concat([s, pd.DataFrame([macro])], ignore_index=True)
    s.to_csv(OUT / "01b_random_multiseed_task_summary.csv", index=False, encoding="utf-8-sig")
    m0_macro = m[m.feature_set == "M0"].groupby("seed").R2.mean()
    print("multi-seed macro R2 (M0):", {int(k): round(v, 4) for k, v in m0_macro.items()})
    print("multi-seed macro R2 (M2):", {int(k): round(v, 4) for k, v in m[m.feature_set == 'M2'].groupby('seed').R2.mean().items()})
    print("per-seed macro delta_R2:", {int(k): round(v, 4) for k, v in d.groupby('seed').delta_R2.mean().items()})
    return m, d, s


def station_holdout():
    rows, frames = [], []
    for task in TASKS:
        per = {}
        for fold in FOLDS:
            for fs in SETS:
                p = pd.read_parquet(SH / f"{task}__{fs}__{fold}.parquet")
                p["fold"] = fold
                per[(fs, fold)] = p
                frames.append(p[["obs_id", "task", "station_id", "sample_id", "fold", "feature_set", "model", "y_true", "y_pred"]])
        ref_obs = None
        for fs in SETS:
            union = pd.concat([per[(fs, f)] for f in FOLDS], ignore_index=True)
            if len(union) != union.obs_id.nunique():
                raise ValueError(f"{task}/{fs}: duplicated OOF obs_id")
            if ref_obs is None:
                ref_obs = set(union.obs_id)
            elif set(union.obs_id) != ref_obs:
                raise ValueError(f"{task}/{fs}: OOF coverage differs across feature sets")
        for fs in SETS:
            for fold in FOLDS:
                p = per[(fs, fold)]
                meta = json.loads((SH / f"{task}__{fs}__{fold}.json").read_text(encoding="utf-8"))
                rows.append(dict(protocol="station_holdout", task=task, model="TabICLv2", feature_set=fs, fold=fold,
                                 n_train=meta["n_train"], n_test=meta["n_test"],
                                 n_train_stations=meta["n_train_stations"], n_test_stations=meta["n_test_stations"],
                                 R2=r2_score(p.y_true, p.y_pred), RMSE=mean_squared_error(p.y_true, p.y_pred) ** .5,
                                 MAE=mean_absolute_error(p.y_true, p.y_pred)))
        for fs in SETS:
            union = pd.concat([per[(fs, f)] for f in FOLDS], ignore_index=True)
            rows.append(dict(protocol="station_holdout", task=task, model="TabICLv2", feature_set=fs, fold="pooled_OOF",
                             n_train=np.nan, n_test=len(union),
                             n_train_stations=np.nan, n_test_stations=union.station_id.nunique(),
                             R2=r2_score(union.y_true, union.y_pred),
                             RMSE=mean_squared_error(union.y_true, union.y_pred) ** .5,
                             MAE=mean_absolute_error(union.y_true, union.y_pred)))
    m = pd.DataFrame(rows)
    d = m[m.feature_set == "M2"].merge(m[m.feature_set == "M0"], on=["task", "fold"], suffixes=("", "_M0"))
    d["delta_R2"] = d.R2 - d.R2_M0
    d["delta_RMSE"] = d.RMSE - d.RMSE_M0
    d["delta_MAE"] = d.MAE - d.MAE_M0
    m = m.merge(d[["task", "fold", "delta_R2", "delta_RMSE", "delta_MAE"]], on=["task", "fold"], how="left")
    m.loc[m.feature_set == "M0", ["delta_R2", "delta_RMSE", "delta_MAE"]] = 0.0
    m.to_csv(OUT / "02_station_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    pred = pd.concat(frames, ignore_index=True).rename(columns={"y_true": "target", "y_pred": "prediction"})
    pred.to_parquet(OUT / "03_station_holdout_predictions.parquet", index=False)
    pooled = m[m.fold == "pooled_OOF"]
    print("holdout pooled macro R2 (M0):", round(pooled[pooled.feature_set == 'M0'].R2.mean(), 4),
          "| (M2):", round(pooled[pooled.feature_set == 'M2'].R2.mean(), 4))
    print("holdout pooled per-task delta_R2:", {r.task: round(r.delta_R2, 4) for r in pooled[pooled.feature_set == 'M2'].itertuples()})
    return m, d, pred


def bootstrap_all():
    rows = []
    for seed in SEEDS:
        for task in TASKS:
            left = v2_pred(task, "M2") if seed == 42 else ms_pred(task, "M2", seed)
            right = v2_pred(task, "M0") if seed == 42 else ms_pred(task, "M0", seed)
            point, values, sites = B.bootstrap(left, right)
            rows.append(dict(protocol=f"random_strict_seed{seed}", task=task, model="TabICLv2", contrast="M2-M0",
                             n_stations=sites, n_bootstrap=len(values), **point,
                             **{k: np.quantile(values[:, j], [.025, .975])[0] for j, k in
                                enumerate(["delta_R2_ci_low", "delta_RMSE_ci_low", "delta_MAE_ci_low"])},
                             **{k: np.quantile(values[:, j], [.025, .975])[1] for j, k in
                                enumerate(["delta_R2_ci_high", "delta_RMSE_ci_high", "delta_MAE_ci_high"])}))
    for task in TASKS:
        left = pd.concat([pd.read_parquet(SH / f"{task}__M2__{f}.parquet")[["obs_id", "station_id", "y_true", "y_pred"]]
                          for f in FOLDS], ignore_index=True)
        right = pd.concat([pd.read_parquet(SH / f"{task}__M0__{f}.parquet")[["obs_id", "station_id", "y_true", "y_pred"]]
                           for f in FOLDS], ignore_index=True)
        point, values, sites = B.bootstrap(left, right)
        rows.append(dict(protocol="station_holdout", task=task, model="TabICLv2", contrast="M2-M0",
                         n_stations=sites, n_bootstrap=len(values), **point,
                         **{k: np.quantile(values[:, j], [.025, .975])[0] for j, k in
                            enumerate(["delta_R2_ci_low", "delta_RMSE_ci_low", "delta_MAE_ci_low"])},
                         **{k: np.quantile(values[:, j], [.025, .975])[1] for j, k in
                            enumerate(["delta_R2_ci_high", "delta_RMSE_ci_high", "delta_MAE_ci_high"])}))
    boot = pd.DataFrame(rows)
    boot.to_csv(OUT / "04_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    print("bootstrap rows:", len(boot))
    return boot


def gates(d_ms, d_sh):
    print("\n=== gate A: multi-seed (random strict) ===")
    per_seed = d_ms.groupby("seed").delta_R2.mean()
    task_mean = d_ms.groupby("task").delta_R2.mean()
    print("per-seed macro delta_R2:", {int(k): round(v, 4) for k, v in per_seed.items()},
          "| three-seed mean:", round(per_seed.mean(), 4))
    print("tasks mean delta_R2 > 0:", int((task_mean > 0).sum()), "/10 | >= 0.02:", int((task_mean >= 0.02).sum()), "/10")
    print("macro delta_RMSE:", round(d_ms.groupby("seed").delta_RMSE.mean().mean(), 4))
    print("\n=== gate B: station holdout (pooled OOF) ===")
    p = d_sh[d_sh.fold == "pooled_OOF"].set_index("task")
    print("pooled macro delta_R2:", round(p.delta_R2.mean(), 4),
          "| tasks positive:", int((p.delta_R2 > 0).sum()), "/10",
          "| macro delta_RMSE:", round(p.delta_RMSE.mean(), 4))
    print("\nper-task pooled delta_R2:")
    print(p[["R2", "R2_M0", "delta_R2", "delta_RMSE"]].round(4).to_string())
    print("\nper-fold delta_R2 (M2-M0):")
    print(d_sh[d_sh.fold != "pooled_OOF"].pivot_table(index="fold", columns="task", values="delta_R2").round(4).to_string())


def main():
    m_ms, d_ms, s_ms = random_multiseed()
    m_sh, d_sh, _ = station_holdout()
    bootstrap_all()
    gates(d_ms, d_sh)


if __name__ == "__main__":
    main()
