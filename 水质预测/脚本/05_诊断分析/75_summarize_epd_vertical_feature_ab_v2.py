"""Summarize paired EPD vertical-feature experiment without model inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/epd_vertical_feature_ab_v2_20261126"
IND = OUT / "individual_predictions"
CONTRASTS = [("paired_SB", "M1", "M0"), ("complete_SMB", "M1", "M0"),
             ("complete_SMB", "M2", "M0"), ("complete_SMB", "M3", "M0"),
             ("complete_SMB", "M3", "M1"), ("complete_SMB", "M3", "M2")]


def score(y, p):
    return {"R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** .5,
            "MAE": mean_absolute_error(y, p)}


def bootstrap(left, right, n=1000, seed=42):
    a = left.merge(right[["obs_id", "y_true", "y_pred"]], on="obs_id", validate="one_to_one",
                   suffixes=("_a", "_b"))
    if len(a) != len(left) or len(a) != len(right) or not np.allclose(a.y_true_a, a.y_true_b):
        raise ValueError("配对预测的obs_id或目标值不一致")
    station = a.station_id.to_numpy()
    groups = {s: np.flatnonzero(station == s) for s in pd.unique(station)}
    keys = np.array(list(groups), dtype=object)
    y = a.y_true_a.to_numpy(float); p = a.y_pred_a.to_numpy(float); q = a.y_pred_b.to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty((n, 3), float)
    for i in range(n):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in chosen])
        sa = score(y[idx], p[idx]); sb = score(y[idx], q[idx])
        values[i] = [sa[k] - sb[k] for k in ("R2", "RMSE", "MAE")]
    point_a = score(y, p); point_b = score(y, q)
    return {f"delta_{k}": point_a[k] - point_b[k] for k in ("R2", "RMSE", "MAE")}, values, len(groups)


def main():
    files = sorted(IND.glob("*.parquet"))
    preds = [pd.read_parquet(p) for p in files]
    if len(files) != 120:
        raise ValueError(f"预计120份预测，实际{len(files)}份；未完成时不汇总")
    all_pred = pd.concat(preds, ignore_index=True)
    all_pred.to_parquet(OUT / "04_逐样本预测.parquet", index=False)
    meta = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(IND.glob("*.json"))]
    metrics = pd.DataFrame(meta).sort_values(["model", "task", "cohort", "feature_set"])
    metrics.to_csv(OUT / "03_十任务实验结果.csv", index=False, encoding="utf-8-sig")
    coverage = (metrics.loc[metrics.feature_set.eq("M0"),
                        ["task", "cohort", "n_train", "n_test", "n_stations", "n_test_stations"]]
                .drop_duplicates(["task", "cohort"]).sort_values(["task", "cohort"]))
    coverage.to_csv(OUT / "01_十任务样本覆盖.csv", index=False, encoding="utf-8-sig")
    rows = []
    for feature_set, cols in {"M0":77,"M1":81,"M2":89,"M3":93}.items():
        from importlib.machinery import SourceFileLoader
        runner = SourceFileLoader("vertical_runner", str(ROOT / "脚本/04_模型训练/74_run_epd_vertical_feature_ab_v2.py")).load_module()
        for pos, col in enumerate(runner.SETS[feature_set], 1):
            rows.append({"方案": feature_set, "序号": pos, "特征": col, "特征总数": cols})
    pd.DataFrame(rows).to_csv(OUT / "02_特征集清单.csv", index=False, encoding="utf-8-sig")
    index = {(d.task.iloc[0], d.model.iloc[0], d.cohort.iloc[0], d.feature_set.iloc[0]): d
             for d in preds}
    boot = []
    for task in metrics.task.unique():
        for model in metrics.model.unique():
            for cohort, a, b in CONTRASTS:
                point, values, sites = bootstrap(index[task, model, cohort, a], index[task, model, cohort, b])
                item = {"task": task, "model": model, "cohort": cohort, "contrast": f"{a}-{b}",
                        "n_stations": sites, "n_bootstrap": len(values), **point}
                for j, name in enumerate(("R2", "RMSE", "MAE")):
                    item[f"delta_{name}_ci_low"], item[f"delta_{name}_ci_high"] = np.quantile(values[:, j], [.025, .975])
                boot.append(item)
    boot = pd.DataFrame(boot)
    boot.to_csv(OUT / "05_配对Bootstrap.csv", index=False, encoding="utf-8-sig")
    return metrics, boot, coverage


if __name__ == "__main__":
    main()
