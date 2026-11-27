"""汇总 CHLA_FAMILY_V2，不训练模型。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "chla_family_v2"
OLD = ROOT / "runs" / "tabiclv2_pilot"
DEPTHS = ("surface", "middle", "bottom")
TASKS = [f"chla_{d}" for d in DEPTHS]
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
N_BOOT = 1000
SEED = 20260921


def metrics(y, p):
    return {"R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** 0.5, "MAE": mean_absolute_error(y, p)}


def pred(task, model, fold):
    return RUN / "predictions" / f"{task}_{model}_{fold}.csv"


def paired(task):
    frames = {}
    for model in ("xgboost", "tabiclv2"):
        parts = []
        for fold in FOLDS:
            p = pred(task, model, fold)
            if not p.exists():
                raise FileNotFoundError(p)
            parts.append(pd.read_csv(p, dtype={"obs_id": "string", "station_id": "string"}))
        frame = pd.concat(parts, ignore_index=True)
        if frame["obs_id"].duplicated().any():
            raise RuntimeError(f"{task}/{model} obs_id 重复")
        frames[model] = frame
    x = frames["xgboost"].rename(columns={"y_true": "y_true_xgb", "y_pred": "y_pred_xgb"})
    t = frames["tabiclv2"].rename(columns={"y_true": "y_true_tabicl", "y_pred": "y_pred_tabicl"})
    keys = ["obs_id", "station_id", "date", "depth", "fold"]
    out = x.merge(t[keys + ["y_true_tabicl", "y_pred_tabicl"]], on=keys, validate="one_to_one")
    if len(out) != len(x) or not np.allclose(out.y_true_xgb, out.y_true_tabicl):
        raise RuntimeError(f"{task} 两个模型目标值不一致")
    return out.rename(columns={"y_true_xgb": "y_true"})


def summarize_pooled(data):
    pooled, comp = [], []
    for task, df in data.items():
        depth = task.removeprefix("chla_")
        vals = {}
        for model, col in (("xgboost", "y_pred_xgb"), ("tabiclv2", "y_pred_tabicl")):
            m = metrics(df.y_true.to_numpy(float), df[col].to_numpy(float)); vals[model] = m
            pooled.append({"task": task, "depth": depth, "model": model, "n_obs": len(df), "n_station": df.station_id.nunique(), **m})
        comp.append({"task": task, "depth": depth, "XGB_R2": vals["xgboost"]["R2"], "TabICL_R2": vals["tabiclv2"]["R2"], "delta_R2": vals["tabiclv2"]["R2"]-vals["xgboost"]["R2"], "XGB_RMSE": vals["xgboost"]["RMSE"], "TabICL_RMSE": vals["tabiclv2"]["RMSE"], "delta_RMSE": vals["tabiclv2"]["RMSE"]-vals["xgboost"]["RMSE"], "XGB_MAE": vals["xgboost"]["MAE"], "TabICL_MAE": vals["tabiclv2"]["MAE"], "delta_MAE": vals["tabiclv2"]["MAE"]-vals["xgboost"]["MAE"], "n_obs": len(df), "n_station": df.station_id.nunique()})
    return pd.DataFrame(pooled), pd.DataFrame(comp)


def station_summary(data):
    rows, sums = [], []
    for task, df in data.items():
        for station, g in df.groupby("station_id", sort=True):
            y = g.y_true.to_numpy(float); x = g.y_pred_xgb.to_numpy(float); t = g.y_pred_tabicl.to_numpy(float)
            mx, mt = metrics(y, x), metrics(y, t); reliable = len(g) >= 5 and np.std(y) > 1e-12
            rows.append({"task": task, "depth": task.removeprefix("chla_"), "station_id": station, "n_obs": len(g), "XGB_R2": mx["R2"] if reliable else np.nan, "TabICL_R2": mt["R2"] if reliable else np.nan, "delta_R2": mt["R2"]-mx["R2"] if reliable else np.nan, "XGB_RMSE": mx["RMSE"], "TabICL_RMSE": mt["RMSE"], "delta_RMSE": mt["RMSE"]-mx["RMSE"], "XGB_MAE": mx["MAE"], "TabICL_MAE": mt["MAE"], "delta_MAE": mt["MAE"]-mx["MAE"], "r2_status": "OK" if reliable else "R2_UNSTABLE"})
        s = pd.DataFrame([r for r in rows if r["task"] == task]); valid = s[s.r2_status == "OK"]
        sums.append({"task": task, "depth": task.removeprefix("chla_"), "n_station": len(s), "n_station_win_rmse": int((s.delta_RMSE < 0).sum()), "win_rate_rmse": float((s.delta_RMSE < 0).mean()), "n_station_win_mae": int((s.delta_MAE < 0).sum()), "win_rate_mae": float((s.delta_MAE < 0).mean()), "n_station_valid_r2": len(valid), "n_station_win_r2": int((valid.delta_R2 > 0).sum()), "win_rate_r2": float((valid.delta_R2 > 0).mean()) if len(valid) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(sums)


def bootstrap(data):
    rng = np.random.default_rng(SEED); rows = []
    for task, df in data.items():
        stations = np.array(sorted(df.station_id.astype(str).unique())); groups = {s: np.flatnonzero(df.station_id.astype(str).to_numpy() == s) for s in stations}; vals = []
        for _ in range(N_BOOT):
            chosen = rng.choice(stations, len(stations), replace=True); ix = np.concatenate([groups[s] for s in chosen]); y = df.y_true.to_numpy(float)[ix]
            x, t = metrics(y, df.y_pred_xgb.to_numpy(float)[ix]), metrics(y, df.y_pred_tabicl.to_numpy(float)[ix]); vals.append({"delta_R2": t["R2"]-x["R2"], "delta_RMSE": t["RMSE"]-x["RMSE"], "delta_MAE": t["MAE"]-x["MAE"]})
        v = pd.DataFrame(vals)
        for metric in ("delta_R2", "delta_RMSE", "delta_MAE"):
            rows.append({"task": task, "depth": task.removeprefix("chla_"), "metric": metric, "n_bootstrap": N_BOOT, "seed": SEED, "median": v[metric].median(), "low_2p5": v[metric].quantile(.025), "high_97p5": v[metric].quantile(.975)})
    return pd.DataFrame(rows)


def fold_table(data):
    rows = []
    for task, df in data.items():
        for fold, g in df.groupby("fold", sort=True):
            x, t = metrics(g.y_true.to_numpy(float), g.y_pred_xgb.to_numpy(float)), metrics(g.y_true.to_numpy(float), g.y_pred_tabicl.to_numpy(float)); rows.append({"task": task, "depth": task.removeprefix("chla_"), "fold": fold, "n_obs": len(g), "n_station": g.station_id.nunique(), "XGB_R2": x["R2"], "TabICL_R2": t["R2"], "delta_R2": t["R2"]-x["R2"], "XGB_RMSE": x["RMSE"], "TabICL_RMSE": t["RMSE"], "delta_RMSE": t["RMSE"]-x["RMSE"], "XGB_MAE": x["MAE"], "TabICL_MAE": t["MAE"], "delta_MAE": t["MAE"]-x["MAE"]})
    return pd.DataFrame(rows)


def classify(comp, wins, boot):
    r = boot[(boot.task == comp.task) & (boot.metric == "delta_RMSE")].iloc[0]; m = boot[(boot.task == comp.task) & (boot.metric == "delta_MAE")].iloc[0]
    robust = comp.delta_RMSE < 0 and comp.delta_MAE < 0 and wins.win_rate_rmse >= .60 and wins.win_rate_mae >= .60 and r.high_97p5 < 0 and m.high_97p5 < 0
    gain = comp.delta_R2 > 0 or comp.delta_RMSE < 0 or comp.delta_MAE < 0
    moderate = not robust and gain and wins.win_rate_rmse >= .50 and wins.win_rate_mae >= .50
    return "ROBUST" if robust else "MODERATE" if moderate else "MIXED" if gain else "NO_GAIN"


def old_direction(depth):
    names = {"surface": "33_chla_surface_station_predictions.csv", "middle": "34_chla_middle_station_predictions.csv", "bottom": "35_chla_bottom_station_predictions.csv"}
    p = OLD / names[depth]
    if not p.exists(): return None
    f = pd.read_csv(p, dtype={"obs_id": "string", "station_id": "string"}); f = f[f["validation"].astype(str).isin(["station", "station_holdout"])].copy(); f["std"] = f["model"].astype(str).str.lower().map({"xgb":"xgboost", "xgboost":"xgboost", "tabicl":"tabiclv2", "tabiclv2":"tabiclv2"}); f = f[f["std"].notna()]
    x = f[f["std"] == "xgboost"].rename(columns={"y_true":"y", "y_pred":"px"}); t = f[f["std"] == "tabiclv2"].rename(columns={"y_true":"yt", "y_pred":"pt"}); q = x.merge(t[["obs_id","station_id","fold","yt","pt"]], on=["obs_id","station_id","fold"], validate="one_to_one");
    if len(q) == 0: return None
    mx, mt = metrics(q.y.to_numpy(float), q.px.to_numpy(float)), metrics(q.y.to_numpy(float), q.pt.to_numpy(float)); ss=[]
    for _, g in q.groupby("station_id"): ss.append((metrics(g.y.to_numpy(float),g.px.to_numpy(float)), metrics(g.y.to_numpy(float),g.pt.to_numpy(float))))
    return {"old_delta_R2": mt["R2"]-mx["R2"], "old_delta_RMSE": mt["RMSE"]-mx["RMSE"], "old_delta_MAE": mt["MAE"]-mx["MAE"], "old_RMSE_win_rate": np.mean([b["RMSE"] < a["RMSE"] for a,b in ss]), "old_MAE_win_rate": np.mean([b["MAE"] < a["MAE"] for a,b in ss]), "old_source": str(p)}


def main():
    data = {task: paired(task) for task in TASKS}; pooled, comp = summarize_pooled(data); stations, wins = station_summary(data); boot = bootstrap(data); folds = fold_table(data); depth = [] ; rep = []
    for row in comp.itertuples(index=False):
        w = wins[wins.task == row.task].iloc[0]; b = boot[boot.task == row.task]; cls = classify(row, w, boot); depth.append({**row._asdict(), "RMSE_win_rate": w.win_rate_rmse, "MAE_win_rate": w.win_rate_mae, "bootstrap_RMSE_low": b[b.metric=="delta_RMSE"].low_2p5.iloc[0], "bootstrap_RMSE_high": b[b.metric=="delta_RMSE"].high_97p5.iloc[0], "bootstrap_MAE_low": b[b.metric=="delta_MAE"].low_2p5.iloc[0], "bootstrap_MAE_high": b[b.metric=="delta_MAE"].high_97p5.iloc[0], "classification": cls})
        old = old_direction(row.depth); new_pos = row.delta_RMSE < 0 and row.delta_MAE < 0 and w.win_rate_rmse > .5 and w.win_rate_mae > .5; old_pos = old and old["old_delta_RMSE"] < 0 and old["old_delta_MAE"] < 0 and old["old_RMSE_win_rate"] > .5 and old["old_MAE_win_rate"] > .5; rc = "OLD_RESULT_UNAVAILABLE" if old is None else "REPLICATED" if old_pos and new_pos else "PARTIALLY_REPLICATED" if old_pos != new_pos or (old["old_delta_R2"] * row.delta_R2 >= 0) else "NOT_REPLICATED"; rep.append({"depth": row.depth, "new_delta_R2": row.delta_R2, "new_delta_RMSE": row.delta_RMSE, "new_delta_MAE": row.delta_MAE, "new_RMSE_win_rate": w.win_rate_rmse, "new_MAE_win_rate": w.win_rate_mae, **(old or {}), "replication_classification": rc})
    pooled.to_csv(RUN / "08_chla_family_v2_pooled_metrics.csv", index=False, encoding="utf-8-sig"); comp.to_csv(RUN / "08_chla_family_v2_pooled_comparison.csv", index=False, encoding="utf-8-sig"); stations.to_csv(RUN / "09_chla_family_v2_station_metrics.csv", index=False, encoding="utf-8-sig"); wins.to_csv(RUN / "09_chla_family_v2_station_win_rates.csv", index=False, encoding="utf-8-sig"); boot.to_csv(RUN / "10_chla_family_v2_bootstrap.csv", index=False, encoding="utf-8-sig"); folds.to_csv(RUN / "11_chla_family_v2_fold_consistency.csv", index=False, encoding="utf-8-sig")
    depth_df, rep_df = pd.DataFrame(depth), pd.DataFrame(rep); depth_df.to_csv(RUN / "chla_family_v2_depth_generalization.csv", index=False, encoding="utf-8-sig"); rep_df.to_csv(RUN / "chla_family_v2_replication.csv", index=False, encoding="utf-8-sig")
    supported = sum(depth_df.classification.isin(["ROBUST","MODERATE"])); family = "CHLA_FAMILY_SUPPORTED" if supported >= 2 else "CHLA_FAMILY_PARTIAL" if supported == 1 else "CHLA_FAMILY_NOT_SUPPORTED"; lines = ["# CHLA_FAMILY_V2 决策报告", "", f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}", "- 协议：`CHLA_FAMILY_V2_UNIFIED_STATION_SPLIT`", f"- family 判定：**{family}**", "", "| 水层 | XGB R² | TabICL R² | ΔR² | ΔRMSE | ΔMAE | RMSE胜率 | MAE胜率 | 分类 |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in depth_df.itertuples(index=False): lines.append(f"| {r.depth} | {r.XGB_R2:.4f} | {r.TabICL_R2:.4f} | {r.delta_R2:+.4f} | {r.delta_RMSE:+.4f} | {r.delta_MAE:+.4f} | {r.RMSE_win_rate:.3f} | {r.MAE_win_rate:.3f} | {r.classification} |")
    lines += ["", "## 历史结果方向复现", "", rep_df.to_string(index=False), "", "本实验完成后停止新增模型和验证方式，不自动进入其他任务。"]
    (RUN / "chla_family_v2_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(depth_df.to_string(index=False)); print(rep_df.to_string(index=False)); print(f"FAMILY={family}")


if __name__ == "__main__": main()
