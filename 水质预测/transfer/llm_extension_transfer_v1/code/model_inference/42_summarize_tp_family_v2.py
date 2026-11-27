"""Summarize TP_FAMILY_V2 and compare the old TP middle pilot."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tp_family_v2"
OLD_RUN = ROOT / "runs" / "tabiclv2_pilot"
BASE_PATH = ROOT / "脚本" / "05_诊断分析" / "39_summarize_tn_family_v2.py"
TASKS = ["tp_surface", "tp_middle", "tp_bottom"]
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))


def load_base():
    spec = importlib.util.spec_from_file_location("tp_family_v2_summary_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import summary base: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()
BASE.RUN = RUN
BASE.PREDICTION_DIR = RUN / "predictions"
BASE.TASKS = TASKS
BASE.FOLDS = FOLDS
BASE.BOOTSTRAP_SEED = 20260921
BASE.N_BOOTSTRAP = 1000


def old_tp_middle() -> dict[str, object] | None:
    path = OLD_RUN / "17_tp_middle_predictions.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={"obs_id": "string", "station_id": "string"})
    frame = frame.loc[frame["validation"].astype(str).isin({"station", "station_holdout"})].copy()
    frame["model_std"] = frame["model"].astype(str).str.lower().map({"xgb": "xgboost", "xgboost": "xgboost", "tabicl": "tabiclv2", "tabiclv2": "tabiclv2"})
    frame = frame.loc[frame["model_std"].notna()]
    xgb = frame.loc[frame["model_std"].eq("xgboost")].rename(columns={"y_true": "y_true_xgb", "y_pred": "y_pred_xgb"})
    tab = frame.loc[frame["model_std"].eq("tabiclv2")].rename(columns={"y_true": "y_true_tabicl", "y_pred": "y_pred_tabicl"})
    keys = ["obs_id", "station_id", "fold"]
    paired = xgb.merge(tab[keys + ["y_true_tabicl", "y_pred_tabicl"]], on=keys, validate="one_to_one").rename(columns={"y_true_xgb": "y_true"})
    xgb_metrics = BASE.metric_values(paired["y_true"].to_numpy(float), paired["y_pred_xgb"].to_numpy(float))
    tab_metrics = BASE.metric_values(paired["y_true"].to_numpy(float), paired["y_pred_tabicl"].to_numpy(float))
    station = []
    for station_id, group in paired.groupby("station_id", sort=True):
        x = BASE.metric_values(group["y_true"].to_numpy(float), group["y_pred_xgb"].to_numpy(float))
        t = BASE.metric_values(group["y_true"].to_numpy(float), group["y_pred_tabicl"].to_numpy(float))
        station.append({"station_id": station_id, "delta_RMSE": t["RMSE"] - x["RMSE"], "delta_MAE": t["MAE"] - x["MAE"]})
    station = pd.DataFrame(station)
    return {
        "split_label": "old_tp_middle_pilot",
        "n_obs": len(paired),
        "n_station": paired["station_id"].nunique(),
        "XGB_R2": xgb_metrics["R2"],
        "TabICL_R2": tab_metrics["R2"],
        "delta_R2": tab_metrics["R2"] - xgb_metrics["R2"],
        "delta_RMSE": tab_metrics["RMSE"] - xgb_metrics["RMSE"],
        "delta_MAE": tab_metrics["MAE"] - xgb_metrics["MAE"],
        "RMSE_win_rate": float((station["delta_RMSE"] < 0).mean()),
        "MAE_win_rate": float((station["delta_MAE"] < 0).mean()),
    }


def replication(new_row: pd.Series, new_station: pd.Series) -> pd.DataFrame:
    old = old_tp_middle()
    new = {
        "split_label": "TP_FAMILY_V2_UNIFIED_STATION_SPLIT",
        "n_obs": new_row["n_obs"],
        "n_station": new_row["n_station"],
        "XGB_R2": new_row["XGB_R2"],
        "TabICL_R2": new_row["TabICL_R2"],
        "delta_R2": new_row["delta_R2"],
        "delta_RMSE": new_row["delta_RMSE"],
        "delta_MAE": new_row["delta_MAE"],
        "RMSE_win_rate": new_station["win_rate_rmse"],
        "MAE_win_rate": new_station["win_rate_mae"],
    }
    rows = [] if old is None else [old]
    rows.append(new)
    frame = pd.DataFrame(rows)
    if old is None:
        frame["replication_classification"] = "OLD_RESULT_UNAVAILABLE"
    else:
        old_positive = old["delta_RMSE"] < 0 and old["delta_MAE"] < 0 and old["RMSE_win_rate"] > 0.5 and old["MAE_win_rate"] > 0.5
        new_positive = new["delta_RMSE"] < 0 and new["delta_MAE"] < 0 and new["RMSE_win_rate"] > 0.5 and new["MAE_win_rate"] > 0.5
        same_direction = old["delta_RMSE"] < 0 and new["delta_RMSE"] < 0
        classification = "REPLICATED" if old_positive and new_positive else "PARTIALLY_REPLICATED" if same_direction or old_positive != new_positive else "NOT_REPLICATED"
        frame["replication_classification"] = classification
    return frame


def write_decision(depth: pd.DataFrame, rep: pd.DataFrame) -> str:
    supported = int(depth["classification"].isin(["ROBUST", "MODERATE"]).sum())
    no_gain = int(depth["classification"].eq("NO_GAIN").sum())
    family = "TP_FAMILY_SUPPORTED" if supported >= 2 and no_gain <= 1 else "TP_FAMILY_PARTIAL" if supported == 1 else "TP_FAMILY_NOT_SUPPORTED"
    rep_class = rep["replication_classification"].iloc[0]
    lines = [
        "# TP_FAMILY_V2 决策报告",
        "",
        f"- 生成时间：`{datetime.now().isoformat(timespec='seconds')}`",
        "- 协议：`TP_FAMILY_V2_UNIFIED_STATION_SPLIT`",
        f"- TP family 总判定：**{family}**",
        f"- 旧 TP middle pilot 方向重复验证：**{rep_class}**",
        "",
        "## 三层结果",
        "",
        "| 水层 | XGBoost R² | TabICLv2 R² | ΔR² | ΔRMSE | ΔMAE | RMSE胜率 | MAE胜率 | 分类 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in depth.itertuples(index=False):
        lines.append(f"| {row.depth} | {row.XGB_R2:.4f} | {row.TabICL_R2:.4f} | {row.delta_R2:+.4f} | {row.delta_RMSE:+.4f} | {row.delta_MAE:+.4f} | {row.RMSE_win_rate:.3f} | {row.MAE_win_rate:.3f} | {row.classification} |")
    lines += [
        "",
        "## TP middle 旧结果对照",
        "",
        rep.to_string(index=False),
        "",
        "TP_FAMILY_V2 完成后停止新增实验，不自动进入其他指标、其他模型或区域适配路线。",
    ]
    (RUN / "tp_family_v2_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return family


def main() -> int:
    paired = {task: BASE.build_paired(task) for task in TASKS}
    pooled_models, pooled = BASE.pooled_table(paired)
    station_frame, station = BASE.station_metrics(paired)
    boot = BASE.bootstrap(paired)
    folds = BASE.fold_consistency(paired)
    rows = []
    for row in pooled.itertuples(index=False):
        summary = station.loc[station["task"].eq(row.task)].iloc[0]
        rows.append(BASE.classify(row.depth, pd.Series(row._asdict()), summary, boot.loc[boot["task"].eq(row.task)]))
    depth = pd.DataFrame(rows)
    rep = replication(pooled.loc[pooled["task"].eq("tp_middle")].iloc[0], station.loc[station["task"].eq("tp_middle")].iloc[0])
    pooled_models.to_csv(RUN / "08_tp_family_v2_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(RUN / "08_tp_family_v2_pooled_comparison.csv", index=False, encoding="utf-8-sig")
    station_frame.to_csv(RUN / "09_tp_family_v2_station_metrics.csv", index=False, encoding="utf-8-sig")
    station.to_csv(RUN / "09_tp_family_v2_station_win_rates.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(RUN / "10_tp_family_v2_bootstrap.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(RUN / "11_tp_family_v2_fold_consistency.csv", index=False, encoding="utf-8-sig")
    rep.to_csv(RUN / "tp_middle_replication.csv", index=False, encoding="utf-8-sig")
    depth.to_csv(RUN / "tp_family_v2_depth_generalization.csv", index=False, encoding="utf-8-sig")
    decision = write_decision(depth, rep)
    print("TP_FAMILY_V2_SUMMARY")
    print(depth.to_string(index=False))
    print(f"TP_FAMILY_DECISION={decision}")
    print(f"TP_MIDDLE_REPLICATION={rep['replication_classification'].iloc[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
