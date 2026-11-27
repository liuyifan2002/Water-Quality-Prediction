"""Summarize Chl-a family station-holdout spatial generalization.

This script only reads Phase A predictions and metrics. It does not train any
model and does not modify the formal mainline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tabiclv2_pilot"
AUDIT_PATH = ROOT / "脚本" / "05_诊断分析" / "34_audit_tabiclv2_spatial_robustness.py"

TASKS = {
    "chla_surface": {"depth": "surface", "prediction": "33_chla_surface_station_predictions.csv", "metrics": "30_chla_surface_station_metrics.csv"},
    "chla_middle": {"depth": "middle", "prediction": "34_chla_middle_station_predictions.csv", "metrics": "31_chla_middle_station_metrics.csv"},
    "chla_bottom": {"depth": "bottom", "prediction": "35_chla_bottom_station_predictions.csv", "metrics": "32_chla_bottom_station_metrics.csv"},
}


def load_audit_module():
    spec = importlib.util.spec_from_file_location("tabiclv2_spatial_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import audit module: {AUDIT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit_module()
for task, spec in TASKS.items():
    AUDIT.TASKS[task] = {
        "target_type": "optical",
        "prediction": spec["prediction"],
        "fold_results": spec["metrics"],
        "target_column": "chla_ug_l",
        "depth": spec["depth"],
    }


def top2_error_share(station: pd.DataFrame, column: str) -> float:
    improvement = -station[column].loc[station[column] < 0]
    total = float(improvement.sum())
    if total <= 0:
        return 0.0
    return float(improvement.nlargest(2).sum() / total)


def load_family_predictions(task: str) -> pd.DataFrame:
    data = pd.read_csv(RUN / TASKS[task]["prediction"], low_memory=False)
    data = data.loc[data["validation"].eq("station")].copy()
    data["station_id"] = data["station_id"].astype(str).str.strip()
    data["obs_id"] = data["obs_id"].astype(str)
    duplicate_count = int(data.duplicated(["obs_id", "model"]).sum())
    if duplicate_count:
        raise RuntimeError(f"{task}: duplicated station prediction rows: {duplicate_count}")
    return data


def classify(pooled: dict[str, object], ci: pd.DataFrame, station: pd.DataFrame) -> tuple[str, dict[str, float]]:
    rmse_ci = ci.loc[ci["metric"].eq("delta_RMSE")].iloc[0]
    mae_ci = ci.loc[ci["metric"].eq("delta_MAE")].iloc[0]
    share_rmse = top2_error_share(station, "delta_RMSE")
    share_mae = top2_error_share(station, "delta_MAE")
    share = max(share_rmse, share_mae)
    stable_error = bool(rmse_ci["ci_upper_97_5"] < 0 or mae_ci["ci_upper_97_5"] < 0)
    majority_error = bool(pooled["station_win_rate_rmse"] > 0.5 and pooled["station_win_rate_mae"] > 0.5)
    positive_r2 = bool(pooled["delta_pooled_R2"] > 0)
    if positive_r2 and stable_error and majority_error and share < 0.25:
        label = "ROBUST"
    elif positive_r2 and (stable_error or majority_error) and share < 0.40:
        label = "MODERATE"
    elif positive_r2 or majority_error:
        label = "MIXED"
    else:
        label = "NO_GAIN"
    return label, {"top2_rmse_share": share_rmse, "top2_mae_share": share_mae, "top2_error_share": share}


def main() -> None:
    pooled_rows = []
    station_frames = []
    bootstrap_frames = []
    depth_rows = []
    decisions = []
    for task, spec in TASKS.items():
        predictions = load_family_predictions(task)
        if predictions is None or predictions.empty:
            raise RuntimeError(f"Missing Phase A predictions for {task}")
        fold_results = AUDIT.load_fold_results(task)
        station = AUDIT.station_comparison(predictions, task)
        pooled = AUDIT.pooled_metrics(predictions, task, fold_results)
        ci = AUDIT.bootstrap_ci(predictions, task)
        label, concentration = classify(pooled, ci, station)
        pooled["classification"] = label
        pooled["top2_rmse_improvement_share"] = concentration["top2_rmse_share"]
        pooled["top2_mae_improvement_share"] = concentration["top2_mae_share"]
        pooled["top2_error_improvement_share"] = concentration["top2_error_share"]
        pooled_rows.append(pooled)
        station = station.assign(depth=spec["depth"], target_family="Chl-a", classification=label)
        station_frames.append(station)
        ci = ci.assign(depth=spec["depth"], target_family="Chl-a", classification=label)
        bootstrap_frames.append(ci)
        rmse_ci = ci.loc[ci["metric"].eq("delta_RMSE")].iloc[0]
        mae_ci = ci.loc[ci["metric"].eq("delta_MAE")].iloc[0]
        depth_rows.append(
            {
                "depth": spec["depth"],
                "pooled_XGB_R2": pooled["XGBoost_pooled_R2"],
                "pooled_TabICL_R2": pooled["TabICLv2_pooled_R2"],
                "delta_R2": pooled["delta_pooled_R2"],
                "delta_RMSE": pooled["delta_pooled_RMSE"],
                "delta_MAE": pooled["delta_pooled_MAE"],
                "station_win_rate_RMSE": pooled["station_win_rate_rmse"],
                "station_win_rate_MAE": pooled["station_win_rate_mae"],
                "station_valid_R2": pooled["station_valid_r2"],
                "station_win_rate_R2": pooled["station_win_rate_r2"],
                "bootstrap_RMSE_low": rmse_ci["ci_lower_2_5"],
                "bootstrap_RMSE_high": rmse_ci["ci_upper_97_5"],
                "bootstrap_MAE_low": mae_ci["ci_lower_2_5"],
                "bootstrap_MAE_high": mae_ci["ci_upper_97_5"],
                "top2_error_improvement_share": concentration["top2_error_share"],
                "classification": label,
            }
        )
        decisions.append((spec["depth"], label))

    pooled_frame = pd.DataFrame(pooled_rows)
    pooled_frame.to_csv(RUN / "chla_family_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(station_frames, ignore_index=True).to_csv(RUN / "chla_family_station_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(RUN / "chla_family_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    depth_frame = pd.DataFrame(depth_rows)
    depth_frame.to_csv(RUN / "chla_depth_generalization.csv", index=False, encoding="utf-8-sig")

    robust_count = int((depth_frame["classification"] == "ROBUST").sum())
    surface_label = depth_frame.loc[depth_frame["depth"].eq("surface"), "classification"].iloc[0]
    middle_bottom = depth_frame.loc[depth_frame["depth"].isin(["middle", "bottom"]), "classification"]
    if robust_count >= 2:
        gate = "CHLA_FAMILY_ROBUST"
        gate_reason = "三层中至少两层具有 pooled 改善、站点误差改善和非少数站点集中的证据。"
    elif surface_label in {"ROBUST", "MODERATE"} and not middle_bottom.isin(["NO_GAIN"]).all():
        gate = "CHLA_FAMILY_MIXED"
        gate_reason = "表层具有稳定或中等空间增益，但中层或底层存在垂向差异；仍可进入 TN family，但不得宣称 Chl-a 各层一致有效。"
    else:
        gate = "CHLA_FAMILY_FAIL"
        gate_reason = "表层以外的两个水层均未形成可接受的空间增益，暂停后续指标族扩展。"

    repro = pd.read_csv(RUN / "chla_surface_repro_check.csv")
    repro_status = ", ".join(f"{row.metric}={row.status}" for row in repro.itertuples())
    lines = [
        "# Chl-a 三层空间泛化 Gate",
        "",
        f"## Phase A 判定：{gate}",
        "",
        gate_reason,
        "",
        f"Chl-a surface 复现检查：{repro_status}。仅使用 station holdout；本阶段未运行 random/year，也未运行 TN。",
        "",
        "## 三层结果",
        "",
        depth_frame.to_string(index=False),
        "",
        "## 判定口径",
        "",
        "分类同时参考 pooled OOF、站点 RMSE/MAE 胜率、1000 次站点级 bootstrap 和前两个站点对误差改善量的贡献。前两个站点贡献低于 0.25 记为 ROBUST，0.25 至 0.40 记为 MODERATE，达到 0.40 记为 MIXED；这只是本轮描述性 Gate 规则，不是显著性检验。",
        "",
        "## 下一步",
        "",
        "Phase A 结果完成后再决定是否运行 TN surface 和 TN middle；本脚本不会自动启动 TN。",
        "",
        "本报告属于扩展实验，不修改正式主链。",
    ]
    (RUN / "chla_family_decision.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Chl-a family summary complete: {gate}")


if __name__ == "__main__":
    main()
