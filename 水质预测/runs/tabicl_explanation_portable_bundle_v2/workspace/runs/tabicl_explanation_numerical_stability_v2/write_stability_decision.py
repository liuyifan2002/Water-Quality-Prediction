"""Write V2 assessment from saved probes and existing SHAP/ALE artifacts only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_explanation_numerical_stability_v2"
RUN = ROOT / "runs/tabicl_v2_final_explanation_complete_v1"


def main() -> None:
    summ = pd.read_csv(OUT / "numerical_jitter_summary.csv")
    task = pd.read_csv(OUT / "jitter_vs_task_scale.csv")
    exp = pd.read_csv(OUT / "jitter_vs_explanation_scale.csv")
    raw = pd.read_csv(OUT / "p1_repeat_predictions.csv")
    audit = json.loads((RUN / "fold_1_completion_audit.json").read_text(encoding="utf-8"))
    chla = summ.loc[summ.task == "chla_surface"]
    tn = summ.loc[summ.task == "tn_surface"]
    historical_task = task.loc[task.probe_id == "V1_historical_ssrd_repeatability_failure"].iloc[0]
    hist = exp.loc[exp.probe_id == "V1_historical_ssrd_repeatability_failure"]

    decision = {
        "experiment": "TABICL_EXPLANATION_NUMERICAL_STABILITY_V2",
        "final_conclusion": "NUMERICAL_JITTER_NONNEGLIGIBLE_FOR_SMALL_EFFECTS",
        "v1_frozen_conclusion": "INFERENCE_NONDETERMINISM_CONFIRMED",
        "v1_fold1_status": "FOLD1_NOT_COMPLETE_VALIDATION_FAILURE",
        "v1_threshold": 1e-6,
        "v1_threshold_modified": False,
        "v1_historical_predictions": [6.8550059382, 6.8565043045, 6.8565043045],
        "v1_historical_max_delta": 0.0014983664,
        "probe_count": int(summ.probe_id.nunique()), "repeats_per_probe": sorted(summ.n_repeats.unique().astype(int).tolist()),
        "actual_model_calls": int(raw.actual_model_calls.sum()),
        "all_probe_input_context_checkpoint_feature_and_model_hashes_unique_counts_are_one": bool(
            (summ[["input_hash_nunique", "context_hash_nunique", "checkpoint_hash_nunique",
                   "feature_order_hash_nunique", "model_state_hash_nunique"]] == 1).all().all()),
        "v2_probe_ranges": {str(r.probe_id): float(r.range) for r in summ.itertuples(index=False)},
        "v2_max_range": float(summ.range.max()),
        "v2_max_abs_deviation_from_mean": float(summ.max_abs_deviation_from_mean.max()),
        "v2_max_pairwise_abs_delta": float(summ.max_pairwise_abs_delta.max()),
        "historical_jitter_over_task_scales": {
            "target_iqr": float(historical_task.jitter_over_iqr), "target_sd": float(historical_task.jitter_over_sd),
            "official_rmse": float(historical_task.jitter_over_rmse), "official_mae": float(historical_task.jitter_over_mae)},
        "historical_jitter_vs_explanation_scale_ratio_ranges": {
            f"{method}:{metric}": [float(g.jitter_over_effect_scale.min()), float(g.jitter_over_effect_scale.max())]
            for (method, metric), g in hist.groupby(["method", "effect_scale_type"])},
        "interpretation": "The current 10-call immediate repetitions were bitwise stable for these five points, but the frozen V1 event remains real. Its 0.0014984 target-unit jitter is 34.2% of the smallest observed ALE bin-to-bin change, about 1.6-2.5% of median absolute ALE local-effect scales, 4.97-6.81% of the two-sample median absolute SHAP scales, and under 0.2% of the full ALE curve ranges. This is non-negligible for small effects, but evidence does not show that major rankings or curve shapes change.",
        "shap_repeatability": "SHAP_REPEAT_NOT_AVAILABLE",
        "ale_full_curve_repeatability": "ALE_CURVE_REPEAT_NOT_AVAILABLE; existing single curves were only read",
        "execution_context_caveat": "The new V2 probes immediately repeated each query after rebuilding context. The V1 historical jitter occurred after an ALE workload; the V1 diagnostic probe launcher explicitly set CUBLAS_WORKSPACE_CONFIG=:4096:8, whereas this V2 launcher did not set it (the original full ALE process value was not preserved). No deterministic CUDA mode was enabled in V2. Therefore V2 zero-range observations do not replace or invalidate V1 evidence.",
        "shap_ale_recomputed": False, "other_folds_started": False, "migration_bundle_created": False,
        "split_features_checkpoint_or_prediction_function_changed": False,
    }
    (OUT / "numerical_stability_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TabICLv2 数值稳定性诊断 V2", "",
        f"最终结论：`{decision['final_conclusion']}`。本结论是V2的经验噪声尺度判断，不是新的PASS标准。", "",
        "## V1 冻结失败记录（未更改）", "",
        "- V1结论：`INFERENCE_NONDETERMINISM_CONFIRMED`；fold验收：`FOLD1_NOT_COMPLETE_VALIDATION_FAILURE`。",
        "- 原阈值仍为`1e-6`，未修改；原三次下界预测为`6.8550059382, 6.8565043045, 6.8565043045`，最大差`0.0014983664`。",
        "- 本轮不以V2结果覆盖、重解释或升级V1。", "",
        "## V2 探针设计与结果", "",
        f"固定5个P1 query，每个连续10次单行调用，共{decision['actual_model_calls']}次实测模型调用。每次`X_query.shape=(1,77)`、cache off、无batch；每组的input、feature-order、checkpoint、context、imputer和model-state哈希各自唯一值数均为1。未设置deterministic CUDA、未切CPU、未改dtype或模型函数。",
        "", "| 任务/query | n | 均值 | SD | min | max | range | max pairwise |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summ.itertuples(index=False):
        lines.append(f"| {r.probe_id} | {int(r.n_repeats)} | {r.mean:.12g} | {r.std_sample:.4g} | {r.min:.12g} | {r.max:.12g} | {r.range:.4g} | {r.max_pairwise_abs_delta:.4g} |")
    lines += [
        "", "本轮五组查询经高精度求均值后，range、样本SD、max absolute deviation和max pairwise delta均为0；十次输出逐位相同。",
        "", "## 任务尺度", "",
        "目标统计量基于正式五折OOF预测文件中的全部`y_true`，仅用于尺度比较；RMSE/MAE直接取正式结果表。",
        "", "| 任务 | target | median | IQR | SD | 正式 pooled RMSE | 正式 pooled MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task_name, g in task.loc[task.probe_id != "V1_historical_ssrd_repeatability_failure"].groupby("task", sort=False):
        r = g.iloc[0]
        lines.append(f"| {task_name} | {r.target_name} | {r.target_median:.6g} | {r.target_iqr:.6g} | {r.target_sd:.6g} | {r.official_pooled_rmse:.6g} | {r.official_pooled_mae:.6g} |")
    lines += [
        "", "V2当前五组query的jitter相对上述IQR、SD、RMSE、MAE均为0。另把V1历史非零事件单独保留：该历史抖动/IQR=`{:.4g}`，/SD=`{:.4g}`，/RMSE=`{:.4g}`，/MAE=`{:.4g}`。".format(historical_task.jitter_over_iqr, historical_task.jitter_over_sd, historical_task.jitter_over_rmse, historical_task.jitter_over_mae),
        "", "## 与已完成解释信号的尺度对照", "",
        "只读现有fold1文件；不重跑SHAP/ALE。历史V1抖动相对解释尺度如下（统计范围来自两条SHAP样本和两条ALE曲线）：", "",
        "| 方法/尺度 | effect scale范围 | 历史jitter/effect范围 |", "|---|---:|---:|",
    ]
    for (method, metric), g in hist.groupby(["method", "effect_scale_type"], sort=False):
        lines.append(f"| {method} / {metric} | {g.effect_scale.min():.6g}–{g.effect_scale.max():.6g} | {g.jitter_over_effect_scale.min():.4%}–{g.jitter_over_effect_scale.max():.4%} |")
    lines += [
        "", "其中历史抖动约为最小ALE相邻bin变化的34.2%；相对median absolute ALE local effect为1.60%–2.49%，相对两样本median absolute SHAP为4.97%–6.81%；相对完整ALE曲线range为0.118%–0.152%。这表明小局部效应/弱bin变化可能受影响；没有证据表明主要ALE幅度或主导SHAP贡献排序已改变。",
        "", "## 稳定性范围与限制", "",
        "- SHAP没有合法重复版本：`SHAP_REPEAT_NOT_AVAILABLE`。两条已完成SHAP来自不同样本，不能当作重复归因比较。",
        "- ALE没有重复完整曲线：`ALE_CURVE_REPEAT_NOT_AVAILABLE`；本轮只比较已存曲线尺度。",
        "- 当前V2立即连续重复的五组探针为零range；V1的非零事件发生在不同调用历史/环境记录条件下。V1诊断探针显式设置过`CUBLAS_WORKSPACE_CONFIG=:4096:8`，V2启动器未设置，而原ALE运行进程环境变量没有保存。因此本轮不能据零range否定历史事件，也不能把该样本外推成所有进程状态下的噪声分布。",
        "- 未启动剩余折次、未构建迁移包、未启用确定性CUDA、未修改推理函数或科学结果。",
        "", "## 最终判断", "",
        f"`{decision['final_conclusion']}`：当前测试query的即时重复预测稳定，但被冻结的历史抖动相对较小解释信号并非可忽略；其量级不足以据此声称主要解释结论被推翻，亦没有SHAP重复证据支持排序稳定性结论。V1仍保持失败。", "",
        "数据文件：`probe_manifest.csv`、`p1_repeat_predictions.csv`、`numerical_jitter_summary.csv`、`jitter_vs_task_scale.csv`、`jitter_vs_explanation_scale.csv`。",
    ]
    (OUT / "numerical_stability_decision.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": decision["final_conclusion"], "report": str(OUT / "numerical_stability_decision.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
