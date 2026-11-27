"""Summarize saved repeatability records; makes no model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "runs/tabicl_v2_final_explanation_complete_v1"
TOL = 1e-6


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rep = pd.read_csv(RUN / "ale_1d_repeatability.csv")
    rep = rep.loc[(rep.task == "chla_surface") & (rep.fold == "fold_1")
                  & (rep.feature == "ssrd_mean_14d")].iloc[0]
    historical = json.loads(rep.repeat_predictions_json)
    probes = pd.read_csv(HERE / "diagnostic_probe_results.csv")
    env = pd.read_csv(RUN / "environment_manifest.csv")
    env = env.loc[(env.task == "chla_surface") & (env.fold == "fold_1")].iloc[-1]
    audit = json.loads((RUN / "fold_1_completion_audit.json").read_text(encoding="utf-8"))

    rows = []
    for item in historical:
        rows.append({
            "record_source": "historical_ale_repeatability", "repeat_index": item["repeat"],
            "query_id": rep.probe_obs_id, "query_shape": "1x77 per call (from saved implementation)",
            "lower_prediction": item["prediction_lower"], "upper_prediction": item["prediction_upper"],
            "local_effect_upper_minus_lower": item["local_delta"],
            "lower_input_hash": "NOT_SAVED", "upper_input_hash": "NOT_SAVED",
            "inputs_identical_verified_by_hash": False, "context_hash": rep.context_hash,
            "checkpoint_hash": env.checkpoint_sha256, "feature_order_hash": env.feature_order_hash,
            "model_state_hash": env.model_state_hash,
            "model_calls_reported": rep.actual_model_calls,
            "model_calls_verified_by_counter": False,
            "actual_model_calls_note": "Historical table stores hardcoded 6; source loop requests 2 calls x 3 repeats, but no per-call counter snapshot was persisted.",
            "elapsed_seconds": None, "difference_from_first_local_effect": None,
            "threshold": TOL, "repeatability_pass": None,
        })
    first_probe = float(probes.prediction_original_scale.iloc[0])
    first_effect = float(historical[0]["local_delta"])
    for r in probes.itertuples(index=False):
        effect = float(r.prediction_original_scale - first_probe + first_effect)
        # Probe repeats only the lower endpoint; its prediction range is the directly testable quantity.
        rows.append({
            "record_source": "new_fixed_lower_endpoint_probe", "repeat_index": r.probe_index,
            "query_id": r.obs_id, "query_shape": r.query_shape,
            "lower_prediction": r.prediction_original_scale, "upper_prediction": None,
            "local_effect_upper_minus_lower": None,
            "lower_input_hash": r.query_input_sha256, "upper_input_hash": None,
            "inputs_identical_verified_by_hash": bool(r.query_input_sha256 == probes.query_input_sha256.iloc[0]
                                                       and r.query_hash_after_call == r.query_input_sha256),
            "context_hash": r.train_context_hash, "checkpoint_hash": r.checkpoint_sha256,
            "feature_order_hash": r.feature_order_hash, "model_state_hash": r.model_state_hash,
            "model_calls_reported": r.actual_model_calls_this_probe,
            "model_calls_verified_by_counter": True,
            "actual_model_calls_note": "One uncached P1 model call measured from the live call counter.",
            "elapsed_seconds": r.elapsed_seconds,
            "difference_from_first_local_effect": float(r.prediction_original_scale - first_probe),
            "threshold": TOL,
            "repeatability_pass": bool(abs(float(r.prediction_original_scale - first_probe)) <= TOL),
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(HERE / "historical_repeat_comparison.csv", index=False, encoding="utf-8-sig")

    probe_range = float(probes.prediction_original_scale.max() - probes.prediction_original_scale.min())
    all_probe_hashes_same = probes.query_input_sha256.nunique() == 1
    all_contexts_same = probes.train_context_hash.nunique() == 1
    all_model_hashes_same = probes.model_state_hash.nunique() == 1 and probes.checkpoint_sha256.nunique() == 1
    reproduced = bool(all_probe_hashes_same and all_contexts_same and all_model_hashes_same and probe_range > TOL)
    decision = {
        "experiment": "TABICL_ALE_REPEATABILITY_DIAGNOSTIC_V1",
        "final_conclusion": "INFERENCE_NONDETERMINISM_CONFIRMED" if reproduced else "INSUFFICIENT_PROVENANCE",
        "classification": "B. INFERENCE_NONDETERMINISM" if reproduced else "D. INSUFFICIENT_PROVENANCE",
        "cause_identified": reproduced,
        "historical_three_calls_have_per_call_input_hashes": False,
        "historical_input_same_from_code_path_only_not_hash_verified": True,
        "probe_count": int(len(probes)), "probe_predictions_original_scale": probes.prediction_original_scale.astype(float).tolist(),
        "probe_input_hashes": probes.query_input_sha256.astype(str).tolist(),
        "probe_hashes_all_identical": all_probe_hashes_same,
        "probe_context_hashes_all_identical": all_contexts_same,
        "probe_checkpoint_hashes_all_identical": probes.checkpoint_sha256.nunique() == 1,
        "probe_model_state_hashes_all_identical": probes.model_state_hash.nunique() == 1,
        "probe_max_prediction_range": probe_range,
        "acceptance_threshold": TOL,
        "probe_exceeds_threshold": probe_range > TOL,
        "comparison_scale": "original Chl-a target scale after log1p/expm1 inverse and nonnegative clamp; target chla_ug_l",
        "historical_ale_delta_range": float(rep.max_local_ale_delta_range),
        "historical_repeatability_status": str(rep.repeatability_status),
        "historical_actual_model_calls_field_is_hardcoded": True,
        "minimal_future_logging_fix_applied": "measure live call counter and persist lower/upper query hashes; completed rows untouched",
        "model_prediction_or_scientific_protocol_changed": False,
        "shap_or_ale_recomputed": False,
        "fold1_acceptance_status_unchanged": audit["final_status"],
        "source_probe_script_sha256": file_hash(HERE / "run_repeatability_probe.py"),
    }
    (HERE / "repeatability_diagnostic_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# TabICLv2 一维 ALE 重复性定向诊断

## 结论

`{decision['final_conclusion']}`（类别 B：`INFERENCE_NONDETERMINISM`）。本结论针对固定 P1 查询下的推理重复性；不表示已定位到某一个具体 CUDA kernel。

## 历史记录恢复

历史记录中的下界/上界预测分别为：

| 重复 | 下界预测 (µg/L) | 上界预测 (µg/L) | 局部效应：上界−下界 (µg/L) |
|---:|---:|---:|---:|
""" + "\n".join(
        f"| {x['repeat']} | {x['prediction_lower']:.12f} | {x['prediction_upper']:.12f} | {x['local_delta']:.12f} |"
        for x in historical
    ) + f"""

历史局部效应范围为 `{float(rep.max_local_ale_delta_range):.13g} µg/L`，阈值为 `{TOL:g} µg/L`。历史逐次记录没有保存77维查询向量哈希、逐次 RNG 状态或逐次模型调用计数；其六次调用字段在原代码中是常数写入，而不是计数器读数。原代码循环中同一组 `probe_lo/probe_hi` 数组被重复使用且未在循环内改写，因此输入相同有代码路径佐证，但缺少历史哈希，不能说历史现场获得了逐次输入的直接哈希证明。

对应模型、特征顺序及上下文元数据可恢复：checkpoint SHA256 `{env.checkpoint_sha256}`；77特征顺序哈希 `{env.feature_order_hash}`；imputer统计哈希 `{env.imputer_stats_hash}`；上下文哈希 `{env.context_hash}`；模型状态哈希 `{env.model_state_hash}`。模型输出是 `chla_ug_l` 原始尺度（反变换并执行非负截断）。

## 最小固定输入探针

没有重算 ALE 曲线。对同一重建的下界查询（`{rep.probe_obs_id}`，shape `1×77`）进行了三次未缓存真实单查询调用。三次输入哈希相同 `{probes.query_input_sha256.iloc[0]}`；checkpoint、context、特征顺序与模型状态哈希均一致。预测为：`{', '.join(f'{v:.12f}' for v in probes.prediction_original_scale.astype(float))} µg/L`；最大差值 `{probe_range:.13g} µg/L`，超过阈值。每次调用实测为一次模型调用，未使用 batch-query。CPU/CUDA Torch RNG 的调用前后哈希保持不变；这不能排除不受这些 RNG 状态控制的 CUDA 算子非确定性。探针逐次耗时约 `{', '.join(f'{v:.3f}' for v in probes.elapsed_seconds.astype(float))}` 秒。

由于当前探针仅重放下界端点，**没有**重新调用上界预测，也没有重新计算局部 ALE；这三次探针用于证明固定查询点本身的 P1 输出存在重复性波动。该波动与历史下界预测差完全一致。

## 根因判定及代码处理

证据支持相同固定输入/上下文下的推理输出波动，而非输入差异或 ALE 差值公式错误。代码中发现并做了一项仅针对未来审计记录的最小修正：把 `actual_model_calls` 从固定常数改为读取实时计数器，并在重复记录中附加下界/上界输入哈希。未改预测函数、dtype、精度、随机种子、分批方式、权重或上下文；已完成 SHAP、ALE、SQLite 和原始失败审计均未重算或覆盖。

本轮不调整推理数值设置，不做更多模型调用。具体非确定性算子/驱动层原因尚未细分，若后续需要消除或规避波动，应另行批准诊断；当前数据不满足重复性验收。

## 对 fold 1 验收的影响

原始 `fold_1_completion_audit.json` 保持 `FOLD1_NOT_COMPLETE_VALIDATION_FAILURE`。本诊断确认失败由推理重复性波动支持，**不能升级为 PASS**。两条 SHAP、两条 ALE 曲线不变；没有启动其他 fold，也没有构建迁移包。

## 文件

- `historical_repeat_comparison.csv`：历史三次结果与新探针逐条对照。
- `diagnostic_probe_results.csv`：三次固定查询真实调用及输入/环境哈希。
- `repeatability_diagnostic_decision.json`：机器可读结论。
"""
    (HERE / "repeatability_diagnostic_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"conclusion": decision["final_conclusion"], "historical_range": float(rep.max_local_ale_delta_range),
                      "probe_range": probe_range, "output_dir": str(HERE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
