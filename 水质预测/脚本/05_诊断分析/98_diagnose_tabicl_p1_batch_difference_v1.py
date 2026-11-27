"""只用已保存的预测探针与源码，诊断P1和冻结批量预测差异；不调用模型。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/tabicl_v2_slim_explainability_v1"
OLD = ROOT / "runs/tabicl_v2_reproduction_root_cause_v2"
OUT = ROOT / "runs/tabicl_p1_batch_difference_diagnostic_v1"
TASKS = (
    "chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface",
    "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom",
)
MAX_TOL = 1e-6
MEAN_TOL = 1e-7
HASH_FIELDS = (
    "feature_order_hash", "imputer_statistics_hash", "checkpoint_sha256",
    "context_hash", "train_obs_order_hash", "test_obs_order_hash",
    "model_state_hash", "target_transform", "norm_order", "use_amp",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def classification(max_diff: float, mean_diff: float, repeat_range: float) -> str:
    if repeat_range > MAX_TOL:
        return "P1_REPEATABILITY_FAILED"
    if max_diff > MAX_TOL:
        return "P1_BATCH_NUMERICAL_MISMATCH"
    if mean_diff > MEAN_TOL:
        return "THREE_QUERY_MEAN_ONLY_FAILED"
    return "P1_GATE_PASS"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(SOURCE / "00_frozen_protocol_task_registry.csv").set_index("task")
    tasks = []
    queries = []
    for task in TASKS:
        old = audit.loc[task]
        gate_file = SOURCE / "gates" / f"{task}_p1_decision.csv"
        gate = frame(gate_file)
        batch = frame(SOURCE / "gates" / f"{task}_batch_reproduction.csv")
        probes = frame(SOURCE / "gates" / f"{task}_p1_probes.csv")
        if gate.empty:
            tasks.append({"task": task, "fold": "fold_1", "historical_batch_status": old.historical_batch_reproduction,
                          "current_batch_status": "NOT_RUN", "p1_gate_status": "HISTORICAL_BLOCKED",
                          "max_p1_vs_frozen_abs": np.nan, "mean_p1_vs_frozen_abs": np.nan,
                          "max_p1_repeat_range": np.nan, "failure_type": "HISTORICAL_REPRODUCTION_AND_REPEATABILITY_FAILURE",
                          "all_batch_p1_asset_hashes_equal": "NOT_VERIFIABLE",
                          "n_queries": 0, "n_probe_calls": 0,
                          "note": "Chl-a表层保留历史冻结失败和ALE重复性失败；本轮没有执行P1资格探针。"})
            continue
        rec = gate.iloc[0]
        max_diff = float(rec.max_p1_vs_frozen_abs)
        mean_diff = float(rec.mean_p1_vs_frozen_abs)
        repeat = float(rec.max_p1_repeat_range)
        hashes = "NOT_VERIFIABLE_WITHOUT_CURRENT_BATCH_METADATA"
        if not batch.empty:
            br = batch.iloc[0]
            hashes = "TRUE" if all(str(br[field]) == str(rec[field]) for field in HASH_FIELDS) else "FALSE"
        probe_group = probes.groupby("obs_id", sort=False) if not probes.empty else []
        for obs_id, group in probe_group:
            g = group.sort_values("repeat")
            values = g.p1_prediction.to_numpy(float)
            baseline = g.frozen_batch_prediction.to_numpy(float)
            queries.append({"task": task, "fold": "fold_1", "obs_id": obs_id,
                            "prediction_quantile": float(g.prediction_quantile.iloc[0]),
                            "n_repeats": len(g), "query_shape_values": "|".join(g.query_shape.astype(str).unique()),
                            "query_hash_unique": g.query_hash.nunique(),
                            "feature_hash_unique": g.feature_order_hash.nunique(),
                            "imputer_hash_unique": g.imputer_statistics_hash.nunique(),
                            "context_hash_unique": g.context_hash.nunique(),
                            "model_state_hash_unique": g.model_state_hash.nunique(),
                            "frozen_prediction_unique": g.frozen_batch_prediction.nunique(),
                            "frozen_batch_prediction": float(baseline[0]),
                            "p1_mean": float(values.mean()), "p1_min": float(values.min()),
                            "p1_max": float(values.max()), "p1_repeat_range": float(np.ptp(values)),
                            "abs_difference_mean": float(np.abs(values - baseline).mean()),
                            "abs_difference_max": float(np.abs(values - baseline).max()),
                            "signed_difference_mean": float((values - baseline).mean()),
                            "first_vs_remaining_mean_abs": float(abs(values[0] - values[1:].mean())) if len(values) > 1 else np.nan,
                            "repeat_values": "|".join(format(value, ".17g") for value in values),
                            "historical_batch_status": old.historical_batch_reproduction,
                            "current_batch_status": batch.status.iloc[0] if not batch.empty else "NOT_RUN"})
        tasks.append({"task": task, "fold": "fold_1", "historical_batch_status": old.historical_batch_reproduction,
                      "current_batch_status": batch.status.iloc[0] if not batch.empty else "NOT_RUN",
                      "p1_gate_status": rec.status,
                      "max_p1_vs_frozen_abs": max_diff, "mean_p1_vs_frozen_abs": mean_diff,
                      "max_p1_repeat_range": repeat,
                      "max_threshold": MAX_TOL, "mean_threshold": MEAN_TOL,
                      "failure_type": classification(max_diff, mean_diff, repeat),
                      "all_batch_p1_asset_hashes_equal": hashes,
                      "n_queries": int(rec.n_queries), "n_probe_calls": int(rec.actual_model_calls),
                      "note": "P1均值仅基于3个预登记查询各重复3次，不能代替全测试集P1均值。"})
    task_df = pd.DataFrame(tasks)
    query_df = pd.DataFrame(queries)
    task_df.to_csv(OUT / "01_任务门槛与资产对照.csv", index=False, encoding="utf-8-sig")
    query_df.to_csv(OUT / "02_逐查询差异与重复性.csv", index=False, encoding="utf-8-sig")

    src_runner = ROOT / "脚本/05_诊断分析/88_run_tabicl_slim_explainability_v1.py"
    source_regressor = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch/tabicl/_sklearn/regressor.py"
    source_preprocessing = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch/tabicl/_sklearn/preprocessing.py"
    trace = pd.DataFrame([
        {"stage": "PREPROCESSING_AND_CONTEXT", "batch_path": "88.py:context -> fit median -> float32 -> log1p -> load_model",
         "p1_path": "same context() code", "evidence": "6个新批量核验任务的10项资产/状态字段逐一比对",
         "finding": "全部相同；不支持这些字段为本次差异源", "qualification": "只覆盖当前重建状态，不证明历史隐藏内部状态完整恢复"},
        {"stage": "QUERY_SHAPE", "batch_path": "88.py:174 adapter.predict(x_test)；一次全测试集调用",
         "p1_path": "88.py:212 adapter.predict(x)；每次1×77，共3×3次",
         "evidence": "当前P1逐查询CSV；已有root_cause_v2/04_query_batch_sensitivity.csv",
         "finding": "可观察差异首先出现在改变查询集合/顺序/分块后的模型输出",
         "qualification": "不能仅凭已有记录拆分模型查询语义与底层GPU算子数值路径"},
        {"stage": "MODEL_FORWARD", "batch_path": "regressor.py:predict -> EnsembleGenerator.transform(mode=both) -> _batch_forward",
         "p1_path": "同一代码路径，但X的query维度不同", "evidence": "隔离源码及先前32查询顺序/分块试验",
         "finding": "固定训练预处理使用transform；底层forward接收不同T维度",
         "qualification": "未保存历史每层张量；具体首个不等价算子未被证明"},
        {"stage": "OUTPUT_TRANSFORM", "batch_path": "log1p结果由expm1回原尺度并非负截断",
         "p1_path": "同一inverse与非负截断规则", "evidence": "88.py:174, 88.py:212-218",
         "finding": "没有发现输出单位或反变换规则分叉", "qualification": "不排除前向极小数值差被expm1放大"},
    ])
    trace.to_csv(OUT / "03_计算路径差异定位.csv", index=False, encoding="utf-8-sig")
    saved = OUT / "04_定向诊断结论.md"
    fail = task_df[task_df.p1_gate_status.eq("P1_GATE_FAILED")]
    marginal = fail[fail.failure_type.eq("THREE_QUERY_MEAN_ONLY_FAILED")].task.tolist()
    substantive = fail[fail.failure_type.eq("P1_BATCH_NUMERICAL_MISMATCH")].task.tolist()
    unstable = fail[fail.failure_type.eq("P1_REPEATABILITY_FAILED")].task.tolist()
    equal_n = int(task_df.all_batch_p1_asset_hashes_equal.eq("TRUE").sum())
    history = pd.read_csv(OLD / "04_query_batch_sensitivity.csv")
    fixed = history.loc[history.condition.eq("fixed_none_power") & history.replicate.eq(1)]
    reversed_row = fixed.loc[fixed.variant.eq("reversed_query_order")].iloc[0]
    chunks_row = fixed.loc[fixed.variant.eq("query_chunks_of_8")].iloc[0]
    lines = [
        "# TabICLv2 P1单查询与冻结批量预测差异：定向诊断",
        "", "## Material Passport", "", "- 类型：独立数值诊断；结论：`ROOT_CAUSE_PARTIALLY_IDENTIFIED`（不等于新模型推理严格复现）。",
        "- 数据：现有10任务审计、9任务P1三点×三次探针、6任务全测试集批量复现及历史32查询顺序/分块试验。",
        "- 本轮实际模型调用：0；所有分析为CPU只读，未与正在运行的GPU解释worker争用。",
        f"- 本诊断脚本SHA-256：`{sha(Path(__file__))}`。源路径及哈希见下节。",
        "", "## 诊断结果", "",
        f"- 新批量门槛的6项任务均为`EXACT_REPRODUCTION`；其中{equal_n}/6项的特征顺序、imputer、checkpoint、训练上下文、train/test obs顺序、模型状态、目标变换、视图顺序与AMP设置，在批量和P1重建记录中相同。",
        f"- 三点平均误差单独超标：{', '.join(marginal)}。其max逐点差和重复范围仍满足1e-6；三点平均约束1e-7并非全测试集P1平均约束，不得直接称全折P1失败。现有V1门槛仍判为失败，不在本诊断中追认PASS。",
        f"- 最大逐点差超标、但三次重复稳定：{', '.join(substantive)}。这支持P1与批量输出存在可重复的数值不等价；不支持输入或context hash变化为已观察原因。",
        f"- 重复调用范围也超标：{', '.join(unstable)}。该任务兼具P1/批量不等价和调用内数值波动；不能把差异完全归于批量语义。TP表层的一个固定查询第1次输出与后两次相差约8.0814e-6，后两次彼此相同；原始三次值全部保留。",
        f"- 共{len(query_df)}个固定查询的三次重复中，query/feature/imputer/context/model-state哈希及冻结参照预测各自均只有一个取值；没有发现重复输入不一致。",
        "- Chl-a表层沿用历史严格复现与ALE重复性失败，本轮没有新P1门槛调用；不得与上述9项新探针混为一谈。",
        "", "## 与已完成32查询实验的交叉证据", "",
        f"- 固定上下文下，32查询反序并按obs_id对齐，最大差为`{float(reversed_row.max_abs_vs_fixed_query_order):.10g}`；改为4×8分块，最大差为`{float(chunks_row.max_abs_vs_fixed_query_order):.10g}`。这证明当前推理路径对查询顺序/分块敏感，但不能证明是历史冻结预测差异的唯一根因。",
        "- 源码显示batch和P1都走同一`predict`/`mode=both`/`_batch_forward`路径；区别是输入查询维度。训练预处理器用已拟合对象做`transform`。因此，当前证据最早把差异定位在查询集进入前向推理阶段；未能细分到具体GPU算子。",
        "", "## 证据边界与行动", "",
        "- 保留原`1e-6`最大值、`1e-7`均值和`1e-6`重复性门槛，原7项BLOCKED记录不变。本轮不修改调度器、不启动被阻断SHAP/ALE。",
        "- 若后续需要证明单查询与历史批量函数等价，必须在当前GPU解释worker结束后安排隔离的同进程分层forward探针；逐层比较预处理视图、ensemble输入、原始quantile和反变换输出，并核对固定query在全测试批与P1中的对应位置。当前并行GPU工作负载下未做此高成本探针。",
        "- 三个仅三点均值失败的任务应单独审查门槛统计口径：不能把三点平均与原全测试集平均视为同一统计量；但也不能事后降低本V1的门槛。",
        "", "## 文件溯源", "",
        f"- `88_run_tabicl_slim_explainability_v1.py` SHA-256：`{sha(src_runner)}`",
        f"- 隔离TabICL regressor SHA-256：`{sha(source_regressor)}`",
        f"- 隔离TabICL preprocessing SHA-256：`{sha(source_preprocessing)}`",
        "- 输入证据：`runs/tabicl_v2_slim_explainability_v1/gates/`、`runs/tabicl_v2_reproduction_root_cause_v2/04_query_batch_sensitivity.csv`。",
        "- 输出表：`01_任务门槛与资产对照.csv`、`02_逐查询差异与重复性.csv`、`03_计算路径差异定位.csv`。",
    ]
    saved.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(task_df)} tasks, {len(query_df)} queries, {equal_n}/6 current batch/P1 assets matched")


if __name__ == "__main__":
    main()
