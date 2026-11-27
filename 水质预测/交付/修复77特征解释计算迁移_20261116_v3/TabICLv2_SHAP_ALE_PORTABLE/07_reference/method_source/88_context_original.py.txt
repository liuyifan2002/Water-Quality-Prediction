"""TabICLv2十任务单折P1解释：逐任务资格门槛与可恢复计算入口。

旧实验文件只读；每个阶段在独立进程中构造完全相同的冻结训练上下文。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v1"
BASE_PATH = ROOT / "脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py"
ORIGINAL_RUNNER = ROOT / "脚本/05_诊断分析/79_run_all_tabicl_explanation_experiments.py"
PATCH = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
FOLD = "fold_1"
P1 = "TABICLV2_POINTWISE_EXPLANATION_V1"
MAX_ABS = 1e-6
MEAN_ABS = 1e-7
ADD_TOL = 0.02
TASKS = (
    "chla_surface", "chla_middle", "chla_bottom", "secchi",
    "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom",
)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_array(value) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(str(arr.dtype).encode() + str(arr.shape).encode() + arr.tobytes()).hexdigest()


def hash_ids(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base():
    base = load_module(BASE_PATH, "tabicl_slim_base")
    for task in TASKS:
        if task in base.TASKS:
            continue
        family, depth = task.split("_")
        folder = ROOT / f"runs/{family}_family_v2"
        target = {"chla": "chla_ug_l", "tn": "tn_mg_l", "tp": "tp_mg_l"}[family]
        number = {"surface": "03", "middle": "04", "bottom": "05"}[depth]
        base.TASKS[task] = {
            "target": target, "depth": depth,
            "manifest": folder / f"{number}_{task}_manifest.csv",
            "prediction": folder / "predictions" / f"{task}_tabiclv2_{{fold}}.csv",
        }
    return base


def audit_row(task: str) -> pd.Series:
    frame = pd.read_csv(OUT / "00_frozen_protocol_task_registry.csv")
    return frame.loc[frame.task.eq(task)].iloc[0]


def sample_rows(task: str, phase: str = "CORE_3") -> pd.DataFrame:
    frame = pd.read_csv(OUT / "01_preregistered_shap_samples.csv", dtype={"obs_id": "string"})
    return frame.loc[frame.task.eq(task) & frame.phase.eq(phase)].copy()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def context(task: str):
    if str(PATCH) not in sys.path:
        sys.path.insert(0, str(PATCH))
    import torch
    from sklearn.impute import SimpleImputer

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    base = load_base()
    features = base.load_features()
    train, test = base.load_fold(task, FOLD, features)
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imp.fit_transform(train[features]).astype(np.float32)
    x_test = imp.transform(test[features]).astype(np.float32)
    target = base.TASKS[task]["target"]
    target_raw = train[target].to_numpy(float)
    use_log = bool(np.nanmin(target_raw) >= 0)
    y_train = np.log1p(target_raw).astype(np.float32) if use_log else target_raw.astype(np.float32)
    adapter = base.load_model(x_train, y_train)
    estimator = adapter.model
    checkpoint = Path(str(estimator.model_path_))
    ctx_hash = hash_array(estimator.ensemble_generator_.X_) + ":" + hash_array(estimator.ensemble_generator_.y_)
    expected = audit_row(task)
    if hash_ids(features) != expected.feature_order_hash:
        raise RuntimeError("77特征顺序哈希与预登记不一致")
    if hash_array(imp.statistics_) != expected.imputer_statistics_hash:
        raise RuntimeError("逐折median imputer哈希与只读审计不一致")
    if file_sha(checkpoint) != expected.checkpoint_sha256:
        raise RuntimeError("预训练模型权重哈希与预登记不一致")
    view_order = list(estimator.ensemble_generator_.ensemble_configs_.keys())
    if view_order != ["none", "power"]:
        raise RuntimeError(f"预处理视图顺序发生变化: {view_order}")
    if getattr(estimator, "use_amp", None) != "auto":
        raise RuntimeError("模型use_amp配置与冻结的auto设置不同")
    if len(features) != 77 or len(train) != expected.train_n or len(test) != expected.test_n:
        raise RuntimeError("训练/测试样本或特征数量不一致")
    if (task == "chla_surface" and pd.notna(expected.historical_p1_context_hash)
            and ctx_hash != expected.historical_p1_context_hash):
        raise RuntimeError("Chl-a历史P1训练上下文哈希不一致")
    inv = np.expm1 if use_log else (lambda value: value)
    metadata = {
        "task": task, "fold": FOLD, "protocol": P1,
        "feature_order_hash": hash_ids(features),
        "imputer_statistics_hash": hash_array(imp.statistics_),
        "checkpoint_sha256": file_sha(checkpoint),
        "context_hash": ctx_hash,
        "train_obs_order_hash": hash_ids(train.obs_id.astype(str)),
        "test_obs_order_hash": hash_ids(test.obs_id.astype(str)),
        "target_transform": "log1p/expm1" if use_log else "identity",
        "norm_order": json.dumps(view_order),
        "use_amp": getattr(estimator, "use_amp", None),
        "model_state_hash": hashlib.sha256("|".join(
            f"{key}:{hash_array(value.detach().cpu().numpy())}"
            for key, value in sorted(estimator.model_.state_dict().items())).encode()).hexdigest(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "runner_sha256": file_sha(Path(__file__)),
    }
    return base, features, train, test, x_train, x_test, adapter, inv, metadata


def frozen_for_task(base, task: str) -> pd.DataFrame:
    return base.frozen(task, FOLD)


def batch_gate(task: str) -> None:
    expected = audit_row(task)
    if expected.status != "PENDING_REPRODUCTION_GATE":
        raise RuntimeError("只有缺少历史严格复现记录的任务可执行批量复现门槛")
    base, features, train, test, x_train, x_test, adapter, inv, meta = context(task)
    frozen = frozen_for_task(base, task)
    t0 = time.perf_counter()
    prediction = np.maximum(inv(adapter.predict(x_test)), 0)
    aligned = pd.DataFrame({"obs_id": test.obs_id.astype(str), "reproduced": prediction}).merge(
        frozen[["obs_id", "y_pred"]].rename(columns={"y_pred": "frozen"}),
        on="obs_id", validate="one_to_one")
    delta = np.abs(aligned.reproduced.to_numpy(float) - aligned.frozen.to_numpy(float))
    same_order = test.obs_id.astype(str).tolist() == frozen.obs_id.astype(str).tolist()
    passed = bool(same_order and len(aligned) == len(test) and delta.max() <= MAX_ABS and delta.mean() <= MEAN_ABS)
    row = {**meta, "status": "EXACT_REPRODUCTION" if passed else "REPRODUCTION_FAILED",
        "max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean()),
        "obs_order_match": same_order, "n_test": len(test),
        "elapsed_seconds": time.perf_counter() - t0, "accept_max_abs": MAX_ABS,
        "accept_mean_abs": MEAN_ABS, "actual_model_calls": 1,
        "note": "一次正式全测试查询批次，仅用于既定历史接口数值资格核验"}
    atomic_csv(OUT / "gates" / f"{task}_batch_reproduction.csv", pd.DataFrame([row]))
    print(json.dumps({"task": task, "stage": "batch_gate", "status": row["status"],
        "max_abs_error": row["max_abs_error"]}, ensure_ascii=False), flush=True)


def p1_gate(task: str) -> None:
    expected = audit_row(task)
    if expected.status == "BLOCKED":
        raise RuntimeError("预登记fold 1已被历史记录阻断")
    if expected.status == "PENDING_REPRODUCTION_GATE":
        gate_path = OUT / "gates" / f"{task}_batch_reproduction.csv"
        if not gate_path.is_file() or pd.read_csv(gate_path).status.iloc[0] != "EXACT_REPRODUCTION":
            raise RuntimeError("尚未通过该任务fold 1历史接口全样本复现资格门槛")
    base, features, train, test, x_train, x_test, adapter, inv, meta = context(task)
    frozen = frozen_for_task(base, task).set_index("obs_id")
    index = {str(obs): i for i, obs in enumerate(test.obs_id.astype(str))}
    rows = []
    for sample in sample_rows(task).itertuples():
        x = x_test[index[str(sample.obs_id)]:index[str(sample.obs_id)] + 1]
        if x.shape != (1, 77) or not np.isfinite(x).all():
            raise RuntimeError(f"{sample.obs_id}: P1输入维度或有限值检查失败")
        baseline = float(frozen.loc[str(sample.obs_id), "y_pred"])
        predictions = []
        for repeat in (1, 2, 3):
            start = time.perf_counter()
            result = np.asarray(adapter.predict(x), float).reshape(-1)
            if result.size != 1 or not np.isfinite(result[0]):
                raise RuntimeError(f"{sample.obs_id}: P1输出维度或有限值检查失败")
            value = float(max(float(inv(result)[0]), 0.0))
            predictions.append(value)
            rows.append({**meta, "obs_id": str(sample.obs_id), "prediction_quantile": sample.target_quantile,
                "repeat": repeat, "query_shape": "1x77", "query_hash": hash_array(x),
                "p1_prediction": value, "frozen_batch_prediction": baseline,
                "absolute_difference": abs(value - baseline), "elapsed_seconds": time.perf_counter() - start,
                "actual_model_calls": 1})
        if max(predictions) - min(predictions) > MAX_ABS:
            print(f"{task}/{sample.obs_id}: P1重复性超过1e-6", flush=True)
    frame = pd.DataFrame(rows)
    atomic_csv(OUT / "gates" / f"{task}_p1_probes.csv", frame)
    max_frozen = float(frame.absolute_difference.max())
    mean_frozen = float(frame.absolute_difference.mean())
    max_repeat = float(frame.groupby("obs_id").p1_prediction.agg(lambda s: s.max() - s.min()).max())
    passed = max_frozen <= MAX_ABS and mean_frozen <= MEAN_ABS and max_repeat <= MAX_ABS
    status = "P1_GATE_PASS" if passed else "P1_GATE_FAILED"
    atomic_csv(OUT / "gates" / f"{task}_p1_decision.csv", pd.DataFrame([{
        **meta, "status": status, "n_queries": len(frame) // 3,
        "actual_model_calls": len(frame), "max_p1_vs_frozen_abs": max_frozen,
        "mean_p1_vs_frozen_abs": mean_frozen, "max_p1_repeat_range": max_repeat,
        "max_threshold": MAX_ABS, "mean_threshold": MEAN_ABS,
        "reason": "固定查询数值核验通过" if passed else "固定查询偏离冻结批量预测或重复性超标；正式解释阻断",
    }]))
    print(json.dumps({"task": task, "stage": "p1_gate", "status": status,
        "max_p1_vs_frozen_abs": max_frozen, "mean_p1_vs_frozen_abs": mean_frozen,
        "max_p1_repeat_range": max_repeat}, ensure_ascii=False), flush=True)


def gate_all() -> None:
    table = pd.read_csv(OUT / "00_frozen_protocol_task_registry.csv")
    for record in table.itertuples():
        task = record.task
        if record.status == "BLOCKED":
            continue
        if record.status == "PENDING_REPRODUCTION_GATE":
            path = OUT / "gates" / f"{task}_batch_reproduction.csv"
            if not path.is_file():
                run_child("--batch-gate", task)
            if not path.is_file():
                continue
            if pd.read_csv(path).status.iloc[0] != "EXACT_REPRODUCTION":
                continue
        path = OUT / "gates" / f"{task}_p1_decision.csv"
        if not path.is_file():
            run_child("--p1-gate", task)


def run_child(flag: str, task: str) -> None:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env["PYTHONPATH"] = str(PATCH) + os.pathsep + env.get("PYTHONPATH", "")
    path = OUT / "logs" / f"{task}_{flag.lstrip('-')}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        completed = subprocess.run([sys.executable, str(Path(__file__)), flag, task],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        print(json.dumps({"task": task, "stage": flag, "exit_code": completed.returncode,
            "log": str(path)}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-gate", choices=TASKS)
    parser.add_argument("--p1-gate", choices=TASKS)
    parser.add_argument("--gate-all", action="store_true")
    args = parser.parse_args()
    if args.batch_gate:
        batch_gate(args.batch_gate)
    elif args.p1_gate:
        p1_gate(args.p1_gate)
    elif args.gate_all:
        gate_all()
    else:
        parser.error("必须指定资格核验阶段")


if __name__ == "__main__":
    main()
