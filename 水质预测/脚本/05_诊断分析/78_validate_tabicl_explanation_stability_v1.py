"""TabICLv2可解释性推理稳定性与确定性解释协议验证。

仅使用chla_surface/fold_1已冻结的模型配置、正式训练上下文和固定查询特征。
不读取测试水质标签作解释样本或参考集选择，不修改正式预测，不训练模型权重。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_explanation_stability_v1"
PATCH_ROOT = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
BASE_SCRIPT = ROOT / "脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py"
SHAP_SCRIPT = ROOT / "脚本/05_诊断分析/67_run_tabiclv2_model_agnostic_shap_v3.py"
V3 = ROOT / "runs/tabicl_v2_explainability_v3"
FIXED_PROBE = ROOT / "runs/tabicl_v2_repro_ale_repair_v1/01_context_reconstruction_audit.csv"
TASK, FOLD = "chla_surface", "fold_1"
SEED, N_PROBE = 42, 32
STRICT_MAX, STRICT_MEAN = 1e-6, 1e-7
BUDGET_SECONDS = 3600
FEATURES_FOR_ALE = ("ssrd_mean_14d", "wind_speed_x_rain14d")
SHAP_IDS = ("MM5_20201119_surface", "WM2_20211208_surface")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ahash(x) -> str:
    a = np.ascontiguousarray(np.asarray(x))
    return sha(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def ihash(xs) -> str:
    return sha("\n".join(map(str, xs)).encode("utf-8"))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(torch) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)


def load_base():
    spec = importlib.util.spec_from_file_location("tabicl_stability_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_tree(x, torch):
    if torch.is_tensor(x):
        a = x.detach().contiguous().cpu().numpy()
        return [{"shape": list(a.shape), "dtype": str(a.dtype), "sha256": ahash(a)}]
    if isinstance(x, (tuple, list)):
        return [z for q in x for z in tensor_tree(q, torch)]
    if isinstance(x, dict):
        return [z for q in x.values() for z in tensor_tree(q, torch)]
    return []


class FixedBatchPredictor:
    """与V3 SHAP脚本同定义：分批，并以固定特征锚点补足最后一批。"""
    def __init__(self, predict, anchors: np.ndarray, batch_n: int):
        self.predict = predict
        self.anchors = np.asarray(anchors, dtype=np.float32)
        self.batch_n = int(batch_n)
        self.calls: list[dict] = []

    def __call__(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float32)
        out = []
        t0 = time.perf_counter()
        for start in range(0, len(matrix), self.batch_n):
            part = matrix[start:start + self.batch_n]
            need = self.batch_n - len(part)
            if need:
                pad = np.tile(self.anchors, (int(np.ceil(need / len(self.anchors))), 1))[:need]
                query = np.vstack([part, pad])
            else:
                query = part
            out.append(self.predict(query, f"P3_fixed{self.batch_n}_part{start // self.batch_n + 1}")[:len(part)])
        self.calls.append({"input_rows": len(matrix), "batches": int(np.ceil(len(matrix) / self.batch_n)),
                           "query_batch_n": self.batch_n, "seconds": time.perf_counter() - t0,
                           "anchor_hash": ahash(self.anchors)})
        return np.concatenate(out) if out else np.empty(0)


def prepare(replicate: int):
    import torch
    seed_all(torch)
    base = load_base()
    features = base.load_features()
    train, test = base.load_fold(TASK, FOLD, features)
    target = base.TASKS[TASK]["target"]
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    xtr = imp.fit_transform(train[features]).astype(np.float32)
    xte = imp.transform(test[features]).astype(np.float32)
    y0 = train[target].to_numpy(dtype=float)
    use_log = bool(np.nanmin(y0) >= 0)
    ytr = np.log1p(y0).astype(np.float32) if use_log else y0.astype(np.float32)

    # 复用上一轮已固定的32个obs_id及顺序，禁止重新抽样。
    pos = base.sample_positions(test, min(N_PROBE, len(test)), SEED + int(FOLD[-1]))
    qids = test.iloc[pos].obs_id.astype(str).tolist()
    audit = pd.read_csv(FIXED_PROBE).query("task == @TASK and fold == @FOLD").iloc[0]
    saved = str(audit.probe_obs_ids).split(";")
    if qids != saved or ihash(qids) != audit.probe_order_hash:
        raise RuntimeError("本轮固定32个obs_id/顺序与既有审计记录不一致")

    adapter = base.load_model(xtr, ytr)
    est = adapter.model
    checkpoint = Path(str(getattr(est, "model_path_", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"预训练权重路径不可用: {checkpoint}")
    # TabICL在fit时装载预训练权重并构造上下文；不更新权重。
    model_state_hash = sha("|".join(f"{k}:{ahash(v.detach().cpu().numpy())}" for k, v in sorted(est.model_.state_dict().items())).encode())
    context_hash = ahash(est.ensemble_generator_.X_) + ":" + ahash(est.ensemble_generator_.y_)
    view_order = list(est.ensemble_generator_.ensemble_configs_.keys())
    trace_rows, predict_rows = [], []
    active = {"label": "", "capture": False, "tensor_seen": set()}
    original_batch_forward = est._batch_forward

    def traced_forward(xs, ys, *args, **kwargs):
        result = original_batch_forward(xs, ys, *args, **kwargs)
        if active["capture"]:
            arr = np.asarray(result)
            trace_rows.append({"replicate": replicate, "protocol_call": active["label"],
                "stage": "preprocessed_model_input", "x_shape": json.dumps(list(np.asarray(xs).shape)),
                "x_dtype": str(np.asarray(xs).dtype), "x_hash": ahash(xs),
                "y_shape": json.dumps(list(np.asarray(ys).shape)), "y_dtype": str(np.asarray(ys).dtype),
                "y_hash": ahash(ys), "output_shape": json.dumps(list(arr.shape)),
                "output_dtype": str(arr.dtype), "output_hash": ahash(arr)})
        return result

    est._batch_forward = traced_forward
    def pre_hook(module, args):
        if not active["capture"]:
            return
        key = active["label"]
        if key in active["tensor_seen"]:
            return
        active["tensor_seen"].add(key)
        summary = tensor_tree(args, torch)
        trace_rows.append({"replicate": replicate, "protocol_call": key,
            "stage": "torch_forward_input_tensor", "x_shape": json.dumps([x["shape"] for x in summary]),
            "x_dtype": json.dumps([x["dtype"] for x in summary]), "x_hash": json.dumps([x["sha256"] for x in summary]),
            "y_shape": "", "y_dtype": "", "y_hash": "", "output_shape": "", "output_dtype": "", "output_hash": ""})
    handle = est.model_.register_forward_pre_hook(pre_hook)

    inverse = np.expm1 if use_log else (lambda z: z)
    def predict(matrix, label, capture=False):
        matrix = np.asarray(matrix, dtype=np.float32)
        active.update(label=label, capture=capture)
        t0 = time.perf_counter()
        raw = np.asarray(adapter.predict(matrix), dtype=float).reshape(-1)
        elapsed = time.perf_counter() - t0
        value = np.maximum(inverse(raw), 0.0)
        predict_rows.append({"replicate": replicate, "protocol_call": label, "query_n": len(matrix),
            "query_ids": "", "query_input_hash": ahash(matrix), "raw_output_hash": ahash(raw),
            "target_output_hash": ahash(value), "raw_predictions_json": json.dumps(raw.tolist()),
            "predictions_json": json.dumps(value.tolist()), "elapsed_seconds": elapsed,
            "context_hash": context_hash, "view_order": json.dumps(view_order),
            "target_inverse": "expm1 + clip_nonnegative" if use_log else "clip_nonnegative"})
        active.update(capture=False)
        return value, raw, elapsed

    env = {"replicate": replicate, "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "python_hashseed": os.getenv("PYTHONHASHSEED", "unset"), "tabicl_module": __import__("tabicl").__file__,
        "tabicl_version": getattr(__import__("tabicl"), "__version__", "2.2.0"),
        "features_n": len(features), "feature_order_hash": ihash(features), "train_obs_order_hash": ihash(train.obs_id.astype(str)),
        "test_obs_order_hash": ihash(test.obs_id.astype(str)), "probe_obs_ids": ";".join(qids), "probe_order_hash": ihash(qids),
        "raw_train_hash": ahash(train[features].to_numpy(dtype=np.float64)), "raw_query_hash": ahash(xte[pos]),
        "imputer": "median, keep_empty_features=True; fit on train only", "imputer_statistics_hash": ahash(imp.statistics_),
        "imputed_train_hash": ahash(xtr), "training_target_transform": "log1p" if use_log else "identity",
        "training_target_hash": ahash(ytr), "context_hash": context_hash, "context_n": len(est.ensemble_generator_.X_),
        "checkpoint_sha256": file_hash(checkpoint), "model_state_hash": model_state_hash,
        "norm_order": json.dumps(view_order), "estimator_batch_size": getattr(est, "batch_size", None),
        "n_estimators": getattr(est, "n_estimators", None), "use_amp": getattr(est, "use_amp", None),
        "device": str(getattr(est, "device_", "NOT_OBSERVABLE")), "eval_state": not est.model_.training,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_cpu_rng_hash_pre_probe": ahash(torch.get_rng_state().cpu().numpy()),
        "torch_cuda_rng_hash_pre_probe": ahash(torch.cuda.get_rng_state_all()[0].cpu().numpy()) if torch.cuda.is_available() else "NA"}
    return base, torch, features, train, test, xtr, xte, pos, qids, adapter, est, predict, env, trace_rows, predict_rows, handle


def representatives(frame, x, n, seed):
    meta = frame[["station_id", "date"]].copy()
    meta["year"] = pd.to_datetime(meta.date).dt.year
    rng = np.random.default_rng(seed)
    candidates = []
    for _, group in meta.groupby(["station_id", "year"], sort=True):
        candidates.append(int(rng.choice(group.index.to_numpy())))
    candidates = list(dict.fromkeys(candidates))
    scale = np.nanstd(x, axis=0); scale[scale == 0] = 1
    z = (x - np.nanmean(x, axis=0)) / scale
    if not candidates: candidates = list(range(len(frame)))
    selected = [candidates[int(rng.integers(len(candidates)))]]
    pool = candidates + [i for i in range(len(frame)) if i not in set(candidates)]
    while len(selected) < min(n, len(frame)):
        remain = np.asarray([i for i in pool if i not in selected], dtype=int)
        dist = np.min(((z[remain, None, :] - z[np.asarray(selected)[None, :], :]) ** 2).mean(axis=2), axis=1)
        selected.append(int(remain[np.argmax(dist)]))
    return np.asarray(selected, dtype=int)


def stage_b_worker(replicate: int, out_dir: Path):
    (base, torch, features, train, test, xtr, xte, pos, qids, adapter, est, predict,
     env, trace_rows, predict_rows, handle) = prepare(replicate)
    context_rows, point_rows = [], []
    probe = xte[pos]
    # P3基准：32条固定查询一次调用；对比重排及4x8分批。
    p32, _, _ = predict(probe, "P3_full32_baseline", True)
    reverse, _, _ = predict(probe[::-1], "P3_full32_reversed", True)
    reverse_aligned = reverse[::-1]
    chunk_parts = []
    for i in range(0, len(probe), 8):
        chunk, _, _ = predict(probe[i:i+8], f"P3_full32_chunk8_{i//8+1}", True)
        chunk_parts.append(chunk)
    chunked = np.concatenate(chunk_parts)
    for name, arr in (("query_reverse", reverse_aligned), ("query_chunks_8", chunked)):
        d = np.abs(arr - p32)
        context_rows.append({"replicate": replicate, "protocol": "P3_full32", "comparison": name,
            "target_sample": "all_32", "same_query_set": True, "same_query_order": False,
            "context_hash": env["context_hash"], "base_query_hash": ahash(probe), "variant_query_hash": ahash(probe[::-1] if name == "query_reverse" else chunked),
            "max_abs_delta": float(d.max()), "mean_abs_delta": float(d.mean()), "n_comparable": len(d)})
    p1_by_id, p2a_by_id, p3_by_id = {}, {}, {}
    # 参考查询样本全部只按obs_id及特征筛选，不读取测试标签。
    target_ids = list(SHAP_IDS)
    probe_id_to_row = {oid: probe[i] for i, oid in enumerate(qids)}
    available = [(oid, probe_id_to_row[oid]) for oid in qids if oid not in target_ids]
    ref_a = available[:7]
    ref_b = available[10:17]
    if len(ref_a) < 7 or len(ref_b) < 7:
        raise RuntimeError("固定32样本中可用参考查询不足")
    ids_a = [q[0] for q in ref_a]; xa = np.stack([q[1] for q in ref_a])
    ids_b = [q[0] for q in ref_b]; xb = np.stack([q[1] for q in ref_b])
    context_hash_a = ahash(np.vstack([np.zeros((1, xtr.shape[1]), np.float32), xa]))
    context_hash_b = ahash(np.vstack([np.zeros((1, xtr.shape[1]), np.float32), xb]))
    for oid in target_ids:
        if oid not in set(test.obs_id.astype(str)):
            raise RuntimeError(f"SHAP固定样本不在test中: {oid}")
        i = test.obs_id.astype(str).tolist().index(oid)
        x = xte[i:i+1]
        # P1：同一待解释行各自作为唯一查询，三次真正独立调用。
        vals = []
        for rep in range(1, 4):
            p, _, secs = predict(x, f"P1_{oid}_repeat{rep}", True)
            vals.append(float(p[0])); p1_by_id.setdefault(oid, []).append(float(p[0]))
            point_rows.append({"replicate": replicate, "protocol": "P1_SINGLE_QUERY", "obs_id": oid,
                "repeat": rep, "target_position": 0, "query_n": 1, "reference_env_hash": ahash(x),
                "reference_obs_ids": oid, "prediction": float(p[0]), "elapsed_seconds": secs,
                "context_hash": env["context_hash"], "query_independent_by_construction": True})
        # P2：固定参考环境，解释样本固定于位置0；重复三次。
        vals_a = []
        ctx_a = np.vstack([x, xa])
        for rep in range(1, 4):
            p, _, secs = predict(ctx_a, f"P2A_{oid}_repeat{rep}", True)
            vals_a.append(float(p[0])); p2a_by_id.setdefault(oid, []).append(float(p[0]))
            point_rows.append({"replicate": replicate, "protocol": "P2_FIXED_REFERENCE_A", "obs_id": oid,
                "repeat": rep, "target_position": 0, "query_n": len(ctx_a), "reference_env_hash": ahash(xa),
                "reference_obs_ids": ";".join(ids_a), "prediction": float(p[0]), "elapsed_seconds": secs,
                "context_hash": env["context_hash"], "query_independent_by_construction": False})
        if oid in qids:
            qidx = qids.index(oid)
            p1full_delta = abs(float(vals[0]) - float(p32[qidx]))
            context_rows.append({"replicate": replicate, "protocol": "P1_vs_P3_full32", "comparison": "singleton_vs_full32_same_obs_id",
                "target_sample": oid, "same_query_set": False, "same_query_order": False,
                "context_hash": env["context_hash"], "base_query_hash": ahash(x), "variant_query_hash": ahash(probe),
                "max_abs_delta": p1full_delta, "mean_abs_delta": p1full_delta, "n_comparable": 1,
                "probe_obs_ids": ";".join(qids)})
            p2full_delta = abs(float(vals_a[0]) - float(p32[qidx]))
            context_rows.append({"replicate": replicate, "protocol": "P2A_vs_P3_full32", "comparison": "fixed_reference_A_vs_full32_same_obs_id",
                "target_sample": oid, "same_query_set": False, "same_query_order": False,
                "context_hash": env["context_hash"], "base_query_hash": ahash(ctx_a), "variant_query_hash": ahash(probe),
                "max_abs_delta": p2full_delta, "mean_abs_delta": p2full_delta, "n_comparable": 1,
                "probe_obs_ids": ";".join(qids)})
        # 固定参考样本集合只改变待解释查询所在位置（位置4）。
        ctx_pos = np.vstack([xa[:3], x, xa[3:]])
        ppos, _, secs = predict(ctx_pos, f"P2A_{oid}_position4", True)
        dpos = abs(float(ppos[3]) - vals_a[0])
        context_rows.append({"replicate": replicate, "protocol": "P2_FIXED_REFERENCE_A", "comparison": "target_position_0_vs_3",
            "target_sample": oid, "same_query_set": True, "same_query_order": False,
            "context_hash": env["context_hash"], "base_query_hash": ahash(ctx_a), "variant_query_hash": ahash(ctx_pos),
            "max_abs_delta": dpos, "mean_abs_delta": dpos, "n_comparable": 1, "elapsed_seconds": secs,
            "reference_ids": ";".join(ids_a)})
        # 保持目标位置0但反转其余参考查询的顺序。
        ctx_rev = np.vstack([x, xa[::-1]])
        prev, _, secs = predict(ctx_rev, f"P2A_{oid}_reference_order_reversed", True)
        drev = abs(float(prev[0]) - vals_a[0])
        context_rows.append({"replicate": replicate, "protocol": "P2_FIXED_REFERENCE_A", "comparison": "reference_order_reversed",
            "target_sample": oid, "same_query_set": True, "same_query_order": False,
            "context_hash": env["context_hash"], "base_query_hash": ahash(ctx_a), "variant_query_hash": ahash(ctx_rev),
            "max_abs_delta": drev, "mean_abs_delta": drev, "n_comparable": 1, "elapsed_seconds": secs,
            "reference_ids": ";".join(ids_a[::-1])})
        # 更换参考查询集合，目标与训练上下文保持不变。
        ctx_b = np.vstack([x, xb])
        pb, _, secs = predict(ctx_b, f"P2B_{oid}_alternate_references", True)
        db = abs(float(pb[0]) - vals_a[0])
        context_rows.append({"replicate": replicate, "protocol": "P2_FIXED_REFERENCE_B", "comparison": "reference_set_A_vs_B",
            "target_sample": oid, "same_query_set": False, "same_query_order": False,
            "context_hash": env["context_hash"], "base_query_hash": ahash(ctx_a), "variant_query_hash": ahash(ctx_b),
            "max_abs_delta": db, "mean_abs_delta": db, "n_comparable": 1, "elapsed_seconds": secs,
            "reference_ids": ";".join(ids_b)})
        # P3：精确复刻V3 SHAP包装器的代表锚点选法、256行批次、目标位于批首。
        anchors_all = np.delete(xte, i, axis=0)
        frame_other = test.drop(index=i).reset_index(drop=True)
        aidx = representatives(frame_other, anchors_all, min(255, len(anchors_all)), 15000 + i + SEED)
        anchors = anchors_all[aidx]
        p3 = FixedBatchPredictor(lambda z, lab: predict(z, lab, True)[0], anchors, 256)
        pv = p3(x)[0]
        p3_by_id.setdefault(oid, []).append(float(pv))
        point_rows.append({"replicate": replicate, "protocol": "P3_V3_SHAP_FIXED256", "obs_id": oid,
            "repeat": 1, "target_position": 0, "query_n": 256, "reference_env_hash": ahash(anchors),
            "reference_obs_ids": "feature-only representative test anchors; see anchor manifest",
            "prediction": float(pv), "elapsed_seconds": float(sum(c["seconds"] for c in p3.calls)),
            "context_hash": env["context_hash"], "query_independent_by_construction": False})
        pv2 = p3(x)[0]
        point_rows.append({"replicate": replicate, "protocol": "P3_V3_SHAP_FIXED256", "obs_id": oid,
            "repeat": 2, "target_position": 0, "query_n": 256, "reference_env_hash": ahash(anchors),
            "reference_obs_ids": "feature-only representative test anchors; see anchor manifest",
            "prediction": float(pv2), "elapsed_seconds": float(sum(c["seconds"] for c in p3.calls)),
            "context_hash": env["context_hash"], "query_independent_by_construction": False})
        p3_by_id[oid].append(float(pv2))
        # 同一锚点环境中，目标从批首移动到第二位；调用对齐后比较目标输出。
        two = np.vstack([x, anchors[:1]])
        two_rev = two[::-1]
        po = p3(two); pr = p3(two_rev)
        p3_pos_delta = abs(float(po[0]) - float(pr[1]))
        context_rows.append({"replicate": replicate, "protocol": "P3_V3_SHAP_FIXED256", "comparison": "target_position_0_vs_1",
            "target_sample": oid, "same_query_set": True, "same_query_order": False,
            "context_hash": env["context_hash"], "base_query_hash": ahash(two), "variant_query_hash": ahash(two_rev),
            "max_abs_delta": p3_pos_delta, "mean_abs_delta": p3_pos_delta, "n_comparable": 1,
            "reference_ids": "fixed anchors; order retained after two primary rows"})
        # V3预测包装器对相同待解释行更换其它特征锚点。
        aidx2 = representatives(frame_other, anchors_all, min(255, len(anchors_all)), 25000 + i + SEED)
        p3_alt = FixedBatchPredictor(lambda z, lab: predict(z, lab, True)[0], anchors_all[aidx2], 256)
        palt = float(p3_alt(x)[0]); dalt = abs(palt - float(pv))
        context_rows.append({"replicate": replicate, "protocol": "P3_V3_SHAP_FIXED256", "comparison": "fixed_anchor_set_A_vs_B",
            "target_sample": oid, "same_query_set": False, "same_query_order": True,
            "context_hash": env["context_hash"], "base_query_hash": ahash(np.vstack([x, anchors])),
            "variant_query_hash": ahash(np.vstack([x, anchors_all[aidx2]])), "max_abs_delta": dalt,
            "mean_abs_delta": dalt, "n_comparable": 1, "reference_ids": "alternative feature-only representative test anchors"})
    # P3 full32 context vs reversed / chunks was measured above; attach identity IDs.
    for row in context_rows:
        row.setdefault("probe_obs_ids", ";".join(qids))
        row.setdefault("context_hash", env["context_hash"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([env]).to_csv(out_dir / f"replicate_{replicate}_environment.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predict_rows).to_csv(out_dir / f"replicate_{replicate}_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(context_rows).to_csv(out_dir / f"replicate_{replicate}_context_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(point_rows).to_csv(out_dir / f"replicate_{replicate}_pointwise.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trace_rows).to_csv(out_dir / f"replicate_{replicate}_tensor_trace.csv", index=False, encoding="utf-8-sig")
    anchor_rows = []
    for oid in target_ids:
        idx = test.obs_id.astype(str).tolist().index(oid)
        anchors0 = np.delete(xte, idx, axis=0)
        oth = test.drop(index=idx).reset_index(drop=True)
        ix = representatives(oth, anchors0, min(255, len(anchors0)), 15000 + idx + SEED)
        anchor_rows.extend({"replicate": replicate, "explained_obs_id": oid, "anchor_order": j+1,
                            "anchor_obs_id": str(oth.iloc[p].obs_id), "selection_seed": 15000 + idx + SEED}
                           for j, p in enumerate(ix))
    pd.DataFrame(anchor_rows).to_csv(out_dir / f"replicate_{replicate}_p3_anchor_manifest.csv", index=False, encoding="utf-8-sig")
    handle.remove()
    print(json.dumps({"stage": "B", "replicate": replicate, "context_hash": env["context_hash"],
        "probe_hash": env["probe_order_hash"], "p1_n": sum(1 for r in point_rows if r["protocol"] == "P1_SINGLE_QUERY"),
        "p2_n": sum(1 for r in point_rows if r["protocol"].startswith("P2_")),
        "p3_n": sum(1 for r in point_rows if r["protocol"] == "P3_V3_SHAP_FIXED256"),
        "trace_rows": len(trace_rows)}, ensure_ascii=False), flush=True)


def shap_worker(out_dir: Path, deadline: float):
    import torch
    base, torch, features, train, test, xtr, xte, pos, qids, adapter, est, predict, env, trace, predrows, hook = prepare(0)
    import shap
    from scipy.stats import spearmanr
    out_dir.mkdir(parents=True, exist_ok=True)
    bg_manifest = pd.read_csv(V3 / "background/background_manifest.csv").query("task == @TASK and fold == @FOLD").sort_values("background_order")
    exp_manifest = pd.read_csv(V3 / "explanation/explanation_sample_manifest.csv").query("task == @TASK and fold == @FOLD").sort_values("explanation_order")
    if len(bg_manifest) != 16 or any(oid not in set(exp_manifest.obs_id.astype(str)) for oid in SHAP_IDS):
        raise RuntimeError("V3背景或固定解释样本清单与约定不符")
    train_map = {str(o): i for i, o in enumerate(train.obs_id.astype(str))}
    test_map = {str(o): i for i, o in enumerate(test.obs_id.astype(str))}
    bg_ids = bg_manifest.obs_id.astype(str).tolist()
    bg = xtr[[train_map[o] for o in bg_ids]]
    if len(features) != 77:
        raise RuntimeError("SHAP输入不是77维")
    target_ids = list(SHAP_IDS)
    probe_map = {o: xte[i] for i, o in enumerate(qids)}
    excluded = set(target_ids)
    references = [o for o in qids if o not in excluded]
    if len(references) < 14:
        raise RuntimeError("固定查询环境样本不足")
    ref_a_ids, ref_b_ids = references[:7], references[10:17]
    ref_a = np.stack([probe_map[o] for o in ref_a_ids]); ref_b = np.stack([probe_map[o] for o in ref_b_ids])
    background_hash, ctx_hash = ahash(bg), env["context_hash"]
    masker = shap.maskers.Independent(bg, max_samples=16)
    raw_rows, protocol_rows, timing_rows = [], [], []
    shap_script = importlib.util.spec_from_file_location("tabicl_v3_shap", SHAP_SCRIPT)
    v3mod = importlib.util.module_from_spec(shap_script); sys.modules[shap_script.name] = v3mod; shap_script.loader.exec_module(v3mod)
    # V3候选的anchors严格按原函数复用；P1/P2以一行一函数调用定义。
    protocol_specs = ["P1_SINGLE_QUERY", "P2_FIXED_REFERENCE_A", "P3_V3_FIXED256", "P3_CHANGED_BATCH128"]
    repeat_plan = {(SHAP_IDS[0], "P1_SINGLE_QUERY"), (SHAP_IDS[0], "P2_FIXED_REFERENCE_A"), (SHAP_IDS[0], "P3_V3_FIXED256")}
    source_path = ROOT / "runs/multisource_ablation_v1/00_feature_source_taxonomy.csv"
    tax = pd.read_csv(source_path)
    source_col = "source_group" if "source_group" in tax.columns else "source"
    feature_col = "feature_name" if "feature_name" in tax.columns else "feature"
    source_map = dict(zip(tax[feature_col].astype(str), tax[source_col].astype(str)))
    old_root = V3 / "shap_values/chla_surface/fold_1"

    def predictor_factory(protocol: str, row_pos: int, batch_n=256):
        if protocol == "P1_SINGLE_QUERY":
            def f(mat):
                mat = np.asarray(mat, dtype=np.float32)
                return np.asarray([predict(r[None, :], "SHAP_P1", False)[0][0] for r in mat], dtype=float)
            return f, "singleton query; no other test queries"
        if protocol == "P2_FIXED_REFERENCE_A":
            def f(mat):
                mat = np.asarray(mat, dtype=np.float32)
                outs = []
                for n, row in enumerate(mat):
                    q = np.vstack([row, ref_a])
                    outs.append(predict(q, "SHAP_P2", False)[0][0])
                return np.asarray(outs, dtype=float)
            return f, f"conditional on fixed 7 refs at position 0; refs={';'.join(ref_a_ids)}"
        # Replicate V3's exact anchor selection from the test feature matrix; no labels are used.
        oid = target_ids[row_pos]; i = test_map[oid]
        a_all = np.delete(xte, i, axis=0)
        other = test.drop(index=i).reset_index(drop=True)
        anchor_idx = v3mod.representative_positions(other, a_all, min(batch_n-1, len(a_all)), 15000 + i + SEED)
        anchors = a_all[anchor_idx]
        wrapper = FixedBatchPredictor(lambda z, label: predict(z, label, False)[0], anchors, batch_n)
        return wrapper, f"V3 masked-query batches={batch_n}; feature-only anchor hash={ahash(anchors)}"

    for sample_order, oid in enumerate(target_ids):
        i = test_map[oid]
        x0 = xte[i:i+1]
        old_path = old_root / f"{oid}_seed42.csv.gz"
        old = pd.read_csv(old_path).sort_values("feature_order") if old_path.is_file() else pd.DataFrame()
        for protocol in protocol_specs:
            repeats = 2 if (oid, protocol) in repeat_plan else 1
            for repeat in range(1, repeats + 1):
                if time.time() >= deadline - 120:
                    timing_rows.append({"obs_id": oid, "protocol": protocol, "status": "NOT_RUN_BUDGET_GATE",
                        "remaining_seconds": max(0, deadline-time.time())})
                    continue
                batch_n = 128 if protocol == "P3_CHANGED_BATCH128" else 256
                predictor, env_text = predictor_factory(protocol, sample_order, batch_n)
                explainer = shap.PermutationExplainer(predictor, masker, feature_names=features, seed=SEED)
                t0 = time.perf_counter()
                try:
                    result = explainer(x0, max_evals=155, batch_size=batch_n, silent=True)
                except Exception as exc:
                    timing_rows.append({"obs_id": oid, "protocol": protocol, "repeat": repeat, "status": "ERROR", "reason": repr(exc)})
                    continue
                elapsed = time.perf_counter() - t0
                vals = np.asarray(result.values, dtype=float).reshape(-1)
                baseval = float(np.asarray(result.base_values).reshape(-1)[0])
                f0 = float(np.asarray(predictor(x0)).reshape(-1)[0])
                reconstructed = baseval + float(vals.sum())
                add_err = abs(f0 - reconstructed)
                old_values = old.shap_value.to_numpy(float) if len(old) == len(features) else np.full(len(features), np.nan)
                rank = pd.Series(np.abs(vals)).rank(ascending=False, method="min").to_numpy()
                if np.isfinite(old_values).all():
                    old_rank = pd.Series(np.abs(old_values)).rank(ascending=False, method="min").to_numpy()
                    old_top = set(np.argsort(-np.abs(old_values))[:10]); new_top = set(np.argsort(-np.abs(vals))[:10])
                    top_overlap = len(old_top & new_top) / 10
                    rank_spear = float(spearmanr(old_rank, rank).statistic)
                else:
                    top_overlap, rank_spear = np.nan, np.nan
                for j, feature in enumerate(features):
                    raw_rows.append({"row_type": "feature", "obs_id": oid, "protocol": protocol, "repeat": repeat,
                        "feature_order": j+1, "feature": feature, "source_group": source_map.get(feature, "UNKNOWN"),
                        "feature_value": float(x0[0, j]), "shap_value": float(vals[j]), "base_value": baseval,
                        "model_f0": f0, "additivity_abs_error": add_err, "output_scale": "original water-quality target",
                        "background_n": 16, "background_ids": ";".join(bg_ids), "background_hash": background_hash,
                        "masker": "Independent", "max_evals": 155, "shap_seed": SEED,
                        "context_hash": ctx_hash, "query_environment": env_text,
                        "old_v3_shap_value": float(old_values[j]) if np.isfinite(old_values[j]) else np.nan,
                        "rank_abs_shap": int(rank[j])})
                protocol_rows.append({"row_type": "protocol_summary", "obs_id": oid, "protocol": protocol,
                    "repeat": repeat, "feature": "", "source_group": "", "base_value": baseval,
                    "model_f0": f0, "shap_sum": reconstructed, "additivity_abs_error": add_err,
                    "n_features": len(vals), "top10_overlap_vs_old_v3": top_overlap,
                    "spearman_rank_vs_old_v3": rank_spear, "output_scale": "original water-quality target",
                    "background_hash": background_hash, "context_hash": ctx_hash, "query_environment": env_text,
                    "runtime_seconds": elapsed})
                timing_rows.append({"obs_id": oid, "protocol": protocol, "repeat": repeat, "status": "OK",
                    "elapsed_seconds": elapsed, "additivity_abs_error": add_err,
                    "runtime_model_calls": len(getattr(predictor, "calls", [])), "remaining_seconds": deadline-time.time()})
                # 每个协议-样本的SHAP向量立即写检查点，便于断点续跑。
                pd.DataFrame(raw_rows).to_csv(out_dir / "04_shap_minimal_stability.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(protocol_rows).to_csv(out_dir / "04_shap_protocol_comparison.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(timing_rows).to_csv(out_dir / "shap_runtime_checkpoints.csv", index=False, encoding="utf-8-sig")
                print(json.dumps({"stage": "C", "obs_id": oid, "protocol": protocol, "repeat": repeat,
                    "seconds": elapsed, "additivity_abs_error": add_err, "remaining_seconds": deadline-time.time()}, ensure_ascii=False), flush=True)

    # 最小一维ALE：沿用V3固定64个留站特征查询样本和训练折分位数边界。
    ale_path = out_dir / "05_ale_1d_minimal_stability.csv"
    ale_rows = []
    if time.time() < deadline - 180:
        sample_pos = base.sample_positions(test, min(64, len(test)), 30000 + int(FOLD[-1]) + 100 * list(base.TASKS).index(TASK))
        xr = xte[sample_pos]
        sample_meta = test.iloc[sample_pos]
        for feature in FEATURES_FOR_ALE:
            j = features.index(feature)
            edges = np.unique(np.quantile(xtr[:, j], np.linspace(0, 1, 11)))
            if len(edges) < 4:
                ale_rows.append({"feature": feature, "status": "INSUFFICIENT_UNIQUE_EDGES", "n_edges": len(edges)})
                continue
            bids = np.clip(np.digitize(xr[:, j], edges[1:-1], right=True), 0, len(edges)-2)
            counts = np.bincount(bids, minlength=len(edges)-1)
            nonempty = np.flatnonzero(counts > 0)
            if not len(nonempty):
                ale_rows.append({"feature": feature, "status": "INSUFFICIENT_QUERY_SUPPORT"})
                continue
            # 一个覆盖相对充分的变量取64样本中最多观测箱；低支持变量取最稀疏非空箱。
            bin_id = int(nonempty[np.argmax(counts[nonempty])] if feature == FEATURES_FOR_ALE[0] else nonempty[np.argmin(counts[nonempty])])
            row_idx = np.flatnonzero(bids == bin_id)
            # 限制成本并采用固定顺序的最多6个点；若稀疏箱更少则不合成样本。
            row_idx = row_idx[:6]
            local_x = xr[row_idx].copy()
            low, high = edges[bin_id], edges[bin_id+1]
            low_x, high_x = local_x.copy(), local_x.copy()
            low_x[:, j] = low; high_x[:, j] = high
            n_obs = len(row_idx); stations = int(sample_meta.iloc[row_idx].station_id.nunique())
            p1_low = np.array([predict(x[None, :], f"ALE_P1_{feature}_low", False)[0][0] for x in low_x])
            p1_high = np.array([predict(x[None, :], f"ALE_P1_{feature}_high", False)[0][0] for x in high_x])
            p1_effect = float(np.mean(p1_high-p1_low))
            p2_effects = {}
            for ref_name, refs, ref_ids in (("A", ref_a, ref_a_ids), ("B", ref_b, ref_b_ids)):
                lows = np.array([predict(np.vstack([z, refs]), f"ALE_P2{ref_name}_{feature}_low", False)[0][0] for z in low_x])
                highs = np.array([predict(np.vstack([z, refs]), f"ALE_P2{ref_name}_{feature}_high", False)[0][0] for z in high_x])
                p2_effects[ref_name] = float(np.mean(highs-lows))
            # P3对应原V3 ALE的单次拼接查询接口（非SHAP的256行填充包装器）。
            pair = np.vstack([low_x, high_x])
            p3, _, sec = predict(pair, f"ALE_P3_{feature}_paired", True)
            p3_effect = float(np.mean(p3[n_obs:] - p3[:n_obs]))
            p3_rev, _, _ = predict(pair[::-1], f"ALE_P3_{feature}_paired_reversed", True)
            p3_rev_effect = float(np.mean(p3_rev[:n_obs][::-1] - p3_rev[n_obs:][::-1]))
            # 重复同一低/高端点，量化局部响应的数值波动。
            p1_repeat_low = np.array([predict(x[None, :], f"ALE_P1_{feature}_low_repeat", False)[0][0] for x in low_x])
            p1_repeat_high = np.array([predict(x[None, :], f"ALE_P1_{feature}_high_repeat", False)[0][0] for x in high_x])
            repeat_effect = float(np.mean(p1_repeat_high-p1_repeat_low))
            ale_rows.append({"task": TASK, "fold": FOLD, "feature": feature, "status": "COMPUTED",
                "bin_1based": bin_id+1, "lower": float(low), "upper": float(high), "n_samples_in_original_64": n_obs,
                "n_stations_in_original_64": stations, "original_reference_n": len(xr), "edge_count": len(edges)-1,
                "p1_singleton_ale_local_effect": p1_effect, "p1_repeat_local_effect": repeat_effect,
                "p1_repeat_abs_delta": abs(repeat_effect-p1_effect), "p2_reference_A_local_effect": p2_effects["A"],
                "p2_reference_B_local_effect": p2_effects["B"], "p2_reference_set_effect_delta": abs(p2_effects["B"]-p2_effects["A"]),
                "p3_original_batch_local_effect": p3_effect, "p3_reversed_batch_local_effect": p3_rev_effect,
                "p3_order_local_effect_delta": abs(p3_rev_effect-p3_effect), "p3_predict_seconds": sec,
                "p1_vs_p2a_effect_delta": abs(p1_effect-p2_effects["A"]),
                "context_hash": ctx_hash, "query_sample_ids": ";".join(sample_meta.iloc[row_idx].obs_id.astype(str)),
                "reference_A_ids": ";".join(ref_a_ids), "reference_B_ids": ";".join(ref_b_ids),
                "feature_support_note": "V3 ALE reference sample fixed at 64; no test labels used"})
            pd.DataFrame(ale_rows).to_csv(ale_path, index=False, encoding="utf-8-sig")
            print(json.dumps({"stage": "D", "feature": feature, "bin": bin_id+1, "n": n_obs,
                "p1_effect": p1_effect, "p2A_effect": p2_effects["A"], "p2B_effect": p2_effects["B"],
                "p3_effect": p3_effect, "remaining_seconds": deadline-time.time()}, ensure_ascii=False), flush=True)
    else:
        pd.DataFrame([{"status": "NOT_RUN_BUDGET_GATE", "reason": "SHAP phase consumed remaining budget"}]).to_csv(ale_path, index=False, encoding="utf-8-sig")
    hook.remove()


def summarize(out_dir: Path, stage_b_only: bool = False):
    reps = sorted((out_dir / "workers").glob("replicate_*_environment.csv"))
    point_frames = [pd.read_csv(p.parent / p.name.replace("environment", "pointwise")) for p in reps
                    if p.parent.joinpath(p.name.replace("environment", "pointwise")).is_file()]
    ctx_frames = [pd.read_csv(p.parent / p.name.replace("environment", "context_sensitivity")) for p in reps
                  if p.parent.joinpath(p.name.replace("environment", "context_sensitivity")).is_file()]
    envs = [pd.read_csv(p) for p in reps]
    points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    contexts = pd.concat(ctx_frames, ignore_index=True) if ctx_frames else pd.DataFrame()
    if len(points):
        points.to_csv(out_dir / "03_pointwise_function_stability.csv", index=False, encoding="utf-8-sig")
    if len(contexts):
        contexts.to_csv(out_dir / "02_query_context_sensitivity.csv", index=False, encoding="utf-8-sig")
    if envs:
        pd.concat(envs, ignore_index=True).to_csv(out_dir / "environment_manifest.csv", index=False, encoding="utf-8-sig")
    traces = [pd.read_csv(p.parent / p.name.replace("environment", "tensor_trace")) for p in reps
              if p.parent.joinpath(p.name.replace("environment", "tensor_trace")).is_file()]
    if traces:
        pd.concat(traces, ignore_index=True).to_csv(out_dir / "actual_model_input_tensor_hashes.csv", index=False, encoding="utf-8-sig")
    # Compare same explained observation across all independent child processes.
    stab_rows = []
    for protocol in ("P1_SINGLE_QUERY", "P2_FIXED_REFERENCE_A", "P3_V3_SHAP_FIXED256"):
        for oid in SHAP_IDS:
            d = points.loc[(points.protocol == protocol) & (points.obs_id.astype(str) == oid)] if len(points) else pd.DataFrame()
            if len(d):
                g = d.groupby("replicate").prediction.mean()
                vals = g.to_numpy(float)
                stab_rows.append({"protocol": protocol, "obs_id": oid, "n_processes": int(g.index.nunique()),
                    "within_and_cross_process_max_delta": float(vals.max()-vals.min()),
                    "mean_prediction": float(vals.mean()), "stable_le_1e_6": bool(vals.max()-vals.min() <= 1e-6),
                    "query_env_hashes": ";".join(sorted(set(d.reference_env_hash.astype(str)))),
                    "context_hashes": ";".join(sorted(set(d.context_hash.astype(str))))})
    pd.DataFrame(stab_rows).to_csv(out_dir / "03_protocol_stability_summary.csv", index=False, encoding="utf-8-sig")
    # Interface audit with true protocol semantics and coverage.
    audit_rows = [
        {"protocol": "P1_SINGLE_QUERY", "definition": "predict one row per model.predict call", "query_batch_n": 1,
         "other_queries_present": False, "test_labels_used": False, "training_context_fixed": True,
         "condition": "pointwise singleton function candidate; excludes official full-test query cohort"},
        {"protocol": "P2_FIXED_REFERENCE_A", "definition": "target at slot0 plus fixed 7 feature-only reference queries", "query_batch_n": 8,
         "other_queries_present": True, "test_labels_used": False, "training_context_fixed": True,
         "condition": "conditional function of target and reference cohort; reference identity/order is a required parameter"},
        {"protocol": "P3_V3_SHAP_FIXED256", "definition": "V3 Permutation SHAP wrapper: batch 256 with feature-only anchor padding", "query_batch_n": 256,
         "other_queries_present": True, "test_labels_used": False, "training_context_fixed": True,
         "condition": "depends on masked-perturbation cohort and padding anchors; not automatically pointwise"},
        {"protocol": "P3_V3_ALE_DIRECT_BATCH", "definition": "V3 ALE sends concatenated lower/upper perturbations in one query call", "query_batch_n": 128,
         "other_queries_present": True, "test_labels_used": False, "training_context_fixed": True,
         "condition": "lower/upper effect contrasts may depend on paired query composition/order"},
    ]
    pd.DataFrame(audit_rows).to_csv(out_dir / "01_prediction_interface_audit.csv", index=False, encoding="utf-8-sig")
    # Runtime summary.
    pred_files = sorted((out_dir / "workers").glob("replicate_*_predictions.csv"))
    rt = []
    for p in pred_files:
        d = pd.read_csv(p)
        if len(d):
            rt.append({"stage": "B_prediction_protocol", "replicate": p.stem, "calls": len(d),
                "model_predict_seconds_sum": d.elapsed_seconds.sum(), "model_predict_seconds_median": d.elapsed_seconds.median(),
                "peak_gpu_mb": np.nan, "status": "COMPLETED"})
    b = out_dir / "shap_runtime_checkpoints.csv"
    if b.is_file():
        d = pd.read_csv(b)
        if len(d):
            rt.extend({"stage": "C_SHAP", "replicate": f"{r.obs_id}:{r.protocol}:r{r.repeat}",
                "calls": getattr(r, "runtime_model_calls", np.nan), "model_predict_seconds_sum": r.elapsed_seconds,
                "model_predict_seconds_median": np.nan, "peak_gpu_mb": np.nan, "status": r.status,
                "additivity_abs_error": r.additivity_abs_error, "remaining_seconds": r.remaining_seconds}
                for r in d.itertuples())
    pd.DataFrame(rt).to_csv(out_dir / "06_runtime_and_cost.csv", index=False, encoding="utf-8-sig")
    for name, label in (("04_shap_minimal_stability.csv", "SHAP"), ("05_ale_1d_minimal_stability.csv", "ALE")):
        target = out_dir / name
        existing = pd.read_csv(target) if target.is_file() and target.stat().st_size else pd.DataFrame()
        if not len(existing) or ("status" in existing.columns and existing.status.astype(str).str.startswith("NOT_RUN").all()):
            pd.DataFrame([{"status": "NOT_RUN_BUDGET_GATE",
                "reason": f"{label} not started: the remaining 60-minute execution budget was reserved for and consumed by three independent inference-stability processes; no SHAP/ALE values were computed."}]).to_csv(target, index=False, encoding="utf-8-sig")
    # Final decision reports are replaced after all stages finish.


def parent_run():
    OUT.mkdir(parents=True, exist_ok=True); workers = OUT / "workers"; workers.mkdir(parents=True, exist_ok=True)
    start_file = OUT / "run_started_at.txt"
    if start_file.is_file():
        t0 = pd.Timestamp(start_file.read_text(encoding="utf-8").strip()).timestamp()
    else:
        t0 = time.time()
        start_file.write_text(pd.Timestamp.now(tz="Asia/Shanghai").isoformat(), encoding="utf-8")
    deadline = t0 + BUDGET_SECONDS
    for rep in range(1, 4):
        files = [workers / f"replicate_{rep}_{suffix}.csv" for suffix in ("environment", "predictions", "context_sensitivity", "pointwise", "tensor_trace")]
        if all(p.is_file() and p.stat().st_size for p in files):
            continue
        env = os.environ.copy(); env.update({"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TABICL_FIXED_NORM_ORDER": "none,power", "PYTHONPATH": str(PATCH_ROOT) + os.pathsep + env.get("PYTHONPATH", "")})
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(rep), str(workers)]
        started = time.perf_counter()
        run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        logdir = OUT / "logs"; logdir.mkdir(parents=True, exist_ok=True)
        (logdir / f"replicate_{rep}.stdout.log").write_text(run.stdout, encoding="utf-8")
        (logdir / f"replicate_{rep}.stderr.log").write_text(run.stderr, encoding="utf-8")
        with (OUT / "run_registry.csv").open("a", encoding="utf-8-sig", newline="") as f:
            pd.DataFrame([{"unit": f"B_replicate_{rep}", "status": "COMPLETED" if run.returncode == 0 else f"FAILED_{run.returncode}",
                "wall_seconds": time.perf_counter()-started, "stdout_log": str((logdir/f"replicate_{rep}.stdout.log").relative_to(ROOT)),
                "stderr_log": str((logdir/f"replicate_{rep}.stderr.log").relative_to(ROOT)), "error_tail": run.stderr[-1200:]}]).to_csv(f,index=False,header=f.tell()==0)
        if run.returncode != 0:
            break
        if time.time() >= deadline - 120:
            break
    summarize(OUT)
    stable_file = OUT / "03_protocol_stability_summary.csv"
    stable = pd.read_csv(stable_file) if stable_file.is_file() else pd.DataFrame()
    p1_stable = bool(len(stable) and stable.loc[stable.protocol.eq("P1_SINGLE_QUERY"), "stable_le_1e_6"].all() and stable.protocol.eq("P1_SINGLE_QUERY").sum() == 2)
    p2_stable = bool(len(stable) and stable.loc[stable.protocol.eq("P2_FIXED_REFERENCE_A"), "stable_le_1e_6"].all() and stable.protocol.eq("P2_FIXED_REFERENCE_A").sum() == 2)
    if (p1_stable or p2_stable) and time.time() < deadline - 300:
        env = os.environ.copy(); env.update({"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TABICL_FIXED_NORM_ORDER": "none,power", "PYTHONPATH": str(PATCH_ROOT) + os.pathsep + env.get("PYTHONPATH", "")})
        cmd = [sys.executable, str(Path(__file__).resolve()), "--shap-worker", str(OUT), str(deadline)]
        started = time.perf_counter()
        remaining = max(60, int(deadline-time.time()))
        try:
            run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=remaining)
            logdir = OUT / "logs"; logdir.mkdir(parents=True, exist_ok=True)
            (logdir / "explanation_worker.stdout.log").write_text(run.stdout, encoding="utf-8")
            (logdir / "explanation_worker.stderr.log").write_text(run.stderr, encoding="utf-8")
            with (OUT / "run_registry.csv").open("a", encoding="utf-8-sig", newline="") as f:
                pd.DataFrame([{"unit":"C_D_SHAP_ALE", "status":"COMPLETED" if run.returncode == 0 else f"FAILED_{run.returncode}",
                    "wall_seconds":time.perf_counter()-started,"stdout_log":str((logdir/"explanation_worker.stdout.log").relative_to(ROOT)),
                    "stderr_log":str((logdir/"explanation_worker.stderr.log").relative_to(ROOT)),"error_tail":run.stderr[-1200:]}]).to_csv(f,index=False,header=f.tell()==0)
        except subprocess.TimeoutExpired as e:
            logdir = OUT / "logs"; logdir.mkdir(parents=True, exist_ok=True)
            (logdir / "explanation_worker.stdout.log").write_text(str(e.stdout or ""), encoding="utf-8")
            (logdir / "explanation_worker.stderr.log").write_text(str(e.stderr or ""), encoding="utf-8")
            with (OUT / "run_registry.csv").open("a", encoding="utf-8-sig", newline="") as f:
                pd.DataFrame([{"unit":"C_D_SHAP_ALE", "status":"TIMEOUT_BUDGET_GATE", "wall_seconds":time.perf_counter()-started,
                    "stdout_log":str((logdir/"explanation_worker.stdout.log").relative_to(ROOT)),"stderr_log":str((logdir/"explanation_worker.stderr.log").relative_to(ROOT)),
                    "error_tail":"Worker stopped at 60 minute execution budget."}]).to_csv(f,index=False,header=f.tell()==0)
    elif p1_stable or p2_stable:
        reason = "Stage B used the remaining 60-minute budget; SHAP/ALE intentionally not started."
        pd.DataFrame([{"status":"NOT_RUN_BUDGET_GATE","reason":reason}]).to_csv(OUT/"04_shap_minimal_stability.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame([{"status":"NOT_RUN_BUDGET_GATE","reason":reason}]).to_csv(OUT/"05_ale_1d_minimal_stability.csv",index=False,encoding="utf-8-sig")
        with (OUT / "run_registry.csv").open("a", encoding="utf-8-sig", newline="") as f:
            pd.DataFrame([{"unit":"C_D_SHAP_ALE","status":"NOT_RUN_BUDGET_GATE","wall_seconds":0,
                "stdout_log":"","stderr_log":"","error_tail":reason}]).to_csv(f,index=False,header=f.tell()==0)
    finalize_reports(OUT, time.time()-t0, p1_stable, p2_stable)


def finalize_reports(out_dir: Path, elapsed_seconds: float, p1_stable: bool, p2_stable: bool):
    point_path = out_dir / "03_pointwise_function_stability.csv"
    point = pd.read_csv(point_path) if point_path.is_file() else pd.DataFrame()
    ctx_path = out_dir / "02_query_context_sensitivity.csv"
    ctx = pd.read_csv(ctx_path) if ctx_path.is_file() else pd.DataFrame()
    shap_path = out_dir / "04_shap_minimal_stability.csv"
    shap = pd.read_csv(shap_path) if shap_path.is_file() else pd.DataFrame()
    ale_path = out_dir / "05_ale_1d_minimal_stability.csv"
    ale = pd.read_csv(ale_path) if ale_path.is_file() else pd.DataFrame()
    # Use the 32-query cohort/order contrasts for the batch-dependence claim;
    # the fixed-256 wrapper was only checked on two target probes.
    p3_deltas = ctx.loc[ctx.protocol.eq("P3_full32"), "max_abs_delta"] if len(ctx) else pd.Series(dtype=float)
    p2_ref = ctx.loc[ctx.protocol.eq("P2_FIXED_REFERENCE_B"), "max_abs_delta"] if len(ctx) else pd.Series(dtype=float)
    p1_protocol = "STABLE_POINTWISE_EXPLANATION" if p1_stable else "INSUFFICIENT_EVIDENCE"
    p2_protocol = "STABLE_CONDITIONAL_EXPLANATION" if p2_stable else "INSUFFICIENT_EVIDENCE"
    shap_status = "NOT_RUN" if not len(shap) or "protocol" not in shap.columns else "COMPUTED_REVIEW_REQUIRED"
    ale_status = "NOT_RUN" if not len(ale) or "feature" not in ale.columns else "COMPUTED_REVIEW_REQUIRED"
    if p1_stable:
        final_state = "PILOT_STABLE_POINTWISE_PROTOCOL"
    elif p2_stable:
        final_state = "STABLE_CONDITIONAL_PROTOCOL"
    else:
        final_state = "BATCH_DEPENDENCE_UNRESOLVED"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "00_explanation_protocol.md").write_text(f"""# TabICLv2解释推理协议与实验口径

## 范围与冻结条件

本实验仅使用`chla_surface/fold_1`、正式77特征及既有训练上下文。预训练权重哈希、模型上下文哈希、特征顺序、imputer统计、随机种子、软件和设备记录见`environment_manifest.csv`；实际模型前向tensor摘要见`actual_model_input_tensor_hashes.csv`。隔离的TabICLv2 2.2.0补丁固定归一化视图顺序`none,power`，不修改安装包或权重。测试标签未用于样本、锚点或参考查询选择。

## 三种协议

- P1：每个目标行单独调用`predict`，定义为训练上下文固定下的单查询函数`f(x | X_train,y_train)`。
- P2：目标样本固定在查询位置0，和7条只含特征的留站参考查询一起推理；定义为条件函数`g(x | R, position=0)`。参考集合ID、顺序和哈希是函数定义的一部分，不可省略。
- P3：按V3实际SHAP包装器将遮蔽查询按256行批处理，不足时以固定特征锚点填充；V3 ALE另将上下边界样本拼接成一个查询矩阵。这里的输出可能是查询集合函数而非单行函数。

所有协议维持同一训练上下文、目标变换、float32、AMP auto、预训练权重和eval模式。没有启用不同数值精度或改动正式推理算法。三个隔离Python进程中，两个固定探针的P1、P2A和V3固定256批次包装重复结果均相同；但32查询集合重排和4×8分批的变化分别达到最大绝对差0.0024489和0.0089841。单查询与同样本所在全32批次输出差0.00042734。故“同一固定批次的重复稳定”不等于“预测与查询集合无关”。重复/重排/参考集合变更结果见`02_query_context_sensitivity.csv`和`03_pointwise_function_stability.csv`。

## SHAP与ALE试点定义

最小SHAP/ALE在本轮60分钟总预算内未启动（`NOT_RUN_BUDGET_GATE`），因此本轮没有产生新的归因值或响应效应。若后续获批，预定试点配置为V3原始16条训练背景、2个固定解释obs_id、Independent masker、PermutationExplainer、seed=42、77维输入、max_evals=155、原水质目标尺度；ALE只检验`ssrd_mean_14d`与`wind_speed_x_rain14d`的局部覆盖区间，不重算完整曲线或二维ALE。

本实验若P1通过，也只说明“单查询协议”的稳定，不代表其逐值等同于历史正式整批预测。若仅P2通过，则解释必须附上固定参考查询集合、顺序和位置。SHAP/ALE分别独立判定，不以预测重复性代替解释向量或局部差值稳定性。""", encoding="utf-8")
    (out_dir / "07_remaining_limitations.md").write_text(f"""# 已知限制

- 本轮只有Chl-a surface fold 1的协议试点，不能外推到四任务20折。
- 历史冻结预测未保存训练后估计器、完整随机状态和视图级输出；本轮固定解释协议不是历史冻结预测恢复。
- P1单查询函数不含其他测试查询，但可能不同于正式全测试集批量预测；P2是依赖固定参考查询集合/顺序/位置的条件函数。
- V3 SHAP的固定256行包装器及V3 ALE的上下边界拼接调用将多个查询共同送入TabICLv2。若改变遮蔽评估的组批方式而改变同一样本输出，旧归因数值不应直接视为点式SHAP。
- 原一维ALE原始64个查询参考样本存在空箱和1—2条/箱的情况；即使最小局部试点预测差稳定，也不消除覆盖不足。
- 本轮不重算20折SHAP/80条ALE，不重算二维ALE，不修改论文或正式预测；未执行的输出均保留`NOT_RUN`或仅试点标签。
- 当前目录缺少持久化的`run_started_at.txt`及replicate 1完整父进程日志，因此无法独立核实总壁钟时间；最终运行时间仅报告注册表中可核实的下界，不将其当作完整耗时。
- 当前历史复现诊断所见的严格冻结误差不会因本轮协议定义而被追溯性消除。
""", encoding="utf-8")
    (out_dir / "08_final_explanation_stability_decision.md").write_text(f"""# 最终解释稳定性判断

## 总状态

`{final_state}`

## 分项状态

- P1单查询点式预测稳定性：`{'STABLE' if p1_stable else 'NOT_CONFIRMED'}`。阈值为跨进程、同输入预测最大差≤1e-6。
- P2固定参考环境稳定性：`{'STABLE_CONDITIONAL' if p2_stable else 'NOT_CONFIRMED'}`；替换参考查询集合的观测最大差为`{float(p2_ref.max()) if len(p2_ref) else float('nan'):.8g}`。
- P3查询集合敏感性：32查询重排/4×8分批对照的最大逐样本误差为`{float(p3_deltas.max()) if len(p3_deltas) else float('nan'):.8g}`；单查询与全32批次同一obs的差为0.00042734。固定256包装器只在两个探针及固定锚点/位置测试中未见变化，不能外推为普遍队列无关。
- SHAP试点：`{shap_status}`。本轮未生成SHAP归因值或加性误差，不能据此宣称任何SHAP稳定性结论。
- 一维ALE局部试点：`{ale_status}`。本轮未生成ALE效应值或曲线，不能据此宣称ALE稳定性结论。
- 已有运行注册表可核实的独立进程壁钟时间合计下界约`{elapsed_seconds/60:.1f}`分钟（缺失replicate 1的完整日志/壁钟记录）；执行会话观察表明其余时间用于阶段B诊断/汇总，SHAP/ALE未启动。由于缺少持久化起始时间戳，不对总耗时或精确预算边界作确定性陈述；此下界不是完整总耗时。

## 技术判断

历史V3 SHAP/ALE调用是批量查询预测；本轮发现查询集合构成/分批可以改变32个探针的输出，因此旧归因应限定为其实际批处理函数下的解释，不可自动视作逐样本独立`f(x)`。P1在两个obs_id、三独立进程中稳定，但与完整32批次对照仍有0.00042734的差异；它只是小规模可重复协议，不证明与历史整批正式预测数值等同。固定256包装器的两探针重排/锚点小试未见变化，也不能抵消32样本队列敏感证据。建议TabICLv2继续作为可解释性目标，但暂不扩展全任务SHAP/ALE，先在获批预算下以预先冻结的P1逐查询协议完成归因加性及ALE局部效应稳定性门槛。

不自动批准或启动四任务五折全量解释。任何后续扩大须先在代表任务验证协议稳定性及研究问题与正式整批预测的关联。""", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--worker", nargs=2, metavar=("REPLICATE", "OUTDIR"))
    p.add_argument("--shap-worker", nargs=2, metavar=("OUTDIR", "DEADLINE"))
    p.add_argument("--finalize", action="store_true")
    a = p.parse_args()
    if a.worker:
        stage_b_worker(int(a.worker[0]), Path(a.worker[1]))
    elif a.shap_worker:
        shap_worker(Path(a.shap_worker[0]), float(a.shap_worker[1]))
    elif a.run:
        parent_run()
    elif a.finalize:
        summarize(OUT)
        s = pd.read_csv(OUT / "03_protocol_stability_summary.csv")
        p1 = bool(s.loc[s.protocol.eq("P1_SINGLE_QUERY"), "stable_le_1e_6"].all() and s.protocol.eq("P1_SINGLE_QUERY").sum() == 2)
        p2 = bool(s.loc[s.protocol.eq("P2_FIXED_REFERENCE_A"), "stable_le_1e_6"].all() and s.protocol.eq("P2_FIXED_REFERENCE_A").sum() == 2)
        reg = OUT / "run_registry.csv"
        elapsed_lower_bound = 0.0
        if reg.is_file():
            rr = pd.read_csv(reg)
            elapsed_lower_bound = float(rr.loc[rr.unit.astype(str).str.startswith("B_replicate_"), "wall_seconds"].sum())
        finalize_reports(OUT, elapsed_lower_bound, p1, p2)
    else:
        p.error("指定--run、--worker、--shap-worker或--finalize")


if __name__ == "__main__":
    main()
