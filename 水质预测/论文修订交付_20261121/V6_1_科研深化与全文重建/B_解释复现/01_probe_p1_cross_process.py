"""在原实验环境中对两条预登记样本做跨进程 P1 数值探针。"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[3]
OUT = HERE.parent
PORT = PROJECT / "交付/修复77特征解释计算迁移_20261116_v3/TabICLv2_SHAP_ALE_PORTABLE"
AUDIT = PROJECT / "结果/诊断结果/修复77特征SHAP回传核验_20261120_v2/02_30核心SHAP核验.csv"
TASK = "chla_surface"
OBS_IDS = ("SM13_20150515_surface", "VM7_20170406_surface")


def child() -> None:
    sys.path.insert(0, str(PORT / "05_scripts"))
    from portable_context import context, file_sha
    import numpy as np
    import sklearn
    import torch
    import tabicl

    _, features, train, test, x_train, x_test, model, inverse, meta = context(TASK)
    by_id = {str(v): i for i, v in enumerate(test.obs_id)}
    with AUDIT.open(encoding="utf-8-sig", newline="") as handle:
        refs = {row["obs_id"]: row for row in csv.DictReader(handle) if row["task"] == TASK}
    view_order = list(model.model.ensemble_generator_.ensemble_configs_)
    result = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "hashseed": os.environ.get("PYTHONHASHSEED"),
        "torch": torch.__version__,
        "tabicl": importlib.metadata.version("tabicl"),
        "sklearn": sklearn.__version__,
        "cuda": torch.cuda.is_available(),
        "tabicl_module": tabicl.__file__,
        "view_order": view_order,
        "context_hash": meta["context_hash"],
        "feature_order_hash": meta["feature_order_hash"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "portable_config_sha256": file_sha(PORT / "04_explanation_config/portable_config.json"),
        "samples": [],
    }
    for obs_id in OBS_IDS:
        idx = by_id[obs_id]
        value = float(np.maximum(inverse(model.predict(x_test[idx:idx+1])), 0)[0])
        ref = refs[obs_id]
        result["samples"].append({
            "obs_id": obs_id,
            "p1_prediction": value,
            "shap_saved_prediction": float(ref["SHAP模型预测"]),
            "prior_independent_p1": float(ref["审查P1预测"]),
            "frozen_batch_prediction": float(ref["正式批量预测"]),
            "delta_vs_shap": value - float(ref["SHAP模型预测"]),
            "delta_vs_prior_p1": value - float(ref["审查P1预测"]),
            "delta_vs_frozen_batch": value - float(ref["正式批量预测"]),
        })
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in ("0", "1"):
        env = os.environ.copy()
        env.update(PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        proc = subprocess.run(
            [sys.executable, "-B", str(HERE), "--child"],
            cwd=PROJECT, env=env, text=True, encoding="utf-8", errors="replace",
            capture_output=True, timeout=1800,
        )
        (OUT / f"P1_probe_hashseed_{seed}.log").write_text(
            proc.stdout + "\nSTDERR\n" + proc.stderr, encoding="utf-8"
        )
        if proc.returncode:
            raise RuntimeError(f"P1 probe hashseed={seed} failed: {proc.returncode}")
        records.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    (OUT / "两样本跨进程P1探针.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT / "两样本跨进程P1探针.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "hashseed", "view_order", "obs_id", "p1_prediction", "shap_saved_prediction",
            "prior_independent_p1", "frozen_batch_prediction", "delta_vs_shap",
            "delta_vs_prior_p1", "delta_vs_frozen_batch",
        ])
        writer.writeheader()
        for record in records:
            for row in record["samples"]:
                writer.writerow({"hashseed": record["hashseed"], "view_order": ",".join(record["view_order"]), **row})
    print(json.dumps(records, ensure_ascii=False))


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
