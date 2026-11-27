"""先续算已有核心 SHAP 的 ALE，再交给原 V2 串行调度器。"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "脚本" / "05_诊断分析"
V2 = ROOT / "runs" / "tabicl_v2_slim_explainability_v2_all10"
LOCAL = Path(__file__).resolve().parent
PAUSE = V2 / "PAUSE_V2.flag"
PRIORITY = ("tn_surface", "tn_bottom", "tp_bottom")


def load_orchestrator():
    source = SCRIPTS / "101_resume_tabicl_all10_explanation_v2.py"
    spec = importlib.util.spec_from_file_location("ale_local_orchestrator", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run() -> int:
    orch = load_orchestrator()
    if PAUSE.exists():
        raise RuntimeError("V2 暂停标记存在，不启动计算")
    if orch.v1_running():
        raise RuntimeError("V1 GPU 解释队列仍在运行")
    for task in PRIORITY:
        core = orch.completed_ids(task, "CORE_3")
        if len(core) != 3:
            raise RuntimeError(f"{task}: 只找到 {len(core)}/3 个合格核心 SHAP")
        screening = V2 / "ale_1d" / f"{task}_feature_screening.csv"
        if not screening.is_file():
            raise RuntimeError(f"{task}: 缺少 ALE 支持度审查")
        selected = pd.read_csv(screening).loc[lambda frame: frame.proposed.astype(bool), "feature"].astype(str).tolist()
        if len(selected) > 2:
            raise RuntimeError(f"{task}: ALE 候选超过两项")
        for feature in selected:
            support = V2 / "ale_1d" / f"{task}_{feature}_bin_support.csv"
            sample = V2 / "ale_1d" / f"{task}_{feature}_query_manifest.csv"
            if not support.is_file() or not sample.is_file():
                raise RuntimeError(f"{task}/{feature}: 缺少分箱或查询清单")
            if not pd.read_csv(support).selected_support.astype(bool).all():
                orch.register(task, "ALE_1D_CURVE", feature, "UNSUPPORTED", support, "样本或站点支持度不足")
                continue
            if orch.curve_done(task, feature):
                print(json.dumps({"task": task, "feature": feature, "status": "REUSED"}, ensure_ascii=False), flush=True)
                continue
            if PAUSE.exists():
                orch.state("PAUSED_BY_USER", task=task, unit=feature)
                return 0
            orch.state("RUNNING", task=task, unit=feature, detail="本机优先续算已有核心 SHAP 的一维 ALE")
            log = LOCAL / f"{task}_{feature}.log"
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = "42"
            env["PYTHONPATH"] = str(orch.PATCH) + os.pathsep + env.get("PYTHONPATH", "")
            command = [sys.executable, str(SCRIPTS / "100_compute_tabicl_all10_ale_1d_v2.py"),
                       "--task", task, "--calculate", "--feature", feature]
            with log.open("a", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=ROOT, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT, check=False)
            curve = V2 / "ale_1d" / f"{task}_{feature}_curve.csv"
            success = result.returncode == 0 and orch.curve_done(task, feature)
            orch.register(task, "ALE_1D_CURVE", feature, "COMPUTED" if success else "FAILED",
                          curve if success else log, f"exit_code={result.returncode}")
            print(json.dumps({"task": task, "feature": feature,
                              "status": "COMPUTED" if success else "FAILED",
                              "exit_code": result.returncode, "log": str(log)}, ensure_ascii=False), flush=True)
            if not success:
                orch.state("STOPPED_ALE_FAILURE", task=task, unit=feature, detail=f"检查 {log}")
                return 1
    if PAUSE.exists():
        orch.state("PAUSED_BY_USER", detail="优先 ALE 完成后在任务边界暂停")
        return 0
    orch.state("ALL10_QUEUE_STARTING", detail="优先 ALE 完成；交给原 V2 串行调度器")
    with (LOCAL / "all10_orchestrator.log").open("a", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, str(SCRIPTS / "101_resume_tabicl_all10_explanation_v2.py")],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as error:
        print(json.dumps({"status": "STOPPED", "error": repr(error)}, ensure_ascii=False), flush=True)
        raise
