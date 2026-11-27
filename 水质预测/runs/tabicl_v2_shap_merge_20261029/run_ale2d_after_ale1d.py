"""等待唯一失败的ALE 1D重试结束，再执行V2 2D联合支持审查与合格组合。"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil


ROOT = Path(r"E:\研究生\codex\水质预测")
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
PYTHON = ROOT / "runs/tabiclv2_pilot/环境/.venv/Scripts/python.exe"
SOURCE = ROOT / "脚本/05_诊断分析/92_compute_tabicl_slim_ale_2d_v1.py"
RETRY_MARKER = "retry_chla_bottom_t2m.py"
ALE1D = "100_compute_tabicl_all10_ale_1d_v2.py"
LOG = OUT / "logs/ale2d_v2_orchestrator.log"


def state(stage: str, detail: str) -> None:
    payload = {
        "experiment": "TABICLV2_SLIM_EXPLAINABILITY_V2_ALL10",
        "stage": stage,
        "task": "",
        "unit": "ALE_2D",
        "detail": detail,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "v2_ale2d": True,
        "v1_gate_is_quality_evidence_not_compute_gate": True,
    }
    temporary = OUT / "progress_status.ale2d.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUT / "progress_status.json")


def active_previous_work() -> list[str]:
    found = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in (proc.info["name"] or "").lower():
                continue
            command = " ".join(proc.info["cmdline"] or [])
            if str(ROOT).lower() not in command.lower():
                continue
            if RETRY_MARKER in command or ALE1D in command:
                found.append(f"{proc.pid}:{command}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    return found


def load_module():
    spec = importlib.util.spec_from_file_location("tabicl_v2_ale2d_runner", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载ALE 2D脚本")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = OUT
    return module


def main() -> None:
    if not PYTHON.is_file() or not SOURCE.is_file():
        raise FileNotFoundError("V2 Python环境或ALE 2D脚本不存在")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8") as log:
        state("WAITING_FOR_ALE1D", "等待chla_bottom/t2m ALE 1D重试结束，不并行启动ALE 2D")
        log.write(json.dumps({"event": "wait_started", "active": active_previous_work()}, ensure_ascii=False) + "\n")
        log.flush()
        while active_previous_work():
            time.sleep(15)
        state("RUNNING", "ALE 1D重试已结束，执行ALE 2D联合支持审查")
        module = load_module()
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "42"
        env["PYTHONPATH"] = str(ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch") + os.pathsep + env.get("PYTHONPATH", "")
        # screen_all 使用V2输出目录，复用冻结任务、特征和联合支持规则；不调用模型。
        module.screen_all()
        decisions_path = OUT / "ale_2d/candidate_pair_decisions.csv"
        decisions = pd.read_csv(decisions_path)
        selected = decisions.loc[decisions.selected.astype(bool), "task"].astype(str).tolist()
        log.write(json.dumps({"event": "screen_complete", "selected_tasks": selected, "decisions": decisions.to_dict("records")}, ensure_ascii=False) + "\n")
        log.flush()
        if not selected:
            state("ALE_2D_BLOCKED_BY_SUPPORT", "联合支持审查无合格组合；未启动ALE 2D模型计算")
            return
        for task in selected:
            state("RUNNING", f"ALE 2D计算：{task}")
            module.calculate(task)
            log.write(json.dumps({"event": "calculate_complete", "task": task}, ensure_ascii=False) + "\n")
            log.flush()
        state("ALE_2D_FINISHED", f"ALE 2D完成{len(selected)}个合格组合")


if __name__ == "__main__":
    main()
