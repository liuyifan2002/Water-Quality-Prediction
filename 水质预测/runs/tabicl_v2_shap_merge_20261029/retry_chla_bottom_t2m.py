"""单次重试唯一失败的ALE单元；沿用已有SQLite检查点，不覆盖局部差值。"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(r"E:\研究生\codex\水质预测")
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
PYTHON = ROOT / "runs/tabiclv2_pilot/环境/.venv/Scripts/python.exe"
SCRIPT = ROOT / "脚本/05_诊断分析/100_compute_tabicl_all10_ale_1d_v2.py"
LOG = OUT / "logs/chla_bottom_ALE_1D_t2m_retry.log"


def state(stage: str, detail: str) -> None:
    payload = {
        "experiment": "TABICLV2_SLIM_EXPLAINABILITY_V2_ALL10",
        "stage": stage,
        "task": "chla_bottom",
        "unit": "t2m",
        "detail": detail,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "retry": True,
        "v1_gate_is_quality_evidence_not_compute_gate": True,
    }
    temporary = OUT / "progress_status.retry.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUT / "progress_status.json")


def main() -> None:
    if not PYTHON.is_file() or not SCRIPT.is_file():
        raise FileNotFoundError("恢复环境或ALE脚本不存在")
    state("RUNNING", "重试 chla_bottom/t2m ALE；沿用SQLite已有检查点")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env["PYTHONPATH"] = str(ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch") + os.pathsep + env.get("PYTHONPATH", "")
    with LOG.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(PYTHON), str(SCRIPT), "--task", "chla_bottom", "--calculate", "--feature", "t2m"],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode == 0:
        state("RETRY_FINISHED", "chla_bottom/t2m ALE 重试完成")
    else:
        state("RETRY_FAILED", f"chla_bottom/t2m ALE 重试失败，exit_code={result.returncode}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
