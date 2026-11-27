"""Launch the supplied serial pipeline and record its lifecycle for the local monitor."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
STATE = OUT / "monitor_run.json"


def write(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def main():
    state = {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(), "status": "RUNNING", "command": "05_scripts/run_all.py"}
    write(state)
    with (OUT / "monitor_launcher.log").open("a", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(ROOT / "05_scripts" / "run_all.py")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    state.update(status="FINISHED" if result.returncode == 0 else "COMPLETED_WITH_ISSUES", exit_code=result.returncode, ended_at=datetime.now(timezone.utc).isoformat())
    write(state)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
