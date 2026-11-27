"""Resume only ALE and final quality review after verified precision repair."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    from importlib.util import module_from_spec, spec_from_file_location
    sys.path.insert(0, str(ROOT / "05_scripts"))
    spec = spec_from_file_location("preflight", ROOT / "05_scripts" / "run_preflight.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.preflight(dependencies=True)
    for name in ("monitor_run.json", "run_status.json"):
        source = OUT / name
        backup = OUT / name.replace(".json", "_first_attempt.json")
        if source.exists() and not backup.exists():
            shutil.copy2(source, backup)
    original = json.loads((OUT / "run_status_first_attempt.json").read_text(encoding="utf-8"))
    completed = [r for r in original if r["script"] in ("run_prediction_audit.py", "run_shap.py")]
    write_json(OUT / "run_status.json", completed)
    state = {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(), "status": "RUNNING", "command": "ALE精度修复续跑"}
    write_json(OUT / "monitor_run.json", state)
    for script in ("run_ale.py", "run_quality_audit.py"):
        log = OUT / "logs" / (script + ".log")
        with log.open("a", encoding="utf-8") as file:
            file.write("\n=== ALE precision repair resume " + datetime.now(timezone.utc).isoformat() + " ===\n")
            file.flush()
            result = subprocess.run([sys.executable, str(ROOT / "05_scripts" / script)], cwd=ROOT, stdout=file, stderr=subprocess.STDOUT)
        completed.append({"script": script, "exit_code": result.returncode})
        write_json(OUT / "run_status.json", completed)
    state.update(status="FINISHED" if all(r["exit_code"] == 0 for r in completed) else "COMPLETED_WITH_ISSUES",
                 exit_code=0 if all(r["exit_code"] == 0 for r in completed) else 1,
                 ended_at=datetime.now(timezone.utc).isoformat())
    write_json(OUT / "monitor_run.json", state)
    return state["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
