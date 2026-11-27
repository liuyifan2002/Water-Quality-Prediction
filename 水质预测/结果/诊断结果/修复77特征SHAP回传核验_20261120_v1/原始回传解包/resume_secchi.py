"""Continue the explicitly revised Secchi boundary registration and final audit."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    from importlib.util import module_from_spec, spec_from_file_location
    sys.path.insert(0, str(ROOT / "05_scripts"))
    spec = spec_from_file_location("preflight", ROOT / "05_scripts" / "run_preflight.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.preflight(dependencies=True)
    for name in ("monitor_run.json", "run_status.json"):
        source = OUT / name
        backup = OUT / name.replace(".json", "_before_secchi_revision.json")
        if source.exists() and not backup.exists():
            shutil.copy2(source, backup)
    state = {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
             "status": "RUNNING", "command": "Secchi边界登记v2续跑"}
    save(OUT / "monitor_run.json", state)
    results = []
    for script, arguments in (("run_ale.py", ["--task", "secchi", "--amend-secchi-boundary"]),
                              ("run_quality_audit.py", [])):
        log = OUT / "logs" / (script + ".log")
        with log.open("a", encoding="utf-8") as file:
            file.write("\n=== Secchi registration revision " + datetime.now(timezone.utc).isoformat() + " ===\n")
            file.flush()
            result = subprocess.run([sys.executable, str(ROOT / "05_scripts" / script), *arguments],
                                    cwd=ROOT, stdout=file, stderr=subprocess.STDOUT)
        results.append({"script": script, "exit_code": result.returncode, "revision": "secchi_boundary_v2"})
        save(OUT / "run_status_secchi_revision.json", results)
        if result.returncode and script == "run_ale.py":
            break
    state.update(status="FINISHED" if len(results) == 2 and all(r["exit_code"] == 0 for r in results)
                 else "COMPLETED_WITH_ISSUES", exit_code=0 if len(results) == 2 and all(r["exit_code"] == 0 for r in results) else 1,
                 ended_at=datetime.now(timezone.utc).isoformat())
    save(OUT / "monitor_run.json", state)
    return state["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
