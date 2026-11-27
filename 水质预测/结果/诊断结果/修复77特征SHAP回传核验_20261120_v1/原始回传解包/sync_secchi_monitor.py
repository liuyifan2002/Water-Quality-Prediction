"""Keep the existing progress page aligned with the Secchi continuation."""
import json
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent / "outputs"
BASE = [{"script": "run_prediction_audit.py", "exit_code": 0},
        {"script": "run_shap.py", "exit_code": 0}]

for _ in range(3600):
    try:
        revised = json.loads((OUT / "run_status_secchi_revision.json").read_text(encoding="utf-8"))
        rows = BASE + [{"script": item["script"], "exit_code": item["exit_code"]} for item in revised]
        target = OUT / "run_status.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
        if len(revised) == 2 or revised[-1]["exit_code"]:
            break
    except (OSError, ValueError, IndexError, KeyError):
        pass
    time.sleep(10)
