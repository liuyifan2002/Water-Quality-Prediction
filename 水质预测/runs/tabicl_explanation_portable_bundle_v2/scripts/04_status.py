"""显示checkpoint队列、动态ETA与当前硬件资源。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import psutil

from portable_common import BUNDLE_ROOT, refresh_atomic_manifest, unit_counts


def main() -> int:
    d = refresh_atomic_manifest()
    c = unit_counts(d)
    p = BUNDLE_ROOT / "progress_status.json"
    status = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    gpu = status.get("gpu_name") or "尚未检测"
    util = status.get("gpu_utilization_pct")
    vram = status.get("peak_vram_mb")
    ram = status.get("process_ram_mb")
    eta = status.get("estimated_remaining_seconds")
    last = status.get("last_checkpoint_time")
    current = status.get("current_atomic_unit") or "idle / not started"
    print(f"TOTAL_UNITS={c['total']}\nCOMPLETED={c['completed']}\nPENDING={c['pending']}\nFAILED={c['failed']}\nSKIPPED={c['skipped']}")
    print("\nCurrent:", current)
    print("Completed:", c["completed"])
    print("Pending:", c["pending"])
    print("Last checkpoint:", last or "not yet written")
    print("Average SHAP/sample:", "destination rate not yet measured")
    print("Average ALE/query:", "destination rate not yet measured")
    print("Estimated remaining:", f"{eta:.0f} sec" if isinstance(eta,(int,float)) else "pending destination timing samples")
    print("GPU:", gpu, "utilization:", util, "%")
    ram_text = f"{float(ram):.0f} MB" if isinstance(ram, (int, float)) else f"current process unavailable (system used {psutil.virtual_memory().used/1024**3:.1f} GB)"
    print("VRAM:", f"{vram} MB" if vram is not None else "not sampled", "; RAM:", ram_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
