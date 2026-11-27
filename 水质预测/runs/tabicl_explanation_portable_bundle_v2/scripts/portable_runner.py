"""单GPU串行、可断点续跑的便携调度入口。"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psutil

from portable_common import BUNDLE_ROOT, load_config, refresh_atomic_manifest, resolve, unit_counts


def write_progress(d, start, process, current=None, gpu_info=None):
    counts = unit_counts(d)
    reg_path = resolve("workspace/runs/tabicl_v2_final_explanation_complete_v1/13_final_explanation_master_registry.csv")
    if reg_path.is_file():
        reg = pd.read_csv(reg_path)
        ok = reg.loc[reg.status.astype(str).isin(["COMPLETED", "COMPLETED_2_FIXED_SAMPLES", "OUTPUT_RECOVERED_AFTER_SCHEDULER_PAUSE"])]
        recent = pd.to_numeric(ok.elapsed_seconds, errors="coerce").dropna().tail(8).tolist() if "elapsed_seconds" in ok else []
    else:
        recent = []
    moving = sum(recent[-5:]) / len(recent[-5:]) if recent else None
    eta = counts["pending"] * moving if moving is not None else None
    proc_ram = None
    if process and process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            proc_ram = (parent.memory_info().rss + sum(c.memory_info().rss for c in parent.children(recursive=True)
                                                       if c.is_running())) / 1024**2
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_ram = None
    status = {
        "current_task": (current or {}).get("task"), "current_fold": (current or {}).get("fold"),
        "current_method": (current or {}).get("method"),
        "current_atomic_unit": (current or {}).get("unit"), "completed_units": counts["completed"], "pending_units": counts["pending"],
        "failed_units": counts["failed"], "skipped_units": counts["skipped"],
        "elapsed_seconds": time.time() - start, "recent_unit_seconds": recent,
        "moving_average_seconds": moving, "estimated_remaining_seconds": eta,
        "last_checkpoint_time": datetime.now(timezone.utc).isoformat(),
        "gpu_name": (gpu_info or {}).get("name"), "gpu_utilization_pct": (gpu_info or {}).get("util"),
        "peak_vram_mb": (gpu_info or {}).get("vram_used_mb"), "process_ram_mb": proc_ram,
        "RAM_CACHE_MODE": "ON", "gpu_workers": 1, "query_protocol": "P1_SINGLE_QUERY",
        "ram_soft_limit_gb": 24,
        "eta_basis": "destination worker registry elapsed_seconds; no RTX2070 fixed estimates",
    }
    p = BUNDLE_ROOT / "progress_status.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def gpu_info() -> dict:
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        if p.returncode == 0:
            vals = [x.strip() for x in p.stdout.strip().split(",")]
            if len(vals) == 4:
                return {"name": vals[0], "util": vals[1], "vram_used_mb": vals[2], "vram_total_mb": vals[3]}
    except Exception:
        pass
    return {}


def active_unit_from_log(path: Path) -> dict:
    stem = path.stem
    tasks = ("chla_surface", "secchi", "tn_surface", "tp_middle")
    task = next((t for t in tasks if t in stem), None)
    method = None
    fold = None
    unit = stem
    if task:
        before, after = stem.split(task, 1)
        method = before.rstrip("_")
        match = re.search(r"(fold_[1-5])", after)
        fold = match.group(1) if match else None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines[-20:]):
            try:
                event = json.loads(line)
                if isinstance(event, dict) and event.get("unit"):
                    unit = str(event["unit"])
                    break
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return {"task":task,"fold":fold,"method":method,"unit":unit}


def main() -> int:
    cfg = load_config()
    cross = BUNDLE_ROOT / "logs/cross_device_validation.json"
    if not cross.is_file() or json.loads(cross.read_text(encoding="utf-8")).get("status") != "CROSS_DEVICE_NUMERICALLY_COMPATIBLE":
        print("STOP: cross-device validation is missing or not compatible; no worker launched.", file=sys.stderr)
        return 4
    lock = BUNDLE_ROOT / "checkpoints/gpu_worker.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        print(f"STOP: another GPU worker may be running: {lock}", file=sys.stderr)
        return 5
    try:
        py = sys.executable
        workspace = resolve(cfg["workspace_root"])
        main_script = workspace / "脚本/05_诊断分析/79_run_all_tabicl_explanation_experiments.py"
        if not main_script.is_file():
            raise FileNotFoundError(main_script)
        cpu_n = os.cpu_count() or 4
        threads = max(1, min(8, cpu_n // 4 or 1))
        env = os.environ.copy()
        env.update({"PYTHONHASHSEED":"42", "CUBLAS_WORKSPACE_CONFIG":":4096:8",
                    "TABICL_FIXED_NORM_ORDER":"none,power", "RAM_CACHE_MODE":"ON",
                    "OMP_NUM_THREADS":str(threads), "MKL_NUM_THREADS":str(threads),
                    "OPENBLAS_NUM_THREADS":str(threads), "TABICL_PRETRAINED_CHECKPOINT":str(resolve("model/tabicl-regressor-v2-20260212.ckpt")),
                    "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"})
        patch = workspace / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
        env["PYTHONPATH"] = str(patch) + os.pathsep + env.get("PYTHONPATH", "")
        logs = resolve("workspace/runs/tabicl_v2_final_explanation_complete_v1/logs")
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / "portable_master_resumable_run.log"
        retry_state = BUNDLE_ROOT / "checkpoints/portable_process_retry.json"
        retry_n = int(json.loads(retry_state.read_text(encoding="utf-8")).get("retry_count", 0)) if retry_state.is_file() else 0
        launches = 0
        process_retry_used = False
        final_code = 0
        start = time.time()
        peak_vram = 0
        while True:
            launches += 1
            with log.open("a", encoding="utf-8", buffering=1) as fp:
                fp.write(f"\n[{datetime.now(timezone.utc).isoformat()}] launch={launches} RAM_CACHE_MODE=ON OMP_NUM_THREADS={threads}\n")
                proc = subprocess.Popen([py, str(main_script), "--run-all"], cwd=workspace, env=env,
                                        stdout=fp, stderr=subprocess.STDOUT, text=True)
                while proc.poll() is None:
                    time.sleep(10)
                    d = refresh_atomic_manifest()
                    active_logs = sorted(logs.glob("*.stdout.log"), key=lambda q: q.stat().st_mtime, reverse=True)
                    if active_logs and time.time() - active_logs[0].stat().st_mtime < 90:
                        current = active_unit_from_log(active_logs[0])
                    else:
                        current = {"unit":"worker running; waiting for first checkpoint"}
                    gi = gpu_info()
                    try:
                        peak_vram = max(peak_vram, int(gi.get("vram_used_mb") or 0))
                    except (TypeError, ValueError):
                        pass
                    gi["peak_vram_mb"] = peak_vram or None
                    write_progress(d, start, proc, current, gi)
                final_code = int(proc.returncode)
            d = refresh_atomic_manifest()
            write_progress(d, start, None, None, gpu_info())
            if final_code == 0 or launches >= 2 or retry_n >= 1:
                break
            # One whole-process recovery is permitted. Per-unit retries are independently capped by the main runner.
            retry_n += 1
            retry_state.write_text(json.dumps({"retry_count":retry_n,"reason":"main scheduler exited nonzero; resume from atomic checkpoints",
                                               "time_utc":datetime.now(timezone.utc).isoformat()},ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"portable runner exit code={final_code}; launches={launches}")
        return final_code
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

