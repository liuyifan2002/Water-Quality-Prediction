"""本机只读实时看板；不调用TabICLv2模型，也不写实验结果。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sqlite3
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import psutil


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v1"
V2 = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
PAGE = Path(__file__).resolve().parent / "实时进度看板" / "index.html"
TASKS = (
    ("chla_surface", "Chl-a 表层"), ("chla_middle", "Chl-a 中层"),
    ("chla_bottom", "Chl-a 底层"), ("secchi", "透明度 Secchi"),
    ("tn_surface", "TN 表层"), ("tn_middle", "TN 中层"),
    ("tn_bottom", "TN 底层"), ("tp_surface", "TP 表层"),
    ("tp_middle", "TP 中层"), ("tp_bottom", "TP 底层"),
)
PROCESS_SCRIPTS = (
    "93_resume_tabicl_slim_explainability_v1.py",
    "89_compute_tabicl_slim_shap_v1.py",
    "90_compute_tabicl_slim_ale_1d_v1.py",
    "92_compute_tabicl_slim_ale_2d_v1.py",
    "101_resume_tabicl_all10_explanation_v2.py",
    "99_compute_tabicl_all10_shap_v2.py",
    "100_compute_tabicl_all10_ale_1d_v2.py",
    "run_ale2d_after_ale1d.py", "92_compute_tabicl_slim_ale_2d_v1.py",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def valid_shap(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 77 or len({row["feature"] for row in rows}) != 77:
            return None
        if any(row.get("status") != "COMPUTED_PASS" for row in rows):
            return None
        return rows[0]
    except (OSError, UnicodeError, ValueError, KeyError):
        return None


def process_state() -> dict:
    active = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if "python" not in (proc.info["name"] or "").lower():
                continue
            args = proc.info["cmdline"] or []
            command = " ".join(args)
            if str(ROOT).lower() not in command.lower():
                continue
            script = next((name for name in PROCESS_SCRIPTS if name in command), None)
            if script is not None:
                active.append({"pid": proc.pid, "script": script,
                               "args": command.rsplit(script, 1)[-1].strip(),
                               "started": datetime.fromtimestamp(proc.info["create_time"]).isoformat(timespec="seconds")})
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    workers = [item for item in active if item["script"] not in
               (PROCESS_SCRIPTS[0], "101_resume_tabicl_all10_explanation_v2.py")]
    # Windows venv启动器和真实Python进程可能同时出现；它们属于同一计算单元。
    worker = max(workers, key=lambda item: item["started"]) if workers else None
    return {"running": bool(active), "worker": worker,
            "orchestrator_running": any(item["script"] == PROCESS_SCRIPTS[0] for item in active),
            "v2_orchestrator_running": any(item["script"] == "101_resume_tabicl_all10_explanation_v2.py" for item in active),
            "process_count_including_venv_launchers": len(active)}


def combined_snapshot(manifest: list[dict[str, str]], processes: dict) -> dict:
    """仅汇总已落盘的V1/V2检查点；历史门槛和V2计算覆盖分开。"""
    progress_path = V2 / "progress_status.json"
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    except (OSError, UnicodeError, ValueError):
        progress = {"stage": "STATUS_UNREADABLE"}
    registry_rows = read_csv(V2 / "run_registry.csv")
    coverage = {row["task"]: row for row in registry_rows if row.get("method") == "COVERAGE"}
    active_worker = processes.get("worker") or {}
    active_task = (active_worker.get("args", "").split("--task ", 1)[-1].split()[0]
                   if "--task " in active_worker.get("args", "") else progress.get("task", ""))
    active_feature = (active_worker.get("args", "").split("--feature ", 1)[-1].split()[0]
                      if "--feature " in active_worker.get("args", "") else progress.get("unit", ""))
    ale_query_progress = None
    if active_task and active_feature and active_worker.get("script") == "100_compute_tabicl_all10_ale_1d_v2.py":
        query_path = V2 / "ale_1d" / f"{active_task}_{active_feature}_query_manifest.csv"
        database = V2 / "checkpoints" / "ale_1d_local_effects.sqlite"
        if query_path.is_file() and database.is_file():
            try:
                total = len(read_csv(query_path))
                with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=0.2) as conn:
                    done, model_seconds = conn.execute(
                        "SELECT COUNT(*), COALESCE(SUM(elapsed_seconds), 0) FROM local_effect "
                        "WHERE task=? AND fold=? AND feature=?",
                        (active_task, "fold_1", active_feature),
                    ).fetchone()
                average = model_seconds / done if done >= 10 else None
                ale_query_progress = {"done": done, "total": total, "feature": active_feature,
                                      "avg_seconds_per_query": average,
                                      "eta_seconds": max(0, total - done) * average if average is not None else None}
            except (OSError, sqlite3.Error, ValueError):
                pass
    task_rows = []
    recent = []
    for task, label in TASKS:
        gate = read_csv(OUT / "gates" / f"{task}_p1_decision.csv")
        historical_gate = gate[0].get("status", "PENDING") if gate else (
            "HISTORICAL_BLOCKED" if task == "chla_surface" else "PENDING")
        samples = [row for row in manifest if row["task"] == task]
        core = optional = sensitivity = 0
        core_source = set()
        for sample in samples:
            filename = f"{sample['obs_id']}_seed42.csv.gz"
            v2_file = V2 / "shap" / task / filename
            v1_file = OUT / "shap" / task / filename
            source = v2_file if valid_shap(v2_file) else v1_file if valid_shap(v1_file) else None
            if source is None:
                continue
            if sample["phase"] == "CORE_3":
                core += 1
                core_source.add("V2" if source == v2_file else "V1")
            else:
                optional += 1
            recent.append((source.stat().st_mtime, label, sample["obs_id"], "SHAP"))
        for root in (V2, OUT):
            for path in (root / "shap" / task).glob("*_seed43.csv.gz"):
                if valid_shap(path):
                    sensitivity = 1
                    recent.append((path.stat().st_mtime, label, path.stem, "SHAP种子敏感性"))
        curves = {}
        for root in (OUT, V2):
            for path in (root / "ale_1d").glob(f"{task}_*_curve.csv"):
                rows = read_csv(path)
                if rows and rows[0].get("status") in (
                    "COMPUTED_PASS", "COMPUTED_WITH_NUMERICAL_LIMITATION"):
                    curves[path.name] = path
        for path in curves.values():
            recent.append((path.stat().st_mtime, label, path.name, "一维ALE"))
        outcome = coverage.get(task, {}).get("status", "")
        if outcome == "COMPUTED" and core == 3 and len(curves) >= 2:
            compute_status = "COMPUTED"
        elif active_task == task and active_worker:
            compute_status = "RUNNING"
        elif outcome in ("FAILED", "INCOMPLETE"):
            compute_status = "FAILED"
        elif outcome == "PARTIAL" or core or curves:
            compute_status = "PARTIAL"
        else:
            compute_status = "PENDING"
        task_rows.append({"key": task, "label": label,
                          "historical_gate": historical_gate,
                          "compute_status": compute_status,
                          "coverage_reason": coverage.get(task, {}).get("reason", ""),
                          "core": core, "core_total": 3,
                          "core_source": ", ".join(sorted(core_source)),
                          "optional": optional, "optional_max": 2,
                          "sensitivity": sensitivity,
                          "ale_1d": len(curves), "ale_1d_max": 2})
    recent.sort(key=lambda item: item[0], reverse=True)
    latest = None
    if recent:
        stamp, label, unit, method = recent[0]
        latest = {"task": label, "unit": unit, "method": method,
                  "saved_at": datetime.fromtimestamp(stamp).isoformat(timespec="seconds")}
    stage_detail = progress.get("detail", "")
    if ale_query_progress:
        stage_detail += (f" · {active_task}/{active_feature} 已落盘查询 "
                         f"{ale_query_progress['done']}/{ale_query_progress['total']}")
        if ale_query_progress["eta_seconds"] is not None:
            stage_detail += f" · 当前曲线预计余 {ale_query_progress['eta_seconds'] / 3600:.1f} 小时"
    return {"experiment": "TABICLV2_SLIM_EXPLAINABILITY_V2_ALL10",
            "stage": progress.get("stage", "NOT_STARTED"),
            "stage_detail": stage_detail,
            "active_ale_queries": ale_query_progress,
            "stage_updated_at": progress.get("updated_at", ""),
            "v2_orchestrator_running": processes["v2_orchestrator_running"],
            "v1_orchestrator_running": processes["orchestrator_running"],
            "active_task": active_task, "tasks": task_rows,
            "tasks_computed": sum(row["compute_status"] == "COMPUTED" for row in task_rows),
            "tasks_with_results": sum(row["core"] or row["ale_1d"] for row in task_rows),
            "core_done": sum(row["core"] for row in task_rows), "core_total": 30,
            "ale_1d_done": sum(row["ale_1d"] for row in task_rows), "ale_1d_max": 20,
            "latest_completed": latest}


def snapshot() -> dict:
    manifest = read_csv(OUT / "01_preregistered_shap_samples.csv")
    if not manifest:
        raise FileNotFoundError("预登记SHAP样本清单不存在")
    task_items = []
    completed = []
    actual_calls = 0
    total_completed_seconds = 0.0
    for task, label in TASKS:
        gate_rows = read_csv(OUT / "gates" / f"{task}_p1_decision.csv")
        gate = gate_rows[0].get("status", "PENDING") if gate_rows else "HISTORICAL_BLOCKED" if task == "chla_surface" else "PENDING"
        passed = gate == "P1_GATE_PASS"
        samples = [row for row in manifest if row["task"] == task]
        core = optional = sensitivity = 0
        for sample in samples:
            path = OUT / "shap" / task / f"{sample['obs_id']}_seed42.csv.gz"
            record = valid_shap(path)
            if record is None:
                continue
            if sample["phase"] == "CORE_3":
                core += 1
            else:
                optional += 1
            elapsed = float(record.get("elapsed_seconds") or 0)
            calls = int(float(record.get("actual_model_calls") or 0))
            actual_calls += calls
            total_completed_seconds += elapsed
            completed.append({"task": task, "label": label, "obs_id": sample["obs_id"],
                              "phase": sample["phase"], "elapsed_seconds": elapsed,
                              "calls": calls, "saved_at": path.stat().st_mtime})
        if passed:
            sensitivity = sum(valid_shap(path) is not None for path in
                              (OUT / "shap" / task).glob("*_seed43.csv.gz"))
        curves = 0
        ale_dir = OUT / "ale_1d"
        if passed and ale_dir.is_dir():
            for path in ale_dir.glob(f"{task}_*_curve.csv"):
                records = read_csv(path)
                if records and records[0].get("status") in (
                    "COMPUTED_PASS", "COMPUTED_WITH_NUMERICAL_LIMITATION"
                ):
                    curves += 1
        task_items.append({"key": task, "label": label, "gate": gate,
                           "eligible": passed, "core": core, "core_total": 3,
                           "optional": optional, "optional_total": 2,
                           "sensitivity": sensitivity, "ale_1d": curves,
                           "ale_1d_max": 2})
    eligible = sum(item["eligible"] for item in task_items)
    core_done = sum(item["core"] for item in task_items)
    latest = max(completed, key=lambda item: item["saved_at"]) if completed else None
    if latest:
        latest["saved_at"] = datetime.fromtimestamp(latest["saved_at"]).isoformat(timespec="seconds")
    logs = sorted((OUT / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    recent_log = None
    if logs:
        path = logs[0]
        recent_log = {"name": path.name, "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")}
    core_seconds = [item["elapsed_seconds"] for item in completed if item["phase"] == "CORE_3"]
    average_core = sum(core_seconds) / len(core_seconds) if core_seconds else None
    core_remaining = max(0, eligible * 3 - core_done)
    proc = process_state()
    return {"generated_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": "TABICLV2_SLIM_EXPLAINABILITY_V1", "tasks": task_items,
            "eligible_tasks": eligible, "blocked_tasks": len(TASKS) - eligible,
            "core_done": core_done, "core_total_eligible": eligible * 3,
            "optional_done": sum(item["optional"] for item in task_items),
            "sensitivity_done": sum(item["sensitivity"] for item in task_items),
            "ale_1d_done": sum(item["ale_1d"] for item in task_items),
            "actual_calls_shap": actual_calls,
            "average_core_seconds": average_core,
            "estimated_remaining_core_seconds": core_remaining * average_core if average_core else None,
            "latest_completed": latest, "recent_log": recent_log,
            "compute": proc, "combined": combined_snapshot(manifest, proc),
            "caveat": "仅统计已落盘且通过结构检查的结果；ETA仅针对合格任务的核心SHAP，不含敏感性、可选样本与ALE。"}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            data = PAGE.read_bytes()
            content_type = "text/html; charset=utf-8"
        elif route == "/api/status":
            try:
                data = json.dumps(snapshot(), ensure_ascii=False, allow_nan=False).encode("utf-8")
                status = 200
            except Exception as error:
                data = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")
                status = 503
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        elif route == "/health":
            data = b"OK"
            content_type = "text/plain; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        # 高频前端轮询不向磁盘写日志。
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    options = parser.parse_args()
    if options.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("安全限制：进度服务只允许绑定本机127.0.0.1")
    if not PAGE.is_file():
        raise SystemExit(f"找不到页面：{PAGE}")
    with ThreadingHTTPServer(("127.0.0.1", options.port), Handler) as server:
        print(f"只读实时看板：http://127.0.0.1:{options.port}", flush=True)
        server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
