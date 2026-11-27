"""十任务P1解释V2串行续跑；V1运行时等待，不修改V1门槛或结果。"""

from __future__ import annotations

import csv
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import msvcrt
import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "脚本/05_诊断分析"
V1 = ROOT / "runs/tabicl_v2_slim_explainability_v1"
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
PATCH = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
PAUSE_FLAG = OUT / "PAUSE_V2.flag"
TASKS = (
    "chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface",
    "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom",
)
V1_PROCESSES = (
    "93_resume_tabicl_slim_explainability_v1.py", "89_compute_tabicl_slim_shap_v1.py",
    "90_compute_tabicl_slim_ale_1d_v1.py", "92_compute_tabicl_slim_ale_2d_v1.py",
)


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def state(stage: str, task: str = "", unit: str = "", detail: str = "") -> None:
    atomic_text(OUT / "progress_status.json", json.dumps({
        "experiment": "TABICLV2_SLIM_EXPLAINABILITY_V2_ALL10",
        "stage": stage, "task": task, "unit": unit, "detail": detail,
        "updated_at": utc_now(), "pid": os.getpid(), "v1_gate_is_quality_evidence_not_compute_gate": True,
    }, ensure_ascii=False, indent=2))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def v1_running() -> list[dict]:
    matches = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in (process.info["name"] or "").lower():
                continue
            command = " ".join(process.info["cmdline"] or [])
            if str(ROOT).lower() not in command.lower():
                continue
            matched = next((name for name in V1_PROCESSES if name in command), None)
            if matched:
                matches.append({"pid": process.pid, "script": matched})
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    return matches


def verified_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha(source) != sha(destination):
            raise RuntimeError(f"已存在不同内容的V2文件，禁止覆盖：{destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha(source) != sha(destination):
        raise RuntimeError(f"复制后哈希不一致：{destination}")


def copy_v1_assets() -> None:
    if v1_running():
        raise RuntimeError("V1进程仍运行，不复制活动检查点")
    for name in ("00_frozen_protocol_task_registry.csv", "01_preregistered_shap_samples.csv",
                 "02_compute_budget.csv", "03_reuse_pending_blocked.csv"):
        verified_copy(V1 / name, OUT / "inputs" / name)
        verified_copy(V1 / name, OUT / name)
    for folder in ("gates", "shap", "ale_1d", "ale_2d"):
        source = V1 / folder
        if source.is_dir():
            for item in source.rglob("*"):
                if item.is_file():
                    verified_copy(item, OUT / folder / item.relative_to(source))
    source_db = V1 / "checkpoints" / "ale_1d_local_effects.sqlite"
    target_db = OUT / "checkpoints" / "ale_1d_local_effects.sqlite"
    if source_db.is_file() and not target_db.is_file():
        target_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True) as source:
            with sqlite3.connect(target_db) as target:
                source.backup(target)
                if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("复制的ALE SQLite完整性检查失败")
    state("V1_RESULTS_IMPORTED", detail="V1已结束；已按哈希复制已完成结果与可恢复ALE检查点")


def manifest() -> pd.DataFrame:
    return pd.read_csv(OUT / "inputs" / "01_preregistered_shap_samples.csv", dtype={"obs_id": "string"})


def valid_shap(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = pd.read_csv(path)
        return (len(value) == 77 and value.feature.nunique() == 77
                and value.status.eq("COMPUTED_PASS").all()
                and float(value.additivity_abs_error.iloc[0]) <= 0.02)
    except (OSError, KeyError, ValueError):
        return False


def core_ids(task: str) -> list[str]:
    samples = manifest()
    return samples.loc[samples.task.eq(task) & samples.phase.eq("CORE_3"), "obs_id"].astype(str).tolist()


def optional_ids(task: str) -> list[str]:
    samples = manifest()
    return samples.loc[samples.task.eq(task) & samples.phase.eq("OPTIONAL_2"), "obs_id"].astype(str).tolist()


def completed_ids(task: str, kind: str) -> list[str]:
    ids = core_ids(task) if kind == "CORE_3" else optional_ids(task)
    return [obs for obs in ids if valid_shap(OUT / "shap" / task / f"{obs}_seed42.csv.gz")]


def median_id(task: str) -> str:
    samples = manifest()
    match = samples.loc[samples.task.eq(task) & samples.target_quantile.eq(0.5), "obs_id"].astype(str)
    if len(match) != 1:
        raise RuntimeError(f"{task}: 预登记中位预测样本不是唯一")
    return match.iloc[0]


def needs_optional(task: str) -> bool:
    source = SCRIPTS / "93_resume_tabicl_slim_explainability_v1.py"
    spec = importlib.util.spec_from_file_location("v2_reused_budget_rule", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = OUT
    return bool(module.need_optional_two(task))


def register(task: str, method: str, unit: str, status: str, output_path: Path | None, reason: str = "") -> None:
    registry = OUT / "run_registry.csv"
    columns = ["task", "fold", "method", "unit", "status", "output_path", "reason", "updated_at"]
    if registry.is_file():
        records = pd.read_csv(registry, dtype=str).fillna("")
    else:
        records = pd.DataFrame(columns=columns)
    keep = ~((records.task == task) & (records.method == method) & (records.unit == unit))
    records = records.loc[keep]
    records = pd.concat([records, pd.DataFrame([{"task": task, "fold": "fold_1",
        "method": method, "unit": unit, "status": status,
        "output_path": str(output_path) if output_path else "", "reason": reason,
        "updated_at": utc_now()}])], ignore_index=True)
    temporary = registry.with_suffix(".csv.tmp")
    records.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(registry)


def run_unit(script: str, args: list[str], task: str, method: str, unit: str) -> bool:
    state("RUNNING", task, unit, f"{script} {' '.join(args)}")
    log = OUT / "logs" / f"{task}_{method}_{unit}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"
    env["PYTHONPATH"] = str(PATCH) + os.pathsep + env.get("PYTHONPATH", "")
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    status = "FINISHED" if process.returncode == 0 else "FAILED"
    register(task, method, unit, status, log, f"exit_code={process.returncode}")
    print(json.dumps({"task": task, "method": method, "unit": unit,
        "status": status, "exit_code": process.returncode, "log": str(log)}, ensure_ascii=False), flush=True)
    return process.returncode == 0


def curve_done(task: str, feature: str) -> bool:
    path = OUT / "ale_1d" / f"{task}_{feature}_curve.csv"
    if not path.is_file():
        return False
    try:
        curve = pd.read_csv(path)
        return len(curve) > 0 and curve.status.iloc[0] in (
            "COMPUTED_PASS", "COMPUTED_WITH_NUMERICAL_LIMITATION")
    except (OSError, KeyError, ValueError):
        return False


def process_task(task: str) -> None:
    historical = OUT / "gates" / f"{task}_p1_decision.csv"
    prior = pd.read_csv(historical).status.iloc[0] if historical.is_file() else "HISTORICAL_BLOCKED"
    register(task, "LEGACY_P1_GATE", task, prior, historical if historical.is_file() else None,
             "V1门槛作为质量证据保留；V2计算不把它当禁令")
    if len(completed_ids(task, "CORE_3")) < 3:
        if not run_unit("99_compute_tabicl_all10_shap_v2.py", ["--task", task, "--phase", "CORE_3"],
                        task, "SHAP", "CORE_3"):
            register(task, "COVERAGE", task, "INCOMPLETE", None, "核心SHAP单元失败；不阻断其他任务")
            return
    for obs in core_ids(task):
        path = OUT / "shap" / task / f"{obs}_seed42.csv.gz"
        register(task, "SHAP_CORE", obs, "COMPUTED" if valid_shap(path) else "FAILED", path)
    if len(completed_ids(task, "CORE_3")) != 3:
        register(task, "COVERAGE", task, "INCOMPLETE", None, "核心SHAP不足3个样本")
        return
    median = median_id(task)
    sensitivity = OUT / "shap" / task / f"{median}_seed43.csv.gz"
    if not valid_shap(sensitivity):
        run_unit("99_compute_tabicl_all10_shap_v2.py", ["--task", task, "--phase", "SENSITIVITY_1"],
                 task, "SHAP", "SENSITIVITY_1")
    register(task, "SHAP_SENSITIVITY", median, "COMPUTED" if valid_shap(sensitivity) else "FAILED", sensitivity)
    if valid_shap(sensitivity):
        optional = needs_optional(task)
        if optional and len(completed_ids(task, "OPTIONAL_2")) < 2:
            run_unit("99_compute_tabicl_all10_shap_v2.py", ["--task", task, "--phase", "OPTIONAL_2"],
                     task, "SHAP", "OPTIONAL_2")
        for obs in optional_ids(task):
            path = OUT / "shap" / task / f"{obs}_seed42.csv.gz"
            register(task, "SHAP_OPTIONAL", obs,
                     "COMPUTED" if valid_shap(path) else "NOT_REQUIRED" if not optional else "FAILED", path)
    screening = OUT / "ale_1d" / f"{task}_feature_screening.csv"
    if not screening.is_file():
        if not run_unit("100_compute_tabicl_all10_ale_1d_v2.py", ["--task", task, "--screen"],
                        task, "ALE_SCREEN", "FEATURES"):
            register(task, "COVERAGE", task, "PARTIAL", None, "ALE覆盖审查失败")
            return
    candidates = pd.read_csv(screening)
    selected = candidates.loc[candidates.proposed.astype(bool), "feature"].astype(str).tolist()
    for feature in selected:
        if not curve_done(task, feature):
            run_unit("100_compute_tabicl_all10_ale_1d_v2.py",
                     ["--task", task, "--calculate", "--feature", feature],
                     task, "ALE_1D", feature)
        path = OUT / "ale_1d" / f"{task}_{feature}_curve.csv"
        register(task, "ALE_1D_CURVE", feature,
                 "COMPUTED" if curve_done(task, feature) else "FAILED", path)
    coverage = "COMPUTED" if len(selected) == 2 and all(curve_done(task, item) for item in selected) else "PARTIAL"
    register(task, "COVERAGE", task, coverage, screening,
             f"core3=3; selected_ale={len(selected)}; computed_ale={sum(curve_done(task, item) for item in selected)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "orchestrator.lock"
    with lock_path.open("a+b") as lock:
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("已有V2串行调度器运行；不会启动第二个GPU队列")
        if PAUSE_FLAG.exists():
            state("PAUSED_BY_USER", detail="发现PAUSE_V2.flag；未开始导入或GPU计算")
            return
        state("WAITING_FOR_V1")
        while True:
            if PAUSE_FLAG.exists():
                state("PAUSED_BY_USER", detail="V1仍运行；V2在等待阶段安全暂停")
                return
            active = v1_running()
            if not active:
                time.sleep(12)
                if not v1_running():
                    break
            state("WAITING_FOR_V1", detail=f"{len(active)}个V1相关进程仍在运行；不会并行调用GPU")
            time.sleep(30)
        if PAUSE_FLAG.exists():
            state("PAUSED_BY_USER", detail="V1已结束；暂停标记阻止后续计算")
            return
        copy_v1_assets()
        for task in TASKS:
            if PAUSE_FLAG.exists():
                state("PAUSED_BY_USER", task=task, detail="在任务边界检测到暂停标记；当前最小单元已结束")
                return
            try:
                process_task(task)
            except Exception as error:
                register(task, "COVERAGE", task, "FAILED", None, repr(error))
                print(json.dumps({"task": task, "status": "FAILED", "reason": repr(error)},
                                 ensure_ascii=False), flush=True)
                continue
        status = pd.read_csv(OUT / "run_registry.csv", dtype=str)
        coverage = status.loc[status.method.eq("COVERAGE")]
        done = int(coverage.status.eq("COMPUTED").sum())
        state("COMPUTATION_QUEUE_FINISHED", detail=f"{done}/10任务完成核心SHAP及两条ALE；解释可靠性待单独评估")
        atomic_text(OUT / "00_计算覆盖说明.md", "\n".join([
            "# TabICLv2十任务P1解释V2：计算覆盖说明", "",
            "本版本仅改变V1资格门槛在调度中的作用：历史批量预测严格等价失败仍作为质量证据保存，不再作为计算禁令。",
            "77特征、预登记查询、训练背景、P1单查询预测、SHAP预算、ALE分箱和样本支持标准均沿用V1。",
            "V1完成结果按文件哈希复制复用，未重新计算；新结果与V1目录隔离。",
            "`COMPUTED`表示计算覆盖，不表示结果科学可靠、与历史批量预测严格等价或环境因果关系。",
            f"本次调度结束时核心SHAP与2条ALE均已落盘的任务：{done}/10。",
            "详细失败或不足支持原因见run_registry.csv和logs/；不得据此改写V1的BLOCKED记录。",
        ]) + "\n")


def preflight() -> None:
    required = [SCRIPTS / name for name in (
        "88_run_tabicl_slim_explainability_v1.py",
        "93_resume_tabicl_slim_explainability_v1.py",
        "99_compute_tabicl_all10_shap_v2.py",
        "100_compute_tabicl_all10_ale_1d_v2.py",
    )]
    required.extend(V1 / name for name in (
        "00_frozen_protocol_task_registry.csv",
        "01_preregistered_shap_samples.csv", "02_compute_budget.csv",
    ))
    missing = [str(path) for path in required if not path.is_file()]
    samples = pd.read_csv(V1 / "01_preregistered_shap_samples.csv") if not missing else pd.DataFrame()
    listed = sorted(samples.task.dropna().unique().tolist()) if len(samples) else []
    if listed != sorted(TASKS):
        missing.append(f"预登记任务不等于十项：{listed}")
    result = {"status": "PREFLIGHT_PASS" if not missing else "PREFLIGHT_FAILED",
              "task_count": len(listed), "v1_active_processes": v1_running(), "issues": missing}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="只读检查，不启动GPU计算或队列")
    arguments = parser.parse_args()
    preflight() if arguments.preflight else main()
