"""将已通过合并校验的外机独有SHAP无覆盖导入V2续跑目录。"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path

import psutil


HERE = Path(__file__).resolve().parent
ROOT = Path(r"E:\研究生\codex\水质预测")
RUN = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_no_writer() -> None:
    scripts = ("101_resume_tabicl_all10_explanation_v2.py", "99_compute_tabicl_all10_shap_v2.py", "100_compute_tabicl_all10_ale_1d_v2.py")
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in (proc.info["name"] or "").lower():
                continue
            cmd = " ".join(proc.info["cmdline"] or [])
            if str(ROOT).lower() in cmd.lower() and any(s in cmd for s in scripts):
                raise RuntimeError(f"仍有V2写入进程PID={proc.pid}，停止导入")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def main() -> None:
    mapping_path = HERE / "summary/source_path_mapping.csv"
    manifest_path = HERE / "summary/sha256_manifest.csv"
    with mapping_path.open(encoding="utf-8-sig", newline="") as f:
        mapping = list(csv.DictReader(f))
    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        manifest = {row["relative_path"]: row for row in csv.DictReader(f)}
    chosen = [row for row in mapping if row["source"] == "remote" and row["decision"] == "UNIQUE" and row["merged_relative_path"].startswith("merged/shap/")]
    if len(chosen) != 20:
        raise RuntimeError(f"预期20份外机独有SHAP，实得{len(chosen)}")
    check_no_writer()
    checks = []
    for row in chosen:
        rel = row["merged_relative_path"]
        src = HERE / rel
        dst = RUN / rel.removeprefix("merged/")
        if not src.is_file() or rel not in manifest or digest(src) != row["sha256"] or digest(src) != manifest[rel]["sha256"]:
            raise RuntimeError(f"合并源文件缺失或SHA256不符：{rel}")
        if dst.exists() and digest(dst) != row["sha256"]:
            raise RuntimeError(f"目标已有不同内容，禁止覆盖：{dst}")
        checks.append((src, dst, row["sha256"]))
    log = []
    for src, dst, expected in checks:
        if dst.exists():
            status = "ALREADY_IDENTICAL"
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY, 0o666)
            except FileExistsError:
                if digest(dst) != expected:
                    raise RuntimeError(f"并发写入不同目标：{dst}")
                status = "ALREADY_IDENTICAL"
            else:
                with os.fdopen(fd, "wb") as output, src.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
                status = "IMPORTED"
        if digest(dst) != expected:
            raise RuntimeError(f"导入后SHA256不符：{dst}")
        log.append({"source": str(src), "destination": str(dst), "sha256": expected, "status": status})
    with (HERE / "summary/import_to_v2_run.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "destination", "sha256", "status"])
        w.writeheader()
        w.writerows(log)
    print(f"IMPORTED {sum(x['status'] == 'IMPORTED' for x in log)}; IDENTICAL {sum(x['status'] == 'ALREADY_IDENTICAL' for x in log)}; VERIFIED 20")


if __name__ == "__main__":
    main()
