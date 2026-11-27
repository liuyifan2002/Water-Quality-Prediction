"""构建十任务P1解释实验的可迁移运行包并生成哈希自检。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/tabicl_explanation_portable_bundle_v2"
BUNDLE = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10_portable_bundle_v1"
ZIP_PATH = BUNDLE.with_suffix(".zip")
REPORT = BUNDLE.with_name("PORTABLE_BUNDLE_REPORT.md")
V1 = ROOT / "runs/tabicl_v2_slim_explainability_v1"
V2 = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
TASKS = ("chla_surface", "chla_middle", "chla_bottom", "secchi",
         "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(source: Path, relative: str, required: bool = True) -> bool:
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"迁移包缺少必需输入：{source}")
        return False
    target = BUNDLE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(source) != digest(target):
        raise RuntimeError(f"复制后SHA256不一致：{relative}")
    return True


def valid_shap(path: Path) -> bool:
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return (len(rows) == 77 and len({row.get("feature") for row in rows}) == 77
                and all(row.get("status") == "COMPUTED_PASS" for row in rows))
    except (OSError, EOFError, UnicodeError, csv.Error):
        return False


def copy_v1_checkpoint() -> tuple[int, int]:
    for name in ("00_frozen_protocol_task_registry.csv", "01_preregistered_shap_samples.csv",
                 "02_compute_budget.csv", "03_reuse_pending_blocked.csv"):
        copy_file(V1 / name, f"runs/tabicl_v2_slim_explainability_v1/{name}")
    for name in ("04_asset_audit_decision.md", "04_task_decisions.csv", "05_xgb_rank_reference.csv",
                 "06_figure_inventory.csv", "07_最终实验决策.md", "08_actual_calls_and_reuse.csv",
                 "09_run_registry.csv", "10_numerical_sensitivity.csv"):
        copy_file(V1 / name, f"runs/tabicl_v2_slim_explainability_v1/{name}", required=False)
    shap_n = 0
    for path in (V1 / "shap").rglob("*.csv.gz"):
        if ".tmp" in path.name or not valid_shap(path):
            continue
        relative = path.relative_to(ROOT).as_posix()
        copy_file(path, relative)
        shap_n += 1
    ale_n = 0
    for folder in ("gates", "ale_1d", "ale_2d"):
        source_dir = V1 / folder
        if not source_dir.is_dir():
            continue
        for path in source_dir.rglob("*"):
            if not path.is_file() or path.name.endswith(".tmp") or path.name.endswith("-wal") or path.name.endswith("-shm"):
                continue
            copy_file(path, path.relative_to(ROOT).as_posix())
            if folder == "ale_1d" and path.name.endswith("_curve.csv"):
                ale_n += 1
    db = V1 / "checkpoints/ale_1d_local_effects.sqlite"
    if db.is_file():
        target = BUNDLE / "runs/tabicl_v2_slim_explainability_v1/checkpoints/ale_1d_local_effects.sqlite"
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as source:
            with sqlite3.connect(target) as dest:
                source.backup(dest)
                if dest.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("V1 ALE checkpoint SQLite完整性检查失败")
    return shap_n, ale_n


def prepare() -> tuple[int, int]:
    if BUNDLE.exists() or ZIP_PATH.exists() or ZIP_PATH.with_suffix(".zip.sha256").exists():
        raise FileExistsError(f"目标包已存在，拒绝覆盖：{BUNDLE}")
    if not (V2 / "PAUSE_V2.flag").is_file():
        raise RuntimeError("本机V2没有暂停标记；拒绝制作迁移快照")
    active = []
    import psutil
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            command = " ".join(proc.info["cmdline"] or [])
            if "101_resume_tabicl_all10_explanation_v2.py" in command and str(ROOT).lower() in command.lower():
                active.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    if active:
        raise RuntimeError(f"本机V2调度器仍运行，拒绝打包：{active}")

    BUNDLE.mkdir(parents=True)
    # 已验证便携包中的依赖锁、运行时清单与CUDA checkpoint作为来源，输入仍以当前冻结项目为准。
    for name in ("requirements-lock.txt", "pip_freeze.txt", "environment_manifest.json"):
        copy_file(BASE / name, name)
    copy_file(BASE / "model/tabicl-regressor-v2-20260212.ckpt", "model/tabicl-regressor-v2-20260212.ckpt")
    copy_file(BASE / "workspace/models/tabiclv2_adapter.py", "models/tabiclv2_adapter.py")
    copy_file(ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv",
              "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv")
    copy_file(ROOT / "runs/tabiclv2_pilot/01_strict_feature_manifest.csv",
              "runs/tabiclv2_pilot/01_strict_feature_manifest.csv")
    copy_file(ROOT / "runs/multisource_ablation_v1/00_feature_source_taxonomy.csv",
              "runs/multisource_ablation_v1/00_feature_source_taxonomy.csv")
    copy_file(ROOT / "runs/modern_baseline_gate_v1/01_secchi_manifest.csv",
              "runs/modern_baseline_gate_v1/01_secchi_manifest.csv")
    for family in ("chla", "tn", "tp"):
        family_root = ROOT / f"runs/{family}_family_v2"
        for depth, number in (("surface", "03"), ("middle", "04"), ("bottom", "05")):
            manifest = family_root / f"{number}_{family}_{depth}_manifest.csv"
            copy_file(manifest, f"runs/{family}_family_v2/{manifest.name}")
    patch_root = ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
    patch_count = 0
    for path in patch_root.rglob("*"):
        if path.is_file():
            copy_file(path, path.relative_to(ROOT).as_posix())
            patch_count += 1
    for number, name in (("66", "66_run_tabiclv2_explainability_v2.py"),
                         ("79", "79_run_all_tabicl_explanation_experiments.py"),
                         ("88", "88_run_tabicl_slim_explainability_v1.py"),
                         ("96", "96_serve_tabicl_slim_dashboard_v1.py"),
                         ("99", "99_compute_tabicl_all10_shap_v2.py"),
                         ("100", "100_compute_tabicl_all10_ale_1d_v2.py"),
                         ("101", "101_resume_tabicl_all10_explanation_v2.py")):
        copy_file(ROOT / "脚本/05_诊断分析" / name, f"脚本/05_诊断分析/{name}")
    copy_file(ROOT / "脚本/05_诊断分析/实时进度看板/index.html",
              "脚本/05_诊断分析/实时进度看板/index.html")
    legacy = ROOT / "runs/tabicl_v2_final_explanation_complete_v1"
    if legacy.is_dir():
        for path in legacy.rglob("*"):
            if path.is_file() and not path.name.endswith((".tmp", "-wal", "-shm")):
                copy_file(path, path.relative_to(ROOT).as_posix())

    # V2 begins ready on the destination PC. The local pause marker is intentionally not included.
    for path in ("logs", "shap", "ale_1d", "checkpoints", "figures"):
        (BUNDLE / "runs/tabicl_v2_slim_explainability_v2_all10" / path).mkdir(parents=True, exist_ok=True)
    v2_config = {"version": "TABICLV2_SLIM_EXPLAINABILITY_V2_ALL10",
                           "tasks": list(TASKS), "fold": "fold_1", "root_relative_paths": True,
                           "query_protocol": "P1_SINGLE_QUERY", "query_shape": [1, 77],
                           "dtype": "float32", "shap": {"algorithm": "shap.PermutationExplainer",
                           "max_evals": 155, "background_n": 16, "core_samples_per_task": 3},
                           "ale": {"max_features_per_task": 2, "support_from_full_test_features": True},
                           "historical_blocked_is_quality_evidence_not_compute_gate": True,
                           "gpu_workers": 1,
                           "pause_flag": "runs/tabicl_v2_slim_explainability_v2_all10/PAUSE_V2.flag"}
    (BUNDLE / "portable_config.json").write_text(json.dumps(v2_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (BUNDLE / "runs/tabicl_v2_slim_explainability_v2_all10/00_实验协议说明.md").write_text(
        (V2 / "00_实验协议说明.md").read_text(encoding="utf-8"), encoding="utf-8")
    (BUNDLE / "runs/tabicl_v2_slim_explainability_v2_all10/progress_status.json").write_text(
        json.dumps({"experiment": "TABICLV2_SLIM_EXPLAINABILITY_V2_ALL10", "stage": "READY_ON_DESTINATION",
                    "task": "", "unit": "", "detail": "目标机启动后读取检查点并续跑十任务",
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "pid": None,
                    "v1_gate_is_quality_evidence_not_compute_gate": True}, ensure_ascii=False, indent=2), encoding="utf-8")

    scripts_root = BUNDLE / "scripts"
    scripts_root.mkdir()
    create_portable_scripts(scripts_root)
    shap_n, ale_n = copy_v1_checkpoint()
    return shap_n, ale_n


def create_portable_scripts(directory: Path) -> None:
    # Templates are committed beside the builder to keep output creation reproducible.
    template_root = Path(__file__).resolve().parent / "tabicl_all10_portable_templates"
    if not template_root.is_dir():
        raise FileNotFoundError(f"缺少便携脚本模板目录：{template_root}")
    for path in template_root.rglob("*"):
        if path.is_file():
            copy_file(path, (directory / path.relative_to(template_root)).relative_to(BUNDLE).as_posix())


def make_manifest() -> list[dict]:
    rows = []
    for path in sorted(BUNDLE.rglob("*")):
        if not path.is_file() or path.name in {"file_manifest_sha256.csv", "portable_bundle_self_check.json"}:
            continue
        if any(part in {"logs", ".venv"} for part in path.relative_to(BUNDLE).parts):
            continue
        if path.name == "progress_status.json" or path.name == "run_registry.csv":
            continue
        rows.append({"relative_path": path.relative_to(BUNDLE).as_posix(), "sha256": digest(path),
                     "bytes": path.stat().st_size, "mutable": False})
    return rows


def self_check(rows: list[dict]) -> dict:
    errors = []
    for row in rows:
        path = BUNDLE / row["relative_path"]
        if not path.is_file() or digest(path) != row["sha256"]:
            errors.append(f"哈希或文件缺失：{row['relative_path']}")
    tasks = pd.read_csv(BUNDLE / "runs/tabicl_v2_slim_explainability_v1/00_frozen_protocol_task_registry.csv")
    samples = pd.read_csv(BUNDLE / "runs/tabicl_v2_slim_explainability_v1/01_preregistered_shap_samples.csv")
    features = pd.read_csv(BUNDLE / "runs/tabiclv2_pilot/01_strict_feature_manifest.csv")
    if set(tasks.task.astype(str)) != set(TASKS):
        errors.append("冻结任务注册表与十任务名单不一致")
    if samples.task.nunique() != 10:
        errors.append("预登记解释样本未覆盖十任务")
    selected = features.loc[features.included_in_pilot.astype(bool), "feature"].astype(str).tolist()
    if len(selected) != 77 or len(set(selected)) != 77:
        errors.append("77特征清单检查失败")
    checkpoint = BUNDLE / "model/tabicl-regressor-v2-20260212.ckpt"
    expected_checkpoint = set(tasks.checkpoint_sha256.astype(str))
    if digest(checkpoint) not in expected_checkpoint:
        errors.append("模型checkpoint hash与冻结注册表不一致")
    pause = BUNDLE / "runs/tabicl_v2_slim_explainability_v2_all10/PAUSE_V2.flag"
    if pause.exists():
        errors.append("迁移包错误包含本机暂停标记")
    for required in ("scripts/00_check_hardware.ps1", "scripts/01_setup_environment.ps1",
                     "scripts/02_resume_remaining.ps1", "scripts/03_start_dashboard.ps1",
                     "scripts/04_status.ps1", "scripts/validate_bundle.py"):
        if not (BUNDLE / required).is_file():
            errors.append(f"入口脚本缺失：{required}")
    runnable = list((BUNDLE / "脚本/05_诊断分析").glob("*.py"))
    runnable.extend((BUNDLE / "portable_config.json", BUNDLE / "scripts/02_resume_remaining.ps1"))
    for path in runnable:
        content = path.read_text(encoding="utf-8", errors="ignore")
        if "E:/研究生/codex/水质预测" in content or "E:\\研究生\\codex\\水质预测" in content:
            errors.append(f"运行代码含原电脑绝对路径：{path.relative_to(BUNDLE).as_posix()}")
    bundle_db = BUNDLE / "runs/tabicl_v2_slim_explainability_v1/checkpoints/ale_1d_local_effects.sqlite"
    if bundle_db.is_file():
        with sqlite3.connect(f"file:{bundle_db.as_posix()}?mode=ro", uri=True) as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                errors.append("SQLite完整性检查失败")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "task_count": int(tasks.task.nunique()), "feature_count": len(selected),
            "v1_shap_reused": int(sum(1 for path in (BUNDLE / "runs/tabicl_v2_slim_explainability_v1/shap").rglob("*.csv.gz") if valid_shap(path))),
            "manifest_files": len(rows), "pause_marker_in_bundle": pause.exists(),
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds")}


def write_scripts_and_readme() -> None:
    # Bundle launcher templates are materialized before validation and included in the static manifest.
    pass


def main() -> None:
    reused_shap, reused_ale = prepare()
    (BUNDLE / "README_FIRST.md").write_text(
        "# TabICLv2十任务解释运行包\n\n"
        "1. 将ZIP复制到Windows电脑并解压到英文路径，例如 `D:\\tabicl_all10\\`。\n"
        "2. 在PowerShell进入解压目录。\n"
        "3. 执行 `powershell -ExecutionPolicy Bypass -File .\\scripts\\00_check_hardware.ps1`。\n"
        "4. 执行 `powershell -ExecutionPolicy Bypass -File .\\scripts\\01_setup_environment.ps1`。\n"
        "5. 执行 `powershell -ExecutionPolicy Bypass -File .\\scripts\\02_resume_remaining.ps1` 开始/续跑。\n"
        "6. 执行 `powershell -ExecutionPolicy Bypass -File .\\scripts\\03_start_dashboard.ps1`，浏览器打开 `http://127.0.0.1:8765/`。\n"
        "7. 用 `powershell -ExecutionPolicy Bypass -File .\\scripts\\04_status.ps1` 查看十任务状态。\n\n"
        "任务范围为十项任务各自fold 1。每个查询仍为P1单查询、77特征、float32；SHAP为PermutationExplainer，155 evaluations、16训练背景。V1历史失败保留，V2先计算再按证据判断。模型不重训。\n",
        encoding="utf-8")
    rows = make_manifest()
    pd.DataFrame(rows).to_csv(BUNDLE / "file_manifest_sha256.csv", index=False, encoding="utf-8-sig")
    check = self_check(rows)
    (BUNDLE / "portable_bundle_self_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    if check["status"] != "PASS":
        raise RuntimeError("运行包self-check失败：" + "; ".join(check["errors"]))
    with zipfile.ZipFile(ZIP_PATH, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in BUNDLE.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(BUNDLE.parent).as_posix())
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP完整性检查失败：{bad}")
    zip_hash = digest(ZIP_PATH)
    ZIP_PATH.with_suffix(ZIP_PATH.suffix + ".sha256").write_text(f"{zip_hash}  {ZIP_PATH.name}\n", encoding="ascii")
    report = [
        "# TabICLv2十任务解释迁移包报告", "",
        f"- 生成状态：`PORTABLE_BUNDLE_READY`", f"- Self-check：`{check['status']}`",
        f"- 本机V2：`PAUSED_FOR_PORTABLE_HANDOFF`（V2 GPU计算单元数0）",
        f"- V1完成结果快照：{reused_shap}个V1注册SHAP样本、{reused_ale}条V1 ALE曲线；另包含历史fold1 pilot两条SHAP及两条ALE完整记录。V1仍在本机运行，其后续新增结果不会自动同步进本压缩包。",
        f"- 任务：十项预登记任务，各自fold 1；现有V1结果在新机按哈希复用。",
        f"- ZIP大小：{ZIP_PATH.stat().st_size:,} bytes；SHA256：`{zip_hash}`",
        f"- 模型SHA256：`{digest(BUNDLE / 'model/tabicl-regressor-v2-20260212.ckpt')}`",
        f"- 77 strict feature manifest SHA256：`{digest(BUNDLE / 'runs/tabiclv2_pilot/01_strict_feature_manifest.csv')}`",
        f"- 已保存完成SHAP数量：{reused_shap}；后续剩余数量以目标机状态页为准。",
        "- 本机V1当前任务不在压缩包中运行；迁移包包含打包时已完成的只读快照，不能含打包后本机新增检查点。",
        "- 新机需联网安装锁定依赖；运行包不含venv。",
        "- 不启用batch query、FP16、autocast或其他改变预测语义的优化；单GPU串行。",
        "- 运行代码和配置使用解压根目录相对路径；旧V1审计CSV里的绝对路径仅是历史来源元数据。",
        "", "## 新电脑启动", "", "```powershell",
        "powershell -ExecutionPolicy Bypass -File .\\scripts\\00_check_hardware.ps1",
        "powershell -ExecutionPolicy Bypass -File .\\scripts\\01_setup_environment.ps1",
        "powershell -ExecutionPolicy Bypass -File .\\scripts\\02_resume_remaining.ps1",
        "powershell -ExecutionPolicy Bypass -File .\\scripts\\03_start_dashboard.ps1",
        "```", "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "PORTABLE_BUNDLE_READY", "bundle": str(BUNDLE),
                      "zip": str(ZIP_PATH), "zip_bytes": ZIP_PATH.stat().st_size,
                      "sha256": zip_hash, "reused_shap": reused_shap, "reused_ale": reused_ale,
                      "self_check": check}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
