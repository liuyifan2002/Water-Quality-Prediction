"""构建TabICLv2解释实验可迁移运行包；仅复制/校验资产，不调用模型预测。"""
from __future__ import annotations

import csv
import ast
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "runs/tabicl_explanation_portable_bundle_v2"
WORKSPACE = BUNDLE / "workspace"
ZIP_PATH = ROOT / "runs/tabicl_explanation_portable_bundle_v2.zip"
REPORT_PATH = BUNDLE / "PORTABLE_BUNDLE_REPORT_V2.md"
EXTERNAL_REPORT = ROOT / "runs/PORTABLE_BUNDLE_REPORT_V2.md"
CHECKPOINT = Path.home() / ".cache/huggingface/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-regressor-v2-20260212.ckpt"
EXPECTED_CHECKPOINT_SHA = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
TASKS = ("chla_surface", "secchi", "tn_surface", "tp_middle")
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
EXPECTED_JITTER = 0.0014983664


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_ids(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def hash_array(x) -> str:
    a = np.ascontiguousarray(np.asarray(x))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    for p in src.rglob("*"):
        if not p.is_file() or any(x in {"__pycache__", ".pytest_cache"} for x in p.parts):
            continue
        if p.suffix in {".pyc", ".pyo", ".tmp"} or p.name.endswith(("-wal", "-shm")) or p.name == "PAUSE_AFTER_CURRENT_FOLD.flag":
            continue
        copy_file(p, dst / p.relative_to(src))


def verify_existing_states() -> dict:
    audit_p = ROOT / "runs/tabicl_v2_final_explanation_complete_v1/fold_1_completion_audit.json"
    decision_p = ROOT / "runs/tabicl_explanation_numerical_stability_v2/numerical_stability_decision.json"
    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    decision = json.loads(decision_p.read_text(encoding="utf-8"))
    if audit.get("final_status") != "FOLD1_NOT_COMPLETE_VALIDATION_FAILURE":
        raise RuntimeError("fold1历史验收状态发生变化；为防止改写冻结结论，中止打包")
    if decision.get("final_conclusion") != "NUMERICAL_JITTER_NONNEGLIGIBLE_FOR_SMALL_EFFECTS":
        raise RuntimeError("V2数值稳定性结论与冻结记录不一致")
    if abs(float(audit["ale1_checks"][0]["max_local_ale_delta_range"]) - EXPECTED_JITTER) > 1e-9:
        raise RuntimeError("历史观测噪声参考不匹配")
    if not audit.get("fold2_rep4_registry_recorded"):
        raise RuntimeError("fold2 rep4恢复记录缺失")
    dbp = ROOT / "runs/tabicl_v2_final_explanation_complete_v1/checkpoints/local_effects.sqlite"
    if not dbp.is_file() or dbp.with_name(dbp.name + "-wal").exists():
        raise RuntimeError("源SQLite缺失或存在未checkpoint WAL；拒绝打包")
    con = sqlite3.connect(f"file:{dbp.as_posix()}?mode=ro", uri=True)
    db_integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    db_rows = con.execute("SELECT COUNT(*) FROM ale1_local").fetchone()[0]
    con.close()
    if db_integrity != "ok" or int(db_rows) != int(audit.get("checkpoint_db_rows", -1)):
        raise RuntimeError(f"源SQLite校验未通过：integrity={db_integrity}, rows={db_rows}")
    audit["source_sqlite_integrity"] = db_integrity
    audit["source_sqlite_rows_verified"] = int(db_rows)
    return audit


def copy_assets() -> None:
    if BUNDLE.exists() or ZIP_PATH.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖：{BUNDLE} 或 {ZIP_PATH}")
    BUNDLE.mkdir(parents=True)
    (WORKSPACE / "runs").mkdir(parents=True)

    # 本轮实验本身、数值稳定性证据和历史已完成解释结果。
    for rel in (
        "runs/tabicl_v2_final_explanation_complete_v1",
        "runs/tabicl_explanation_numerical_stability_v2",
        "runs/tabicl_v2_explanation_stability_v1",
        "runs/tabicl_v2_explainability_v3",
        "runs/tabicl_v2_repro_ale_repair_v1",
    ):
        copy_tree(ROOT / rel, WORKSPACE / rel)

    # 正式预测输入仅复制解释恢复脚本直接使用的模型表、split及TabICLv2冻结预测。
    copy_file(ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv",
              WORKSPACE / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv")
    copy_file(ROOT / "runs/tabiclv2_pilot/01_strict_feature_manifest.csv",
              WORKSPACE / "runs/tabiclv2_pilot/01_strict_feature_manifest.csv")
    copy_file(ROOT / "runs/multisource_ablation_v1/00_feature_source_taxonomy.csv",
              WORKSPACE / "runs/multisource_ablation_v1/00_feature_source_taxonomy.csv")
    split_files = (
        "runs/chla_family_v2/03_chla_surface_manifest.csv",
        "runs/tn_family_v2/03_tn_surface_manifest.csv",
        "runs/tp_family_v2/04_tp_middle_manifest.csv",
        "runs/modern_baseline_gate_v1/01_secchi_manifest.csv",
    )
    for rel in split_files:
        copy_file(ROOT / rel, WORKSPACE / rel)
    pred_files = []
    for task_prefix, folder in (
        ("chla_surface_tabiclv2_", "runs/chla_family_v2/predictions"),
        ("tn_surface_tabiclv2_", "runs/tn_family_v2/predictions"),
        ("tp_middle_tabiclv2_", "runs/tp_family_v2/predictions"),
    ):
        for fold in FOLDS:
            name = f"{task_prefix}{fold}.csv"
            pred_files.append(f"{folder}/{name}")
    pred_files.append("runs/modern_baseline_gate_v1/predictions/secchi_tabiclv2_recovered.csv")
    for rel in pred_files:
        copy_file(ROOT / rel, WORKSPACE / rel)

    # 只带入已有XGBoost解释对照表，不复制XGBoost模型或无关产物。
    xgb_cmp = ROOT / "runs/xgb_shap_ale_v2/04_integrated/04_shap_ale_ablation_comparison.csv"
    if xgb_cmp.is_file():
        copy_file(xgb_cmp, WORKSPACE / "runs/xgb_shap_ale_v2/04_integrated/04_shap_ale_ablation_comparison.csv")

    # TabICL源码补丁及预训练权重原样归档；适配器只将权重定位到包内文件。
    copy_tree(ROOT / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch",
              WORKSPACE / "runs/tabicl_v2_deterministic_repair_v1/isolated_patch")
    adapter_src = ROOT / "models/tabiclv2_adapter.py"
    adapter_text = adapter_src.read_text(encoding="utf-8")
    old = '        defaults = {\n            "device": "cuda",'
    new = ('        defaults = {\n'
           '            "device": "cuda",')
    if old not in adapter_text:
        raise RuntimeError("模型adapter结构与已审计版本不符；拒绝猜测修改")
    adapter_text = adapter_text.replace(old, new, 1)
    # The checkpoint path is injected by the portable launcher. This changes location only, not weights or inference settings.
    anchor = '        defaults.update(kwargs)\n        self.model = TabICLRegressor(**defaults)'
    replacement = ('        checkpoint = os.environ.get("TABICL_PRETRAINED_CHECKPOINT")\n'
                   '        if checkpoint:\n'
                   '            defaults["model_path"] = checkpoint\n'
                   '            defaults["allow_auto_download"] = False\n'
                   '        defaults.update(kwargs)\n'
                   '        self.model = TabICLRegressor(**defaults)')
    if anchor not in adapter_text:
        raise RuntimeError("无法在adapter中定位参数注入点")
    adapter_text = adapter_text.replace("from typing import Any\n", "from typing import Any\nimport os\n", 1)
    adapter_text = adapter_text.replace(anchor, replacement, 1)
    adst = WORKSPACE / "models/tabiclv2_adapter.py"
    adst.parent.mkdir(parents=True, exist_ok=True)
    adst.write_text(adapter_text, encoding="utf-8")
    copy_file(CHECKPOINT, BUNDLE / "model/tabicl-regressor-v2-20260212.ckpt")
    if sha_file(BUNDLE / "model/tabicl-regressor-v2-20260212.ckpt") != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("包内TabICLv2 checkpoint SHA256不匹配")

    # 运行代码保留workspace相对根结构，ROOT仍由Path(__file__).resolve()解析。
    source_scripts = ROOT / "脚本/05_诊断分析"
    for name in (
        "79_run_all_tabicl_explanation_experiments.py",
        "81_validate_tabicl_explanation_results.py",
        "66_run_tabiclv2_explainability_v2.py",
    ):
        p = source_scripts / name
        if p.is_file():
            copy_file(p, WORKSPACE / "脚本/05_诊断分析" / name)
    portable_runner = WORKSPACE / "脚本/05_诊断分析/79_run_all_tabicl_explanation_experiments.py"
    text = portable_runner.read_text(encoding="utf-8")
    predictor_anchor = "    predictor = Predictor()\n    env = {"
    if predictor_anchor not in text:
        raise RuntimeError("79脚本Predictor结构发生变化，无法安全登记真实调用次数")
    text = text.replace(predictor_anchor, "    predictor = Predictor()\n    predictor.calls = calls\n    env = {", 1)
    context_anchor = "def make_context(task: str, fold: str):\n    import torch\n    seed_all(torch)"
    context_replacement = ("def make_context(task: str, fold: str):\n"
                          "    import torch\n"
                          "    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '4'))))\n"
                          "    seed_all(torch)")
    if context_anchor not in text:
        raise RuntimeError("79脚本上下文初始化结构发生变化；不能安全应用CPU线程调度设置")
    text = text.replace(context_anchor, context_replacement, 1)
    ale_repeat_anchor = "        if probe_bin is not None:\n            probe_i=int(np.flatnonzero(eligible & (bins==probe_bin))[0]);probe_oid=query_ids[probe_i]"
    ale_repeat_replacement = (
        "        repeat_file=OUT/\"ale_1d_repeatability.csv\"\n"
        "        already_repeat=False\n"
        "        if repeat_file.is_file():\n"
        "            prior_repeat=pd.read_csv(repeat_file)\n"
        "            already_repeat=bool(((prior_repeat.task==task)&(prior_repeat.fold==fold)&(prior_repeat.feature==feature)).any())\n"
        "        if probe_bin is not None and not already_repeat:\n"
        "            probe_i=int(np.flatnonzero(eligible & (bins==probe_bin))[0]);probe_oid=query_ids[probe_i]"
    )
    if ale_repeat_anchor not in text:
        raise RuntimeError("79脚本ALE重复探针结构发生变化，不能确保断点不重复计算")
    text = text.replace(ale_repeat_anchor, ale_repeat_replacement, 1)
    portable_runner.write_text(text, encoding="utf-8")


def build_golden_queries() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """仅做已冻结输入的CPU重建与hash核对；不加载模型、不执行predict。"""
    base_path = WORKSPACE / "脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py"
    spec = importlib.util.spec_from_file_location("portable_base_data_only", base_path)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    assert spec.loader is not None
    spec.loader.exec_module(base)
    probes = pd.read_csv(WORKSPACE / "runs/tabicl_explanation_numerical_stability_v2/probe_manifest.csv")
    repeat = pd.read_csv(WORKSPACE / "runs/tabicl_v2_final_explanation_complete_v1/ale_1d_repeatability.csv")
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for r in probes.itertuples(index=False):
        features = base.load_features()
        train, test = base.load_fold(str(r.task), str(r.fold), features)
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        xtr = imp.fit_transform(train[features]).astype(np.float32)
        xte = imp.transform(test[features]).astype(np.float32)
        match = np.flatnonzero(test.obs_id.astype(str).to_numpy() == str(r.obs_id))
        if len(match) != 1:
            raise RuntimeError(f"golden obs_id不是唯一或不存在：{r.probe_id}")
        x = xte[int(match[0])].copy()
        if str(r.probe_id) == "chla_fold1_failed_ssrd_lower":
            rr = repeat.loc[(repeat.task == r.task) & (repeat.fold == r.fold) & (repeat.feature == "ssrd_mean_14d")]
            if rr.empty:
                raise RuntimeError("无法从已存ALE repeatability记录恢复失败探针下界")
            x[features.index("ssrd_mean_14d")] = float(rr.iloc[0].x_lower)
        key = str(r.probe_id)
        actual_hash = hash_array(x)
        if actual_hash != str(r.query_input_sha256):
            raise RuntimeError(f"golden输入hash不匹配 {key}: expected={r.query_input_sha256}, actual={actual_hash}")
        if sha_ids(features) != str(r.feature_order_hash):
            raise RuntimeError(f"77特征顺序hash不匹配：{key}")
        imp_hash = hash_array(imp.statistics_)
        if imp_hash != str(r.imputer_stats_hash):
            raise RuntimeError(f"imputer统计hash不匹配：{key}")
        rows.append({**r._asdict(), "query_hash_verified": actual_hash, "imputer_hash_verified": imp_hash,
                     "train_obs_order_hash": sha_ids(train.obs_id.astype(str)),
                     "test_obs_order_hash": sha_ids(test.obs_id.astype(str)),
                     "query_shape_verified": "1x77", "query_dtype_verified": str(x.dtype),
                     "context_rebuilt_on_new_host": True,
                     "relevant_ale_curve_range": relevant_ale_range(str(r.task), str(r.fold))})
        arrays[key] = x
    out = BUNDLE / "inputs/golden_queries.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(BUNDLE / "inputs/golden_query_manifest.csv", index=False, encoding="utf-8-sig")
    return manifest, arrays


def relevant_ale_range(task: str, fold: str) -> float:
    """返回已有解释曲线尺度作跨设备对照；不据此重算曲线。"""
    for rel in (
        "runs/tabicl_v2_final_explanation_complete_v1/05_ale_1d_results.csv",
        "runs/tabicl_v2_explainability_v3/ale_1d/ale_1d_by_fold.csv",
    ):
        p = WORKSPACE / rel
        if p.is_file():
            d = pd.read_csv(p)
            d = d.loc[(d.task == task) & (d.fold == fold)]
            if not d.empty and "ale" in d:
                ranges = d.groupby("feature").ale.agg(lambda s: float(s.max() - s.min()))
                if len(ranges):
                    return float(ranges.max())
    return float("nan")


def build_atomic_manifest() -> pd.DataFrame:
    run = WORKSPACE / "runs/tabicl_v2_final_explanation_complete_v1"
    v2 = WORKSPACE / "runs/tabicl_explanation_numerical_stability_v2"
    repair = WORKSPACE / "runs/tabicl_v2_repro_ale_repair_v1"
    shap_manifest = pd.read_csv(run / "shap_sample_manifest.csv")
    ale1_audit = pd.read_csv(repair / "06_ale_1d_rebinning_comparison.csv")
    ale2_audit = pd.read_csv(repair / "07_ale_2d_grid_sensitivity.csv")
    old = pd.read_csv(run / "13_final_explanation_master_registry.csv")
    rows = []
    # Frozen 3-repeat P1 protocol; fold1 existing probes and fold2 rep4 remain completed.
    for task in TASKS:
        for fold in FOLDS:
            if (task, fold) == ("chla_surface", "fold_1"):
                for rep in (1, 2, 3):
                    p = run / "pointwise" / task / fold / f"replicate_{rep}.csv"
                    status = "COMPUTED" if p.is_file() else "PENDING"
                    rows.append({"unit_type":"P1_REP","task":task,"fold":fold,"unit_id":f"P1:{task}:{fold}:rep{rep}",
                                 "status":status,"output_path":str(p.relative_to(BUNDLE)),"reason":"原有fold1点式探针；不重跑"})
                continue
            for rep in (4, 5, 6):
                p = run / "pointwise" / task / fold / f"replicate_{rep}.csv"
                status = "OUTPUT_RECOVERED" if (task, fold, rep) == ("chla_surface", "fold_2", 4) and p.is_file() else ("COMPUTED" if p.is_file() else "PENDING")
                rows.append({"unit_type":"P1_REP","task":task,"fold":fold,"unit_id":f"P1:{task}:{fold}:rep{rep}",
                             "status":status,"output_path":str(p.relative_to(BUNDLE)),"reason":"rep4恢复记录单独保留" if rep == 4 and status == "OUTPUT_RECOVERED" else ""})

    # SHAP原子单位按obs_id登记；仅文件已存在且77特征与加性检查通过才算完成。
    for r in shap_manifest.itertuples(index=False):
        p = run / "shap_values" / str(r.task) / str(r.fold) / f"{r.obs_id}_seed42.csv.gz"
        status = "PENDING"
        reason = ""
        if p.is_file():
            d = pd.read_csv(p)
            if len(d) == 77 and d.feature_order.tolist() == list(range(1,78)) and float(d.additivity_abs_error.max()) <= 0.02:
                status, reason = "COMPUTED_PASS", "已有文件、77维顺序及加性误差核验通过"
        rows.append({"unit_type":"SHAP_SAMPLE","task":r.task,"fold":r.fold,"unit_id":f"SHAP:{r.task}:{r.fold}:{r.obs_id}",
                     "status":status,"output_path":str(p.relative_to(BUNDLE)),"reason":reason})

    support = pd.read_csv(run / "06_ale_1d_support_and_stability.csv") if (run / "06_ale_1d_support_and_stability.csv").is_file() else pd.DataFrame()
    repeat = pd.read_csv(run / "ale_1d_repeatability.csv") if (run / "ale_1d_repeatability.csv").is_file() else pd.DataFrame()
    existing = ale1_audit.drop_duplicates(["task","fold","feature"])
    for r in existing.itertuples(index=False):
        key = (str(r.task), str(r.fold), str(r.feature))
        s = support.loc[(support.task == key[0]) & (support.fold == key[1]) & (support.feature == key[2])] if len(support) else pd.DataFrame()
        rep = repeat.loc[(repeat.task == key[0]) & (repeat.fold == key[1]) & (repeat.feature == key[2])] if len(repeat) else pd.DataFrame()
        if not s.empty and str(s.iloc[-1].status) == "INSUFFICIENT_SUPPORT":
            status = "SKIPPED_INSUFFICIENT_SUPPORT"
        elif not s.empty and str(s.iloc[-1].status).startswith("COMPUTED") and not rep.empty and float(rep.iloc[-1].max_local_ale_delta_range) > 1e-6:
            status = "COMPUTED_WITH_LIMITATION"
        elif not s.empty and str(s.iloc[-1].status).startswith("COMPUTED") and not rep.empty:
            status = "COMPUTED_PASS"
        else:
            status = "PENDING"
        rows.append({"unit_type":"ALE1_FEATURE","task":key[0],"fold":key[1],"unit_id":f"ALE1:{key[0]}:{key[1]}:{key[2]}",
                     "status":status,"output_path":str((run / "05_ale_1d_results.csv").relative_to(BUNDLE)),
                     "reason":"保留原始数值敏感性状态；不因已计算而改写为PASS" if status == "COMPUTED_WITH_LIMITATION" else ""})

    # 记录没有覆盖审计行的预设任务折变量，以明确标注未审计为支持不足，而不是虚构待计算输入。
    all_features = {t: sorted(ale1_audit.loc[ale1_audit.task == t, "feature"].astype(str).unique()) for t in TASKS}
    for task in TASKS:
        if not all_features[task]:
            continue
        for fold in FOLDS:
            for feature in all_features[task]:
                if ((ale1_audit.task == task) & (ale1_audit.fold == fold) & (ale1_audit.feature == feature)).any():
                    continue
                rows.append({"unit_type":"ALE1_FEATURE","task":task,"fold":fold,"unit_id":f"ALE1:{task}:{fold}:{feature}",
                             "status":"NOT_IN_AUDIT","output_path":"","reason":"无预注册重分箱审计行；不从未审计范围自动生成计算"})

    a2out = run / "08_ale_2d_support_and_stability.csv"
    a2done = pd.read_csv(a2out) if a2out.is_file() and a2out.stat().st_size else pd.DataFrame()
    a2unique = ale2_audit.drop_duplicates(["task","fold","feature_1","feature_2","grid_actual"])
    for r in a2unique.itertuples(index=False):
        status = "PENDING"
        if not a2done.empty:
            got = a2done.loc[(a2done.task == r.task) & (a2done.fold == r.fold) & (a2done.feature_1 == r.feature_1) &
                             (a2done.feature_2 == r.feature_2) & (a2done.grid.astype(str) == str(r.grid_actual))]
            if not got.empty:
                status = "COMPUTED_OR_TERMINAL"
        rows.append({"unit_type":"ALE2_PAIR_GRID","task":r.task,"fold":r.fold,
                     "unit_id":f"ALE2:{r.task}:{r.fold}:{r.feature_1}:{r.feature_2}:{r.grid_actual}",
                     "status":status,"output_path":str(a2out.relative_to(BUNDLE)),"reason":"只对既有预审组合处理"})

    for task in TASKS:
        for fold in FOLDS:
            pred = run / f"02_original_vs_pointwise_full_{task}_{fold}.csv"
            status = "COMPUTED" if pred.is_file() else "PENDING_DEPENDENCY"
            rows.append({"unit_type":"FULL_FOLD_COMPARE","task":task,"fold":fold,"unit_id":f"FULL:{task}:{fold}",
                         "status":status,"output_path":str(pred.relative_to(BUNDLE)),"reason":"仅在该折P1验证STABLE后执行"})
    for task in TASKS:
        fold = "fold_1"
        sample = shap_manifest.loc[(shap_manifest.task == task) & (shap_manifest.fold == fold)]
        if sample.empty:
            continue
        obs = str(sample.iloc[0].obs_id)
        p = run / "shap_sensitivity" / task / fold / f"{obs}_seed43_eval309.csv.gz"
        rows.append({"unit_type":"SHAP_SENSITIVITY","task":task,"fold":fold,"unit_id":f"SHAP_SENS:{task}:{fold}:{obs}",
                     "status":"COMPUTED" if p.is_file() else "PENDING","output_path":str(p.relative_to(BUNDLE)),"reason":"既定预算/种子敏感性分析"})
    return pd.DataFrame(rows)


def write_runtime_files(golden: pd.DataFrame, atomic: pd.DataFrame, audit: dict) -> None:
    bundle = BUNDLE
    # The lock describes the verified runtime used for the earlier calculations; the old venv is deliberately excluded.
    project_python = ROOT / "runs/tabiclv2_pilot/环境/.venv/Scripts/python.exe"
    if not project_python.is_file():
        raise FileNotFoundError("已验证项目Python环境不存在，无法生成真实pip freeze；不复制venv")
    freeze = subprocess.run([str(project_python), "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    (bundle / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    core_versions = subprocess.run([str(project_python), "-c",
        "import sys,numpy,pandas,scipy,sklearn,shap,torch; import importlib.metadata as m; "
        "print('|'.join([sys.version.split()[0],numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__,shap.__version__,torch.__version__,str(torch.version.cuda),m.version('tabicl')]))"],
        check=True, capture_output=True, text=True).stdout.strip().split("|")
    if len(core_versions) != 9:
        raise RuntimeError(f"无法解析当前真实环境版本：{core_versions}")
    v = dict(zip(("python","numpy","pandas","scipy","scikit_learn","shap","torch","cuda_runtime","tabicl"),core_versions))
    v["psutil"] = subprocess.run([str(project_python), "-c", "import psutil; print(psutil.__version__)"],
                                 check=True, capture_output=True, text=True).stdout.strip()
    requirements = """--extra-index-url https://download.pytorch.org/whl/cu124
numpy=={numpy}
pandas=={pandas}
scipy=={scipy}
scikit-learn=={scikit_learn}
shap=={shap}
torch=={torch}
tabicl=={tabicl}
psutil=={psutil}
""".format(**v)
    (bundle / "requirements-lock.txt").write_text(requirements, encoding="utf-8")
    source_driver = None
    try:
        gpu_q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10)
        if gpu_q.returncode == 0:
            source_driver = gpu_q.stdout.strip()
    except Exception:
        pass
    env_manifest = {
        "verified_source_environment": v,
        "python_executable_kind": "Python 3.11 x64; the source .venv is not copied",
        "torch_cuda_runtime": v["cuda_runtime"],
        "cudnn_version": "9.1.0 (reported by source environment)" if v["torch"].startswith("2.6.0+cu124") else "recorded in source fold manifest",
        "source_gpu": "NVIDIA GeForce RTX 2070 with Max-Q Design (from V1/V2 run manifests)",
        "source_gpu_driver_query": source_driver or "not available from current shell",
        "target_gpu_expected_by_user": "NVIDIA GeForce RTX 4060 8GB; must be detected and compatibility-gated on destination",
        "source_os": platform.platform(),
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "portable_adapter_checkpoint_path": "../model/tabicl-regressor-v2-20260212.ckpt via TABICL_PRETRAINED_CHECKPOINT",
        "historical_python_executable_paths_are_provenance_only": True,
    }
    (bundle / "environment_manifest.json").write_text(json.dumps(env_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "bundle_root": ".", "workspace_root": "workspace", "input_dir": "workspace",
        "model_dir": "model", "checkpoint_dir": "workspace/runs/tabicl_v2_final_explanation_complete_v1/checkpoints",
        "output_dir": "workspace/runs/tabicl_v2_final_explanation_complete_v1", "log_dir": "workspace/runs/tabicl_v2_final_explanation_complete_v1/logs",
        "RAM_CACHE_MODE": "ON", "max_process_ram_gb": 24, "gpu_workers": 1,
        "query_protocol": "P1_SINGLE_QUERY", "query_shape": [1,77], "dtype": "float32",
        "reference_jitter_abs": EXPECTED_JITTER, "jitter_reference_kind": "observed numerical jitter reference, not a theoretical bound",
        "cross_device_status_required_before_resume": "CROSS_DEVICE_NUMERICALLY_COMPATIBLE",
        "frozen_v1_fold1_status": audit["final_status"],
        "v1_stability_status": "INFERENCE_NONDETERMINISM_CONFIRMED",
        "v2_stability_status": "NUMERICAL_JITTER_NONNEGLIGIBLE_FOR_SMALL_EFFECTS",
        "fold2_rep4_status": "OUTPUT_RECOVERED_AFTER_SCHEDULER_PAUSE",
        "portable_paths_are_relative": True,
    }
    (bundle / "portable_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (bundle / "checkpoints").mkdir(parents=True, exist_ok=True)
    (bundle / "logs").mkdir(parents=True, exist_ok=True)
    (bundle / "completed_results").mkdir(parents=True, exist_ok=True)
    atomic.to_csv(bundle / "checkpoints/atomic_unit_manifest.csv", index=False, encoding="utf-8-sig")
    # Current successful outputs are already copied under workspace/runs; add a concise directory index, not duplicates.
    (bundle / "completed_results/README.txt").write_text(
        "规范结果位于workspace/runs/tabicl_v2_final_explanation_complete_v1；本目录不复制第二份数据，避免检查点分叉。\n",
        encoding="utf-8")
    (bundle / "checkpoints/README.txt").write_text(
        "权威SQLite与原run registry位于workspace/runs/tabicl_v2_final_explanation_complete_v1/checkpoints及13_final_explanation_master_registry.csv。\n",
        encoding="utf-8")
    (bundle / "logs/README.txt").write_text("新电脑续跑日志将落入workspace/runs/.../logs；旧RTX2070日志保存在同目录。\n", encoding="utf-8")
    status = {
        "current_task": None, "current_fold": None, "current_method": None, "current_atomic_unit": None,
        "completed_units": int(atomic.status.isin(["COMPUTED","COMPUTED_PASS","COMPUTED_WITH_LIMITATION","OUTPUT_RECOVERED","COMPUTED_OR_TERMINAL"]).sum()),
        "pending_units": int(atomic.status.isin(["PENDING","PENDING_DEPENDENCY"]).sum()),
        "failed_units": int(atomic.status.astype(str).str.startswith("FAILED").sum()),
        "skipped_units": int(atomic.status.astype(str).str.startswith(("SKIPPED","NOT_IN_AUDIT")).sum()),
        "elapsed_seconds": 0.0, "recent_unit_seconds": [], "moving_average_seconds": None,
        "estimated_remaining_seconds": None, "last_checkpoint_time": None, "gpu_name": "to be detected on destination",
        "gpu_utilization_pct": None, "peak_vram_mb": None, "process_ram_mb": None,
        "source_reference_jitter_abs": EXPECTED_JITTER,
        "note": "ETA will be populated only from actual destination-machine completed units.",
    }
    (bundle / "progress_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme() -> None:
    text = """# TabICLv2解释实验可迁移运行包 V2

## 新电脑最短操作

1. 将ZIP复制到新电脑并解压到英文短路径，例如 `D:\\tabicl_explain\\`。
2. PowerShell进入该目录。
3. 硬件检查：

   `powershell -ExecutionPolicy Bypass -File .\\scripts\\00_check_hardware.ps1`

4. 创建独立环境：

   `powershell -ExecutionPolicy Bypass -File .\\scripts\\01_setup_environment.ps1`

5. 先运行跨设备兼容检查：

   `powershell -ExecutionPolicy Bypass -File .\\scripts\\02_validate_cross_device.ps1`

6. 只有输出 `CROSS_DEVICE_NUMERICALLY_COMPATIBLE` 才续跑：

   `powershell -ExecutionPolicy Bypass -File .\\scripts\\03_resume_remaining.ps1`

7. 查看进度：

   `powershell -ExecutionPolicy Bypass -File .\\scripts\\04_status.ps1`

## 冻结结论与协议

- 状态为 `PORTABLE_BUNDLE_CREATED_WITH_KNOWN_NUMERICAL_NONDETERMINISM`；fold1仍为 `{audit_status}`，不能写成PASS。
- 原严格重复阈值仍为 `1e-6`；V1失败和V2结论均保留。`{jitter}` 是目标变量原单位下的历史观测噪声参考，不是理论上界。
- 仅使用P1单查询，每次模型输入必须为 `(1,77)`、float32；不batch、不换模型/特征/split/context/imputer/数值精度。
- RAM缓存用于每个fold-method worker内部复用已加载的数组、imputer输出和固定上下文；模型不按单个query重新加载。GPU只允许一个worker串行。
- 当前调度器按方法启动独立worker。跨方法共用同一Python进程/模型状态没有在原RTX2070上完成等价性验证，因此不启用该优化。
- 小于或接近观测噪声尺度的ALE局部变化标记为 `NUMERICALLY_SENSITIVE_SMALL_EFFECT`；小SHAP贡献标记为 `SMALL_EFFECT_INTERPRET_WITH_CAUTION`。不得删除此类结果。
- 所有冻结输入、V1/V2运行记录和完成结果均位于 `workspace/`。原历史环境清单中出现的绝对路径仅作为不可变来源记录，不是运行依赖；新脚本与配置使用包内相对路径。
- 原运行没有找到可直接复制的20折序列化imputer/context对象。包内包括完整输入、训练规则、fold1黄金hash及预测参考；其余fold的imputer/context由原始冻结输入和原脚本在新机重建。
- 当前包不声称RTX4060已通过兼容性。只有目标机运行第二步预检后才可决定是否继续。绝不输出 `BITWISE_EQUIVALENT`。

## 状态计数

初始checkpoint atomic manifest：{atomic_count}项；已计算/完成 {completed_count}，待处理或依赖待定 {pending_count}，支持不足/未审计跳过 {skipped_count}。fold2 rep4保留为独立 `OUTPUT_RECOVERED` 单元。
""".format(audit_status=json.loads((BUNDLE/"portable_config.json").read_text(encoding="utf-8"))["frozen_v1_fold1_status"],
           jitter=EXPECTED_JITTER,
           atomic_count=len(pd.read_csv(BUNDLE/"checkpoints/atomic_unit_manifest.csv")),
           completed_count=int(pd.read_csv(BUNDLE/"checkpoints/atomic_unit_manifest.csv").status.isin(["COMPUTED","COMPUTED_PASS","COMPUTED_WITH_LIMITATION","OUTPUT_RECOVERED","COMPUTED_OR_TERMINAL"]).sum()),
           pending_count=int(pd.read_csv(BUNDLE/"checkpoints/atomic_unit_manifest.csv").status.isin(["PENDING","PENDING_DEPENDENCY"]).sum()),
           skipped_count=int(pd.read_csv(BUNDLE/"checkpoints/atomic_unit_manifest.csv").status.astype(str).str.startswith(("SKIPPED","NOT_IN_AUDIT")).sum()))
    (BUNDLE / "README_FIRST.md").write_text(text, encoding="utf-8")


def make_file_manifest() -> pd.DataFrame:
    rows = []
    for p in sorted(BUNDLE.rglob("*")):
        if not p.is_file() or p.name == "file_manifest_sha256.csv":
            continue
        rel = p.relative_to(BUNDLE).as_posix()
        mutable = (rel.startswith("workspace/runs/tabicl_v2_final_explanation_complete_v1/") or
                   rel.startswith("logs/") or rel in {"progress_status.json", "checkpoints/atomic_unit_manifest.csv", "checkpoints/portable_process_retry.json"})
        rows.append({"relative_path":rel,"size_bytes":p.stat().st_size,"sha256":sha_file(p),"mutable":mutable})
    d = pd.DataFrame(rows)
    d.to_csv(BUNDLE / "file_manifest_sha256.csv", index=False, encoding="utf-8-sig")
    return d


def self_check(golden: pd.DataFrame, manifest: pd.DataFrame, audit: dict) -> dict:
    failures, warnings = [], []
    for p in BUNDLE.rglob("*.sqlite"):
        try:
            con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
            integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok": failures.append(f"sqlite integrity: {p.relative_to(BUNDLE)}={integrity}")
            if con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal" and p.with_name(p.name+"-wal").exists():
                failures.append(f"uncheckpointed WAL: {p.relative_to(BUNDLE)}")
            con.close()
        except Exception as e:
            failures.append(f"sqlite read error {p.relative_to(BUNDLE)}: {e!r}")
    checkpoint = BUNDLE / "model/tabicl-regressor-v2-20260212.ckpt"
    if not checkpoint.is_file() or sha_file(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        failures.append("checkpoint missing or hash mismatch")
    checkpoint_parse_ok = False
    try:
        import torch
        obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checkpoint_parse_ok = isinstance(obj, dict) and "config" in obj and "state_dict" in obj
        if not checkpoint_parse_ok:
            failures.append("checkpoint config/state_dict parsing failed")
    except Exception as e:
        failures.append(f"checkpoint parse failed: {e!r}")
    python_files = list(BUNDLE.rglob("*.py"))
    for p in python_files:
        try:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except SyntaxError as e:
            failures.append(f"Python syntax error {p.relative_to(BUNDLE)}: {e}")
    run = WORKSPACE / "runs/tabicl_v2_final_explanation_complete_v1"
    r = pd.read_csv(run / "13_final_explanation_master_registry.csv")
    rep4 = r.loc[r.unit.eq("A:chla_surface:fold_2:rep4")]
    rep4_out = run / "pointwise/chla_surface/fold_2/replicate_4.csv"
    if rep4.empty or rep4.iloc[-1].status != "OUTPUT_RECOVERED_AFTER_SCHEDULER_PAUSE" or not rep4_out.is_file():
        failures.append("fold2 rep4 recovery record/output missing")
    audit_copy = json.loads((run / "fold_1_completion_audit.json").read_text(encoding="utf-8"))
    decision_copy = json.loads((WORKSPACE / "runs/tabicl_explanation_numerical_stability_v2/numerical_stability_decision.json").read_text(encoding="utf-8"))
    if audit_copy.get("final_status") != "FOLD1_NOT_COMPLETE_VALIDATION_FAILURE": failures.append("V1 fold1 status rewritten")
    if decision_copy.get("final_conclusion") != "NUMERICAL_JITTER_NONNEGLIGIBLE_FOR_SMALL_EFFECTS": failures.append("V2 conclusion mismatch")
    if int(len(golden)) < 5 or not golden.query_hash_verified.eq(golden.query_input_sha256).all(): failures.append("golden query set/hash invalid")
    atomic = pd.read_csv(BUNDLE / "checkpoints/atomic_unit_manifest.csv")
    if atomic.unit_id.duplicated().any(): failures.append("duplicate atomic units")
    if not ((atomic.unit_id == "P1:chla_surface:fold_2:rep4") & (atomic.status == "OUTPUT_RECOVERED")).any(): failures.append("rep4 not marked recovered in queue")
    # Absolute path scan is limited to executable/configuration files; immutable historical provenance may mention old locations.
    for p in list(BUNDLE.rglob("*.py")) + list(BUNDLE.rglob("*.ps1")) + list(BUNDLE.rglob("*.json")):
        if p.name in {"file_manifest_sha256.csv"}:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "E:\\研究生\\codex\\水质预测" in text or "E:/研究生/codex/水质预测" in text:
            failures.append(f"hard-coded project absolute path in executable/config: {p.relative_to(BUNDLE)}")
    # Ensure relative bundle paths and required portable tools are in place.
    required = ["README_FIRST.md","portable_config.json","requirements-lock.txt","pip_freeze.txt","environment_manifest.json",
        "file_manifest_sha256.csv","scripts/00_check_hardware.ps1","scripts/01_setup_environment.ps1",
        "scripts/02_validate_cross_device.ps1","scripts/03_resume_remaining.ps1","scripts/04_status.ps1",
        "scripts/portable_runner.py","scripts/validate_cross_device.py","scripts/final_validate.py"]
    for rel in required:
        if not (BUNDLE / rel).is_file(): failures.append(f"required file missing: {rel}")
    check = {
        "status": "PASS" if not failures else "FAIL", "failures": failures, "warnings": warnings,
        "bundle_created_status": "PORTABLE_BUNDLE_CREATED_WITH_KNOWN_NUMERICAL_NONDETERMINISM",
        "fold1_status_preserved": audit_copy["final_status"],
        "v1_stability_status_preserved": decision_copy["v1_frozen_conclusion"],
        "v2_stability_status_preserved": decision_copy["final_conclusion"],
        "fold2_rep4_preserved": bool(not rep4.empty and rep4_out.is_file()),
        "golden_queries": int(len(golden)), "golden_hashes_verified": bool(golden.query_hash_verified.eq(golden.query_input_sha256).all()),
        "checkpoint_sha256": sha_file(checkpoint), "sqlite_files_checked": int(len(list(BUNDLE.rglob("*.sqlite")))),
        "checkpoint_parsed_config_and_state_dict": checkpoint_parse_ok,
        "python_scripts_syntax_checked": int(len(python_files)),
        "feature_order_hash": str(audit_copy.get("feature_order_hash")),
        "split_manifest_hashes": {r.relative_path: r.sha256 for r in manifest.itertuples(index=False)
                                  if r.relative_path.endswith(("03_chla_surface_manifest.csv", "03_tn_surface_manifest.csv",
                                                              "04_tp_middle_manifest.csv", "01_secchi_manifest.csv"))},
        "atomic_units": int(len(atomic)), "completed_units": int(atomic.status.isin(["COMPUTED","COMPUTED_PASS","COMPUTED_WITH_LIMITATION","OUTPUT_RECOVERED","COMPUTED_OR_TERMINAL"]).sum()),
        "pending_units": int(atomic.status.isin(["PENDING","PENDING_DEPENDENCY"]).sum()),
        "skipped_units": int(atomic.status.astype(str).str.startswith(("SKIPPED","NOT_IN_AUDIT")).sum()),
        "source_v1_strict_threshold": 1e-6, "source_v1_threshold_changed": False,
        "no_model_prediction_called_during_bundle_build": True,
    }
    (BUNDLE / "portable_bundle_self_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    return check


def write_report(check: dict) -> None:
    atomic = pd.read_csv(BUNDLE / "checkpoints/atomic_unit_manifest.csv")
    completed = int(atomic.status.isin(["COMPUTED","COMPUTED_PASS","COMPUTED_WITH_LIMITATION","OUTPUT_RECOVERED","COMPUTED_OR_TERMINAL"]).sum())
    pending = int(atomic.status.isin(["PENDING","PENDING_DEPENDENCY"]).sum())
    skipped = int(atomic.status.astype(str).str.startswith(("SKIPPED","NOT_IN_AUDIT")).sum())
    folds_pending = 19
    content = f"""# TabICLv2解释实验便携运行包 V2 报告

- 打包状态：`PORTABLE_BUNDLE_CREATED_WITH_KNOWN_NUMERICAL_NONDETERMINISM`
- Self-check：`{check['status']}`
- fold1历史状态：`FOLD1_NOT_COMPLETE_VALIDATION_FAILURE`（原1e-6阈值和失败记录保持）
- V1稳定性结论：`INFERENCE_NONDETERMINISM_CONFIRMED`
- V2数值稳定性结论：`NUMERICAL_JITTER_NONNEGLIGIBLE_FOR_SMALL_EFFECTS`
- 历史观测噪声参考：`{EXPECTED_JITTER}`（不是理论上界）
- Fold1已完成：2个SHAP样本、2条ALE曲线；第一条ALE重复性失败未改写，第二条保留PASS。
- Fold2 rep4：`OUTPUT_RECOVERED_AFTER_SCHEDULER_PAUSE`，模型输入/输出文件及registry已打包。
- Atomic units：总数 {len(atomic)}；已完成/已计算 {completed}；待处理或依赖待定 {pending}；支持不足/未审计终止 {skipped}。
- 还未开始的任务折：{folds_pending}/19（fold1是历史pilot且失败状态保留；fold2 rep4单独恢复）。
- ZIP大小及SHA256由同目录 `.zip` 与 `.zip.sha256` 给出；为避免自引用，ZIP内本报告不嵌入ZIP自身hash。
- Checkpoint SHA256：`{EXPECTED_CHECKPOINT_SHA}`
- Python环境：保存了真实来源 `pip freeze` 和版本manifest；没有复制旧 `.venv`。
- 新电脑第一次必须运行跨GPU兼容检查。RTX4060尚未在本机实测，包不宣称跨GPU已验证，也不输出bitwise等价。
- 本机打包过程未调用TabICLv2预测、未计算SHAP/ALE、未启动任何剩余fold。
- RAM缓存：按fold-method worker驻留X/y、77特征矩阵、imputer输出、support索引和固定context；同一query不重复读CSV/重建模型。进程上限目标24GB。
- 长生命周期：模型/上下文在每个fold-method worker内只初始化一次，所有query仍逐条P1调用；未经数值等价检查，不跨method共用同一模型实例。
- 运行脚本和配置无项目E盘硬编码；历史冻结日志中的路径只作来源元数据，不参与新机解析。
- imputer/context序列化对象：原运行目录未发现完整20折对象文件。运行包携带全部训练输入、manifest、拟合规则和可核对hash，在目标机由原协议重建；fold1黄金查询覆盖五个既有探针。

## 新电脑启动命令

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\00_check_hardware.ps1
powershell -ExecutionPolicy Bypass -File .\\scripts\\01_setup_environment.ps1
powershell -ExecutionPolicy Bypass -File .\\scripts\\02_validate_cross_device.ps1
# 只有报告为CROSS_DEVICE_NUMERICALLY_COMPATIBLE时才继续：
powershell -ExecutionPolicy Bypass -File .\\scripts\\03_resume_remaining.ps1
powershell -ExecutionPolicy Bypass -File .\\scripts\\04_status.ps1
```
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    audit = verify_existing_states()
    copy_assets()
    golden, _ = build_golden_queries()
    atomic = build_atomic_manifest()
    write_runtime_files(golden, atomic, audit)
    write_readme()
    # Bundle templates (portable scripts/config templates) are staged beside this builder.
    template_dir = ROOT / "脚本/05_诊断分析/tabicl_portable_v2_templates"
    if not template_dir.is_dir():
        raise FileNotFoundError("便携脚本模板目录不存在")
    copy_tree(template_dir, BUNDLE / "scripts")
    # Re-write generated lock/config outputs after copying the templates, since templates contain no runtime artifacts.
    (BUNDLE / "scripts/portable_config.template.json").unlink(missing_ok=True)
    manifest = make_file_manifest()
    check = self_check(golden, manifest, audit)
    write_report(check)
    if check["status"] != "PASS":
        raise RuntimeError("portable_bundle_self_check失败：" + "; ".join(check["failures"]))
    # Include final report/self-check hashes; refresh manifest then self-check once more.
    make_file_manifest()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in sorted(BUNDLE.rglob("*")):
            if p.is_file():
                z.write(p, Path(BUNDLE.name) / p.relative_to(BUNDLE))
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        bad = z.testzip()
    if bad:
        raise RuntimeError(f"ZIP CRC校验失败：{bad}")
    zip_sha = sha_file(ZIP_PATH)
    sidecar = ZIP_PATH.with_suffix(ZIP_PATH.suffix + ".sha256")
    sidecar.write_text(f"{zip_sha}  {ZIP_PATH.name}\n", encoding="ascii")
    external = REPORT_PATH.read_text(encoding="utf-8") + f"\nZIP SHA256：`{zip_sha}`\nZIP大小：`{ZIP_PATH.stat().st_size}` bytes。\n"
    EXTERNAL_REPORT.write_text(external, encoding="utf-8")
    print(json.dumps({"status":"PORTABLE_BUNDLE_CREATED_WITH_KNOWN_NUMERICAL_NONDETERMINISM",
        "self_check":check["status"],"bundle":str(BUNDLE),"zip":str(ZIP_PATH),"zip_size_bytes":ZIP_PATH.stat().st_size,
        "zip_sha256":zip_sha,"completed_atomic_units":check["completed_units"],"pending_atomic_units":check["pending_units"],
        "skipped_atomic_units":check["skipped_units"],"pending_folds":19,
        "fold1_status":audit["final_status"],"v1_stability":"INFERENCE_NONDETERMINISM_CONFIRMED",
        "v2_stability":"NUMERICAL_JITTER_NONNEGLIGIBLE_FOR_SMALL_EFFECTS","fold2_rep4_preserved":check["fold2_rep4_preserved"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
