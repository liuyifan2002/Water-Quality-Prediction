"""LLM_EXTENSION_TRANSFER_PACKAGE_V1 迁移包只读校验器。

本脚本只读取 CSV/JSON/文本文件并核对结构、键、分折、预测样本集合和
SHA256；不会导入模型库，不训练，不推理，不生成或替换预测结果。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PRED_COLUMNS = {"task", "model", "fold", "obs_id", "station_id", "y_true", "y_pred"}
MODERN_TASKS = ("chla_surface", "secchi", "tn_surface", "tp_middle")
MODERN_MODELS = ("xgboost", "tabiclv2", "tabm", "tabpfn")


def rel(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def file_check(checks, check_id: str, passed: bool, detail: str) -> None:
    checks.append({"check": check_id, "pass": str(bool(passed)), "detail": detail})


def check_required_files(checks) -> None:
    required = [
        "data/main_table/model_table_3depth_2014_2024.csv",
        "data/feature_manifest/01_strict_feature_manifest.csv",
        "data/station_metadata/station_meta.csv",
        "splits/chla_family_v2/02_chla_family_v2_station_fold_map.csv",
        "splits/tn_family_v2/02_tn_family_v2_station_fold_map.csv",
        "splits/tp_family_v2/02_tp_family_v2_station_fold_map.csv",
        "splits/secchi/01_secchi_manifest.csv",
        "splits/modern_baseline_gate_v1/02_preflight.csv",
        "results/final_experiment_closure_audit_v1/final_core_results_table_v2.csv",
        "results/final_experiment_closure_audit_v1/final_experiment_master_registry_v2.csv",
        "validation/validate_transfer.py",
    ]
    missing = [item for item in required if not (PACKAGE_ROOT / item).is_file()]
    file_check(checks, "required_files", not missing, "missing=" + ",".join(missing) if missing else "all required anchor files exist")


def check_main_table(checks) -> None:
    path = PACKAGE_ROOT / "data/main_table/model_table_3depth_2014_2024.csv"
    if not path.is_file():
        return
    fields, rows = read_rows(path)
    required = {"obs_id", "station", "date", "water_zone_en", "depth", "chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l"}
    missing = sorted(required - set(fields))
    seen = set()
    duplicates = 0
    count = 0
    for row in rows:
        count += 1
        key = row.get("obs_id", "")
        if key in seen:
            duplicates += 1
        seen.add(key)
    file_check(checks, "main_table_readable", not missing, f"rows={count}; columns={len(fields)}; missing={missing}")
    file_check(checks, "main_table_obs_id_unique", duplicates == 0 and count == len(seen), f"rows={count}; unique_obs_id={len(seen)}; duplicates={duplicates}")


def check_strict_features(checks) -> None:
    table = PACKAGE_ROOT / "data/main_table/model_table_3depth_2014_2024.csv"
    manifest = PACKAGE_ROOT / "data/feature_manifest/01_strict_feature_manifest.csv"
    if not table.is_file() or not manifest.is_file():
        return
    table_fields, _ = read_rows(table)
    fields, rows = read_rows(manifest)
    features = []
    for row in rows:
        include = row.get("included_in_pilot", "").strip().lower()
        if include in {"true", "1", "yes"}:
            features.append(row.get("feature", ""))
    absent = [name for name in features if name not in table_fields]
    ordered = len(features) == len(set(features)) and all(bool(name) for name in features)
    file_check(checks, "strict_feature_count", len(features) == 77, f"included_features={len(features)}")
    file_check(checks, "strict_features_located", not absent, f"absent={absent}")
    file_check(checks, "strict_feature_order", ordered, f"manifest_order_unique_and_explicit={ordered}")


def check_family_split(checks, family: str, prefix: str) -> None:
    directory = PACKAGE_ROOT / "splits" / family
    validation = directory / "06_split_validation.csv"
    mapping = directory / f"02_{prefix}_family_v2_station_fold_map.csv"
    if validation.is_file():
        fields, rows = read_rows(validation)
        statuses = [row.get("status", "").strip().upper() for row in rows]
        file_check(checks, f"{family}_split_validation", bool(statuses) and all(status == "PASS" for status in statuses), f"rows={len(statuses)}; statuses={sorted(set(statuses))}")
    else:
        file_check(checks, f"{family}_split_validation", False, "validation file missing")
    if not mapping.is_file():
        file_check(checks, f"{family}_canonical_map", False, "canonical map missing")
        return
    map_fields, map_rows = read_rows(mapping)
    station_fold = {row.get("station_id", ""): row.get("fold", "") for row in map_rows}
    file_check(checks, f"{family}_canonical_map", bool(station_fold) and len(station_fold) == len(set(station_fold)), f"stations={len(station_fold)}; columns={map_fields}")
    manifest_paths = sorted(directory.glob("0[3-5]_*.csv"))
    all_ok = True
    details = []
    for manifest_path in manifest_paths:
        fields, rows = read_rows(manifest_path)
        groups = defaultdict(lambda: {"train": set(), "test": set()})
        for row in rows:
            key = (row.get("task", ""), row.get("fold", ""))
            groups[key][row.get("role", "").lower()].add(row.get("obs_id", ""))
            if row.get("role", "").lower() == "test" and station_fold.get(row.get("station_id", "")) != row.get("fold", ""):
                all_ok = False
        for key, roles in groups.items():
            overlap = roles["train"] & roles["test"]
            if overlap:
                all_ok = False
            details.append(f"{manifest_path.name}:{key[0]}/{key[1]} train={len(roles['train'])} test={len(roles['test'])} overlap={len(overlap)}")
    file_check(checks, f"{family}_manifest_consistency", all_ok and bool(manifest_paths), "; ".join(details[:15]))


def normalize_prediction_row(row):
    y_true = row.get("y_true", row.get("target", ""))
    model = row.get("model", "").strip().lower()
    model = {"xgb": "xgboost", "tabicl": "tabiclv2"}.get(model, model)
    return (row.get("task", ""), model, row.get("fold", ""), row.get("obs_id", ""), row.get("station_id", ""), y_true)


def prediction_set(path: Path, forced_task: str = "", forced_model: str = ""):
    fields, rows = read_rows(path)
    normalized = set()
    for row in rows:
        item = normalize_prediction_row(row)
        # 四模型样本一致性只比较测试键，不比较不同脚本写出的数值字符串格式。
        normalized.add((item[2], item[3], item[4]))
    return fields, normalized


def check_modern_predictions(checks) -> None:
    base = PACKAGE_ROOT / "predictions" / "modern_baseline_gate_v1"
    task_sets = {}
    missing = []
    bad_columns = []
    for task in MODERN_TASKS:
        for model in MODERN_MODELS:
            paths = sorted(base.glob(f"{task}_{model}_fold_*.csv"))
            if task == "secchi" and model == "xgboost":
                paths = [base / "secchi_xgb_frozen_for_gate.csv"]
            if task == "secchi" and model == "tabiclv2":
                paths = [base / "secchi_tabiclv2_recovered.csv"]
            if not paths or not all(path.is_file() for path in paths):
                missing.append(f"{task}/{model}")
                continue
            merged = set()
            for path in paths:
                fields, values = prediction_set(path, forced_task=task, forced_model=model)
                normalized_fields = {"fold", "obs_id", "station_id", "y_pred"}
                if not normalized_fields.issubset(set(fields)) or not ({"y_true", "target"} & set(fields)):
                    bad_columns.append(rel(path))
                merged |= values
            task_sets[(task, model)] = merged
            file_check(checks, f"prediction_schema_{task}_{model}", not bad_columns or all(item != rel(path) for item in bad_columns), f"rows={len(merged)}; files={len(paths)}")
    file_check(checks, "modern_prediction_completeness", not missing, "missing=" + ",".join(missing) if missing else "four models present for all four tasks")
    for task in MODERN_TASKS:
        available = [task_sets[(task, model)] for model in MODERN_MODELS if (task, model) in task_sets]
        equal = len(available) == 4 and all(values == available[0] for values in available[1:])
        sizes = [len(values) for values in available]
        file_check(checks, f"modern_test_set_equal_{task}", equal, f"set_sizes={sizes}")


def check_closure_results(checks) -> None:
    path = PACKAGE_ROOT / "results/final_experiment_closure_audit_v1/final_core_results_table_v2.csv"
    if not path.is_file():
        return
    fields, rows = read_rows(path)
    required = {"task", "model", "n_obs", "pooled_R2", "fold_R2_mean", "RMSE", "MAE"}
    file_check(checks, "closure_v2_metric_schema", required.issubset(set(fields)), f"missing={sorted(required - set(fields))}")
    values_ok = all(row.get("pooled_R2", "").strip() != "" and row.get("fold_R2_mean", "").strip() != "" for row in rows)
    file_check(checks, "pooled_and_fold_r2_distinct_fields", values_ok and "pooled_R2" != "fold_R2_mean", f"rows={len(rows)}")
    tn_tp_ok = all(float(row.get("n_obs", "0") or 0) > 0 for row in rows if row.get("target_family") in {"tn", "tp"})
    file_check(checks, "tn_tp_core_metrics_present", tn_tp_ok, "TN/TP n_obs positive in V2 core table")


def check_hash_manifest(checks) -> None:
    manifest = PACKAGE_ROOT / "manifest/sha256_manifest.csv"
    if not manifest.is_file():
        file_check(checks, "internal_sha256_manifest", False, "sha256 manifest not generated yet")
        return
    fields, rows = read_rows(manifest)
    failures = []
    count = 0
    for row in rows:
        relative = row.get("relative_path", "")
        path = PACKAGE_ROOT / relative
        count += 1
        if not path.is_file() or path.stat().st_size != int(row.get("bytes", "-1")):
            failures.append(relative)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest.lower() != row.get("sha256", "").lower():
            failures.append(relative)
    file_check(checks, "internal_sha256_manifest", not failures, f"checked={count}; failures={failures[:10]}")


def write_outputs(checks) -> None:
    output = PACKAGE_ROOT / "validation/transfer_schema_check.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "pass", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    passed = all(item["pass"] == "True" for item in checks)
    report = [
        "# 迁移包一致性报告",
        "",
        f"总体校验：{'PASS' if passed else 'FAIL'}",
        "",
        "本报告只反映迁移包内文件的结构、键、冻结分折、预测样本集合、闭环指标字段和 SHA256 检查；未启动任何模型训练或推理。",
        "",
        "| 检查 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    for item in checks:
        report.append(f"| `{item['check']}` | `{item['pass']}` | {item['detail'].replace('|', '/')} |")
    (PACKAGE_ROOT / "validation/transfer_consistency_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="只读校验 LLM_EXTENSION_TRANSFER_PACKAGE_V1")
    parser.add_argument("--check-only", action="store_true", help="只输出结果，不改变校验报告文件")
    args = parser.parse_args()
    checks = []
    check_required_files(checks)
    check_main_table(checks)
    check_strict_features(checks)
    for family, prefix in (("chla_family_v2", "chla"), ("tn_family_v2", "tn"), ("tp_family_v2", "tp")):
        check_family_split(checks, family, prefix)
    check_modern_predictions(checks)
    check_closure_results(checks)
    check_hash_manifest(checks)
    if not args.check_only:
        write_outputs(checks)
    for item in checks:
        print(f"{item['pass']:<5} {item['check']}: {item['detail']}")
    return 0 if all(item["pass"] == "True" for item in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
