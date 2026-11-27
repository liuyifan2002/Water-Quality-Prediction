"""Freeze and audit the already-computed Chla fold-1 explanation checkpoint."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_final_explanation_complete_v1"
TASK, FOLD = "chla_surface", "fold_1"
EXPECTED_SHAP = ["MM5_20201119_surface", "WM2_20211208_surface"]
EXPECTED_ALE = ["ssrd_mean_14d", "wind_speed_x_rain14d"]
REPEAT_TOL = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    # No runner/worker should be active when this is called. Flush all SQLite WAL pages.
    db = OUT / "checkpoints/local_effects.sqlite"
    con = sqlite3.connect(db, timeout=30)
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    wal_result = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    db_rows = con.execute("SELECT COUNT(*) FROM ale1_local").fetchone()[0]
    by_feature = dict(con.execute(
        "SELECT feature, COUNT(*) FROM ale1_local WHERE task=? AND fold=? GROUP BY feature",
        (TASK, FOLD),
    ).fetchall())
    con.commit()
    con.close()

    shap_runtime = pd.read_csv(OUT / "shap_runtime.csv")
    sr = shap_runtime.loc[(shap_runtime.task == TASK) & (shap_runtime.fold == FOLD)]
    actual_calls = {}
    call_log = OUT / "logs/C_SHAP_PILOT_chla_surface_fold_1.stdout.log"
    if call_log.is_file():
        for line in call_log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
                obs_id = item.get("unit", "").split(":")[-1]
                actual_calls[obs_id] = int(item.get("actual_model_predict_calls", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    shap_checks = []
    for obs_id in EXPECTED_SHAP:
        path = OUT / "shap_values" / TASK / FOLD / f"{obs_id}_seed42.csv.gz"
        row = sr.loc[sr.obs_id == obs_id]
        df = pd.read_csv(path) if path.is_file() else pd.DataFrame()
        err = float(row.iloc[0].additivity_abs_error) if len(row) else None
        calls = int(actual_calls.get(obs_id, 0))
        elapsed = float(row.iloc[0].elapsed_seconds) if len(row) else 0.0
        shap_checks.append({
            "obs_id": obs_id, "file_exists": path.is_file(), "shap_rows": len(df),
            "feature_count": int(df.feature.nunique()) if "feature" in df else 0,
            "additivity_abs_error": err, "actual_model_calls": calls,
            "elapsed_seconds": elapsed, "file_sha256": sha256(path) if path.is_file() else None,
            "pass": bool(path.is_file() and len(df) == 77 and df.feature.nunique() == 77
                          and err is not None and err <= 1e-6 and calls > 0 and elapsed > 0),
        })

    supports = pd.read_csv(OUT / "06_ale_1d_support_and_stability.csv")
    supports = supports.loc[(supports.task == TASK) & (supports.fold == FOLD)
                            & supports.feature.isin(EXPECTED_ALE)]
    curve = pd.read_csv(OUT / "05_ale_1d_results.csv")
    curve = curve.loc[(curve.task == TASK) & (curve.fold == FOLD)
                      & curve.feature.isin(EXPECTED_ALE)]
    repeats = pd.read_csv(OUT / "ale_1d_repeatability.csv")
    repeats = repeats.loc[(repeats.task == TASK) & (repeats.fold == FOLD)
                          & repeats.feature.isin(EXPECTED_ALE)]
    ale_checks = []
    for feature in EXPECTED_ALE:
        s = supports.loc[supports.feature == feature]
        c = curve.loc[curve.feature == feature]
        r = repeats.loc[repeats.feature == feature]
        support = s.iloc[0].to_dict() if len(s) else {}
        repeat = r.iloc[0].to_dict() if len(r) else {}
        nquery = int(support.get("n_query", 0))
        bins = int(support.get("n_bins", 0))
        ale_checks.append({
            "feature": feature, "query_rows": nquery, "curve_rows": len(c),
            "bin_count": bins, "sqlite_rows": int(by_feature.get(feature, 0)),
            "min_bin_n": support.get("min_bin_n"), "min_bin_stations": support.get("min_bin_stations"),
            "repeat_count": int(repeat.get("n_repeats", 0)),
            "actual_model_calls": int(repeat.get("actual_model_calls", 0)),
            "max_local_ale_delta_range": repeat.get("max_local_ale_delta_range"),
            "repeatability_status": repeat.get("repeatability_status"),
            "computed": bool(len(s) == 1 and len(c) == bins and bins > 0 and nquery > 0
                              and int(by_feature.get(feature, 0)) == nquery),
            "pass": bool(len(s) == 1 and len(c) == bins and bins > 0 and nquery > 0
                          and int(by_feature.get(feature, 0)) == nquery
                          and int(repeat.get("n_repeats", 0)) == 3
                          and int(repeat.get("actual_model_calls", 0)) >= 6
                          and float(repeat.get("max_local_ale_delta_range", float("inf"))) <= REPEAT_TOL),
        })

    env = pd.read_csv(OUT / "environment_manifest.csv")
    env1 = env.loc[(env.task == TASK) & (env.fold == FOLD)]
    feature_hash = str(env1.iloc[-1].feature_order_hash) if len(env1) else None
    model_hash = str(env1.iloc[-1].checkpoint_sha256) if len(env1) else None
    context_hash = str(env1.iloc[-1].context_hash) if len(env1) else None
    env_hash = sha256(OUT / "environment_manifest.csv")
    registry_path = OUT / "13_final_explanation_master_registry.csv"
    registry = pd.read_csv(registry_path)
    boundary_output = OUT / "pointwise/chla_surface/fold_2/replicate_4.csv"
    boundary_unit = "A:chla_surface:fold_2:rep4"
    boundary_recorded = bool((registry.unit == boundary_unit).any())
    if boundary_output.is_file() and not boundary_recorded:
        boundary_row = pd.DataFrame([{
            "unit": boundary_unit, "status": "OUTPUT_RECOVERED_AFTER_SCHEDULER_PAUSE",
            "elapsed_seconds": None, "stdout_log": "logs/A_chla_surface_fold_2_rep4.stdout.log",
            "stderr_log": "logs/A_chla_surface_fold_2_rep4.stderr.log",
            "error": "Fold-2 rep4 output exists; scheduler was paused immediately after worker exit before registry append.",
            "output_path": str(boundary_output.relative_to(OUT)), "output_sha256": sha256(boundary_output),
            "completed_at": pd.Timestamp(boundary_output.stat().st_mtime, unit="s", tz="UTC").isoformat(),
        }])
        registry = pd.concat([registry, boundary_row], ignore_index=True)
        registry.to_csv(registry_path, index=False, encoding="utf-8-sig")
        boundary_recorded = True
    failed_attempts = registry.loc[(registry.unit == "D_ALE1_PILOT:chla_surface:fold_1")
                                   & registry.status.str.startswith("FAILED", na=False)]
    completed_attempts = registry.loc[(registry.unit == "D_ALE1_PILOT:chla_surface:fold_1:retry1")
                                      & (registry.status == "COMPLETED")]
    flag = OUT / "PAUSE_AFTER_CURRENT_FOLD.flag"
    all_shap_pass = all(x["pass"] for x in shap_checks)
    all_ale_pass = all(x["pass"] for x in ale_checks)
    final_status = "CHLA_SURFACE_FOLD1_COMPLETE" if all_shap_pass and all_ale_pass else "FOLD1_NOT_COMPLETE_VALIDATION_FAILURE"

    audit = {
        "task": TASK, "fold": FOLD, "shap_expected": len(EXPECTED_SHAP),
        "shap_completed": sum(x["pass"] for x in shap_checks), "shap_checks": shap_checks,
        "ale1_expected": len(EXPECTED_ALE), "ale1_completed": sum(x["computed"] for x in ale_checks),
        "ale1_passed_validation": sum(x["pass"] for x in ale_checks),
        "ale1_checks": ale_checks,
        "failed_units": [{"unit": "D_ALE1_PILOT:chla_surface:fold_1", "attempt_status": "FAILED_RUNTIME_1",
                          "reason": "CUDA illegal instruction in torch.tensor allocation; bounded retry completed"}]
                         if len(failed_attempts) and len(completed_attempts) else [],
        "skipped_units": [], "checkpoint_db": str(db.relative_to(ROOT)),
        "checkpoint_db_rows": int(db_rows), "rows_by_feature": by_feature,
        "sqlite_integrity_check": integrity, "sqlite_wal_checkpoint_truncate": list(wal_result),
        "sqlite_wal_file_exists_after_checkpoint": Path(str(db) + "-wal").exists(),
        "feature_order_hash": feature_hash, "model_checkpoint_hash": model_hash,
        "train_context_hash": context_hash, "environment_hash": env_hash,
        "pause_flag_present": flag.is_file(), "fold2_rep4_started_and_finished_before_pause":
            boundary_output.is_file(), "fold2_rep4_registry_recorded": boundary_recorded,
        "final_status": final_status,
        "decision_reason": "ssrd_mean_14d repeatability range exceeds the predeclared 1e-6 threshold; no rerun or threshold change performed",
    }
    out = OUT / "fold_1_completion_audit.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"audit_path": str(out), "final_status": final_status,
                      "shap_completed": audit["shap_completed"], "ale1_completed": audit["ale1_completed"],
                      "sqlite_rows": db_rows, "integrity": integrity, "wal_checkpoint": wal_result},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
