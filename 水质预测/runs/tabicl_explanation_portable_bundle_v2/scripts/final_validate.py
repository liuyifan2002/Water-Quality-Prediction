"""检查断点完整性、历史状态和结果队列；不重算解释。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from portable_common import BUNDLE_ROOT, refresh_atomic_manifest, unit_counts, verify_static_manifest


def main() -> int:
    errors = verify_static_manifest()
    run = BUNDLE_ROOT / "workspace/runs/tabicl_v2_final_explanation_complete_v1"
    audit = json.loads((run / "fold_1_completion_audit.json").read_text(encoding="utf-8"))
    v2 = json.loads((BUNDLE_ROOT / "workspace/runs/tabicl_explanation_numerical_stability_v2/numerical_stability_decision.json").read_text(encoding="utf-8"))
    registry = pd.read_csv(run / "13_final_explanation_master_registry.csv")
    rep4 = registry.loc[registry.unit.eq("A:chla_surface:fold_2:rep4")]
    if audit.get("final_status") != "FOLD1_NOT_COMPLETE_VALIDATION_FAILURE": errors.append("fold1历史状态被修改")
    if v2.get("final_conclusion") != "NUMERICAL_JITTER_NONNEGLIGIBLE_FOR_SMALL_EFFECTS": errors.append("V2数值稳定性状态不匹配")
    if rep4.empty or rep4.iloc[-1].status != "OUTPUT_RECOVERED_AFTER_SCHEDULER_PAUSE": errors.append("fold2 rep4 registry记录缺失")
    if not (run / "pointwise/chla_surface/fold_2/replicate_4.csv").is_file(): errors.append("fold2 rep4结果文件缺失")
    dbp = run / "checkpoints/local_effects.sqlite"
    try:
        con = sqlite3.connect(f"file:{dbp.as_posix()}?mode=ro", uri=True)
        check = con.execute("PRAGMA integrity_check").fetchone()[0]
        n = con.execute("SELECT COUNT(*) FROM ale1_local").fetchone()[0]
        con.close()
        if check != "ok": errors.append(f"SQLite integrity_check={check}")
    except Exception as exc:
        n, check = None, f"ERROR {exc!r}"
        errors.append(check)
    units = refresh_atomic_manifest()
    counts = unit_counts(units)
    report = {"status":"PORTABLE_RESULTS_VALID" if not errors else "PORTABLE_RESULTS_NEED_REVIEW",
              "errors":errors,"bundle_created_status":"PORTABLE_BUNDLE_CREATED_WITH_KNOWN_NUMERICAL_NONDETERMINISM",
              "fold1_historical_status":audit.get("final_status"),"v1_status":"INFERENCE_NONDETERMINISM_CONFIRMED",
              "v2_status":v2.get("final_conclusion"),"fold2_rep4_preserved":bool(not rep4.empty),
              "sqlite_integrity":check,"sqlite_ale1_rows":n,"atomic_units":counts}
    p = BUNDLE_ROOT / "logs/final_validation.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

