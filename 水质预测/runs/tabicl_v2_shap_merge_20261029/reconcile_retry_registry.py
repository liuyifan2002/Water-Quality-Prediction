"""回写已验证的手工ALE重试状态，不改曲线内容或冻结结果。"""

from pathlib import Path
import pandas as pd


ROOT = Path(r"E:\研究生\codex\水质预测")
RUN = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
REGISTRY = RUN / "run_registry.csv"


def main() -> None:
    records = pd.read_csv(REGISTRY, dtype=str).fillna("")
    now = "2026-10-30T00:53:43+08:00"
    curve = RUN / "ale_1d/chla_bottom_t2m_curve.csv"
    checks = [
        ("ALE_1D", "t2m", "FINISHED", str(RUN / "logs/chla_bottom_ALE_1D_t2m_retry.log"), "retry exit_code=0"),
        ("ALE_1D_CURVE", "t2m", "COMPUTED", str(curve), "manual retry verified COMPUTED_PASS"),
        ("COVERAGE", "chla_bottom", "COMPUTED", str(RUN / "ale_1d/chla_bottom_feature_screening.csv"), "core3=3; selected_ale=2; computed_ale=2; retry reconciled"),
    ]
    for method, unit, status, output_path, reason in checks:
        mask = (records.task == "chla_bottom") & (records.method == method) & (records.unit == unit)
        row = {"task": "chla_bottom", "fold": "fold_1", "method": method, "unit": unit,
               "status": status, "output_path": output_path, "reason": reason, "updated_at": now}
        records = records.loc[~mask]
        records = pd.concat([records, pd.DataFrame([row])], ignore_index=True)
    temporary = REGISTRY.with_suffix(".csv.reconcile.tmp")
    records.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(REGISTRY)
    print("REGISTRY_RECONCILED chla_bottom ale_1d=2 coverage=COMPUTED")


if __name__ == "__main__":
    main()
