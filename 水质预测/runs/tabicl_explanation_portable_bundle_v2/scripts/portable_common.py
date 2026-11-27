"""可迁移运行包公共路径、哈希及checkpoint工具。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.loads((BUNDLE_ROOT / "portable_config.json").read_text(encoding="utf-8"))


def resolve(rel: str) -> Path:
    p = (BUNDLE_ROOT / rel).resolve()
    if BUNDLE_ROOT.resolve() not in p.parents and p != BUNDLE_ROOT.resolve():
        raise ValueError(f"路径越出运行包根目录：{rel}")
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_static_manifest() -> list[str]:
    manifest = BUNDLE_ROOT / "file_manifest_sha256.csv"
    if not manifest.is_file():
        return ["file_manifest_sha256.csv缺失"]
    errors = []
    d = pd.read_csv(manifest)
    for r in d.itertuples(index=False):
        is_mutable = str(r.mutable).strip().lower() in {"true", "1", "yes"}
        if is_mutable:
            continue
        p = resolve(r.relative_path)
        if not p.is_file():
            errors.append(f"缺失文件：{r.relative_path}")
        elif sha256(p) != str(r.sha256):
            errors.append(f"SHA256不匹配：{r.relative_path}")
    return errors


def atomic_manifest_path() -> Path:
    return BUNDLE_ROOT / "checkpoints/atomic_unit_manifest.csv"


def refresh_atomic_manifest() -> pd.DataFrame:
    """根据原子输出与落盘状态刷新队列，不触发任何模型调用。"""
    p = atomic_manifest_path()
    d = pd.read_csv(p, keep_default_na=False)
    root = BUNDLE_ROOT
    run = resolve("workspace/runs/tabicl_v2_final_explanation_complete_v1")
    for i, r in d.iterrows():
        typ = str(r.unit_type)
        status = str(r.status)
        if typ in {"P1_REP", "SHAP_SAMPLE", "FULL_FOLD_COMPARE", "SHAP_SENSITIVITY"}:
            out = resolve(str(r.output_path)) if str(r.output_path) else None
            if out and out.is_file() and typ == "P1_REP" and out.stat().st_size:
                if r.unit_id == "P1:chla_surface:fold_2:rep4":
                    d.at[i, "status"] = "OUTPUT_RECOVERED"
                else:
                    d.at[i, "status"] = "COMPUTED"
            elif out and out.is_file() and typ in {"SHAP_SAMPLE", "SHAP_SENSITIVITY", "FULL_FOLD_COMPARE"}:
                d.at[i, "status"] = "COMPUTED"
            elif status in {"COMPUTED", "COMPUTED_PASS", "COMPUTED_WITH_LIMITATION", "OUTPUT_RECOVERED"}:
                d.at[i, "status"] = "PENDING"
        elif typ == "ALE1_FEATURE":
            parts = str(r.unit_id).split(":", 3)
            if len(parts) == 4:
                _, task, fold, feature = parts
                sup = run / "06_ale_1d_support_and_stability.csv"
                rep = run / "ale_1d_repeatability.csv"
                curve = run / "05_ale_1d_results.csv"
                if sup.is_file():
                    s = pd.read_csv(sup)
                    s = s.loc[(s.task == task) & (s.fold == fold) & (s.feature == feature)]
                else:
                    s = pd.DataFrame()
                if not s.empty and str(s.iloc[-1].status) == "INSUFFICIENT_SUPPORT":
                    d.at[i, "status"] = "SKIPPED_INSUFFICIENT_SUPPORT"
                elif not s.empty and str(s.iloc[-1].status).startswith("COMPUTED") and curve.is_file():
                    curves = pd.read_csv(curve)
                    curves = curves.loc[(curves.task == task) & (curves.fold == fold) & (curves.feature == feature)]
                    if not curves.empty:
                        if rep.is_file():
                            rr = pd.read_csv(rep)
                            rr = rr.loc[(rr.task == task) & (rr.fold == fold) & (rr.feature == feature)]
                        else:
                            rr = pd.DataFrame()
                        if not rr.empty and float(rr.iloc[-1].max_local_ale_delta_range) > 1e-6:
                            d.at[i, "status"] = "COMPUTED_WITH_LIMITATION"
                        elif not rr.empty:
                            d.at[i, "status"] = "COMPUTED_PASS"
        elif typ == "ALE2_PAIR_GRID":
            out = resolve(str(r.output_path)) if str(r.output_path) else None
            if out and out.is_file() and out.stat().st_size:
                a2 = pd.read_csv(out)
                parts = str(r.unit_id).split(":", 5)
                if len(parts) == 6:
                    _, task, fold, f1, f2, grid = parts
                    hit = a2.loc[(a2.task == task) & (a2.fold == fold) & (a2.feature_1 == f1) &
                                 (a2.feature_2 == f2) & (a2.grid.astype(str) == grid)]
                    if not hit.empty and str(hit.iloc[-1].status) in {"COMPUTED_POINTWISE", "INSUFFICIENT_SUPPORT", "COMPUTED_WITH_LIMITATION"}:
                        d.at[i, "status"] = "COMPUTED_OR_TERMINAL"
    d.to_csv(p, index=False, encoding="utf-8-sig")
    return d


def unit_counts(d: pd.DataFrame) -> dict:
    completed = d.status.isin(["COMPUTED", "COMPUTED_PASS", "COMPUTED_WITH_LIMITATION", "OUTPUT_RECOVERED", "COMPUTED_OR_TERMINAL"])
    skipped = d.status.astype(str).str.startswith(("SKIPPED", "NOT_IN_AUDIT"))
    failed = d.status.astype(str).str.startswith("FAILED")
    return {"total": int(len(d)), "completed": int(completed.sum()), "pending": int((~completed & ~skipped & ~failed).sum()),
            "failed": int(failed.sum()), "skipped": int(skipped.sum())}

