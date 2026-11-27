"""只读校验远端/本地 TabICLv2 SHAP，并在本目录生成可追溯合并结果。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


ROOT = Path(r"E:\研究生\codex\水质预测")
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/tabicl_v2_slim_explainability_v2_all10"
ZIP = ROOT / "tabicl_v2_shap_results_for_return_20261029.zip"
SIDECAR = ZIP.with_suffix(ZIP.suffix + ".sha256")
PREFIX = "tabicl_v2_shap_return_20261029/"
STRICT = ROOT / "transfer/llm_extension_transfer_v1/data/feature_manifest/01_strict_feature_manifest.csv"
TASKS = ["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def copy_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ensure(digest(path) == sha256(data), f"目标已存在且内容不同：{path}")
        return
    with path.open("xb") as f:
        f.write(data)
    ensure(digest(path) == sha256(data), f"复制后校验失败：{path}")


def read_shap(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(gzip.decompress(data)))


def single(df: pd.DataFrame, field: str):
    values = df[field].drop_duplicates().tolist()
    ensure(len(values) == 1, f"{field} 不是单值：{values[:3]}")
    return values[0]


def validate_shap(data: bytes, rel: str, source: str, features: list[str], prereg: pd.DataFrame, frozen: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    m = re.fullmatch(r"shap/([^/]+)/(.+)_seed(42|43)\.csv\.gz", rel)
    ensure(m is not None, f"SHAP 路径格式异常：{rel}")
    task, obs_id, seed = m.group(1), m.group(2), int(m.group(3))
    ensure(task in TASKS, f"未知任务：{task}")
    df = read_shap(data)
    required = {"task", "fold", "obs_id", "station_id", "feature_order", "feature", "shap_value", "base_value", "model_prediction", "additivity_abs_error", "status", "selection_phase", "shap_seed", "checkpoint_sha256", "feature_order_hash", "context_hash", "query_hash", "imputer_statistics_hash", "train_obs_order_hash", "test_obs_order_hash"}
    ensure(required <= set(df.columns), f"字段缺失：{rel} {required - set(df.columns)}")
    ensure(len(df) == 77, f"不是77行：{rel}")
    ensure(df.feature_order.astype(int).tolist() == list(range(1, 78)), f"特征序号错误：{rel}")
    ensure(df.feature.tolist() == features, f"77特征顺序错误：{rel}")
    for col, expected in [("task", task), ("fold", "fold_1"), ("obs_id", obs_id), ("shap_seed", seed), ("status", "COMPUTED_PASS")]:
        ensure(str(single(df, col)) == str(expected), f"{col} 不匹配：{rel}")
    ensure(str(single(df, "station_id")) == obs_id.split("_")[0], f"站点ID不匹配：{rel}")
    phase = "CORE_3" if seed == 42 else "SENSITIVITY_1"
    ensure(single(df, "selection_phase") == phase, f"phase不匹配：{rel}")
    # SENSITIVITY_1 使用 CORE_3 的中位样本重算，不是另一个预登记行。
    registered_phase = "CORE_3" if seed == 43 else phase
    row = prereg[(prereg.task == task) & (prereg.obs_id == obs_id) & (prereg.phase == registered_phase)]
    ensure(len(row) == 1 and row.iloc[0].fold == "fold_1", f"不在预登记清单：{rel}")
    if seed == 43:
        ensure(float(row.iloc[0].target_quantile) == 0.5, f"敏感性分析不是中位样本：{rel}")
    frow = frozen[frozen.task == task]
    ensure(len(frow) == 1, f"冻结任务注册表缺失：{task}")
    frow = frow.iloc[0]
    for field in ["checkpoint_sha256", "feature_order_hash", "imputer_statistics_hash", "train_obs_order_hash", "test_obs_order_hash"]:
        ensure(str(single(df, field)) == str(frow[field]), f"{field} 与冻结注册表不一致：{rel}")
    for field in ["shap_value", "base_value", "model_prediction", "additivity_abs_error"]:
        ensure(np.isfinite(pd.to_numeric(df[field], errors="coerce")).all(), f"数值非有限：{rel} {field}")
    max_add = float(df.additivity_abs_error.abs().max())
    ensure(max_add <= 0.02, f"加和误差超限：{rel} {max_add}")
    metadata = {"relative_path": rel, "source": source, "task": task, "obs_id": obs_id, "phase": phase, "seed": seed, "sha256": sha256(data), "bytes": len(data), "query_hash": str(single(df, "query_hash")), "context_hash": str(single(df, "context_hash")), "max_additivity_error": max_add, "model_prediction": float(single(df, "model_prediction")), "base_value": float(single(df, "base_value"))}
    return metadata, df


def main() -> None:
    ensure(ZIP.is_file() and SIDECAR.is_file() and RUN.is_dir() and STRICT.is_file(), "必要输入不存在")
    expected_zip_hash = SIDECAR.read_text(encoding="utf-8-sig").split()[0].lower()
    ensure(digest(ZIP) == expected_zip_hash, "外部ZIP SHA256不匹配")
    features = pd.read_csv(STRICT).query("included_in_pilot == True").feature.tolist()
    ensure(len(features) == 77 and len(set(features)) == 77, "冻结严格特征不是77个唯一项")
    prereg_path = RUN / "01_preregistered_shap_samples.csv"
    frozen_path = RUN / "00_frozen_protocol_task_registry.csv"
    prereg = pd.read_csv(prereg_path)
    frozen = pd.read_csv(frozen_path)
    local = {}
    remote = {}
    remote_bytes = {}
    remote_frames = {}
    with zipfile.ZipFile(ZIP) as z:
        ensure(z.testzip() is None, "ZIP CRC校验失败")
        names = z.namelist()
        ensure(len(names) == len(set(names)), "ZIP存在重复成员名")
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            ensure(not p.is_absolute() and ".." not in p.parts and info.filename.startswith(PREFIX), f"ZIP路径不安全：{info.filename}")
            ensure((info.external_attr >> 16) & 0o170000 != 0o120000, f"ZIP存在软链接：{info.filename}")
            ensure(info.file_size < 20_000_000, f"ZIP成员异常大：{info.filename}")
        remote_prereg = z.read(PREFIX + "provenance/01_preregistered_shap_samples.csv")
        ensure(sha256(remote_prereg) == digest(prereg_path), "远端与本地预登记样本清单不一致")
        manifest_bytes = z.read(PREFIX + "manifest_sha256.csv")
        manifest = pd.read_csv(io.BytesIO(manifest_bytes))
        ensure(len(manifest) == 28 and manifest.relative_path.is_unique, "远端清单不是28个唯一SHAP文件")
        listed = set(manifest.relative_path)
        actual = {n[len(PREFIX):] for n in names if n.startswith(PREFIX + "shap/") and n.endswith(".csv.gz")}
        ensure(listed == actual, "远端SHAP清单与ZIP内容不一致")
        for _, item in manifest.iterrows():
            rel = item.relative_path
            data = z.read(PREFIX + rel)
            ensure(len(data) == int(item["bytes"]) and sha256(data) == str(item.sha256), f"远端文件hash不匹配：{rel}")
            meta, df = validate_shap(data, rel, "remote", features, prereg, frozen)
            for field in ["task", "query_hash", "context_hash", "checkpoint_sha256", "feature_order_hash", "status"]:
                value = meta.get(field, str(single(df, field)))
                ensure(str(value) == str(item[field]), f"远端清单{field}不匹配：{rel}")
            ensure(int(item.feature_count) == 77 and str(item.kind) == meta["phase"], f"远端清单特征数或phase不匹配：{rel}")
            remote[rel] = meta
            remote_bytes[rel] = data
            remote_frames[rel] = df
        summary = pd.read_csv(io.BytesIO(z.read(PREFIX + "task_summary.csv")))
        ensure(len(summary) == 7 and set(summary.task) == {remote[x]["task"] for x in remote}, "远端任务汇总不一致")
        for _, row in summary.iterrows():
            entries = [v for v in remote.values() if v["task"] == row.task]
            ensure(len(entries) == int(row.file_count) == 4 and sum(v["phase"] == "CORE_3" for v in entries) == 3 and sum(v["phase"] == "SENSITIVITY_1" for v in entries) == 1, f"远端任务计数不一致：{row.task}")
        for name in ["README_TRANSFER.md", "manifest_sha256.csv", "task_summary.csv", "provenance/01_preregistered_shap_samples.csv", "provenance/run_registry_local_shap_tasks.csv"]:
            copy_bytes(z.read(PREFIX + name), OUT / "sources/remote" / name)
    local_frames = {}
    for p in sorted((RUN / "shap").rglob("*.csv.gz")):
        rel = p.relative_to(RUN).as_posix()
        data = p.read_bytes()
        meta, df = validate_shap(data, rel, "local", features, prereg, frozen)
        local[rel] = meta
        local_frames[rel] = df
    ensure(len(local) == 19, f"本地SHAP数量不是审计时的19份：{len(local)}；可能有并发写入")
    overlaps = sorted(set(local) & set(remote))
    remote_unique = sorted(set(remote) - set(local))
    ensure(len(overlaps) == 8 and len(remote_unique) == 20, "覆盖关系与审计时不同，停止以避免覆盖")
    comparisons = []
    for rel in overlaps:
        a, b = local_frames[rel], remote_frames[rel]
        ensure(local[rel]["query_hash"] == remote[rel]["query_hash"] and local[rel]["context_hash"] == remote[rel]["context_hash"], f"重名文件模型上下文冲突：{rel}")
        diffs = (a.shap_value.to_numpy() - b.shap_value.to_numpy()).astype(float)
        top_a = set(a.iloc[np.argsort(-np.abs(a.shap_value.to_numpy()))[:10]].feature)
        top_b = set(b.iloc[np.argsort(-np.abs(b.shap_value.to_numpy()))[:10]].feature)
        comparisons.append({"relative_path": rel, "local_sha256": local[rel]["sha256"], "remote_sha256": remote[rel]["sha256"], "max_abs_shap_delta": float(np.max(np.abs(diffs))), "mean_abs_shap_delta": float(np.mean(np.abs(diffs))), "prediction_delta": abs(local[rel]["model_prediction"] - remote[rel]["model_prediction"]), "base_delta": abs(local[rel]["base_value"] - remote[rel]["base_value"]), "top10_feature_overlap": len(top_a & top_b), "decision": "LOCAL_CANONICAL_REMOTE_VARIANT"})
    mapping = []
    selected_frames = {}
    for rel in sorted(set(local) | set(remote)):
        if rel in local:
            src, meta, data, frame = "local", local[rel], (RUN / rel).read_bytes(), local_frames[rel]
        else:
            src, meta, data, frame = "remote", remote[rel], remote_bytes[rel], remote_frames[rel]
        ensure(sha256(data) == meta["sha256"], f"源文件在合并期间变化：{rel}")
        dest = OUT / "merged" / rel
        copy_bytes(data, dest)
        mapping.append({"merged_relative_path": dest.relative_to(OUT).as_posix(), "source": src, "source_path": str(RUN / rel) if src == "local" else str(ZIP) + "!" + PREFIX + rel, "sha256": meta["sha256"], "bytes": meta["bytes"], "decision": "LOCAL_CANONICAL_OVERLAP" if rel in overlaps else "UNIQUE"})
        selected_frames[rel] = frame
    for rel in overlaps:
        copy_bytes(remote_bytes[rel], OUT / "remote_overlap_variants" / rel)
    ale_files = sorted(p for p in (RUN / "ale_1d").rglob("*") if p.is_file())
    for p in ale_files:
        data = p.read_bytes()
        dest = OUT / "merged/ale_1d" / p.relative_to(RUN / "ale_1d")
        copy_bytes(data, dest)
        mapping.append({"merged_relative_path": dest.relative_to(OUT).as_posix(), "source": "local", "source_path": str(p), "sha256": sha256(data), "bytes": len(data), "decision": "ALE_LOCAL_SNAPSHOT"})
    ale2_files = sorted(p for p in (RUN / "ale_2d").rglob("*") if p.is_file())
    for p in ale2_files:
        data = p.read_bytes()
        dest = OUT / "merged/ale_2d" / p.relative_to(RUN / "ale_2d")
        copy_bytes(data, dest)
        mapping.append({"merged_relative_path": dest.relative_to(OUT).as_posix(), "source": "local", "source_path": str(p), "sha256": sha256(data), "bytes": len(data), "decision": "ALE_2D_SUPPORT_ONLY"})
    for p in [prereg_path, frozen_path, RUN / "run_registry.csv", STRICT]:
        copy_bytes(p.read_bytes(), OUT / "sources/local" / p.name)
    importance = []
    sample_top = []
    for task in TASKS:
        task_frames = [frame for rel, frame in selected_frames.items() if rel.split("/")[1] == task]
        if not task_frames:
            continue
        stack = np.stack([frame.shap_value.to_numpy(dtype=float) for frame in task_frames])
        means = np.mean(np.abs(stack), axis=0)
        ranks = np.argsort(-means)
        for rank, idx in enumerate(ranks, 1):
            importance.append({"task": task, "rank": rank, "feature": features[idx], "mean_abs_shap": float(means[idx]), "n_samples": len(task_frames), "scope": "仅已预登记的fold_1样本，不代表全测试集"})
        for rel, frame in selected_frames.items():
            if rel.split("/")[1] != task:
                continue
            idxs = np.argsort(-np.abs(frame.shap_value.to_numpy(dtype=float)))[:10]
            for rank, idx in enumerate(idxs, 1):
                sample_top.append({"task": task, "obs_id": single(frame, "obs_id"), "seed": int(single(frame, "shap_seed")), "rank": rank, "feature": features[idx], "shap_value": float(frame.iloc[idx].shap_value), "source": "local" if rel in local else "remote"})
    write_csv(OUT / "summary/feature_importance_by_task.csv", importance, ["task", "rank", "feature", "mean_abs_shap", "n_samples", "scope"])
    write_csv(OUT / "summary/sample_top10_features.csv", sample_top, ["task", "obs_id", "seed", "rank", "feature", "shap_value", "source"])
    write_csv(OUT / "summary/overlap_comparison.csv", comparisons, list(comparisons[0]))
    write_csv(OUT / "summary/source_path_mapping.csv", mapping, list(mapping[0]))
    coverage = []
    for task in TASKS:
        entries = [selected_frames[r] for r in selected_frames if r.split("/")[1] == task]
        core = sum(int(single(d, "shap_seed")) == 42 for d in entries)
        sensitivity = sum(int(single(d, "shap_seed")) == 43 for d in entries)
        curves = sorted((OUT / "merged/ale_1d").glob(task + "_*_curve.csv"))
        coverage.append({"task": task, "core_shap": core, "core_expected": 3, "sensitivity_shap": sensitivity, "sensitivity_expected": 1, "ale_1d_completed_curves": len(curves), "status": "SHAP_COMPLETE" if core == 3 and sensitivity == 1 else "SHAP_PARTIAL", "ale_curve_files": ";".join(p.name for p in curves)})
    write_csv(OUT / "summary/task_coverage.csv", coverage, list(coverage[0]))
    ale_summary = []
    for p in sorted((OUT / "merged/ale_1d").glob("*_curve.csv")):
        frame = pd.read_csv(p)
        ensure(len(frame) == 10 and sorted(frame.bin.astype(int)) == list(range(1, 11)), f"ALE曲线不是完整10箱：{p.name}")
        ensure(frame.status.nunique() == 1 and frame.task.nunique() == 1 and frame.feature.nunique() == 1, f"ALE曲线元数据不一致：{p.name}")
        ale_summary.append({"task": frame.task.iloc[0], "feature": frame.feature.iloc[0], "status": frame.status.iloc[0], "bins": len(frame), "samples_total": int(frame.n_samples.sum()), "repeatability_max_local_delta": float(frame.repeatability_max_local_delta.max()), "relative_path": p.relative_to(OUT).as_posix()})
    write_csv(OUT / "summary/ale_1d_curve_summary.csv", ale_summary, ["task", "feature", "status", "bins", "samples_total", "repeatability_max_local_delta", "relative_path"])
    total_core = sum(row["core_shap"] for row in coverage)
    total_sensitivity = sum(row["sensitivity_shap"] for row in coverage)
    total_curves = sum(row["ale_1d_completed_curves"] for row in coverage)
    report = f"""# TabICLv2 SHAP 回传与本地合并校验报告\n\n- 外部ZIP SHA256：`{expected_zip_hash}`，与侧车文件一致；ZIP CRC、路径安全、内部28项SHA256均通过。\n- 预登记样本清单：回传与本地SHA256一致；冻结77特征及顺序、fold、obs_id、seed、checkpoint、imputer、train/test顺序哈希、加和误差均通过。\n- 原始本地SHAP：{len(local)}份；回传SHAP：{len(remote)}份；重名：{len(overlaps)}份；回传独有：{len(remote_unique)}份。\n- 合并SHAP：{len(selected_frames)}份，覆盖10任务；CORE_3 {total_core}/30，SENSITIVITY_1 {total_sensitivity}/10。\n- 本地ALE 1D已成曲线：{total_curves}份；全部`ale_1d`文件已作为快照复制，未将不完整的支持文件视为曲线。\n- 重名8份的上下文和query哈希一致，但文件SHA256不同；保留本地为主版本，回传副本在`remote_overlap_variants/shap/`，差异详见`summary/overlap_comparison.csv`。跨GPU输出不声明位级一致。\n- 本合并仅整合解释结果；不运行模型，不修改冻结预测、split、原始SHAP或ALE，也不改本地运行注册表和看板状态。\n- `feature_importance_by_task.csv`仅反映每任务已预登记的3–4个样本，不是全测试集全局重要性，也不能替代P1质量门槛。\n\n## 未完成与注意事项\n\n- TP底层的SENSITIVITY_1仍缺1份；ALE 1D和ALE 2D未完成全部预登记范围。\n- `chla_middle_wind_speed_sq_curve.csv`原状态为`COMPUTED_WITH_NUMERICAL_LIMITATION`，本合并原样保留；不解释为稳定通过。\n- 本机`progress_status.json`仍显示旧`RUNNING`时间戳；合并未据此推断正在运行，亦未自动重启实验。\n"""
    report += f"\n补充：本地 ALE 2D 的 {len(ale2_files)} 份候选/支持文件也已复制，但没有已完成曲线；8 份重名 SHAP 的样本内前十特征集合均相同。\n"
    (OUT / "summary/merge_validation_report.md").write_text(report, encoding="utf-8")
    manifest_rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "sha256_manifest.csv":
            manifest_rows.append({"relative_path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)})
    write_csv(OUT / "summary/sha256_manifest.csv", manifest_rows, ["relative_path", "bytes", "sha256"])
    print(f"MERGE_VALIDATED: SHAP {len(selected_frames)}; CORE {total_core}/30; SENSITIVITY {total_sensitivity}/10; ALE curves {total_curves}; overlaps {len(overlaps)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"MERGE_FAILED: {exc}", file=sys.stderr)
        raise
