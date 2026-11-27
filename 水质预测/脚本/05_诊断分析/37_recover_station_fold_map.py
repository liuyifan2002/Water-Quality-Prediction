"""Recover a historical station-to-fold map without creating a new split.

The script only reads existing station manifests and saved station-holdout
predictions. If the evidence conflicts, it keeps TN surface/middle blocked.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tabiclv2_pilot"
MAIN_TABLE = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
CURRENT_SPLIT = RUN / "02_split_manifest.csv"
HISTORICAL_SPLIT = (
    ROOT
    / "结果"
    / "扩展实验"
    / "候选模型稳健性验证_20260914"
    / "回传结果_20260914"
    / "outputs_station_5fold"
    / "split_manifest.csv"
)

PREDICTION_SOURCES = {
    "chla_surface": ["33_chla_surface_station_predictions.csv"],
    "chla_middle": ["34_chla_middle_station_predictions.csv"],
    "chla_bottom": ["35_chla_bottom_station_predictions.csv"],
    "secchi": ["09_secchi_predictions.csv", "09_secchi_predictions.parquet"],
    "tn_bottom": ["14_tn_bottom_predictions.csv", "14_tn_bottom_predictions.parquet"],
    "tp_middle": ["17_tp_middle_predictions.csv", "17_tp_middle_predictions.parquet"],
}


def norm_station(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def norm_fold(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.search(r"(?:fold|station_fold)?[_ -]?(\d+)", str(value).strip().lower())
    return f"fold_{int(match.group(1))}" if match else None


def task_from_target_depth(target: object, depth: object) -> str | None:
    target_text = str(target).strip().lower()
    depth_text = str(depth).strip().lower()
    if target_text == "chla" and depth_text in {"surface", "middle", "bottom"}:
        return f"chla_{depth_text}"
    if target_text == "secchi":
        return "secchi"
    if target_text in {"tn", "tp"} and depth_text in {"surface", "middle", "bottom"}:
        return f"{target_text}_{depth_text}"
    return None


def add_evidence(rows: list[dict[str, object]], frame: pd.DataFrame, source: str, source_kind: str, task_column: str | None = None) -> None:
    data = frame.copy()
    if task_column:
        data["recovered_task"] = data[task_column].astype(str)
    else:
        data["recovered_task"] = [task_from_target_depth(t, d) for t, d in zip(data["target"], data["depth"])]
    data["station_id"] = data["station"].map(norm_station) if "station" in data.columns else data["station_id"].map(norm_station)
    data["fold_name"] = data["fold"].map(norm_fold)
    data = data.loc[data["recovered_task"].isin(["chla_surface", "chla_middle", "chla_bottom", "secchi", "tn_bottom", "tp_middle"])]
    data = data.loc[data["station_id"].notna() & data["fold_name"].notna()]
    for row in data[["recovered_task", "station_id", "fold_name"]].drop_duplicates().itertuples(index=False):
        rows.append(
            {
                "task": row.recovered_task,
                "station_id": row.station_id,
                "fold": row.fold_name,
                "recovery_source": source,
                "source_kind": source_kind,
            }
        )


def read_prediction(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=["station_id", "fold", "validation", "model"], low_memory=False)
    return pd.read_parquet(path, columns=["station_id", "fold", "validation", "model"])


def collect_evidence() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if CURRENT_SPLIT.exists():
        current = pd.read_csv(CURRENT_SPLIT, usecols=["task", "validation", "fold", "split", "station"])
        current = current.loc[current["validation"].eq("station_holdout") & current["split"].eq("test")].copy()
        add_evidence(rows, current.rename(columns={"task": "recovered_task"}), str(CURRENT_SPLIT), "current_split_manifest", task_column="recovered_task")
    if HISTORICAL_SPLIT.exists():
        historical = pd.read_csv(HISTORICAL_SPLIT, usecols=["evaluation_mode", "target", "depth", "fold", "split", "station"])
        historical = historical.loc[historical["evaluation_mode"].eq("station_5fold") & historical["split"].eq("test")].copy()
        # Secchi is a single formal task in this project. The historical
        # package repeats it across depth rows, so retain only its surface
        # records when recovering the task-level station assignment.
        historical = historical.loc[
            ~historical["target"].astype(str).str.lower().eq("secchi")
            | historical["depth"].astype(str).str.lower().eq("surface")
        ].copy()
        add_evidence(rows, historical, str(HISTORICAL_SPLIT), "historical_station_split_manifest")
    for task, candidates in PREDICTION_SOURCES.items():
        for filename in candidates:
            path = RUN / filename
            if not path.exists():
                continue
            prediction = read_prediction(path)
            prediction = prediction.loc[prediction["validation"].eq("station")].copy()
            prediction["recovered_task"] = task
            add_evidence(rows, prediction, str(path), "saved_station_predictions", task_column="recovered_task")
            break
    evidence = pd.DataFrame(rows).drop_duplicates(["task", "station_id", "fold", "recovery_source"])
    if evidence.empty:
        raise RuntimeError("No station-fold evidence was found")
    return evidence.sort_values(["station_id", "task", "fold", "source_kind"]).reset_index(drop=True)


def build_audit(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_id, frame in evidence.groupby("station_id", sort=True):
        folds = sorted(frame["fold"].unique())
        rows.append(
            {
                "station_id": station_id,
                "observed_tasks": ";".join(sorted(frame["task"].unique())),
                "observed_folds": ";".join(folds),
                "n_unique_folds": len(folds),
                "consistent": len(folds) == 1,
                "evidence_count": len(frame),
                "recovery_sources": ";".join(sorted(frame["recovery_source"].unique())),
            }
        )
    return pd.DataFrame(rows)


def build_canonical(audit: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    if not audit["consistent"].all():
        return pd.DataFrame()
    rows = []
    for row in audit.itertuples(index=False):
        subset = evidence.loc[evidence["station_id"].eq(row.station_id)]
        rows.append(
            {
                "station_id": row.station_id,
                "fold": row.observed_folds,
                "evidence_tasks": row.observed_tasks,
                "evidence_count": row.evidence_count,
                "recovery_source": ";".join(sorted(subset["source_kind"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)


def derive_tn_manifest(canonical: pd.DataFrame, depth: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["obs_id", "sample_id", "station", "date", "depth", "tn_mg_l"]
    data = pd.read_csv(MAIN_TABLE, usecols=columns, low_memory=False)
    data["obs_id"] = data["obs_id"].astype("string")
    data["station_id"] = data["station"].map(norm_station)
    data["date"] = pd.to_datetime(data["date"])
    data["tn_mg_l"] = pd.to_numeric(data["tn_mg_l"], errors="coerce")
    data = data.loc[data["depth"].eq(depth) & data["tn_mg_l"].notna()].copy()
    if data["obs_id"].duplicated().any():
        raise RuntimeError(f"tn_{depth}: duplicated obs_id in target data")
    canonical_map = canonical.set_index("station_id")["fold"].to_dict()
    data["assigned_fold"] = data["station_id"].map(canonical_map)
    missing = sorted(data.loc[data["assigned_fold"].isna(), "station_id"].unique())
    if missing:
        raise RuntimeError(f"tn_{depth}: {len(missing)} stations are absent from canonical map")
    rows = []
    task = f"tn_{depth}"
    source = "recovered canonical station-fold map; no new split generated"
    for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:
        test = data.loc[data["assigned_fold"].eq(fold)].copy()
        train = data.loc[~data["assigned_fold"].eq(fold)].copy()
        if test.empty or train.empty:
            raise RuntimeError(f"{task}/{fold}: empty train or test split")
        for split_name, subset in [("train", train), ("test", test)]:
            for row in subset.itertuples(index=False):
                rows.append(
                    {
                        "obs_id": str(row.obs_id),
                        "sample_id": str(row.sample_id),
                        "station_id": row.station_id,
                        "date": row.date.strftime("%Y-%m-%d"),
                        "depth": depth,
                        "target": task,
                        "validation": "station",
                        "fold": fold,
                        "split": split_name,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_station_train": train["station_id"].nunique(),
                        "n_station_test": test["station_id"].nunique(),
                        "split_source": source,
                    }
                )
    manifest = pd.DataFrame(rows)
    return manifest, data


def validate_manifest(manifest: pd.DataFrame) -> None:
    for (task, fold), frame in manifest.groupby(["target", "fold"], sort=True):
        train_stations = set(frame.loc[frame["split"].eq("train"), "station_id"])
        test_stations = set(frame.loc[frame["split"].eq("test"), "station_id"])
        if train_stations & test_stations:
            raise RuntimeError(f"{task}/{fold}: train/test station overlap")
        if frame.loc[frame["split"].eq("test"), "obs_id"].duplicated().any():
            raise RuntimeError(f"{task}/{fold}: duplicated test obs_id")


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    evidence = collect_evidence()
    evidence.to_csv(RUN / "30_station_fold_recovery_evidence.csv", index=False, encoding="utf-8-sig")
    audit = build_audit(evidence)
    audit.to_csv(RUN / "30_station_fold_consistency_audit.csv", index=False, encoding="utf-8-sig")
    conflicts = audit.loc[~audit["consistent"]].copy()
    conflicts.to_csv(RUN / "30_station_fold_conflicts.csv", index=False, encoding="utf-8-sig")
    canonical = build_canonical(audit, evidence)
    status_rows = []
    if canonical.empty:
        decision = "BLOCKED_CONFLICT" if not conflicts.empty else "BLOCKED_NO_CANONICAL_MAP"
        reason = "存在 station_id 跨任务折次冲突，不能自行选择折次；TN surface/middle 保持 BLOCKED。" if not conflicts.empty else "没有足够的一致 station-fold 证据；TN surface/middle 保持 BLOCKED。"
    else:
        canonical.to_csv(RUN / "31_canonical_station_fold_map.csv", index=False, encoding="utf-8-sig")
        derived = []
        for depth, filename in [("surface", "32_tn_surface_recovered_station_manifest.csv"), ("middle", "33_tn_middle_recovered_station_manifest.csv")]:
            manifest, data = derive_tn_manifest(canonical, depth)
            validate_manifest(manifest)
            manifest.to_csv(RUN / filename, index=False, encoding="utf-8-sig")
            derived.append((depth, manifest))
            for fold, frame in manifest.groupby("fold", sort=True):
                status_rows.append(
                    {
                        "task": f"tn_{depth}",
                        "fold": fold,
                        "n_train": int(frame.loc[frame["split"].eq("train"), "obs_id"].nunique()),
                        "n_test": int(frame.loc[frame["split"].eq("test"), "obs_id"].nunique()),
                        "split_source": "canonical station-fold recovery",
                        "status": "RECOVERED",
                    }
                )
        decision = "CANONICAL_SPLIT_RECOVERED"
        reason = "已有任务和逐样本 station-holdout 证据中的 station_id -> fold assignment 一致，已按实际 TN surface/middle obs_id 派生固定 manifest；未生成新的随机划分。"
    status_columns = ["task", "fold", "n_train", "n_test", "split_source", "status"]
    pd.DataFrame(status_rows, columns=status_columns).to_csv(RUN / "34_tn_recovered_split_status.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# TN station split 恢复审计",
        "",
        f"## 结论：{decision}",
        "",
        reason,
        "",
        f"证据行数：`{len(evidence)}`；涉及站点：`{evidence['station_id'].nunique()}`；冲突站点：`{len(conflicts)}`。",
        "",
        "## 审计文件",
        "",
        "- `30_station_fold_recovery_evidence.csv`：各任务、各来源提取的 station-fold 证据。",
        "- `30_station_fold_consistency_audit.csv`：按 station_id 汇总的跨任务一致性。",
        "- `30_station_fold_conflicts.csv`：冲突清单；为空表示未发现冲突。",
        "- `31_canonical_station_fold_map.csv`：仅在一致性通过时生成。",
        "",
        "## 约束",
        "",
        "本次只恢复历史固定站点分组，没有调用 GroupKFold、KFold 或随机 shuffle；没有运行模型、安装依赖或修改正式主链。",
    ]
    (RUN / "34_station_fold_recovery_decision.md").write_text("\n".join(lines), encoding="utf-8")
    print(decision)
    if status_rows:
        print(pd.DataFrame(status_rows).to_string(index=False))


if __name__ == "__main__":
    main()
