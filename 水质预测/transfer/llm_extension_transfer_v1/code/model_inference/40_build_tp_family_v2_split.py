"""Create the TP_FAMILY_V2 unified station-holdout protocol only."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tp_family_v2"
MAIN_TABLE = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
STRICT = ROOT / "runs" / "tabiclv2_pilot" / "01_strict_feature_manifest.csv"
TARGET = "tp_mg_l"
DEPTHS = ("surface", "middle", "bottom")
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
SEED = 20260921
PROTOCOL = "TP_FAMILY_V2_UNIFIED_STATION_SPLIT"


def features() -> list[str]:
    frame = pd.read_csv(STRICT)
    selected = frame.loc[frame["included_in_pilot"].astype(bool), "feature"].dropna().astype(str).tolist()
    if len(selected) != 77 or len(set(selected)) != 77:
        raise RuntimeError(f"strict features expected 77, found {len(selected)}")
    header = set(pd.read_csv(MAIN_TABLE, nrows=0).columns)
    missing = sorted(set(selected) - header)
    if missing:
        raise RuntimeError(f"strict features missing from master table: {missing}")
    return selected


def load_data(selected: list[str]) -> pd.DataFrame:
    cols = ["obs_id", "sample_id", "station", "date", "depth", TARGET, *selected]
    data = pd.read_csv(MAIN_TABLE, usecols=list(dict.fromkeys(cols)), dtype={"obs_id": "string", "sample_id": "string", "station": "string", "depth": "string"}, low_memory=False)
    data["obs_id"] = data["obs_id"].astype("string").str.strip()
    data["sample_id"] = data["sample_id"].astype("string").str.strip()
    data["station_id"] = data["station"].astype("string").str.strip()
    data["depth"] = data["depth"].astype("string").str.strip().str.lower()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data.loc[~np.isfinite(data[TARGET].to_numpy(float)), TARGET] = np.nan
    return data


def layer(data: pd.DataFrame, depth: str) -> pd.DataFrame:
    result = data.loc[data["depth"].eq(depth) & data["obs_id"].notna() & data["sample_id"].notna() & data["station_id"].notna() & data["date"].notna() & data[TARGET].notna()].copy()
    if result["obs_id"].duplicated().any():
        raise RuntimeError(f"{depth}: duplicated valid obs_id")
    return result


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    selected = features()
    data = load_data(selected)
    layers = {depth: layer(data, depth) for depth in DEPTHS}
    stations = {depth: set(frame["station_id"].astype(str)) for depth, frame in layers.items()}
    all_stations = sorted(set(data["station_id"].dropna().astype(str)))
    common = set.intersection(*(stations[depth] for depth in DEPTHS))
    universe_rows = []
    for station_id in all_stations:
        available = {depth: station_id in stations[depth] for depth in DEPTHS}
        reasons = ";".join(f"missing_{depth}_valid_target" for depth in DEPTHS if not available[depth])
        universe_rows.append({
            "station_id": station_id,
            **{f"{depth}_valid_rows": int((layers[depth]["station_id"] == station_id).sum()) for depth in DEPTHS},
            **{f"{depth}_available": available[depth] for depth in DEPTHS},
            "common_station": station_id in common,
            "exclusion_reason": reasons,
        })
    universe = pd.DataFrame(universe_rows).sort_values("station_id", kind="mergesort").reset_index(drop=True)
    ordered = sorted(common)
    station_frame = pd.DataFrame({"station_id": ordered})
    fold_series = pd.Series(index=station_frame.index, dtype="string")
    for number, (_, test_index) in enumerate(KFold(n_splits=5, shuffle=True, random_state=SEED).split(station_frame), 1):
        fold_series.loc[test_index] = f"fold_{number}"
    canonical = station_frame.assign(
        station_order=np.arange(1, len(station_frame) + 1),
        fold=fold_series.astype(str),
        protocol=PROTOCOL,
        seed=SEED,
        assignment_algorithm="sklearn.model_selection.KFold on sorted unique common stations",
    )
    manifests = {}
    map_by_station = canonical.set_index("station_id")["fold"].to_dict()
    for depth, number in zip(DEPTHS, (3, 4, 5)):
        frame = layers[depth].loc[layers[depth]["station_id"].isin(common)].copy()
        frame["assigned_fold"] = frame["station_id"].map(map_by_station)
        parts = []
        for fold in FOLDS:
            test = frame.loc[frame["assigned_fold"].eq(fold)]
            train = frame.loc[~frame["assigned_fold"].eq(fold)]
            for role, subset in (("train", train), ("test", test)):
                part = subset[["obs_id", "sample_id", "station_id", "date", TARGET]].copy()
                part.insert(0, "task", f"tp_{depth}")
                part.insert(1, "depth", depth)
                part.insert(2, "fold", fold)
                part.insert(3, "role", role)
                part["date"] = part["date"].dt.strftime("%Y-%m-%d")
                part["target_value"] = part.pop(TARGET)
                part["split_source"] = PROTOCOL
                parts.append(part)
        manifests[depth] = pd.concat(parts, ignore_index=True)
        manifests[depth].to_csv(RUN / f"0{number}_tp_{depth}_manifest.csv", index=False, encoding="utf-8-sig")
    summary = []
    validation = []
    reference = {}
    for depth, manifest in manifests.items():
        for fold in FOLDS:
            part = manifest.loc[manifest["fold"].eq(fold)]
            train = part.loc[part["role"].eq("train")]
            test = part.loc[part["role"].eq("test")]
            train_stations = set(train["station_id"].astype(str)); test_stations = set(test["station_id"].astype(str))
            train_ids = set(train["obs_id"].astype(str)); test_ids = set(test["obs_id"].astype(str))
            checks = {
                "train_test_station_overlap_zero": not train_stations.intersection(test_stations),
                "train_test_obs_overlap_zero": not train_ids.intersection(test_ids),
                "test_station_matches_map": test_stations == {s for s, f in map_by_station.items() if f == fold},
                "test_stations_match_other_tp_layers": fold not in reference or test_stations == reference[fold],
                "train_obs_unique": not train["obs_id"].duplicated().any(),
                "test_obs_unique": not test["obs_id"].duplicated().any(),
                "target_nonmissing": bool(part["target_value"].notna().all()),
            }
            reference[fold] = test_stations
            status = "PASS" if all(checks.values()) else "FAIL"
            summary.append({"task": f"tp_{depth}", "depth": depth, "fold": fold, "train_rows": len(train), "test_rows": len(test), "train_stations": len(train_stations), "test_stations": len(test_stations), "overlap": len(train_stations & test_stations), "status": status})
            validation.append({"task": f"tp_{depth}", "depth": depth, "fold": fold, "train_rows": len(train), "test_rows": len(test), "train_stations": len(train_stations), "test_stations": len(test_stations), "overlap": len(train_stations & test_stations), **checks, "status": status})
    summary = pd.DataFrame(summary); validation = pd.DataFrame(validation)
    universe.to_csv(RUN / "01_station_universe.csv", index=False, encoding="utf-8-sig")
    canonical.to_csv(RUN / "02_tp_family_v2_station_fold_map.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(RUN / "06_split_validation.csv", index=False, encoding="utf-8-sig")
    status = "SPLIT_APPROVED" if validation["status"].eq("PASS").all() and summary["overlap"].eq(0).all() else "SPLIT_FAILED"
    protocol = [
        "# TP_FAMILY_V2_UNIFIED_STATION_SPLIT",
        "",
        f"- status: **{status}**",
        f"- generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- master table: `{MAIN_TABLE}`",
        f"- strict feature count: `{len(selected)}`",
        f"- station counts: surface={len(stations['surface'])}, middle={len(stations['middle'])}, bottom={len(stations['bottom'])}, common={len(common)}",
        f"- seed: `{SEED}`; sklearn: `{sklearn.__version__}`; Python: `{sys.version.split()[0]}`",
        "- sorted common station IDs lexicographically; one KFold call on the unique station list; all three depths reuse the frozen map.",
        "- target filter: valid finite `tp_mg_l`, non-null obs_id/sample_id/station/date and matching depth.",
        "",
        "## Fold summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "`06_split_validation.csv` must be entirely PASS before model execution.",
    ]
    (RUN / "00_protocol.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")
    print(f"STATUS={status}")
    print(f"STATIONS surface={len(stations['surface'])} middle={len(stations['middle'])} bottom={len(stations['bottom'])} common={len(common)}")
    print(canonical.groupby("fold").size().to_string())
    print(summary.to_string(index=False))
    return 0 if status == "SPLIT_APPROVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
