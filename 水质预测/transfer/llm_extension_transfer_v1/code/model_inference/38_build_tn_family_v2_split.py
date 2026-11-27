"""Build the TN_FAMILY_V2 unified station-holdout protocol.

This is a new extension experiment. It does not read or recover historical
station folds, and it does not fit any model. One canonical station-level
five-fold map is created from the current formal master table, then reused by
the three TN depth tasks.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tn_family_v2"
MAIN_TABLE = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
STRICT_MANIFEST = ROOT / "runs" / "tabiclv2_pilot" / "01_strict_feature_manifest.csv"

SEED = 20260921
N_SPLITS = 5
TARGET_COLUMN = "tn_mg_l"
DEPTHS = ("surface", "middle", "bottom")
TASKS = {depth: f"tn_{depth}" for depth in DEPTHS}
REQUIRED_BASE_COLUMNS = ("obs_id", "sample_id", "station", "date", "depth", TARGET_COLUMN)
FOLD_NAMES = tuple(f"fold_{i}" for i in range(1, N_SPLITS + 1))


def find_main_table() -> Path:
    if not MAIN_TABLE.exists():
        raise FileNotFoundError(f"正式主表不存在: {MAIN_TABLE}")
    return MAIN_TABLE


def load_strict_features() -> tuple[list[str], list[str]]:
    if not STRICT_MANIFEST.exists():
        raise FileNotFoundError(f"冻结特征清单不存在: {STRICT_MANIFEST}")
    manifest = pd.read_csv(STRICT_MANIFEST, dtype={"feature": "string"})
    if "included_in_pilot" not in manifest.columns:
        raise ValueError("冻结特征清单缺少 included_in_pilot 字段")
    included = manifest.loc[manifest["included_in_pilot"].astype(bool), "feature"].dropna().astype(str).tolist()
    if len(included) != 77 or len(set(included)) != 77:
        raise ValueError(f"冻结特征清单不是 77 个唯一可用特征: {len(included)}")
    return included, manifest["feature"].dropna().astype(str).tolist()


def load_base_table(features: list[str]) -> pd.DataFrame:
    header = pd.read_csv(find_main_table(), nrows=0).columns.tolist()
    missing_columns = [column for column in (*REQUIRED_BASE_COLUMNS, *features) if column not in header]
    if missing_columns:
        raise ValueError(f"正式主表缺少字段: {missing_columns}")

    # Only the identifiers, target and frozen feature columns are read. The
    # features are checked for schema compatibility here but are not used to
    # create or alter the station split.
    data = pd.read_csv(
        MAIN_TABLE,
        usecols=list((*REQUIRED_BASE_COLUMNS, *features)),
        dtype={"obs_id": "string", "sample_id": "string", "station": "string", "depth": "string"},
        low_memory=False,
    )
    data["obs_id"] = data["obs_id"].astype("string").str.strip()
    data["sample_id"] = data["sample_id"].astype("string").str.strip()
    data["station_id"] = data["station"].astype("string").str.strip()
    data["depth"] = data["depth"].astype("string").str.strip().str.lower()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data.loc[~np.isfinite(data[TARGET_COLUMN].to_numpy(dtype=float)), TARGET_COLUMN] = np.nan
    return data


def layer_frame(data: pd.DataFrame, depth: str) -> pd.DataFrame:
    result = data.loc[
        data["depth"].eq(depth)
        & data["obs_id"].notna()
        & data["sample_id"].notna()
        & data["station_id"].notna()
        & data["date"].notna()
        & data[TARGET_COLUMN].notna()
    ].copy()
    if result["obs_id"].duplicated().any():
        duplicate_count = int(result["obs_id"].duplicated().sum())
        raise ValueError(f"TN {depth} 有重复有效 obs_id: {duplicate_count}")
    return result


def build_station_universe(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    layers = {depth: layer_frame(data, depth) for depth in DEPTHS}
    station_sets = {depth: set(frame["station_id"].astype(str)) for depth, frame in layers.items()}
    all_stations = sorted(set(data["station_id"].dropna().astype(str)))
    common = set.intersection(*(station_sets[depth] for depth in DEPTHS))

    rows: list[dict[str, object]] = []
    for station_id in all_stations:
        present = {depth: station_id in station_sets[depth] for depth in DEPTHS}
        reasons = [f"missing_{depth}_valid_target" for depth in DEPTHS if not present[depth]]
        rows.append(
            {
                "station_id": station_id,
                "surface_valid_rows": int((layers["surface"]["station_id"] == station_id).sum()),
                "middle_valid_rows": int((layers["middle"]["station_id"] == station_id).sum()),
                "bottom_valid_rows": int((layers["bottom"]["station_id"] == station_id).sum()),
                "surface_available": present["surface"],
                "middle_available": present["middle"],
                "bottom_available": present["bottom"],
                "common_station": station_id in common,
                "exclusion_reason": ";".join(reasons) if reasons else "",
            }
        )
    universe = pd.DataFrame(rows).sort_values("station_id", kind="mergesort").reset_index(drop=True)
    return universe, layers


def create_canonical_map(universe: pd.DataFrame) -> pd.DataFrame:
    stations = universe.loc[universe["common_station"], "station_id"].astype(str).sort_values(kind="mergesort").tolist()
    if len(stations) < N_SPLITS:
        raise ValueError(f"三层共同 station 数不足 {N_SPLITS}: {len(stations)}")
    station_frame = pd.DataFrame({"station_id": stations})
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_values = pd.Series(index=station_frame.index, dtype="string")
    for fold_number, (_, test_index) in enumerate(splitter.split(station_frame), start=1):
        fold_values.loc[test_index] = f"fold_{fold_number}"
    result = station_frame.copy()
    result["station_order"] = np.arange(1, len(result) + 1, dtype=int)
    result["fold"] = fold_values.astype(str)
    result["protocol"] = "TN_FAMILY_V2_UNIFIED_STATION_SPLIT"
    result["seed"] = SEED
    result["assignment_algorithm"] = "sklearn.model_selection.KFold on sorted unique common stations"
    return result


def build_manifest(layer: pd.DataFrame, depth: str, canonical: pd.DataFrame) -> pd.DataFrame:
    task = TASKS[depth]
    fold_by_station = canonical.set_index("station_id")["fold"].to_dict()
    layer = layer.loc[layer["station_id"].isin(fold_by_station)].copy()
    layer["canonical_fold"] = layer["station_id"].map(fold_by_station)
    pieces: list[pd.DataFrame] = []
    for fold in FOLD_NAMES:
        test = layer.loc[layer["canonical_fold"].eq(fold)].copy()
        train = layer.loc[~layer["canonical_fold"].eq(fold)].copy()
        for role, subset in (("train", train), ("test", test)):
            part = subset[["obs_id", "station_id", "date", "sample_id", TARGET_COLUMN]].copy()
            part.insert(0, "task", task)
            part.insert(1, "depth", depth)
            part.insert(2, "fold", fold)
            part.insert(3, "role", role)
            part["date"] = part["date"].dt.strftime("%Y-%m-%d")
            part["target_value"] = part.pop(TARGET_COLUMN)
            part["split_source"] = "TN_FAMILY_V2_UNIFIED_STATION_SPLIT"
            pieces.append(part)
    manifest = pd.concat(pieces, ignore_index=True)
    return manifest[
        [
            "task",
            "depth",
            "fold",
            "role",
            "obs_id",
            "station_id",
            "date",
            "sample_id",
            "target_value",
            "split_source",
        ]
    ]


def validate_protocol(
    manifests: dict[str, pd.DataFrame], canonical: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    map_by_station = canonical.set_index("station_id")["fold"].to_dict()
    fold_station_sets = {
        fold: set(canonical.loc[canonical["fold"].eq(fold), "station_id"].astype(str)) for fold in FOLD_NAMES
    }
    summary_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    reference_test_sets: dict[str, set[str]] = {}

    for depth, manifest in manifests.items():
        task = TASKS[depth]
        for fold in FOLD_NAMES:
            part = manifest.loc[manifest["fold"].eq(fold)]
            train = part.loc[part["role"].eq("train")].copy()
            test = part.loc[part["role"].eq("test")].copy()
            train_stations = set(train["station_id"].astype(str))
            test_stations = set(test["station_id"].astype(str))
            train_obs = set(train["obs_id"].astype(str))
            test_obs = set(test["obs_id"].astype(str))
            expected_test_stations = fold_station_sets[fold]
            overlap = train_stations & test_stations
            test_station_match = test_stations == expected_test_stations
            if fold not in reference_test_sets:
                reference_test_sets[fold] = test_stations
            cross_layer_match = test_stations == reference_test_sets[fold]
            checks = {
                "train_test_station_overlap_zero": len(overlap) == 0,
                "train_test_obs_overlap_zero": len(train_obs & test_obs) == 0,
                "test_station_matches_canonical_map": test_station_match,
                "test_stations_match_other_tn_layers": cross_layer_match,
                "train_obs_unique": not train["obs_id"].duplicated().any(),
                "test_obs_unique": not test["obs_id"].duplicated().any(),
                "train_target_nonmissing": bool(train["target_value"].notna().all()),
                "test_target_nonmissing": bool(test["target_value"].notna().all()),
                "station_assignment_matches_map": bool(
                    all(map_by_station.get(station) != fold for station in train_stations)
                    and all(map_by_station.get(station) == fold for station in test_stations)
                ),
            }
            status = "PASS" if all(checks.values()) else "FAIL"
            summary_rows.append(
                {
                    "task": task,
                    "depth": depth,
                    "fold": fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_stations": len(train_stations),
                    "test_stations": len(test_stations),
                    "overlap": len(overlap),
                    "status": status,
                }
            )
            validation_rows.append(
                {
                    "task": task,
                    "depth": depth,
                    "fold": fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_stations": len(train_stations),
                    "test_stations": len(test_stations),
                    "overlap": len(overlap),
                    **checks,
                    "status": status,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(validation_rows)


def write_protocol(
    *,
    status: str,
    data: pd.DataFrame,
    strict_features: list[str],
    universe: pd.DataFrame,
    canonical: pd.DataFrame,
    summary: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    counts = {
        depth: int(universe.loc[universe[f"{depth}_available"], "station_id"].nunique())
        for depth in DEPTHS
    }
    common_count = int(universe["common_station"].sum())
    lines = [
        "# TN_FAMILY_V2_UNIFIED_STATION_SPLIT 协议",
        "",
        f"- 最终状态：**{status}**",
        f"- 生成时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 正式主表：`{MAIN_TABLE}`",
        f"- 冻结特征清单：`{STRICT_MANIFEST}`",
        f"- 实际冻结输入特征数：`{len(strict_features)}`",
        f"- 目标字段：`{TARGET_COLUMN}`；有效目标定义为可解析且为有限数值。",
        f"- 水层：`surface`、`middle`、`bottom`。",
        "",
        "## 共同站点范围",
        "",
        f"- surface 有效目标站点：`{counts['surface']}`",
        f"- middle 有效目标站点：`{counts['middle']}`",
        f"- bottom 有效目标站点：`{counts['bottom']}`",
        f"- 三层共同站点：`{common_count}`",
        "- 单层缺少有效目标的站点保留在 `01_station_universe.csv`，不进入本协议共同站点。",
        "",
        "## 固定分折规则",
        "",
        f"- 固定随机种子：`{SEED}`",
        "- 分组单位：`station_id`；同一站点所有日期和三层观测使用同一折次。",
        "- 站点排序：先将站点 ID 转为去除首尾空格的字符串，再按稳定字典序升序排列。",
        "- 算法：对排序后的共同站点唯一列表只调用一次 `sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=20260921)`。",
        f"- sklearn 版本：`{sklearn.__version__}`；Python：`{sys.version.split()[0]}`。",
        "- 三个水层只从 `02_tn_family_v2_station_fold_map.csv` 派生 manifest，不重新分折。",
        "",
        "## 目标过滤与验证",
        "",
        "- 每层保留 `depth` 匹配、`obs_id/sample_id/station_id/date` 非空、`tn_mg_l` 非空且有限的观测。",
        "- 个别月份没有目标值不会删除该站点；只有整层没有任何有效目标时才会排除站点。",
        "- `06_split_validation.csv` 的每一行必须为 `PASS`，且 `overlap` 必须为 0，才允许进入模型阶段。",
        "",
        "## 输出",
        "",
        "- `01_station_universe.csv`：三层站点统计及排除原因。",
        "- `02_tn_family_v2_station_fold_map.csv`：永久 canonical station→fold 映射。",
        "- `03_tn_surface_manifest.csv`、`04_tn_middle_manifest.csv`、`05_tn_bottom_manifest.csv`：三层 train/test obs_id 清单。",
        "- `06_split_validation.csv`：逐检查项验证结果。",
        "",
        "## 分折汇总",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## 验证结果",
        "",
        "```text",
        validation.to_string(index=False),
        "```",
        "",
        "本协议不读取上一轮历史 split，不覆盖旧 TN bottom pilot，不运行任何模型。",
    ]
    (RUN / "00_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    strict_features, _ = load_strict_features()
    data = load_base_table(strict_features)
    universe, layers = build_station_universe(data)
    canonical = create_canonical_map(universe)
    manifests = {depth: build_manifest(layers[depth], depth, canonical) for depth in DEPTHS}
    summary, validation = validate_protocol(manifests, canonical)

    universe.to_csv(RUN / "01_station_universe.csv", index=False, encoding="utf-8-sig")
    canonical.to_csv(RUN / "02_tn_family_v2_station_fold_map.csv", index=False, encoding="utf-8-sig")
    for depth, number in zip(DEPTHS, (3, 4, 5)):
        manifests[depth].to_csv(RUN / f"0{number}_tn_{depth}_manifest.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(RUN / "06_split_validation.csv", index=False, encoding="utf-8-sig")
    status = "SPLIT_APPROVED" if (validation["status"] == "PASS").all() and (summary["overlap"] == 0).all() else "SPLIT_FAILED"
    write_protocol(
        status=status,
        data=data,
        strict_features=strict_features,
        universe=universe,
        canonical=canonical,
        summary=summary,
        validation=validation,
    )

    print(f"STATUS={status}")
    print("STATION_COUNTS")
    for depth in DEPTHS:
        count = int(universe.loc[universe[f"{depth}_available"], "station_id"].nunique())
        print(f"  {depth}: {count}")
    print(f"  common: {int(universe['common_station'].sum())}")
    print("FOLD_STATION_COUNTS")
    print(canonical.groupby("fold", sort=False).size().to_string())
    print("SPLIT_SUMMARY")
    print(summary.to_string(index=False))
    print(f"VALIDATION_FILE={RUN / '06_split_validation.csv'}")
    print(f"PROTOCOL_FILE={RUN / '00_protocol.md'}")
    return 0 if status == "SPLIT_APPROVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
