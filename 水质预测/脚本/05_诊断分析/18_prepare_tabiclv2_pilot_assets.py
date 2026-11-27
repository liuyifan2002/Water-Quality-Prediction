"""Prepare TabICLv2 pilot environment, strict features, and saved splits.

This script does not fit a model. It only audits existing assets and writes a
small, reproducible pilot manifest under runs/tabiclv2_pilot/.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tabiclv2_pilot"
MAIN_TABLE = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
STRICT_FEATURES = ROOT / "结果" / "扩展实验" / "分海区严格验证_20260906" / "严格验证特征清单.csv"
RANDOM_SPLIT = ROOT / "结果" / "扩展实验" / "表格预训练算法对比_20260908" / "smoke_test_full_xgboost" / "split_manifest.csv"
YEAR_PREDICTIONS = ROOT / "结果" / "扩展实验" / "分海区严格验证_20260906" / "分海区严格验证_预测.csv"
STATION_PREDICTIONS = ROOT / "结果" / "扩展实验" / "分海区五折留站验证_20260906" / "五折留站点验证_预测.csv"
SAVED_CANDIDATE_ROOT = ROOT / "结果" / "扩展实验" / "候选模型稳健性验证_20260914" / "回传结果_20260914"
PORTABLE_PYTHON = ROOT / "结果" / "扩展实验" / "候选模型稳健性验证_20260914" / "portable_package" / ".venv" / "Scripts" / "python.exe"

TASKS = {
    "chla_surface": {"target": "chla", "target_column": "chla_ug_l", "depth": "surface"},
    "secchi": {"target": "secchi", "target_column": "secchi_depth_m", "depth": "surface"},
    "tn_bottom": {"target": "tn", "target_column": "tn_mg_l", "depth": "bottom"},
    "tp_middle": {"target": "tp", "target_column": "tp_mg_l", "depth": "middle"},
}

TARGET_COLUMNS = {"chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l"}
SAME_DAY_WQ_TOKENS = {
    "salinity", "temperature_c", "turbidity", "suspended_solids", "do_mg_l", "ph",
    "inorganic_nitrogen", "orthophosphate", "ammonia", "nitrate", "nitrite",
}


def run_capture(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except Exception as exc:  # pragma: no cover - platform-dependent diagnostics
        return 1, repr(exc)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def package_probe(python: Path) -> dict[str, object]:
    code = (
        "import importlib.util, sys; "
        "mods=['torch','sklearn','tabicl','tabpfn','tabm']; "
        "print('python='+sys.version.replace('\\n',' ')); "
        "[print(m+'='+str(importlib.util.find_spec(m) is not None)) for m in mods]"
    )
    if not python.exists():
        return {"python": str(python), "exists": False}
    code_value, output = run_capture([str(python), "-c", code])
    return {"python": str(python), "exists": True, "return_code": code_value, "probe": output}


def write_environment_check() -> None:
    torch_spec = importlib.util.find_spec("torch")
    tabicl_spec = importlib.util.find_spec("tabicl")
    sklearn_spec = importlib.util.find_spec("sklearn")
    nvidia_code, nvidia_output = run_capture(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"]
    )
    text = [
        "# TabICLv2 pilot 环境检查",
        "",
        f"检查时间：{datetime.now().isoformat(timespec='seconds')}",
        f"当前 Python：`{sys.executable}`",
        f"Python 版本：`{platform.python_version()}`",
        f"当前环境 sklearn：`{'可导入' if sklearn_spec else '缺失'}`",
        f"当前环境 torch：`{'可导入' if torch_spec else '缺失'}`",
        f"当前环境 tabicl：`{'可导入' if tabicl_spec else '缺失'}`",
        "",
        "## 隔离环境探测",
        "",
        "```text",
        json.dumps(package_probe(PORTABLE_PYTHON), ensure_ascii=False, indent=2),
        "```",
        "",
        "## GPU",
        "",
        f"nvidia-smi 返回码：`{nvidia_code}`",
        "```text",
        nvidia_output or "无输出",
        "```",
        "",
        "## 当前判定",
        "",
        "- Python 3.11 与 TabICLv2 官方 Python API 方向兼容。",
        "- 当前项目环境和已有候选模型便携环境都没有可导入的 `torch`/`tabicl`，不能直接开始模型试验。",
        "- GPU 驱动可见，但这不等于 Python 环境已经具备 CUDA PyTorch。",
        "- 需要在 `runs/tabiclv2_pilot/环境/.venv` 建立隔离环境；不得升级当前主环境的 NumPy、pandas 或 sklearn。",
        "- 官方回归入口为 `from tabicl import TabICLRegressor`，默认回归权重为 `tabicl-regressor-v2-20260212.ckpt`。",
        "- 官方文档：<https://tabicl.readthedocs.io/en/latest/api.html>。",
        "",
        "结论：环境存在可修复依赖缺口，目前不是硬件阻塞；在隔离环境安装完成并通过最小 import/单批量 smoke test 前，不进入 TabICLv2 正式 pilot。",
    ]
    (RUN / "00_environment_check.md").write_text("\n".join(text), encoding="utf-8")


def feature_group(feature: str) -> str:
    if feature in {"lon", "lat", "water_zone_code"}:
        return "空间与站点"
    if feature in {"year_offset", "month_sin", "month_cos", "is_wet_season", "quarter", "day_of_year", "days_since_prev_sample"}:
        return "时间"
    if feature.startswith(("rain",)):
        return "降雨"
    if feature.startswith(("t2m", "u10", "v10", "tcc", "wind")):
        return "气象"
    if feature.startswith(("sst", "ssrd")):
        return "ERA5"
    if feature.startswith("tidal"):
        return "潮汐"
    if feature.startswith(("ntl", "viirs")):
        return "夜间灯光"
    if feature.startswith(("landsat", "s2", "hq_satellite")):
        return "遥感可用性"
    return "环境派生"


def load_common_features() -> tuple[list[str], dict[str, list[str]]]:
    source = pd.read_csv(STRICT_FEATURES)
    feature_sets = {
        str(row.target): list(json.loads(row.features_json))
        for row in source.itertuples(index=False)
    }
    common = sorted(set.intersection(*(set(values) for values in feature_sets.values())))
    return common, feature_sets


def write_feature_manifest() -> list[str]:
    common, feature_sets = load_common_features()
    table_columns = set(pd.read_csv(MAIN_TABLE, nrows=0).columns)
    rows = []
    for feature in common:
        forbidden = feature in TARGET_COLUMNS or any(token in feature.lower() for token in SAME_DAY_WQ_TOKENS)
        present = feature in table_columns
        if forbidden:
            reason = "禁止进入 pilot：疑似目标或同期水质字段。"
        elif not present:
            reason = "历史严格清单有此字段，但当前正式主表没有；本轮不自行补造。"
        else:
            reason = "四指标共同严格特征；不含目标历史项和同期水质协变量。"
        rows.append(
            {
                "feature": feature,
                "group": feature_group(feature),
                "source": "分海区严格验证共同特征清单",
                "used_random": True,
                "used_year": True,
                "used_station": True,
                "deployment_safe": bool(not forbidden),
                "included_in_pilot": bool(present and not forbidden),
                "reason": reason,
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["group", "feature"])
    manifest.to_csv(RUN / "01_strict_feature_manifest.csv", index=False, encoding="utf-8-sig")
    if len(common) == 0:
        raise RuntimeError("严格特征共同交集为空。")
    return manifest.loc[manifest["included_in_pilot"], "feature"].tolist()


def load_metadata() -> pd.DataFrame:
    usecols = ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"]
    usecols += sorted(TARGET_COLUMNS)
    data = pd.read_csv(MAIN_TABLE, usecols=usecols, low_memory=False)
    data["date"] = pd.to_datetime(data["date"])
    return data


def task_obs(data: pd.DataFrame, task: str) -> pd.DataFrame:
    spec = TASKS[task]
    result = data.loc[(data["depth"].eq(spec["depth"])) & data[spec["target_column"]].notna()].copy()
    return result


def append_split(rows: list[pd.DataFrame], data: pd.DataFrame, task: str, validation: str, fold: str, test_ids: set[str], source: str) -> None:
    eligible = task_obs(data, task)
    test_ids = set(test_ids).intersection(set(eligible["obs_id"]))
    if not test_ids:
        raise RuntimeError(f"{task} {validation} 没有恢复到测试 obs_id。来源：{source}")
    frame = eligible[["obs_id", "sample_id", "station", "date"]].copy()
    frame["split"] = frame["obs_id"].isin(test_ids).map({True: "test", False: "train"})
    train = frame.loc[frame["split"].eq("train")]
    test = frame.loc[frame["split"].eq("test")]
    summary = {
        "task": task,
        "validation": validation,
        "fold": fold,
        "n_train": len(train),
        "n_test": len(test),
        "n_station_train": train["station"].nunique(),
        "n_station_test": test["station"].nunique(),
        "date_train_min": train["date"].min().date().isoformat(),
        "date_train_max": train["date"].max().date().isoformat(),
        "date_test_min": test["date"].min().date().isoformat(),
        "date_test_max": test["date"].max().date().isoformat(),
        "split_source": source,
    }
    for key, value in summary.items():
        frame[key] = value
    frame["fold"] = fold
    rows.append(frame)


def write_split_manifest() -> None:
    data = load_metadata()
    rows: list[pd.DataFrame] = []
    random = pd.read_csv(RANDOM_SPLIT, usecols=["target", "depth", "obs_id", "split", "seed"])
    saved_random = pd.read_csv(SAVED_CANDIDATE_ROOT / "outputs_random_seeds" / "split_manifest.csv", low_memory=False)
    saved_year = pd.read_csv(SAVED_CANDIDATE_ROOT / "outputs_time_holdout" / "split_manifest.csv", low_memory=False)
    saved_station = pd.read_csv(SAVED_CANDIDATE_ROOT / "outputs_station_5fold" / "split_manifest.csv", low_memory=False)
    candidate_tasks = {"chla_surface", "secchi", "tn_bottom"}
    for task in TASKS:
        spec = TASKS[task]
        if task in candidate_tasks:
            selected = saved_random.loc[
                saved_random["target"].eq(spec["target"])
                & saved_random["depth"].eq(spec["depth"])
                & saved_random["seed"].eq(42)
            ]
            source = "候选模型稳健性验证保存 random seed_42 obs_id"
        else:
            selected = random.loc[random["target"].eq(spec["target"]) & random["depth"].eq(spec["depth"]) & random["seed"].eq(42)]
            source = "已有 smoke_test_full_xgboost split_manifest.csv；复用 obs_id"
        if selected.empty:
            raise RuntimeError(f"随机保存划分缺少 {task}")
        append_split(rows, data, task, "random_strict", "seed_42", set(selected.loc[selected["split"].eq("test"), "obs_id"]), source)

    year = pd.read_csv(YEAR_PREDICTIONS, usecols=["obs_id", "target", "depth", "evaluation_mode"])
    year = year.loc[year["evaluation_mode"].eq("time_holdout_2023_2024")]
    for task in TASKS:
        spec = TASKS[task]
        if task in candidate_tasks:
            selected = saved_year.loc[
                saved_year["target"].eq(spec["target"])
                & saved_year["depth"].eq(spec["depth"])
                & saved_year["split"].eq("test")
            ]
            source = "候选模型稳健性验证保存年份留出测试 obs_id"
        else:
            selected = year.loc[year["target"].eq(spec["target"]) & year["depth"].eq(spec["depth"])]
            source = "已有分海区严格验证预测表测试 obs_id"
        append_split(rows, data, task, "year_holdout", "2014_2022_to_2023_2024", set(selected["obs_id"]), source)

    station = pd.read_csv(STATION_PREDICTIONS, usecols=["obs_id", "target", "depth", "fold"])
    for task in TASKS:
        spec = TASKS[task]
        if task in candidate_tasks:
            selected_task = saved_station.loc[
                saved_station["target"].eq(spec["target"])
                & saved_station["depth"].eq(spec["depth"])
                & saved_station["split"].eq("test")
            ]
            source = "候选模型稳健性验证保存五折留站测试 obs_id"
        else:
            selected_task = station.loc[station["target"].eq(spec["target"]) & station["depth"].eq(spec["depth"])]
            source = "已有分海区五折留站预测表测试 obs_id"
        for fold in sorted(selected_task["fold"].dropna().unique()):
            selected = selected_task.loc[selected_task["fold"].eq(fold)]
            append_split(rows, data, task, "station_holdout", f"fold_{int(fold)}", set(selected["obs_id"]), source)

    manifest = pd.concat(rows, ignore_index=True)
    key = ["task", "validation", "fold", "obs_id"]
    if manifest.duplicated(key).any():
        raise RuntimeError("恢复的 split manifest 存在重复 task/validation/fold/obs_id。")
    manifest.to_csv(RUN / "02_split_manifest.csv", index=False, encoding="utf-8-sig")
    summary = manifest.groupby(["task", "validation", "fold"], as_index=False).agg(
        n_train=("split", lambda value: int((value == "train").sum())),
        n_test=("split", lambda value: int((value == "test").sum())),
        n_station_train=("station", lambda value: int(manifest.loc[value.index]["station"].loc[manifest.loc[value.index, "split"].eq("train")].nunique())),
        n_station_test=("station", lambda value: int(manifest.loc[value.index]["station"].loc[manifest.loc[value.index, "split"].eq("test")].nunique())),
    )
    summary.to_csv(RUN / "02_split_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    for path in [MAIN_TABLE, STRICT_FEATURES, RANDOM_SPLIT, YEAR_PREDICTIONS, STATION_PREDICTIONS]:
        if not path.exists():
            raise FileNotFoundError(path)
    write_environment_check()
    features = write_feature_manifest()
    write_split_manifest()
    print(f"TabICLv2 pilot assets ready: features={len(features)}, output={RUN}")


if __name__ == "__main__":
    main()
