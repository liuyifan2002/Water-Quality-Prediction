"""Recover the frozen Random Strict obs_id allocation for ten tasks."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/random_10task_v1"
PILOT = ROOT / "runs/tabiclv2_pilot/02_split_manifest.csv"
GLOBAL = ROOT / "结果/扩展实验/表格预训练算法对比_20260908/smoke_test_full_xgboost/split_manifest.csv"
TABLE = ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv"
TASKS = {
    "chla_surface": ("chla_ug_l", "surface", "chla"),
    "chla_middle": ("chla_ug_l", "middle", "chla"),
    "chla_bottom": ("chla_ug_l", "bottom", "chla"),
    "secchi": ("secchi_depth_m", "surface", "secchi"),
    "tn_surface": ("tn_mg_l", "surface", "tn"),
    "tn_middle": ("tn_mg_l", "middle", "tn"),
    "tn_bottom": ("tn_mg_l", "bottom", "tn"),
    "tp_surface": ("tp_mg_l", "surface", "tp"),
    "tp_middle": ("tp_mg_l", "middle", "tp"),
    "tp_bottom": ("tp_mg_l", "bottom", "tp"),
}
HISTORICAL = {"chla_surface", "secchi", "tn_bottom", "tp_middle"}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TABLE, usecols=["obs_id", "sample_id", "station", "date", "depth", "chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l"], low_memory=False)
    data["obs_id"] = data["obs_id"].astype(str)
    if data.obs_id.duplicated().any():
        raise ValueError("正式主表 obs_id 不唯一")
    pilot = pd.read_csv(PILOT, dtype={"obs_id": str}, low_memory=False)
    pilot = pilot.loc[pilot.validation.eq("random_strict") & pilot.fold.eq("seed_42")]
    global_split = pd.read_csv(GLOBAL, dtype={"obs_id": str}, low_memory=False)
    global_split = global_split.loc[global_split.seed.eq(42)]
    outputs = []
    for task, (target, depth, global_target) in TASKS.items():
        eligible = data.loc[data.depth.eq(depth) & data[target].notna(), ["obs_id", "sample_id", "station", "date"]].copy()
        if task in HISTORICAL:
            saved = pilot.loc[pilot.task.eq(task), ["obs_id", "split", "split_source"]]
            source = "pilot 原冻结 random_strict seed_42"
        else:
            saved = global_split.loc[global_split.target.eq(global_target) & global_split.depth.eq(depth), ["obs_id", "split"]].copy()
            saved["split_source"] = str(GLOBAL.relative_to(ROOT))
            source = "全局冻结 seed_42 映射"
        if saved.obs_id.duplicated().any():
            raise ValueError(f"{task}: split obs_id 重复")
        merged = eligible.merge(saved, on="obs_id", how="left", validate="one_to_one")
        if merged.split.isna().any() or set(merged.split) != {"train", "test"} or len(merged) != len(saved):
            raise ValueError(f"{task}: 划分与有效样本不一致 (eligible={len(merged)}, saved={len(saved)}, missing={merged.split.isna().sum()})")
        merged.insert(0, "task", task)
        merged["fold"] = "seed_42"
        merged["validation"] = "random_strict"
        merged["split_origin"] = source
        outputs.append(merged)
    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(OUT / "01_random_split_manifest.csv", index=False, encoding="utf-8-sig")
    print(result.groupby(["task", "split"]).size().unstack().to_string())

if __name__ == "__main__":
    main()
