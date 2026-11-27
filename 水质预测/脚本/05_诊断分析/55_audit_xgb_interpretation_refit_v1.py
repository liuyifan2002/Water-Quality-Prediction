"""只读审计 XGBoost 解释模型受控复现所需的配置与训练输入。"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "xgb_interpretation_refit_v1"
MAIN = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
FEATURE_MANIFEST = ROOT / "runs" / "tabiclv2_pilot" / "01_strict_feature_manifest.csv"
CONFIG = ROOT / "结果" / "扩展实验" / "候选模型稳健性验证_20260914" / "portable_package" / "config.json"
SEED = 42
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))

TASKS = {
    "chla_surface": {
        "depth": "surface", "target": "chla_ug_l", "run": "chla_family_v2",
        "manifest": "03_chla_surface_manifest.csv", "registry": "07_run_registry.csv",
        "script": "脚本/04_模型训练/44_run_chla_family_v2_models.py",
        "base_script": "脚本/04_模型训练/36_run_tn_family_v2_models.py",
        "split_version": "CHLA_FAMILY_V2_UNIFIED_STATION_SPLIT",
    },
    "tn_surface": {
        "depth": "surface", "target": "tn_mg_l", "run": "tn_family_v2",
        "manifest": "03_tn_surface_manifest.csv", "registry": "07_run_registry.csv",
        "script": "脚本/04_模型训练/36_run_tn_family_v2_models.py",
        "base_script": "脚本/04_模型训练/36_run_tn_family_v2_models.py",
        "split_version": "TN_FAMILY_V2_UNIFIED_STATION_SPLIT",
    },
    "tp_middle": {
        "depth": "middle", "target": "tp_mg_l", "run": "tp_family_v2",
        "manifest": "04_tp_middle_manifest.csv", "registry": "07_run_registry.csv",
        "script": "脚本/04_模型训练/41_run_tp_family_v2_models.py",
        "base_script": "脚本/04_模型训练/36_run_tn_family_v2_models.py",
        "split_version": "TP_FAMILY_V2_UNIFIED_STATION_SPLIT",
    },
    "secchi": {
        "depth": None, "target": "secchi_depth_m", "run": "tabiclv2_pilot",
        "manifest": "02_split_manifest.csv", "registry": "07_secchi_strict_metrics.csv",
        "script": "脚本/04_模型训练/33_run_tabiclv2_chla_smoke.py",
        "base_script": "脚本/04_模型训练/33_run_tabiclv2_chla_smoke.py",
        "split_version": "SECCHI_HISTORICAL_STATION_HOLDOUT",
    },
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
    }


def load_features() -> list[str]:
    frame = pd.read_csv(FEATURE_MANIFEST)
    features = frame.loc[frame["included_in_pilot"].astype(bool), "feature"].dropna().astype(str).tolist()
    if len(features) != 77 or len(set(features)) != 77:
        raise RuntimeError(f"strict feature manifest异常: {len(features)}")
    return features


def load_secchi_manifest() -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(ROOT / "runs/tabiclv2_pilot/02_split_manifest.csv", dtype={"obs_id": "string"}, chunksize=200_000):
        keep = chunk["task"].eq("secchi") & chunk["validation"].eq("station_holdout") & chunk["fold"].isin(FOLDS)
        if keep.any():
            parts.append(chunk.loc[keep, ["obs_id", "split", "fold"]])
    result = pd.concat(parts, ignore_index=True)
    return result.rename(columns={"split": "role"})


def load_manifest(task: str, spec: dict[str, object]) -> pd.DataFrame:
    if task == "secchi":
        return load_secchi_manifest()
    path = ROOT / "runs" / str(spec["run"]) / str(spec["manifest"])
    return pd.read_csv(path, dtype={"obs_id": "string"}, usecols=["obs_id", "role", "fold"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = load_features()
    versions = package_versions()
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    xgb_params = dict(config["model_parameters"]["xgboost"])
    effective_params = {**xgb_params, "random_state": SEED, "objective": "reg:squarederror"}

    header = pd.read_csv(MAIN, nrows=0).columns.tolist()
    required = ["obs_id", "station", "depth", *(spec["target"] for spec in TASKS.values()), *features]
    required = list(dict.fromkeys(required))
    missing_columns = [c for c in required if c not in header]
    if missing_columns:
        raise RuntimeError(f"正式主表缺少字段: {missing_columns}")
    data = pd.read_csv(MAIN, usecols=required, dtype={"obs_id": "string", "station": "string", "depth": "string"}, low_memory=False)
    data["obs_id"] = data["obs_id"].astype("string").str.strip()
    data["depth"] = data["depth"].astype("string").str.strip().str.lower()

    config_rows, input_rows = [], []
    for task, spec in TASKS.items():
        run_dir = ROOT / "runs" / str(spec["run"])
        manifest_path = run_dir / str(spec["manifest"])
        script_path = ROOT / str(spec["script"])
        base_script_path = ROOT / str(spec["base_script"])
        manifest = load_manifest(task, spec)
        task_data = data.loc[data[str(spec["target"])].notna()].copy()
        if spec["depth"] is not None:
            task_data = task_data.loc[task_data["depth"].eq(spec["depth"])].copy()
        task_ids = set(task_data["obs_id"].astype(str))

        if task == "secchi":
            registry = pd.read_csv(run_dir / str(spec["registry"])).query("validation == 'station' and model == 'xgb'")
            prediction_all = pd.read_csv(run_dir / "09_secchi_predictions.csv", dtype={"obs_id": "string"})
            prediction_all = prediction_all.loc[prediction_all["validation"].eq("station") & prediction_all["model"].astype(str).str.lower().eq("xgb")]
        else:
            registry = pd.read_csv(run_dir / str(spec["registry"])).query("task == @task and model == 'xgboost'")
            prediction_all = None

        for fold in FOLDS:
            split = manifest.loc[manifest["fold"].eq(fold)]
            train_ids = split.loc[split["role"].eq("train"), "obs_id"].astype(str).tolist()
            test_ids = split.loc[split["role"].eq("test"), "obs_id"].astype(str).tolist()
            reg = registry.loc[registry["fold"].eq(fold)].iloc[0]
            if task == "secchi":
                pred = prediction_all.loc[prediction_all["fold"].eq(fold)].copy()
                prediction_path = run_dir / "09_secchi_predictions.csv"
            else:
                prediction_path = run_dir / "predictions" / f"{task}_xgboost_{fold}.csv"
                pred = pd.read_csv(prediction_path, dtype={"obs_id": "string"})
            prediction_ids = pred["obs_id"].astype(str).tolist()
            train_set, test_set = set(train_ids), set(test_ids)
            config_rows.append({
                "task": task, "fold": fold, "training_script": rel(script_path),
                "training_script_sha256": digest(script_path), "shared_base_script": rel(base_script_path),
                "shared_base_script_sha256": digest(base_script_path), "split_manifest": rel(manifest_path),
                "split_version": spec["split_version"], "feature_manifest": rel(FEATURE_MANIFEST),
                "feature_manifest_sha256": digest(FEATURE_MANIFEST), "feature_count": len(features),
                "feature_order_sha256": hashlib.sha256("\n".join(features).encode()).hexdigest(),
                "xgb_config": rel(CONFIG), "xgb_config_sha256": digest(CONFIG),
                "xgb_parameters_json": json.dumps(effective_params, sort_keys=True), "random_seed": SEED,
                "python_version_evidence": versions["python"], "xgboost_version_evidence": versions["xgboost"],
                "sklearn_version_evidence": versions["scikit_learn"], "numpy_version_evidence": versions["numpy"],
                "pandas_version_evidence": versions["pandas"], "environment_evidence_quality": "RETAINED_VENV_NOT_PER_FOLD_LOCKFILE",
                "imputer_rule": "SimpleImputer(strategy=median, keep_empty_features=True), fit on fold train only",
                "input_dtype": "float32 after imputation", "target_transform": "log1p if fold train target minimum >= 0; inverse expm1",
                "postprocess": "maximum(inverse_prediction, 0)", "sample_weight": "NONE",
                "early_stopping": "NONE", "eval_set": "NONE", "hyperparameter_search": "NONE",
                "extra_sampling": "NONE after target/depth filter and saved obs_id selection",
                "frozen_prediction": rel(prediction_path), "frozen_prediction_sha256": digest(prediction_path),
                "frozen_R2": reg["R2"], "frozen_RMSE": reg["RMSE"], "frozen_MAE": reg["MAE"],
                "configuration_status": "COMPLETE_WITH_ENVIRONMENT_PROVENANCE_RISK",
            })
            input_rows.append({
                "task": task, "fold": fold, "target": spec["target"], "depth": spec["depth"] or "single_task",
                "manifest_train_count": len(train_ids), "manifest_test_count": len(test_ids),
                "manifest_train_unique": len(train_set), "manifest_test_unique": len(test_set),
                "train_test_obs_overlap": len(train_set & test_set),
                "train_ids_found_in_main_table": len(train_set & task_ids), "test_ids_found_in_main_table": len(test_set & task_ids),
                "registry_n_train": int(reg["n_train"]), "registry_n_test": int(reg["n_test"]),
                "prediction_count": len(prediction_ids), "prediction_unique_count": len(set(prediction_ids)),
                "prediction_obs_set_matches_test": set(prediction_ids) == test_set,
                "prediction_obs_order_sha256": hashlib.sha256("\n".join(prediction_ids).encode()).hexdigest(),
                "manifest_test_order_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
                "all_77_features_present": all(f in task_data.columns for f in features),
                "target_complete_for_saved_ids": bool(task_data.loc[task_data["obs_id"].astype(str).isin(train_set | test_set), str(spec["target"])].notna().all()),
                "input_reconstruction_status": "PASS" if (
                    len(train_ids) == len(train_set) == int(reg["n_train"])
                    and len(test_ids) == len(test_set) == int(reg["n_test"])
                    and len(train_set & task_ids) == len(train_set) and len(test_set & task_ids) == len(test_set)
                    and set(prediction_ids) == test_set and not (train_set & test_set)
                ) else "FAIL",
            })

    config_df = pd.DataFrame(config_rows)
    input_df = pd.DataFrame(input_rows)
    config_df.to_csv(OUT / "00_original_training_configuration_audit.csv", index=False, encoding="utf-8-sig")
    input_df.to_csv(OUT / "01_training_input_reconstruction_check.csv", index=False, encoding="utf-8-sig")

    feasibility = []
    for task in TASKS:
        cfg = config_df.loc[config_df["task"].eq(task)]
        inp = input_df.loc[input_df["task"].eq(task)]
        feasible = len(cfg) == 5 and inp["input_reconstruction_status"].eq("PASS").all()
        feasibility.append({
            "task": task, "folds_expected": 5, "folds_configuration_recovered": len(cfg),
            "folds_input_reconstruction_pass": int(inp["input_reconstruction_status"].eq("PASS").sum()),
            "strict_feature_count": 77, "split_recovered": True, "predictions_recovered": True,
            "hyperparameters_recovered": True, "preprocessing_rule_recovered": True,
            "target_transform_recovered": True, "software_versions_per_fold_locked": False,
            "controlled_refit_feasible": feasible,
            "decision": "CONFIGURATION_SUFFICIENT_FOR_CONTROLLED_REFIT" if feasible else "INSUFFICIENT_CONFIGURATION",
            "remaining_gap": "原运行未保存逐折模型/imputer，且无每折不可变环境锁；精确一致性必须通过受控复现实测证明",
        })
    pd.DataFrame(feasibility).to_csv(OUT / "02_reproduction_feasibility_matrix.csv", index=False, encoding="utf-8-sig")
    print(f"AUDIT_ONLY rows={len(config_df)} input_pass={input_df['input_reconstruction_status'].eq('PASS').sum()}/20")


if __name__ == "__main__":
    main()
