"""按冻结协议重新拟合用于解释的XGBoost模型；不会写入任何正式结果目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "xgb_interpretation_refit_v1"
MAIN = ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv"
FEATURE_MANIFEST = ROOT / "runs/tabiclv2_pilot/01_strict_feature_manifest.csv"
CONFIG = ROOT / "结果/扩展实验/候选模型稳健性验证_20260914/portable_package/config.json"
SEED = 42
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
TASKS = {
    "chla_surface": ("chla_ug_l", "surface", ROOT / "runs/chla_family_v2/03_chla_surface_manifest.csv"),
    "tn_surface": ("tn_mg_l", "surface", ROOT / "runs/tn_family_v2/03_tn_surface_manifest.csv"),
    "tp_middle": ("tp_mg_l", "middle", ROOT / "runs/tp_family_v2/04_tp_middle_manifest.csv"),
    "secchi": ("secchi_depth_m", None, ROOT / "runs/tabiclv2_pilot/02_split_manifest.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--all-folds", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def list_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def features() -> list[str]:
    frame = pd.read_csv(FEATURE_MANIFEST)
    result = frame.loc[frame["included_in_pilot"].astype(bool), "feature"].dropna().astype(str).tolist()
    if len(result) != 77 or len(set(result)) != 77:
        raise RuntimeError("冻结特征不是77个唯一特征")
    return result


def split_manifest(task: str) -> pd.DataFrame:
    path = TASKS[task][2]
    if task != "secchi":
        return pd.read_csv(path, dtype={"obs_id": "string"}, usecols=["obs_id", "role", "fold"])
    parts = []
    for chunk in pd.read_csv(path, dtype={"obs_id": "string"}, chunksize=200_000):
        keep = chunk["task"].eq("secchi") & chunk["validation"].eq("station_holdout") & chunk["fold"].isin(FOLDS)
        if keep.any():
            parts.append(chunk.loc[keep, ["obs_id", "split", "fold"]])
    return pd.concat(parts, ignore_index=True).rename(columns={"split": "role"})


def load_task_data(task: str, feature_names: list[str]) -> pd.DataFrame:
    target, depth, _ = TASKS[task]
    usecols = ["obs_id", "station", "date", "depth", target, *feature_names]
    frame = pd.read_csv(MAIN, usecols=usecols, dtype={"obs_id": "string", "station": "string", "depth": "string"}, low_memory=False)
    frame["obs_id"] = frame["obs_id"].astype("string").str.strip()
    frame["depth"] = frame["depth"].astype("string").str.strip().str.lower()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.loc[frame[target].notna()].copy()
    if depth is not None:
        frame = frame.loc[frame["depth"].eq(depth)].copy()
    for name in feature_names:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame


def update_registry(row: dict[str, object]) -> None:
    path = RUN / "01_xgb_refit_run_registry.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        old = old.loc[~(old["task"].eq(row["task"]) & old["fold"].eq(row["fold"]))]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def run_fold(task: str, fold: str, data: pd.DataFrame, manifest: pd.DataFrame, feature_names: list[str]) -> None:
    started = time.perf_counter()
    out = RUN / "artifacts" / task / fold
    out.mkdir(parents=True, exist_ok=True)
    log_path = RUN / "logs" / f"{task}_{fold}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"task": task, "fold": fold, "status": "RUNNING", "started_at": datetime.now().isoformat(timespec="seconds")}
    update_registry(row)
    try:
        split = manifest.loc[manifest["fold"].eq(fold)]
        train_ids = set(split.loc[split["role"].eq("train"), "obs_id"].astype(str))
        test_ids = set(split.loc[split["role"].eq("test"), "obs_id"].astype(str))
        train = data.loc[data["obs_id"].astype(str).isin(train_ids)].copy()
        test = data.loc[data["obs_id"].astype(str).isin(test_ids)].copy()
        if len(train) != len(train_ids) or len(test) != len(test_ids) or train_ids & test_ids:
            raise RuntimeError("保存的obs_id无法完整、互斥地重建训练/测试输入")

        x_train_raw = train[feature_names].to_numpy(dtype=float)
        x_test_raw = test[feature_names].to_numpy(dtype=float)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x_train = imputer.fit_transform(x_train_raw).astype(np.float32)
        x_test = imputer.transform(x_test_raw).astype(np.float32)
        target = TASKS[task][0]
        y_train_original = train[target].to_numpy(dtype=float)
        y_true = test[target].to_numpy(dtype=float)
        use_log1p = bool(np.nanmin(y_train_original) >= 0)
        y_train = np.log1p(y_train_original) if use_log1p else y_train_original.copy()
        params = dict(json.loads(CONFIG.read_text(encoding="utf-8-sig"))["model_parameters"]["xgboost"])
        params.update({"random_state": SEED, "objective": "reg:squarederror"})
        model = XGBRegressor(**params)
        model.fit(x_train, y_train)
        raw = np.asarray(model.predict(x_test), dtype=float)
        prediction = np.maximum(np.expm1(raw) if use_log1p else raw, 0)

        model_path = out / "xgboost_refit_model.ubj"
        imputer_path = out / "median_imputer.joblib"
        model.save_model(model_path)
        joblib.dump(imputer, imputer_path)
        (out / "feature_order.json").write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame({"obs_id": train["obs_id"].astype(str)}).to_csv(out / "train_obs_ids.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"obs_id": test["obs_id"].astype(str)}).to_csv(out / "test_obs_ids.csv", index=False, encoding="utf-8-sig")
        pred_path = RUN / "predictions" / f"{task}_{fold}_refit_predictions.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"obs_id": test["obs_id"].astype(str), "y_true": y_true, "y_pred_refit": prediction}).to_csv(pred_path, index=False, encoding="utf-8-sig")
        metadata = {
            "asset_type": "CONTROLLED_REFIT_NOT_ORIGINAL_FROZEN_MODEL", "task": task, "fold": fold,
            "parameters": params, "random_seed": SEED, "feature_count": 77, "feature_order_sha256": list_hash(feature_names),
            "train_obs_count": len(train), "test_obs_count": len(test),
            "train_obs_order_sha256": list_hash(train["obs_id"].astype(str).tolist()),
            "test_obs_order_sha256": list_hash(test["obs_id"].astype(str).tolist()),
            "imputer": "SimpleImputer(strategy=median, keep_empty_features=True); fit on fold train only",
            "input_dtype": "float32", "target_transform": "log1p/expm1" if use_log1p else "identity", "postprocess": "maximum(prediction,0)",
            "environment": {"python": platform.python_version(), "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
            "input_hashes": {"main_table": sha256(MAIN), "feature_manifest": sha256(FEATURE_MANIFEST), "split_manifest": sha256(TASKS[task][2]), "config": sha256(CONFIG)},
            "output_hashes": {"model": sha256(model_path), "imputer": sha256(imputer_path), "prediction": sha256(pred_path)},
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        metadata_path = out / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        row.update({"status": "COMPLETED_PENDING_VALIDATION", "ended_at": metadata["completed_at"], "elapsed_seconds": time.perf_counter()-started,
                    "model_path": str(model_path.relative_to(ROOT)), "imputer_path": str(imputer_path.relative_to(ROOT)),
                    "prediction_path": str(pred_path.relative_to(ROOT)), "metadata_path": str(metadata_path.relative_to(ROOT)), "reason": ""})
        log_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        row.update({"status": "REFIT_FAILED", "ended_at": datetime.now().isoformat(timespec="seconds"), "elapsed_seconds": time.perf_counter()-started, "reason": repr(exc)})
        log_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        update_registry(row)
        raise
    update_registry(row)
    print(f"REFIT_COMPLETE {task} {fold} seconds={row['elapsed_seconds']:.3f}", flush=True)


def main() -> None:
    args = parse_args()
    if bool(args.fold) == bool(args.all_folds):
        raise SystemExit("必须且只能指定 --fold 或 --all-folds")
    feature_names = features()
    data = load_task_data(args.task, feature_names)
    manifest = split_manifest(args.task)
    for fold in FOLDS if args.all_folds else (args.fold,):
        run_fold(args.task, fold, data, manifest, feature_names)


if __name__ == "__main__":
    main()
