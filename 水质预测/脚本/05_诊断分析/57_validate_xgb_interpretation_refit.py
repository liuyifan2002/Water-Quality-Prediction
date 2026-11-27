"""独立重载受控复现模型与imputer，并按事前固定阈值验证冻结预测一致性。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import importlib.util


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/xgb_interpretation_refit_v1"
REFIT_SCRIPT = ROOT / "脚本/05_诊断分析/56_refit_frozen_xgb_for_interpretation.py"
spec = importlib.util.spec_from_file_location("refit_protocol", REFIT_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("无法加载复现协议")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

EXACT = {"max_abs": 1e-6, "mean_abs": 1e-7, "metric": 1e-10}
APPROX = {"max_abs": 1e-4, "mean_abs": 1e-5, "metric": 1e-6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=P.TASKS)
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_predictions(task: str, fold: str) -> pd.DataFrame:
    if task == "secchi":
        frame = pd.read_csv(ROOT / "runs/tabiclv2_pilot/09_secchi_predictions.csv", dtype={"obs_id": "string"})
        return frame.loc[frame["validation"].eq("station") & frame["model"].astype(str).str.lower().eq("xgb") & frame["fold"].eq(fold), ["obs_id", "y_true", "y_pred"]].copy()
    run = {"chla_surface": "chla_family_v2", "tn_surface": "tn_family_v2", "tp_middle": "tp_family_v2"}[task]
    return pd.read_csv(ROOT / f"runs/{run}/predictions/{task}_xgboost_{fold}.csv", dtype={"obs_id": "string"})[["obs_id", "y_true", "y_pred"]]


def metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    return r2_score(y, pred), float(np.sqrt(mean_squared_error(y, pred))), mean_absolute_error(y, pred)


def validate_fold(task: str, fold: str, data: pd.DataFrame, features: list[str]) -> dict[str, object]:
    out = RUN / "artifacts" / task / fold
    model_path, imputer_path = out / "xgboost_refit_model.ubj", out / "median_imputer.joblib"
    feature_path, meta_path = out / "feature_order.json", out / "metadata.json"
    pred_path = RUN / "predictions" / f"{task}_{fold}_refit_predictions.csv"
    required = [model_path, imputer_path, feature_path, meta_path, pred_path, out / "train_obs_ids.csv", out / "test_obs_ids.csv"]
    if not all(p.exists() for p in required):
        return {"task": task, "fold": fold, "classification": "INSUFFICIENT_CONFIGURATION", "reason": "复现资产不完整"}
    saved_features = json.loads(feature_path.read_text(encoding="utf-8"))
    feature_match = saved_features == features
    test_ids = pd.read_csv(out / "test_obs_ids.csv", dtype={"obs_id": "string"})["obs_id"].astype(str).tolist()
    test = data.loc[data["obs_id"].astype(str).isin(set(test_ids))].copy()
    model = XGBRegressor()
    model.load_model(model_path)
    imputer = joblib.load(imputer_path)
    x_test = imputer.transform(test[features].to_numpy(dtype=float)).astype(np.float32)
    raw = np.asarray(model.predict(x_test), dtype=float)
    prediction = np.maximum(np.expm1(raw), 0)
    frozen = frozen_predictions(task, fold)
    obs_count_match = len(test) == len(test_ids) == len(frozen)
    obs_order_match = test["obs_id"].astype(str).tolist() == test_ids == frozen["obs_id"].astype(str).tolist()
    y_match = np.allclose(test[P.TASKS[task][0]].to_numpy(float), frozen["y_true"].to_numpy(float), rtol=0, atol=1e-12)
    diff = np.abs(prediction - frozen["y_pred"].to_numpy(float))
    max_abs, mean_abs = float(diff.max()), float(diff.mean())
    old_r2, old_rmse, old_mae = metrics(frozen["y_true"].to_numpy(float), frozen["y_pred"].to_numpy(float))
    new_r2, new_rmse, new_mae = metrics(frozen["y_true"].to_numpy(float), prediction)
    metric_diff = max(abs(new_r2-old_r2), abs(new_rmse-old_rmse), abs(new_mae-old_mae))
    integrity = sha256(model_path) == json.loads(meta_path.read_text(encoding="utf-8"))["output_hashes"]["model"] and sha256(imputer_path) == json.loads(meta_path.read_text(encoding="utf-8"))["output_hashes"]["imputer"]
    prerequisites = obs_count_match and obs_order_match and feature_match and y_match and integrity
    if prerequisites and max_abs <= EXACT["max_abs"] and mean_abs <= EXACT["mean_abs"] and metric_diff <= EXACT["metric"]:
        classification = "EXACT_REPRODUCTION"
    elif prerequisites and max_abs <= APPROX["max_abs"] and mean_abs <= APPROX["mean_abs"] and metric_diff <= APPROX["metric"]:
        classification = "APPROXIMATE_REPRODUCTION"
    else:
        classification = "REPRODUCTION_FAILED"
    return {
        "task": task, "fold": fold, "classification": classification, "n_predictions": len(prediction),
        "obs_count_match": obs_count_match, "obs_order_match": obs_order_match, "feature_order_match": feature_match,
        "y_true_match": y_match, "model_imputer_hash_match": integrity, "independent_reload_pass": True,
        "max_abs_prediction_error": max_abs, "mean_abs_prediction_error": mean_abs,
        "frozen_R2": old_r2, "refit_R2": new_r2, "R2_abs_diff": abs(new_r2-old_r2),
        "frozen_RMSE": old_rmse, "refit_RMSE": new_rmse, "RMSE_abs_diff": abs(new_rmse-old_rmse),
        "frozen_MAE": old_mae, "refit_MAE": new_mae, "MAE_abs_diff": abs(new_mae-old_mae),
        "exact_max_abs_threshold": EXACT["max_abs"], "exact_mean_abs_threshold": EXACT["mean_abs"], "exact_metric_threshold": EXACT["metric"],
        "reason": "" if classification == "EXACT_REPRODUCTION" else "未达到事前固定的严格阈值" if classification == "APPROXIMATE_REPRODUCTION" else "超过验收阈值或完整性前提失败",
    }


def refresh_asset_manifests(validation: pd.DataFrame) -> None:
    model_rows, environment_rows = [], []
    for task in P.TASKS:
        for fold in P.FOLDS:
            out = RUN / "artifacts" / task / fold
            meta_path = out / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            files = {
                "model": out / "xgboost_refit_model.ubj", "imputer": out / "median_imputer.joblib",
                "features": out / "feature_order.json", "train_obs": out / "train_obs_ids.csv",
                "test_obs": out / "test_obs_ids.csv", "metadata": meta_path,
                "prediction": RUN / "predictions" / f"{task}_{fold}_refit_predictions.csv",
                "log": RUN / "logs" / f"{task}_{fold}.log",
            }
            check = validation.loc[validation["task"].eq(task) & validation["fold"].eq(fold)]
            model_rows.append({
                "task": task, "fold": fold, "asset_type": "CONTROLLED_REFIT_NOT_ORIGINAL_FROZEN_MODEL",
                **{f"{name}_path": str(path.relative_to(ROOT)) for name, path in files.items()},
                **{f"{name}_sha256": sha256(path) if path.exists() else "MISSING" for name, path in files.items()},
                "all_assets_present": all(path.exists() for path in files.values()),
                "independent_reload_status": check.iloc[0]["classification"] if len(check) else "NOT_VALIDATED",
            })
            environment_rows.append({"task": task, "fold": fold, **meta["environment"],
                                     "environment_provenance": "retained isolated venv; no original per-fold lockfile",
                                     "parameters_json": json.dumps(meta["parameters"], sort_keys=True),
                                     "random_seed": meta["random_seed"], "input_dtype": meta["input_dtype"],
                                     "target_transform": meta["target_transform"], "postprocess": meta["postprocess"]})
    pd.DataFrame(model_rows).to_csv(RUN / "03_xgb_refit_model_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(environment_rows).to_csv(RUN / "04_xgb_refit_environment_manifest.csv", index=False, encoding="utf-8-sig")
    registry_path = RUN / "01_xgb_refit_run_registry.csv"
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        classes = validation[["task", "fold", "classification"]]
        registry = registry.drop(columns=["classification"], errors="ignore").merge(classes, on=["task", "fold"], how="left")
        registry["status"] = registry["classification"].fillna(registry["status"])
        registry.to_csv(registry_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    features = P.features()
    data = P.load_task_data(args.task, features)
    rows = [validate_fold(args.task, fold, data, features) for fold in P.FOLDS]
    path = RUN / "02_xgb_refit_prediction_validation.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        old = old.loc[~old["task"].eq(args.task)]
    result = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    refresh_asset_manifests(result)
    for row in rows:
        print(f"{row['task']} {row['fold']} {row['classification']} max_abs={row.get('max_abs_prediction_error','NA')}")
    if not all(row["classification"] == "EXACT_REPRODUCTION" for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
