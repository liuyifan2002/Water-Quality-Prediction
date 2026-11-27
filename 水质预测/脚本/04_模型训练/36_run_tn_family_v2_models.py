"""Run the TN_FAMILY_V2 unified station-holdout experiment.

This extension uses only the frozen V2 manifests. It does not create or
modify a split, and it never writes to formal mainline results.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "tn_family_v2"
PREDICTION_DIR = RUN / "predictions"
METRICS_DIR = RUN / "fold_metrics"
MAIN_TABLE = ROOT / "数据" / "处理后数据" / "正式主表" / "model_table_3depth_2014_2024.csv"
STRICT_MANIFEST = ROOT / "runs" / "tabiclv2_pilot" / "01_strict_feature_manifest.csv"
CANONICAL_MAP = RUN / "02_tn_family_v2_station_fold_map.csv"
SPLIT_VALIDATION = RUN / "06_split_validation.csv"
XGB_CONFIG = ROOT / "结果" / "扩展实验" / "候选模型稳健性验证_20260914" / "portable_package" / "config.json"
REGISTRY = RUN / "07_run_registry.csv"

SEED = 42
FEATURE_SET = "X_strict_77"
SPLIT_VERSION = "TN_FAMILY_V2_UNIFIED_STATION_SPLIT"
DEPTHS = ("surface", "middle", "bottom")
MODELS = ("xgboost", "tabiclv2")
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
TARGET_COLUMN = "tn_mg_l"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def load_features() -> list[str]:
    manifest = pd.read_csv(STRICT_MANIFEST)
    selected = manifest.loc[manifest["included_in_pilot"].astype(bool), "feature"].dropna().astype(str).tolist()
    if len(selected) != 77 or len(set(selected)) != 77:
        raise RuntimeError(f"冻结特征不是 77 个唯一特征: {len(selected)}")
    return selected


def load_main_data(features: list[str], include_zone: bool = False) -> pd.DataFrame:
    if not MAIN_TABLE.exists():
        raise FileNotFoundError(MAIN_TABLE)
    header = pd.read_csv(MAIN_TABLE, nrows=0).columns.tolist()
    required = ["obs_id", "sample_id", "station", "date", "depth", TARGET_COLUMN, *features]
    if include_zone and "water_zone_en" in header:
        required.append("water_zone_en")
    missing = [column for column in required if column not in header]
    if missing:
        raise RuntimeError(f"正式主表缺少字段: {missing}")
    data = pd.read_csv(
        MAIN_TABLE,
        usecols=list(dict.fromkeys(required)),
        dtype={"obs_id": "string", "sample_id": "string", "station": "string", "depth": "string"},
        low_memory=False,
    )
    data["obs_id"] = data["obs_id"].astype("string").str.strip()
    data["station_id"] = data["station"].astype("string").str.strip()
    data["depth"] = data["depth"].astype("string").str.strip().str.lower()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data.loc[~np.isfinite(data[TARGET_COLUMN].to_numpy(dtype=float)), TARGET_COLUMN] = np.nan
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return data


def load_canonical() -> pd.DataFrame:
    canonical = pd.read_csv(CANONICAL_MAP, dtype={"station_id": "string"})
    canonical["station_id"] = canonical["station_id"].astype("string").str.strip()
    required = {"station_id", "fold"}
    if not required.issubset(canonical.columns):
        raise RuntimeError(f"canonical map 缺少字段: {sorted(required - set(canonical.columns))}")
    if canonical["station_id"].duplicated().any() or canonical["fold"].isna().any():
        raise RuntimeError("canonical map station_id 不唯一或 fold 缺失")
    if sorted(canonical["fold"].unique().tolist()) != list(FOLDS):
        raise RuntimeError("canonical map 不是 fold_1 到 fold_5")
    return canonical


def manifest_path(depth: str) -> Path:
    return RUN / {
        "surface": "03_tn_surface_manifest.csv",
        "middle": "04_tn_middle_manifest.csv",
        "bottom": "05_tn_bottom_manifest.csv",
    }[depth]


def load_manifest(depth: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path(depth), dtype={"obs_id": "string", "station_id": "string"})
    manifest["obs_id"] = manifest["obs_id"].astype("string").str.strip()
    manifest["station_id"] = manifest["station_id"].astype("string").str.strip()
    required = {"task", "depth", "fold", "role", "obs_id", "station_id", "date", "target_value", "split_source"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"{depth} manifest 缺少字段: {sorted(required - set(manifest.columns))}")
    return manifest


def validate_split_and_build_balance(
    data: pd.DataFrame, canonical: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical_stations = set(canonical["station_id"].astype(str))
    fold_stations = {
        fold: set(canonical.loc[canonical["fold"].eq(fold), "station_id"].astype(str)) for fold in FOLDS
    }
    validation = pd.read_csv(SPLIT_VALIDATION)
    if len(validation) != 15 or not validation["status"].eq("PASS").all() or not validation["overlap"].eq(0).all():
        raise RuntimeError("06_split_validation.csv 未通过 15/15 PASS 或 overlap=0")
    balance_rows: list[dict[str, object]] = []
    reference_test_stations: dict[str, set[str]] = {}
    for depth in DEPTHS:
        manifest = load_manifest(depth)
        layer = data.loc[data["depth"].eq(depth) & data[TARGET_COLUMN].notna()].copy()
        if layer["obs_id"].duplicated().any():
            raise RuntimeError(f"{depth}: 正式主表有效目标 obs_id 重复")
        for fold in FOLDS:
            part = manifest.loc[manifest["fold"].eq(fold)]
            train = part.loc[part["role"].eq("train")]
            test = part.loc[part["role"].eq("test")]
            train_stations = set(train["station_id"].astype(str))
            test_stations = set(test["station_id"].astype(str))
            train_ids = set(train["obs_id"].astype(str))
            test_ids = set(test["obs_id"].astype(str))
            if train_stations & test_stations:
                raise RuntimeError(f"{depth}/{fold}: train/test station overlap")
            if train_ids & test_ids:
                raise RuntimeError(f"{depth}/{fold}: train/test obs_id overlap")
            if not test_stations.issubset(canonical_stations) or test_stations != fold_stations[fold]:
                raise RuntimeError(f"{depth}/{fold}: test station 与 canonical map 不一致")
            if fold in reference_test_stations and test_stations != reference_test_stations[fold]:
                raise RuntimeError(f"{depth}/{fold}: 三层 test station 不一致")
            reference_test_stations[fold] = test_stations
            if train["obs_id"].duplicated().any() or test["obs_id"].duplicated().any():
                raise RuntimeError(f"{depth}/{fold}: manifest obs_id 重复")
            if train["target_value"].isna().any() or test["target_value"].isna().any():
                raise RuntimeError(f"{depth}/{fold}: manifest target 缺失")
            train_data = layer.loc[layer["obs_id"].isin(train_ids)]
            test_data = layer.loc[layer["obs_id"].isin(test_ids)]
            if len(train_data) != len(train_ids) or len(test_data) != len(test_ids):
                raise RuntimeError(f"{depth}/{fold}: manifest obs_id 无法完整匹配正式主表")
            row = {
                "task": f"tn_{depth}",
                "depth": depth,
                "fold": fold,
                "train_rows": len(train_ids),
                "test_rows": len(test_ids),
                "train_stations": len(train_stations),
                "test_stations": len(test_stations),
                "train_target_mean": float(train_data[TARGET_COLUMN].mean()),
                "train_target_sd": float(train_data[TARGET_COLUMN].std(ddof=1)),
                "test_target_mean": float(test_data[TARGET_COLUMN].mean()),
                "test_target_sd": float(test_data[TARGET_COLUMN].std(ddof=1)),
                "n_features": len(features),
            }
            if "water_zone_en" in data.columns:
                for zone, count in train_data.groupby("water_zone_en", dropna=False)["station_id"].nunique().items():
                    label = "missing" if pd.isna(zone) else str(zone)
                    row[f"train_zone_{label}"] = int(count)
                for zone, count in test_data.groupby("water_zone_en", dropna=False)["station_id"].nunique().items():
                    label = "missing" if pd.isna(zone) else str(zone)
                    row[f"test_zone_{label}"] = int(count)
            balance_rows.append(row)
    return pd.DataFrame(balance_rows), validation


def check_cuda() -> dict[str, object]:
    try:
        import torch
        import tabicl  # noqa: F401

        cuda = bool(torch.cuda.is_available())
        return {
            "torch_version": torch.__version__,
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_available": cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else "",
            "tabicl_version": package_version("tabicl"),
        }
    except Exception as exc:
        return {
            "torch_version": "ERROR",
            "torch_cuda_version": "ERROR",
            "cuda_available": False,
            "gpu_name": "",
            "tabicl_version": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_preflight(features: list[str], balance: pd.DataFrame, validation: pd.DataFrame, cuda: dict[str, object]) -> None:
    balance.to_csv(RUN / "06_fold_balance.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# TN_FAMILY_V2 模型启动前检查",
        "",
        f"- 检查时间：`{now()}`",
        "- split_version：`TN_FAMILY_V2_UNIFIED_STATION_SPLIT`",
        f"- strict features：`{len(features)}`，预期为 77",
        f"- split validation：`{len(validation)}/15 PASS`",
        f"- canonical map：`{CANONICAL_MAP}`",
        f"- XGBoost 配置：`{XGB_CONFIG}`",
        "",
        "## CUDA / TabICLv2",
        "",
        "```text",
        json.dumps(cuda, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 检查结论",
        "",
        "- 77 个特征字段已在正式主表中确认可读取。",
        "- 三层 manifest 均与 canonical station→fold map 一致。",
        "- target 无缺失，train/test station 与 obs_id 无重叠。",
        "- fold balance 仅作描述记录，不据此调整分折。",
        "",
        "## Fold balance",
        "",
        "```text",
        balance.to_string(index=False),
        "```",
        "",
        "模型启动条件：以上检查全部通过，且 `cuda_available=True`、`tabicl` 可导入。",
    ]
    (RUN / "06_preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preflight() -> tuple[list[str], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    features = load_features()
    data = load_main_data(features, include_zone=True)
    canonical = load_canonical()
    balance, validation = validate_split_and_build_balance(data, canonical, features)
    cuda = check_cuda()
    write_preflight(features, balance, validation, cuda)
    if not bool(cuda.get("cuda_available")) or cuda.get("tabicl_version") in {"ERROR", "NOT_INSTALLED"}:
        raise RuntimeError(f"TabICLv2 CUDA preflight failed: {cuda}")
    print(f"PREFLIGHT_PASS features={len(features)} validation={len(validation)}/15 cuda=True gpu={cuda['gpu_name']}", flush=True)
    return features, data, canonical, cuda


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, prediction)),
        "RMSE": float(mean_squared_error(y_true, prediction) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, prediction)),
    }


def target_transform(y_train: np.ndarray) -> tuple[np.ndarray, callable]:
    if np.nanmin(y_train) >= 0:
        return np.log1p(y_train), np.expm1
    return y_train.copy(), lambda values: values


def fit_xgb(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    from xgboost import XGBRegressor

    if not XGB_CONFIG.exists():
        raise FileNotFoundError(XGB_CONFIG)
    config = json.loads(XGB_CONFIG.read_text(encoding="utf-8-sig"))
    params = dict(config["model_parameters"]["xgboost"])
    params.update({"random_state": SEED, "objective": "reg:squarederror"})
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_test), dtype=float)
    del model
    return prediction


def fit_tabicl(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    adapter_path = ROOT / "models" / "tabiclv2_adapter.py"
    spec = importlib.util.spec_from_file_location("tn_family_v2_tabiclv2_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"TabICLv2 adapter 不存在: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.TabICLv2Regressor(random_state=SEED)
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
    del model, module
    return prediction


def clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def prediction_file(task: str, model: str, fold: str) -> Path:
    return PREDICTION_DIR / f"{task}_{model}_{fold}.csv"


def metrics_file(task: str, model: str, fold: str) -> Path:
    return METRICS_DIR / f"{task}_{model}_{fold}.csv"


def registry_columns() -> list[str]:
    return [
        "task", "depth", "model", "fold", "status", "start_time", "end_time", "runtime",
        "prediction_file", "metrics_file", "R2", "RMSE", "MAE", "n_train", "n_test",
        "n_features", "feature_set", "split_version", "error",
    ]


def build_registry() -> pd.DataFrame:
    rows = []
    for depth in DEPTHS:
        for model in MODELS:
            for fold in FOLDS:
                task = f"tn_{depth}"
                rows.append(
                    {
                        "task": task,
                        "depth": depth,
                        "model": model,
                        "fold": fold,
                        "status": "PENDING",
                        "start_time": "",
                        "end_time": "",
                        "runtime": np.nan,
                        "prediction_file": str(prediction_file(task, model, fold)),
                        "metrics_file": str(metrics_file(task, model, fold)),
                        "R2": np.nan,
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "n_train": np.nan,
                        "n_test": np.nan,
                        "n_features": 77,
                        "feature_set": FEATURE_SET,
                        "split_version": SPLIT_VERSION,
                        "error": "",
                    }
                )
    return pd.DataFrame(rows, columns=registry_columns())


def artifacts_valid(row: pd.Series) -> bool:
    pred_path = Path(str(row["prediction_file"]))
    metric_path = Path(str(row["metrics_file"]))
    if not pred_path.exists() or not metric_path.exists():
        return False
    try:
        pred = pd.read_csv(pred_path, nrows=2)
        metric = pd.read_csv(metric_path)
        required_pred = {"obs_id", "station_id", "date", "depth", "fold", "y_true", "y_pred", "residual", "model", "feature_set", "split_version"}
        required_metric = {"task", "depth", "model", "fold", "R2", "RMSE", "MAE", "status"}
        return not metric.empty and required_pred.issubset(pred.columns) and required_metric.issubset(metric.columns) and metric.iloc[0]["status"] == "OK"
    except Exception:
        return False


def load_or_create_registry() -> pd.DataFrame:
    fresh = build_registry()
    if not REGISTRY.exists():
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        fresh.to_csv(REGISTRY, index=False, encoding="utf-8-sig")
        return fresh
    registry = pd.read_csv(REGISTRY)
    if set(registry_columns()) - set(registry.columns):
        raise RuntimeError("07_run_registry.csv 字段与本轮协议不一致，不自动覆盖")
    registry = registry[registry_columns()].copy()
    for index, row in registry.iterrows():
        if str(row["status"]) == "OK" and not artifacts_valid(row):
            registry.loc[index, "status"] = "PENDING"
        elif str(row["status"]) == "RUNNING":
            registry.loc[index, "status"] = "PENDING"
    registry.to_csv(REGISTRY, index=False, encoding="utf-8-sig")
    return registry


def save_registry(registry: pd.DataFrame) -> None:
    temp = REGISTRY.with_suffix(".tmp.csv")
    registry.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(REGISTRY)


def set_registry(registry: pd.DataFrame, task: str, model: str, fold: str, **values: object) -> None:
    mask = registry["task"].eq(task) & registry["model"].eq(model) & registry["fold"].eq(fold)
    if int(mask.sum()) != 1:
        raise RuntimeError(f"registry key 不唯一: {task}/{model}/{fold}")
    for key, value in values.items():
        registry.loc[mask, key] = value
    save_registry(registry)


def run_one(
    task: str,
    depth: str,
    model_name: str,
    fold: str,
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    features: list[str],
) -> dict[str, object]:
    started = time.perf_counter()
    split = manifest.loc[manifest["fold"].eq(fold)]
    train_ids = set(split.loc[split["role"].eq("train"), "obs_id"].astype(str))
    test_ids = set(split.loc[split["role"].eq("test"), "obs_id"].astype(str))
    train = data.loc[data["obs_id"].astype(str).isin(train_ids)].copy()
    test = data.loc[data["obs_id"].astype(str).isin(test_ids)].copy()
    if len(train) != len(train_ids) or len(test) != len(test_ids):
        raise RuntimeError(f"{task}/{model_name}/{fold}: obs_id 无法完整匹配")
    if set(train["station_id"].astype(str)) & set(test["station_id"].astype(str)):
        raise RuntimeError(f"{task}/{model_name}/{fold}: station overlap")

    x_train_raw = train[features].to_numpy(dtype=float)
    x_test_raw = test[features].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(x_train_raw).astype(np.float32)
    x_test = imputer.transform(x_test_raw).astype(np.float32)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_true = test[TARGET_COLUMN].to_numpy(dtype=float)
    y_train_transformed, inverse = target_transform(y_train)
    raw_prediction = fit_xgb(x_train, y_train_transformed, x_test) if model_name == "xgboost" else fit_tabicl(x_train, y_train_transformed, x_test)
    y_pred = np.maximum(inverse(raw_prediction), 0)
    if len(y_pred) != len(y_true) or not np.isfinite(y_pred).all():
        raise RuntimeError(f"{task}/{model_name}/{fold}: prediction shape 或数值异常")
    scores = metrics(y_true, y_pred)
    runtime = time.perf_counter() - started

    pred = pd.DataFrame(
        {
            "task": task,
            "obs_id": test["obs_id"].astype(str).to_numpy(),
            "station_id": test["station_id"].astype(str).to_numpy(),
            "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "depth": depth,
            "fold": fold,
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_pred - y_true,
            "model": model_name,
            "feature_set": FEATURE_SET,
            "split_version": SPLIT_VERSION,
        }
    )
    pred_path = prediction_file(task, model_name, fold)
    metric_path = metrics_file(task, model_name, fold)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    metric = pd.DataFrame(
        [
            {
                "task": task,
                "depth": depth,
                "model": model_name,
                "fold": fold,
                **scores,
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(features),
                "feature_set": FEATURE_SET,
                "split_version": SPLIT_VERSION,
                "runtime": runtime,
                "status": "OK",
            }
        ]
    )
    metric.to_csv(metric_path, index=False, encoding="utf-8-sig")
    return {**scores, "n_train": len(train), "n_test": len(test), "runtime": runtime, "prediction_file": str(pred_path), "metrics_file": str(metric_path)}


def run_models(features: list[str], data: pd.DataFrame) -> None:
    registry = load_or_create_registry()
    manifests = {depth: load_manifest(depth) for depth in DEPTHS}
    for depth in DEPTHS:
        task = f"tn_{depth}"
        task_data = data.loc[data["depth"].eq(depth) & data[TARGET_COLUMN].notna()].copy()
        for model_name in MODELS:
            for fold in FOLDS:
                row = registry.loc[
                    registry["task"].eq(task) & registry["model"].eq(model_name) & registry["fold"].eq(fold)
                ].iloc[0]
                if str(row["status"]) == "OK" and artifacts_valid(row):
                    print(f"SKIP {task} {model_name} {fold}: already OK", flush=True)
                    continue
                start = now()
                started = time.perf_counter()
                set_registry(registry, task, model_name, fold, status="RUNNING", start_time=start, end_time="", error="")
                print(f"START {task} {model_name} {fold}", flush=True)
                try:
                    result = run_one(task, depth, model_name, fold, task_data, manifests[depth], features)
                    set_registry(
                        registry,
                        task,
                        model_name,
                        fold,
                        status="OK",
                        end_time=now(),
                        runtime=result["runtime"],
                        prediction_file=result["prediction_file"],
                        metrics_file=result["metrics_file"],
                        R2=result["R2"],
                        RMSE=result["RMSE"],
                        MAE=result["MAE"],
                        n_train=result["n_train"],
                        n_test=result["n_test"],
                        error="",
                    )
                    print(f"OK {task} {model_name} {fold} R2={result['R2']:.4f} runtime={result['runtime']:.1f}s", flush=True)
                except Exception as exc:
                    set_registry(registry, task, model_name, fold, status="FAILED", end_time=now(), runtime=time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}")
                    print(f"FAILED {task} {model_name} {fold}: {type(exc).__name__}: {exc}", flush=True)
                    raise
                finally:
                    clear_cuda()
    print("TN_FAMILY_V2 model runs complete", flush=True)


def main() -> int:
    args = parse_args()
    if not args.preflight_only and not args.run:
        args.preflight_only = True
    features, data, _, _ = run_preflight()
    if args.preflight_only:
        return 0
    run_models(features, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
