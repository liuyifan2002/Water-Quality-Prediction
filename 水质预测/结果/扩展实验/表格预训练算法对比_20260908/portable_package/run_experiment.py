from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_MAP = {
    "chla": "chla_ug_l",
    "secchi": "secchi_depth_m",
    "tn": "tn_mg_l",
    "tp": "tp_mg_l",
}
DEPTHS = ["surface", "middle", "bottom"]
MODELS = ["xgboost", "catboost", "tabpfn", "tabm"]

BASELINE_FEATURES = [
    "lon",
    "lat",
    "year_offset",
    "month_sin",
    "month_cos",
    "is_wet_season",
    "quarter",
    "day_of_year",
    "water_zone_code",
]
HISTORY_BASE_FEATURES = ["days_since_prev_sample"]
HISTORY_SUFFIXES = ["_lag1", "_lag2", "_roll3", "_delta1"]
PROCESS_BASE_FEATURES = [
    "t2m",
    "u10",
    "v10",
    "tcc",
    "wind_speed",
    "wind_speed_sq",
    "wind_dir_sin",
    "wind_dir_cos",
    "rain_3d",
    "rain_14d",
    "rain_month",
    "tidal_range",
    "ntl_avg_rad",
    "viirs_count",
    "t2m_x_rain14d",
    "wind_speed_x_rain14d",
    "rain14d_x_tidal_range",
    "t2m_x_tidal_range",
    "ssrd_sum_3d",
    "ssrd_mean_3d",
    "ssrd_sum_7d",
    "ssrd_mean_7d",
    "ssrd_sum_14d",
    "ssrd_mean_14d",
    "sst_mean_3d",
    "sst_max_3d",
    "sst_anom_3d",
    "sst_mean_7d",
    "sst_max_7d",
    "sst_anom_7d",
    "sst_mean_14d",
    "sst_max_14d",
    "sst_anom_14d",
    "sst_mean_7d_x_ssrd_mean_7d",
    "sst_anom_7d_x_ssrd_mean_7d",
    "rain14d_x_ssrd_mean_7d",
]
PROCESS_LAG_BASES = [
    "t2m",
    "tcc",
    "wind_speed",
    "rain_3d",
    "rain_14d",
    "rain_month",
    "tidal_range",
    "ntl_avg_rad",
]
REMOTE_FEATURES = [
    "landsat_hq_blue",
    "landsat_hq_green",
    "landsat_hq_red",
    "landsat_hq_nir",
    "landsat_hq_swir1",
    "landsat_hq_swir2",
    "landsat_hq_ndvi",
    "landsat_hq_ndwi",
    "landsat_hq_mndwi",
    "landsat_hq_ndti",
    "landsat_image_count",
    "landsat_clear_pixel_count",
    "landsat_strict_valid_gee",
    "landsat_hq",
    "landsat_available",
    "s2_hq_b2_mean",
    "s2_hq_b3_mean",
    "s2_hq_b4_mean",
    "s2_hq_b5_mean",
    "s2_hq_b6_mean",
    "s2_hq_b8_mean",
    "s2_hq_b8a_mean",
    "s2_hq_b11_mean",
    "s2_hq_b12_mean",
    "s2_hq_ndwi_mean",
    "s2_hq_mndwi_mean",
    "s2_hq_ndci_mean",
    "s2_hq_ndti_mean",
    "s2_has_images",
    "s2_image_count",
    "s2_strict_valid",
    "s2_valid_pixel_count",
    "s2_hq",
    "s2_available",
    "hq_satellite_available_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在固定数据、固定特征和固定划分下比较 XGBoost、CatBoost、TabPFN 与 TabM。"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/model_table.csv"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGET_MAP), default=sorted(TARGET_MAP))
    parser.add_argument("--depths", nargs="+", choices=DEPTHS, default=DEPTHS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="仅运行四个代表性任务，先检查环境和流程。")
    parser.add_argument("--resume", action="store_true", help="跳过输出中已经成功完成的模型任务。")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    tabpfn_params = config.get("model_parameters", {}).get("tabpfn", {})
    model_path = tabpfn_params.get("model_path")
    if model_path:
        resolved_model_path = Path(model_path)
        if not resolved_model_path.is_absolute():
            resolved_model_path = path.resolve().parent / resolved_model_path
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"找不到 TabPFN 本地权重文件：{resolved_model_path}")
        tabpfn_params["model_path"] = str(resolved_model_path)
    return config


def normalize_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        values = out[col].dropna().astype(str).str.strip().str.lower()
        if values.empty:
            continue
        if set(values.unique()).issubset({"true", "false", "0", "1"}):
            out[col] = (
                out[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
            )
    return out


def prepare_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    data = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    data = data.rename(columns={c: c.replace("\ufeff", "") for c in data.columns})
    data = normalize_bool_columns(data)
    data["water_zone_en"] = data["water_zone_en"].fillna("unknown")
    data["water_zone_code"] = pd.Categorical(data["water_zone_en"]).codes.astype("int16")

    numeric_columns = [
        c
        for c in data.columns
        if c not in {"obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "depth"}
    ]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def available(data: pd.DataFrame, features: list[str]) -> list[str]:
    return [name for name in features if name in data.columns]


def build_feature_sets(data: pd.DataFrame, target_column: str) -> dict[str, list[str]]:
    baseline = available(data, BASELINE_FEATURES)
    history = available(
        data,
        HISTORY_BASE_FEATURES + [f"{target_column}{suffix}" for suffix in HISTORY_SUFFIXES],
    )
    process = available(data, PROCESS_BASE_FEATURES)
    process_lags = available(
        data,
        [f"{base}{suffix}" for base in PROCESS_LAG_BASES for suffix in HISTORY_SUFFIXES],
    )
    remote = available(data, REMOTE_FEATURES)

    def unique(items: list[str]) -> list[str]:
        return list(dict.fromkeys(items))

    return {
        "baseline": baseline,
        "f1_history": unique(baseline + history + process + process_lags),
        "f2_remote": remote,
        "f3_history_remote": unique(baseline + history + process + process_lags + remote),
    }


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator == 0:
        return float("nan")
    return float(1 - np.sum((y_true - y_pred) ** 2) / denominator)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "nse": nse(y_true, y_pred),
    }


def target_transform(values: np.ndarray) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    values = np.asarray(values, dtype=float)
    if np.nanmin(values) >= 0:
        return np.log1p(values), np.expm1
    return values.copy(), lambda x: x


def fit_xgboost(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    seed: int,
    config: dict[str, Any],
) -> np.ndarray:
    from xgboost import XGBRegressor

    params = dict(config["model_parameters"]["xgboost"])
    params.update({"random_state": seed, "objective": "reg:squarederror"})
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def fit_catboost(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    seed: int,
    config: dict[str, Any],
) -> np.ndarray:
    from catboost import CatBoostRegressor

    params = dict(config["model_parameters"]["catboost"])
    params.update(
        {
            "random_seed": seed,
            "loss_function": "RMSE",
            "verbose": False,
            "allow_writing_files": False,
        }
    )
    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def fit_tabpfn(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    seed: int,
    config: dict[str, Any],
) -> np.ndarray:
    from tabpfn import TabPFNRegressor

    params = dict(config["model_parameters"].get("tabpfn", {}))
    signature = inspect.signature(TabPFNRegressor)
    if "random_state" in signature.parameters:
        params["random_state"] = seed
    if "device" in signature.parameters and "device" not in params:
        try:
            import torch

            params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            params["device"] = "cpu"
    params = {key: value for key, value in params.items() if key in signature.parameters}
    model = TabPFNRegressor(**params)
    model.fit(x_train.to_numpy(dtype=np.float32), y_train.astype(np.float32))
    return np.asarray(model.predict(x_test.to_numpy(dtype=np.float32)), dtype=float)


def fit_tabm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    seed: int,
    config: dict[str, Any],
) -> np.ndarray:
    import torch
    from tabm import TabM
    from torch.utils.data import DataLoader, TensorDataset

    params = dict(config["model_parameters"]["tabm"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inner_indices = np.arange(len(x_train))
    fit_indices, val_indices = train_test_split(
        inner_indices,
        test_size=float(params["validation_fraction"]),
        random_state=seed,
        shuffle=True,
    )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(imputer.fit_transform(x_train.iloc[fit_indices])).astype("float32")
    x_val = scaler.transform(imputer.transform(x_train.iloc[val_indices])).astype("float32")
    x_all = scaler.fit_transform(imputer.fit_transform(x_train)).astype("float32")
    x_out = scaler.transform(imputer.transform(x_test)).astype("float32")

    y_mean = float(np.mean(y_train[fit_indices]))
    y_std = float(np.std(y_train[fit_indices]))
    if y_std < 1e-8:
        y_std = 1.0
    y_fit = ((y_train[fit_indices] - y_mean) / y_std).astype("float32")
    y_val = ((y_train[val_indices] - y_mean) / y_std).astype("float32")

    make_kwargs: dict[str, Any] = {
        "n_num_features": x_fit.shape[1],
        "cat_cardinalities": [],
        "d_out": 1,
    }
    make_signature = inspect.signature(TabM.make)
    if "arch_type" in make_signature.parameters:
        make_kwargs["arch_type"] = params.get("arch_type", "tabm-mini")
    if "k" in make_signature.parameters:
        make_kwargs["k"] = int(params.get("k", 16))
    model = TabM.make(**make_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
        batch_size=int(params["batch_size"]),
        shuffle=True,
    )
    x_val_tensor = torch.from_numpy(x_val).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    remaining_patience = int(params["patience"])

    for epoch in range(int(params["max_epochs"])):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch).squeeze(-1)
            loss = ((prediction - y_batch[:, None]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prediction = model(x_val_tensor).squeeze(-1).mean(dim=1)
            val_loss = float(torch.mean((val_prediction - y_val_tensor) ** 2).item())
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = int(params["patience"])
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    # 按已选轮数重新使用全部外层训练集，避免丢弃内部验证样本。
    best_epochs = best_epoch + 1
    y_all_mean = float(np.mean(y_train))
    y_all_std = float(np.std(y_train))
    if y_all_std < 1e-8:
        y_all_std = 1.0
    y_all = ((y_train - y_all_mean) / y_all_std).astype("float32")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = TabM.make(**make_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    full_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_all), torch.from_numpy(y_all)),
        batch_size=int(params["batch_size"]),
        shuffle=True,
    )
    for _ in range(best_epochs):
        model.train()
        for x_batch, y_batch in full_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch).squeeze(-1)
            loss = ((prediction - y_batch[:, None]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    model.eval()
    predictions: list[np.ndarray] = []
    output_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_out)),
        batch_size=int(params["batch_size"]),
        shuffle=False,
    )
    with torch.no_grad():
        for (x_batch,) in output_loader:
            raw = model(x_batch.to(device)).squeeze(-1).mean(dim=1)
            predictions.append(raw.cpu().numpy())
    return np.concatenate(predictions) * y_all_std + y_all_mean


FIT_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "xgboost": fit_xgboost,
    "catboost": fit_catboost,
    "tabpfn": fit_tabpfn,
    "tabm": fit_tabm,
}


def safe_package_version(package_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return "not_installed"


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            name: safe_package_version(name)
            for name in ["numpy", "pandas", "scikit-learn", "xgboost", "catboost", "torch", "tabpfn", "tabm"]
        },
    }
    try:
        import torch

        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda_version"] = torch.version.cuda
        report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        report["nvidia_smi"] = completed.stdout.strip() or completed.stderr.strip()
    except Exception as exc:
        report["nvidia_smi"] = f"unavailable: {type(exc).__name__}: {exc}"
    return report


def write_checkpoint(metrics_rows: list[dict[str, Any]], predictions: list[pd.DataFrame], output_dir: Path) -> None:
    pd.DataFrame(metrics_rows).to_csv(output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            output_dir / "model_predictions.csv", index=False, encoding="utf-8-sig"
        )


def create_plots(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    successful = metrics_df[metrics_df["status"] == "ok"].copy()
    if successful.empty:
        return
    try:
        import matplotlib.pyplot as plt

        successful["task"] = successful["target"] + "-" + successful["depth"]
        pivot = successful.pivot_table(index="task", columns="model", values="r2", aggfunc="first")
        ordered_models = [name for name in MODELS if name in pivot.columns]
        pivot = pivot[ordered_models]
        ax = pivot.plot(kind="bar", figsize=(14, 6), width=0.82)
        ax.axhline(0.6, color="black", linestyle="--", linewidth=1, label="R2=0.60")
        ax.set_ylabel("R2")
        ax.set_xlabel("Target-depth task")
        ax.set_title("Model comparison under identical data and split")
        ax.legend(frameon=False, ncol=max(1, len(ordered_models)))
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "figure_model_r2_comparison.png", dpi=220, bbox_inches="tight")
        plt.close()

        if "xgboost" in pivot.columns:
            gain = pivot.subtract(pivot["xgboost"], axis=0).drop(columns=["xgboost"])
            if not gain.empty:
                ax = gain.plot(kind="bar", figsize=(14, 6), width=0.82)
                ax.axhline(0, color="black", linewidth=1)
                ax.set_ylabel("R2 change relative to XGBoost")
                ax.set_xlabel("Target-depth task")
                ax.set_title("Algorithm gain under identical data and split")
                ax.legend(frameon=False, ncol=max(1, len(gain.columns)))
                ax.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                plt.savefig(output_dir / "figure_r2_gain_vs_xgboost.png", dpi=220, bbox_inches="tight")
                plt.close()
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", encoding="utf-8"
        )


def create_summary(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    successful = metrics_df[metrics_df["status"] == "ok"].copy()
    lines = ["# 表格预训练算法对比结果", ""]
    if successful.empty:
        lines.extend(["没有成功完成的模型。请查看 `model_errors.log` 和 `environment.json`。", ""])
    else:
        best = successful.loc[successful.groupby(["target", "depth"])["r2"].idxmax()].copy()
        lines.extend(
            [
                "## 每个任务的最佳模型",
                "",
                "| 指标 | 水层 | 最佳模型 | R2 | RMSE | MAE |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in best.sort_values(["target", "depth"]).itertuples():
            lines.append(
                f"| {row.target} | {row.depth} | {row.model} | {row.r2:.4f} | {row.rmse:.4f} | {row.mae:.4f} |"
            )
        lines.extend(["", "## 相对 XGBoost 的变化", ""])
        if "xgboost" in set(successful["model"]):
            pivot = successful.pivot_table(index=["target", "depth"], columns="model", values="r2", aggfunc="first")
            lines.extend(["| 指标 | 水层 | 候选模型 | R2变化 |", "|---|---|---|---:|"])
            for (target, depth), row in pivot.iterrows():
                for model in [name for name in MODELS if name != "xgboost" and name in pivot.columns]:
                    if pd.notna(row.get(model)) and pd.notna(row.get("xgboost")):
                        lines.append(f"| {target} | {depth} | {model} | {row[model] - row['xgboost']:+.4f} |")
        else:
            lines.append("XGBoost 未成功运行，无法计算相对变化。")
        lines.extend(
            [
                "",
                "## 判定口径",
                "",
                "建议进入下一轮的最低条件：平均 R2 相对 XGBoost 提高不低于 0.02，至少 7 个任务提高，且单任务下降不超过 0.03。",
                "本文件仅汇总本次固定随机划分结果，不能替代年份留出或留站点检验。",
                "",
            ]
        )
    failed = metrics_df[metrics_df["status"] != "ok"]
    if not failed.empty:
        lines.extend(["## 未完成项目", "", "| 指标 | 水层 | 模型 | 状态 | 错误 |", "|---|---|---|---|---|"])
        for row in failed.itertuples():
            message = str(getattr(row, "error_message", "")).replace("|", "/").replace("\n", " ")[:180]
            lines.append(f"| {row.target} | {row.depth} | {row.model} | {row.status} | {message} |")
        lines.append("")
    (output_dir / "结果摘要.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["split"]["random_state"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment_report(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60, check=False
        )
        (args.output_dir / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    except Exception as exc:
        (args.output_dir / "pip_freeze.txt").write_text(str(exc), encoding="utf-8")

    data = prepare_dataset(args.dataset)
    task_policy = config["task_feature_sets"]
    reference = pd.read_csv(config["reference_metrics_file"])
    reference_lookup = {
        (row.target, row.depth): float(row.r2)
        for row in reference.itertuples()
    }

    selected_tasks = [(target, depth) for target in args.targets for depth in args.depths]
    if args.quick:
        quick_tasks = {("chla", "surface"), ("secchi", "middle"), ("tn", "middle"), ("tp", "middle")}
        selected_tasks = [task for task in selected_tasks if task in quick_tasks]

    existing_ok: set[tuple[str, str, str, int]] = set()
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    metrics_path = args.output_dir / "model_metrics.csv"
    predictions_path = args.output_dir / "model_predictions.csv"
    if args.resume and metrics_path.exists():
        old_metrics = pd.read_csv(metrics_path)
        metrics_rows = old_metrics.to_dict("records")
        existing_ok = {
            (str(row.target), str(row.depth), str(row.model), int(row.seed))
            for row in old_metrics[old_metrics["status"] == "ok"].itertuples()
        }
        if predictions_path.exists():
            prediction_frames.append(pd.read_csv(predictions_path))

    split_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    error_log = args.output_dir / "model_errors.log"
    total_jobs = len(selected_tasks) * len(args.models)
    completed_jobs = 0

    for target, depth in selected_tasks:
        target_column = TARGET_MAP[target]
        subset = data[(data["depth"] == depth) & data[target_column].notna()].copy()
        subset = subset.sort_values(["date", "station"]).reset_index(drop=True)
        train, test = train_test_split(
            subset,
            test_size=float(config["split"]["test_size"]),
            random_state=seed,
            shuffle=True,
        )
        train = train.sort_values(["date", "station"]).reset_index(drop=True)
        test = test.sort_values(["date", "station"]).reset_index(drop=True)
        feature_set_name = task_policy[f"{target}_{depth}"]
        features = build_feature_sets(data, target_column)[feature_set_name]
        feature_rows.append(
            {
                "target": target,
                "depth": depth,
                "feature_set": feature_set_name,
                "feature_count": len(features),
                "features_json": json.dumps(features, ensure_ascii=False),
            }
        )
        for frame, split_name in [(train, "train"), (test, "test")]:
            part = frame[["obs_id", "sample_id", "station", "date"]].copy()
            part.insert(0, "target", target)
            part.insert(1, "depth", depth)
            part["split"] = split_name
            part["seed"] = seed
            split_rows.append(part)

        x_train = train[features]
        x_test = test[features]
        y_true = test[target_column].to_numpy(dtype=float)
        y_train, inverse_transform = target_transform(train[target_column].to_numpy(dtype=float))

        for model_name in args.models:
            completed_jobs += 1
            key = (target, depth, model_name, seed)
            if key in existing_ok:
                print(f"[{completed_jobs}/{total_jobs}] 跳过已完成：{target}-{depth}-{model_name}", flush=True)
                continue
            print(f"[{completed_jobs}/{total_jobs}] 开始：{target}-{depth}-{model_name}", flush=True)
            started = time.perf_counter()
            row: dict[str, Any] = {
                "target": target,
                "target_column": target_column,
                "depth": depth,
                "feature_set": feature_set_name,
                "feature_count": len(features),
                "model": model_name,
                "seed": seed,
                "split_mode": "random_80_20",
                "train_rows": len(train),
                "test_rows": len(test),
                "same_day_water_quality_covariates": "excluded",
                "reference_r2": reference_lookup.get((target, depth), np.nan),
            }
            try:
                raw_prediction = FIT_FUNCTIONS[model_name](x_train, y_train, x_test, seed, config)
                prediction = np.maximum(inverse_transform(raw_prediction), 0)
                row.update(metrics(y_true, prediction))
                row["status"] = "ok"
                row["runtime_seconds"] = time.perf_counter() - started
                row["reproduction_abs_diff"] = (
                    abs(row["r2"] - row["reference_r2"]) if model_name == "xgboost" else np.nan
                )
                prediction_frame = test[
                    ["obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "year", "month"]
                ].copy()
                prediction_frame["target"] = target
                prediction_frame["depth"] = depth
                prediction_frame["model"] = model_name
                prediction_frame["seed"] = seed
                prediction_frame["observed"] = y_true
                prediction_frame["predicted"] = prediction
                prediction_frame["residual"] = y_true - prediction
                prediction_frames.append(prediction_frame)
                print(f"    完成：R2={row['r2']:.4f}，耗时={row['runtime_seconds']:.1f}s", flush=True)
            except Exception as exc:
                row.update(
                    {
                        "status": "error",
                        "runtime_seconds": time.perf_counter() - started,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                with error_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n{'=' * 80}\n{target}-{depth}-{model_name}\n{traceback.format_exc()}\n")
                print(f"    失败：{type(exc).__name__}: {exc}", flush=True)
            metrics_rows = [
                item
                for item in metrics_rows
                if (str(item.get("target")), str(item.get("depth")), str(item.get("model")), int(item.get("seed", seed)))
                != key
            ]
            metrics_rows.append(row)
            write_checkpoint(metrics_rows, prediction_frames, args.output_dir)

    pd.concat(split_rows, ignore_index=True).drop_duplicates().to_csv(
        args.output_dir / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(feature_rows).drop_duplicates().to_csv(
        args.output_dir / "feature_manifest.csv", index=False, encoding="utf-8-sig"
    )
    metrics_df = pd.DataFrame(metrics_rows)
    create_plots(metrics_df, args.output_dir)
    create_summary(metrics_df, args.output_dir)
    run_status = {
        "finished": True,
        "successful_jobs": int((metrics_df["status"] == "ok").sum()),
        "failed_jobs": int((metrics_df["status"] != "ok").sum()),
        "expected_jobs": total_jobs,
        "seed": seed,
        "quick": bool(args.quick),
    }
    (args.output_dir / "run_status.json").write_text(
        json.dumps(run_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("全部任务处理结束。请运行 pack_results.bat 打包 outputs 目录。", flush=True)


if __name__ == "__main__":
    main()
