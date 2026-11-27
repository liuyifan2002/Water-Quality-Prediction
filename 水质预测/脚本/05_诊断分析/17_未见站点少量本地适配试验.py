"""Leakage-free zero-shot and few-shot pilot for unseen monitoring stations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "结果" / "扩展实验" / "候选模型稳健性验证_20260914" / "portable_package"
DEFAULT_OUTPUT = ROOT / "结果" / "扩展实验" / "论文主线审计_20260916" / "少量本地适配试验"
TASKS = ["chla_surface", "secchi_surface", "tn_bottom"]
LEVELS = [0, 1, 3, 6, 12, 24, "all_pretest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a chronological few-shot station-adaptation pilot.")
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--n-stations", type=int, default=20)
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--test-start-year", type=int, default=2023)
    parser.add_argument("--test-end-year", type=int, default=2024)
    parser.add_argument("--min-local-history", type=int, default=24)
    parser.add_argument("--min-test-rows", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="只根据已完成的试验明细重建汇总表，不重新训练模型。",
    )
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_runner_for_fewshot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_parts(task: str) -> tuple[str, str]:
    return task.rsplit("_", maxsplit=1)


def select_stations(
    data: pd.DataFrame,
    tasks: list[str],
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    min_history: int,
    min_test_rows: int,
    n_stations: int,
    random_state: int,
    target_map: dict[str, str],
) -> pd.DataFrame:
    eligible_sets: list[set[str]] = []
    station_zone = data[["station", "water_zone_en"]].drop_duplicates().set_index("station")["water_zone_en"]
    for task in tasks:
        target, depth = task_parts(task)
        subset = data.loc[data["depth"].eq(depth) & data[target_map[target]].notna()].copy()
        counts = subset.groupby("station").agg(
            history=("year", lambda year: int((year <= train_end_year).sum())),
            test=("year", lambda year: int(year.between(test_start_year, test_end_year).sum())),
        )
        eligible_sets.append(set(counts.index[(counts["history"] >= min_history) & (counts["test"] >= min_test_rows)]))
    common = sorted(set.intersection(*eligible_sets))
    frame = pd.DataFrame({"station": common})
    frame["water_zone_en"] = frame["station"].map(station_zone)
    rng = np.random.default_rng(random_state)
    selected: list[str] = []
    for _, group in frame.groupby("water_zone_en", sort=True):
        values = group["station"].to_numpy(dtype=str).copy()
        rng.shuffle(values)
        selected.extend(values[: min(2, len(values))].tolist())
    if len(selected) < n_stations:
        remaining = np.array(sorted(set(common) - set(selected)), dtype=str)
        rng.shuffle(remaining)
        selected.extend(remaining[: n_stations - len(selected)].tolist())
    selected = selected[:n_stations]
    out = frame.loc[frame["station"].isin(selected)].copy()
    out["selection_rule"] = "每水区最多2站，随后按固定种子补足"
    return out.sort_values(["water_zone_en", "station"]).reset_index(drop=True)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed station-level metrics without touching model outputs."""
    summary = metrics.groupby(["task", "adaptation_level"], dropna=False).agg(
        station_count=("heldout_station", "nunique"),
        r2_mean=("r2", "mean"),
        r2_median=("r2", "median"),
        r2_std=("r2", "std"),
        r2_q25=("r2", lambda value: value.quantile(0.25)),
        r2_q75=("r2", lambda value: value.quantile(0.75)),
        rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
    ).reset_index()
    # CSV round-trips may read the numeric level 0 as an integer.  Compare the
    # display values as strings so the zero-shot reference is always retained.
    zero = summary.loc[summary["adaptation_level"].astype(str).eq("0"), ["task", "r2_median"]].rename(
        columns={"r2_median": "zero_shot_r2_median"}
    )
    summary = summary.merge(zero, on="task", how="left")
    summary["median_r2_gain_vs_zero"] = summary["r2_median"] - summary["zero_shot_r2_median"]
    return summary


def local_rows(history: pd.DataFrame, level: int | str) -> pd.DataFrame:
    if level == "all_pretest":
        return history
    return history.iloc[: int(level)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_existing:
        metrics_path = args.output_dir / "少量本地适配试验结果.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"未找到已完成试验明细：{metrics_path}")
        summary = build_summary(pd.read_csv(metrics_path))
        summary.to_csv(args.output_dir / "少量本地适配汇总.csv", index=False, encoding="utf-8-sig")
        print(summary.to_string(index=False))
        return
    runner = load_module(args.package / "run_experiment.py")
    config = runner.load_config(args.package / "config.json")
    validation = load_json(args.package / "validation_config.json")
    data = runner.prepare_dataset(args.package / "data" / "model_table.csv")
    stations = select_stations(
        data,
        args.tasks,
        args.train_end_year,
        args.test_start_year,
        args.test_end_year,
        args.min_local_history,
        args.min_test_rows,
        args.n_stations,
        args.random_state,
        runner.TARGET_MAP,
    )
    stations.to_csv(args.output_dir / "试验站点清单.csv", index=False, encoding="utf-8-sig")

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    station_set = set(stations["station"])
    for task in args.tasks:
        target, depth = task_parts(task)
        target_column = runner.TARGET_MAP[target]
        feature_set = validation["task_feature_sets"][task]
        features = list(runner.build_feature_sets(data, target_column)[feature_set])
        features = [name for name in features if not name.startswith(f"{target_column}_")]
        subset = data.loc[data["depth"].eq(depth) & data[target_column].notna()].copy()
        for station in stations["station"]:
            held = subset.loc[subset["station"].eq(station)].sort_values("date").reset_index(drop=True)
            history = held.loc[held["year"] <= args.train_end_year].copy().reset_index(drop=True)
            test = held.loc[held["year"].between(args.test_start_year, args.test_end_year)].copy().reset_index(drop=True)
            external = subset.loc[subset["station"].ne(station) & subset["year"].le(args.train_end_year)].copy()
            if len(history) < args.min_local_history or len(test) < args.min_test_rows:
                continue
            for level in LEVELS:
                local = local_rows(history, level)
                train = pd.concat([external, local], ignore_index=True)
                y_fit, inverse = runner.target_transform(train[target_column].to_numpy(dtype=float))
                prediction = runner.FIT_FUNCTIONS["xgboost"](
                    train[features], y_fit, test[features], args.random_state, config
                )
                prediction = np.maximum(inverse(np.asarray(prediction, dtype=float)), 0.0)
                metric_rows.append(
                    {
                        "task": task,
                        "target": target,
                        "depth": depth,
                        "model": "xgboost",
                        "heldout_station": station,
                        "region": stations.loc[stations["station"].eq(station), "water_zone_en"].iloc[0],
                        "adaptation_level": level,
                        "N_local": len(local),
                        "local_last_date": local["date"].max().date().isoformat() if not local.empty else "",
                        "test_start_date": test["date"].min().date().isoformat(),
                        "test_end_date": test["date"].max().date().isoformat(),
                        "N_test": len(test),
                        "external_train_rows": len(external),
                        "train_rows": len(train),
                        "feature_set": feature_set,
                        "feature_count": len(features),
                        "target_history_features": "excluded",
                        "same_day_water_quality_covariates": "excluded",
                        **metric_row(test[target_column].to_numpy(dtype=float), prediction),
                    }
                )
                output = test[["obs_id", "station", "water_zone_en", "date", "year", "month"]].copy()
                output["task"] = task
                output["adaptation_level"] = level
                output["N_local"] = len(local)
                output["observed"] = test[target_column].to_numpy(dtype=float)
                output["predicted"] = prediction
                prediction_rows.append(output)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "少量本地适配试验结果.csv", index=False, encoding="utf-8-sig")
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            args.output_dir / "少量本地适配预测.csv", index=False, encoding="utf-8-sig"
        )
    summary = build_summary(metrics)
    summary.to_csv(args.output_dir / "少量本地适配汇总.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "tasks": args.tasks,
        "levels": LEVELS,
        "station_count": int(len(stations)),
        "train_period": f"2014-{args.train_end_year}",
        "test_period": f"{args.test_start_year}-{args.test_end_year}",
        "target_history_features": "excluded",
        "same_day_water_quality_covariates": "excluded",
        "chronology": "目标站点仅使用测试期之前最早的本地观测；测试为2023-2024。",
    }
    (args.output_dir / "试验元数据.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
