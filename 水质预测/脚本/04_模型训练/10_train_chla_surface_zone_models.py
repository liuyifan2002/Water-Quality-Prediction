from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATASET = Path("数据/处理后数据/衍生版本/model_table_3depth_2014_2024_zoneenhanced.csv")
DEFAULT_REPORT_DIR = Path("结果/扩展实验/chla_surface_zone_models")

FEATURE_SET_CHOICES = [
    "zone_baseline",
    "zone_history_process",
    "zone_landsat",
    "zone_history_landsat",
]

BASELINE_FEATURES = [
    "lon",
    "lat",
    "year_offset",
    "month_sin",
    "month_cos",
    "is_wet_season",
    "quarter",
    "day_of_year",
]

PROCESS_FEATURES = [
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
    "wind_speed_x_tidal_range",
    "wind_dir_sin_x_tidal_range",
    "wind_dir_cos_x_tidal_range",
    "wind_speed_x_sst_mean_3d",
    "wind_speed_x_ssrd_mean_3d",
]

LANDSAT_FEATURES = [
    "landsat_hq_blue_filled",
    "landsat_hq_green_filled",
    "landsat_hq_red_filled",
    "landsat_hq_nir_filled",
    "landsat_hq_swir1_filled",
    "landsat_hq_swir2_filled",
    "landsat_hq_ndvi_filled",
    "landsat_hq_ndwi_filled",
    "landsat_hq_mndwi_filled",
    "landsat_hq_ndti_filled",
    "landsat_image_count",
    "landsat_clear_pixel_count",
    "landsat_strict_valid_gee",
    "landsat_hq",
    "landsat_hq_signal_count",
    "landsat_hq_filled_signal_count",
    "landsat_hq_fill_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train surface Chl-a models separately for each water zone with enhanced history and Landsat features."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--feature-sets", nargs="*", choices=FEATURE_SET_CHOICES, default=FEATURE_SET_CHOICES)
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--test-start-year", type=int, default=2023)
    parser.add_argument("--test-end-year", type=int, default=2024)
    parser.add_argument("--val-start-year", type=int, default=2022)
    parser.add_argument("--min-train-rows", type=int, default=160)
    parser.add_argument("--min-test-rows", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--use-log1p-target", action="store_true", default=True)
    parser.add_argument("--tune", action="store_true", default=True)
    parser.add_argument("--tune-n-estimators", nargs="*", type=int, default=[300, 500])
    parser.add_argument("--tune-max-depth", nargs="*", type=int, default=[3, 4, 5])
    parser.add_argument("--tune-learning-rate", nargs="*", type=float, default=[0.03, 0.05])
    parser.add_argument("--tune-subsample", nargs="*", type=float, default=[0.8, 1.0])
    parser.add_argument("--tune-colsample-bytree", nargs="*", type=float, default=[0.8, 1.0])
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False})


def load_train_module():
    path = Path("脚本/04_模型训练/05_train_full_station_models.py")
    spec = importlib.util.spec_from_file_location("train_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def available_feature_list(data: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in data.columns]


def collect_history_features(data: pd.DataFrame) -> list[str]:
    target = "chla_ug_l"
    features = ["days_since_prev_sample"]
    for suffix in ["_lag1", "_lag2", "_lag3", "_lag7", "_roll3", "_roll5", "_delta1", "_delta7"]:
        col = f"{target}{suffix}"
        if col in data.columns:
            features.append(col)
    return features


def collect_process_lag_features(data: pd.DataFrame) -> list[str]:
    prefixes = [
        "t2m",
        "tcc",
        "wind_speed",
        "rain_3d",
        "rain_14d",
        "rain_month",
        "tidal_range",
        "ntl_avg_rad",
    ]
    suffixes = ["_lag1", "_lag2", "_lag3", "_roll3", "_roll5", "_delta1", "_delta7"]
    features: list[str] = []
    for prefix in prefixes:
        for suffix in suffixes:
            col = f"{prefix}{suffix}"
            if col in data.columns:
                features.append(col)
    return features


def build_feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    baseline = available_feature_list(data, BASELINE_FEATURES)
    history = available_feature_list(data, collect_history_features(data))
    process = available_feature_list(data, PROCESS_FEATURES)
    process_lags = available_feature_list(data, collect_process_lag_features(data))
    landsat = available_feature_list(data, LANDSAT_FEATURES)
    return {
        "zone_baseline": baseline,
        "zone_history_process": baseline + [c for c in history + process + process_lags if c not in baseline],
        "zone_landsat": landsat,
        "zone_history_landsat": baseline + [c for c in history + process + process_lags + landsat if c not in baseline],
    }


def transform_target(values: pd.Series, use_log1p: bool) -> tuple[np.ndarray, callable]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if use_log1p and np.nanmin(arr) >= 0:
        return np.log1p(arr), np.expm1
    return arr, lambda x: x


def tuning_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    combos: list[dict[str, object]] = []
    for n_estimators in args.tune_n_estimators:
        for max_depth in args.tune_max_depth:
            for learning_rate in args.tune_learning_rate:
                for subsample in args.tune_subsample:
                    for colsample_bytree in args.tune_colsample_bytree:
                        combos.append(
                            {
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "learning_rate": learning_rate,
                                "subsample": subsample,
                                "colsample_bytree": colsample_bytree,
                            }
                        )
    return combos


def select_model_params(train_mod, train: pd.DataFrame, features: list[str], args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    base = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    meta = {"tuned": False, "validation_rows": 0, "validation_r2": np.nan}
    if not args.tune:
        return base, meta

    inner_train = train[train["year"] < args.val_start_year].copy()
    inner_val = train[train["year"] >= args.val_start_year].copy()
    if len(inner_train) < args.min_train_rows or len(inner_val) < args.min_test_rows:
        return base, meta

    y_inner, inverse_transform = transform_target(inner_train["chla_ug_l"], args.use_log1p_target)
    y_val = inner_val["chla_ug_l"].to_numpy(dtype=float)
    model_args = argparse.Namespace(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    best_params = base.copy()
    best_score = -np.inf
    for candidate in tuning_grid(args):
        model = train_mod.make_model(model_args, candidate)
        model.fit(inner_train[features], y_inner)
        pred_val = inverse_transform(model.predict(inner_val[features]))
        pred_val = np.maximum(pred_val, 0)
        score = float(r2_score(y_val, pred_val))
        if score > best_score:
            best_score = score
            best_params = candidate.copy()
    return best_params, {"tuned": True, "validation_rows": len(inner_val), "validation_r2": best_score}


def fit_one_zone(
    train_mod,
    zone_df: pd.DataFrame,
    zone: str,
    feature_set_name: str,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    subset = zone_df.sort_values(["date", "station"]).reset_index(drop=True)
    train = subset[subset["year"] <= args.train_end_year].copy()
    test = subset[subset["year"].between(args.test_start_year, args.test_end_year)].copy()
    result: dict[str, object] = {
        "water_zone_en": zone,
        "feature_set": feature_set_name,
        "feature_count": len(features),
        "features_json": json.dumps(features, ensure_ascii=False),
        "stations": int(subset["station"].nunique()),
        "train_rows": len(train),
        "test_rows": len(test),
    }
    if not features:
        result["status"] = "skipped_no_features"
        return result, pd.DataFrame()
    if len(train) < args.min_train_rows or len(test) < args.min_test_rows:
        result["status"] = "skipped_insufficient_rows"
        return result, pd.DataFrame()

    params, tune_meta = select_model_params(train_mod, train, features, args)
    result.update(
        {
            "tuned": tune_meta["tuned"],
            "validation_rows": tune_meta["validation_rows"],
            "validation_r2": tune_meta["validation_r2"],
            "model_params_json": json.dumps(params, ensure_ascii=False),
        }
    )

    model_args = argparse.Namespace(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    model = train_mod.make_model(model_args, params)
    y_train, inverse_transform = transform_target(train["chla_ug_l"], args.use_log1p_target)
    model.fit(train[features], y_train)
    pred = inverse_transform(model.predict(test[features]))
    pred = np.maximum(pred, 0)
    y = test["chla_ug_l"].to_numpy(dtype=float)

    result.update(
        {
            "status": "ok",
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }
    )

    pred_df = test[["obs_id", "sample_id", "station", "water_zone_en", "date", "year", "month"]].copy()
    pred_df["feature_set"] = feature_set_name
    pred_df["observed"] = y
    pred_df["predicted"] = pred
    return result, pred_df


def write_summary(metrics: pd.DataFrame, report_dir: Path) -> None:
    ok = metrics[metrics["status"] == "ok"].copy()
    best = ok.sort_values(["water_zone_en", "r2"], ascending=[True, False]).drop_duplicates("water_zone_en")
    lines = [
        "# Chl-a表层分海区建模结果",
        "",
        "## 结果概览",
        "",
        f"- 成功模型数：{len(ok)}",
        f"- 水区数：{best['water_zone_en'].nunique()}",
        f"- 最优模型平均R2：{best['r2'].mean():.3f}",
        f"- 最优模型中位R2：{best['r2'].median():.3f}",
        "",
        "## 各水区最优结果",
        "",
    ]
    table = best[["water_zone_en", "feature_set", "stations", "train_rows", "test_rows", "rmse", "mae", "r2"]].copy()
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    body: list[str] = []
    for _, row in table.iterrows():
        vals: list[str] = []
        for col in table.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    lines.extend([header, sep, *body, ""])
    lines.extend(
        [
            "## 简要解释",
            "",
            "1. 分海区建模的作用，是把不同海域的水动力和营养盐背景拆开，减少一个统一模型同时解释全部机制时的混杂。",
            "2. 如果局地R2明显高于全站点表层Chl-a结果，说明空间异质性确实是当前主瓶颈之一。",
            "3. 若某些海区仍然偏低，则更可能是峰值事件、采样频率和高值样本稀缺共同造成的限制。",
            "",
        ]
    )
    (report_dir / "叶绿素表层分海区结果汇总.md").write_text("\n".join(lines), encoding="utf-8")


def plot_best_r2(metrics: pd.DataFrame, report_dir: Path) -> None:
    ok = metrics[metrics["status"] == "ok"].copy()
    best = ok.sort_values(["water_zone_en", "r2"], ascending=[True, False]).drop_duplicates("water_zone_en")
    best = best.sort_values("r2", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(best["water_zone_en"], best["r2"], color="#2a9d8f")
    ax.set_xlabel("R2")
    ax.set_title("Chl-a表层分海区最优R2")
    fig.savefig(report_dir / "fig_chla_surface_zone_best_r2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    train_mod = load_train_module()
    data = train_mod.prepare_dataset(args.dataset)
    data = data[(data["depth"] == "surface") & data["chla_ug_l"].notna()].copy()

    feature_sets = build_feature_sets(data)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for zone, zone_df in data.groupby("water_zone_en", sort=True):
        for feature_set_name in args.feature_sets:
            features = feature_sets[feature_set_name]
            manifest_rows.append(
                {
                    "water_zone_en": zone,
                    "feature_set": feature_set_name,
                    "feature_count": len(features),
                    "features_json": json.dumps(features, ensure_ascii=False),
                }
            )
            result, pred_df = fit_one_zone(train_mod, zone_df, zone, feature_set_name, features, args)
            metric_rows.append(result)
            if not pred_df.empty:
                prediction_frames.append(pred_df)

    metrics = pd.DataFrame(metric_rows)
    manifest = pd.DataFrame(manifest_rows).drop_duplicates()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    metrics.to_csv(args.report_dir / "chla_surface_zone_metrics.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(args.report_dir / "chla_surface_zone_feature_manifest.csv", index=False, encoding="utf-8-sig")
    if not predictions.empty:
        predictions.to_csv(args.report_dir / "chla_surface_zone_predictions.csv", index=False, encoding="utf-8-sig")
    write_summary(metrics, args.report_dir)
    plot_best_r2(metrics, args.report_dir)
    print(f"Wrote zone-model outputs to {args.report_dir}")


if __name__ == "__main__":
    main()
