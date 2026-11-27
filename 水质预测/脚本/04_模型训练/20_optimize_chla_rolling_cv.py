from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


DATASET = Path("数据/处理后数据/衍生版本/model_dataset_surface_2014_2024_enhanced.csv")
OUT_DIR = Path("结果/扩展实验/chla_optimization")
TARGET = "Chl_a"

F1_FEATURES = [
    "month", "season_sin", "season_cos",
    "t2m", "u10", "v10", "tcc", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "rain_3d", "rain_14d", "rain_month", "tidal_range", "ntl_avg_rad",
    "t2m_lag1", "t2m_lag2", "t2m_roll3",
    "tcc_lag1", "tcc_lag2", "tcc_roll3",
    "wind_speed_lag1", "wind_speed_lag2", "wind_speed_roll3",
    "rain_month_lag1", "rain_month_lag2", "rain_month_roll3",
    "rain_14d_lag1", "rain_14d_lag2", "rain_14d_roll3",
    "tidal_range_lag1", "tidal_range_lag2", "tidal_range_roll3",
    "ntl_avg_rad_lag1", "ntl_avg_rad_lag2", "ntl_avg_rad_roll3",
]
F2_FEATURES = [
    "blue", "green", "red", "nir", "swir1", "swir2",
    "ndvi", "ndwi", "mndwi", "ndti",
    "blue_green", "green_red", "red_green", "nir_red", "red_nir",
    "valid_count", "landsat_missing", "landsat_imputed", "landsat_no_data",
    "ndwi_lag1", "ndwi_lag2", "ndwi_roll3",
    "mndwi_lag1", "mndwi_lag2", "mndwi_roll3",
    "ndti_lag1", "ndti_lag2", "ndti_roll3",
    "valid_count_lag1", "valid_count_lag2", "valid_count_roll3",
]
AR_FEATURES = ["Chl_a_lag1", "Chl_a_lag2", "Chl_a_roll3", "Chl_a_delta1"]


def add_chla_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["station", "date"]).copy()
    grouped = out.groupby("station", sort=False)[TARGET]
    out["Chl_a_lag1"] = grouped.shift(1)
    out["Chl_a_lag2"] = grouped.shift(2)
    out["Chl_a_roll3"] = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    out["Chl_a_delta1"] = out["Chl_a_lag1"] - out["Chl_a_lag2"]
    return out


def feature_sets() -> dict[str, list[str]]:
    compact = [
        "month", "season_sin", "season_cos", "t2m", "tcc", "wind_speed",
        "rain_3d", "rain_14d", "rain_month", "tidal_range", "ntl_avg_rad",
        "green", "red", "nir", "swir1", "ndwi", "mndwi", "ndti",
        "blue_green", "green_red", "valid_count",
        "rain_month_lag1", "rain_month_roll3", "wind_speed_lag1", "tcc_lag1",
        "ndwi_lag1", "mndwi_lag1", "ndti_lag1",
    ]
    return {
        "F1_process": F1_FEATURES,
        "F2_remote": F2_FEATURES,
        "F3_fusion": F1_FEATURES + F2_FEATURES,
        "F3_AR": F1_FEATURES + F2_FEATURES + AR_FEATURES,
        "compact_AR": compact + AR_FEATURES,
        "history_only": AR_FEATURES + ["month", "season_sin", "season_cos"],
    }


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = np.nan if denom == 0 else 1 - np.sum((y_true - pred) ** 2) / denom
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "nse": float(nse),
    }


def transform_y(y: pd.Series, transform: str) -> tuple[np.ndarray, callable]:
    values = y.to_numpy()
    if transform == "log1p":
        return np.log1p(values), np.expm1
    if transform == "sqrt":
        return np.sqrt(np.maximum(values, 0)), lambda p: np.square(np.maximum(p, 0))
    return values, lambda p: p


def fill_xy(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fill = train[features].median(numeric_only=True).fillna(0)
    return train[features].fillna(fill), val[features].fillna(fill)


def clip_pred(pred: np.ndarray, train_y: pd.Series) -> np.ndarray:
    upper = max(float(train_y.quantile(0.99) * 1.25), float(train_y.max()))
    return np.clip(pred, 0, upper)


def month_climatology(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    by_month = train.groupby("month")[TARGET].median()
    return val["month"].map(by_month).fillna(train[TARGET].median()).to_numpy()


def persistence(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    return val["Chl_a_lag1"].fillna(val["Chl_a_roll3"]).fillna(train[TARGET].median()).to_numpy()


def model_factory(name: str):
    from xgboost import XGBRegressor

    if name == "xgb_shallow":
        return XGBRegressor(
            n_estimators=180, max_depth=1, learning_rate=0.04,
            min_child_weight=6, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=8.0, reg_alpha=0.1, objective="reg:squarederror",
            random_state=42, n_jobs=1,
        )
    if name == "xgb_regularized":
        return XGBRegressor(
            n_estimators=260, max_depth=2, learning_rate=0.03,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=10.0, reg_alpha=0.15, objective="reg:squarederror",
            random_state=42, n_jobs=1,
        )
    if name == "gbr_huber":
        return GradientBoostingRegressor(
            loss="huber", n_estimators=160, max_depth=2, learning_rate=0.035,
            min_samples_leaf=6, subsample=0.85, random_state=42,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=500, max_depth=4, min_samples_leaf=5,
            max_features=0.55, random_state=42, n_jobs=1,
        )
    if name == "ridge_robust":
        return make_pipeline(RobustScaler(), Ridge(alpha=8.0))
    raise ValueError(name)


def candidate_grid() -> list[dict]:
    rows = [
        {"candidate": "month_climatology", "feature_set": "baseline", "transform": "none"},
        {"candidate": "persistence_lag1", "feature_set": "baseline", "transform": "none"},
    ]
    specs = [
        ("xgb_shallow", "F1_process", ["log1p", "sqrt", "raw"]),
        ("xgb_shallow", "compact_AR", ["log1p", "raw"]),
        ("xgb_shallow", "F3_fusion", ["raw"]),
        ("xgb_regularized", "F3_fusion", ["raw"]),
        ("xgb_regularized", "F3_AR", ["raw"]),
        ("xgb_regularized", "history_only", ["raw"]),
        ("gbr_huber", "history_only", ["log1p", "sqrt", "raw"]),
        ("gbr_huber", "F2_remote", ["log1p", "sqrt", "raw"]),
        ("gbr_huber", "compact_AR", ["log1p", "raw"]),
        ("extra_trees", "compact_AR", ["log1p", "raw"]),
        ("extra_trees", "F3_fusion", ["raw"]),
        ("ridge_robust", "history_only", ["log1p"]),
        ("ridge_robust", "F1_process", ["log1p"]),
    ]
    for model, fs, transforms in specs:
        for transform in transforms:
            rows.append({"candidate": model, "feature_set": fs, "transform": transform})
    return rows


def predict_candidate(spec: dict, train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    if spec["candidate"] == "month_climatology":
        return month_climatology(train, val)
    if spec["candidate"] == "persistence_lag1":
        return persistence(train, val)
    features = feature_sets()[spec["feature_set"]]
    x_train, x_val = fill_xy(train, val, features)
    y_train, inv = transform_y(train[TARGET], spec["transform"])
    model = model_factory(spec["candidate"])
    model.fit(x_train, y_train)
    return clip_pred(inv(model.predict(x_val)), train[TARGET])


def score_spec_cv(group: pd.DataFrame, spec: dict) -> dict | None:
    rmses = []
    maes = []
    years_used = []
    for year in [2018, 2019, 2020, 2021, 2022]:
        train = group[group["year"].between(2014, year - 1)].dropna(subset=[TARGET]).copy()
        val = group[group["year"].eq(year)].dropna(subset=[TARGET]).copy()
        if len(train) < 36 or len(val) < 6:
            continue
        pred = predict_candidate(spec, train, val)
        rmses.append(np.sqrt(mean_squared_error(val[TARGET], pred)))
        maes.append(mean_absolute_error(val[TARGET], pred))
        years_used.append(year)
    if len(rmses) < 3:
        return None
    return {
        **spec,
        "cv_years": ",".join(map(str, years_used)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_median": float(np.median(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "cv_mae_mean": float(np.mean(maes)),
        "selection_score": float(np.mean(rmses) + 0.25 * np.std(rmses)),
    }


def optimize_station(station: str, group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = group[group["year"].between(2014, 2022)].dropna(subset=[TARGET]).copy()
    test = group[group["year"].between(2023, 2024)].dropna(subset=[TARGET]).copy()
    cv_rows = []
    for spec in candidate_grid():
        row = score_spec_cv(group, spec)
        if row is not None:
            cv_rows.append({"station": station, **row})
    cv = pd.DataFrame(cv_rows).sort_values(["selection_score", "cv_mae_mean"]).reset_index(drop=True)

    best = cv.iloc[0].to_dict()
    pred = predict_candidate(best, train_all, test)
    pred_df = test[["station", "date", "year", "month", TARGET]].copy()
    pred_df["pred"] = pred
    pred_df["abs_error"] = np.abs(pred_df[TARGET] - pred_df["pred"])
    pred_df["selected_candidate"] = best["candidate"]
    pred_df["selected_feature_set"] = best["feature_set"]
    pred_df["selected_transform"] = best["transform"]

    m = metrics(test[TARGET].to_numpy(), pred)
    selected = pd.DataFrame(
        [
            {
                "station": station,
                "candidate": best["candidate"],
                "feature_set": best["feature_set"],
                "transform": best["transform"],
                "cv_rmse_mean": best["cv_rmse_mean"],
                "cv_rmse_std": best["cv_rmse_std"],
                "selection_score": best["selection_score"],
                "test_rows": len(test),
                **m,
            }
        ]
    )
    return cv, selected, pred_df


def plot_outputs(pred: pd.DataFrame, station_metrics: pd.DataFrame) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    order = station_metrics.sort_values("rmse")["station"]
    ax.bar(order, station_metrics.set_index("station").loc[order, "rmse"], color="#2b8cbe")
    ax.set_ylabel("RMSE")
    ax.set_title("Rolling-CV selected Chl-a model by station")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "chla_rolling_cv_station_rmse.png")
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 12), dpi=160)
    axes = axes.ravel()
    for ax, station in zip(axes, sorted(pred["station"].unique())):
        sub = pred[pred["station"].eq(station)].sort_values("date")
        ax.plot(sub["date"], sub[TARGET], color="#222222", marker="o", lw=1.5, label="Observed")
        ax.plot(sub["date"], sub["pred"], color="#2b8cbe", marker="s", lw=1.25, label="Predicted")
        ax.set_title(station)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Rolling-CV optimized Chl-a predictions, 2023-2024", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / "chla_rolling_cv_station_predictions.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame, selected: pd.DataFrame, old_summary: pd.DataFrame) -> None:
    old = old_summary[old_summary["target"].eq("Chl_a")].iloc[0]
    new = summary.iloc[0]
    lines = [
        "# Chl-a 滚动交叉验证优化报告",
        "",
        "## 一、为什么改用滚动交叉验证",
        "",
        "上一轮只用 2021-2022 作为验证期选择模型，结果在 2023-2024 测试期出现退化。这说明 Chl-a 的年际变化较强，单一验证期不能稳定代表未来。因此本轮改为 2018-2022 多年份滚动验证，并用平均 RMSE 加 0.25 倍标准差作为选择分数，优先选择误差低且跨年份稳定的模型。",
        "",
        "## 二、总体表现",
        "",
        "| 模型版本 | RMSE | MAE | R2 / NSE |",
        "|---|---:|---:|---:|",
        f"| 原站点独立最优 F3 调参 XGBoost | {old['rmse']:.4f} | {old['mae']:.4f} | {old['r2']:.3f} |",
        f"| Chl-a 滚动 CV 优化 | {new['rmse']:.4f} | {new['mae']:.4f} | {new['r2']:.3f} |",
        "",
        "## 三、各站点选择结果",
        "",
        "| 站点 | 模型 | 特征组 | 变换 | CV RMSE | Test RMSE | Test MAE | Test R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.sort_values("station").iterrows():
        lines.append(
            f"| {row['station']} | {row['candidate']} | {row['feature_set']} | {row['transform']} | "
            f"{row['cv_rmse_mean']:.3f} | {row['rmse']:.3f} | {row['mae']:.3f} | {row['r2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 四、解释",
            "",
            "Chl-a 的优化重点应是提高跨年份稳定性，而不是追求某一验证期最优。若滚动 CV 结果仍低于原 F3 调参模型，应保留原模型作为当前正式版本，并把本轮结果作为诊断证据：Chl-a 的短期藻华峰值仅靠月尺度气象、降水、潮汐、夜光和 Landsat 特征仍难以稳定预测。",
            "",
            "下一步针对 Chl-a 最有价值的改进不是继续扩大模型网格，而是补充过程变量：采样日前 7-14 天太阳辐射、连续低风速、短期水温异常、降水后滞后窗口、以及 Sentinel-2 或更严格的近岸水体遥感质量控制。",
            "",
        ]
    )
    (OUT_DIR / "chla_rolling_cv_optimization_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET, parse_dates=["date"])
    df = add_chla_history(df)

    cv_frames = []
    selected_frames = []
    pred_frames = []
    for station, group in df.groupby("station"):
        cv, selected, pred = optimize_station(station, group)
        cv_frames.append(cv)
        selected_frames.append(selected)
        pred_frames.append(pred)

    cv_all = pd.concat(cv_frames, ignore_index=True)
    selected_all = pd.concat(selected_frames, ignore_index=True)
    pred_all = pd.concat(pred_frames, ignore_index=True)
    summary = pd.DataFrame(
        [
            {
                "target": TARGET,
                "model": "Chl_a_rolling_cv_selected",
                "test_rows": len(pred_all),
                **metrics(pred_all[TARGET].to_numpy(), pred_all["pred"].to_numpy()),
            }
        ]
    )

    old_summary = pd.read_csv("结果/扩展实验/station_specific/station_specific_best_model_overall_updated.csv", encoding="utf-8-sig")
    write_report(summary, selected_all, old_summary)
    plot_outputs(pred_all, selected_all)

    cv_all.to_csv(OUT_DIR / "chla_rolling_cv_candidate_scores.csv", index=False, encoding="utf-8-sig")
    selected_all.to_csv(OUT_DIR / "chla_rolling_cv_selected_by_station.csv", index=False, encoding="utf-8-sig")
    pred_all.to_csv(OUT_DIR / "chla_rolling_cv_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "chla_rolling_cv_summary.csv", index=False, encoding="utf-8-sig")

    print("Chl-a rolling CV summary:")
    print(summary.round(4).to_string(index=False))
    print("Selected by station:")
    print(selected_all[["station", "candidate", "feature_set", "transform", "cv_rmse_mean", "rmse", "mae", "r2"]].round(4).to_string(index=False))
    print(f"写出 {OUT_DIR / 'chla_rolling_cv_optimization_report.md'}")


if __name__ == "__main__":
    main()
