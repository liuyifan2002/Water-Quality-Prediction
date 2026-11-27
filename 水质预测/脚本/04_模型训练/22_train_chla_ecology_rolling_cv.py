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


DATASET = Path("数据/处理后数据/衍生版本/model_dataset_surface_2014_2024_chla_ecology.csv")
OUT_DIR = Path("结果/扩展实验/chla_optimization")
TARGET = "Chl_a"

BASE_PROCESS = [
    "month", "season_sin", "season_cos", "t2m", "u10", "v10", "tcc", "wind_speed",
    "rain_3d", "rain_14d", "rain_month", "tidal_range", "ntl_avg_rad",
    "t2m_lag1", "tcc_lag1", "wind_speed_lag1", "rain_month_lag1", "rain_14d_lag1",
    "tidal_range_lag1", "ntl_avg_rad_lag1",
]
REMOTE = [
    "blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "ndwi", "mndwi", "ndti",
    "blue_green", "green_red", "red_green", "nir_red", "red_nir",
    "valid_count", "landsat_missing", "landsat_imputed", "landsat_no_data",
    "landsat_strict_valid", "landsat_moderate_valid", "landsat_low_quality", "landsat_valid_count_log1p",
]
ECO = [
    "t2m_mean_7d", "t2m_max_7d", "t2m_anom_mean_7d",
    "wind_mean_7d", "wind_max_7d", "low_wind_hours_7d", "low_wind_days_7d",
    "tcc_mean_7d", "solar_proxy_mean_7d", "sunny_proxy_days_7d",
    "rain_sum_7d", "rain_max_7d", "heavy_rain_days_7d",
    "t2m_mean_14d", "t2m_max_14d", "t2m_anom_mean_14d",
    "wind_mean_14d", "wind_max_14d", "low_wind_hours_14d", "low_wind_days_14d",
    "tcc_mean_14d", "solar_proxy_mean_14d", "sunny_proxy_days_14d",
    "rain_sum_14d", "rain_max_14d", "heavy_rain_days_14d",
    "consecutive_low_wind_days",
]
AR = ["Chl_a_lag1", "Chl_a_lag2", "Chl_a_roll3", "Chl_a_delta1"]


def add_chla_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["station", "date"]).copy()
    g = out.groupby("station", sort=False)[TARGET]
    out["Chl_a_lag1"] = g.shift(1)
    out["Chl_a_lag2"] = g.shift(2)
    out["Chl_a_roll3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    out["Chl_a_delta1"] = out["Chl_a_lag1"] - out["Chl_a_lag2"]
    return out


def feature_sets() -> dict[str, list[str]]:
    compact_remote = ["green", "red", "nir", "swir1", "ndwi", "mndwi", "ndti", "green_red", "valid_count",
                      "landsat_low_quality", "landsat_valid_count_log1p"]
    compact_eco = [
        "t2m_mean_7d", "t2m_anom_mean_7d", "wind_mean_7d", "low_wind_days_7d",
        "solar_proxy_mean_7d", "rain_sum_7d", "rain_max_7d",
        "t2m_mean_14d", "wind_mean_14d", "low_wind_days_14d",
        "solar_proxy_mean_14d", "rain_sum_14d", "consecutive_low_wind_days",
    ]
    return {
        "history_only": AR + ["month", "season_sin", "season_cos"],
        "F3_previous": BASE_PROCESS + REMOTE,
        "eco_process": BASE_PROCESS + ECO,
        "eco_remote": ECO + REMOTE,
        "eco_fusion": BASE_PROCESS + ECO + REMOTE,
        "eco_fusion_AR": BASE_PROCESS + ECO + REMOTE + AR,
        "compact_eco_AR": ["month", "season_sin", "season_cos"] + compact_eco + compact_remote + AR,
    }


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    denom = np.sum((y - np.mean(y)) ** 2)
    nse = np.nan if denom == 0 else 1 - np.sum((y - pred) ** 2) / denom
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "nse": float(nse),
    }


def transform_y(y: pd.Series, transform: str) -> tuple[np.ndarray, callable]:
    values = y.to_numpy()
    if transform == "log1p":
        return np.log1p(values), np.expm1
    if transform == "sqrt":
        return np.sqrt(np.maximum(values, 0)), lambda p: np.square(np.maximum(p, 0))
    return values, lambda p: p


def xy(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fill = train[features].median(numeric_only=True).fillna(0)
    return train[features].fillna(fill), test[features].fillna(fill)


def clip_pred(pred: np.ndarray, train_y: pd.Series) -> np.ndarray:
    upper = max(float(train_y.quantile(0.99) * 1.25), float(train_y.max()))
    return np.clip(pred, 0, upper)


def month_climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    by_month = train.groupby("month")[TARGET].median()
    return test["month"].map(by_month).fillna(train[TARGET].median()).to_numpy()


def make_model(name: str):
    from xgboost import XGBRegressor

    if name == "xgb_shallow":
        return XGBRegressor(
            n_estimators=180, max_depth=1, learning_rate=0.04, min_child_weight=6,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=8.0, reg_alpha=0.1,
            objective="reg:squarederror", random_state=42, n_jobs=1,
        )
    if name == "xgb_regularized":
        return XGBRegressor(
            n_estimators=260, max_depth=2, learning_rate=0.03, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, reg_alpha=0.15,
            objective="reg:squarederror", random_state=42, n_jobs=1,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=500, max_depth=4, min_samples_leaf=5,
            max_features=0.55, random_state=42, n_jobs=1,
        )
    if name == "gbr_huber":
        return GradientBoostingRegressor(
            loss="huber", n_estimators=160, max_depth=2, learning_rate=0.035,
            min_samples_leaf=6, subsample=0.85, random_state=42,
        )
    if name == "ridge_robust":
        return make_pipeline(RobustScaler(), Ridge(alpha=8.0))
    raise ValueError(name)


def candidates() -> list[dict]:
    rows = [{"candidate": "month_climatology", "feature_set": "baseline", "transform": "none"}]
    specs = [
        ("ridge_robust", "history_only", ["log1p"]),
        ("gbr_huber", "history_only", ["log1p", "raw"]),
        ("xgb_regularized", "history_only", ["raw"]),
        ("xgb_shallow", "eco_process", ["log1p", "sqrt", "raw"]),
        ("xgb_regularized", "eco_fusion", ["raw", "log1p"]),
        ("xgb_regularized", "eco_fusion_AR", ["raw", "log1p"]),
        ("extra_trees", "eco_fusion", ["raw", "log1p"]),
        ("extra_trees", "eco_fusion_AR", ["raw", "log1p"]),
        ("extra_trees", "compact_eco_AR", ["raw", "log1p"]),
        ("gbr_huber", "eco_remote", ["log1p", "sqrt", "raw"]),
        ("gbr_huber", "compact_eco_AR", ["log1p", "raw"]),
    ]
    for model, fs, transforms in specs:
        for transform in transforms:
            rows.append({"candidate": model, "feature_set": fs, "transform": transform})
    return rows


def predict(spec: dict, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if spec["candidate"] == "month_climatology":
        return month_climatology(train, test)
    features = feature_sets()[spec["feature_set"]]
    x_train, x_test = xy(train, test, features)
    y_train, inv = transform_y(train[TARGET], spec["transform"])
    model = make_model(spec["candidate"])
    model.fit(x_train, y_train)
    return clip_pred(inv(model.predict(x_test)), train[TARGET])


def score_cv(group: pd.DataFrame, spec: dict) -> dict | None:
    rmses, maes, years = [], [], []
    for year in [2018, 2019, 2020, 2021, 2022]:
        train = group[group["year"].between(2014, year - 1)].dropna(subset=[TARGET]).copy()
        val = group[group["year"].eq(year)].dropna(subset=[TARGET]).copy()
        if len(train) < 36 or len(val) < 6:
            continue
        pred = predict(spec, train, val)
        rmses.append(np.sqrt(mean_squared_error(val[TARGET], pred)))
        maes.append(mean_absolute_error(val[TARGET], pred))
        years.append(year)
    if len(rmses) < 3:
        return None
    return {
        **spec,
        "cv_years": ",".join(map(str, years)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "cv_mae_mean": float(np.mean(maes)),
        "selection_score": float(np.mean(rmses) + 0.25 * np.std(rmses)),
    }


def optimize_station(station: str, group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv_rows = []
    for spec in candidates():
        row = score_cv(group, spec)
        if row is not None:
            cv_rows.append({"station": station, **row})
    cv = pd.DataFrame(cv_rows).sort_values(["selection_score", "cv_mae_mean"]).reset_index(drop=True)
    best = cv.iloc[0].to_dict()

    train = group[group["year"].between(2014, 2022)].dropna(subset=[TARGET]).copy()
    test = group[group["year"].between(2023, 2024)].dropna(subset=[TARGET]).copy()
    pred = predict(best, train, test)
    m = metrics(test[TARGET].to_numpy(), pred)

    selected = pd.DataFrame([{**best, "test_rows": len(test), **m}])
    pred_df = test[["station", "date", "year", "month", TARGET]].copy()
    pred_df["pred"] = pred
    pred_df["abs_error"] = np.abs(pred_df[TARGET] - pred_df["pred"])
    pred_df["selected_candidate"] = best["candidate"]
    pred_df["selected_feature_set"] = best["feature_set"]
    pred_df["selected_transform"] = best["transform"]
    return cv, selected, pred_df


def plot(pred: pd.DataFrame, selected: pd.DataFrame) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    order = selected.sort_values("rmse")["station"]
    ax.bar(order, selected.set_index("station").loc[order, "rmse"], color="#238b45")
    ax.set_ylabel("RMSE")
    ax.set_title("Chl-a ecology-enhanced model by station")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "chla_ecology_station_rmse.png")
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 12), dpi=160)
    axes = axes.ravel()
    for ax, station in zip(axes, sorted(pred["station"].unique())):
        sub = pred[pred["station"].eq(station)].sort_values("date")
        ax.plot(sub["date"], sub[TARGET], marker="o", lw=1.5, color="#222222", label="Observed")
        ax.plot(sub["date"], sub["pred"], marker="s", lw=1.25, color="#238b45", label="Predicted")
        ax.set_title(station)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Ecology-enhanced Chl-a predictions, 2023-2024", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / "chla_ecology_station_predictions.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame, selected: pd.DataFrame, old: pd.DataFrame) -> None:
    old_chla = old[old["target"].eq("Chl_a")].iloc[0]
    new = summary.iloc[0]
    lines = [
        "# Chl-a 短期生态过程变量增强报告",
        "",
        "## 一、新增变量",
        "",
        "本轮加入采样日前 7 天和 14 天窗口的短期生态过程变量：气温均值/最高值/距平、平均风速/最大风速、低风速小时数和天数、连续低风速天数、云量、太阳辐射代理、晴天代理天数、累计降水、最大日降水和强降水日数。同时新增 Landsat 严格质量控制标记，包括 strict_valid、moderate_valid、low_quality 和 valid_count_log1p。",
        "",
        "注意：当前 ERA5 本地文件没有真实太阳辐射和海表水温变量，因此 solar_proxy 使用 1 - total_cloud_cover，水温窗口暂用 2m 气温作为代理。后续需要补下载 ssrd 和 sea_surface_temperature 后再替换代理变量。",
        "",
        "## 二、总体结果",
        "",
        "| 版本 | RMSE | MAE | R2 / NSE |",
        "|---|---:|---:|---:|",
        f"| 原站点独立最优 | {old_chla['rmse']:.4f} | {old_chla['mae']:.4f} | {old_chla['r2']:.3f} |",
        f"| 短期生态增强 | {new['rmse']:.4f} | {new['mae']:.4f} | {new['r2']:.3f} |",
        "",
        "## 三、各站点结果",
        "",
        "| 站点 | 模型 | 特征组 | 变换 | RMSE | MAE | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.sort_values("station").iterrows():
        lines.append(
            f"| {row['station']} | {row['candidate']} | {row['feature_set']} | {row['transform']} | "
            f"{row['rmse']:.3f} | {row['mae']:.3f} | {row['r2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 四、解释",
            "",
            "如果增强模型优于上一版，说明 Chl-a 的短期变化确实包含可被 7-14 天窗口捕捉的过程信号，例如低风速导致的水体稳定、降水后的营养盐输入、以及较高光照条件下的藻类增长潜势。若部分站点退化，则说明这些站点仍受局地水动力、遥感质量或突发事件控制，不能用统一窗口解释。",
            "",
        ]
    )
    (OUT_DIR / "chla_ecology_enhancement_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET, parse_dates=["date"])
    df = add_chla_history(df)

    cv_frames, selected_frames, pred_frames = [], [], []
    for station, group in df.groupby("station"):
        cv, selected, pred = optimize_station(station, group)
        cv_frames.append(cv)
        selected_frames.append(selected)
        pred_frames.append(pred)

    cv_all = pd.concat(cv_frames, ignore_index=True)
    selected_all = pd.concat(selected_frames, ignore_index=True)
    pred_all = pd.concat(pred_frames, ignore_index=True)
    summary = pd.DataFrame(
        [{"target": TARGET, "model": "Chl_a_ecology_rolling_cv", "test_rows": len(pred_all),
          **metrics(pred_all[TARGET].to_numpy(), pred_all["pred"].to_numpy())}]
    )
    old = pd.read_csv("结果/扩展实验/station_specific/station_specific_best_model_overall_updated.csv", encoding="utf-8-sig")
    write_report(summary, selected_all, old)
    plot(pred_all, selected_all)

    cv_all.to_csv(OUT_DIR / "chla_ecology_candidate_scores.csv", index=False, encoding="utf-8-sig")
    selected_all.to_csv(OUT_DIR / "chla_ecology_selected_by_station.csv", index=False, encoding="utf-8-sig")
    pred_all.to_csv(OUT_DIR / "chla_ecology_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "chla_ecology_summary.csv", index=False, encoding="utf-8-sig")

    print("Chl-a ecology-enhanced summary:")
    print(summary.round(4).to_string(index=False))
    print("Selected by station:")
    print(selected_all[["station", "candidate", "feature_set", "transform", "rmse", "mae", "r2"]].round(4).to_string(index=False))
    print(f"写出 {OUT_DIR / 'chla_ecology_enhancement_report.md'}")


if __name__ == "__main__":
    main()
