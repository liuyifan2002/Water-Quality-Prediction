from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


DATASET = Path("data/processed/model_dataset_surface_2014_2024_enhanced.csv")
OUT_DIR = Path("reports/chla_optimization")
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


@dataclass
class CandidateResult:
    station: str
    candidate: str
    feature_set: str
    transform: str
    val_rmse: float
    val_mae: float
    val_r2: float
    test_rmse: float
    test_mae: float
    test_r2: float


def nse(y_true: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - pred) ** 2) / denom


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "nse": float(nse(y_true, pred)),
    }


def add_chla_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["station", "date"]).copy()
    grouped = out.groupby("station", sort=False)[TARGET]
    out["Chl_a_lag1"] = grouped.shift(1)
    out["Chl_a_lag2"] = grouped.shift(2)
    out["Chl_a_roll3"] = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    out["Chl_a_delta1"] = out["Chl_a_lag1"] - out["Chl_a_lag2"]
    return out


def feature_sets() -> dict[str, list[str]]:
    f3 = F1_FEATURES + F2_FEATURES
    compact = [
        "month", "season_sin", "season_cos",
        "t2m", "tcc", "wind_speed", "rain_3d", "rain_14d", "rain_month",
        "tidal_range", "ntl_avg_rad",
        "green", "red", "nir", "swir1", "ndwi", "mndwi", "ndti",
        "blue_green", "green_red", "valid_count",
        "rain_month_lag1", "rain_month_roll3", "wind_speed_lag1", "tcc_lag1",
        "ndwi_lag1", "mndwi_lag1", "ndti_lag1",
    ]
    return {
        "F1_process": F1_FEATURES,
        "F2_remote": F2_FEATURES,
        "F3_fusion": f3,
        "F3_AR": f3 + AR_FEATURES,
        "compact_AR": compact + AR_FEATURES,
        "history_only": AR_FEATURES + ["month", "season_sin", "season_cos"],
    }


def transform_y(y: pd.Series, transform: str) -> tuple[np.ndarray, callable]:
    values = y.to_numpy()
    if transform == "log1p":
        return np.log1p(values), np.expm1
    if transform == "sqrt":
        return np.sqrt(np.maximum(values, 0)), lambda p: np.square(np.maximum(p, 0))
    return values, lambda p: p


def clean_prediction(pred: np.ndarray, train_y: pd.Series) -> np.ndarray:
    upper = max(float(train_y.quantile(0.99) * 1.35), float(train_y.max()))
    return np.clip(pred, 0, upper)


def make_models() -> dict[str, object]:
    from xgboost import XGBRegressor

    return {
        "xgb_shallow": XGBRegressor(
            n_estimators=220,
            max_depth=1,
            learning_rate=0.04,
            min_child_weight=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            reg_alpha=0.1,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
        "xgb_regularized": XGBRegressor(
            n_estimators=320,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            reg_alpha=0.15,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=600,
            max_depth=4,
            min_samples_leaf=5,
            max_features=0.55,
            random_state=42,
            n_jobs=1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=600,
            max_depth=5,
            min_samples_leaf=5,
            max_features=0.65,
            random_state=42,
            n_jobs=1,
        ),
        "gbr_huber": GradientBoostingRegressor(
            loss="huber",
            n_estimators=180,
            max_depth=2,
            learning_rate=0.035,
            min_samples_leaf=6,
            subsample=0.85,
            random_state=42,
        ),
        "ridge_robust": make_pipeline(RobustScaler(), Ridge(alpha=8.0)),
        "huber_linear": make_pipeline(RobustScaler(), HuberRegressor(alpha=0.2, epsilon=1.5, max_iter=1000)),
    }


def matrix(train: pd.DataFrame, other: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fill = train[features].median(numeric_only=True).fillna(0)
    return train[features].fillna(fill), other[features].fillna(fill)


def climatology_predict(train: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    month_med = train.groupby("month")[TARGET].median()
    fallback = float(train[TARGET].median())
    return other["month"].map(month_med).fillna(fallback).to_numpy()


def persistence_predict(train: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    fallback = float(train[TARGET].median())
    return other["Chl_a_lag1"].fillna(other["Chl_a_roll3"]).fillna(fallback).to_numpy()


def evaluate_station(station: str, group: pd.DataFrame) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    train_all = group[group["year"].between(2014, 2022)].dropna(subset=[TARGET]).copy()
    fit = train_all[train_all["year"].between(2014, 2020)].copy()
    val = train_all[train_all["year"].between(2021, 2022)].copy()
    test = group[group["year"].between(2023, 2024)].dropna(subset=[TARGET]).copy()

    if len(fit) < 40 or len(val) < 10 or len(test) < 10:
        return [], pd.DataFrame(), pd.DataFrame()

    candidates = []
    pred_rows = []

    base_predictions = {
        "month_climatology": (climatology_predict(fit, val), climatology_predict(train_all, test)),
        "persistence_lag1": (persistence_predict(fit, val), persistence_predict(train_all, test)),
    }
    for name, (val_pred, test_pred) in base_predictions.items():
        val_pred = clean_prediction(val_pred, fit[TARGET])
        test_pred = clean_prediction(test_pred, train_all[TARGET])
        val_m = metrics(val[TARGET].to_numpy(), val_pred)
        test_m = metrics(test[TARGET].to_numpy(), test_pred)
        candidates.append(
            {
                "station": station,
                "candidate": name,
                "feature_set": "baseline",
                "transform": "none",
                "val_rmse": val_m["rmse"],
                "val_mae": val_m["mae"],
                "val_r2": val_m["r2"],
                "test_rmse": test_m["rmse"],
                "test_mae": test_m["mae"],
                "test_r2": test_m["r2"],
                "test_nse": test_m["nse"],
            }
        )

    model_val_preds: dict[str, np.ndarray] = {}
    model_test_preds: dict[str, np.ndarray] = {}
    model_meta: dict[str, tuple[str, str, str]] = {}
    for fs_name, features in feature_sets().items():
        x_fit, x_val = matrix(fit, val, features)
        _, x_test_final = matrix(train_all, test, features)
        x_train_all, _ = matrix(train_all, test, features)
        for transform in ["log1p", "sqrt", "raw"]:
            y_fit, inv_fit = transform_y(fit[TARGET], transform)
            y_train_all, inv_all = transform_y(train_all[TARGET], transform)
            for model_name, model in make_models().items():
                key = f"{model_name}|{fs_name}|{transform}"
                model.fit(x_fit, y_fit)
                val_pred = clean_prediction(inv_fit(model.predict(x_val)), fit[TARGET])
                final_model = make_models()[model_name]
                final_model.fit(x_train_all, y_train_all)
                test_pred = clean_prediction(inv_all(final_model.predict(x_test_final)), train_all[TARGET])

                val_m = metrics(val[TARGET].to_numpy(), val_pred)
                test_m = metrics(test[TARGET].to_numpy(), test_pred)
                candidates.append(
                    {
                        "station": station,
                        "candidate": model_name,
                        "feature_set": fs_name,
                        "transform": transform,
                        "val_rmse": val_m["rmse"],
                        "val_mae": val_m["mae"],
                        "val_r2": val_m["r2"],
                        "test_rmse": test_m["rmse"],
                        "test_mae": test_m["mae"],
                        "test_r2": test_m["r2"],
                        "test_nse": test_m["nse"],
                    }
                )
                model_val_preds[key] = val_pred
                model_test_preds[key] = test_pred
                model_meta[key] = (model_name, fs_name, transform)

    # 月气候态在 Chl-a 上很重要；融合候选用验证集选择权重。
    val_clim = climatology_predict(fit, val)
    test_clim = climatology_predict(train_all, test)
    for key, val_pred in model_val_preds.items():
        for weight in [0.25, 0.5, 0.75]:
            blended_val = weight * val_pred + (1 - weight) * val_clim
            blended_test = weight * model_test_preds[key] + (1 - weight) * test_clim
            val_m = metrics(val[TARGET].to_numpy(), blended_val)
            test_m = metrics(test[TARGET].to_numpy(), blended_test)
            model_name, fs_name, transform = model_meta[key]
            candidates.append(
                {
                    "station": station,
                    "candidate": f"blend_{weight:.2f}_{model_name}",
                    "feature_set": fs_name,
                    "transform": transform,
                    "val_rmse": val_m["rmse"],
                    "val_mae": val_m["mae"],
                    "val_r2": val_m["r2"],
                    "test_rmse": test_m["rmse"],
                    "test_mae": test_m["mae"],
                    "test_r2": test_m["r2"],
                    "test_nse": test_m["nse"],
                }
            )

    cand_df = pd.DataFrame(candidates).sort_values(["val_rmse", "val_mae"]).reset_index(drop=True)
    best = cand_df.iloc[0]

    if best["candidate"] == "month_climatology":
        final_pred = climatology_predict(train_all, test)
    elif best["candidate"] == "persistence_lag1":
        final_pred = persistence_predict(train_all, test)
    elif str(best["candidate"]).startswith("blend_"):
        weight = float(str(best["candidate"]).split("_")[1])
        model_name = "_".join(str(best["candidate"]).split("_")[2:])
        features = feature_sets()[best["feature_set"]]
        x_train_all, x_test = matrix(train_all, test, features)
        y_train_all, inv = transform_y(train_all[TARGET], best["transform"])
        model = make_models()[model_name]
        model.fit(x_train_all, y_train_all)
        model_pred = clean_prediction(inv(model.predict(x_test)), train_all[TARGET])
        final_pred = weight * model_pred + (1 - weight) * climatology_predict(train_all, test)
    else:
        features = feature_sets()[best["feature_set"]]
        x_train_all, x_test = matrix(train_all, test, features)
        y_train_all, inv = transform_y(train_all[TARGET], best["transform"])
        model = make_models()[best["candidate"]]
        model.fit(x_train_all, y_train_all)
        final_pred = clean_prediction(inv(model.predict(x_test)), train_all[TARGET])

    pred_df = test[["station", "date", "year", "month", TARGET]].copy()
    pred_df["pred"] = final_pred
    pred_df["abs_error"] = np.abs(pred_df[TARGET] - pred_df["pred"])
    pred_df["selected_candidate"] = best["candidate"]
    pred_df["selected_feature_set"] = best["feature_set"]
    pred_df["selected_transform"] = best["transform"]
    return candidates, pred_df, cand_df.head(10)


def plot_predictions(pred: pd.DataFrame) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stations = sorted(pred["station"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 12), dpi=160, sharex=False)
    axes = axes.ravel()
    for ax, station in zip(axes, stations):
        sub = pred[pred["station"] == station].sort_values("date")
        ax.plot(sub["date"], sub[TARGET], marker="o", lw=1.6, label="Observed", color="#222222")
        ax.plot(sub["date"], sub["pred"], marker="s", lw=1.3, label="Predicted", color="#2b8cbe")
        ax.set_title(station)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Optimized station-specific Chl-a predictions, 2023-2024", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / "chla_optimized_station_predictions.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=160)
    ax.scatter(pred[TARGET], pred["pred"], c="#2b8cbe", alpha=0.78, edgecolor="white", linewidth=0.4)
    lim = max(pred[TARGET].max(), pred["pred"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#333333", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed Chl-a")
    ax.set_ylabel("Predicted Chl-a")
    ax.set_title("Observed vs predicted Chl-a")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "chla_optimized_observed_vs_predicted.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame, selected: pd.DataFrame, old: pd.DataFrame) -> None:
    old_chla = old[old["target"].eq("Chl_a")].sort_values("rmse").iloc[0]
    new = summary.iloc[0]
    lines = [
        "# Chl-a 专项优化报告",
        "",
        "## 一、优化思路",
        "",
        "本轮只针对表层叶绿素 a（Chl-a）优化，并保持站点独立建模。由于 Chl-a 分布高度偏态，且少数高值月份会明显拉高 RMSE，本轮没有简单加深模型，而是比较了：月气候态基线、上一期持续性基线、F1 过程特征、F2 遥感特征、F3 融合特征、F3+历史 Chl-a 特征、紧凑特征集，以及月气候态与机器学习预测的融合方案。",
        "",
        "训练和测试仍保持时间外推设置：2014-2020 年用于拟合候选模型，2021-2022 年用于每个站点选择模型，2014-2022 年重新训练最终模型，2023-2024 年作为独立测试。",
        "",
        "## 二、总体结果",
        "",
        "| 版本 | RMSE | MAE | R2 / NSE |",
        "|---|---:|---:|---:|",
        f"| 原站点独立最优 | {old_chla['rmse']:.4f} | {old_chla['mae']:.4f} | {old_chla['r2']:.3f} |",
        f"| Chl-a 专项优化 | {new['rmse']:.4f} | {new['mae']:.4f} | {new['r2']:.3f} |",
        "",
        "## 三、各站点最终选择",
        "",
        "| 站点 | 选择模型 | 特征组 | 目标变换 | RMSE | MAE | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.sort_values("station").iterrows():
        lines.append(
            f"| {row['station']} | {row['candidate']} | {row['feature_set']} | {row['transform']} | "
            f"{row['test_rmse']:.3f} | {row['test_mae']:.3f} | {row['test_r2']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 四、结果解释",
            "",
            "1. Chl-a 的主要难点不是模型复杂度不足，而是月尺度特征难以捕捉短期藻华峰值。DM1、SM6、VM5、WM2 等站点存在明显高值和突发波动，模型容易低估峰值。",
            "2. 如果某些站点最终选择月气候态或气候态融合模型，说明这些站点 2023-2024 年的可预测信号主要来自季节性背景，而不是 Landsat 或气象变量的即时变化。",
            "3. 如果某些站点选择 F3_AR 或 compact_AR，说明该站点 Chl-a 具有一定短期记忆效应；但这种效应弱于 TN/TP，不能作为所有站点的统一假设。",
            "4. Chl-a 后续优化的科学重点应从单纯调参转向过程补充：太阳辐射、连续低风速、采样日前 7-14 天水温/降水窗口、藻华异常月份识别，以及更严格的近岸水体遥感质量控制。",
            "",
            "## 五、对研究问题的影响",
            "",
            "Chl-a 专项优化可以帮助回答：在香港近岸海域，月尺度多源数据是否足以预测藻类生物量变化？当前结果倾向于说明，常规月尺度气象、降水、潮汐、夜光和 Landsat 特征能够解释一部分背景变化，但对突发高 Chl-a 事件仍不足。因此，Chl-a 是本研究中最能体现近岸生态过程复杂性的指标，也是后续加入更高频气象、太阳辐射和 Sentinel-2 增强实验的主要理由。",
            "",
        ]
    )
    (OUT_DIR / "chla_optimization_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET, parse_dates=["date"])
    df = add_chla_history(df)

    all_candidates = []
    selected_rows = []
    pred_frames = []
    top10_frames = []
    for station, group in df.groupby("station"):
        candidates, pred_df, top10 = evaluate_station(station, group)
        if not candidates:
            continue
        cand = pd.DataFrame(candidates)
        best = cand.sort_values(["val_rmse", "val_mae"]).iloc[0].copy()
        selected_rows.append(best)
        all_candidates.append(cand)
        pred_frames.append(pred_df)
        top10["rank"] = np.arange(1, len(top10) + 1)
        top10_frames.append(top10)

    candidates_all = pd.concat(all_candidates, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    pred = pd.concat(pred_frames, ignore_index=True)
    top10_all = pd.concat(top10_frames, ignore_index=True)

    summary = pd.DataFrame(
        [
            {
                "target": TARGET,
                "model": "Chl_a_station_selected",
                "test_rows": len(pred),
                **metrics(pred[TARGET].to_numpy(), pred["pred"].to_numpy()),
            }
        ]
    )

    old = pd.read_csv("reports/station_specific/station_specific_best_model_overall_updated.csv", encoding="utf-8-sig")
    write_report(summary, selected, old)
    plot_predictions(pred)

    candidates_all.to_csv(OUT_DIR / "chla_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "chla_selected_models_by_station.csv", index=False, encoding="utf-8-sig")
    top10_all.to_csv(OUT_DIR / "chla_top10_candidates_by_station.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT_DIR / "chla_optimized_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "chla_optimized_summary.csv", index=False, encoding="utf-8-sig")

    print("Chl-a optimized summary:")
    print(summary.round(4).to_string(index=False))
    print("Selected models by station:")
    print(selected[["station", "candidate", "feature_set", "transform", "test_rmse", "test_mae", "test_r2"]].round(4).to_string(index=False))
    print(f"写出 {OUT_DIR / 'chla_optimization_report.md'}")


if __name__ == "__main__":
    main()
