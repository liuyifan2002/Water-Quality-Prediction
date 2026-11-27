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


DEFAULT_DATASET = Path("数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv")
DEFAULT_PREDICTIONS = Path("结果/正式结果/full_station/full_station_predictions.csv")
DEFAULT_METRICS = Path("结果/正式结果/full_station/full_station_training_metrics.csv")
DEFAULT_BEST = Path("结果/正式结果/full_station/full_station_best_by_target_depth.csv")
DEFAULT_OUT_DIR = Path("结果/诊断结果/chla_surface_diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Chl-a surface variability, temporal persistence, coverage, and residual structure."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--best-table", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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


def load_surface_chla(dataset: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset, parse_dates=["date"], low_memory=False)
    subset = df[(df["depth"] == "surface") & pd.to_numeric(df["chla_ug_l"], errors="coerce").notna()].copy()
    numeric_cols = [
        "chla_ug_l",
        "chla_ug_l_lag1",
        "chla_ug_l_lag2",
        "chla_ug_l_roll3",
        "chla_ug_l_delta1",
        "days_since_prev_sample",
        "landsat_valid_count",
        "landsat_available",
        "ssrd_sum_3d",
        "ssrd_sum_7d",
        "ssrd_sum_14d",
        "sst_mean_3d",
        "sst_mean_7d",
        "sst_mean_14d",
    ]
    for col in numeric_cols:
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset["year"] = pd.to_numeric(subset["year"], errors="coerce")
    subset["month"] = pd.to_numeric(subset["month"], errors="coerce")
    return subset.sort_values(["station", "date"]).reset_index(drop=True)


def compute_station_autocorr(surface: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station, grp in surface.groupby("station", sort=True):
        grp = grp.sort_values("date")
        lag1 = grp[["chla_ug_l", "chla_ug_l_lag1"]].dropna()
        lag2 = grp[["chla_ug_l", "chla_ug_l_lag2"]].dropna()
        rows.append(
            {
                "station": station,
                "water_zone_en": grp["water_zone_en"].iloc[0],
                "n": len(grp),
                "lag1_corr": lag1["chla_ug_l"].corr(lag1["chla_ug_l_lag1"]) if len(lag1) >= 4 else np.nan,
                "lag2_corr": lag2["chla_ug_l"].corr(lag2["chla_ug_l_lag2"]) if len(lag2) >= 4 else np.nan,
                "mean_chla": grp["chla_ug_l"].mean(),
                "std_chla": grp["chla_ug_l"].std(),
                "cv_chla": grp["chla_ug_l"].std() / grp["chla_ug_l"].mean() if grp["chla_ug_l"].mean() > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["lag1_corr", "n"], ascending=[False, False]).reset_index(drop=True)


def compute_peakiness(surface: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    change_rows: list[dict[str, object]] = []
    for station, grp in surface.groupby("station", sort=True):
        grp = grp.sort_values("date").copy()
        grp["abs_change"] = grp["chla_ug_l"].diff().abs()
        grp["ratio_change"] = grp["chla_ug_l"] / grp["chla_ug_l_lag1"]
        rows.append(
            {
                "station": station,
                "water_zone_en": grp["water_zone_en"].iloc[0],
                "n": len(grp),
                "max_chla": grp["chla_ug_l"].max(),
                "p95_chla": grp["chla_ug_l"].quantile(0.95),
                "max_abs_change": grp["abs_change"].max(),
                "median_abs_change": grp["abs_change"].median(),
                "pct_abs_change_ge_5": (grp["abs_change"] >= 5).mean(),
                "pct_ratio_ge_2": (grp["ratio_change"] >= 2).mean(),
                "median_days_between_samples": grp["days_since_prev_sample"].median(),
            }
        )
        top_changes = grp.nlargest(3, "abs_change")[["date", "chla_ug_l", "chla_ug_l_lag1", "abs_change", "days_since_prev_sample"]]
        for _, row in top_changes.iterrows():
            change_rows.append(
                {
                    "station": station,
                    "water_zone_en": grp["water_zone_en"].iloc[0],
                    "date": row["date"],
                    "observed": row["chla_ug_l"],
                    "lag1": row["chla_ug_l_lag1"],
                    "abs_change": row["abs_change"],
                    "days_since_prev_sample": row["days_since_prev_sample"],
                }
            )
    peakiness = pd.DataFrame(rows).sort_values(["max_abs_change", "max_chla"], ascending=[False, False]).reset_index(drop=True)
    biggest_changes = pd.DataFrame(change_rows).sort_values("abs_change", ascending=False).reset_index(drop=True)
    return peakiness, biggest_changes


def compute_coverage(surface: pd.DataFrame) -> pd.DataFrame:
    def share_at_least_one(column: str) -> float:
        if column not in surface.columns:
            return float("nan")
        return float((pd.to_numeric(surface[column], errors="coerce") >= 1).mean())

    def median_value(column: str) -> float:
        if column not in surface.columns:
            return float("nan")
        return float(pd.to_numeric(surface[column], errors="coerce").median())

    coverage_rows = [
        {
            "metric": "landsat_available_share",
            "value": share_at_least_one("landsat_available"),
        },
        {
            "metric": "landsat_valid_count_median",
            "value": median_value("landsat_valid_count"),
        },
        {
            "metric": "s2_feature_fields_present",
            "value": float(any(column.startswith("s2_") for column in surface.columns)),
        },
        {
            "metric": "ssrd_7d_missing_share",
            "value": float(surface["ssrd_sum_7d"].isna().mean()),
        },
        {
            "metric": "sst_7d_missing_share",
            "value": float(surface["sst_mean_7d"].isna().mean()),
        },
    ]
    return pd.DataFrame(coverage_rows)


def load_best_predictions(predictions_path: Path, best_table_path: Path) -> tuple[str, float, pd.DataFrame]:
    pred = pd.read_csv(predictions_path, parse_dates=["date"])
    best = pd.read_csv(best_table_path)
    row = best[(best["target"] == "chla") & (best["depth"] == "surface")].iloc[0]
    best_feature_set = row["feature_set"]
    best_r2 = float(row["r2"])
    out = pred[(pred["target"] == "chla") & (pred["depth"] == "surface") & (pred["feature_set"] == best_feature_set)].copy()
    out["residual"] = out["predicted"] - out["observed"]
    out["abs_error"] = out["residual"].abs()
    return best_feature_set, best_r2, out.sort_values(["date", "station"]).reset_index(drop=True)


def fit_best_model_for_importance(surface: pd.DataFrame, metrics_path: Path):
    metrics = pd.read_csv(metrics_path)
    row = (
        metrics[
            (metrics["target"] == "chla")
            & (metrics["depth"] == "surface")
            & (metrics["status"] == "ok")
        ]
        .sort_values("r2", ascending=False)
        .iloc[0]
    )
    params = json.loads(row["model_params_json"])
    feature_set = row["feature_set"]
    split_mode = str(row.get("split_mode", "random"))
    test_size = float(row.get("test_size", 0.2)) if split_mode == "random" else 0.2
    random_state = int(row.get("split_random_state", 42)) if split_mode == "random" else 42

    train_mod = load_train_module()
    feature_sets = train_mod.build_feature_sets(surface, train_mod.TARGET_MAP["chla"])
    features = feature_sets[feature_set]
    split_args = argparse.Namespace(
        split_mode=split_mode,
        test_size=test_size,
        random_state=random_state,
        train_end_year=int(row.get("train_end_year", 2022)) if split_mode == "time" else 2022,
        test_start_year=int(row.get("test_start_year", 2023)) if split_mode == "time" else 2023,
        test_end_year=int(row.get("test_end_year", 2024)) if split_mode == "time" else 2024,
    )
    train, _ = train_mod.split_for_evaluation(surface, split_args)

    model_args = argparse.Namespace(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=4,
    )
    model = train_mod.make_model(model_args, params)
    y_train, _ = train_mod.transform_target(train["chla_ug_l"], True)
    model.fit(train[features], y_train)
    return feature_set, features, model


def plot_station_timeseries(surface: pd.DataFrame, peakiness: pd.DataFrame, out_dir: Path) -> None:
    top = peakiness.head(6)
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), constrained_layout=True)
    for ax, (_, row) in zip(axes.flatten(), top.iterrows()):
        grp = surface[surface["station"] == row["station"]].sort_values("date")
        ax.plot(grp["date"], grp["chla_ug_l"], marker="o", linewidth=1.4, markersize=3)
        ax.set_title(f"{row['station']} | {row['water_zone_en']}")
        ax.set_ylabel("Chl-a (ug/L)")
    fig.suptitle("Chl-a表层高波动站点时间序列", fontsize=14)
    fig.savefig(out_dir / "fig_chla_surface_timeseries_top6.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(pred: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    monthly = (
        pred.groupby(pd.to_datetime(pred["date"]).dt.to_period("M"))
        .agg(mean_residual=("residual", "mean"), mean_abs_error=("abs_error", "mean"), n=("residual", "size"))
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    axes[0].plot(monthly["date"], monthly["mean_residual"], marker="o")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Chl-a表层按月平均残差")
    axes[0].set_ylabel("Pred - Obs")

    axes[1].plot(monthly["date"], monthly["mean_abs_error"], marker="o", color="#d62828")
    axes[1].set_title("Chl-a表层按月平均绝对误差")
    axes[1].set_ylabel("MAE")
    axes[1].set_xlabel("Date")
    fig.savefig(out_dir / "fig_chla_surface_residual_monthly.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pred["observed"], pred["predicted"], s=18, alpha=0.4)
    line_min = min(pred["observed"].min(), pred["predicted"].min())
    line_max = max(pred["observed"].max(), pred["predicted"].max())
    ax.plot([line_min, line_max], [line_min, line_max], color="black", linewidth=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Chl-a表层 Observed vs Predicted")
    fig.savefig(out_dir / "fig_chla_surface_scatter_best.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return monthly


def plot_importance(features: list[str], model, out_dir: Path) -> pd.DataFrame:
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["importance"], color="#2a9d8f")
    ax.set_title("Chl-a表层最优模型特征重要性")
    ax.set_xlabel("Importance")
    fig.savefig(out_dir / "fig_chla_surface_feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return importance


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(col) for col in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body: list[str] = []
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def write_summary(
    surface: pd.DataFrame,
    coverage: pd.DataFrame,
    autocorr: pd.DataFrame,
    peakiness: pd.DataFrame,
    biggest_changes: pd.DataFrame,
    pred: pd.DataFrame,
    importance: pd.DataFrame,
    best_feature_set: str,
    best_r2: float,
    out_path: Path,
) -> None:
    lag1_pairs = surface[["chla_ug_l", "chla_ug_l_lag1"]].dropna()
    lag2_pairs = surface[["chla_ug_l", "chla_ug_l_lag2"]].dropna()
    roll3_pairs = surface[["chla_ug_l", "chla_ug_l_roll3"]].dropna()

    zone_resid = (
        pred.groupby("water_zone_en")
        .agg(n=("residual", "size"), mean_residual=("residual", "mean"), mae=("abs_error", "mean"))
        .sort_values("mae", ascending=False)
    )

    top_changes_md = biggest_changes.head(10).copy()
    top_features_md = importance.head(12).copy()
    coverage_map = dict(zip(coverage["metric"], coverage["value"]))

    lines = [
        "# Chl-a表层诊断",
        "",
        "## 样本与波动概况",
        "",
        f"- 样本数：{len(surface)}",
        f"- 站点数：{surface['station'].nunique()}",
        f"- 时间范围：{surface['date'].min().date()} 至 {surface['date'].max().date()}",
        f"- Chl-a均值：{surface['chla_ug_l'].mean():.3f} ug/L",
        f"- Chl-a标准差：{surface['chla_ug_l'].std():.3f} ug/L",
        f"- Chl-a最大值：{surface['chla_ug_l'].max():.3f} ug/L",
        "",
        "## 历史自相关",
        "",
        f"- 全样本 lag1 相关系数：{lag1_pairs['chla_ug_l'].corr(lag1_pairs['chla_ug_l_lag1']):.3f}",
        f"- 全样本 lag2 相关系数：{lag2_pairs['chla_ug_l'].corr(lag2_pairs['chla_ug_l_lag2']):.3f}",
        f"- 全样本 roll3 相关系数：{roll3_pairs['chla_ug_l'].corr(roll3_pairs['chla_ug_l_roll3']):.3f}",
        f"- 站点 lag1 中位相关：{autocorr['lag1_corr'].median():.3f}",
        f"- 站点 lag1 四分位范围：{autocorr['lag1_corr'].quantile(0.25):.3f} 到 {autocorr['lag1_corr'].quantile(0.75):.3f}",
        "",
        "## 遥感与过程覆盖度",
        "",
        f"- Landsat 可用样本比例：{coverage_map['landsat_available_share']:.3f}",
        f"- Landsat 有效影像数中位数：{coverage_map['landsat_valid_count_median']:.1f}",
        f"- Sentinel-2 特征字段是否进入正式主表：{coverage_map['s2_feature_fields_present']:.0f}",
        f"- ssrd_7d 缺失率：{coverage_map['ssrd_7d_missing_share']:.3f}",
        f"- sst_7d 缺失率：{coverage_map['sst_7d_missing_share']:.3f}",
        "",
        "## 当前最优模型",
        "",
        f"- 最优特征组：`{best_feature_set}`",
        f"- 测试集样本数：{len(pred)}",
        f"- 正式测试集 R2：{best_r2:.3f}",
        f"- 测试集残差均值：{pred['residual'].mean():.3f}",
        f"- 测试集绝对误差均值：{pred['abs_error'].mean():.3f}",
        "",
        "## 诊断结论",
        "",
        "1. Chl-a表层存在明显短期跳变；部分站点相邻两次采样之间会出现较大的浓度跃升，说明短期扰动确实强。",
        "2. lag1和roll3对当前值有一定解释力，但并不高到可以稳定锁定峰值，因此仅靠历史惯性难以得到高R2。",
        "3. 当前正式主表保留 Landsat 遥感字段，但未包含 Sentinel-2 特征字段，因此本轮正式 SHAP 仅解释实际进入模型的遥感变量。",
        "4. ssrd/sst窗口已实现全覆盖，但它们是离岸海面代表值，对站点尺度的近岸微环境仍有平滑效应。",
        "5. 如果后续残差在部分海区持续偏大，应优先做分海区表层Chl-a建模，而不是继续把全部站点完全混合。",
        "",
        "## 波动最大的样本变化（前10）",
        "",
        markdown_table(top_changes_md),
        "",
        "## 当前最重要的前12个特征",
        "",
        markdown_table(top_features_md),
        "",
        "## 分海区残差概览",
        "",
        markdown_table(zone_resid.reset_index()),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    surface = load_surface_chla(args.dataset)
    coverage = compute_coverage(surface)
    autocorr = compute_station_autocorr(surface)
    peakiness, biggest_changes = compute_peakiness(surface)
    best_feature_set, best_r2, pred = load_best_predictions(args.predictions, args.best_table)
    _, features, model = fit_best_model_for_importance(surface, args.metrics)
    monthly = plot_residuals(pred, args.out_dir)
    plot_station_timeseries(surface, peakiness, args.out_dir)
    importance = plot_importance(features, model, args.out_dir)

    coverage.to_csv(args.out_dir / "coverage_summary.csv", index=False, encoding="utf-8-sig")
    autocorr.to_csv(args.out_dir / "station_autocorrelation.csv", index=False, encoding="utf-8-sig")
    peakiness.to_csv(args.out_dir / "station_peakiness.csv", index=False, encoding="utf-8-sig")
    biggest_changes.to_csv(args.out_dir / "top_changes.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(args.out_dir / "chla_surface_residuals.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(args.out_dir / "chla_surface_monthly_residuals.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(args.out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    write_summary(
        surface=surface,
        coverage=coverage,
        autocorr=autocorr,
        peakiness=peakiness,
        biggest_changes=biggest_changes,
        pred=pred,
        importance=importance,
        best_feature_set=best_feature_set,
        best_r2=best_r2,
        out_path=args.out_dir / "叶绿素表层诊断总结.md",
    )
    print(f"Wrote diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
