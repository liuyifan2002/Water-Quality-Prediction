from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_DIAG_DIR = Path("结果/诊断结果/chla_surface_diagnostics")
DEFAULT_ZONE_DIR = Path("结果/扩展实验/chla_surface_zone_models")
DEFAULT_FIG_DIR = Path("结果/论文图表/论文图版_随机比例划分_20260905/第五章")
DEFAULT_TABLE_DIR = Path("结果/论文图表/论文表格_随机比例划分_20260905/第五章")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理论文第五章的 Chl-a 诊断与分海区扩展实验图表。")
    parser.add_argument("--diag-dir", type=Path, default=DEFAULT_DIAG_DIR)
    parser.add_argument("--zone-dir", type=Path, default=DEFAULT_ZONE_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少输入文件: {path}")
    return pd.read_csv(path, low_memory=False)


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def copy_figure(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"缺少已有图件: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build_autocorrelation_table(autocorr: pd.DataFrame, peakiness: pd.DataFrame) -> pd.DataFrame:
    merged = autocorr.merge(peakiness, on=["station", "water_zone_en", "n"], how="left", suffixes=("", "_peak"))
    summary = (
        merged.groupby("water_zone_en")
        .agg(
            站点数=("station", "nunique"),
            样本数=("n", "sum"),
            lag1相关性中位数=("lag1_corr", "median"),
            lag2相关性中位数=("lag2_corr", "median"),
            Chla均值=("mean_chla", "mean"),
            Chla变异系数中位数=("cv_chla", "median"),
            最大跳变中位数=("max_abs_change", "median"),
            相邻采样间隔中位数_天=("median_days_between_samples", "median"),
        )
        .reset_index()
        .rename(columns={"water_zone_en": "海区"})
        .sort_values("lag1相关性中位数", ascending=False)
    )
    return summary


def build_high_value_table(residuals: pd.DataFrame) -> pd.DataFrame:
    residuals = residuals.copy()
    threshold = residuals["observed"].quantile(0.95)
    high = residuals[residuals["observed"] >= threshold].copy()
    high["低估"] = high["predicted"] < high["observed"]
    summary = pd.DataFrame(
        [
            {
                "高值阈值(ug/L)": round(float(threshold), 4),
                "高值样本数": int(len(high)),
                "高值观测均值(ug/L)": round(float(high["observed"].mean()), 4),
                "高值预测均值(ug/L)": round(float(high["predicted"].mean()), 4),
                "平均残差(预测-观测)": round(float((high["predicted"] - high["observed"]).mean()), 4),
                "低估比例(%)": round(float(high["低估"].mean() * 100), 2),
            }
        ]
    )
    return summary


def build_zone_best_table(metrics: pd.DataFrame) -> pd.DataFrame:
    ok = metrics[metrics["status"] == "ok"].copy()
    best = ok.sort_values(["water_zone_en", "r2"], ascending=[True, False]).drop_duplicates("water_zone_en")
    best = best.rename(
        columns={
            "water_zone_en": "海区",
            "feature_set": "最优特征组",
            "stations": "站点数",
            "train_rows": "训练样本数",
            "test_rows": "测试样本数",
            "r2": "R2",
            "rmse": "RMSE",
            "mae": "MAE",
        }
    )
    return best[["海区", "站点数", "训练样本数", "测试样本数", "最优特征组", "R2", "RMSE", "MAE"]].sort_values("R2", ascending=False)


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    figure_map = {
        "fig_chla_surface_timeseries_top6.png": "图5-1_Chla表层代表站点时间序列.png",
        "fig_chla_surface_scatter_best.png": "图5-2_Chla表层观测值预测值散点图.png",
        "fig_chla_surface_residual_monthly.png": "图5-3_Chla表层月尺度残差变化.png",
        "fig_chla_surface_feature_importance.png": "图5-4_Chla表层主要特征重要性.png",
    }
    for source_name, target_name in figure_map.items():
        copy_figure(args.diag_dir / source_name, args.fig_dir / target_name)
    copy_figure(args.zone_dir / "fig_chla_surface_zone_best_r2.png", args.fig_dir / "图5-5_分海区与全站点模型表现比较.png")

    autocorr = read_csv(args.diag_dir / "station_autocorrelation.csv")
    peakiness = read_csv(args.diag_dir / "station_peakiness.csv")
    residuals = read_csv(args.diag_dir / "chla_surface_residuals.csv")
    zone_metrics = read_csv(args.zone_dir / "chla_surface_zone_metrics.csv")

    write_csv(build_autocorrelation_table(autocorr, peakiness), args.table_dir / "表5-1_Chla表层自相关与峰值波动统计.csv")
    write_csv(build_high_value_table(residuals), args.table_dir / "表5-2_Chla表层高值样本误差统计.csv")
    write_csv(build_zone_best_table(zone_metrics), args.table_dir / "表5-3_分海区Chla表层最佳模型结果.csv")

    pending = pd.DataFrame(
        [
            ["图5-6", "典型日期表层水质连续空间分布", "待补", "需要经空间外推验证的栅格预测因子和正式制图结果；当前不可作为正式精度结论。"],
            ["表5-4", "连续空间制图不确定性来源", "待补", "需在制图实验完成后补充遥感缺失、近岸混合像元、空间外推和水层不可见性等内容。"],
        ],
        columns=["编号", "图表名称", "状态", "补充条件"],
    )
    write_csv(pending, args.table_dir / "第五章待补图表与前提条件.csv")
    print(f"已生成第五章图表: {args.fig_dir} / {args.table_dir}")


if __name__ == "__main__":
    main()
