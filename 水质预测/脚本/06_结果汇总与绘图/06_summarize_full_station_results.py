from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_METRICS = Path("结果/正式结果/full_station/full_station_training_metrics.csv")
DEFAULT_OUT_DIR = Path("结果/正式结果/full_station")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize full-station model results into comparison tables and a markdown brief."
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}. Run 脚本/04_模型训练/05_train_full_station_models.py first.")
    metrics = pd.read_csv(path)
    metrics = metrics.rename(columns={c: c.replace("\ufeff", "") for c in metrics.columns})
    required = {"target", "depth", "feature_set", "status", "rmse", "mae", "r2", "nse"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metrics file is missing columns: {sorted(missing)}")
    return metrics


def build_best_table(ok: pd.DataFrame) -> pd.DataFrame:
    best_idx = ok.groupby(["target", "depth"])["r2"].idxmax()
    best = ok.loc[best_idx].copy().sort_values(["target", "depth"]).reset_index(drop=True)
    return best


def build_feature_set_summary(ok: pd.DataFrame) -> pd.DataFrame:
    summary = (
        ok.groupby("feature_set")
        .agg(
            runs=("feature_set", "size"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            mean_nse=("nse", "mean"),
            median_r2=("r2", "median"),
            median_nse=("nse", "median"),
        )
        .reset_index()
        .sort_values("mean_r2", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def build_gain_table(ok: pd.DataFrame) -> pd.DataFrame:
    pivot = ok.pivot_table(
        index=["target", "depth"],
        columns="feature_set",
        values=["rmse", "r2", "nse"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{feature_set}" for metric, feature_set in pivot.columns]
    gain = pivot.reset_index()

    history_remote_key = "f3_history_remote"
    history_key = "f1_history"
    if {f"r2_{history_remote_key}", f"r2_{history_key}"}.issubset(gain.columns):
        gain["history_remote_minus_history_r2"] = gain[f"r2_{history_remote_key}"] - gain[f"r2_{history_key}"]
    if {f"rmse_{history_remote_key}", f"rmse_{history_key}"}.issubset(gain.columns):
        gain["history_remote_minus_history_rmse"] = gain[f"rmse_{history_remote_key}"] - gain[f"rmse_{history_key}"]
    if {f"r2_{history_remote_key}", "r2_baseline"}.issubset(gain.columns):
        gain["history_remote_minus_baseline_r2"] = gain[f"r2_{history_remote_key}"] - gain["r2_baseline"]
    if {f"rmse_{history_remote_key}", "rmse_baseline"}.issubset(gain.columns):
        gain["history_remote_minus_baseline_rmse"] = gain[f"rmse_{history_remote_key}"] - gain["rmse_baseline"]

    return gain.sort_values(["target", "depth"]).reset_index(drop=True)


def build_pivot_table(ok: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = ok.pivot_table(index=["target", "depth"], columns="feature_set", values=metric, aggfunc="first")
    return table.reset_index().sort_values(["target", "depth"]).reset_index(drop=True)


def write_markdown_summary(
    best: pd.DataFrame,
    feature_summary: pd.DataFrame,
    gains: pd.DataFrame,
    out_path: Path,
) -> None:
    def df_to_md(df: pd.DataFrame) -> str:
        header = "| " + " | ".join(df.columns.astype(str)) + " |"
        sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = [
            "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep, *rows])

    lines: list[str] = []
    lines.append("# 全站点正式结果汇总")
    lines.append("")

    if not best.empty:
        lines.append("## 各指标各水层最优模型")
        lines.append("")
        lines.append(df_to_md(best[["target", "depth", "feature_set", "rmse", "mae", "r2", "nse"]]))
        lines.append("")

    if not feature_summary.empty:
        lines.append("## 各特征组平均表现")
        lines.append("")
        lines.append(
            df_to_md(
                feature_summary[
                    ["feature_set", "runs", "mean_rmse", "mean_mae", "mean_r2", "mean_nse", "median_r2", "median_nse"]
                ]
            )
        )
        lines.append("")

    if not gains.empty:
        keep_cols = [
            c
            for c in [
                "target",
                "depth",
                "history_remote_minus_history_r2",
                "history_remote_minus_history_rmse",
                "history_remote_minus_baseline_r2",
                "history_remote_minus_baseline_rmse",
            ]
            if c in gains.columns
        ]
        lines.append("## 历史信息与遥感融合增益")
        lines.append("")
        lines.append(df_to_md(gains[keep_cols]))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.metrics)
    ok = metrics[metrics["status"] == "ok"].copy()
    if ok.empty:
        raise SystemExit("No successful runs found in the metrics file.")

    best = build_best_table(ok)
    feature_summary = build_feature_set_summary(ok)
    gains = build_gain_table(ok)
    r2_pivot = build_pivot_table(ok, "r2")
    rmse_pivot = build_pivot_table(ok, "rmse")

    best_out = args.out_dir / "full_station_best_by_target_depth.csv"
    feature_summary_out = args.out_dir / "full_station_feature_set_summary.csv"
    gains_out = args.out_dir / "full_station_fusion_gain.csv"
    r2_out = args.out_dir / "full_station_r2_pivot.csv"
    rmse_out = args.out_dir / "full_station_rmse_pivot.csv"
    md_out = args.out_dir / "正式主结果汇总.md"

    best.to_csv(best_out, index=False, encoding="utf-8-sig")
    feature_summary.to_csv(feature_summary_out, index=False, encoding="utf-8-sig")
    gains.to_csv(gains_out, index=False, encoding="utf-8-sig")
    r2_pivot.to_csv(r2_out, index=False, encoding="utf-8-sig")
    rmse_pivot.to_csv(rmse_out, index=False, encoding="utf-8-sig")
    write_markdown_summary(best, feature_summary, gains, md_out)

    print(f"Wrote {best_out}")
    print(f"Wrote {feature_summary_out}")
    print(f"Wrote {gains_out}")
    print(f"Wrote {r2_out}")
    print(f"Wrote {rmse_out}")
    print(f"Wrote {md_out}")
    print(best[["target", "depth", "feature_set", "r2", "rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
