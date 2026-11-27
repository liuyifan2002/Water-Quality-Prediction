from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb


DEFAULT_DATASET = Path("数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv")
DEFAULT_OUT_DIR = Path("结果/论文图表/full_station_shap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 Chl-a 和 Secchi 的融合模型生成 SHAP 图和解释表。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--targets", nargs="*", default=["chla", "secchi"], choices=["chla", "secchi", "tn", "tp"])
    parser.add_argument("--depths", nargs="*", default=["surface", "middle", "bottom"], choices=["surface", "middle", "bottom"])
    parser.add_argument(
        "--feature-set",
        default="f3_history_remote",
        choices=["baseline", "f1_history", "f2_remote", "f3_history_remote"],
    )
    parser.add_argument("--split-mode", choices=["random", "time"], default="random")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-end-year", type=int, default=2022, help="仅 split-mode=time 时使用。")
    parser.add_argument("--test-start-year", type=int, default=2023, help="仅 split-mode=time 时使用。")
    parser.add_argument("--test-end-year", type=int, default=2024, help="仅 split-mode=time 时使用。")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
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


def top_remote_summary(contrib_df: pd.DataFrame, feature_df: pd.DataFrame, combo_meta: dict[str, str]) -> pd.DataFrame:
    remote_features = [c for c in contrib_df.columns if c.startswith("s2_") or c.startswith("landsat_")]
    rows: list[dict[str, object]] = []
    for feature in remote_features:
        feature_values = pd.to_numeric(feature_df[feature], errors="coerce")
        shap_values = pd.to_numeric(contrib_df[feature], errors="coerce")
        valid = feature_values.notna() & shap_values.notna()
        if not valid.any():
            continue

        feature_valid = feature_values[valid]
        shap_valid = shap_values[valid]
        q25 = feature_valid.quantile(0.25)
        q75 = feature_valid.quantile(0.75)
        low_mean = shap_valid[feature_valid <= q25].mean()
        high_mean = shap_valid[feature_valid >= q75].mean()
        if pd.isna(low_mean) or pd.isna(high_mean):
            direction = "关系不稳定"
        elif high_mean > low_mean:
            direction = "高值更推动预测升高"
        elif high_mean < low_mean:
            direction = "高值更推动预测降低"
        else:
            direction = "高低值贡献接近"

        rows.append(
            {
                **combo_meta,
                "feature": feature,
                "mean_abs_shap": float(shap_valid.abs().mean()),
                "mean_shap": float(shap_valid.mean()),
                "feature_mean": float(feature_valid.mean()),
                "feature_nonnull": int(valid.sum()),
                "q25": float(q25),
                "q75": float(q75),
                "low_q_mean_shap": float(low_mean),
                "high_q_mean_shap": float(high_mean),
                "effect_direction": direction,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def make_remote_bar_grid(summary_df: pd.DataFrame, out_dir: Path) -> None:
    combos = summary_df[["target", "depth"]].drop_duplicates().sort_values(["target", "depth"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (_, combo) in zip(axes_flat, combos.iterrows()):
        panel = summary_df[(summary_df["target"] == combo["target"]) & (summary_df["depth"] == combo["depth"])].head(8)
        if panel.empty:
            ax.set_axis_off()
            continue
        plot_df = panel.iloc[::-1]
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#2a9d8f")
        ax.set_title(f"{combo['target'].upper()} | {combo['depth']}")
        ax.set_xlabel("Mean |SHAP|")
        ax.set_ylabel("")

    fig.savefig(out_dir / "fig05_remote_shap_importance_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_dependence_grid(summary_df: pd.DataFrame, dependence_rows: list[dict[str, object]], out_dir: Path) -> None:
    dep = pd.DataFrame(dependence_rows)
    combos = summary_df[["target", "depth"]].drop_duplicates().sort_values(["target", "depth"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (_, combo) in zip(axes_flat, combos.iterrows()):
        panel = dep[(dep["target"] == combo["target"]) & (dep["depth"] == combo["depth"])].copy()
        if panel.empty:
            ax.set_axis_off()
            continue

        feature_name = panel["feature"].iloc[0]
        ax.scatter(panel["feature_value"], panel["shap_value"], s=12, alpha=0.45, color="#457b9d", edgecolors="none")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{combo['target'].upper()} | {combo['depth']} | {feature_name}")
        ax.set_xlabel(feature_name)
        ax.set_ylabel("SHAP value")

    fig.savefig(out_dir / "fig06_remote_top_feature_dependence_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# 遥感特征 SHAP 结果摘要",
        "",
        "说明：以下结论基于正式主链 `f3_history_remote` 模型在固定随机比例留出测试集上的 XGBoost SHAP 值。当前正式主表中实际可用的遥感变量纳入解释；未进入正式主表的 Sentinel-2 字段不生成虚假解释。",
        "",
    ]

    for (target, depth), panel in summary_df.groupby(["target", "depth"], sort=True):
        top = panel.head(3)
        lines.append(f"## {target.upper()} | {depth}")
        if top.empty:
            lines.append("- 没有可用的遥感特征解释结果。")
            lines.append("")
            continue
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['feature']}`: mean|SHAP|={row['mean_abs_shap']:.4f}，"
                f"高值区间相对低值区间的作用为“{row['effect_direction']}”。"
            )
        lines.append("")

    (out_dir / "SHAP结果解读.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    train_mod = load_train_module()
    data = train_mod.prepare_dataset(args.dataset)

    summary_frames: list[pd.DataFrame] = []
    dependence_rows: list[dict[str, object]] = []

    for target_key in args.targets:
        target_col = train_mod.TARGET_MAP[target_key]
        feature_sets = train_mod.build_feature_sets(data, target_col)
        features = feature_sets[args.feature_set]

        for depth in args.depths:
            subset = data[(data["depth"] == depth) & data[target_col].notna()].copy()
            subset = subset.sort_values(["date", "station"]).reset_index(drop=True)
            train, test = train_mod.split_for_evaluation(subset, args)
            if train.empty or test.empty or not features:
                continue

            model = train_mod.make_model(args)
            y_train, _ = train_mod.transform_target(train[target_col], True)
            model.fit(train[features], y_train)

            dtest = xgb.DMatrix(test[features], feature_names=features)
            contrib = model.get_booster().predict(dtest, pred_contribs=True)
            contrib_df = pd.DataFrame(contrib[:, :-1], columns=features, index=test.index)

            combo_meta = {"target": target_key, "depth": depth, "feature_set": args.feature_set}
            summary_df = top_remote_summary(contrib_df, test[features], combo_meta)
            if summary_df.empty:
                continue

            summary_frames.append(summary_df)

            top_feature = summary_df.iloc[0]["feature"]
            feature_values = pd.to_numeric(test[top_feature], errors="coerce")
            shap_values = pd.to_numeric(contrib_df[top_feature], errors="coerce")
            valid = feature_values.notna() & shap_values.notna()
            dep_df = pd.DataFrame(
                {
                    "target": target_key,
                    "depth": depth,
                    "feature": top_feature,
                    "feature_value": feature_values[valid],
                    "shap_value": shap_values[valid],
                }
            )
            dependence_rows.extend(dep_df.to_dict("records"))

    if not summary_frames:
        raise SystemExit("没有生成可用的遥感 SHAP 结果，请检查正式主表中的遥感特征是否成功合并。")

    summary_all = pd.concat(summary_frames, ignore_index=True)
    summary_all.to_csv(args.out_dir / "remote_shap_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dependence_rows).to_csv(args.out_dir / "remote_top_feature_dependence_points.csv", index=False, encoding="utf-8-sig")

    make_remote_bar_grid(summary_all, args.out_dir)
    make_dependence_grid(summary_all, dependence_rows, args.out_dir)
    write_markdown_summary(summary_all, args.out_dir)

    print(f"已生成 SHAP 输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
