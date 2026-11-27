"""比较 SHAP 排名与 ALE 影响曲线：Chl-a 表层严格年份留出预览。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv"
DEFAULT_MODEL_CONFIG = ROOT / "结果/扩展实验/表格预训练算法对比_20260908/portable_package/config.json"
DEFAULT_RUNNER = ROOT / "脚本/04_模型训练/31_run_tabular_pretrained_ab.py"
DEFAULT_OUT_DIR = ROOT / "结果/扩展实验/候选模型稳健性验证_20260914/可解释性预览_Chla表层"

CURVE_FEATURES = ["rain_14d", "wind_speed", "ssrd_mean_7d", "sst_mean_7d"]
FEATURE_LABELS = {
    "rain_14d": "前14天累计降雨",
    "wind_speed": "风速",
    "ssrd_mean_7d": "前7天平均太阳辐射",
    "sst_mean_7d": "前7天平均海表温度",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Chl-a 表层的 SHAP 排名与 ALE 影响曲线预览。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bins", type=int, default=12)
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei", rc={"axes.unicode_minus": False, "text.color": "black"})
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"


def load_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("effect_curve_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模型运行器：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def predict_log(model: xgb.XGBRegressor, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(frame), dtype=float)


def ale_curve(model: xgb.XGBRegressor, data: pd.DataFrame, features: list[str], feature: str, bins: int) -> pd.DataFrame:
    """用测试集条件分布计算一阶 ALE；输出为居中后的 log1p 预测效应。"""
    values = pd.to_numeric(data[feature], errors="coerce")
    valid = values.notna()
    work = data.loc[valid, features].copy()
    values = values.loc[valid]
    if len(work) < 50 or values.nunique() < 4:
        return pd.DataFrame()

    edges = np.unique(values.quantile(np.linspace(0, 1, bins + 1)).to_numpy(dtype=float))
    if len(edges) < 4:
        return pd.DataFrame()

    rows: list[dict[str, float]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        is_last = index == len(edges) - 2
        mask = (values >= lower) & ((values <= upper) if is_last else (values < upper))
        sample = work.loc[mask].copy()
        if sample.empty:
            continue
        low = sample.copy()
        high = sample.copy()
        low[feature] = lower
        high[feature] = upper
        rows.append(
            {
                "feature": feature,
                "x": float((lower + upper) / 2),
                "lower": float(lower),
                "upper": float(upper),
                "count": int(len(sample)),
                "local_difference": float(np.mean(predict_log(model, high) - predict_log(model, low))),
            }
        )

    curve = pd.DataFrame(rows)
    if curve.empty:
        return curve
    curve["ale_log1p"] = curve["local_difference"].cumsum()
    curve["ale_log1p"] -= np.average(curve["ale_log1p"], weights=curve["count"])
    return curve


def plot_shap_ranking(summary: pd.DataFrame, out_dir: Path) -> None:
    panel = summary.head(12).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    ax.barh(panel["feature"], panel["mean_abs_shap"], color="#2f6f8f")
    ax.set_title("SHAP 排名：变量重要性大小，不表示作用方向")
    ax.set_xlabel("平均绝对 SHAP 值（log1p Chl-a）")
    ax.set_ylabel("")
    fig.savefig(out_dir / "图1_SHAP重要性排名_表层Chla.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_ale_curves(curves: pd.DataFrame, out_dir: Path) -> None:
    features = [feature for feature in CURVE_FEATURES if feature in set(curves["feature"])]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    for ax, feature in zip(axes.flatten(), features):
        panel = curves[curves["feature"].eq(feature)].sort_values("x")
        ax.plot(panel["x"], panel["ale_log1p"], marker="o", markersize=3.5, linewidth=1.7, color="#b04a32")
        ax.axhline(0, linewidth=0.8, color="black")
        ax.vlines(panel["x"], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[0] + 0.03 * np.diff(ax.get_ylim())[0], color="#777777", alpha=.45, linewidth=.8)
        ax.set_title(FEATURE_LABELS.get(feature, feature))
        ax.set_xlabel(feature)
        ax.set_ylabel("相对预测效应\nlog(1 + Chl-a)")
    for ax in axes.flatten()[len(features):]:
        ax.set_axis_off()
    fig.suptitle("ALE 影响曲线：变量取值变化对应的局部预测响应", fontsize=14)
    fig.savefig(out_dir / "图2_ALE影响曲线_表层Chla.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir: Path, importance: pd.DataFrame, curves: pd.DataFrame, r2: float) -> None:
    top = importance.head(5)
    lines = [
        "# Chl-a 表层影响曲线预览",
        "",
        "## 分析边界",
        "",
        "- 模型：XGBoost；目标：表层 Chl-a；特征组：f3_history_remote。",
        "- 划分：2014-2022 年训练、2023-2024 年测试。",
        "- 已排除目标自身的历史、滚动和差分特征；同期水质协变量未使用。",
        f"- 本机重训的年份留出测试 R2 为 {r2:.4f}。本图用于解释方法预览，不替代异机固定环境的正式指标。",
        "",
        "## 两种图的作用",
        "",
        "- 图1回答“哪个变量在该模型中较重要”，但不说明升降方向或响应阈值。",
        "- 图2回答“在其他变量保持为该样本条件时，变量在其本地取值范围内改变会怎样改变预测”。",
        "- ALE 曲线不应解释为真实因果关系；它描述的是模型学习到的条件关联。",
        "",
        "## 排名靠前的变量",
        "",
    ]
    for row in top.itertuples():
        lines.append(f"- `{row.feature}`：平均绝对 SHAP 值 {row.mean_abs_shap:.4f}。")
    if not curves.empty:
        lines.extend(["", "## 曲线说明", "", "- 曲线纵轴为相对平均预测的局部效应，零线表示平均水平。", "- 曲线弯折处表示模型对该变量响应变化更明显的区间。"])
    (out_dir / "影响曲线说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    runner = load_runner(args.runner)
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    data = runner.prepare_dataset(args.dataset)
    target_column = runner.TARGET_MAP["chla"]
    features = runner.build_feature_sets(data, target_column)["f3_history_remote"]
    features = [feature for feature in features if not feature.startswith(f"{target_column}_")]

    subset = data[(data["depth"].eq("surface")) & data[target_column].notna()].copy()
    train = subset[subset["year"].le(2022)].sort_values(["date", "station"]).reset_index(drop=True)
    test = subset[subset["year"].between(2023, 2024)].sort_values(["date", "station"]).reset_index(drop=True)
    y_train, inverse = runner.target_transform(train[target_column].to_numpy(dtype=float))
    params = dict(config["model_parameters"]["xgboost"])
    params.update({"random_state": 42, "objective": "reg:squarederror"})
    model = xgb.XGBRegressor(**params)
    model.fit(train[features], y_train)

    prediction_log = predict_log(model, test[features])
    prediction = inverse(prediction_log)
    r2 = float(r2_score(test[target_column].to_numpy(dtype=float), prediction))
    dtest = xgb.DMatrix(test[features], feature_names=features)
    contributions = model.get_booster().predict(dtest, pred_contribs=True)
    importance = pd.DataFrame(
        {"feature": features, "mean_abs_shap": np.abs(contributions[:, :-1]).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    curves = pd.concat(
        [
            curve
            for feature in CURVE_FEATURES
            if feature in features
            for curve in [ale_curve(model, test, features, feature, args.bins)]
            if not curve.empty
        ],
        ignore_index=True,
    )

    importance.to_csv(args.out_dir / "SHAP重要性排名.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(args.out_dir / "ALE影响曲线数据.csv", index=False, encoding="utf-8-sig")
    plot_shap_ranking(importance, args.out_dir)
    plot_ale_curves(curves, args.out_dir)
    write_report(args.out_dir, importance, curves, r2)
    print(f"已生成影响曲线预览：{args.out_dir}")


if __name__ == "__main__":
    main()
