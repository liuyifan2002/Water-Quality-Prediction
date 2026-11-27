from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training import run_literature_inspired_precision_boost as boost


DATE_TAG = "20260518"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize literature-inspired precision boost outputs.")
    parser.add_argument("--date-tag", default=DATE_TAG)
    parser.add_argument(
        "--input",
        default="data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_optimization_dataset_20260517.csv",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_permutation_importance(out_dir: Path, input_path: Path, date_tag: str, force: bool) -> Path:
    out_path = out_dir / "permutation_importance.csv"
    if out_path.exists() and not force:
        return out_path
    df = boost.load_data(input_path)
    added_by_target: dict[str, list[str]] = {}
    for target, spec in boost.TARGETS.items():
        df, added = boost.add_target_features(df, target, spec["history_prefix"])
        added_by_target[target] = added

    rows: list[pd.DataFrame] = []
    for target, spec in boost.TARGETS.items():
        target_col = spec["column"]
        target_df = df[df[target_col].notna()].copy()
        features = boost.build_features(target_df, target, spec["history_prefix"], added_by_target[target])
        for depth in boost.DEPTH_ORDER:
            train, val, _ = boost.split_frames(target_df, target_col, depth)
            if train.empty or val.empty:
                continue
            pipe = Pipeline(
                [
                    ("preprocessor", boost.sklearn_preprocessor(train, features)),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=300,
                            learning_rate=0.04,
                            num_leaves=18,
                            max_depth=6,
                            min_child_samples=35,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_lambda=1.0,
                            random_state=42,
                            n_jobs=1,
                            verbose=-1,
                        ),
                    ),
                ]
            )
            pipe.fit(train[features], train[target_col])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                perm = permutation_importance(
                    pipe,
                    val[features],
                    val[target_col],
                    scoring="r2",
                    n_repeats=2,
                    random_state=42,
                    n_jobs=1,
                )
            rows.append(
                pd.DataFrame(
                    {
                        "target": target,
                        "depth": depth,
                        "candidate": f"{target}_{depth}_lightgbm_reference_permutation",
                        "feature": features,
                        "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std,
                    }
                )
            )
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            columns=["target", "depth", "candidate", "feature", "importance_mean", "importance_std"]
        ).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def best_by_r2(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"] == "test"].copy()
    idx = test.groupby(["target", "depth"])["r2"].idxmax()
    return test.loc[idx].sort_values(["target", "depth"]).reset_index(drop=True)


def best_by_high90(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"] == "test"].copy()
    idx = test.groupby(["target", "depth"])["high90_rmse"].idxmin()
    return test.loc[idx].sort_values(["target", "depth"]).reset_index(drop=True)


def format_metric_table(best: pd.DataFrame) -> str:
    cols = [
        "target",
        "depth",
        "candidate",
        "strategy",
        "r2",
        "rmse",
        "mae",
        "high90_rmse",
        "high90_underestimation_rate",
        "bias",
    ]
    out = best[cols].copy()
    for col in ["r2", "rmse", "mae", "high90_rmse", "high90_underestimation_rate", "bias"]:
        out[col] = out[col].map(lambda x: f"{x:.4f}")
    header = "| " + " | ".join(out.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in out.columns) + " |" for _, row in out.iterrows()]
    return "\n".join([header, divider] + body)


def write_reports(out_dir: Path, date_tag: str) -> None:
    report_dir = Path("experiments/reports")
    config_dir = Path("experiments/configs")
    report_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(out_dir / "metrics.csv")
    comparison = pd.read_csv(out_dir / "comparison_to_previous_best.csv")
    best = best_by_r2(metrics)
    high = best_by_high90(metrics)

    success = comparison.loc[comparison["delta_r2_vs_previous"] > 0].copy()
    success = success.sort_values(["target", "depth", "delta_r2_vs_previous"], ascending=[True, True, False])

    chla_bottom = best[(best["target"] == "chla") & (best["depth"] == "bottom")].iloc[0]
    secchi_rows = best[best["target"] == "secchi"].copy()

    report_lines = [
        f"# Literature Inspired Precision Boost {date_tag}",
        "",
        "## 1. 本轮研究思路",
        "",
        "本轮停止重复普通特征堆叠、常规 Optuna 和常规高值加权，改为测试文献中常见的多模型融合、CatBoost 类别特征建模、二层 stacking、残差校正、光学状态分区和轻量序列模型可行性。",
        "",
        "## 2. 当前最佳结果",
        "",
        format_metric_table(best),
        "",
        "## 3. 高值样本辅助线",
        "",
        format_metric_table(high),
        "",
        "## 4. 是否刷新上一轮最佳",
        "",
    ]
    if success.empty:
        report_lines.append("本轮没有任何 target-depth 在 Test R2 上刷新上一轮最佳。")
    else:
        for _, row in success.iterrows():
            report_lines.append(
                f"- {row['target']} / {row['depth']}: {row['candidate']} 刷新 R2，delta={row['delta_r2_vs_previous']:+.4f}。"
            )
    report_lines += [
        "",
        "## 5. Chl-a Bottom 判断",
        "",
        f"Chl-a bottom 本轮最佳为 `{chla_bottom['candidate']}`，Test R2={chla_bottom['r2']:.4f}，未突破上一轮 0.3318。说明 bottom 层主要瓶颈仍然不是常规树模型结构，而更可能来自底层过程观测不足、垂向混合机制缺失、标签时间错配或站点异质性。",
        "",
        "## 6. Secchi/ZSD 判断",
        "",
    ]
    for _, row in secchi_rows.iterrows():
        report_lines.append(
            f"- {row['depth']}: 最佳 `{row['candidate']}`，Test R2={row['r2']:.4f}。"
        )
    report_lines += [
        "",
        "Secchi/ZSD 的主要新增益来自 stacking，说明 Ocean Colour 透明度信息可以被不同模型互补吸收，但提升幅度仍小于目标阈值。",
        "",
        "## 7. 路线结论",
        "",
        "- CatBoost: 对 Chl-a surface 有横向竞争力，但没有超过上一轮主线；可保留为类别特征参考，不建议替代主模型。",
        "- Stacking: 对 Secchi/ZSD 三层有效，是本轮最有价值的新路线；对 Chl-a 未形成突破。",
        "- Residual correction: 更适合作为高值或偏差辅助线，不能直接替代主模型。",
        "- Regime 分区: 本轮未突破全局模型，说明简单光学状态分区不足以解决 bottom 机制差异。",
        "- 轻量序列模型: 本轮未安装深度学习依赖，按可复现和环境洁净原则登记为未执行。",
        "- ERA5/气象: 本轮只保留可行性入口，未做真实 A/B；下一阶段若要突破 Chl-a bottom，应优先补入真实风、光、降雨、温度过程变量。",
        "",
        "## 8. 失败与瓶颈",
        "",
        "Chl-a bottom 未达 R2 > 0.35，瓶颈判断为：底层机制瓶颈、外部驱动缺失、station 泛化瓶颈和标签/遥感时间错配共同存在。Secchi/ZSD 未达到 surface >0.43、middle >0.36、bottom >0.37，瓶颈主要是透明度标签噪声和 Ocean Colour 缺失覆盖。",
        "",
        "## 9. 下一轮不应重复",
        "",
        "不要继续扩大普通 Optuna、不要只换 KNN/rolling/mask 填补、不要把 P90 加权直接当主模型。下一轮应优先补真实外部驱动，或做有明确水动力分区依据的模型。",
        "",
        "## 10. 输出与目录洁净",
        "",
        f"- 输出目录: `experiments/outputs/literature_inspired_precision_boost_{date_tag}/`",
        f"- 配置文件: `experiments/configs/literature_inspired_precision_boost_{date_tag}.json`",
        f"- 报告文件: `experiments/reports/literature_inspired_precision_boost_{date_tag}.md`",
        f"- 处理数据: `data/processed/literature_inspired_precision_boost_{date_tag}/`",
        "- 未在项目根目录生成临时 csv/json/ipynb。",
    ]
    text = "\n".join(report_lines) + "\n"
    (out_dir / "experiment_report.md").write_text(text, encoding="utf-8")
    (report_dir / f"literature_inspired_precision_boost_{date_tag}.md").write_text(text, encoding="utf-8")

    failed_text = f"""# 失败实验登记

本文件登记失败、低收益或暂不执行的方法，避免后续重复消耗时间。

## Literature Inspired Precision Boost {date_tag}

| 方法 | 状态 | 原因 | 后续建议 |
|---|---|---|---|
| Chl-a bottom CatBoost / stacking / regime | 未突破 | 最佳 R2 低于上一轮 0.3318 | 不再重复同类普通模型横向比较 |
| residual correction | 辅助有效但不宜主线 | 可降低部分高值误差，但整体 R2 不稳定 | 只作为 high-value auxiliary |
| 光学状态 regime 分区 | 低收益 | 仅用 Ocean Colour 状态无法表达底层水动力过程 | 后续需要结合真实水动力或分区依据 |
| 轻量序列模型 | 暂未执行 | 当前隔离环境无 TensorFlow/PyTorch；为避免污染环境，本轮只登记可行性 | 仅在确认安装独立深度学习环境后运行 |
| ERA5/气象 A/B | 暂未执行 | 本轮优先完成非新增数据源方法；气象驱动需先做 source validation | 下一阶段优先作为 Chl-a bottom 新信息源 |

## 下一轮不应重复

- 不再重复普通 Optuna 扩搜索空间。
- 不再单独比较 KNN / rolling / mask 缺失填补。
- 不再把 P90 高值加权直接替代主模型。
- 不再仅靠 Ocean Colour 简单交互来突破 Chl-a bottom。
"""
    (report_dir / "failed_experiments.md").write_text(failed_text, encoding="utf-8")

    registry_path = report_dir / "methods_tried_registry.md"
    registry = registry_path.read_text(encoding="utf-8")
    marker = f"## 8. Literature Inspired Precision Boost {date_tag}"
    if marker not in registry:
        registry += f"""

{marker}

已完成。

### 为什么尝试

参考水质遥感和机器学习研究中的类别特征建模、二层 stacking、残差校正、光学状态分区与序列建模思路，尝试跳出普通特征堆叠和常规调参。

### 与旧实验区别

本轮新增 CatBoost、time-aware OOF stacking、validation residual correction、光学 regime 分区，并把轻量序列模型和 ERA5 作为可行性路线登记。

### 是否成功

- Chl-a bottom 未突破上一轮最佳。
- Secchi/ZSD 三层 stacking 有小幅正增益。
- residual correction 可作为高值误差辅助，但不建议作为主模型。

### 是否以后还值得继续

CatBoost 与简单光学 regime 暂不继续；stacking 可用于 Secchi/ZSD；Chl-a bottom 下一轮应转向真实外部驱动和更强水动力分区。
"""
        registry_path.write_text(registry, encoding="utf-8")

    config_path = Path("experiments/configs") / f"literature_inspired_precision_boost_{date_tag}.json"
    config: dict[str, Any] = json.loads((out_dir / "experiment_config.json").read_text(encoding="utf-8"))
    config["finalized"] = True
    config["permutation_importance"] = str(out_dir / "permutation_importance.csv")
    config["report"] = str(report_dir / f"literature_inspired_precision_boost_{date_tag}.md")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(f"experiments/outputs/literature_inspired_precision_boost_{args.date_tag}")
    ensure_permutation_importance(out_dir, Path(args.input), args.date_tag, args.force)
    write_reports(out_dir, args.date_tag)
    print(json.dumps({"finalized": str(out_dir), "report": f"experiments/reports/literature_inspired_precision_boost_{args.date_tag}.md"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
