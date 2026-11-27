"""汇总精度提升扩展实验的跨阶段结果，不改动正式结果。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("结果/扩展实验/精度提升_20260905")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总指定扩展实验根目录中的精度筛选结果。")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="待汇总的扩展实验根目录。")
    return parser.parse_args()


def evaluation_lines(evaluation_mode: str) -> list[str]:
    if evaluation_mode == "random":
        return [
            "- 候选选择：固定外层随机划分的开发集（80%）内，再按 random_state=43 随机分为 75%/25%（全样本约 60%/20%）。",
            "- 最终评估：固定外层随机 80%/20%，random_state=42；所有阶段复用同一外层测试集。",
            "- 本结果衡量的是同分布插补/监测更新能力，而不是严格的未来年份时间外推能力。",
        ]
    return [
        "- 候选选择：2014–2020 训练，2021–2022 验证。",
        "- 最终外推：2014–2022 训练，2023–2024 测试。",
        "- 本结果衡量严格时间外推能力。",
    ]


def main() -> None:
    args = parse_args()
    root = args.root
    stage_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name[:2] in {"01", "02", "03", "04", "05"}
    )
    frames: list[pd.DataFrame] = []
    for stage_dir in stage_dirs:
        path = stage_dir / "精度提升_全部候选指标.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["source_stage"] = stage_dir.name
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"未在 {root} 找到阶段结果。")

    all_metrics = pd.concat(frames, ignore_index=True)
    final_metrics = all_metrics.loc[all_metrics["stage"].eq("final_test")].copy()
    selection_metrics = all_metrics.loc[all_metrics["stage"].eq("selection_validation")].copy()
    evaluation_modes = final_metrics.get("evaluation_mode", pd.Series(dtype=str)).dropna().unique().tolist()
    evaluation_mode = evaluation_modes[0] if len(evaluation_modes) == 1 else "mixed"
    exploratory_best = (
        final_metrics.sort_values(["target", "depth", "r2"], ascending=[True, True, False])
        .groupby(["target", "depth"], as_index=False)
        .first()
    )
    validation_best = (
        selection_metrics.sort_values(["target", "depth", "r2"], ascending=[True, True, False])
        .groupby(["target", "depth"], as_index=False)
        .first()
    )
    recommended = validation_best.merge(
        final_metrics[["target", "depth", "source_stage", "candidate", "r2", "rmse", "mae", "nse", "bias"]],
        on=["target", "depth", "source_stage", "candidate"], how="left", suffixes=("_validation", "_test"),
    ).rename(columns={"r2_validation": "validation_r2", "r2_test": "test_r2"})

    root.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(root / "精度提升_跨阶段全部指标.csv", index=False, encoding="utf-8-sig")
    exploratory_best.to_csv(root / "精度提升_跨阶段探索最优组合.csv", index=False, encoding="utf-8-sig")
    recommended.to_csv(root / "精度提升_跨阶段验证期推荐组合.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 固定随机划分精度提升扩展实验汇总（2026-09-05）",
        "",
        "## 评价设置",
        "",
        *evaluation_lines(evaluation_mode),
        "- 不使用同日其他水质指标；历史特征限定为同站点、同水层且日期更早的目标观测。",
        "- 本文件所有结果均为扩展实验，不替换 `full_station` 正式主链。",
        "",
        "## 跨阶段探索最优（按最终测试 R² 排序）",
        "",
        "此表用于定位值得复验的组合。虽然外层测试集固定且未参与候选选择，但多阶段复用同一测试集会产生比较选择偏差，不能直接作为正式主链升级依据。",
        "",
        "| 指标 | 水层 | 方法阶段 | 候选组合 | R² | RMSE | MAE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in exploratory_best.sort_values(["target", "depth"]).iterrows():
        lines.append(
            f"| {row['target']} | {row['depth']} | {row['source_stage']} | {row['candidate']} | "
            f"{row['r2']:.4f} | {row['rmse']:.4f} | {row['mae']:.4f} |"
        )
    lines.extend([
        "",
        "## 结果解读",
        "",
        "- 以“跨阶段验证期推荐组合”作为候选入围的主要依据；它由内层验证集选择，避免直接用外层测试集挑模型。",
        "- “跨阶段探索最优组合”仅用于发现信号，不应被解释为独立泛化性能。",
        "- 下一步应将每个任务的少量入围组合送入重复随机划分或严格时间外推复验；只有稳定优于基线，才考虑改动正式主链。",
        "",
    ])
    (root / "精度提升_实验结论.md").write_text("\n".join(lines), encoding="utf-8")
    print(exploratory_best[["target", "depth", "source_stage", "candidate", "r2", "rmse", "mae"]].round(4).to_string(index=False))
    print(f"已写出：{root}")


if __name__ == "__main__":
    main()
