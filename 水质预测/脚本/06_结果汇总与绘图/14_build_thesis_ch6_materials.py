from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_REPORT_DIR = Path("结果/正式结果/full_station")
DEFAULT_DIAG_DIR = Path("结果/诊断结果/chla_surface_diagnostics")
DEFAULT_TABLE_DIR = Path("结果/论文图表/论文表格_随机比例划分_20260905/第六章")
DEFAULT_CATALOG_PATH = Path("结果/论文图表/论文表格_随机比例划分_20260905/论文图表总清单.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成论文第六章结论证据表、数据缺口表和论文图表总清单。")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--diag-dir", type=Path, default=DEFAULT_DIAG_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少输入文件: {path}")
    return pd.read_csv(path, low_memory=False)


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def build_conclusion_evidence(best: pd.DataFrame, fusion: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    best = best.copy()
    best["指标"] = best["target"].str.upper()
    indicator_best = (
        best.sort_values(["指标", "r2"], ascending=[True, False])
        .groupby("指标", as_index=False)
        .first()[["指标", "depth", "feature_set", "r2", "rmse"]]
    )
    rows: list[dict[str, object]] = []
    for _, row in indicator_best.iterrows():
        rows.append(
            {
                "结论主题": f"{row['指标']} 的最佳正式随机留出表现",
                "关键证据": f"最佳水层={row['depth']}；特征组={row['feature_set']}；R2={row['r2']:.3f}；RMSE={row['rmse']:.3f}",
                "证据来源": "表3-4 与图3-2至图3-5",
                "适用边界": "固定随机比例留出；不使用同期其他水质指标；不代表未来年份外推能力。",
            }
        )
    gain_mean = float(fusion["history_remote_minus_history_r2"].mean())
    positive_count = int((fusion["history_remote_minus_history_r2"] > 0).sum())
    rows.append(
        {
            "结论主题": "遥感信息的总体增益",
            "关键证据": f"融合组相对历史组的平均 R2 变化={gain_mean:.3f}；正增益任务数={positive_count}/{len(fusion)}。",
            "证据来源": "图4-5、图4-6与表4-3",
            "适用边界": "不同任务与水层存在差异，不应解释为统一且稳定的提升。",
        }
    )
    quality = dict(zip(coverage["metric"], coverage["value"]))
    rows.append(
        {
            "结论主题": "正式主表中的遥感覆盖边界",
            "关键证据": f"Landsat 可用样本比例={quality.get('landsat_available_share', float('nan')):.3f}；Sentinel-2 特征字段进入正式主表={quality.get('s2_feature_fields_present', float('nan')):.0f}。",
            "证据来源": "图4-1与表4-1",
            "适用边界": "正式解释只针对实际进入主表和模型的遥感变量。",
        }
    )
    return pd.DataFrame(rows)


def build_data_gap_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["准确采样时刻", "短时间窗辐射、风浪、潮汐与遥感近同日匹配误差", "Chl-a 表层及高值事件", "高", "获取采样时分与现场测量时刻"],
            ["河流流量、排放与水位流速", "陆源输入和潮汐交换过程未被直接刻画", "TN、TP 与近岸高值事件", "高", "接入可复现的水文或排放过程数据"],
            ["垂向温盐、流速与底层过程", "中层和底层仅由表面遥感间接代理", "中层、底层 Chl-a、TN、TP", "高", "增加剖面观测或水动力背景变量"],
            ["严格近同日高质量遥感影像", "云、混合像元与低有效像元降低遥感增益", "Chl-a、Secchi", "中", "优化大气校正、时间窗口和像元质量控制"],
            ["站点级河口羽状流影响量化", "海区类别无法完全替代淡水输入路径", "TN、TP、Secchi", "中", "构建距河口、径流与羽状流代理的受控 A/B"],
        ],
        columns=["数据缺口", "对机制表达的影响", "主要受影响任务", "优先级", "后续建议"],
    )


def build_catalog(catalog_path: Path) -> pd.DataFrame:
    root = catalog_path.parent.parent
    figure_root = root / "论文图版_随机比例划分_20260905"
    table_root = root / "论文表格_随机比例划分_20260905"
    rows: list[dict[str, object]] = []
    for path in sorted(figure_root.rglob("*.png")):
        chapter = path.parent.name
        rows.append({"类型": "图", "章节目录": chapter, "文件名": path.name, "相对路径": str(path.relative_to(root.parent))})
    for path in sorted(table_root.rglob("*.csv")):
        if path == catalog_path:
            continue
        chapter = path.parent.name
        rows.append({"类型": "表", "章节目录": chapter, "文件名": path.name, "相对路径": str(path.relative_to(root.parent))})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    best = read_csv(args.report_dir / "full_station_best_by_target_depth.csv")
    fusion = read_csv(args.report_dir / "full_station_fusion_gain.csv")
    coverage = read_csv(args.diag_dir / "coverage_summary.csv")
    write_csv(build_conclusion_evidence(best, fusion, coverage), args.table_dir / "表6-1_主要结论与对应证据.csv")
    write_csv(build_data_gap_table(), args.table_dir / "表6-2_数据缺口与后续改进方向.csv")
    write_csv(build_catalog(args.catalog_path), args.catalog_path)
    print(f"已生成第六章表格与总清单: {args.table_dir} / {args.catalog_path}")


if __name__ == "__main__":
    main()
