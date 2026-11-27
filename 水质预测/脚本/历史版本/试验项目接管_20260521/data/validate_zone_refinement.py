from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATION_META = PROJECT_ROOT / "data" / "model_input" / "station_meta.csv"
MODEL_DATASET = PROJECT_ROOT / "data" / "interim" / "current" / "current_model_dataset_with_highres_and_mechanism.csv"
COASTAL_STATION = PROJECT_ROOT / "data" / "processed" / "current" / "coastal_distance_station_features.csv"
STATION_DEPTH = PROJECT_ROOT / "data" / "processed" / "current" / "station_depth_features.csv"
MOUTH_VALIDATION = PROJECT_ROOT / "data" / "processed" / "current" / "distance_to_mouth_station_validation.csv"

OUT_STATION = PROJECT_ROOT / "data" / "processed" / "current" / "zone_refinement_station_validation.csv"
OUT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "current" / "zone_refinement_candidate_summary.csv"
OUT_REPORT = PROJECT_ROOT / "experiments" / "reports" / "current" / "zone_refinement_source_validation_report.md"

TARGET_COLUMNS = {
    "chla": "chla_ug_l",
    "secchi": "secchi_depth_m",
    "tn": "tn_mg_l",
    "tp": "tp_mg_l",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 zone refinement 的候选方案")
    parser.add_argument("--station-meta", default=str(STATION_META))
    parser.add_argument("--model-dataset", default=str(MODEL_DATASET))
    parser.add_argument("--coastal-station", default=str(COASTAL_STATION))
    parser.add_argument("--station-depth", default=str(STATION_DEPTH))
    parser.add_argument("--mouth-validation", default=str(MOUTH_VALIDATION))
    parser.add_argument("--station-out", default=str(OUT_STATION))
    parser.add_argument("--summary-out", default=str(OUT_SUMMARY))
    parser.add_argument("--report-out", default=str(OUT_REPORT))
    return parser.parse_args()


def load_station_frame(args: argparse.Namespace) -> pd.DataFrame:
    meta = pd.read_csv(args.station_meta).rename(columns={"Station": "station"})
    coast = pd.read_csv(args.coastal_station)
    depth = pd.read_csv(args.station_depth)
    mouth = pd.read_csv(args.mouth_validation)

    cols = ["station", "distance_to_coast_km", "coastal_exposure_bucket"]
    depth_cols = ["station", "station_depth_approx_m", "is_shallow_station_lt5m", "is_shallow_station_lt10m"]
    mouth_cols = ["station", "distance_to_river_mouth_km", "mouth_distance_bucket", "nearest_mouth_id", "mouth_mapping_ambiguity_flag"]
    out = meta.merge(coast[cols], on="station", how="left", validate="one_to_one")
    out = out.merge(depth[depth_cols], on="station", how="left", validate="one_to_one")
    out = out.merge(mouth[mouth_cols], on="station", how="left", validate="one_to_one")
    return out


def assign_scheme_columns(station_df: pd.DataFrame) -> pd.DataFrame:
    out = station_df.copy()

    out["scheme1_zone_id"] = (
        out["water_zone_en"].str.replace(" ", "_", regex=False)
        + "__mouth__"
        + out["mouth_distance_bucket"].fillna("unknown")
    )
    out["scheme1_zone_name"] = out["water_zone_en"] + " + mouth " + out["mouth_distance_bucket"].fillna("unknown")
    out["scheme1_rule"] = "原 water_zone_en + mouth_distance_bucket"

    out["scheme2_zone_id"] = (
        out["water_zone_en"].str.replace(" ", "_", regex=False)
        + "__coast__"
        + out["coastal_exposure_bucket"].fillna("unknown")
    )
    out["scheme2_zone_name"] = out["water_zone_en"] + " + coast " + out["coastal_exposure_bucket"].fillna("unknown")
    out["scheme2_rule"] = "原 water_zone_en + coastal_exposure_bucket"

    def estuary_position(row: pd.Series) -> str:
        dist = row["distance_to_river_mouth_km"]
        if pd.isna(dist):
            return "unknown"
        if dist < 10:
            return "inner_estuary"
        if dist < 30:
            return "middle_estuary"
        return "outer_estuary"

    out["scheme3_zone_id"] = out.apply(estuary_position, axis=1)
    out["scheme3_zone_name"] = out["scheme3_zone_id"].str.replace("_", " ", regex=False)
    out["scheme3_rule"] = "仅基于 distance_to_river_mouth_km 的 inner/middle/outer estuary 分层"

    macro_map = {
        "Deep Bay": "western_estuarine_inflow",
        "North Western": "western_estuarine_inflow",
        "Western Buffer": "western_estuarine_inflow",
        "Victoria Harbour": "harbour_buffer",
        "Eastern Buffer": "harbour_buffer",
        "Junk Bay": "harbour_buffer",
        "Tolo Harbour and Channel": "northeast_semi_enclosed",
        "Mirs Bay": "northeast_semi_enclosed",
        "Port Shelter": "northeast_semi_enclosed",
        "Southern": "southern_open",
    }
    out["scheme4_zone_id"] = out["water_zone_en"].map(macro_map).fillna("other_open")
    out["scheme4_zone_name"] = out["scheme4_zone_id"].str.replace("_", " ", regex=False)
    out["scheme4_rule"] = "基于现有 water_zone_en 的可解释宏分区：西部河口输入 / 港内缓冲 / 东北半封闭 / 南部外海"
    return out


def zone_long_summary(station_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    scheme_meta = {
        "scheme1": {
            "zone_id_col": "scheme1_zone_id",
            "zone_name_col": "scheme1_zone_name",
            "rule_col": "scheme1_rule",
            "candidate_name": "water_zone_plus_mouth_bucket",
            "target_help": "TN / TP / Secchi 优先",
            "risk": "某些 zone 可能偏碎，但机制解释直观。",
        },
        "scheme2": {
            "zone_id_col": "scheme2_zone_id",
            "zone_name_col": "scheme2_zone_name",
            "rule_col": "scheme2_rule",
            "candidate_name": "water_zone_plus_coastal_bucket",
            "target_help": "Secchi / TN / TP 优先",
            "risk": "与已有 coastal-distance 信息相关性较高，新增信息可能有限。",
        },
        "scheme3": {
            "zone_id_col": "scheme3_zone_id",
            "zone_name_col": "scheme3_zone_name",
            "rule_col": "scheme3_rule",
            "candidate_name": "mouth_distance_estuary_position",
            "target_help": "TN / TP / Secchi，兼顾 Chl-a",
            "risk": "对非典型河口型站点可能解释不足。",
        },
        "scheme4": {
            "zone_id_col": "scheme4_zone_id",
            "zone_name_col": "scheme4_zone_name",
            "rule_col": "scheme4_rule",
            "candidate_name": "macro_hydromorphology_zone",
            "target_help": "四指标统一",
            "risk": "宏分区较粗，可能压缩局地差异。",
        },
    }

    obs = model_df[["obs_id", "sample_id", "station", "depth", "split"] + list(TARGET_COLUMNS.values())].copy()
    obs = obs.merge(station_df, on="station", how="left", validate="many_to_one")

    for scheme_id, meta in scheme_meta.items():
        zone_id_col = meta["zone_id_col"]
        zone_name_col = meta["zone_name_col"]
        rule = station_df[meta["rule_col"]].iloc[0]
        for zone_id, sub in obs.groupby(zone_id_col, dropna=False):
            station_count = sub["station"].nunique()
            observation_count = len(sub)
            supported_depths = sorted([d for d in sub["depth"].dropna().unique().tolist()])
            targets_covered = [
                target for target, col in TARGET_COLUMNS.items() if col in sub.columns and sub[col].notna().sum() > 0
            ]
            zone_name = sub[zone_name_col].dropna().iloc[0] if sub[zone_name_col].notna().any() else str(zone_id)
            too_fragmented = bool(station_count < 3 or observation_count < 200)
            suitable = not too_fragmented
            records.append(
                {
                    "scheme_id": scheme_id,
                    "candidate_name": meta["candidate_name"],
                    "zone_id": zone_id,
                    "zone_name": zone_name,
                    "construction_rule": rule,
                    "station_count": int(station_count),
                    "observation_count": int(observation_count),
                    "depths_covered": ",".join(supported_depths),
                    "targets_covered": ",".join(targets_covered),
                    "sample_too_fragmented": int(too_fragmented),
                    "suitable_for_ab": int(suitable),
                    "likely_help_targets": meta["target_help"],
                    "negative_migration_risk": meta["risk"],
                }
            )
    return pd.DataFrame(records)


def write_report(summary_df: pd.DataFrame, report_out: Path) -> None:
    lines = [
        "# zone refinement source validation 报告",
        "",
        "## 1. 本轮候选方案",
        "",
        "- 方案 1：原 water_zone_en + distance_to_mouth 分箱",
        "- 方案 2：原 water_zone_en + coastal_distance 分箱",
        "- 方案 3：仅基于 distance_to_mouth 的 inner / middle / outer estuary",
        "- 方案 4：基于 water_zone_en 的可解释宏分区",
        "",
        "## 2. 样本碎片化判定",
        "",
        "- 本轮使用保守阈值：`station_count < 3` 或 `observation_count < 200` 记为样本过碎。",
        "",
        "## 3. 各方案汇总",
        "",
    ]

    scheme_rank = []
    for scheme_id, sub in summary_df.groupby("scheme_id"):
        total_zones = len(sub)
        fragmented = int(sub["sample_too_fragmented"].sum())
        suitable_rate = float(sub["suitable_for_ab"].mean())
        scheme_rank.append((scheme_id, fragmented, suitable_rate))
        lines.extend(
            [
                f"### {scheme_id}",
                "",
                f"- zone 数：{total_zones}",
                f"- 样本过碎 zone 数：{fragmented}",
                f"- 适合进入 A/B 的 zone 比例：{suitable_rate:.2%}",
                "",
            ]
        )

    scheme_rank.sort(key=lambda x: (x[1], -x[2], x[0]))
    best_scheme = scheme_rank[0][0] if scheme_rank else "unknown"
    lines.extend(
        [
            "## 4. 初步建议",
            "",
            f"- 本轮从样本规模与解释性平衡来看，初步最适合进入 A/B 的方案是：`{best_scheme}`。",
            "- 若后续要与 distance-to-river-mouth 组合，优先考虑分区规则最简单且不过碎的方案。",
        ]
    )
    report_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    station_df = load_station_frame(args)
    station_df = assign_scheme_columns(station_df)
    model_df = pd.read_csv(args.model_dataset, low_memory=False)
    summary_df = zone_long_summary(station_df, model_df)

    Path(args.station_out).parent.mkdir(parents=True, exist_ok=True)
    station_df.to_csv(args.station_out, index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.summary_out, index=False, encoding="utf-8-sig")
    write_report(summary_df, Path(args.report_out))
    print(
        json.dumps(
            {
                "station_count": int(len(station_df)),
                "scheme_count": 4,
                "summary_rows": int(len(summary_df)),
                "best_scheme": summary_df.groupby("scheme_id")["sample_too_fragmented"].sum().sort_values().index[0],
                "station_out": str(args.station_out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
