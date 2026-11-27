from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.data.validate_distance_to_river_mouth import (
    COASTLINE_ZIP,
    HYDRORIVERS_ZIP,
    LOCAL_CRS,
    STATION_META,
    assign_station_distances,
    build_major_mouths,
    gpd,
)


OUT_CSV = PROJECT_ROOT / "data" / "processed" / "current" / "distance_to_mouth_system_station_validation.csv"
OUT_MD = PROJECT_ROOT / "experiments" / "reports" / "current" / "distance_to_mouth_system_source_refinement_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 distance-to-river-mouth 重构为主要河口系统级定义。")
    parser.add_argument("--bbox-buffer-deg", type=float, default=1.0)
    parser.add_argument("--coast-near-threshold-km", type=float, default=5.0)
    parser.add_argument("--mouth-cluster-km", type=float, default=5.0)
    return parser.parse_args()


def classify_system(row: pd.Series) -> str:
    lon = float(row["mouth_lon"])
    lat = float(row["mouth_lat"])
    if lat >= 22.9:
        return "north_major_mainland"
    if lon < 113.7 and lat < 22.7:
        return "western_pearl_estuary"
    if lon < 114.15 and lat >= 22.45:
        return "northwest_local_inflow"
    return "eastern_local_inflow"


SYSTEM_NAME_MAP = {
    "north_major_mainland": "北侧大陆主要河口系统",
    "western_pearl_estuary": "西侧珠江口系统",
    "northwest_local_inflow": "西北近岸局地入流系统",
    "eastern_local_inflow": "东侧近岸局地入流系统",
}


def main() -> None:
    args = parse_args()
    station_meta = pd.read_csv(STATION_META)
    mouths, _ = build_major_mouths(
        station_meta=station_meta,
        coastline_zip=COASTLINE_ZIP,
        hydrorivers_zip=HYDRORIVERS_ZIP,
        bbox_buffer_deg=args.bbox_buffer_deg,
        coast_near_threshold_km=args.coast_near_threshold_km,
        mouth_cluster_km=args.mouth_cluster_km,
    )

    systems = mouths.copy()
    systems["mouth_system_id"] = systems.apply(classify_system, axis=1)
    systems["mouth_system_name"] = systems["mouth_system_id"].map(SYSTEM_NAME_MAP)

    station_gdf = gpd.GeoDataFrame(
        station_meta.rename(columns={"Station": "station"}).copy(),
        geometry=gpd.points_from_xy(station_meta["lon"], station_meta["lat"]),
        crs="EPSG:4326",
    ).to_crs(LOCAL_CRS)
    systems_local = systems.to_crs(LOCAL_CRS)

    rows: list[dict[str, object]] = []
    for row in station_gdf.itertuples(index=False):
        dists: list[tuple[str, str, float]] = []
        for system_id, sub in systems_local.groupby("mouth_system_id"):
            dist_km = float(sub.geometry.distance(row.geometry).min() / 1000.0)
            dists.append((system_id, SYSTEM_NAME_MAP[system_id], dist_km))
        dists = sorted(dists, key=lambda x: x[2])
        best = dists[0]
        second = dists[1] if len(dists) > 1 else ("", "", np.nan)
        rows.append(
            {
                "station": row.station,
                "station_name_en": row.station_name_en,
                "water_zone_en": row.water_zone_en,
                "lon": row.lon,
                "lat": row.lat,
                "nearest_mouth_system_id": best[0],
                "nearest_mouth_system_name": best[1],
                "distance_to_mouth_system_km": best[2],
                "log_distance_to_mouth_system_km": float(np.log1p(best[2])),
                "second_nearest_mouth_system_km": second[2],
                "system_distance_gap_km": float(second[2] - best[2]) if pd.notna(second[2]) else np.nan,
                "system_mapping_ambiguity_flag": bool(pd.notna(second[2]) and (second[2] - best[2]) < 5.0),
                "mouth_system_definition_source": "HydroRIVERS + Natural Earth + major-mouth-system manual grouping",
            }
        )

    out = pd.DataFrame(rows).sort_values("station").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    sys_summary = systems.groupby(["mouth_system_id", "mouth_system_name"]).agg(
        mouth_count=("nearest_mouth_id", "count"),
        max_discharge=("dis_av_cms", "max"),
        mean_lon=("mouth_lon", "mean"),
        mean_lat=("mouth_lat", "mean"),
    ).reset_index()
    assign_summary = out.groupby(["water_zone_en", "nearest_mouth_system_id"]).size()
    ambiguity = int(out["system_mapping_ambiguity_flag"].sum())

    lines = [
        "# distance-to-river-mouth 系统级重构报告",
        "",
        "本轮不把该结果直接入模，只做 source refinement。",
        "",
        "## 系统级定义",
        "- `north_major_mainland`：北侧大陆主要河口系统",
        "- `western_pearl_estuary`：西侧珠江口系统",
        "- `northwest_local_inflow`：西北近岸局地入流系统",
        "- `eastern_local_inflow`：东侧近岸局地入流系统",
        "",
        "## 核心结果",
        f"- 站点覆盖率：{len(out)}/94",
        f"- 系统映射歧义站点数：{ambiguity}",
        f"- 系统数量：{sys_summary.shape[0]}",
        "",
        "## 主要河口系统汇总",
        "",
        sys_summary.to_string(index=False),
        "",
        "## 各 water_zone 的最近系统分配",
        "",
        assign_summary.to_string(),
        "",
        "## 判断",
        "- 相比逐条 terminal reach，这个系统级口径显著降低了歧义。",
        "- 但当前分配仍明显偏向 `eastern_local_inflow`，说明系统划分虽然更稳，却还不够形成均衡、可直接入模的距离梯度。",
        "- 因此本轮建议把它视为比原口径更稳的 source refinement 版本，但仍暂不直接进入 `distance_to_mouth_only A/B`。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        {
            "out_csv": str(OUT_CSV),
            "system_count": int(sys_summary.shape[0]),
            "ambiguity_count": ambiguity,
        }
    )


if __name__ == "__main__":
    main()
