from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATION_META = PROJECT_ROOT / "data" / "model_input" / "station_meta.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "manual" / "river_context"
RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR = RAW_DIR / "extracted"

COASTLINE_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"
COASTLINE_ZIP = RAW_DIR / "ne_10m_coastline.zip"
HYDRORIVERS_URL = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_as_shp.zip"
HYDRORIVERS_ZIP = RAW_DIR / "HydroRIVERS_v10_as_shp.zip"

OUT_CSV = PROJECT_ROOT / "data" / "processed" / "current" / "distance_to_mouth_station_validation.csv"
OUT_MD = PROJECT_ROOT / "experiments" / "reports" / "current" / "distance_to_mouth_source_validation_report.md"

LOCAL_CRS = "EPSG:2326"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证香港近岸站点 distance-to-river-mouth 静态空间特征")
    parser.add_argument("--station-meta", default=str(STATION_META))
    parser.add_argument("--coastline-url", default=COASTLINE_URL)
    parser.add_argument("--coastline-zip", default=str(COASTLINE_ZIP))
    parser.add_argument("--hydrorivers-url", default=HYDRORIVERS_URL)
    parser.add_argument("--hydrorivers-zip", default=str(HYDRORIVERS_ZIP))
    parser.add_argument("--station-out", default=str(OUT_CSV))
    parser.add_argument("--report-out", default=str(OUT_MD))
    parser.add_argument("--bbox-buffer-deg", type=float, default=1.0)
    parser.add_argument("--coast-near-threshold-km", type=float, default=5.0)
    parser.add_argument("--mouth-cluster-km", type=float, default=5.0)
    return parser.parse_args()


def download_if_needed(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    import requests

    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resp.content)


def extract_zip_if_needed(zip_path: Path, expected_member: str) -> Path:
    out_path = EXTRACT_DIR / expected_member
    if out_path.exists():
        return out_path
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf:
        zf.extractall(EXTRACT_DIR)
    if not out_path.exists():
        raise FileNotFoundError(f"未能从压缩包解出预期文件: {expected_member}")
    return out_path


def get_bbox(stations: pd.DataFrame, buffer_deg: float) -> tuple[float, float, float, float]:
    minx = stations["lon"].min() - buffer_deg
    maxx = stations["lon"].max() + buffer_deg
    miny = stations["lat"].min() - buffer_deg
    maxy = stations["lat"].max() + buffer_deg
    return minx, miny, maxx, maxy


def cluster_points(points_xy: np.ndarray, eps_km: float) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    if len(points_xy) == 0:
        return np.array([], dtype=int)
    model = DBSCAN(eps=eps_km * 1000.0, min_samples=1)
    return model.fit_predict(points_xy)


def pick_best_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    upper_map = {col.upper(): col for col in df.columns}
    for cand in candidates:
        if cand.upper() in upper_map:
            return upper_map[cand.upper()]
    return None


def safe_linestring_endpoints(geom) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        if len(coords) < 2:
            return None
        return (coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])
    if geom.geom_type == "MultiLineString":
        longest = max(list(geom.geoms), key=lambda g: g.length, default=None)
        if longest is None:
            return None
        coords = list(longest.coords)
        if len(coords) < 2:
            return None
        return (coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])
    return None


def build_major_mouths(station_meta: pd.DataFrame, coastline_zip: Path, hydrorivers_zip: Path, bbox_buffer_deg: float, coast_near_threshold_km: float, mouth_cluster_km: float) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    bbox = get_bbox(station_meta, bbox_buffer_deg)
    bbox_geom = box(*bbox)

    coastline_path = extract_zip_if_needed(coastline_zip, "ne_10m_coastline.shp")
    hydrorivers_path = extract_zip_if_needed(hydrorivers_zip, "HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shp")

    coastline = gpd.read_file(coastline_path).to_crs("EPSG:4326")
    rivers = gpd.read_file(hydrorivers_path).to_crs("EPSG:4326")

    rivers = rivers[rivers.geometry.intersects(bbox_geom)].copy()
    coast_local = coastline.to_crs(LOCAL_CRS)
    coast_union = coast_local.geometry.unary_union

    dis_col = pick_best_column(rivers, ["DIS_AV_CMS", "DIS_AV_CMS2"])
    ord_col = pick_best_column(rivers, ["ORD_STRA", "ORD_FLOW"])
    len_col = pick_best_column(rivers, ["LENGTH_KM", "LENGTH_KM2"])

    rows: list[dict[str, object]] = []
    river_local = rivers.to_crs(LOCAL_CRS)
    for idx, row in river_local.iterrows():
        endpoints = safe_linestring_endpoints(row.geometry)
        if endpoints is None:
            continue
        start_pt = gpd.GeoSeries.from_xy([endpoints[0][0]], [endpoints[0][1]], crs=LOCAL_CRS).iloc[0]
        end_pt = gpd.GeoSeries.from_xy([endpoints[1][0]], [endpoints[1][1]], crs=LOCAL_CRS).iloc[0]
        d_start = start_pt.distance(coast_union) / 1000.0
        d_end = end_pt.distance(coast_union) / 1000.0
        if d_start <= d_end:
            mouth_geom = start_pt
            mouth_dist = d_start
            mouth_xy = endpoints[0]
        else:
            mouth_geom = end_pt
            mouth_dist = d_end
            mouth_xy = endpoints[1]
        rows.append(
            {
                "row_id": idx,
                "mouth_distance_to_coast_km": mouth_dist,
                "geometry": mouth_geom,
                "mouth_x": mouth_xy[0],
                "mouth_y": mouth_xy[1],
                "dis_av_cms": float(row[dis_col]) if dis_col and pd.notna(row[dis_col]) else np.nan,
                "river_order": int(row[ord_col]) if ord_col and pd.notna(row[ord_col]) else np.nan,
                "length_km": float(row[len_col]) if len_col and pd.notna(row[len_col]) else np.nan,
            }
        )

    candidates = gpd.GeoDataFrame(rows, geometry="geometry", crs=LOCAL_CRS)
    if candidates.empty:
        raise RuntimeError("HydroRIVERS 未生成任何河口候选点。")

    candidates = candidates[candidates["mouth_distance_to_coast_km"] <= coast_near_threshold_km].copy()
    if candidates.empty:
        raise RuntimeError("HydroRIVERS 候选河口点全部不满足近岸阈值。")

    if "dis_av_cms" in candidates.columns:
        discharge_threshold = np.nanquantile(candidates["dis_av_cms"].fillna(0.0), 0.60)
        candidates = candidates[
            (candidates["dis_av_cms"].fillna(0.0) >= discharge_threshold)
            | (candidates["river_order"].fillna(0.0) >= 4)
        ].copy()
    if candidates.empty:
        raise RuntimeError("主要河口筛选后为空。")

    clusters = cluster_points(candidates[["mouth_x", "mouth_y"]].to_numpy(), mouth_cluster_km)
    candidates["mouth_cluster_id"] = clusters
    candidates["major_score"] = (
        np.log1p(candidates["dis_av_cms"].fillna(0.0)) * 3.0
        + candidates["river_order"].fillna(0.0)
        + np.log1p(candidates["length_km"].fillna(0.0))
    )

    picked = (
        candidates.sort_values(["mouth_cluster_id", "major_score"], ascending=[True, False])
        .groupby("mouth_cluster_id", as_index=False)
        .head(1)
        .copy()
    )
    picked["nearest_mouth_id"] = picked["mouth_cluster_id"].apply(lambda x: f"mouth_{int(x):02d}")
    picked["mouth_definition_source"] = "HydroRIVERS Asia terminal reaches + Natural Earth coastline + major-mouth filter"
    picked = picked.sort_values("nearest_mouth_id").reset_index(drop=True)
    picked["mouth_lon"] = picked.to_crs("EPSG:4326").geometry.x
    picked["mouth_lat"] = picked.to_crs("EPSG:4326").geometry.y

    meta = {
        "candidate_count_pre_cluster": int(len(candidates)),
        "major_mouth_count": int(len(picked)),
        "discharge_threshold": float(discharge_threshold) if "discharge_threshold" in locals() else None,
        "bbox": bbox,
        "coast_near_threshold_km": coast_near_threshold_km,
        "mouth_cluster_km": mouth_cluster_km,
        "used_columns": {"discharge": dis_col, "order": ord_col, "length": len_col},
    }
    return picked, meta


def assign_station_distances(station_meta: pd.DataFrame, mouths: gpd.GeoDataFrame) -> pd.DataFrame:
    stations = station_meta.rename(columns={"Station": "station"}).copy()
    station_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs="EPSG:4326",
    ).to_crs(LOCAL_CRS)

    mouth_local = mouths.to_crs(LOCAL_CRS).copy()
    mouth_coords = np.column_stack([mouth_local.geometry.x.to_numpy(), mouth_local.geometry.y.to_numpy()])
    mouth_ids = mouth_local["nearest_mouth_id"].tolist()

    rows: list[dict[str, object]] = []
    for row in station_gdf.itertuples(index=False):
        sx = row.geometry.x
        sy = row.geometry.y
        dists_km = np.sqrt((mouth_coords[:, 0] - sx) ** 2 + (mouth_coords[:, 1] - sy) ** 2) / 1000.0
        order = np.argsort(dists_km)
        best_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else None
        best_dist = float(dists_km[best_idx])
        second_dist = float(dists_km[second_idx]) if second_idx is not None else np.nan
        ambiguity_flag = bool(second_idx is not None and (second_dist - best_dist) < 5.0)
        rows.append(
            {
                "station": row.station,
                "station_name_en": row.station_name_en,
                "water_zone_en": row.water_zone_en,
                "lon": row.lon,
                "lat": row.lat,
                "distance_to_river_mouth_km": best_dist,
                "log_distance_to_river_mouth_km": float(np.log1p(best_dist)),
                "mouth_distance_bucket": pd.cut(
                    pd.Series([best_dist]),
                    bins=[-np.inf, 5, 15, 30, np.inf],
                    labels=["lt5km", "5to15km", "15to30km", "ge30km"],
                ).astype(str).iloc[0],
                "nearest_mouth_id": mouth_ids[best_idx],
                "nearest_mouth_distance_km": best_dist,
                "second_nearest_mouth_distance_km": second_dist,
                "mouth_mapping_ambiguity_flag": int(ambiguity_flag),
                "mouth_definition_source": "HydroRIVERS Asia terminal reaches + Natural Earth coastline + major-mouth filter",
            }
        )
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, mouths: gpd.GeoDataFrame, meta: dict[str, object], report_out: Path) -> None:
    lines = [
        "# distance-to-river-mouth source validation 报告",
        "",
        "## 1. 数据源与定义",
        "",
        "- 主河口数据源：HydroRIVERS Asia（公开免费）",
        "- 海岸线数据源：Natural Earth 10m coastline（公开免费）",
        "- 本轮 river mouth 口径：主要河口",
        "- 主要河口定义：HydroRIVERS 在研究区扩展范围内的近海终端河段，按近岸阈值、河流量级与空间聚合去重后保留。",
        "",
        "## 2. 站点覆盖情况",
        "",
        f"- 站点总数：{len(df)}",
        f"- 成功生成距离值的站点数：{int(df['distance_to_river_mouth_km'].notna().sum())}",
        f"- 覆盖率：{df['distance_to_river_mouth_km'].notna().mean():.4f}",
        f"- 主要河口候选数（去重后）：{meta['major_mouth_count']}",
        "",
        "## 3. 数值分布与歧义检查",
        "",
        f"- 最小距离：{df['distance_to_river_mouth_km'].min():.3f} km",
        f"- 中位数距离：{df['distance_to_river_mouth_km'].median():.3f} km",
        f"- 最大距离：{df['distance_to_river_mouth_km'].max():.3f} km",
        f"- 映射歧义站点数：{int(df['mouth_mapping_ambiguity_flag'].sum())}",
        "",
        "## 4. 结论",
        "",
        "- 该特征是静态空间背景输入，不使用未来信息，不构成时间泄漏。",
        "- 若主要河口数量适中、歧义站点比例可接受，则建议扩展到 observation-level 并进入下一轮单源 A/B。",
        "- 若歧义站点过多或明显映射到错误河口，则应先修正主要河口定义，不直接进入 A/B。",
        "",
        "## 5. 本轮建议",
        "",
        "- 如果覆盖率达到 94/94 且异常值、歧义风险可控，则建议下一轮执行 `distance_to_mouth_only A/B`。",
        "- 若 zone refinement 同时通过，则再考虑 `distance_to_mouth + zone A/B`。",
        "",
        "## 6. 主要河口列表",
        "",
    ]
    for row in mouths[["nearest_mouth_id", "mouth_lon", "mouth_lat", "dis_av_cms", "river_order"]].itertuples(index=False):
        lines.append(
            f"- `{row.nearest_mouth_id}`: lon={row.mouth_lon:.4f}, lat={row.mouth_lat:.4f}, discharge={float(row.dis_av_cms):.3f}, order={row.river_order}"
        )
    report_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    station_meta = pd.read_csv(args.station_meta)
    coastline_zip = Path(args.coastline_zip)
    hydrorivers_zip = Path(args.hydrorivers_zip)
    download_if_needed(args.coastline_url, coastline_zip)
    download_if_needed(args.hydrorivers_url, hydrorivers_zip)

    mouths, meta = build_major_mouths(
        station_meta=station_meta,
        coastline_zip=coastline_zip,
        hydrorivers_zip=hydrorivers_zip,
        bbox_buffer_deg=args.bbox_buffer_deg,
        coast_near_threshold_km=args.coast_near_threshold_km,
        mouth_cluster_km=args.mouth_cluster_km,
    )
    station_df = assign_station_distances(station_meta, mouths)
    Path(args.station_out).parent.mkdir(parents=True, exist_ok=True)
    station_df.to_csv(args.station_out, index=False, encoding="utf-8-sig")
    write_report(station_df, mouths, meta, Path(args.report_out))
    print(
        json.dumps(
            {
                "station_count": int(len(station_df)),
                "coverage_rate": float(station_df["distance_to_river_mouth_km"].notna().mean()),
                "major_mouth_count": int(meta["major_mouth_count"]),
                "ambiguity_count": int(station_df["mouth_mapping_ambiguity_flag"].sum()),
                "station_out": str(args.station_out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
