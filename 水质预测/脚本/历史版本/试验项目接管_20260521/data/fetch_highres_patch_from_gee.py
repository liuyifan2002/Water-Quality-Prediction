from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ee
import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CURRENT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
CURRENT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"
CURRENT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "current"
CURRENT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "current"
CURRENT_EXTERNAL_GEE_DIR = PROJECT_ROOT / "data" / "external" / "gee_exports"

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_INDEX_BANDS = ["NDVI", "NDWI", "MNDWI", "NDCI", "FAI", "red_green_ratio"]
S2_ALL_BANDS = S2_BANDS + S2_INDEX_BANDS

LANDSAT_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "LST_K"]
LANDSAT_INDEX_BANDS = ["NDVI", "NDWI", "MNDWI", "red_green_ratio", "nir_red_ratio"]
LANDSAT_ALL_BANDS = LANDSAT_BANDS + LANDSAT_INDEX_BANDS

S2_PATCH_SPECS = {
    "3x3": {"scale": 10, "half_size_m": 15, "expected_count": 9},
    "5x5": {"scale": 10, "half_size_m": 25, "expected_count": 25},
}
LANDSAT_PATCH_SPECS = {
    "3x3": {"scale": 30, "half_size_m": 45, "expected_count": 9},
    "5x5": {"scale": 30, "half_size_m": 75, "expected_count": 25},
}


@dataclass
class SensorConfig:
    sensor: str
    bands: list[str]
    patch_specs: dict[str, dict[str, int]]
    local_prefix: str
    final_csv: Path
    validation_json: Path
    missing_csv: Path
    chunk_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 and Landsat high-resolution patch features from GEE.")
    parser.add_argument(
        "--requests",
        default="data/processed/current/gee_station_date_requests.csv",
        help="Station-date request table.",
    )
    parser.add_argument(
        "--chunk-size-s2",
        type=int,
        default=300,
        help="Sentinel-2 direct-download chunk size.",
    )
    parser.add_argument(
        "--chunk-size-landsat",
        type=int,
        default=400,
        help="Landsat direct-download chunk size.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Pause between chunk downloads.")
    parser.add_argument(
        "--sensors",
        nargs="+",
        choices=["s2", "landsat"],
        default=["s2", "landsat"],
        help="Sensors to fetch in this run.",
    )
    parser.add_argument(
        "--max-chunks-per-sensor",
        type=int,
        default=None,
        help="Optional cap for probe runs; leave empty for full run.",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=25,
        help="Smallest recursive retry chunk size when GEE reports chunk-level failure.",
    )
    parser.add_argument(
        "--start-base-chunk",
        type=int,
        default=1,
        help="Resume from this 1-based base chunk index before recursive splitting.",
    )
    parser.add_argument(
        "--end-base-chunk",
        type=int,
        default=None,
        help="Optional end base chunk index for partial reruns.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed/current outputs.")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        CURRENT_REPORT_DIR,
        CURRENT_PROCESSED_DIR,
        CURRENT_INTERIM_DIR,
        CURRENT_OUTPUT_DIR,
        CURRENT_EXTERNAL_GEE_DIR,
        CURRENT_INTERIM_DIR / "gee_chunks",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def assert_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output exists, use --force to overwrite: {path}")


def initialize_ee() -> dict[str, Any]:
    report: dict[str, Any] = {
        "earthengine_cli_found": False,
        "ee_initialize_ok": False,
        "cloud_project": None,
        "datasets": {},
    }
    try:
        import shutil

        cli_path = shutil.which("earthengine")
        report["earthengine_cli_found"] = cli_path is not None
        report["earthengine_cli_path"] = cli_path
    except Exception:
        report["earthengine_cli_found"] = False

    ee.Initialize()
    report["ee_initialize_ok"] = True
    data = ee.data.getAssetRoots()
    report["asset_roots_count"] = len(data) if isinstance(data, list) else None
    cloud_project = None
    try:
        cloud_project = ee.data._cloud_api_user_project
    except Exception:
        cloud_project = None
    report["cloud_project"] = cloud_project

    for dataset_id in [
        "COPERNICUS/S2_SR_HARMONIZED",
        "LANDSAT/LC08/C02/T1_L2",
        "LANDSAT/LC09/C02/T1_L2",
    ]:
        try:
            size = ee.ImageCollection(dataset_id).limit(1).size().getInfo()
            report["datasets"][dataset_id] = {"accessible": True, "size_probe": size}
        except Exception as exc:  # pragma: no cover - live GEE failure path
            report["datasets"][dataset_id] = {"accessible": False, "error": str(exc)}
    return report


def write_gee_status_md(status: dict[str, Any]) -> None:
    lines = [
        "# GEE 本机认证与初始化状态",
        "",
        "## 总体状态",
        "",
        f"- `earthengine` CLI 可用: {status.get('earthengine_cli_found')}",
        f"- Earth Engine Python API 初始化成功: {status.get('ee_initialize_ok')}",
        f"- Cloud Project: `{status.get('cloud_project') or '未显式指定，但当前初始化可用'}`",
        "",
        "## 数据集访问探针",
        "",
    ]
    for dataset_id, info in status.get("datasets", {}).items():
        if info.get("accessible"):
            lines.append(f"- `{dataset_id}`: 可访问，size probe={info.get('size_probe')}")
        else:
            lines.append(f"- `{dataset_id}`: 不可访问，错误={info.get('error')}")
    lines += [
        "",
        "## 安全说明",
        "",
        "- 本报告不打印 token。",
        "- 本报告不复制 credential 文件。",
        "- 认证文件未写入项目目录。",
    ]
    (CURRENT_REPORT_DIR / "gee_local_auth_status.md").write_text("\n".join(lines), encoding="utf-8")


def load_requests(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = ["request_id", "sample_id", "station", "date", "lon", "lat", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Request table missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df[df["date"].notna() & df["lon"].notna() & df["lat"].notna()].copy()
    df = df.sort_values(["date", "station", "request_id"]).reset_index(drop=True)
    if df.duplicated(subset=["request_id"]).any():
        raise ValueError("request_id is not unique in gee_station_date_requests.csv")
    return df


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    return [df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)]


def ee_feature_collection(chunk: pd.DataFrame) -> ee.FeatureCollection:
    features = []
    for row in chunk.itertuples(index=False):
        geom = ee.Geometry.Point([float(row.lon), float(row.lat)])
        props = {
            "request_id": str(row.request_id),
            "sample_id": str(row.sample_id),
            "station": str(row.station),
            "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
            "lon": float(row.lon),
            "lat": float(row.lat),
            "split": str(row.split),
        }
        features.append(ee.Feature(geom, props))
    return ee.FeatureCollection(features)


def prefix_dictionary(dictionary: ee.Dictionary, prefix: str) -> ee.Dictionary:
    dictionary = ee.Dictionary(dictionary)
    keys = ee.List(dictionary.keys())
    new_keys = keys.map(lambda k: ee.String(prefix).cat(ee.String(k)))
    values = keys.map(lambda k: dictionary.get(k))
    return ee.Dictionary.fromLists(new_keys, values)


def s2_mask_and_indices(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    scaled = image.select(S2_BANDS).multiply(0.0001)
    ndvi = scaled.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = scaled.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = scaled.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndci = scaled.normalizedDifference(["B5", "B4"]).rename("NDCI")
    fai = scaled.expression(
        "nir - (red + (swir - red) * ((nir_wl - red_wl) / (swir_wl - red_wl)))",
        {
            "nir": scaled.select("B8"),
            "red": scaled.select("B4"),
            "swir": scaled.select("B11"),
            "nir_wl": 842,
            "red_wl": 665,
            "swir_wl": 1610,
        },
    ).rename("FAI")
    red_green = scaled.select("B4").divide(scaled.select("B3")).rename("red_green_ratio")
    return (
        scaled.addBands([ndvi, ndwi, mndwi, ndci, fai, red_green])
        .updateMask(mask)
        .copyProperties(image, ["system:time_start", "PRODUCT_ID", "CLOUDY_PIXEL_PERCENTAGE"])
    )


def landsat_mask_scale_and_indices(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(1 << 3).eq(0)
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
        .And(qa.bitwiseAnd(1 << 7).eq(0))
    )
    optical = (
        image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        .multiply(0.0000275)
        .add(-0.2)
    )
    thermal = image.select("ST_B10").multiply(0.00341802).add(149.0).rename("LST_K")
    ndvi = optical.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    ndwi = optical.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
    mndwi = optical.normalizedDifference(["SR_B3", "SR_B6"]).rename("MNDWI")
    red_green = optical.select("SR_B4").divide(optical.select("SR_B3")).rename("red_green_ratio")
    nir_red = optical.select("SR_B5").divide(optical.select("SR_B4")).rename("nir_red_ratio")
    return (
        optical.addBands([thermal, ndvi, ndwi, mndwi, red_green, nir_red])
        .updateMask(clear)
        .copyProperties(image, ["system:time_start", "LANDSAT_PRODUCT_ID", "CLOUD_COVER", "SPACECRAFT_ID"])
    )


def combined_reducer() -> ee.Reducer:
    return (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.percentile([10, 90]), sharedInputs=True)
        .combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.count(), sharedInputs=True)
    )


def patch_stats(image: ee.Image, point: ee.Geometry, patch_name: str, patch_spec: dict[str, int], bands: list[str], ref_band: str) -> ee.Dictionary:
    region = point.buffer(float(patch_spec["half_size_m"]), 1).bounds(1)
    stats = ee.Dictionary(
        image.select(bands).reduceRegion(
            reducer=combined_reducer(),
            geometry=region,
            scale=int(patch_spec["scale"]),
            maxPixels=1_000_000,
            bestEffort=False,
        )
    )
    prefixed = prefix_dictionary(stats, f"{patch_name}_")
    count_key = f"{patch_name}_{ref_band}_count"
    valid_ratio = ee.Algorithms.If(
        prefixed.contains(count_key),
        ee.Number(prefixed.get(count_key)).divide(float(patch_spec["expected_count"])),
        None,
    )
    missing_ratio = ee.Algorithms.If(
        prefixed.contains(count_key),
        ee.Number(1).subtract(ee.Number(prefixed.get(count_key)).divide(float(patch_spec["expected_count"]))),
        None,
    )
    return ee.Dictionary(prefixed).set(f"{patch_name}_valid_ratio", valid_ratio).set(f"{patch_name}_missing_ratio", missing_ratio)


def reference_patch_count(image: ee.Image, point: ee.Geometry, band: str, scale: int, half_size_m: int) -> ee.Number:
    region = point.buffer(float(half_size_m), 1).bounds(1)
    reduced = ee.Dictionary(
        image.select([band]).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=scale,
            maxPixels=1_000_000,
            bestEffort=False,
        )
    )
    return ee.Number(ee.Algorithms.If(reduced.contains(band), reduced.get(band), 0))


def build_s2_collection(point: ee.Geometry, date_str: str) -> ee.ImageCollection:
    date = ee.Date(date_str)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(date.advance(-7, "day"), date.advance(8, "day"))
        .map(s2_mask_and_indices)
        .map(
            lambda img: img.set(
                {
                    "patch_ref_count_3x3": reference_patch_count(img, point, "B2", 10, 15),
                    "patch_ref_count_5x5": reference_patch_count(img, point, "B2", 10, 25),
                }
            )
        )
        .map(
            lambda img: img.set(
                {
                    "date_diff_abs": ee.Number(img.date().difference(date, "day")).abs(),
                }
            )
        )
        .map(
            lambda img: img.set(
                {
                    "window_rank": ee.Number(
                        ee.Algorithms.If(
                            ee.Number(img.get("date_diff_abs")).lte(1),
                            1,
                            ee.Algorithms.If(ee.Number(img.get("date_diff_abs")).lte(3), 2, 3),
                        )
                    ),
                    "window_days_used": ee.Number(
                        ee.Algorithms.If(
                            ee.Number(img.get("date_diff_abs")).lte(1),
                            1,
                            ee.Algorithms.If(ee.Number(img.get("date_diff_abs")).lte(3), 3, 7),
                        )
                    ),
                }
            )
        )
        .filter(ee.Filter.gt("patch_ref_count_5x5", 0))
        .map(
            lambda img: img.set(
                {
                    "patch_valid_priority": ee.Number(img.get("patch_ref_count_5x5")).multiply(-1),
                    "sort_score": ee.Number(img.get("window_rank"))
                    .multiply(100000)
                    .add(ee.Number(img.get("date_diff_abs")).multiply(100))
                    .add(
                        ee.Number(
                            ee.Algorithms.If(
                                img.propertyNames().contains("CLOUDY_PIXEL_PERCENTAGE"),
                                img.get("CLOUDY_PIXEL_PERCENTAGE"),
                                100,
                            )
                        )
                    )
                }
            )
        )
    )
    return collection


def build_landsat_collection(point: ee.Geometry, date_str: str) -> ee.ImageCollection:
    date = ee.Date(date_str)
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    collection = (
        l8.merge(l9)
        .filterBounds(point)
        .filterDate(date.advance(-16, "day"), date.advance(17, "day"))
        .map(landsat_mask_scale_and_indices)
        .map(
            lambda img: img.set(
                {
                    "patch_ref_count_3x3": reference_patch_count(img, point, "SR_B2", 30, 45),
                    "patch_ref_count_5x5": reference_patch_count(img, point, "SR_B2", 30, 75),
                }
            )
        )
        .map(
            lambda img: img.set(
                {
                    "date_diff_abs": ee.Number(img.date().difference(date, "day")).abs(),
                }
            )
        )
        .map(
            lambda img: img.set(
                {
                    "window_rank": ee.Number(
                        ee.Algorithms.If(ee.Number(img.get("date_diff_abs")).lte(8), 1, 2)
                    ),
                    "window_days_used": ee.Number(
                        ee.Algorithms.If(ee.Number(img.get("date_diff_abs")).lte(8), 8, 16)
                    ),
                }
            )
        )
        .filter(ee.Filter.gt("patch_ref_count_5x5", 0))
        .map(
            lambda img: img.set(
                {
                    "patch_valid_priority": ee.Number(img.get("patch_ref_count_5x5")).multiply(-1),
                    "sort_score": ee.Number(img.get("window_rank"))
                    .multiply(100000)
                    .add(ee.Number(img.get("date_diff_abs")).multiply(100))
                    .add(
                        ee.Number(
                            ee.Algorithms.If(
                                img.propertyNames().contains("CLOUD_COVER"),
                                img.get("CLOUD_COVER"),
                                100,
                            )
                        )
                    )
                }
            )
        )
    )
    return collection


def build_empty_feature(point: ee.Geometry, props: ee.Dictionary, sensor: str) -> ee.Feature:
    base = ee.Dictionary(props)
    if sensor == "s2":
        base = base.combine(
            ee.Dictionary(
                {
                "s2_image_id": None,
                "s2_image_date": None,
                "s2_date_diff_abs": None,
                "s2_cloudy_pixel_percentage": None,
                "s2_image_count_in_window": 0,
                "s2_window_days_used": None,
                }
            ),
            True,
        )
    else:
        base = base.combine(
            ee.Dictionary(
                {
                "landsat_image_id": None,
                "landsat_image_date": None,
                "landsat_date_diff_abs": None,
                "landsat_cloud_cover": None,
                "landsat_image_count_in_window": 0,
                "landsat_window_days_used": None,
                }
            ),
            True,
        )
    return ee.Feature(point, base)


def s2_feature_for_request(feature: ee.Feature) -> ee.Feature:
    point = feature.geometry()
    props = ee.Dictionary(feature.toDictionary(["request_id", "sample_id", "station", "date", "lon", "lat", "split"]))
    collection = build_s2_collection(point, ee.String(feature.get("date")))
    size = collection.size()
    selected = ee.Image(collection.sort("sort_score").first())
    empty = size.eq(0)

    def when_present() -> ee.Feature:
        window_rank = ee.Number(selected.get("window_rank"))
        window_days = ee.Number(selected.get("window_days_used"))
        image_count = collection.filter(ee.Filter.eq("window_rank", window_rank)).size()
        meta = ee.Dictionary(
            {
                "s2_image_id": selected.get("PRODUCT_ID"),
                "s2_image_date": selected.date().format("YYYY-MM-dd"),
                "s2_date_diff_abs": selected.get("date_diff_abs"),
                "s2_cloudy_pixel_percentage": selected.get("CLOUDY_PIXEL_PERCENTAGE"),
                "s2_image_count_in_window": image_count,
                "s2_window_days_used": window_days,
            }
        )
        stats_3 = patch_stats(selected, point, "s2_3x3", S2_PATCH_SPECS["3x3"], S2_ALL_BANDS, "B2")
        stats_5 = patch_stats(selected, point, "s2_5x5", S2_PATCH_SPECS["5x5"], S2_ALL_BANDS, "B2")
        return ee.Feature(point, props.combine(meta, True).combine(stats_3, True).combine(stats_5, True))

    return ee.Feature(ee.Algorithms.If(empty, build_empty_feature(point, props, "s2"), when_present()))


def landsat_feature_for_request(feature: ee.Feature) -> ee.Feature:
    point = feature.geometry()
    props = ee.Dictionary(feature.toDictionary(["request_id", "sample_id", "station", "date", "lon", "lat", "split"]))
    collection = build_landsat_collection(point, ee.String(feature.get("date")))
    size = collection.size()
    selected = ee.Image(collection.sort("sort_score").first())
    empty = size.eq(0)

    def when_present() -> ee.Feature:
        window_rank = ee.Number(selected.get("window_rank"))
        window_days = ee.Number(selected.get("window_days_used"))
        image_count = collection.filter(ee.Filter.eq("window_rank", window_rank)).size()
        meta = ee.Dictionary(
            {
                "landsat_image_id": selected.get("LANDSAT_PRODUCT_ID"),
                "landsat_image_date": selected.date().format("YYYY-MM-dd"),
                "landsat_date_diff_abs": selected.get("date_diff_abs"),
                "landsat_cloud_cover": selected.get("CLOUD_COVER"),
                "landsat_spacecraft_id": selected.get("SPACECRAFT_ID"),
                "landsat_image_count_in_window": image_count,
                "landsat_window_days_used": window_days,
            }
        )
        stats_3 = patch_stats(selected, point, "landsat_3x3", LANDSAT_PATCH_SPECS["3x3"], LANDSAT_ALL_BANDS, "SR_B2")
        stats_5 = patch_stats(selected, point, "landsat_5x5", LANDSAT_PATCH_SPECS["5x5"], LANDSAT_ALL_BANDS, "SR_B2")
        return ee.Feature(point, props.combine(meta, True).combine(stats_3, True).combine(stats_5, True))

    return ee.Feature(ee.Algorithms.If(empty, build_empty_feature(point, props, "landsat"), when_present()))


def download_fc_csv(fc: ee.FeatureCollection, selectors: list[str] | None = None) -> str:
    kwargs: dict[str, Any] = {"filetype": "CSV"}
    if selectors:
        kwargs["selectors"] = selectors
    url = fc.getDownloadURL(**kwargs)
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    return response.text


def execute_sensor_download(
    requests_df: pd.DataFrame,
    config: SensorConfig,
    feature_builder,
    max_supported_date: str | None = None,
    min_supported_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work_df = requests_df.copy()
    if min_supported_date is not None:
        min_ts = pd.Timestamp(min_supported_date)
        unsupported = work_df["date"] < min_ts
    else:
        unsupported = pd.Series(False, index=work_df.index)
    if max_supported_date is not None:
        max_ts = pd.Timestamp(max_supported_date)
        unsupported = unsupported | (work_df["date"] > max_ts)

    direct_df = work_df.loc[~unsupported].copy()
    unsupported_df = work_df.loc[unsupported].copy()
    chunks = chunk_dataframe(direct_df, config.chunk_size)
    if args.max_chunks_per_sensor is not None:
        chunks = chunks[: args.max_chunks_per_sensor]
    chunk_dir = CURRENT_INTERIM_DIR / "gee_chunks" / config.sensor
    chunk_dir.mkdir(parents=True, exist_ok=True)
    partial_manifest_path = CURRENT_INTERIM_DIR / f"{config.sensor}_gee_manifest_partial.csv"

    manifest_rows: list[dict[str, Any]] = []
    chunk_paths: list[Path] = []

    def process_chunk(chunk: pd.DataFrame, chunk_id: str) -> None:
        expected_filename = f"{config.local_prefix}_part{chunk_id}.csv"
        local_path = chunk_dir / expected_filename
        fc = ee_feature_collection(chunk).map(feature_builder)
        started_at = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
        status = "success"
        note = ""
        try:
            csv_text = download_fc_csv(fc)
            local_path.write_text(csv_text, encoding="utf-8")
            chunk_paths.append(local_path)
            completed_at = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
            manifest_rows.append(
                {
                    "task_name": f"{config.local_prefix}_part{chunk_id}",
                    "sensor": config.sensor,
                    "chunk_id": chunk_id,
                    "request_count": int(len(chunk)),
                    "status": status,
                    "export_destination": "local_direct_download",
                    "expected_filename": expected_filename,
                    "local_file_found": local_path.exists(),
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "note": note,
                }
            )
        except Exception as exc:  # pragma: no cover - live ee failure path
            note = str(exc)
            if len(chunk) > args.min_chunk_size:
                manifest_rows.append(
                    {
                        "task_name": f"{config.local_prefix}_part{chunk_id}",
                        "sensor": config.sensor,
                        "chunk_id": chunk_id,
                        "request_count": int(len(chunk)),
                        "status": "split_retry",
                        "export_destination": "local_direct_download",
                        "expected_filename": expected_filename,
                        "local_file_found": local_path.exists(),
                        "started_at": started_at,
                        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                        "note": f"{note} | retry_split",
                    }
                )
                mid = len(chunk) // 2
                left = chunk.iloc[:mid].copy()
                right = chunk.iloc[mid:].copy()
                process_chunk(left, f"{chunk_id}a")
                process_chunk(right, f"{chunk_id}b")
            else:
                completed_at = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
                manifest_rows.append(
                    {
                        "task_name": f"{config.local_prefix}_part{chunk_id}",
                        "sensor": config.sensor,
                        "chunk_id": chunk_id,
                        "request_count": int(len(chunk)),
                        "status": "failed",
                        "export_destination": "local_direct_download",
                        "expected_filename": expected_filename,
                        "local_file_found": local_path.exists(),
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "note": note,
                    }
                )
        pd.DataFrame(manifest_rows).to_csv(partial_manifest_path, index=False, encoding="utf-8-sig")
        time.sleep(args.sleep_seconds)

    end_base_chunk = args.end_base_chunk or len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        if idx < args.start_base_chunk or idx > end_base_chunk:
            continue
        process_chunk(chunk, f"{idx:03d}")

    if any(row["status"] == "failed" for row in manifest_rows):
        raise RuntimeError(f"{config.sensor} direct download failed for one or more chunks; see gee_export_task_manifest.csv")

    frame_files = sorted(chunk_dir.glob(f"{config.local_prefix}_part*.csv"))
    frame_list = [pd.read_csv(path, low_memory=False) for path in frame_files]
    sensor_df = pd.concat(frame_list, ignore_index=True) if frame_list else pd.DataFrame()
    if not unsupported_df.empty:
        unsupported_rows = unsupported_df[["request_id", "sample_id", "station", "date", "lon", "lat", "split"]].copy()
        if config.sensor == "s2":
            unsupported_rows["s2_image_id"] = np.nan
            unsupported_rows["s2_image_date"] = np.nan
            unsupported_rows["s2_date_diff_abs"] = np.nan
            unsupported_rows["s2_cloudy_pixel_percentage"] = np.nan
            unsupported_rows["s2_image_count_in_window"] = 0
            unsupported_rows["s2_window_days_used"] = np.nan
        else:
            unsupported_rows["landsat_image_id"] = np.nan
            unsupported_rows["landsat_image_date"] = np.nan
            unsupported_rows["landsat_date_diff_abs"] = np.nan
            unsupported_rows["landsat_cloud_cover"] = np.nan
            unsupported_rows["landsat_spacecraft_id"] = np.nan
            unsupported_rows["landsat_image_count_in_window"] = 0
            unsupported_rows["landsat_window_days_used"] = np.nan
        sensor_df = pd.concat([sensor_df, unsupported_rows], ignore_index=True, sort=False)
        manifest_rows.append(
            {
                "task_name": f"{config.local_prefix}_unsupported_dates",
                "sensor": config.sensor,
                "chunk_id": "unsupported",
                "request_count": int(len(unsupported_df)),
                "status": "skipped_unsupported_dates",
                "export_destination": "local_direct_download",
                "expected_filename": "",
                "local_file_found": False,
                "started_at": "",
                "completed_at": "",
                "note": f"outside supported range {min_supported_date or '-inf'} to {max_supported_date or '+inf'}",
            }
        )
        pd.DataFrame(manifest_rows).to_csv(partial_manifest_path, index=False, encoding="utf-8-sig")

    sensor_df["date"] = pd.to_datetime(sensor_df["date"]).dt.strftime("%Y-%m-%d")
    sensor_df = sensor_df.sort_values(["date", "station", "request_id"]).reset_index(drop=True)
    manifest_df = pd.DataFrame(manifest_rows)
    return sensor_df, manifest_df


def validate_sensor_output(sensor_df: pd.DataFrame, config: SensorConfig, date_diff_col: str, image_id_col: str) -> dict[str, Any]:
    report = {
        "rows": int(len(sensor_df)),
        "columns": list(sensor_df.columns),
        "station_count": int(sensor_df["station"].nunique()) if "station" in sensor_df.columns else 0,
        "date_range": [
            str(pd.to_datetime(sensor_df["date"]).min().date()) if len(sensor_df) else None,
            str(pd.to_datetime(sensor_df["date"]).max().date()) if len(sensor_df) else None,
        ],
        "request_id_unique": not sensor_df["request_id"].duplicated().any(),
        "duplicate_request_ids": int(sensor_df["request_id"].duplicated().sum()),
        "coverage_rate": float(sensor_df[image_id_col].notna().mean()) if image_id_col in sensor_df.columns else 0.0,
        "date_diff_summary": sensor_df[date_diff_col].describe(percentiles=[0.5, 0.9, 0.95]).to_dict() if date_diff_col in sensor_df.columns else {},
    }
    missing = pd.DataFrame(
        {
            "column": sensor_df.columns,
            "missing_rate": sensor_df.isna().mean().values,
            "non_null_count": sensor_df.notna().sum().values,
        }
    ).sort_values(["missing_rate", "column"], ascending=[False, True])
    sensor_df.to_csv(config.final_csv, index=False, encoding="utf-8-sig")
    missing.to_csv(config.missing_csv, index=False, encoding="utf-8-sig")
    config.validation_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_existing_sensor_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def merge_highres_features(s2_df: pd.DataFrame | None, landsat_df: pd.DataFrame | None) -> dict[str, Any]:
    merge_keys = ["request_id", "sample_id", "station", "date", "lon", "lat", "split"]
    if s2_df is not None and not s2_df.empty:
        s2_df = s2_df.copy()
        s2_df["date"] = pd.to_datetime(s2_df["date"])
    if landsat_df is not None and not landsat_df.empty:
        landsat_df = landsat_df.copy()
        landsat_df["date"] = pd.to_datetime(landsat_df["date"])
    if s2_df is not None and not s2_df.empty:
        merged = s2_df.copy()
    elif landsat_df is not None and not landsat_df.empty:
        merged = landsat_df.copy()
    else:
        raise RuntimeError("Neither Sentinel-2 nor Landsat highres patch features were produced.")

    if landsat_df is not None and not landsat_df.empty:
        if s2_df is None or s2_df.empty:
            merged = landsat_df.copy()
        else:
            merged = merged.merge(landsat_df, on=merge_keys, how="outer", validate="one_to_one")

    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["date", "station", "request_id"]).reset_index(drop=True)
    if merged.duplicated(subset=["request_id"]).any():
        raise ValueError("Merged highres patch table has duplicate request_id values.")

    output_path = CURRENT_PROCESSED_DIR / "highres_patch_features_2014_2024.csv"
    validation_path = CURRENT_PROCESSED_DIR / "highres_patch_validation_report.json"
    missing_path = CURRENT_PROCESSED_DIR / "highres_patch_missing_summary.csv"
    dictionary_path = CURRENT_PROCESSED_DIR / "highres_patch_feature_dictionary.csv"

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    missing = pd.DataFrame(
        {
            "column": merged.columns,
            "missing_rate": merged.isna().mean().values,
            "non_null_count": merged.notna().sum().values,
        }
    ).sort_values(["missing_rate", "column"], ascending=[False, True])
    missing.to_csv(missing_path, index=False, encoding="utf-8-sig")

    feature_rows = []
    for column in merged.columns:
        if column in merge_keys:
            meaning = "join_key"
        elif column.endswith("_valid_ratio"):
            meaning = "patch_valid_pixel_ratio"
        elif column.endswith("_missing_ratio"):
            meaning = "patch_missing_pixel_ratio"
        elif "_mean" in column:
            meaning = "patch_mean"
        elif "_stdDev" in column or column.endswith("_std"):
            meaning = "patch_std"
        elif "_median" in column:
            meaning = "patch_median"
        elif "_p10" in column:
            meaning = "patch_percentile_10"
        elif "_p90" in column:
            meaning = "patch_percentile_90"
        elif column.endswith("_min"):
            meaning = "patch_min"
        elif column.endswith("_max"):
            meaning = "patch_max"
        elif column.endswith("_count"):
            meaning = "patch_valid_pixel_count"
        else:
            meaning = "metadata_or_raw"
        feature_rows.append({"column": column, "meaning": meaning})
    pd.DataFrame(feature_rows).to_csv(dictionary_path, index=False, encoding="utf-8-sig")

    report = {
        "rows": int(len(merged)),
        "station_count": int(merged["station"].nunique()),
        "date_range": [str(merged["date"].min().date()), str(merged["date"].max().date())],
        "request_id_unique": not merged["request_id"].duplicated().any(),
        "s2_coverage_rate": float(merged["s2_image_id"].notna().mean()) if "s2_image_id" in merged.columns else 0.0,
        "landsat_coverage_rate": float(merged["landsat_image_id"].notna().mean()) if "landsat_image_id" in merged.columns else 0.0,
        "s2_date_diff_summary": merged["s2_date_diff_abs"].describe(percentiles=[0.5, 0.9, 0.95]).to_dict() if "s2_date_diff_abs" in merged.columns else {},
        "landsat_date_diff_summary": merged["landsat_date_diff_abs"].describe(percentiles=[0.5, 0.9, 0.95]).to_dict() if "landsat_date_diff_abs" in merged.columns else {},
        "s2_valid_ratio_columns": [c for c in merged.columns if c.startswith("s2_") and c.endswith("_valid_ratio")],
        "landsat_valid_ratio_columns": [c for c in merged.columns if c.startswith("landsat_") and c.endswith("_valid_ratio")],
    }
    validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def append_cleanup_entry(lines: list[str]) -> None:
    path = CURRENT_REPORT_DIR / "cleanup_log.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# 清理与汇总日志\n"
    block = "\n".join(lines).rstrip() + "\n"
    path.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")


def main() -> None:
    global args
    args = parse_args()
    ensure_dirs()

    request_path = PROJECT_ROOT / args.requests
    configs = {
        "s2": SensorConfig(
            sensor="s2",
            bands=S2_ALL_BANDS,
            patch_specs=S2_PATCH_SPECS,
            local_prefix="s2_patch_features_2014_2024",
            final_csv=CURRENT_PROCESSED_DIR / "s2_highres_patch_features_2014_2024.csv",
            validation_json=CURRENT_PROCESSED_DIR / "s2_highres_patch_validation_report.json",
            missing_csv=CURRENT_PROCESSED_DIR / "s2_highres_patch_missing_summary.csv",
            chunk_size=args.chunk_size_s2,
        ),
        "landsat": SensorConfig(
            sensor="landsat",
            bands=LANDSAT_ALL_BANDS,
            patch_specs=LANDSAT_PATCH_SPECS,
            local_prefix="landsat_patch_features_2014_2024",
            final_csv=CURRENT_PROCESSED_DIR / "landsat_highres_patch_features_2014_2024.csv",
            validation_json=CURRENT_PROCESSED_DIR / "landsat_highres_patch_validation_report.json",
            missing_csv=CURRENT_PROCESSED_DIR / "landsat_highres_patch_missing_summary.csv",
            chunk_size=args.chunk_size_landsat,
        ),
    }

    overwrite_targets = [
        configs["s2"].final_csv,
        configs["s2"].validation_json,
        configs["s2"].missing_csv,
        configs["landsat"].final_csv,
        configs["landsat"].validation_json,
        configs["landsat"].missing_csv,
        CURRENT_PROCESSED_DIR / "highres_patch_features_2014_2024.csv",
        CURRENT_PROCESSED_DIR / "highres_patch_validation_report.json",
        CURRENT_PROCESSED_DIR / "highres_patch_missing_summary.csv",
        CURRENT_PROCESSED_DIR / "highres_patch_feature_dictionary.csv",
        CURRENT_PROCESSED_DIR / "gee_export_task_manifest.csv",
        CURRENT_REPORT_DIR / "gee_local_auth_status.md",
    ]
    for path in overwrite_targets:
        assert_can_write(path, args.force)

    status = initialize_ee()
    write_gee_status_md(status)
    requests_df = load_requests(request_path)

    manifest_frames: list[pd.DataFrame] = []

    s2_df, s2_manifest = execute_sensor_download(
        requests_df,
        configs["s2"],
        s2_feature_for_request,
        min_supported_date="2015-06-23",
    ) if "s2" in args.sensors else (None, pd.DataFrame())
    if s2_df is not None:
        manifest_frames.append(s2_manifest)
        validate_sensor_output(s2_df, configs["s2"], "s2_date_diff_abs", "s2_image_id")

    landsat_df, landsat_manifest = execute_sensor_download(
        requests_df,
        configs["landsat"],
        landsat_feature_for_request,
        min_supported_date="2014-01-01",
    ) if "landsat" in args.sensors else (None, pd.DataFrame())
    if landsat_df is not None:
        manifest_frames.append(landsat_manifest)
        validate_sensor_output(landsat_df, configs["landsat"], "landsat_date_diff_abs", "landsat_image_id")

    manifest_df = pd.concat(manifest_frames, ignore_index=True) if manifest_frames else pd.DataFrame()
    manifest_df.to_csv(CURRENT_PROCESSED_DIR / "gee_export_task_manifest.csv", index=False, encoding="utf-8-sig")

    if s2_df is None:
        s2_df = load_existing_sensor_csv(configs["s2"].final_csv)
    if landsat_df is None:
        landsat_df = load_existing_sensor_csv(configs["landsat"].final_csv)

    merge_report = merge_highres_features(s2_df, landsat_df)
    append_cleanup_entry(
        [
            "## 本轮 GEE highres patch 抓取与合并",
            "",
            f"- 新增/更新 `data/processed/current/s2_highres_patch_features_2014_2024.csv`，行数 `{len(s2_df) if s2_df is not None else 0}`。",
            f"- 新增/更新 `data/processed/current/landsat_highres_patch_features_2014_2024.csv`，行数 `{len(landsat_df) if landsat_df is not None else 0}`。",
            f"- 新增/更新 `data/processed/current/highres_patch_features_2014_2024.csv`，行数 `{merge_report['rows']}`。",
            "- 新增/更新 `data/processed/current/gee_export_task_manifest.csv`。",
            "- 新增/更新 `experiments/reports/current/gee_local_auth_status.md`。",
        ]
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "gee_auth_ok": status.get("ee_initialize_ok"),
                "s2_rows": len(s2_df) if s2_df is not None else 0,
                "landsat_rows": len(landsat_df) if landsat_df is not None else 0,
                "merged_rows": merge_report["rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
