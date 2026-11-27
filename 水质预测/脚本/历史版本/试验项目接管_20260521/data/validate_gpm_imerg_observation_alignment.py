from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NASA_VENV = PROJECT_ROOT / "tmp" / "venvs" / "nasa_oceancolor_20260520" / "Lib" / "site-packages"
if NASA_VENV.exists() and str(NASA_VENV) not in sys.path:
    sys.path.insert(0, str(NASA_VENV))

import numpy as np
import pandas as pd
import xarray as xr

from scripts.new_data_sources_utils import load_env_file


CURRENT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "current"
CURRENT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"
CURRENT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "manual" / "gpm_imerg_validation"

DATASET_PATH = CURRENT_INTERIM_DIR / "current_model_dataset_with_highres_and_mechanism.csv"
OUT_CSV = CURRENT_PROCESSED_DIR / "gpm_imerg_observation_alignment_validation.csv"
OUT_JSON = CURRENT_PROCESSED_DIR / "gpm_imerg_observation_alignment_validation.json"
OUT_MD = CURRENT_REPORT_DIR / "gpm_imerg_observation_alignment_validation_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPM IMERG observation-date 对齐验证")
    parser.add_argument("--sample-count", type=int, default=6)
    return parser.parse_args()


def maybe_import_earthaccess():
    try:
        import earthaccess  # type: ignore
        return earthaccess
    except Exception:
        return None


def select_sample_dates(sample_count: int) -> list[pd.Timestamp]:
    df = pd.read_csv(DATASET_PATH, usecols=["date"])
    dates = pd.to_datetime(df["date"]).dropna().sort_values().drop_duplicates().to_list()
    if len(dates) <= sample_count:
        return list(dates)
    positions = np.linspace(0, len(dates) - 1, sample_count, dtype=int)
    return [dates[pos] for pos in positions]


def load_station_points(date_value: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, usecols=["station", "date", "lon", "lat"], low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    day = df[df["date"] == date_value].copy()
    return day.drop_duplicates(subset=["station"]).sort_values("station").reset_index(drop=True)


def find_precip_var(ds: xr.Dataset) -> str:
    candidates = ["precipitation", "precipitationCal", "precipitationUncal", "HQprecipitation"]
    for name in candidates:
        if name in ds.data_vars:
            return name
    for name in ds.data_vars:
        if "precip" in name.lower():
            return name
    raise KeyError("no_precipitation_like_variable_found")


def nearest_extract(ds: xr.Dataset, stations: pd.DataFrame, value_var: str) -> pd.DataFrame:
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    out = []
    for row in stations.itertuples(index=False):
        point = ds[value_var].sel({lat_name: row.lat, lon_name: row.lon}, method="nearest")
        value = point.values
        if np.isscalar(value):
            extracted = float(value)
        else:
            extracted = float(np.asarray(value).squeeze())
        nearest_lat = float(point[lat_name].values)
        nearest_lon = float(point[lon_name].values)
        out.append(
            {
                "station": row.station,
                "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                "lon": float(row.lon),
                "lat": float(row.lat),
                "extracted_value": extracted,
                "nearest_grid_lat": nearest_lat,
                "nearest_grid_lon": nearest_lon,
                "grid_lat_diff_deg": abs(float(row.lat) - nearest_lat),
                "grid_lon_diff_deg": abs(float(row.lon) - nearest_lon),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    env = load_env_file()
    user = env.get("NASA_EARTHDATA_USERNAME", "")
    pwd = env.get("NASA_EARTHDATA_PASSWORD", "")
    earthaccess = maybe_import_earthaccess()
    if earthaccess is None:
        raise RuntimeError("earthaccess_missing")
    if not user or not pwd:
        raise RuntimeError("earthdata_credentials_missing")

    os.environ["EARTHDATA_USERNAME"] = user
    os.environ["EARTHDATA_PASSWORD"] = pwd
    earthaccess.login(strategy="environment", persist=False)

    sample_dates = select_sample_dates(args.sample_count)
    alignment_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []
    variable_name = ""
    file_success = 0

    for date_value in sample_dates:
        date_str = pd.Timestamp(date_value).strftime("%Y-%m-%d")
        results = earthaccess.search_data(
            short_name="GPM_3IMERGDF",
            version="07",
            temporal=(date_str, date_str),
            bounding_box=(113.8, 22.1, 114.5, 22.6),
            count=1,
        )
        if not results:
            date_rows.append(
                {
                    "date": date_str,
                    "granule_found": False,
                    "download_ok": False,
                    "open_ok": False,
                    "extraction_success_rate": 0.0,
                    "note": "no_granule_found",
                }
            )
            continue

        try:
            local_paths = earthaccess.download([results[0]], local_path=str(RAW_DIR))
        except Exception as exc:
            text = str(exc)
            note = "earthdata_eula_not_accepted" if ("Eula" in text or "EULA" in text) else f"{type(exc).__name__}: {text[:240]}"
            date_rows.append(
                {
                    "date": date_str,
                    "granule_found": True,
                    "download_ok": False,
                    "open_ok": False,
                    "extraction_success_rate": 0.0,
                    "note": note,
                }
            )
            continue
        if not local_paths:
            date_rows.append(
                {
                    "date": date_str,
                    "granule_found": True,
                    "download_ok": False,
                    "open_ok": False,
                    "extraction_success_rate": 0.0,
                    "note": "download_returned_empty",
                }
            )
            continue

        file_path = Path(local_paths[0])
        file_success += 1
        ds = xr.open_dataset(file_path)
        variable_name = variable_name or find_precip_var(ds)
        stations = load_station_points(pd.Timestamp(date_value))
        extracted = nearest_extract(ds, stations, variable_name)
        extracted["granule_title"] = getattr(results[0], "data", {}).get("meta", {}).get("native-id", "")
        extracted["source_file"] = str(file_path)
        extracted["variable_name"] = variable_name
        extracted["download_status"] = "success"
        extracted["open_status"] = "success"
        extracted["missing_flag"] = extracted["extracted_value"].isna()
        alignment_rows.extend(extracted.to_dict(orient="records"))

        success_rate = float((~extracted["missing_flag"]).mean())
        date_rows.append(
            {
                "date": date_str,
                "granule_found": True,
                "download_ok": True,
                "open_ok": True,
                "extraction_success_rate": success_rate,
                "note": "sample_alignment_ok",
                "station_count": int(len(extracted)),
                "source_file": str(file_path),
            }
        )
        ds.close()

    align_df = pd.DataFrame(alignment_rows)
    date_df = pd.DataFrame(date_rows)
    if not align_df.empty:
        align_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        overall_success = float((~align_df["missing_flag"]).mean())
        mean_lat_diff = float(align_df["grid_lat_diff_deg"].mean())
        mean_lon_diff = float(align_df["grid_lon_diff_deg"].mean())
    else:
        pd.DataFrame(columns=["station", "date"]).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        overall_success = 0.0
        mean_lat_diff = float("nan")
        mean_lon_diff = float("nan")

    payload = {
        "source_name": "gpm_imerg_final_daily",
        "sample_date_count": len(sample_dates),
        "sample_dates": [pd.Timestamp(x).strftime("%Y-%m-%d") for x in sample_dates],
        "successful_download_count": file_success,
        "variable_name": variable_name,
        "overall_extraction_success_rate": overall_success,
        "mean_grid_lat_diff_deg": mean_lat_diff,
        "mean_grid_lon_diff_deg": mean_lon_diff,
        "date_level_results": date_rows,
        "next_step_recommendation": "若样本 granule 下载成功，可进入下一步 runoff proxy observation-date 特征设计；若仍被 GES DISC 条款拦截，则需先确认该账号已完成对应数据中心的一次性条款接受。",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# GPM IMERG observation-date 对齐验证报告",
        "",
        "本轮只做样本下载与 observation-date 对齐验证，不构建正式 runoff 特征，不进入 A/B。",
        "",
        "## 核心结果",
        f"- 抽样日期数：{len(sample_dates)}",
        f"- 成功下载 granule 数：{file_success}",
        f"- 使用变量：`{variable_name or '未识别'}`",
        f"- 总体提取成功率：`{overall_success:.4f}`",
        f"- 平均纬度网格偏差（度）：`{mean_lat_diff:.4f}`" if not np.isnan(mean_lat_diff) else "- 平均纬度网格偏差：未计算",
        f"- 平均经度网格偏差（度）：`{mean_lon_diff:.4f}`" if not np.isnan(mean_lon_diff) else "- 平均经度网格偏差：未计算",
        "",
        "## 日期级结果",
        "",
        date_df.to_string(index=False) if not date_df.empty else "- 无",
        "",
        "## 判断",
        "- 若抽样 granule 均可成功下载并完成站点最近邻提取，则说明 GPM IMERG 已从“可发现”推进到“可用于 observation-date 对齐验证”。",
        "- 若仍被 GES DISC 条款拦截，则说明登录已通，但实际数据下载链仍未打通；此时不建议进入正式 runoff proxy 特征构建。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
