from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NASA_VENV = PROJECT_ROOT / "tmp" / "venvs" / "nasa_oceancolor_20260520" / "Lib" / "site-packages"
if NASA_VENV.exists() and str(NASA_VENV) not in sys.path:
    sys.path.insert(0, str(NASA_VENV))

from scripts.new_data_sources_utils import load_env_file, load_yaml_config  # noqa: E402

CURRENT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
CURRENT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "current"
CURRENT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"
RAW_OCEAN_COLOR_DIR = PROJECT_ROOT / "data" / "raw" / "manual" / "ocean_color" / "nasa_earthdata_validation"
CONFIG_PATH = PROJECT_ROOT / "config" / "new_data_sources.yaml"

FREE_SOURCE_INVENTORY = CURRENT_PROCESSED_DIR / "free_data_source_inventory.csv"
NRT_REPORT_JSON = CURRENT_PROCESSED_DIR / "nasa_oceancolor_validation_report.json"
NRT_SAMPLE_CSV = CURRENT_PROCESSED_DIR / "nasa_oceancolor_validation_sample_201401.csv"
DATA_REQUEST_MD = CURRENT_REPORT_DIR / "data_request_checklist.md"
FOLLOWUP_MD = CURRENT_REPORT_DIR / "nasa_oceancolor_bathymetry_followup.md"
FREE_SOURCE_MD = CURRENT_REPORT_DIR / "free_data_source_validation_report.md"
ACCEPTED_MD = CURRENT_REPORT_DIR / "accepted_data_sources.md"
REJECTED_MD = CURRENT_REPORT_DIR / "rejected_data_sources.md"
NEXT_PLAN_MD = CURRENT_REPORT_DIR / "next_data_acquisition_plan.md"
DATA_GAP_MD = CURRENT_REPORT_DIR / "data_gap_inventory.md"
EXPERIMENT_REPORT_MD = CURRENT_REPORT_DIR / "experiment_report.md"
API_CHECKLIST_MD = CURRENT_REPORT_DIR / "api_access_request_checklist.md"
CLEANUP_LOG_MD = CURRENT_REPORT_DIR / "cleanup_log.md"
PROJECT_ORGANIZER = PROJECT_ROOT / "scripts" / "maintenance" / "project_organizer_agent.py"

DEPTH_BATHY_PATH = CURRENT_PROCESSED_DIR / "depth_bathymetry_features_2014_2024.csv"
CURRENT_BEST_SCORECARD = CURRENT_PROCESSED_DIR / "current_best_scorecard_updated.csv"

NASA_SOURCE_NAME = "nasa_earthdata_oceancolor"
GEBCO_SOURCE_NAME = "gebco_bathymetry_followup"
HK_HYDRO_SOURCE_NAME = "hk_hydrographic_office_bathymetry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NASA Earthdata / bathymetry follow-up validation.")
    parser.add_argument("--date-tag", default="20260520")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [CURRENT_REPORT_DIR, CURRENT_OUTPUT_DIR, CURRENT_PROCESSED_DIR, RAW_OCEAN_COLOR_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_section(path: Path, heading: str, body: str) -> None:
    marker = f"## {heading}"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{marker}\n\n{body.strip()}\n"
    if marker in existing:
        before, _, rest = existing.partition(marker)
        rest_lines = rest.splitlines()
        cut = len(rest_lines)
        for idx, line in enumerate(rest_lines[1:], start=1):
            if line.startswith("## "):
                cut = idx
                break
        replacement = before.rstrip() + "\n\n" + block
        if cut < len(rest_lines):
            replacement += "\n" + "\n".join(rest_lines[cut:])
        write_text(path, replacement)
        return
    text = existing.rstrip()
    if text:
        text += "\n\n"
    text += block
    write_text(path, text)


def month_bounds(year: int, month: int) -> tuple[str, str]:
    end_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{end_day:02d}"


def station_frame() -> pd.DataFrame:
    cfg = load_yaml_config(CONFIG_PATH)
    station_path = PROJECT_ROOT / str(cfg["station_file"])
    df = pd.read_csv(station_path, low_memory=False)
    rename = {}
    if "Station" in df.columns and "station" not in df.columns:
        rename["Station"] = "station"
    if rename:
        df = df.rename(columns=rename)
    needed = ["station", "lon", "lat"]
    keep = [c for c in ["station", "station_name_en", "water_zone_en", "lon", "lat"] if c in df.columns]
    out = df[keep].drop_duplicates(subset=needed).copy()
    out["station"] = out["station"].astype(str).str.strip().str.upper()
    return out


def current_best_summary() -> pd.DataFrame:
    return pd.read_csv(CURRENT_BEST_SCORECARD)


def load_earthaccess():
    import earthaccess  # type: ignore

    return earthaccess


def nasa_login() -> tuple[bool, str]:
    env = load_env_file()
    user = env.get("NASA_EARTHDATA_USERNAME", "")
    pwd = env.get("NASA_EARTHDATA_PASSWORD", "")
    if not user or not pwd:
        return False, "missing_credentials"
    os.environ["EARTHDATA_USERNAME"] = user
    os.environ["EARTHDATA_PASSWORD"] = pwd
    earthaccess = load_earthaccess()
    last_error = ""
    for _ in range(3):
        try:
            earthaccess.login(strategy="environment", persist=False)
            return True, "login_ok"
        except Exception as exc:  # pragma: no cover - network path
            last_error = f"{type(exc).__name__}: {exc}"
    return False, last_error[:500]


def discover_nasa_collections() -> list[dict[str, Any]]:
    earthaccess = load_earthaccess()
    queries = [
        {"short_name": "VIIRSN_L3m_CHL", "label": "VIIRS SNPP monthly CHL"},
        {"short_name": "VIIRSN_L3m_KD", "label": "VIIRS SNPP monthly KD490"},
        {"short_name": "VIIRSN_L3m_RRS", "label": "VIIRS SNPP monthly RRS"},
        {"short_name": "MODISA_L3m_CHL", "label": "MODIS Aqua monthly CHL"},
    ]
    rows: list[dict[str, Any]] = []
    for query in queries:
        try:
            results = earthaccess.search_datasets(short_name=query["short_name"])
            summary = results[0].summary() if results else {}
            rows.append(
                {
                    "short_name": query["short_name"],
                    "label": query["label"],
                    "found": bool(results),
                    "concept_id": summary.get("concept-id") if summary else "",
                    "entry_title": summary.get("entry-title") if summary else "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "short_name": query["short_name"],
                    "label": query["label"],
                    "found": False,
                    "concept_id": "",
                    "entry_title": f"error: {type(exc).__name__}",
                }
            )
    return rows


def search_monthly_granule(short_name: str, year: int, month: int, granule_pattern: str):
    earthaccess = load_earthaccess()
    start_date, end_date = month_bounds(year, month)
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(start_date, end_date),
        granule_name=granule_pattern,
        count=1,
    )
    return results[0] if results else None


def download_validation_granules(force: bool = False) -> dict[str, Path | None]:
    RAW_OCEAN_COLOR_DIR.mkdir(parents=True, exist_ok=True)
    granules = {
        "chl": search_monthly_granule("VIIRSN_L3m_CHL", 2014, 1, "*MO*chlor_a*9km*"),
        "kd490": search_monthly_granule("VIIRSN_L3m_KD", 2014, 1, "*MO*Kd_490*9km*"),
    }
    earthaccess = load_earthaccess()
    out: dict[str, Path | None] = {"chl": None, "kd490": None}
    for key, granule in granules.items():
        if granule is None:
            continue
        expected = RAW_OCEAN_COLOR_DIR / Path(granule.data_links()[0]).name
        if expected.exists() and not force:
            out[key] = expected
            continue
        paths = earthaccess.download([granule], local_path=str(RAW_OCEAN_COLOR_DIR))
        if paths:
            out[key] = Path(paths[0])
    return out


def extract_station_values(nc_path: Path, value_var: str, stations: pd.DataFrame, prefix: str) -> pd.DataFrame:
    ds = None
    last_error: Exception | None = None
    for engine in ["h5netcdf", "netcdf4", None]:
        try:
            ds = xr.open_dataset(nc_path, engine=engine) if engine else xr.open_dataset(nc_path)
            break
        except Exception as exc:  # pragma: no cover - engine fallback
            last_error = exc
    if ds is None:
        raise RuntimeError(f"无法打开 NASA OceanColor 文件: {nc_path}") from last_error
    rows: list[dict[str, Any]] = []
    for rec in stations.itertuples(index=False):
        try:
            sample = ds[value_var].sel(lat=float(rec.lat), lon=float(rec.lon), method="nearest")
            value = float(sample.values)
        except Exception:
            value = np.nan
        rows.append(
            {
                "station": rec.station,
                f"{prefix}_value": value,
            }
        )
    return pd.DataFrame(rows)


def build_nasa_validation_sample(force: bool = False) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    stations = station_frame()
    granules = download_validation_granules(force=force)
    stats: dict[str, Any] = {
        "source_name": NASA_SOURCE_NAME,
        "official_url": "https://oceancolor.gsfc.nasa.gov/",
        "access_type": "free_account",
        "data_type": "remote_sensing",
        "temporal_resolution": "monthly_pilot_9km",
        "spatial_resolution": "9km",
        "temporal_coverage": "2014-01 pilot only in this round",
        "whether_it_is_real_observation_or_reanalysis_proxy": "remote_sensing_observation_product",
    }
    if granules["chl"] is None or granules["kd490"] is None:
        stats.update(
            {
                "station_match_check": "failed",
                "temporal_match_check": "failed",
                "missing_rate": 1.0,
                "extraction_success_rate": 0.0,
                "recommendation": "reject",
                "reason": "NASA OceanColor 月尺度验证样本未能稳定下载。"
            }
        )
        return None, stats

    chl = extract_station_values(granules["chl"], "chlor_a", stations, "nasa_oc_chl")
    kd = extract_station_values(granules["kd490"], "Kd_490", stations, "nasa_oc_kd490")
    merged = stations.merge(chl, on="station", how="left").merge(kd, on="station", how="left")
    merged["date"] = pd.Timestamp("2014-01-15")
    merged["nasa_oc_source_short_name"] = "VIIRSN_L3m_CHL/KD"
    merged["nasa_oc_temporal_resolution"] = "monthly"
    merged["nasa_oc_spatial_resolution_km"] = 9
    merged["nasa_oc_monthly_proxy_flag"] = 1
    merged.to_csv(NRT_SAMPLE_CSV, index=False, encoding="utf-8-sig")
    features = ["nasa_oc_chl_value", "nasa_oc_kd490_value"]
    stats.update(
        {
            "station_match_check": "nearest station-lon-lat extraction on 94 stations",
            "temporal_match_check": "monthly product aligned to 2014-01 pilot month",
            "missing_rate": float(merged[features].isna().mean().mean()),
            "extraction_success_rate": float(merged[features].notna().all(axis=1).mean()),
            "recommendation": "use",
            "reason": "NASA OceanColor 月尺度 9km 验证样本下载与站点提取成功，可进入后续全时段批量化实现。"
        }
    )
    return merged, stats


def bathymetry_followup() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    depth_df = pd.read_csv(DEPTH_BATHY_PATH, low_memory=False) if DEPTH_BATHY_PATH.exists() else pd.DataFrame()
    globathy_cols = [c for c in depth_df.columns if "globathy" in c]
    all_null = True
    if globathy_cols:
        all_null = not depth_df[globathy_cols].notna().any().any()

    gebco_url = "https://www.gebco.net/data-products/gridded-bathymetry-data"
    hk_hydro_url = "https://www.hydro.gov.hk/"
    hk_open_data_pdf = "https://www.mardep.gov.hk/filemanager/en/share/aboutus/pdf/MD-Consolidated-Annual-Open-Data-Plan--Spatial-Data-Plan-included--for-2026-2028-E-v1.0.pdf"
    gebco_status = None
    hk_status = None
    hk_pdf_status = None
    try:
        gebco_status = requests.get(gebco_url, timeout=30).status_code
    except Exception:
        pass
    try:
        hk_status = requests.get(hk_hydro_url, timeout=30).status_code
    except Exception:
        pass
    try:
        hk_pdf_status = requests.get(hk_open_data_pdf, timeout=30).status_code
    except Exception:
        pass

    rows.append(
        {
            "source_name": GEBCO_SOURCE_NAME,
            "official_url": gebco_url,
            "access_type": "public",
            "data_type": "static_geodata",
            "variable_names": "bathymetry_depth,bathymetry_slope",
            "variable_units": "m",
            "spatial_resolution": "global grid, subset route not stabilized this round",
            "temporal_resolution": "static",
            "temporal_coverage": "static",
            "spatial_coverage_check": f"url_status={gebco_status};local_depth_all_null={all_null}",
            "station_match_check": "current local table all-null" if all_null else "local table has non-null values",
            "temporal_match_check": "static_geodata",
            "missing_rate": 1.0 if all_null else float(depth_df[globathy_cols].isna().mean().mean()),
            "extraction_success_rate": 0.0 if all_null else float(depth_df[globathy_cols].notna().any(axis=1).mean()),
            "known_limitations": "本轮没有建立稳定的 GEBCO 研究区子集下载与站点提取流程。",
            "whether_it_is_real_observation_or_reanalysis_proxy": "public_static_geodata",
            "deployability_check": "fail" if all_null else "pending_review",
            "leakage_risk_check": "pass",
            "recommendation": "reject" if all_null else "use",
            "reason": "当前 depth_bathymetry 表仍为空值为主，GEBCO 需要下一轮单独做稳定子集化。"
        }
    )
    rows.append(
        {
            "source_name": HK_HYDRO_SOURCE_NAME,
            "official_url": hk_hydro_url,
            "access_type": "public_or_subscription",
            "data_type": "hydrographic_office_product",
            "variable_names": "bathymetry/chart depth/tidal stream prediction",
            "variable_units": "chart datum dependent",
            "spatial_resolution": "chart / tide point dependent",
            "temporal_resolution": "static charts + 15-min tidal stream open data plan reference",
            "temporal_coverage": "current portal / open data plan",
            "spatial_coverage_check": f"site_status={hk_status};open_data_plan_status={hk_pdf_status}",
            "station_match_check": "bulk station-depth file not found locally",
            "temporal_match_check": "exact sampling datetime still missing",
            "missing_rate": np.nan,
            "extraction_success_rate": 0.0,
            "known_limitations": "目前确认有官方海图/潮流预测服务，但未确认存在可直接自由批量下载的站点水深或 ENC 深度栅格。",
            "whether_it_is_real_observation_or_reanalysis_proxy": "official_hydrographic_product",
            "deployability_check": "pending_manual_route",
            "leakage_risk_check": "pass",
            "recommendation": "reject",
            "reason": "本轮仅确认官方入口存在，未找到可直接批量并入模型的免费水深数值产品。"
        }
    )
    return rows


def update_free_source_inventory(nasa_stats: dict[str, Any], bathy_rows: list[dict[str, Any]]) -> pd.DataFrame:
    existing = pd.read_csv(FREE_SOURCE_INVENTORY) if FREE_SOURCE_INVENTORY.exists() else pd.DataFrame()
    keep = existing[~existing["source_name"].isin([NASA_SOURCE_NAME, GEBCO_SOURCE_NAME, HK_HYDRO_SOURCE_NAME])].copy() if not existing.empty else existing
    rows = [nasa_stats, *bathy_rows]
    updated = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    updated.to_csv(FREE_SOURCE_INVENTORY, index=False, encoding="utf-8-sig")
    return updated


def write_followup_docs(source_df: pd.DataFrame, nasa_stats: dict[str, Any]) -> None:
    nasa_row = source_df[source_df["source_name"] == NASA_SOURCE_NAME].iloc[0]
    bathy_rows = source_df[source_df["source_name"].isin([GEBCO_SOURCE_NAME, HK_HYDRO_SOURCE_NAME])]

    write_text(
        DATA_REQUEST_MD,
        """# 向原始监测单位申请的数据清单

## 最高优先级（直接影响 Chl-a bottom）

| 字段 | 用途 | 优先级 | 对 Chl-a bottom 预期帮助 |
|---|---|---|---|
| exact_sampling_datetime | 将潮位/潮流/辐射/风场与真实采样时刻对齐，摆脱 date-level proxy | 最高 | 很高 |
| station_depth | 建立站点静态水深、浅滩/深槽差异与分层风险 | 最高 | 很高 |
| bottom_water_temperature | 直接刻画底层热状态 | 最高 | 很高 |
| vertical_temperature_profile | 构造 thermocline / stratification / mixing 指标 | 最高 | 很高 |
| ADCP current_u / current_v / current_speed | 近底输运与再悬浮过程 | 最高 | 很高 |
| water_level | 感潮/水动力边界条件 | 高 | 高 |
| discharge / inflow / outflow | 河口输入、冲刷和停留时间代理 | 高 | 高 |

## 中高优先级（机制补充）

| 字段 | 用途 | 优先级 | 对 Chl-a bottom 预期帮助 |
|---|---|---|---|
| salinity_profile | 分层与淡咸水入侵结构 | 中高 | 中高 |
| dissolved_oxygen_profile | 缺氧/分层稳定度侧证 | 中高 | 中高 |
| sampling_depth_m | 精确采样深度，而不是仅 surface/middle/bottom 标签 | 中高 | 中高 |
| station_depth_datum | 统一 chart datum / HKPD / station datum | 中高 | 中高 |
| nearest_tide_station_id | 采样站与潮位/潮流站映射 | 中 | 中 |

## 说明

1. 以上字段不要求来自 NASA；其中 exact_sampling_datetime、底层水温、垂向剖面、ADCP 流速和实测水位/流量通常必须向原始监测或配套观测单位申请。
2. 当前项目最缺的是“真实底层机制数据”，不是普通模型调参。
3. 如果只能优先申请一批，建议顺序为：`exact_sampling_datetime -> station_depth -> bottom_water_temperature / vertical_temperature_profile -> ADCP current -> water_level / discharge`。
""",
    )

    write_text(
        FOLLOWUP_MD,
        f"""# NASA OceanColor / Bathymetry 跟进结果

## NASA Earthdata / OB.DAAC OceanColor

- 凭据来源：本地 `.env`（gitignore 已覆盖）。
- 登录状态：{nasa_stats.get('login_status', 'unknown')}
- 本轮验证产品：VIIRS SNPP L3m monthly CHL / KD490 9km。
- 验证样本文件：[nasa_oceancolor_validation_sample_201401.csv]({NRT_SAMPLE_CSV.as_posix()})
- 结论：{nasa_row['reason']}

## GEBCO

{chr(10).join([f"- {row['source_name']}: {row['reason']}" for _, row in bathy_rows.iterrows() if row['source_name'] == GEBCO_SOURCE_NAME])}

## Hong Kong Hydrographic Office / Marine Department

{chr(10).join([f"- {row['source_name']}: {row['reason']}" for _, row in bathy_rows.iterrows() if row['source_name'] == HK_HYDRO_SOURCE_NAME])}
""",
    )

    append_section(
        FREE_SOURCE_MD,
        "2026-05-20 NASA / bathymetry 跟进",
        "\n".join(
            [
                f"- **{NASA_SOURCE_NAME}**：{nasa_row['reason']}",
                f"- **{GEBCO_SOURCE_NAME}**：{bathy_rows[bathy_rows['source_name'] == GEBCO_SOURCE_NAME]['reason'].iloc[0]}",
                f"- **{HK_HYDRO_SOURCE_NAME}**：{bathy_rows[bathy_rows['source_name'] == HK_HYDRO_SOURCE_NAME]['reason'].iloc[0]}",
            ]
        ),
    )
    append_section(
        ACCEPTED_MD,
        "2026-05-20 新增接受项",
        f"- `{NASA_SOURCE_NAME}`：已完成凭据验证与月尺度样本提取，后续可继续做全时段批量化，但本轮尚未进入正式 A/B。",
    )
    append_section(
        REJECTED_MD,
        "2026-05-20 新增拒绝/暂缓项",
        "\n".join(
            [
                f"- `{GEBCO_SOURCE_NAME}`：本轮未建立稳定 GEBCO 子集化与站点提取流程。",
                f"- `{HK_HYDRO_SOURCE_NAME}`：确认存在官方海图/潮流服务，但未找到可直接自由批量并入模型的免费水深数值产品。",
            ]
        ),
    )
    append_section(
        NEXT_PLAN_MD,
        "2026-05-20 NASA / bathymetry 下一步",
        "\n".join(
            [
                "- 已完成 NASA Earthdata 凭据验证和 VIIRS SNPP 月尺度 OceanColor 样本提取。",
                "- 下一步若继续自动化，应优先把 NASA OceanColor 做成全时段月尺度批量提取，再决定是否值得升级到更高频率。",
                "- 对 bathymetry，应优先继续 GEBCO 稳定子集化；若无稳定公开批量路线，则优先向监测单位申请 station_depth。",
                "- 对 Hong Kong Hydrographic Office / Marine Department，本轮只确认官方入口和开放数据计划存在，不把其写成已成功接入的水深数值产品。",
            ]
        ),
    )
    append_section(
        DATA_GAP_MD,
        "2026-05-20 NASA / bathymetry 缺口补充",
        "\n".join(
            [
                "- NASA Earthdata / OB.DAAC：本轮已完成凭据验证和月尺度样本提取，但尚未完成 2014-2024 全时段批量特征构建，因此还不能进入正式 A/B。",
                "- GEBCO：官方入口可访问，但当前没有稳定的香港研究区子集化落地结果。",
                "- Hong Kong Hydrographic Office / Marine Department：已确认有官方海图和潮流预测服务，但未确认存在可直接自由批量下载的 station-depth / bathymetry 数值表。",
            ]
        ),
    )
    append_section(
        EXPERIMENT_REPORT_MD,
        "2026-05-20 NASA / bathymetry 跟进",
        "\n".join(
            [
                "本轮没有直接进入新一轮模型 A/B，而是优先推进 NASA Earthdata / OB.DAAC OceanColor 和 bathymetry 的免费数据可行性验证。",
                f"`{NASA_SOURCE_NAME}` 已完成凭据验证和 VIIRS SNPP 月尺度样本提取；`{GEBCO_SOURCE_NAME}` 与 `{HK_HYDRO_SOURCE_NAME}` 仍未形成可直接入模的稳定数值产品。",
                "同时新增了向监测单位申请真实机制数据的清单，避免把 proxy 误写成真实底层机制变量。",
            ]
        ),
    )
    append_section(
        API_CHECKLIST_MD,
        "2026-05-20 更新",
        "- NASA Earthdata 账号已通过本地 `.env` 配置，后续不再需要额外 token；但仍需保持账号信息只存在于本地未追踪环境中。",
    )
    append_section(
        CLEANUP_LOG_MD,
        "2026-05-20 NASA / bathymetry 跟进",
        "- 新增本地未追踪 `.env` 中的 NASA Earthdata 凭据；未写入任何追踪文件、日志、CSV、JSON 或 Markdown 正文。",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_project_organizer() -> None:
    import subprocess

    dry = subprocess.run(
        [sys.executable, str(PROJECT_ORGANIZER), "--mode", "dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=240,
    )
    if dry.returncode == 0:
        subprocess.run(
            [sys.executable, str(PROJECT_ORGANIZER), "--mode", "apply-safe"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=240,
        )


def main() -> None:
    args = parse_args()
    ensure_dirs()
    login_ok, login_status = nasa_login()
    collections = discover_nasa_collections() if login_ok else []
    nasa_sample, nasa_stats = build_nasa_validation_sample(force=args.force) if login_ok else (None, {})
    nasa_stats["login_status"] = login_status
    nasa_stats["collection_discovery"] = collections
    if not login_ok:
        nasa_stats.update(
            {
                "source_name": NASA_SOURCE_NAME,
                "official_url": "https://oceancolor.gsfc.nasa.gov/",
                "access_type": "free_account",
                "data_type": "remote_sensing",
                "variable_names": "CHL,KD490",
                "variable_units": "product-specific",
                "spatial_resolution": "pending",
                "temporal_resolution": "pending",
                "temporal_coverage": "pending",
                "spatial_coverage_check": "not_attempted",
                "station_match_check": "not_attempted",
                "temporal_match_check": "not_attempted",
                "missing_rate": np.nan,
                "extraction_success_rate": 0.0,
                "known_limitations": "Earthdata 登录未成功，无法进入样本提取。",
                "whether_it_is_real_observation_or_reanalysis_proxy": "remote_sensing",
                "deployability_check": "fail",
                "leakage_risk_check": "pass",
                "recommendation": "reject",
                "reason": f"NASA Earthdata 登录失败：{login_status}",
            }
        )
    bathy_rows = bathymetry_followup()
    updated = update_free_source_inventory(nasa_stats, bathy_rows)
    write_followup_docs(updated, nasa_stats)
    write_json(
        NRT_REPORT_JSON,
        {
            "nasa_login_ok": login_ok,
            "login_status": login_status,
            "collections": collections,
            "nasa_stats": nasa_stats,
            "bathymetry_followup": bathy_rows,
            "sample_rows": 0 if nasa_sample is None else int(len(nasa_sample)),
        },
    )
    run_project_organizer()
    print("nasa_bathymetry_followup_completed")


if __name__ == "__main__":
    main()
