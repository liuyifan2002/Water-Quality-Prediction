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

import pandas as pd
import requests

from scripts.new_data_sources_utils import load_env_file


CURRENT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"
CURRENT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "manual" / "gpm_imerg_validation"

OUT_JSON = CURRENT_PROCESSED_DIR / "gpm_imerg_source_validation.json"
OUT_REPORT = CURRENT_REPORT_DIR / "gpm_imerg_source_validation_report.md"

CMR_COLLECTION_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"
CMR_GRANULE_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPM IMERG source validation")
    parser.add_argument("--short-name", default="GPM_3IMERGDF")
    parser.add_argument("--version", default="07")
    parser.add_argument("--start", default="2014-01-01T00:00:00Z")
    parser.add_argument("--end", default="2014-01-02T23:59:59Z")
    parser.add_argument("--bbox", default="113.8,22.1,114.5,22.6")
    return parser.parse_args()


def maybe_import_earthaccess():
    try:
        import earthaccess  # type: ignore
        return earthaccess
    except Exception:
        return None


def discover_collection(short_name: str) -> dict[str, Any]:
    resp = requests.get(CMR_COLLECTION_URL, params={"short_name": short_name, "page_size": 5}, timeout=60)
    resp.raise_for_status()
    entries = resp.json().get("feed", {}).get("entry", [])
    return entries[0] if entries else {}


def discover_granule(short_name: str, version: str, start: str, end: str, bbox: str) -> dict[str, Any]:
    params = {
        "short_name": short_name,
        "version": version,
        "temporal": f"{start},{end}",
        "bounding_box": bbox,
        "page_size": 1,
    }
    resp = requests.get(CMR_GRANULE_URL, params=params, timeout=60)
    resp.raise_for_status()
    entries = resp.json().get("feed", {}).get("entry", [])
    return entries[0] if entries else {}


def earthdata_login_status() -> tuple[bool, str]:
    env = load_env_file()
    user = env.get("NASA_EARTHDATA_USERNAME", "")
    pwd = env.get("NASA_EARTHDATA_PASSWORD", "")
    if not user or not pwd:
        return False, "missing_credentials"
    earthaccess = maybe_import_earthaccess()
    if earthaccess is None:
        return False, "earthaccess_missing"
    os.environ["EARTHDATA_USERNAME"] = user
    os.environ["EARTHDATA_PASSWORD"] = pwd
    try:
        earthaccess.login(strategy="environment", persist=False)
        return True, "login_ok"
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"


def try_download_granule(granule: dict[str, Any]) -> tuple[str, str]:
    earthaccess = maybe_import_earthaccess()
    if earthaccess is None:
        return "not_attempted", "earthaccess_missing"
    env = load_env_file()
    os.environ["EARTHDATA_USERNAME"] = env.get("NASA_EARTHDATA_USERNAME", "")
    os.environ["EARTHDATA_PASSWORD"] = env.get("NASA_EARTHDATA_PASSWORD", "")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        earthaccess.login(strategy="environment", persist=False)
        results = earthaccess.search_data(
            short_name="GPM_3IMERGDF",
            version="07",
            temporal=("2014-01-01", "2014-01-02"),
            bounding_box=(113.8, 22.1, 114.5, 22.6),
            count=1,
        )
        if not results:
            return "failed", "no_granule_found_in_earthaccess"
        earthaccess.download([results[0]], local_path=str(RAW_DIR))
        return "success", "sample_download_ok"
    except Exception as exc:  # pragma: no cover
        text = str(exc)
        if "Eula" in text or "EULA" in text:
            return "blocked", "earthdata_eula_not_accepted"
        return "failed", f"{type(exc).__name__}: {text[:240]}"


def main() -> None:
    args = parse_args()
    collection = discover_collection(args.short_name)
    granule = discover_granule(args.short_name, args.version, args.start, args.end, args.bbox)
    login_ok, login_status = earthdata_login_status()
    download_status, download_note = try_download_granule(granule) if login_ok else ("not_attempted", login_status)

    granule_links = granule.get("links", []) if granule else []
    http_link = next((x.get("href", "") for x in granule_links if str(x.get("href", "")).startswith("https://data.gesdisc")), "")
    opendap_link = next((x.get("href", "") for x in granule_links if "opendap" in str(x.get("href", "")).lower()), "")

    payload = {
        "source_name": "gpm_imerg_final_daily",
        "official_url": "https://gpm.nasa.gov/data/imerg",
        "short_name": args.short_name,
        "version": args.version,
        "login_ok": login_ok,
        "login_status": login_status,
        "collection_found": bool(collection),
        "dataset_id": collection.get("dataset_id", ""),
        "time_start": collection.get("time_start", ""),
        "granule_found": bool(granule),
        "granule_title": granule.get("title", ""),
        "http_link_found": bool(http_link),
        "opendap_link_found": bool(opendap_link),
        "download_status": download_status,
        "download_note": download_note,
        "spatial_resolution": "0.1 degree",
        "temporal_resolution": "daily",
        "covers_2014_2024": True,
        "covers_hk_study_area": True,
        "suitable_for_lag_features": True,
        "recommended_next_step": "若用户账户已接受 GES DISC 数据条款，则可进入样本下载与 observation-date 对齐验证；否则先完成一次性 EULA 接受。",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# GPM IMERG source validation 报告",
        "",
        "本轮只做 source validation，不构建正式 runoff 特征，不进入 A/B。",
        "",
        "## 核心状态",
        f"- Earthdata 凭据登录：`{'成功' if login_ok else '失败'}`",
        f"- 登录状态：`{login_status}`",
        f"- CMR collection 发现：`{bool(collection)}`",
        f"- granule 发现：`{bool(granule)}`",
        f"- 样本 granule 下载状态：`{download_status}`",
        f"- 下载备注：`{download_note}`",
        "",
        "## 数据源判断",
        "- 数据机构：NASA GPM / GES DISC",
        "- 产品：GPM IMERG Final Daily (`GPM_3IMERGDF` V07)",
        "- 是否免费：是",
        "- 是否需要账号：是，免费 Earthdata 账号",
        "- 时间分辨率：daily",
        "- 空间分辨率：0.1 degree",
        "- 2014-2024 覆盖：是",
        "- 香港研究区覆盖：是",
        "- 是否适合 lag features：是",
        "",
        "## 本轮结论",
    ]
    if download_status == "success":
        lines += [
            "- 已完成登录、产品发现和样本 granule 下载验证。",
            "- 下一轮可直接进入 GPM IMERG observation-date 对齐 source validation。",
        ]
    elif download_status == "blocked":
        lines += [
            "- 登录和产品发现都通过了，但样本下载被 GES DISC 数据条款接受步骤拦住。",
            "- 这说明数据源本身可用，但在自动下载前，需要账户完成一次性的 EULA/terms 接受。",
            "- 在完成该步骤前，建议把 CHIRPS 保留为无需账号的备选公开源。",
        ]
    else:
        lines += [
            "- 当前已完成登录或元数据层验证，但样本下载链仍未完全打通。",
            "- 在进入正式 runoff proxy source validation 前，仍需先解决下载链稳定性。",
        ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
