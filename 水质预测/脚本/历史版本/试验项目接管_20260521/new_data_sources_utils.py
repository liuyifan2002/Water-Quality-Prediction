from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "new_data_sources.yaml"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_scalar(value: str) -> Any:
    raw = value.strip()
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw == "":
        return ""
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _fallback_yaml_parse(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            parent[key] = {}
            stack.append((indent, parent[key]))
        else:
            parent[key] = parse_scalar(value)
    return root


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return _fallback_yaml_parse(text)


def load_env_file(env_path: Path | None = None) -> dict[str, str]:
    env_path = env_path or ROOT / ".env"
    values = dict(os.environ)
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values.setdefault(key.strip(), value.strip())
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path)
        values.update({k: v for k, v in os.environ.items() if v is not None})
    except Exception:
        pass
    return values


def parse_date(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).normalize()


def date_tag_or_today(date_tag: str | None = None) -> str:
    return date_tag or datetime.now().strftime("%Y%m%d")


def experiment_dir(prefix: str, date_tag: str | None = None) -> Path:
    tag = date_tag_or_today(date_tag)
    return ROOT / "outputs" / "modeling" / "experiments" / f"{prefix}_{tag}"


def normalize_station_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def normalize_depth(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if "surface" in text or text in {"s", "sur"}:
        return "surface"
    if "middle" in text or text in {"m", "mid"}:
        return "middle"
    if "bottom" in text or text in {"b", "bot"}:
        return "bottom"
    return text


def standardize_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def station_meta_frame(config: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / config["station_file"]
    df = pd.read_csv(path, low_memory=False)
    cols = {col.lower(): col for col in df.columns}
    station_col = cols.get("station", "Station")
    zone_col = cols.get("water_zone_en", "water_zone_en")
    lon_col = cols.get("lon", "lon")
    lat_col = cols.get("lat", "lat")
    out = df[[station_col, zone_col, lon_col, lat_col]].copy()
    out.columns = ["station", "water_zone_en", "lon", "lat"]
    out["station"] = out["station"].map(normalize_station_name)
    return out.drop_duplicates(subset=["station"])


def sample_frame(config: dict[str, Any], start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    path = ROOT / config["sample_file"]
    df = pd.read_csv(path, low_memory=False)
    cols = {col.lower(): col for col in df.columns}
    station_col = cols.get("station", "Station")
    date_col = cols.get("date", "date")
    df["station"] = df[station_col].map(normalize_station_name)
    df["date"] = standardize_date_series(df[date_col])
    if start_date:
        df = df[df["date"] >= parse_date(start_date)]
    if end_date:
        df = df[df["date"] <= parse_date(end_date)]
    return df


def water_quality_master_frame(config: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / config["water_quality_master_file"]
    df = pd.read_csv(path, low_memory=False)
    df["station"] = df["station"].map(normalize_station_name)
    df["date"] = standardize_date_series(df["date"])
    if "depth" in df.columns:
        df["depth"] = df["depth"].map(normalize_depth)
    return df


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, title: str, lines: list[str]) -> None:
    content = [f"# {title}", ""]
    content.extend(lines)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def credentials_present(values: dict[str, str], keys: list[str]) -> tuple[bool, list[str]]:
    missing = [key for key in keys if not values.get(key)]
    return len(missing) == 0, missing


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator.astype(float).copy()
    den = denominator.astype(float)
    mask = den.notna() & (den != 0)
    out.loc[:] = pd.NA
    out.loc[mask] = numerator.loc[mask] / den.loc[mask]
    return out


def angle_degrees(u: pd.Series, v: pd.Series) -> pd.Series:
    return pd.Series(
        [math.degrees(math.atan2(vv, uu)) % 360 if pd.notna(uu) and pd.notna(vv) else pd.NA for uu, vv in zip(u, v)],
        index=u.index,
    )


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0
    lon1r = math.radians(lon1)
    lat1r = math.radians(lat1)
    lon2r = math.radians(lon2)
    lat2r = math.radians(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def parse_csv_like_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def run_script(script_name: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(ROOT / "scripts" / script_name)] + args
    return subprocess.run(command, capture_output=True, text=True, check=False)


@dataclass
class StepResult:
    step: str
    script: str
    returncode: int
    status: str
    stdout: str
    stderr: str
