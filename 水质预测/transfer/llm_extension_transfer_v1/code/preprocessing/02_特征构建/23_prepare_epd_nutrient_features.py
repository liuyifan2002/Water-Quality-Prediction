from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG = Path("配置/历史_v0配置.json")
OUT = Path("数据/处理后数据/诊断数据/epd_surface_nutrient_features_2014_2024.csv")
REPORT = Path("结果/扩展实验/chla_optimization/epd_nutrient_feature_report.csv")


RENAME = {
    "Station": "station",
    "Dates": "date",
    "Depth": "layer",
    "Chlorophyll-a (μg/L)": "Chl_a",
    "Ammonia Nitrogen (mg/L)": "NH4_N",
    "Nitrate Nitrogen (mg/L)": "NO3_N",
    "Nitrite Nitrogen (mg/L)": "NO2_N",
    "Orthophosphate Phosphorus (mg/L)": "PO4_P",
    "Total Inorganic Nitrogen (mg/L)": "TIN",
    "Total Kjeldahl Nitrogen (mg/L)": "TKN",
    "Total Nitrogen (mg/L)": "TN",
    "Total Phosphorus (mg/L)": "TP",
    "Silica (mg/L)": "SiO2",
    "Temperature (°C)": "water_temp_epd",
    "Salinity (psu)": "salinity",
    "Turbidity (NTU)": "turbidity",
    "Suspended Solids (mg/L)": "SS",
    "Dissolved Oxygen (mg/L)": "DO_mgL",
    "Dissolved Oxygen (%saturation)": "DO_sat",
    "pH": "pH",
    "Phaeo-pigments (μg/L)": "phaeo_pigments",
}


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def load_raw() -> pd.DataFrame:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    water_dir = Path(cfg["epd_water_dir"])
    frames = []
    for year in range(int(cfg["year_start"]), int(cfg["year_end"]) + 1):
        path = water_dir / f"marine-historical-{year}-en.csv"
        frame = pd.read_csv(path)
        frame["source_year"] = year
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data = data[
        data["station"].isin(cfg["representative_stations"])
        & data["year"].between(int(cfg["year_start"]), int(cfg["year_end"]))
        & data["layer"].eq(cfg["layer"])
    ].copy()
    numeric_cols = [col for col in RENAME.values() if col not in {"station", "date", "layer"}]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = to_number(data[col])
    return data


def build_features() -> pd.DataFrame:
    data = load_raw()
    keep_cols = [
        "date", "station",
        "NH4_N", "NO3_N", "NO2_N", "PO4_P", "TIN", "TKN",
        "SiO2", "water_temp_epd", "salinity", "turbidity", "SS",
        "DO_mgL", "DO_sat", "pH", "phaeo_pigments",
    ]
    out = data[keep_cols].sort_values(["station", "date"]).reset_index(drop=True)
    out["DIN_calc"] = out[["NH4_N", "NO3_N", "NO2_N"]].sum(axis=1, min_count=1)
    out["NOx_N"] = out[["NO3_N", "NO2_N"]].sum(axis=1, min_count=1)
    out["DIN_TP_ratio"] = out["DIN_calc"] / out["PO4_P"].replace(0, np.nan)
    out["TN_TP_ratio"] = data["TN"].to_numpy() / data["TP"].replace(0, np.nan).to_numpy()

    lag_cols = [
        "NH4_N", "NO3_N", "NO2_N", "PO4_P", "TIN", "DIN_calc", "NOx_N",
        "DIN_TP_ratio", "TN_TP_ratio", "SiO2", "water_temp_epd",
        "salinity", "turbidity", "SS", "DO_mgL", "pH", "phaeo_pigments",
    ]
    grouped = out.groupby("station", sort=False)
    for col in lag_cols:
        out[f"{col}_lag1"] = grouped[col].shift(1)
        out[f"{col}_lag2"] = grouped[col].shift(2)
        out[f"{col}_roll3"] = grouped[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        out[f"{col}_delta1"] = out[f"{col}_lag1"] - out[f"{col}_lag2"]
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    features = build_features()
    features.to_csv(OUT, index=False, encoding="utf-8-sig")
    report_rows = []
    for col in features.columns:
        if col in {"date", "station"}:
            continue
        report_rows.append(
            {
                "feature": col,
                "missing_rate": float(features[col].isna().mean()),
                "non_missing": int(features[col].notna().sum()),
            }
        )
    report = pd.DataFrame(report_rows).sort_values("missing_rate")
    report.to_csv(REPORT, index=False, encoding="utf-8-sig")
    print(f"写出 {OUT}: {features.shape}")
    print(f"写出 {REPORT}")
    print(report.head(20).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
