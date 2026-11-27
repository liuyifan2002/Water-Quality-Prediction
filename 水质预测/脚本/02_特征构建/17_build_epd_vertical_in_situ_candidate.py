"""Build an isolated EPD in-situ vertical candidate table; never alter strict features."""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "结果/诊断结果/全量数据来源与质量复核_v1_20261115/真实补取/EPD真实海水年度_1986_2024"
MASTER = ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv"
OUT = ROOT / "数据/处理后数据/衍生版本/EPD原位垂向候选_v1_20261126"
KEY = ["station", "date", "sample_no", "depth"]
FIELDS = {
    "temperature_c": "Temperature (°C)",
    "salinity_psu": "Salinity (psu)",
    "do_mg_l": "Dissolved Oxygen (mg/L)",
    "turbidity_ntu": "Turbidity (NTU)",
    "ph": "pH",
}
DEPTH = {"Surface Water": "surface", "Middle Water": "middle", "Bottom Water": "bottom"}
TASKS = [
    ("Chl-a表层", "chla_ug_l", "surface", "chla"),
    ("Chl-a中层", "chla_ug_l", "middle", "chla"),
    ("Chl-a底层", "chla_ug_l", "bottom", "chla"),
    ("Secchi", "secchi_depth_m", "surface", None),
    ("TN表层", "tn_mg_l", "surface", "tn"),
    ("TN中层", "tn_mg_l", "middle", "tn"),
    ("TN底层", "tn_mg_l", "bottom", "tn"),
    ("TP表层", "tp_mg_l", "surface", "tp"),
    ("TP中层", "tp_mg_l", "middle", "tp"),
    ("TP底层", "tp_mg_l", "bottom", "tp"),
]


def density0(t: pd.Series, s: pd.Series) -> pd.Series:
    """EOS-80 density at reference pressure 0 dbar, kg/m3; not in-situ density."""
    rw = 999.842594 + 6.793952e-2*t - 9.095290e-3*t**2 + 1.001685e-4*t**3 - 1.120083e-6*t**4 + 6.536332e-9*t**5
    a = 0.824493 - 4.0899e-3*t + 7.6438e-5*t**2 - 8.2467e-7*t**3 + 5.3875e-9*t**4
    b = -5.72466e-3 + 1.0227e-4*t - 1.6546e-6*t**2
    return rw + a*s + b*s.pow(1.5) + 4.8314e-4*s**2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_files = [RAW / f"marine-historical-{year}-en.csv" for year in range(2014, 2025)]
    if not all(p.is_file() for p in raw_files):
        raise FileNotFoundError("2014—2024 年 EPD 原始 CSV 不齐全")
    raw = pd.concat([pd.read_csv(p, encoding="utf-8-sig", low_memory=False) for p in raw_files], ignore_index=True)
    raw = raw.rename(columns={"Station": "station", "Dates": "date", "Sample No": "sample_no", "Depth": "depth"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    raw["depth"] = raw["depth"].map(DEPTH)
    if raw["depth"].isna().any() or raw.duplicated(KEY).any():
        raise ValueError("EPD 水层或原始样本键无法唯一确定")
    for field, original in FIELDS.items():
        raw[field] = pd.to_numeric(raw[original], errors="coerce")
    master_cols = ["obs_id", "sample_id", *KEY, "chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l", *FIELDS]
    master = pd.read_csv(MASTER, usecols=master_cols, low_memory=False)
    if not master["obs_id"].is_unique or master.duplicated(KEY).any():
        raise ValueError("主表 obs_id 或原始样本键不唯一")
    aligned = master.merge(raw[KEY + list(FIELDS)], on=KEY, how="left", validate="one_to_one", indicator=True, suffixes=("_master", ""))
    if (aligned["_merge"] != "both").any() or len(aligned) != len(raw):
        raise ValueError("原始 EPD CSV 与主表无法逐行对齐")
    for field in FIELDS:
        a = aligned[field].to_numpy(dtype=float)
        b = aligned[f"{field}_master"].to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True, atol=1e-10, rtol=0):
            raise ValueError(f"主表与 EPD 原始值不一致：{field}")
    aligned = aligned.drop(columns=["_merge"])
    if (aligned.groupby("sample_id")["sample_no"].nunique() > 1).any():
        raise ValueError("sample_id 与样品编号的关系不唯一")

    event_key = ["sample_id", "station", "date", "sample_no"]
    profile = aligned[event_key + ["depth", *FIELDS]].pivot(index=event_key, columns="depth", values=list(FIELDS)[:-1])
    profile.columns = [f"epd_{field}_{depth}" for field, depth in profile.columns]
    profile = profile.reset_index()
    for field, unit in [("temperature_c", "c"), ("salinity_psu", "psu"), ("do_mg_l", "mg_l")]:
        profile[f"epd_surface_minus_bottom_{field}"] = profile[f"epd_{field}_surface"] - profile[f"epd_{field}_bottom"]
    tsb = [f"epd_{f}_{z}" for f in ("temperature_c", "salinity_psu") for z in ("surface", "bottom")]
    profile["epd_density0_surface_minus_bottom_kg_m3"] = np.where(
        profile[tsb].notna().all(axis=1),
        density0(profile["epd_temperature_c_surface"], profile["epd_salinity_psu_surface"])
        - density0(profile["epd_temperature_c_bottom"], profile["epd_salinity_psu_bottom"]),
        np.nan,
    )
    features = [c for c in profile if c.startswith("epd_")]
    candidate = master[["obs_id", "sample_id", *KEY]].merge(profile, on=event_key, how="left", validate="many_to_one")
    assert len(candidate) == len(master) and candidate.obs_id.equals(master.obs_id)
    candidate_path = OUT / "epd_vertical_candidate_features_2014_2024.csv"
    candidate.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    universes = {}
    for family in ("chla", "tn", "tp"):
        p = ROOT / f"runs/{family}_family_v2/01_station_universe.csv"
        u = pd.read_csv(p)
        universes[family] = set(u.loc[u.common_station, "station_id"])
    checks = []
    joined = master[["obs_id", "station", "depth", "chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l"]].merge(
        candidate[["obs_id", *features]], on="obs_id", validate="one_to_one"
    )
    for task, target, depth, family in TASKS:
        rows = joined[(joined.depth == depth) & joined[target].notna()]
        if family:
            rows = rows[rows.station.isin(universes[family])]
        for field in features:
            good = rows[field].notna()
            checks.append({"任务": task, "候选变量": field, "任务有效记录数": len(rows), "有值记录数": int(good.sum()),
                           "有效覆盖率": float(good.mean()) if len(rows) else np.nan,
                           "任务站点数": rows.station.nunique(), "有值站点数": rows.loc[good, "station"].nunique()})
    coverage = pd.DataFrame(checks)
    coverage.to_csv(OUT / "十任务候选特征覆盖率.csv", index=False, encoding="utf-8-sig")
    event_complete = profile[[f"epd_{f}_{z}" for f in list(FIELDS)[:4] for z in ("surface", "middle", "bottom")]].notna().all(axis=1)
    lines = ["# EPD 原位垂向候选特征说明", "", "- 来源：本地 EPD 2014—2024 年度原始 CSV；按站点、日期、样品编号、水层与正式主表逐行核对。", "- 此表是独立候选版本，保留原始 obs_id；未修改正式 77 特征、目标、划分或模型。", "- 同期多水层输入只适用于完成同日原位采样后的回顾性预测，不代表提前可获得的环境数据。", "- 差值方向均为表层减底层；缺少对应层实测值时保持空值。", "- 密度差采用 EOS-80 在 0 dbar 参考压力下的密度差；因没有每层实测压力，不是原位压力下的真实密度差。", "", f"- 候选表：{len(candidate)} 行、{candidate.station.nunique()} 站；新增 {len(features)} 个候选变量。", f"- 三层四项变量全部齐全的采样事件：{int(event_complete.sum())}/{len(profile)}。", "", "## 变量非缺失数（按采样事件）", ""]
    lines += [f"- `{field}`：{int(profile[field].notna().sum())}/{len(profile)}" for field in features]
    lines += ["", "## 文件 SHA-256", "", f"- 候选表：`{sha256(candidate_path)}`", f"- 正式主表（只读来源）：`{sha256(MASTER)}`", ""]
    (OUT / "候选特征说明.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"候选表 {len(candidate)} 行，{candidate.station.nunique()} 站，{len(features)} 个变量；完整三层事件 {int(event_complete.sum())}/{len(profile)}")
    print(candidate_path)


if __name__ == "__main__":
    main()
