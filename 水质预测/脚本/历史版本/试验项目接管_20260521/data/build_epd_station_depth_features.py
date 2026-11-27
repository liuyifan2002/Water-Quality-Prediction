from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PDF = PROJECT_ROOT / "tmp" / "official_depth_sources" / "epd_2002_report.pdf"
DEFAULT_STATION_META = PROJECT_ROOT / "data" / "model_input" / "station_meta.csv"
DEFAULT_MODEL_DATASET = (
    PROJECT_ROOT / "data" / "interim" / "current" / "current_model_dataset_with_highres_and_mechanism.csv"
)
DEFAULT_STATION_OUT = PROJECT_ROOT / "data" / "processed" / "current" / "station_depth_features.csv"
DEFAULT_DEPTH_OUT = (
    PROJECT_ROOT / "data" / "processed" / "current" / "depth_bathymetry_features_2014_2024.csv"
)
DEFAULT_JSON_OUT = (
    PROJECT_ROOT / "data" / "processed" / "current" / "station_depth_feature_validation.json"
)
DEFAULT_REPORT_OUT = (
    PROJECT_ROOT / "experiments" / "reports" / "current" / "station_depth_feature_validation_report.md"
)


LAYER_FRACTIONS = {
    "surface": 0.2,
    "middle": 0.6,
    "bottom": 0.9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从香港 EPD 年报附录抽取站点近似水深并构建当前特征表")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--station-meta", default=str(DEFAULT_STATION_META))
    parser.add_argument("--model-dataset", default=str(DEFAULT_MODEL_DATASET))
    parser.add_argument("--station-out", default=str(DEFAULT_STATION_OUT))
    parser.add_argument("--depth-out", default=str(DEFAULT_DEPTH_OUT))
    parser.add_argument("--validation-json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--validation-report-out", default=str(DEFAULT_REPORT_OUT))
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")
    text = text.replace("Ёу", " ")
    text = text.replace("КЃ", " ")
    return text


def load_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pieces: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            pieces.append(normalize_text(text))
    return "\n".join(pieces)


def extract_station_depths(full_text: str, station_codes: list[str]) -> dict[str, float]:
    depth_map: dict[str, float] = {}
    for station in station_codes:
        pattern = re.compile(
            rf"(?<![A-Z0-9])\*?{re.escape(station)}(?![A-Z0-9]).{{0,180}}?E\s+(\d{{1,2}})(?!\d)",
            re.S,
        )
        match = pattern.search(full_text)
        if match:
            depth_map[station] = float(match.group(1))
    return depth_map


def build_station_depth_table(station_meta: pd.DataFrame, depth_map: dict[str, float]) -> pd.DataFrame:
    out = station_meta.rename(columns={"Station": "station"}).copy()
    out["station_depth_approx_m"] = out["station"].map(depth_map)
    out["station_depth_source"] = np.where(
        out["station_depth_approx_m"].notna(),
        "Hong Kong EPD 2002 Appendix A approximate depth",
        "not_found_in_epd_2002_appendix_a",
    )
    out["station_depth_missing_flag"] = out["station_depth_approx_m"].isna().astype(int)
    out["is_shallow_station_lt5m"] = (
        out["station_depth_approx_m"].notna() & (out["station_depth_approx_m"] < 5)
    ).astype(int)
    out["is_shallow_station_lt10m"] = (
        out["station_depth_approx_m"].notna() & (out["station_depth_approx_m"] < 10)
    ).astype(int)
    out["is_deep_station_ge20m"] = (
        out["station_depth_approx_m"].notna() & (out["station_depth_approx_m"] >= 20)
    ).astype(int)
    out["log_station_depth_approx_m"] = np.log1p(out["station_depth_approx_m"])
    return out


def build_observation_depth_features(model_df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "obs_id",
        "sample_id",
        "station",
        "date",
        "depth",
        "split",
        "lon",
        "lat",
        "water_zone_en",
        "year",
        "month",
        "is_wet_season",
    ]
    base = model_df[keep].copy()
    merged = base.merge(
        station_df[
            [
                "station",
                "station_depth_approx_m",
                "station_depth_source",
                "station_depth_missing_flag",
                "is_shallow_station_lt5m",
                "is_shallow_station_lt10m",
                "is_deep_station_ge20m",
                "log_station_depth_approx_m",
            ]
        ],
        on="station",
        how="left",
        validate="many_to_one",
    )
    merged["layer_fraction_proxy"] = merged["depth"].map(LAYER_FRACTIONS)
    merged["estimated_sample_depth_m_proxy"] = (
        merged["station_depth_approx_m"] * merged["layer_fraction_proxy"]
    )
    merged["bottom_gap_m_proxy"] = merged["station_depth_approx_m"] * (
        1.0 - merged["layer_fraction_proxy"]
    )
    merged["station_depth_x_wet_season"] = (
        merged["station_depth_approx_m"] * merged["is_wet_season"].fillna(0)
    )
    merged["station_depth_x_month"] = merged["station_depth_approx_m"] * merged["month"]
    merged["depth_source_group"] = np.where(
        merged["station_depth_approx_m"].notna(),
        "epd_station_depth",
        "missing_station_depth",
    )
    merged["bathymetry_source"] = "station_depth_only_no_gebco_yet"
    merged["bathymetry_risk_flag"] = np.where(
        merged["station_depth_approx_m"].notna(),
        "approximate_station_depth_not_gridded_bathymetry",
        "station_depth_unavailable",
    )
    return merged


def write_validation(
    station_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    json_path: Path,
    report_path: Path,
) -> None:
    station_count = int(len(station_df))
    found_station_count = int(station_df["station_depth_approx_m"].notna().sum())
    coverage_rate = float(found_station_count / station_count) if station_count else 0.0
    obs_count = int(len(obs_df))
    obs_covered = int(obs_df["station_depth_approx_m"].notna().sum())
    obs_coverage_rate = float(obs_covered / obs_count) if obs_count else 0.0

    by_zone = (
        obs_df.groupby("water_zone_en", dropna=False)["station_depth_approx_m"]
        .apply(lambda s: float(s.notna().mean()))
        .reset_index(name="observation_coverage_rate")
        .sort_values("observation_coverage_rate", ascending=True)
    )
    shallow_examples = (
        station_df[station_df["station_depth_approx_m"].notna()]
        .sort_values("station_depth_approx_m")
        .head(8)[["station", "station_depth_approx_m"]]
        .to_dict(orient="records")
    )
    deep_examples = (
        station_df[station_df["station_depth_approx_m"].notna()]
        .sort_values("station_depth_approx_m", ascending=False)
        .head(8)[["station", "station_depth_approx_m"]]
        .to_dict(orient="records")
    )

    payload = {
        "source_name": "Hong Kong EPD 2002 Appendix A approximate station depth",
        "source_type": "official_report_table",
        "station_count": station_count,
        "station_depth_found_count": found_station_count,
        "station_coverage_rate": coverage_rate,
        "observation_count": obs_count,
        "observation_covered_count": obs_covered,
        "observation_coverage_rate": obs_coverage_rate,
        "notes": [
            "该来源提供的是站点近似总水深，不是逐日 bathymetry 栅格。",
            "该来源可作为 station depth / relative sampling depth proxy 使用。",
            "当前不应将其表述为 GEBCO gridded bathymetry 或真实底层过程观测。",
        ],
        "zone_observation_coverage": by_zone.to_dict(orient="records"),
        "shallow_examples": shallow_examples,
        "deep_examples": deep_examples,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 香港 EPD 站点水深特征验证报告",
        "",
        "## 1. 数据来源",
        "",
        "- 来源：香港 EPD 2002 年 marine water quality 年报附录 A",
        "- 性质：官方报告中的 **站点近似总水深**",
        "- 适用口径：`station depth / relative depth proxy`",
        "- 不应表述为：GEBCO 栅格 bathymetry、逐日底层观测、ADCP 或垂向剖面",
        "",
        "## 2. 覆盖情况",
        "",
        f"- 站点总数：{station_count}",
        f"- 成功匹配近似水深的站点数：{found_station_count}",
        f"- 站点覆盖率：{coverage_rate:.4f}",
        f"- 当前观测样本总数：{obs_count}",
        f"- 当前观测样本覆盖数：{obs_covered}",
        f"- 当前观测样本覆盖率：{obs_coverage_rate:.4f}",
        "",
        "## 3. 风险与限制",
        "",
        "1. 该表给的是近似总水深，不是高分辨率 bathymetry 栅格。",
        "2. 该表不能替代 station exact sampling datetime、bottom temperature、ADCP current 等真实底层机制数据。",
        "3. 该表可以帮助四指标模型学习“深浅 / 内湾外湾 / 水体交换潜势”的长期静态差异。",
        "",
        "## 4. 代表性站点",
        "",
        "### 较浅站点示例",
        "",
    ]
    for item in shallow_examples:
        lines.append(f"- {item['station']}: {item['station_depth_approx_m']:.1f} m")
    lines.extend(["", "### 较深站点示例", ""])
    for item in deep_examples:
        lines.append(f"- {item['station']}: {item['station_depth_approx_m']:.1f} m")
    lines.extend(["", "## 5. 结论", ""])
    lines.append("- 该来源已经足够进入四指标统一单源 A/B，优先验证其对 TN surface / TP surface / TN middle / TN bottom 的增益。")
    lines.append("- 若后续拿到 GEBCO 或更细的香港海图/测深数值产品，应把它作为独立 bathymetry 数据源再次做 source validation。")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf)
    station_meta_path = Path(args.station_meta)
    model_dataset_path = Path(args.model_dataset)
    station_out = Path(args.station_out)
    depth_out = Path(args.depth_out)
    validation_json_out = Path(args.validation_json_out)
    validation_report_out = Path(args.validation_report_out)

    station_meta = pd.read_csv(station_meta_path)
    model_df = pd.read_csv(model_dataset_path, low_memory=False)
    model_df["date"] = pd.to_datetime(model_df["date"])
    full_text = load_pdf_text(pdf_path)
    depth_map = extract_station_depths(full_text, station_meta["Station"].astype(str).tolist())

    station_depth_df = build_station_depth_table(station_meta, depth_map)
    observation_depth_df = build_observation_depth_features(model_df, station_depth_df)

    station_out.parent.mkdir(parents=True, exist_ok=True)
    depth_out.parent.mkdir(parents=True, exist_ok=True)
    station_depth_df.to_csv(station_out, index=False, encoding="utf-8-sig")
    observation_depth_df.to_csv(depth_out, index=False, encoding="utf-8-sig")
    write_validation(station_depth_df, observation_depth_df, validation_json_out, validation_report_out)

    print(
        json.dumps(
            {
                "station_count": int(len(station_depth_df)),
                "station_depth_found_count": int(station_depth_df["station_depth_approx_m"].notna().sum()),
                "observation_count": int(len(observation_depth_df)),
                "observation_covered_count": int(observation_depth_df["station_depth_approx_m"].notna().sum()),
                "station_out": str(station_out),
                "depth_out": str(depth_out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
