from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from new_data_sources_utils import credentials_present, date_tag_or_today, ensure_dir, experiment_dir, load_env_file, load_yaml_config, write_json, write_markdown


DEFAULT_CANDIDATES = [
    {
        "product_family": "Ocean Colour L4 MY daily plankton",
        "keyword": "GlobColour",
        "dataset_id": "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        "temporal_resolution": "P1D",
        "spatial_resolution": "4km",
        "target_start_year": 1997,
        "target_end_year": 2024,
        "variables": ["CHL"],
        "notes": "全球 L4 gapfree 日尺度 plankton 产品，适合提供 CHL 主变量。",
    },
    {
        "product_family": "Ocean Colour L4 MY daily transparency",
        "keyword": "GlobColour",
        "dataset_id": "cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D",
        "temporal_resolution": "P1D",
        "spatial_resolution": "4km",
        "target_start_year": 1997,
        "target_end_year": 2024,
        "variables": ["KD490", "SPM", "ZSD"],
        "notes": "全球 L4 gapfree 日尺度 transparency 产品，适合提供 KD490、SPM 和 ZSD。",
    },
    {
        "product_family": "Ocean Colour L3 MY daily reflectance",
        "keyword": "GlobColour",
        "dataset_id": "cmems_obs-oc_glo_bgc-reflectance_my_l3-multi-4km_P1D",
        "temporal_resolution": "P1D",
        "spatial_resolution": "4km",
        "target_start_year": 1997,
        "target_end_year": 2024,
        "variables": ["RRS443", "RRS490", "RRS510", "RRS560", "RRS665", "RRS709"],
        "notes": "全球 L3 日尺度 reflectance 产品，适合补充 RRS 波段以构造 NDCI/MCI。",
    },
    {
        "product_family": "Ocean Colour L4 MY monthly plankton fallback",
        "keyword": "GlobColour",
        "dataset_id": "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M",
        "temporal_resolution": "P1M",
        "spatial_resolution": "4km",
        "target_start_year": 1997,
        "target_end_year": 2024,
        "variables": ["CHL"],
        "notes": "月尺度 plankton fallback。",
    },
    {
        "product_family": "Ocean Colour L4 MY monthly transparency fallback",
        "keyword": "GlobColour",
        "dataset_id": "cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M",
        "temporal_resolution": "P1M",
        "spatial_resolution": "4km",
        "target_start_year": 1997,
        "target_end_year": 2024,
        "variables": ["KD490", "ZSD", "SPM"],
        "notes": "月尺度 transparency fallback。",
    },
    {
        "product_family": "Ocean Colour L4 MY monthly reflectance fallback",
        "keyword": "GlobColour",
        "dataset_id": "cmems_obs-oc_glo_bgc-reflectance_my_l4-multi-4km_P1M",
        "temporal_resolution": "P1M",
        "spatial_resolution": "4km",
        "target_start_year": 1997,
        "target_end_year": 2024,
        "variables": ["RRS443", "RRS490", "RRS510", "RRS560", "RRS665", "RRS709"],
        "notes": "月尺度 reflectance fallback。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="发现 Copernicus Marine Ocean Colour 候选产品。")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--date-tag", default="20260516")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def probe_copernicusmarine() -> tuple[bool, str]:
    try:
        import copernicusmarine  # type: ignore

        attrs = sorted([name for name in dir(copernicusmarine) if not name.startswith("_")])
        return True, ", ".join(attrs[:80])
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def build_candidate_tables(candidates: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows = []
    variable_rows = []
    for item in candidates:
        row = dict(item)
        start_year = row.get("target_start_year")
        end_year = row.get("target_end_year")
        row["covers_2014_2024"] = bool(start_year is not None and end_year is not None and start_year <= 2014 and end_year >= 2024)
        row["recommended_priority"] = 1 if row["covers_2014_2024"] and row["temporal_resolution"] == "P1D" else 2
        candidate_rows.append(row)
        for variable in row["variables"]:
            variable_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "product_family": row["product_family"],
                    "variable": variable,
                    "temporal_resolution": row["temporal_resolution"],
                    "spatial_resolution": row["spatial_resolution"],
                }
            )
    return pd.DataFrame(candidate_rows), pd.DataFrame(variable_rows)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    env = load_env_file()
    date_tag = date_tag_or_today(args.date_tag)
    out_dir = ensure_dir(args.out_dir or experiment_dir("cmems_ocean_colour_discovery", date_tag))

    username_key = config.get("copernicus_marine", {}).get("username_env_var", "COPERNICUSMARINE_USERNAME")
    password_key = config.get("copernicus_marine", {}).get("password_env_var", "COPERNICUSMARINE_PASSWORD")
    has_creds, missing_keys = credentials_present(env, [username_key, password_key])

    candidate_df, variable_df = build_candidate_tables(DEFAULT_CANDIDATES)
    candidate_df.to_csv(out_dir / "cmems_ocean_colour_candidate_datasets.csv", index=False, encoding="utf-8-sig")
    variable_df.to_csv(out_dir / "cmems_ocean_colour_variable_inventory.csv", index=False, encoding="utf-8-sig")

    recommendation = {
        "label": "Copernicus Marine Ocean Colour 数据集推荐",
        "credentials_required": True,
        "credentials_present": has_creds,
        "missing_credentials": missing_keys,
        "recommended_daily_dataset_id": "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        "recommended_monthly_dataset_id": "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M",
        "recommended_daily_dataset_bundle": [
            "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
            "cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D",
            "cmems_obs-oc_glo_bgc-reflectance_my_l3-multi-4km_P1D",
        ],
        "recommended_monthly_dataset_bundle": [
            "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M",
            "cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M",
            "cmems_obs-oc_glo_bgc-reflectance_my_l4-multi-4km_P1M",
        ],
        "recommended_variables": config.get("cmems_ocean_colour", {}).get(
            "preferred_variables",
            ["CHL", "KD490", "SPM", "ZSD", "RRS443", "RRS490", "RRS510", "RRS560", "RRS665", "RRS681", "RRS709", "PP", "PFT"],
        ),
        "recommended_reason": "优先选择覆盖 2014-2024 的全球历史 MY 产品；CHL、transparency 和 reflectance 需要由多个数据集联合提供，若日尺度可用则优先 P1D，否则 fallback 到 P1M。",
        "covers_2014_2024_candidates": candidate_df.loc[candidate_df["covers_2014_2024"], "dataset_id"].tolist(),
    }

    if args.dry_run:
        recommendation["status"] = "dry_run_only"
        write_json(out_dir / "cmems_ocean_colour_recommendation.json", recommendation)
        write_markdown(
            out_dir / "cmems_ocean_colour_discovery_report.md",
            "CMEMS Ocean Colour 产品发现报告",
            [
                "1. 当前为 dry-run，只输出候选数据集与推荐优先级，不执行真实 catalog 查询。",
                "2. 推荐日尺度组合：`plankton_my_l4-gapfree-multi-4km_P1D + transp_my_l4-gapfree-multi-4km_P1D + reflectance_my_l3-multi-4km_P1D`。",
                "3. 推荐月尺度 fallback：`plankton_my_l4-multi-4km_P1M + transp_my_l4-multi-4km_P1M + reflectance_my_l4-multi-4km_P1M`。",
                "4. 真实 discover/describe 仍需要已配置的 Copernicus Marine 账号和联网环境。",
            ],
        )
        print(f"Dry-run only. Wrote Ocean Colour candidate tables to {out_dir}")
        return

    if not has_creds:
        recommendation["status"] = "missing_credentials"
        write_json(out_dir / "cmems_ocean_colour_recommendation.json", recommendation)
        raise RuntimeError(f"未配置 Copernicus Marine 账号，缺少: {missing_keys}")

    ok, detail = probe_copernicusmarine()
    recommendation["copernicusmarine_probe_ok"] = ok
    recommendation["copernicusmarine_probe_detail"] = detail
    recommendation["status"] = "catalog_probe_only"
    write_json(out_dir / "cmems_ocean_colour_recommendation.json", recommendation)
    write_markdown(
        out_dir / "cmems_ocean_colour_discovery_report.md",
        "CMEMS Ocean Colour 产品发现报告",
        [
            "1. 当前脚本已完成候选清单、推荐结果和账号检查。",
            f"2. `copernicusmarine` 探测结果：`{detail}`。",
            "3. 若后续需要更精确的 dataset inventory，请在真实联网环境中追加 search/describe 调用。",
        ],
    )
    print(f"Saved Ocean Colour discovery outputs to {out_dir}")


if __name__ == "__main__":
    main()
