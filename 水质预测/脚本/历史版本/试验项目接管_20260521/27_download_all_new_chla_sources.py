from __future__ import annotations

import argparse
import json
from pathlib import Path

from new_data_sources_utils import StepResult, date_tag_or_today, ensure_dir, experiment_dir, load_yaml_config, run_script, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按顺序执行新增 Chl-a 数据源的下载、制备与校验。")
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--date-tag", default="20260516")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-epd", action="store_true")
    parser.add_argument("--skip-lagged-cross-wq", action="store_true")
    parser.add_argument("--skip-gee", action="store_true")
    parser.add_argument("--skip-cmems-physics", action="store_true")
    parser.add_argument("--skip-cmems-bgc", action="store_true")
    parser.add_argument("--run-cmems-ocean-colour", action="store_true")
    parser.add_argument("--skip-cmems-ocean-colour-download", action="store_true")
    parser.add_argument("--skip-cmems-download", action="store_true")
    parser.add_argument("--gee-station-asset-id", default="")
    parser.add_argument("--gee-dataset-id-olci", default="")
    parser.add_argument("--gee-dataset-id-kd490", default="")
    parser.add_argument("--cmems-physics-dataset-id", default="")
    parser.add_argument("--cmems-bgc-dataset-id", default="")
    parser.add_argument("--cmems-ocean-colour-dataset-id", default="")
    parser.add_argument("--ocean-colour-variables", default="")
    return parser.parse_args()


def run_step(step: str, script_name: str, args: list[str]) -> StepResult:
    completed = run_script(script_name, args)
    return StepResult(
        step=step,
        script=script_name,
        returncode=completed.returncode,
        status="success" if completed.returncode == 0 else "failed",
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def format_step_line(item: StepResult) -> list[str]:
    lines = [f"- `{item.step}`: `{item.status}`"]
    if item.stdout.strip():
        lines.append(f"  stdout: `{item.stdout.strip()}`")
    if item.stderr.strip():
        lines.append(f"  stderr: `{item.stderr.strip()}`")
    return lines


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    date_tag = date_tag_or_today(args.date_tag)
    report_dir = ensure_dir(experiment_dir("chla_new_data_download", date_tag))

    common_args = ["--start-date", args.start_date, "--end-date", args.end_date, "--date-tag", date_tag]
    long_history_args = [
        "--start-date",
        str(config["long_history_period"]["start_date"]),
        "--end-date",
        str(config["long_history_period"]["end_date"]),
        "--date-tag",
        date_tag,
    ]
    if args.force:
        common_args.append("--force")
        long_history_args.append("--force")
    if args.dry_run:
        common_args.append("--dry-run")
        long_history_args.append("--dry-run")

    steps: list[tuple[str, str, bool]] = [
        ("epd_long_history", "22_download_epd_long_history.py", not args.skip_epd and config["source_switches"].get("download_epd_long_history", True)),
        ("lagged_cross_wq", "23_prepare_lagged_cross_wq_features.py", not args.skip_lagged_cross_wq),
        ("gee_ocean_color", "24_extract_gee_ocean_color_features.py", not args.skip_gee and config["source_switches"].get("extract_gee_olci", True)),
        ("cmems_physics", "25_download_cmems_physics_features.py", not args.skip_cmems_physics and config["source_switches"].get("download_cmems_physics", True)),
        ("cmems_bgc", "26_download_cmems_bgc_features.py", not args.skip_cmems_bgc and config["source_switches"].get("download_cmems_bgc", True)),
        ("cmems_ocean_colour_discovery", "29_discover_cmems_ocean_colour_products.py", args.run_cmems_ocean_colour and config["source_switches"].get("download_cmems_ocean_colour", True)),
        ("cmems_ocean_colour_features", "30_download_cmems_ocean_colour_features.py", args.run_cmems_ocean_colour and config["source_switches"].get("download_cmems_ocean_colour", True)),
    ]

    results: list[StepResult] = []
    for step_name, script_name, enabled in steps:
        if not enabled:
            results.append(StepResult(step=step_name, script=script_name, returncode=0, status="skipped", stdout="", stderr=""))
            continue

        if step_name == "epd_long_history":
            extra_args = list(long_history_args)
        elif step_name == "cmems_ocean_colour_discovery":
            extra_args = ["--date-tag", date_tag]
            if args.dry_run:
                extra_args.append("--dry-run")
        else:
            extra_args = list(common_args)
        if step_name == "gee_ocean_color":
            extra_args.append("--export-to-drive")
            if args.gee_station_asset_id:
                extra_args.extend(["--station-asset-id", args.gee_station_asset_id])
            if args.gee_dataset_id_olci:
                extra_args.extend(["--dataset-id-olci", args.gee_dataset_id_olci])
            if args.gee_dataset_id_kd490:
                extra_args.extend(["--dataset-id-kd490", args.gee_dataset_id_kd490])
        elif step_name == "cmems_physics":
            if args.skip_cmems_download:
                extra_args.append("--skip-download")
            if args.cmems_physics_dataset_id:
                extra_args.extend(["--dataset-id", args.cmems_physics_dataset_id])
        elif step_name == "cmems_bgc":
            if args.skip_cmems_download:
                extra_args.append("--skip-download")
            if args.cmems_bgc_dataset_id:
                extra_args.extend(["--dataset-id", args.cmems_bgc_dataset_id])
        elif step_name == "cmems_ocean_colour_features":
            if args.skip_cmems_ocean_colour_download:
                extra_args.append("--skip-download")
            if args.cmems_ocean_colour_dataset_id:
                extra_args.extend(["--dataset-id", args.cmems_ocean_colour_dataset_id])
            if args.ocean_colour_variables:
                extra_args.extend(["--variables", args.ocean_colour_variables])

        results.append(run_step(step_name, script_name, extra_args))

    validator = run_step("validate_new_sources", "21_validate_new_chla_data_sources.py", ["--date-tag", date_tag])
    results.append(validator)

    readiness_summary = {"ready_sources": [], "template_only_sources": [], "missing_real_input_sources": []}
    validation_report = report_dir.parent / f"chla_new_data_source_validation_{date_tag}" / "new_data_source_validation_report.json"
    if validation_report.exists():
        readiness_summary = json.loads(validation_report.read_text(encoding="utf-8"))

    report_json = {
        "label": "Chl-a 新增数据源下载总控",
        "date_tag": date_tag,
        "dry_run": args.dry_run,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "steps": [
            {
                "step": item.step,
                "script": item.script,
                "status": item.status,
                "returncode": item.returncode,
                "stdout": item.stdout.strip(),
                "stderr": item.stderr.strip(),
            }
            for item in results
        ],
        "readiness_summary": {
            "ready_sources": readiness_summary.get("ready_sources", []),
            "template_only_sources": readiness_summary.get("template_only_sources", []),
            "missing_real_input_sources": readiness_summary.get("missing_real_input_sources", []),
        },
    }
    write_json(report_dir / "new_data_download_report.json", report_json)

    lines = [
        f"- 日期标签：`{date_tag}`",
        f"- dry-run：`{args.dry_run}`",
        f"- 时间范围：`{args.start_date}` 到 `{args.end_date}`",
        "",
        "## 步骤结果",
    ]
    for item in results:
        lines.extend(format_step_line(item))
    lines.extend(
        [
            "",
            "## 校验结果",
            f"- ready_for_ab_experiment：`{', '.join(readiness_summary.get('ready_sources', [])) or 'none'}`",
            f"- template_only：`{', '.join(readiness_summary.get('template_only_sources', [])) or 'none'}`",
            f"- missing_real_input：`{', '.join(readiness_summary.get('missing_real_input_sources', [])) or 'none'}`",
        ]
    )
    write_markdown(report_dir / "new_data_download_report.md", "Chl-a 新增数据源下载总报告", lines)
    print(f"Saved report to {report_dir}")


if __name__ == "__main__":
    main()
