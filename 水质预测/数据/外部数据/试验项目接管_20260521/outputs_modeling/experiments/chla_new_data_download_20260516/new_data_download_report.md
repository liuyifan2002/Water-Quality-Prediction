# Chl-a 新增数据源下载总报告

- 日期标签：`20260516`
- dry-run：`True`
- 时间范围：`2014-01-01` 到 `2024-12-31`

## 步骤结果
- `epd_long_history`: `success`
  stdout: `Dry-run only. Planned EPD long-history preparation for 1986-2024.`
- `lagged_cross_wq`: `success`
  stdout: `Dry-run only. Planned lagged cross-water-quality feature preparation.`
- `gee_ocean_color`: `success`
  stdout: `Dry-run only. Generated GEE JS scaffold and export instructions.`
- `cmems_physics`: `success`
  stdout: `Dry-run only. Planned Copernicus Marine physics download/extraction.`
- `cmems_bgc`: `success`
  stdout: `Dry-run only. Planned Copernicus Marine BGC download/extraction.`
- `cmems_ocean_colour_discovery`: `success`
  stdout: `Dry-run only. Wrote Ocean Colour candidate tables to C:\Users\刘逸凡\.codex\worktrees\9ff7\codex\水质预测\outputs\modeling\experiments\cmems_ocean_colour_discovery_20260516`
- `cmems_ocean_colour_features`: `success`
  stdout: `Dry-run only. Planned CMEMS Ocean Colour download/extraction.`
- `validate_new_sources`: `success`
  stdout: `Saved validation outputs to C:\Users\刘逸凡\.codex\worktrees\9ff7\codex\水质预测\outputs\modeling\experiments\chla_new_data_source_validation_20260516
ready_for_ab_experiment: lagged_cross_water_quality`

## 校验结果
- ready_for_ab_experiment：`lagged_cross_water_quality`
- template_only：`copernicus_bgc, copernicus_physics, long_history_chla, ocean_color_kd490_tss, sentinel3_olci`
- missing_real_input：`none`
