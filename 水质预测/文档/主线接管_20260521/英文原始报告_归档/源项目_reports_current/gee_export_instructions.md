# GEE 导出说明

- 本轮 Sentinel-2 / Landsat highres patch 已经成功导入 current/，因此这份说明转为“如何重跑或补跑”。
- 如果后续需要重跑，请继续使用：
  - `scripts/data/export_highres_patch_gee_sentinel2.js`
  - `scripts/data/export_highres_patch_gee_landsat.js`
- 导出后建议文件名：
  - `s2_patch_features_2014_2024.csv`
  - `landsat_patch_features_2014_2024.csv`
- 放回目录：`data/external/gee_exports/`，然后再由 Codex 读取、校验并合并。