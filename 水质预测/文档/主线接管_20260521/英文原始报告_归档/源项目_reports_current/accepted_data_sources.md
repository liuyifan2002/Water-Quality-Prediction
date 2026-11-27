# 通过验证的数据源

- **copernicus_bgc**：成功提取站点-日期尺度 BGC 代理特征。
- **enhanced_tide_proxy**：作为 date-level tide proxy 可部署，但不得写成真实采样潮相。
- **high_quality_patch_qc_refined**：用于验证上一轮 patch 负迁移是否主要来自低质量 patch 噪声。

## 2026-05-20 新增接受项

- `nasa_earthdata_oceancolor`：已完成凭据验证与月尺度样本提取，后续可继续做全时段批量化，但本轮尚未进入正式 A/B。
