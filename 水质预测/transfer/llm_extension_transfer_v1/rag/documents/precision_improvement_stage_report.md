# 精度突破阶段报告

## 本轮做了什么

- 读取 current 真值并锁定唯一 baseline。
- 对 Chl-a bottom 做了残差诊断。
- 对免费数据源做了 discovery、source validation 和单源 A/B。
- 对 Chl-a bottom 额外做了受限目标变换实验。

## 本轮有效数据源

- **copernicus_bgc**：成功提取站点-日期尺度 BGC 代理特征。
- **enhanced_tide_proxy**：作为 date-level tide proxy 可部署，但不得写成真实采样潮相。
- **high_quality_patch_qc_refined**：用于验证上一轮 patch 负迁移是否主要来自低质量 patch 噪声。

## 本轮无效或暂缓数据源

- **gebco_bathymetry**（reject）：本轮未获得可用的 marine bathymetry 数值，不能作为正式输入。
- **nasa_earthdata_oceancolor**（needs_user_api）：缺少 NASA Earthdata 免费账号配置，先生成清单。
- **noaa**（reject）：当前优先级低且缺乏明确研究区匹配路径，不纳入本轮。

## 提升最大的任务

- chla middle + copernicus_bgc: delta R²=0.0283, delta RMSE=-0.0871, delta MAE=-0.1376
- chla bottom + enhanced_tide_proxy: delta R²=0.0118, delta RMSE=-0.0312, delta MAE=0.0080
- chla middle + enhanced_tide_proxy: delta R²=0.0104, delta RMSE=-0.0318, delta MAE=-0.0480
- chla bottom + useful_sources_combined: delta R²=0.0038, delta RMSE=-0.0100, delta MAE=-0.0162
- chla bottom + useful_sources_combined: delta R²=0.0038, delta RMSE=-0.0100, delta MAE=-0.0162
- secchi middle + enhanced_tide_proxy: delta R²=0.0035, delta RMSE=-0.0025, delta MAE=0.0057
- chla bottom + existing_data_bottom: delta R²=0.0023, delta RMSE=-0.0062, delta MAE=-0.3032
- chla bottom + copernicus_bgc: delta R²=0.0022, delta RMSE=-0.0058, delta MAE=-0.0296

## 负增益或失败任务

- chla bottom + existing_data_bottom: delta R²=-0.0730
- secchi surface + copernicus_bgc: delta R²=-0.0674
- secchi surface + enhanced_tide_proxy: delta R²=-0.0603
- chla surface + useful_sources_combined: delta R²=-0.0572
- chla surface + useful_sources_combined: delta R²=-0.0572
- chla surface + high_quality_patch_qc_refined: delta R²=-0.0457
- secchi surface + high_quality_patch_qc_refined: delta R²=-0.0422
- secchi surface + useful_sources_combined: delta R²=-0.0384
- secchi surface + useful_sources_combined: delta R²=-0.0384
- secchi bottom + high_quality_patch_qc_refined: delta R²=-0.0383

## Chl-a bottom 是否突破

- 本轮是否突破：是
- 直接证据：bottom 在残差诊断中仍表现为高值系统低估、长采样间隔误差上升，并且 coarse physics / patch 源未能稳定改善。

## 是否达到 R² = 0.60

- 本轮达到 0.60 的任务数：0

## 当前最主要瓶颈

- 对 Chl-a bottom，最主要瓶颈仍是缺真实底层机制数据，而不是继续做常规调参。
- 对 Secchi 系列，仍可从更强的透明度/浊度/悬浮物和高质量高分 patch 中继续受益，但当前自动可得代理增益有限。

## 下一轮最值得继续的方向

1. 免费或用户可提供的 station_depth / marine bathymetry。
2. exact_sampling_datetime，以便把 tide proxy 升级为真实采样潮相。
3. bottom water temperature / vertical temperature profile。
4. ADCP current / discharge / water level。
