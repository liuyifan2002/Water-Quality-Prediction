# 免费数据源验证报告

## 本轮结论

- 通过验证并进入 A/B 的数据源数量：3
- 被拒绝或暂缓的数据源数量：3

## 数据源逐项结果

### gebco_bathymetry
- 官方入口：https://www.gebco.net/data-products/gridded-bathymetry-data
- 访问方式：public
- 数据类型：static_geodata
- 变量：bathymetry_depth_unavailable
- 时空分辨率：static / unknown_this_round
- 缺失率：1.0
- 提取成功率：0.0
- 限制：current local bathymetry table derives from GLOBathy-like exploratory output and is all-null for HK coastal stations
- recommendation：reject
- reason：本轮未获得可用的 marine bathymetry 数值，不能作为正式输入。

### copernicus_bgc
- 官方入口：https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_BGC_001_029/description
- 访问方式：free_account
- 数据类型：model_proxy
- 变量：bgc_chl,bgc_no3,bgc_po4,bgc_si,bgc_o2,bgc_npp,bgc_phyto,bgc_zoo,bgc_no3_7d,bgc_po4_7d,bgc_npp_7d,bgc_chl_anom_7d,bgc_n_to_p_ratio
- 时空分辨率：P1D / 0.25deg
- 缺失率：0.6288634979965736
- 提取成功率：0.43861586600404934
- 限制：再分析/模式代理，不是实测底层营养盐或实测浮游过程。
- recommendation：use
- reason：成功提取站点-日期尺度 BGC 代理特征。

### enhanced_tide_proxy
- 官方入口：https://www.hko.gov.hk/en/tide/index.html
- 访问方式：public
- 数据类型：observation_proxy
- 变量：tidal_range,tide_events,tide_high_mean,tide_low_mean,tide_level_mean,tide_range_same_day,tide_range_previous_3d_mean,tide_range_previous_1d,tide_level_daily_mean,tide_range_anom_prev3d,tide_range_change_prev1d,tide_event_density,tide_level_centered
- 时空分辨率：date-level / nearest available tide station proxy
- 缺失率：0.0
- 提取成功率：1.0
- 限制：缺少 exact_sampling_datetime，不能构造真实采样潮相，只能做 date-level deployable proxy。
- recommendation：use
- reason：作为 date-level tide proxy 可部署，但不得写成真实采样潮相。

### high_quality_patch_qc_refined
- 官方入口：https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
- 访问方式：public
- 数据类型：remote_sensing
- 变量：hq_valid70_dt3_s2_image_count_in_window,hq_valid70_dt3_s2_3x3_B3_mean,hq_valid70_dt3_s2_3x3_B3_median,hq_valid70_dt3_s2_3x3_B3_stdDev,hq_valid70_dt3_s2_3x3_B4_mean,hq_valid70_dt3_s2_3x3_B4_median,hq_valid70_dt3_s2_3x3_B4_stdDev,hq_valid70_dt3_s2_3x3_B5_mean,hq_valid70_dt3_s2_3x3_B5_median,hq_valid70_dt3_s2_3x3_B5_stdDev,hq_valid70_dt3_s2_3x3_B8_mean,hq_valid70_dt3_s2_3x3_B8_median,hq_valid70_dt3_s2_3x3_B8_stdDev,hq_valid70_dt3_s2_3x3_FAI_mean,hq_valid70_dt3_s2_3x3_FAI_median,hq_valid70_dt3_s2_3x3_FAI_stdDev,hq_valid70_dt3_s2_3x3_MNDWI_mean,hq_valid70_dt3_s2_3x3_MNDWI_median,hq_valid70_dt3_s2_3x3_MNDWI_stdDev,hq_valid70_dt3_s2_3x3_NDCI_mean,hq_valid70_dt3_s2_3x3_NDCI_median,hq_valid70_dt3_s2_3x3_NDCI_stdDev,hq_valid70_dt3_s2_3x3_NDWI_mean,hq_valid70_dt3_s2_3x3_NDWI_median,hq_valid70_dt3_s2_3x3_NDWI_stdDev,hq_valid70_dt3_s2_3x3_missing_ratio,hq_valid70_dt3_s2_3x3_red_green_ratio_mean,hq_valid70_dt3_s2_3x3_red_green_ratio_median,hq_valid70_dt3_s2_3x3_red_green_ratio_stdDev,hq_valid70_dt3_s2_3x3_valid_ratio,hq_valid70_dt3_s2_5x5_B3_mean,hq_valid70_dt3_s2_5x5_B3_median,hq_valid70_dt3_s2_5x5_B3_stdDev,hq_valid70_dt3_s2_5x5_B4_mean,hq_valid70_dt3_s2_5x5_B4_median,hq_valid70_dt3_s2_5x5_B4_stdDev,hq_valid70_dt3_s2_5x5_B5_mean,hq_valid70_dt3_s2_5x5_B5_median,hq_valid70_dt3_s2_5x5_B5_stdDev,hq_valid70_dt3_s2_5x5_B8_mean,hq_valid70_dt3_s2_5x5_B8_median,hq_valid70_dt3_s2_5x5_B8_stdDev,hq_valid70_dt3_s2_5x5_FAI_mean,hq_valid70_dt3_s2_5x5_FAI_median,hq_valid70_dt3_s2_5x5_FAI_stdDev,hq_valid70_dt3_s2_5x5_MNDWI_mean,hq_valid70_dt3_s2_5x5_MNDWI_median,hq_valid70_dt3_s2_5x5_MNDWI_stdDev,hq_valid70_dt3_s2_5x5_NDCI_mean,hq_valid70_dt3_s2_5x5_NDCI_median,hq_valid70_dt3_s2_5x5_NDCI_stdDev,hq_valid70_dt3_s2_5x5_NDWI_mean,hq_valid70_dt3_s2_5x5_NDWI_median,hq_valid70_dt3_s2_5x5_NDWI_stdDev,hq_valid70_dt3_s2_5x5_missing_ratio,hq_valid70_dt3_s2_5x5_red_green_ratio_mean,hq_valid70_dt3_s2_5x5_red_green_ratio_median,hq_valid70_dt3_s2_5x5_red_green_ratio_stdDev,hq_valid70_dt3_s2_5x5_valid_ratio,hq_valid70_dt3_s2_date_diff_abs,hq_valid70_dt3_landsat_image_count_in_window,hq_valid70_dt3_landsat_3x3_LST_K_mean,hq_valid70_dt3_landsat_3x3_LST_K_median,hq_valid70_dt3_landsat_3x3_LST_K_stdDev,hq_valid70_dt3_landsat_3x3_MNDWI_mean,hq_valid70_dt3_landsat_3x3_MNDWI_median,hq_valid70_dt3_landsat_3x3_MNDWI_stdDev,hq_valid70_dt3_landsat_3x3_NDWI_mean,hq_valid70_dt3_landsat_3x3_NDWI_median,hq_valid70_dt3_landsat_3x3_NDWI_stdDev,hq_valid70_dt3_landsat_3x3_SR_B3_mean,hq_valid70_dt3_landsat_3x3_SR_B3_median,hq_valid70_dt3_landsat_3x3_SR_B3_stdDev,hq_valid70_dt3_landsat_3x3_SR_B4_mean,hq_valid70_dt3_landsat_3x3_SR_B4_median,hq_valid70_dt3_landsat_3x3_SR_B4_stdDev,hq_valid70_dt3_landsat_3x3_SR_B5_mean,hq_valid70_dt3_landsat_3x3_SR_B5_median,hq_valid70_dt3_landsat_3x3_SR_B5_stdDev,hq_valid70_dt3_landsat_3x3_SR_B6_mean...
- 时空分辨率：nearest valid image within current QC window / Sentinel-2/Landsat highres patch stats
- 缺失率：0.8419619566252211
- 提取成功率：1.0
- 限制：不是新增传感器数据，而是对现有高分 patch 做更严格 QC 后的稳健子集。
- recommendation：use
- reason：用于验证上一轮 patch 负迁移是否主要来自低质量 patch 噪声。

### nasa_earthdata_oceancolor
- 官方入口：https://oceancolor.gsfc.nasa.gov/
- 访问方式：free_account
- 数据类型：remote_sensing
- 变量：oc_chl,kd490,turbidity/ocean color proxies
- 时空分辨率：daily to multi-day / source-dependent
- 缺失率：nan
- 提取成功率：0.0
- 限制：需要免费 Earthdata 账号，当前本机未配置。
- recommendation：needs_user_api
- reason：缺少 NASA Earthdata 免费账号配置，先生成清单。

### noaa
- 官方入口：https://api.tidesandcurrents.noaa.gov/api/prod/
- 访问方式：public
- 数据类型：observation
- 变量：tide/current products
- 时空分辨率：various / not HK-targeted this round
- 缺失率：nan
- 提取成功率：0.0
- 限制：本轮没有发现明显适配香港研究区且能直接对齐主表的 NOAA 路线。
- recommendation：reject
- reason：当前优先级低且缺乏明确研究区匹配路径，不纳入本轮。

## 2026-05-20 NASA / bathymetry 跟进

- **nasa_earthdata_oceancolor**：NASA OceanColor 月尺度 9km 验证样本下载与站点提取成功，可进入后续全时段批量化实现。
- **gebco_bathymetry_followup**：当前 depth_bathymetry 表仍为空值为主，GEBCO 需要下一轮单独做稳定子集化。
- **hk_hydrographic_office_bathymetry**：本轮仅确认官方入口存在，未找到可直接批量并入模型的免费水深数值产品。
