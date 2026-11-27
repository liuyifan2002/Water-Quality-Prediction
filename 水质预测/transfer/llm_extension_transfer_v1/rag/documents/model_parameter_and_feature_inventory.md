# 模型参数与输入特征清单

## 本轮 strong mechanism candidate 模型
- 模型框架沿用 current best family（station_calibration / ensemble_lgbm_rf_avg / impute_knn_lightgbm / stacking_ridge）。
- 未新增 Optuna，未切换模型家族。
- 新增特征组：`globathy_*`、`cmems_*`、`surface_temp`、`near_bottom_temp_proxy`、`tide_*`、`hq_*`。
- 这些特征的具体含义见 `strong_mechanism_feature_dictionary.csv`。

## 2026-05-20 新增候选模型与特征补充

- `copernicus_bgc` 单源 A/B：沿用 current best family，不做大规模调参；新增字段主要为 `bgc_chl/bgc_no3/bgc_po4/bgc_si/bgc_o2/bgc_npp/bgc_*_7d/bgc_chl_anom_7d/bgc_n_to_p_ratio`。
- `enhanced_tide_proxy` 单源 A/B：沿用 current best family；新增字段为 `tide_range_same_day/tide_range_previous_3d_mean/tide_range_anom_prev3d/tide_range_change_prev1d/tide_level_daily_mean/tide_event_density` 等。
- `high_quality_patch_qc_refined` 单源 A/B：沿用 current best family；只保留高分 patch 中与水体指数、valid_ratio、date_diff_abs 相关的更稳健子集，不再把全部 patch 统计重新堆回模型。
- `chla_bottom_log1p` 与 `chla_bottom_yeojohnson`：保持 bottom 当前 LightGBM-family 框架，仅做目标变换验证。
