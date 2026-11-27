# SHAP、ALE与来源消融综合分析

所有SHAP解释均处于`log1p(target)`尺度；ALE处于反变换后的原始目标单位。以下关系是模型统计关系，不是因果证据。

## Chl-a表层

SHAP前5变量为：lon（SPATIAL_STATIC）、lat（SPATIAL_STATIC）、t2m（METEOROLOGY）、month_cos（TEMPORAL_PROCESS）、tcc_roll3（METEOROLOGY）。
一维ALE候选均表现出较高折间相关，具体幅度与共同支持范围见 `02_ale_1d_fold_consistency.csv`。
二维交互中：t2m×ssrd_sum_3d为SUPPORTED；t2m×tcc_roll3为SUPPORTED。

## 透明度

SHAP前5变量为：lon（SPATIAL_STATIC）、lat（SPATIAL_STATIC）、day_of_year（TEMPORAL_PROCESS）、year_offset（TEMPORAL_PROCESS）、wind_dir_cos（METEOROLOGY）。
一维ALE候选均表现出较高折间相关，具体幅度与共同支持范围见 `02_ale_1d_fold_consistency.csv`。
二维交互中：t2m_delta1×days_since_prev_sample为SUPPORTED；v10×days_since_prev_sample为SUPPORTED。

## TN表层

SHAP前5变量为：lon（SPATIAL_STATIC）、lat（SPATIAL_STATIC）、year_offset（TEMPORAL_PROCESS）、v10（METEOROLOGY）、u10（METEOROLOGY）。
一维ALE候选均表现出较高折间相关，具体幅度与共同支持范围见 `02_ale_1d_fold_consistency.csv`。
二维交互中：v10×t2m_delta1为SUPPORTED；v10×u10为PARTIALLY_SUPPORTED。

## TP中层

SHAP前5变量为：year_offset（TEMPORAL_PROCESS）、lon（SPATIAL_STATIC）、lat（SPATIAL_STATIC）、t2m_roll3（METEOROLOGY）、day_of_year（TEMPORAL_PROCESS）。
一维ALE候选均表现出较高折间相关，具体幅度与共同支持范围见 `02_ale_1d_fold_consistency.csv`。
二维交互中：t2m_roll3×t2m_lag2为PARTIALLY_SUPPORTED；t2m_roll3×tcc_roll3为SUPPORTED。
