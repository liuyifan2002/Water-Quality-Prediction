# Chl-a 短期生态过程变量增强报告

## 一、新增变量

本轮加入采样日前 7 天和 14 天窗口的短期生态过程变量：气温均值/最高值/距平、平均风速/最大风速、低风速小时数和天数、连续低风速天数、云量、太阳辐射代理、晴天代理天数、累计降水、最大日降水和强降水日数。同时新增 Landsat 严格质量控制标记，包括 strict_valid、moderate_valid、low_quality 和 valid_count_log1p。

注意：当前 ERA5 本地文件没有真实太阳辐射和海表水温变量，因此 solar_proxy 使用 1 - total_cloud_cover，水温窗口暂用 2m 气温作为代理。后续需要补下载 ssrd 和 sea_surface_temperature 后再替换代理变量。

## 二、总体结果

| 版本 | RMSE | MAE | R2 / NSE |
|---|---:|---:|---:|
| 原站点独立最优 | 4.9210 | 2.8619 | 0.073 |
| 短期生态增强 | 4.6920 | 2.9569 | 0.158 |

## 三、各站点结果

| 站点 | 模型 | 特征组 | 变换 | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|---:|---:|
| DM1 | ridge_robust | history_only | log1p | 3.843 | 3.026 | -0.161 |
| MM8 | extra_trees | eco_fusion | raw | 3.563 | 2.680 | 0.299 |
| NM3 | xgb_regularized | history_only | raw | 5.905 | 3.515 | -0.007 |
| PM7 | ridge_robust | history_only | log1p | 1.386 | 1.027 | -0.035 |
| SM6 | extra_trees | eco_fusion | log1p | 5.432 | 3.672 | -0.160 |
| TM4 | xgb_shallow | eco_process | log1p | 2.785 | 2.200 | 0.213 |
| VM5 | extra_trees | eco_fusion_AR | raw | 4.843 | 3.042 | 0.219 |
| WM2 | extra_trees | compact_eco_AR | raw | 7.152 | 4.494 | 0.179 |

## 四、解释

如果增强模型优于上一版，说明 Chl-a 的短期变化确实包含可被 7-14 天窗口捕捉的过程信号，例如低风速导致的水体稳定、降水后的营养盐输入、以及较高光照条件下的藻类增长潜势。若部分站点退化，则说明这些站点仍受局地水动力、遥感质量或突发事件控制，不能用统一窗口解释。
