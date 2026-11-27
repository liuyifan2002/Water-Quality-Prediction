# TN_FAMILY_V2 决策报告

- 生成时间：`2026-09-21T18:01:55`
- 协议：`TN_FAMILY_V2_UNIFIED_STATION_SPLIT`
- TN family 总判定：**TN_FAMILY_SUPPORTED**
- 旧 TN bottom 方向重复验证：**REPLICATED**

## 三层结果

| 水层 | XGBoost R² | TabICLv2 R² | ΔR² | ΔRMSE | ΔMAE | RMSE胜率 | MAE胜率 | 分类 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| surface | 0.2593 | 0.4022 | +0.1429 | -0.0369 | -0.0280 | 0.750 | 0.798 | ROBUST |
| middle | 0.4783 | 0.4975 | +0.0192 | -0.0043 | -0.0091 | 0.738 | 0.750 | MODERATE |
| bottom | 0.5078 | 0.5401 | +0.0323 | -0.0072 | -0.0094 | 0.726 | 0.798 | MODERATE |

## 判定口径

- ROBUST：pooled ΔRMSE 与 ΔMAE 均小于 0，站点 RMSE/MAE 胜率均至少 0.60，且两项 bootstrap 95% 区间上限均小于 0。
- MODERATE：存在总体误差改善，站点两项误差胜率均不低于 0.50，但未同时满足 ROBUST 条件。
- MIXED：存在局部改善，但总体误差或站点胜率证据不一致。
- NO_GAIN：没有稳定的总体和站点级改善证据。

## 旧 TN bottom 对照

                       split_label  n_obs  n_station   XGB_R2  TabICL_R2  delta_R2  delta_RMSE  delta_MAE  RMSE_win_rate  MAE_win_rate replication_classification
               old_tn_bottom_pilot   9889         91 0.475900   0.552987  0.077087   -0.017399  -0.017804       0.692308      0.780220                 REPLICATED
TN_FAMILY_V2_UNIFIED_STATION_SPLIT   9307         84 0.507777   0.540122  0.032344   -0.007166  -0.009394       0.726190      0.797619                 REPLICATED

## 结论

TabICLv2 的空间泛化优势在 TN 三个水层中属于 `TN_FAMILY_SUPPORTED`。该判断基于统一的 84 个共同站点和同一 canonical station fold，不与旧 pilot 的绝对 R² 直接等同。

TN_FAMILY_V2 完成后停止新增模型实验，不自动进入 TP 或其他算法路线。
