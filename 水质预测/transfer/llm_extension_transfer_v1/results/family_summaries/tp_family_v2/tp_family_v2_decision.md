# TP_FAMILY_V2 决策报告

- 生成时间：`2026-09-21T20:06:59`
- 协议：`TP_FAMILY_V2_UNIFIED_STATION_SPLIT`
- TP family 总判定：**TP_FAMILY_SUPPORTED**
- 旧 TP middle pilot 方向重复验证：**REPLICATED**

## 三层结果

| 水层 | XGBoost R² | TabICLv2 R² | ΔR² | ΔRMSE | ΔMAE | RMSE胜率 | MAE胜率 | 分类 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tp_surface | -0.1266 | 0.2474 | +0.3741 | -0.0110 | -0.0057 | 0.750 | 0.845 | MODERATE |
| tp_middle | 0.3042 | 0.4716 | +0.1674 | -0.0047 | -0.0029 | 0.750 | 0.798 | MODERATE |
| tp_bottom | 0.3830 | 0.5070 | +0.1239 | -0.0034 | -0.0024 | 0.702 | 0.821 | MODERATE |

## TP middle 旧结果对照

                       split_label  n_obs  n_station   XGB_R2  TabICL_R2  delta_R2  delta_RMSE  delta_MAE  RMSE_win_rate  MAE_win_rate replication_classification
               old_tp_middle_pilot   9000         84 0.251874   0.467977  0.216103   -0.005982  -0.004419       0.702381      0.833333                 REPLICATED
TP_FAMILY_V2_UNIFIED_STATION_SPLIT   9000         84 0.304207   0.471625  0.167419   -0.004733  -0.002873       0.750000      0.797619                 REPLICATED

TP_FAMILY_V2 完成后停止新增实验，不自动进入其他指标、其他模型或区域适配路线。
