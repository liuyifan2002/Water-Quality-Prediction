# CHLA_FAMILY_V2 决策报告

- 生成时间：2026-09-22T14:22:55
- 协议：`CHLA_FAMILY_V2_UNIFIED_STATION_SPLIT`
- family 判定：**CHLA_FAMILY_SUPPORTED**

| 水层 | XGB R² | TabICL R² | ΔR² | ΔRMSE | ΔMAE | RMSE胜率 | MAE胜率 | 分类 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| surface | 0.4575 | 0.5890 | +0.1315 | -0.6232 | -0.4262 | 0.881 | 0.929 | ROBUST |
| middle | 0.3826 | 0.4885 | +0.1059 | -0.3392 | -0.2536 | 0.798 | 0.917 | ROBUST |
| bottom | 0.3535 | 0.4898 | +0.1363 | -0.3275 | -0.1817 | 0.833 | 0.905 | ROBUST |

## 历史结果方向复现

  depth  new_delta_R2  new_delta_RMSE  new_delta_MAE  new_RMSE_win_rate  new_MAE_win_rate  old_delta_R2  old_delta_RMSE  old_delta_MAE  old_RMSE_win_rate  old_MAE_win_rate                                                                    old_source replication_classification
surface      0.131548       -0.623185      -0.426204           0.880952          0.928571      0.154434       -0.739281      -0.427996           0.861702          0.861702 E:\研究生\codex\水质预测\runs\tabiclv2_pilot\33_chla_surface_station_predictions.csv                 REPLICATED
 middle      0.105939       -0.339190      -0.253649           0.797619          0.916667      0.081476       -0.253768      -0.178301           0.690476          0.785714  E:\研究生\codex\水质预测\runs\tabiclv2_pilot\34_chla_middle_station_predictions.csv                 REPLICATED
 bottom      0.136319       -0.327484      -0.181749           0.833333          0.904762      0.137811       -0.340948      -0.194992           0.824176          0.857143  E:\研究生\codex\水质预测\runs\tabiclv2_pilot\35_chla_bottom_station_predictions.csv                 REPLICATED

本实验完成后停止新增模型和验证方式，不自动进入其他任务。
