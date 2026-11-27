# distance_to_mouth_only 统一 A/B 报告

本轮按指令仅做诊断性 A/B，不 promote，不修改 current best。

## 特征口径
- 额外特征：distance_to_mouth_system_km, log_distance_to_mouth_system_km, nearest_mouth_system_id, mouth_system_distance_bucket, system_mapping_ambiguity_flag
- observation 级缺失率：0.0000
- extraction success rate：1.0000

## 正信号任务
target   layer  delta_R2  delta_RMSE  delta_MAE
  chla  bottom  0.003157   -0.008206  -0.028331
secchi  bottom  0.007095   -0.004869  -0.003472
secchi surface  0.003788   -0.002712  -0.000849
    tn  bottom  0.007135   -0.001290  -0.001059
    tn surface  0.002269   -0.000807  -0.000525

## 负迁移任务
target   layer  delta_R2  delta_RMSE  delta_MAE
  chla  middle -0.000921    0.002812   0.004966
  chla surface -0.006977    0.026097   0.001316
secchi  middle -0.019542    0.013654   0.008544
    tn  middle -0.007373    0.001314  -0.000215
    tp  bottom -0.025406    0.000674   0.000662
    tp  middle -0.024209    0.000649   0.000061
    tp surface -0.003825    0.000169   0.000179

## 判断
- 该结果仅用于判断后续是否继续组合实验，不作为 current best 更新依据。
