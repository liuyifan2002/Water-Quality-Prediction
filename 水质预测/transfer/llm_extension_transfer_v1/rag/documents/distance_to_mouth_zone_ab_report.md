# distance_to_mouth_plus_zone 统一 A/B 报告

本轮按指令仅做诊断性 A/B，不 promote，不修改 current best。

## 特征口径
- 额外特征：distance_to_mouth_system_km, log_distance_to_mouth_system_km, nearest_mouth_system_id, mouth_system_distance_bucket, system_mapping_ambiguity_flag, scheme4_zone_id
- observation 级缺失率：0.0000
- extraction success rate：1.0000

## 正信号任务
target   layer  delta_R2  delta_RMSE  delta_MAE
  chla surface  0.008299   -0.031209  -0.017563
secchi  bottom  0.000333   -0.000228   0.001879
secchi surface  0.002794   -0.002000   0.000855
    tn  bottom  0.006242   -0.001128  -0.000804

## 负迁移任务
target   layer  delta_R2  delta_RMSE  delta_MAE
  chla  bottom -0.001438    0.003731  -0.014550
  chla  middle -0.000134    0.000410   0.006204
secchi  middle -0.009255    0.006491   0.003530
    tn  middle -0.000840    0.000150  -0.001423
    tn surface -0.016007    0.005643   0.003349
    tp  bottom -0.056966    0.001500   0.001353
    tp  middle -0.007024    0.000189  -0.000094
    tp surface -0.008787    0.000386   0.000455

## 判断
- 该结果仅用于判断后续是否继续组合实验，不作为 current best 更新依据。
