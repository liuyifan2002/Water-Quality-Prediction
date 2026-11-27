# MODERN_BASELINE_GATE_V1 决策报告

- 完成时间：2026-09-22T22:52:07
- 任务：Chl-a surface、Secchi、TN surface、TP middle
- 新增模型：TabM、TabPFN
- 所有新增模型均使用相同 station holdout、77 strict features 和 test obs_id。

## 核心指标

        task    model  pooled_R2     RMSE      MAE  fold_R2_mean  fold_R2_sd                                  source
chla_surface  xgboost   0.457487 4.806943 2.262060      0.453587    0.053165               FROZEN_FAMILY_PREDICTIONS
chla_surface tabiclv2   0.589035 4.183758 1.835856      0.586955    0.052863               FROZEN_FAMILY_PREDICTIONS
chla_surface     tabm   0.574797 4.255613 1.889244      0.570620    0.067677             NEW_MODERN_BASELINE_GATE_V1
chla_surface   tabpfn   0.556957 4.343970 1.916206      0.553973    0.058301             NEW_MODERN_BASELINE_GATE_V1
      secchi  xgboost   0.535241 0.756189 0.495572      0.515322    0.087172           FROZEN_SECHHI_XGB_PREDICTIONS
      secchi tabiclv2   0.602049 0.699731 0.448556      0.584848    0.094002 RECOVERED_HISTORICAL_TABICL_PREDICTIONS
      secchi     tabm   0.581826 0.717290 0.440704      0.556866    0.112545             NEW_MODERN_BASELINE_GATE_V1
      secchi   tabpfn   0.575004 0.723117 0.470225      0.555624    0.079548             NEW_MODERN_BASELINE_GATE_V1
  tn_surface  xgboost   0.259296 0.363036 0.170082      0.375641    0.290478               FROZEN_FAMILY_PREDICTIONS
  tn_surface tabiclv2   0.402173 0.326148 0.142043      0.556561    0.246662               FROZEN_FAMILY_PREDICTIONS
  tn_surface     tabm   0.272619 0.359756 0.145259      0.369048    0.530276             NEW_MODERN_BASELINE_GATE_V1
  tn_surface   tabpfn   0.417464 0.321950 0.138226      0.584266    0.258363             NEW_MODERN_BASELINE_GATE_V1
   tp_middle  xgboost   0.304207 0.036811 0.016066      0.343188    0.309416               FROZEN_FAMILY_PREDICTIONS
   tp_middle tabiclv2   0.471625 0.032078 0.013193      0.516548    0.139384               FROZEN_FAMILY_PREDICTIONS
   tp_middle     tabm   0.317599 0.036455 0.015193      0.321256    0.415442             NEW_MODERN_BASELINE_GATE_V1
   tp_middle   tabpfn   0.468001 0.032188 0.013271      0.528134    0.143038             NEW_MODERN_BASELINE_GATE_V1

## 站点级胜率

        task  model_a  model_b  n_station  b_wins_RMSE  win_rate_RMSE_b_vs_a  b_wins_MAE  win_rate_MAE_b_vs_a
chla_surface  xgboost tabiclv2         84           74              0.880952          78             0.928571
chla_surface  xgboost     tabm         84           67              0.797619          73             0.869048
chla_surface  xgboost   tabpfn         84           69              0.821429          77             0.916667
chla_surface tabiclv2     tabm         84           35              0.416667          43             0.511905
chla_surface tabiclv2   tabpfn         84           25              0.297619          25             0.297619
chla_surface     tabm   tabpfn         84           34              0.404762          31             0.369048
      secchi  xgboost tabiclv2         94           76              0.808511          77             0.819149
      secchi  xgboost     tabm         94           76              0.808511          79             0.840426
      secchi  xgboost   tabpfn         94           66              0.702128          74             0.787234
      secchi tabiclv2     tabm         94           54              0.574468          55             0.585106
      secchi tabiclv2   tabpfn         94           26              0.276596          28             0.297872
      secchi     tabm   tabpfn         94           27              0.287234          27             0.287234
  tn_surface  xgboost tabiclv2         84           63              0.750000          67             0.797619
  tn_surface  xgboost     tabm         84           63              0.750000          70             0.833333
  tn_surface  xgboost   tabpfn         84           68              0.809524          70             0.833333
  tn_surface tabiclv2     tabm         84           42              0.500000          44             0.523810
  tn_surface tabiclv2   tabpfn         84           44              0.523810          49             0.583333
  tn_surface     tabm   tabpfn         84           44              0.523810          45             0.535714
   tp_middle  xgboost tabiclv2         84           63              0.750000          67             0.797619
   tp_middle  xgboost     tabm         84           40              0.476190          40             0.476190
   tp_middle  xgboost   tabpfn         84           58              0.690476          66             0.785714
   tp_middle tabiclv2     tabm         84           22              0.261905          19             0.226190
   tp_middle tabiclv2   tabpfn         84           30              0.357143          29             0.345238
   tp_middle     tabm   tabpfn         84           55              0.654762          69             0.821429

## 站点配对 bootstrap

        task  model_a  model_b status  n_bootstrap  delta_RMSE  RMSE_CI_low  RMSE_CI_high  delta_MAE  MAE_CI_low  MAE_CI_high
chla_surface  xgboost tabiclv2     OK         1000   -0.621440    -0.777194     -0.479781  -0.424973   -0.529090    -0.335683
chla_surface  xgboost     tabm     OK         1000   -0.562161    -0.771728     -0.305104  -0.382029   -0.514680    -0.205716
chla_surface  xgboost   tabpfn     OK         1000   -0.464123    -0.635796     -0.310427  -0.345499   -0.438655    -0.262672
chla_surface tabiclv2     tabm     OK         1000    0.064888    -0.062900      0.278848   0.049008   -0.033323     0.194373
chla_surface tabiclv2   tabpfn     OK         1000    0.162593     0.076080      0.247684   0.080305    0.039124     0.119751
chla_surface     tabm   tabpfn     OK         1000    0.092229    -0.103774      0.257038   0.032161   -0.110059     0.119527
      secchi  xgboost tabiclv2     OK         1000   -0.057605    -0.082296     -0.032563  -0.047262   -0.063795    -0.031766
      secchi  xgboost     tabm     OK         1000   -0.045270    -0.102565      0.052761  -0.056106   -0.076450    -0.028490
      secchi  xgboost   tabpfn     OK         1000   -0.032774    -0.056244     -0.011952  -0.025648   -0.041548    -0.008678
      secchi tabiclv2     tabm     OK         1000    0.015156    -0.032231      0.099250  -0.007817   -0.022095     0.007276
      secchi tabiclv2   tabpfn     OK         1000    0.023865     0.007723      0.040295   0.021393    0.008576     0.036676
      secchi     tabm   tabpfn     OK         1000    0.008221    -0.088606      0.060545   0.030284    0.005068     0.050238
  tn_surface  xgboost tabiclv2     OK         1000   -0.035416    -0.115003     -0.004074  -0.026159   -0.058994    -0.008378
  tn_surface  xgboost     tabm     OK         1000   -0.007401    -0.106269      0.102024  -0.024152   -0.059466     0.004267
  tn_surface  xgboost   tabpfn     OK         1000   -0.039415    -0.125512     -0.009570  -0.029891   -0.063972    -0.012146
  tn_surface tabiclv2     tabm     OK         1000    0.032161    -0.015465      0.143654   0.002510   -0.010299     0.022439
  tn_surface tabiclv2   tabpfn     OK         1000   -0.004214    -0.018025      0.000751  -0.003543   -0.009766     0.000380
  tn_surface     tabm   tabpfn     OK         1000   -0.036445    -0.158484      0.005693  -0.006173   -0.027061     0.005122
   tp_middle  xgboost tabiclv2     OK         1000   -0.004682    -0.013369      0.000241  -0.002815   -0.006912    -0.000553
   tp_middle  xgboost     tabm     OK         1000   -0.000330    -0.010820      0.009577  -0.000699   -0.005128     0.002035
   tp_middle  xgboost   tabpfn     OK         1000   -0.004471    -0.013489     -0.000109  -0.002693   -0.006455    -0.000762
   tp_middle tabiclv2     tabm     OK         1000    0.004157     0.000411      0.011285   0.001914    0.001147     0.003106
   tp_middle tabiclv2   tabpfn     OK         1000    0.000249    -0.002232      0.001402   0.000107   -0.000834     0.000728
   tp_middle     tabm   tabpfn     OK         1000   -0.004205    -0.012570      0.000318  -0.001911   -0.003574    -0.000675

## Secchi 说明

Secchi 的历史冻结结果包含 TabICLv2 的 fold-level 指标，但没有可与本轮逐样本 TabM/TabPFN 完全配对的 TabICLv2 prediction 文件。因此 Secchi 中涉及 TabICLv2 的 paired bootstrap 标记为 UNAVAILABLE_SAMPLE_PREDICTIONS；不把非配对结果伪装成严格的站点级比较。

## 方法建议

**REPRESENTATIVE_BENCHMARK_SUFFICIENT**：本轮四项代表任务足以完成现代表格模型的代表性公平比较，但在补齐 Secchi 的 TabICLv2 逐样本预测前，不将该任务的站点 bootstrap 用于强结论。是否把 TabICLv2 作为统一主模型，应按任务依赖性报告，而不是简单统计冠军数量。

本 Gate 完成后停止，不自动启动全 10 任务扩展。
