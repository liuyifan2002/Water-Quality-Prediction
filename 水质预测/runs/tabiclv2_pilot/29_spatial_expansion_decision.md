# TabICLv2 空间泛化稳健性审计与扩展决策

## 最终建议：EXPAND SELECTIVELY

光学任务的空间证据较强，但营养盐任务存在站点异质性或极端站点驱动，先扩展光学指标族。

Chl-a 缺少逐样本预测，本轮不纳入站点胜率、bootstrap 和 pooled OOF 审计；已有 fold-level 结果只能作为补充证据。

## 判定口径

稳健性首先依据 pooled OOF、站点胜率和站点级 bootstrap 误差区间判断。为区分‘误差改善普遍存在’与‘少数站点贡献了大部分改善’，另计算 RMSE 和 MAE 改善量中贡献最大的两个站点所占比例：低于 0.25 标记为改善分布较广，0.25 至 0.40 标记为存在站点异质性，达到 0.40 标记为少数站点集中。该分层是本次描述性审计的透明判定规则，不是显著性检验阈值。R² 极端值单独保留为警告，因为低站点目标方差会放大 R² 变化。

## 四任务证据状态

```text
        task target_type                      status              classification                                              note  n_station  pooled_delta_R2  pooled_delta_RMSE  pooled_delta_MAE  n_station_win_rmse  win_rate_rmse  n_station_win_mae  win_rate_mae  n_station_valid_r2  n_station_win_r2  win_rate_r2  bootstrap_delta_RMSE_median  bootstrap_delta_RMSE_ci_lower  bootstrap_delta_RMSE_ci_upper  bootstrap_delta_MAE_median  bootstrap_delta_MAE_ci_lower  bootstrap_delta_MAE_ci_upper  top2_rmse_improvement_share  top2_mae_improvement_share  top2_error_improvement_share r2_extreme_station_driver error_extreme_station_driver extreme_station_driver
chla_surface     optical NO_SAMPLE_LEVEL_PREDICTIONS NO_SAMPLE_LEVEL_PREDICTIONS                              未保存逐样本留站点预测，本轮不重训补齐。        NaN              NaN                NaN               NaN                 NaN            NaN                NaN           NaN                 NaN               NaN          NaN                          NaN                            NaN                            NaN                         NaN                           NaN                           NaN                          NaN                         NaN                           NaN                       NaN                          NaN                    NaN
      secchi     optical                          OK         ROBUST SPATIAL GAIN R2 极端站点与 RMSE/MAE 改善集中度分开判断；站点级结果为描述性比较，未进行显著性检验。       94.0         0.066808          -0.056458         -0.047016                76.0       0.808511               77.0      0.819149                94.0              76.0     0.808511                    -0.056562                      -0.084057                      -0.031886                   -0.047063                     -0.063608                     -0.031226                     0.093557                    0.095620                      0.095620                       YES                           NO                     NO
   tn_bottom    nutrient                          OK       MODERATE SPATIAL GAIN R2 极端站点与 RMSE/MAE 改善集中度分开判断；站点级结果为描述性比较，未进行显著性检验。       91.0         0.077087          -0.017399         -0.017804                63.0       0.692308               71.0      0.780220                90.0              62.0     0.688889                    -0.016817                      -0.033595                      -0.005756                   -0.017345                     -0.030034                     -0.008909                     0.289970                    0.303708                      0.303708                       YES                           NO                     NO
   tp_middle    nutrient                          OK                       MIXED R2 极端站点与 RMSE/MAE 改善集中度分开判断；站点级结果为描述性比较，未进行显著性检验。       84.0         0.216103          -0.005982         -0.004419                59.0       0.702381               70.0      0.833333                79.0              56.0     0.708861                    -0.005770                      -0.014432                      -0.000736                   -0.004283                     -0.008599                     -0.001520                     0.450981                    0.478803                      0.478803                       YES                          YES                    YES
```

## 三个假设

- H1（未见站点上 TabICLv2 具有更稳定优势）：**PARTIALLY SUPPORTED**。Secchi、TN bottom、TP middle 的 pooled OOF 与站点误差需要结合各自分类判断；TP 若出现极端站点驱动，不能作为普遍证据。Chl-a 缺少逐样本审计。
- H2（优势主要来自空间泛化而非时间外推）：**SUPPORTED**。前三个任务和当前四任务 fold-level 记录均显示 station 方向通常优于 year，但这描述的是本数据集验证模式，不等同于因果结论。
- H3（空间优势具有任务依赖性，光学强于营养盐）：**PARTIALLY SUPPORTED**。光学任务的 fold-level 证据较一致，营养盐任务的站点差异更大；是否成立以本报告的 station-level bootstrap 和胜率为准。

## 解释边界

本审计只回答预测误差在未见站点上的分布情况，不解释为什么某个站点改善，也不把目标分布变化直接视为模型失败原因。后续正式任务应优先使用 station holdout，并同时报告 pooled OOF 与站点级胜率。

本报告不改变正式主链，也没有重新训练任何模型。