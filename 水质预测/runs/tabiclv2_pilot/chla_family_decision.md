# Chl-a 三层空间泛化 Gate

## Phase A 判定：CHLA_FAMILY_ROBUST

三层中至少两层具有 pooled 改善、站点误差改善和非少数站点集中的证据。

Chl-a surface 复现检查：XGBoost_R2=ACCEPTABLE, TabICLv2_R2=ACCEPTABLE。仅使用 station holdout；本阶段未运行 random/year，也未运行 TN。

## 三层结果

  depth  pooled_XGB_R2  pooled_TabICL_R2  delta_R2  delta_RMSE  delta_MAE  station_win_rate_RMSE  station_win_rate_MAE  station_valid_R2  station_win_rate_R2  bootstrap_RMSE_low  bootstrap_RMSE_high  bootstrap_MAE_low  bootstrap_MAE_high  top2_error_improvement_share classification
surface       0.423514          0.577947  0.154434   -0.739281  -0.427996               0.861702              0.861702                94             0.861702           -0.904570            -0.571614          -0.536872           -0.327166                      0.097302         ROBUST
 middle       0.363378          0.444854  0.081476   -0.253768  -0.178301               0.690476              0.785714                79             0.696203           -0.378813            -0.137249          -0.253547           -0.108798                      0.125578         ROBUST
 bottom       0.332893          0.470704  0.137811   -0.340948  -0.194992               0.824176              0.857143                90             0.833333           -0.514276            -0.181320          -0.266833           -0.130494                      0.164297         ROBUST

## 判定口径

分类同时参考 pooled OOF、站点 RMSE/MAE 胜率、1000 次站点级 bootstrap 和前两个站点对误差改善量的贡献。前两个站点贡献低于 0.25 记为 ROBUST，0.25 至 0.40 记为 MODERATE，达到 0.40 记为 MIXED；这只是本轮描述性 Gate 规则，不是显著性检验。

## 下一步

Phase A 结果完成后再决定是否运行 TN surface 和 TN middle；本脚本不会自动启动 TN。

本报告属于扩展实验，不修改正式主链。