# RANDOM_TABULAR_MODEL_ZOO_V2 最终判定

- 内部 OOF 锁定的任务级单模型/非负融合在固定 Random 测试集上达到 R²≥0.5：10/10。
- 表中最优配置由训练集内部验证选择，外部测试结果只用于最终报告。
- Random 单次样本划分与 Station V2 五折留站 pooled OOF 不可作为同协议提升解释。

| 任务 | 锁定模型 | 测试R² | RMSE | MAE |
|---|---|---:|---:|---:|
| chla_surface | tabicl;lightgbm_tuned | 0.6280 | 4.0200 | 1.7955 |
| chla_middle | lightgbm_tuned | 0.5027 | 3.2672 | 1.4699 |
| chla_bottom | lightgbm_tuned;tabicl;extra_trees_tuned | 0.5591 | 2.6500 | 1.2441 |
| secchi | tabicl;hist_gb_tuned | 0.7095 | 0.6062 | 0.4043 |
| tn_surface | tabicl;hist_gb_tuned;lightgbm_tuned | 0.8230 | 0.2462 | 0.1262 |
| tn_middle | hist_gb_tuned;lightgbm_tuned;random_forest | 0.6275 | 0.2183 | 0.1216 |
| tn_bottom | hist_gb_tuned;lightgbm_tuned;tabpfn | 0.6427 | 0.2021 | 0.1208 |
| tp_surface | xgboost_tuned | 0.7559 | 0.0321 | 0.0145 |
| tp_middle | lightgbm_tuned;hist_gb_tuned;catboost | 0.5970 | 0.0286 | 0.0128 |
| tp_bottom | lightgbm_tuned | 0.5636 | 0.0294 | 0.0132 |

## 模型家族与融合判读

- 传统、深度和预训练模型的逐任务内部排名见 `02_model_registry.csv`，逐模型外部表现见 `08_all_models_test_comparison.csv`；不能用外部测试排名事后替换锁定方案。
- 内部 OOF 的调参前后差值及搜索耗时见 `05_tuned_model_metrics.csv`；融合权重与二级交叉验证见 `06_ensemble_weights.csv`。
- 最终测试 R² 与内部最佳单模型测试 R² 的差异只描述已经锁定方案的结果，不是选择融合的依据。
- TabDPT 的官方权重和完整样本单折试跑通过，但十任务五折规模运行本轮未完成；TabFM、Mitra-v2 未完成本机独立整规模运行，均未填入虚构分数。
- 本机实际 GPU 为 RTX 2070 Max-Q 8GB。内部和外部运行耗时、GPU 峰值内存记录分别见 `01_model_runtime_status.csv` 与逐模型预测 JSON；原脚本未拆分计时的模型仅报告总耗时，训练/推理分项为 NA。
- Random 单次样本划分结果不能作为五折留站空间泛化结果。

| 任务 | 内部最佳单模型 | 单模型测试R² | 锁定方案测试R² | 差值 |
|---|---|---:|---:|---:|
| chla_surface | tabicl | 0.6410 | 0.6280 | -0.0130 |
| chla_middle | lightgbm_tuned | 0.5027 | 0.5027 | +0.0000 |
| chla_bottom | lightgbm_tuned | 0.5510 | 0.5591 | +0.0081 |
| secchi | tabicl | 0.7008 | 0.7095 | +0.0087 |
| tn_surface | tabicl | 0.8111 | 0.8230 | +0.0119 |
| tn_middle | hist_gb_tuned | 0.6334 | 0.6275 | -0.0059 |
| tn_bottom | hist_gb_tuned | 0.6527 | 0.6427 | -0.0100 |
| tp_surface | xgboost_tuned | 0.7559 | 0.7559 | +0.0000 |
| tp_middle | lightgbm_tuned | 0.6001 | 0.5970 | -0.0031 |
| tp_bottom | lightgbm_tuned | 0.5636 | 0.5636 | -0.0000 |
