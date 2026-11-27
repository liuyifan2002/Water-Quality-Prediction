# TabICLv2 数值稳定性诊断 V2

最终结论：`NUMERICAL_JITTER_NONNEGLIGIBLE_FOR_SMALL_EFFECTS`。本结论是V2的经验噪声尺度判断，不是新的PASS标准。

## V1 冻结失败记录（未更改）

- V1结论：`INFERENCE_NONDETERMINISM_CONFIRMED`；fold验收：`FOLD1_NOT_COMPLETE_VALIDATION_FAILURE`。
- 原阈值仍为`1e-6`，未修改；原三次下界预测为`6.8550059382, 6.8565043045, 6.8565043045`，最大差`0.0014983664`。
- 本轮不以V2结果覆盖、重解释或升级V1。

## V2 探针设计与结果

固定5个P1 query，每个连续10次单行调用，共50次实测模型调用。每次`X_query.shape=(1,77)`、cache off、无batch；每组的input、feature-order、checkpoint、context、imputer和model-state哈希各自唯一值数均为1。未设置deterministic CUDA、未切CPU、未改dtype或模型函数。

| 任务/query | n | 均值 | SD | min | max | range | max pairwise |
|---|---:|---:|---:|---:|---:|---:|---:|
| chla_fold1_failed_ssrd_lower | 10 | 6.85500593816 | 0 | 6.85500593816 | 6.85500593816 | 0 | 0 |
| chla_surface_fold1_frozen_pred_low | 10 | 0.20463174673 | 0 | 0.20463174673 | 0.20463174673 | 0 | 0 |
| chla_surface_fold1_frozen_pred_high | 10 | 30.382485851 | 0 | 30.382485851 | 30.382485851 | 0 | 0 |
| tn_surface_fold1_frozen_pred_low | 10 | 0.0710966479257 | 0 | 0.0710966479257 | 0.0710966479257 | 0 | 0 |
| tn_surface_fold1_frozen_pred_high | 10 | 2.29668515837 | 0 | 2.29668515837 | 2.29668515837 | 0 | 0 |

本轮五组查询经高精度求均值后，range、样本SD、max absolute deviation和max pairwise delta均为0；十次输出逐位相同。

## 任务尺度

目标统计量基于正式五折OOF预测文件中的全部`y_true`，仅用于尺度比较；RMSE/MAE直接取正式结果表。

| 任务 | target | median | IQR | SD | 正式 pooled RMSE | 正式 pooled MAE |
|---|---|---:|---:|---:|---:|---:|
| chla_surface | chla_ug_l | 1.9 | 3.8 | 6.52659 | 4.18376 | 1.83586 |
| tn_surface | tn_mg_l | 0.49 | 0.41 | 0.421843 | 0.326148 | 0.142043 |

V2当前五组query的jitter相对上述IQR、SD、RMSE、MAE均为0。另把V1历史非零事件单独保留：该历史抖动/IQR=`0.0003943`，/SD=`0.0002296`，/RMSE=`0.0003581`，/MAE=`0.0008162`。

## 与已完成解释信号的尺度对照

只读现有fold1文件；不重跑SHAP/ALE。历史V1抖动相对解释尺度如下（统计范围来自两条SHAP样本和两条ALE曲线）：

| 方法/尺度 | effect scale范围 | 历史jitter/effect范围 |
|---|---:|---:|
| SHAP / median_abs_shap | 0.0219985–0.0301368 | 4.9719%–6.8112% |
| SHAP / p75_abs_shap | 0.0497879–0.0539715 | 2.7762%–3.0095% |
| SHAP / p90_abs_shap | 0.112967–0.11432 | 1.3107%–1.3264% |
| SHAP / top10_feature_abs_shap | 0.0921164–0.694828 | 0.2156%–1.6266% |
| ALE / ale_curve_range | 0.987567–1.26667 | 0.1183%–0.1517% |
| ALE / median_abs_local_effect | 0.0601996–0.0933719 | 1.6047%–2.4890% |
| ALE / p90_abs_local_effect | 0.259175–0.271245 | 0.5524%–0.5781% |
| ALE / median_abs_bin_to_bin_change | 0.0940129–0.0954438 | 1.5699%–1.5938% |
| ALE / p90_abs_bin_to_bin_change | 0.273015–0.285192 | 0.5254%–0.5488% |
| ALE / bin_to_bin_abs_change | 0.00437575–0.493322 | 0.3037%–34.2425% |

其中历史抖动约为最小ALE相邻bin变化的34.2%；相对median absolute ALE local effect为1.60%–2.49%，相对两样本median absolute SHAP为4.97%–6.81%；相对完整ALE曲线range为0.118%–0.152%。这表明小局部效应/弱bin变化可能受影响；没有证据表明主要ALE幅度或主导SHAP贡献排序已改变。

## 稳定性范围与限制

- SHAP没有合法重复版本：`SHAP_REPEAT_NOT_AVAILABLE`。两条已完成SHAP来自不同样本，不能当作重复归因比较。
- ALE没有重复完整曲线：`ALE_CURVE_REPEAT_NOT_AVAILABLE`；本轮只比较已存曲线尺度。
- 当前V2立即连续重复的五组探针为零range；V1的非零事件发生在不同调用历史/环境记录条件下。V1诊断探针显式设置过`CUBLAS_WORKSPACE_CONFIG=:4096:8`，V2启动器未设置，而原ALE运行进程环境变量没有保存。因此本轮不能据零range否定历史事件，也不能把该样本外推成所有进程状态下的噪声分布。
- 未启动剩余折次、未构建迁移包、未启用确定性CUDA、未修改推理函数或科学结果。

## 最终判断

`NUMERICAL_JITTER_NONNEGLIGIBLE_FOR_SMALL_EFFECTS`：当前测试query的即时重复预测稳定，但被冻结的历史抖动相对较小解释信号并非可忽略；其量级不足以据此声称主要解释结论被推翻，亦没有SHAP重复证据支持排序稳定性结论。V1仍保持失败。

数据文件：`probe_manifest.csv`、`p1_repeat_predictions.csv`、`numerical_jitter_summary.csv`、`jitter_vs_task_scale.csv`、`jitter_vs_explanation_scale.csv`。