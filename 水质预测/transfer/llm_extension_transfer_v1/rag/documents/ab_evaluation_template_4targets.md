# 四指标统一 A/B 评价模板

## 1. 适用范围

本模板用于后续所有：

- 新数据源验证
- 新特征组验证
- 新时间窗口验证
- 新交互项验证
- 小范围模型补充验证

统一要求：**必须同时面向 Chl-a、Secchi、TN、TP 四项指标**，而不是只围绕单个局部任务汇报结果。

## 2. 统一任务清单

后续 A/B 默认任务为：

1. Chl-a surface
2. Chl-a middle
3. Chl-a bottom
4. Secchi surface
5. Secchi middle
6. Secchi bottom
7. TN surface
8. TN middle
9. TN bottom
10. TP surface
11. TP middle
12. TP bottom

如果某一指标/水层本轮无法运行，必须显式填写：

- `not_run_reason`
- `missing_dependency`
- `whether_to_carry_forward`

不允许静默省略。

## 3. 输出表字段

后续统一 A/B 表至少包含以下字段：

- `experiment_id`
- `experiment_group`
- `source_type`
- `source_name`
- `target`
- `depth_layer`
- `task_name`
- `baseline_model`
- `candidate_model`
- `baseline_feature_set`
- `candidate_feature_set`
- `baseline_R2`
- `candidate_R2`
- `delta_R2`
- `baseline_RMSE`
- `candidate_RMSE`
- `delta_RMSE`
- `baseline_MAE`
- `candidate_MAE`
- `delta_MAE`
- `n_train`
- `n_valid`
- `n_test`
- `missing_rate`
- `coverage_rate`
- `leakage_check_status`
- `deployability_check`
- `promote_to_current_best`
- `reason`
- `not_run_reason`

## 4. 评价顺序

每个实验的正式结论按照以下顺序判断：

1. **四指标整体是否存在普遍正增益**
2. **哪些指标受益最大**
3. **哪些水层受益最大**
4. **是否存在负迁移**
5. **是否值得 promote**

## 5. promote 规则

只有满足以下条件，才允许 `promote_to_current_best = true`：

1. `delta_R2 > 0`
2. `RMSE` 或 `MAE` 至少一个同步改善
3. 无数据泄漏
4. 输入可部署
5. 结果可复现
6. 不是通过删样本制造出来的提升

## 6. 当前推荐的分层口径

### 6.1 通用强输入数据

优先看是否对四指标都有增益：

- bathymetry / depth
- tide / water level / exchange proxy
- rainfall / wind / sea temperature
- discharge / runoff / plume proxy

### 6.2 光学增强数据

主要看：

- Chl-a
- Secchi

并额外记录：

- 对 TN/TP 是否无效
- 是否只是局部改善

### 6.3 营养盐增强数据

重点看：

- TN
- TP

并记录是否对 Chl-a / Secchi 有联动增益。

## 7. 后续报告要求

以后所有阶段报告，必须至少回答：

1. 本轮新增输入是否对四指标整体有帮助；
2. 是否只对光学指标有效；
3. 是否对 TN / TP 有独立增益；
4. 是否有任何任务接近或达到 `R² = 0.60`；
5. 若没有，真正缺的是哪类数据。

