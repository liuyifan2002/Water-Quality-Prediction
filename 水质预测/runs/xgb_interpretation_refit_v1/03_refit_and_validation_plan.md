# XGBoost 解释模型受控复现与验收方案

## 1. 当前结论

四个任务均具备按现存训练代码和配置开展受控复现的条件。20/20 折的训练/测试 obs_id、77 特征及顺序、XGBoost 参数、随机种子、预处理规则、目标变换、冻结预测和指标均已定位；本轮未运行任何模型。

新生成的文件必须称为“受控复现模型”，不能称为“找回的原始冻结模型”。正式预测 CSV 和正式评价指标保持只读。

## 2. 固定复现协议

每个 `task → fold` 独立执行：

1. 从现有 manifest 读取原始 train/test obs_id，不重新划分、不排序重构 split。
2. 按 `runs/tabiclv2_pilot/01_strict_feature_manifest.csv` 中 `included_in_pilot=True` 的原始行顺序读取77个特征。
3. 从正式主表按目标非缺失及指定水层筛选，再严格以保存的 obs_id 提取训练和测试行。
4. 只在该折训练集拟合 `SimpleImputer(strategy="median", keep_empty_features=True)`；变换后转换为 `float32`。
5. 因四任务训练目标均非负，使用训练代码规定的 `log1p`，预测后 `expm1` 并执行 `maximum(prediction, 0)`。
6. 使用冻结参数：`n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42, objective="reg:squarederror"`。
7. 不设置样本权重、`eval_set`、早停，不抽样、不调参、不训练其他模型。
8. 每折完成后立即保存模型、imputer、特征清单、元数据、复现预测和验收结果；成功折不得因后续失败而重复运行。

## 3. 独立输出结构

```text
runs/xgb_interpretation_refit_v1/
  artifacts/<task>/<fold>/xgboost_model.ubj
  artifacts/<task>/<fold>/median_imputer.joblib
  artifacts/<task>/<fold>/feature_order.json
  artifacts/<task>/<fold>/metadata.json
  predictions/<task>_<fold>_refit_predictions.csv
  validation/<task>_<fold>_reproduction_check.csv
  registry/refit_run_registry.csv
```

`metadata.json` 至少记录输入文件 SHA-256、脚本 SHA-256、环境版本、有效参数、训练/测试 obs_id 哈希、特征顺序哈希、目标变换和运行时间。

## 4. 事先固定的数值容差

以下阈值在查看复现差异前固定，不得事后放宽：

| 等级 | obs_id要求 | 最大绝对预测误差 | 平均绝对预测误差 | R²/RMSE/MAE最大绝对差 |
|---|---|---:|---:|---:|
| `EXACT_REPRODUCTION` | 数量、集合、顺序完全一致 | ≤ 1e-6 | ≤ 1e-7 | ≤ 1e-10 |
| `APPROXIMATE_REPRODUCTION` | 数量、集合、顺序完全一致 | ≤ 1e-4 | ≤ 1e-5 | ≤ 1e-6 |
| `REPRODUCTION_FAILED` | 任一 obs_id 条件失败，或超过近似阈值 | — | — | — |
| `INSUFFICIENT_CONFIGURATION` | 在训练前发现关键数据、参数或规则不可恢复 | — | — | — |

指标必须由逐样本复现预测重新计算，不能只比较 registry 中四舍五入后的显示值。比较前按 obs_id 一对一连接，同时单独验证原始行顺序。

## 5. 拟运行脚本

人工批准后再新增并运行：

1. `脚本/05_诊断分析/56_refit_frozen_xgb_for_interpretation.py`：逐任务逐折受控拟合并保存模型、imputer和元数据。
2. `脚本/05_诊断分析/57_validate_xgb_interpretation_refit.py`：只读新旧预测，执行 obs_id、逐样本误差、指标差异和资产完整性验收。

当前的 `55_audit_xgb_interpretation_refit_v1.py` 只负责配置和输入审计，不包含 `.fit()` 调用。

## 6. 与后续 SHAP/ALE 的衔接门槛

- 只有通过 `EXACT_REPRODUCTION` 的折可直接作为正式预测模型的数值等价复现，用于主解释结果。
- `APPROXIMATE_REPRODUCTION` 的折应先评估四舍五入误差、环境差异和 SHAP/ALE排序稳定性；论文中必须称为“近似复现解释模型”，不得暗示为原始模型内部解释。
- 任一任务存在 `REPRODUCTION_FAILED` 折时，该任务不得汇总为完整五折正式解释结果。
- 无论达到何种等级，SHAP/ALE仍只解释模型统计关系，不证明环境因果机制。
