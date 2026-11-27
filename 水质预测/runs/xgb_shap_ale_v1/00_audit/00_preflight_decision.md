# XGB_SHAP_ALE_V1 Step 0 预检决策

- 生成时间：2026-10-18T01:30:10
- 决策：`BLOCKED_MODEL_ARTIFACTS`
- 审计范围：Chl-a surface、Secchi、TN surface、TP middle，共 20 个 task-fold。
- strict feature manifest：`runs/tabiclv2_pilot/01_strict_feature_manifest.csv`，纳入特征 77 个，唯一特征 77 个。
- 来源分类：`runs/multisource_ablation_v1/00_feature_source_taxonomy.csv`，77 特征匹配 0 个。

## 结论

四个任务均没有定位到完整五折、可加载且与冻结预测对应的 XGBoost 模型文件。现有 prediction CSV、fold metrics、split manifest 和参数配置不能替代训练完成后的模型对象。根据实验禁令，不重新训练、不根据预测拟合替代模型，因此不执行 SHAP、1D ALE、2D ALE及正式论文图。

## 上游证据

- Family V2 训练入口在内存中拟合并输出预测，未调用 XGBoost `save_model`、pickle 或 joblib 持久化接口。
- Secchi 历史入口同样只返回预测；`runs/tabiclv2_pilot/09_secchi_predictions.csv` 是预测文件，不是模型文件。
- 77 特征清单与来源分类文件存在，但模型缺失导致无法核验模型内特征名称、输入顺序及加载后预测一致性。

## 解阻条件

需提供原始冻结运行产生的 20 个模型文件（四任务 × 五折），以及每折拟合后的 median imputer 参数或包含预处理器的完整 pipeline；随后才能执行小样本预测复现核验。重新训练得到的新模型不属于本任务允许的解阻方式。
