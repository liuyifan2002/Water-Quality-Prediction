# 最终实验闭环决策

## 状态

`CLOSED_WITH_LIMITATIONS`

## 审计结果

- 十项正式预测任务：逐任务、逐fold核验split、训练/测试obs_id和监测站隔离、77特征、模型间测试样本配对，并从冻结预测重算pooled R²/RMSE/MAE；结果见`06_prediction_experiment_integrity.csv`。
- Family V2：90个模型-fold登记、9项任务五折站点隔离；登记与split审计见`family_v2_split_audit.csv`。
- TabM/TabPFN：4项代表任务×2模型×5折预测与station-holdout manifest核对。其协议JSON的split mode与实际manifest不一致，须按实际manifest读数，JSON为元数据缺口。
- Random/Year/Station历史严格总结：每类仅4个代表任务，Random/Year保留汇总指标而无逐样本审计表，Station为fold-R²均值；更广的候选稳健性验证使用80/84特征，不能混作strict-77证据。
- 分海区结果、来源消融和XGBoost解释均保留原试验范围；来源消融未覆盖真实遥感变量。
- TabICLv2解释：175个task-fold SHAP样本记录、每样本77特征完整归因；初次严格推理15/20折。All-vs-strict敏感性与各变量排名见`03_shap_validation_and_sensitivity.csv`；5折有界复测在`tabicl_reproduction_probe.csv`。
- 1D ALE共16变量×5折，但覆盖强度不均；2D ALE七组均部分支持。

## 关键判断

没有发现正式十任务预测主结果的split泄漏、测试obs_id跨fold重复、XGBoost/TabICLv2测试样本不配对或pooled指标重算不符。当前限制主要涉及TabICLv2解释分支的复现波动和历史扩展实验元数据/粒度，不足以推翻冻结主结果；但不能声称所有解释折均严格复现。

已规划实验的计算基本完成。随机/年份总结与TabICLv2部分解释折的严格证据仍有限。可冻结正式实验主链及已明确标注限制的扩展结果库，再进入论文写作前应保留这些边界；不需要为消除所有限制而新增无预案训练。


## 复现差异探针补充

5/5折完成有界探针；同上下文复测及查询反序差异均为0，batch8/singleton最大差2.45e-06；重建同种子上下文的预测差最大值范围0–0.0104。这支持上下文重建路径的跨运行波动，但没有隔离具体底层原因。仅作为技术诊断，首次严格复现失败折不改判；主预测结果未被修改。
