# RANDOM_10TASK_V1：Random Strict 协议

- 数据：`数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv`；使用 Pilot 冻结 manifest 中 `included_in_pilot=True` 的77项特征，排除 `water_zone_code`。
- 划分：沿用历史 `random_strict`、`seed_42`、约80%训练/20%测试的逐样本 obs_id 分配。Chl-a表层、Secchi、TN底层、TP中层严格复制 Pilot manifest；其余六项直接复制历史全局 seed42 manifest。未生成新随机种子或重抽测试集。
- 历史四任务划分的源脚本为 `脚本/05_诊断分析/18_prepare_tabiclv2_pilot_assets.py`；Pilot 的前三项源于候选模型稳健性验证保存的 seed42 obs_id，TP中层源于全局 smoke manifest。任务的有效样本为指定水层、目标值非缺失的主表记录。Secchi仅作为 surface 单任务。
- 模型：XGBoost 原 Pilot 参数和 TabICLv2 原适配器；对新任务 XGBoost 只在外部训练集内部固定20%验证划分比较三个预设参数组合，再以完整外部训练集拟合一次，外部测试集只评估。TabICLv2 使用原默认推理设置。训练集拟合中位数填补；非负目标使用 log1p 拟合、expm1 逆变换并把负预测截为零，和 Pilot 一致。
- 已有逐样本结果完全同协议者复用：Secchi、TN底层、TP中层的 XGBoost 和 TabICLv2。Chl-a表层仅有指标、没有同协议逐样本预测文件，按 Pilot 原配置重新运行；六个新增任务运行两模型。
- 指标：单次 Random 测试集的 R²、RMSE、MAE。此协议没有五折，不能计算 pooled OOF 或 fold mean。与 Station V2 的五折 pooled OOF 只做并列描述，不能当作相同评价对象的直接增益。
- 十任务各有冻结测试集预测文件。`test_best_observed_model` 只是事后描述两模型测试结果，不是以测试集调参或选择可部署模型的规则。
