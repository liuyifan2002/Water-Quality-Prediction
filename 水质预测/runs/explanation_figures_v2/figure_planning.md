# 多样化学位论文图件 V2：数据资产审计与图件规划

> 本轮仅完成资产审计和规划，不生成图件，不运行模型，不重新计算SHAP/ALE。已有 MAIN-01～03 V1/V2 未被覆盖。

## 审计摘要
- SHAP：40组，10个任务各4组；每组应为77行、逐特征值和逐特征SHAP值。合格文件：40/40。
- ALE 1D：20条曲线；19条为10 bins，1条存在自适应bin数。数值敏感候选：chla_middle_wind_speed, chla_middle_wind_speed_sq。
- P1：9个任务探针文件、9个决策文件。具备9个任务的3次重复探针和P1决策字段；缺少chla_surface；图件只能表达探针重复范围，不代表全样本预测置信区间。
- ALE 2D：3个候选均只有联合支持审查，无response surface；分别为TN surface 13/16、TN bottom 7/16、TP bottom 13/16，状态为ALE_2D_BLOCKED_BY_SUPPORT。
- publication feature mapping：存在，77行，SHA256=d66c26e91df0b5af4d86e9fcc8119d5f9717c7e1de8c1f02cd534fe75cd7bb69。

## 图件分类边界
- COMPUTED_RESULT_FIGURE：由已完成的SHAP/ALE数值结果直接组织，不改变模型结果。
- SUPPORT_AUDIT_FIGURE：只表达样本数、独立站点数和预登记支持状态，不称为模型解释结果。
- NUMERICAL_STABILITY_FIGURE：只表达P1重复探针、ALE重复性或已有跨运行比较；coverage不完整时必须标注partial。

## 可复现样本规则
- Waterfall固定选择：CORE_3 且 target_quantile=0.5；10个任务样本；不按图形美观人工挑选。样本ID已在审计脚本运行时读取，未按图形美观人工选择。
- 10任务图件按任务单独保存；不为压缩数量而把不同科学问题强行合图。

## 分批执行计划
1. 第一批：保留已有MAIN-01～03 V1/V2，不覆盖。
2. 第二批：SHAP全局与分布解释（热图、任务独立排名、排名变化、来源组、Beeswarm、Violin）。
3. 第三批：10个任务的预登记典型样本Waterfall。
4. 第四批：10个任务的ALE 1D图，每张展示该任务已完成的两个特征。
5. 第五批：ALE支持度、数值稳定性和3个ALE 2D联合支持审查图。

## 当前阻塞/需在出图前固定的事项
- SMALL_EFFECT_INTERPRET_WITH_CAUTION 的数值阈值尚未在本轮重新规定；生成前应读取历史协议或明确登记阈值，不能临时从图形中推断。
- P1图件的语义限定为重复探针范围，不得写成预测置信区间。
- ALE 2D只能画支持矩阵，禁止生成或暗示响应曲面。
