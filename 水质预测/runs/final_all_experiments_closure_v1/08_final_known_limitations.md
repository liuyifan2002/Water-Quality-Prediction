# 实验闭环审计摘要

本轮审计以最终注册表V2为正式主链基准，并检查Family V2、modern baseline、历史随机/年份/分区实验、来源消融以及XGBoost/TabICLv2解释扩展。核心任务逐样本指标重新计算并和冻结汇总表核对；所有审计只读正式预测。

TabICLv2解释计算文件覆盖20/20折，但初次推理严格复现只有15/20折。后续ALE程序在同样任务/折上独立推理均记录严格通过，支持运行级波动判断，但原始失败不改判。SHAP A/B排名敏感性见03表；一维ALE覆盖不足见04表；二维ALE七组均保留部分支持见05表。

`modern_baseline_gate_v1/00_protocol.json`仍写random split，但实际manifest及预测注册指向station holdout；实际结果按manifest复核通过，配置JSON需在实验元数据层修订后才可作为自包含复现协议。历史Random/Year只有聚合指标，区域实验与主链split/features不同，不与正式结果合并。

