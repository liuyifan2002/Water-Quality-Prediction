# TN station split 恢复审计

## 结论：BLOCKED_CONFLICT

存在 station_id 跨任务折次冲突，不能自行选择折次；TN surface/middle 保持 BLOCKED。

证据行数：`1355`；涉及站点：`94`；冲突站点：`91`。

## 审计文件

- `30_station_fold_recovery_evidence.csv`：各任务、各来源提取的 station-fold 证据。
- `30_station_fold_consistency_audit.csv`：按 station_id 汇总的跨任务一致性。
- `30_station_fold_conflicts.csv`：冲突清单；为空表示未发现冲突。
- `31_canonical_station_fold_map.csv`：仅在一致性通过时生成。

## 约束

本次只恢复历史固定站点分组，没有调用 GroupKFold、KFold 或随机 shuffle；没有运行模型、安装依赖或修改正式主链。