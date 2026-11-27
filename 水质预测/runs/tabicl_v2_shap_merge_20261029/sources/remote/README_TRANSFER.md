# TabICLv2 V2 SHAP 回传包

生成时间：2026-10-29（北京时间）

## 内容

本包含本机完成的 7 个任务：Chl-a 表层、中层、底层，Secchi，TN 中层，TP 表层、TP 中层。共 21 份预登记 CORE_3 SHAP 和 7 份 seed43 敏感性 SHAP；每份均为 77 特征且所有行状态为 `COMPUTED_PASS`。

包内还附每个 SHAP 文件的 SHA-256 清单、V2预登记样本表、实验协议、这 7 项任务对应的运行登记筛选副本，以及核心和敏感性运行日志。

TN 表层、TN 底层、TP 底层由用户确认在另一台电脑完成；本机未把它们作为本次回传计算结果打包。所有 ALE 均按用户要求不在本机计算，因此本包不含 ALE 曲线。

## 导入到原电脑

1. 先备份原电脑对应目录 `runs/tabicl_v2_slim_explainability_v2_all10/shap`。
2. 解压本 ZIP。
3. 将解压后的 `tabicl_v2_shap_return_20261029/shap/` 与原目录的 `shap/` 合并，保留已有文件；不要用包内筛选版 `run_registry_local_tasks.csv` 覆盖原始完整 `run_registry.csv`。
4. 用 `manifest_sha256.csv` 校验文件。若同名文件的哈希不同，先保留两份并查明来源，不要直接覆盖。

文件完整性可用 ZIP 旁边的 `.sha256` 文件校验。包不包含模型 checkpoint、训练数据或环境，适用于回传 SHAP 结果和审计记录。
