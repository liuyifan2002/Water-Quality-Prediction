# LLM_EXTENSION_TRANSFER_PACKAGE_V1

本目录是从原始研究项目只读复制出的实验迁移包，面向另一台 Windows 电脑上的 RTX 4060 / 32 GB RAM 环境。迁移目标为：

- `LLM_DYNAMIC_ROUTING_V1`：本地 LLM 辅助的动态模型选择与融合；
- `LLM_KNOWLEDGE_RAG_V1`：环境知识增强的水质预测解释和诊断。

本次工作只完成盘点、复制、校验和打包，未启动任何模型训练或推理，未改变既有 train/test split、canonical station-fold map、77 strict features 或冻结预测结果。

## 正式来源判定

正式主表确认采用：

`E:\研究生\codex\水质预测\数据\处理后数据\正式主表\model_table_3depth_2014_2024.csv`

判定依据是 Family V2 协议、正式主链配置和现有模型脚本引用关系共同指向该文件；同名衍生表、外部试验表和 portable package 副本未被替代为正式主表。

冻结 77 特征清单为：

`runs\tabiclv2_pilot\01_strict_feature_manifest.csv`

## 关键目录

- `data/`：正式主表、站点元数据、正式静态辅助表和 77 特征清单；
- `splits/`：CHLA/TN/TP Family V2、Secchi station-holdout、现代基准四任务的冻结 split 来源；
- `predictions/`：现代基准四任务的四模型逐样本预测，以及十任务核心 XGBoost/TabICLv2 OOF 预测；
- `results/`：V2 闭环结果、现代基准汇总、Family V2 汇总、分海区结果、多源消融结果；
- `code/`：已有预处理、模型运行、预测恢复、结果审计脚本和配置；
- `model_assets/`：TabPFN v2 checkpoint 及缓存元数据；
- `rag/`：已有水质来源、变量/输入说明、误差/分区/消融分析文档。完整文献库尚未形成，见 `manifest/unresolved_items.md`；
- `validation/`：可直接运行的迁移包校验器与校验输出。

## 迁移后校验

在本目录的上一级或任意位置运行：

```powershell
python validation\validate_transfer.py
```

该脚本只读取文件并检查 schema、obs_id、77 特征、split PASS、四模型逐折测试样本集合、V2 指标字段和内部 SHA256，不导入模型库，不训练，不推理。

若只查看终端结果而不更新校验报告，可运行：

```powershell
python validation\validate_transfer.py --check-only
```

## 模型资产说明

- XGBoost：正式预测已迁移；未来重跑还需按原配置安装 XGBoost；
- TabICLv2：正式预测和适配器已迁移；TabICLv2 运行环境需要按原记录安装；
- TabM：四个代表任务的冻结预测已迁移；新电脑需要安装 `tabm`；
- TabPFN：冻结预测和本地 `tabpfn-v2-regressor.ckpt` 已迁移。该权重属于可按固定版本重新获取的预训练权重，但本包保留本地副本以支持离线检查；
- 未复制 Python 虚拟环境、旧 CUDA 环境、完整 Hugging Face 缓存、`__pycache__` 和历史 portable package 的虚拟环境。

## 结果口径

V2 闭环结果优先于历史 V1 结果；`pooled_R2` 与 `fold_R2_mean` 在结果表中保持为两个独立字段。TN/TP 的 V2 核心结果已经有正样本量和核心指标，校验信息保存在 `results/final_experiment_closure_audit_v1/final_registry_validation_checks_v2.csv`。

现代基准四任务的 XGBoost/TabICLv2 来源于核心冻结预测，TabM/TabPFN 来源于 `MODERN_BASELINE_GATE_V1`。Secchi 的 TabICLv2 使用已恢复预测；其恢复检查文件一并迁移。各任务各折四模型的 `(fold, obs_id)` 测试集合由校验器严格比较。

## 许可与敏感信息

本包没有自动下载文献，没有生成参考文献，没有复制账号、密钥、认证文件或旧虚拟环境。被排除项目及原因见 `manifest/excluded_items.csv`；原始路径到迁移路径映射见 `manifest/source_path_mapping.csv`。
