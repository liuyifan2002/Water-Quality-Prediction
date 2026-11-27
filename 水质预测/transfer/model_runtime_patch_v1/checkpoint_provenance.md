# Checkpoint 来源与兼容性证据

## TabICLv2

- 文件：`checkpoints/tabicl-regressor-v2-20260212.ckpt`
- 来源：本机 Hugging Face 缓存仓库 `jingang/TabICL`
- 缓存 snapshot revision：`4dcd344ece2c00be9e831fdd35bed57b5ad83e19`
- 大小：114,324,594 bytes
- SHA256：`0DB9CB538F114E79026BF08F45F41AD8DD7AD2DE2AACA9A5CA8CD3BD9748AE7A`
- 参考核对：与任务给出的官方参考 SHA256 完全一致。
- 历史匹配证据：`runs/tabiclv2_pilot/00_environment_check.md` 记录官方默认 checkpoint 自动下载成功；`脚本/05_诊断分析/18_prepare_tabiclv2_pilot_assets.py` 将同名文件作为官方默认 checkpoint；原始适配器调用 `tabicl.TabICLRegressor`。
- 复制方式：从原始 Hugging Face 缓存复制，未移动、删除或改写原缓存。

## TabPFN V2

- 文件：原项目 `models/tabpfn-v2-regressor.ckpt`
- 大小：44,390,977 bytes
- SHA256：`2AB5A07D5C41DFE6DB9AA7AE106FC6DE898326C2765BE66505A07E2868C10736`
- 历史包版本：`tabpfn==9.0.0`
- 历史配置：`n_estimators=8`，`model_path=models/tabpfn-v2-regressor.ckpt`
- 兼容性证据：原始现代 baseline 配置和 runner 明确调用 `TabPFNRegressor` 并指向该本地 checkpoint；项目内全部历史副本哈希一致；新电脑已有相同 SHA256 的 V2 权重。
- 迁移决策：不重复复制；由新电脑现有匹配权重承担该资产。若新电脑实际文件哈希不同，应停止恢复并重新核对，不得用其他 checkpoint 替换。

## TabM

TabM 不使用单独的预训练 checkpoint。历史环境为 `tabm==0.0.3`，配置使用 `arch_type=tabm-mini`、`k=16`，并在 runner 中通过 `TabM.make` 构造模型；训练相关参数和预处理脚本已作为只读源码/配置复制。当前任务不启动训练，因此未生成或复制新的 TabM 权重。
