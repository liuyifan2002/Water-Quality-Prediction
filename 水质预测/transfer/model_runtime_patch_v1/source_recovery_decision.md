# SOURCE_MODEL_DEPENDENCY_RECOVERY_V1 决策记录

本目录只保存历史运行时资产、来源证据和迁移校验信息。原始实验已冻结；本次未启动模型训练或推理，未修改正式预测结果、split、特征清单或原始模型文件。

## 已确认可恢复资产

- 原始隔离环境：`runs/tabiclv2_pilot/环境/.venv`。
- Python：3.11.9；Windows x64；原机实际验证 GPU 为 NVIDIA GeForce RTX 2070 with Max-Q Design。
- PyTorch：`2.6.0+cu124`，CUDA 构建 12.4；原环境探针记录 CUDA 可用。
- TabICL：`2.2.0`；TabICLv2 官方回归 checkpoint 已从本机 Hugging Face 缓存复制到 `checkpoints/`。
- TabICLv2 checkpoint：`tabicl-regressor-v2-20260212.ckpt`，SHA256 与用户提供的参考值完全一致；缓存快照 revision 为 `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`，对应仓库 `jingang/TabICL`。
- TabPFN：`9.0.0`；历史运行配置为 `n_estimators=8`、本地 `models/tabpfn-v2-regressor.ckpt`。项目内全部历史副本哈希一致，且新电脑已有匹配权重，因此本补丁不重复复制该权重。
- TabM：`0.0.3`；为已安装的 PyPI distribution，不是 editable/source install；未发现可追溯的 Git commit，未用其他实现替代。正式配置和原始运行脚本已复制到本目录。

## 版本恢复判断

TabICLv2、TabPFN、TabM 的版本以原始隔离环境的 `pip freeze`、`pip show`、`importlib.metadata` 和模块文件位置交叉记录。原始代码调用分别为 `tabicl.TabICLRegressor`、`tabpfn.TabPFNRegressor` 和 `tabm.TabM`。其中 TabM 的安装元数据提供官方仓库地址 `https://github.com/yandex-research/tabm`，但没有 source commit；新电脑应优先安装精确 distribution 版本 `tabm==0.0.3`，不得替换成另一套 TabM 实现。

## 不复制的资产

- 不复制原始 `.venv`、`site-packages`、旧 CUDA 环境或无关缓存。
- TabPFN V2 checkpoint 不重复复制，因为项目历史副本和新电脑已有文件的 SHA256 已匹配：`2AB5A07D5C41DFE6DB9AA7AE106FC6DE898326C2765BE66505A07E2868C10736`。
- 原始项目中未发现可直接复用的完整 Windows x64 / Python 3.11 / `torch==2.6.0+cu124` wheel；本机 pip cache 仅有 HTTP 缓存和约 4.8 kB 的本地构建 wheel，不能作为完整 PyTorch 安装介质。

## 迁移结论

TabICLv2 权重、历史版本和配置证据已恢复；TabPFN 历史权重已核验为可复用；TabM 精确包版本已确定但没有源码 commit；PyTorch CUDA wheel 需要在新电脑按官方 CUDA 12.4 wheel 源重新获取。因此本补丁是**部分完成**，结论为 `SOURCE_RUNTIME_ASSETS_PARTIAL`，不是 `SOURCE_RUNTIME_ASSETS_READY`。
