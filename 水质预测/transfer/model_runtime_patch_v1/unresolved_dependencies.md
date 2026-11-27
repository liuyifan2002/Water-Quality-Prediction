# 未解决依赖与恢复阻塞项

## `TORCH_WHEEL_DOWNLOAD_REQUIRED`

原始隔离环境实际使用 `torch==2.6.0+cu124`，但本机没有发现可直接复制的完整 Windows x64 / Python 3.11 wheel。检查到的 pip cache 约 759.2 MB 为 HTTP 缓存，locally built wheels 仅约 4.8 kB，不能替代 PyTorch CUDA wheel。不要从原 `.venv` 直接复制 `site-packages`。

新电脑需要在独立的 Python 3.11 x64 环境中，按原记录从 PyTorch 官方 CUDA 12.4 wheel 源获取精确版本 `torch==2.6.0+cu124`，然后再安装本清单中的 TabICL、TabM、TabPFN 精确版本。安装后应先执行环境探针和包元数据校验，不启动正式实验。

## `TABM_SOURCE_COMMIT_NOT_RECORDED`

历史环境中的 `tabm==0.0.3` 是非 editable 的已安装 distribution，`direct_url.json` 未提供源码地址或 commit。包元数据指向官方仓库 `yandex-research/tabm`，但无法从当前环境证明具体 Git revision。迁移时必须固定 PyPI distribution 版本，不得声称已经恢复某个 Git commit。

## `CATBOOST_NOT_INSTALLED_IN_HISTORICAL_ENV`

原始隔离环境的 `pip freeze` 和 `importlib.metadata` 未发现 CatBoost，且 `import catboost` 不可用；但便携式现代模型配置的约束文件仍列出 `catboost>=1.2,<2`。这属于便携式 runner 的依赖缺口，不应伪装成原始环境已安装。若新电脑只恢复 XGBoost、TabICLv2、TabM、TabPFN 四类目标模型，可按实际脚本依赖评估是否需要额外安装；本补丁不自动安装。

## `TABPFN_SOURCE_REVISION_NOT_RECORDED`

原始环境可确认 `tabpfn==9.0.0`、项目来源 URL 和本地 V2 checkpoint；当前环境没有记录 TabPFN 源码 Git revision。checkpoint 的历史使用关系由配置、脚本和既有结果记录支持，未重新运行模型验证。

## 其他说明

- 没有发现需要迁移的 TabICLv2 项目内 checkpoint 副本；Hugging Face 缓存中的精确文件已经复制到 `checkpoints/`。
- 没有复制整个虚拟环境、旧 CUDA runtime、pip HTTP 缓存或无关模型缓存。
- 本轮未下载大文件、未运行任何模型训练或推理、未修改任何正式实验结果。
