# 源码根因定位与隔离修复

## 范围和代码路径

运行环境为TabICLv2 2.2.0，Python 3.11.9、PyTorch 2.6.0+cu124、CUDA runtime 12.4，GPU为NVIDIA GeForce RTX 2070 with Max-Q Design。实际加载的上游文件路径为`E:\研究生\codex\水质预测\runs\tabiclv2_pilot\环境\.venv\Lib\site-packages\tabicl\_sklearn\preprocessing.py`。本轮只把TabICLv2包复制到`runs/tabicl_v2_deterministic_repair_v1/isolated_patch/tabicl/`，不改安装环境。

调用链为`TabICLRegressor.fit` → `EnsembleGenerator.fit` → `_generate_ensemble` → `ensemble_configs_`有序分组 → `EnsembleGenerator.transform`依序生成各归一化视图 → 推理侧按这些视图构造批次。原实现第1129行的`list(set([config[1] ...]))`将归一化方法重排；由于后续预处理器字典、视图拼接和随机数消耗顺序均沿该顺序执行，Python哈希种子变化可改变上下文及预测。

## 修复与对照证据

隔离副本将分组顺序固定为估计器声明的`norm_methods_`顺序（该任务默认为`none,power`），并允许诊断运行显式比较已配置的两种顺序。原始安装包SHA-256与副本SHA-256见`01_deterministic_patch_manifest.csv`。

fold 1的三次独立进程在固定`none,power`下得到相同上下文哈希、相同实际视图顺序，逐样本预测两两最大差为0；固定`power,none`下也得到另一组稳定上下文和稳定输出。两种视图顺序间32样本预测最大差为0.0035162776、平均差为0.00075040013，两组均未达到冻结预测阈值（详见`02_context_hash_comparison.csv`和`03_chla_fold1_reproduction.csv`）。结合上一轮五折仅改变`PYTHONHASHSEED`的配对结果，确认无序集合是跨进程顺序不稳定的直接来源之一；不能据此认定它是全部冻结数值差异的唯一来源。

## 结论边界

本轮只证明32个固定查询样本下的重建确定性，不是全测试集严格复现。由于探针预测相对冻结值不等价，且按本轮实测吞吐量估算规定的15次全测试集验证超过6小时，未启动完整测试集、其他四折、SHAP或ALE敏感性复核。