# 剩余限制

- 32样本固定顺序在此前三次独立进程中可重复，但当前与历史冻结预测不满足max≤1e-6、mean≤1e-7；不能启动完整测试集复现。
- 本轮`original_ambient`是在为隔离控制统一设置PYTHONHASHSEED=42的进程中运行，仅未额外初始化全局Python/NumPy/PyTorch RNG；它不是当年历史进程的原样副本。
- 查询重排和拆分后预测变化说明模型实现对查询集合/顺序/执行形状敏感。当前探针只能识别这一类依赖，不能证明历史完整测试矩阵中具体是哪一项造成差异。
- 原历史拟合估计器、imputer对象、查询/训练实际序列化上下文、完整pre-fit RNG state、逐视图模型输出及原PYTHONHASHSEED均未保存，无法严格还原。
- 内部batch_size=4与8的输出差异较大；查询分块还改变模型看到的查询集合，既不是安全等价加速，也不能用于正式预测替代。
- RTX 2070 8GB环境中未启用TabICLv2 kv_cache/representation cache，因此不报告推测性加速或峰值显存收益。
- 未运行fold1全测试集，未重新计算其他fold，未重算SHAP/ALE，未修改任何冻结预测和论文Word。
