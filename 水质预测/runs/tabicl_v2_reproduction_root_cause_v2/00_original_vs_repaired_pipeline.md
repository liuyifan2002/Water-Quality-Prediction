# 原正式推理与隔离修复流程对照

## 正式入口与不变量

Chl-a Family V2通过`脚本/04_模型训练/33_run_tabiclv2_chla_smoke.py::fit_tabicl`和`models/tabiclv2_adapter.py`调用TabICLv2。模型适配器使用预训练权重、random_state=42、CUDA、AMP auto。正式脚本按77特征顺序构造float32输入，逐折median填补，训练目标采用log1p；模型预测内部逆标准化后，外层执行expm1与非负截断。正式路径将完整测试矩阵一次传入predict。

本轮所有条件复用相同fold1训练行、相同训练上下文构造规则、同一固定32个测试obs_id。填补后训练/query矩阵、目标变换、特征编码和imputer统计量跨条件最大数值差均为0。checkpoint哈希保持一致。固定顺序隔离补丁仅确定视图排列；诊断变体只改变查询行排列、查询分块或集成成员batch大小。

## 本轮可观测差异

固定`PYTHONHASHSEED=42`下，原路径未额外初始化Python/NumPy/PyTorch全局RNG，与显式全局seed=42的预测一致：max/mean冻结误差分别为0.0048978687/0.00080113545。因此本次没有证据表明额外全局seed初始化可解释剩余差异。

在固定`none,power`视图顺序、同一拟合对象下：基准32样本一次调用耗时6.60s；按原测试表顺序重排后对齐回原obs_id的max/mean变化为0.0024489054/0.00050298651；反序变化为0.0024489054/0.00050298651；4批×8查询变化为0.0089841061/0.00092484801；内部集成成员batch_size 8→4变化为0.0024489054/0.00050298651。这些差异远高于1e-6，说明预测依赖查询集合/顺序或前向分批数值；该诊断不能证明它是历史全测试差异的唯一成因。

## 证据与解释边界

阶段数值比较见`array_comparison.csv`，完整trace见`02_intermediate_state_trace.csv`，视图级输入与模型输出见`workers/*_trace_forward.csv`，所有固定查询批次比较见`04_query_batch_sensitivity.csv`。历史冻结运行没有保存拟合估计器、预处理对象、原始Python hash seed、pre-fit RNG bytes或逐视图输出。因此当前能定位到“查询行集合/顺序进入模型前向计算后产生了逐样本输出差异”，不能反推当年具体查询批次或唯一恢复历史输出。