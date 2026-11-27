# TabICLv2最终可解释性实验协议

实验协议：`TABICLV2_POINTWISE_EXPLANATION_V1`。四项任务、20折；77个冻结特征及正式fold manifest；每折训练样本单独拟合median imputer；只构造TabICLv2预训练权重的推理上下文，不更新预训练权重；固定seed=42、Python hash seed=42、隔离补丁中的`none,power`视图顺序、float32、AMP=auto、eval模式；目标按原正式任务逻辑使用log1p/expm1并非负截断。逐查询函数将每个SHAP/ALE输入行单独调用一次模型，不把其他查询或标签放入上下文。所有原批量预测及新解释结果分开保存。

冻结资产来源、77特征/obs顺序哈希、上下文与权重SHA256、环境和预测接口参数记录在`environment_manifest.csv`与逐折worker输出。Python/NumPy/PyTorch/CUDA/GPU版本在运行时采集。原始冻结预测按obs_id配对；任何新点式结果均不替换正式预测。

计算状态和已固定协议列于`13_final_explanation_master_registry.csv`。新解释结果必须通过独立脚本`81_validate_tabicl_explanation_results.py`核验后才可标记`COMPLETE_VALIDATED`。任务支持SQLite/逐样本检查点；重复调用调度器会跳过已有有效输出。
