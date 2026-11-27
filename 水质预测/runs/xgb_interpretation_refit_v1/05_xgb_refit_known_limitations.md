# XGBoost受控复现模型已知限制

1. 本目录中的20个模型是按照冻结协议重新拟合的受控复现模型，不是找回的原始冻结模型二进制。
2. `EXACT_REPRODUCTION`仅表示逐样本预测达到事前固定的严格数值阈值，不证明每棵树的内部二进制结构与原模型逐字节相同。
3. 原始训练运行没有保存Git commit标识；本次通过当前训练脚本、配置、split、特征清单和输入文件的SHA-256固化复现依据。
4. 原始运行没有逐折环境锁文件。复现使用保留的隔离环境：Python 3.11.9、XGBoost 3.2.0、NumPy 2.4.6、pandas 2.3.3、scikit-learn 1.9.1。
5. 本次没有修改或覆盖Family V2、Secchi、最终实验审计及正式预测文件；新预测只保存在 `runs/xgb_interpretation_refit_v1/predictions/`。
6. 20折均完成模型和imputer独立重载推理，但这不消除未来软件升级、硬件变化或输入文件变化造成漂移的风险。后续使用前应校验manifest中的SHA-256。
7. 后续SHAP/ALE可以解释这些数值等价复现的XGBoost模型，但不能解释TabICLv2内部机理，也不能将统计关联表述为环境因果关系。
8. 本轮没有执行SHAP、1D ALE或2D ALE。
