# TabICLv2近岸水质领域适配实验协议

- 实验：TABICLV2_WATER_QUALITY_ADAPTATION_V1，独立扩展实验。
- 数据：正式主表原样读取，77项strict特征；不含water_zone_code。
- 外部划分：复用RANDOM_10TASK_V1，split SHA256=bb9269039de23f8fb4412eb71bf8268a972ec40ca17c3d0b333a6e2fa791b359。
- 内部划分：复用Model Zoo V2的01_inner_fold_manifest.csv，SHA256=c4fea605966709a29baf014ed7eb6c8c557d06122728017ea127ef9ae21213cc；全部调参仅使用外部训练集。
- 特征顺序：Pilot 77项manifest，SHA256=115cee492668ac4da82560279b9d5a8fb9c664a0cb740a2cbc46ddedd15a1573。
- 模型身份：tabicl==2.2.0，torch==2.6.0+cu124，回归权重SHA256=0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a；权重与本机官方缓存哈希一致。
- 原始基准：复用Model Zoo V2 TabICLv2内部OOF及外部测试预测；Z1与默认none+power相同，不重算。
- 选择：首轮最多五种推理配置、同一内部折次初筛；最多两种进入完整五折。最终以完整五折pooled OOF R²为主，RMSE/MAE辅助。
- 目标变换、缺失填补与特征选择均仅在各内部训练折拟合；原始水质单位上评价。
- 外部测试集只在内部配置锁定后评价一次；不能根据测试分数改选。
- 官方微调在隔离依赖目录内完成TN中层一个内部折代表试验；早停后未优于零样本基线，未扩展至十任务或参与外部测试。
- 不修改正式主链、冻结预测、split、77项特征或历史模型资产。
