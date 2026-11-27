# ADAPTIVE_ENSEMBLE_V2 固定协议

沿用 RANDOM_10TASK_V1 外部划分、Model Zoo V2 五折及 V1 的全部基础预测与四种融合二层交叉拟合预测。单模型按训练集 pooled OOF R² 选择；融合仅当其 OOF R² 优势严格大于0.005且至少3折胜出时入选。选择表先写出，再读取外部测试预测。外部测试仅作评价，不参与选择。V1二层交叉拟合不是基础模型严格嵌套OOF；独立外部测试为主要性能证据。

外部划分 SHA256：`bb9269039de23f8fb4412eb71bf8268a972ec40ca17c3d0b333a6e2fa791b359`；内部划分 SHA256：`c4fea605966709a29baf014ed7eb6c8c557d06122728017ea127ef9ae21213cc`。
