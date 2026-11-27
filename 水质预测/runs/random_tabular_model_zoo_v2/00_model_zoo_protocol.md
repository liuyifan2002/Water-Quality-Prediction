# RANDOM_TABULAR_MODEL_ZOO_V2 实验协议

## 冻结边界

- 外部训练/测试样本直接取自 `runs/random_10task_v1/01_random_split_manifest.csv` 的 `random_strict / seed_42`，十任务 obs_id 不重划。该文件的 SHA-256 已写入 `00_protocol.json`。
- 输入为正式主表 `model_table_3depth_2014_2024.csv` 中 Pilot strict manifest 标记纳入的77项特征，不含 `water_zone_code`。不读取测试目标用于任何内部选择。
- 内部五折：对各任务外部训练 obs_id 排序后，使用 `KFold(n_splits=5, shuffle=True, random_state=42)`；十项任务的内部折次记录在 `01_inner_fold_manifest.csv`。不同任务的折号并不表示同一批样本。
- 所有填补、缩放均在当前训练折内拟合；目标非负时在拟合端使用 log1p，预测端使用 expm1 并截断负预测。所有 R²、RMSE、MAE 在原始目标单位计算。
- 基准模型先产生完整训练集 inner OOF。传统模型仅对 OOF 表现前两位做有限候选调参。最终单模型或非负融合通过内部 OOF 和二级交叉验证确定，外部测试集不参与候选、参数或融合权重选择。
- 外部测试结果仅在方案锁定后计算；历史 V1 逐样本结果只有在数据、特征、划分和模型参数均一致时复用。Random 与 Station V2 仅作并列参考，不能视为同一泛化协议。

## 模型身份与环境

- XGBoost、CatBoost、LightGBM、随机森林、极端随机树、HistGradientBoosting、SVR、Ridge 和 MLP 组成首轮基准。
- TabICLv2：现有 `tabicl==2.2.0` 和 Pilot 适配器；固定查询顺序，不按结果调整批次。
- TabPFN：现有 `tabpfn==9.0.0` 和本地 V2 checkpoint。
- TabM：`tabm==0.0.3`，本轮明确使用 `arch_type=tabm, k=32`。旧 `tabm-mini, k=16` 结果不复用。
- 本机实测 GPU 为 RTX 2070 Max-Q 8GB，并非任务书预期的 RTX 4060 8GB。GPU 模型顺序运行。TabDPT、TabFM、Mitra-v2 的纳入以权重、许可、独立环境和实际规模运行检查为准，失败保留 BLOCKED 状态。

## 结果含义

- 内部 OOF 用于任务级模型和融合选择；外部 Random 测试集用于最终报告。
- 融合权重为非负且和为1。二级五折在 OOF 预测上交叉拟合权重，并与单模型比较；若融合增益不足，保留单模型。
- R² ≥ 0.5 是研究目标，不是筛选测试样本或改变指标的准则。不利结果同样保留。

## 条件纳入模型的官方来源

- TabDPT：官方回归实现与 v1.3 权重见 [Layer6 推理仓库](https://github.com/layer6ai-labs/TabDPT-inference) 和 [模型页](https://huggingface.co/Layer6/TabDPT)。已在隔离环境安装 `tabdpt==1.3.2`，整规模运行通过后才能列入比较。
- TabFM：官方[代码与使用说明](https://github.com/google-research/tabfm)确认支持回归；代码为 Apache-2.0，默认预训练权重另受非商业许可约束。其当前 PyTorch 后端要求与现有冻结环境不同，不能在原环境直接升级核心依赖。
- Mitra-v2：官方[回归模型页](https://huggingface.co/autogluon/mitra-regressor-2)标注 Apache-2.0，并要求专用 `mitra-finetune` 推理实现；不能将旧版 Mitra 回归接口冒充 v2。
