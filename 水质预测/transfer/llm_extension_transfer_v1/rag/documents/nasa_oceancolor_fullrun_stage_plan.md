# NASA OceanColor 全时段构建与严格 A/B 计划

## 本轮使用的 NASA 产品

- 第一优先级：`VIIRSN_L3m_CHL`、`VIIRSN_L3m_KD`。
- 第二优先级：`VIIRSN_L3m_RRS`，仅在产品结构稳定且提取成功率可接受时纳入。
- 第三优先级：`MODISA_L3m_CHL`，仅作为传感器对照与 `chl_sensor_diff` 来源。

## 为什么先做月尺度

1. 当前主表缺少 `exact_sampling_datetime`，无法可靠做真实采样时刻匹配。
2. 月尺度先验证 NASA 表层光学代理是否在总体上提供稳定正信号。
3. 日尺度和 8-day 在近岸云、缺测和采样时刻不确定条件下更容易引入噪声。
4. 如果月尺度都没有正增益，高频扩展的收益通常有限。

## 为什么先用 CHL / KD490 / RRS

1. `CHL` 是最直接的表层叶绿素光学代理。
2. `KD490` 与透明度/光衰减机制更直接相关，理论上更可能帮助 Secchi。
3. `RRS` 可以补充波段形状信息，但复杂度更高，因此放在稳定下载后再纳入。
4. 本轮不无限扩展产品，避免文件和特征失控。

## 任务优先级

- 最高优先级：`Chl-a bottom`。
- 次优先级：`Secchi middle`、`Secchi bottom`。
- 第三优先级：`Chl-a middle`。
- 其余任务：按同口径完整评估，但不额外扩大战线。

## 预期帮助机制

- NASA OceanColor 是**表层遥感/光学代理**，不是底层实测数据。
- `CHL` 可能帮助 Chl-a surface / middle，也可能通过表层-底层弱相关间接帮助 bottom。
- `KD490` 与 Secchi / 透明度机制最直接，优先关注 Secchi middle / bottom。
- `RRS` 若成功，可能比单一 CHL 更能捕捉近岸水色和浊度差异。

## 当前 best 任务状态

- `chla surface`: 当前 best R² = 0.3298，距离 0.60 还差 0.2702。
- `chla middle`: 当前 best R² = 0.3646，距离 0.60 还差 0.2354。
- `chla bottom`: 当前 best R² = 0.3436，距离 0.60 还差 0.2564。
- `secchi surface`: 当前 best R² = 0.4080，距离 0.60 还差 0.1920。
- `secchi middle`: 当前 best R² = 0.3482，距离 0.60 还差 0.2518。
- `secchi bottom`: 当前 best R² = 0.3574，距离 0.60 还差 0.2426。

## NASA 产品发现状态

- `VIIRSN_L3m_CHL`: found = True, concept_id = `C3388381565-OB_CLOUD`。
- `VIIRSN_L3m_KD`: found = True, concept_id = `C3388381638-OB_CLOUD`。
- `VIIRSN_L3m_RRS`: found = True, concept_id = `C3388381848-OB_CLOUD`。
- `MODISA_L3m_CHL`: found = True, concept_id = `C3380709133-OB_CLOUD`。

## 明确边界

- NASA OceanColor 只能提供表层/光学遥感代理，不能替代 `bottom water temperature`、`ADCP current`、`vertical profile` 或真实采样潮相。
- 本轮不会同时开启 GEBCO、HKHO、ADCP 或 bottom profile 新方向；这些仅保留在现有文档中。
