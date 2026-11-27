# distance-to-mouth / zone / GPM 本轮执行摘要

本轮严格遵守以下边界：

- 不 promote current best
- 不修改 current best
- 不 cleanup
- `distance-to-mouth` 先跑单源，再按结果决定是否跑组合
- `GPM IMERG` 只做样本下载与 observation-date 对齐验证，不进入正式 A/B

## 1. distance_to_mouth_only

基于“主要河口系统级定义”运行了四指标 12 任务统一 A/B。

### 主要正信号

- `chla bottom`: `delta R² = +0.0032`
- `secchi bottom`: `delta R² = +0.0071`
- `secchi surface`: `delta R² = +0.0038`
- `tn bottom`: `delta R² = +0.0071`
- `tn surface`: `delta R² = +0.0023`

### 主要负迁移

- `secchi middle`: `delta R² = -0.0195`
- `tp middle`: `delta R² = -0.0242`
- `tp bottom`: `delta R² = -0.0254`
- `chla surface`: `delta R² = -0.0070`

### 判断

- 单源确实出现正信号，因此满足“可以继续跑 controlled combination”的前提。
- 但正信号偏向 `tn bottom / secchi bottom`，而不是你最关心的 `TN surface`。

## 2. distance_to_mouth + zone

在 `distance_to_mouth_only` 出现正信号后，继续运行了 `distance_to_mouth + zone` 的 controlled combination A/B。

### 主要正信号

- `chla surface`: `delta R² = +0.0083`
- `tn bottom`: `delta R² = +0.0062`
- `secchi surface`: `delta R² = +0.0028`
- `secchi bottom`: `delta R² = +0.0003`

### 主要负迁移

- `tn surface`: `delta R² = -0.0160`
- `tp bottom`: `delta R² = -0.0570`
- `secchi middle`: `delta R² = -0.0093`
- `tp surface`: `delta R² = -0.0088`

### 判断

- 组合后没有形成统一提升。
- 尤其 `TN surface` 从 `0.5201` 降到 `0.5041`，说明“distance-to-mouth + zone” 目前不适合作为整体推进四指标精度的主线。

## 3. GPM IMERG observation-date 对齐验证

本轮继续执行了 `GPM IMERG` 的样本下载与 observation-date 对齐验证。

### 已确认通过

- Earthdata 登录可用
- `GPM_3IMERGDF` V07 产品可发现
- 2014-2024 覆盖存在
- 香港研究区覆盖存在
- 适合后续构造 lag features

### 当前卡点

- 抽样 6 个观测日期都能发现 granule
- 但 6 个日期的样本下载全部被 `GES DISC` 条款接受步骤拦住
- 因此本轮仍未进入真正的 observation-date 数值提取

### 判断

- `GPM IMERG` 这条线在“产品存在、时空覆盖可行”层面是通过的
- 但在“样本 granule 可稳定下载”这一关仍未打通
- 因此下一轮还不能把它推进到正式 runoff proxy 特征构建

## 4. 本轮总体结论

1. `distance_to_mouth_only` 比 `distance_to_mouth + zone` 更值得保留为下一步候选。
2. `distance_to_mouth + zone` 对 `TN surface / TP` 的负迁移较明显，不建议继续作为主线。
3. `GPM IMERG` 目前最大问题不是建模，而是数据中心条款下载链未真正打通。
4. 本轮所有结果都只作为诊断性 A/B 和 source validation 使用，不进入 current best。
