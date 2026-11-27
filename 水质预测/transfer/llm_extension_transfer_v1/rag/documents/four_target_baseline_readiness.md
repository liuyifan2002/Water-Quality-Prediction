# 四指标 baseline readiness 报告

## 1. 本轮新增的实质进展

在完成四指标统一目标改写后，本轮没有停留在文档层面，而是进一步完成了：

- 基于 `data/interim/current/current_model_dataset_with_highres_and_mechanism.csv`
- 对 **TN / TP** 两项指标、三个水层
- 进行了一轮 **baseline bootstrap**

输出文件：

- `experiments/outputs/current/tn_tp_baseline_bootstrap_metrics.csv`
- `data/processed/current/tn_tp_baseline_bootstrap_scorecard.csv`
- `data/processed/current/four_target_bootstrap_scorecard.csv`
- `experiments/reports/current/tn_tp_baseline_bootstrap_report.md`

这意味着当前项目已经不再只有 `Chl-a + Secchi` 的 current 主线，而是第一次形成了可以继续扩展到 **四指标统一 A/B** 的 baseline 底盘。

## 2. TN baseline bootstrap 结果

### TN surface

- best candidate: `tn_surface_stable_core_lightgbm`
- validation R² = `0.5820`
- test R² = `0.5201`
- test RMSE = `0.3411`
- test MAE = `0.2473`

判断：

- 这是当前四指标中**最接近 0.60** 的新增任务；
- 说明 TN surface 在现有数据边界内已经具备较强可预测性；
- 后续新增通用机制数据如果能继续带来 `+0.03 ~ +0.08` 级别增益，TN surface 是最有希望先冲到 `0.60` 的任务之一。

### TN middle

- best candidate: `tn_middle_baseline_lightgbm`
- validation R² = `0.2703`
- test R² = `0.1823`
- test RMSE = `0.2921`
- test MAE = `0.2399`

判断：

- 中层 TN 明显比表层更难；
- 仅靠当前公开输入和既有机制代理，解释力有限；
- 后续更需要空间分区、交换、输入负荷和水动力相关数据。

### TN bottom

- validation-selected best: `tn_bottom_mechanism_context_lightgbm`
- validation R² = `0.2305`
- test R² = `0.2353`

补充说明：

- 若只看 test，`tn_bottom_baseline_random_forest` 与 `tn_bottom_stable_core_random_forest` 实际 test R² 更高；
- 但本轮仍按 **validation R² 优先** 选 baseline，避免把 test 上的偶然性直接写成 current best。

判断：

- TN bottom 目前仍然偏低；
- 但比 TP 深层略好，说明底层营养盐并不是完全不可预测，只是还缺更强输入。

## 3. TP baseline bootstrap 结果

### TP surface

- best candidate: `tp_surface_mechanism_context_random_forest`
- validation R² = `0.6389`
- test R² = `0.3986`
- test RMSE = `0.0531`
- test MAE = `0.0298`

判断：

- TP surface 具备一定可预测性；
- 但 validation 和 test 间存在明显落差，说明跨年泛化仍不稳；
- 这类任务后续需要更稳的输入，而不是直接拿 validation 高分当成正式突破。

### TP middle

- best candidate: `tp_middle_stable_core_lightgbm`
- validation R² = `0.1725`
- test R² = `0.0739`

### TP bottom

- best candidate: `tp_bottom_baseline_random_forest`
- validation R² = `0.2084`
- test R² = `0.0328`

判断：

- TP 的中下层目前仍明显偏弱；
- 这更支持“仅靠现有公开光学/弱机制输入，难以解释 TP 深层变化”的判断；
- 后续要真正提升 TP，优先级应放在：
  - bathymetry / station depth / zone
  - tide / exchange / water level
  - discharge / inflow / outflow / runoff

## 4. 当前四指标整体状态

结合原有 `Chl-a / Secchi` current best 与本轮 TN / TP bootstrap，可以得到当前四指标统一现状：

### 已有 current best

- Chl-a surface: `0.3298`
- Chl-a middle: `0.3646`
- Chl-a bottom: `0.3436`
- Secchi surface: `0.4080`
- Secchi middle: `0.3482`
- Secchi bottom: `0.3574`

### 新增 baseline ready

- TN surface: `0.5201`
- TN middle: `0.1823`
- TN bottom: `0.2353`
- TP surface: `0.3986`
- TP middle: `0.0739`
- TP bottom: `0.0328`

当前最接近 `R² = 0.60` 的任务已经变成：

1. `TN surface`：`0.5201`
2. `Secchi surface`：`0.4080`
3. `TP surface`：`0.3986`
4. `Chl-a middle`：`0.3646`

## 5. 对下一步提精度的意义

这组结果让后续“增加预测精度”的主线变得更具体了：

1. **四指标框架已经真正能跑，不再只是概念**
2. **TN surface 是当前最值得冲 0.60 的优先任务**
3. **TP middle / bottom 是当前最薄弱的任务**
4. **Chl-a bottom 仍然难，但已经不再是唯一焦点**

## 6. 下一步建议

从提精度角度，下一轮最值得直接做的是：

1. 用统一四指标模板，对 `bathymetry / depth / zone` 做单源 A/B
2. 用统一四指标模板，对 `tide / exchange proxy` 做单源 A/B
3. 对 `TN surface` 做针对性的强输入增强，优先尝试把它推进到 `R² > 0.60`
4. 对 `TP surface / middle / bottom` 优先补输入，不优先做调参

