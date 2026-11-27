# zone refinement 单源 A/B 报告

本轮仅执行 `scheme4` 的 `zone_refinement_only` 统一四指标 12 任务 A/B。
按用户要求，本轮结果只用于诊断，不用于 promote，也不修改 current best。

- 有效 zone 特征：scheme4_zone_id
- 缺失率：0.0000
- 成功覆盖率：1.0000

## R² 提升任务
- `chla bottom`: baseline R² `0.3130` -> candidate R² `0.3155`，delta `+0.0025`
- `chla middle`: baseline R² `0.3667` -> candidate R² `0.3687`，delta `+0.0019`
- `tn bottom`: baseline R² `0.2353` -> candidate R² `0.2388`，delta `+0.0035`
- `tp middle`: baseline R² `0.0739` -> candidate R² `0.0870`，delta `+0.0130`
- `tp surface`: baseline R² `0.3986` -> candidate R² `0.3996`，delta `+0.0011`

## R² 下降或持平任务
- `chla surface`: baseline R² `0.2833` -> candidate R² `0.2817`，delta `-0.0016`
- `secchi bottom`: baseline R² `0.3284` -> candidate R² `0.3219`，delta `-0.0064`
- `secchi middle`: baseline R² `0.3391` -> candidate R² `0.3213`，delta `-0.0178`
- `secchi surface`: baseline R² `0.3674` -> candidate R² `0.3584`，delta `-0.0090`
- `tn middle`: baseline R² `0.1823` -> candidate R² `0.1821`，delta `-0.0002`
- `tn surface`: baseline R² `0.5201` -> candidate R² `0.5121`，delta `-0.0080`
- `tp bottom`: baseline R² `0.0328` -> candidate R² `0.0024`，delta `-0.0303`

## 结论
- `scheme4` 作为保守分区方案，优点是解释性强、样本不碎。
- 但它本质上是对现有 `water_zone_en` 的宏观重表达，新增信息强度可能有限。
- 本轮不做 promote；是否值得继续保留为组合特征，取决于本次统一 A/B 的实际增益结构。
