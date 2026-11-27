# Chl-a 精度提升新增数据源方案

## 1. 当前为什么暂停历史其他水质协变量

`lagged_cross_water_quality` 单源 A/B 已经做完，结论比较明确：

1. 它没有提升当前 Chl-a 最强候选主线
2. `surface / middle / bottom` 的总体 R2 都下降
3. P90 高值样本 RMSE 和低估比例也没有改善

因此，历史其他水质变量这条线暂时停止，不进入 Chl-a 增强主线。

## 2. 为什么当前优先 Ocean Colour

现阶段更值得补的是“外部海洋水色信息”，而不是继续堆历史水质协变量。原因主要有三点：

1. 目前最稳模型已经充分利用了 Chl-a 自身历史、环境驱动和 Landsat/S2 遥感
2. 高值样本和爆发态样本仍然偏低估，说明缺少更直接的藻华/水色过程表征
3. Ocean Colour 产品可以直接补充 `CHL / KD490 / SPM / ZSD / RRS`，更贴近藻类、水体透明度和悬浮颗粒物过程

## 3. 为什么 GEE 的 OLCI TOA radiance 不是第一优先正式输入

本阶段不把 GEE 中偏 TOA radiance 的 `COPERNICUS/S3/OLCI` 作为主入口，主要是因为：

1. 它更接近原始观测层，不是已经完成海色反演的标准 Chl/Kd490/SPM 产品
2. 需要额外大气校正和产品级解释，正式主链不够简洁
3. 对近岸复杂水体，直接使用标准 Ocean Colour L4 产品更利于稳定比较和后续 A/B

它仍然可以作为辅助探索源，但不是本阶段正式优先级。

## 4. 推荐新增数据源

### 4.1 Copernicus Marine Ocean Colour

优先级最高。目标变量包括：

1. `CHL`
2. `KD490`
3. `SPM`
4. `ZSD`
5. `RRS400 / 412 / 443 / 490 / 510 / 560 / 620 / 665 / 681 / 709`
6. `PP`
7. `PFT`

预期作用：

1. 更直接补充藻类背景状态
2. 表达光场限制
3. 反映浑浊与悬浮扰动
4. 提供红边响应以增强高值识别

### 4.2 Copernicus Marine Physics

优先级第二。重点变量包括：

1. `SST`
2. `SSS`
3. `MLD`
4. `SSH`
5. `uo / vo / current_speed`
6. `bottom_temp`

预期作用：

1. 表达低盐径流输入
2. 表达混合层深浅
3. 表达海流输运和滞留
4. 帮助解释 middle/bottom 的结构差异

### 4.3 Copernicus Marine BGC

优先级第三。重点变量包括：

1. `chl`
2. `no3`
3. `po4`
4. `si`
5. `o2`
6. `npp`
7. `phyto`
8. `zoo`

预期作用：

1. 直接补营养盐背景
2. 补生态生产力状态
3. 改善藻华高值样本解释

但要注意：它属于再分析/模式背景代理，正式论文里要单独说明这一属性。

### 4.4 EPD 长历史水质

优先级第四。当前还没打通官方长历史链接，但一旦补齐，可以提供：

1. `station-month` 长期气候态
2. `zone-month` 高值风险背景
3. 更长时间尺度的站点惯性

## 5. 物理意义概览

| 数据源 | 主要物理意义 |
|---|---|
| Ocean Colour | 直接表达藻类浓度、透明度、悬浮颗粒物和光衰减 |
| CMEMS Physics | 表达水团结构、径流淡化、混合层状态和输运过程 |
| CMEMS BGC | 表达营养盐、生物生产力和生态背景 |
| EPD 长历史 | 表达长期季节气候态和历史高值风险背景 |

## 6. 哪些适合进入正式预测，哪些更适合增强实验

### 更适合正式预测

1. Ocean Colour
2. CMEMS Physics

这两类更接近“预测时可以获得的外部环境/水色信息”。

### 更适合增强实验

1. CMEMS BGC
2. 全时段统计的 EPD 长历史气候态

原因是它们更偏模式背景或长期统计背景，解释时要更谨慎。

## 7. 空间与时间覆盖风险

### 空间分辨率风险

1. OLCI / Global Ocean Colour 常见分辨率约为 300m 到 4km
2. 对香港近岸窄湾、河口和港池，4km 产品会有明显混合像元风险
3. CMEMS Physics/BGC 网格更粗，更适合作为背景场而非局地观测替代

### 时间覆盖风险

1. Sentinel-3 OLCI 自 2016 年后覆盖更稳定，无法自然补足 2014-2015
2. 多源 L4 Ocean Colour 历史产品更适合覆盖 2014-2024 主实验
3. 月尺度产品只能表达背景状态，不适合直接解释短期突发事件

## 8. 推荐执行顺序

1. 先做 `Copernicus Marine Ocean Colour`
2. 再做 `CMEMS Physics`
3. 再做 `CMEMS BGC`
4. 最后回头补 `EPD 长历史`

顺序原因很简单：Ocean Colour 最有机会直接改善当前 Chl-a 高值识别和总体表现。

## 9. 下一步实验矩阵

| 实验 | 输入文件 | 预期作用 | 风险 | 是否优先 |
|---|---|---|---|---|
| 只加 Ocean Colour | `ocean_color_kd490_tss_features_2014_2024.csv` | 提升 Chl-a 与透明度过程表达 | 近岸混合像元、缺测 | 是 |
| 只加 OLCI 子表 | `sentinel3_olci_features_2014_2024.csv` | 提升红边与高值藻华识别 | 2014-2015 覆盖不足 | 是 |
| 只加 CMEMS Physics | `copernicus_physics_features_2014_2024.csv` | 提升盐度、混合、输运解释 | 网格较粗 | 是 |
| 只加 CMEMS BGC | `copernicus_bgc_features_2014_2024.csv` | 提升营养盐和生态背景解释 | 再分析属性较强 | 次优先 |
| Ocean Colour + Physics | 两类文件 | 结合水色和水动力 | 维度升高、缺失叠加 | 高优先 |
| Physics + BGC | 两类 CMEMS 文件 | 机制解释更完整 | 模式误差叠加 | 次优先 |
| 所有新增源综合 | 全部新文件 | 信息最全 | 贡献难分解，不适合作为首轮实验 | 否 |

## 10. 当前建议

当前最合理的推进方式不是继续修补历史协变量，而是尽快把 Ocean Colour 这条线做成真实输入，再用 `21_validate_new_chla_data_sources.py` 判定是否 `ready_for_ab_experiment`。只有当真实 Ocean Colour 文件通过校验后，才值得进入下一轮 Chl-a 单源 A/B。
