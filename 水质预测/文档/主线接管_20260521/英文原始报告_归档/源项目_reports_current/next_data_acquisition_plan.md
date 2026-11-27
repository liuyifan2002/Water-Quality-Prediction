# 下一阶段数据获取优先级（四指标统一版）

## 1. 目标重定义

下一阶段数据获取的目标，不再是优先服务某一个局部任务，而是：

> 优先补充对 **Chl-a、Secchi、TN、TP** 四项指标都可能带来普遍增益的输入数据，并尽可能推动整体预测精度向 `R² > 0.60` 靠近。

因此，后续数据源优先级按照“对四指标的共同价值”排序，而不是按某一个专项瓶颈排序。

## 2. 第一优先级：四指标通用机制底盘

### 2.1 bathymetry / station depth / distance-to-shore / zone

这是当前最值得优先补的输入组。

理由：

1. 对 Chl-a、Secchi、TN、TP 都有空间结构解释力；
2. 能表达近岸/外海、深浅、交换强弱、分区差异；
3. 对营养盐和光学指标都不是“偏科”输入。

当前建议顺序：

1. 继续尝试稳定的 `GEBCO` 子集提取；
2. 若 GEBCO 仍不能稳定服务研究区，则继续寻找可公开复现的香港近岸 bathymetry；
3. 若公开路径仍不足，最终仍需向监测单位或相关部门申请 `station_depth` 或监测站深度元数据。

## 3. 第二优先级：近岸交换与潮汐/水动力

### 3.1 tide / water level / exchange proxy

原因：

1. 对 Chl-a、Secchi、TN、TP 都可能有输运和交换意义；
2. 比单纯表层光学遥感更接近近岸过程；
3. 目前已有 `date-level tide proxy`，但还可以继续增强。

当前可做：

- HKO / data.gov.hk 的预测潮位、实时潮位、潮差、潮事件数

当前不能伪装成已拥有：

- `exact_sampling_datetime`
- `tide_level_at_sampling`
- `flood_or_ebb_tide_at_sampling`

如果没有采样具体时刻，这条线就只能继续作为 **date-level proxy**。

## 4. 第三优先级：对 TN / TP 尤其关键的输入

### 4.1 discharge / inflow / outflow / runoff proxy

如果研究目标是四指标整体提升，这一类输入比继续扩写光学遥感更值得优先。

原因：

1. TN / TP 更依赖输入负荷、河口交换和径流扰动；
2. 这类变量同时也可能影响 Chl-a 和透明度；
3. 它们比单纯光学代理更有希望带来跨指标共同增益。

当前状态：

- 本地尚无稳定、可直接统一接入的正式数据；
- 后续如果能找到公开免费水文站数据或可靠 proxy，应优先纳入统一四指标 A/B。

## 5. 第四优先级：气象与海气背景

### 5.1 rainfall / wind / sea temperature / pressure

这一类仍值得保留，但定位要更清楚：

- 它们是**通用背景驱动**；
- 对四指标都可能有帮助；
- 但单独使用时往往难以打穿精度天花板。

后续原则：

1. 可继续作为四指标统一输入组的一部分；
2. 不再把它们当作“专项突破主角”；
3. 重点看它们是否与水动力、地形、营养盐输入类数据形成组合增益。

## 6. 第五优先级：光学遥感增强

### 6.1 high-quality patch / OceanColor / KD490 / RRS

这条线继续保留，但必须降级为：

- 主要服务 **Chl-a / Secchi**
- 不是四指标整体突破的唯一主线

当前判断：

1. NASA OceanColor 月尺度单源 A/B 已经证明，作为表层月尺度光学代理，它没有带来稳定正增益；
2. 高分 patch 和质量筛选 patch 也没有形成跨指标强增益；
3. 因此后续只有在拿到更强时间对齐条件（如 `exact_sampling_datetime`）或更高质量空间匹配时，才值得继续扩展。

## 7. 暂不再优先投入的方向

以下方向当前不应再作为下一轮主攻：

1. 仅围绕 Chl-a bottom 设计整轮实验；
2. 只靠月尺度表层 OceanColor 继续扩写；
3. 只靠弱 ERA5 / rain / tide proxy 重复实验；
4. 只做普通调参、普通特征堆叠、普通 stacking。

## 8. 下一轮推荐顺序

按照四指标统一目标，下一轮最值得继续的顺序建议为：

1. **bathymetry / station depth / distance-to-shore / zone**
2. **tide / water level / nearshore exchange proxy**
3. **discharge / runoff / inflow-outflow / plume proxy**
4. **rainfall / wind / sea temperature 作为通用背景驱动**
5. **仅在明确有新对齐条件时，再继续高频率光学遥感扩展**

## 9. 仍需人工或原始监测单位支持的关键数据

如果目标真的是把四指标整体往 `R² > 0.60` 推进，目前最关键且难以由公开源完全替代的仍然是：

1. `exact_sampling_datetime`
2. `station_depth`
3. `bottom water temperature / vertical profile`
4. `ADCP current`
5. `water level / discharge / inflow / outflow`

这些数据不只是为了 Chl-a bottom，也很可能决定 TN / TP 和近岸透明度的上限。


## Coastal-distance ??????

1. **GEBCO ???????**?????? bathymetry ????????????
2. **Coastal context ??**?`zone refinement`?`distance-to-river-mouth`?`distance-to-outlet`?
3. **????????**?`runoff / discharge / plume proxy`????? `TN / TP / Secchi`?
4. **????????**?`exact_sampling_datetime`?`bottom profile / bottom temperature`?`ADCP current / water level`?
