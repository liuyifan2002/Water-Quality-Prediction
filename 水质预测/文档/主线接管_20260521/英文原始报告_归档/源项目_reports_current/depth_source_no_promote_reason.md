# 官方站点水深未 promote 原因

## 结论

本轮新增源 `epd_station_depth` **没有进入 current best**。

## 直接原因

1. `TN surface` 作为当前最接近 `R² = 0.60` 的四指标任务，本轮从 `0.5201` 降到 `0.5162`，没有向 `0.60` 推进。
2. `TN bottom` 虽然从 `0.2353` 提升到 `0.2467`，`delta R² = +0.0114`，但 validation 没有同步改善，因此不满足稳定 promote 条件。
3. `Chl-a bottom` 虽然从 `0.3436` 提升到 `0.3519`，`delta R² = +0.0083`，但本轮研究主目标不是单独追某一个专项任务，而是提升四指标整体精度；同时该任务 validation 也不支持 promote。
4. `TP surface / TP middle / TP bottom` 与 `Secchi surface / middle / bottom` 大多出现负增益。

## 解释

- `station_depth_approx_m` 能帮助模型识别深浅和空间分区差异；
- 但它本质上仍是 **静态站点总水深**，不能替代：
  - `exact_sampling_datetime`
  - `bottom temperature / vertical profile`
  - `ADCP current`
  - `water level / discharge`
- 因此，它更像是“有信息但不够强”的空间背景输入，而不是足以推动四指标整体跃迁的强机制数据。

## 下一步

更值得继续的是：

1. `GEBCO` 或其他可复现近岸 bathymetry 栅格；
2. `distance-to-shore / coastal exposure / zone refinement`；
3. `runoff / discharge / plume proxy`；
4. 若能补齐原始监测数据，则优先补：
   - `exact_sampling_datetime`
   - `bottom profile / bottom temperature`
   - `ADCP current / water level / discharge`
