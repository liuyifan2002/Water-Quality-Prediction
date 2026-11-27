# 精度突破阶段计划（四指标统一版）

## 1. 研究总目标

本项目当前阶段的总目标，不再是单独追求某一个局部任务的突破，而是：

- 在**香港近岸全站点**范围内，
- 同时提升 **Chl-a、Secchi/透明度、TN、TP** 四项指标的预测精度，
- 并尽可能把各指标、各水层的 **Test R² 推进到 0.60 以上**。

新增数据源、特征工程和后续 A/B 的评价口径，从本文件起统一改为：

1. 先看四指标整体是否有普遍增益；
2. 再看哪些指标/水层受益最大；
3. 最后才看某个专项难点是否被突破。

这意味着：

- `Chl-a bottom` 仍然是重要难点，但它是**专项瓶颈**，不是当前唯一主线；
- 后续数据源优先级，必须优先考虑对 **Chl-a + Secchi + TN + TP** 具有共同解释力的输入；
- 只有当某个数据源对四指标整体没有普遍价值、但对个别关键任务有显著增益时，才降级为专项支线。

## 2. 当前基线状态

截至当前 `current/`，已有稳定 scorecard 的任务主要仍是光学相关两项：

| 指标 | 水层 | 当前 best R² | 当前状态 |
|---|---|---:|---|
| Chl-a | surface | 0.3298 | 已纳入 current best |
| Chl-a | middle | 0.3646 | 已纳入 current best |
| Chl-a | bottom | 0.3436 | 已纳入 current best |
| Secchi | surface | 0.4080 | 已纳入 current best |
| Secchi | middle | 0.3482 | 已纳入 current best |
| Secchi | bottom | 0.3574 | 已纳入 current best |

与 `R² = 0.60` 的差距仍然明显：

- Chl-a surface：差 `0.2702`
- Chl-a middle：差 `0.2354`
- Chl-a bottom：差 `0.2564`
- Secchi surface：差 `0.1920`
- Secchi middle：差 `0.2518`
- Secchi bottom：差 `0.2426`

对 **TN / TP**，当前 `current/` 还没有形成与上述完全同口径的统一 scorecard。因此从现在开始：

- TN / TP 不再被当作“以后再说”的附属项；
- 后续任何新数据源的 A/B 模板都必须**同时预留 TN / TP**；
- 一旦 TN / TP 的统一 baseline 补齐，就直接纳入同一套四指标评价表。

目前这一步已经从“计划”推进为“部分落地”：

- `TN / TP baseline bootstrap` 已完成；
- 对应结果见 `data/processed/current/tn_tp_baseline_bootstrap_scorecard.csv`；
- 四指标统一现状卡已生成：`data/processed/current/four_target_bootstrap_scorecard.csv`。

其中当前最接近 `R² = 0.60` 的新增任务是：

- `TN surface`：test R² = `0.5201`
- `TP surface`：test R² = `0.3986`

## 3. 当前研究判断

### 3.1 已经确认的事实

1. 继续做普通调参、普通 Optuna、普通 stacking、普通弱特征堆叠，已经不是主要突破路线。
2. NASA OceanColor 月尺度单源 A/B 已经证明：作为**表层月尺度光学代理**，它对当前主线任务整体没有稳定正增益。
3. Copernicus physics 对部分任务有小幅帮助，但还不足以支撑整体跃迁。
4. `Chl-a bottom` 的问题依旧突出，但它更像是“真实底层机制信息缺失”，而不是单纯模型容量不足。
5. 如果研究目标是四指标整体提升，那么后续新增数据源必须优先看：
   - 是否对 **TN / TP** 也有解释力；
   - 是否能同时帮助光学指标与营养盐指标；
   - 是否具备稳定、可复现、可部署的站点-日期对齐方式。
6. `TN / TP baseline bootstrap` 已证明：
   - TN surface 已具备较强 baseline，可作为后续冲 `0.60` 的优先任务；
   - TP surface 有一定基础，但 validation/test 落差较大；
   - TN / TP 的中下层当前仍明显受限于更强机制输入不足。

### 3.2 纠偏后的主线

从这一版计划开始，项目的主线重新收敛为：

> 用更强、更可解释、对四指标普遍有用的输入数据，逐轮做 source validation 和统一 A/B，逐步逼近 `R² > 0.60`。

而不是：

> 围绕单个局部任务反复试弱代理或普通模型技巧。

## 4. 后续 A/B 的统一评价口径

从现在开始，所有新增数据源、特征组和实验方案，都按四指标统一口径评价：

### 4.1 必跑任务

1. Chl-a surface / middle / bottom
2. Secchi surface / middle / bottom
3. TN surface / middle / bottom
4. TP surface / middle / bottom

如果某个指标暂时没有可用 baseline，也必须在报告里明确写：

- 缺什么；
- 为什么缺；
- 这一轮为什么还没有纳入正式统一比较。

### 4.2 必看指标

每个 A/B 至少输出：

- `R²`
- `RMSE`
- `MAE`
- 高值样本表现（如适用）
- 分站点表现
- 是否存在系统负迁移

### 4.3 promote 原则

只有在以下条件同时满足时，才允许进入 current best 候选：

1. 至少一个核心任务 `delta R² > 0`
2. `RMSE` 或 `MAE` 至少一个同步改善
3. 无数据泄漏
4. 可复现
5. 输入在部署场景中可获得
6. 结论不是靠删难样本得到的

## 5. 新增数据源优先级（四指标统一版）

### 第一优先级：对四指标都可能有普遍帮助的通用机制数据

1. **bathymetry / station depth / distance to shore / zone**
   - 对 Chl-a、Secchi、TN、TP 都可能有共同解释力；
   - 能表达近岸/外海、深浅、交换强弱和空间分区差异；
   - 这是当前最值得优先补的通用静态底盘。

2. **tide / water level / hydrodynamic exchange proxy**
   - 对近岸交换、悬浮物、透明度、营养盐输运都有意义；
   - 即使暂时只有 `date-level proxy`，也比纯模型调参更值得。

3. **rainfall / wind / sea temperature / meteorology**
   - 对 TN / TP 的输入和扰动过程、对 Chl-a 的混合与气候背景、对 Secchi 的浑浊过程都可能有帮助；
   - 但要明确区分真实观测与再分析代理。

### 第二优先级：更偏光学类指标的外部数据

4. **high-quality remote sensing patch**
   - 主要服务 Chl-a 和 Secchi；
   - 对 TN / TP 的帮助通常更弱；
   - 不能再把这类数据误当成四指标通杀源。

5. **OceanColor / KD490 / RRS / turbidity proxy**
   - 继续保留为光学增强支线；
   - 但必须承认：如果目标是四指标整体提升，它们只是部分拼图，不是唯一主线。

### 第三优先级：营养盐更敏感的输入与人类活动代理

6. **river / runoff / discharge / inflow-outflow**
   - 对 TN / TP 尤其关键；
   - 这类数据如果能拿到，往往比继续加表层遥感更有希望提升四指标整体表现。

7. **land-use / impervious / plume / human activity proxy**
   - 更适合解释 TN / TP 和部分 Chl-a 异常；
   - 仅在来源清楚、口径稳定时纳入。

## 6. 当前阶段最值得做的事情

按新的总目标，下一阶段不再默认“先盯住 Chl-a bottom”。新的顺序应该是：

1. 补强对四指标都可能有帮助的通用数据底盘；
2. 用统一 A/B 模板评估四指标整体是否受益；
3. 只有在通用数据不能解释时，再把 `Chl-a bottom` 作为专项瓶颈单独攻关。

## 7. 当前阶段不应继续重复的事情

1. 不再只围绕 Chl-a bottom 组织全部路线。
2. 不再把对光学指标有效的数据，默认视为对 TN / TP 也有效。
3. 不再重复普通调参、普通 stacking、普通弱代理扩写。
4. 不再用单个局部任务的改善，替代四指标整体结论。

## 8. 本轮改写后的阶段目标

从这一版开始，项目的阶段目标正式改写为：

> 用统一的四指标评价框架，逐轮验证新增输入数据是否能同时提升香港近岸 Chl-a、Secchi、TN、TP 的预测精度，并尽可能把各指标、各水层 Test R² 推进到 0.60 以上。

## GEBCO ? Coastal-distance ????

- `GEBCO`??????????????? THREDDS / OPeNDAP ?????????????????? A/B?
- `coastal_distance_only`???? source validation ???????? A/B?
- `TN surface`?`0.5201 -> 0.5245`?`delta R? = +0.0043`????? useful ??????????
- `station_depth_plus_coastal_distance`?? `TN/TP` ??????????????????
- ????????`GEBCO ???????`?`zone refinement`?`distance-to-river-mouth`?`runoff/discharge/plume proxy`?
