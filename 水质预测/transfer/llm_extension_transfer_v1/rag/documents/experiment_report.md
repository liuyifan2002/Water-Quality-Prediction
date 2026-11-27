# 当前实验总说明

## 1. 当前项目主目标

当前项目的正式主目标已经重新统一为：

> 在香港近岸全站点范围内，同时提升 **Chl-a、Secchi/透明度、TN、TP** 四项水质指标的预测精度，并尽可能把各指标、各水层的 `Test R²` 推进到 `0.60` 以上。

这意味着：

- 任何新增数据源、特征工程或模型补充，都必须优先看它对四指标整体是否有帮助；
- `Chl-a bottom` 仍然保留为专项难点，但不再作为整个项目唯一主线；
- 不再用单个局部任务的提升，替代四指标整体结论。

## 2. 当前已完成的主线工作

当前 `current/` 已完成并沉淀的内容包括：

1. `Copernicus Marine physics` 特征构建与 A/B；
2. 高分 patch 与质量筛选 patch 特征构建与 A/B；
3. `NASA OceanColor` 月尺度全时段特征构建与单源 A/B；
4. `Chl-a bottom` 残差诊断；
5. 强输入数据能力盘点、source validation 与 promote 审计；
6. current 目录的 canonical 报告、索引、治理与清理。

## 3. 当前最重要的阶段判断

### 3.1 已经确认有效的事情

- `enhanced_tide_proxy` 对部分现有任务有小幅稳定帮助；
- 部分光学/物理代理对局部任务存在弱正信号；
- current best 已经从完全依赖普通调参，转向“以输入数据质量和机制解释为主”的推进方式。

### 3.2 已经确认无效或低收益的事情

- 普通调参、普通 Optuna、普通 stacking；
- 只靠表层月尺度 OceanColor 扩写；
- 只靠弱代理不断加特征；
- 只围绕单个局部任务循环试错。

### 3.3 当前必须正视的现实

1. 当前四指标整体精度仍明显低于 `R² = 0.60`；
2. 公开免费数据里，真正能跨四指标提供强增益的输入仍然不够多；
3. 如果 TN / TP 也要一起提升，后续优先级必须转向：
   - 地形/空间结构
   - 交换/潮汐/水动力
   - 径流/输入负荷/河口过程
4. 如果没有更强的机制数据，继续小修小补模型很难形成本质突破。

## 4. 当前主线指标状态

当前已有统一 current best 的任务主要是：

- Chl-a：surface / middle / bottom
- Secchi：surface / middle / bottom

TN / TP 仍需补齐到与上述同口径的统一 baseline / A/B 框架中。

因此，从现在开始：

- TN / TP 必须进入后续统一 A/B 模板；
- 后续阶段报告必须明确写出 TN / TP 是否已纳入、未纳入原因以及下一步计划；
- 不允许再默认只围绕 Chl-a 和 Secchi 写“总体提升”结论。

补充进展：

- 本轮已完成 `TN / TP baseline bootstrap`；
- 当前四指标统一现状卡见：
  - `data/processed/current/tn_tp_baseline_bootstrap_scorecard.csv`
  - `data/processed/current/four_target_bootstrap_scorecard.csv`
- 当前最接近 `R² = 0.60` 的新增任务不再是 Chl-a，而是 `TN surface`，其 baseline bootstrap 的 test R² 已达到 `0.5201`。

## 5. 当前阶段下一步总方向

新的正式顺序是：

1. 优先补对四指标都可能有共同帮助的通用输入；
2. 用四指标统一模板做 source validation 和 A/B；
3. 只有在通用输入不能解释时，再单独把某个专项难点拉出来深挖。

按当前结果看，下一步最值得优先拉升精度的具体任务顺序应调整为：

1. `TN surface`
2. `Secchi surface`
3. `TP surface`
4. `Chl-a middle`
5. 其后再继续看 `Chl-a bottom / TN bottom / TP middle / TP bottom`

## 6. 当前目录中的规范入口

后续阅读和维护，优先以以下文件为准：

- `experiments/reports/current/precision_improvement_stage_plan.md`
- `experiments/reports/current/ab_evaluation_template_4targets.md`
- `experiments/reports/current/next_data_acquisition_plan.md`
- `data/processed/current/current_best_scorecard_updated.csv`
- `experiments/outputs/current/precision_improvement_ab_metrics.csv`

## 7. 当前阶段的工作边界

当前不再做的事情：

1. 不把普通调参当主线；
2. 不把弱代理写成真实机制变量；
3. 不把局部提升夸大成整体突破；
4. 不把光学遥感默认视为 TN / TP 的主要解释源；
5. 不在没有统一四指标评价口径的情况下报告“总体成功”。

## GEBCO / Coastal-distance ????

????? `coastal-distance` ?????????????????? 12 ????????? A/B?

?????
- `TN surface`?`0.5201 -> 0.5245`?`delta R? = +0.0043`?RMSE ? MAE ???????? useful ???
- `TN bottom`?`0.2353 -> 0.2322`???????
- `TP surface`?`0.3986 -> 0.3946`?????
- `Chl-a middle`?`0.3646 -> 0.3672`???????????????????

??????? promote ? current best?

???`GEBCO` ???????? source validation??????????? bathymetry ?????????????
