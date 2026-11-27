# 根据站点分类的预测结果整理

本文档整理 `station_group_highres_patch_alignment_20260518` 这轮实验中“站点分类 / 分区建模”相关的预测结果，目标是回答三个问题：

1. 不同站点分类方式是否能整体提升模型精度；
2. 哪些目标变量/水层更适合采用 grouped model；
3. 哪些站点明显受益，哪些站点反而变差。

## 1. 数据来源

- 原始实验目录：`experiments/outputs/station_group_highres_patch_alignment_20260518/`
- 整体分组建模指标：`station_grouping_metrics.csv`
- 逐站点比较：`station_level_prediction_comparison.csv`
- 站点分组定义：`station_group_definitions.csv`
- current 汇总表：`experiments/outputs/current/station_group_prediction_summary.csv`

## 2. 本轮采用的四类站点分类方式

### 2.1 空间位置分类 `spatial`

基于站点空间位置特征聚类，核心使用：

- `lon`
- `lat`
- 若存在则附加空间区域信息

### 2.2 历史水质统计分类 `water_quality_train_only`

只使用 train 期统计，不使用 validation / test 标签，核心使用：

- train 均值
- train 标准差
- train 高分位值
- train 季节振幅

### 2.3 光学统计分类 `optical_train_only`

只使用 train 期遥感光学统计，不使用 test 标签，核心使用：

- Ocean Colour 平均状态
- Ocean Colour 波动强度
- 遥感有效观测情况

### 2.4 混合分类 `hybrid_train_only`

同时使用：

- 空间位置
- train 期水质统计
- train 期光学统计

## 3. 整体 test 表现最好的 grouped model

下表中的“整体最好”是指：在所有 grouped model 方案中，对应目标变量/水层的 **test R² 最高** 的那一个结果；它不一定优于 current 主线。

| 目标 | 水层 | 整体最优 grouped 方案 | 分组方式 | 聚类方法 | 组数 | test R² | 相对 previous best |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Chl-a | surface | `group_as_feature` | `optical_train_only` | `kmeans` | 4 | 0.3183 | -0.0115 |
| Chl-a | middle | `group_as_feature` | `spatial` | `kmeans` | 5 | 0.3810 | +0.0164 |
| Chl-a | bottom | `global_group_residual` | `hybrid_train_only` | `agglomerative` | 2 | 0.3192 | -0.0126 |
| Secchi/ZSD | surface | `group_as_feature` | `spatial` | `kmeans` | 5 | 0.4086 | +0.0006 |
| Secchi/ZSD | middle | `global_group_residual` | `water_quality_train_only` | `agglomerative` | 2 | 0.2756 | -0.0691 |
| Secchi/ZSD | bottom | `global_group_residual` | `water_quality_train_only` | `agglomerative` | 2 | 0.3294 | -0.0280 |

### 3.1 整体结论

- **只有 Chl-a middle** 出现了比较明确的整体正增益，`R² +0.0164`，但仍未达到“整体提升 ≥ 0.03”的主线替代标准。
- **Secchi/ZSD surface** 基本持平，只有 `+0.0006`，属于非常轻微的改善。
- 其余 4 个任务整体上都不如 current 主线。

因此，从“整体 pooled test R²”角度看，站点分类模型**目前不适合替代 current 全局主线**。

## 4. 逐站点改善比例最高的分类方式

下表中的“最佳分类方式”不是看整体 pooled R²，而是看 **逐站点 improved 比例最高** 的方案。

| 目标 | 水层 | 最佳分类方式 | 站点数 | improved 站点数 | improved 比例 | 平均 ΔR² | 平均 ΔRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Chl-a | surface | `spatial` | 94 | 36 | 38.30% | -0.1273 | +0.1094 |
| Chl-a | middle | `hybrid_train_only` | 79 | 49 | 62.03% | +0.0163 | -0.0454 |
| Chl-a | bottom | `optical_train_only` | 90 | 72 | 80.00% | +0.0538 | -0.0126 |
| Secchi/ZSD | surface | `spatial` | 94 | 54 | 57.45% | +0.0653 | -0.0317 |
| Secchi/ZSD | middle | `spatial` | 79 | 38 | 48.10% | -0.0398 | +0.0132 |
| Secchi/ZSD | bottom | `hybrid_train_only` | 90 | 63 | 70.00% | +0.0947 | -0.0335 |

### 4.1 逐站点视角下的结论

- **Chl-a bottom**：`optical_train_only` 虽然整体 pooled R² 没有刷新，但有 **80% 站点 improved**，说明这条路线更像 `station_specific_auxiliary`，适合做站点级辅助。
- **Secchi/ZSD bottom**：`hybrid_train_only` 也有 **70% 站点 improved**，同样更适合作为逐站点辅助线。
- **Chl-a middle**：`hybrid_train_only` 同时具备一定整体增益和 62.03% 的 improved 比例，是目前最值得保留观察的 grouped 路线。
- **Chl-a surface** 和 **Secchi/ZSD middle** 的分组收益不稳定，不建议继续作为重点路线。

## 5. 明显改善的代表站点

以下站点是在“最佳分类方式”下，改善较明显的代表例子。

### 5.1 Chl-a bottom：`optical_train_only`

改善较明显的站点：

- `PT3`：`ΔR² = +0.9782`
- `TM8`：`ΔR² = +0.4893`
- `VM4`：`ΔR² = +0.4096`
- `SM19`：`ΔR² = +0.4006`
- `VM5`：`ΔR² = +0.3187`

变差较明显的站点：

- `VM12`：`ΔR² = -0.4881`
- `PM9`：`ΔR² = -0.2590`
- `PM7`：`ΔR² = -0.2306`

### 5.2 Chl-a middle：`hybrid_train_only`

改善较明显的站点：

- `MM1`：`ΔR² = +0.4379`
- `WM1`：`ΔR² = +0.3143`
- `EM1`：`ΔR² = +0.2859`
- `DM5`：`ΔR² = +0.2809`
- `WT3`：`ΔR² = +0.2404`

### 5.3 Secchi/ZSD surface：`spatial`

改善较明显的站点：

- `PT4`：`ΔR² = +1.7765`
- `PM6`：`ΔR² = +1.5963`
- `EM2`：`ΔR² = +0.9396`
- `EM1`：`ΔR² = +0.5541`
- `SM13`：`ΔR² = +0.5031`

### 5.4 Secchi/ZSD bottom：`hybrid_train_only`

改善较明显的站点：

- `PM6`：`ΔR² = +2.0691`
- `PT4`：`ΔR² = +1.2014`
- `NM8`：`ΔR² = +0.5025`
- `PM11`：`ΔR² = +0.3143`
- `WT1`：`ΔR² = +0.2756`

## 6. 推荐解释口径

### 6.1 当前不建议写成“站点分类模型整体优于全局模型”

因为：

- 整体 pooled test R² 只有 `Chl-a middle` 出现了相对明确的正增益；
- 其余任务大多没有整体提升，甚至下降；
- 因此 current 正式主线仍应保留全局模型。

### 6.2 更合理的表述

当前更合理的结论是：

- 站点分类模型对**部分站点**确有帮助；
- 它在 `Chl-a bottom` 和 `Secchi/ZSD bottom` 上表现出较强的站点异质性修正潜力；
- 但现阶段更适合作为 **station-specific auxiliary**，而不是统一替代全局主模型。

## 7. 当前推荐

### 7.1 保留为 current 主线的任务

- Chl-a surface：保留全局主线
- Chl-a middle：保留全局主线，但可记录 grouped model 为观察候选
- Chl-a bottom：保留全局主线，分组模型作为 `station_specific_auxiliary`
- Secchi/ZSD surface：保留全局主线
- Secchi/ZSD middle：保留全局主线
- Secchi/ZSD bottom：保留全局主线，分组模型作为 `station_specific_auxiliary`

### 7.2 当前最值得继续跟踪的 grouped 路线

1. `Chl-a middle + spatial group_as_feature`
2. `Chl-a bottom + optical_train_only`
3. `Secchi/ZSD bottom + hybrid_train_only`

## 8. 对论文写作的直接用法

如果要在论文里写这部分结果，建议用以下口径：

- “站点分类建模在整体精度上未稳定优于全站点统一建模，但在若干目标变量和站点上表现出明显的局地改善潜力。”
- “其中，Chl-a middle 在空间分组下获得了小幅整体增益，而 Chl-a bottom 与 Secchi/ZSD bottom 在站点层面表现出更高比例的改善站点，说明站点异质性对底层水质预测具有重要影响。”
- “因此，站点分类模型更适合作为辅助建模框架，而非当前阶段的统一主模型。”
