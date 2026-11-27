# 项目当前入口

## 当前目标

项目已进入“强输入数据驱动的精度突破阶段”。当前主策略不是继续调参，而是优先补充更强、更加接近水质形成机制的输入数据，并对每个新增数据源执行 source validation 和同口径 A/B。

## 当前最佳主线

| 指标 | 水层 | 当前正式主线 R² |
| --- | --- | --- |
| Chl-a | surface | 0.3298 |
| Chl-a | middle | 0.3646 |
| Chl-a | bottom | 0.3318 |
| Secchi/ZSD | surface | 0.4080 |
| Secchi/ZSD | middle | 0.3447 |
| Secchi/ZSD | bottom | 0.3574 |

说明：
- 以上仍是 `current_best_scorecard.csv` 保留的正式主线。
- 本轮强输入阶段没有触发 `mainline_update`，因此 current best 暂不刷新。

## 本轮强输入阶段结论

- 已确认本地可直接处理的强输入主要有：Sentinel-2 / Landsat 高分 patch、已有 bottom mechanism proxy 表、station meta、ERA5 / rainfall / tide 原始环境表。
- 高分 patch 已完成 source validation，但相对当前最佳主线没有形成实质增益。
- Chl-a bottom 的 mechanism 表当前仍以弱代理为主，只带来 `+0.0027` 的小幅增益，不足以证明底层机制瓶颈被补齐。
- 连续两轮强输入尝试的最优 `delta R² < 0.01`，按停止规则暂停自动迭代。
- 当前最缺的仍是：`station_depth / bathymetry_depth / bottom_temp / vertical_temperature_profile / exact_sampling_datetime / discharge / water_level / current_speed`。

## 当前最重要的文件

- 结果总览：[experiment_report.md](/C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/experiments/reports/current/experiment_report.md)
- 强输入行动计划：[strong_input_data_action_plan.md](/C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/experiments/reports/current/strong_input_data_action_plan.md)
- R² 0.60 进度日志：[r2_0p6_breakthrough_log.md](/C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/experiments/reports/current/r2_0p6_breakthrough_log.md)
- 输入对齐说明：[model_input_alignment_documentation.md](/C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/experiments/reports/current/model_input_alignment_documentation.md)
- 模型与特征清单：[model_parameter_and_feature_inventory.md](/C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/experiments/reports/current/model_parameter_and_feature_inventory.md)
- 数据缺口：[data_gap_inventory.md](/C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/experiments/reports/current/data_gap_inventory.md)
- 预测上限判断：[prediction_ceiling_analysis.md](/C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/experiments/reports/current/prediction_ceiling_analysis.md)
- 项目整理状态：[project_structure_health_report.md](/C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/experiments/reports/current/project_structure_health_report.md)
