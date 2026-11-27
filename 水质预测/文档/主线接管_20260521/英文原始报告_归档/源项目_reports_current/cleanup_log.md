# 清理日志

## 2026-05-18 current 高分 patch / mechanism 正式验证

### 本轮更新

- current_best_scorecard.csv
- highres patch 与 mechanism 的 station-level 对比文件
- current/ 下所有核心 MD
- data/external/gee_exports/ 下的高分 patch 副本

### 复制到 external 目录

- `C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/processed/current/s2_highres_patch_features_2014_2024.csv -> C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/external/gee_exports/s2_patch_features_2014_2024.csv`
- `C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/processed/current/landsat_highres_patch_features_2014_2024.csv -> C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/external/gee_exports/landsat_patch_features_2014_2024.csv`

### 清理与校验

- `__pycache__` 清理数量：154
- 根目录污染检查：通过
- 本轮没有删除历史实验结果，只更新 current canonical 文件。

## 本轮 current 高分 patch / bottom mechanism A/B

- 更新 `experiments/outputs/current/s2_patch_ab_metrics.csv`、`s2_patch_predictions.csv`、`s2_patch_feature_importance.csv`、`s2_patch_station_level_comparison.csv`。
- 更新 `experiments/outputs/current/landsat_patch_ab_metrics.csv`、`landsat_patch_predictions.csv`、`landsat_patch_feature_importance.csv`、`landsat_patch_station_level_comparison.csv`。
- 更新 `experiments/outputs/current/highres_patch_ab_metrics.csv`、`highres_patch_predictions.csv`、`highres_patch_feature_importance.csv`、`highres_patch_station_level_comparison.csv`。
- 更新 `experiments/outputs/current/chla_bottom_mechanism_ab_metrics.csv` 及其逐站点比较输出。
- 更新 `experiments/outputs/current/station_level_prediction_comparison.csv`、`station_level_predictions.csv`。
- 按最新规则重写 `current_best_scorecard.csv`，避免把小幅 uplift 直接写成主线刷新。

## 本轮 current 汇总与文档更新

- 更新 S2-only、Landsat-only、S2+Landsat 三组 highres patch A/B 输出。
- 更新 Chl-a bottom mechanism A/B 输出。
- 更新 current_best_scorecard、逐站点比较、模型参数与输入特征清单、数据对齐说明。
- 复制 highres patch 正式副本到 `data/external/gee_exports/`。
- C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/processed/current/s2_highres_patch_features_2014_2024.csv -> C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/external/gee_exports/s2_patch_features_2014_2024.csv
- C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/processed/current/landsat_highres_patch_features_2014_2024.csv -> C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/external/gee_exports/landsat_patch_features_2014_2024.csv
- `__pycache__` 清理数量：152
- 根目录污染检查：通过

## 2026-05-18 current 文档澄清补充

- 重新补写 `model_input_alignment_documentation.md`，明确 ERA5、风、降雨、潮汐 proxy 的原始输入字段、`date` 合并键、派生公式与解释边界。
- 在 `model_parameter_and_feature_inventory.md` 顶部新增“机制变量输入与处理补充说明”，避免把弱代理变量误写成真实底层机制变量。
- 重写 `bottom_mechanism_data_status.md`，逐条列出当前已进入的弱代理变量与仍缺失的强机制变量。

## 2026-05-19 站点分类结果整理

- 新增 `experiments/reports/current/station_group_prediction_summary.md`，汇总站点分类建模的整体结果与逐站点改善情况。
- 新增 `experiments/outputs/current/station_group_prediction_summary.csv`，压缩输出“整体最优 grouped 方案”和“逐站点 improved 比例最高方案”的摘要结果。
- 更新 `experiment_master_index.md`，加入站点分类结果摘要文件入口。

## 本轮 current 汇总与文档更新

- 更新 S2-only、Landsat-only、S2+Landsat 三组 highres patch A/B 输出。
- 更新 Chl-a bottom mechanism A/B 输出。
- 更新 current_best_scorecard、逐站点比较、模型参数与输入特征清单、数据对齐说明。
- 复制 highres patch 正式副本到 `data/external/gee_exports/`。
- C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/processed/current/s2_highres_patch_features_2014_2024.csv -> C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/external/gee_exports/s2_patch_features_2014_2024.csv
- C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/processed/current/landsat_highres_patch_features_2014_2024.csv -> C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/external/gee_exports/landsat_patch_features_2014_2024.csv
- `__pycache__` 清理数量：152
- 根目录污染检查：通过

## 本轮 current 汇总与文档更新

- 更新 S2-only、Landsat-only、S2+Landsat 三组 highres patch A/B 输出。
- 更新 Chl-a bottom mechanism A/B 输出。
- 更新 current_best_scorecard、逐站点比较、模型参数与输入特征清单、数据对齐说明。
- 复制 highres patch 正式副本到 `data/external/gee_exports/`。
- C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/processed/current/s2_highres_patch_features_2014_2024.csv -> C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/external/gee_exports/s2_patch_features_2014_2024.csv
- C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/processed/current/landsat_highres_patch_features_2014_2024.csv -> C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/external/gee_exports/landsat_patch_features_2014_2024.csv
- `__pycache__` 清理数量：152
- 根目录污染检查：通过

## 2026-05-19 特征瓶颈诊断与最小有效特征集

- 新增 `scripts/evaluation/run_current_feature_bottleneck_diagnostics.py`。
- 更新 `feature_stability_analysis.csv`、`task_specific_feature_sets.json`、`feature_set_ab_metrics.csv`、误差分解表、站点推荐表和精度上限分析文档。
- 本轮未新增数据源，未覆盖历史实验目录，输出集中在 current/。

## Project Organizer Agent 运行记录：2026-05-19 07:52:03

- 运行模式：`dry-run`
- 扫描文件数：11956
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：1
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-19 07:52:44

- 运行模式：`apply-safe`
- 扫描文件数：11966
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：1 个

### 清理的缓存

- `scripts/maintenance/__pycache__`：1 个文件，0.0476 MB

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-19 12:43:11

- 运行模式：`dry-run`
- 扫描文件数：11981
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：1
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-19 12:43:56

- 运行模式：`apply-safe`
- 扫描文件数：11980
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：1 个

### 清理的缓存

- `scripts/data/__pycache__`：1 个文件，0.0383 MB

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-19 12:44:58

- 运行模式：`dry-run`
- 扫描文件数：11980
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-19 12:45:38

- 运行模式：`apply-safe`
- 扫描文件数：11980
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## 2026-05-19 strong mechanism stage
- 新增/更新 auto_data_acquisition_capability.csv
- 新增/更新 depth_bathymetry_features_2014_2024.csv
- 新增/更新 copernicus_physics_features_2014_2024.csv
- 新增/更新 tide_phase_features_2014_2024.csv
- 新增/更新 high_quality_patch_features_2014_2024.csv
- 新增/更新 strong_mechanism_input_features_2014_2024.csv

## 2026-05-19 strong mechanism stage
- 新增/更新 auto_data_acquisition_capability.csv
- 新增/更新 depth_bathymetry_features_2014_2024.csv
- 新增/更新 copernicus_physics_features_2014_2024.csv
- 新增/更新 tide_phase_features_2014_2024.csv
- 新增/更新 high_quality_patch_features_2014_2024.csv
- 新增/更新 strong_mechanism_input_features_2014_2024.csv

## 2026-05-20 strong mechanism current sync
- ?? auto_data_acquisition_capability.csv??? station_depth / current_discharge_water_level ????
- ?? Copernicus Marine physics ?? 2014-2024 ??? station-date ???
- ???? strong_mechanism A/B ??? current ???????

## Project Organizer Agent 运行记录：2026-05-20 11:26:03

- 运行模式：`dry-run`
- 扫描文件数：12419
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：71
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-20 11:26:49

- 运行模式：`apply-safe`
- 扫描文件数：12003
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：71 个

### 清理的缓存

- `scripts/__pycache__`：2 个文件，0.0297 MB
- `scripts/data/__pycache__`：1 个文件，0.0628 MB
- `scripts/maintenance/__pycache__`：1 个文件，0.0476 MB
- `scripts/training/__pycache__`：3 个文件，0.1327 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/__pycache__`：1 个文件，0.0445 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/_distutils_hack/__pycache__`：1 个文件，0.0115 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/catboost/__pycache__`：5 个文件，0.4058 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/catboost/widget/__pycache__`：2 个文件，0.0090 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/__pycache__`：4 个文件，0.0290 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/parser/__pycache__`：3 个文件，0.0862 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/tz/__pycache__`：5 个文件，0.1053 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/__pycache__`：17 个文件，0.0926 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/backend/__pycache__`：9 个文件，0.0484 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/parameters/__pycache__`：7 个文件，0.0166 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/lightgbm/__pycache__`：9 个文件，0.5070 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/__pycache__`：10 个文件，0.0764 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_core/__pycache__`：27 个文件，0.9264 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_typing/__pycache__`：10 个文件，0.0389 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_utils/__pycache__`：3 个文件，0.0150 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/lib/__pycache__`：28 个文件，0.9009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/linalg/__pycache__`：3 个文件，0.1326 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/ma/__pycache__`：3 个文件，0.4014 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/matrixlib/__pycache__`：2 个文件，0.0366 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/random/__pycache__`：2 个文件，0.0106 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/__pycache__`：4 个文件，0.0265 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_config/__pycache__`：5 个文件，0.0476 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/tslibs/__pycache__`：1 个文件，0.0023 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/window/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_testing/__pycache__`：6 个文件，0.0979 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/__pycache__`：1 个文件，0.0005 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/extensions/__pycache__`：1 个文件，0.0009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/indexers/__pycache__`：1 个文件，0.0006 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/interchange/__pycache__`：1 个文件，0.0005 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/types/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/typing/__pycache__`：1 个文件，0.0014 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/arrays/__pycache__`：1 个文件，0.0016 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/compat/__pycache__`：6 个文件，0.0253 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/compat/numpy/__pycache__`：2 个文件，0.0162 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/__pycache__`：22 个文件，1.8349 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/_numba/__pycache__`：2 个文件，0.0096 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/array_algos/__pycache__`：9 个文件，0.0532 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/__pycache__`：20 个文件，0.7842 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/arrow/__pycache__`：3 个文件，0.1616 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/sparse/__pycache__`：3 个文件，0.0896 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/computation/__pycache__`：13 个文件，0.1559 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/dtypes/__pycache__`：11 个文件，0.3049 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/groupby/__pycache__`：9 个文件，0.4216 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/indexers/__pycache__`：3 个文件，0.0357 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/indexes/__pycache__`：14 个文件，0.7186 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/interchange/__pycache__`：4 个文件，0.0443 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/internals/__pycache__`：9 个文件，0.3463 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/methods/__pycache__`：3 个文件，0.0262 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/ops/__pycache__`：8 个文件，0.0563 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/reshape/__pycache__`：9 个文件，0.2385 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/strings/__pycache__`：4 个文件，0.1680 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/tools/__pycache__`：5 个文件，0.0735 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/util/__pycache__`：3 个文件，0.0162 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/window/__pycache__`：8 个文件，0.1977 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/errors/__pycache__`：1 个文件，0.0318 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/__pycache__`：16 个文件，0.7004 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/excel/__pycache__`：10 个文件，0.1495 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/formats/__pycache__`：5 个文件，0.1665 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/json/__pycache__`：4 个文件，0.0897 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/parsers/__pycache__`：6 个文件，0.2195 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/sas/__pycache__`：2 个文件，0.0065 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/plotting/__pycache__`：3 个文件，0.0910 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/tseries/__pycache__`：4 个文件，0.0255 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/util/__pycache__`：6 个文件，0.0502 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/util/version/__pycache__`：1 个文件，0.0238 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pytz/__pycache__`：5 个文件，0.0678 MB

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## 2026-05-20 精度突破阶段更新

- 新增本轮计划、残差诊断、免费数据源验证、A/B 对比、updated scorecard 和阶段报告；随后会运行 Project Organizer Agent 做 dry-run 与 apply-safe。

## Project Organizer Agent 运行记录：2026-05-20 16:52:53

- 运行模式：`dry-run`
- 扫描文件数：12079
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：9
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-20 16:53:27

- 运行模式：`apply-safe`
- 扫描文件数：12024
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：9 个

### 清理的缓存

- `scripts/__pycache__`：2 个文件，0.0291 MB
- `scripts/data/__pycache__`：1 个文件，0.0907 MB
- `scripts/training/__pycache__`：3 个文件，0.1327 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/catboost/__pycache__`：5 个文件，0.4058 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/catboost/widget/__pycache__`：2 个文件，0.0090 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/__pycache__`：17 个文件，0.0926 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/backend/__pycache__`：9 个文件，0.0484 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/parameters/__pycache__`：7 个文件，0.0166 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/lightgbm/__pycache__`：9 个文件，0.5070 MB

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-20 16:58:12

- 运行模式：`dry-run`
- 扫描文件数：12027
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：2
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-20 16:58:50

- 运行模式：`apply-safe`
- 扫描文件数：12025
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：2 个

### 清理的缓存

- `scripts/__pycache__`：1 个文件，0.0136 MB
- `scripts/data/__pycache__`：1 个文件，0.0907 MB

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## 2026-05-20 NASA / bathymetry 跟进

- 新增本地未追踪 `.env` 中的 NASA Earthdata 凭据；未写入任何追踪文件、日志、CSV、JSON 或 Markdown 正文。

## Project Organizer Agent 运行记录：2026-05-20 21:36:28

- 运行模式：`dry-run`
- 扫描文件数：16731
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：152
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-20 21:37:18

- 运行模式：`apply-safe`
- 扫描文件数：15492
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：152 个

### 清理的缓存

- `scripts/__pycache__`：1 个文件，0.0155 MB
- `scripts/data/__pycache__`：1 个文件，0.0325 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/__pycache__`：3 个文件，0.2219 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/_distutils_hack/__pycache__`：2 个文件，0.0111 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiobotocore/__pycache__`：30 个文件，0.2995 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiobotocore/retries/__pycache__`：5 个文件，0.0179 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiohappyeyeballs/__pycache__`：5 个文件，0.0248 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiohttp/__pycache__`：48 个文件，1.0949 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiohttp/_websocket/__pycache__`：7 个文件，0.0580 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aioitertools/__pycache__`：8 个文件，0.0578 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aioitertools/tests/__pycache__`：7 个文件，0.1239 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiosignal/__pycache__`：1 个文件，0.0030 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/attr/__pycache__`：13 个文件，0.2139 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/attrs/__pycache__`：6 个文件，0.0028 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/__pycache__`：42 个文件，1.2947 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/crt/__pycache__`：2 个文件，0.0223 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/docs/__pycache__`：13 个文件，0.1208 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/docs/bcdoc/__pycache__`：4 个文件，0.0576 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/retries/__pycache__`：8 个文件，0.0431 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/vendored/__pycache__`：2 个文件，0.0446 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/vendored/requests/__pycache__`：2 个文件，0.0061 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/vendored/requests/packages/__pycache__`：1 个文件，0.0004 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/vendored/requests/packages/urllib3/__pycache__`：2 个文件，0.0093 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/bounded_pool_executor/__pycache__`：1 个文件，0.0028 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/certifi/__pycache__`：3 个文件，0.0034 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/charset_normalizer/__pycache__`：10 个文件，0.1595 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/charset_normalizer/cli/__pycache__`：2 个文件，0.0150 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/cmr/__pycache__`：2 个文件，0.0470 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/colorama/__pycache__`：6 个文件，0.0407 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/colorama/tests/__pycache__`：7 个文件，0.0561 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/dateutil/__pycache__`：8 个文件，0.1055 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/dateutil/parser/__pycache__`：3 个文件，0.0863 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/dateutil/tz/__pycache__`：5 个文件，0.1053 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/dateutil/zoneinfo/__pycache__`：2 个文件，0.0126 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/earthaccess/__pycache__`：13 个文件，0.1793 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/earthaccess/utils/__pycache__`：2 个文件，0.0029 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/earthaccess/virtual/__pycache__`：5 个文件，0.0243 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/frozenlist/__pycache__`：1 个文件，0.0042 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/fsspec/__pycache__`：22 个文件，0.4202 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/fsspec/implementations/__pycache__`：27 个文件，0.4938 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/fsspec/tests/abstract/__pycache__`：8 个文件，0.0996 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/idna/__pycache__`：8 个文件，0.2784 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/importlib_resources/__pycache__`：8 个文件，0.0546 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/importlib_resources/compat/__pycache__`：2 个文件，0.0007 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/importlib_resources/future/__pycache__`：2 个文件，0.0060 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/importlib_resources/tests/__pycache__`：15 个文件，0.1158 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/importlib_resources/tests/compat/__pycache__`：3 个文件，0.0020 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/jmespath/__pycache__`：8 个文件，0.0817 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/multidict/__pycache__`：4 个文件，0.0774 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/multimethod/__pycache__`：1 个文件，0.0343 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/__pycache__`：3 个文件，0.0041 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/__pycache__`：9 个文件，0.1167 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/cli/__pycache__`：12 个文件，0.1113 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/commands/__pycache__`：18 个文件，0.1571 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/distributions/__pycache__`：5 个文件，0.0173 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/index/__pycache__`：4 个文件，0.0793 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/locations/__pycache__`：4 个文件，0.0371 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/metadata/__pycache__`：4 个文件，0.0634 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/metadata/importlib/__pycache__`：4 个文件，0.0301 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/models/__pycache__`：12 个文件，0.0710 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/network/__pycache__`：8 个文件，0.0785 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/operations/__pycache__`：4 个文件，0.0461 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/operations/build/__pycache__`：8 个文件，0.0255 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/operations/install/__pycache__`：3 个文件，0.0408 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/req/__pycache__`：6 个文件，0.1304 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/resolution/__pycache__`：2 个文件，0.0016 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/resolution/legacy/__pycache__`：2 个文件，0.0229 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/resolution/resolvelib/__pycache__`：9 个文件，0.1207 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/utils/__pycache__`：27 个文件，0.1676 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_internal/vcs/__pycache__`：6 个文件，0.0794 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/__pycache__`：3 个文件，0.1753 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/cachecontrol/__pycache__`：9 个文件，0.0528 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/cachecontrol/caches/__pycache__`：3 个文件，0.0121 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/certifi/__pycache__`：3 个文件，0.0044 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/chardet/__pycache__`：44 个文件，0.9558 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/chardet/cli/__pycache__`：2 个文件，0.0044 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/chardet/metadata/__pycache__`：2 个文件，0.0106 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/colorama/__pycache__`：6 个文件，0.0407 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/colorama/tests/__pycache__`：7 个文件，0.0562 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/distlib/__pycache__`：13 个文件，0.5007 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/distro/__pycache__`：3 个文件，0.0567 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/idna/__pycache__`：8 个文件，0.2220 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/msgpack/__pycache__`：4 个文件，0.0582 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/packaging/__pycache__`：11 个文件，0.1289 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/pkg_resources/__pycache__`：1 个文件，0.1528 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/platformdirs/__pycache__`：8 个文件，0.0718 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/pygments/__pycache__`：16 个文件，0.1737 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/pygments/filters/__pycache__`：1 个文件，0.0383 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/pygments/formatters/__pycache__`：14 个文件，0.1662 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/pygments/lexers/__pycache__`：3 个文件，0.1188 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/pygments/styles/__pycache__`：1 个文件，0.0045 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/pyparsing/__pycache__`：10 个文件，0.4629 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/pyparsing/diagram/__pycache__`：1 个文件，0.0275 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/pyproject_hooks/__pycache__`：3 个文件，0.0171 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/pyproject_hooks/_in_process/__pycache__`：2 个文件，0.0169 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/requests/__pycache__`：18 个文件，0.2115 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/resolvelib/__pycache__`：5 个文件，0.0493 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/resolvelib/compat/__pycache__`：2 个文件，0.0008 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/rich/__pycache__`：77 个文件，1.1635 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/tenacity/__pycache__`：11 个文件，0.0794 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/tomli/__pycache__`：4 个文件，0.0347 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/truststore/__pycache__`：6 个文件，0.0533 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/urllib3/__pycache__`：11 个文件，0.1727 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/urllib3/contrib/__pycache__`：7 个文件，0.0874 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/urllib3/contrib/_securetransport/__pycache__`：3 个文件，0.0314 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/urllib3/packages/__pycache__`：2 个文件，0.0446 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/urllib3/packages/backports/__pycache__`：3 个文件，0.0099 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/urllib3/util/__pycache__`：13 个文件，0.1045 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pip/_vendor/webencodings/__pycache__`：5 个文件，0.0366 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pkg_resources/__pycache__`：1 个文件，0.1522 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pkg_resources/_vendor/__pycache__`：3 个文件，0.0437 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pkg_resources/_vendor/importlib_resources/__pycache__`：9 个文件，0.0497 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pkg_resources/_vendor/jaraco/__pycache__`：3 个文件，0.0287 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pkg_resources/_vendor/jaraco/text/__pycache__`：1 个文件，0.0254 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pkg_resources/_vendor/more_itertools/__pycache__`：3 个文件，0.1863 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pkg_resources/_vendor/packaging/__pycache__`：11 个文件，0.1290 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pkg_resources/_vendor/pyparsing/__pycache__`：10 个文件，0.4405 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pkg_resources/_vendor/pyparsing/diagram/__pycache__`：1 个文件，0.0268 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pkg_resources/extern/__pycache__`：1 个文件，0.0042 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pqdm/__pycache__`：6 个文件，0.0114 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/propcache/__pycache__`：4 个文件，0.0064 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/requests/__pycache__`：19 个文件，0.2549 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/s3fs/__pycache__`：6 个文件，0.1459 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/__pycache__`：30 个文件，0.3648 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_distutils/__pycache__`：33 个文件，0.4237 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_distutils/command/__pycache__`：23 个文件，0.2294 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_vendor/__pycache__`：4 个文件，0.1391 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_vendor/importlib_metadata/__pycache__`：8 个文件，0.0773 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_vendor/importlib_resources/__pycache__`：9 个文件，0.0497 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_vendor/jaraco/__pycache__`：3 个文件，0.0287 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_vendor/jaraco/text/__pycache__`：1 个文件，0.0254 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_vendor/more_itertools/__pycache__`：3 个文件，0.1654 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_vendor/packaging/__pycache__`：11 个文件，0.1290 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_vendor/pyparsing/__pycache__`：10 个文件，0.4405 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_vendor/pyparsing/diagram/__pycache__`：1 个文件，0.0268 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/_vendor/tomli/__pycache__`：4 个文件，0.0347 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/command/__pycache__`：26 个文件，0.3860 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/config/__pycache__`：5 个文件，0.1083 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/config/_validate_pyproject/__pycache__`：6 个文件，0.2241 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/setuptools/extern/__pycache__`：1 个文件，0.0042 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/tenacity/__pycache__`：10 个文件，0.0884 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/tenacity/asyncio/__pycache__`：2 个文件，0.0159 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/tqdm/__pycache__`：22 个文件，0.1761 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/tqdm/contrib/__pycache__`：9 个文件，0.0443 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/urllib3/__pycache__`：12 个文件，0.2483 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/urllib3/contrib/__pycache__`：3 个文件，0.0368 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/urllib3/contrib/emscripten/__pycache__`：5 个文件，0.0555 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/urllib3/http2/__pycache__`：3 个文件，0.0244 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/urllib3/util/__pycache__`：13 个文件，0.1104 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/wrapt/__pycache__`：9 个文件，0.1002 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/yarl/__pycache__`：8 个文件，0.0947 MB
- `tmp/venvs/nasa_oceancolor_20260520/Scripts/__pycache__`：1 个文件，0.0034 MB

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-21 02:18:57

- 运行模式：`apply-safe`
- 扫描文件数：17506
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：195 个

### 清理的缓存

- `scripts/__pycache__`：1 个文件，0.0155 MB
- `scripts/data/__pycache__`：1 个文件，0.0886 MB
- `scripts/training/__pycache__`：3 个文件，0.1327 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/__pycache__`：3 个文件，0.2219 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiobotocore/__pycache__`：29 个文件，0.2951 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiobotocore/retries/__pycache__`：5 个文件，0.0179 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiohappyeyeballs/__pycache__`：5 个文件，0.0248 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiohttp/__pycache__`：31 个文件，0.6725 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiohttp/_websocket/__pycache__`：5 个文件，0.0225 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aioitertools/__pycache__`：7 个文件，0.0534 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/aiosignal/__pycache__`：1 个文件，0.0030 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/attr/__pycache__`：13 个文件，0.2139 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/__pycache__`：41 个文件，1.2782 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/crt/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/docs/__pycache__`：12 个文件，0.1177 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/docs/bcdoc/__pycache__`：4 个文件，0.0576 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/retries/__pycache__`：8 个文件，0.0431 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/vendored/__pycache__`：2 个文件，0.0446 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/vendored/requests/__pycache__`：2 个文件，0.0061 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/vendored/requests/packages/__pycache__`：1 个文件，0.0004 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/botocore/vendored/requests/packages/urllib3/__pycache__`：2 个文件，0.0093 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/bounded_pool_executor/__pycache__`：1 个文件，0.0028 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/certifi/__pycache__`：2 个文件，0.0027 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/charset_normalizer/__pycache__`：7 个文件，0.1125 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/cmr/__pycache__`：2 个文件，0.0470 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/colorama/__pycache__`：6 个文件，0.0407 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/dateutil/__pycache__`：4 个文件，0.0291 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/dateutil/parser/__pycache__`：3 个文件，0.0863 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/dateutil/tz/__pycache__`：5 个文件，0.1053 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/earthaccess/__pycache__`：12 个文件，0.1791 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/earthaccess/utils/__pycache__`：2 个文件，0.0029 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/earthaccess/virtual/__pycache__`：5 个文件，0.0243 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/frozenlist/__pycache__`：1 个文件，0.0042 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/fsspec/__pycache__`：16 个文件，0.3428 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/fsspec/implementations/__pycache__`：2 个文件，0.0276 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/idna/__pycache__`：5 个文件，0.1320 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/importlib_resources/__pycache__`：4 个文件，0.0239 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/jmespath/__pycache__`：8 个文件，0.0817 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/multidict/__pycache__`：3 个文件，0.0085 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/multimethod/__pycache__`：1 个文件，0.0343 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/pqdm/__pycache__`：5 个文件，0.0099 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/propcache/__pycache__`：3 个文件，0.0030 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/requests/__pycache__`：18 个文件，0.2505 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/s3fs/__pycache__`：6 个文件，0.1459 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/tenacity/__pycache__`：9 个文件，0.0856 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/tenacity/asyncio/__pycache__`：2 个文件，0.0159 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/tqdm/__pycache__`：11 个文件，0.1316 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/urllib3/__pycache__`：12 个文件，0.2483 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/urllib3/contrib/__pycache__`：3 个文件，0.0368 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/urllib3/http2/__pycache__`：2 个文件，0.0062 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/urllib3/util/__pycache__`：13 个文件，0.1104 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/wrapt/__pycache__`：9 个文件，0.1002 MB
- `tmp/venvs/nasa_oceancolor_20260520/Lib/site-packages/yarl/__pycache__`：7 个文件，0.0848 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/__pycache__`：3 个文件，0.2774 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/_distutils_hack/__pycache__`：1 个文件，0.0115 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/catboost/__pycache__`：5 个文件，0.4058 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/catboost/widget/__pycache__`：2 个文件，0.0090 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/colorama/__pycache__`：6 个文件，0.0406 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/cycler/__pycache__`：1 个文件，0.0245 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/__pycache__`：4 个文件，0.0290 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/parser/__pycache__`：3 个文件，0.0862 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/tz/__pycache__`：5 个文件，0.1053 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/__pycache__`：17 个文件，0.0926 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/backend/__pycache__`：9 个文件，0.0484 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/parameters/__pycache__`：7 个文件，0.0166 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/__pycache__`：20 个文件，0.3548 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/cloudpickle/__pycache__`：2 个文件，0.0572 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/loky/__pycache__`：6 个文件，0.0775 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/loky/backend/__pycache__`：9 个文件，0.0673 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/kiwisolver/__pycache__`：2 个文件，0.0031 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/lightgbm/__pycache__`：9 个文件，0.5070 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/__pycache__`：21 个文件，1.2300 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/_api/__pycache__`：2 个文件，0.0438 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/backends/__pycache__`：2 个文件，0.0158 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/__pycache__`：11 个文件，0.0986 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_core/__pycache__`：29 个文件，1.0223 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_core/tests/__pycache__`：1 个文件，0.0083 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_typing/__pycache__`：11 个文件，0.0439 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_utils/__pycache__`：3 个文件，0.0150 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/char/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/core/__pycache__`：2 个文件，0.0025 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/f2py/__pycache__`：16 个文件，0.5509 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/f2py/_backends/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/fft/__pycache__`：4 个文件，0.0813 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/lib/__pycache__`：28 个文件，0.9009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/linalg/__pycache__`：3 个文件，0.1326 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/ma/__pycache__`：3 个文件，0.4014 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/matrixlib/__pycache__`：2 个文件，0.0366 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/polynomial/__pycache__`：9 个文件，0.4482 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/random/__pycache__`：2 个文件，0.0106 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/rec/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/strings/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/testing/__pycache__`：2 个文件，0.0039 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/testing/_private/__pycache__`：3 个文件，0.1221 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/typing/__pycache__`：1 个文件，0.0052 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/packaging/__pycache__`：2 个文件，0.0425 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/__pycache__`：4 个文件，0.0265 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_config/__pycache__`：5 个文件，0.0476 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/tslibs/__pycache__`：1 个文件，0.0023 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/window/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_testing/__pycache__`：6 个文件，0.0979 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/__pycache__`：1 个文件，0.0005 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/extensions/__pycache__`：1 个文件，0.0009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/indexers/__pycache__`：1 个文件，0.0006 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/interchange/__pycache__`：1 个文件，0.0005 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/types/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/typing/__pycache__`：1 个文件，0.0014 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/arrays/__pycache__`：1 个文件，0.0016 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/compat/__pycache__`：6 个文件，0.0253 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/compat/numpy/__pycache__`：2 个文件，0.0162 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/__pycache__`：22 个文件，1.8349 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/_numba/__pycache__`：2 个文件，0.0096 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/array_algos/__pycache__`：9 个文件，0.0532 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/__pycache__`：20 个文件，0.7842 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/arrow/__pycache__`：3 个文件，0.1616 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/sparse/__pycache__`：3 个文件，0.0896 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/computation/__pycache__`：13 个文件，0.1559 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/dtypes/__pycache__`：11 个文件，0.3049 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/groupby/__pycache__`：9 个文件，0.4216 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/indexers/__pycache__`：3 个文件，0.0357 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/indexes/__pycache__`：14 个文件，0.7186 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/interchange/__pycache__`：4 个文件，0.0443 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/internals/__pycache__`：9 个文件，0.3463 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/methods/__pycache__`：3 个文件，0.0262 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/ops/__pycache__`：8 个文件，0.0563 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/reshape/__pycache__`：9 个文件，0.2385 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/strings/__pycache__`：4 个文件，0.1680 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/tools/__pycache__`：5 个文件，0.0735 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/util/__pycache__`：3 个文件，0.0162 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/window/__pycache__`：8 个文件，0.1977 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/errors/__pycache__`：1 个文件，0.0318 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/__pycache__`：16 个文件，0.7004 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/excel/__pycache__`：10 个文件，0.1495 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/formats/__pycache__`：5 个文件，0.1665 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/json/__pycache__`：4 个文件，0.0897 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/parsers/__pycache__`：6 个文件，0.2195 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/sas/__pycache__`：2 个文件，0.0065 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/plotting/__pycache__`：3 个文件，0.0910 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/tseries/__pycache__`：4 个文件，0.0255 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/util/__pycache__`：6 个文件，0.0502 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/util/version/__pycache__`：1 个文件，0.0238 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/PIL/__pycache__`：18 个文件，0.3827 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pyparsing/__pycache__`：11 个文件，0.5119 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pytz/__pycache__`：5 个文件，0.0678 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/__pycache__`：4 个文件，0.0116 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/__pycache__`：14 个文件，0.2012 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/_uarray/__pycache__`：2 个文件，0.0295 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/__pycache__`：2 个文件，0.0031 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/common/__pycache__`：5 个文件，0.0512 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/numpy/__pycache__`：4 个文件，0.0105 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/constants/__pycache__`：5 个文件，0.1768 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/fft/__pycache__`：9 个文件，0.1381 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/fft/_pocketfft/__pycache__`：4 个文件，0.0240 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/integrate/__pycache__`：12 个文件，0.2632 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/integrate/_ivp/__pycache__`：9 个文件，0.1609 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/interpolate/__pycache__`：20 个文件，0.5416 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/linalg/__pycache__`：34 个文件，0.5369 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/ndimage/__pycache__`：13 个文件，0.2946 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/__pycache__`：53 个文件，1.2896 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_highs/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_lsq/__pycache__`：8 个文件，0.1351 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_shgo_lib/__pycache__`：3 个文件，0.0702 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_trlib/__pycache__`：1 个文件，0.0009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_trustregion_constr/__pycache__`：8 个文件，0.1128 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/__pycache__`：31 个文件，0.5053 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/csgraph/__pycache__`：3 个文件，0.0317 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/__pycache__`：13 个文件，0.2060 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_dsolve/__pycache__`：3 个文件，0.0356 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/__pycache__`：2 个文件，0.0211 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/arpack/__pycache__`：2 个文件，0.0673 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/lobpcg/__pycache__`：2 个文件，0.0405 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_isolve/__pycache__`：9 个文件，0.1223 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/spatial/__pycache__`：10 个文件，0.1850 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/spatial/transform/__pycache__`：4 个文件，0.0285 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/special/__pycache__`：16 个文件，0.2937 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/__pycache__`：54 个文件，2.8931 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_levy_stable/__pycache__`：1 个文件，0.0516 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_rcont/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_unuran/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/__check_build/__pycache__`：1 个文件，0.0024 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/__pycache__`：6 个文件，0.0819 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/externals/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/externals/_packaging/__pycache__`：3 个文件，0.0255 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/__pycache__`：7 个文件，0.4100 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/_pairwise_distances_reduction/__pycache__`：2 个文件，0.0301 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/_plot/__pycache__`：6 个文件，0.0753 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/cluster/__pycache__`：4 个文件，0.0712 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/model_selection/__pycache__`：6 个文件，0.3561 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/preprocessing/__pycache__`：8 个文件，0.3429 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/utils/__pycache__`：33 个文件，0.5806 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/tqdm/__pycache__`：11 个文件，0.1316 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/xgboost/__pycache__`：15 个文件，0.4877 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/yaml/__pycache__`：17 个文件，0.2637 MB

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## NASA OceanColor 月尺度全时段构建与 A/B

- 更新 `data\processed\current\nasa_oceancolor_features_2014_2024_long.csv`、`data\processed\current\nasa_oceancolor_features_2014_2024_wide.csv`。
- 更新 `experiments\outputs\current\nasa_oceancolor_ab_metrics.csv`、`experiments\outputs\current\nasa_oceancolor_comparison_to_current_best.csv`、`experiments\outputs\current\nasa_oceancolor_feature_importance.csv`。
- 更新 `experiments\reports\current\nasa_oceancolor_fullrun_report.md` 与 `nasa_oceancolor_full_validation_report.md`。
- 生成 `data\processed\current\current_best_scorecard_after_nasa_oceancolor.csv` 与 `data\processed\current\r2_0p6_progress_scorecard_after_nasa_oceancolor.csv`。

## Project Organizer Agent 运行记录：2026-05-21 03:02:15

- 运行模式：`dry-run`
- 扫描文件数：17514
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。
## 2026-05-21 四指标口径重写

- 重写 `precision_improvement_stage_plan.md`，将阶段总目标从偏重 `Chl-a bottom` 调整为 **Chl-a / Secchi / TN / TP 四指标统一冲击 R² > 0.60**。
- 新增 `ab_evaluation_template_4targets.md` 和 `ab_evaluation_template_4targets.csv`，作为后续统一 A/B 评价模板。
- 重写 `next_data_acquisition_plan.md`，将数据源优先级改为“对四指标普遍有帮助的通用数据优先”。
- 更新 `experiment_report.md`，将 current 主目标和主线判断与上述新口径对齐。
- 本轮仅做 `current/` 入口文档和模板收敛，未新增训练、未改模型、未覆盖历史实验结果。

## 2026-05-21 TN/TP baseline bootstrap

- 新增脚本 `scripts/training/run_tn_tp_baseline_bootstrap.py`。
- 基于 `data/interim/current/current_model_dataset_with_highres_and_mechanism.csv` 完成 TN / TP 三水层 baseline bootstrap。
- 新增输出：
  - `experiments/outputs/current/tn_tp_baseline_bootstrap_metrics.csv`
  - `data/processed/current/tn_tp_baseline_bootstrap_scorecard.csv`
  - `data/processed/current/four_target_bootstrap_scorecard.csv`
  - `experiments/reports/current/tn_tp_baseline_bootstrap_report.md`
  - `experiments/reports/current/four_target_baseline_readiness.md`
- 关键结果：
  - `TN surface` baseline bootstrap test R² = `0.5201`
  - `TP surface` baseline bootstrap test R² = `0.3986`
- 本轮开始把“四指标统一评价”从文档层推进到可运行 baseline 层。

## Project Organizer Agent 运行记录：2026-05-21 03:31:27

- 运行模式：`dry-run`
- 扫描文件数：17522
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-21 03:31:33

- 运行模式：`apply-safe`
- 扫描文件数：17522
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：148 个

### 清理的缓存

- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/__pycache__`：2 个文件，0.1062 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/_distutils_hack/__pycache__`：1 个文件，0.0115 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/cycler/__pycache__`：1 个文件，0.0245 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/__pycache__`：4 个文件，0.0290 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/parser/__pycache__`：3 个文件，0.0862 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/tz/__pycache__`：5 个文件，0.1053 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/__pycache__`：17 个文件，0.0926 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/backend/__pycache__`：9 个文件，0.0484 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/parameters/__pycache__`：7 个文件，0.0166 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/__pycache__`：20 个文件，0.3548 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/cloudpickle/__pycache__`：2 个文件，0.0572 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/loky/__pycache__`：6 个文件，0.0775 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/loky/backend/__pycache__`：9 个文件，0.0673 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/kiwisolver/__pycache__`：2 个文件，0.0031 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/lightgbm/__pycache__`：9 个文件，0.5070 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/__pycache__`：21 个文件，1.2300 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/_api/__pycache__`：2 个文件，0.0438 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/backends/__pycache__`：2 个文件，0.0158 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/__pycache__`：11 个文件，0.0986 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_core/__pycache__`：29 个文件，1.0223 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_core/tests/__pycache__`：1 个文件，0.0083 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_typing/__pycache__`：11 个文件，0.0439 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_utils/__pycache__`：3 个文件，0.0150 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/char/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/core/__pycache__`：2 个文件，0.0025 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/f2py/__pycache__`：16 个文件，0.5509 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/f2py/_backends/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/fft/__pycache__`：4 个文件，0.0813 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/lib/__pycache__`：28 个文件，0.9009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/linalg/__pycache__`：3 个文件，0.1326 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/ma/__pycache__`：3 个文件，0.4014 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/matrixlib/__pycache__`：2 个文件，0.0366 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/polynomial/__pycache__`：9 个文件，0.4482 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/random/__pycache__`：2 个文件，0.0106 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/rec/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/strings/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/testing/__pycache__`：2 个文件，0.0039 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/testing/_private/__pycache__`：3 个文件，0.1221 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/typing/__pycache__`：1 个文件，0.0052 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/packaging/__pycache__`：2 个文件，0.0425 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/__pycache__`：4 个文件，0.0265 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_config/__pycache__`：5 个文件，0.0476 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/tslibs/__pycache__`：1 个文件，0.0023 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/window/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_testing/__pycache__`：6 个文件，0.0979 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/__pycache__`：1 个文件，0.0005 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/extensions/__pycache__`：1 个文件，0.0009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/indexers/__pycache__`：1 个文件，0.0006 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/interchange/__pycache__`：1 个文件，0.0005 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/types/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/typing/__pycache__`：1 个文件，0.0014 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/arrays/__pycache__`：1 个文件，0.0016 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/compat/__pycache__`：6 个文件，0.0253 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/compat/numpy/__pycache__`：2 个文件，0.0162 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/__pycache__`：22 个文件，1.8349 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/_numba/__pycache__`：2 个文件，0.0096 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/array_algos/__pycache__`：9 个文件，0.0532 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/__pycache__`：20 个文件，0.7842 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/arrow/__pycache__`：3 个文件，0.1616 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/sparse/__pycache__`：3 个文件，0.0896 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/computation/__pycache__`：13 个文件，0.1559 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/dtypes/__pycache__`：11 个文件，0.3049 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/groupby/__pycache__`：9 个文件，0.4216 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/indexers/__pycache__`：3 个文件，0.0357 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/indexes/__pycache__`：14 个文件，0.7186 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/interchange/__pycache__`：4 个文件，0.0443 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/internals/__pycache__`：9 个文件，0.3463 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/methods/__pycache__`：3 个文件，0.0262 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/ops/__pycache__`：8 个文件，0.0563 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/reshape/__pycache__`：9 个文件，0.2385 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/strings/__pycache__`：4 个文件，0.1680 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/tools/__pycache__`：5 个文件，0.0735 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/util/__pycache__`：3 个文件，0.0162 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/window/__pycache__`：8 个文件，0.1977 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/errors/__pycache__`：1 个文件，0.0318 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/__pycache__`：16 个文件，0.7004 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/excel/__pycache__`：10 个文件，0.1495 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/formats/__pycache__`：6 个文件，0.1820 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/json/__pycache__`：4 个文件，0.0897 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/parsers/__pycache__`：6 个文件，0.2195 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/sas/__pycache__`：2 个文件，0.0065 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/plotting/__pycache__`：3 个文件，0.0910 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/tseries/__pycache__`：4 个文件，0.0255 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/util/__pycache__`：6 个文件，0.0502 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/util/version/__pycache__`：1 个文件，0.0238 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/PIL/__pycache__`：18 个文件，0.3827 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pyparsing/__pycache__`：11 个文件，0.5119 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pytz/__pycache__`：5 个文件，0.0678 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/__pycache__`：4 个文件，0.0116 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/__pycache__`：14 个文件，0.2012 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/_uarray/__pycache__`：2 个文件，0.0295 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/__pycache__`：2 个文件，0.0031 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/common/__pycache__`：5 个文件，0.0512 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/numpy/__pycache__`：4 个文件，0.0105 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/constants/__pycache__`：5 个文件，0.1768 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/fft/__pycache__`：9 个文件，0.1381 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/fft/_pocketfft/__pycache__`：4 个文件，0.0240 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/integrate/__pycache__`：12 个文件，0.2632 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/integrate/_ivp/__pycache__`：9 个文件，0.1609 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/interpolate/__pycache__`：20 个文件，0.5416 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/linalg/__pycache__`：34 个文件，0.5369 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/ndimage/__pycache__`：13 个文件，0.2946 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/__pycache__`：53 个文件，1.2896 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_highs/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_lsq/__pycache__`：8 个文件，0.1351 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_shgo_lib/__pycache__`：3 个文件，0.0702 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_trlib/__pycache__`：1 个文件，0.0009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_trustregion_constr/__pycache__`：8 个文件，0.1128 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/__pycache__`：31 个文件，0.5053 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/csgraph/__pycache__`：3 个文件，0.0317 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/__pycache__`：13 个文件，0.2060 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_dsolve/__pycache__`：3 个文件，0.0356 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/__pycache__`：2 个文件，0.0211 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/arpack/__pycache__`：2 个文件，0.0673 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/lobpcg/__pycache__`：2 个文件，0.0405 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_isolve/__pycache__`：9 个文件，0.1223 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/spatial/__pycache__`：10 个文件，0.1850 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/spatial/transform/__pycache__`：4 个文件，0.0285 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/special/__pycache__`：16 个文件，0.2937 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/__pycache__`：54 个文件，2.8931 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_levy_stable/__pycache__`：1 个文件，0.0516 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_rcont/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_unuran/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/__check_build/__pycache__`：1 个文件，0.0024 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/__pycache__`：9 个文件，0.2519 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/_loss/__pycache__`：3 个文件，0.0558 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/compose/__pycache__`：3 个文件，0.0908 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/covariance/__pycache__`：6 个文件，0.1283 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/decomposition/__pycache__`：12 个文件，0.3549 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/ensemble/__pycache__`：9 个文件，0.4161 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/ensemble/_hist_gradient_boosting/__pycache__`：5 个文件，0.1352 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/externals/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/externals/_packaging/__pycache__`：3 个文件，0.0255 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/impute/__pycache__`：3 个文件，0.0603 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/linear_model/__pycache__`：17 个文件，0.7096 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/linear_model/_glm/__pycache__`：3 个文件，0.0526 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/__pycache__`：7 个文件，0.4100 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/_pairwise_distances_reduction/__pycache__`：2 个文件，0.0301 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/_plot/__pycache__`：6 个文件，0.0753 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/cluster/__pycache__`：4 个文件，0.0712 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/model_selection/__pycache__`：6 个文件，0.3561 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/neighbors/__pycache__`：10 个文件，0.2075 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/preprocessing/__pycache__`：8 个文件，0.3429 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/svm/__pycache__`：4 个文件，0.1152 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/tree/__pycache__`：4 个文件，0.1287 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/utils/__pycache__`：36 个文件，0.5956 MB

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-21 10:30:13

- 运行模式：`apply-safe`
- 扫描文件数：17539
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：152 个

### 清理的缓存

- `scripts/training/__pycache__`：4 个文件，0.1507 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/__pycache__`：2 个文件，0.1062 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/_distutils_hack/__pycache__`：1 个文件，0.0115 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/catboost/__pycache__`：5 个文件，0.4058 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/catboost/widget/__pycache__`：2 个文件，0.0090 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/cycler/__pycache__`：1 个文件，0.0245 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/__pycache__`：4 个文件，0.0290 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/parser/__pycache__`：3 个文件，0.0862 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/tz/__pycache__`：5 个文件，0.1053 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/__pycache__`：17 个文件，0.0926 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/backend/__pycache__`：9 个文件，0.0484 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/parameters/__pycache__`：7 个文件，0.0166 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/__pycache__`：20 个文件，0.3548 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/cloudpickle/__pycache__`：2 个文件，0.0572 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/loky/__pycache__`：6 个文件，0.0775 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/loky/backend/__pycache__`：9 个文件，0.0673 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/kiwisolver/__pycache__`：2 个文件，0.0031 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/lightgbm/__pycache__`：9 个文件，0.5070 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/__pycache__`：21 个文件，1.2300 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/_api/__pycache__`：2 个文件，0.0438 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/backends/__pycache__`：2 个文件，0.0158 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/__pycache__`：11 个文件，0.0986 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_core/__pycache__`：29 个文件，1.0223 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_core/tests/__pycache__`：1 个文件，0.0083 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_typing/__pycache__`：11 个文件，0.0439 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_utils/__pycache__`：3 个文件，0.0150 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/char/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/core/__pycache__`：2 个文件，0.0025 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/f2py/__pycache__`：16 个文件，0.5509 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/f2py/_backends/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/fft/__pycache__`：4 个文件，0.0813 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/lib/__pycache__`：28 个文件，0.9009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/linalg/__pycache__`：3 个文件，0.1326 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/ma/__pycache__`：3 个文件，0.4014 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/matrixlib/__pycache__`：2 个文件，0.0366 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/polynomial/__pycache__`：9 个文件，0.4482 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/random/__pycache__`：2 个文件，0.0106 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/rec/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/strings/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/testing/__pycache__`：2 个文件，0.0039 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/testing/_private/__pycache__`：3 个文件，0.1221 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/typing/__pycache__`：1 个文件，0.0052 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/packaging/__pycache__`：2 个文件，0.0425 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/__pycache__`：4 个文件，0.0265 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_config/__pycache__`：5 个文件，0.0476 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/tslibs/__pycache__`：1 个文件，0.0023 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/window/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_testing/__pycache__`：6 个文件，0.0979 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/__pycache__`：1 个文件，0.0005 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/extensions/__pycache__`：1 个文件，0.0009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/indexers/__pycache__`：1 个文件，0.0006 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/interchange/__pycache__`：1 个文件，0.0005 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/types/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/typing/__pycache__`：1 个文件，0.0014 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/arrays/__pycache__`：1 个文件，0.0016 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/compat/__pycache__`：6 个文件，0.0253 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/compat/numpy/__pycache__`：2 个文件，0.0162 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/__pycache__`：22 个文件，1.8349 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/_numba/__pycache__`：2 个文件，0.0096 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/array_algos/__pycache__`：9 个文件，0.0532 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/__pycache__`：20 个文件，0.7842 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/arrow/__pycache__`：3 个文件，0.1616 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/sparse/__pycache__`：3 个文件，0.0896 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/computation/__pycache__`：13 个文件，0.1559 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/dtypes/__pycache__`：11 个文件，0.3049 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/groupby/__pycache__`：9 个文件，0.4216 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/indexers/__pycache__`：3 个文件，0.0357 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/indexes/__pycache__`：14 个文件，0.7186 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/interchange/__pycache__`：4 个文件，0.0443 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/internals/__pycache__`：9 个文件，0.3463 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/methods/__pycache__`：4 个文件，0.0395 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/ops/__pycache__`：8 个文件，0.0563 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/reshape/__pycache__`：9 个文件，0.2385 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/strings/__pycache__`：4 个文件，0.1680 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/tools/__pycache__`：5 个文件，0.0735 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/util/__pycache__`：3 个文件，0.0162 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/window/__pycache__`：8 个文件，0.1977 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/errors/__pycache__`：1 个文件，0.0318 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/__pycache__`：16 个文件，0.7004 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/excel/__pycache__`：10 个文件，0.1495 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/formats/__pycache__`：6 个文件，0.1820 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/json/__pycache__`：4 个文件，0.0897 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/parsers/__pycache__`：6 个文件，0.2195 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/sas/__pycache__`：2 个文件，0.0065 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/plotting/__pycache__`：3 个文件，0.0910 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/tseries/__pycache__`：4 个文件，0.0255 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/util/__pycache__`：6 个文件，0.0502 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/util/version/__pycache__`：1 个文件，0.0238 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/PIL/__pycache__`：18 个文件，0.3827 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pyparsing/__pycache__`：11 个文件，0.5119 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pytz/__pycache__`：5 个文件，0.0678 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/__pycache__`：4 个文件，0.0116 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/__pycache__`：14 个文件，0.2012 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/_uarray/__pycache__`：2 个文件，0.0295 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/__pycache__`：2 个文件，0.0031 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/common/__pycache__`：5 个文件，0.0512 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/numpy/__pycache__`：4 个文件，0.0105 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/constants/__pycache__`：5 个文件，0.1768 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/fft/__pycache__`：9 个文件，0.1381 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/fft/_pocketfft/__pycache__`：4 个文件，0.0240 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/integrate/__pycache__`：12 个文件，0.2632 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/integrate/_ivp/__pycache__`：9 个文件，0.1609 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/interpolate/__pycache__`：20 个文件，0.5416 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/linalg/__pycache__`：34 个文件，0.5369 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/ndimage/__pycache__`：13 个文件，0.2946 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/__pycache__`：53 个文件，1.2896 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_highs/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_lsq/__pycache__`：8 个文件，0.1351 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_shgo_lib/__pycache__`：3 个文件，0.0702 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_trlib/__pycache__`：1 个文件，0.0009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_trustregion_constr/__pycache__`：8 个文件，0.1128 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/__pycache__`：31 个文件，0.5053 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/csgraph/__pycache__`：3 个文件，0.0317 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/__pycache__`：13 个文件，0.2060 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_dsolve/__pycache__`：3 个文件，0.0356 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/__pycache__`：2 个文件，0.0211 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/arpack/__pycache__`：2 个文件，0.0673 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/lobpcg/__pycache__`：2 个文件，0.0405 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_isolve/__pycache__`：9 个文件，0.1223 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/spatial/__pycache__`：10 个文件，0.1850 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/spatial/transform/__pycache__`：4 个文件，0.0285 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/special/__pycache__`：16 个文件，0.2937 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/__pycache__`：54 个文件，2.8931 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_levy_stable/__pycache__`：1 个文件，0.0516 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_rcont/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_unuran/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/__check_build/__pycache__`：1 个文件，0.0024 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/__pycache__`：9 个文件，0.2519 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/_loss/__pycache__`：3 个文件，0.0558 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/compose/__pycache__`：3 个文件，0.0908 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/covariance/__pycache__`：6 个文件，0.1283 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/decomposition/__pycache__`：12 个文件，0.3549 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/ensemble/__pycache__`：9 个文件，0.4161 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/ensemble/_hist_gradient_boosting/__pycache__`：5 个文件，0.1352 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/externals/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/externals/_packaging/__pycache__`：3 个文件，0.0255 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/impute/__pycache__`：3 个文件，0.0603 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/linear_model/__pycache__`：17 个文件，0.7096 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/linear_model/_glm/__pycache__`：3 个文件，0.0526 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/__pycache__`：7 个文件，0.4100 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/_pairwise_distances_reduction/__pycache__`：2 个文件，0.0301 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/_plot/__pycache__`：6 个文件，0.0753 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/cluster/__pycache__`：4 个文件，0.0712 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/model_selection/__pycache__`：6 个文件，0.3561 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/neighbors/__pycache__`：10 个文件，0.2075 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/preprocessing/__pycache__`：8 个文件，0.3429 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/svm/__pycache__`：4 个文件，0.1152 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/tree/__pycache__`：4 个文件，0.1287 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/utils/__pycache__`：36 个文件，0.5956 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/xgboost/__pycache__`：15 个文件，0.4877 MB

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-21 10:30:29

- 运行模式：`dry-run`
- 扫描文件数：17539
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## 2026-05-21 GEBCO / Coastal-distance ??

- ?? `scripts/data/build_coastal_distance_features.py`?????? `coastal_distance_features_2014_2024.csv`?
- ?? `scripts/training/run_coastal_distance_four_target_ab.py`???????? A/B?
- ?? A/B ??????? comparison ???? bug?
- ?? `coastal_distance_no_promote_reason.md`?????? promote ???
- ? coastal-distance ???? `precision_improvement_ab_metrics.csv` ? `precision_improvement_comparison_to_current_best.csv`?

## Project Organizer Agent 运行记录：2026-05-21 16:04:25

- 运行模式：`dry-run`
- 扫描文件数：18654
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：152
- 创建 MD 骨架：0 个
- 清理缓存目录：0 个

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。

## Project Organizer Agent 运行记录：2026-05-21 16:05:32

- 运行模式：`apply-safe`
- 扫描文件数：17555
- 根目录污染候选：0
- 重复输出候选组：9
- 缓存目录候选：0
- 创建 MD 骨架：0 个
- 清理缓存目录：152 个

### 清理的缓存

- `scripts/training/__pycache__`：4 个文件，0.1507 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/__pycache__`：2 个文件，0.1062 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/_distutils_hack/__pycache__`：1 个文件，0.0115 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/catboost/__pycache__`：5 个文件，0.4058 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/catboost/widget/__pycache__`：2 个文件，0.0090 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/cycler/__pycache__`：1 个文件，0.0245 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/__pycache__`：4 个文件，0.0290 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/parser/__pycache__`：3 个文件，0.0862 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/dateutil/tz/__pycache__`：5 个文件，0.1053 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/__pycache__`：17 个文件，0.0926 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/backend/__pycache__`：9 个文件，0.0484 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/graphviz/parameters/__pycache__`：7 个文件，0.0166 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/__pycache__`：20 个文件，0.3548 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/cloudpickle/__pycache__`：2 个文件，0.0572 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/loky/__pycache__`：6 个文件，0.0775 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/joblib/externals/loky/backend/__pycache__`：9 个文件，0.0673 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/kiwisolver/__pycache__`：2 个文件，0.0031 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/lightgbm/__pycache__`：9 个文件，0.5070 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/__pycache__`：21 个文件，1.2300 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/_api/__pycache__`：2 个文件，0.0438 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/matplotlib/backends/__pycache__`：2 个文件，0.0158 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/__pycache__`：11 个文件，0.0986 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_core/__pycache__`：29 个文件，1.0223 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_core/tests/__pycache__`：1 个文件，0.0083 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_typing/__pycache__`：11 个文件，0.0439 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/_utils/__pycache__`：3 个文件，0.0150 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/char/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/core/__pycache__`：2 个文件，0.0025 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/f2py/__pycache__`：16 个文件，0.5509 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/f2py/_backends/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/fft/__pycache__`：4 个文件，0.0813 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/lib/__pycache__`：28 个文件，0.9009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/linalg/__pycache__`：3 个文件，0.1326 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/ma/__pycache__`：3 个文件，0.4014 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/matrixlib/__pycache__`：2 个文件，0.0366 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/polynomial/__pycache__`：9 个文件，0.4482 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/random/__pycache__`：2 个文件，0.0106 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/rec/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/strings/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/testing/__pycache__`：2 个文件，0.0039 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/testing/_private/__pycache__`：3 个文件，0.1221 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/numpy/typing/__pycache__`：1 个文件，0.0052 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/packaging/__pycache__`：2 个文件，0.0425 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/__pycache__`：4 个文件，0.0265 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_config/__pycache__`：5 个文件，0.0476 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/tslibs/__pycache__`：1 个文件，0.0023 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_libs/window/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/_testing/__pycache__`：6 个文件，0.0979 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/__pycache__`：1 个文件，0.0005 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/extensions/__pycache__`：1 个文件，0.0009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/indexers/__pycache__`：1 个文件，0.0006 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/interchange/__pycache__`：1 个文件，0.0005 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/types/__pycache__`：1 个文件，0.0007 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/api/typing/__pycache__`：1 个文件，0.0014 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/arrays/__pycache__`：1 个文件，0.0016 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/compat/__pycache__`：6 个文件，0.0253 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/compat/numpy/__pycache__`：2 个文件，0.0162 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/__pycache__`：22 个文件，1.8349 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/_numba/__pycache__`：2 个文件，0.0096 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/array_algos/__pycache__`：9 个文件，0.0532 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/__pycache__`：20 个文件，0.7842 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/arrow/__pycache__`：3 个文件，0.1616 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/arrays/sparse/__pycache__`：3 个文件，0.0896 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/computation/__pycache__`：13 个文件，0.1559 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/dtypes/__pycache__`：11 个文件，0.3049 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/groupby/__pycache__`：9 个文件，0.4216 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/indexers/__pycache__`：3 个文件，0.0357 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/indexes/__pycache__`：14 个文件，0.7186 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/interchange/__pycache__`：4 个文件，0.0443 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/internals/__pycache__`：9 个文件，0.3463 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/methods/__pycache__`：4 个文件，0.0395 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/ops/__pycache__`：8 个文件，0.0563 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/reshape/__pycache__`：9 个文件，0.2385 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/strings/__pycache__`：4 个文件，0.1680 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/tools/__pycache__`：5 个文件，0.0735 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/util/__pycache__`：3 个文件，0.0162 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/core/window/__pycache__`：8 个文件，0.1977 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/errors/__pycache__`：1 个文件，0.0318 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/__pycache__`：16 个文件，0.7004 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/excel/__pycache__`：10 个文件，0.1495 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/formats/__pycache__`：6 个文件，0.1820 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/json/__pycache__`：4 个文件，0.0897 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/parsers/__pycache__`：6 个文件，0.2195 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/io/sas/__pycache__`：2 个文件，0.0065 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/plotting/__pycache__`：3 个文件，0.0910 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/tseries/__pycache__`：4 个文件，0.0255 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/util/__pycache__`：6 个文件，0.0502 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pandas/util/version/__pycache__`：1 个文件，0.0238 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/PIL/__pycache__`：18 个文件，0.3827 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pyparsing/__pycache__`：11 个文件，0.5119 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/pytz/__pycache__`：5 个文件，0.0678 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/__pycache__`：4 个文件，0.0116 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/__pycache__`：14 个文件，0.2012 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/_uarray/__pycache__`：2 个文件，0.0295 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/__pycache__`：2 个文件，0.0031 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/common/__pycache__`：5 个文件，0.0512 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/_lib/array_api_compat/numpy/__pycache__`：4 个文件，0.0105 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/constants/__pycache__`：5 个文件，0.1768 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/fft/__pycache__`：9 个文件，0.1381 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/fft/_pocketfft/__pycache__`：4 个文件，0.0240 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/integrate/__pycache__`：12 个文件，0.2632 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/integrate/_ivp/__pycache__`：9 个文件，0.1609 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/interpolate/__pycache__`：20 个文件，0.5416 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/linalg/__pycache__`：34 个文件，0.5369 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/ndimage/__pycache__`：13 个文件，0.2946 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/__pycache__`：53 个文件，1.2896 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_highs/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_lsq/__pycache__`：8 个文件，0.1351 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_shgo_lib/__pycache__`：3 个文件，0.0702 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_trlib/__pycache__`：1 个文件，0.0009 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/optimize/_trustregion_constr/__pycache__`：8 个文件，0.1128 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/__pycache__`：31 个文件，0.5053 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/csgraph/__pycache__`：3 个文件，0.0317 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/__pycache__`：13 个文件，0.2060 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_dsolve/__pycache__`：3 个文件，0.0356 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/__pycache__`：2 个文件，0.0211 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/arpack/__pycache__`：2 个文件，0.0673 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_eigen/lobpcg/__pycache__`：2 个文件，0.0405 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/sparse/linalg/_isolve/__pycache__`：9 个文件，0.1223 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/spatial/__pycache__`：10 个文件，0.1850 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/spatial/transform/__pycache__`：4 个文件，0.0285 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/special/__pycache__`：16 个文件，0.2937 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/__pycache__`：54 个文件，2.8931 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_levy_stable/__pycache__`：1 个文件，0.0516 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_rcont/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/scipy/stats/_unuran/__pycache__`：1 个文件，0.0002 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/__check_build/__pycache__`：1 个文件，0.0024 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/__pycache__`：9 个文件，0.2519 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/_loss/__pycache__`：3 个文件，0.0558 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/compose/__pycache__`：3 个文件，0.0908 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/covariance/__pycache__`：6 个文件，0.1283 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/decomposition/__pycache__`：12 个文件，0.3549 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/ensemble/__pycache__`：9 个文件，0.4161 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/ensemble/_hist_gradient_boosting/__pycache__`：5 个文件，0.1352 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/externals/__pycache__`：1 个文件，0.0003 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/externals/_packaging/__pycache__`：3 个文件，0.0255 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/impute/__pycache__`：3 个文件，0.0603 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/linear_model/__pycache__`：17 个文件，0.7096 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/linear_model/_glm/__pycache__`：3 个文件，0.0526 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/__pycache__`：7 个文件，0.4100 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/_pairwise_distances_reduction/__pycache__`：2 个文件，0.0301 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/_plot/__pycache__`：6 个文件，0.0753 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/metrics/cluster/__pycache__`：4 个文件，0.0712 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/model_selection/__pycache__`：6 个文件，0.3561 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/neighbors/__pycache__`：10 个文件，0.2075 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/preprocessing/__pycache__`：8 个文件，0.3429 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/svm/__pycache__`：4 个文件，0.1152 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/tree/__pycache__`：4 个文件，0.1287 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/sklearn/utils/__pycache__`：36 个文件，0.5956 MB
- `tmp/venvs/ocean_ab_20260517/Lib/site-packages/xgboost/__pycache__`：15 个文件，0.4877 MB

### 说明

- 本次未移动或删除原始数据、外部下载数据、历史实验结果、历史配置、凭证或认证文件。
- 所有整理建议已写入 current/ 报告。
