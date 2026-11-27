# Chl-a 新增数据下载与制备指南

## 1. 当前策略

当前 `lagged_cross_water_quality` 单源 A/B 已完成，结果显示它会同时拉低 `surface / middle / bottom` 的总体 R2 和 P90 高值表现，因此这条线暂时停止，不再作为 Chl-a 增强主线优先方向。

现阶段优先级改为：

1. Copernicus Marine Ocean Colour
2. Copernicus Marine Physics
3. Copernicus Marine BGC
4. EPD 长历史水质补链
5. NASA OceanColor 作为后备方案

## 2. 为什么优先 Ocean Colour

Ocean Colour 产品能直接提供对 Chl-a 更贴近过程的外部信息，重点包括：

1. `CHL`：外部叶绿素背景状态
2. `KD490`：光衰减强度，反映水体透明度与光场条件
3. `SPM`：悬浮颗粒物，反映再悬浮、河流输入和浑浊扰动
4. `ZSD`：透明度/塞氏深度代理
5. `RRS` 波段：用于构造红边和水色指数

相比之下，GEE 里的 `COPERNICUS/S3/OLCI` 更偏 TOA radiance / 原始观测层，不是本阶段最优的正式输入源。它仍可用于辅助探索，但不作为当前第一优先正式数据入口。

## 3. 需要哪些账号

### 必需

1. `Copernicus Marine` 账号
2. 若后续运行 GEE 辅助脚本，则还需要 `Earth Engine` 项目

### 当前不是必需

1. `NASA Earthdata` 账号

## 4. 如何配置 `.env`

在仓库根目录新建 `.env`，变量名参见 [`.env.example`](C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/.env.example)。

建议至少配置：

```bash
COPERNICUSMARINE_USERNAME=your_username
COPERNICUSMARINE_PASSWORD=your_password
EARTHENGINE_PROJECT=your-ee-project
NASA_EARTHDATA_USERNAME=
NASA_EARTHDATA_PASSWORD=
```

不要把真实账号、密码或 Token 写进脚本，也不要提交 `.env`。

## 5. 如何发现 Ocean Colour dataset_id

先运行产品发现脚本：

```bash
python scripts/29_discover_cmems_ocean_colour_products.py --dry-run --date-tag 20260516
```

输出目录：

`outputs/modeling/experiments/cmems_ocean_colour_discovery_20260516/`

重点看：

1. `cmems_ocean_colour_candidate_datasets.csv`
2. `cmems_ocean_colour_variable_inventory.csv`
3. `cmems_ocean_colour_recommendation.json`
4. `cmems_ocean_colour_discovery_report.md`

当前默认推荐：

1. 日尺度主候选：`cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D`
2. 月尺度 fallback：`cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M`

## 6. 如何下载 Ocean Colour

### 6.1 dry-run

```bash
python scripts/30_download_cmems_ocean_colour_features.py --dry-run --date-tag 20260516
```

### 6.2 真实运行

```bash
python scripts/30_download_cmems_ocean_colour_features.py \
  --date-tag 20260516 \
  --dataset-id cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D
```

### 6.3 使用本地已有 NetCDF 只做提取

```bash
python scripts/30_download_cmems_ocean_colour_features.py \
  --date-tag 20260516 \
  --dataset-id cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D \
  --skip-download
```

### 6.4 指定变量

```bash
python scripts/30_download_cmems_ocean_colour_features.py \
  --date-tag 20260516 \
  --dataset-id cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D \
  --variables CHL,KD490,SPM,ZSD,RRS443,RRS490,RRS510,RRS560,RRS665,RRS681,RRS709,PP
```

## 7. 总控脚本怎么跑

仅运行 Ocean Colour 相关步骤：

```bash
python scripts/27_download_all_new_chla_sources.py \
  --dry-run \
  --date-tag 20260516 \
  --run-cmems-ocean-colour
```

真实运行：

```bash
python scripts/27_download_all_new_chla_sources.py \
  --date-tag 20260516 \
  --run-cmems-ocean-colour \
  --cmems-ocean-colour-dataset-id cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D
```

## 8. 下载后的文件在哪里

### 原始 NetCDF

`data/raw/manual/ocean_color/cmems/`

### 标准化输入文件

1. `data/model_input/new_sources/ocean_color_kd490_tss_features_2014_2024.csv`
2. `data/model_input/new_sources/sentinel3_olci_features_2014_2024.csv`

## 9. 如何运行校验

```bash
python scripts/21_validate_new_chla_data_sources.py --date-tag 20260516
```

校验输出目录：

`outputs/modeling/experiments/chla_new_data_source_validation_20260516/`

重点看：

1. `new_data_source_readiness_table.csv`
2. `new_data_source_field_summary.csv`
3. `new_data_source_missing_rate.csv`
4. `new_data_source_validation_report.json`

## 10. 如何判断是否可以进入 A/B 实验

当 `ocean_color_kd490_tss` 或 `sentinel3_olci` 满足以下条件时，可以进入单源 A/B：

1. `readiness_level = ready_for_ab_experiment`
2. 存在 `date` 和 `station`
3. 时间范围覆盖 `2014-2024`
4. `station_coverage_rate` 可接受
5. 关键字段不是全空列
6. `oc_valid` 或 `olci_valid` 比例不是 0

如果只有月尺度产品，则可以进入“背景状态增强实验”，但要在报告里明确它不适合解释短期事件。

## 11. 哪些大文件不要提交 Git

以下文件已在 `.gitignore` 排除：

1. `.env`
2. `*.netrc`
3. `data/raw/**/*.nc`
4. `data/raw/**/*.zip`
5. `tmp_downloads/`
6. `tmp_env_source/`

## 12. 常见错误

### 未配置 Copernicus Marine 账号

说明 `.env` 缺少 `COPERNICUSMARINE_USERNAME` 或 `COPERNICUSMARINE_PASSWORD`。

### 未提供 dataset_id

说明 discovery 推荐结果不存在，且命令行也没有显式传入 `--dataset-id`。

### 未找到本地 NetCDF

说明你使用了 `--skip-download`，但原始 `NetCDF` 还没有落到 `data/raw/manual/ocean_color/cmems/`。

### 校验仍然显示 template_only

说明真实输出 CSV 还没有生成，或者输出文件名与规范不一致。

### 月尺度 fallback

说明日尺度产品不可用，脚本退回月尺度。此时可用于背景水色状态增强，但不宜直接解释短期藻华事件。
