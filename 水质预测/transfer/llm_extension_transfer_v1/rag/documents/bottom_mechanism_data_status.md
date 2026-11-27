# Chl-a bottom 机制变量状态

## 1. 当前文件

- 文件：`data/processed/current/bottom_mechanism_features_2014_2024.csv`
- 当前可用 mechanism 字段数：28

## 2. 原始输入来源

当前 bottom mechanism 表主要由以下三类环境表按 `date` 单键合并后生成：

- `data/raw/manual/environment/era5_features_2014_2024.csv`
- `data/raw/manual/environment/rainfall_features_2014_2024.csv`
- `data/raw/manual/environment/tide_features_2014_2024.csv`

这意味着这些值反映的是同日区域背景环境，不是站点专属底层实测机制变量。

## 3. 当前已经真正进入表中的弱代理变量

### 温度 / 热状态

- `surface_temp = sst_mean_3d`
- `temp_7d_mean = sst_mean_7d`
- `temp_14d_mean = sst_mean_14d`
- `temp_30d_mean = sst_mean_14d`（当前实现近似，不是真实 30 天窗口）
- `temp_change_rate = sst_mean_3d - sst_mean_14d`
- `stratification_strength = sst_mean_7d / (abs(wind_speed) + 1.0)`

### 风混合

- `wind_speed_3d_mean = wind_speed`
- `wind_speed_7d_mean = wind_speed`
- `wind_speed_14d_mean = wind_speed`
- `wind_speed_3d_max = wind_speed`
- `wind_direction_u = u10`
- `wind_direction_v = v10`
- `wind_mixing_proxy = wind_speed * tidal_range`

说明：当前 `3d/7d/14d/max` 风速字段还不是独立滚动统计，而是对同一背景风速值的复用。

### 降雨扰动

- `precip_3d_sum = rain_3d`
- `precip_7d_sum = rain_7d`
- `storm_event_flag = float(rain_3d >= 50)`

### 潮汐背景

- `tide_range_proxy = tidal_range`

### 站点区域

- `zone = water_zone_en`

## 4. 当前缺失的强机制变量

以下变量依旧没有补齐，因此本轮 mechanism A/B 不能被表述为“真实底层机制变量验证完成”：

- `station_depth`
- `bathymetry_depth`
- `bottom_temp`
- `surface_bottom_temp_diff`
- `temp_gradient`
- `thermocline_proxy`
- `water_level`
- `water_level_change_3d`
- `discharge_7d_mean`
- `flushing_proxy`
- `current_speed`
- `current_speed_7d_mean`

## 5. 当前缺失率最高的字段

- `station_depth`: 100.00%
- `discharge_7d_mean`: 100.00%
- `water_level_change_3d`: 100.00%
- `water_level`: 100.00%
- `thermocline_proxy`: 100.00%
- `surface_bottom_temp_diff`: 100.00%
- `temp_gradient`: 100.00%
- `bottom_temp`: 100.00%
- `distance_to_river_mouth`: 100.00%

## 6. 结论

- 本轮 Chl-a bottom mechanism A/B 只能验证弱代理变量是否提供增益。
- 这些 proxy 更适合解释背景热状态、风混合、降雨扰动和潮差作用。
- 它们不能替代真实 `station_depth / bathymetry / bottom_temp / vertical profile / current / water level / discharge`。
