# 向原始监测单位申请的数据清单

## 最高优先级（直接影响 Chl-a bottom）

| 字段 | 用途 | 优先级 | 对 Chl-a bottom 预期帮助 |
|---|---|---|---|
| exact_sampling_datetime | 将潮位/潮流/辐射/风场与真实采样时刻对齐，摆脱 date-level proxy | 最高 | 很高 |
| station_depth | 建立站点静态水深、浅滩/深槽差异与分层风险 | 最高 | 很高 |
| bottom_water_temperature | 直接刻画底层热状态 | 最高 | 很高 |
| vertical_temperature_profile | 构造 thermocline / stratification / mixing 指标 | 最高 | 很高 |
| ADCP current_u / current_v / current_speed | 近底输运与再悬浮过程 | 最高 | 很高 |
| water_level | 感潮/水动力边界条件 | 高 | 高 |
| discharge / inflow / outflow | 河口输入、冲刷和停留时间代理 | 高 | 高 |

## 中高优先级（机制补充）

| 字段 | 用途 | 优先级 | 对 Chl-a bottom 预期帮助 |
|---|---|---|---|
| salinity_profile | 分层与淡咸水入侵结构 | 中高 | 中高 |
| dissolved_oxygen_profile | 缺氧/分层稳定度侧证 | 中高 | 中高 |
| sampling_depth_m | 精确采样深度，而不是仅 surface/middle/bottom 标签 | 中高 | 中高 |
| station_depth_datum | 统一 chart datum / HKPD / station datum | 中高 | 中高 |
| nearest_tide_station_id | 采样站与潮位/潮流站映射 | 中 | 中 |

## 说明

1. 以上字段不要求来自 NASA；其中 exact_sampling_datetime、底层水温、垂向剖面、ADCP 流速和实测水位/流量通常必须向原始监测或配套观测单位申请。
2. 当前项目最缺的是“真实底层机制数据”，不是普通模型调参。
3. 如果只能优先申请一批，建议顺序为：`exact_sampling_datetime -> station_depth -> bottom_water_temperature / vertical_temperature_profile -> ADCP current -> water_level / discharge`。
