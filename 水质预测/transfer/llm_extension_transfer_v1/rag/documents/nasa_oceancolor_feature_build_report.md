# NASA OceanColor 特征构建报告

## 构建范围

- 时间范围：2014-01 至 2024-12。
- 时间尺度：monthly L3m。
- 空间提取：nearest valid pixel，必要时 fallback 到 3x3 / 5x5 median。
- NASA OceanColor 被明确视为表层光学遥感代理，不是底层实测数据。

## 产品执行情况

- `VIIRSN_L3m_CHL`: status=`skip_download_use_local`, discovered=132, downloaded=132, rate=1.0000。
- `VIIRSN_L3m_KD`: status=`skip_download_use_local`, discovered=132, downloaded=132, rate=1.0000。
- `VIIRSN_L3m_RRS`: status=`deferred_structure_complex`, discovered=913, downloaded=913, rate=6.9167。
- `MODISA_L3m_CHL`: status=`deferred_sensor_control`, discovered=19, downloaded=19, rate=0.1439。

## 产物规模

- 长表行数：`24816`
- 宽表行数：`12408`
- 宽表列数：`23`
- 是否纳入 RRS：`False`
