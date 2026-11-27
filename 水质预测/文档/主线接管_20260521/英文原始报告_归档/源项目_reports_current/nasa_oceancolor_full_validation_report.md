# NASA OceanColor 全时段验证报告

- 2014-2024 月尺度是否完整覆盖：`True`
- 全体变量总体缺失率：`0.0529`
- 全体变量总体提取成功率：`0.9471`

## 各产品下载与提取状态

### VIIRSN_L3m_CHL

- 下载状态：`skip_download_use_local`
- 发现 granules 数：`132`
- 成功下载文件数：`132`
- 下载成功率：`1.0000`
- 说明：使用本地已有文件

### VIIRSN_L3m_KD

- 下载状态：`skip_download_use_local`
- 发现 granules 数：`132`
- 成功下载文件数：`132`
- 下载成功率：`1.0000`
- 说明：使用本地已有文件

### VIIRSN_L3m_RRS

- 下载状态：`deferred_structure_complex`
- 发现 granules 数：`913`
- 成功下载文件数：`913`
- 下载成功率：`6.9167`
- 说明：RRS 当前为按波段拆分文件结构，先不在本轮全量并入；待 CHL/KD 单源 A/B 给出明确正信号后再单独扩展。

### MODISA_L3m_CHL

- 下载状态：`deferred_sensor_control`
- 发现 granules 数：`19`
- 成功下载文件数：`19`
- 下载成功率：`0.1439`
- 说明：MODIS Aqua monthly CHL 当前只做传感器对照；在 NASA 主线月尺度 CHL/KD 单源 A/B 明确前，本轮暂不并入正式特征构建。

## 建议

- 建议先以 `VIIRSN_L3m_CHL` 和 `VIIRSN_L3m_KD` 进入单源 A/B。
- `VIIRSN_L3m_RRS` 仅在变量结构稳定、缺失率可接受时纳入。
- `MODISA_L3m_CHL` 主要用于传感器对照，不应喧宾夺主。
- NASA OceanColor 是表层光学代理，不应表述为底层实测数据。
