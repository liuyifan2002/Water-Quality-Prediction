# 拒绝或暂缓的数据源

- **gebco_bathymetry**（reject）：本轮未获得可用的 marine bathymetry 数值，不能作为正式输入。
- **nasa_earthdata_oceancolor**（needs_user_api）：缺少 NASA Earthdata 免费账号配置，先生成清单。
- **noaa**（reject）：当前优先级低且缺乏明确研究区匹配路径，不纳入本轮。

## 2026-05-20 新增拒绝/暂缓项

- `gebco_bathymetry_followup`：本轮未建立稳定 GEBCO 子集化与站点提取流程。
- `hk_hydrographic_office_bathymetry`：确认存在官方海图/潮流服务，但未找到可直接自由批量并入模型的免费水深数值产品。
