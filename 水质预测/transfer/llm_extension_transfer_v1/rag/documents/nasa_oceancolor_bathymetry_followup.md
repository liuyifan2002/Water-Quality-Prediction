# NASA OceanColor / Bathymetry 跟进结果

## NASA Earthdata / OB.DAAC OceanColor

- 凭据来源：本地 `.env`（gitignore 已覆盖）。
- 登录状态：login_ok
- 本轮验证产品：VIIRS SNPP L3m monthly CHL / KD490 9km。
- 验证样本文件：[nasa_oceancolor_validation_sample_201401.csv](C:/Users/刘逸凡/.codex/worktrees/9ff7/codex/水质预测/data/processed/current/nasa_oceancolor_validation_sample_201401.csv)
- 结论：NASA OceanColor 月尺度 9km 验证样本下载与站点提取成功，可进入后续全时段批量化实现。

## GEBCO

- gebco_bathymetry_followup: 当前 depth_bathymetry 表仍为空值为主，GEBCO 需要下一轮单独做稳定子集化。

## Hong Kong Hydrographic Office / Marine Department

- hk_hydrographic_office_bathymetry: 本轮仅确认官方入口存在，未找到可直接批量并入模型的免费水深数值产品。
