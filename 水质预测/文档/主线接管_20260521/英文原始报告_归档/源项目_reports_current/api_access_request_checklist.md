# 免费 API / 账号清单

## nasa_earthdata_oceancolor

1. API / 数据源名称：nasa_earthdata_oceancolor
2. 官方入口：https://oceancolor.gsfc.nasa.gov/
3. 是否免费：是
4. 是否需要账号：是
5. 是否需要 token：通常不需要单独 token，但需要免费账号登录
6. 是否有调用配额：取决于官方免费账户策略
7. 我需要提供什么：本机 `.env` 中的免费账号用户名/密码
8. 这个数据源能补充什么机制变量：oc_chl,kd490,turbidity/ocean color proxies
9. 预计对哪些任务有帮助：nasa_earthdata_oceancolor
10. 不提供该 API 时是否有替代数据源：可继续用当前 Copernicus / GEE / HKO 路线，但 OceanColor 这条线会暂停。

## 2026-05-20 更新

- NASA Earthdata 账号已通过本地 `.env` 配置，后续不再需要额外 token；但仍需保持账号信息只存在于本地未追踪环境中。
