# Coastal-distance 特征验证报告

## 1. 数据来源

- 来源：Natural Earth 10m coastline（公开免费）
- 官方入口：https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip
- 类型：静态海岸线矢量，不是逐日海岸观测

## 2. 特征口径

- `distance_to_coast_km`: 站点到最近海岸线的平面距离（km）
- `coastal_exposure_bucket`: 近岸暴露分组
- 可解释为近岸/外海、内湾/开放水域差异代理

## 3. 覆盖情况

- 站点覆盖率：1.0000
- 观测覆盖率：1.0000

## 4. 结论

- 该来源满足 source validation，可进入四指标统一单源 A/B。
- 它不是 GEBCO bathymetry，也不能替代 station depth / bottom profile / current 等真实机制变量。