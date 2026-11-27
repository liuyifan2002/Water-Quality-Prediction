# GEBCO / coastal-distance 线 A/B 报告

## 1. 本轮口径

- `GEBCO`：本轮完成官方 source validation，但自动数值子集化仍不稳定，未进入正式入模。
- `coastal_distance_only`：使用公开 coastline 生成站点到海岸距离特征。
- `station_depth_plus_coastal_distance`：把已验证的官方站点水深与 coastal-distance 联合使用。

## 2. Top delta R²

- station_depth_plus_coastal_distance | chla bottom: delta R² = +0.0125, candidate R² = 0.3561
- coastal_distance_only | tn surface: delta R² = +0.0043, candidate R² = 0.5245
- coastal_distance_only | chla middle: delta R² = +0.0026, candidate R² = 0.3672
- coastal_distance_only | tp middle: delta R² = -0.0001, candidate R² = 0.0738
- coastal_distance_only | secchi middle: delta R² = -0.0004, candidate R² = 0.3479
- station_depth_plus_coastal_distance | tp middle: delta R² = -0.0029, candidate R² = 0.0711
- station_depth_plus_coastal_distance | secchi middle: delta R² = -0.0029, candidate R² = 0.3453
- coastal_distance_only | tn bottom: delta R² = -0.0031, candidate R² = 0.2322
- station_depth_plus_coastal_distance | chla middle: delta R² = -0.0039, candidate R² = 0.3606
- coastal_distance_only | tp surface: delta R² = -0.0039, candidate R² = 0.3946
- station_depth_plus_coastal_distance | tp surface: delta R² = -0.0094, candidate R² = 0.3892
- station_depth_plus_coastal_distance | tn bottom: delta R² = -0.0109, candidate R² = 0.2244
- coastal_distance_only | tn middle: delta R² = -0.0152, candidate R² = 0.1672
- station_depth_plus_coastal_distance | tn surface: delta R² = -0.0156, candidate R² = 0.5046
- coastal_distance_only | tp bottom: delta R² = -0.0186, candidate R² = 0.0141
- coastal_distance_only | secchi bottom: delta R² = -0.0215, candidate R² = 0.3359
- station_depth_plus_coastal_distance | secchi bottom: delta R² = -0.0290, candidate R² = 0.3285
- station_depth_plus_coastal_distance | secchi surface: delta R² = -0.0314, candidate R² = 0.3766