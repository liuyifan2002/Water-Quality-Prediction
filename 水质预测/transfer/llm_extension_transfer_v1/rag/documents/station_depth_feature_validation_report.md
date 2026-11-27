# 香港 EPD 站点水深特征验证报告

## 1. 数据来源

- 来源：香港 EPD 2002 年 marine water quality 年报附录 A
- 性质：官方报告中的 **站点近似总水深**
- 适用口径：`station depth / relative depth proxy`
- 不应表述为：GEBCO 栅格 bathymetry、逐日底层观测、ADCP 或垂向剖面

## 2. 覆盖情况

- 站点总数：94
- 成功匹配近似水深的站点数：94
- 站点覆盖率：1.0000
- 当前观测样本总数：30719
- 当前观测样本覆盖数：30719
- 当前观测样本覆盖率：1.0000

## 3. 风险与限制

1. 该表给的是近似总水深，不是高分辨率 bathymetry 栅格。
2. 该表不能替代 station exact sampling datetime、bottom temperature、ADCP current 等真实底层机制数据。
3. 该表可以帮助四指标模型学习“深浅 / 内湾外湾 / 水体交换潜势”的长期静态差异。

## 4. 代表性站点

### 较浅站点示例

- DM1: 2.0 m
- DM2: 2.0 m
- DM3: 3.0 m
- PT2: 3.0 m
- NT1: 4.0 m
- TM2: 4.0 m
- DM4: 4.0 m
- TM5: 4.0 m

### 较深站点示例

- VM1: 38.0 m
- WM1: 35.0 m
- NM1: 34.0 m
- SM3: 33.0 m
- MM8: 31.0 m
- MM13: 28.0 m
- MM19: 28.0 m
- WM4: 26.0 m

## 5. 结论

- 该来源已经足够进入四指标统一单源 A/B，优先验证其对 TN surface / TP surface / TN middle / TN bottom 的增益。
- 若后续拿到 GEBCO 或更细的香港海图/测深数值产品，应把它作为独立 bathymetry 数据源再次做 source validation。