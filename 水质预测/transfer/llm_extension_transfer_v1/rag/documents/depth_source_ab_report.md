# 站点水深单源 A/B 报告

## 1. 本轮目标

- 使用香港 EPD 官方附录中的站点近似总水深，验证其能否作为四指标统一增益源。
- 重点关注 TN surface，同时同步比较 TP surface、TN middle、TN bottom 以及其余八个任务。

## 2. 数据口径

- 新增源：`station_depth_approx_m`，来源于香港 EPD 2002 附录 A。
- 该变量是站点近似总水深，不是 GEBCO gridded bathymetry，也不是逐日底层观测。
- 为了让模型可利用层位信息，本轮额外构造了：`layer_fraction_proxy`、`estimated_sample_depth_m_proxy`、`bottom_gap_m_proxy`。

## 3. 四指标统一结果概览

- tn bottom: baseline R²=0.2353, candidate R²=0.2467, delta=+0.0114
- chla bottom: baseline R²=0.3436, candidate R²=0.3519, delta=+0.0083
- chla middle: baseline R²=0.3646, candidate R²=0.3624, delta=-0.0021
- tn surface: baseline R²=0.5201, candidate R²=0.5162, delta=-0.0040
- secchi middle: baseline R²=0.3482, candidate R²=0.3439, delta=-0.0043
- tn middle: baseline R²=0.1823, candidate R²=0.1712, delta=-0.0111
- tp surface: baseline R²=0.3986, candidate R²=0.3862, delta=-0.0124
- tp middle: baseline R²=0.0739, candidate R²=0.0616, delta=-0.0124
- secchi bottom: baseline R²=0.3574, candidate R²=0.3276, delta=-0.0298
- secchi surface: baseline R²=0.4080, candidate R²=0.3705, delta=-0.0375
- tp bottom: baseline R²=0.0328, candidate R²=-0.0050, delta=-0.0377
- chla surface: baseline R²=0.3298, candidate R²=0.2860, delta=-0.0439

## 4. 解释边界

- 若 station depth 对 TN/TP 有增益，解释应偏向“空间分区 / 深浅差异 / 交换潜势”，而不是底层实测过程。
- 若对 Chl-a bottom 仍帮助有限，则进一步支持“需要真实 bottom temperature / current / profile / sampling time”的判断。