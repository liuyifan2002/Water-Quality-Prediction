# GPM IMERG observation-date 对齐验证报告

本轮只做样本下载与 observation-date 对齐验证，不构建正式 runoff 特征，不进入 A/B。

## 核心结果
- 抽样日期数：6
- 成功下载 granule 数：0
- 使用变量：`未识别`
- 总体提取成功率：`0.0000`
- 平均纬度网格偏差：未计算
- 平均经度网格偏差：未计算

## 日期级结果

      date  granule_found  download_ok  open_ok  extraction_success_rate                        note
2014-01-02           True        False    False                      0.0 earthdata_eula_not_accepted
2016-03-10           True        False    False                      0.0 earthdata_eula_not_accepted
2018-06-06           True        False    False                      0.0 earthdata_eula_not_accepted
2020-10-15           True        False    False                      0.0 earthdata_eula_not_accepted
2022-12-08           True        False    False                      0.0 earthdata_eula_not_accepted
2024-12-30           True        False    False                      0.0 earthdata_eula_not_accepted

## 判断
- 若抽样 granule 均可成功下载并完成站点最近邻提取，则说明 GPM IMERG 已从“可发现”推进到“可用于 observation-date 对齐验证”。
- 若仍被 GES DISC 条款拦截，则说明登录已通，但实际数据下载链仍未打通；此时不建议进入正式 runoff proxy 特征构建。
