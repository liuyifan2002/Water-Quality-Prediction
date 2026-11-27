# Chl-a bottom 残差诊断

## 关键结论

- 高值区间（>P90）underestimation_rate：0.912
- 61-90d 采样间隔 RMSE：5.758

## 误差最重的站点（Top 5 by RMSE）

- ET2: RMSE=12.486, MAE=5.315, bias=-2.332, n=12
- VT11: RMSE=9.883, MAE=5.524, bias=-2.469, n=18
- SM10: RMSE=7.431, MAE=6.396, bias=0.083, n=24
- VT3: RMSE=6.971, MAE=4.794, bias=-0.273, n=12
- ST3: RMSE=6.316, MAE=5.566, bias=1.914, n=12

## 误差最重的月份（Top 5 by RMSE）

- 7: RMSE=6.284, bias=1.191, n=180
- 9: RMSE=4.671, bias=-0.499, n=180
- 6: RMSE=4.280, bias=0.613, n=148
- 8: RMSE=3.940, bias=0.846, n=148
- 5: RMSE=3.910, bias=0.193, n=180

## 解释判断

- 当前 bottom 误差仍明显集中在高值段和部分站点，且高值样本存在系统低估。
- Copernicus physics 对 bottom 的负迁移，更像是 coarse reanalysis 不能代表真实近底混合和局地输运，而不是该类变量完全无信息。
- high-quality patch 对 bottom 没有形成稳定帮助，说明 bottom 更像机制数据缺口，不像单纯光学状态缺口。
- 如果没有 exact_sampling_datetime、station_depth、底层温度或垂向剖面，bottom 很难突破到更高 R² 档位。
