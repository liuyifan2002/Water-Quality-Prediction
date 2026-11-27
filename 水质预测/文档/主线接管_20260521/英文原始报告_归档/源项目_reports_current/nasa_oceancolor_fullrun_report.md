# NASA OceanColor 全时段运行报告

## 本轮是否完成 2014-2024 全时段特征构建

- 完成状态：`True`

## 产品执行情况

- `VIIRSN_L3m_CHL`: status=`skip_download_use_local`, discovered=132, downloaded=132, rate=1.0000。
- `VIIRSN_L3m_KD`: status=`skip_download_use_local`, discovered=132, downloaded=132, rate=1.0000。
- `VIIRSN_L3m_RRS`: status=`deferred_structure_complex`, discovered=913, downloaded=913, rate=6.9167。
- `MODISA_L3m_CHL`: status=`deferred_sensor_control`, discovered=19, downloaded=19, rate=0.1439。

## 单源 A/B 结果

- `secchi middle / nasa_chl_kd`: delta R² = -0.0082, delta RMSE = 0.0058, delta MAE = 0.0050, validation delta R² = -0.0278, promote = `False`。
- `secchi bottom / nasa_chl_kd`: delta R² = -0.0114, delta RMSE = 0.0079, delta MAE = 0.0080, validation delta R² = -0.0065, promote = `False`。
- `secchi middle / nasa_chl_kd_lag1`: delta R² = -0.0131, delta RMSE = 0.0092, delta MAE = 0.0086, validation delta R² = -0.0197, promote = `False`。
- `secchi bottom / nasa_chl_kd_lag1`: delta R² = -0.0167, delta RMSE = 0.0116, delta MAE = 0.0131, validation delta R² = -0.0031, promote = `False`。
- `chla middle / nasa_chl_kd`: delta R² = -0.0212, delta RMSE = 0.0641, delta MAE = -0.1090, validation delta R² = -0.0504, promote = `False`。
- `secchi middle / nasa_chl_kd_rolling3`: delta R² = -0.0232, delta RMSE = 0.0163, delta MAE = 0.0129, validation delta R² = -0.0225, promote = `False`。
- `chla bottom / nasa_chl_kd_rolling3`: delta R² = -0.0282, delta RMSE = 0.0742, delta MAE = -0.0529, validation delta R² = -0.0052, promote = `False`。
- `chla middle / nasa_chl_kd_lag1`: delta R² = -0.0301, delta RMSE = 0.0907, delta MAE = -0.0922, validation delta R² = -0.0535, promote = `False`。
- `chla middle / nasa_chl_kd_rolling3`: delta R² = -0.0376, delta RMSE = 0.1131, delta MAE = -0.1001, validation delta R² = -0.0472, promote = `False`。
- `chla bottom / nasa_chl_kd`: delta R² = -0.0390, delta RMSE = 0.1021, delta MAE = -0.0316, validation delta R² = -0.0165, promote = `False`。
- `secchi surface / nasa_chl_kd`: delta R² = -0.0392, delta RMSE = 0.0285, delta MAE = 0.0130, validation delta R² = 0.0086, promote = `False`。
- `secchi bottom / nasa_chl_kd_rolling3`: delta R² = -0.0438, delta RMSE = 0.0301, delta MAE = 0.0239, validation delta R² = 0.0018, promote = `False`。
- `chla bottom / nasa_chl_kd_lag1`: delta R² = -0.0525, delta RMSE = 0.1368, delta MAE = -0.0259, validation delta R² = -0.0203, promote = `False`。
- `secchi surface / nasa_chl_kd_lag1`: delta R² = -0.0567, delta RMSE = 0.0410, delta MAE = 0.0249, validation delta R² = 0.0054, promote = `False`。
- `secchi surface / nasa_chl_kd_rolling3`: delta R² = -0.0729, delta RMSE = 0.0523, delta MAE = 0.0285, validation delta R² = 0.0063, promote = `False`。
- `chla surface / nasa_chl_kd_lag1`: delta R² = -0.0959, delta RMSE = 0.3594, delta MAE = 0.1972, validation delta R² = -0.0295, promote = `False`。
- `chla surface / nasa_chl_kd`: delta R² = -0.1066, delta RMSE = 0.3982, delta MAE = 0.2614, validation delta R² = -0.0121, promote = `False`。
- `chla surface / nasa_chl_kd_rolling3`: delta R² = -0.1081, delta RMSE = 0.4034, delta MAE = 0.2264, validation delta R² = -0.0181, promote = `False`。

## Chl-a bottom 结论

- 最佳 NASA 候选为 `nasa_chl_kd_rolling3`，delta R² = -0.0282。
- NASA OceanColor 仍然是表层光学代理；若 bottom 没有明显提升，主要说明底层过程仍受垂向结构、底层温度、流速和再悬浮控制。

## Secchi 结论

- `Secchi middle` 最佳 NASA 候选 `nasa_chl_kd`: delta R² = -0.0082。
- `Secchi bottom` 最佳 NASA 候选 `nasa_chl_kd`: delta R² = -0.0114。
- `Secchi surface` 最佳 NASA 候选 `nasa_chl_kd`: delta R² = -0.0392。

## 是否 promote 到 current best

- promote 数量：`0`
- 本轮没有任何 NASA 候选通过 promote 审计。

## 下一步建议

- 如果月尺度已经出现清晰正信号，再考虑升级到 8-day / daily。
- 如果月尺度无效或高缺失，应优先继续申请 `exact_sampling_datetime`、`station_depth`、`bottom profile`、`ADCP / water level`，而不是盲目升频。
