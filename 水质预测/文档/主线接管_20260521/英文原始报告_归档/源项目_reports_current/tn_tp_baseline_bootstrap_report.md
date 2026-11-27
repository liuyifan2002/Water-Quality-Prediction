# TN/TP baseline bootstrap 报告

本轮目标：在不引入新增模型大调参的前提下，确认 TN / TP 是否已经具备纳入四指标统一框架的 baseline 条件。

## TN

- `surface`: baseline ready，best=tn_surface_stable_core_lightgbm，validation R²=0.5820，test R²=0.5201，RMSE=0.3411，MAE=0.2473
- `middle`: baseline ready，best=tn_middle_baseline_lightgbm，validation R²=0.2703，test R²=0.1823，RMSE=0.2921，MAE=0.2399
- `bottom`: baseline ready，best=tn_bottom_mechanism_context_lightgbm，validation R²=0.2305，test R²=0.2353，RMSE=0.2759，MAE=0.2231

## TP

- `surface`: baseline ready，best=tp_surface_mechanism_context_random_forest，validation R²=0.6389，test R²=0.3986，RMSE=0.0531，MAE=0.0298
- `middle`: baseline ready，best=tp_middle_stable_core_lightgbm，validation R²=0.1725，test R²=0.0739，RMSE=0.0500，MAE=0.0308
- `bottom`: baseline ready，best=tp_bottom_baseline_random_forest，validation R²=0.2084，test R²=0.0328，RMSE=0.0517，MAE=0.0288
