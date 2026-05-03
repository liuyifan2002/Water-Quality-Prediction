# 香港近岸水质预测最小可跑版本

本项目先做一个能跑通的 v0 流程，不同时展开 ACOLITE、C2RCC、Sentinel-2、三层水体、全部站点、SHAP 和完整消融实验。

## v0 范围

- 目标变量：`Chl_a`、`SD`、`TN`、`TP`
- 水层：表层，EPD 字段为 `Surface Water`
- 时间：2014-2024
- 代表站点：`DM1, NM3, VM5, WM2, SM6, TM4, PM7, MM8`
- 模型：XGBoost
- 第一版输入：月份、ERA5 气象、HKO 降水、HKO 潮汐、VIIRS 夜光、Landsat 8/9 月尺度反射率与指数

配置文件在 `config/minimal_v0.json`，如果要换站点或数据路径，优先改这个文件。

## 环境依赖

```powershell
python -m pip install -r requirements.txt
```

## 当前执行顺序

1. 整理 EPD 表层水质数据：

   ```powershell
   python scripts/01_prepare_water_quality_surface.py
   ```

   输出：

   - `data/processed/water_quality_surface_2014_2024.csv`
   - `data/processed/gee_station_months_surface_2014_2024.csv`

2. 下载并整理 ERA5 月尺度气象，生成：

   - `data/processed/era5_monthly_2014_2024.csv`
   - 字段：`year, month, t2m, u10, v10, tcc`

   配好 CDS API 后运行：

   ```powershell
   python scripts/02_download_prepare_era5_monthly.py
   ```

3. 整理 HKO 降水，生成：

   - `data/processed/rainfall_features_2014_2024.csv`
   - 字段：`date, rain_3d, rain_14d, rain_month`

   已使用 HKO 日雨量 CSV：

   ```powershell
   python scripts/03_prepare_hko_rainfall_features.py
   ```

4. 整理 HKO 潮汐，生成：

   - `data/processed/tide_features_2014_2024.csv`
   - 字段：`date, tidal_range`

   已使用 HKO Quarry Bay `HLT` 潮汐表：

   ```powershell
   python scripts/04_prepare_hko_tide_features.py
   ```

5. 将 `data/processed/gee_station_months_surface_2014_2024.csv` 上传到 GEE asset，然后运行：

   - `scripts/06_extract_viirs_ntl_monthly_gee.js`
   - `scripts/05_extract_landsat_features_gee.js`

   对应导出文件放回 `data/processed`：

   - `viirs_ntl_station_monthly_2014_2024.csv`
   - `landsat_station_monthly_2014_2024.csv`

6. 合并总数据表：

   ```powershell
   python scripts/07_merge_model_dataset.py
   ```

   输出：

   - `data/processed/model_dataset_surface_2014_2024.csv`

7. 跑第一个 Chl-a XGBoost：

   ```powershell
   python scripts/08_train_xgboost_chla.py
   ```

   输出：

   - `reports/xgboost_chla_surface_metrics.csv`
   - `reports/xgboost_chla_surface_predictions.csv`

8. 跑 Chl-a 的 F1/F2/F3 消融：

   ```powershell
   python scripts/09_train_xgboost_ablation_chla.py
   ```

   输出：

   - `reports/xgboost_chla_surface_ablation_metrics.csv`

## 暂缓事项

Sentinel-2、更多站点、中层/底层、SHAP、空间外推、ACOLITE/C2RCC 都等 Landsat v0 跑通后再加。
