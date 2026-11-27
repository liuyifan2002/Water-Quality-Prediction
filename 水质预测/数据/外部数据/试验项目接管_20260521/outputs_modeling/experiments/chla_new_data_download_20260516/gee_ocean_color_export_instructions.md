# GEE 海洋水色导出说明

1. 已生成 GEE JS 脚本，请先把 `data/model_input/gee_station_date_samples.csv` 上传为 GEE 表资产。
2. 脚本位置：`C:\Users\刘逸凡\.codex\worktrees\9ff7\codex\水质预测\scripts\24_extract_gee_ocean_color_features.js`。
3. 将 `STATION_TABLE_ASSET`、`OLCI_DATASET_ID`、`KD490_DATASET_ID` 替换为实际值后，在 Earth Engine Code Editor 运行。
4. 真实导出完成后，将 CSV 标准化为 `data/model_input/new_sources/sentinel3_olci_features_2014_2024.csv` 和 `ocean_color_kd490_tss_features_2014_2024.csv`。
