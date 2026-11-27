# TP_FAMILY_V2_UNIFIED_STATION_SPLIT

- status: **SPLIT_APPROVED**
- generated: `2026-09-21T20:00:54`
- master table: `E:\研究生\codex\水质预测\数据\处理后数据\正式主表\model_table_3depth_2014_2024.csv`
- strict feature count: `77`
- station counts: surface=94, middle=84, bottom=91, common=84
- seed: `20260921`; sklearn: `1.6.1`; Python: `3.11.9`
- sorted common station IDs lexicographically; one KFold call on the unique station list; all three depths reuse the frozen map.
- target filter: valid finite `tp_mg_l`, non-null obs_id/sample_id/station/date and matching depth.

## Fold summary

```text
      task   depth   fold  train_rows  test_rows  train_stations  test_stations  overlap status
tp_surface surface fold_1        7366       1941              67             17        0   PASS
tp_surface surface fold_2        7363       1944              67             17        0   PASS
tp_surface surface fold_3        7540       1767              67             17        0   PASS
tp_surface surface fold_4        7416       1891              67             17        0   PASS
tp_surface surface fold_5        7543       1764              68             16        0   PASS
 tp_middle  middle fold_1        7059       1941              67             17        0   PASS
 tp_middle  middle fold_2        7117       1883              67             17        0   PASS
 tp_middle  middle fold_3        7354       1646              67             17        0   PASS
 tp_middle  middle fold_4        7170       1830              67             17        0   PASS
 tp_middle  middle fold_5        7300       1700              68             16        0   PASS
 tp_bottom  bottom fold_1        7366       1941              67             17        0   PASS
 tp_bottom  bottom fold_2        7363       1944              67             17        0   PASS
 tp_bottom  bottom fold_3        7540       1767              67             17        0   PASS
 tp_bottom  bottom fold_4        7416       1891              67             17        0   PASS
 tp_bottom  bottom fold_5        7543       1764              68             16        0   PASS
```

`06_split_validation.csv` must be entirely PASS before model execution.
