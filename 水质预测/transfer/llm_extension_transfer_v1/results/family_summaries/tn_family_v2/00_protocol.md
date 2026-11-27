# TN_FAMILY_V2_UNIFIED_STATION_SPLIT 协议

- 最终状态：**SPLIT_APPROVED**
- 生成时间：`2026-09-21T01:02:50`
- 正式主表：`E:\研究生\codex\水质预测\数据\处理后数据\正式主表\model_table_3depth_2014_2024.csv`
- 冻结特征清单：`E:\研究生\codex\水质预测\runs\tabiclv2_pilot\01_strict_feature_manifest.csv`
- 实际冻结输入特征数：`77`
- 目标字段：`tn_mg_l`；有效目标定义为可解析且为有限数值。
- 水层：`surface`、`middle`、`bottom`。

## 共同站点范围

- surface 有效目标站点：`94`
- middle 有效目标站点：`84`
- bottom 有效目标站点：`91`
- 三层共同站点：`84`
- 单层缺少有效目标的站点保留在 `01_station_universe.csv`，不进入本协议共同站点。

## 固定分折规则

- 固定随机种子：`20260921`
- 分组单位：`station_id`；同一站点所有日期和三层观测使用同一折次。
- 站点排序：先将站点 ID 转为去除首尾空格的字符串，再按稳定字典序升序排列。
- 算法：对排序后的共同站点唯一列表只调用一次 `sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=20260921)`。
- sklearn 版本：`1.6.1`；Python：`3.11.9`。
- 三个水层只从 `02_tn_family_v2_station_fold_map.csv` 派生 manifest，不重新分折。

## 目标过滤与验证

- 每层保留 `depth` 匹配、`obs_id/sample_id/station_id/date` 非空、`tn_mg_l` 非空且有限的观测。
- 个别月份没有目标值不会删除该站点；只有整层没有任何有效目标时才会排除站点。
- `06_split_validation.csv` 的每一行必须为 `PASS`，且 `overlap` 必须为 0，才允许进入模型阶段。

## 输出

- `01_station_universe.csv`：三层站点统计及排除原因。
- `02_tn_family_v2_station_fold_map.csv`：永久 canonical station→fold 映射。
- `03_tn_surface_manifest.csv`、`04_tn_middle_manifest.csv`、`05_tn_bottom_manifest.csv`：三层 train/test obs_id 清单。
- `06_split_validation.csv`：逐检查项验证结果。

## 分折汇总

```text
      task   depth   fold  train_rows  test_rows  train_stations  test_stations  overlap status
tn_surface surface fold_1        7366       1941              67             17        0   PASS
tn_surface surface fold_2        7363       1944              67             17        0   PASS
tn_surface surface fold_3        7540       1767              67             17        0   PASS
tn_surface surface fold_4        7416       1891              67             17        0   PASS
tn_surface surface fold_5        7543       1764              68             16        0   PASS
 tn_middle  middle fold_1        7059       1941              67             17        0   PASS
 tn_middle  middle fold_2        7117       1883              67             17        0   PASS
 tn_middle  middle fold_3        7354       1646              67             17        0   PASS
 tn_middle  middle fold_4        7170       1830              67             17        0   PASS
 tn_middle  middle fold_5        7300       1700              68             16        0   PASS
 tn_bottom  bottom fold_1        7366       1941              67             17        0   PASS
 tn_bottom  bottom fold_2        7363       1944              67             17        0   PASS
 tn_bottom  bottom fold_3        7540       1767              67             17        0   PASS
 tn_bottom  bottom fold_4        7416       1891              67             17        0   PASS
 tn_bottom  bottom fold_5        7543       1764              68             16        0   PASS
```

## 验证结果

```text
      task   depth   fold  train_rows  test_rows  train_stations  test_stations  overlap  train_test_station_overlap_zero  train_test_obs_overlap_zero  test_station_matches_canonical_map  test_stations_match_other_tn_layers  train_obs_unique  test_obs_unique  train_target_nonmissing  test_target_nonmissing  station_assignment_matches_map status
tn_surface surface fold_1        7366       1941              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
tn_surface surface fold_2        7363       1944              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
tn_surface surface fold_3        7540       1767              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
tn_surface surface fold_4        7416       1891              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
tn_surface surface fold_5        7543       1764              68             16        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
 tn_middle  middle fold_1        7059       1941              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
 tn_middle  middle fold_2        7117       1883              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
 tn_middle  middle fold_3        7354       1646              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
 tn_middle  middle fold_4        7170       1830              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
 tn_middle  middle fold_5        7300       1700              68             16        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
 tn_bottom  bottom fold_1        7366       1941              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
 tn_bottom  bottom fold_2        7363       1944              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
 tn_bottom  bottom fold_3        7540       1767              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
 tn_bottom  bottom fold_4        7416       1891              67             17        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
 tn_bottom  bottom fold_5        7543       1764              68             16        0                             True                         True                                True                                 True              True             True                     True                    True                            True   PASS
```

本协议不读取上一轮历史 split，不覆盖旧 TN bottom pilot，不运行任何模型。
