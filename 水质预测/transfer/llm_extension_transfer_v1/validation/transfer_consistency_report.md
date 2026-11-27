# 迁移包一致性报告

总体校验：PASS

本报告只反映迁移包内文件的结构、键、冻结分折、预测样本集合、闭环指标字段和 SHA256 检查；未启动任何模型训练或推理。

| 检查 | 状态 | 说明 |
|---|---|---|
| `required_files` | `True` | all required anchor files exist |
| `main_table_readable` | `True` | rows=30719; columns=272; missing=[] |
| `main_table_obs_id_unique` | `True` | rows=30719; unique_obs_id=30719; duplicates=0 |
| `strict_feature_count` | `True` | included_features=77 |
| `strict_features_located` | `True` | absent=[] |
| `strict_feature_order` | `True` | manifest_order_unique_and_explicit=True |
| `chla_family_v2_split_validation` | `True` | rows=15; statuses=['PASS'] |
| `chla_family_v2_canonical_map` | `True` | stations=84; columns=['station_id', 'station_order', 'fold', 'protocol', 'seed', 'assignment_algorithm'] |
| `chla_family_v2_manifest_consistency` | `True` | 03_chla_surface_manifest.csv:chla_surface/fold_1 train=7744 test=2039 overlap=0; 03_chla_surface_manifest.csv:chla_surface/fold_2 train=7740 test=2043 overlap=0; 03_chla_surface_manifest.csv:chla_surface/fold_3 train=7923 test=1860 overlap=0; 03_chla_surface_manifest.csv:chla_surface/fold_4 train=7796 test=1987 overlap=0; 03_chla_surface_manifest.csv:chla_surface/fold_5 train=7929 test=1854 overlap=0; 04_chla_middle_manifest.csv:chla_middle/fold_1 train=7417 test=2039 overlap=0; 04_chla_middle_manifest.csv:chla_middle/fold_2 train=7478 test=1978 overlap=0; 04_chla_middle_manifest.csv:chla_middle/fold_3 train=7725 test=1731 overlap=0; 04_chla_middle_manifest.csv:chla_middle/fold_4 train=7534 test=1922 overlap=0; 04_chla_middle_manifest.csv:chla_middle/fold_5 train=7670 test=1786 overlap=0; 05_chla_bottom_manifest.csv:chla_bottom/fold_1 train=7744 test=2039 overlap=0; 05_chla_bottom_manifest.csv:chla_bottom/fold_2 train=7740 test=2043 overlap=0; 05_chla_bottom_manifest.csv:chla_bottom/fold_3 train=7923 test=1860 overlap=0; 05_chla_bottom_manifest.csv:chla_bottom/fold_4 train=7796 test=1987 overlap=0; 05_chla_bottom_manifest.csv:chla_bottom/fold_5 train=7929 test=1854 overlap=0 |
| `tn_family_v2_split_validation` | `True` | rows=15; statuses=['PASS'] |
| `tn_family_v2_canonical_map` | `True` | stations=84; columns=['station_id', 'station_order', 'fold', 'protocol', 'seed', 'assignment_algorithm'] |
| `tn_family_v2_manifest_consistency` | `True` | 03_tn_surface_manifest.csv:tn_surface/fold_1 train=7366 test=1941 overlap=0; 03_tn_surface_manifest.csv:tn_surface/fold_2 train=7363 test=1944 overlap=0; 03_tn_surface_manifest.csv:tn_surface/fold_3 train=7540 test=1767 overlap=0; 03_tn_surface_manifest.csv:tn_surface/fold_4 train=7416 test=1891 overlap=0; 03_tn_surface_manifest.csv:tn_surface/fold_5 train=7543 test=1764 overlap=0; 04_tn_middle_manifest.csv:tn_middle/fold_1 train=7059 test=1941 overlap=0; 04_tn_middle_manifest.csv:tn_middle/fold_2 train=7117 test=1883 overlap=0; 04_tn_middle_manifest.csv:tn_middle/fold_3 train=7354 test=1646 overlap=0; 04_tn_middle_manifest.csv:tn_middle/fold_4 train=7170 test=1830 overlap=0; 04_tn_middle_manifest.csv:tn_middle/fold_5 train=7300 test=1700 overlap=0; 05_tn_bottom_manifest.csv:tn_bottom/fold_1 train=7366 test=1941 overlap=0; 05_tn_bottom_manifest.csv:tn_bottom/fold_2 train=7363 test=1944 overlap=0; 05_tn_bottom_manifest.csv:tn_bottom/fold_3 train=7540 test=1767 overlap=0; 05_tn_bottom_manifest.csv:tn_bottom/fold_4 train=7416 test=1891 overlap=0; 05_tn_bottom_manifest.csv:tn_bottom/fold_5 train=7543 test=1764 overlap=0 |
| `tp_family_v2_split_validation` | `True` | rows=15; statuses=['PASS'] |
| `tp_family_v2_canonical_map` | `True` | stations=84; columns=['station_id', 'station_order', 'fold', 'protocol', 'seed', 'assignment_algorithm'] |
| `tp_family_v2_manifest_consistency` | `True` | 03_tp_surface_manifest.csv:tp_surface/fold_1 train=7366 test=1941 overlap=0; 03_tp_surface_manifest.csv:tp_surface/fold_2 train=7363 test=1944 overlap=0; 03_tp_surface_manifest.csv:tp_surface/fold_3 train=7540 test=1767 overlap=0; 03_tp_surface_manifest.csv:tp_surface/fold_4 train=7416 test=1891 overlap=0; 03_tp_surface_manifest.csv:tp_surface/fold_5 train=7543 test=1764 overlap=0; 04_tp_middle_manifest.csv:tp_middle/fold_1 train=7059 test=1941 overlap=0; 04_tp_middle_manifest.csv:tp_middle/fold_2 train=7117 test=1883 overlap=0; 04_tp_middle_manifest.csv:tp_middle/fold_3 train=7354 test=1646 overlap=0; 04_tp_middle_manifest.csv:tp_middle/fold_4 train=7170 test=1830 overlap=0; 04_tp_middle_manifest.csv:tp_middle/fold_5 train=7300 test=1700 overlap=0; 05_tp_bottom_manifest.csv:tp_bottom/fold_1 train=7366 test=1941 overlap=0; 05_tp_bottom_manifest.csv:tp_bottom/fold_2 train=7363 test=1944 overlap=0; 05_tp_bottom_manifest.csv:tp_bottom/fold_3 train=7540 test=1767 overlap=0; 05_tp_bottom_manifest.csv:tp_bottom/fold_4 train=7416 test=1891 overlap=0; 05_tp_bottom_manifest.csv:tp_bottom/fold_5 train=7543 test=1764 overlap=0 |
| `prediction_schema_chla_surface_xgboost` | `True` | rows=9783; files=5 |
| `prediction_schema_chla_surface_tabiclv2` | `True` | rows=9783; files=5 |
| `prediction_schema_chla_surface_tabm` | `True` | rows=9783; files=5 |
| `prediction_schema_chla_surface_tabpfn` | `True` | rows=9783; files=5 |
| `prediction_schema_secchi_xgboost` | `True` | rows=10866; files=1 |
| `prediction_schema_secchi_tabiclv2` | `True` | rows=10866; files=1 |
| `prediction_schema_secchi_tabm` | `True` | rows=10866; files=5 |
| `prediction_schema_secchi_tabpfn` | `True` | rows=10866; files=5 |
| `prediction_schema_tn_surface_xgboost` | `True` | rows=9307; files=5 |
| `prediction_schema_tn_surface_tabiclv2` | `True` | rows=9307; files=5 |
| `prediction_schema_tn_surface_tabm` | `True` | rows=9307; files=5 |
| `prediction_schema_tn_surface_tabpfn` | `True` | rows=9307; files=5 |
| `prediction_schema_tp_middle_xgboost` | `True` | rows=9000; files=5 |
| `prediction_schema_tp_middle_tabiclv2` | `True` | rows=9000; files=5 |
| `prediction_schema_tp_middle_tabm` | `True` | rows=9000; files=5 |
| `prediction_schema_tp_middle_tabpfn` | `True` | rows=9000; files=5 |
| `modern_prediction_completeness` | `True` | four models present for all four tasks |
| `modern_test_set_equal_chla_surface` | `True` | set_sizes=[9783, 9783, 9783, 9783] |
| `modern_test_set_equal_secchi` | `True` | set_sizes=[10866, 10866, 10866, 10866] |
| `modern_test_set_equal_tn_surface` | `True` | set_sizes=[9307, 9307, 9307, 9307] |
| `modern_test_set_equal_tp_middle` | `True` | set_sizes=[9000, 9000, 9000, 9000] |
| `closure_v2_metric_schema` | `True` | missing=[] |
| `pooled_and_fold_r2_distinct_fields` | `True` | rows=20 |
| `tn_tp_core_metrics_present` | `True` | TN/TP n_obs positive in V2 core table |
| `internal_sha256_manifest` | `True` | checked=665; failures=[] |
