# 修复77特征SHAP回传核验报告

核验对象：TabICLv2_WQ77_EXPLANATION_RESULTS_20261120.zip。附件中的脚本和说明只作为审查证据，没有执行；独立计算均来自已有数值。原ZIP、迁移包及正式实验文件不改写。

## 审查结论

回传覆盖十任务30个预登记核心样本×77特征，共2310个局部归因值；另含20条一维ALE。硬性血缘/算术失败1项，SHAP预测与独立P1探针的1e-6数值比对失败18项。
**本结果必须以P1单查询、3核心样本任务内解释使用；不能称为对正式完整测试批预测的严格等价解释，也不能称为十任务总体稳定性验证通过。**十项批量复现审查均未达到原最大1e-6/均值1e-7阈值，不能被run_status.exit_code=0或quality_audit的50 PASS覆盖。

## 1. 文件与版本

ZIP SHA256与旁路文件相同，CRC、成员路径及PACKAGE_MANIFEST逐项哈希已检查。portable_config与本地原迁移版相同；30核心ID/顺序不变，未把种子重复算作独立样本。逐任务核对本地修复版NPZ、训练/测试ID顺序、77列顺序、官方checkpoint、新imputer统计哈希、每个SHAP查询float32值/哈希以及16训练背景矩阵哈希。
目的机未回传独立pip freeze、GPU型号/驱动完整快照；配置锁版本和入口逻辑可以检查，但不据此推断具体设备或把数值差全部归为GPU。

## 2. SHAP数值完整性

最大独立复算可加误差=3.5527136788e-15。每文件77列排序、特征值、原目标尺度、seed42、155评估预算、16背景、有限值及保存的可加性残差逐一核验。平均绝对SHAP按每任务3核心聚合，全部770项与回传汇总及任务内名次对照。
30个文件记录的实际模型调用合计70,247次；保存的计算时间合计79.48小时（只是各文件记录的合计，不等于独立验证了GPU占用或墙钟日历时间）。

| task | obs_id | 可加性复算误差 | SHAP减审查P1 | P1减正式批量 | 非零特征数 |
| --- | --- | --- | --- | --- | --- |
| chla_surface | MM5_20141015_surface | 1.110223e-15 | -0.002682181 | -9.858771e-08 | 76 |
| chla_surface | VM7_20170406_surface | 1.110223e-15 | 0.0008842882 | 0 | 76 |
| chla_surface | SM13_20150515_surface | 0 | 0.01431616 | 2.985483e-06 | 76 |
| chla_middle | NM6_20200529_middle | 8.881784e-16 | 0.0007216955 | 0.0006496001 | 76 |
| chla_middle | SM19_20160118_middle | 4.440892e-16 | -0.007742898 | -0.007654731 | 76 |
| chla_middle | VM7_20220506_middle | 0 | 0.002305707 | 0.00307557 | 76 |
| chla_bottom | PM6_20230527_bottom | 6.661338e-16 | 0 | 1.105168e-07 | 76 |
| chla_bottom | NM5_20150112_bottom | 6.661338e-16 | 0 | 0 | 76 |
| chla_bottom | SM13_20190517_bottom | 3.552714e-15 | 0 | -1.890874e-06 | 76 |
| secchi | TM5_20140526_surface | 1.776357e-15 | 0 | -4.757046e-05 | 76 |
| secchi | VM4_20170303_surface | 1.332268e-15 | 0 | 8.82995e-07 | 76 |
| secchi | SM4_20180423_surface | 3.552714e-15 | 0 | 3.654793e-05 | 76 |
| tn_surface | PM4_20240708_surface | 8.326673e-16 | -0.0001164384 | -6.651317e-05 | 76 |
| tn_surface | SM19_20191206_surface | 5.551115e-17 | 0.0005501522 | 0.0006299297 | 76 |
| tn_surface | VM7_20220106_surface | 4.440892e-16 | 4.874343e-05 | -0.0001956573 | 76 |
| tn_middle | SM19_20170721_middle | 5.551115e-17 | 0 | 0 | 76 |
| tn_middle | TM6_20200826_middle | 7.21645e-16 | 0 | 7.353159e-06 | 76 |
| tn_middle | WT1_20230517_middle | 1.776357e-15 | 0 | 1.126357e-07 | 76 |
| tn_bottom | TM6_20220427_bottom | 1.665335e-16 | 2.761028e-05 | 0.0002897138 | 76 |
| tn_bottom | SM13_20180321_bottom | 4.996004e-16 | 0.0002123797 | 0.0001091799 | 76 |
| tn_bottom | NM6_20170619_bottom | 8.881784e-16 | 0.001189137 | 0.0004511405 | 76 |
| tp_surface | TM6_20181105_surface | 1.561251e-16 | 6.003313e-06 | -2.401929e-05 | 76 |
| tp_surface | VM7_20220606_surface | 1.040834e-16 | 4.22514e-05 | 5.838069e-05 | 76 |
| tp_surface | SM3_20170707_surface | 9.853229e-16 | 5.451386e-05 | 0.0001346692 | 76 |
| tp_middle | PM4_20140110_middle | 1.082467e-15 | 0 | -7.325347e-06 | 76 |
| tp_middle | TM6_20230216_middle | 1.103284e-15 | 0 | 5.987883e-06 | 76 |
| tp_middle | WM2_20191204_middle | 5.273559e-16 | 0 | 7.707658e-06 | 76 |
| tp_bottom | PM6_20180205_bottom | 1.004405e-15 | -3.901332e-05 | -3.901143e-05 | 76 |
| tp_bottom | VM7_20140306_bottom | 5.967449e-16 | 0.0001342416 | 0.0001074545 | 76 |
| tp_bottom | SM3_20230114_bottom | 3.885781e-16 | -1.464899e-05 | -4.149392e-05 | 76 |

可加性接近零只能证明当前SHAP分解与当前预测数值相加一致，不能证明解释稳定、代表总体或有因果含义。仅单次155预算Permutation归因，不是多次排列一致性结论。当前未见seed43结果，故未完成该种子敏感性审查，不补造重复。

## 3. 正式批量预测与P1边界

| task | 严格批量复现状态 | 最大绝对误差 | 平均绝对误差 | 固定P1三次范围 | SHAP与审查P1最大差 | 数值阈值_最大 | 数值阈值_均值 | 批量样本数 | 核心数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chla_surface | NUMERICAL_REPRODUCTION_FAILED | 0.004712158 | 1.342457e-05 | 0 | 0.01431616 | 1e-06 | 1e-07 | 2039 | 3 |
| chla_middle | NUMERICAL_REPRODUCTION_FAILED | 0.01142272 | 0.0003497195 | 0 | 0.007742898 | 1e-06 | 1e-07 | 2039 | 3 |
| chla_bottom | NUMERICAL_REPRODUCTION_FAILED | 0.0006926955 | 3.961651e-06 | 0 | 0 | 1e-06 | 1e-07 | 2039 | 3 |
| secchi | NUMERICAL_REPRODUCTION_FAILED | 0.0005879957 | 4.588212e-05 | 0 | 0 | 1e-06 | 1e-07 | 2173 | 3 |
| tn_surface | NUMERICAL_REPRODUCTION_FAILED | 0.003672928 | 0.0003372013 | 0 | 0.0005501522 | 1e-06 | 1e-07 | 1941 | 3 |
| tn_middle | NUMERICAL_REPRODUCTION_FAILED | 0.000460119 | 3.95714e-05 | 0 | 0 | 1e-06 | 1e-07 | 1941 | 3 |
| tn_bottom | NUMERICAL_REPRODUCTION_FAILED | 0.002347778 | 0.000344728 | 0 | 0.001189137 | 1e-06 | 1e-07 | 1941 | 3 |
| tp_surface | NUMERICAL_REPRODUCTION_FAILED | 0.0005185624 | 3.626711e-05 | 0 | 5.451386e-05 | 1e-06 | 1e-07 | 1941 | 3 |
| tp_middle | NUMERICAL_REPRODUCTION_FAILED | 5.47056e-05 | 4.248212e-06 | 0 | 0 | 1e-06 | 1e-07 | 1941 | 3 |
| tp_bottom | NUMERICAL_REPRODUCTION_FAILED | 0.0008105193 | 5.468172e-05 | 0 | 0.0001342416 | 1e-06 | 1e-07 | 1941 | 3 |

表中最大/平均批量误差为目的机已有审查记录，回传未含全批新复现预测向量，因此本机不能独立重算这两个全批统计量。已独立核对固定3样本的原冻结参考、三次P1范围及SHAP保存预测；对应明细见02/03号表。
三次P1范围为0只覆盖30个固定核心输入的该次执行，不能外推到所有扰动样本、所有站点或历史失败事件。需要区分：数据/模型身份一致；全批严格数值复现失败；固定点P1重复范围为0；归因可加性通过。这四件事可以同时成立。

## 4. 本批前五特征（描述性）

| task | Top5 |
| --- | --- |
| chla_surface | ssrd_sum_3d；wind_speed_x_rain14d；ssrd_mean_3d；lon；sst_mean_3d |
| chla_middle | ssrd_sum_3d；ssrd_mean_3d；wind_speed_x_rain14d；sst_mean_3d；month_cos |
| chla_bottom | ssrd_sum_3d；wind_speed_x_rain14d；ssrd_mean_7d；lon；ssrd_mean_3d |
| secchi | lat；lon；month_sin；ntl_avg_rad_roll3；ntl_avg_rad |
| tn_surface | ntl_avg_rad；lon；year_offset；lat；ntl_avg_rad_lag1 |
| tn_middle | ntl_avg_rad；lon；lat；ntl_avg_rad_lag1；ntl_avg_rad_delta1 |
| tn_bottom | lon；ntl_avg_rad；lat；ntl_avg_rad_delta1；ntl_avg_rad_roll3 |
| tp_surface | ntl_avg_rad_roll3；ntl_avg_rad；year_offset；month_sin；ntl_avg_rad_lag2 |
| tp_middle | year_offset；ntl_avg_rad；lon；ntl_avg_rad_delta1；lat |
| tp_bottom | year_offset；ntl_avg_rad；wind_speed_roll3；lon；t2m_lag1 |

Top5来自本批3核心平均绝对SHAP，不用来重新挑选解释样本。不同水质任务原始SHAP量纲不同，不能横向比较绝对贡献；排名仅任务内部相对次序。

## 5. 随包ALE核验

| task | feature | 分箱数 | 查询数 | 最小bin_n | 最小bin_站点 | 局部差值复算最大误差 | ALE复算最大误差 | 重复局部差值范围 | 登记改变行数 | 数值状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chla_surface | wind_speed_lag1 | 10 | 600 | 60 | 17 | 9.020562e-17 | 1.387779e-16 | 0 | 0 | COMPUTED_PASS |
| chla_surface | wind_speed | 10 | 600 | 60 | 17 | 9.714451e-17 | 7.459311e-17 | 0 | 0 | COMPUTED_PASS |
| chla_middle | wind_speed | 10 | 600 | 60 | 17 | 1.110223e-16 | 8.326673e-17 | 0 | 0 | COMPUTED_PASS |
| chla_middle | wind_speed_sq | 10 | 600 | 60 | 17 | 9.584347e-17 | 9.020562e-17 | 0 | 0 | COMPUTED_PASS |
| chla_bottom | t2m_lag1 | 10 | 600 | 60 | 17 | 8.326673e-17 | 1.110223e-16 | 0 | 0 | COMPUTED_PASS |
| chla_bottom | t2m | 10 | 600 | 60 | 17 | 8.326673e-17 | 3.330669e-16 | 0 | 0 | COMPUTED_PASS |
| secchi | ssrd_sum_3d | 6 | 328 | 42 | 11 | 6.245005e-17 | 8.326673e-17 | 0 | 2 | COMPUTED_PASS |
| secchi | t2m_roll3 | 10 | 600 | 60 | 19 | 9.887924e-17 | 7.372575e-17 | 0 | 0 | COMPUTED_PASS |
| tn_surface | t2m_delta1 | 10 | 600 | 60 | 17 | 9.887924e-17 | 8.023096e-17 | 0 | 0 | COMPUTED_PASS |
| tn_surface | wind_speed_lag1 | 10 | 600 | 60 | 17 | 9.97466e-17 | 9.280771e-17 | 0 | 0 | COMPUTED_PASS |
| tn_middle | t2m_delta1 | 10 | 600 | 60 | 17 | 8.586881e-17 | 9.714451e-17 | 0 | 0 | COMPUTED_PASS |
| tn_middle | v10 | 10 | 600 | 60 | 17 | 9.280771e-17 | 9.714451e-17 | 0 | 0 | COMPUTED_PASS |
| tn_bottom | t2m_lag2 | 10 | 600 | 60 | 17 | 9.649399e-17 | 9.627715e-17 | 0 | 0 | COMPUTED_PASS |
| tn_bottom | t2m_roll3 | 10 | 600 | 60 | 17 | 8.218252e-17 | 8.673617e-17 | 0 | 0 | COMPUTED_PASS |
| tp_surface | t2m_lag2 | 10 | 600 | 60 | 17 | 9.86624e-17 | 8.164042e-17 | 0 | 0 | COMPUTED_PASS |
| tp_surface | u10 | 10 | 600 | 60 | 17 | 9.177771e-17 | 9.194034e-17 | 0 | 0 | COMPUTED_PASS |
| tp_middle | t2m_lag1 | 10 | 600 | 60 | 17 | 8.679038e-17 | 9.833714e-17 | 0 | 0 | COMPUTED_PASS |
| tp_middle | tcc | 10 | 600 | 60 | 17 | 8.969062e-17 | 9.779504e-17 | 0 | 0 | COMPUTED_PASS |
| tp_bottom | t2m_lag1 | 10 | 600 | 60 | 17 | 9.70903e-17 | 9.931292e-17 | 0 | 0 | COMPUTED_PASS |
| tp_bottom | wind_speed_lag1 | 10 | 600 | 60 | 17 | 9.237403e-17 | 8.868774e-17 | 0 | 0 | COMPUTED_PASS |

SQLite用只读immutable模式打开；逐曲线检查保存的局部上下界预测差、查询ID/哈希、新上下文、bin内均值、累积与按样本数加权中心化、n与独立站点数，未调用模型或重新计算ALE预测。表中ALE复算只是对已有局部差值作算术核验。
Secchi ssrd_sum_3d已在目的机以secchi_boundary_v2修订两条登记：NT1_20161107_surface、NM6_20161107_surface由第3箱移至第4箱；查询ID与六个边界保留。这是可追溯的登记修订，不能写成原登记毫无变化，也不能把原DATA_DRIFT记录删除或改称从未失败。其余19曲线边界精度修复记录已与源边界比对。
Secchi曲线应在图注/数据说明中注明登记v2；来源配置中的原登记状态仍是需核对，最终结果与修订JSON要一同引用。

## 6. 代码差异和是否可用

SHAP核心算法、背景规则、上下文适配器与预测审查代码均按AST与原迁移版比对。ALE入口新增round_trip精度读写和显式Secchi两条边界修订；质量审查相应增加修订校验。原SHAP核心未被替换成TreeSHAP、旧模型或历史82/111特征计算。
若硬性核验均通过，可作为修复版模型的P1局部归因素材进入下一步绘图/科学审阅；严格批量等价、种子敏感性与总体代表性仍不具备通过结论。论文不应只据50个PASS或进程完成状态写“正式预测解释已完全验证”。

## 7. 本轮未执行事项

未运行回传的任何Python/PowerShell脚本；未加载TabICLv2模型；未训练、推理或计算SHAP/ALE。仅加载本地可信SimpleImputer序列化对象读取statistics_，未调用fit/transform/predict；原数据、预测和论文均不修改。

## 必须先处理的硬性问题

| 检查项 | 结果 | 类别 | 依据 |
| --- | --- | --- | --- |
| 回传清单全覆盖 | FAIL | HARD |  |

## SHAP对固定P1探针的额外数值不一致

| 检查项 | 结果 | 类别 | 依据 |
| --- | --- | --- | --- |
| chla_surfaceMM5_20141015_surfaceSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | -0.002682180526931255 |
| chla_surfaceVM7_20170406_surfaceSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 0.0008842881947357384 |
| chla_surfaceSM13_20150515_surfaceSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 0.014316160781685738 |
| chla_middleNM6_20200529_middleSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 0.0007216955193432018 |
| chla_middleSM19_20160118_middleSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | -0.007742897614913424 |
| chla_middleVM7_20220506_middleSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 0.0023057066892686606 |
| tn_surfacePM4_20240708_surfaceSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | -0.00011643837846209149 |
| tn_surfaceSM19_20191206_surfaceSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 0.0005501522011306936 |
| tn_surfaceVM7_20220106_surfaceSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 4.8743432368159034e-05 |
| tn_bottomTM6_20220427_bottomSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 2.76102805928935e-05 |
| tn_bottomSM13_20180321_bottomSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 0.00021237974083598843 |
| tn_bottomNM6_20170619_bottomSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 0.001189136965628057 |
| tp_surfaceTM6_20181105_surfaceSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 6.003313202197769e-06 |
| tp_surfaceVM7_20220606_surfaceSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 4.225139545890183e-05 |
| tp_surfaceSM3_20170707_surfaceSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 5.451385801930053e-05 |
| tp_bottomPM6_20180205_bottomSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | -3.901331822430028e-05 |
| tp_bottomVM7_20140306_bottomSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | 0.00013424164418519846 |
| tp_bottomSM3_20230114_bottomSHAP预测与独立固定P1探针 | FAIL | NUMERICAL | -1.4648988463997203e-05 |
