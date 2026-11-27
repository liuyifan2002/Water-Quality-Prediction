# MULTISOURCE_ABLATION_V1 决策报告

- 完成时间：2026-09-23T01:03:44
- 模型：XGBoost
- 任务：Chl-a surface、Secchi、TN surface、TP middle
- 评价：统一 station holdout，来源级 leave-one-source-out，1000 次站点 bootstrap

## 来源分组
source_group
HUMAN_ACTIVITY          6
METEOROLOGY            29
OCEAN_PHYSICS          11
RAINFALL_HYDRO_TIDE    22
SPATIAL_STATIC          2
TEMPORAL_PROCESS        7

## 结果
        task        source_group  n_stations  station_win_rate_RMSE  station_win_rate_MAE  pooled_delta_R2  pooled_delta_RMSE  pooled_delta_MAE  bootstrap_RMSE_CI_low  bootstrap_RMSE_CI_high  bootstrap_MAE_CI_low  bootstrap_MAE_CI_high classification
chla_surface      HUMAN_ACTIVITY          84               0.404762              0.428571         0.002929          -0.012992         -0.007809              -0.037801                0.014960             -0.021803               0.004718        NEUTRAL
chla_surface         METEOROLOGY          84               0.642857              0.595238        -0.026146           0.114470          0.061088               0.066750                0.162947              0.033086               0.089365     SUPPORTIVE
chla_surface       OCEAN_PHYSICS          84               0.583333              0.583333        -0.008679           0.038296          0.014739               0.009098                0.069267              0.000477               0.029455     SUPPORTIVE
chla_surface RAINFALL_HYDRO_TIDE          84               0.607143              0.500000        -0.009353           0.041260          0.018014               0.008009                0.071847              0.000214               0.033801     SUPPORTIVE
chla_surface      SPATIAL_STATIC          84               0.750000              0.797619        -0.087248           0.372129          0.196458               0.253314                0.498952              0.133352               0.254577      IMPORTANT
chla_surface    TEMPORAL_PROCESS          84               0.666667              0.642857        -0.004122           0.018228          0.018946              -0.012409                0.044746              0.004186               0.032631     SUPPORTIVE
      secchi      HUMAN_ACTIVITY          94               0.500000              0.457447         0.004061          -0.003311         -0.003783              -0.006318               -0.000673             -0.006232              -0.001640        NEUTRAL
      secchi         METEOROLOGY          94               0.670213              0.659574        -0.032207           0.025762          0.013522               0.011883                0.042416              0.006744               0.021679      IMPORTANT
      secchi       OCEAN_PHYSICS          94               0.489362              0.500000        -0.007148           0.005793          0.000847               0.000164                0.011278             -0.002580               0.004237     SUPPORTIVE
      secchi RAINFALL_HYDRO_TIDE          94               0.585106              0.542553        -0.007181           0.005819          0.001589               0.001280                0.010792             -0.001939               0.004629     SUPPORTIVE
      secchi      SPATIAL_STATIC          94               0.787234              0.797872        -0.171028           0.128259          0.095549               0.086398                0.168225              0.064507               0.129633      IMPORTANT
      secchi    TEMPORAL_PROCESS          94               0.638298              0.638298        -0.004256           0.003455          0.001492              -0.000723                0.006949             -0.002147               0.004625     SUPPORTIVE
  tn_surface      HUMAN_ACTIVITY          84               0.511905              0.511905        -0.016506           0.004023          0.001601              -0.000250                0.011171             -0.000283               0.004215     SUPPORTIVE
  tn_surface         METEOROLOGY          84               0.785714              0.678571        -0.033022           0.008004          0.008205               0.000906                0.020325              0.005182               0.012186      IMPORTANT
  tn_surface       OCEAN_PHYSICS          84               0.523810              0.595238        -0.009076           0.002217          0.001618              -0.001409                0.008937              0.000070               0.003465     SUPPORTIVE
  tn_surface RAINFALL_HYDRO_TIDE          84               0.583333              0.583333         0.007849          -0.001929          0.000191              -0.004337                0.001159             -0.001907               0.002062     SUPPORTIVE
  tn_surface      SPATIAL_STATIC          84               0.773810              0.833333         0.040370          -0.010032          0.010966              -0.081689                0.039202             -0.025411               0.034685     SUPPORTIVE
  tn_surface    TEMPORAL_PROCESS          84               0.678571              0.666667         0.012495          -0.003075          0.000025              -0.012892                0.002536             -0.003915               0.003007     SUPPORTIVE
   tp_middle      HUMAN_ACTIVITY          84               0.464286              0.404762         0.017793          -0.000474         -0.000147              -0.001624                0.000356             -0.000625               0.000164        NEUTRAL
   tp_middle         METEOROLOGY          84               0.654762              0.678571        -0.082602           0.002124          0.000885               0.000130                0.005387              0.000248               0.001882      IMPORTANT
   tp_middle       OCEAN_PHYSICS          84               0.500000              0.559524        -0.008650           0.000228          0.000031              -0.000002                0.000472             -0.000158               0.000193     SUPPORTIVE
   tp_middle RAINFALL_HYDRO_TIDE          84               0.428571              0.440476         0.006885          -0.000183         -0.000090              -0.000495                0.000176             -0.000263               0.000080        NEUTRAL
   tp_middle      SPATIAL_STATIC          84               0.750000              0.785714         0.085078          -0.002324         -0.000150              -0.011503                0.002349             -0.003822               0.002024        NEUTRAL
   tp_middle    TEMPORAL_PROCESS          84               0.642857              0.654762         0.055153          -0.001489         -0.000329              -0.004776                0.000310             -0.001418               0.000229        NEUTRAL

## 遥感结论
当前冻结的 77 个 strict features 中没有真实遥感波段、Ocean Colour 产品或遥感 availability flag，因此本轮不能运行独立 NO_REMOTE_SENSING 消融。不能把 ERA5、SST 或夜间灯光的贡献解释为遥感贡献。

## 总体判断
本轮仅完成来源级 XGBoost 消融，不开展 TabICLv2 消融、SHAP、ALE 或新模型实验。
