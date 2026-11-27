# distance-to-river-mouth 系统级重构报告

本轮不把该结果直接入模，只做 source refinement。

## 系统级定义
- `north_major_mainland`：北侧大陆主要河口系统
- `western_pearl_estuary`：西侧珠江口系统
- `northwest_local_inflow`：西北近岸局地入流系统
- `eastern_local_inflow`：东侧近岸局地入流系统

## 核心结果
- 站点覆盖率：94/94
- 系统映射歧义站点数：3
- 系统数量：4

## 主要河口系统汇总

       mouth_system_id mouth_system_name  mouth_count  max_discharge   mean_lon  mean_lat
  eastern_local_inflow        东侧近岸局地入流系统           22         34.316 114.609943 22.608428
  north_major_mainland        北侧大陆主要河口系统           10       1041.516 113.733750 23.025833
northwest_local_inflow        西北近岸局地入流系统           10         13.951 113.720833 22.690417
 western_pearl_estuary           西侧珠江口系统           27       9176.198 113.260185 22.224691

## 各 water_zone 的最近系统分配

water_zone_en             nearest_mouth_system_id
Deep Bay                  eastern_local_inflow        1
                          northwest_local_inflow      4
Eastern Buffer            eastern_local_inflow        5
Junk Bay                  eastern_local_inflow        2
Mirs Bay                  eastern_local_inflow       14
North Western             eastern_local_inflow        7
Port Shelter              eastern_local_inflow       12
Southern                  eastern_local_inflow       18
Tolo Harbour and Channel  eastern_local_inflow        8
Victoria Harbour          eastern_local_inflow       17
Western Buffer            eastern_local_inflow        6

## 判断
- 相比逐条 terminal reach，这个系统级口径显著降低了歧义。
- 但当前分配仍明显偏向 `eastern_local_inflow`，说明系统划分虽然更稳，却还不够形成均衡、可直接入模的距离梯度。
- 因此本轮建议把它视为比原口径更稳的 source refinement 版本，但仍暂不直接进入 `distance_to_mouth_only A/B`。
