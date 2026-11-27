"""从已经读取的历史记录登记候选源；只整理证据，不运行实验。"""
from pathlib import Path
import csv

ROOT=Path(r'E:\研究生\codex\水质预测')
OUT=Path(__file__).resolve().parent
E='数据/外部数据/试验项目接管_20260521/experiments_outputs_current/'
D='数据/外部数据/试验项目接管_20260521/data_processed_current/'
N='数据/外部数据/试验项目接管_20260521/data_model_input_new_sources/'
R='文档/主线接管_20260521/英文原始报告_归档/源项目_reports_current/'
V='数据/外部数据/试验项目接管_20260521/outputs_modeling/experiments/chla_new_data_source_validation_20260520/'
rows=[]
def add(source,fields,resolution,coverage,stage,validation,entered,performance,decision,evidence,class_code):
    paths=[x.strip() for x in evidence.split(';')]
    for p in paths:
        if p and not (ROOT/p).exists():raise FileNotFoundError(ROOT/p)
    rows.append(dict(data_source=source,representative_fields=fields,spatiotemporal_resolution=resolution,
                     effective_coverage=coverage,experiment_stage=stage,validation_protocol=validation,
                     entered_model=entered,performance_change=performance,final_treatment=decision,
                     evidence_files=';'.join(paths),evidence_class=class_code))

add('CMEMS BGC','bgc_chl; bgc_no3; bgc_po4; bgc_npp','0.25°、日尺度',
    '10,866站点日期行；核心字段非缺失43.9%','已提取并完成历史A/B','历史年份留出；Chl-a/Secchi六任务；非正式站点五折',
    '仅历史候选；未进X_strict_77','Chl-a中层测试R² 0.3646→0.3928，验证R² 0.2467→0.2161',
    '局部改善但验证退化，不提升为统一正式输入',
    E+'new_source_comparison_to_current_best.csv;'+N+'copernicus_bgc_features_2014_2024.csv','B/C')
add('CMEMS Physics','cmems_sss; cmems_uo; cmems_vo; cmems_mld','0.083°、日尺度',
    '10,866站点日期行；主要字段非缺失72.4%','已提取并完成历史A/B','历史年份留出；六任务；非正式站点五折',
    '仅历史候选；未进X_strict_77','Chl-a中层测试R² 0.3646→0.3772，验证R² 0.2467→0.2156',
    '局部改善但验证退化，保留背景代理定位',
    E+'strong_mechanism_comparison_to_current_best.csv;'+D+'copernicus_physics_validation_report.json','B/C')
add('NASA OceanColor','nasa_chl; nasa_kd490; Rrs产品','约9 km、月尺度',
    '全时段A/B匹配率97.89%；缺失2.11%','已提取并完成历史A/B','历史年份留出；Chl-a/Secchi多任务；非正式站点五折',
    '仅历史候选；未进X_strict_77','Chl-a表层测试R² 0.3298→0.2232，RMSE及MAE均变差',
    '月尺度单源未形成稳定增益，不统一纳入',
    E+'nasa_oceancolor_comparison_to_current_best.csv;'+D+'nasa_oceancolor_validation_report.json','A')
add('CMEMS Ocean Colour 4 km patch','oc_chl; oc_kd490; oc_zsd; 3×3/5×5统计','约4 km、日切片与站点日期匹配',
    '基础OC关键字段非缺失76.8%；部分Rrs仅约24.3%','已构造并作覆盖审查','本轮未将独立CMEMS 4 km patch A/B与其它patch混同',
    '仅历史候选；未进X_strict_77','本次未定位到可独立归属的同口径A/B指标，不判定预测无效',
    '保留为已构建、单独预测效应待核的光学候选',
    N+'ocean_color_kd490_tss_features_2014_2024.csv;'+V+'new_data_source_readiness_table.csv','D/C')
add('Sentinel-2高分影像','s2_3x3_B4_mean; s2_5x5_valid_ratio','站点邻域3×3/5×5 patch、影像时间窗',
    '10,866站点日期；有效影像覆盖39.03%','已构造并完成历史A/B','历史年份留出；六任务；非正式站点五折',
    '仅历史候选；未进X_strict_77','Chl-a表层测试R² 0.3298→0.2821',
    '覆盖有限且未形成稳定增益，不统一纳入',
    E+'current_candidate_registry.csv;'+D+'highres_patch_validation_report.json','A/C')
add('Landsat高分影像','landsat_3x3_SR_B4_mean; landsat_5x5_valid_ratio','站点邻域3×3/5×5 patch、影像时间窗',
    '10,866站点日期；有效影像覆盖10.12%','已构造并完成历史A/B','历史年份留出；六任务；非正式站点五折',
    '仅历史候选；未进X_strict_77','六任务中有局部正值，范围ΔR²=-0.0491至+0.0051',
    '局部小幅改善未达统一输入依据',
    E+'current_candidate_registry.csv;'+D+'highres_patch_validation_report.json','B/C')
add('Sentinel-3 OLCI候选表','olci_chla; olci_kd490; olci_rrs_443','候选表按站点日期；原产品分辨率需追溯',
    'olci_chla非缺失76.76%；指数列全空','已构建候选表并做来源审查','未定位到独立OLCI正式预测A/B',
    '否','无可比预测指标，不能判定无效',
    '保留为来源/覆盖已核、预测未核的候选',
    N+'sentinel3_olci_features_2014_2024.csv;'+V+'new_data_source_readiness_table.csv','D/C')
add('Natural Earth岸线距离','distance_to_coast_km; coastal_exposure_bucket','10m岸线矢量、站点静态',
    '94/94站；缺失0%','已构建并完成历史A/B','历史年份留出；12任务/多组合；非正式站点五折',
    '仅历史候选；未进X_strict_77','TN表层测试R² 0.5201→0.5245，幅度+0.0043',
    '局部改善但无跨任务稳定性，不统一纳入',
    E+'coastal_distance_comparison_to_current_best.csv;'+D+'coastal_distance_validation_report.json','B')
add('香港环保署站点近似水深','station_depth_approx_m; relative_depth','站点静态、官方历史表近似值',
    '94/94站；观测表30,719/30,719匹配','已构建并完成历史A/B','历史年份留出；12任务；非正式站点五折',
    '仅历史候选；未进X_strict_77','Chl-a底层测试R² 0.3436→0.3519；其它任务并非普遍改善',
    '局部改善，非真实连续海底地形',
    E+'depth_source_comparison_to_current_best.csv;'+D+'station_depth_feature_validation.json','B')
add('GEBCO海底地形','拟提取bathymetry','公开格网；本地稳定子集分辨率未定',
    '未形成可用香港近岸特征表','完成来源验证，未形成稳定特征','无正式A/B',
    '否','无模型指标，不能判定无效',
    '保留数据获取缺口，与站点近似水深区分',
    R+'coastal_distance_ab_report.md;'+R+'data_gap_inventory.md','D')
add('HKO增强潮汐代理','enhanced_tide_proxy; tide_phase','采样日/日期级潮位代理',
    '具体覆盖见历史A/B记录；采样时刻不可得','已构造并完成历史A/B','历史年份留出；六任务；非正式站点五折',
    '增强候选未统一纳入；正式77仍有tidal_range','六任务测试ΔR²范围-0.0603至+0.0118',
    '仅保留历史探索；不把日期级代理写成采样瞬时潮位',
    E+'new_source_comparison_to_current_best.csv;'+D+'tide_phase_validation_report.json','B')
add('GPM IMERG降雨代理','拟构建1/3/7/14/30日降雨窗','0.1°、日尺度（计划产品）',
    '产品发现成功；样本下载受条款接受阻断','仅来源验证','无正式A/B',
    '否','无模型指标，不能判定无效',
    '保留待获取，不替代正式HKO降雨字段',
    R+'gpm_imerg_source_validation_report.md','D')
add('CHIRPS降雨代理','拟构建逐日/逐月降雨','0.05°、日/月尺度（可行性报告）',
    '未见本地全时段匹配表','可行性研究','无正式A/B',
    '否','无模型指标，不能判定无效',
    '保留为备选数据源',
    R+'runoff_discharge_proxy_data_feasibility_report.md','D')
add('ERA5-Land额外水文气象','拟用径流/地表水文代理','约0.1°、小时尺度（可行性报告）',
    '未见本地全时段匹配表','可行性研究','无正式A/B',
    '否；正式77的ERA5变量不是ERA5-Land','无模型指标，不能判定无效',
    '保留为备选，不混写成已测试',
    R+'runoff_discharge_proxy_data_feasibility_report.md','D')
add('GloFAS/WRTDS类流量','未见已构造字段','未核实','未见本地可追溯输入表',
    '限定项目文件盘点','无正式A/B','否','无模型指标，不能判定无效',
    '列为未取得的候选方向，不写预测结果',
    R+'runoff_discharge_proxy_data_feasibility_report.md','F')
add('DSD/STP污水与流量代理','未见已构造字段','未核实','未见本地可追溯输入表',
    '限定项目文件盘点','无正式A/B','否','无模型指标，不能判定无效',
    '列为未取得的候选方向，不写预测结果',
    R+'next_data_acquisition_plan.md','F')
add('VIIRS夜间灯光','ntl_avg_rad; viirs_count及历史统计','站点—月份匹配',
    '正式主表可定位六项相关字段','正式77组成部分','正式站点留出模型输入',
    '是：HUM六项','本表不重算或比较其单源指标',
    '保留正式输入，不标记为未采用',
    'runs/tabiclv2_pilot/01_strict_feature_manifest.csv;'+
    'runs/explanation_figures_v2/metadata/feature_display_name_mapping.csv','E')

with (OUT/'候选环境数据来源核验表.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader();w.writerows(rows)
print('CANDIDATES_REGISTERED',len(rows))
