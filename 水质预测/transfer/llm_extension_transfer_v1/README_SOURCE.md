# 香港近岸多源数据融合水质预测项目

本项目当前以“全站点三水层 `full_station` 正式主链”为准，服务于香港近岸水质预测、遥感信息敏感性分析与论文材料整理。项目重点围绕 `Chl-a / Secchi / TN / TP` 四类指标，在 `Surface / Middle / Bottom` 三水层上，融合水质监测、遥感、ERA5、降雨、潮汐和夜间灯光等多源信息，构建可复跑、可审计的研究流水线。

## 当前项目定位

- **正式主链**：94 站点、三水层、2014-2024 年 `full_station` 建模主流程
- **正式评价口径**：每个指标—水层任务采用固定 `80%/20%` 随机比例留出，`random_state=42`；用于评价同分布留出样本表现，不表述为未来年份外推。
- **扩展实验**：`Chl-a` 优化、站点独立建模、分海区建模、滚动 CV、增强特征等
- **诊断分析**：`Chl-a surface` 峰值、残差、自相关、覆盖率等专项诊断
- **论文材料**：正式结果解读、流程图说明、研究定位与论文图表整理

## 根目录结构

```text
项目根目录/
├─ 配置/
├─ 数据/
├─ 脚本/
├─ 结果/
├─ 文档/
├─ 归档_历史版本/
└─ 待确认_未归类/
```

详细规范见：

- [AGENTS.md](AGENTS.md)
- [项目架构说明](文档/项目架构说明.md)
- [目录结构规范](文档/目录结构规范.md)
- [正式主链运行说明](文档/正式主链运行说明.md)

## 正式主链入口

正式主链脚本位于 `脚本/01_数据准备`、`脚本/02_特征构建`、`脚本/04_模型训练`、`脚本/06_结果汇总与绘图`。

推荐运行顺序：

1. `python 脚本/01_数据准备/01_prepare_station_samples.py`
2. `python 脚本/01_数据准备/02_build_water_quality_master.py`
3. `python 脚本/02_特征构建/03_merge_satellite_features.py`
4. `python 脚本/02_特征构建/04_build_model_table.py`
5. `python 脚本/04_模型训练/05_train_full_station_models.py`
6. `python 脚本/06_结果汇总与绘图/06_summarize_full_station_results.py`
7. `python 脚本/06_结果汇总与绘图/07_plot_full_station_results.py`
8. `python 脚本/06_结果汇总与绘图/08_shap_full_station_results.py`
9. `python 脚本/05_诊断分析/09_diagnose_chla_surface.py`
10. `python 脚本/06_结果汇总与绘图/11_build_thesis_ch1_ch2_materials.py`
11. `python 脚本/06_结果汇总与绘图/12_build_thesis_ch3_ch4_materials.py`
12. `python 脚本/06_结果汇总与绘图/13_build_thesis_ch5_materials.py`
13. `python 脚本/06_结果汇总与绘图/14_build_thesis_ch6_materials.py`
14. `python 脚本/06_结果汇总与绘图/15_build_thesis_draft_docx.py`

## 数据与结果位置

- 正式主表：`数据/处理后数据/正式主表/`
- 扩展版本：`数据/处理后数据/衍生版本/`
- 诊断数据：`数据/处理后数据/诊断数据/`
- 正式主结果：`结果/正式结果/full_station/`
- 扩展实验：`结果/扩展实验/`
- 论文图表：`结果/论文图表/`
- 诊断结果：`结果/诊断结果/`
- 论文材料：`文档/论文材料/`

## 历史版本与待确认内容

- 早期表层 `v0` 脚本已归入：`脚本/历史版本/`
- 过期输出已归入：`归档_历史版本/过期输出/`
- 暂不确定归属的文件统一放入：`待确认_未归类/`

## 环境依赖

```powershell
python -m pip install -r requirements.txt
```

如果涉及 ERA5 下载，还需要本地可用的 `cdsapi` 与对应的 CDS 凭据配置。

## 当前扩展实验入口

以下命令只生成扩展实验结果，不覆盖 `结果/正式结果/full_station/`：

```powershell
python 脚本/04_模型训练/26_run_quality_aware_remote_ab.py
python 脚本/04_模型训练/27_run_chla_gpboost_uncertainty.py
python 脚本/04_模型训练/28_run_zone_specific_full_station_ab.py
```

- 质量感知遥感门控融合结果：`结果/扩展实验/质量感知遥感门控融合_20260905/`。
- Chl-a 点预测与不确定性结果：`结果/扩展实验/叶绿素不确定性预测_20260905/`。
- 分海区建模结果：`结果/扩展实验/分海区建模_20260906/`。这是当前最有希望进入下一轮正式候选比较的扩展模型。
- GPBoost 专项应在 `.venv-gpboost` 中安装 `配置/GPBoost_扩展实验requirements.txt`；它不属于正式主链通用依赖。
- MMoE 不得使用当前 GPBoost 环境。后续应新建 `.venv-mmoe`，并仅在其中安装 `配置/MMoE_独立环境requirements.txt`。

## 说明

- 本次整理仅调整目录结构、路径引用和说明文档，不改变模型算法、训练参数、特征工程逻辑和绘图风格。
- 新增功能前，请先遵守 [AGENTS.md](AGENTS.md) 中的提问与归类规则。
