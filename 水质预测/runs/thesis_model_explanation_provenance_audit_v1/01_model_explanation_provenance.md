# 第三章模型解释对象溯源报告

核查对象为 `文档/论文材料/硕士论文TabICLv2_SHAP_ALE_插图整合初稿_20261031_v4/水质预测硕士学位论文_TabICLv2_SHAP_ALE_插图整合初稿.docx`。本轮仅读取论文、脚本、日志、模型资产清单和冻结结果，并核对 DOCX 内媒体图片与原图的 SHA256；没有运行任何模型、预测、SHAP 或 ALE。

## 一、结论

1. **SHAP并非只解释XGBoost。** 图3-1、图3-2及图3-8至图3-18、图C-1至图C-20解释TabICLv2；图3-3至图3-7及图3-60、图3-61的SHAP侧来自数值等价受控复现的XGBoost。后两图分别为XGBoost SHAP与XGBoost来源消融对照、XGBoost任务内排名汇总；不能写成TabICLv2 SHAP。
2. **ALE也不是单一模型。** 图3-19至图3-22为历史TabICLv2五折一维ALE；图3-23至图3-27为受控复现XGBoost一维ALE；图3-28至图3-47为TabICLv2十任务各fold_1的P1单查询一维ALE。图3-52至图3-55是XGBoost二维ALE交互组图，图3-56为相应覆盖审查。图3-57至图3-59只是TabICLv2 P1候选的4×4联合支持审查，**没有计算该三组二维响应曲面**。
3. 当前Word共核对**81张解释相关插图**（第三章61张、附录C20张）：TabICLv2相关64张（其中3张仅为二维候选支持审查）、XGBoost相关17张。每张图片均与上述三处原图目录之一字节级SHA256一致。此数目是图件归属统计，不是独立实验或质量通过数量。
4. 第3.1节第301段“SHAP和ALE解释对象明确限定为XGBoost”与实际插图冲突，属于模型归属错误；第3.5.1节将“20个通过数值一致性核验的折模型”及“TreeSHAP 20/20”置于TabICLv2方法之下，也混淆了两套审查结果。第3.6节和第3.7节需按图号、模型、预测协议分别限定，不应把不同模型或不同折次的曲线连成一套结果。

## 二、模型和方法调用证据

| 证据链 | 真实模型与预测函数 | SHAP/ALE方法与尺度 | 当前图件 |
|---|---|---|---|
| `脚本/05_诊断分析/58_compute_xgb_shap_ale_v2.py` → `runs/xgb_interpretation_refit_v1/artifacts/{task}/{fold}/xgboost_refit_model.ubj`、`median_imputer.joblib` → `runs/xgb_shap_ale_v2/` | 加载 `XGBRegressor` 和该折median imputer；`pred_final` 用 `model.predict` 后 `expm1` 与非负截断 | `get_booster().predict(DMatrix, pred_contribs=True)` 为XGBoost原生TreeSHAP，归因在`log1p(target)`内部尺度；ALE调用`pred_final`，在原始物理尺度 | 图3-3至7、23至27、52至56、60至61 |
| `脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py` → `67_run_tabiclv2_model_agnostic_shap_v3.py`、`69_run_tabiclv2_ale_v3.py` → `runs/tabicl_v2_explainability_v3/` → `72_finalize_tabiclv2_explanation_closure_v1.py` | `models/tabiclv2_adapter.py` 的 `TabICLv2Regressor`；基于各折训练上下文和median imputer；输出经`expm1`与非负截断 | `PermutationExplainer`使用边际masker与固定256批次锚点包装；ALE脚本的扰动预测调用批查询函数。两者为原始物理尺度，**不是无条件独立逐样本函数** | 图3-1至2、19至22 |
| `脚本/05_诊断分析/88_run_tabicl_slim_explainability_v1.py` → `99_compute_tabicl_all10_shap_v2.py`、`100_compute_tabicl_all10_ale_1d_v2.py` → `runs/tabicl_v2_slim_explainability_v2_all10/` → `runs/thesis_figure_production_v3/scripts/produce_v3_figures.py` | 同一TabICLv2适配器；每次`adapter.predict(row[None,:])`为P1单查询，checkpoint、训练上下文和特征顺序由预检核验 | 模型无关`PermutationExplainer`和一维ALE，均在原始目标尺度；各任务只用fold_1。与历史整批冻结预测严格等价**未获统一证明** | 图3-8至18、28至51、57至59、图C-1至20 |

`shap.TreeExplainer`并非本项目XGBoost路线的实际API调用；脚本调用的是原生`pred_contribs`，其算法性质为TreeSHAP。不能因两个路线都使用“SHAP”而认定模型相同。TabM、TabPFN用于现代基线比较；所审解释脚本没有加载它们用于上述图件。

## 三、模型资产真实性与历史调整

- 2026-10-18 01:30 的 `runs/xgb_shap_ale_v1/xgb_shap_ale_v1_decision.md` 判定原始冻结XGBoost模型及预处理器未存档，0/20折可加载；早期以原始冻结资产直接解释的方案没有执行。
- 之后 `runs/xgb_interpretation_refit_v1/06_xgb_refit_final_decision.md` 记录20/20折受控重新拟合模型与正式冻结测试预测严格数值一致，最大逐样本误差约 `3.55e-15`；但它们**不是找回的原始二进制模型**，原始Git commit和逐折环境锁仍缺失。`xgb_shap_ale_v2_final_decision.md`确认其产生20折TreeSHAP、16条一维ALE和8组二维候选；原始XGBoost权重不可与这些复现资产混称。
- TabICLv2使用预训练回归checkpoint及各折训练上下文，并非另有一套可直接视为“本地重新训练”的独立冻结权重。迁移清单 `transfer/model_runtime_patch_v1/checkpoint_inventory.csv` 记录 `tabicl-regressor-v2-20260212.ckpt`，大小114,324,594字节，SHA256=`0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`，来源快照revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`。P1脚本核对加载checkpoint的SHA256；历史20折每个运行进程的实际加载哈希证据仍需按原始日志逐项确认，不能仅凭当前迁移副本反推。
- TabICLv2历史四任务五折解释在 `runs/tabicl_v2_explainability_closure_v1/12_final_decision.md` 记录20折175个77维SHAP样本、80条逐折一维ALE曲线，但最终状态为 `PARTIALLY_COMPLETE`：Chl-a surface的1、2、3、5折及Secchi第4折未通过严格逐样本复现；七组历史二维ALE为部分支持。故“完成计算”不能写成“20折全部通过质量门槛”。
- 后续稳定性审查揭示预测函数随查询集合变化：P1两探针三进程达到小样本重复性阈值；P3的32查询重排/拆批最大差`0.0089841061`，P1与整批同一观测差`0.00042734`。固定256包装器仅有有限探针稳定证据。历史批查询归因与后续P1归因的**模型身份一致、条件预测函数不相同**。
- 十任务P1扩展明确写为 `P1_FUNCTION_EXPLANATION_NOT_FROZEN_BATCH_EQUIVALENCE`。原始方案、P1门槛失败和数值抖动审查均未被后续落盘结果自动改写为通过。`runs/tabicl_explanation_numerical_stability_v2/numerical_stability_decision.md`保留历史抖动`0.0014983664`和小效应谨慎解释结论。
- 2026-10-30的V3绘图脚本读取冻结文件生成新图；2026-10-31的插图整合稿在旧TabICLv2图、旧XGBoost图基础上新增P1图，**图件增加而第3.1节原句未同步修改**。此为图文冲突的直接来源之一；不以文件名或修改日期代替脚本、数据和图片哈希证据。

## 四、输出尺度与科学边界

XGBoost TreeSHAP在`log1p(target)`尺度；XGBoost ALE在`expm1`并非负截断后的原始目标尺度。TabICLv2上述SHAP和ALE均使用原始目标尺度的包装预测，但历史固定批查询与新P1单查询代表不同**条件预测函数**。不能比较两模型原始SHAP绝对值，不能用XGBoost归因解释TabICLv2内部决策，不能把统计归因或ALE响应视为环境因果效应。图3-48、49、56至59是支持审查，不应表述为模型响应；图3-50是九任务三次P1重复探针范围，不是预测置信区间；图3-51为历史抖动尺度对照，不代表新阈值。

逐图证据、原图路径、媒体哈希和结果路径见 `02_figure_model_mapping.csv`；所读取脚本、报告、结果、模型与图片的SHA256见 `03_artifact_evidence_inventory.csv`。本审计状态为**ANALYZED（只读核查）**，不等同于重新执行模型或验证每条历史数值结果。
