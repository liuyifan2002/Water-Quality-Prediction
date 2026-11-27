# TABICLV2_EXPLAINABILITY_V3 最终决策

## 决策

`PARTIALLY_COMPLETE`

## 完成范围

- 四项任务均获得真实调用冻结TabICLv2预测函数的77维模型无关SHAP；总计15个合格折、75个解释样本。
- TN surface与TP middle完成5折SHAP；Chl-a surface完成3折；Secchi完成2折。
- 一维ALE完成13个再次严格复现的折；二维ALE只保留通过联合覆盖预审的结果。
- 已生成TabICLv2—XGBoost排名/来源组对照，未直接比较不同输出尺度的原始SHAP值。
- 已生成论文图件并写入Word新副本；原稿和正式预测结果均未修改。

## 未达到EXPLANATION_COMPLETE的原因

5个折在独立进程中未达到事前严格数值复现阈值，且Word仍需人工逐页视觉QA。不得将当前结果表述为四任务全部五折解释完成。
