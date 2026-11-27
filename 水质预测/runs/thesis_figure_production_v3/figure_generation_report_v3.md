# V3图件生成报告

本轮只进行了既有SHAP/ALE结果的绘图和质量验收；未运行模型、预测、SHAP或ALE计算。

- 生成PNG/PDF图件：59组；另有1条GLOBAL-01既有图引用、1条DIST-02暂不生成记录。
- 已生成成果分类统计：`COMPUTED_RESULT_FIGURE=52`、`SUPPORT_AUDIT_FIGURE=5`、`NUMERICAL_STABILITY_FIGURE=2`。
- DIST-02 Violin按要求暂不生成。
- MAIN-01～03 V1/V2未覆盖。
- ALE 2D仅生成支持审查矩阵，未生成响应曲面。
- STABILITY-01仅9个任务，缺少Chl-a surface；图注限定为三次重复探针范围，不是预测置信区间。
- STABILITY-02使用历史冻结文件 `runs/tabicl_explanation_numerical_stability_v2/jitter_vs_explanation_scale.csv`，未重新定义阈值。
