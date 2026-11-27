TabICLv2 WQ77 可解释性计算结果包
生成日期：2026-11-20（北京时间）

目录：
- outputs/shap：30个核心样本SHAP结果、背景样本登记和运行时记录。
- outputs/ale_1d：20条ALE曲线、分箱支持、查询清单、数值重复审查及Secchi登记修订。
- outputs/audit：10个任务的模型身份、冻结整批预测复现和P1重复审查。
- outputs/summary：汇总排名、ALE真实支持和50项质量审查。
- outputs/checkpoints：ALE逐查询局部效应SQLite，便于核查和断点分析。
- outputs/logs、outputs/status：运行日志及逐项状态；早期失败日志为历史记录，请以最终状态和quality_audit.csv为准。
- 04_explanation_config：原登记配置与查询清单；05_scripts：计算及审查脚本；08_reports：源包静态核验材料。

最终执行状态：30/30核心SHAP、20/20 ALE曲线，质量完整性审查50/50通过。
Secchi ssrd_sum_3d使用显式登记修订v2：两条样本原值50345940、修复值50345942 J/m²，箱号3改4；原查询ID与分箱边界均保留。详见outputs/ale_1d/secchi_ssrd_sum_3d_registration_revision.json。
重要限制：10/10任务的冻结整批预测数值复现均未达到严格阈值，虽模型身份审查及P1重复审查通过，不能宣称与冻结预测完全一致。详见outputs/audit/*.json。
本包用于结果分析，不包含约870MB的模型、原始训练/测试数据和划分资产；如需在另一台电脑重新推理，请使用原始可移植包与相同环境。
PACKAGE_MANIFEST.csv列出包内每个文件的大小与SHA-256；压缩包本身的SHA-256另见同目录.sha256文件。
