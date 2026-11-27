# 修复后77特征TabICLv2解释计算迁移包

数据版本：WQ77_REPAIRED_20261115_V1。先阅读08_reports/MISSING_ASSETS.md。包复制到任意目录解压即可；勿只复制05_scripts。

## 顺序

1. 根据06_environment/INSTALL_WINDOWS.md安装源版本依赖。
2. `python 05_scripts/run_preflight.py --static-only`：仅文件/哈希/语法。
3. `python 05_scripts/run_preflight.py --dependencies`：版本与CUDA可用性，不预测。
4. **以下仅在另一台计算机执行**：`python 05_scripts/run_prediction_audit.py`。每任务一次原完整第1折测试批，外加3核心各3次P1重复；原完整批是必要协议，不能改成仅3行批再声称复现。
5. `python 05_scripts/run_shap.py`：十任务各3核心，共30个77维结果；可单任务`--task tp_middle`。
6. `python 05_scripts/run_ale.py`：原20变量/分箱/查询ID逐曲线；Secchi ssrd_sum_3d登记冲突会被记录为阻塞，其余继续。数据问题未解决不能声称20/20。
7. `python 05_scripts/run_quality_audit.py`：完整性、排名和支持汇总。需种子审查时显式运行`python 05_scripts/run_shap.py --sensitivity`，10次重复不计独立核心样本。

串行总入口：`python 05_scripts/run_all.py`。它依次预检、预测审查、SHAP、ALE、质量审查；若部分任务失败继续其他任务并返回非零。不会执行二维ALE或可选20样本扩展。

结果只写outputs/；shap按样本检查点，ALE按SQLite局部查询检查点续跑；输入哈希不同不覆盖。每次退出查看outputs/status、outputs/logs、summary/quality_audit.csv。请勿同时启动多个GPU入口。断电后重跑同一命令；活跃PID锁防并发，失效锁会清理。

50折完整上下文已打包，但本轮原登记解释仅第1折。run_prediction_audit不重新训练权重，官方fit只恢复保存上下文；不重算正式预测实验指标，也不改写冻结预测。

30核心汇总为每任务3样本的mean(abs(SHAP))，对全部77特征任务内排序；跨任务原SHAP绝对值不可直接比较。ALE真实分箱样本/站点数另列，空箱不填零，不插值，不将其解释为因果或总体稳定性通过。

当前电脑构建包后立即停止；目的端安装/模型加载/数值资格仍待实际执行。
