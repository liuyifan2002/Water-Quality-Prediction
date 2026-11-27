# 表格预训练算法对比便携运行包

## 一、实验目的

在完全相同的数据、特征和固定随机 80%/20% 划分下，对比 XGBoost、CatBoost、TabPFN 和 TabM。实验不使用同日其他水质指标，不覆盖原项目正式结果。

## 二、数据说明

- 数据行数：30,719
- 数据列数：108
- 时间范围：2014—2024 年
- 研究对象：香港近岸 94 个站点
- 目标：Chl-a、透明度、TN、TP
- 水层：surface、middle、bottom
- 已删除同期水质协变量，只保留四个目标值、目标自身历史特征、环境特征、时间空间特征和正式模型实际使用的遥感可用性字段。
- 透明度在三个水层任务中沿用同一站点日期观测值；论文解释时不能把三层记录当作三次独立透明度测量。
- 正式主表中未找到的候选字段数量：0。这些字段不会在本实验中临时补造。

## 三、推荐电脑

- Windows 10/11，64 位 Python 3.11 或 3.12。
- NVIDIA 显卡，建议显存不少于 8 GB。
- 至少 16 GB 内存，磁盘剩余空间建议 15 GB 以上。
- 首次安装依赖时需要联网；TabPFN 回归权重已随本包提供，运行时不再访问 Hugging Face。

## 四、运行步骤

1. 解压整个 ZIP，不要只打开压缩包直接运行。
2. 双击 `setup_nvidia.bat`。没有 NVIDIA 显卡时可运行 `setup_cpu.bat`。
3. 先双击 `run_quick.bat`，确认四个代表任务能完成。
4. 快速测试无严重错误后，双击 `run_all.bat` 运行全部任务。
5. 如果 TabPFN 或 TabM 安装失败，可先运行 `run_tree_only.bat`，检查数据和固定划分是否可复现。
6. 全部结束后双击 `pack_results.bat`，得到 `results_to_return.zip`。
7. 将 `results_to_return.zip` 复制回原电脑交给 Codex 分析。

脚本支持断点继续。重新双击 `run_all.bat` 时，已经成功的任务会跳过，失败任务会再次尝试。

## 五、必须带回的结果

`results_to_return.zip` 中应至少包含：

- `model_metrics.csv`：全部模型指标；
- `model_predictions.csv`：逐样本实测值和预测值；
- `split_manifest.csv`：训练集和测试集编号；
- `feature_manifest.csv`：每个任务的完整特征；
- `environment.json` 和 `pip_freeze.txt`：电脑与软件环境；
- `run_status.json`：完成和失败数量；
- `model_errors.log`：如有失败，保存完整报错；
- `结果摘要.md` 和两张对比图。

## 六、重要限制

- 不要修改 `data/model_table.csv`、`config.json` 或 `reference_metrics.csv`。
- 不要把盐度、DO、pH、浊度等同期水质重新加入输入。
- 不要根据测试集结果修改参数后重复报告同一测试集成绩。
- TabPFN 必须使用本地运行方式，本包不会把数据上传到第三方接口。
- `models/tabpfn-v2-regressor.ckpt` 是 TabPFN 的固定本地回归权重，不要移动、改名或删除。
- 本轮只判断新算法是否值得继续。通过后才进行多水层共享模型和年份/留站点验证。

## 七、远程协作

当前 Codex 不能自动接管另一台未连接的电脑。最简单的方式是在另一台电脑安装 Codex，打开本目录并让它运行 `run_all.bat`；或者运行完成后把 `results_to_return.zip` 带回当前项目。不要通过聊天发送账号、密码或远程控制验证码。
