# GEE 本机认证状态报告

更新时间：2026-05-18

## 检查结论

- Python 环境：`Python 3.11.9`
- `earthengine-api`：已安装
- `earthengine-api` 版本：`1.7.26`
- `earthengine` CLI：可用
- `earthengine` CLI 版本：`earthengine 1.7.26`
- GEE 认证：已完成
- `ee.Initialize()`：成功
- 是否需要 Google Cloud Project ID：当前不需要显式指定 project
- 使用的 Project ID：未使用显式 project id
- Sentinel-2 集合访问：成功，可访问 `COPERNICUS/S2_SR_HARMONIZED`
- Sentinel-2 最小访问测试：`limit(1).size().getInfo()` 返回 `1`
- 下一步是否可以生成 / 运行 GEE 导出脚本：可以生成脚本；运行导出脚本前仍应逐项确认导出范围、输出资产或 Drive 路径、任务命名和配额。

## 执行记录

1. 检查 `python --version`，确认当前 Python 为 `3.11.9`。
2. 检查 `pip show earthengine-api`，初始状态未安装。
3. 执行 `pip install earthengine-api`，安装完成，版本为 `1.7.26`。
4. 检查 `earthengine --help` 和 `earthengine --version`，CLI 可用。
5. 直接运行 `earthengine authenticate` 时，本机 Chrome 自动启动出现应用程序错误；随后改为手动授权链接流程。
6. 手动授权完成后，成功保存 Earth Engine authorization token。
7. 运行 `ee.Initialize()`，不指定 project 即可初始化成功。
8. 访问 `COPERNICUS/S2_SR_HARMONIZED` 的最小集合元数据，确认 Sentinel-2 集合可访问。
9. 执行 `earthengine task list` 只读检查历史任务状态；本轮未启动任何新任务。

## 安全与边界

- 本报告未记录授权码、token、Google 账号信息或 credential 文件内容。
- 本轮未下载 GEE 数据。
- 本轮未运行 Sentinel-2 / Landsat 导出任务。
- 本轮未修改已有实验结果。
- 本轮未把认证文件放入项目目录。
- Earth Engine 凭证使用官方默认用户级凭证位置，由 `earthengine authenticate` 管理。

## 注意事项

- CLI 调用时出现 TensorFlow / SciPy / NumPy 版本提示：当前提示来自 `tensorflow.python.keras` 导入链，未阻止 `earthengine` CLI、认证、`ee.Initialize()` 或 Sentinel-2 集合访问。
- 本机 Chrome 自动启动认证流程存在崩溃问题；后续如再次认证，建议继续使用手动生成授权链接和 authorization code 的流程，避免自动拉起 Chrome。
