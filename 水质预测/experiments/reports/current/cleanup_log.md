# 清理记录

更新时间：2026-05-18

## 本轮 GEE 本机认证检查

- 已安装 `earthengine-api==1.7.26`。
- 已确认 `earthengine` CLI 可用。
- 已完成 Earth Engine 本机认证。
- 已验证 `ee.Initialize()` 不指定 project 可以成功。
- 已验证 `COPERNICUS/S2_SR_HARMONIZED` 最小集合元数据可访问。
- 未新增检测脚本，因此未向 `scripts/data/` 写入文件。
- 未下载 Sentinel-2 / Landsat 数据。
- 未启动任何 GEE Export task。
- 未修改已有实验结果。
- 未将授权码、token、账号信息或 credential 文件内容写入项目文件。

## 清理状态

- 检查项目目录内 `__pycache__`：未发现需要清理的目录。
- 未在项目根目录生成临时文件。
