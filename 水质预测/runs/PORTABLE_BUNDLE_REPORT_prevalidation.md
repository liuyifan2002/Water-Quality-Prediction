# TabICLv2十任务解释迁移包报告

- 生成状态：`PORTABLE_BUNDLE_READY`
- Self-check：`PASS`
- 本机V2：`PAUSED_FOR_PORTABLE_HANDOFF`（V2 GPU计算单元数0）
- V1完成结果快照：3个V1注册SHAP样本、0条V1 ALE曲线；另包含历史fold1 pilot两条SHAP及两条ALE完整记录。V1仍在本机运行，其后续新增结果不会自动同步进本压缩包。
- 任务：十项预登记任务，各自fold 1；现有V1结果在新机按哈希复用。
- ZIP大小：122,358,047 bytes；SHA256：`876f73413c62246474a75eb0c68d704dd3c1ae909d23946ced1d21741934400d`
- 模型SHA256：`0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`
- 77 strict feature manifest SHA256：`115cee492668ac4da82560279b9d5a8fb9c664a0cb740a2cbc46ddedd15a1573`
- 已保存完成SHAP数量：3；后续剩余数量以目标机状态页为准。
- 本机V1当前任务不在压缩包中运行；迁移包包含打包时已完成的只读快照，不能含打包后本机新增检查点。
- 新机需联网安装锁定依赖；运行包不含venv。
- 不启用batch query、FP16、autocast或其他改变预测语义的优化；单GPU串行。
- 运行代码和配置使用解压根目录相对路径；旧V1审计CSV里的绝对路径仅是历史来源元数据。

## 新电脑启动

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\00_check_hardware.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\01_setup_environment.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\02_resume_remaining.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\03_start_dashboard.ps1
```
