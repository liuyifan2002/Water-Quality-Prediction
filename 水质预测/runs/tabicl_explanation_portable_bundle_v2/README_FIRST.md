# TabICLv2解释实验可迁移运行包 V2

## 新电脑最短操作

1. 将ZIP复制到新电脑并解压到英文短路径，例如 `D:\tabicl_explain\`。
2. PowerShell进入该目录。
3. 硬件检查：

   `powershell -ExecutionPolicy Bypass -File .\scripts\00_check_hardware.ps1`

4. 创建独立环境：

   `powershell -ExecutionPolicy Bypass -File .\scripts\01_setup_environment.ps1`

5. 先运行跨设备兼容检查：

   `powershell -ExecutionPolicy Bypass -File .\scripts\02_validate_cross_device.ps1`

6. 只有输出 `CROSS_DEVICE_NUMERICALLY_COMPATIBLE` 才续跑：

   `powershell -ExecutionPolicy Bypass -File .\scripts\03_resume_remaining.ps1`

7. 查看进度：

   `powershell -ExecutionPolicy Bypass -File .\scripts\04_status.ps1`

## 冻结结论与协议

- 状态为 `PORTABLE_BUNDLE_CREATED_WITH_KNOWN_NUMERICAL_NONDETERMINISM`；fold1仍为 `FOLD1_NOT_COMPLETE_VALIDATION_FAILURE`，不能写成PASS。
- 原严格重复阈值仍为 `1e-6`；V1失败和V2结论均保留。`0.0014983664` 是目标变量原单位下的历史观测噪声参考，不是理论上界。
- 仅使用P1单查询，每次模型输入必须为 `(1,77)`、float32；不batch、不换模型/特征/split/context/imputer/数值精度。
- RAM缓存用于每个fold-method worker内部复用已加载的数组、imputer输出和固定上下文；模型不按单个query重新加载。GPU只允许一个worker串行。
- 当前调度器按方法启动独立worker。跨方法共用同一Python进程/模型状态没有在原RTX2070上完成等价性验证，因此不启用该优化。
- 小于或接近观测噪声尺度的ALE局部变化标记为 `NUMERICALLY_SENSITIVE_SMALL_EFFECT`；小SHAP贡献标记为 `SMALL_EFFECT_INTERPRET_WITH_CAUTION`。不得删除此类结果。
- 所有冻结输入、V1/V2运行记录和完成结果均位于 `workspace/`。原历史环境清单中出现的绝对路径仅作为不可变来源记录，不是运行依赖；新脚本与配置使用包内相对路径。
- 原运行没有找到可直接复制的20折序列化imputer/context对象。包内包括完整输入、训练规则、fold1黄金hash及预测参考；其余fold的imputer/context由原始冻结输入和原脚本在新机重建。
- 当前包不声称RTX4060已通过兼容性。只有目标机运行第二步预检后才可决定是否继续。绝不输出 `BITWISE_EQUIVALENT`。

## 状态计数

初始checkpoint atomic manifest：419项；已计算/完成 8，待处理或依赖待定 411，支持不足/未审计跳过 0。fold2 rep4保留为独立 `OUTPUT_RECOVERED` 单元。
