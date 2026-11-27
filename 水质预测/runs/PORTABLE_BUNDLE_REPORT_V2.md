# TabICLv2解释实验便携运行包 V2 报告

- 打包状态：`PORTABLE_BUNDLE_CREATED_WITH_KNOWN_NUMERICAL_NONDETERMINISM`
- Self-check：`PASS`
- fold1历史状态：`FOLD1_NOT_COMPLETE_VALIDATION_FAILURE`（原1e-6阈值和失败记录保持）
- V1稳定性结论：`INFERENCE_NONDETERMINISM_CONFIRMED`
- V2数值稳定性结论：`NUMERICAL_JITTER_NONNEGLIGIBLE_FOR_SMALL_EFFECTS`
- 历史观测噪声参考：`0.0014983664`（不是理论上界）
- Fold1已完成：2个SHAP样本、2条ALE曲线；第一条ALE重复性失败未改写，第二条保留PASS。
- Fold2 rep4：`OUTPUT_RECOVERED_AFTER_SCHEDULER_PAUSE`，模型输入/输出文件及registry已打包。
- Atomic units：总数 419；已完成/已计算 8；待处理或依赖待定 411；支持不足/未审计终止 0。
- 还未开始的任务折：19/19（fold1是历史pilot且失败状态保留；fold2 rep4单独恢复）。
- ZIP大小及SHA256由同目录 `.zip` 与 `.zip.sha256` 给出；为避免自引用，ZIP内本报告不嵌入ZIP自身hash。
- Checkpoint SHA256：`0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`
- Python环境：保存了真实来源 `pip freeze` 和版本manifest；没有复制旧 `.venv`。
- 新电脑第一次必须运行跨GPU兼容检查。RTX4060尚未在本机实测，包不宣称跨GPU已验证，也不输出bitwise等价。
- 本机打包过程未调用TabICLv2预测、未计算SHAP/ALE、未启动任何剩余fold。
- RAM缓存：按fold-method worker驻留X/y、77特征矩阵、imputer输出、support索引和固定context；同一query不重复读CSV/重建模型。进程上限目标24GB。
- 长生命周期：模型/上下文在每个fold-method worker内只初始化一次，所有query仍逐条P1调用；未经数值等价检查，不跨method共用同一模型实例。
- 运行脚本和配置无项目E盘硬编码；历史冻结日志中的路径只作来源元数据，不参与新机解析。
- imputer/context序列化对象：原运行目录未发现完整20折对象文件。运行包携带全部训练输入、manifest、拟合规则和可核对hash，在目标机由原协议重建；fold1黄金查询覆盖五个既有探针。

## 新电脑启动命令

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\00_check_hardware.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\01_setup_environment.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\02_validate_cross_device.ps1
# 只有报告为CROSS_DEVICE_NUMERICALLY_COMPATIBLE时才继续：
powershell -ExecutionPolicy Bypass -File .\scripts\03_resume_remaining.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\04_status.ps1
```

ZIP SHA256：`f687af9247c793b4e00504b1324b0d031558620f8b8678c4391c8a4bec4d4e44`
ZIP大小：`124323570` bytes。
