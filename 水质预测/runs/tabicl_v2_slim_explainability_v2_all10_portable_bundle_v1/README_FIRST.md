# TabICLv2十任务解释运行包

1. 将ZIP复制到Windows电脑并解压到英文路径，例如 `D:\tabicl_all10\`。
2. 在PowerShell进入解压目录。
3. 执行 `powershell -ExecutionPolicy Bypass -File .\scripts\00_check_hardware.ps1`。
4. 执行 `powershell -ExecutionPolicy Bypass -File .\scripts\01_setup_environment.ps1`。
5. 执行 `powershell -ExecutionPolicy Bypass -File .\scripts\02_resume_remaining.ps1` 开始/续跑。
6. 执行 `powershell -ExecutionPolicy Bypass -File .\scripts\03_start_dashboard.ps1`，浏览器打开 `http://127.0.0.1:8765/`。
7. 用 `powershell -ExecutionPolicy Bypass -File .\scripts\04_status.ps1` 查看十任务状态。

任务范围为十项任务各自fold 1。每个查询仍为P1单查询、77特征、float32；SHAP为PermutationExplainer，155 evaluations、16训练背景。V1历史失败保留，V2先计算再按证据判断。模型不重训。
