# 十任务V2安全续跑入口；不会启动第二个同名调度器，也不会并发V1 GPU计算。
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$program = Join-Path $PSScriptRoot '101_resume_tabicl_all10_explanation_v2.py'
$python = Join-Path $projectRoot 'runs\tabiclv2_pilot\环境\.venv\Scripts\python.exe'
$output = Join-Path $projectRoot 'runs\tabicl_v2_slim_explainability_v2_all10'
if (-not (Test-Path -LiteralPath $python)) { throw "缺少已验证的Python环境：$python" }
if (-not (Test-Path -LiteralPath $program)) { throw "缺少续跑脚本：$program" }
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*101_resume_tabicl_all10_explanation_v2.py*" }
if ($existing) {
    Write-Output "V2调度器已运行：$($existing.ProcessId -join ', ')"
    exit 0
}
New-Item -ItemType Directory -Path (Join-Path $output 'logs') -Force | Out-Null
$worker = Start-Process -FilePath $python -ArgumentList @($program) -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $output 'logs\orchestrator_stdout.log') `
    -RedirectStandardError (Join-Path $output 'logs\orchestrator_stderr.log')
Write-Output "V2调度器已启动，launcher PID=$($worker.Id)。状态：$output\progress_status.json"
