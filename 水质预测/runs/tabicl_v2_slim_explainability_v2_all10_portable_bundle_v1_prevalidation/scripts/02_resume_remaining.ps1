$ErrorActionPreference = 'Stop'
$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $bundleRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw '请先运行 scripts\01_setup_environment.ps1。' }
& $python (Join-Path $PSScriptRoot 'validate_bundle.py')
if ($LASTEXITCODE -ne 0) { throw '迁移包校验失败，停止续跑。' }
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*101_resume_tabicl_all10_explanation_v2.py*' -and $_.CommandLine -like "*$bundleRoot*" }
if ($existing) { Write-Output "十任务调度器已运行：$($existing.ProcessId -join ', ')"; exit 0 }
$env:PYTHONHASHSEED = '42'
$env:PYTHONPATH = Join-Path $bundleRoot 'runs\tabicl_v2_deterministic_repair_v1\isolated_patch'
$env:TABICL_PRETRAINED_CHECKPOINT = Join-Path $bundleRoot 'model\tabicl-regressor-v2-20260212.ckpt'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$output = Join-Path $bundleRoot 'runs\tabicl_v2_slim_explainability_v2_all10'
New-Item -ItemType Directory -Force (Join-Path $output 'logs') | Out-Null
$proc = Start-Process -FilePath $python -ArgumentList @((Join-Path $bundleRoot '脚本\05_诊断分析\101_resume_tabicl_all10_explanation_v2.py')) `
    -WorkingDirectory $bundleRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $output 'logs\orchestrator_stdout.log') `
    -RedirectStandardError (Join-Path $output 'logs\orchestrator_stderr.log')
Write-Output "十任务续跑器已启动，launcher PID=$($proc.Id)。状态见 runs\tabicl_v2_slim_explainability_v2_all10\progress_status.json。"
