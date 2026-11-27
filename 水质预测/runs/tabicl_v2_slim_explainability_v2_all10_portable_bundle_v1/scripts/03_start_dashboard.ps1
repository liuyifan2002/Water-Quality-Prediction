$ErrorActionPreference = 'Stop'
$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$url = 'http://127.0.0.1:8765/'
try {
    $health = Invoke-WebRequest -Uri ($url + 'health') -TimeoutSec 2 -UseBasicParsing
    if ($health.Content -eq 'OK') { Write-Output "看板已运行：$url"; Start-Process $url; return }
} catch { }
$python = Join-Path $bundleRoot '.venv\Scripts\python.exe'
$server = Join-Path $bundleRoot '脚本\05_诊断分析\96_serve_tabicl_slim_dashboard_v1.py'
$output = Join-Path $bundleRoot 'runs\tabicl_v2_slim_explainability_v2_all10\logs'
New-Item -ItemType Directory -Force $output | Out-Null
$proc = Start-Process -FilePath $python -ArgumentList @("`"$server`"", '--port', '8765') -WorkingDirectory $bundleRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'dashboard.stdout.log') `
    -RedirectStandardError (Join-Path $output 'dashboard.stderr.log')
Start-Sleep -Seconds 2
$health = Invoke-WebRequest -Uri ($url + 'health') -TimeoutSec 4 -UseBasicParsing
if ($health.Content -ne 'OK') { throw '看板健康检查失败，请查看dashboard.stderr.log。' }
Write-Output "看板已启动：$url （PID $($proc.Id)）"
Start-Process $url
