$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = Join-Path $projectRoot 'runs\tabiclv2_pilot\环境\.venv\Scripts\python.exe'
$serverScript = Join-Path $PSScriptRoot '96_serve_tabicl_slim_dashboard_v1.py'
$outputDir = Join-Path $projectRoot 'runs\tabicl_v2_slim_explainability_v1'
$url = 'http://127.0.0.1:8765/'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "找不到现有Python环境：$pythonExe" }
try {
    $health = Invoke-WebRequest -Uri ($url + 'health') -TimeoutSec 2 -UseBasicParsing
    if ($health.Content -eq 'OK') {
        Write-Host "看板已运行：$url"
        Start-Process $url
        return
    }
} catch { }
$stdout = Join-Path $outputDir 'dashboard.stdout.log'
$stderr = Join-Path $outputDir 'dashboard.stderr.log'
$process = Start-Process -FilePath $pythonExe -ArgumentList @("`"$serverScript`"", '--port', '8765') `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Start-Sleep -Seconds 2
try {
    $health = Invoke-WebRequest -Uri ($url + 'health') -TimeoutSec 3 -UseBasicParsing
    if ($health.Content -ne 'OK') { throw '健康检查没有返回OK' }
} catch {
    throw "看板启动失败或端口8765被其他程序占用；请查看 $stderr。原进程ID：$($process.Id)"
}
Write-Host "看板已启动：$url （PID $($process.Id)）"
Start-Process $url
