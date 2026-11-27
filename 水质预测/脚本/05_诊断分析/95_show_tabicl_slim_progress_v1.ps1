$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = Join-Path $projectRoot 'runs\tabiclv2_pilot\环境\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "找不到项目现有Python环境：$pythonExe"
}
& $pythonExe (Join-Path $PSScriptRoot '94_show_tabicl_slim_progress_v1.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "`n当前相关Python进程："
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match '(89_compute_tabicl_slim_shap_v1|90_compute_tabicl_slim_ale_1d_v1|92_compute_tabicl_slim_ale_2d_v1|93_resume_tabicl_slim_explainability_v1)\.py'
})
if ($active.Count -eq 0) {
    Write-Host '未发现运行中的调度器或计算worker；请检查实验日志。'
} else {
    $active | Select-Object ProcessId, ParentProcessId, @{Name='脚本';Expression={
        if ($_.CommandLine -match '(\d{2}_[\w]+\.py)') { $Matches[1] } else { $_.Name }
    }} | Format-Table -AutoSize
}
