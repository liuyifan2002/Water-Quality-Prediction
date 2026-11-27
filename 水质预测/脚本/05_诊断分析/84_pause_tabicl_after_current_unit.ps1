param(
    [Parameter(Mandatory=$true)][int]$WorkerProcessId,
    [Parameter(Mandatory=$true)][int]$RunnerProcessId,
    [Parameter(Mandatory=$true)][int]$LauncherProcessId,
    [Parameter(Mandatory=$true)][int]$WatchdogProcessId
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runRoot = Join-Path $projectRoot 'runs/tabicl_v2_final_explanation_complete_v1'
$log = Join-Path $runRoot 'logs/pause_after_current_unit.log'
"$(Get-Date -Format o) waiting_worker=$WorkerProcessId" | Add-Content -LiteralPath $log -Encoding utf8
while (Get-Process -Id $WorkerProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Milliseconds 100 }
"$(Get-Date -Format o) atomic_worker_exited; pausing_scheduler" | Add-Content -LiteralPath $log -Encoding utf8
foreach ($id in @($RunnerProcessId,$LauncherProcessId,$WatchdogProcessId)) {
    if (Get-Process -Id $id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $id -Force
        "$(Get-Date -Format o) stopped_scheduler_pid=$id" | Add-Content -LiteralPath $log -Encoding utf8
    }
}
