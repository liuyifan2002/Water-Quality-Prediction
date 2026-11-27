param([Parameter(Mandatory=$true)][int]$RunProcessId)
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $projectRoot 'runs/tabiclv2_pilot/环境/.venv/Scripts/python.exe'
$runner = Join-Path $PSScriptRoot '79_run_all_tabicl_explanation_experiments.py'
$validator = Join-Path $PSScriptRoot '81_validate_tabicl_explanation_results.py'
$runRoot = Join-Path $projectRoot 'runs/tabicl_v2_final_explanation_complete_v1'
$logDir = Join-Path $runRoot 'logs'
$watchLog = Join-Path $logDir 'completion_watchdog.log'
$statusPath = Join-Path $runRoot 'completion_watchdog_status.txt'
$pauseFlag = Join-Path $runRoot 'PAUSE_AFTER_CURRENT_FOLD.flag'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

"$(Get-Date -Format o) waiting_for_pid=$RunProcessId" | Add-Content -LiteralPath $watchLog -Encoding utf8
while (Get-Process -Id $RunProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
"$(Get-Date -Format o) main_runner_exited" | Add-Content -LiteralPath $watchLog -Encoding utf8
if (Test-Path -LiteralPath $pauseFlag) {
    "$(Get-Date -Format o) pause_flag_detected_no_resume" | Add-Content -LiteralPath $watchLog -Encoding utf8
    Set-Content -LiteralPath $statusPath -Value 'PAUSED_BY_FLAG' -Encoding ascii
    exit 0
}

$env:PYTHONHASHSEED = '42'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:TABICL_FIXED_NORM_ORDER = 'none,power'
$patchRoot = Join-Path $projectRoot 'runs/tabicl_v2_deterministic_repair_v1/isolated_patch'
$env:PYTHONPATH = $patchRoot + [IO.Path]::PathSeparator + $env:PYTHONPATH

& $pythonExe $runner --summarize *>> $watchLog
$code = $LASTEXITCODE
$decisionPath = Join-Path $runRoot '14_final_explanation_closure_decision.md'
$decision = if (Test-Path -LiteralPath $decisionPath) { Get-Content -LiteralPath $decisionPath -Raw } else { '' }
if ($code -eq 0 -and $decision -notmatch 'EXPLANATION_COMPLETE_WITH_LIMITATIONS') {
    "$(Get-Date -Format o) one_checkpoint_resume_started" | Add-Content -LiteralPath $watchLog -Encoding utf8
    & $pythonExe $runner --run-all *>> $watchLog
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        & $pythonExe $runner --summarize *>> $watchLog
        $code = $LASTEXITCODE
    }
}
if ($code -eq 0) {
    & $pythonExe $validator *>> $watchLog
    $code = $LASTEXITCODE
}
Set-Content -LiteralPath $statusPath -Value $code -Encoding ascii
"$(Get-Date -Format o) watchdog_exit_code=$code" | Add-Content -LiteralPath $watchLog -Encoding utf8
exit $code
