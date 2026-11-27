$ErrorActionPreference = 'Stop'

# 可恢复的隐藏批处理入口：复用79号总控及其逐样本/SQLite检查点。
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $projectRoot 'runs/tabiclv2_pilot/环境/.venv/Scripts/python.exe'
$runner = Join-Path $PSScriptRoot '79_run_all_tabicl_explanation_experiments.py'
$validator = Join-Path $PSScriptRoot '81_validate_tabicl_explanation_results.py'
$runRoot = Join-Path $projectRoot 'runs/tabicl_v2_final_explanation_complete_v1'
$logDir = Join-Path $runRoot 'logs'
$logPath = Join-Path $logDir 'master_resumable_run.log'
$exitPath = Join-Path $runRoot 'master_resumable_exit_code.txt'

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "找不到项目Python环境：$pythonExe" }
if (-not (Test-Path -LiteralPath $runner)) { throw "找不到总控脚本：$runner" }

$env:PYTHONHASHSEED = '42'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:TABICL_FIXED_NORM_ORDER = 'none,power'
$patchRoot = Join-Path $projectRoot 'runs/tabicl_v2_deterministic_repair_v1/isolated_patch'
$env:PYTHONPATH = $patchRoot + [IO.Path]::PathSeparator + $env:PYTHONPATH

try {
    & $pythonExe $runner --run-all *>> $logPath
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        $validationLog = Join-Path $logDir 'independent_validation.log'
        & $pythonExe $validator *>> $validationLog
        $code = $LASTEXITCODE
    }
} catch {
    $_ | Out-String | Add-Content -LiteralPath $logPath -Encoding utf8
    $code = 1
}
Set-Content -LiteralPath $exitPath -Value $code -Encoding ascii
exit $code
