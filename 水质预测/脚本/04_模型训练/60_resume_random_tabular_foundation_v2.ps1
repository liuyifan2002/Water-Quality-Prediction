$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $projectRoot 'runs/tabiclv2_pilot/环境/.venv/Scripts/python.exe'
$runner = Join-Path $PSScriptRoot '59_run_random_tabular_foundation_v2.py'
$tasks = @('chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom')
$models = @('tabicl','tabpfn','tabm')
foreach ($model in $models) {
    foreach ($task in $tasks) {
        & $pythonExe $runner --task $task --model $model
        if ($LASTEXITCODE -ne 0) { Write-Output "FAILED $task $model exit=$LASTEXITCODE" }
    }
}
