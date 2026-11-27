$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $projectRoot 'runs/tabiclv2_pilot/环境/.venv/Scripts/python.exe'
$runner = Join-Path $PSScriptRoot '61_tune_random_tabular_v2.py'
$tasks = @('chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom')
foreach ($task in $tasks) {
    & $pythonExe $runner --task $task
    if ($LASTEXITCODE -ne 0) { Write-Output "FAILED tuning $task exit=$LASTEXITCODE" }
}
