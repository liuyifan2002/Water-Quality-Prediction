$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $projectRoot 'runs/tabiclv2_pilot/环境/.venv/Scripts/python.exe'
$runner = Join-Path $PSScriptRoot '57_run_random_tabular_model_zoo_v2.py'
$tasks = @('chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom')
$models = @('ridge','xgboost','lightgbm','catboost','random_forest','extra_trees','hist_gb','mlp','svr_rbf')
foreach ($model in $models) {
    foreach ($task in $tasks) {
        & $pythonExe $runner --task $task --model $model
        if ($LASTEXITCODE -ne 0) { Write-Output "FAILED $task $model exit=$LASTEXITCODE" }
    }
}
