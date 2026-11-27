$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonExe = Join-Path $projectRoot 'runs/tabiclv2_pilot/环境/.venv/Scripts/python.exe'
$runner = Join-Path $PSScriptRoot '62_evaluate_random_tabular_final_v2.py'
$outputDir = Join-Path $projectRoot 'runs/random_tabular_model_zoo_v2'
$plan = Join-Path $outputDir '06_final_selection_plan.json'
if (-not (Test-Path -LiteralPath $plan)) { throw 'Internal selection plan is absent; external test is prohibited.' }
$tasks = @('chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom')
foreach ($task in $tasks) {
    $files = Get-ChildItem -LiteralPath (Join-Path $outputDir '03_inner_metrics') -Filter "$task`_*.json" -File
    foreach ($file in $files) {
        $model = $file.BaseName.Substring($task.Length + 1)
        if ($model -in @('tabpfn','tabm','tabdpt')) { continue }
        & $pythonExe $runner --task $task --model $model
        if ($LASTEXITCODE -ne 0) { Write-Output "FAILED final $task $model exit=$LASTEXITCODE" }
    }
}
foreach ($model in @('tabpfn','tabm')) {
    foreach ($task in $tasks) {
        $inner = Join-Path $outputDir "03_inner_metrics/$task`_$model.json"
        if (-not (Test-Path -LiteralPath $inner)) { continue }
        & $pythonExe $runner --task $task --model $model
        if ($LASTEXITCODE -ne 0) { Write-Output "FAILED final $task $model exit=$LASTEXITCODE" }
    }
}
