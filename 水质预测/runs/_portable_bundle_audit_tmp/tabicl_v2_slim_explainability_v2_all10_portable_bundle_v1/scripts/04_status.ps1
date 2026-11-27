$ErrorActionPreference = 'Stop'
$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$statusPath = Join-Path $bundleRoot 'runs\tabicl_v2_slim_explainability_v2_all10\progress_status.json'
if (Test-Path -LiteralPath $statusPath) {
    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    Write-Output "Queue stage: $($status.stage)"
    Write-Output "Current task: $($status.task)"
    Write-Output "Current unit: $($status.unit)"
    Write-Output "Updated: $($status.updated_at)"
    Write-Output "Detail: $($status.detail)"
}
$registryPath = Join-Path $bundleRoot 'runs\tabicl_v2_slim_explainability_v2_all10\run_registry.csv'
if (Test-Path -LiteralPath $registryPath) {
    $registry = Import-Csv -LiteralPath $registryPath
    $registry | Group-Object status | Sort-Object Name | ForEach-Object { Write-Output ("{0}: {1}" -f $_.Name,$_.Count) }
    foreach ($task in @('chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom')) {
        $coverage = $registry | Where-Object { $_.task -eq $task -and $_.method -eq 'COVERAGE' } | Select-Object -Last 1
        $state = if ($coverage) { $coverage.status } else { 'PENDING' }
        Write-Output "$task : $state"
    }
}
$v2 = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*101_resume_tabicl_all10_explanation_v2.py*' }
Write-Output "V2 process count: $(@($v2).Count)"
