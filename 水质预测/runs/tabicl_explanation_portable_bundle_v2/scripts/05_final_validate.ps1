$ErrorActionPreference = 'Stop'
$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $bundleRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run 01_setup_environment.ps1 first.' }
$env:PYTHONPATH = Join-Path $bundleRoot 'scripts'
& $python (Join-Path $bundleRoot 'scripts\final_validate.py')
if ($LASTEXITCODE -ne 0) { throw 'Final validation failed; see logs/final_validation.json.' }
