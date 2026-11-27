$ErrorActionPreference = 'Stop'
$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $bundleRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Environment not installed; progress file can still be viewed directly.' }
$env:PYTHONPATH = Join-Path $bundleRoot 'scripts'
& $python (Join-Path $bundleRoot 'scripts\04_status.py')
