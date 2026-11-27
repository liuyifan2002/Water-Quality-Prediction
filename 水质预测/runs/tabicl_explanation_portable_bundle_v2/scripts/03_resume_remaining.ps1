$ErrorActionPreference = 'Stop'
$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $bundleRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run 01_setup_environment.ps1 first.' }
$gate = Join-Path $bundleRoot 'logs\cross_device_validation.json'
if (-not (Test-Path -LiteralPath $gate)) { throw 'Run 02_validate_cross_device.ps1 first.' }
$decision = Get-Content -LiteralPath $gate -Raw | ConvertFrom-Json
if ($decision.status -ne 'CROSS_DEVICE_NUMERICALLY_COMPATIBLE') { throw 'Cross-device gate is not compatible; stopped without running experiments.' }
$env:PYTHONPATH = (Join-Path $bundleRoot 'scripts') + [IO.Path]::PathSeparator + (Join-Path $bundleRoot 'workspace\runs\tabicl_v2_deterministic_repair_v1\isolated_patch')
$env:PYTHONHASHSEED = '42'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:TABICL_FIXED_NORM_ORDER = 'none,power'
$env:TABICL_PRETRAINED_CHECKPOINT = Join-Path $bundleRoot 'model\tabicl-regressor-v2-20260212.ckpt'
$env:RAM_CACHE_MODE = 'ON'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$counts = & $python (Join-Path $bundleRoot 'scripts\04_status.py') --machine
Write-Host $counts
& $python (Join-Path $bundleRoot 'scripts\portable_runner.py')
if ($LASTEXITCODE -ne 0) { throw "Portable runner exited with code $LASTEXITCODE. Rerun this same script to resume from checkpoints." }
& $python (Join-Path $bundleRoot 'scripts\final_validate.py')
if ($LASTEXITCODE -ne 0) { throw 'Final validation reports unresolved issues; inspect logs/final_validation.json.' }
