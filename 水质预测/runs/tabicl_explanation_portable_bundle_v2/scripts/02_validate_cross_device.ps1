$ErrorActionPreference = 'Stop'
$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $bundleRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run 01_setup_environment.ps1 first.' }
$env:PYTHONPATH = (Join-Path $bundleRoot 'scripts') + [IO.Path]::PathSeparator + (Join-Path $bundleRoot 'workspace\runs\tabicl_v2_deterministic_repair_v1\isolated_patch')
$env:PYTHONHASHSEED = '42'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:TABICL_FIXED_NORM_ORDER = 'none,power'
$env:TABICL_PRETRAINED_CHECKPOINT = Join-Path $bundleRoot 'model\tabicl-regressor-v2-20260212.ckpt'
& $python (Join-Path $bundleRoot 'scripts\validate_cross_device.py')
if ($LASTEXITCODE -ne 0) { throw 'CROSS_DEVICE_DIFFERENCE_REQUIRES_REVIEW; no remaining explanation units were started.' }
