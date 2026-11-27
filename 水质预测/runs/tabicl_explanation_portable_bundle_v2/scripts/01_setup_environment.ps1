$ErrorActionPreference = 'Stop'
$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venv = Join-Path $bundleRoot '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { & py -3.11 -m venv $venv }
    else {
        $globalPython = Get-Command python -ErrorAction SilentlyContinue
        if (-not $globalPython) { throw 'Python 3.11 x64 is required; install it, then rerun this script.' }
        & python -m venv $venv
    }
}
if (-not (Test-Path -LiteralPath $python)) { throw 'Could not create the bundle-local Python 3.11 environment.' }
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $bundleRoot 'requirements-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. No experiment was started.' }
$env:PYTHONPATH = Join-Path $bundleRoot 'scripts'
& $python (Join-Path $bundleRoot 'scripts\validate_environment.py')
if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed; do not continue to cross-device check.' }
& $python (Join-Path $bundleRoot 'scripts\validate_checkpoint.py')
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint validation failed; do not continue.' }
Write-Host 'Environment and checkpoint checks passed. Next: scripts\02_validate_cross_device.ps1'
