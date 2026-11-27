$ErrorActionPreference = 'Stop'
$bundleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venv = Join-Path $bundleRoot '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.11 -m venv $venv }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { & python -m venv $venv }
    else { throw '请先安装Python 3.11 x64后重试。' }
}
if (-not (Test-Path -LiteralPath $python)) { throw 'Python 3.11 venv创建失败。' }
& $python -m pip install -r (Join-Path $bundleRoot 'requirements-lock.txt')
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败；未启动解释计算。' }
& $python (Join-Path $PSScriptRoot 'validate_bundle.py')
if ($LASTEXITCODE -ne 0) { throw '运行包资产核验失败。' }
Write-Output '环境依赖与运行包文件核验通过。'
