$ErrorActionPreference = 'Stop'
Write-Host '=== TabICLv2 portable V2 hardware check ==='
Write-Host ("Computer: {0}" -f $env:COMPUTERNAME)
Write-Host ("Logical CPU: {0}" -f [Environment]::ProcessorCount)
try {
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Host ("Physical RAM: {0:N1} GB" -f ($os.TotalVisibleMemorySize / 1MB))
} catch { Write-Warning "Could not read physical RAM: $_" }
try {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
} catch { Write-Warning 'nvidia-smi unavailable; CUDA environment validation will fail closed.' }
Write-Host 'Next: powershell -ExecutionPolicy Bypass -File .\scripts\01_setup_environment.ps1'
