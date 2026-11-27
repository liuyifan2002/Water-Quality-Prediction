$ErrorActionPreference = 'Stop'
$computer = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum
$ram = [math]::Round($computer.TotalPhysicalMemory / 1GB, 1)
Write-Output "CPU logical threads: $($cpu.Sum)"
Write-Output "Physical RAM: $ram GB"
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) { & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv }
else { Write-Warning '未检测到nvidia-smi；确认已安装NVIDIA驱动和CUDA兼容GPU。' }
if ($ram -lt 16) { Write-Warning 'RAM低于16GB，建议提高Windows虚拟内存并监控运行进程。' }
