@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 请先运行 setup_nvidia.bat。TabM 和 TabPFN 的完整验证需要 NVIDIA GPU。
  pause
  exit /b 1
)

.venv\Scripts\python.exe run_candidate_validation.py --modes station_5fold --output-dir outputs_station_5fold
pause
