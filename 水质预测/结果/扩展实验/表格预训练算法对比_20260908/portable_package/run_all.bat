@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 请先运行 setup_nvidia.bat。
  pause
  exit /b 1
)
.venv\Scripts\python.exe run_experiment.py --resume
pause
