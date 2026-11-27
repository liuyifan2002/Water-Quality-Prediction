@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 请先运行安装脚本。
  pause
  exit /b 1
)
.venv\Scripts\python.exe run_experiment.py --models xgboost catboost --resume
pause
