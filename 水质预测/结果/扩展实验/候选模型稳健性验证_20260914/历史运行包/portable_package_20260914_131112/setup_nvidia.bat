@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_CMD="
py -3.11 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD (
  python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
  echo 未找到可用的 Python 3.11 或 3.12。请先安装 64 位 Python。
  pause
  exit /b 1
)
%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
  echo 虚拟环境创建失败，安装已停止。
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo CUDA PyTorch is unavailable. Do not run TabPFN or TabM until this is fixed.
  pause
  exit /b 1
)
echo 环境安装结束。TabPFN 第一次运行时还会下载本地模型权重。
pause
