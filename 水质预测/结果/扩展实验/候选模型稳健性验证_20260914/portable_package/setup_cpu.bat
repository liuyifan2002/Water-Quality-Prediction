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
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo CPU 环境安装结束。CPU 模式不建议运行 TabPFN 和 TabM 全量实验。
pause
