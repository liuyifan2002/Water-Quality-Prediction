@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python Launcher was not found. Install Python 3.11 first.
  pause
  exit /b 1
)
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
echo CPU 环境安装结束。CPU 模式不建议运行 TabPFN 和 TabM 全量实验。
pause
