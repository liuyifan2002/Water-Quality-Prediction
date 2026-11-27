@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path 'results_to_return.zip') { $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'; Move-Item 'results_to_return.zip' ('results_to_return_previous_' + $stamp + '.zip') }"
powershell -NoProfile -Command "Compress-Archive -Path outputs\* -DestinationPath results_to_return.zip -Force"
echo 已生成 results_to_return.zip，请把这个文件复制回原电脑。
pause
