@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$items = @('outputs_random_seeds','outputs_time_holdout','outputs_station_5fold') | Where-Object { Test-Path $_ }; if ($items.Count -eq 0) { throw '未找到正式验证输出目录。请先完成至少一项正式验证。' }; if (Test-Path '候选模型稳健性验证结果.zip') { $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'; Move-Item -LiteralPath '候选模型稳健性验证结果.zip' -Destination ('历史回传包_' + $stamp + '.zip') }; Compress-Archive -Path $items -DestinationPath '候选模型稳健性验证结果.zip' -Force"
if errorlevel 1 (
  echo 打包失败，请保留屏幕上的错误信息。
  pause
  exit /b 1
)
echo 已生成 候选模型稳健性验证结果.zip
pause
