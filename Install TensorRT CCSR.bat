@echo off
setlocal
title Install TensorRT CCSR for ComfyUI
cd /d "%~dp0"
echo Installing TensorRT CCSR for ComfyUI and its GPU runtime...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
set "exit_code=%errorlevel%"
echo.
if not "%exit_code%"=="0" (
  echo Installation failed. See the message above and outputs\install.log.
) else (
  echo Installation complete.
  echo TensorRT acceleration is now active for CCSR in ComfyUI.
)
pause
exit /b %exit_code%
