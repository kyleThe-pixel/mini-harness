@echo off
REM Photo Tagger launcher - opens the search UI at http://127.0.0.1:7862
cd /d "%~dp0"
set "PY=C:\ForgeNeo\sd-webui-forge-neo\system\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] Forge python not found at:
  echo   %PY%
  echo Edit this file and set PY to your Python, then re-run.
  pause
  exit /b 1
)
"%PY%" app.py
pause
