@echo off
REM Photo Tagger first-time setup.
REM Reuses your Forge Neo portable Python (has torch+CUDA, gradio, pillow already)
REM and adds the tagger's extra packages into it.
setlocal
set "PY=C:\ForgeNeo\sd-webui-forge-neo\system\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] Forge python not found at:
  echo   %PY%
  echo Edit this file and set PY to a Python 3.10+ on your machine, then re-run.
  pause
  exit /b 1
)
echo Using %PY%
"%PY%" -m pip install --upgrade easyocr pillow-heif requests
echo.
echo Setup done. Now double-click start_tagger_v1.bat
pause
