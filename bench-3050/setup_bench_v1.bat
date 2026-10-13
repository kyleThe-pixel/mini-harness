@echo off
REM One-time setup: isolated venv for the harness runner (psutil, requests, gradio_client).
REM Does NOT touch Forge or wan2gp envs.
cd /d "%~dp0"
python -m venv bench-venv
call bench-venv\Scripts\activate.bat
pip install --upgrade pip
pip install psutil requests gradio_client
echo.
echo Setup done. Edit bench_config.json paths, then run run_bench_v1.bat
pause
