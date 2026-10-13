@echo off
REM Run the benchmark harness. Pass through args, e.g.:
REM   run_bench_v1.bat --skip Wan-2.2-5B-TI2V
REM   run_bench_v1.bat --models Bonsai-27B-Q1_0,MiMo-V2.6-Distill-Qwen-9B
cd /d "%~dp0"
if not exist bench-venv\Scripts\activate.bat (
    echo bench-venv not found - run setup_bench_v1.bat first
    pause
    exit /b 1
)
call bench-venv\Scripts\activate.bat
python bench.py %*
pause
