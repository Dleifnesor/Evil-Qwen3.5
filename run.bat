@echo off
rem Evil Qwen3.5 - start chatting.  Extra options are passed through, e.g.:
rem   run.bat --chat --preset lyra
rem   run.bat --compare -p "What do you think of humans?"
setlocal
cd /d "%~dp0"
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
if not exist ".venv\Scripts\python.exe" (
    echo Not set up yet - running setup.bat first...
    call setup.bat || exit /b 1
)
if "%~1"=="" (
    ".venv\Scripts\python.exe" evil_qwen.py --chat
) else (
    ".venv\Scripts\python.exe" evil_qwen.py %*
)
if errorlevel 1 (
    echo.
    echo Something went wrong - see the message above. Running setup.bat again usually fixes it.
    pause
)
endlocal
