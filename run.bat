@echo off
rem Evil Qwen3.5 - start chatting.  Extra options are passed through, e.g.:
rem   run.bat --chat --preset lyra
rem   run.bat --compare -p "What do you think of humans?"
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
if exist ".venv\Scripts\python.exe" goto :run
echo Not set up yet - running setup.bat first...
call setup.bat || exit /b 1
:run
if "%~1"=="" goto :chat
".venv\Scripts\python.exe" evil_qwen.py %*
goto :after
:chat
".venv\Scripts\python.exe" evil_qwen.py --chat
:after
if not errorlevel 1 exit /b 0
echo.
echo Something went wrong - see the message above.
echo If it mentions a missing package or PyTorch, run setup.bat again.
pause
exit /b 1
