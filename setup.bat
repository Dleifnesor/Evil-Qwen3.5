@echo off
rem ============================================================================
rem  Evil Qwen3.5 - one-time setup for Windows.
rem  Creates a private Python environment in .venv (your other Python packages
rem  are never touched), installs the right PyTorch for your GPU, then checks it.
rem
rem    setup.bat                 auto-detect GPU
rem    setup.bat nvidia          force NVIDIA (RTX 20-series or newer)
rem    setup.bat nvidia-old      force NVIDIA GTX 10-series / older driver
rem    setup.bat amd             force AMD RX 7000 / 9000 (ROCm)
rem    setup.bat cpu             no GPU (slow)
rem
rem  To use a specific Python:  set PYTHON=C:\path\to\python.exe  then run setup.bat
rem  (wrap the path in quotes if it contains spaces)
rem ============================================================================
rem  Note: no delayed expansion, and no %vars% echoed inside ( ) blocks, so folder
rem  names with spaces, ( ) or ! (e.g. "Evil-Qwen3.5-main (1)") are safe.
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
rem keep any globally installed / user packages out of this environment
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_NO_INPUT=1"

rem ---- 1. which PyTorch build? ------------------------------------------------
set "BACKEND=%~1"
if not "%BACKEND%"=="" goto :backend_chosen
set "BACKEND=cpu"
where nvidia-smi >nul 2>&1 || goto :not_nvidia
set "BACKEND=nvidia"
nvidia-smi --query-gpu=name --format=csv,noheader 2>nul | findstr /i /r "GTX Quadro.P Tesla.P TITAN.X" >nul && set "BACKEND=nvidia-old"
goto :backend_chosen
:not_nvidia
powershell -NoProfile -Command "if ((Get-CimInstance Win32_VideoController).Name -match 'Radeon.*RX\s*[79]\d{3}') { exit 0 } else { exit 1 }" >nul 2>&1 && set "BACKEND=amd"
:backend_chosen
if /i "%BACKEND%"=="nvidia" goto :backend_ok
if /i "%BACKEND%"=="nvidia-old" goto :backend_ok
if /i "%BACKEND%"=="amd" goto :backend_ok
if /i "%BACKEND%"=="cpu" goto :backend_ok
echo Unknown option "%BACKEND%". Use: nvidia, nvidia-old, amd or cpu.
goto :fail
:backend_ok
echo [1/5] GPU backend: %BACKEND%
if /i "%BACKEND%"=="cpu" echo       No supported GPU found - it will run on the CPU, which is slow but works with 16 GB+ RAM.

rem ---- 2. Python + private environment in .venv ----------------------------------
rem AMD's PyTorch for Windows only exists for Python 3.12. NVIDIA/CPU work with 3.11-3.13
rem (transformers needs 3.11+).
set "PYOK=import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 13)))"
set "PYRANGE=3.11 - 3.13"
if /i not "%BACKEND%"=="amd" goto :pyok_set
set "PYOK=import sys; sys.exit(sys.version_info[:2] != (3, 12))"
set "PYRANGE=3.12"
:pyok_set
set "VPY=%CD%\.venv\Scripts\python.exe"
if not exist "%VPY%" goto :find_python
"%VPY%" -c "%PYOK%" >nul 2>&1 && goto :venv_ready
echo       Existing .venv is broken or uses an unsupported Python - recreating it.
rmdir /s /q .venv
:find_python
if not defined PYTHON goto :search_python
call %PYTHON% -c "%PYOK%" >nul 2>&1 && goto :have_python
echo ERROR: PYTHON=%PYTHON% is not a working Python %PYRANGE%.
goto :fail
:search_python
for %%C in ("py -3.12" "python3.12" "python" "py -3.13" "py -3.11" "python3.13" "python3.11" "py -3") do (
    %%~C -c "%PYOK%" >nul 2>&1 && ( set "PYTHON=%%~C" & goto :have_python )
)
echo.
echo ERROR: Python 3.12 was not found.
echo   Install it from https://www.python.org/downloads/  - tick "Add python.exe to PATH" -
echo   then run setup.bat again.
goto :fail
:have_python
echo [2/5] Creating .venv with: %PYTHON%
call %PYTHON% -m venv .venv
if not exist "%VPY%" goto :venv_failed
goto :venv_made
:venv_failed
echo ERROR: could not create the .venv folder.
goto :fail
:venv_ready
echo [2/5] Reusing existing .venv
:venv_made
for /f "delims=" %%v in ('call "%VPY%" -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%v"
echo [3/5] Environment: %CD%\.venv  - Python %PYVER%
"%VPY%" -m pip install --quiet --upgrade pip || goto :pip_broken
goto :pip_ok
:pip_broken
echo ERROR: pip does not work inside .venv. Delete the .venv folder and run setup.bat again.
goto :fail
:pip_ok

rem ---- 3. PyTorch ------------------------------------------------------------------
set "TORCH_TEST=import torch, sys; sys.exit(not torch.cuda.is_available())"
if /i "%BACKEND%"=="cpu" set "TORCH_TEST=import torch"
"%VPY%" -c "%TORCH_TEST%" >nul 2>&1 || goto :install_torch
echo [4/5] PyTorch already installed and working - skipping.
goto :torch_done
:install_torch
echo [4/5] Installing PyTorch for %BACKEND% - a few GB, this can take a while...
set "R=https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
if /i "%BACKEND%"=="nvidia"     set "TORCH_ARGS=torch --index-url https://download.pytorch.org/whl/cu130"
if /i "%BACKEND%"=="nvidia-old" set "TORCH_ARGS=torch --index-url https://download.pytorch.org/whl/cu126"
if /i "%BACKEND%"=="cpu"        set "TORCH_ARGS=torch --index-url https://download.pytorch.org/whl/cpu"
if /i "%BACKEND%"=="amd"        set "TORCH_ARGS=%R%/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl %R%/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl %R%/rocm-7.2.1.tar.gz %R%/torch-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
"%VPY%" -m pip install --force-reinstall %TORCH_ARGS% || goto :torch_failed
goto :torch_done
:torch_failed
echo ERROR: PyTorch install failed. Check your internet connection and run setup.bat again.
goto :fail
:torch_done

rem ---- 4. everything else + self-check ------------------------------------------
echo [5/5] Installing the other packages...
"%VPY%" -m pip install --quiet -r requirements.txt || goto :req_failed
goto :req_ok
:req_failed
echo ERROR: installing requirements.txt failed.
goto :fail
:req_ok
echo.
"%VPY%" evil_qwen.py --check || goto :fail
if /i "%BACKEND%"=="cpu" goto :done
"%VPY%" -c "import torch, sys; sys.exit(not torch.cuda.is_available())" >nul 2>&1 && goto :done
echo.
echo WARNING: PyTorch was installed but cannot see your GPU. Update your graphics driver
echo          and run setup.bat again. It will still work on the CPU, just slowly.
:done
echo.
echo Setup complete. Start chatting with:  run.bat
echo The first start downloads the ~9 GB model.
endlocal
pause
exit /b 0

:fail
echo.
echo Setup did not finish. Scroll up for the error.
endlocal
pause
exit /b 1
