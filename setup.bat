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
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
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
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "BACKEND=nvidia"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>nul | findstr /i /r "GTX Quadro.P Tesla.P TITAN.X" >nul && set "BACKEND=nvidia-old"
    goto :backend_chosen
)
powershell -NoProfile -Command "if ((Get-CimInstance Win32_VideoController).Name -match 'Radeon.*RX\s*[79]\d{3}') { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 ( set "BACKEND=amd" & goto :backend_chosen )
set "BACKEND=cpu"
:backend_chosen
if /i not "%BACKEND%"=="nvidia" if /i not "%BACKEND%"=="nvidia-old" if /i not "%BACKEND%"=="amd" if /i not "%BACKEND%"=="cpu" (
    echo Unknown option "%BACKEND%". Use: nvidia, nvidia-old, amd or cpu.
    goto :fail
)
echo [1/5] GPU backend: %BACKEND%
if /i "%BACKEND%"=="cpu" echo       No supported GPU found - it will run on the CPU, which is slow but works with 16 GB+ RAM.

rem ---- 2. find Python -----------------------------------------------------------
rem AMD's PyTorch for Windows only exists for Python 3.12; NVIDIA/CPU work with 3.10-3.13.
if defined PYTHON goto :have_python
py -3.12 -c "import sys" >nul 2>&1 && ( set "PYTHON=py -3.12" & goto :have_python )
python3.12 -c "import sys" >nul 2>&1 && ( set "PYTHON=python3.12" & goto :have_python )
python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1 && ( set "PYTHON=python" & goto :have_python )
if /i "%BACKEND%"=="amd" goto :no_python
for %%V in (3.13 3.11 3.10) do (
    py -%%V -c "import sys" >nul 2>&1 && ( set "PYTHON=py -%%V" & goto :have_python )
)
python -c "import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 13)))" >nul 2>&1 && ( set "PYTHON=python" & goto :have_python )
:no_python
echo.
echo ERROR: Python 3.12 was not found.
echo   Install it from https://www.python.org/downloads/  (tick "Add python.exe to PATH"),
echo   then run setup.bat again.
goto :fail
:have_python
for /f "delims=" %%v in ('%PYTHON% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%v"
echo [2/5] Python: %PYTHON%  (%PYVER%)

rem ---- 3. private environment in .venv ------------------------------------------
set "VPY=%CD%\.venv\Scripts\python.exe"
if exist "%VPY%" (
    "%VPY%" -c "import sys" >nul 2>&1 || ( echo       Existing .venv is broken - recreating it. & rmdir /s /q .venv )
)
if /i "%BACKEND%"=="amd" if exist "%VPY%" (
    "%VPY%" -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1 || ( echo       Existing .venv is not Python 3.12 - recreating it for AMD. & rmdir /s /q .venv )
)
if not exist "%VPY%" (
    %PYTHON% -m venv .venv || ( echo ERROR: could not create the .venv folder. & goto :fail )
)
echo [3/5] Environment: %CD%\.venv
"%VPY%" -m pip install --quiet --upgrade pip || ( echo ERROR: pip does not work inside .venv. & goto :fail )

rem ---- 4. PyTorch ------------------------------------------------------------------
set "TORCH_OK=0"
if /i "%BACKEND%"=="cpu" (
    "%VPY%" -c "import torch" >nul 2>&1 && set "TORCH_OK=1"
) else (
    "%VPY%" -c "import torch, sys; sys.exit(not torch.cuda.is_available())" >nul 2>&1 && set "TORCH_OK=1"
)
if "%TORCH_OK%"=="1" (
    echo [4/5] PyTorch already installed and working - skipping.
    goto :torch_done
)
echo [4/5] Installing PyTorch for %BACKEND% (a few GB, this can take a while^)...
set "R=https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
if /i "%BACKEND%"=="nvidia"     set "TORCH_ARGS=torch --index-url https://download.pytorch.org/whl/cu130"
if /i "%BACKEND%"=="nvidia-old" set "TORCH_ARGS=torch --index-url https://download.pytorch.org/whl/cu126"
if /i "%BACKEND%"=="cpu"        set "TORCH_ARGS=torch --index-url https://download.pytorch.org/whl/cpu"
if /i "%BACKEND%"=="amd"        set "TORCH_ARGS=!R!/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl !R!/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl !R!/rocm-7.2.1.tar.gz !R!/torch-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
"%VPY%" -m pip install --force-reinstall !TORCH_ARGS! || ( echo ERROR: PyTorch install failed. & goto :fail )
:torch_done

rem ---- 5. everything else + self-check ------------------------------------------
echo [5/5] Installing the other packages...
"%VPY%" -m pip install --quiet -r requirements.txt || ( echo ERROR: installing requirements.txt failed. & goto :fail )
echo.
"%VPY%" evil_qwen.py --check || goto :fail
if /i not "%BACKEND%"=="cpu" (
    "%VPY%" -c "import torch, sys; sys.exit(not torch.cuda.is_available())" >nul 2>&1 || (
        echo.
        echo WARNING: PyTorch was installed but cannot see your GPU. Update your graphics driver
        echo          and run setup.bat again. It will still work on the CPU, just slowly.
    )
)
echo.
echo Setup complete. Start chatting with:  run.bat
echo (the first start downloads the ~9 GB model)
endlocal
pause
exit /b 0

:fail
echo.
echo Setup did not finish. Scroll up for the error.
endlocal
pause
exit /b 1
