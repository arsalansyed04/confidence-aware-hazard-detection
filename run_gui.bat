@echo off
setlocal enabledelayedexpansion

:: Change directory to the repository root where this batch file is located
cd /d "%~dp0"

title Weather-Adaptive Hazard Detection - GUI Demo

echo =====================================================================
echo    Weather-Adaptive Hazard Detection - Desktop GUI Demo Launcher
echo =====================================================================
echo.

:: Detect Python executable (prefer workspace .venv, fallback to venv or system python)
set "PYTHON_EXE="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=*" %%i in ('where python') do (
            if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
        )
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python environment was not found!
    echo Please make sure Python is installed or run setup in the .venv folder.
    echo.
    pause
    exit /b 1
)

:: Verify GUI script exists
if not exist "%~dp0src_core\gui.py" (
    echo [ERROR] GUI script not found at "%~dp0src_core\gui.py".
    echo Please make sure this file is placed in the project root directory.
    echo.
    pause
    exit /b 1
)

:: Set recommended environment variables
set "PYTHONUNBUFFERED=1"
set "HF_HUB_DISABLE_XET=1"

echo Using Python: !PYTHON_EXE!
echo Launching GUI demo interface...
echo.

"!PYTHON_EXE!" "%~dp0src_core\gui.py" %*
set "EXIT_CODE=!ERRORLEVEL!"

if not "!EXIT_CODE!"=="0" (
    echo.
    echo [WARNING] GUI application exited with exit code !EXIT_CODE!.
    pause
)

exit /b !EXIT_CODE!
