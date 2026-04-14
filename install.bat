@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM  Surabaya Public Facility Routing - ONE-CLICK SETUP
REM  Just double-click or run:  install.bat
REM =====================================================

set ENV_NAME=surabaya-routing
set VENV_DIR=venv
set PYTHON_MIN=3.10

echo.
echo  =====================================================
echo   Surabaya Public Facility Routing - Setup
echo  =====================================================
echo.

REM ---------- Try Conda first ----------
where conda >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo  [INFO] Conda detected. Using conda environment.
    echo.
    goto :CONDA_SETUP
)

REM ---------- Fallback: plain Python venv ----------
echo  [INFO] Conda not found. Using Python venv instead.
echo.
goto :VENV_SETUP


REM =====================================================
REM  CONDA PATH
REM =====================================================
:CONDA_SETUP

REM Remove old env if it exists so we start clean
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo  [INFO] Removing existing conda env '%ENV_NAME%'...
    conda env remove -n %ENV_NAME% -y >nul 2>&1
)

echo  [STEP 1/3] Creating conda env '%ENV_NAME%' (Python 3.11)...
conda create -n %ENV_NAME% python=3.11 pip -y
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Failed to create conda env. Check your conda installation.
    goto :FAIL
)

echo.
echo  [STEP 2/3] Installing packages via pip inside conda env...
conda run -n %ENV_NAME% pip install --upgrade pip -q
conda run -n %ENV_NAME% pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Package installation failed. See errors above.
    goto :FAIL
)

echo.
echo  [STEP 3/3] Verifying installation...
conda run -n %ENV_NAME% python -c "import osmnx, geopandas, folium, networkx, sklearn; print('  [OK] All packages verified.')"
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Verification failed. Some packages may be missing.
    goto :FAIL
)

echo.
echo  =====================================================
echo   SETUP COMPLETE (conda)
echo  =====================================================
echo.
echo  To use this project:
echo.
echo    1. Activate the environment:
echo         conda activate %ENV_NAME%
echo.
echo    2. Run the scripts in order:
echo         python 01_extract_facilities.py
echo         python 02_explore_data.py
echo         python 03_routing_demo.py
echo.
goto :END


REM =====================================================
REM  VENV (pip) PATH
REM =====================================================
:VENV_SETUP

REM Check Python version
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    py --version >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo  [ERROR] Python not found.
        echo  Download Python 3.11 from: https://www.python.org/downloads/
        echo  Make sure to check "Add Python to PATH" during install.
        goto :FAIL
    )
    set PYTHON_CMD=py
) else (
    set PYTHON_CMD=python
)

echo  Python found:
%PYTHON_CMD% --version
echo.

REM Remove old venv if present
if exist "%VENV_DIR%" (
    echo  [INFO] Removing existing venv...
    rmdir /s /q "%VENV_DIR%"
)

echo  [STEP 1/3] Creating virtual environment '%VENV_DIR%'...
%PYTHON_CMD% -m venv %VENV_DIR%
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Failed to create virtual environment.
    goto :FAIL
)

echo.
echo  [STEP 2/3] Installing packages (may take 3-5 minutes)...
call %VENV_DIR%\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Package installation failed. See errors above.
    goto :FAIL
)

echo.
echo  [STEP 3/3] Verifying installation...
python -c "import osmnx, geopandas, folium, networkx, sklearn; print('  [OK] All packages verified.')"
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Verification failed. Some packages may be missing.
    goto :FAIL
)

echo.
echo  =====================================================
echo   SETUP COMPLETE (venv)
echo  =====================================================
echo.
echo  To use this project:
echo.
echo    1. Activate the environment:
echo         %VENV_DIR%\Scripts\activate.bat
echo.
echo    2. Run the scripts in order:
echo         python 01_extract_facilities.py
echo         python 02_explore_data.py
echo         python 03_routing_demo.py
echo.
goto :END


REM =====================================================
REM  FAIL / END
REM =====================================================
:FAIL
echo.
echo  Setup did not complete. Fix the errors above and re-run install.bat
echo.
pause
exit /b 1

:END
if not exist data mkdir data
echo  Project folder is ready. Good luck!
echo.
pause
exit /b 0
