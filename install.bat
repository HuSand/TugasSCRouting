@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM  Surabaya Public Facility Routing - Setup
REM  Aman dijalankan berkali-kali (idempotent)
REM =====================================================

set "ENV_NAME=surabaya-routing"
set "VENV_DIR=venv"

echo.
echo  =====================================================
echo   Surabaya Public Facility Routing - Setup
echo  =====================================================
echo.

where conda >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo  [INFO] Conda ditemukan.
    echo.
    goto :CONDA_SETUP
)

echo  [INFO] Conda tidak ditemukan. Menggunakan Python venv.
echo.
goto :VENV_SETUP


REM =====================================================
REM  CONDA PATH
REM =====================================================
:CONDA_SETUP

REM Cek apakah env sudah ada
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo  [INFO] Env '%ENV_NAME%' sudah ada, skip pembuatan ulang.
    echo  [INFO] Hanya update packages.
    echo.
    goto :FIND_PY
)

echo  [STEP 1/3] Membuat conda env '%ENV_NAME%' (Python 3.11)...
conda create -n %ENV_NAME% python=3.11 pip -y
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Gagal membuat conda env.
    goto :FAIL
)
echo.

:FIND_PY
REM Cari python.exe langsung di folder env
set "PY="
for %%d in (
    "%USERPROFILE%\anaconda3\envs\%ENV_NAME%"
    "%USERPROFILE%\Anaconda3\envs\%ENV_NAME%"
    "%USERPROFILE%\miniconda3\envs\%ENV_NAME%"
    "%USERPROFILE%\Miniconda3\envs\%ENV_NAME%"
    "%USERPROFILE%\.conda\envs\%ENV_NAME%"
    "%LOCALAPPDATA%\conda\conda\envs\%ENV_NAME%"
    "C:\ProgramData\anaconda3\envs\%ENV_NAME%"
    "C:\ProgramData\miniconda3\envs\%ENV_NAME%"
) do (
    if exist "%%~d\python.exe" (
        set "PY=%%~d\python.exe"
        goto :INSTALL_PACKAGES
    )
)

REM Fallback: tanya conda
for /f "tokens=*" %%p in ('conda run -n %ENV_NAME% python -c "import sys;print(sys.executable)" 2^>nul') do (
    if exist "%%p" (
        set "PY=%%p"
        goto :INSTALL_PACKAGES
    )
)

echo  [ERROR] Tidak dapat menemukan python.exe di env '%ENV_NAME%'.
goto :FAIL

:INSTALL_PACKAGES
echo  [INFO] Menggunakan: %PY%
echo.
echo  [STEP 2/3] Install / update semua package...
"%PY%" -m pip install --upgrade pip -q
"%PY%" -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Gagal install packages.
    goto :FAIL
)

echo.
echo  [STEP 3/3] Verifikasi...
"%PY%" -c "import osmnx, geopandas, folium, folium.plugins, networkx, pandas, numpy, matplotlib, seaborn, sklearn, shapely; print('  [OK] Semua package OK.')"
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Verifikasi gagal. Ada package yang belum terinstall.
    goto :FAIL
)

goto :SUCCESS_CONDA


REM =====================================================
REM  VENV PATH
REM =====================================================
:VENV_SETUP

python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=python"
) else (
    py --version >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_CMD=py"
    ) else (
        echo  [ERROR] Python tidak ditemukan.
        echo  Download Python 3.11: https://www.python.org/downloads/
        goto :FAIL
    )
)

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo  [INFO] venv sudah ada, skip pembuatan ulang.
    set "PY=%VENV_DIR%\Scripts\python.exe"
    goto :INSTALL_PACKAGES
)

echo  [STEP 1/3] Membuat venv '%VENV_DIR%'...
%PYTHON_CMD% -m venv %VENV_DIR%
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Gagal membuat venv.
    goto :FAIL
)
set "PY=%VENV_DIR%\Scripts\python.exe"
goto :INSTALL_PACKAGES


REM =====================================================
REM  SUCCESS / FAIL
REM =====================================================
:SUCCESS_CONDA
echo.
echo  =====================================================
echo   SETUP SELESAI
echo  =====================================================
echo.
echo  Jalankan: run.bat
echo.
echo  Atau manual (conda aktif):
echo    conda activate %ENV_NAME%
echo    python main.py extract
echo    python main.py compare
echo.
goto :END

:FAIL
echo.
echo  [GAGAL] Perbaiki error di atas lalu jalankan ulang install.bat
echo.
pause
exit /b 1

:END
if not exist data  mkdir data
if not exist logs  mkdir logs
if not exist cache mkdir cache
echo  Folder siap.
echo.
pause
exit /b 0
