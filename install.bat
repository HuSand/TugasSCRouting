@echo off
setlocal enabledelayedexpansion

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
REM  PENTING: semua perintah conda harus pakai "call"
REM  karena conda di Windows adalah conda.bat, bukan .exe
REM =====================================================
:CONDA_SETUP

call conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo  [INFO] Env '%ENV_NAME%' sudah ada, skip pembuatan ulang.
    echo.
    goto :CONDA_PIP
)

echo  [STEP 1/3] Membuat conda env '%ENV_NAME%' (Python 3.11)...
call conda create -n %ENV_NAME% python=3.11 pip -y
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Gagal membuat conda env.
    goto :FAIL
)
echo.

:CONDA_PIP
echo  [STEP 2/3] Install semua package...
call conda run -n %ENV_NAME% pip install --upgrade pip -q
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Gagal upgrade pip.
    goto :FAIL
)
call conda run -n %ENV_NAME% pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Gagal install packages. Lihat error di atas.
    goto :FAIL
)

echo.
echo  [STEP 3/3] Verifikasi...
call conda run -n %ENV_NAME% python -c "import osmnx, geopandas, folium, networkx, pandas, numpy, matplotlib, seaborn, sklearn, shapely; print('  [OK] Semua package terverifikasi.')"
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Verifikasi gagal.
    goto :FAIL
)

goto :SUCCESS


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
        goto :FAIL
    )
)

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo  [INFO] venv sudah ada, skip pembuatan ulang.
    set "PY=%VENV_DIR%\Scripts\python.exe"
    goto :VENV_PIP
)

echo  [STEP 1/3] Membuat venv...
%PYTHON_CMD% -m venv %VENV_DIR%
if %ERRORLEVEL% neq 0 ( goto :FAIL )
set "PY=%VENV_DIR%\Scripts\python.exe"

:VENV_PIP
echo  [STEP 2/3] Install semua package...
"%PY%" -m pip install --upgrade pip -q
"%PY%" -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 ( goto :FAIL )

echo.
echo  [STEP 3/3] Verifikasi...
"%PY%" -c "import osmnx, geopandas, folium, networkx, pandas, numpy, matplotlib, seaborn, sklearn, shapely; print('  [OK] Semua package terverifikasi.')"
if %ERRORLEVEL% neq 0 ( goto :FAIL )

goto :SUCCESS


REM =====================================================
REM  SUCCESS / FAIL
REM =====================================================
:SUCCESS
echo.
echo  =====================================================
echo   SETUP SELESAI - jalankan run.bat
echo  =====================================================
echo.
if not exist data  mkdir data
if not exist logs  mkdir logs
if not exist cache mkdir cache
pause
exit /b 0

:FAIL
echo.
echo  Setup gagal. Perbaiki error di atas lalu jalankan ulang install.bat
echo.
pause
exit /b 1
