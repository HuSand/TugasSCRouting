@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM  Surabaya Routing Platform - Run Menu
REM =====================================================

set "ENV_NAME=surabaya-routing"

REM --- 1. Cek venv dulu ---
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
    set "ENV_LABEL=venv"
    goto :MENU
)

REM --- 2. Cari python.exe langsung di conda env dirs ---
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
        set "ENV_LABEL=conda: %ENV_NAME%"
        goto :MENU
    )
)

REM --- 3. Fallback: tanya conda langsung ---
where conda >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%p in ('conda run -n %ENV_NAME% python -c "import sys;print(sys.executable)" 2^>nul') do (
        if exist "%%p" (
            set "PY=%%p"
            set "ENV_LABEL=conda: %ENV_NAME%"
            goto :MENU
        )
    )
)

echo.
echo  [ERROR] Environment '%ENV_NAME%' tidak ditemukan.
echo  Jalankan install.bat terlebih dahulu.
echo.
pause
exit /b 1


:MENU
cls
echo.
echo  =========================================================
echo   Surabaya Public Facility Routing Platform
echo   Env: %ENV_LABEL%
echo  =========================================================
echo.
echo   1   Extract facilities + road network  (internet, ~10 min)
echo   2   Explore and profile data
echo   3   Routing demonstrations
echo   4   Algorithm comparison benchmark
echo   7   Algorithm comparison benchmark  (parallel legs, max CPU)
echo   ---------------------------------------------------------
echo   5   Run full pipeline  (1 -^> 2 -^> 3 -^> 4)
echo   6   Open GA Evolution Viewer  (run 4 first)
echo.
echo   0   Exit
echo.
set /p "CHOICE= Select [0-7]: "

if "%CHOICE%"=="1" goto :DO1
if "%CHOICE%"=="2" goto :DO2
if "%CHOICE%"=="3" goto :DO3
if "%CHOICE%"=="4" goto :DO4
if "%CHOICE%"=="5" goto :DO5
if "%CHOICE%"=="6" goto :DO6
if "%CHOICE%"=="7" goto :DO7
if "%CHOICE%"=="0" exit /b 0
echo  Pilihan tidak valid.
timeout /t 1 >nul
goto :MENU

:DO1
echo.
"%PY%" main.py extract
goto :BACK

:DO2
echo.
"%PY%" main.py explore
goto :BACK

:DO3
echo.
"%PY%" main.py demo
goto :BACK

:DO4
echo.
"%PY%" main.py compare
goto :BACK

:DO5
echo.
"%PY%" main.py all
goto :BACK

:DO7
echo.
"%PY%" main.py compare --parallel-legs
goto :BACK

:DO6
echo.
if exist "data\evolution_viewer.html" (
    echo  Opening GA Evolution Viewer in browser...
    start "" "data\evolution_viewer.html"
) else (
    echo  [INFO] evolution_viewer.html not found.
    echo  Run option 4 ^(Algorithm comparison^) first to generate it.
    pause
)
goto :BACK

:BACK
echo.
echo  Press any key to return to menu...
pause >nul
goto :MENU
