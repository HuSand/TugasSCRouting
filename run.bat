@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM  Surabaya Routing Platform — Run Menu
REM =====================================================

REM --- Detect environment ---
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
    set "ENV=venv"
    goto :MENU
)
where conda >nul 2>&1
if %ERRORLEVEL% equ 0 (
    conda env list | findstr /C:"surabaya-routing" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PY=conda run --no-capture-output -n surabaya-routing python"
        set "ENV=conda: surabaya-routing"
        goto :MENU
    )
)
echo.
echo  [ERROR] No environment found. Run install.bat first.
echo.
pause & exit /b 1


:MENU
cls
echo.
echo  =========================================================
echo   Surabaya Public Facility Routing Platform
echo   Env: %ENV%
echo  =========================================================
echo.
echo   1   Extract facilities + road network  (internet, ~10 min)
echo   2   Explore and profile data
echo   3   Routing demonstrations
echo   4   Algorithm comparison benchmark
echo   ─────────────────────────────────────────────────────
echo   5   Run full pipeline  (1 → 2 → 3 → 4)
echo.
echo   0   Exit
echo.
set /p "CHOICE= Select [0-5]: "

if "%CHOICE%"=="1" ( call :RUN extract   & goto :BACK )
if "%CHOICE%"=="2" ( call :RUN explore   & goto :BACK )
if "%CHOICE%"=="3" ( call :RUN demo      & goto :BACK )
if "%CHOICE%"=="4" ( call :RUN compare   & goto :BACK )
if "%CHOICE%"=="5" ( call :RUN all       & goto :BACK )
if "%CHOICE%"=="0" exit /b 0
echo  Invalid choice. & timeout /t 1 >nul & goto :MENU


:RUN
echo.
%PY% main.py %1
exit /b %ERRORLEVEL%

:BACK
echo.
echo  Press any key to return to menu...
pause >nul
goto :MENU
