@echo off
REM Build script for Monte Carlo CLI
REM Builds a single-file executable via PyInstaller
REM
REM Usage:
REM   build          — build with defaults
REM   build clean    — remove build/dist artifacts first
REM   build test     — build then run the exe to verify

setlocal

cd /d "%~dp0"

if "%1"=="clean" (
    echo Cleaning build artifacts...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    if exist "*.spec" del /q "*.spec"
    echo Done.
    if "%2"=="" exit /b 0
)

echo Building montecarlo-cli.exe...
echo UPX compression: enabled
echo Icon: assets\icon.ico
echo.

python -m PyInstaller build.spec --clean --noconfirm

if %ERRORLEVEL% neq 0 (
    echo.
    echo Build failed.
    exit /b %ERRORLEVEL%
)

echo.
echo Build complete: dist\montecarlo-cli.exe

if "%1"=="test" (
    echo.
    echo Testing executable...
    echo 4 | dist\montecarlo-cli.exe
)
