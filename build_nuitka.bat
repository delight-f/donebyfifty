@echo off
REM Build script for Monte Carlo CLI via Nuitka
REM
REM Nuitka compiles Python to C then to machine code — this is naturally
REM slower than PyInstaller's bundle-only approach (~5-15 min first build).
REM
REM Usage (double-click to prompt, or run from terminal with arguments):
REM   build_nuitka                      — prompts: onefile or onefolder
REM   build_nuitka onefile              — single-file .exe (for distribution)
REM   build_nuitka standalone           — onefolder (faster dev iteration)
REM   build_nuitka onefile test         — build then run to verify
REM   build_nuitka standalone clean     — clean artifacts then build
REM   build_nuitka clean                — remove all build artifacts only

setlocal enabledelayedexpansion
cd /d "%~dp0"

set MODE=
set DO_CLEAN=
set DO_TEST=

:parse_args
if "%1"=="" goto :done_parse
if /i "%1"=="clean" set DO_CLEAN=1
if /i "%1"=="standalone" set MODE=standalone
if /i "%1"=="onefile" set MODE=onefile
if /i "%1"=="test" set DO_TEST=1
shift
goto :parse_args
:done_parse

REM --- Clean mode (standalone, no build) ------------------------------------
if defined DO_CLEAN (
    echo Cleaning build artifacts...
    for %%d in (montecarlo-cli.dist main.dist main.build main.onefile-build __pycache__) do (
        if exist "%%d" rmdir /s /q "%%d"
    )
    for %%f in (dist\main.exe dist\montecarlo-cli.exe) do if exist "%%f" del /q "%%f"
    for %%d in (dist\main.dist dist\montecarlo-cli.dist dist\main.build dist\main.onefile-build) do if exist "%%d" rmdir /s /q "%%d"
    echo Done.
    if not defined DO_TEST exit /b 0
)

REM --- Interactive prompt if no mode supplied (double-click scenario) -------
if not defined MODE (
    echo.
    echo === Monte Carlo CLI — Nuitka Builder ===
    echo.
    choice /c OF /n /t 10 /d O /m "[O]nefile (single .exe)  or  [F]older (onefolder)  [default O in 10s]: "
    if errorlevel 2 (
        set MODE=standalone
        echo Selected: onefolder
    ) else (
        set MODE=onefile
        echo Selected: onefile
    )
    echo.
)

echo === Building montecarlo-cli.exe via Nuitka ===
echo Mode:       %MODE%
echo Entry:      main.py
echo Icon:       assets\icon.ico
echo.

REM Detect CPU core count for parallel compilation
set NUM_CPUS=%NUMBER_OF_PROCESSORS%
if not defined NUM_CPUS set NUM_CPUS=4

set FLAGS=--assume-yes-for-downloads --show-progress
set FLAGS=%FLAGS% --output-filename=montecarlo-cli.exe
set FLAGS=%FLAGS% --windows-icon-from-ico="assets\icon.ico"
set FLAGS=%FLAGS% --output-dir=dist
set FLAGS=%FLAGS% --jobs=%NUM_CPUS%

REM Enable UPX compression if upx.exe is present
if exist "%~dp0upx.exe" (
    set FLAGS=%FLAGS% --enable-plugin=upx --upx-binary="%~dp0upx.exe"
) else (
    echo [INFO] upx.exe not found -- skipping UPX compression
)

REM Include Rich submodules that Rich lazy-loads at runtime.
REM Must use !FLAGS! (delayed expansion) inside the for loop so each
REM iteration appends to the growing value rather than the initial one.
set RICH_MODULES=panel table progress progress_bar prompt text layout box markup style syntax highlighter ansi live live_render measure emoji json tree columns
for %%m in (%RICH_MODULES%) do set FLAGS=!FLAGS! --include-module=rich.%%m

if /i "%MODE%"=="onefile" (
    set FLAGS=%FLAGS% --onefile
) else (
    set FLAGS=%FLAGS% --standalone
)

echo Flags:  %FLAGS%
echo.
echo Note: First Nuitka build compiles ALL dependencies to C -- expect
echo       5-15 minutes (uses %NUM_CPUS% CPU cores).
echo       Python 3.14 support is experimental in Nuitka 4.1.3 --
echo       if the build fails, try Python 3.13 instead.
echo.

echo Starting build...
echo.

python -m nuitka main.py %FLAGS%

if %ERRORLEVEL% neq 0 (
    echo.
    echo === BUILD FAILED ===
    exit /b %ERRORLEVEL%
)

echo.
if /i "%MODE%"=="onefile" (
    if exist "dist\montecarlo-cli.exe" (
        echo === Build complete: dist\montecarlo-cli.exe (%MODE%) ===
    ) else (
        echo === Build complete (check dist\ for output) ===
    )
) else (
    if exist "dist\montecarlo-cli.dist" (
        dir /b "dist\montecarlo-cli.dist\*.exe" 2>nul
        echo === Build complete: dist\montecarlo-cli.dist\ (onefolder) ===
        echo Run with: dist\montecarlo-cli.dist\montecarlo-cli.exe
    ) else (
        echo === Build complete (check dist\ for output) ===
    )
)

if defined DO_TEST (
    echo.
    echo === Testing executable ===
    if /i "%MODE%"=="onefile" (
        echo First run is slower (onefile extracts itself)...
        echo 4 | "dist\montecarlo-cli.exe"
        if errorlevel 1 echo [WARNING] Non-zero exit code: !errorlevel!
    ) else (
        echo 4 | "dist\montecarlo-cli.dist\montecarlo-cli.exe"
        if errorlevel 1 echo [WARNING] Non-zero exit code: !errorlevel!
    )
    echo.
    echo === Test complete ===
)

echo.
echo Done.
exit /b 0
