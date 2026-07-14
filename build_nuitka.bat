@echo off
REM Build script for Monte Carlo CLI via Nuitka (onefile only)
REM
REM Nuitka compiles Python to C then to machine code — this is naturally
REM slower than PyInstaller's bundle-only approach (~5-15 min first build).
REM
REM Usage:
REM   build_nuitka                — build
REM   build_nuitka test           — build then run to verify
REM   build_nuitka clean          — remove build artifacts only
REM   build_nuitka clean test     — clean, build, then run

setlocal enabledelayedexpansion
cd /d "%~dp0"

set DO_CLEAN=
set DO_TEST=

:parse_args
if "%1"=="" goto :done_parse
if /i "%1"=="clean" set DO_CLEAN=1
if /i "%1"=="test" set DO_TEST=1
shift
goto :parse_args
:done_parse

REM --- Clean mode -------------------------------------------------------
if defined DO_CLEAN (
    echo Cleaning build artifacts...
    for %%d in (main.build main.onefile-build main.dist __pycache__) do (
        if exist "%%d" rmdir /s /q "%%d"
    )
    if exist "dist\montecarlo-cli.exe" del /q "dist\montecarlo-cli.exe"
    if exist "dist\main.build" rmdir /s /q "dist\main.build"
    if exist "dist\main.onefile-build" rmdir /s /q "dist\main.onefile-build"
    echo Done.
    if not defined DO_TEST exit /b 0
)

echo === Building montecarlo-cli.exe via Nuitka (onefile) ===
echo Entry:      main.py
echo Icon:       assets\icon.ico
echo.

REM Detect CPU core count for parallel compilation
set NUM_CPUS=%NUMBER_OF_PROCESSORS%
if not defined NUM_CPUS set NUM_CPUS=4

set FLAGS=--onefile
set FLAGS=%FLAGS% --assume-yes-for-downloads
set FLAGS=%FLAGS% --output-filename=montecarlo-cli.exe
set FLAGS=%FLAGS% --windows-icon-from-ico="assets\icon.ico"
set FLAGS=%FLAGS% --output-dir=dist
set FLAGS=%FLAGS% --include-package-data=rich
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
set RICH_MODULES=panel table progress prompt text layout box markup style syntax highlighter ansi live live_render measure emoji json tree columns
for %%m in (%RICH_MODULES%) do set FLAGS=!FLAGS! --include-module=rich.%%m

REM --- Explicitly do NOT bundle a profiles/ directory --------------------
REM Profiles are user data that must persist next to the exe on disk.
REM They must never be baked into the compiled binary as default content,
REM or a fresh copy could be extracted over user data on some future run.
REM If a profiles/ folder exists in this project directory at build time,
REM it is dev/test data only — it is NOT included by this script.

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
if exist "dist\montecarlo-cli.exe" (
    echo === Build complete: dist\montecarlo-cli.exe ===
) else (
    echo === Build complete (check dist\ for output) ===
)

if defined DO_TEST (
    echo.
    echo === Testing executable ===
    echo First run is slower (onefile extracts itself)...
    echo 4 | "dist\montecarlo-cli.exe"
    if errorlevel 1 echo [WARNING] Non-zero exit code: !errorlevel!
    echo.
    echo === Test complete ===
    echo.
    echo IMPORTANT: verify dist\profiles\ was created next to the exe,
    echo not somewhere else. Then move the exe to another folder and
    echo confirm profiles\ is created THERE, not left behind in dist\.
)

echo.
echo Done.
exit /b 0
