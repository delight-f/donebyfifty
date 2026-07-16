# build.ps1
#
# Builds montecarlo-cli.exe from source using PyInstaller.
# Run from the repo root: .\build.ps1
#
# What this does, step by step:
#   1. Checks python is on PATH
#   2. Installs/updates project dependencies + PyInstaller
#   3. Runs PyInstaller against build.spec
#   4. Reports where the finished exe landed

$ErrorActionPreference = "Stop"   # any failed command stops the script immediately

Write-Host "== Checking Python ==" -ForegroundColor Cyan
$pythonVersion = python --version
if (-not $?) {
    Write-Error "python not found on PATH. Install Python 3.11+ and try again."
    exit 1
}
Write-Host "Using $pythonVersion"

Write-Host "== Installing dependencies ==" -ForegroundColor Cyan
# Installs the project itself plus the [build] extras group defined in
# pyproject.toml (PyInstaller + its hook contrib package).
python -m pip install --upgrade pip
python -m pip install -e ".[build]"

Write-Host "== Running PyInstaller ==" -ForegroundColor Cyan
python -m PyInstaller build.spec --noconfirm --clean

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed. See output above."
    exit 1
}

$exePath = Join-Path (Get-Location) "dist\montecarlo-cli.exe"
if (Test-Path $exePath) {
    Write-Host "== Build succeeded ==" -ForegroundColor Green
    Write-Host "Executable: $exePath"
} else {
    Write-Error "Build reported success but exe not found at expected path: $exePath"
    exit 1
}
