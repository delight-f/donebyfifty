# build.ps1
#
# Builds montecarlo-cli.exe from source using PyInstaller.
# Run from anywhere: .\build.ps1 (the script changes to its own directory).
#
# What this does, step by step:
#   1. Checks python is on PATH
#   2. Creates/uses a .venv and installs project + build dependencies
#   3. Runs PyInstaller against build.spec
#   4. Reports where the finished exe landed

$ErrorActionPreference = "Stop"   # any failed cmdlet stops the script immediately

Set-Location $PSScriptRoot

# Native commands do not honour $ErrorActionPreference, so every call goes
# through here and its $LASTEXITCODE is checked.
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$What,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$What failed (exit code $LASTEXITCODE). See output above." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "== Checking Python ==" -ForegroundColor Cyan
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "python not found on PATH. Install Python 3.11+ and try again." -ForegroundColor Red
    exit 1
}
Invoke-Native "python --version" { python --version }

Write-Host "== Setting up build environment ==" -ForegroundColor Cyan
$VenvDir = Join-Path $PSScriptRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Invoke-Native "venv creation" { python -m venv $VenvDir }
}

# Installs the project itself plus the [build] extras group defined in
# pyproject.toml (PyInstaller + its hook contrib package).
Invoke-Native "pip upgrade" { & $VenvPython -m pip install --upgrade pip }
Invoke-Native "dependency install" { & $VenvPython -m pip install -e ".[build]" }

Write-Host "== Running PyInstaller ==" -ForegroundColor Cyan
Invoke-Native "PyInstaller build" { & $VenvPython -m PyInstaller build.spec --noconfirm --clean }

$exePath = Join-Path $PSScriptRoot "dist\montecarlo-cli.exe"
if (Test-Path $exePath) {
    Write-Host "== Build succeeded ==" -ForegroundColor Green
    Write-Host "Executable: $exePath"
} else {
    Write-Host "Build reported success but exe not found at expected path: $exePath" -ForegroundColor Red
    exit 1
}
