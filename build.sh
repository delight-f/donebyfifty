#!/usr/bin/env bash
#
# Builds montecarlo-cli from source using PyInstaller.
# Run from the repo root: ./build.sh
#
# What this does, step by step:
#   1. Checks python3 is on PATH
#   2. Creates/uses a .venv and installs project + build dependencies
#   3. Runs PyInstaller against build.spec
#   4. Reports where the finished binary landed

set -euo pipefail

cd "$(dirname "$0")"

echo "== Checking Python =="
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "python3 not found on PATH. Install Python 3.11+ and try again." >&2
    exit 1
fi
echo "Using $("$PYTHON" --version)"

echo "== Setting up build environment =="
VENV=.venv
if [ ! -d "$VENV" ]; then
    "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[build]"

echo "== Running PyInstaller =="
python -m PyInstaller build.spec --noconfirm --clean

EXE=dist/montecarlo-cli
if [ -x "$EXE" ]; then
    echo "== Build succeeded =="
    echo "Executable: $(pwd)/$EXE"
else
    echo "Build reported success but no executable found at: $EXE" >&2
    exit 1
fi
