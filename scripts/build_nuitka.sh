#!/usr/bin/env bash
# Build script for Monte Carlo CLI via Nuitka (macOS / Linux)
# Creates a single-file executable (or standalone directory)
#
# Usage:
#   ./build_nuitka.sh            — build standalone (default, works with profiles)
#   ./build_nuitka.sh onefile    — build onefile exe (note: cannot resolve
#                                   exe dir — profiles won't be found)
#   ./build_nuitka.sh clean      — remove all build artifacts
#   ./build_nuitka.sh test       — build onefile then run to verify
#   ./build_nuitka.sh clean test — clean, build, test

set -euo pipefail
cd "$(dirname "$0")"

MODE="standalone"
DO_CLEAN=""
DO_TEST=""

for arg in "$@"; do
    case "$arg" in
        clean) DO_CLEAN=1 ;;
        standalone) MODE="standalone" ;;
        onefile) MODE="onefile" ;;
        test) DO_TEST=1 ;;
    esac
done

if [ -n "$DO_CLEAN" ]; then
    echo "Cleaning build artifacts..."
    rm -rf montecarlo-cli.dist main.dist main.build main.onefile-build __pycache__
    rm -f dist/main.exe dist/montecarlo-cli.exe
    rm -rf dist/main.dist dist/montecarlo-cli.dist dist/main.build dist/main.onefile-build
    echo "Done."
    [ -z "$DO_TEST" ] && exit 0
fi

echo "=== Building montecarlo-cli via Nuitka ==="
echo "Mode:       $MODE"
echo "Entry:      main.py"
echo "Python:     $(python -m nuitka --version 2>&1 | head -1)"
echo ""
echo "Starting build (this will take a few minutes)..."
echo ""

FLAGS="--assume-yes-for-downloads"
FLAGS="$FLAGS --output-filename=montecarlo-cli"
FLAGS="$FLAGS --output-dir=dist"

# Parallel compilation
NUM_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
FLAGS="$FLAGS --jobs=$NUM_CPUS"

# Include Rich's data files (themes, etc.)
FLAGS="$FLAGS --include-package-data=rich"

# Include Rich submodules that Rich lazy-loads at runtime
RICH_MODS="panel table progress prompt text layout box markup style syntax highlighter ansi live live_render measure emoji json tree columns"
for mod in $RICH_MODS; do
    FLAGS="$FLAGS --include-module=rich.$mod"
done

# UPX compression (optional — requires upx to be installed)
if command -v upx &>/dev/null; then
    FLAGS="$FLAGS --enable-plugin=upx"
else
    echo "[INFO] upx not found — skipping UPX compression"
fi

# Windows icon (Windows only)
if [ "$(uname -s 2>/dev/null)" = "MINGW" ] || [ "${OS:-}" = "Windows_NT" ]; then
    if [ -f "assets/icon.ico" ]; then
        FLAGS="$FLAGS --windows-icon-from-ico=assets/icon.ico"
    fi
fi

if [ "$MODE" = "onefile" ]; then
    FLAGS="$FLAGS --onefile"
else
    FLAGS="$FLAGS --standalone"
fi

echo "Nuitka flags: $FLAGS"
echo ""

python -m nuitka main.py $FLAGS

echo ""
if [ "$MODE" = "onefile" ]; then
    if [ -f "dist/montecarlo-cli" ]; then
        echo "=== Build complete: dist/montecarlo-cli ($MODE) ==="
    elif [ -f "dist/montecarlo-cli.exe" ]; then
        echo "=== Build complete: dist/montecarlo-cli.exe ($MODE) ==="
    else
        echo "=== Build complete (check dist/ for output) ==="
    fi
else
    if [ -d "dist/montecarlo-cli.dist" ]; then
        echo "=== Build complete: dist/montecarlo-cli.dist/ ($MODE) ==="
        echo "Run with: dist/montecarlo-cli.dist/montecarlo-cli"
    else
        echo "=== Build complete (check dist/ for output) ==="
    fi
fi

if [ -n "$DO_TEST" ]; then
    echo ""
    echo "=== Testing executable ==="
    if [ "$MODE" = "onefile" ]; then
        echo "First run is slower (onefile extracts itself)..."
        EXE="dist/montecarlo-cli"
        [ -f "dist/montecarlo-cli.exe" ] && EXE="dist/montecarlo-cli.exe"
        echo "4" | "$EXE"
    else
        echo "4" | "dist/montecarlo-cli.dist/montecarlo-cli"
    fi
    echo ""
    echo "=== Test complete ==="
fi

echo ""
echo "Done."
