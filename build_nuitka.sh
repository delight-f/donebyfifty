#!/usr/bin/env bash
# Build script for Monte Carlo CLI via Nuitka (macOS / Linux)
# Creates a single-file executable (or standalone directory)
#
# Usage:
#   ./build_nuitka.sh            — build onefile exe (default)
#   ./build_nuitka.sh standalone — build standalone directory (faster, easier debug)
#   ./build_nuitka.sh clean      — remove all build artifacts
#   ./build_nuitka.sh test       — build onefile then run to verify
#   ./build_nuitka.sh clean test — clean, build, test

set -euo pipefail
cd "$(dirname "$0")"

MODE="onefile"
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

FLAGS="--assume-yes-for-downloads --show-progress"
FLAGS="$FLAGS --output-filename=montecarlo-cli"
FLAGS="$FLAGS --output-dir=dist"
FLAGS="$FLAGS --enable-plugin=upx"

# Include Rich submodules that Rich lazy-loads at runtime
FLAGS="$FLAGS --include-module=rich.panel"
FLAGS="$FLAGS --include-module=rich.table"
FLAGS="$FLAGS --include-module=rich.progress"
FLAGS="$FLAGS --include-module=rich.progress_bar"
FLAGS="$FLAGS --include-module=rich.prompt"
FLAGS="$FLAGS --include-module=rich.text"
FLAGS="$FLAGS --include-module=rich.layout"
FLAGS="$FLAGS --include-module=rich.box"
FLAGS="$FLAGS --include-module=rich.markup"
FLAGS="$FLAGS --include-module=rich.style"
FLAGS="$FLAGS --include-module=rich.syntax"
FLAGS="$FLAGS --include-module=rich.highlighter"
FLAGS="$FLAGS --include-module=rich.ansi"
FLAGS="$FLAGS --include-module=rich.live"
FLAGS="$FLAGS --include-module=rich.live_render"
FLAGS="$FLAGS --include-module=rich.measure"
FLAGS="$FLAGS --include-module=rich.emoji"
FLAGS="$FLAGS --include-module=rich.json"
FLAGS="$FLAGS --include-module=rich.tree"
FLAGS="$FLAGS --include-module=rich.columns"

# On macOS, include the icon if available
if [ -f "assets/icon.ico" ]; then
    FLAGS="$FLAGS --windows-icon-from-ico=assets/icon.ico"
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
