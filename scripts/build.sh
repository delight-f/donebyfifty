#!/usr/bin/env bash
# Build script for Monte Carlo CLI (macOS / Linux)
# Builds a single-file executable via PyInstaller
#
# Usage:
#   ./build.sh          — build with defaults
#   ./build.sh clean    — remove build/dist artifacts first
#   ./build.sh test     — build then run the exe to verify

set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "clean" ]; then
    echo "Cleaning build artifacts..."
    rm -rf build dist *.spec
    echo "Done."
    [ $# -gt 1 ] || exit 0
fi

echo "Building montecarlo-cli..."
echo "UPX compression: enabled"
echo "Icon: assets/icon.ico"
echo

python -m PyInstaller build.spec --clean --noconfirm

echo
echo "Build complete: dist/montecarlo-cli"

if [ "${1:-}" = "test" ]; then
    echo
    echo "Testing executable..."
    echo "4" | ./dist/montecarlo-cli
fi
