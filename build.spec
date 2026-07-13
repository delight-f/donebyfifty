# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller build spec for Monte Carlo CLI
#
# Build:
#   pyinstaller build.spec
#
# Build flags used:
#   --onefile         Single executable (not a directory)
#   --console         Keep as console app (not --windowed)
#   --upx             Apply UPX compression when available
#
# Rich has an official hook in pyinstaller-hooks-contrib so no explicit
# hiddenimports are needed for standard Rich usage. Test on a clean
# machine if upgrading Rich.
#
# PIL (Pillow) is excluded via EXCLUDES. Pygments may pull in PIL
# through optional Image/ImageFilter format plugins, but those are
# never loaded at runtime by this app.

import sys
from pathlib import Path

# ── Project root (resolved from working directory, since spec files are
#    exec'd without ``__file__`` by PyInstaller's build machinery) ─────
ROOT = Path().resolve()

# ── Block modules not needed by this app ───────────────────────────────
EXCLUDES = [
    "PIL",           # Pillow — pulled by Pygments optional deps, never used here
    "PIL.Image",     #   explicitly block submodules to stop hooks
    "PIL.ImageFilter",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "PIL.ImageOps",
    "tkinter",       # GUI toolkit — console app only
    "matplotlib",    # No plotting in the app
    "scipy",         # No scientific computing
    "pandas",        # No dataframe work
    "curses",        # Rich replaces curses
    "curses.ascii",
    "curses.panel",
    "curses.textpad",
]

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Rich lazy-loads some internal modules via __getattr__
        "rich.panel",
        "rich.table",
        "rich.progress",
        "rich.progress_bar",
        "rich.prompt",
        "rich.text",
        "rich.layout",
        "rich.box",
        "rich.markup",
        "rich.style",
        "rich.syntax",
        "rich.highlighter",
        "rich.ansi",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="montecarlo-cli",
    icon=str(ROOT / "assets" / "icon.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
