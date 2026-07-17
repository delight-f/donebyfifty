"""Test configuration: add src/donebyfifty to sys.path so flat imports resolve.

The project uses a ``src/`` layout but imports are flat (``from models import``
rather than ``from donebyfifty.models import``). This conftest ensures pytest
can discover the modules regardless of the working directory.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "donebyfifty"
if SRC not in sys.path:
    sys.path.insert(0, str(SRC))
