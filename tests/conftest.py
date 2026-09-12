"""Test configuration: add src/donebyfifty to sys.path so flat imports resolve.

The project uses a ``src/`` layout but imports are flat (``from models import``
rather than ``from donebyfifty.models import``). This conftest ensures pytest
can discover the modules regardless of the working directory.
"""

import random
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "donebyfifty"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _seed_module_rng() -> None:
    """M14: seed the module RNG before every test.

    Monte Carlo entry points fall back to the module-level ``random`` when no
    seed argument is given, so this makes unseeded tests reproducible instead
    of drawing OS entropy.
    """
    random.seed(12345)
