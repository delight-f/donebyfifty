"""M14 — collect the reference regression harness under pytest.

``regression.py`` is a runnable script (``python -m tests.regression``) rather
than a collected test module, so a regression in it — or in the engine paths it
exercises — never failed the suite. This wrapper runs the same reference
household through the deterministic and Monte Carlo paths and checks the
monotonicity/range invariants the script asserts.
"""

from __future__ import annotations

from regression import build_reference_household, run_deterministic, run_monte_carlo_report


def test_reference_household_deterministic() -> None:
    """The reference household produces positive bridge and super."""
    det = run_deterministic()
    assert det["bridge"] > 0
    assert det["total_super"] > 0


def test_reference_household_monte_carlo() -> None:
    """Monte Carlo summary percentiles are ordered and in range."""
    mc = run_monte_carlo_report(200)
    assert 0.0 < mc["p_success"] <= 1.0
    assert mc["bridge_p5"] <= mc["bridge_median"] <= mc["bridge_p95"]
    assert mc["bridge_min"] <= mc["bridge_p5"]


def test_reference_household_builder() -> None:
    """The builder is deterministic and exposes the expected shape."""
    h = build_reference_household()
    assert len(h.earners) == 2
    assert len(h.mortgages) == 1
    assert h.base_living_expenses > 0
