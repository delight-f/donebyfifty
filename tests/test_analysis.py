"""Tests for the analysis layer: drawdown ordering, paired scenarios, and the age search.

The search tests replace the Monte Carlo driver with a closed-form success curve so the
pass/fail pattern is known exactly. The pairing and engine tests run the real engine at
small trial counts with a fixed seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from conftest import reference_household

from donebyfifty import analysis
from donebyfifty.analysis import (
    _drawdown_start_index,
    run_retirement_search,
    run_scenario_comparison,
)
from donebyfifty.models import Earner, Household, SimulationInputs
from donebyfifty.montecarlo import run_monte_carlo, simulate_trials


def _earner(label: str, retirement_age: int) -> Earner:
    return Earner(
        label=label,
        salary=200_000.0,
        super_balance=300_000.0,
        salary_growth_rate=0.03,
        retirement_age=retirement_age,
        super_access_age=60,
        sg_rate=0.12,
    )


def _household(*earners: Earner, expenses: float = 80_000.0) -> Household:
    return Household(
        earners=earners,
        base_living_expenses=expenses,
        retirement_target=60_000.0,
    )


def _fake_monte_carlo(
    p_at_age: Callable[[int], float],
    calls: list[tuple[int, ...]] | None = None,
) -> Callable[..., Any]:
    """Stand-in for run_monte_carlo whose success depends on the last earner's age.

    Every call's earner retirement ages are appended to ``calls`` when provided.
    """

    def fake(
        inputs: SimulationInputs, *, seed: int | None = None, compute_se: bool = True
    ) -> SimpleNamespace:
        ages = tuple(e.retirement_age for e in inputs.household.earners)
        if calls is not None:
            calls.append(ages)
        return SimpleNamespace(p_success=p_at_age(ages[-1]))

    return fake


# =============================================================================
# RETIREMENT AGE SEARCH
# =============================================================================


class TestRetirementSearch:
    """run_retirement_search over a closed-form pass/fail curve."""

    def test_monotone_curve_returns_earliest_passing_age(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ages 48..55 pass (p=0.99); ages 45..47 fail (p=0.5).
        # The scan runs 55 -> 45 from the entered age; the contiguous passing run from the
        # top ends at 48, so the earliest passing age is 48.
        monkeypatch.setattr(
            analysis,
            "run_monte_carlo",
            _fake_monte_carlo(lambda age: 0.99 if age >= 48 else 0.5),
        )
        inputs = SimulationInputs(household=_household(_earner("Earner 1", 55)), n_iterations=10)

        result = run_retirement_search(
            inputs, seed=1, n_trials=10, min_search_age=45, success_threshold=0.95
        )

        assert result.entered_age == 55
        assert result.floor_age == 45
        assert result.feasible
        assert result.entered_plan_feasible
        assert result.earliest_feasible_age == 48
        assert result.earliest_age == 48
        assert result.earliest_p_success == 0.99
        assert not result.non_monotonic
        assert result.non_monotonic_ages == []
        assert result.tolerance == 0.0
        assert dict(result.evaluated_ages) == {
            age: (0.99 if age >= 48 else 0.5) for age in range(45, 56)
        }

    def test_non_monotone_curve_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Age 52 fails while 48..51 and 53..55 pass; ages below 48 fail.
        # Top-down run: 55, 54, 53 pass, then 52 fails, so earliest is 53.
        # The lowest passing age below a failing one is 48, with failing age 52 above it.
        monkeypatch.setattr(
            analysis,
            "run_monte_carlo",
            _fake_monte_carlo(lambda age: 0.5 if age == 52 or age < 48 else 0.99),
        )
        inputs = SimulationInputs(household=_household(_earner("Earner 1", 55)), n_iterations=10)

        result = run_retirement_search(
            inputs, seed=1, n_trials=10, min_search_age=45, success_threshold=0.95
        )

        assert result.earliest_age == 53
        assert result.earliest_feasible_age == 53
        assert result.non_monotonic
        assert result.non_monotonic_ages == [48, 52]

    def test_infeasible_when_no_age_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(analysis, "run_monte_carlo", _fake_monte_carlo(lambda age: 0.5))
        inputs = SimulationInputs(household=_household(_earner("Earner 1", 55)), n_iterations=10)

        result = run_retirement_search(
            inputs, seed=1, n_trials=10, min_search_age=45, success_threshold=0.95
        )

        assert not result.feasible
        assert not result.entered_plan_feasible
        assert result.earliest_feasible_age is None
        # Sentinel: one below the floor, never the entered age.
        assert result.earliest_age == 44
        assert result.earliest_p_success == 0.5

    def test_per_earner_searches_only_target_and_holds_others_fixed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[int, ...]] = []
        monkeypatch.setattr(
            analysis,
            "run_monte_carlo",
            _fake_monte_carlo(lambda age: 0.99 if age >= 48 else 0.5, calls),
        )
        inputs = SimulationInputs(
            household=_household(_earner("Earner 1", 55), _earner("Earner 2", 55)),
            n_iterations=10,
        )

        result = run_retirement_search(
            inputs,
            seed=1,
            n_trials=10,
            min_search_age=45,
            success_threshold=0.95,
            mode="per_earner",
            target_earner_index=1,
        )

        assert result.target_earner_label == "Earner 2"
        assert result.earliest_age == 48
        # Earner 1 is never moved away from its entered age of 55.
        assert all(ages[0] == 55 for ages in calls)

    def test_per_earner_rejects_unemployed_target(self) -> None:
        not_working = replace(_earner("Earner 1", 55), employment_type="not_employed")
        inputs = SimulationInputs(
            household=_household(not_working, _earner("Earner 2", 55)), n_iterations=10
        )

        with pytest.raises(ValueError, match="not employed"):
            run_retirement_search(
                inputs, seed=1, n_trials=10, mode="per_earner", target_earner_index=0
            )


# =============================================================================
# DRAWDOWN START (M12)
# =============================================================================


class TestDrawdownStart:
    """Reordering applies only to drawdown-period years."""

    def test_first_working_earner_stop_year(self) -> None:
        # Earners stop at 50 and 58; simulation starts at 37.
        # Drawdown starts at min(50 - 37, 58 - 37) = 13 years in.
        inputs = SimulationInputs(
            household=_household(_earner("Earner 1", 50), _earner("Earner 2", 58)),
            simulation_start_age=37,
        )
        assert _drawdown_start_index(inputs, n_years=60) == 13

    def test_no_working_earner_reorders_from_first_year(self) -> None:
        not_working = replace(_earner("Earner 1", 50), employment_type="not_employed")
        inputs = SimulationInputs(household=_household(not_working), simulation_start_age=37)
        assert _drawdown_start_index(inputs, n_years=60) == 0

    def test_clamped_to_horizon(self) -> None:
        # Stopping at 90 is 53 years in, but only 10 simulated years exist.
        inputs = SimulationInputs(
            household=_household(_earner("Earner 1", 90)), simulation_start_age=37
        )
        assert _drawdown_start_index(inputs, n_years=10) == 10


# =============================================================================
# PAIRED SCENARIOS
# =============================================================================


class TestScenarioPairing:
    """Scenario deltas are paired against the base case on the same draws.

    Uses the reference household retiring at 45: its success rate sits strictly between
    0 and 1, so the standard errors are non-degenerate.
    """

    def test_delta_p_is_scenario_minus_base(self) -> None:
        h = reference_household(retire_age=45)
        inputs = SimulationInputs(household=h, sell_order=("AU ETFs",))
        n = 200

        scens = run_scenario_comparison(inputs, seed=7, n_trials=n)
        base = run_monte_carlo(replace(inputs, n_iterations=n), seed=7, compute_se=False)

        higher = scens["Expenses 10% higher"]
        assert higher.delta_p == pytest.approx(higher.p_success - base.p_success, abs=1e-12)

    def test_delta_se_matches_paired_standard_error(self) -> None:
        h = reference_household(retire_age=45)
        n = 200
        inputs = SimulationInputs(household=h, sell_order=("AU ETFs",))

        scens = run_scenario_comparison(inputs, seed=7, n_trials=n)

        # Independent recompute: per-trial success differences on the same seed.
        # Each diff is in {-1, 0, +1}; delta_se = sd(diff, ddof=1) / sqrt(n).
        earner1_stops = replace(
            h.earners[0],
            employment_type="not_employed",
            salary=0.0,
            sg_rate=0.0,
            self_employed_income=0.0,
            retirement_age=inputs.simulation_start_age,
            pt_days_per_week=0.0,
        )
        stop_h = replace(h, earners=(earner1_stops, h.earners[1]))
        base_ok = simulate_trials(replace(inputs, n_iterations=n), seed=7).succeeded.astype(
            np.float64
        )
        scen_ok = simulate_trials(
            replace(inputs, household=stop_h, n_iterations=n), seed=7
        ).succeeded.astype(np.float64)
        expected = float(np.std(scen_ok - base_ok, ddof=1) / math.sqrt(n))

        label = f"Earner 1 stops working now (age {inputs.simulation_start_age})"
        assert scens[label].delta_se == pytest.approx(expected, rel=1e-12)

    def test_paired_se_is_below_unpaired_se(self) -> None:
        # The unpaired SE treats the two success estimates as independent samples.
        # Common draws make the per-trial differences small, so the paired SE is smaller.
        h = reference_household(retire_age=45)
        n = 200
        inputs = SimulationInputs(household=h, sell_order=("AU ETFs",))

        scens = run_scenario_comparison(inputs, seed=7, n_trials=n)
        base = run_monte_carlo(replace(inputs, n_iterations=n), seed=7, compute_se=False)
        lower = scens["Expenses 10% lower"]
        p_base, p_scen = base.p_success, lower.p_success
        unpaired = math.sqrt(p_base * (1 - p_base) / n + p_scen * (1 - p_scen) / n)

        assert lower.delta_se < unpaired

    def test_stops_working_scenario_retires_earner_now(self) -> None:
        """Pre-fix: the 'stops working' scenario zeroed income but kept the earner's original
        retirement age, and the label named no age. Part-time work could still run because its
        default window starts at retirement age. The scenario now retires the earner at the
        current age, clears part-time work, and labels that age.
        """
        h = reference_household(retire_age=45)
        earners = (replace(h.earners[0], pt_days_per_week=2.0), h.earners[1])
        h = replace(h, earners=earners)
        start = SimulationInputs().simulation_start_age

        stop = analysis._scenario_households(h, start)[f"Earner 1 stops working now (age {start})"]

        assert stop.earners[0].retirement_age == start
        assert stop.earners[0].pt_days_per_week == 0.0
        assert stop.earners[0].employment_type == "not_employed"
        assert stop.earners[1] == h.earners[1]


# =============================================================================
# MONTE CARLO SUMMARY BEHAVIOUR
# =============================================================================


class TestSummaryBehaviour:
    """Standard-error switches, near-miss default, validation, and warnings."""

    def test_compute_se_false_zeroes_standard_errors(self) -> None:
        inputs = SimulationInputs(household=_household(_earner("Earner 1", 50)), n_iterations=200)

        full = run_monte_carlo(inputs, seed=42)
        fast = run_monte_carlo(inputs, seed=42, compute_se=False)

        assert fast.p_success == full.p_success
        assert fast.bridge_median == full.bridge_median
        assert fast.bridge_mean_se == 0.0
        assert fast.bridge_median_se == 0.0
        assert fast.bridge_median_ci_lo == 0.0
        assert fast.bridge_median_ci_hi == 0.0
        assert fast.super_median_se == 0.0

    def test_near_miss_threshold_defaults_to_one_year_of_target(self) -> None:
        h = _household(_earner("Earner 1", 50))
        result = run_monte_carlo(
            SimulationInputs(household=h, n_iterations=100), seed=42, compute_se=False
        )

        assert result.near_miss_threshold == h.retirement_target

    def test_single_iteration_rejected(self) -> None:
        with pytest.raises(ValueError):
            SimulationInputs(n_iterations=1)

    def test_terminal_bridge_matches_last_trajectory_year_under_unmet_spending(self) -> None:
        # Expenses far above income force unmet spending, so the terminal bridge and the
        # final-year trajectory must use the same units and net definition (M10).
        h = _household(_earner("Earner 1", 50), expenses=300_000.0)
        result = run_monte_carlo(
            SimulationInputs(household=h, n_iterations=100), seed=42, compute_se=False
        )

        assert result.p_success < 1.0
        assert result.bridge_p5 == result.bridge_by_age_p5[-1]

    def test_no_low_variance_warning_when_median_not_positive(self) -> None:
        # Spending above all assets leaves the terminal bridge at or below zero, so the
        # low-variance guardrail (which needs a positive median and mean) must stay quiet.
        h = _household(_earner("Earner 1", 50), expenses=300_000.0)
        result = run_monte_carlo(
            SimulationInputs(household=h, n_iterations=100), seed=42, compute_se=False
        )

        assert result.bridge_median <= 0
        assert result.warnings == ()
