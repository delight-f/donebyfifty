"""Analyses built on the Monte Carlo driver.

Sequencing risk, scenario comparison and retirement-age search. Each reads its
household from ``inputs.household`` and fixes one seed for all of its passes, so
the passes share common random numbers (identical draws) and comparisons between
them are paired.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from .engine import run_trials
from .models import Household, SimulationInputs
from .montecarlo import (
    _percentile,
    generate_draws,
    run_monte_carlo,
    simulate_trials,
    summarise_trials,
)
from .state import DrawsBatch, TrialBatchResult, horizon_years

_NEVER_RETIRE = 999
"""Retirement-age sentinel meaning "never retires"."""


def _resolve_seed(seed: int | None) -> int:
    """Fix one seed for a whole analysis so every pass uses the same draws."""
    if seed is not None:
        return seed
    return int(np.random.SeedSequence().generate_state(1)[0])


def _success_indicators(result: TrialBatchResult) -> NDArray[np.float64]:
    """Per-trial success as 0/1 floats, aligned with the trial order."""
    return result.succeeded.astype(np.float64)


# =============================================================================
# SEQUENCING RISK
# =============================================================================


@dataclass
class SequencingRiskResult:
    """Success and real P5 under worst-first, best-first and the original order."""

    worst_first_p_success: float
    worst_first_p5: float
    best_first_p_success: float
    best_first_p5: float
    original_p_success: float
    original_p5: float

    @property
    def spread(self) -> float:
        """Success-probability gap between best-first and worst-first ordering."""
        return self.best_first_p_success - self.worst_first_p_success


def _drawdown_start_index(inputs: SimulationInputs, n_years: int) -> int:
    """First simulation year whose returns may be reordered (M12).

    Drawdown starts when the first working earner stops work, so the index is the
    earliest ``retirement_age - resolved_start_age`` over employed earners, clamped
    to the horizon. With no working earner, drawdown is already running and every
    year is eligible for reordering.
    """
    employed = [e for e in inputs.household.earners if e.employment_type != "not_employed"]
    if not employed:
        return 0
    start = min(
        e.retirement_age - e.resolved_start_age(inputs.simulation_start_age) for e in employed
    )
    return min(max(start, 0), n_years)


def _sequenced_draws(draws: DrawsBatch, drawdown_start: int, direction: str) -> DrawsBatch:
    """Reorder each trial's drawdown-period years by equity shock; working years keep their order.

    Every per-year shock moves together, so within-year pairing is preserved.
    ``eq_z`` is monotone in the equity real return, so sorting on it orders the
    years by return. Sorting runs per trial: row ``i`` of the order permutes trial ``i``.
    """
    key = draws.eq_z[:, drawdown_start:]
    if direction == "best_first":
        key = -key
    drawdown_order = drawdown_start + np.argsort(key, axis=1, kind="stable")
    working_order = np.broadcast_to(np.arange(drawdown_start), (draws.n_trials, drawdown_start))
    order = np.concatenate([working_order, drawdown_order], axis=1)
    return draws.reordered(order)


def run_sequencing_analysis(
    inputs: SimulationInputs, seed: int | None = None
) -> SequencingRiskResult:
    """Compare success under worst-first, best-first and the original return order.

    All three orderings run on the same base draws (common random numbers), so the
    differences reflect ordering alone.

    Args:
        inputs: Simulation inputs; the household is ``inputs.household``.
        seed: Random seed. ``None`` draws a fresh one.
    """
    household = inputs.household
    n_years = horizon_years(household, inputs)
    n_trials = inputs.n_iterations
    seed = _resolve_seed(seed)
    drawdown_start = _drawdown_start_index(inputs, n_years)
    base_draws = generate_draws(household, inputs, n_trials, n_years, seed)

    def _run(direction: str | None) -> tuple[float, float]:
        draws = base_draws
        if direction is not None:
            draws = _sequenced_draws(draws, drawdown_start, direction)
        result = run_trials(household, inputs, draws)
        bridges = result.bridge / result.horizon_deflator
        return float(np.mean(result.succeeded)), _percentile(bridges, 5.0)

    original_p, original_p5 = _run(None)
    worst_p, worst_p5 = _run("worst_first")
    best_p, best_p5 = _run("best_first")
    return SequencingRiskResult(
        worst_first_p_success=worst_p,
        worst_first_p5=worst_p5,
        best_first_p_success=best_p,
        best_first_p5=best_p5,
        original_p_success=original_p,
        original_p5=original_p5,
    )


# =============================================================================
# SCENARIO COMPARISON
# =============================================================================


@dataclass
class ScenarioComparisonResult:
    """One scenario's results, paired against the base case on the same draws."""

    label: str
    p_success: float
    bridge_p5: float
    bridge_median: float
    delta_p: float
    """Scenario success probability minus the base case's."""
    delta_se: float
    """Standard error of ``delta_p`` from per-trial success differences (paired)."""


def _scaled_expenses(household: Household, factor: float) -> Household:
    """Scale living costs, the retirement target and child education costs by ``factor``."""
    children = tuple(
        replace(
            child,
            education_schedule=tuple(
                (age, cost * factor) for age, cost in child.education_schedule
            ),
        )
        for child in household.children
    )
    return replace(
        household,
        base_living_expenses=household.base_living_expenses * factor,
        retirement_target=household.retirement_target * factor,
        children=children,
    )


def _scenario_households(household: Household, simulation_start_age: int) -> dict[str, Household]:
    """Named scenario households, built from the household's actual configuration."""
    scenarios: dict[str, Household] = {}
    if any(e.pt_days_per_week > 0 for e in household.earners):
        new_earners = tuple(replace(e, pt_days_per_week=0.0) for e in household.earners)
        scenarios["No PT income"] = replace(household, earners=new_earners)
    for i, earner in enumerate(household.earners):
        age = earner.resolved_start_age(simulation_start_age)
        label = f"{earner.label} stops working now (age {age})"
        if label in scenarios:
            continue  # deduplicate labels
        # Retirement is set to the earner's current age, so the stop is effective now. Part-time
        # work is cleared too, since its default window starts at retirement age.
        new_e = replace(
            earner,
            retirement_age=age,
            salary=0.0,
            sg_rate=0.0,
            employment_type="not_employed",
            self_employed_income=0.0,
            pt_days_per_week=0.0,
        )
        earners_list = list(household.earners)
        earners_list[i] = new_e
        scenarios[label] = replace(household, earners=tuple(earners_list))
    scenarios["Expenses 10% higher"] = _scaled_expenses(household, 1.1)
    scenarios["Expenses 10% lower"] = _scaled_expenses(household, 0.9)
    return scenarios


def run_scenario_comparison(
    inputs: SimulationInputs,
    seed: int | None = None,
    n_trials: int = 10_000,
) -> dict[str, ScenarioComparisonResult]:
    """Run dynamic scenario comparisons against the base case.

    Scenarios are built from the household's actual configuration:
    - "No PT income": only when at least one earner has PT work
    - "{earner} stops working now (age N)": one per earner, retires at the current age,
      sets them to not_employed and clears their part-time work
    - "Expenses 10% higher" / "Expenses 10% lower": scales living costs, the
      retirement target and child education costs

    The base case and every scenario run on the same draws, so each scenario's
    ``delta_p`` and ``delta_se`` are paired comparisons.

    Args:
        inputs: Base simulation inputs; the household is ``inputs.household``.
        seed: Random seed. ``None`` draws a fresh one shared by all passes.
        n_trials: Trials per run (default 10,000).

    Returns:
        Dict mapping scenario name to ScenarioComparisonResult.
    """
    seed = _resolve_seed(seed)
    base_inputs = replace(inputs, n_iterations=n_trials)
    base_trials = simulate_trials(base_inputs, seed=seed)
    base_ok = _success_indicators(base_trials)
    base_p = float(np.mean(base_ok))

    results: dict[str, ScenarioComparisonResult] = {}
    scenarios = _scenario_households(inputs.household, inputs.simulation_start_age)
    for label, household in scenarios.items():
        scen_inputs = replace(inputs, household=household, n_iterations=n_trials)
        trials = simulate_trials(scen_inputs, seed=seed)
        summary = summarise_trials(scen_inputs, trials, compute_se=False)
        diff = _success_indicators(trials) - base_ok
        results[label] = ScenarioComparisonResult(
            label=label,
            p_success=summary.p_success,
            bridge_p5=summary.bridge_p5,
            bridge_median=summary.bridge_median,
            delta_p=summary.p_success - base_p,
            delta_se=float(np.std(diff, ddof=1) / math.sqrt(diff.size)),
        )
    return results


# =============================================================================
# RETIREMENT AGE SEARCH
# =============================================================================


@dataclass
class RetirementSearchResult:
    """Outcome of a retirement-age search.

    Pass means ``p_success >= threshold``. ``earliest_age`` is the lowest scanned
    age in the contiguous passing run that starts at the top of the scan.
    """

    entered_age: int
    entered_p_success: float
    earliest_age: int
    earliest_p_success: float
    floor_age: int
    threshold: float
    mode: str
    target_earner_label: str | None
    entered_ages_by_earner: dict[str, int]
    entered_plan_feasible: bool
    feasible: bool
    """True when the top of the scan passes, so a passing run exists."""
    earliest_feasible_age: int | None
    """Lowest passing age in the top-down run; ``None`` when ``feasible`` is False."""
    non_monotonic: bool
    non_monotonic_ages: list[int]
    """``[lower, higher]``: a passing age below a failing age (first found), or empty."""
    trials: int
    """Trials per candidate in the search (the search's own count, not the main run's)."""
    evaluated_ages: Mapping[int, float] = field(default_factory=dict)
    """Scanned age -> success probability."""
    tolerance: float = 0.0
    """Pass/fail noise allowance; always 0.0 because the threshold is applied exactly."""


def _is_working(earner_employment: str, retirement_age: int) -> bool:
    """True for an earner whose retirement age the search may move."""
    return earner_employment != "not_employed" and retirement_age < _NEVER_RETIRE


def run_retirement_search(
    inputs: SimulationInputs,
    seed: int | None = None,
    n_trials: int = 10_000,
    min_search_age: int = 40,
    success_threshold: float | None = None,
    mode: str = "both_together",
    target_earner_index: int = 0,
) -> RetirementSearchResult:
    """Find the earliest retirement age that meets the success threshold.

    Ages are scanned from the top of the range (the entered plan, capped at the
    horizon) down to the search floor. Every candidate runs on the same seed
    (common random numbers), so the pass/fail boundary is not blurred by noise
    between candidates.

    Args:
        inputs: Simulation inputs; the household is ``inputs.household``.
        seed: Random seed. ``None`` draws a fresh one shared by all candidates.
        n_trials: Trials per candidate (default 10,000).
        min_search_age: Lowest age considered (the floor is at least sim start + 5).
        success_threshold: Pass threshold; ``None`` uses ``inputs.success_threshold``.
        mode: ``"both_together"`` moves every working earner's retirement age to the
            candidate. ``"per_earner"`` moves only ``target_earner_index``.
        target_earner_index: Earner searched in ``"per_earner"`` mode.

    Raises:
        ValueError: If ``mode == "per_earner"`` and the target earner is not employed.
    """
    household = inputs.household
    if success_threshold is None:
        success_threshold = inputs.success_threshold
    seed = _resolve_seed(seed)
    search_floor = max(min_search_age, inputs.simulation_start_age + 5)

    if mode == "per_earner":
        target = household.earners[target_earner_index]
        if target.employment_type == "not_employed":
            raise ValueError(
                f"Earner {target.label!r} is not employed, so there is no retirement age to search."
            )
        entered_age = target.retirement_age
        target_label: str | None = target.label
    else:
        working_ages = [
            e.retirement_age
            for e in household.earners
            if _is_working(e.employment_type, e.retirement_age)
        ]
        entered_age = max(working_ages) if working_ages else inputs.simulation_start_age
        target_label = None

    def _candidate(test_age: int) -> Household:
        if mode == "per_earner":
            earners = tuple(
                replace(e, retirement_age=test_age) if i == target_earner_index else e
                for i, e in enumerate(household.earners)
            )
        else:
            earners = tuple(
                (
                    replace(e, retirement_age=test_age)
                    if _is_working(e.employment_type, e.retirement_age)
                    else e
                )
                for e in household.earners
            )
        return replace(household, earners=earners)

    def _evaluate(hh: Household) -> float:
        hh_inputs = replace(inputs, household=hh, n_iterations=n_trials)
        return run_monte_carlo(hh_inputs, seed=seed, compute_se=False).p_success

    entered_p = _evaluate(household)
    top = min(entered_age, inputs.simulation_start_age + horizon_years(household, inputs))
    descending = list(range(top, search_floor - 1, -1))
    evaluated = {age: _evaluate(_candidate(age)) for age in descending}
    passes = {age: p >= success_threshold for age, p in evaluated.items()}

    # Contiguous passing run from the top of the scan.
    earliest: int | None = None
    for age in descending:
        if not passes[age]:
            break
        earliest = age

    # First inversion: a passing age below a failing one.
    ascending = sorted(evaluated)
    non_monotonic_ages: list[int] = []
    for i, lower in enumerate(ascending):
        if not passes[lower]:
            continue
        failing_above = [h for h in ascending[i + 1 :] if not passes[h]]
        if failing_above:
            non_monotonic_ages = [lower, failing_above[0]]
            break

    return RetirementSearchResult(
        entered_age=entered_age,
        entered_p_success=entered_p,
        earliest_age=earliest if earliest is not None else search_floor - 1,
        earliest_p_success=evaluated[earliest] if earliest is not None else entered_p,
        floor_age=search_floor,
        threshold=success_threshold,
        mode=mode,
        target_earner_label=target_label,
        entered_ages_by_earner={e.label: e.retirement_age for e in household.earners},
        entered_plan_feasible=entered_p >= success_threshold,
        feasible=earliest is not None,
        earliest_feasible_age=earliest,
        non_monotonic=bool(non_monotonic_ages),
        non_monotonic_ages=non_monotonic_ages,
        evaluated_ages=evaluated,
        trials=n_trials,
        tolerance=0.0,
    )
