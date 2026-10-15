"""Monte Carlo driver: runs many trials, percentiles and bootstrap intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import BK_RHO
from .engine import run_trials
from .models import Household, SimulationInputs, SimulationResults, validate_horizon
from .returns import ASSET_CLASS_PARAMS, INFLATION_EQ_CORR, SUPER_EQ_CORR, inflation_rate
from .state import CLASS_NAMES, DrawsBatch, TrialBatchResult, horizon_years


def _correlated(
    driver: NDArray[np.float64], independent: NDArray[np.float64], rho: float | NDArray[np.float64]
) -> NDArray[np.float64]:
    """Standard-normal shock with correlation ``rho`` to ``driver``, elementwise."""
    return np.asarray(rho * driver + np.sqrt(1.0 - rho * rho) * independent, dtype=np.float64)


def generate_draws(
    household: Household,
    inputs: SimulationInputs,
    n_trials: int,
    n_years: int,
    seed: int | None,
) -> DrawsBatch:
    """Pre-generate every shock for ``n_trials`` trials as one ``DrawsBatch``.

    Four independent streams come from ``SeedSequence(seed).spawn(4)``, so the
    equity path does not change when, say, a mortgage is added:

    - stream 0: equity shock and inflation (inflation correlated to equity).
    - stream 1: per-asset-class shocks, correlated to equity by ``corr_with_eq``.
    - stream 2: per-earner super override shocks, correlated to equity at ``SUPER_EQ_CORR``.
    - stream 3: per-mortgage rate shocks, correlated to equity at ``interest_rate_corr``
      (``BK_RHO`` when unset).

    ``seed=None`` draws fresh OS entropy.
    """
    eq_seq, class_seq, super_seq, mortgage_seq = np.random.SeedSequence(seed).spawn(4)

    rng = np.random.default_rng(eq_seq)
    eq_z = rng.standard_normal((n_trials, n_years))
    infl_ind = rng.standard_normal((n_trials, n_years))
    if inputs.stochastic_inflation:
        inflation = inflation_rate(inputs.inflation, _correlated(eq_z, infl_ind, INFLATION_EQ_CORR))
    else:
        inflation = np.full((n_trials, n_years), inputs.inflation)

    # Columns follow ``CLASS_NAMES``; each class draws its own stream in that order.
    rng = np.random.default_rng(class_seq)
    class_z = np.stack(
        [
            _correlated(
                eq_z,
                rng.standard_normal((n_trials, n_years)),
                ASSET_CLASS_PARAMS[name]["corr_with_eq"],
            )
            for name in CLASS_NAMES
        ],
        axis=2,
    )

    rng = np.random.default_rng(super_seq)
    super_ind = rng.standard_normal((n_trials, len(household.earners), n_years))
    super_z = _correlated(eq_z[:, None, :], super_ind, SUPER_EQ_CORR)

    rng = np.random.default_rng(mortgage_seq)
    mortgage_ind = rng.standard_normal((n_trials, len(household.mortgages), n_years))
    mortgage_rho = np.array(
        [
            BK_RHO if m.interest_rate_corr is None else m.interest_rate_corr
            for m in household.mortgages
        ],
        dtype=np.float64,
    )[None, :, None]
    mortgage_z = _correlated(eq_z[:, None, :], mortgage_ind, mortgage_rho)

    return DrawsBatch(
        eq_z=eq_z,
        inflation=inflation,
        class_z=class_z,
        super_override_z=super_z,
        mortgage_z=mortgage_z,
    )


# =============================================================================
# MONTE CARLO RUNNER
# =============================================================================


# SeedSequence spawn key for the bootstrap stream. ``generate_draws`` spawns keys 0-3.
_BOOTSTRAP_SPAWN_KEY = 4
# Element budget per bootstrap chunk: bounds the resample matrix to a few MB.
_BOOTSTRAP_BLOCK_ELEMENTS = 1_000_000
_TRAJECTORY_PCTS = (5, 10, 25, 50, 75, 90, 95)


def _percentile_index(n: int, pct: float) -> int:
    """Return the index for ``pct`` in a sorted sequence of length ``n``.

    Single percentile definition used everywhere in the engine (M1):
    ``round(n * pct / 100)``, clamped to the last element. Previously the
    bootstrap used a floor while reportable percentiles used ``round``, so the
    standard error described a different order statistic than the value it
    annotated.
    """
    if n <= 0:
        return 0
    return min(int(round(n * pct / 100.0)), n - 1)


def _percentile(values: ArrayLike, pct: float) -> float:
    """Percentile of ``values`` (any order) using the shared definition."""
    arr = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if arr.size == 0:
        return 0.0
    return float(arr[_percentile_index(arr.size, pct)])


def _percentile_over_trials(values: NDArray[np.float64], pct: float) -> NDArray[np.float64]:
    """Percentile down axis 0 (the trials axis) of ``values``, one per remaining index."""
    sorted_values = np.sort(values, axis=0)
    return np.asarray(sorted_values[_percentile_index(values.shape[0], pct)], dtype=np.float64)


def _bootstrap_percentile(
    values: NDArray[np.float64],
    pct: float,
    *,
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap SE and percentile interval for a percentile (M1).

    Resamples ``values`` with replacement ``n_bootstrap`` times, takes the
    requested percentile of each resample with the same definition as the
    reported value, and returns ``(se, ci_lo, ci_hi)``. Resamples are drawn in
    vectorised chunks from ``rng``, so the result is reproducible for a fixed
    stream.
    """
    n = values.size
    k = _percentile_index(n, pct)
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    chunk = max(1, _BOOTSTRAP_BLOCK_ELEMENTS // n)
    for start in range(0, n_bootstrap, chunk):
        rows = min(chunk, n_bootstrap - start)
        resampled = values[rng.integers(0, n, size=(rows, n))]
        estimates[start : start + rows] = np.partition(resampled, k, axis=1)[:, k]
    se = float(np.std(estimates, ddof=1))
    ordered = np.sort(estimates)
    lo = float(ordered[_percentile_index(n_bootstrap, (1.0 - ci) / 2.0 * 100.0)])
    hi = float(ordered[_percentile_index(n_bootstrap, (1.0 + ci) / 2.0 * 100.0)])
    return se, lo, hi


@dataclass(frozen=True)
class MonteCarloResults(SimulationResults):
    """``SimulationResults`` extended with the engine's new statistics.

    ``SimulationResults`` lives in ``models.py``, which is owned by another
    worker this round, so the additional engine outputs are exposed by
    subclassing it here (additive only — every field has a default). Since a
    ``MonteCarloResults`` *is* a ``SimulationResults``, all existing consumers
    keep working; new consumers (ui_results.py) can read the extra fields.

    M1 — bootstrap percentile intervals
        ``*_ci_lo`` / ``*_ci_hi`` accompany the existing ``*_se`` fields. All
        are computed with a single percentile definition and a seeded RNG, so
        they are reproducible for a fixed run seed.
    """

    # ── M1 — bootstrap percentile intervals ──────────────────────────────
    bridge_median_ci_lo: float | None = None
    bridge_median_ci_hi: float | None = None
    bridge_p5_ci_lo: float | None = None
    bridge_p5_ci_hi: float | None = None
    bridge_p95_ci_lo: float | None = None
    bridge_p95_ci_hi: float | None = None
    super_median_ci_lo: float | None = None
    super_median_ci_hi: float | None = None

    def summary_dict(self) -> dict[str, object]:
        """Flat dict for serialisation, including the extended fields."""
        data: dict[str, object] = dict(super().summary_dict())
        data.update(
            {
                "bridge_median_ci_lo": self.bridge_median_ci_lo,
                "bridge_median_ci_hi": self.bridge_median_ci_hi,
                "bridge_p5_ci_lo": self.bridge_p5_ci_lo,
                "bridge_p5_ci_hi": self.bridge_p5_ci_hi,
                "bridge_p95_ci_lo": self.bridge_p95_ci_lo,
                "bridge_p95_ci_hi": self.bridge_p95_ci_hi,
                "super_median_ci_lo": self.super_median_ci_lo,
                "super_median_ci_hi": self.super_median_ci_hi,
            }
        )
        return data


def _check_run_inputs(inputs: SimulationInputs) -> None:
    """Reject inputs the engine cannot simulate."""
    if inputs.sell_strategy != "waterfall":
        raise ValueError(
            f"Unsupported sell_strategy '{inputs.sell_strategy}'. "
            f"Only 'waterfall' is currently implemented."
        )
    validate_horizon(inputs.household, inputs)


def simulate_trials(inputs: SimulationInputs, *, seed: int | None) -> TrialBatchResult:
    """Run every trial of ``inputs`` on the draws for ``seed``; return the batch result.

    Draws come from ``generate_draws``, so two calls with the same seed and the
    same household shape see identical shocks (common random numbers). Paired
    comparisons between scenarios rely on this.

    Raises:
        ValueError: If ``inputs.sell_strategy`` is not ``"waterfall"``, or an
            earner's super access age is not above their start age.
    """
    _check_run_inputs(inputs)
    household = inputs.household
    n_years = horizon_years(household, inputs)
    draws = generate_draws(household, inputs, inputs.n_iterations, n_years, seed)
    return run_trials(household, inputs, draws)


def _across_trials(values: NDArray[np.float64], pct: float) -> list[float]:
    """Percentile across trials (axis 0) of a ``(trials, ...)`` array, as a list."""
    return [float(v) for v in _percentile_over_trials(values, pct)]


def _percentile_bands(values: NDArray[np.float64]) -> dict[str, list[float]]:
    """p50, p5 and p95 across trials of a ``(trials, years)`` array."""
    return {
        "p50": _across_trials(values, 50.0),
        "p5": _across_trials(values, 5.0),
        "p95": _across_trials(values, 95.0),
    }


def summarise_trials(
    inputs: SimulationInputs,
    result: TrialBatchResult,
    *,
    seed: int | None = None,
    near_miss_threshold: float | None = None,
    compute_se: bool = True,
) -> MonteCarloResults:
    """Aggregate a batch result into today's-dollar statistics.

    Each trial's balances are deflated with that trial's own deflators (M3):
    balances at the horizon by ``horizon_deflator``, the floor by
    ``floor_deflator``. Percentiles use ``_percentile_index`` throughout.

    Args:
        inputs: The inputs the trials were run with.
        result: The batch result from ``simulate_trials``.
        seed: Seed recorded on the result and used for the bootstrap stream.
        near_miss_threshold: Real floor (today's dollars) at or below which a trial
            counts as a near miss. ``None`` means one year of
            ``household.retirement_target``.
        compute_se: When False, standard errors and percentile intervals are 0.0
            and no bootstrap runs. Internal callers that only need ``p_success``
            use this.

    Raises:
        ValueError: If ``near_miss_threshold`` is negative.
    """
    household = inputs.household
    if near_miss_threshold is None:
        near_miss_threshold = household.retirement_target
    if near_miss_threshold < 0:
        raise ValueError(f"near_miss_threshold must be >= 0 (got {near_miss_threshold})")

    n = result.n_trials
    horizon_age = result.horizon_age
    failed = ~result.succeeded
    terminal = result.bridge / result.horizon_deflator
    floor = result.min_bridge / result.floor_deflator
    super_total = result.total_super / result.horizon_deflator
    per_earner_super = result.super_balances / result.horizon_deflator[:, None]
    remaining_mortgage = result.mortgage_principals

    bridge_by_age_p = {
        pct: _across_trials(result.bridge_by_age, float(pct)) for pct in _TRAJECTORY_PCTS
    }
    mortgage_by_age_out = {
        mortgage.label: _percentile_bands(result.mortgage_by_age[:, :, mi])
        for mi, mortgage in enumerate(household.mortgages)
    }
    offset_labels = [a.label for a in household.investment_accounts if a.is_offset]
    offset_by_age_out = {
        label: _percentile_bands(result.offset_by_age[:, :, oi])
        for oi, label in enumerate(offset_labels)
    }
    mortgage_rate_by_age_out = {
        mortgage.label: _percentile_bands(result.mortgage_rate_by_age[:, :, mi])
        for mi, mortgage in enumerate(household.mortgages)
    }

    # Per-trial drawdown totals (M4a), today's dollars: true quantiles of each total.
    offset_drawn_totals = result.total_offset_drawn
    non_offset_drawn_totals = result.total_non_offset_drawn
    cgt_paid_totals = result.total_cgt_paid
    cgt_without_floor_totals = result.total_cgt_without_floor
    cgt_floor_extra_totals = cgt_paid_totals - cgt_without_floor_totals
    income_tax_totals = result.total_income_tax

    # A trial succeeds when it never fails (see ``TrialBatchResult.succeeded``).
    p_success = float(np.mean(~failed))

    # Worst real floor across trials; ties resolve to the first trial.
    worst = int(np.argmin(floor))

    # ── Per-mortgage term clearance (checkable only when term ends within horizon)
    per_mortgage_term_cleared_pct: dict[str, float] = {}
    per_mortgage_not_checkable: dict[str, int] = {}
    for mi, mortgage in enumerate(household.mortgages):
        if mortgage.loan_term_end_age is None:
            continue
        if result.term_checkable[mi]:
            per_mortgage_term_cleared_pct[mortgage.label] = float(
                np.mean(result.term_cleared[:, mi])
            )
        else:
            per_mortgage_not_checkable[mortgage.label] = n
    mortgage_term_clearance_rate = min(per_mortgage_term_cleared_pct.values(), default=1.0)

    # ── Standard errors and percentile intervals (M1) ────────────────────
    # Bootstrap draws from its own spawn key so it never shares a stream with the
    # shocks in ``generate_draws``.
    bridge_mean_se = 0.0
    bridge_median_se = bridge_median_lo = bridge_median_hi = 0.0
    bridge_p5_se = bridge_p5_lo = bridge_p5_hi = 0.0
    bridge_p95_se = bridge_p95_lo = bridge_p95_hi = 0.0
    super_median_se = super_median_lo = super_median_hi = 0.0
    if compute_se:
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(_BOOTSTRAP_SPAWN_KEY,)))
        bridge_mean_se = float(np.std(terminal, ddof=1)) / math.sqrt(n)
        bridge_median_se, bridge_median_lo, bridge_median_hi = _bootstrap_percentile(
            terminal, 50.0, rng=rng
        )
        bridge_p5_se, bridge_p5_lo, bridge_p5_hi = _bootstrap_percentile(terminal, 5.0, rng=rng)
        bridge_p95_se, bridge_p95_lo, bridge_p95_hi = _bootstrap_percentile(terminal, 95.0, rng=rng)
        super_median_se, super_median_lo, super_median_hi = _bootstrap_percentile(
            super_total, 50.0, rng=rng
        )

    # ── Near-miss / failure depth ─────────────────────────────────────────
    # A near miss is a failed trial or one whose real floor is at or below the
    # threshold. ``near_miss_only_rate`` counts the threshold band alone.
    near_miss_rate = float(np.mean(failed | (floor <= near_miss_threshold)))
    near_miss_only_rate = float(np.mean((floor > 0.0) & (floor <= near_miss_threshold)))
    failed_ages, failed_counts = np.unique(
        result.first_failure_age[result.first_failure_age >= 0], return_counts=True
    )
    failure_age_distribution = {
        int(age): int(count) for age, count in zip(failed_ages, failed_counts)
    }

    # ── Low-variance guardrail (Finding F1, M21) ─────────────────────────
    # Relative spread of the terminal bridge; flags near-deterministic runs.
    run_warnings: list[str] = []
    median = _percentile(terminal, 50.0)
    mean = float(np.mean(terminal))
    if n >= 100 and median > 0 and mean > 0:
        cv = float(np.std(terminal, ddof=1)) / mean
        if cv < 0.02:
            run_warnings.append(
                f"Monte Carlo terminal bridge variance is extremely low (CV={cv:.4%}). "
                "This may indicate a near-deterministic configuration. Check that "
                "investment accounts use asset-class returns rather than fixed "
                "interest_rate overrides."
            )

    ages = [inputs.simulation_start_age + y for y in range(result.bridge_by_age.shape[1])]

    return MonteCarloResults(
        trials=n,
        p_success=p_success,
        bridge_mean=mean,
        bridge_median=_percentile(terminal, 50.0),
        bridge_p5=_percentile(terminal, 5.0),
        bridge_p10=_percentile(terminal, 10.0),
        bridge_p25=_percentile(terminal, 25.0),
        bridge_p75=_percentile(terminal, 75.0),
        bridge_p90=_percentile(terminal, 90.0),
        bridge_p95=_percentile(terminal, 95.0),
        bridge_min=float(np.min(terminal)),
        bridge_floor=float(np.min(floor)),
        floor_age=int(result.floor_age[worst]),
        floor_end_bridge=float(terminal[worst]),
        super_median=_percentile(super_total, 50.0),
        horizon_age=horizon_age,
        per_earner_super_p50={
            earner.label: _percentile(per_earner_super[:, ei], 50.0)
            for ei, earner in enumerate(household.earners)
        },
        remaining_mortgage_p50={
            mortgage.label: _percentile(remaining_mortgage[:, mi], 50.0)
            for mi, mortgage in enumerate(household.mortgages)
        },
        mortgage_term_clearance_rate=mortgage_term_clearance_rate,
        per_mortgage_term_cleared_pct=per_mortgage_term_cleared_pct,
        per_mortgage_not_checkable=per_mortgage_not_checkable,
        seed=seed,
        bridge_mean_se=bridge_mean_se,
        bridge_median_se=bridge_median_se,
        bridge_p5_se=bridge_p5_se,
        bridge_p95_se=bridge_p95_se,
        super_median_se=super_median_se,
        # M1 — bootstrap percentile intervals
        bridge_median_ci_lo=bridge_median_lo,
        bridge_median_ci_hi=bridge_median_hi,
        bridge_p5_ci_lo=bridge_p5_lo,
        bridge_p5_ci_hi=bridge_p5_hi,
        bridge_p95_ci_lo=bridge_p95_lo,
        bridge_p95_ci_hi=bridge_p95_hi,
        super_median_ci_lo=super_median_lo,
        super_median_ci_hi=super_median_hi,
        # Near-miss and failure depth
        near_miss_rate=near_miss_rate,
        near_miss_only_rate=near_miss_only_rate,
        near_miss_threshold=near_miss_threshold,
        failure_age_distribution=failure_age_distribution,
        warnings=tuple(run_warnings),
        # Work Items 4, 5 — per-year drawdown composition (today's dollars)
        offset_drawn_p50=_across_trials(result.offset_drawn_by_age, 50.0),
        non_offset_drawn_p50=_across_trials(result.non_offset_drawn_by_age, 50.0),
        offset_drawn_p5=_across_trials(result.offset_drawn_by_age, 5.0),
        offset_drawn_p95=_across_trials(result.offset_drawn_by_age, 95.0),
        cgt_paid_p50=_across_trials(result.cgt_paid_by_age, 50.0),
        cgt_paid_p5=_across_trials(result.cgt_paid_by_age, 5.0),
        cgt_paid_p95=_across_trials(result.cgt_paid_by_age, 95.0),
        cgt_without_floor_p50=_across_trials(result.cgt_without_floor_by_age, 50.0),
        income_tax_p50=_across_trials(result.income_tax_by_age, 50.0),
        # W2b — per-trial investment-income tax totals (today's dollars)
        income_tax_total_p5=_percentile(income_tax_totals, 5.0),
        income_tax_total_p50=_percentile(income_tax_totals, 50.0),
        income_tax_total_p95=_percentile(income_tax_totals, 95.0),
        # M4a — per-trial drawdown total quantiles (today's dollars)
        offset_drawn_total_p5=_percentile(offset_drawn_totals, 5.0),
        offset_drawn_total_p50=_percentile(offset_drawn_totals, 50.0),
        offset_drawn_total_p95=_percentile(offset_drawn_totals, 95.0),
        non_offset_drawn_total_p5=_percentile(non_offset_drawn_totals, 5.0),
        non_offset_drawn_total_p50=_percentile(non_offset_drawn_totals, 50.0),
        non_offset_drawn_total_p95=_percentile(non_offset_drawn_totals, 95.0),
        cgt_paid_total_p5=_percentile(cgt_paid_totals, 5.0),
        cgt_paid_total_p50=_percentile(cgt_paid_totals, 50.0),
        cgt_paid_total_p95=_percentile(cgt_paid_totals, 95.0),
        cgt_without_floor_total_p5=_percentile(cgt_without_floor_totals, 5.0),
        cgt_without_floor_total_p50=_percentile(cgt_without_floor_totals, 50.0),
        cgt_without_floor_total_p95=_percentile(cgt_without_floor_totals, 95.0),
        cgt_floor_extra_total_p5=_percentile(cgt_floor_extra_totals, 5.0),
        cgt_floor_extra_total_p50=_percentile(cgt_floor_extra_totals, 50.0),
        cgt_floor_extra_total_p95=_percentile(cgt_floor_extra_totals, 95.0),
        # Work Items 1, 10 — per-year trajectory (real, per-trial deflated)
        bridge_by_age_ages=ages,
        bridge_by_age_p5=bridge_by_age_p[5],
        bridge_by_age_p10=bridge_by_age_p[10],
        bridge_by_age_p25=bridge_by_age_p[25],
        bridge_by_age_p50=bridge_by_age_p[50],
        bridge_by_age_p75=bridge_by_age_p[75],
        bridge_by_age_p90=bridge_by_age_p[90],
        bridge_by_age_p95=bridge_by_age_p[95],
        mortgage_by_age_ages=ages,
        mortgage_by_age=mortgage_by_age_out,
        offset_by_age=offset_by_age_out,
        mortgage_rate_by_age=mortgage_rate_by_age_out,
    )


def run_monte_carlo(
    inputs: SimulationInputs,
    *,
    seed: int | None = None,
    near_miss_threshold: float | None = None,
    compute_se: bool = True,
) -> MonteCarloResults:
    """Run ``inputs.n_iterations`` independent trials and summarise them.

    The household is ``inputs.household`` (the single source of truth, H10).
    All shocks are drawn up front by ``generate_draws``; the engine draws no
    random numbers itself.

    Args:
        inputs: Simulation parameters, including the household.
        seed: Seed for reproducibility. ``None`` draws fresh OS entropy.
        near_miss_threshold: Real floor (today's dollars) at or below which a trial
            counts as a near miss. ``None`` means one year of
            ``household.retirement_target``.
        compute_se: When False, skip the bootstrap and report standard errors and
            intervals as 0.0.

    Returns:
        ``MonteCarloResults`` (a ``SimulationResults``).

    Raises:
        ValueError: If ``sell_strategy`` is unsupported, the horizon is not
            positive, or ``near_miss_threshold`` is negative.
    """
    result = simulate_trials(inputs, seed=seed)
    return summarise_trials(
        inputs,
        result,
        seed=seed,
        near_miss_threshold=near_miss_threshold,
        compute_se=compute_se,
    )
