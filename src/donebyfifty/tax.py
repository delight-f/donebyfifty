"""Australian income tax maths: brackets, LITO, Medicare levy and MLS.

Pure tax functions. Every function accepts either a float (and returns a float)
or a float64 array (and returns an array of the same shape), so one code path
serves the scalar helpers and the batch engine (D22). Every function that
evaluates the bracket schedule takes the schedule explicitly (``brackets``) so
callers choose the financial year's schedule via ``brackets_for_fy`` /
``indexed_brackets``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import (
    Final,
    overload,
)

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
FloatOrArray = float | FloatArray

# =============================================================================
# TAX BRACKETS (legislated schedule per financial year)
# =============================================================================

# Resident marginal rates. The $135k/$190k thresholds are the amended Stage 3
# settings. The lowest non-zero rate steps down from 16% (FY2025-26) to 15%
# (FY2026-27) and 14% (FY2027-28 onwards).
TAX_FREE_THRESHOLD: Final[float] = 18_200.0
BRACKET_45K: Final[float] = 45_000.0
BRACKET_135K: Final[float] = 135_000.0
BRACKET_190K: Final[float] = 190_000.0

MEDICARE: Final[float] = 0.02

# Medicare levy low-income reduction, singles (FY2025-26). Below the lower
# threshold no levy is payable; between the thresholds the levy is 10% of the
# excess over the lower threshold, capped at the full 2%.
MEDICARE_LOW_INCOME_THRESHOLD: Final[float] = 28_011.0
MEDICARE_LOW_INCOME_UPPER: Final[float] = 35_013.0

# Medicare levy low-income reduction, families (FY2025-26, ATO "Medicare levy
# reduction - family income"). Family lower threshold plus a per-dependent-child
# add-on. The family reduction applies when the individual's taxable income
# exceeds MEDICARE_LOW_INCOME_UPPER.
MEDICARE_FAMILY_LOWER: Final[float] = 47_238.0
MEDICARE_FAMILY_CHILD_LOWER: Final[float] = 4_338.0

# Low Income Tax Offset (LITO), standard phase-out: full $700 to $37,500,
# then 5c/$ down to $325 at $45,000, then 1.5c/$ to zero at ~$66,667.
LITO_MAX: Final[float] = 700.0
LITO_MID: Final[float] = 325.0  # offset remaining at the second taper start
LITO_FIRST_TAPER_START: Final[float] = 37_500.0
LITO_SECOND_TAPER_START: Final[float] = 45_000.0
LITO_FIRST_TAPER_RATE: Final[float] = 0.05
LITO_SECOND_TAPER_RATE: Final[float] = 0.015

# Part-time work assumptions
PT_WEEKS_PER_YEAR: Final[float] = 48.0


# =============================================================================
# BRACKET SCHEDULES
# =============================================================================


def brackets_for_fy(fy_start_year: int) -> tuple[tuple[float, float], ...]:
    """Legislated resident bracket schedule for the financial year starting 1 July.

    The 18,201-45,000 band is 16% for FY2025-26 and earlier, 15% for FY2026-27
    and 14% from FY2027-28. The other bands (30%, 37%, 45%) are unchanged.

    Args:
        fy_start_year: Calendar year in which the financial year starts
            (e.g. 2025 for FY2025-26).

    Returns:
        ``(threshold, rate)`` pairs. Each rate applies to income up to its
        threshold, above the previous one.

    """
    if fy_start_year <= 2025:
        low_rate = 0.16
    elif fy_start_year == 2026:
        low_rate = 0.15
    else:
        low_rate = 0.14
    return (
        (TAX_FREE_THRESHOLD, 0.0),
        (BRACKET_45K, low_rate),
        (BRACKET_135K, 0.30),
        (BRACKET_190K, 0.37),
        (float("inf"), 0.45),
    )


def indexed_brackets(
    fy_start_year: int, years_elapsed: int, growth_rate: float
) -> tuple[tuple[float, float], ...]:
    """Bracket schedule for simulation year ``years_elapsed``.

    Starts from the legislated schedule for ``fy_start_year`` and scales every
    threshold by ``(1 + growth_rate) ** years_elapsed``. Rates are not indexed.

    Args:
        fy_start_year: Financial year start for the simulation year.
        years_elapsed: Simulation year index (0 for the first year).
        growth_rate: Annual threshold indexation rate (e.g. 0.025).

    Returns:
        Indexed ``(threshold, rate)`` pairs.

    """
    factor = (1.0 + growth_rate) ** years_elapsed
    return tuple((threshold * factor, rate) for threshold, rate in brackets_for_fy(fy_start_year))


# =============================================================================
# HELPERS
# =============================================================================


def _as_output(value: FloatArray) -> FloatOrArray:
    """Return a Python float for a 0-d result, otherwise the array itself."""
    return float(value) if value.ndim == 0 else value


def _step_rate(x: FloatArray, steps: tuple[tuple[float, float], ...]) -> FloatArray:
    """Rate of the first step whose threshold is >= ``x``; the last step above all."""
    thresholds = np.array([threshold for threshold, _ in steps], dtype=np.float64)
    rates = np.array([rate for _, rate in steps], dtype=np.float64)
    index = np.minimum(np.searchsorted(thresholds, x, side="left"), len(steps) - 1)
    return np.asarray(rates[index], dtype=np.float64)


# =============================================================================
# TAX CALCULATION
# =============================================================================


@lru_cache(maxsize=64)
def _bracket_table(
    brackets: tuple[tuple[float, float], ...],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Thresholds, lower bounds, rates and tax owed up to each bracket's lower bound."""
    thresholds = np.array([threshold for threshold, _ in brackets], dtype=np.float64)
    rates = np.array([rate for _, rate in brackets], dtype=np.float64)
    lower = np.concatenate(([0.0], thresholds[:-1]))
    below = np.concatenate(([0.0], np.cumsum(rates[:-1] * (thresholds[:-1] - lower[:-1]))))
    return thresholds, lower, rates, below


@overload
def bracket_tax(taxable_income: float, brackets: tuple[tuple[float, float], ...]) -> float: ...
@overload
def bracket_tax(
    taxable_income: FloatArray, brackets: tuple[tuple[float, float], ...]
) -> FloatArray: ...
def bracket_tax(
    taxable_income: FloatOrArray, brackets: tuple[tuple[float, float], ...]
) -> FloatOrArray:
    """Progressive income tax from the bracket schedule, before offsets/levies.

    Single source of truth for bracket progression: ``tax()`` and the CGT
    bracket-stacking integration both evaluate the schedule through this
    function, so ``tax(b) - tax(a)`` is exactly the tax on the slice ``[a, b]``.

    Args:
        taxable_income: Annual taxable income (float or array). Non-positive
            income yields 0.0.
        brackets: Bracket schedule, e.g. from ``brackets_for_fy``.

    Returns:
        Income tax payable on ``taxable_income``, excluding the Medicare levy,
        the Medicare Levy Surcharge and all offsets. Never negative.

    """
    if not brackets:
        raise ValueError("bracket schedule is empty; pass brackets_for_fy/indexed_brackets output")
    thresholds, lower, rates, below = _bracket_table(brackets)
    x = np.maximum(np.asarray(taxable_income, dtype=np.float64), 0.0)
    # Bracket k covers (thresholds[k-1], thresholds[k]]; the last bracket takes any excess.
    index = np.minimum(np.searchsorted(thresholds, x, side="left"), len(rates) - 1)
    income_tax = below[index] + rates[index] * (x - lower[index])
    return _as_output(np.asarray(income_tax, dtype=np.float64))


@overload
def lito_offset(taxable_income: float) -> float: ...
@overload
def lito_offset(taxable_income: FloatArray) -> FloatArray: ...
def lito_offset(taxable_income: FloatOrArray) -> FloatOrArray:
    """Low Income Tax Offset (LITO) available at a given taxable income.

    Standard phase-out: the full $700 applies up to $37,500, tapers at 5c/$
    to $325 at $45,000, then at 1.5c/$ to zero at about $66,667. The offset is
    non-refundable — it can reduce income tax to, but not below, zero.
    """
    x = np.asarray(taxable_income, dtype=np.float64)
    first_taper = np.maximum(0.0, LITO_MAX - LITO_FIRST_TAPER_RATE * (x - LITO_FIRST_TAPER_START))
    second_taper = np.maximum(
        0.0, LITO_MID - LITO_SECOND_TAPER_RATE * (x - LITO_SECOND_TAPER_START)
    )
    offset = np.where(
        x <= LITO_FIRST_TAPER_START,
        LITO_MAX,
        np.where(x <= LITO_SECOND_TAPER_START, first_taper, second_taper),
    )
    return _as_output(np.asarray(offset, dtype=np.float64))


@overload
def medicare_levy(
    taxable_income: float, *, family_income: float | None = None, dependants: int = 0
) -> float: ...
@overload
def medicare_levy(
    taxable_income: FloatArray,
    *,
    family_income: FloatOrArray | None = None,
    dependants: int = 0,
) -> FloatArray: ...
def medicare_levy(
    taxable_income: FloatOrArray,
    *,
    family_income: FloatOrArray | None = None,
    dependants: int = 0,
) -> FloatOrArray:
    """Medicare levy (2%) with the low-income reduction applied.

    Singles (``family_income is None``, or ``taxable_income`` at or below
    ``MEDICARE_LOW_INCOME_UPPER``): no levy at or below
    ``MEDICARE_LOW_INCOME_THRESHOLD``; between the thresholds the levy is 10% of
    the excess over the lower threshold, capped at the full 2%.

    Families (``family_income`` given and ``taxable_income`` above
    ``MEDICARE_LOW_INCOME_UPPER``): reduced levy per spouse is
    ``min(full levy, 0.10 * (family_income - family_lower) * taxable_income /
    family_income)`` where ``family_lower = MEDICARE_FAMILY_LOWER +
    MEDICARE_FAMILY_CHILD_LOWER * dependants``. The reduction reaches the full
    levy at the family upper threshold ($59,047 + $5,423 per child), so no
    separate upper bound is needed.

    Args:
        taxable_income: The individual's taxable income (float or array).
        family_income: Combined family taxable income (couples). None for singles.
        dependants: Number of dependent children (family threshold add-on).

    Returns:
        Medicare levy payable, never negative.

    Example:
        A spouse with $40,000 taxable and $80,000 family income, no children:
        family_lower = 47,238, so 0.10 * (80,000 - 47,238) * 40,000 / 80,000
        = 1,638; full levy = 800; payable = min(800, 1,638) = 800.
        With $60,000 family income: 0.10 * 12,762 * 40,000 / 60,000 = 850.8,
        so payable = min(800, 850.8) = 800.

    """
    x = np.asarray(taxable_income, dtype=np.float64)
    full_levy = x * MEDICARE
    individual = np.where(
        x <= MEDICARE_LOW_INCOME_THRESHOLD,
        0.0,
        np.where(
            x < MEDICARE_LOW_INCOME_UPPER,
            np.minimum(full_levy, 0.10 * (x - MEDICARE_LOW_INCOME_THRESHOLD)),
            full_levy,
        ),
    )
    if family_income is None:
        levy = individual
    else:
        family = np.asarray(family_income, dtype=np.float64)
        family_lower = MEDICARE_FAMILY_LOWER + MEDICARE_FAMILY_CHILD_LOWER * dependants
        # Guard the division: only families above the lower threshold are reduced.
        above = family > family_lower
        safe_family = np.where(above, family, 1.0)
        reduced = np.where(
            above,
            np.minimum(full_levy, 0.10 * (family - family_lower) * x / safe_family),
            0.0,
        )
        levy = np.where(x > MEDICARE_LOW_INCOME_UPPER, reduced, individual)
    return _as_output(np.asarray(levy, dtype=np.float64))


@overload
def tax(
    taxable_income: float,
    *,
    brackets: tuple[tuple[float, float], ...],
    medicare_surcharge: float = 0.0,
    family_income: float | None = None,
    dependants: int = 0,
) -> float: ...
@overload
def tax(
    taxable_income: FloatArray,
    *,
    brackets: tuple[tuple[float, float], ...],
    medicare_surcharge: FloatOrArray = 0.0,
    family_income: FloatOrArray | None = None,
    dependants: int = 0,
) -> FloatArray: ...
def tax(
    taxable_income: FloatOrArray,
    *,
    brackets: tuple[tuple[float, float], ...],
    medicare_surcharge: FloatOrArray = 0.0,
    family_income: FloatOrArray | None = None,
    dependants: int = 0,
) -> FloatOrArray:
    """Calculate Australian personal income tax plus Medicare levy.

    Applies progressive marginal rates across ``brackets``, deducts the
    non-refundable LITO, then adds the Medicare levy (with the low-income
    reduction) plus any Medicare Levy Surcharge. Non-positive income yields 0.0.

    Args:
        taxable_income: Total taxable income for the year (float or array).
        brackets: Bracket schedule for the year (``brackets_for_fy`` or
            ``indexed_brackets``).
        medicare_surcharge: Additional Medicare Levy Surcharge rate
            (e.g. 0.01 for Tier 1 MLS). Charged on total taxable income.
        family_income: Combined family taxable income for couples, used for the
            family Medicare levy reduction. None for singles.
        dependants: Number of dependent children (family levy threshold add-on).

    Returns:
        Total tax payable (income tax + Medicare levy + MLS), never negative.

    """
    x = np.asarray(taxable_income, dtype=np.float64)
    income_tax = np.maximum(0.0, bracket_tax(x, brackets) - lito_offset(x))
    levy = medicare_levy(x, family_income=family_income, dependants=dependants)
    total = income_tax + levy + x * medicare_surcharge
    return _as_output(np.asarray(np.where(x > 0.0, total, 0.0), dtype=np.float64))


@overload
def marginal_rate(taxable_income: float, *, brackets: tuple[tuple[float, float], ...]) -> float: ...
@overload
def marginal_rate(
    taxable_income: FloatArray, *, brackets: tuple[tuple[float, float], ...]
) -> FloatArray: ...
def marginal_rate(
    taxable_income: FloatOrArray, *, brackets: tuple[tuple[float, float], ...]
) -> FloatOrArray:
    """Marginal (top-bracket) tax rate for a given taxable income.

    Returns the rate of the bracket containing the last dollar of income.

    Args:
        taxable_income: Annual taxable income (float or array).
        brackets: Bracket schedule for the year.

    Returns:
        Marginal tax rate as a decimal (e.g. 0.45 for 45%).

    """
    x = np.asarray(taxable_income, dtype=np.float64)
    return _as_output(_step_rate(x, brackets))


# =============================================================================
# MEDICARE LEVY SURCHARGE TIERS (FY2026-27)
# =============================================================================

# FY2026-27 singles. Source: ATO MLS thresholds (secondary sources; the two ATO
# pages returned HTTP 504 during review, so treat the intermediate boundaries as
# best-available and keep them here as named constants for easy correction).
MLS_TIERS_SINGLE: Final[tuple[tuple[float, float], ...]] = (
    (105_000.0, 0.0),  # below $105k: no MLS
    (123_000.0, 0.01),  # $105k-$123k: Tier 1 (1.0%)
    (164_000.0, 0.0125),  # $123k-$164k: Tier 2 (1.25%)
    (float("inf"), 0.015),  # $164k+: Tier 3 (1.5%)
)

# FY2026-27 family thresholds (combined income). The base already allows for
# one child; each further child adds MLS_FAMILY_CHILD_UPLIFT via ``mls_rate``.
MLS_TIERS_COUPLE: Final[tuple[tuple[float, float], ...]] = (
    (210_000.0, 0.0),  # below $210k: no MLS
    (246_000.0, 0.01),  # $210k-$246k: Tier 1 (1.0%)
    (328_000.0, 0.0125),  # $246k-$328k: Tier 2 (1.25%)
    (float("inf"), 0.015),  # $328k+: Tier 3 (1.5%)
)

MLS_FAMILY_CHILD_UPLIFT: Final[float] = 1_500.0


@overload
def mls_rate(mls_income: float, *, couple: bool, n_children: int) -> float: ...
@overload
def mls_rate(mls_income: FloatArray, *, couple: bool, n_children: int) -> FloatArray: ...
def mls_rate(mls_income: FloatOrArray, *, couple: bool, n_children: int) -> FloatOrArray:
    """Medicare Levy Surcharge rate for an MLS income level (float or array).

    Couples use the family tiers, lifted by ``MLS_FAMILY_CHILD_UPLIFT`` for
    each child beyond the first. Singles ignore ``n_children``.

    Args:
        mls_income: Income tested for MLS. For couples, the combined household
            income for MLS purposes; for singles, the individual's income.
        couple: True for a couple (family tiers), False for a single.
        n_children: Number of dependent children.

    Returns:
        MLS rate as a decimal (e.g. 0.0125 for Tier 2).

    """
    if couple:
        uplift = MLS_FAMILY_CHILD_UPLIFT * max(0, n_children - 1)
        tiers = tuple((threshold + uplift, rate) for threshold, rate in MLS_TIERS_COUPLE)
    else:
        tiers = MLS_TIERS_SINGLE
    x = np.asarray(mls_income, dtype=np.float64)
    return _as_output(_step_rate(x, tiers))
