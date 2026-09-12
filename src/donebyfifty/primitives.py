"""Verified mathematical primitives for the Monte Carlo retirement simulator.

All functions in this module are extracted from a model verified by a Professor
of Finance. They are preserved with zero semantic changes to the core logic.

Two functions (``amortize_mortgage_monthly``, ``handle_offset_overflow``) have
been converted from mutating a dataclass object in-place to pure functions that
accept and return float values. Their internal arithmetic is identical to the
verified original.

Return assumptions: All asset-class return parameters (module-level constants
below) represent real (inflation-adjusted) returns. The generators uplift each
draw to nominal via their ``inflation`` argument
(``(1 + real) * (1 + inflation) - 1``), matching the engine's nominal cash
flows; ``simulation.py`` then deflates the output to today's dollars once.
Returns are applied via lognormal draws using the standard Black-Scholes
parameterisation (``mu = ln(1 + mean) - ½σ²``).
Provenance and confidence for each figure are documented per-constant below.
Not all figures are independently sourced; see individual comments for details.
"""

from __future__ import annotations

import math
import random
from typing import Final, Literal, Sequence, TypedDict, overload

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class AssetHolding(TypedDict):
    """Mutable container for an asset's market value and cost basis."""

    val: float
    basis: float


# =============================================================================
# TAX BRACKETS (FY2025-26, as amended Stage 3)
# =============================================================================

# Resident marginal rates from 1 July 2024 (Income Tax Rates Act 1986, Sch 7).
# The $135k/$190k thresholds are the amended Stage 3 settings — NOT the
# pre-2024 $120k/$180k thresholds.
BRACKETS: Final[tuple[tuple[float, float], ...]] = (
    (18_200, 0.0),
    (45_000, 0.16),
    (135_000, 0.30),
    (190_000, 0.37),
    (float("inf"), 0.45),
)

MEDICARE: Final[float] = 0.02

# Medicare levy low-income reduction (FY2025-26 singles). Below the lower
# threshold no levy is payable; between the thresholds the levy is 10% of the
# excess over the lower threshold, capped at the full 2%. Families use a higher
# threshold plus a per-dependent-child add-on.
MEDICARE_LOW_INCOME_THRESHOLD: Final[float] = 28_011.0
MEDICARE_LOW_INCOME_UPPER: Final[float] = 35_013.0
MEDICARE_FAMILY_UPLIFT: Final[float] = 0.0  # placeholder; family table not sourced
MEDICARE_CHILD_UPLIFT: Final[float] = 1_500.0  # per dependent child

# Low Income Tax Offset (LITO), standard phase-out: full $700 to $37,500,
# then 5c/$ down to $325 at $45,000, then 1.5c/$ to zero at ~$66,667.
LITO_MAX: Final[float] = 700.0
LITO_MID: Final[float] = 325.0  # offset remaining at the second taper start
LITO_FIRST_TAPER_START: Final[float] = 37_500.0
LITO_SECOND_TAPER_START: Final[float] = 45_000.0
LITO_FIRST_TAPER_RATE: Final[float] = 0.05
LITO_SECOND_TAPER_RATE: Final[float] = 0.015
CGT_FLOOR_RATE: Final[float] = (
    0.30  # post-2027 minimum (floor) CGT rate on real indexed gains, per owner
)
# NOTE: This is a FLOOR, not a universal rate. The actual CGT rate per owner is
# max(marginal_rate, CGT_FLOOR_RATE). For owners in the 37% or 45% bracket, their
# marginal rate applies; this floor only catches owners whose marginal rate is below 30%.
# (The 50% CGT discount was abolished post-2027; CPI indexation of cost basis replaces it.)

# =============================================================================
# Asset class return assumptions (real, inflation-adjusted)
# where mu = ln(1 + mean) - 0.5*sigma^2 (Black-Scholes parameterisation).
# Provenance and confidence are documented per-constant below.
# =============================================================================

# --- Equity (Australian) --------------------------------------------------------
# Basis: real (inflation-adjusted).
# Source: Credit Suisse / UBS Global Investment Returns Yearbook (with London
#   Business School), Australian equities annualised real return since 1900:
#   ~6.4-6.7% (edition-dependent; figure varies slightly by edition year and
#   should not be cited as a single precise value).
# Confidence: sourced (unconditional long-run historical; not valuation-adjusted).
# Caveats: This is a 120+ year unconditional average, not a forward-looking
#   forecast. Several major houses (Vanguard VCMM, Fidelity capital markets
#   assumptions) currently forecast below-historical-average equity returns for
#   the next decade due to elevated valuations. Data pre-1980 uses inconsistent
#   methodology across sources (Market Index, 2024) and should be treated as
#   indicative rather than precise. If precision matters more than defensibility
#   for the model's horizon (10-20 years), consider a valuation-conditioned
#   alternative instead.
EQ_MEAN: Final[float] = 0.07

# Confidence: unsourced-placeholder. No citation found for this figure.
EQ_STD: Final[float] = 0.15

# --- Super (equity-like) --------------------------------------------------------
# Basis: real (inflation-adjusted), equity-like by assumption.
# Source: UNSOURCED. Currently set equal to the equity mean by assumption
#   (no independent source for superannuation-specific returns).
# Confidence: unsourced-placeholder. A defensible source would be a specific
#   super fund's published PDS investment-option return assumptions (e.g.
#   balanced or growth option), not yet retrieved. The correlation to equity
#   (SUPER_EQ_CORR = 0.80) is also an unsourced placeholder.
# Caveats: Super funds are not identical to a pure equity index — they hold
#   multi-asset portfolios with fees that reduce net returns. Treating super as
#   "equity minus a discount" (lower mean, lower std, imperfect correlation) is
#   a reasonable modelling choice but the specific numbers are unsourced.
SUPER_MEAN: Final[float] = 0.07

# Confidence: unsourced-placeholder. No citation found for this figure.
SUPER_STD: Final[float] = 0.12

# Confidence: unsourced-placeholder. Target correlation between super and equity.
SUPER_EQ_CORR: Final[float] = 0.80

# --- Bonds ---------------------------------------------------------------------
# Basis: real (inflation-adjusted).
# Source: RBA Bulletin (Fraser, 1991), "Three Decades of Real Interest Rates".
#   Australian bond rate deflated by CPI averaged ~1.5% real over the period to
#   1990.
# Confidence: derived-estimate. The source supports a real bond return in the
#   1-2% range. The current figure of 3.0% sits above this range.
# Caveats: Fraser (1991) predates the low-rate 2010s-2020s period and may not
#   reflect more recent conditions. The current 3.0% figure is higher than the
#   1-2% band the source implies — this gap is flagged explicitly rather than
#   silently adjusting the constant. If a more recent source is found, consider
#   revising this downward.
BOND_MEAN: Final[float] = 0.03

# Confidence: unsourced-placeholder. No citation found for this figure.
BOND_STD: Final[float] = 0.06

# --- Cash ----------------------------------------------------------------------
# Basis: real (inflation-adjusted).
# Source: Same RBA source as Bonds — Fraser (1991) treats short-term / fixed-
#   interest real returns as roughly 1-2% real, without cleanly separating cash
#   from short bonds.
# Confidence: derived-estimate. No cash-specific historical series was found.
#   The current 2.5% figure sits at the upper edge of the 1-2% band the RBA
#   source implies.
# Caveats: Same vintage caveat as Bonds. A more recent cash-rate series (e.g.
#   RBA cash rate target deflated by CPI) would strengthen this estimate.
CASH_MEAN: Final[float] = 0.025

# Confidence: unsourced-placeholder. No citation found for this figure.
CASH_STD: Final[float] = 0.01

# --- Property (Australian residential) -----------------------------------------
# Basis: real (inflation-adjusted).
# Source: CoreLogic — Australian dwelling prices grew ~6.4% p.a. (nominal,
#   capital growth only) over the 30 years to ~2024/2025. A separate CoreLogic
#   figure reports cumulative capital growth of 382% vs cumulative CPI growth of
#   99.5% over the 30 years to 2022, implying comparable annualised nominal
#   capital growth net of inflation.
# Confidence: derived-estimate. The figures are capital growth only — no rental
#   yield included, and they are nominal, requiring deflation to match the
#   model's real-return convention. Derivation: real capital growth (~3.5-4%
#   after deflating) + estimated net rental yield (~2-3%, not independently
#   sourced) ≈ current 5% figure.
# Caveats: The yield component (~2-3%) is not independently sourced; the two
#   proxies (CoreLogic capital growth + unsourced yield estimate) were combined
#   by the reviewer, not drawn from a single series. Capital growth data reflects
#   a specific 30-year window and may not be representative of all holding
#   periods. A total-return property index (e.g. MSCI Australia) would be
#   preferable but was not retrieved.
PROPERTY_MEAN: Final[float] = 0.05

# Confidence: unsourced-placeholder. No citation found for this figure.
PROPERTY_STD: Final[float] = 0.12

# --- Intl Equity ---------------------------------------------------------------
# Basis: real (inflation-adjusted).
# Source: UNSOURCED. No citation found.
# Confidence: unsourced-placeholder.
# Caveats: International equity returns are not independent of Australian equity
#   returns — the model captures this via INTL_EQ_CORR (also unsourced). A
#   defensible source would be MSCI World ex-Australia or equivalent, not yet
#   retrieved.
INTL_EQ_MEAN: Final[float] = 0.065

# Confidence: unsourced-placeholder. No citation found for this figure.
INTL_EQ_STD: Final[float] = 0.16

# --- Cross-asset correlations with equity -------------------------------------
# All correlation figures are unsourced-placeholder. No citation found for any.
# A defensible source would be historical pairwise correlation matrices from
# index providers (e.g. ASX, MSCI, Bloomberg AusBond) over a consistent lookback
# window.
BOND_EQ_CORR: Final[float] = 0.10
CASH_EQ_CORR: Final[float] = 0.0
PROPERTY_EQ_CORR: Final[float] = 0.60
INTL_EQ_CORR: Final[float] = 0.70

ASSET_CLASS_PARAMS: Final[dict[str, dict[str, float]]] = {
    "equity": {"mean": EQ_MEAN, "std": EQ_STD, "corr_with_eq": 1.0},
    "bonds": {"mean": BOND_MEAN, "std": BOND_STD, "corr_with_eq": BOND_EQ_CORR},
    "cash": {"mean": CASH_MEAN, "std": CASH_STD, "corr_with_eq": CASH_EQ_CORR},
    "property": {"mean": PROPERTY_MEAN, "std": PROPERTY_STD, "corr_with_eq": PROPERTY_EQ_CORR},
    "intl_equity": {"mean": INTL_EQ_MEAN, "std": INTL_EQ_STD, "corr_with_eq": INTL_EQ_CORR},
}

# Precomputed Cholesky mu constants (log-mean adjusted for variance)
MU_EQ: Final[float] = math.log(1 + EQ_MEAN) - 0.5 * EQ_STD**2
MU_SUPER: Final[float] = math.log(1 + SUPER_MEAN) - 0.5 * SUPER_STD**2
MU_BOND: Final[float] = math.log(1 + BOND_MEAN) - 0.5 * BOND_STD**2
MU_CASH: Final[float] = math.log(1 + CASH_MEAN) - 0.5 * CASH_STD**2
MU_PROPERTY: Final[float] = math.log(1 + PROPERTY_MEAN) - 0.5 * PROPERTY_STD**2
MU_INTL_EQ: Final[float] = math.log(1 + INTL_EQ_MEAN) - 0.5 * INTL_EQ_STD**2

# Log-mean map for asset class lookup (used in generate_asset_return)
_ASSET_MU: Final[dict[str, float]] = {
    "equity": MU_EQ,
    "bonds": MU_BOND,
    "cash": MU_CASH,
    "property": MU_PROPERTY,
    "intl_equity": MU_INTL_EQ,
}

# Inflation parameters (for stochastic inflation)
# Inflation is modelled as IID lognormal with no mean reversion.
# This is a common simplification in retirement modelling but can
# produce unrealistic cumulative inflation paths over horizons > 15 years.
# For production use, consider a mean-reverting process (Vasicek/CIR).
INFLATION_MEAN: Final[float] = 0.025
INFLATION_STD: Final[float] = 0.015
# Log-mean with half-variance correction so E[inflation] = INFLATION_MEAN
MU_INFLATION: Final[float] = math.log(1 + INFLATION_MEAN) - 0.5 * INFLATION_STD**2
INFLATION_EQ_CORR: Final[float] = (
    -0.15
)  # mild negative correlation: high inflation often coincides with poor equity returns
SUPER_INF_CORR: Final[float] = -0.10  # correlation between super and inflation

# Part-time work assumptions
PT_DAILY_RATE: Final[float] = 3000.0
PT_WEEKS_PER_YEAR: Final[float] = 48.0

# Tax calculation cache: (gross_income, brackets_hash) -> net_income
# The cache uses a content-based key so it works across years even when
# brackets are rebuilt as new tuple objects (as long as the values are the same).
PT_TAX_CALC_CACHE: dict[tuple[float, tuple[tuple[float, float], ...]], float] = {}

# Education cost schedule (reference family's actual fee schedule, today's dollars)
# $292,008 total, from original client's fee schedule.
# Preschool/daycare (ages 0-4): ~$15,000/yr (average)
# Prep (age 5):      $19,168
# Years 1-6 (6-11):  ~$19,972-20,352/yr
# Years 7-12 (12-17): ~$25,224-25,600/yr
EDU_SCHEDULE_TODAY: Final[dict[int, float]] = {
    0: 15000,
    1: 15000,
    2: 15000,
    3: 15000,
    4: 15000,
    5: 19168,
    6: 19972,
    7: 19976,
    8: 20040,
    9: 20196,
    10: 20188,
    11: 20352,
    12: 25308,
    13: 25292,
    14: 25600,
    15: 25224,
    16: 25316,
    17: 25376,
}


# =============================================================================
# TAX CALCULATION
# =============================================================================


def bracket_tax(
    taxable_income: float, brackets: tuple[tuple[float, float], ...] | None = None
) -> float:
    """Progressive income tax from the bracket schedule, before offsets/levies.

    Single source of truth for bracket progression: ``tax()``,
    ``consulting_net_income()`` and the CGT bracket-stacking integration
    (``cgt_rates_on_gain``) all evaluate the schedule through this function,
    so ``tax(b) - tax(a)`` is exactly the tax on the slice ``[a, b]``.

    Args:
        taxable_income: Annual taxable income. Non-positive income yields 0.0.
        brackets: Optional bracket overrides (e.g. indexed brackets).
            If None, uses the module-level ``BRACKETS``.

    Returns:
        Income tax payable on ``taxable_income``, excluding the Medicare levy,
        the Medicare Levy Surcharge and all offsets. Never negative.

    """
    if taxable_income <= 0.0:
        return 0.0
    active = brackets if brackets is not None else BRACKETS
    total_tax: float = 0.0
    prev_threshold: float = 0.0

    for threshold, rate in active:
        if taxable_income > threshold:
            total_tax += (threshold - prev_threshold) * rate
            prev_threshold = threshold
        else:
            total_tax += (taxable_income - prev_threshold) * rate
            break

    return total_tax


def lito_offset(taxable_income: float) -> float:
    """Low Income Tax Offset (LITO) available at a given taxable income.

    Standard phase-out: the full $700 applies up to $37,500, tapers at 5c/$
    to $325 at $45,000, then at 1.5c/$ to zero at about $66,667. The offset is
    non-refundable — it can reduce income tax to, but not below, zero.
    """
    if taxable_income <= LITO_FIRST_TAPER_START:
        return LITO_MAX
    if taxable_income <= LITO_SECOND_TAPER_START:
        tapered = LITO_MAX - LITO_FIRST_TAPER_RATE * (taxable_income - LITO_FIRST_TAPER_START)
        return max(0.0, tapered)
    return max(0.0, LITO_MID - LITO_SECOND_TAPER_RATE * (taxable_income - LITO_SECOND_TAPER_START))


def medicare_levy(
    taxable_income: float,
    *,
    dependants: int = 0,
    has_spouse: bool = False,
) -> float:
    """Medicare levy with the low-income reduction applied.

    Singles: no levy at or below ``MEDICARE_LOW_INCOME_THRESHOLD``; between the
    thresholds the levy is 10% of the excess over the lower threshold, capped at
    the full 2%. The family threshold is lifted when a spouse is present and by
    ``MEDICARE_CHILD_UPLIFT`` per dependent child.

    Note:
        The family threshold uplift is a placeholder (see
        ``MEDICARE_FAMILY_UPLIFT``) pending a sourced ATO figure; the child
        add-on is the commonly cited ~$1,500/child.

    """
    lower = MEDICARE_LOW_INCOME_THRESHOLD + (
        MEDICARE_FAMILY_UPLIFT if has_spouse else 0.0
    ) + MEDICARE_CHILD_UPLIFT * dependants
    upper = MEDICARE_LOW_INCOME_UPPER + (
        MEDICARE_FAMILY_UPLIFT if has_spouse else 0.0
    ) + MEDICARE_CHILD_UPLIFT * dependants
    full_levy = taxable_income * MEDICARE
    if taxable_income <= lower:
        return 0.0
    if taxable_income < upper:
        return min(full_levy, 0.10 * (taxable_income - lower))
    return full_levy


def tax(
    taxable_income: float,
    medicare_surcharge: float = 0.0,
    brackets: tuple[tuple[float, float], ...] | None = None,
    *,
    dependants: int = 0,
    has_spouse: bool = False,
) -> float:
    """Calculate Australian personal income tax plus Medicare levy.

    Applies progressive marginal rates across the bracket schedule, deducts the
    non-refundable LITO, then adds the Medicare levy (with the low-income
    reduction) plus any Medicare Levy Surcharge. Non-positive income yields 0.0.

    Args:
        taxable_income: Total taxable income for the year.
        medicare_surcharge: Additional Medicare Levy Surcharge rate
            (e.g. 0.01 for Tier 1 MLS). Added to the standard 2% levy and
            charged on total taxable income (no low-income reduction applies).
        brackets: Optional bracket overrides for tax indexation.
            If None, uses the module-level ``BRACKETS``.
        dependants: Number of dependent children (family levy threshold add-on).
        has_spouse: Whether the taxpayer has a spouse (family levy threshold).

    Returns:
        Total tax payable (income tax + Medicare levy + MLS), never negative.

    """
    if taxable_income <= 0.0:
        return 0.0

    income_tax = max(0.0, bracket_tax(taxable_income, brackets) - lito_offset(taxable_income))
    levy = medicare_levy(taxable_income, dependants=dependants, has_spouse=has_spouse)

    return income_tax + levy + taxable_income * medicare_surcharge


def marginal_rate(
    taxable_income: float, brackets: tuple[tuple[float, float], ...] | None = None
) -> float:
    """Find the marginal (top-bracket) tax rate for a given taxable income.

    Reuses the same ``BRACKETS`` tuple as ``tax()`` — single source of truth.
    Returns the rate of the bracket containing the last dollar of income.

    Args:
        taxable_income: Annual taxable income.
        brackets: Optional bracket overrides (e.g. indexed brackets).
            If None, uses the module-level ``BRACKETS``.

    Returns:
        Marginal tax rate as a decimal (e.g. 0.45 for 45%).

    """
    active = brackets if brackets is not None else BRACKETS
    for threshold, bracket_rate in active:
        if taxable_income <= threshold:
            return bracket_rate
    return active[-1][1] if active else 0.0


def mls_rate_for_income(
    taxable_income: float,
    n_earners: int = 1,
    n_children: int = 1,
) -> float:
    """Compute the Medicare Levy Surcharge rate for a given taxable income.

    Selects singles or family tiers based on ``n_earners`` (≥ 2 → family).
    Family tier thresholds are lifted by the per-child add-on (the family base
    already covers the first child, so ``n_children`` defaults to 1).
    Returns 0.0 for income below the lowest threshold.

    Args:
        taxable_income: Annual taxable income (combined for couples).
        n_earners: Number of earners in the household (1 = singles tiers).
        n_children: Dependent children (family tiers only).

    Returns:
        MLS rate as a decimal (e.g. 0.0125 for Tier 2).

    """
    if n_earners >= 2:
        uplift = MLS_FAMILY_CHILD_UPLIFT * (n_children - 1)
        tiers = tuple((threshold + uplift, rate) for threshold, rate in MLS_TIERS_COUPLE)
    else:
        tiers = MLS_TIERS_SINGLE
    for threshold, rate in tiers:
        if taxable_income <= threshold:
            return rate
    return tiers[-1][1] if tiers else 0.0


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
# one child; each further child adds MLS_FAMILY_CHILD_UPLIFT via
# ``mls_rate_for_income``.
MLS_TIERS_COUPLE: Final[tuple[tuple[float, float], ...]] = (
    (210_000.0, 0.0),  # below $210k: no MLS
    (246_000.0, 0.01),  # $210k-$246k: Tier 1 (1.0%)
    (328_000.0, 0.0125),  # $246k-$328k: Tier 2 (1.25%)
    (float("inf"), 0.015),  # $328k+: Tier 3 (1.5%)
)

MLS_FAMILY_CHILD_UPLIFT: Final[float] = 1_500.0


# =============================================================================
# PART-TIME CONSULTING INCOME
# =============================================================================


def consulting_net_income(
    days_per_week: float,
    brackets: tuple[tuple[float, float], ...] | None = None,
    daily_rate: float | None = None,
    weeks_per_year: float | None = None,
) -> float:
    """Compute after-tax income from part-time consulting.

    Args:
        days_per_week: Average days worked per week (e.g. 1.0, 2.0, 0.5).
        brackets: Optional bracket overrides for tax indexation.
            If None, uses the module-level ``BRACKETS``.
        daily_rate: Daily consulting rate. If None, uses module-level
            ``PT_DAILY_RATE`` (3,000).
        weeks_per_year: Weeks worked per year. If None, uses module-level
            ``PT_WEEKS_PER_YEAR`` (48).

    Returns:
        Net after-tax income from consulting for one year.

    """
    dr = daily_rate if daily_rate is not None else PT_DAILY_RATE
    wpy = weeks_per_year if weeks_per_year is not None else PT_WEEKS_PER_YEAR
    gross = days_per_week * wpy * dr
    if gross <= 0:
        return 0.0
    # Consult the cache first
    active_brackets = brackets if brackets is not None else BRACKETS
    # Use (gross, brackets) as content-based cache key — brackets tuple is
    # hashable by value, so identical indexed brackets hit across years.
    cache_key = (gross, brackets) if brackets is not None else (gross, BRACKETS)
    if cache_key in PT_TAX_CALC_CACHE:
        return PT_TAX_CALC_CACHE[cache_key]
    total_tax = 0.0
    prev = 0.0
    for threshold, rate in active_brackets:
        if gross > threshold:
            total_tax += (threshold - prev) * rate
            prev = threshold
        else:
            total_tax += (gross - prev) * rate
            break
    total_tax += gross * MEDICARE
    net = gross - total_tax
    PT_TAX_CALC_CACHE[cache_key] = net
    return net


def clear_tax_cache() -> None:
    """Reset the part-time income tax calculation cache.

    Must be called between independent simulation runs to prevent stale
    bracket-indexed entries from one run leaking into another.
    """
    PT_TAX_CALC_CACHE.clear()


# =============================================================================
# CORRELATED RETURN GENERATION (Cholesky decomposition)
# =============================================================================


def _require_correlation(name: str, value: float) -> None:
    """Reject a correlation coefficient outside [-1, 1] with a clear error."""
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"{name}={value!r} is not a valid correlation (must be in [-1, 1])")


def validate_correlations(rho_se: float, rho_ei: float, rho_si: float) -> None:
    """Validate a 3x3 correlation matrix before Cholesky decomposition (M12).

    Each coefficient must lie in [-1, 1]; ``rho_se`` must be strictly inside the
    range because the Cholesky factor divides by ``sqrt(1 - rho_se**2)``. The
    matrix must be positive semi-definite, i.e. its determinant
    ``1 + 2*rho_se*rho_ei*rho_si - rho_se**2 - rho_ei**2 - rho_si**2`` is
    non-negative. This replaces the silent ``max(0.0, l22_sq)`` clamp, so a
    non-PSD matrix fails loudly and names the offending values.

    Args:
        rho_se: Equity-super correlation.
        rho_ei: Equity-inflation correlation.
        rho_si: Super-inflation correlation.

    Raises:
        ValueError: If any coefficient is out of range, ``|rho_se| == 1``, or
            the implied matrix is not positive semi-definite.

    """
    _require_correlation("rho_se", rho_se)
    _require_correlation("rho_ei", rho_ei)
    _require_correlation("rho_si", rho_si)
    if abs(rho_se) >= 1.0:
        raise ValueError(
            f"rho_se={rho_se!r} makes the correlation matrix singular "
            "(division by sqrt(1 - rho_se**2))"
        )
    determinant = (
        1.0
        + 2.0 * rho_se * rho_ei * rho_si
        - rho_se**2
        - rho_ei**2
        - rho_si**2
    )
    if determinant < -1e-12:
        raise ValueError(
            "correlation matrix is not positive semi-definite "
            f"(determinant={determinant:.6g}) for rho_se={rho_se!r}, "
            f"rho_ei={rho_ei!r}, rho_si={rho_si!r}"
        )


@overload
def generate_correlated_returns(
    rho: float = SUPER_EQ_CORR,
    *,
    return_z: Literal[True],
    rng: random.Random | None = None,
    inflation: float = 0.0,
) -> tuple[float, float, float]: ...


@overload
def generate_correlated_returns(
    rho: float = SUPER_EQ_CORR,
    *,
    return_z: Literal[False] = False,
    rng: random.Random | None = None,
    inflation: float = 0.0,
) -> tuple[float, float]: ...


def generate_correlated_returns(
    rho: float = SUPER_EQ_CORR,
    *,
    return_z: bool = False,
    rng: random.Random | None = None,
    inflation: float = 0.0,
) -> tuple[float, float] | tuple[float, float, float]:
    """Generate one year of correlated lognormal equity and super returns.

    Uses Cholesky decomposition to induce the specified correlation
    between the two asset class return series. Note that ``rho`` is the
    correlation of the underlying **log-returns** (the standard normal draws),
    not of the exponentiated returns themselves.

    For a 2x2 correlation matrix [[1, rho], [rho, 1]]:
        L = [[1, 0], [rho, sqrt(1 - rho^2)]]
        x_eq    = z1
        x_super = rho * z1 + sqrt(1 - rho^2) * z2
    Then exponentiate with mean correction: exp(mu + sigma * x) - 1

    The returned returns are uplifted from real to nominal using the
    ``inflation`` parameter: ``(1 + real_gross) * (1 + inflation) - 1``.

    Args:
        rho: Correlation of the underlying equity and super **log-returns**
            (standard normal draws), in [-1, 1].
        return_z: If True, also return the equity standard normal draw ``z1``
            for use in ``generate_asset_return()``.
        rng: Optional ``random.Random`` instance for reproducible
            series generation. Defaults to module-level ``random``.
        inflation: Per-year inflation rate used to uplift real returns
            to nominal. Default 0.0 (no uplift).

    Raises:
        ValueError: If ``rho`` is outside [-1, 1].

    Returns:
        If ``return_z`` is False: (equity_return, super_return).
        If ``return_z`` is True: (equity_return, super_return, eq_z).
        All returns are decimal fractions.

    """
    _require_correlation("rho", rho)
    _rng = rng if rng is not None else random
    z1 = _rng.gauss(0, 1)
    z2 = _rng.gauss(0, 1)

    x_eq = z1
    x_super = rho * z1 + math.sqrt(1 - rho * rho) * z2

    eq_r = math.exp(MU_EQ + EQ_STD * x_eq) * (1 + inflation) - 1
    super_r = math.exp(MU_SUPER + SUPER_STD * x_super) * (1 + inflation) - 1

    if return_z:
        return eq_r, super_r, z1
    return eq_r, super_r


def generate_asset_return(
    asset_class: str,
    eq_z: float,
    eq_r: float,
    deterministic: bool = False,
    inflation: float = 0.0,
) -> float:
    """Generate a lognormal return for a given asset class, correlated with equity.

    Uses the Cholesky decomposition for 2 correlated variables where
    one is equity (already drawn as ``eq_z``) and the other is this asset class.
    The correlation is induced via:
        z_asset = rho * eq_z + sqrt(1 - rho^2) * z_independent

    When ``deterministic`` is True, the function returns the mean return
    without stochastic noise (used when running with mean returns).

    The returned return is uplifted from real to nominal using the
    ``inflation`` parameter. The unknown-class fallback returns ``eq_r``
    which is already uplifted by the caller.

    Args:
        asset_class: One of ``"equity"``, ``"bonds"``, ``"cash"``,
            ``"property"``, ``"intl_equity"``.
        eq_z: The standard normal draw used for equity this year.
        eq_r: The actual equity return (used as fallback for unknown classes).
        deterministic: If True, return the mean return without stochastic noise.
        inflation: Per-year inflation rate used to uplift real returns
            to nominal. Default 0.0 (no uplift).

    Returns:
        The asset class return as a decimal fraction.

    """
    if asset_class not in ASSET_CLASS_PARAMS:
        return eq_r  # fall back to equity for unknown classes (already uplifted)

    params = ASSET_CLASS_PARAMS[asset_class]
    rho = params["corr_with_eq"]
    mean_val = params["mean"]
    std_val = params["std"]

    # Deterministic mode: return mean return without stochastic noise
    if deterministic:
        return (1 + mean_val) * (1 + inflation) - 1

    mu = _ASSET_MU.get(asset_class, math.log(1 + mean_val) - 0.5 * std_val**2)
    z_independent = random.gauss(0, 1)
    z_asset = rho * eq_z + math.sqrt(1 - rho * rho) * z_independent
    return math.exp(mu + std_val * z_asset) * (1 + inflation) - 1


# =============================================================================
# CORRELATED TRIPLET (equity, super, inflation) — 3×3 Cholesky
# =============================================================================


def generate_correlated_triplet(
    rho_se: float = SUPER_EQ_CORR,
    rho_ei: float = INFLATION_EQ_CORR,
    rho_si: float = SUPER_INF_CORR,
    rng: random.Random | None = None,
    inflation: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Generate one year of correlated equity, super, and inflation returns.

    Uses a 3×3 Cholesky decomposition to induce correlations among equity,
    super, and inflation. Handles the general case where all three are
    pairwise correlated. Each ``rho`` is the correlation of the underlying
    **log-returns** (standard normal draws), not of the returns themselves.

    The equity and super returns are uplifted from real to nominal using the
    ``inflation`` parameter. The inflation return (``inf_r``) is not uplifted
    — it *is* the inflation.

    Args:
        rho_se: Equity-super correlation.
        rho_ei: Equity-inflation correlation.
        rho_si: Super-inflation correlation.
        rng: Optional ``random.Random`` instance for reproducible
            series generation. Defaults to module-level ``random``.
        inflation: Per-year inflation rate used to uplift real returns
            to nominal. Default 0.0 (no uplift).

    Raises:
        ValueError: If the correlations are out of range or do not form a
            positive semi-definite matrix.

    Returns:
        ((eq_r, eq_z), (super_r, super_z), inf_r) where eq_r, super_r, inf_r
        are lognormal returns and eq_z, super_z are standard normal draws.

    """
    validate_correlations(rho_se, rho_ei, rho_si)
    _rng = rng if rng is not None else random
    # Three independent standard normals
    z1 = _rng.gauss(0, 1)
    z2_raw = _rng.gauss(0, 1)
    z3_raw = _rng.gauss(0, 1)

    # Cholesky decomposition of 3x3 correlation matrix:
    # [1     rho_se rho_ei]
    # [rho_se 1     rho_si]
    # [rho_ei rho_si 1    ]
    #
    # L[0][0] = 1
    # L[1][0] = rho_se,  L[1][1] = sqrt(1 - rho_se^2)
    # L[2][0] = rho_ei
    # L[2][1] = (rho_si - rho_ei*rho_se) / sqrt(1 - rho_se^2)
    # L[2][2] = sqrt(1 - rho_ei^2 - L[2][1]^2)
    x_eq = z1
    x_super = rho_se * z1 + math.sqrt(1 - rho_se * rho_se) * z2_raw

    l21 = (rho_si - rho_ei * rho_se) / math.sqrt(1 - rho_se * rho_se)
    l22_sq = 1 - rho_ei * rho_ei - l21 * l21
    l22 = math.sqrt(l22_sq)  # PSD validated above, so l22_sq >= 0
    x_inf = rho_ei * z1 + l21 * z2_raw + l22 * z3_raw

    eq_r = math.exp(MU_EQ + EQ_STD * x_eq) * (1 + inflation) - 1
    super_r = math.exp(MU_SUPER + SUPER_STD * x_super) * (1 + inflation) - 1
    inf_r = math.exp(MU_INFLATION + INFLATION_STD * x_inf) - 1

    return (eq_r, x_eq), (super_r, x_super), inf_r


def cgt_on_gain(
    ordinary_income: float,
    gain: float,
    brackets: tuple[tuple[float, float], ...] | None = None,
) -> tuple[float, float]:
    """CGT on a capital gain stacked on top of the owner's ordinary income.

    H6: Australian CGT stacks the gain on the last dollar of ordinary income
    (s102-5 ITAA 1997 — the gain is assessable income, not a flat rate on the
    whole amount). Tax is therefore the exact tax on the slice
    ``[income, income + gain]``, obtained as ``bracket_tax(b) - bracket_tax(a)``.

    Treasury Laws Amendment (Tax Reform No. 1) Act 2026 (Cth): for disposals
    from 1 July 2027 the 50% discount is replaced by CPI indexation, and a
    per-owner 30% minimum effective rate applies — so the slice tax is floored
    at ``CGT_FLOOR_RATE * gain``.

    Args:
        ordinary_income: The owner's taxable income before the gain.
        gain: The (already indexed, post-discount) assessable gain.
        brackets: Optional bracket overrides (e.g. indexed brackets).

    Returns:
        ``(tax, effective_rate)``. ``effective_rate`` is ``tax / gain`` (0 when
        ``gain <= 0``), so callers can weight it across owners if needed.

    """
    if gain <= 0.0:
        return 0.0, 0.0
    slice_tax = bracket_tax(ordinary_income + gain, brackets) - bracket_tax(
        ordinary_income, brackets
    )
    tax = max(slice_tax, CGT_FLOOR_RATE * gain)
    return tax, tax / gain


def cgt_weighted_rate(
    owners: Sequence[tuple[float, float]],
    gain: float,
    brackets: tuple[tuple[float, float], ...] | None = None,
    discount: float = 0.0,
) -> tuple[float, float]:
    """Ownership-weighted CGT effective rate for a jointly-held gain (H6).

    Applies ``cgt_on_gain`` per owner on that owner's gain share, so each owner
    is taxed from their own income level (preserving the per-owner 30% floor),
    then weights the resulting taxes by the shares already embedded in the
    per-owner call. This replaces the previous flat
    ``max(marginal_rate(income), 0.30)`` applied to the whole gain.

    Args:
        owners: ``(ordinary_income, ownership_share)`` per owner. Shares should
            sum to 1.0; owners with a non-positive share are ignored.
        gain: The account's total nominal gain.
        brackets: Optional bracket overrides (e.g. indexed brackets).
        discount: Fraction of the gain excluded before stacking (0.5 models the
            pre-reform 50% CGT discount; 0.0 for post-reform indexed gains).

    Returns:
        ``(weighted_rate, raw_weighted_rate)`` — the floored and un-floored
        effective rates on the whole nominal gain (both 0 when ``gain <= 0``).

    """
    if gain <= 0.0:
        return 0.0, 0.0
    floored_tax = 0.0
    raw_tax = 0.0
    for income, share in owners:
        if share <= 0:
            continue
        assessable = gain * share * (1.0 - discount)
        tax, _ = cgt_on_gain(income, assessable, brackets)
        floored_tax += tax
        raw_tax += bracket_tax(income + assessable, brackets) - bracket_tax(income, brackets)
    return floored_tax / gain, raw_tax / gain


def cgt_split_tax(
    owners: Sequence[tuple[float, float]],
    pre_reform_gain: float,
    post_reform_gain: float,
) -> tuple[float, float]:
    """Ownership-weighted CGT on a disposal straddling 30 June 2027.

    Applies ``cgt_on_2027_disposal`` per owner on that owner's share of each
    portion and sums the tax. Returned as tax amounts (not rates) so the caller
    can divide by whichever gain base its sale machinery uses.

    Args:
        owners: ``(ordinary_income, ownership_share)`` per owner.
        pre_reform_gain: Nominal gain accrued to 30 June 2027 (pre-discount).
        post_reform_gain: CPI-indexed gain accrued from 30 June 2027.

    Returns:
        ``(floored_tax, tax_without_floor)``.

    """
    floored_tax = 0.0
    raw_tax = 0.0
    for income, share in owners:
        if share <= 0:
            continue
        tax, tax_without_floor, _ = cgt_on_2027_disposal(
            income,
            pre_reform_gain=pre_reform_gain * share,
            post_reform_gain=post_reform_gain * share,
        )
        floored_tax += tax
        raw_tax += tax_without_floor
    return floored_tax, raw_tax


# 30 June 2027 reform commencement, expressed as a fractional simulation year.
REFORM_DATE_YEAR: Final[float] = 2027.5


class CostBaseLot(TypedDict):
    """One acquisition-dated tranche of an asset's cost base (H5)."""

    basis: float  # cost-base amount incurred
    incurred: int  # simulation year in which the amount was incurred


def indexed_cost_base(
    lots: Sequence[CostBaseLot],
    disposal_year: int,
    cumulative_inflation: Sequence[float],
) -> float:
    """CPI-index each cost-base lot from its own incurrence date.

    s960-275 ITAA 1997 keys the indexation factor off when each amount was
    incurred. ``cumulative_inflation[y]`` is the cumulative CPI multiplier from
    simulation year 0 to year ``y`` (index 0 == 1.0), so
    ``cumulative_inflation[incurred]`` is the lot's own base index and a
    year-0 lot receives the full horizon indexation rather than none.

    Args:
        lots: Acquisition-dated cost-base tranches.
        disposal_year: Year of disposal (indexes ``cumulative_inflation``).
        cumulative_inflation: Cumulative CPI multipliers by simulation year.

    Returns:
        The indexed cost base (nominal), always >= the nominal cost base.

    """
    disposal_index = cumulative_inflation[disposal_year]
    return sum(
        lot["basis"] * disposal_index / cumulative_inflation[lot["incurred"]] for lot in lots
    )


def cgt_on_2027_disposal(
    ordinary_income: float,
    *,
    pre_reform_gain: float,
    post_reform_gain: float,
) -> tuple[float, float, float]:
    """CGT on a disposal straddling the 30 June 2027 reform line.

    Subdiv 112-E ITAA 1997 deems a disposal and reacquisition at market value on
    30 June 2027. The gain is split and each portion is taxed under its own
    regime — the two benefits are never stacked on the same portion:

    * pre-reform portion: 50% CGT discount retained, no indexation;
    * post-reform portion: CPI-indexed (caller passes the indexed gain), no
      discount, 30% minimum effective rate per owner.

    Both portions are assessable income stacked on ordinary income (H6); the
    discounted pre-reform amount is stacked first, then the post-reform amount
    on top of that.

    Args:
        ordinary_income: Owner's taxable income before the gain.
        pre_reform_gain: Nominal gain accrued to 30 June 2027 (pre-discount).
        post_reform_gain: CPI-indexed gain accrued from 30 June 2027.

    Returns:
        ``(total_tax, tax_without_floor, effective_rate)`` where
        ``tax_without_floor`` omits the 30% floor on the post-reform portion.

    """
    pre_assessable = 0.5 * max(0.0, pre_reform_gain)
    post_assessable = max(0.0, post_reform_gain)

    pre_tax = bracket_tax(ordinary_income + pre_assessable) - bracket_tax(ordinary_income)
    stacked = ordinary_income + pre_assessable
    post_slice = bracket_tax(stacked + post_assessable) - bracket_tax(stacked)

    post_tax = max(post_slice, CGT_FLOOR_RATE * post_assessable)
    total = pre_tax + post_tax
    total_without_floor = pre_tax + post_slice
    gross_gain = pre_reform_gain + post_reform_gain
    effective = total / gross_gain if gross_gain > 0 else 0.0
    return total, total_without_floor, effective


# =============================================================================
# ASSET SELL-DOWN WITH CGT
# =============================================================================


def sell_assets(
    asset: AssetHolding,
    remain: float,
    cgt_on: bool,
    weighted_marginal_rate: float = CGT_FLOOR_RATE,
    raw_marginal_rate: float | None = None,
    cumulative_inflation_factor: float = 1.0,
) -> tuple[float, float, float]:
    """Sell from an asset holding to cover a spending need, applying CGT.

    Computes the gross sale amount required to net ``remain`` after CGT,
    then mutates ``asset`` in-place to reflect the sale.

    For disposals from 1 July 2027:
      Treasury Laws Amendment (Tax Reform No. 1) Act 2026 (Cth)
      - Cost-base indexed by CPI (s 110-25 ITAA 1997 as amended) — Phase 1
      - 50% CGT discount abolished; replaced by CPI indexation — Phase 1
      - Minimum effective tax rate of 30% on real gains, per owner
        (new s 115-100) — Phase 2
      - Effective rate = max(marginal_rate, 0.30) per owner
      - Transitional (30 Jun 2027) split: ``cgt_on_2027_disposal`` implements the
        pre/post split and ``indexed_cost_base`` indexes acquisition-dated lots.
        This function still takes ONE indexation factor, so a disposal that
        straddles the reform line must be routed through those helpers once the
        caller supplies lots and the market value at 30 Jun 2027.

    ``weighted_marginal_rate`` should be the ownership-weighted average of
    max(each_owner_marginal_rate, 0.30) for jointly-held accounts,
    or simply max(owner_marginal_rate, 0.30) for single-owner
    accounts. The 30% floor must be applied at the call site
    (in ``_drawdown``), not inside this function.

    ``raw_marginal_rate`` is the un-floored marginal rate (no 30%
    minimum), used to compute the counterfactual CGT for the CGT
    breakdown display. If None, defaults to ``weighted_marginal_rate``
    (no breakdown).

    Args:
        asset: Mutable ``AssetHolding`` with ``val`` (market value)
               and ``basis`` (cost basis). Modified in-place.
        remain: After-tax cash needed from this sale.
        cgt_on: If True, apply CGT.
        weighted_marginal_rate: Ownership-weighted marginal rate
            floored at 30% (defaults to module-level ``CGT_FLOOR_RATE``
            for backward compatibility with tests).
        raw_marginal_rate: Un-floored marginal rate for without-floor
            CGT computation. If None, no counterfactual is computed.
        cumulative_inflation_factor: Cumulative inflation multiplier for
            cost-base indexation (1.0 = no indexation, 1.03 = 3% inflation).
            Defaults to 1.0 (no indexation) for backward compatibility.

    Returns:
        Tuple of (remaining_spending_need, tax_paid, tax_without_floor).
        ``tax_without_floor`` is 0.0 when ``raw_marginal_rate`` is None.

    """
    val = asset["val"]
    basis = asset["basis"]

    if val <= 0:
        return (remain, 0.0, 0.0)

    # Cost-base indexation: inflate the cost basis so only real gain is taxed
    #   Treasury Laws Amendment (Tax Reform No. 1) Act 2026 (Cth)
    indexed_basis = basis * cumulative_inflation_factor

    # Effective tax rate per dollar sold:
    #   basis_fraction of each $1 is untaxed return of capital
    #   (1 - basis_fraction) is gain, taxed at cgt_rate
    #   The gain is computed on the INDEXED basis (real gain), not nominal.
    if val > 0:
        indexed_basis_fraction = indexed_basis / val
    else:
        indexed_basis_fraction = 1.0
    gain_fraction = max(0.0, 1.0 - indexed_basis_fraction)
    effective_cgt_rate = gain_fraction * weighted_marginal_rate if cgt_on else 0.0

    if effective_cgt_rate >= 1.0:
        # Edge case: indexed basis is zero or very small -- all proceeds are gain
        gross_needed = (
            remain / (1.0 - weighted_marginal_rate)
            if cgt_on and weighted_marginal_rate < 1.0
            else remain
        )
    else:
        gross_needed = remain / (1.0 - effective_cgt_rate) if effective_cgt_rate < 1.0 else remain

    sell = min(val, gross_needed)
    if sell <= 0:
        return (remain, 0.0, 0.0)

    # Compute actual CGT on the real (indexed) gain
    fraction_sold = sell / val
    basis_consumed = basis * fraction_sold
    indexed_basis_consumed = basis_consumed * cumulative_inflation_factor
    real_gain = sell - indexed_basis_consumed
    cgt = max(0.0, real_gain * weighted_marginal_rate) if cgt_on else 0.0
    net_proceeds = sell - cgt

    # Compute CGT without the 30% floor (counterfactual)
    cgt_without_floor = 0.0
    if cgt_on and raw_marginal_rate is not None and raw_marginal_rate != weighted_marginal_rate:
        cgt_without_floor = max(0.0, real_gain * raw_marginal_rate)
    else:
        cgt_without_floor = cgt  # same rate, no floor effect

    # Update asset dict in-place (track actual cost basis, not indexed)
    asset["val"] = val - sell
    asset["basis"] = basis - basis_consumed

    new_remain = remain - net_proceeds
    return (max(0.0, new_remain), cgt, cgt_without_floor)


# =============================================================================
# STOCHASTIC MORTGAGE RATE GENERATION (Black-Karasinski)
# =============================================================================


def generate_mortgage_rate(
    prev_rate: float,
    eq_z: float = 0.0,
    theta: float = 0.065,
    kappa: float = 0.20,
    sigma_tilde: float = 0.18,
    rho: float = 0.20,
) -> float:
    """Generate a one-year mortgage rate using a Black-Karasinski model.

    The mean-reverting lognormal process uses the exact discrete AR(1)

    The continuous-time BK SDE is:
        d(ln r) = kappa * (ln theta - ln r) * dt + sigma_tilde * dW

    Exact discrete transition over dt = 1 year:
        ln(r_{t+1}) = ln(theta) + phi * (ln(r_t) - ln(theta)) + sigma_eps * z

    where:
        phi = exp(-kappa)                                   (AR(1) decay factor)
        sigma_eps = sigma_tilde * sqrt((1 - phi**2) / (2*kappa))   (innovation std dev)
        z = rho * eq_z + sqrt(1 - rho**2) * z_independent   (correlated innovation)

    Convention A (financial mathematics standard): sigma_tilde is the
    continuous-time diffusion coefficient, NOT the stationary log-rate
    std dev.  The factor sqrt(2*kappa) in sigma_eps connects this to the
    discrete-time AR(1) innovation variance.

    Each call draws an independent z_independent.  For a single-rate-
    environment scenario, precompute z_mtg once and pass it in place
    of the two-step draw inside this function.

    Args:
        prev_rate: Mortgage rate from the previous year (r_t).
            On the first call (y=0), this is the user's ``interest_rate``.
            Protected by a ``max(prev_rate, 1e-10)`` floor as a two-layer
            defence: UI validation prevents zero/negative at entry; this
            guard catches residual edge cases (extreme OU z-scores,
            deserialised legacy data, API misuse).
        eq_z: Equity standard normal draw for this year (from the
            existing Cholesky decomposition).
        theta: Long-run mean mortgage rate (decimal).  Default 0.065.
        kappa: Mean-reversion speed per year.  Default 0.20.
        sigma_tilde: Continuous-time diffusion coefficient
            (volatility per sqrt(year)).  Default 0.18.
        rho: Correlation of the mortgage rate innovation with the
            equity innovation eq_z.  Default 0.20.

    Returns:
        Mortgage rate for this year as a decimal (e.g. 0.065 for 6.5%).

    """
    phi = math.exp(-kappa)
    sigma_eps = sigma_tilde * math.sqrt((1.0 - phi * phi) / (2.0 * kappa))

    z_independent = random.gauss(0, 1)
    # Each call draws an independent z_independent.  If a household has
    # multiple stochastic mortgages, their rates will diverge over time
    # (modelling lender-specific repricing risk).
    z_mtg = rho * eq_z + math.sqrt(1.0 - rho * rho) * z_independent

    log_prev = math.log(max(prev_rate, 1e-10))
    log_rate = math.log(theta) + phi * (log_prev - math.log(theta)) + sigma_eps * z_mtg

    return math.exp(log_rate)


# =============================================================================
# MORTGAGE AMORTISATION (with offset benefit)
# =============================================================================


def amortize_mortgage_monthly(
    mortgage: float,
    offset: float,
    monthly_pmt: float,
    monthly_rate: float,
) -> tuple[float, float]:
    """Apply 12 months of mortgage amortisation with offset benefit.

    Interest is charged on effective debt (mortgage - offset, floored at 0),
    not the gross mortgage principal.

    Where the payment does not cover the interest, the shortfall is
    capitalised: the balance grows. This is the honest treatment for the
    rate-stress scenarios (M11); the previous behaviour dropped the unpaid
    interest, understating debt exactly when it mattered most.

    Args:
        mortgage: Current mortgage principal.
        offset: Current offset account balance.
        monthly_pmt: Fixed monthly payment (principal + interest).
        monthly_rate: Monthly interest rate (annual_rate / 12).

    Returns:
        Tuple of (new_mortgage, new_offset) after 12 months of payments.

    """
    m = mortgage
    o = max(0.0, offset)

    for _ in range(12):
        if m <= 0:
            break
        effective_debt = max(0.0, m - o)
        interest = effective_debt * monthly_rate
        # M11: capitalise any interest the payment does not cover, so the
        # balance grows (negative amortisation) rather than the shortfall
        # silently vanishing. A payment larger than the balance clamps to zero.
        m += interest - monthly_pmt
        if m <= 0:
            m = 0.0
            break

    return m, o


# =============================================================================
# OFFSET OVERFLOW HANDLING
# =============================================================================


def handle_offset_overflow(
    offset: float,
    mortgage: float,
    au_etfs: float,
    au_basis: float,
) -> tuple[float, float, float, float]:
    """Redirect offset balance exceeding mortgage into AU ETFs.

    When offset > mortgage, the excess earns the mortgage rate tax-free
    but could earn equity returns if deployed. This function sweeps the
    overflow into AU ETFs, updating the cost basis accordingly.

    This is a pure function conversion from the original which mutated a
    dataclass in-place. The internal logic is identical.

    Args:
        offset: Current offset account balance.
        mortgage: Current mortgage principal.
        au_etfs: Current AU ETF market value.
        au_basis: Current AU ETF cost basis.

    Returns:
        Tuple of (new_offset, new_mortgage, new_au_etfs, new_au_basis).

    """
    o = max(0.0, offset)
    m = mortgage
    e = au_etfs
    b = au_basis

    if o > m and m > 0:
        overflow = o - m
        o = m
        e += overflow
        b += overflow
    elif m <= 0 and o > 0:
        overflow = o
        o = 0.0
        e += overflow
        b += overflow

    return o, m, e, b
