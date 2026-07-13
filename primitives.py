"""Verified mathematical primitives for the Monte Carlo retirement simulator.

All functions in this module are extracted from a model verified by a Professor
of Finance. They are preserved with zero semantic changes to the core logic.

Two functions (``amortize_mortgage_monthly``, ``handle_offset_overflow``) have
been converted from mutating a dataclass object in-place to pure functions that
accept and return float values. Their internal arithmetic is identical to the
verified original.

Return assumptions: All asset-class return parameters (module-level constants
below) represent real (inflation-adjusted) returns, consistent with the
simulation engine's internal convention — the engine compounds in real terms
and deflates dollar figures to today's dollars once at output
(``simulation.py``). Returns are applied via lognormal draws using the standard
Black-Scholes parameterisation (``mu = ln(1 + mean) - ½σ²``).
Provenance and confidence for each figure are documented per-constant below.
Not all figures are independently sourced; see individual comments for details.
"""

from __future__ import annotations

import math
import random
from typing import Final, Literal, TypedDict, overload

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class AssetHolding(TypedDict):
    """Mutable container for an asset's market value and cost basis."""

    val: float
    basis: float


# =============================================================================
# CONSTANTS (Australian financial system, not family-specific)
# =============================================================================

# Tax brackets: (threshold, rate) pairs
BRACKETS: Final[tuple[tuple[float, float], ...]] = (
    (18200, 0.00),
    (45000, 0.16),
    (135000, 0.30),
    (190000, 0.37),
    (float("inf"), 0.45),
)

MEDICARE: Final[float] = 0.02
CGT_FLOOR_RATE: Final[float] = (
    0.30  # post-2027 minimum (floor) CGT rate on real indexed gains, per owner
)
# NOTE: This is a FLOOR, not a universal rate. The actual CGT rate per owner is
# max(marginal_rate, CGT_FLOOR_RATE). For owners in the 37% or 45% bracket, their
# marginal rate applies; this floor only catches owners whose marginal rate is below 30%.
# (The 50% CGT discount was abolished post-2027; CPI indexation of cost basis replaces it.)

# =============================================================================
# Asset class return assumptions (real, inflation-adjusted)
# =============================================================================
# All returns are real (above inflation), consistent with the simulation engine's
# internal convention. Applied via lognormal draws: r = exp(mu + sigma*Z) - 1
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


def tax(
    taxable_income: float,
    medicare_surcharge: float = 0.0,
    brackets: tuple[tuple[float, float], ...] | None = None,
) -> float:
    """Calculate Australian personal income tax plus Medicare levy.

    Applies progressive marginal rates across the bracket schedule
    and adds the 2% Medicare levy plus optional Medicare Levy Surcharge
    on total taxable income.

    Args:
        taxable_income: Total taxable income for the year.
        medicare_surcharge: Additional Medicare Levy Surcharge rate
            (e.g. 0.01 for Tier 1 MLS). Added to the standard 2% levy.
        brackets: Optional bracket overrides for tax indexation.
            If None, uses the module-level ``BRACKETS``.

    Returns:
        Total tax payable (income tax + Medicare levy + MLS).

    """
    active_brackets = brackets if brackets is not None else BRACKETS
    total_tax: float = 0.0
    prev_threshold: float = 0.0

    for threshold, rate in active_brackets:
        if taxable_income > threshold:
            total_tax += (threshold - prev_threshold) * rate
            prev_threshold = threshold
        else:
            total_tax += (taxable_income - prev_threshold) * rate
            break

    return total_tax + taxable_income * (MEDICARE + medicare_surcharge)


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
) -> float:
    """Compute the Medicare Levy Surcharge rate for a given taxable income.

    Selects singles or couple tiers based on ``n_earners`` (≥ 2 → couple).
    Returns 0.0 for income below the lowest threshold.

    Args:
        taxable_income: Annual taxable income (combined for couples).
        n_earners: Number of earners in the household (1 = singles tiers).
        tiers_single: Override singles tier thresholds.
        tiers_couple: Override couple tier thresholds.

    Returns:
        MLS rate as a decimal (e.g. 0.0125 for Tier 2).

    """
    tiers = MLS_TIERS_COUPLE if n_earners >= 2 else MLS_TIERS_SINGLE
    for threshold, rate in tiers:
        if taxable_income <= threshold:
            return rate
    return tiers[-1][1] if tiers else 0.0


# =============================================================================
# MEDICARE LEVY SURCHARGE TIERS (as of FY2025-26)
# =============================================================================

# Singles thresholds and rates
MLS_TIERS_SINGLE: Final[tuple[tuple[float, float], ...]] = (
    (90000.0, 0.0),  # Below $90k: no MLS
    (105000.0, 0.01),  # $90k-$105k: Tier 1 (1.0%)
    (140000.0, 0.0125),  # $105k-$140k: Tier 2 (1.25%)
    (float("inf"), 0.015),  # $140k+: Tier 3 (1.5%)
)

# Couple thresholds and rates
# Note: MLS applies based on combined income for couples/families
MLS_TIERS_COUPLE: Final[tuple[tuple[float, float], ...]] = (
    (180000.0, 0.0),  # Below $180k: no MLS
    (210000.0, 0.01),  # $180k-$210k: Tier 1 (1.0%)
    (280000.0, 0.0125),  # $210k-$280k: Tier 2 (1.25%)
    (float("inf"), 0.015),  # $280k+: Tier 3 (1.5%)
)


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


@overload
def generate_correlated_returns(
    rho: float = SUPER_EQ_CORR,
    *,
    return_z: Literal[True],
    rng: random.Random | None = None,
) -> tuple[float, float, float]: ...


@overload
def generate_correlated_returns(
    rho: float = SUPER_EQ_CORR,
    *,
    return_z: Literal[False] = False,
    rng: random.Random | None = None,
) -> tuple[float, float]: ...


def generate_correlated_returns(
    rho: float = SUPER_EQ_CORR,
    *,
    return_z: bool = False,
    rng: random.Random | None = None,
) -> tuple[float, float] | tuple[float, float, float]:
    """Generate one year of correlated lognormal equity and super returns.

    Uses Cholesky decomposition to induce the specified correlation
    between the two asset class return series.

    For a 2x2 correlation matrix [[1, rho], [rho, 1]]:
        L = [[1, 0], [rho, sqrt(1 - rho^2)]]
        x_eq    = z1
        x_super = rho * z1 + sqrt(1 - rho^2) * z2
    Then exponentiate with mean correction: exp(mu + sigma * x) - 1

    Args:
        rho: Target Pearson correlation between equity and super returns.
        return_z: If True, also return the equity standard normal draw ``z1``
            for use in ``generate_asset_return()``.
        rng: Optional ``random.Random`` instance for reproducible
            series generation. Defaults to module-level ``random``.

    Returns:
        If ``return_z`` is False: (equity_return, super_return).
        If ``return_z`` is True: (equity_return, super_return, eq_z).
        All returns are decimal fractions.

    """
    _rng = rng if rng is not None else random
    z1 = _rng.gauss(0, 1)
    z2 = _rng.gauss(0, 1)

    x_eq = z1
    x_super = rho * z1 + math.sqrt(1 - rho * rho) * z2

    eq_r = math.exp(MU_EQ + EQ_STD * x_eq) - 1
    super_r = math.exp(MU_SUPER + SUPER_STD * x_super) - 1

    if return_z:
        return eq_r, super_r, z1
    return eq_r, super_r


def generate_asset_return(
    asset_class: str, eq_z: float, eq_r: float, deterministic: bool = False
) -> float:
    """Generate a lognormal return for a given asset class, correlated with equity.

    Uses the Cholesky decomposition for 2 correlated variables where
    one is equity (already drawn as ``eq_z``) and the other is this asset class.
    The correlation is induced via:
        z_asset = rho * eq_z + sqrt(1 - rho^2) * z_independent

    When ``deterministic`` is True, the function returns the mean return
    without stochastic noise (used when running with mean returns).

    Args:
        asset_class: One of ``"equity"``, ``"bonds"``, ``"cash"``,
            ``"property"``, ``"intl_equity"``.
        eq_z: The standard normal draw used for equity this year.
        eq_r: The actual equity return (used as fallback for unknown classes).
        deterministic: If True, return the mean return without stochastic noise.

    Returns:
        The asset class return as a decimal fraction.

    """
    if asset_class not in ASSET_CLASS_PARAMS:
        return eq_r  # fall back to equity for unknown classes

    params = ASSET_CLASS_PARAMS[asset_class]
    rho = params["corr_with_eq"]
    mean_val = params["mean"]
    std_val = params["std"]

    # Deterministic mode: return mean return without stochastic noise
    if deterministic:
        return mean_val

    mu = _ASSET_MU.get(asset_class, math.log(1 + mean_val) - 0.5 * std_val**2)
    z_independent = random.gauss(0, 1)
    z_asset = rho * eq_z + math.sqrt(1 - rho * rho) * z_independent
    return math.exp(mu + std_val * z_asset) - 1


# =============================================================================
# CORRELATED TRIPLET (equity, super, inflation) — 3×3 Cholesky
# =============================================================================


def generate_correlated_triplet(
    rho_se: float = SUPER_EQ_CORR,
    rho_ei: float = INFLATION_EQ_CORR,
    rho_si: float = SUPER_INF_CORR,
    rng: random.Random | None = None,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Generate one year of correlated equity, super, and inflation returns.

    Uses a 3×3 Cholesky decomposition to induce correlations among equity,
    super, and inflation. Handles the general case where all three are
    pairwise correlated.

    Args:
        rho_se: Equity-super correlation.
        rho_ei: Equity-inflation correlation.
        rho_si: Super-inflation correlation.
        rng: Optional ``random.Random`` instance for reproducible
            series generation. Defaults to module-level ``random``.

    Returns:
        ((eq_r, eq_z), (super_r, super_z), inf_r) where eq_r, super_r, inf_r
        are lognormal returns and eq_z, super_z are standard normal draws.

    """
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
    l22 = math.sqrt(max(0.0, l22_sq))  # guard against tiny numerical negatives
    x_inf = rho_ei * z1 + l21 * z2_raw + l22 * z3_raw

    eq_r = math.exp(MU_EQ + EQ_STD * x_eq) - 1
    super_r = math.exp(MU_SUPER + SUPER_STD * x_super) - 1
    inf_r = math.exp(MU_INFLATION + INFLATION_STD * x_inf) - 1

    return (eq_r, x_eq), (super_r, x_super), inf_r


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
      - Transitional treatment for gains accrued pre-1/7/2027 — Phase 3,
        deferred, not yet implemented

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

    This is a pure function conversion from the original which mutated a
    dataclass in-place. The internal arithmetic is identical.

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
        principal_payment = monthly_pmt - interest
        if principal_payment < 0:
            principal_payment = 0.0
        elif principal_payment > m:
            principal_payment = m
        m -= principal_payment
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
