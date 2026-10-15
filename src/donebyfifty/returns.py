"""Asset-class return assumptions and pure return transforms.

Return parameters are real (inflation-adjusted) geometric (compound) figures:
``*_GEO_MEAN`` is the compound annual real return, the median path of the
lognormal process is ``exp(mu) - 1 = *_GEO_MEAN``, and the arithmetic mean of
the simulated returns is ``exp(mu + sigma**2 / 2) - 1``. Provenance and
confidence for each figure are documented beside its constant.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Investment income (D14). Yields are on opening account value and are reinvested
# as new cost base. Equity and property growth is total return less yield; bonds
# and cash have no separate yield and their whole positive nominal return is income.
# Australian equity distributions are 75% franked at the company tax rate.
INCOME_YIELD: Final[dict[str, float]] = {
    "equity": 0.040,
    "intl_equity": 0.020,
    "property": 0.035,
}
FRANKED_FRACTION: Final[float] = 0.75
COMPANY_TAX_RATE: Final[float] = 0.30

# =============================================================================
# Asset class return assumptions (real, geometric, inflation-adjusted)
# Each asset's real log-return is Normal(mu, sigma) with mu = ln(1 + geo_mean)
# (geometric parameterisation, H7/D8). sigma is log-return volatility.
# Provenance and confidence are documented per-constant below.
# =============================================================================

# --- Equity (Australian) --------------------------------------------------------
# Basis: real (inflation-adjusted), geometric (compound annual) return.
# Source: Credit Suisse / UBS Global Investment Returns Yearbook (with London
#   Business School), Australian equities annualised real return since 1900:
#   ~6.4-6.7% (edition-dependent; figure varies slightly by edition year and
#   should not be cited as a single precise value). The midpoint, 6.5%, is used.
# Confidence: sourced (unconditional long-run historical; not valuation-adjusted).
# Caveats: This is a 120+ year unconditional average, not a forward-looking
#   forecast. Several major houses (Vanguard VCMM, Fidelity capital markets
#   assumptions) currently forecast below-historical-average equity returns for
#   the next decade due to elevated valuations. Data pre-1980 uses inconsistent
#   methodology across sources (Market Index, 2024) and should be treated as
#   indicative rather than precise. If precision matters more than defensibility
#   for the model's horizon (10-20 years), consider a valuation-conditioned
#   alternative instead.
EQ_GEO_MEAN: Final[float] = 0.065

# Confidence: unsourced-placeholder. No citation found for this figure.
EQ_STD: Final[float] = 0.15

# --- Super (equity-like) --------------------------------------------------------
# Basis: real (inflation-adjusted), geometric, equity-like by assumption.
# Source: UNSOURCED. Currently set equal to the equity geometric mean by
#   assumption (no independent source for superannuation-specific returns).
# Confidence: unsourced-placeholder. A defensible source would be a specific
#   super fund's published PDS investment-option return assumptions (e.g.
#   balanced or growth option), not yet retrieved. The correlation to equity
#   (SUPER_EQ_CORR = 0.80) is also an unsourced placeholder.
# Caveats: Super funds are not identical to a pure equity index — they hold
#   multi-asset portfolios with fees that reduce net returns. Treating super as
#   "equity minus a discount" (lower mean, lower std, imperfect correlation) is
#   a reasonable modelling choice but the specific numbers are unsourced.
SUPER_GEO_MEAN: Final[float] = 0.065

# Confidence: unsourced-placeholder. No citation found for this figure.
SUPER_STD: Final[float] = 0.12

# Confidence: unsourced-placeholder. Target correlation between super and equity.
SUPER_EQ_CORR: Final[float] = 0.80

# --- Bonds ---------------------------------------------------------------------
# Basis: real (inflation-adjusted), geometric.
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
BOND_GEO_MEAN: Final[float] = 0.03

# Confidence: unsourced-placeholder. No citation found for this figure.
BOND_STD: Final[float] = 0.06

# --- Cash ----------------------------------------------------------------------
# Basis: real (inflation-adjusted), geometric.
# Source: Same RBA source as Bonds — Fraser (1991) treats short-term / fixed-
#   interest real returns as roughly 1-2% real, without cleanly separating cash
#   from short bonds.
# Confidence: derived-estimate. No cash-specific historical series was found.
#   The current 2.5% figure sits at the upper edge of the 1-2% band the RBA
#   source implies.
# Caveats: Same vintage caveat as Bonds. A more recent cash-rate series (e.g.
#   RBA cash rate target deflated by CPI) would strengthen this estimate.
CASH_GEO_MEAN: Final[float] = 0.025

# Confidence: unsourced-placeholder. No citation found for this figure.
CASH_STD: Final[float] = 0.01

# --- Property (Australian residential) -----------------------------------------
# Basis: real (inflation-adjusted), geometric.
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
PROPERTY_GEO_MEAN: Final[float] = 0.05

# Confidence: unsourced-placeholder. No citation found for this figure.
PROPERTY_STD: Final[float] = 0.12

# --- Intl Equity ---------------------------------------------------------------
# Basis: real (inflation-adjusted), geometric.
# Source: UNSOURCED. No citation found.
# Confidence: unsourced-placeholder.
# Caveats: International equity returns are not independent of Australian equity
#   returns — the model captures this via INTL_EQ_CORR (also unsourced). A
#   defensible source would be MSCI World ex-Australia or equivalent, not yet
#   retrieved. Set 0.5 percentage points below the Australian equity figure.
INTL_EQ_GEO_MEAN: Final[float] = 0.060

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
    "equity": {"geo_mean": EQ_GEO_MEAN, "std": EQ_STD, "corr_with_eq": 1.0},
    "bonds": {"geo_mean": BOND_GEO_MEAN, "std": BOND_STD, "corr_with_eq": BOND_EQ_CORR},
    "cash": {"geo_mean": CASH_GEO_MEAN, "std": CASH_STD, "corr_with_eq": CASH_EQ_CORR},
    "property": {
        "geo_mean": PROPERTY_GEO_MEAN,
        "std": PROPERTY_STD,
        "corr_with_eq": PROPERTY_EQ_CORR,
    },
    "intl_equity": {
        "geo_mean": INTL_EQ_GEO_MEAN,
        "std": INTL_EQ_STD,
        "corr_with_eq": INTL_EQ_CORR,
    },
}

# Log-mean mu = ln(1 + geo_mean): the median of the lognormal real growth factor.
MU_EQ: Final[float] = math.log(1 + EQ_GEO_MEAN)
MU_SUPER: Final[float] = math.log(1 + SUPER_GEO_MEAN)
MU_BOND: Final[float] = math.log(1 + BOND_GEO_MEAN)
MU_CASH: Final[float] = math.log(1 + CASH_GEO_MEAN)
MU_PROPERTY: Final[float] = math.log(1 + PROPERTY_GEO_MEAN)
MU_INTL_EQ: Final[float] = math.log(1 + INTL_EQ_GEO_MEAN)

# Log-mean map for asset class lookup (used in asset_real_return)
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
INFLATION_STD: Final[float] = 0.015
# Inflation is centred on the arithmetic mean ``inputs.inflation``. The log-mean
# is ln(1 + inflation_mean) - sigma**2 / 2 so that E[exp(mu + sigma*Z)] equals
# 1 + inflation_mean. This half-variance correction is the only one applied in
# this module; the asset-class figures are geometric and have none.
INFLATION_EQ_CORR: Final[float] = (
    -0.15
)  # mild negative correlation: high inflation often coincides with poor equity returns
# =============================================================================
# PURE RETURN TRANSFORMS
# No randomness here: the Monte Carlo draw generator (montecarlo.py) supplies
# standard-normal shocks and these functions map them to returns.
# =============================================================================


def lognormal_real_return(geo_mean: float, std: float, z: ArrayLike) -> NDArray[np.float64]:
    """Real return of a lognormal asset at standard-normal shock ``z``.

    With ``mu = ln(1 + geo_mean)`` the return is ``exp(mu + std * z) - 1``, so
    ``z = 0`` gives the median path, which equals ``geo_mean``.
    """
    return np.asarray((1.0 + geo_mean) * np.exp(std * np.asarray(z, dtype=np.float64)) - 1.0)


def asset_real_return(asset_class: str, z: ArrayLike) -> NDArray[np.float64]:
    """Real return of a standard asset class (``ASSET_CLASS_PARAMS``) at shock ``z``."""
    params = ASSET_CLASS_PARAMS[asset_class]
    return lognormal_real_return(params["geo_mean"], params["std"], z)


def inflation_rate(mean: float, z: ArrayLike) -> NDArray[np.float64]:
    """Annual inflation at shock ``z``: lognormal with arithmetic mean ``mean``.

    The log-mean ``ln(1 + mean) - INFLATION_STD**2 / 2`` makes
    ``E[1 + inflation] = 1 + mean``.
    """
    mu = math.log(1.0 + mean) - 0.5 * INFLATION_STD**2
    return np.asarray(np.exp(mu + INFLATION_STD * np.asarray(z, dtype=np.float64)) - 1.0)


def nominal_return(real: ArrayLike, inflation: ArrayLike) -> NDArray[np.float64]:
    """Convert a real return to nominal: ``(1 + real) * (1 + inflation) - 1``."""
    return np.asarray(
        (1.0 + np.asarray(real, dtype=np.float64)) * (1.0 + np.asarray(inflation, dtype=np.float64))
        - 1.0
    )
