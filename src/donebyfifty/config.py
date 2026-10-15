"""Configuration constants and defaults.

Simulation assumptions that are not part of the verified tax, returns, mortgage
or CGT kernels live here. EDU_SCHEDULE_TODAY is the education cost schedule.
"""

from __future__ import annotations

from typing import (
    Final,
)

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

# Default education schedule as sorted (age, annual cost) pairs. The engine reads an empty
# Child.education_schedule as no costs, so the default is stored as this table, not as ().
DEFAULT_EDU_SCHEDULE: Final[tuple[tuple[int, float], ...]] = tuple(
    sorted(EDU_SCHEDULE_TODAY.items())
)

# =============================================================================
# AUSTRALIAN SUPERANNUATION CONSTANTS
# =============================================================================

CONC_CAP: Final[float] = 32_500.0  # annual concessional contributions cap
SG_MAX_BASE: Final[float] = 270_830.0  # max salary base for Super Guarantee (FY2026-27)
SUPER_TAX_ON_CONTRIBUTIONS: Final[float] = 0.15  # 15% tax on contributions in fund
DIV293_THRESHOLD: Final[float] = 250_000.0  # Division 293 income threshold
DIV293_RATE: Final[float] = 0.15  # Division 293 additional tax rate

# Default indexation rates for policy parameters (used in UI warnings)
DEFAULT_CONC_CAP_GROWTH_RATE: Final[float] = 0.03  # annual growth for concessional cap


# MLS tiers live in tax.py (single source of truth for the Medicare Levy Surcharge).


# =============================================================================
# SIMULATION START AGE DEFAULT
# =============================================================================

DEFAULT_SIM_START_AGE: Final[int] = 37

# =============================================================================
# MORTGAGE RATE (BLACK-KARASINSKI) PARAMETERS
# =============================================================================

# Convention A: all sigma values are continuous-time OU diffusion coefficients
# (volatility per sqrt(year)). The discrete AR(1) innovation std dev is:
#   sigma_eps = sigma_tilde * sqrt((1 - exp(-2*kappa)) / (2*kappa))
# NOT sigma_tilde * sqrt(1 - exp(-2*kappa)).  The sqrt(2*kappa) factor
# connects the continuous-time and discrete-time parameterisations.
#
# Mean-centred: BK_THETA is the stationary arithmetic mean of the rate, so the
# stationary log-rate variance is sigma**2 / (2*kappa). Stationary 95% intervals:
#   Moderate (sigma 0.18): ~[3.6%, 10.9%]
#     (contains the 2000-2024 Australian variable rate range of ~[4.1%, 9.5%])
#   Stress (sigma 0.25):   ~[2.8%, 13.1%]

BK_SIGMA_MODERATE: Final[float] = 0.18  # continuous-time diffusion coeff (sigma_tilde)
BK_SIGMA_STRESS: Final[float] = 0.25  # stress calibration (wider dispersion)
BK_KAPPA: Final[float] = 0.20  # mean-reversion speed (half-life ~3.5 yr)
BK_THETA: Final[float] = 0.065  # long-run mean mortgage rate (6.5%)
BK_RHO: Final[float] = 0.20  # correlation with equity return innovation

# =============================================================================
# UI THEME
# =============================================================================

THEME_COLOR: Final[str] = "green"
THEME_COLOR_BRIGHT: Final[str] = "bright_green"
THEME_COLOR_WARN: Final[str] = "yellow"
THEME_COLOR_ERROR: Final[str] = "red"
THEME_COLOR_ACCENT: Final[str] = "cyan"
