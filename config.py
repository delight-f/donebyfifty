"""Configuration constants and defaults.

All simulation assumptions that are NOT part of the verified mathematical
primitives live here. No simulation, I/O, or UI code in this module.

Constants that ARE part of the verified maths live in ``primitives.py``
(``EQ_MEAN``, ``BRACKETS``, ``CGT_FLOOR_RATE``, ``MEDICARE``, etc.) and are
imported by ``simulation.py`` directly.
"""

from __future__ import annotations

from typing import Final

import primitives as _prim

# =============================================================================
# AUSTRALIAN SUPERANNUATION CONSTANTS
# =============================================================================

CONC_CAP: Final[float] = 32_500.0  # annual concessional contributions cap
SG_MAX_BASE: Final[float] = 260_000.0  # max salary base for Super Guarantee
SUPER_TAX_ON_CONTRIBUTIONS: Final[float] = 0.15  # 15% tax on contributions in fund
DIV293_THRESHOLD: Final[float] = 250_000.0  # Division 293 income threshold
DIV293_RATE: Final[float] = 0.15  # Division 293 additional tax rate

# Default indexation rates for policy parameters (used in UI warnings)
DEFAULT_CONC_CAP_GROWTH_RATE: Final[float] = 0.03  # annual growth for concessional cap
DEFAULT_DIV293_GROWTH_RATE: Final[float] = 0.025  # annual growth for Div 293 threshold

# =============================================================================
# TAX CACHE CLEARING
# =============================================================================


def clear_tax_cache() -> None:
    """Reset the part-time income tax cache between simulation runs.

    Must be called before each independent simulation run to prevent
    stale bracket-indexed entries from leaking across runs.
    """
    _prim.PT_TAX_CALC_CACHE.clear()


# =============================================================================
# SIMULATION START AGE DEFAULT
# =============================================================================

DEFAULT_SIM_START_AGE: Final[int] = 37

# =============================================================================
# FILE / PATH SETTINGS
# =============================================================================

# Convention A: all sigma values are continuous-time OU diffusion coefficients
# (volatility per sqrt(year)). The discrete AR(1) innovation std dev is:
#   sigma_eps = sigma_tilde * sqrt((1 - exp(-2*kappa)) / (2*kappa))
# NOT sigma_tilde * sqrt(1 - exp(-2*kappa)).  The sqrt(2*kappa) factor
# connects the continuous-time and discrete-time parameterisations.
#
# Moderate default: 95% steady-state interval ~[3.7%, 11.4%]
#   (contains the 2000-2024 Australian variable rate range of ~[4.1%, 9.5%])
# Stress calibration:  95% steady-state interval ~[3.0%, 14.1%]

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
