"""Configuration constants, defaults, and input validation ranges.

All simulation assumptions that are NOT part of the verified mathematical
primitives live here. No simulation, I/O, or UI code in this module.

Constants that ARE part of the verified maths live in ``primitives.py``
(``EQ_MEAN``, ``BRACKETS``, ``CGT_FLOOR_RATE``, ``MEDICARE``, etc.) and are
imported by ``simulation.py`` directly.
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# GENERAL SIMULATION DEFAULTS
# =============================================================================

DEFAULT_N_ITERATIONS: Final[int] = 5_000
DEFAULT_INFLATION: Final[float] = 0.025
DEFAULT_SIM_START_AGE: Final[int] = 37
DEFAULT_SIM_END_AGE: Final[int] = 72  # ignored in bridge mode; simulation ends at min(super_access_age)
DEFAULT_CGT_ON_DRAWDOWNS: Final[bool] = True
DEFAULT_SELL_STRATEGY: Final[str] = "waterfall"

# =============================================================================
# AUSTRALIAN SUPERANNUATION CONSTANTS
# =============================================================================

CONC_CAP: Final[float] = 32_500.0  # annual concessional contributions cap
SG_MAX_BASE: Final[float] = 260_000.0  # max salary base for Super Guarantee
SG_RATE: Final[float] = 0.12  # Super Guarantee percentage (12% from July 2025)
SUPER_TAX_ON_CONTRIBUTIONS: Final[float] = 0.15  # 15% tax on contributions in fund
DIV293_THRESHOLD: Final[float] = 250_000.0  # Division 293 income threshold
DIV293_RATE: Final[float] = 0.15  # Division 293 additional tax rate

# Default indexation rates for policy parameters (used in UI warnings)
DEFAULT_CONC_CAP_GROWTH_RATE: Final[float] = 0.03  # annual growth for concessional cap
DEFAULT_DIV293_GROWTH_RATE: Final[float] = 0.025  # annual growth for Div 293 threshold

# =============================================================================
# TAX CACHE CLEARING
# =============================================================================

PT_TAX_CALC_CACHE: dict[tuple[float, tuple], float] = {}
"""Shared cache for part-time tax calculations (cleared between runs)."""

# Import primitives at module level — no circular dependency:
#   config → primitives → (nothing from config)
#   simulation → config, simulation → primitives
import primitives as _prim


def clear_tax_cache() -> None:
    """Reset the part-time income tax cache between simulation runs.

    Clears both the local cache and the primitives module's cache.
    Must be called before each independent simulation run to prevent
    stale bracket-indexed entries from leaking across runs.
    """
    PT_TAX_CALC_CACHE.clear()
    _prim.PT_TAX_CALC_CACHE.clear()


# =============================================================================
# EDUCATION SCHEDULE PRESETS
# =============================================================================

EDUCATION_PRESETS: Final[dict[str, tuple[tuple[int, float], ...]]] = {
    "none": (),
    "public_primary": (
        (5, 500.0),
        (6, 500.0),
        (7, 500.0),
        (8, 500.0),
        (9, 500.0),
        (10, 500.0),
        (11, 500.0),
        (12, 800.0),
        (13, 800.0),
        (14, 800.0),
        (15, 800.0),
        (16, 800.0),
        (17, 800.0),
    ),
    "private_primary": (
        (5, 5_000.0),
        (6, 5_000.0),
        (7, 5_000.0),
        (8, 5_000.0),
        (9, 5_000.0),
        (10, 5_000.0),
        (11, 5_000.0),
        (12, 15_000.0),
        (13, 15_000.0),
        (14, 15_000.0),
        (15, 15_000.0),
        (16, 15_000.0),
        (17, 15_000.0),
    ),
}

# =============================================================================
# INPUT VALIDATION RANGES
# =============================================================================

VALIDATION: Final[dict[str, tuple[float, float]]] = {
    "n_iterations": (100, 100_000),
    "inflation": (0.0, 0.10),
    "simulation_start_age": (18, 65),
    "simulation_end_age": (50, 100),
    # Per-earner
    "salary": (0.0, 5_000_000.0),
    "super_balance": (0.0, 10_000_000.0),
    "salary_growth_rate": (-0.05, 0.15),
    "retirement_age": (30, 75),
    "super_access_age": (55, 70),
    "sg_rate": (0.0, 0.20),
    "pt_days_per_week": (0.0, 7.0),
    "pt_daily_rate": (0.0, 20_000.0),
    # Child
    "child_age": (0, 25),
    # Mortgage
    "mortgage_principal": (0.0, 10_000_000.0),
    "mortgage_rate": (0.01, 0.15),
    "monthly_payment": (0.0, 100_000.0),
    # Investment account
    "market_value": (0.0, 10_000_000.0),
    "cost_basis": (0.0, 10_000_000.0),
    "cgt_rate": (0.0, 0.50),
    "conc_cap_growth_rate": (0.0, 0.10),
    "sg_max_base_growth_rate": (0.0, 0.10),
    "super_fee_rate": (0.0, 0.05),
    "mls_threshold": (0.0, 500_000.0),
    "mls_rate": (0.0, 0.05),
    "bracket_growth_rate": (0.0, 0.10),
    "div293_threshold": (0.0, 1_000_000.0),
    "div293_rate": (0.0, 0.30),
    "div293_growth_rate": (0.0, 0.10),
    # Household
    "living_expenses": (10_000.0, 1_000_000.0),
    "retirement_target": (10_000.0, 1_000_000.0),
}

# =============================================================================
# FILE / PATH SETTINGS
# =============================================================================

PROFILES_DIR: Final[str] = "profiles"

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
BK_SIGMA_STRESS: Final[float] = 0.25    # stress calibration (wider dispersion)
BK_KAPPA: Final[float] = 0.20          # mean-reversion speed (half-life ~3.5 yr)
BK_THETA: Final[float] = 0.065         # long-run mean mortgage rate (6.5%)
BK_RHO: Final[float] = 0.20            # correlation with equity return innovation

# =============================================================================
# UI THEME
# =============================================================================

THEME_COLOR: Final[str] = "green"
THEME_COLOR_BRIGHT: Final[str] = "bright_green"
THEME_COLOR_WARN: Final[str] = "yellow"
THEME_COLOR_ERROR: Final[str] = "red"
THEME_COLOR_ACCENT: Final[str] = "cyan"
