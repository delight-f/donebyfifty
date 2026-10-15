"""Mortgage rate process, monthly amortisation and offset overflow handling.

Every function accepts scalars or arrays (broadcast together) and returns
float64 arrays. The batch engine calls them with one entry per trial.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

# =============================================================================
# STOCHASTIC MORTGAGE RATE GENERATION (Black-Karasinski)
# =============================================================================


def generate_mortgage_rate(
    prev_rate: ArrayLike,
    z: ArrayLike,
    *,
    theta: float,
    kappa: float,
    sigma_tilde: float,
) -> NDArray[np.float64]:
    """Generate a one-year mortgage rate using a Black-Karasinski model.

    The continuous-time BK SDE is:
        d(ln r) = kappa * (ln mu - ln r) * dt + sigma_tilde * dW

    Exact discrete transition over dt = 1 year, with ln mu the stationary
    log-mean:
        ln(r_{t+1}) = ln(mu) + phi * (ln(r_t) - ln(mu)) + sigma_eps * z

    where:
        phi = exp(-kappa)                                          (AR(1) decay)
        sigma_eps = sigma_tilde * sqrt((1 - phi**2) / (2*kappa))   (innovation std dev)
        ln mu = ln(theta) - sigma_tilde**2 / (4*kappa)

    The log-mean correction makes ``theta`` the stationary *arithmetic* mean of
    the rate: the stationary log-rate has variance sigma_tilde**2 / (2*kappa),
    so E[r] = exp(ln mu + variance / 2) = theta.

    The caller supplies the standard normal innovation ``z`` (for example,
    ``rho * eq_z + sqrt(1 - rho**2) * z_independent`` for a rate correlated with
    equity). Each call uses the ``z`` given, so the caller controls correlation
    and independence across mortgages.

    Args:
        prev_rate: Mortgage rate from the previous year (r_t), scalar or per trial.
            On the first call (y=0), this is the user's ``interest_rate``.
            Protected by a ``max(prev_rate, 1e-10)`` floor as a two-layer
            defence: UI validation prevents zero/negative at entry; this
            guard catches residual edge cases (extreme z-scores,
            deserialised legacy data, API misuse).
        z: Standard normal innovation for this year, scalar or per trial.
        theta: Long-run arithmetic mean mortgage rate (decimal, e.g. 0.065).
        kappa: Mean-reversion speed per year. Must be positive.
        sigma_tilde: Continuous-time diffusion coefficient
            (volatility per sqrt(year)).

    Returns:
        Mortgage rate for this year as a decimal (e.g. 0.065 for 6.5%).

    Raises:
        ValueError: If ``kappa`` is not positive.

    """
    if kappa <= 0.0:
        raise ValueError(f"kappa={kappa!r} must be positive (mean-reversion speed)")
    phi = math.exp(-kappa)
    sigma_eps = sigma_tilde * math.sqrt((1.0 - phi * phi) / (2.0 * kappa))
    log_mean = math.log(theta) - sigma_tilde * sigma_tilde / (4.0 * kappa)

    log_prev = np.log(np.maximum(np.asarray(prev_rate, dtype=np.float64), 1e-10))
    log_rate = log_mean + phi * (log_prev - log_mean) + sigma_eps * np.asarray(z, dtype=np.float64)

    return np.asarray(np.exp(log_rate), dtype=np.float64)


# =============================================================================
# MORTGAGE AMORTISATION (with offset benefit)
# =============================================================================


def pi_payment(principal: ArrayLike, annual_rate: ArrayLike, months: int) -> NDArray[np.float64]:
    """Level monthly principal-and-interest payment for a fully amortising loan.

    Args:
        principal: Loan balance at the start of the amortisation (scalar or per trial).
        annual_rate: Annual nominal interest rate (decimal, e.g. 0.06), scalar or per trial.
        months: Number of monthly payments remaining. Must be positive.

    Returns:
        Monthly payment that repays ``principal`` exactly over ``months``.
        Zero-rate loans are repaid in equal principal instalments.

    Raises:
        ValueError: If ``months`` is not positive.

    """
    if months <= 0:
        raise ValueError(f"months={months!r} must be positive")
    p = np.asarray(principal, dtype=np.float64)
    r = np.asarray(annual_rate, dtype=np.float64) / 12.0
    zero = r == 0.0
    safe_r = np.where(zero, 1.0, r)
    annuity = p * safe_r / (1.0 - (1.0 + safe_r) ** -months)
    return np.asarray(np.where(zero, p / months, annuity), dtype=np.float64)


def amortize_mortgage_monthly(
    mortgage: ArrayLike,
    offset: ArrayLike,
    monthly_pmt: ArrayLike,
    monthly_rate: ArrayLike,
    *,
    interest_only: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Apply 12 months of mortgage repayments with offset benefit.

    Interest is charged monthly on effective debt (mortgage - offset, floored at
    0), not the gross mortgage principal.

    Interest-only (``interest_only=True``): principal is unchanged; the full
    monthly interest is paid in cash and ``monthly_pmt`` is ignored.

    Principal and interest (``interest_only=False``): each month's payment covers
    interest first, and the remainder reduces principal. Repayments never exceed
    the outstanding balance. If the payment does not cover the interest, only the
    payment is paid in cash and the shortfall is capitalised onto the balance
    (negative amortisation, M11), so debt is not understated in rate-stress
    scenarios.

    Args:
        mortgage: Current mortgage principal (scalar or per trial).
        offset: Current offset account balance.
        monthly_pmt: Fixed monthly principal-and-interest payment. Ignored when
            ``interest_only`` is True.
        monthly_rate: Monthly interest rate (annual_rate / 12).
        interest_only: True for an interest-only loan.

    Returns:
        Tuple of ``(new_principal, interest_paid, principal_paid)`` over the 12
        months. ``interest_paid`` and ``principal_paid`` are cash amounts
        leaving the household. Capitalised interest is included in
        ``new_principal`` only.

    """
    shape = np.broadcast_shapes(
        np.shape(mortgage), np.shape(offset), np.shape(monthly_pmt), np.shape(monthly_rate)
    )
    m = np.array(np.broadcast_to(np.asarray(mortgage, dtype=np.float64), shape))
    o = np.broadcast_to(np.maximum(0.0, np.asarray(offset, dtype=np.float64)), shape)
    pmt = np.broadcast_to(np.asarray(monthly_pmt, dtype=np.float64), shape)
    rate = np.broadcast_to(np.asarray(monthly_rate, dtype=np.float64), shape)
    interest_paid = np.zeros(shape, dtype=np.float64)
    principal_paid = np.zeros(shape, dtype=np.float64)

    for _ in range(12):
        active = m > 0.0
        interest = np.maximum(0.0, m - o) * rate
        if interest_only:
            interest_paid += np.where(active, interest, 0.0)
            continue
        short = pmt < interest
        interest_paid += np.where(active, np.where(short, pmt, interest), 0.0)
        repayment = np.where(active & ~short, np.minimum(pmt - interest, m), 0.0)
        principal_paid += repayment
        m = m - repayment + np.where(active & short, interest - pmt, 0.0)

    return m, interest_paid, principal_paid


# =============================================================================
# OFFSET OVERFLOW HANDLING
# =============================================================================


def handle_offset_overflow(
    offset: ArrayLike,
    mortgage: ArrayLike,
    au_etfs: ArrayLike,
    au_basis: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Redirect offset balance exceeding mortgage into AU ETFs.

    When offset > mortgage, the excess earns the mortgage rate tax-free
    but could earn equity returns if deployed. This function sweeps the
    overflow into AU ETFs, updating the cost basis accordingly.

    Args:
        offset: Current offset account balance (scalar or per trial).
        mortgage: Current mortgage principal.
        au_etfs: Current AU ETF market value.
        au_basis: Current AU ETF cost basis.

    Returns:
        Tuple of (new_offset, new_mortgage, new_au_etfs, new_au_basis).

    """
    o = np.maximum(0.0, np.asarray(offset, dtype=np.float64))
    m = np.asarray(mortgage, dtype=np.float64)
    e = np.asarray(au_etfs, dtype=np.float64)
    b = np.asarray(au_basis, dtype=np.float64)

    # Offset above a positive mortgage spills the excess; a repaid mortgage
    # sends the whole offset balance across.
    above = (o > m) & (m > 0.0)
    repaid = (m <= 0.0) & (o > 0.0)
    overflow = np.where(above, o - m, np.where(repaid, o, 0.0))
    new_o = np.where(above, m, np.where(repaid, 0.0, o))
    new_e = e + overflow
    new_b = b + overflow
    return (
        np.asarray(new_o, dtype=np.float64),
        np.asarray(m, dtype=np.float64),
        np.asarray(new_e, dtype=np.float64),
        np.asarray(new_b, dtype=np.float64),
    )
