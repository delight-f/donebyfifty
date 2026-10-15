"""Capital gains and investment-income tax on the household tax ledger.

Vectorised over trials (D22): every array has a leading trial axis ``T``. Accounts
are addressed by integer column index, owners by ``O`` columns.

Design (fix-plan D11-D14):

* Every income slice (employment, investment income, realised gains) is taxed as
  the increment of the full ``tax()`` on each owner's running assessable income
  (D12). Medicare and MLS use the household's running family income.
* Cost bases are per-account cost arrays indexed by the year they were incurred
  (``Ledger``). Every cost increase is recorded with ``add_cost``. At the 1 July
  2027 reform the holding becomes one lot at its market value plus a frozen
  pre-reform gain (D13).
* Selling a fraction of a holding realises a pre-reform gain (50% discount, no
  floor) and a post-reform gain (CPI-indexed cost base, no discount, 30%
  per-owner minimum rate).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .returns import COMPANY_TAX_RATE
from .tax import FloatArray, tax

# Minimum effective rate on post-reform gains, per owner (D13).
CGT_FLOOR_RATE: Final[float] = 0.30
# Discount on pre-reform gains (taxable share of a discountable gain).
CGT_DISCOUNT_TAXABLE_SHARE: Final[float] = 0.5
# Fixed bisection count for the gross-up. Each step halves the bracket, so after
# 37 steps the bracket is 2**-37 of the holding: at most ~$0.007 for a $1e9 holding.
SALE_ITERATIONS: Final[int] = 37


@dataclass
class Ledger:
    """Cost bases for every trial and account, and the frozen pre-reform gain (D13).

    ``cost[t, a, y]`` is the cost base of account ``a`` in trial ``t`` incurred in
    year index ``y`` (index 0 is the opening basis). ``pre_reform_gain[t, a]`` is the
    nominal gain accrued to the reform, valid once ``snapped`` is True.
    """

    cost: NDArray[np.float64]  # (T, A, Y+1)
    pre_reform_gain: NDArray[np.float64]  # (T, A)
    snapped: bool = False

    @classmethod
    def empty(cls, n_trials: int, n_accounts: int, n_years: int) -> Ledger:
        """Ledger with no cost base and no pre-reform gain."""
        return cls(
            cost=np.zeros((n_trials, n_accounts, n_years + 1)),
            pre_reform_gain=np.zeros((n_trials, n_accounts)),
        )

    def take(self, rows: NDArray[np.intp]) -> Ledger:
        """Ledger restricted to the trial rows ``rows``, as copies."""
        return Ledger(self.cost[rows], self.pre_reform_gain[rows], self.snapped)


def cost_total(ledger: Ledger) -> FloatArray:
    """Total nominal cost base per trial and account, shape (T, A)."""
    return ledger.cost.sum(axis=2)


def add_cost(ledger: Ledger, account: int, amount: FloatArray, year: int) -> None:
    """Record a cost increase (deposit, reinvested income, overflow) incurred in ``year``.

    Non-positive amounts are ignored.
    """
    ledger.cost[:, account, year] += np.maximum(amount, 0.0)


def reform_snapshot(ledger: Ledger, market_values: FloatArray, year: int) -> None:
    """Freeze the pre-reform gain and reset every account's cost base to market value.

    ``market_values`` has shape (T, A). The reset cost base is booked in ``year``.
    """
    ledger.pre_reform_gain = market_values - cost_total(ledger)
    ledger.cost[...] = 0.0
    ledger.cost[:, :, year] = np.maximum(market_values, 0.0)
    ledger.snapped = True


def scale(ledger: Ledger, account: int, keep: FloatArray) -> None:
    """Scale an account's cost bases and pre-reform gain by ``keep`` (proportional sale)."""
    ledger.cost[:, account, :] *= keep[:, np.newaxis]
    ledger.pre_reform_gain[:, account] *= keep


def realised_gains(
    ledger: Ledger,
    account: int,
    market_value: FloatArray,
    fraction: FloatArray,
    year: int,
    price_index: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Nominal gains realised by selling ``fraction`` of the holding.

    Returns ``(discountable, post_reform)``, each shape (T,). Before the reform
    snapshot the whole gain is discountable. After it, the discountable part is the
    frozen pre-reform gain and the post-reform gain is value less the CPI-indexed
    cost base. ``price_index`` has shape (T, Y+1).
    """
    if not ledger.snapped:
        return fraction * (market_value - cost_total(ledger)[:, account]), np.zeros_like(
            market_value
        )
    discountable = fraction * ledger.pre_reform_gain[:, account]
    indexed_cost = (
        ledger.cost[:, account, :] * price_index[:, year, np.newaxis] / price_index
    ).sum(axis=1)
    return discountable, fraction * (market_value - indexed_cost)


@dataclass(frozen=True)
class TaxContext:
    """Year-level tax settings shared by every income slice in the year."""

    brackets: tuple[tuple[float, float], ...]
    dependants: int
    # Two or more earners.
    couple: bool
    # MLS rate at a family MLS income (taxable plus reportable super contributions).
    surcharge: Callable[[FloatArray], FloatArray]

    @property
    def family_test(self) -> bool:
        """Whether the Medicare family reduction applies (couples or dependants)."""
        return self.couple or self.dependants > 0


@dataclass
class Positions:
    """Per-trial, per-owner running position for the year (D12) and carried losses.

    ``assessable[t, o]`` is owner ``o``'s taxable income so far this year in trial
    ``t``; ``mls_income[t, o]`` is the income the Medicare/MLS family test uses;
    ``loss_pool[t, o]`` is the owner's unapplied capital loss carried forward.
    """

    assessable: FloatArray  # (T, O)
    mls_income: FloatArray  # (T, O)
    loss_pool: FloatArray  # (T, O)

    @classmethod
    def zero(cls, n_trials: int, n_owners: int) -> Positions:
        """Positions with nothing assessed and no losses."""
        return cls(
            np.zeros((n_trials, n_owners)),
            np.zeros((n_trials, n_owners)),
            np.zeros((n_trials, n_owners)),
        )

    def begin_year(self, assessable: FloatArray, mls_income: FloatArray) -> None:
        """Reset this year's assessable and MLS income; carried losses are kept."""
        self.assessable[...] = assessable
        self.mls_income[...] = mls_income

    def take(self, rows: NDArray[np.intp]) -> Positions:
        """Positions for the trial rows ``rows``, as copies."""
        return Positions(self.assessable[rows], self.mls_income[rows], self.loss_pool[rows])


def owner_tax(
    ctx: TaxContext, taxable: FloatArray, family_taxable: FloatArray, family_mls: FloatArray
) -> FloatArray:
    """Full income tax with Medicare and MLS for one owner, per trial.

    The Medicare family reduction uses family taxable income and applies to
    couples and to households with dependants (sole parents). Other singles get
    the individual reduction only. MLS is tested on family MLS income (taxable
    income plus reportable super contributions), which is a separate total.

    Inputs may stack several states on a leading axis; ``ctx.surcharge`` must then be
    elementwise.
    """
    return tax(
        taxable,
        brackets=ctx.brackets,
        medicare_surcharge=ctx.surcharge(family_mls),
        family_income=family_taxable if ctx.family_test else None,
        dependants=ctx.dependants,
    )


def stack_income(
    ctx: TaxContext, positions: Positions, owner: int, amount: FloatArray
) -> FloatArray:
    """Stack ``amount`` of income on ``owner``'s position; return the tax increment per trial.

    The increment is ``owner_tax(a + s) - owner_tax(a)``, with the family taxable
    and family MLS running totals both including the slice (D12). Updates
    ``positions`` in place.
    """
    base = positions.assessable[:, owner]
    family_taxable = positions.assessable.sum(axis=1)
    family_mls = positions.mls_income.sum(axis=1)
    increment = owner_tax(
        ctx, base + amount, family_taxable + amount, family_mls + amount
    ) - owner_tax(ctx, base, family_taxable, family_mls)
    positions.assessable[:, owner] = base + amount
    positions.mls_income[:, owner] += amount
    return increment


@dataclass(frozen=True)
class SaleTax:
    """CGT on one sale, per trial.

    ``tax`` includes the post-reform floor; ``tax_without_floor`` does not.
    """

    tax: FloatArray
    tax_without_floor: FloatArray


# Owner columns after a sale: (owner, assessable, mls_income, loss_pool).
_OwnerColumns = tuple[int, FloatArray, FloatArray, FloatArray]


def _sell(
    ctx: TaxContext,
    positions: Positions,
    ownership: Sequence[tuple[int, float]],
    discountable: FloatArray,
    post_reform: FloatArray,
) -> tuple[SaleTax, list[_OwnerColumns]]:
    """CGT for a sale realising ``discountable`` and ``post_reform`` gains, per trial.

    The gains are already scaled by the fraction sold. Losses and stacked amounts
    depend only on the gains, so every owner's tax is evaluated in one batch. Per
    owner there are three states: before the discountable slice, after it, and
    after the post-reform slice. Each state carries that owner's taxable income and
    the family totals as they stand at that point in the sequence. ``positions`` is
    read but not mutated; the updated owner columns are returned for the caller to
    commit.
    """
    family_taxable = positions.assessable.sum(axis=1)
    family_mls = positions.mls_income.sum(axis=1)
    states_taxable: list[FloatArray] = []
    states_family_taxable: list[FloatArray] = []
    states_family_mls: list[FloatArray] = []
    post_gains: list[FloatArray] = []
    columns: list[_OwnerColumns] = []
    for owner, share in ownership:
        base = positions.assessable[:, owner]
        disc = share * discountable
        post_gain = share * post_reform
        pool = positions.loss_pool[:, owner] + np.maximum(-post_gain, 0.0)
        pool = pool + np.maximum(-disc, 0.0)
        post_gain = np.maximum(post_gain, 0.0)
        disc = np.maximum(disc, 0.0)
        applied = np.minimum(pool, post_gain)
        post_gain = post_gain - applied
        pool = pool - applied
        applied = np.minimum(pool, disc)
        disc = disc - applied
        pool = pool - applied

        discount_amount = CGT_DISCOUNT_TAXABLE_SHARE * disc
        taxable_mid = base + discount_amount
        taxable_end = taxable_mid + post_gain
        family_taxable_mid = family_taxable + discount_amount
        family_taxable_end = family_taxable_mid + post_gain
        family_mls_mid = family_mls + discount_amount
        family_mls_end = family_mls_mid + post_gain
        states_taxable += [base, taxable_mid, taxable_end]
        states_family_taxable += [family_taxable, family_taxable_mid, family_taxable_end]
        states_family_mls += [family_mls, family_mls_mid, family_mls_end]
        post_gains.append(post_gain)
        mls_owner = positions.mls_income[:, owner] + discount_amount + post_gain
        columns.append((owner, taxable_end, mls_owner, pool))
        family_taxable = family_taxable_end
        family_mls = family_mls_end

    values = owner_tax(
        ctx, np.stack(states_taxable), np.stack(states_family_taxable), np.stack(states_family_mls)
    )
    payable = np.zeros_like(discountable)
    without_floor = np.zeros_like(discountable)
    for i, post_gain in enumerate(post_gains):
        before, after_discount, after_post = values[3 * i], values[3 * i + 1], values[3 * i + 2]
        discount_tax = after_discount - before
        post_tax = after_post - after_discount
        payable = payable + discount_tax + np.maximum(post_tax, CGT_FLOOR_RATE * post_gain)
        without_floor = without_floor + discount_tax + post_tax
    return SaleTax(payable, without_floor), columns


def _commit(positions: Positions, columns: Sequence[_OwnerColumns]) -> None:
    """Write the owner columns returned by ``_sell`` into ``positions``."""
    for owner, assessable, mls_income, loss_pool in columns:
        positions.assessable[:, owner] = assessable
        positions.mls_income[:, owner] = mls_income
        positions.loss_pool[:, owner] = loss_pool


def sale_tax(
    ctx: TaxContext,
    positions: Positions,
    ledger: Ledger,
    account: int,
    ownership: Sequence[tuple[int, float]],
    market_value: FloatArray,
    fraction: FloatArray,
    year: int,
    price_index: FloatArray,
) -> SaleTax:
    """CGT on selling ``fraction`` of a holding, per trial. Commits to ``positions``.

    Each owner's share of the discountable and post-reform gains is stacked on
    that owner's position. Losses for that owner are applied to post-reform
    gains first, then to discountable gains before the discount. The 30% floor
    applies to each owner's post-reform gain only.
    """
    discountable, post = realised_gains(ledger, account, market_value, fraction, year, price_index)
    sold, columns = _sell(ctx, positions, ownership, discountable, post)
    _commit(positions, columns)
    return sold


def solve_sale(
    ctx: TaxContext,
    positions: Positions,
    ledger: Ledger,
    account: int,
    ownership: Sequence[tuple[int, float]],
    market_value: FloatArray,
    need: FloatArray,
    year: int,
    price_index: FloatArray,
) -> tuple[FloatArray, SaleTax]:
    """Fraction of a holding to sell so that proceeds net of CGT cover ``need``, per trial.

    Bisects on the fraction in [0, 1] for ``SALE_ITERATIONS`` fixed steps. Trials
    with ``market_value <= 0`` or ``need <= 0`` get fraction 0 and zero tax. Trials
    where selling everything still leaves net proceeds at or below ``need`` sell
    the whole holding (fraction 1). Only the chosen sale is committed to
    ``positions``.
    """
    fraction = np.zeros_like(market_value)
    tax_payable = np.zeros_like(market_value)
    tax_without_floor = np.zeros_like(market_value)
    # Only trials with a holding and a need can sell; the rest keep fraction 0 and zero tax.
    rows = np.flatnonzero((market_value > 0.0) & (need > 0.0))
    if rows.size == 0:
        return fraction, SaleTax(tax_payable, tax_without_floor)
    sub_value = market_value[rows]
    sub_need = need[rows]
    sub_prices = price_index[rows]
    sub_ledger = ledger.take(rows)
    # Candidate sales are evaluated on read-only subset positions.
    sub_positions = positions.take(rows)
    discountable, post = realised_gains(
        sub_ledger, account, sub_value, np.ones_like(sub_value), year, sub_prices
    )

    def net(candidate: FloatArray) -> FloatArray:
        sold, _ = _sell(ctx, sub_positions, ownership, candidate * discountable, candidate * post)
        return candidate * sub_value - sold.tax

    bisect = net(np.ones_like(sub_value)) > sub_need
    lo = np.zeros_like(sub_value)
    hi = np.ones_like(sub_value)
    for _ in range(SALE_ITERATIONS):
        mid = 0.5 * (lo + hi)
        below = net(mid) < sub_need
        lo = np.where(bisect & below, mid, lo)
        hi = np.where(bisect & ~below, mid, hi)
    sub_sold = sale_tax(
        ctx, sub_positions, sub_ledger, account, ownership, sub_value, hi, year, sub_prices
    )
    positions.assessable[rows] = sub_positions.assessable
    positions.mls_income[rows] = sub_positions.mls_income
    positions.loss_pool[rows] = sub_positions.loss_pool
    fraction[rows] = hi
    tax_payable[rows] = sub_sold.tax
    tax_without_floor[rows] = sub_sold.tax_without_floor
    return fraction, SaleTax(tax_payable, tax_without_floor)


def investment_income_tax(
    ctx: TaxContext,
    positions: Positions,
    ownership: Sequence[tuple[int, float]],
    unfranked: FloatArray,
    franked: FloatArray,
) -> FloatArray:
    """Net tax on one account's cash distribution per trial (D14). May be negative.

    Franked distributions are grossed up by ``1 / (1 - COMPANY_TAX_RATE)``. The
    franking credit is ``franked * COMPANY_TAX_RATE / (1 - COMPANY_TAX_RATE)`` and
    is refundable, so a low-income owner's net tax can be negative. Each owner's
    share is stacked on that owner's position.
    """
    gross_up = 1.0 / (1.0 - COMPANY_TAX_RATE)
    grossed = unfranked + franked * gross_up
    credit = franked * (gross_up - 1.0)
    total = np.zeros_like(grossed)
    for owner, share in ownership:
        total = total + stack_income(ctx, positions, owner, share * grossed) - share * credit
    return total
