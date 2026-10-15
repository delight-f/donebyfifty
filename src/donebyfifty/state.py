"""Batch simulation state, per-trial shock batches and batch trial results.

Every array has a leading trial axis ``T``. ``run_single_trial`` runs the same
batch engine with ``T == 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .cgt import Ledger, Positions, add_cost
from .models import Household, InvestmentAccount, SimulationInputs
from .returns import ASSET_CLASS_PARAMS

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

# Column order of the asset-class shock arrays (``DrawsBatch.class_z``).
CLASS_NAMES: tuple[str, ...] = tuple(ASSET_CLASS_PARAMS)
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


# =============================================================================
# UTILITY: FIND ACCOUNT INDICES
# =============================================================================


def _offset_indices(household: Household) -> list[int]:
    """Return indices of offset accounts in the investment_accounts list."""
    return [i for i, a in enumerate(household.investment_accounts) if a.is_offset]


def _non_offset_indices(household: Household) -> list[int]:
    """Return indices of non-offset, sellable accounts."""
    return [i for i, a in enumerate(household.investment_accounts) if not a.is_offset]


def _num_offset_accounts(household: Household) -> int:
    """Return the number of offset accounts."""
    return sum(1 for a in household.investment_accounts if a.is_offset)


def account_ownership(account: InvestmentAccount) -> list[tuple[int, float]]:
    """Return ``(owner index, share)`` pairs for owners with a positive share."""
    return [(ei, share) for ei, share in account.ownership.items() if share > 0]


# =============================================================================
# INTERNAL STATE
# =============================================================================


@dataclass
class _BatchState:
    """Mutable state for ``T`` trials simulated together.

    Parallel arrays indexed to match the ``Household`` tuples: earners (O),
    mortgages (M) and investment accounts (A).
    """

    price_index: FloatArray  # (T, Y + 1) price level path, price_index[:, 0] == 1
    super_balances: FloatArray  # (T, O)
    # Per-owner tax position for the current year (D12); carried losses persist.
    positions: Positions
    # Investment-income tax assessed in the previous year, paid this year (D14).
    tax_payable_carry: FloatArray  # (T,)
    mortgage_principals: FloatArray  # (T, M)
    # Mortgage rates for the current year (decimal). Start at each mortgage's
    # ``interest_rate``; stochastic mortgages are advanced each year.
    current_mortgage_rates: FloatArray  # (T, M)
    # Monthly P&I payment per mortgage, reset when a repricing applies.
    mortgage_payments: FloatArray  # (T, M)
    # Rate each mortgage's current payment was set at.
    payment_rates: FloatArray  # (T, M)
    account_values: FloatArray  # (T, A)
    # Per-account cost-base lots (D13).
    ledger: Ledger
    # Implicit cash balance when no investment account can take a surplus.
    cash_balance: FloatArray  # (T,)
    # Cumulative unmet spending once bridge assets are exhausted.
    unmet_spending: FloatArray  # (T,)
    # Drawdown composition for the current year, reset each year (in year-start
    # nominal dollars).
    year_offset_drawn: FloatArray  # (T,)
    year_non_offset_drawn: FloatArray  # (T,)
    year_cgt_paid: FloatArray  # (T,)
    year_cgt_without_floor: FloatArray  # (T,)
    year_income_tax: FloatArray  # (T,)

    @property
    def total_bridge(self) -> FloatArray:
        """Sum of accessible bridge assets (offset + taxable accounts + cash), per trial."""
        return self.account_values.sum(axis=1) + self.cash_balance


def init_batch_state(
    household: Household,
    n_trials: int,
    n_years: int,
    price_index: FloatArray,
) -> _BatchState:
    """Create the opening state for ``n_trials`` trials of ``household``.

    Each account's opening cost basis is recorded as a lot at year 0.
    """
    n_owners = len(household.earners)
    n_accounts = len(household.investment_accounts)
    ledger = Ledger.empty(n_trials, n_accounts, n_years)
    for ai, account in enumerate(household.investment_accounts):
        add_cost(ledger, ai, np.full(n_trials, account.cost_basis), 0)
    return _BatchState(
        price_index=price_index,
        super_balances=np.tile(
            np.array([e.super_balance for e in household.earners], dtype=np.float64),
            (n_trials, 1),
        ).reshape(n_trials, n_owners),
        positions=Positions.zero(n_trials, n_owners),
        tax_payable_carry=np.zeros(n_trials),
        mortgage_principals=np.tile(
            np.array([m.principal for m in household.mortgages], dtype=np.float64),
            (n_trials, 1),
        ).reshape(n_trials, len(household.mortgages)),
        current_mortgage_rates=np.tile(
            np.array([m.interest_rate for m in household.mortgages], dtype=np.float64),
            (n_trials, 1),
        ).reshape(n_trials, len(household.mortgages)),
        mortgage_payments=np.tile(
            np.array([m.monthly_payment for m in household.mortgages], dtype=np.float64),
            (n_trials, 1),
        ).reshape(n_trials, len(household.mortgages)),
        payment_rates=np.tile(
            np.array([m.interest_rate for m in household.mortgages], dtype=np.float64),
            (n_trials, 1),
        ).reshape(n_trials, len(household.mortgages)),
        account_values=np.tile(
            np.array([a.market_value for a in household.investment_accounts], dtype=np.float64),
            (n_trials, 1),
        ).reshape(n_trials, n_accounts),
        ledger=ledger,
        cash_balance=np.zeros(n_trials),
        unmet_spending=np.zeros(n_trials),
        year_offset_drawn=np.zeros(n_trials),
        year_non_offset_drawn=np.zeros(n_trials),
        year_cgt_paid=np.zeros(n_trials),
        year_cgt_without_floor=np.zeros(n_trials),
        year_income_tax=np.zeros(n_trials),
    )


def horizon_years(household: Household, inputs: SimulationInputs) -> int:
    """Simulation years until the earliest super access age across earners."""
    return min(
        e.effective_super_access_age - e.resolved_start_age(inputs.simulation_start_age)
        for e in household.earners
    )


# =============================================================================
# PER-TRIAL RANDOM SHOCKS
# =============================================================================


@dataclass(frozen=True, eq=False)
class DrawsBatch:
    """Pre-generated standard-normal shocks for ``T`` Monte Carlo trials.

    Attributes:
        eq_z: Equity shock, shape (T, Y). Other correlated shocks derive from it.
        inflation: Realised annual inflation rate, shape (T, Y) (not a shock).
        class_z: Shock per standard asset class, shape (T, Y, C), columns in
            ``CLASS_NAMES`` order.
        super_override_z: Super override shock per earner, shape (T, O, Y).
        mortgage_z: Mortgage-rate shock per mortgage, shape (T, M, Y).

    """

    eq_z: FloatArray
    inflation: FloatArray
    class_z: FloatArray
    super_override_z: FloatArray
    mortgage_z: FloatArray

    @property
    def n_trials(self) -> int:
        """Number of trials in the batch."""
        return int(self.eq_z.shape[0])

    @property
    def n_years(self) -> int:
        """Number of simulated years per trial."""
        return int(self.eq_z.shape[1])

    def trial(self, i: int) -> DrawsBatch:
        """Return trial ``i`` as a batch of one (the shape ``run_single_trial`` uses)."""
        return DrawsBatch(
            eq_z=self.eq_z[i : i + 1],
            inflation=self.inflation[i : i + 1],
            class_z=self.class_z[i : i + 1],
            super_override_z=self.super_override_z[i : i + 1],
            mortgage_z=self.mortgage_z[i : i + 1],
        )

    def reordered(self, order: IntArray) -> DrawsBatch:
        """Return a copy with every year of each trial permuted by ``order`` (T, Y).

        Used by sequencing analysis so that all shocks for a year move together.
        """
        year_order = order[:, None, :]
        return DrawsBatch(
            eq_z=np.take_along_axis(self.eq_z, order, axis=1),
            inflation=np.take_along_axis(self.inflation, order, axis=1),
            class_z=np.take_along_axis(self.class_z, order[:, :, None], axis=1),
            super_override_z=np.take_along_axis(self.super_override_z, year_order, axis=2),
            mortgage_z=np.take_along_axis(self.mortgage_z, year_order, axis=2),
        )

    @classmethod
    def median(
        cls,
        *,
        n_trials: int = 1,
        n_years: int,
        n_earners: int,
        n_mortgages: int,
        inflation: float,
    ) -> DrawsBatch:
        """Zero-shock draws: every return at its median, inflation held constant.

        Used when ``run_single_trial`` is called without draws.
        """
        return cls(
            eq_z=np.zeros((n_trials, n_years)),
            inflation=np.full((n_trials, n_years), inflation),
            class_z=np.zeros((n_trials, n_years, len(CLASS_NAMES))),
            super_override_z=np.zeros((n_trials, n_earners, n_years)),
            mortgage_z=np.zeros((n_trials, n_mortgages, n_years)),
        )


# =============================================================================
# BATCH RESULT
# =============================================================================


@dataclass(frozen=True, eq=False)
class TrialBatchResult:
    """Per-trial results for a batch of ``T`` trials.

    A trial fails in a year when spending cannot be met from available
    resources. ``first_failure_age`` records the first such year, ``-1`` if
    none. Money in trajectories and totals is in year-start dollars; balances
    are in real end-of-year dollars.
    """

    bridge: FloatArray  # (T,) terminal bridge, nominal less unmet and carried tax
    super_balances: FloatArray  # (T, O) at the horizon
    mortgage_principals: FloatArray  # (T, M) at the horizon
    horizon_age: int
    term_checkable: BoolArray  # (M,) mortgage term ends within the horizon
    term_cleared: BoolArray  # (T, M) principal repaid by the horizon
    min_bridge: FloatArray  # (T,) net bridge in the floor year (nominal)
    unmet_spending: FloatArray  # (T,)
    floor_age: IntArray  # (T,)
    first_failure_age: IntArray  # (T,), -1 if the trial never failed
    bridge_by_age: FloatArray  # (T, Y) real net bridge at each year end
    mortgage_by_age: FloatArray  # (T, Y, M)
    offset_by_age: FloatArray  # (T, Y, N_offset)
    mortgage_rate_by_age: FloatArray  # (T, Y, M)
    offset_drawn_by_age: FloatArray  # (T, Y)
    non_offset_drawn_by_age: FloatArray  # (T, Y)
    cgt_paid_by_age: FloatArray  # (T, Y)
    cgt_without_floor_by_age: FloatArray  # (T, Y)
    income_tax_by_age: FloatArray  # (T, Y)
    floor_deflator: FloatArray  # (T,)
    horizon_deflator: FloatArray  # (T,)
    total_offset_drawn: FloatArray  # (T,)
    total_non_offset_drawn: FloatArray  # (T,)
    total_cgt_paid: FloatArray  # (T,)
    total_cgt_without_floor: FloatArray  # (T,)
    total_income_tax: FloatArray  # (T,)

    @property
    def n_trials(self) -> int:
        """Number of trials in the batch."""
        return int(self.bridge.shape[0])

    @property
    def succeeded(self) -> BoolArray:
        """(T,) True where no year in the trial failed to meet spending."""
        return self.first_failure_age < 0

    @property
    def total_super(self) -> FloatArray:
        """(T,) Total super balance at the horizon age."""
        return self.super_balances.sum(axis=1)

    @property
    def total_mortgage(self) -> FloatArray:
        """(T,) Total mortgage principal at the horizon age."""
        return self.mortgage_principals.sum(axis=1)
