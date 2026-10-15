"""Drawdown waterfall: funds a shortfall from offsets, sales, then cash.

Vectorised over trials: every trial follows the same waterfall, with per-trial
amounts. Accounts are sold in the household's ``sell_order`` when one is given,
otherwise per trial from the smallest unrealised gain fraction up.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .cgt import TaxContext, cost_total, scale, solve_sale
from .models import Household, SimulationInputs
from .state import FloatArray, _BatchState, account_ownership

_NO_GAIN_KEY = 1e9


def _offset_floor(state: _BatchState, household: Household, account_index: int) -> FloatArray:
    """Per-trial balance an offset account must keep, given its linked mortgages."""
    label = household.investment_accounts[account_index].label
    floor: FloatArray = np.zeros(state.account_values.shape[0])
    for mi, mortgage in enumerate(household.mortgages):
        if label not in mortgage.offset_accounts:
            continue
        principal = state.mortgage_principals[:, mi]
        if mortgage.offset_reserve_mode == "stall_prevention" and mortgage.monthly_payment > 0:
            monthly_rate = state.current_mortgage_rates[:, mi] / 12.0
            positive = monthly_rate > 0.0
            safe_rate = np.where(positive, monthly_rate, 1.0)
            dynamic = np.where(
                positive,
                np.maximum(0.0, principal - mortgage.monthly_payment / safe_rate),
                0.0,
            )
            floor = np.maximum(floor, dynamic)
        elif mortgage.offset_reserve_mode == "interest_cancelling":
            floor = np.maximum(floor, principal)
        else:
            floor = np.maximum(floor, mortgage.offset_reserve_floor)
    return floor


def _static_sell_order(
    household: Household, inputs: SimulationInputs, non_offset_idxs: list[int]
) -> list[int] | None:
    """Return the labelled sell order, with unlisted accounts appended; None if unset."""
    if not inputs.sell_order:
        return None
    ordered: list[int] = []
    for label in inputs.sell_order:
        for ai in non_offset_idxs:
            if household.investment_accounts[ai].label == label:
                ordered.append(ai)
                break
    for ai in non_offset_idxs:
        if ai not in ordered:
            ordered.append(ai)
    return ordered


def _gain_fraction_keys(state: _BatchState, non_offset_idxs: list[int]) -> FloatArray:
    """Per-trial unrealised gain fraction of each non-offset account (T, N).

    Accounts with no positive value sort last.
    """
    value = state.account_values[:, non_offset_idxs]
    cost = cost_total(state.ledger)[:, non_offset_idxs]
    positive = value > 0.0
    safe_value = np.where(positive, value, 1.0)
    return np.where(positive, (value - cost) / safe_value, _NO_GAIN_KEY)


def _sell_order(
    state: _BatchState, household: Household, inputs: SimulationInputs, non_offset_idxs: list[int]
) -> NDArray[np.intp]:
    """Return the (T, N) account index sold at each rank for each trial."""
    n_trials = state.account_values.shape[0]
    static = _static_sell_order(household, inputs, non_offset_idxs)
    if static is not None:
        return np.broadcast_to(np.asarray(static, dtype=np.intp), (n_trials, len(static)))
    positions = np.argsort(_gain_fraction_keys(state, non_offset_idxs), axis=1, kind="stable")
    return np.asarray(non_offset_idxs, dtype=np.intp)[positions]


def _drawdown(
    state: _BatchState,
    household: Household,
    inputs: SimulationInputs,
    ctx: TaxContext,
    remain: FloatArray,
    year: int,
    offset_idxs: list[int],
    non_offset_idxs: list[int],
) -> FloatArray:
    """Fund ``remain`` (T,) from offsets, then sales, then cash; return the unfunded part."""
    remain = remain.copy()

    # 1. Offset accounts, above their reserve floor.
    for ai in offset_idxs:
        value = state.account_values[:, ai]
        available = np.maximum(0.0, value - _offset_floor(state, household, ai))
        draw = np.where(remain > 0.0, np.minimum(available, remain), 0.0)
        state.account_values[:, ai] = value - draw
        state.year_offset_drawn = state.year_offset_drawn + draw
        remain = remain - draw

    # 2. Sell non-offset accounts in rank order, realising CGT when enabled.
    order = _sell_order(state, household, inputs, non_offset_idxs)
    for rank in range(order.shape[1]):
        rank_account = order[:, rank]
        for ai in non_offset_idxs:
            value = state.account_values[:, ai]
            mask = (rank_account == ai) & (remain > 0.0) & (value > 1.0)
            if not mask.any():
                continue
            if inputs.cgt_on_drawdowns:
                need = np.where(mask, remain, 0.0)
                fraction, sold = solve_sale(
                    ctx,
                    state.positions,
                    state.ledger,
                    ai,
                    account_ownership(household.investment_accounts[ai]),
                    value,
                    need,
                    year,
                    state.price_index,
                )
                tax = sold.tax
                tax_without_floor = sold.tax_without_floor
            else:
                denominator = np.where(mask, value, 1.0)
                fraction = np.where(mask, np.minimum(1.0, remain / denominator), 0.0)
                tax = np.zeros_like(remain)
                tax_without_floor = tax
            proceeds = fraction * value - tax
            state.account_values[:, ai] = value * (1.0 - fraction)
            scale(state.ledger, ai, 1.0 - fraction)
            remain = remain - proceeds
            state.year_non_offset_drawn = state.year_non_offset_drawn + proceeds
            state.year_cgt_paid = state.year_cgt_paid + tax
            state.year_cgt_without_floor = state.year_cgt_without_floor + tax_without_floor

    # 3. Cash that is still available.
    cash_draw = np.where(
        (remain > 0.0) & (state.cash_balance > 0.0),
        np.minimum(state.cash_balance, remain),
        0.0,
    )
    state.cash_balance = state.cash_balance - cash_draw
    return remain - cash_draw
