"""Batch simulation engine: every trial of a run advances through each year together.

Each year runs in cash-flow order: half-year growth, part-time and salary income
with tax, expenses and mortgage repayments, surplus investment or drawdown,
interest-only interest, then offset overflow, then the second half-year of growth
and fees. Shocks come from ``DrawsBatch``; the engine draws no random numbers.
``run_single_trial`` is the ``T == 1`` case of the same engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .cgt import TaxContext, add_cost, cost_total, investment_income_tax, owner_tax, reform_snapshot
from .config import (
    BK_KAPPA,
    BK_SIGMA_MODERATE,
    BK_THETA,
    CONC_CAP,
    SG_MAX_BASE,
)
from .config import SUPER_TAX_ON_CONTRIBUTIONS as SUPER_TAX_RATE
from .drawdown import _drawdown
from .models import Earner, Household, InvestmentAccount, SimulationInputs, validate_horizon
from .mortgage import (
    amortize_mortgage_monthly,
    generate_mortgage_rate,
    handle_offset_overflow,
    pi_payment,
)
from .returns import (
    ASSET_CLASS_PARAMS,
    FRANKED_FRACTION,
    INCOME_YIELD,
    SUPER_STD,
    asset_real_return,
    lognormal_real_return,
    nominal_return,
)
from .state import (
    CLASS_INDEX,
    DrawsBatch,
    FloatArray,
    TrialBatchResult,
    _BatchState,
    _non_offset_indices,
    _offset_indices,
    account_ownership,
    horizon_years,
    init_batch_state,
)
from .tax import PT_WEEKS_PER_YEAR, indexed_brackets, mls_rate

# Drawdown shortfalls at or below this many dollars are floating-point residue from
# the drawdown arithmetic, not unmet spending.
UNMET_TOLERANCE = 1e-6


# =============================================================================
# PER-YEAR SHOCKS
# =============================================================================


@dataclass(frozen=True, eq=False)
class _YearShocks:
    """The shocks and price level for one simulated year, for every trial."""

    y: int
    age: int
    price: FloatArray  # (T,) price level at the start of the year
    inflation: FloatArray  # (T,) inflation realised during the year
    eq_z: FloatArray  # (T,)
    class_z: FloatArray  # (T, C), columns in CLASS_NAMES order
    super_z: FloatArray  # (T, O)
    mortgage_z: FloatArray  # (T, M)
    earner_ages: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class _Earning:
    """One working earner's income and super for the year, per trial."""

    taxable: FloatArray  # gross less salary sacrifice
    gross: FloatArray  # salary plus self-employed income, for the MLS test
    sg: FloatArray  # employer super guarantee
    sacrifice: FloatArray  # personal concessional contributions


def _columns(columns: list[FloatArray], n_trials: int) -> FloatArray:
    """Stack per-owner (T,) arrays into a (T, len(columns)) array."""
    if not columns:
        return np.zeros((n_trials, 0))
    return np.stack(columns, axis=1)


# =============================================================================
# GROWTH
# =============================================================================


def _super_growth_pct(earner: Earner, y: int) -> float:
    """Equity share (percent) of an earner's super in simulation year ``y``.

    Glides linearly, in simulation years, from ``super_growth_pct`` at year 0 to
    ``super_glide_target_pct`` at ``super_glide_end_year``, then holds the target.
    """
    end = earner.super_glide_end_year
    if end is None:
        return earner.super_growth_pct
    if y >= end:
        return earner.super_glide_target_pct
    start = earner.super_growth_pct
    return start + (earner.super_glide_target_pct - start) * (y / end)


def _super_real_return(earner: Earner, index: int, year: _YearShocks) -> FloatArray:
    """Real annual return on one earner's super for the year, per trial."""
    if earner.super_mean_override is not None:
        std = earner.super_std_override if earner.super_std_override is not None else SUPER_STD
        return lognormal_real_return(earner.super_mean_override, std, year.super_z[:, index])
    growth = _super_growth_pct(earner, year.y) / 100.0
    equity = asset_real_return("equity", year.eq_z)
    bonds = asset_real_return("bonds", year.class_z[:, CLASS_INDEX["bonds"]])
    return growth * equity + (1.0 - growth) * bonds


def _account_real_return(account: InvestmentAccount, year: _YearShocks) -> FloatArray:
    """Real annual return on one investment account for the year, per trial."""
    column = year.class_z[:, CLASS_INDEX[account.asset_class]]
    if account.interest_rate is None:
        return asset_real_return(account.asset_class, column)
    params = ASSET_CLASS_PARAMS.get(account.asset_class)
    if params is None:
        return np.full(year.eq_z.shape[0], account.interest_rate)
    return lognormal_real_return(account.interest_rate, params["std"], column)


def _half_year_factors(household: Household, year: _YearShocks) -> tuple[FloatArray, FloatArray]:
    """Growth factors for half a year, as (T, O) super and (T, A) account arrays."""
    n = year.eq_z.shape[0]
    super_factors = _columns(
        [
            np.sqrt(1.0 + nominal_return(_super_real_return(e, i, year), year.inflation))
            for i, e in enumerate(household.earners)
        ],
        n,
    )
    account_factors = _columns(
        [
            (
                np.ones(n)
                if a.is_offset
                else np.sqrt(1.0 + nominal_return(_account_real_return(a, year), year.inflation))
            )
            for a in household.investment_accounts
        ],
        n,
    )
    return super_factors, account_factors


def _apply_growth(
    state: _BatchState, super_factors: FloatArray, account_factors: FloatArray
) -> None:
    """Multiply super balances and investment accounts by their growth factors."""
    state.super_balances *= super_factors
    state.account_values *= account_factors


# =============================================================================
# INCOME, SUPER AND TAX
# =============================================================================


def _mls_rate(
    household: Household, inputs: SimulationInputs, family_income: FloatArray
) -> FloatArray:
    """Medicare levy surcharge rate, tested on household income."""
    if not inputs.mls_enabled:
        return np.zeros_like(family_income)
    if inputs.mls_tiered:
        return mls_rate(
            family_income,
            couple=len(household.earners) > 1,
            n_children=household.num_children,
        )
    return np.where(family_income > inputs.mls_threshold, inputs.mls_rate, 0.0)


def _earning(earner: Earner, year: _YearShocks, sg_cap: float, conc_cap: float) -> _Earning:
    """Salary and self-employed income, super guarantee and salary sacrifice for one earner."""
    t = year.y
    salary = earner.salary * (1.0 + earner.salary_growth_rate) ** t * year.price

    if earner.employment_type == "self_employed":
        sg = np.zeros_like(year.price)  # sole traders do not receive SG
    else:
        sg = np.minimum(salary, sg_cap) * earner.sg_rate

    se_income = np.zeros_like(year.price)
    if earner.employment_type == "both" and earner.self_employed_income > 0:
        se_income = (
            earner.self_employed_income * (1.0 + earner.self_employed_growth_rate) ** t * year.price
        )
        if earner.self_employed_sg_applies:
            sg = sg + np.minimum(se_income, sg_cap) * earner.sg_rate

    gross = salary + se_income
    if earner.personal_super_contributions_total_p_a is not None:
        sacrifice = np.full_like(year.price, earner.personal_super_contributions_total_p_a)
    else:
        sacrifice = np.maximum(0.0, conc_cap - sg)
    return _Earning(taxable=gross - sacrifice, gross=gross, sg=sg, sacrifice=sacrifice)


def _part_time_phase(
    state: _BatchState,
    household: Household,
    year: _YearShocks,
    sg_cap: float,
) -> FloatArray:
    """Part-time gross income and its super guarantee, per earner (T, O).

    An earner's part-time income is paid only once they are not in full employment,
    and within their part-time age window.
    """
    n = year.price.shape[0]
    columns: list[FloatArray] = []
    for i, earner in enumerate(household.earners):
        ea = year.earner_ages[i]
        gross = np.zeros(n)
        in_window = (
            earner.pt_days_per_week > 0
            and earner.effective_pt_start_age <= ea < earner.effective_pt_end_age
        )
        in_full_employment = (
            earner.employment_type in ("employed", "both") and ea < earner.retirement_age
        )
        if in_window and not in_full_employment:
            if earner.pt_rate_mode == "salary_pct":
                annual_gross = earner.salary * earner.pt_salary_pct / 100.0 * year.price
                weeks = (
                    earner.pt_weeks_per_year if earner.pt_weeks_per_year > 0 else PT_WEEKS_PER_YEAR
                )
                daily_rate = annual_gross / (earner.pt_days_per_week * weeks)
            else:
                daily_rate = np.full(n, earner.pt_daily_rate)
                weeks = earner.pt_weeks_per_year
            gross = earner.pt_days_per_week * weeks * daily_rate
            state.super_balances[:, i] += (
                np.minimum(gross, sg_cap) * earner.sg_rate * (1.0 - SUPER_TAX_RATE)
            )
        columns.append(gross)
    return _columns(columns, n)


def _earnings_phase(
    state: _BatchState,
    household: Household,
    inputs: SimulationInputs,
    year: _YearShocks,
    ctx: TaxContext,
    pt_gross: FloatArray,
    conc_cap: float,
    sg_cap: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Take-home pay and per-owner assessable and MLS income for the year.

    Each earner's employment and part-time income are assessed together, since a
    household's tax is computed on the combined position of its owners. Returns
    ``(takehome (T,), assessable (T, O), mls_income (T, O))``.
    """
    n = year.price.shape[0]
    working: dict[int, _Earning] = {}
    for i, earner in enumerate(household.earners):
        if earner.employment_type == "not_employed" or year.earner_ages[i] >= earner.retirement_age:
            continue
        working[i] = _earning(earner, year, sg_cap, conc_cap)

    assessable = pt_gross.copy()
    mls_income = pt_gross.copy()
    for i, earning in working.items():
        assessable[:, i] += earning.taxable
        mls_income[:, i] += earning.gross
    family_taxable = assessable.sum(axis=1)
    family_mls = mls_income.sum(axis=1)

    takehome: FloatArray = np.zeros(n)
    for i, earner in enumerate(household.earners):
        owner_takehome = assessable[:, i] - owner_tax(
            ctx, assessable[:, i], family_taxable, family_mls
        )
        w = working.get(i)
        if w is None:
            # Not working: a trial contributes only if it earns part-time income.
            takehome += np.where(pt_gross[:, i] != 0.0, owner_takehome, 0.0)
            continue
        # Division 293 tax, paid from take-home (worst-case assumption). Division 293 income
        # is taxable income (including part-time gross) plus concessional contributions. The
        # threshold is statutory and not indexed.
        contributions = w.sg + w.sacrifice
        div293_income = w.taxable + pt_gross[:, i] + contributions
        excess = np.where(
            (div293_income > inputs.div293_threshold) & (contributions > 0.0),
            np.minimum(contributions, div293_income - inputs.div293_threshold),
            0.0,
        )
        owner_takehome = owner_takehome - excess * inputs.div293_rate
        state.super_balances[:, i] += contributions * (1.0 - SUPER_TAX_RATE)
        if earner.non_concessional_contributions_p_a > 0:
            state.super_balances[:, i] += earner.non_concessional_contributions_p_a
        takehome += owner_takehome
    return takehome, assessable, mls_income


def _investment_income_tax(
    state: _BatchState,
    household: Household,
    year: _YearShocks,
    opening_values: FloatArray,
    ctx: TaxContext,
) -> FloatArray:
    """Assess tax on this year's investment income and reinvest it (D14).

    Each non-offset account distributes income on its opening value. Equity is
    partly franked. Returns the tax per trial (T,).
    """
    total: FloatArray = np.zeros(opening_values.shape[0])
    for ai, account in enumerate(household.investment_accounts):
        if account.is_offset:
            continue
        opening = opening_values[:, ai]
        active = opening > 0.0
        if not active.any():
            continue
        if account.asset_class in INCOME_YIELD:
            distribution = INCOME_YIELD[account.asset_class] * opening
        else:
            real = _account_real_return(account, year)
            distribution = np.maximum(0.0, nominal_return(real, year.inflation)) * opening
        distribution = np.where(active, distribution, 0.0)
        if account.asset_class == "equity":
            franked = distribution * FRANKED_FRACTION
        else:
            franked = np.zeros_like(distribution)
        total += investment_income_tax(
            ctx, state.positions, account_ownership(account), distribution - franked, franked
        )
        add_cost(state.ledger, ai, distribution, year.y)
    return total


# =============================================================================
# EXPENSES AND MORTGAGES
# =============================================================================


def _living_and_education(household: Household, year: _YearShocks) -> FloatArray:
    """Living expenses plus scheduled education costs for the year, per trial."""
    counted = [
        (i, e) for i, e in enumerate(household.earners) if e.employment_type != "not_employed"
    ]
    n_total = len(counted)
    if n_total == 0:
        expense_base = household.retirement_target
    else:
        n_retired = sum(1 for i, e in counted if year.earner_ages[i] >= e.retirement_age)
        expense_base = (n_retired / n_total) * household.retirement_target + (
            (n_total - n_retired) / n_total
        ) * household.base_living_expenses

    education = 0.0
    for child in household.children:
        if child.education_schedule:
            child_age = child.age + year.y
            for sched_age, sched_cost in child.education_schedule:
                if sched_age == child_age:
                    education += sched_cost
                    break
    return expense_base * year.price + education * year.price


def _offset_total(
    state: _BatchState,
    household: Household,
    offset_labels: Sequence[str],
    offset_idxs: list[int],
) -> FloatArray:
    """Sum of offset account balances linked to a mortgage, per trial."""
    total: FloatArray = np.zeros(state.account_values.shape[0])
    for oi in offset_idxs:
        if household.investment_accounts[oi].label in offset_labels:
            total = total + state.account_values[:, oi]
    return total


def _pi_mortgage_cash(
    state: _BatchState, household: Household, age: int, offset_idxs: list[int]
) -> FloatArray:
    """Cash paid on principal-and-interest mortgages this year. Updates principals.

    The offset balance is the start-of-year balance. A mortgage with
    ``reprice_payment`` set re-amortises its payment at each new rate over the
    remaining term.
    """
    cash: FloatArray = np.zeros(state.account_values.shape[0])
    for mi, mortgage in enumerate(household.mortgages):
        if mortgage.monthly_payment == 0:
            continue
        principal = state.mortgage_principals[:, mi]
        active = principal > 0.0
        if not active.any():
            continue
        rate = state.current_mortgage_rates[:, mi]
        if mortgage.reprice_payment and mortgage.loan_term_end_age is not None:
            months = (mortgage.loan_term_end_age - age) * 12
            if months > 0:
                reprice = active & (rate != state.payment_rates[:, mi])
                new_payment = pi_payment(principal, rate, months)
                state.mortgage_payments[:, mi] = np.where(
                    reprice, new_payment, state.mortgage_payments[:, mi]
                )
                state.payment_rates[:, mi] = np.where(reprice, rate, state.payment_rates[:, mi])

        new_principal, interest_paid, principal_paid = amortize_mortgage_monthly(
            mortgage=principal,
            offset=_offset_total(state, household, mortgage.offset_accounts, offset_idxs),
            monthly_pmt=state.mortgage_payments[:, mi],
            monthly_rate=rate / 12.0,
        )
        state.mortgage_principals[:, mi] = np.where(active, new_principal, principal)
        cash += np.where(active, interest_paid + principal_paid, 0.0)
    return cash


def _update_mortgage_rates(
    state: _BatchState, household: Household, mortgage_z: FloatArray
) -> None:
    """Advance each stochastic mortgage rate one year with the Black-Karasinski process."""
    for mi, mortgage in enumerate(household.mortgages):
        if not mortgage.interest_rate_stochastic:
            continue
        # Resolve None -> config default. Use "is not None" so 0.0 is a valid choice.
        theta = (
            mortgage.interest_rate_theta if mortgage.interest_rate_theta is not None else BK_THETA
        )
        kappa = (
            mortgage.interest_rate_kappa if mortgage.interest_rate_kappa is not None else BK_KAPPA
        )
        sigma_tilde = (
            mortgage.interest_rate_vol
            if mortgage.interest_rate_vol is not None
            else BK_SIGMA_MODERATE
        )
        state.current_mortgage_rates[:, mi] = generate_mortgage_rate(
            prev_rate=state.current_mortgage_rates[:, mi],
            z=mortgage_z[:, mi],
            theta=theta,
            kappa=kappa,
            sigma_tilde=sigma_tilde,
        )


# =============================================================================
# SURPLUS, DRAWDOWN AND OFFSETS
# =============================================================================


def _allocate_surplus_or_draw(
    state: _BatchState,
    household: Household,
    inputs: SimulationInputs,
    ctx: TaxContext,
    surplus: FloatArray,
    year: int,
    offset_idxs: list[int],
    non_offset_idxs: list[int],
) -> None:
    """Invest a surplus per the user's allocation, or draw down to cover a deficit."""
    positive = np.maximum(surplus, 0.0)
    invest_pct = inputs.surplus_investment_pct / 100.0
    if offset_idxs and non_offset_idxs and invest_pct > 0:
        to_invest = positive * invest_pct
        to_offset = positive * (1.0 - invest_pct)
        state.account_values[:, offset_idxs[0]] += to_offset
        state.account_values[:, non_offset_idxs[0]] += to_invest
        add_cost(state.ledger, non_offset_idxs[0], to_invest, year)
    elif offset_idxs:
        state.account_values[:, offset_idxs[0]] += positive
    elif non_offset_idxs:
        state.account_values[:, non_offset_idxs[0]] += positive
        add_cost(state.ledger, non_offset_idxs[0], positive, year)
    else:
        state.cash_balance = state.cash_balance + positive

    deficit = np.maximum(-surplus, 0.0)
    if (deficit > 0.0).any():
        unmet = _drawdown(
            state, household, inputs, ctx, deficit, year, offset_idxs, non_offset_idxs
        )
        state.unmet_spending = state.unmet_spending + np.where(unmet > UNMET_TOLERANCE, unmet, 0.0)


def _interest_only_cash(
    state: _BatchState,
    household: Household,
    inputs: SimulationInputs,
    ctx: TaxContext,
    year: int,
    offset_idxs: list[int],
    non_offset_idxs: list[int],
) -> None:
    """Pay interest on interest-only mortgages in cash, after drawdown.

    Interest is charged on principal net of the post-drawdown offset balance.
    """
    io_interest: FloatArray = np.zeros(state.account_values.shape[0])
    for mi, mortgage in enumerate(household.mortgages):
        if mortgage.monthly_payment > 0:
            continue
        principal = state.mortgage_principals[:, mi]
        offset_total = _offset_total(state, household, mortgage.offset_accounts, offset_idxs)
        effective_debt = np.maximum(0.0, principal - offset_total)
        io_interest = io_interest + np.where(
            principal > 0.0, effective_debt * state.current_mortgage_rates[:, mi], 0.0
        )
    io_interest = np.maximum(io_interest, 0.0)

    from_cash = np.minimum(state.cash_balance, io_interest)
    state.cash_balance = state.cash_balance - from_cash
    remaining = io_interest - from_cash
    if (remaining > 0.0).any():
        unmet = _drawdown(
            state, household, inputs, ctx, remaining, year, offset_idxs, non_offset_idxs
        )
        state.unmet_spending = state.unmet_spending + np.where(unmet > UNMET_TOLERANCE, unmet, 0.0)


def _offset_overflow(
    state: _BatchState,
    household: Household,
    year: int,
    offset_idxs: list[int],
    non_offset_idxs: list[int],
) -> None:
    """Move offset balance above the linked mortgage balance into the first non-offset account."""
    if not (offset_idxs and non_offset_idxs):
        return
    first_offset_idx = offset_idxs[0]
    offset_label = household.investment_accounts[first_offset_idx].label
    linked_mortgage: FloatArray = np.zeros(state.account_values.shape[0])
    for mi, mortgage in enumerate(household.mortgages):
        if offset_label in mortgage.offset_accounts:
            linked_mortgage = linked_mortgage + state.mortgage_principals[:, mi]
    target_idx = non_offset_idxs[0]

    basis = cost_total(state.ledger)[:, target_idx]
    new_offset, _, new_account_value, new_account_basis = handle_offset_overflow(
        offset=state.account_values[:, first_offset_idx],
        mortgage=linked_mortgage,
        au_etfs=state.account_values[:, target_idx],
        au_basis=basis,
    )
    state.account_values[:, first_offset_idx] = new_offset
    state.account_values[:, target_idx] = new_account_value
    # An overflow sweep adds basis incurred this year, as its own lot.
    add_cost(state.ledger, target_idx, new_account_basis - basis, year)


# =============================================================================
# YEAR AND HORIZON
# =============================================================================


def _simulate_year(
    state: _BatchState,
    household: Household,
    inputs: SimulationInputs,
    year: _YearShocks,
    offset_idxs: list[int],
    non_offset_idxs: list[int],
) -> None:
    """Simulate one calendar year for every trial, updating ``state`` in place."""
    y = year.y
    super_factors, account_factors = _half_year_factors(household, year)
    opening_values = state.account_values.copy()
    _apply_growth(state, super_factors, account_factors)

    conc_cap = CONC_CAP * (1 + inputs.conc_cap_growth_rate) ** y
    sg_cap = SG_MAX_BASE * (1 + inputs.sg_max_base_growth_rate) ** y
    year_brackets = indexed_brackets(
        inputs.simulation_start_year + y, y, inputs.bracket_growth_rate
    )

    pt_gross = _part_time_phase(state, household, year, sg_cap)
    ctx = TaxContext(
        brackets=year_brackets,
        dependants=household.num_children,
        couple=len(household.earners) > 1,
        surcharge=lambda family_mls: _mls_rate(household, inputs, family_mls),
    )
    takehome, assessable, mls_income = _earnings_phase(
        state, household, inputs, year, ctx, pt_gross, conc_cap, sg_cap
    )
    state.positions.begin_year(assessable, mls_income)

    # Investment income tax is assessed this year and paid next year (D14).
    income_tax = _investment_income_tax(state, household, year, opening_values, ctx)
    prior_carry = state.tax_payable_carry
    state.tax_payable_carry = income_tax
    state.year_income_tax = income_tax

    expenses = (
        _living_and_education(household, year)
        + _pi_mortgage_cash(state, household, year.age, offset_idxs)
        + prior_carry
    )
    _allocate_surplus_or_draw(
        state, household, inputs, ctx, takehome - expenses, y, offset_idxs, non_offset_idxs
    )
    _interest_only_cash(state, household, inputs, ctx, y, offset_idxs, non_offset_idxs)
    _offset_overflow(state, household, y, offset_idxs, non_offset_idxs)

    _apply_growth(state, super_factors, account_factors)
    if inputs.super_fee_rate > 0:
        state.super_balances *= 1.0 - inputs.super_fee_rate
    for ai, account in enumerate(household.investment_accounts):
        if account.fee_rate > 0:
            state.account_values[:, ai] *= 1.0 - account.fee_rate


def run_trials(
    household: Household,
    inputs: SimulationInputs,
    draws: DrawsBatch,
    windfall: float = 0.0,
) -> TrialBatchResult:
    """Simulate every trial in ``draws`` over the bridge horizon.

    The horizon runs from ``simulation_start_age`` until the earliest super access
    age across earners. ``windfall`` is a one-off cash injection at the start,
    added to the first non-offset account.
    """
    validate_horizon(household, inputs)
    n_years = horizon_years(household, inputs)
    n_trials = draws.n_trials
    start_ages = [e.resolved_start_age(inputs.simulation_start_age) for e in household.earners]

    # price_index[:, y] is the price level at the start of year y.
    price_index = np.ones((n_trials, n_years + 1))
    price_index[:, 1:] = np.cumprod(1.0 + draws.inflation[:, :n_years], axis=1)

    state = init_batch_state(household, n_trials, n_years, price_index)
    offset_idxs = _offset_indices(household)
    non_offset_idxs = _non_offset_indices(household)

    # The windfall goes to the first non-offset account, with its cost base incurred now.
    if windfall > 0 and state.account_values.shape[1] > 0:
        target_idx = non_offset_idxs[0] if non_offset_idxs else 0
        state.account_values[:, target_idx] += windfall
        add_cost(state.ledger, target_idx, np.full(n_trials, windfall), 0)

    n_mortgages = len(household.mortgages)
    n_offset = len(offset_idxs)
    reform_year = max(0, 2027 - inputs.simulation_start_year)

    min_real = np.full(n_trials, np.inf)
    min_bridge = np.full(n_trials, np.inf)
    floor_age = np.full(n_trials, inputs.simulation_start_age, dtype=np.int64)
    floor_deflator = np.ones(n_trials)
    first_failure = np.full(n_trials, -1, dtype=np.int64)

    bridge_by_age = np.zeros((n_trials, n_years))
    mortgage_by_age = np.zeros((n_trials, n_years, n_mortgages))
    offset_by_age = np.zeros((n_trials, n_years, n_offset))
    rate_by_age = np.zeros((n_trials, n_years, n_mortgages))
    offset_drawn_by_age = np.zeros((n_trials, n_years))
    non_offset_drawn_by_age = np.zeros((n_trials, n_years))
    cgt_paid_by_age = np.zeros((n_trials, n_years))
    cgt_without_floor_by_age = np.zeros((n_trials, n_years))
    income_tax_by_age = np.zeros((n_trials, n_years))

    for y in range(n_years):
        age = inputs.simulation_start_age + y
        year_price = price_index[:, y]
        end_price = price_index[:, y + 1]
        year = _YearShocks(
            y=y,
            age=age,
            price=year_price,
            inflation=draws.inflation[:, y],
            eq_z=draws.eq_z[:, y],
            class_z=draws.class_z[:, y, :],
            super_z=draws.super_override_z[:, :, y],
            mortgage_z=draws.mortgage_z[:, :, y],
            earner_ages=tuple(s + y for s in start_ages),
        )

        _update_mortgage_rates(state, household, year.mortgage_z)

        state.year_offset_drawn = np.zeros(n_trials)
        state.year_non_offset_drawn = np.zeros(n_trials)
        state.year_cgt_paid = np.zeros(n_trials)
        state.year_cgt_without_floor = np.zeros(n_trials)
        if y == reform_year:
            reform_snapshot(state.ledger, state.account_values, y)

        unmet_before = state.unmet_spending.copy()
        _simulate_year(state, household, inputs, year, offset_idxs, non_offset_idxs)

        # Year-end balances are deflated by the end-of-year price level.
        net = state.total_bridge - state.unmet_spending
        real_net = net / end_price
        better = real_net < min_real
        min_real = np.where(better, real_net, min_real)
        min_bridge = np.where(better, net, min_bridge)
        floor_age = np.where(better, age, floor_age)
        floor_deflator = np.where(better, end_price, floor_deflator)

        # A year fails if spending went unmet or the net bridge is exhausted.
        failed = (state.unmet_spending > unmet_before) | (net <= 0.0)
        first_failure = np.where(failed & (first_failure < 0), age, first_failure)

        bridge_by_age[:, y] = real_net
        mortgage_by_age[:, y, :] = state.mortgage_principals
        offset_by_age[:, y, :] = state.account_values[:, offset_idxs]
        rate_by_age[:, y, :] = state.current_mortgage_rates

        # Drawdown flows are in year-start prices.
        offset_drawn_by_age[:, y] = state.year_offset_drawn / year_price
        non_offset_drawn_by_age[:, y] = state.year_non_offset_drawn / year_price
        cgt_paid_by_age[:, y] = state.year_cgt_paid / year_price
        cgt_without_floor_by_age[:, y] = state.year_cgt_without_floor / year_price
        income_tax_by_age[:, y] = state.year_income_tax / year_price

    horizon_age = inputs.simulation_start_age + n_years
    term_checkable = np.array(
        [
            m.loan_term_end_age is not None and horizon_age >= m.loan_term_end_age
            for m in household.mortgages
        ],
        dtype=np.bool_,
    )
    return TrialBatchResult(
        bridge=state.total_bridge - state.unmet_spending - state.tax_payable_carry,
        super_balances=state.super_balances,
        mortgage_principals=state.mortgage_principals,
        horizon_age=horizon_age,
        term_checkable=term_checkable,
        term_cleared=state.mortgage_principals <= 0.0,
        min_bridge=min_bridge,
        unmet_spending=state.unmet_spending,
        floor_age=floor_age,
        first_failure_age=first_failure,
        bridge_by_age=bridge_by_age,
        mortgage_by_age=mortgage_by_age,
        offset_by_age=offset_by_age,
        mortgage_rate_by_age=rate_by_age,
        offset_drawn_by_age=offset_drawn_by_age,
        non_offset_drawn_by_age=non_offset_drawn_by_age,
        cgt_paid_by_age=cgt_paid_by_age,
        cgt_without_floor_by_age=cgt_without_floor_by_age,
        income_tax_by_age=income_tax_by_age,
        floor_deflator=floor_deflator,
        horizon_deflator=price_index[:, n_years],
        total_offset_drawn=offset_drawn_by_age.sum(axis=1),
        total_non_offset_drawn=non_offset_drawn_by_age.sum(axis=1),
        total_cgt_paid=cgt_paid_by_age.sum(axis=1),
        total_cgt_without_floor=cgt_without_floor_by_age.sum(axis=1),
        total_income_tax=income_tax_by_age.sum(axis=1),
    )


def run_single_trial(
    household: Household,
    inputs: SimulationInputs,
    draws: DrawsBatch | None = None,
    windfall: float = 0.0,
) -> TrialBatchResult:
    """Run one trial (``T == 1``) of the batch engine.

    With ``draws`` None the median path runs: all shocks zero, inflation held at
    ``inputs.inflation``.
    """
    if draws is None:
        validate_horizon(household, inputs)
        draws = DrawsBatch.median(
            n_years=horizon_years(household, inputs),
            n_earners=len(household.earners),
            n_mortgages=len(household.mortgages),
            inflation=inputs.inflation,
        )
    elif draws.n_trials != 1:
        raise ValueError(f"run_single_trial takes one trial, got {draws.n_trials}; use run_trials")
    return run_trials(household, inputs, draws, windfall)
