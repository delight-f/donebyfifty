"""Monte Carlo retirement simulation engine.

Imports verified mathematical primitives from ``primitives.py`` and drives
the N-of-everything household model from ``models.py`` through stochastic
trials.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

from config import (
    BK_KAPPA,
    BK_RHO,
    BK_SIGMA_MODERATE,
    BK_THETA,
    CONC_CAP,
    SG_MAX_BASE,
)
from config import (
    SUPER_TAX_ON_CONTRIBUTIONS as SUPER_TAX_RATE,
)
from models import Household, SimulationInputs, SimulationResults
from primitives import (
    ASSET_CLASS_PARAMS,
    BRACKETS,
    EDU_SCHEDULE_TODAY,
    EQ_MEAN,
    PT_WEEKS_PER_YEAR,
    SUPER_EQ_CORR,
    AssetHolding,
    CostBaseLot,
    amortize_mortgage_monthly,
    cgt_split_tax,
    cgt_weighted_rate,
    consulting_net_income,
    generate_asset_return,
    generate_correlated_returns,
    generate_correlated_triplet,
    generate_mortgage_rate,
    handle_offset_overflow,
    indexed_cost_base,
    sell_assets,
    tax,
)

# =============================================================================
# INTERNAL STATE
# =============================================================================


@dataclass
class _SimulationState:
    """Mutable state tracked across a single simulation trial.

    Uses parallel lists indexed to match the ``Household`` tuples.
    Not part of the public API (underscore prefix).
    """

    age: int
    year: int  # years since simulation start (0-indexed)

    # Per-earner (index matches household.earners)
    super_balances: list[float] = field(default_factory=list)
    earner_taxable_incomes: list[float] = field(default_factory=list)
    """Per-earner taxable income for the current simulation year.

    Populated in ``simulate_working_year()`` before ``_drawdown()`` is called.
    Used by ``_drawdown()`` to compute per-owner weighted marginal CGT rates.
    Includes salary (if working) + PT gross income (if in PT phase) - sacrifices.
    ``"""

    # Per-mortgage (index matches household.mortgages)
    mortgage_principals: list[float] = field(default_factory=list)
    current_mortgage_rates: list[float] = field(default_factory=list)
    """Mortgage rates for the current simulation year (decimal).
    Initialised to each mortgage's ``interest_rate`` at state init.
    When ``interest_rate_stochastic`` is False, this stays at the
    initial value for all years (static rate).  When True, updated
    each year in ``run_single_trial`` via the BK process.
    """

    # Per-account (index matches household.investment_accounts)
    account_values: list[float] = field(default_factory=list)
    account_bases: list[float] = field(default_factory=list)

    # H5 — acquisition-dated cost-base lots per account. Indexation keys off
    # each lot's own ``incurred`` year (s960-275), not simulation year 0.
    account_lots: list[list[CostBaseLot]] = field(default_factory=list)

    # H5 — cumulative inflation factor per simulation year, so a lot can be
    # indexed by ``cum_now / cum_incurred``.
    cumulative_inflation_by_year: list[float] = field(default_factory=list)

    # H5 — 30 June 2027 reform snapshot per account: market value and nominal
    # basis at the reform line, plus the cumulative inflation factor then.
    reform_values: list[float | None] = field(default_factory=list)
    reform_bases: list[float] = field(default_factory=list)
    reform_inflation: float = 1.0

    # Implicit cash balance for households with no investment accounts
    # Surplus cash accumulates here; drawn from during retirement shortfalls
    cash_balance: float = 0.0

    # Cumulative unmet spending when bridge assets are exhausted
    # (used to measure true drawdown depth, not just the minimum balance)
    unmet_spending: float = 0.0

    # Per-year drawdown composition tracking (Work Items 4, 5)
    # Reset to 0 at the start of each year in run_single_trial
    year_offset_drawn: float = 0.0
    year_non_offset_drawn: float = 0.0
    year_cgt_paid: float = 0.0
    year_cgt_without_floor: float = 0.0

    # Computed properties
    @property
    def total_bridge(self) -> float:
        """Sum of accessible bridge assets (offset + taxable accounts + cash)."""
        return sum(self.account_values) + self.cash_balance


# =============================================================================
# STATE INITIALISATION
# =============================================================================


def init_state(
    household: Household,
    start_age: int,
) -> _SimulationState:
    """Create initial simulation state from a household definition.

    Args:
        household: The household to simulate.
        start_age: Age at simulation start (year 0).

    Returns:
        A new ``_SimulationState`` with initial values populated.

    """
    state = _SimulationState(
        age=start_age,
        year=0,
        super_balances=[e.super_balance for e in household.earners],
        mortgage_principals=[m.principal for m in household.mortgages],
        current_mortgage_rates=[m.interest_rate for m in household.mortgages],
        account_values=[a.market_value for a in household.investment_accounts],
        account_bases=[a.cost_basis for a in household.investment_accounts],
        account_lots=[
            [{"basis": a.cost_basis, "incurred": 0}] if a.cost_basis > 0 else []
            for a in household.investment_accounts
        ],
        reform_values=[None] * len(household.investment_accounts),
        reform_bases=[0.0] * len(household.investment_accounts),
    )
    return state


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


def _series_rng(seed: int | None) -> random.Random:
    """Return the RNG used for pre-generated return series.

    With an explicit seed the generator is fully isolated. Without one it is
    seeded from the module-level RNG, so a global ``random.seed`` — e.g. a test
    fixture — makes unseeded runs reproducible rather than drawing fresh OS
    entropy (M14).
    """
    return random.Random(seed) if seed is not None else random.Random(random.getrandbits(64))


# =============================================================================
# WORKING YEAR
# =============================================================================


def simulate_working_year(
    state: _SimulationState,
    household: Household,
    inputs: SimulationInputs,
    eq_return: float,
    eq_z: float = 0.0,
    deterministic: bool = False,
    cumulative_inflation: float | None = None,
    *,
    offset_idxs: list[int] | None = None,
    non_offset_idxs: list[int] | None = None,
    infl_t: float | None = None,
) -> float:
    """Simulate one calendar year while at least one earner is working.

    Each employed earner generates salary, pays tax, makes super contributions.
    The household surplus (or deficit) is applied to offset accounts first.
    Mortgages are amortised with offset benefit, then assets grow.

    Per-earner super returns are generated independently via
    ``generate_asset_return``, using the equity z-score ``eq_z`` to
    maintain correlation with equity (the correlated super return from
    Cholesky is not used — each earner's super asset class may differ).

    Args:
        state: Mutable simulation state (updated in-place).
        household: The household definition.
        inputs: Simulation parameters (inflation, etc.).
        eq_return: Equity return for this year (decimal).
        eq_z: Standard normal draw for equity (used for per-asset-class returns).
        deterministic: If True, use mean returns without stochastic noise.
        cumulative_inflation: Accumulated inflation factor. If None, uses
            ``(1 + inputs.inflation) ** y`` (fixed rate).
        offset_idxs: Precomputed offset account indices. If None, computed now.
        non_offset_idxs: Precomputed non-offset account indices. If None, computed now.
        infl_t: Per-year inflation for uplifting real returns to nominal.
            If None, uses ``inputs.inflation``.

    Returns:
        Unmet spending shortfall for this year (0 if fully covered).

    """
    y = state.year
    inflation = inputs.inflation
    year_factor = cumulative_inflation if cumulative_inflation is not None else (1 + inflation) ** y
    # Per-year inflation for uplifting real returns to nominal
    per_year_infl = infl_t if infl_t is not None else inputs.inflation

    # ── 0. Grow super FIRST with per-earner returns ─────────────────────
    for si, earner in enumerate(household.earners):
        if earner.super_mean_override is not None:
            if deterministic:
                per_super = (1.0 + earner.super_mean_override) * (1.0 + per_year_infl) - 1.0
            else:
                mu_s = (
                    math.log(1 + earner.super_mean_override)
                    - 0.5 * (earner.super_std_override or 0.12) ** 2
                )
                z_s = random.gauss(0, 1)
                per_super = (
                    math.exp(mu_s + (earner.super_std_override or 0.12) * z_s)
                    * (1.0 + per_year_infl)
                    - 1.0
                )
        else:
            # Blended return: growth_pct% equity + (1 - growth_pct)% bonds
            growth_pct = earner.super_growth_pct
            if earner.super_glide_end_year is not None and y < earner.super_glide_end_year:
                progress = y / earner.super_glide_end_year
                growth_pct = growth_pct + (earner.super_glide_target_pct - growth_pct) * progress
            eq_ret = generate_asset_return(
                "equity", eq_z, eq_return, deterministic=deterministic, inflation=per_year_infl
            )
            bond_ret = generate_asset_return(
                "bonds", eq_z, eq_return, deterministic=deterministic, inflation=per_year_infl
            )
            per_super = (growth_pct / 100.0) * eq_ret + (1.0 - growth_pct / 100.0) * bond_ret
        state.super_balances[si] *= 1.0 + per_super
    if inputs.super_fee_rate > 0:
        for si in range(len(state.super_balances)):
            state.super_balances[si] *= 1.0 - inputs.super_fee_rate

    # ── 0a. Index policy caps ────────────────────────────────────────────
    indexed_conc_cap = CONC_CAP * (1 + inputs.conc_cap_growth_rate) ** y
    indexed_sg_max = SG_MAX_BASE * (1 + inputs.sg_max_base_growth_rate) ** y

    # ── 1. Compute total take-home from all earners ──────────────────────
    # Build indexed tax brackets for this year (bracket creep)
    year_factor_brackets = (1 + inputs.bracket_growth_rate) ** y
    indexed_brackets = tuple(
        (threshold * year_factor_brackets, rate) for threshold, rate in BRACKETS
    )

    total_takehome = 0.0

    # Per-earner taxable income accumulator (salary + PT gross - sacrifices)
    earner_taxable = [0.0] * len(household.earners)

    for i, earner in enumerate(household.earners):
        if earner.employment_type == "not_employed":
            # Not employed — no salary income, no SG this year
            continue
        if (
            earner.employment_type in ("employed", "self_employed", "both")
            and state.age >= earner.retirement_age
        ):
            # Retired — skip income, even if self-employed or both
            continue

        # ── Salary income (employed / both) ──────────────────────────
        # Salary growth is a real (above-inflation) rate.
        # Convert to effective nominal by compounding with inflation:
        #   (1 + real_growth) * (1 + inflation) - 1
        effective_growth = (1 + earner.salary_growth_rate) * (1 + inputs.inflation) - 1
        salary = earner.salary * (1 + effective_growth) ** y

        if earner.employment_type in ("self_employed",):
            # Self-employed only: SG does not apply (sole traders don't receive SG)
            sg = 0.0
            if earner.sg_rate > 0:
                sg = 0.0
        elif earner.employment_type == "both":
            # Both: SG applies to salary portion, plus optionally on self-employed income
            sg = min(salary, indexed_sg_max) * earner.sg_rate
        else:
            # Employed: standard SG
            sg = min(salary, indexed_sg_max) * earner.sg_rate

        # ── Self-employed income (both) ──────────────────────────────
        se_income = 0.0
        if earner.employment_type == "both" and earner.self_employed_income > 0:
            eff_se_growth = (1 + earner.self_employed_growth_rate) * (1 + inputs.inflation) - 1
            se_income = earner.self_employed_income * (1 + eff_se_growth) ** y
            if earner.self_employed_sg_applies:
                sg += min(se_income, indexed_sg_max) * earner.sg_rate

        # ── Total gross income for tax purposes ─────────────────────
        total_gross = salary + se_income

        if earner.personal_super_contributions_total_p_a is not None:
            sacrifice = earner.personal_super_contributions_total_p_a
        else:
            sacrifice = max(0.0, indexed_conc_cap - sg)
        taxable = total_gross - sacrifice
        if inputs.mls_enabled:
            if inputs.mls_tiered:
                # Progressive MLS tiers based on per-earner taxable income.
                # NOTE: Real MLS applies to combined household income for
                # couples.  This per-earner computation using singles tiers
                # is a simplification (conservative — it may over-apply MLS
                # for dual-income households where neither earner crosses
                # the individual threshold but combined they do).
                from primitives import mls_rate_for_income

                mls = mls_rate_for_income(taxable, n_earners=len(household.earners))
            else:
                mls = inputs.mls_rate if taxable > inputs.mls_threshold else 0.0
        else:
            mls = 0.0
        income_tax = tax(taxable, medicare_surcharge=mls, brackets=indexed_brackets)
        takehome = total_gross - income_tax - sacrifice

        # Division 293 tax paid from take-home (worst-case assumption).
        # Division 293: 15% on the LESSER of (concessional contributions,
        # div293_income - $250k threshold). The threshold is statutory and
        # has been $250,000 since 1 July 2017 — it is NOT indexed.
        # In practice, the taxpayer can elect to release excess contributions
        # from super to cover this tax, reducing the cash-flow impact.
        div293_income = taxable + sg + sacrifice  # combined income + contributions
        div293_threshold = inputs.div293_threshold  # statutory, not indexed
        if div293_income > div293_threshold and (sg + sacrifice) > 0:
            excess = min(sg + sacrifice, div293_income - div293_threshold)
            div293_tax = excess * inputs.div293_rate
            takehome -= div293_tax  # paid from take-home (worst-case assumption)

        earner_taxable[i] = taxable  # salary - sacrifice (or 0.0 if not working)

        total_takehome += takehome

        net_contrib = (sg + sacrifice) * (1 - SUPER_TAX_RATE)
        state.super_balances[i] += net_contrib

        # Non-concessional (after-tax) contributions
        if earner.non_concessional_contributions_p_a > 0:
            state.super_balances[i] += earner.non_concessional_contributions_p_a

    # ── 2. Compute total part-time income ─────────────────────────────────
    # PT income is only paid if the earner is NOT in full employment.
    # This prevents double-counting when pt_start_age overlaps with
    # employment years (the earner can't work two full-time jobs).
    total_pt_income = 0.0
    for ei, earner in enumerate(household.earners):
        if earner.pt_days_per_week > 0 and earner.pt_start_age <= state.age < earner.pt_end_age:
            if earner.employment_type in ("employed", "both") and state.age < earner.retirement_age:
                # Earner is still in full employment — skip PT income
                continue

            # Determine the effective daily rate based on the earner's mode
            if earner.pt_rate_mode == "salary_pct":
                # PT income as a percentage of the earner's initial full-time salary
                pt_annual_gross = earner.salary * earner.pt_salary_pct / 100.0
                weeks = (
                    earner.pt_weeks_per_year if earner.pt_weeks_per_year > 0 else PT_WEEKS_PER_YEAR
                )
                days = earner.pt_days_per_week if earner.pt_days_per_week > 0 else 1.0
                eff_daily_rate = pt_annual_gross / (days * weeks)
            else:
                eff_daily_rate = earner.pt_daily_rate
                weeks = earner.pt_weeks_per_year

            total_pt_income += consulting_net_income(
                earner.pt_days_per_week,
                brackets=indexed_brackets,
                daily_rate=eff_daily_rate,
                weeks_per_year=weeks,
            )
            # SG on PT gross income (employer SG obligation)
            pt_gross = earner.pt_days_per_week * weeks * eff_daily_rate
            # Add PT GROSS income (not net) to taxable income for marginal rate calculation
            earner_taxable[ei] += pt_gross
            pt_sg = min(pt_gross, indexed_sg_max) * earner.sg_rate
            state.super_balances[ei] += pt_sg * (1 - SUPER_TAX_RATE)

    total_takehome += total_pt_income

    # Write per-earner taxable incomes into state (used by _drawdown for CGT marginal rates)
    state.earner_taxable_incomes = earner_taxable

    # ── 3. Compute total expenses ────────────────────────────────────────
    # Living expenses
    n_retired = sum(1 for e in household.earners if state.age >= e.retirement_age)
    n_total = len(household.earners)
    expense_base = (n_retired / n_total) * household.retirement_target + (
        (n_total - n_retired) / n_total
    ) * household.base_living_expenses
    living = expense_base * year_factor

    # Education costs for each child
    education = 0.0
    for child in household.children:
        if child.education_schedule:
            # Use custom schedule
            child_age = child.age + y
            for sched_age, sched_cost in child.education_schedule:
                if sched_age == child_age:
                    education += sched_cost * year_factor
                    break
        else:
            # Use default education schedule (inflated by year_factor
            # so stochastic inflation is applied consistently with custom schedules)
            annual_today = EDU_SCHEDULE_TODAY.get(child.age + y, 0.0)
            if annual_today > 0:
                education += annual_today * year_factor

    # Mortgage payments (only for mortgages with outstanding principal)
    mortgage_payments = sum(
        m.monthly_payment * 12
        for mi, m in enumerate(household.mortgages)
        if state.mortgage_principals[mi] > 0
    )

    total_expenses = living + education + mortgage_payments

    # ── 4. Apply surplus/deficit to accounts ────────────────────────────
    offset_idxs = offset_idxs if offset_idxs is not None else _offset_indices(household)
    non_offset_idxs = (
        non_offset_idxs if non_offset_idxs is not None else _non_offset_indices(household)
    )

    # First pass: handle P&I mortgage payments + living + education
    # IO interest is handled separately in step 4b (after drawdown)
    surplus = total_takehome - total_expenses

    if surplus > 0:
        # ── Positive surplus: allocate per user preference ────────────
        # Track how much was allocated so IO interest can reclaim it
        invest_pct = inputs.surplus_investment_pct / 100.0
        if offset_idxs and non_offset_idxs and invest_pct > 0:
            to_invest = surplus * invest_pct
            to_offset = surplus * (1.0 - invest_pct)
            state.account_values[offset_idxs[0]] += to_offset
            state.account_values[non_offset_idxs[0]] += to_invest
            state.account_bases[non_offset_idxs[0]] += to_invest
        elif offset_idxs:
            state.account_values[offset_idxs[0]] += surplus
        elif non_offset_idxs:
            state.account_values[non_offset_idxs[0]] += surplus
            state.account_bases[non_offset_idxs[0]] += surplus
        else:
            state.cash_balance += surplus
        surplus = 0.0  # all allocated; IO interest draws from accounts if needed

    elif surplus < 0:
        deficit = -surplus
        unmet = _drawdown(
            state,
            household,
            deficit,
            inputs,
            offset_idxs=offset_idxs,
            non_offset_idxs=non_offset_idxs,
        )
        if unmet > 0:
            state.unmet_spending += unmet
        surplus = -unmet  # 0 if covered, negative if still unmet

    # ── 4b. Interest on interest-only mortgages (post-drawdown offset) ──
    # Calculated AFTER the first drawdown so offset balance reflects the
    # actual funds available during this period. If the interest cannot be
    # covered from remaining surplus, a second drawdown covers it.
    io_interest = 0.0
    for mi, mortgage in enumerate(household.mortgages):
        if state.mortgage_principals[mi] <= 0:
            continue
        if mortgage.monthly_payment > 0:
            continue  # Interest is embedded in the P&I payment
        offset_total = 0.0
        for oi in offset_idxs:
            acct = household.investment_accounts[oi]
            if acct.label in mortgage.offset_accounts:
                offset_total += state.account_values[oi]
        effective_debt = max(0.0, state.mortgage_principals[mi] - offset_total)
        io_interest += effective_debt * state.current_mortgage_rates[mi]

    if io_interest > 0:
        # IO interest is a cash expense — pay from cash balance first, then draw from accounts.
        if state.cash_balance >= io_interest:
            state.cash_balance -= io_interest
        else:
            remaining_io = io_interest - state.cash_balance
            state.cash_balance = 0.0
            unmet = _drawdown(
                state,
                household,
                remaining_io,
                inputs,
                offset_idxs=offset_idxs,
                non_offset_idxs=non_offset_idxs,
            )
            if unmet > 0:
                state.unmet_spending += unmet

    # ── 5. Amortise mortgages with offset benefit ────────────────────────
    for mi, mortgage in enumerate(household.mortgages):
        if state.mortgage_principals[mi] <= 0:
            continue
        monthly_rate = state.current_mortgage_rates[mi] / 12

        # Find total offset balance linked to this mortgage
        offset_total = 0.0
        for oi in offset_idxs:
            acct = household.investment_accounts[oi]
            if acct.label in mortgage.offset_accounts:
                offset_total += state.account_values[oi]

        new_m, _ = amortize_mortgage_monthly(
            mortgage=state.mortgage_principals[mi],
            offset=offset_total,
            monthly_pmt=mortgage.monthly_payment,
            monthly_rate=monthly_rate,
        )
        state.mortgage_principals[mi] = new_m

    # ── 6. Handle offset overflow (excess offset → first non-offset acct) ─
    if offset_idxs and non_offset_idxs and state.account_values:
        first_offset_idx = offset_idxs[0]
        offset_label = household.investment_accounts[first_offset_idx].label

        # Sum only mortgages linked to this offset account
        linked_mortgage = sum(
            state.mortgage_principals[mi]
            for mi, m in enumerate(household.mortgages)
            if offset_label in m.offset_accounts
        )

        first_non_offset = non_offset_idxs
        target_idx = first_non_offset[0]

        o_val = state.account_values[first_offset_idx]
        acct_val = state.account_values[target_idx]
        acct_basis = state.account_bases[target_idx]

        new_o, new_m, new_acct_val, new_acct_basis = handle_offset_overflow(
            offset=o_val,
            mortgage=linked_mortgage,
            au_etfs=acct_val,
            au_basis=acct_basis,
        )
        state.account_values[first_offset_idx] = new_o
        state.account_values[target_idx] = new_acct_val
        overflow_basis = new_acct_basis - acct_basis
        state.account_bases[target_idx] = new_acct_basis
        # H5 — an overflow sweep adds basis incurred this year, as its own lot.
        if overflow_basis > 0 and target_idx < len(state.account_lots):
            state.account_lots[target_idx].append({"basis": overflow_basis, "incurred": state.year})

    # ── 7. Grow investment accounts (custom interest rates if specified) ────
    for ai, account in enumerate(household.investment_accounts):
        if not account.is_offset:
            # Custom return override: use user-specified mean but retain
            # the asset class's standard deviation and equity correlation.
            # This prevents accidentally disabling Monte Carlo stochasticity
            # by setting a non-zero interest_rate (Finding F1).
            if account.interest_rate > 0:
                params = ASSET_CLASS_PARAMS.get(account.asset_class)
                if params is None:
                    # Unknown asset class — fall back to fixed rate
                    asset_return = (1.0 + account.interest_rate) * (1.0 + per_year_infl) - 1.0
                elif deterministic:
                    asset_return = (1.0 + account.interest_rate) * (1.0 + per_year_infl) - 1.0
                else:
                    mu = math.log(1 + account.interest_rate) - 0.5 * params["std"] ** 2
                    z_independent = random.gauss(0, 1)
                    z_asset = (
                        params["corr_with_eq"] * eq_z
                        + math.sqrt(1 - params["corr_with_eq"] ** 2) * z_independent
                    )
                    asset_return = (
                        math.exp(mu + params["std"] * z_asset) * (1.0 + per_year_infl) - 1.0
                    )
            else:
                # Use asset class returns
                asset_return = generate_asset_return(
                    account.asset_class,
                    eq_z,
                    eq_return,
                    deterministic=deterministic,
                    inflation=per_year_infl,
                )
            state.account_values[ai] *= 1.0 + asset_return
        # Offset accounts do NOT grow — benefit is via reduced mortgage interest

    # ── 7a. Deduct account fees after growth ─────────────────────────────
    if any(a.fee_rate > 0 for a in household.investment_accounts):
        for ai, account in enumerate(household.investment_accounts):
            if account.fee_rate > 0:
                state.account_values[ai] *= 1.0 - account.fee_rate

    # (Super already grown in step 0 — contributions added above)
    return 0.0


# =============================================================================
# DRAWDOWN HELPER
# =============================================================================


def _drawdown(
    state: _SimulationState,
    household: Household,
    remain: float,
    inputs: SimulationInputs,
    *,
    non_offset_idxs: list[int] | None = None,
    offset_idxs: list[int] | None = None,
) -> float:
    """Draw down investment accounts to cover a spending shortfall.

    Draws from offset accounts first (tax-free), then non-offset accounts
    in the order specified by ``inputs.sell_order`` (or worst-tax first if
    no order is specified).

    Args:
        state: Mutable simulation state (updated in-place).
        household: The household definition.
        remain: Amount of cash still needed.
        inputs: Simulation parameters (CGT on/off, sell order).
        non_offset_idxs: Precomputed non-offset indices. If None, computed now.
        offset_idxs: Precomputed offset indices. If None, computed now.

    Returns:
        Remaining shortfall after drawing all available funds (0 if covered).

    """
    cgt_on = inputs.cgt_on_drawdowns
    offset_idxs = offset_idxs if offset_idxs is not None else _offset_indices(household)
    non_offset_idxs = (
        non_offset_idxs if non_offset_idxs is not None else _non_offset_indices(household)
    )

    # Step 1: Draw from offset accounts (tax-free), respecting reserve floors
    for ai in offset_idxs:
        if remain <= 0:
            return 0.0
        val = state.account_values[ai]
        if val <= 0:
            continue
        # Compute reserve floor for this offset account:
        # the max floor across all mortgages linked to this offset label.
        label = household.investment_accounts[ai].label
        floor = 0.0
        for mi, mortgage in enumerate(household.mortgages):
            if label not in mortgage.offset_accounts:
                continue
            if mortgage.offset_reserve_mode == "stall_prevention" and mortgage.monthly_payment > 0:
                # Preserve enough offset so the payment covers the interest
                # on the net debt.  Result: interest = payment, principal = 0.
                # required = max(0, balance - payment / monthly_rate)
                monthly_rate = state.current_mortgage_rates[mi] / 12
                if monthly_rate > 0:
                    dyn_floor = max(
                        0.0,
                        state.mortgage_principals[mi] - (mortgage.monthly_payment / monthly_rate),
                    )
                    floor = max(floor, dyn_floor)
            elif mortgage.offset_reserve_mode == "interest_cancelling":
                # Preserve enough offset to fully cancel all interest.
                # Floor = full mortgage principal.  Result: interest = 0,
                # 100% of payment goes to principal.
                floor = max(floor, state.mortgage_principals[mi])
            else:
                # "fixed" mode — use static offset_reserve_floor
                floor = max(floor, mortgage.offset_reserve_floor)
        available = max(0.0, val - floor)
        if available <= 0:
            continue
        draw = min(available, remain)
        state.account_values[ai] -= draw
        state.year_offset_drawn += draw
        remain -= draw

    # Step 2: Determine sell order for non-offset accounts
    if inputs.sell_order:
        # User-specified order: match by label
        ordered: list[int] = []
        for label in inputs.sell_order:
            for ai in non_offset_idxs:
                if household.investment_accounts[ai].label == label:
                    ordered.append(ai)
                    break
        # Append any unlisted accounts at the end
        for ai in non_offset_idxs:
            if ai not in ordered:
                ordered.append(ai)
        sell_order = ordered
    else:
        # Default: sell in order of lowest taxable gain first (defer CGT,
        # letting larger unrealised gains continue to compound).
        sell_order = sorted(
            non_offset_idxs,
            key=lambda ai: (
                state.account_values[ai] - state.account_bases[ai]
                if state.account_values[ai] > 0
                else 1e9
            ),
        )

    # Step 3: Sell non-offset accounts in order
    for ai in sell_order:
        if remain <= 0:
            return 0.0
        if state.account_values[ai] <= 1.0:
            continue
        account = household.investment_accounts[ai]
        asset: AssetHolding = {
            "val": state.account_values[ai],
            "basis": state.account_bases[ai],
        }
        # Compute per-owner weighted CGT rate.
        #   H6: the gain is stacked on each owner's income and taxed by
        #   integrating the bracket schedule from ``income`` to
        #   ``income + gain`` (per owner), rather than applying one flat
        #   marginal rate to the whole gain.
        #   H5: for a post-reform disposal of an asset held at 30 June 2027 the
        #   gain is split at the reform line — the pre-reform portion gets the
        #   50% discount, the post-reform portion is CPI-indexed from the reform
        #   market value with the 30% floor. The indexation is then embedded in
        #   the rate, so ``sell_assets`` is passed a factor of 1.0 (passing the
        #   full factor as well would double-count indexation).
        weighted_rate = account.cgt_rate
        raw_weighted_rate = account.cgt_rate
        if cgt_on and state.earner_taxable_incomes:
            owners = [
                (
                    (
                        state.earner_taxable_incomes[ei]
                        if ei < len(state.earner_taxable_incomes)
                        else 0.0
                    ),
                    share,
                )
                for ei, share in account.ownership.items()
                if share > 0
            ]
            nominal_gain = asset["val"] - asset["basis"]
            reform_mv = state.reform_values[ai] if ai < len(state.reform_values) else None
            reform_year_index = max(0, 2027 - inputs.simulation_start_year)
            if state.year > reform_year_index and reform_mv is not None and nominal_gain > 0:
                # Pre-reform portion: nominal gain to the reform line.
                pre_gain = max(0.0, reform_mv - state.reform_bases[ai])
                # Post-reform portion: proceeds less the reform-line basis
                # indexed from 30 June 2027 (each lot from its own year).
                lots = state.account_lots[ai] if ai < len(state.account_lots) else []
                post_basis = indexed_cost_base(lots, state.year, state.cumulative_inflation_by_year)
                post_gain = max(0.0, asset["val"] - post_basis)
                split_tax, split_tax_raw = cgt_split_tax(owners, pre_gain, post_gain)
                weighted_rate = split_tax / nominal_gain
                raw_weighted_rate = split_tax_raw / nominal_gain
            else:
                # Pre-reform disposal (on or before the 2027 line): the 50% CGT
                # discount applies and there is no cost-base indexation.
                weighted_rate, raw_weighted_rate = cgt_weighted_rate(
                    owners, nominal_gain, discount=0.5
                )

        old_remain = remain
        old_basis = state.account_bases[ai]
        remain, tax_paid, tax_without_floor = sell_assets(
            asset,
            remain,
            cgt_on,
            weighted_marginal_rate=weighted_rate,
            raw_marginal_rate=raw_weighted_rate,
            # H5 indexation is already embedded in ``weighted_rate`` (via the
            # reform split or the pre-reform discount), so sell_assets must not
            # index the basis again.
            cumulative_inflation_factor=1.0,
        )
        state.account_values[ai] = asset["val"]
        state.account_bases[ai] = asset["basis"]
        # H5 — scale the account's lots so their total tracks the cost basis
        # that remains after the proportional-basis sale.
        if old_basis > 0 and ai < len(state.account_lots):
            keep = max(0.0, asset["basis"]) / old_basis
            for lot in state.account_lots[ai]:
                lot["basis"] *= keep

        # Track per-year drawdown composition (Work Items 4, 5)
        net_proceeds = old_remain - remain
        state.year_non_offset_drawn += net_proceeds
        state.year_cgt_paid += tax_paid
        state.year_cgt_without_floor += tax_without_floor

    # Step 4: Fall back to implicit cash balance (no-CGT, no-basis accounting)
    if remain > 0 and state.cash_balance > 0:
        draw = min(state.cash_balance, remain)
        state.cash_balance -= draw
        remain -= draw

    return remain


# =============================================================================
# SINGLE TRIAL
# =============================================================================


class TrialResult:
    """Result of a single simulation trial at the horizon age."""

    def __init__(
        self,
        bridge: float,
        super_balances: list[float],
        mortgage_principals: list[float],
        horizon_age: int,
        min_bridge: float = 0.0,
        unmet_spending: float = 0.0,
        term_cleared: list[bool | None] | None = None,
        floor_age: int = 0,
        first_failure_age: int | None = None,
        # Per-year trajectory data (Work Items 1, 10)
        bridge_by_age: list[float] | None = None,
        mortgage_by_age: list[list[float]] | None = None,
        offset_by_age: list[list[float]] | None = None,
        # Per-year drawdown data (Work Items 4, 5)
        offset_drawn_by_age: list[float] | None = None,
        non_offset_drawn_by_age: list[float] | None = None,
        cgt_paid_by_age: list[float] | None = None,
        cgt_without_floor_by_age: list[float] | None = None,
        mortgage_rate_by_age: list[list[float]] | None = None,
        # M3 — per-trial deflators for real (today's dollars) conversion
        floor_deflator: float = 1.0,
        horizon_deflator: float = 1.0,
        # M4a — per-trial totals in today's dollars
        total_offset_drawn: float = 0.0,
        total_non_offset_drawn: float = 0.0,
        total_cgt_paid: float = 0.0,
        total_cgt_without_floor: float = 0.0,
    ) -> None:
        self.bridge = bridge
        self.super_balances = list(super_balances)
        self.mortgage_principals = list(mortgage_principals)
        self.horizon_age = horizon_age
        self.total_super = sum(super_balances)
        self.total_mortgage = sum(mortgage_principals)
        self.min_bridge = min_bridge
        self.unmet_spending = unmet_spending
        self.term_cleared = list(term_cleared) if term_cleared else []
        self.floor_age = floor_age
        self.first_failure_age = first_failure_age
        # M3 — deflators for real conversion
        self.floor_deflator = floor_deflator
        self.horizon_deflator = horizon_deflator
        # M4a — per-trial totals in today's dollars
        self.total_offset_drawn = total_offset_drawn
        self.total_non_offset_drawn = total_non_offset_drawn
        self.total_cgt_paid = total_cgt_paid
        self.total_cgt_without_floor = total_cgt_without_floor
        # Per-year trajectory
        self.bridge_by_age = list(bridge_by_age) if bridge_by_age else []
        self.mortgage_by_age = [list(y) for y in (mortgage_by_age or [])]
        self.offset_by_age = [list(y) for y in (offset_by_age or [])]
        # Per-year mortgage rates
        self.mortgage_rate_by_age = [list(y) for y in (mortgage_rate_by_age or [])]
        # Per-year drawdown composition (Work Items 4, 5)
        self.offset_drawn_by_age = list(offset_drawn_by_age) if offset_drawn_by_age else []
        self.non_offset_drawn_by_age = (
            list(non_offset_drawn_by_age) if non_offset_drawn_by_age else []
        )
        self.cgt_paid_by_age = list(cgt_paid_by_age) if cgt_paid_by_age else []
        self.cgt_without_floor_by_age = (
            list(cgt_without_floor_by_age) if cgt_without_floor_by_age else []
        )


def run_single_trial(
    household: Household,
    inputs: SimulationInputs,
    eq_returns: list[float] | None = None,
    eq_zs: list[float] | None = None,
    inf_returns: list[float] | None = None,
    windfall: float = 0.0,
) -> TrialResult:
    """Run a single stochastic simulation trial.

    Simulates from ``simulation_start_age`` to the earliest
    ``super_access_age`` across all earners (the "bridge horizon").
    Only working-year logic is used; the simulation ends when the first
    earner can access super.

    Per-earner super returns are generated independently inside
    ``simulate_working_year`` via ``generate_asset_return``, correlated
    with equity through ``eq_z``. The global Cholesky super return is
    not passed through — each earner may have a different super asset class.

    Args:
        household: The household to simulate.
        inputs: Simulation parameters.
        eq_returns: Pre-generated equity return series. If None, uses
                    deterministic mean returns.
        eq_zs: Pre-generated equity standard normal draws. If None, uses
               zero (deterministic mode).
        inf_returns: Pre-generated inflation return series for stochastic
            inflation. If None, uses fixed ``inputs.inflation``.
        windfall: One-off cash injection at simulation start (e.g. from
                  selling assets before the simulation).

    Returns:
        A ``TrialResult`` with bridge assets, per-earner super, and
        remaining mortgage principals at the bridge horizon age.

    """
    state = init_state(household, inputs.simulation_start_age)
    bridge_end_age = min(e.super_access_age for e in household.earners)
    n_years = bridge_end_age - inputs.simulation_start_age

    # Precompute account indices (invariant across all years)
    offset_idxs = _offset_indices(household)
    non_offset_idxs = _non_offset_indices(household)

    # Apply windfall to first non-offset account
    if windfall > 0 and state.account_values:
        target_idx = non_offset_idxs[0] if non_offset_idxs else 0
        state.account_values[target_idx] += windfall
        state.account_bases[target_idx] += windfall
        if target_idx < len(state.account_lots):
            state.account_lots[target_idx].append({"basis": windfall, "incurred": 0})

    min_bridge: float = float("inf")
    floor_age: int = inputs.simulation_start_age
    first_failure_age: int | None = None
    cumulative_inflation = 1.0
    # M3 — cumulative inflation factor prevailing in the year of the running
    # minimum, and the factor at the horizon (end of the last year).
    floor_deflator: float = 1.0
    horizon_deflator: float = 1.0
    # M4a — per-trial drawdown totals (today's dollars)
    total_offset_drawn = 0.0
    total_non_offset_drawn = 0.0
    total_cgt_paid = 0.0
    total_cgt_without_floor = 0.0

    # Per-year trajectory accumulators (Work Items 1, 10)
    bridge_by_age_trial: list[float] = []  # one entry per year
    mortgage_by_age_trial: list[list[float]] = []  # [year][mortgage_idx]
    offset_by_age_trial: list[list[float]] = []  # [year][offset_idx]
    offset_idxs_local = offset_idxs  # capture for use inside loop
    # Per-year drawdown accumulators (Work Items 4, 5)
    offset_drawn_by_age_trial: list[float] = []
    non_offset_drawn_by_age_trial: list[float] = []
    cgt_paid_by_age_trial: list[float] = []
    cgt_without_floor_by_age_trial: list[float] = []
    # Per-year rate trajectory
    mortgage_rate_by_age_trial: list[list[float]] = []  # [year][mortgage_idx]

    for y in range(n_years):
        age = inputs.simulation_start_age + y
        state.age = age
        state.year = y

        deterministic_mode = eq_returns is None  # True when running mean returns
        eq_r = eq_returns[y] if eq_returns else EQ_MEAN
        eq_z = eq_zs[y] if eq_zs else 0.0

        # ── Generate stochastic mortgage rates for this year ────────
        for mi, mortgage in enumerate(household.mortgages):
            if mortgage.interest_rate_stochastic:
                prev = state.current_mortgage_rates[mi]
                # Resolve None -> config default for each BK parameter.
                # Using "is not None" (not "or") so that 0.0 is a
                # valid user choice, especially for interest_rate_corr.
                theta = (
                    mortgage.interest_rate_theta
                    if mortgage.interest_rate_theta is not None
                    else BK_THETA
                )
                kappa = (
                    mortgage.interest_rate_kappa
                    if mortgage.interest_rate_kappa is not None
                    else BK_KAPPA
                )
                sigma_tilde = (
                    mortgage.interest_rate_vol
                    if mortgage.interest_rate_vol is not None
                    else BK_SIGMA_MODERATE
                )
                rho = (
                    mortgage.interest_rate_corr
                    if mortgage.interest_rate_corr is not None
                    else BK_RHO
                )
                state.current_mortgage_rates[mi] = generate_mortgage_rate(
                    prev_rate=prev,
                    eq_z=eq_z,
                    theta=theta,
                    kappa=kappa,
                    sigma_tilde=sigma_tilde,
                    rho=rho,
                )

        # Reset per-year drawdown counters (Work Items 4, 5)
        state.year_offset_drawn = 0.0
        state.year_non_offset_drawn = 0.0
        state.year_cgt_paid = 0.0
        state.year_cgt_without_floor = 0.0

        # Track cumulative inflation (stochastic or fixed)
        if inf_returns is not None:
            cumulative_inflation *= 1.0 + inf_returns[y]
        else:
            cumulative_inflation = (1 + inputs.inflation) ** y

        # Determine per-year inflation for uplifting real returns
        if inf_returns is not None:
            year_infl = inf_returns[y]
        else:
            year_infl = inputs.inflation

        # M3 — this year's cumulative deflator (factor prevailing this year)
        # and the running horizon deflator (one step further than the last
        # year's factor, matching the end-of-horizon balance).
        year_deflator = cumulative_inflation
        horizon_deflator *= 1.0 + year_infl
        # H5 — record this year's cumulative factor before the year is simulated
        # so drawdown inside the year can index lots from their incurrence date.
        state.cumulative_inflation_by_year.append(year_deflator)

        simulate_working_year(
            state,
            household,
            inputs,
            eq_r,
            eq_z=eq_z,
            deterministic=deterministic_mode,
            cumulative_inflation=cumulative_inflation,
            offset_idxs=offset_idxs,
            non_offset_idxs=non_offset_idxs,
            infl_t=year_infl,
        )

        # Include unmet spending in effective bridge: a household that
        # can't meet expenses has a bridge that is effectively negative
        # by the amount of the shortfall, even if accounts show $0.
        current_bridge = state.total_bridge - state.unmet_spending
        if current_bridge < min_bridge:
            min_bridge = current_bridge
            floor_age = age
            floor_deflator = year_deflator

        # Track first age bridge goes to zero (Work Item 2)
        if first_failure_age is None and current_bridge <= 0:
            first_failure_age = age

        # H5 — snapshot each account at the 30 June 2027 reform line. The year
        # containing 30 June 2027 is ``2027 - simulation_start_year``; its
        # market value and nominal basis become the deemed reacquisition basis
        # for any later disposal that straddles the reform.
        if y == 2027 - inputs.simulation_start_year:
            state.reform_values = list(state.account_values)
            state.reform_bases = list(state.account_bases)
            state.reform_inflation = year_deflator
            # Subdiv 112-E deemed reacquisition: from the reform line each
            # account's cost base is its market value then, incurred at that
            # year — so later indexation runs from 2027, not from year 0.
            for ai, mv in enumerate(state.account_values):
                if ai < len(state.account_lots):
                    state.account_lots[ai] = [{"basis": mv, "incurred": y}] if mv > 0 else []

        # Capture per-year trajectory in today's dollars (M3: deflate by the
        # factor prevailing in that year, so it is comparable to the
        # per-trial floor deflated by its own year).
        bridge_by_age_trial.append(current_bridge / year_deflator)
        mortgage_by_age_trial.append(list(state.mortgage_principals))
        offset_row = [state.account_values[oi] for oi in offset_idxs_local]
        offset_by_age_trial.append(offset_row)

        # Capture per-year mortgage rates
        mortgage_rate_by_age_trial.append(list(state.current_mortgage_rates))

        # Capture per-year drawdown composition in today's dollars (M4a/M3)
        offset_drawn_by_age_trial.append(state.year_offset_drawn / year_deflator)
        non_offset_drawn_by_age_trial.append(state.year_non_offset_drawn / year_deflator)
        cgt_paid_by_age_trial.append(state.year_cgt_paid / year_deflator)
        cgt_without_floor_by_age_trial.append(state.year_cgt_without_floor / year_deflator)

        # M4a — accumulate per-trial totals (today's dollars)
        total_offset_drawn += state.year_offset_drawn / year_deflator
        total_non_offset_drawn += state.year_non_offset_drawn / year_deflator
        total_cgt_paid += state.year_cgt_paid / year_deflator
        total_cgt_without_floor += state.year_cgt_without_floor / year_deflator

    # ── Term-clearance check ────────────────────────────────────────────
    term_cleared: list[bool | None] = []
    for mi, mortgage in enumerate(household.mortgages):
        if mortgage.loan_term_end_age is None:
            term_cleared.append(None)  # No term set — not checkable
        elif bridge_end_age < mortgage.loan_term_end_age:
            # Term end is beyond simulation horizon — not checkable yet
            term_cleared.append(None)
        elif state.mortgage_principals[mi] <= 0:
            term_cleared.append(True)  # Cleared by term end
        else:
            term_cleared.append(False)  # Failed: still outstanding at term end

    return TrialResult(
        bridge=state.total_bridge,
        super_balances=list(state.super_balances),
        mortgage_principals=list(state.mortgage_principals),
        horizon_age=bridge_end_age,
        min_bridge=min_bridge,
        unmet_spending=state.unmet_spending,
        term_cleared=term_cleared,
        floor_age=floor_age,
        first_failure_age=first_failure_age,
        # Per-year trajectory (Work Items 1, 10)
        bridge_by_age=bridge_by_age_trial,
        mortgage_by_age=mortgage_by_age_trial,
        offset_by_age=offset_by_age_trial,
        # Per-year drawdown composition (Work Items 4, 5)
        offset_drawn_by_age=offset_drawn_by_age_trial,
        non_offset_drawn_by_age=non_offset_drawn_by_age_trial,
        cgt_paid_by_age=cgt_paid_by_age_trial,
        cgt_without_floor_by_age=cgt_without_floor_by_age_trial,
        mortgage_rate_by_age=mortgage_rate_by_age_trial,
        # M3 — real-value deflators
        floor_deflator=floor_deflator,
        horizon_deflator=horizon_deflator,
        # M4a — per-trial totals
        total_offset_drawn=total_offset_drawn,
        total_non_offset_drawn=total_non_offset_drawn,
        total_cgt_paid=total_cgt_paid,
        total_cgt_without_floor=total_cgt_without_floor,
    )


# =============================================================================
# MONTE CARLO RUNNER
# =============================================================================


def _percentile_index(n: int, pct: float) -> int:
    """Return the index for ``pct`` in a sorted list of length ``n``.

    Single percentile definition used everywhere in the engine (M1):
    ``round(n * pct / 100)``, clamped to the last element. Previously the
    bootstrap used a floor while reportable percentiles used ``round``, so the
    standard error described a different order statistic than the value it
    annotated.
    """
    if n <= 0:
        return 0
    return min(int(round(n * pct / 100.0)), n - 1)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Percentile of an already-sorted list, using the shared definition."""
    if not sorted_values:
        return 0.0
    return sorted_values[_percentile_index(len(sorted_values), pct)]


def _bootstrap_percentile(
    values: list[float],
    pct: float,
    *,
    n_bootstrap: int = 1000,
    seed: int | None = None,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap SE and percentile interval for a percentile (M1).

    Resamples the observed ``values`` with replacement ``n_bootstrap`` times
    and returns ``(se, ci_lo, ci_hi)`` for the requested percentile.

    - Uses the same percentile definition as the reported value
      (``_percentile``) — so the SE describes the statistic it annotates.
    - Uses a dedicated, seeded ``random.Random``: reproducible for a fixed
      seed and isolated from the run's stream.
    - ``n_bootstrap`` defaults to 1000 (the SE's own relative error at 200
      resamples was ~5%).
    - Reports a two-sided percentile interval alongside the SE.

    Args:
        values: Observed values (need not be sorted).
        pct: Percentile to estimate (e.g. 50.0 for the median).
        n_bootstrap: Number of bootstrap resamples (default 1000).
        seed: Seed for the bootstrap RNG. Deterministic for a fixed seed;
            ``None`` also yields a deterministic (default-seeded) estimate.
        ci: Confidence level for the returned interval (default 0.95).

    Returns:
        ``(standard_error, interval_low, interval_high)``.

    """
    rng = random.Random(0 if seed is None else seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        sample = sorted(rng.choices(values, k=n))
        estimates.append(_percentile(sample, pct))
    se = statistics.stdev(estimates)
    estimates.sort()
    lo = _percentile(estimates, (1.0 - ci) / 2.0 * 100.0)
    hi = _percentile(estimates, (1.0 + ci) / 2.0 * 100.0)
    return se, lo, hi


@dataclass(frozen=True)
class MonteCarloResults(SimulationResults):
    """``SimulationResults`` extended with the engine's new statistics.

    ``SimulationResults`` lives in ``models.py``, which is owned by another
    worker this round, so the additional engine outputs are exposed by
    subclassing it here (additive only — every field has a default). Since a
    ``MonteCarloResults`` *is* a ``SimulationResults``, all existing consumers
    keep working; new consumers (B3/ui.py) can read the extra fields.

    M1 — bootstrap percentile intervals
        ``*_ci_lo`` / ``*_ci_hi`` accompany the existing ``*_se`` fields. All
        are computed with a single percentile definition and a seeded RNG, so
        they are reproducible for a fixed run seed.

    M4a — per-trial drawdown totals (today's dollars)
        Quantiles of the **per-trial total** drawdown components, replacing
        the UI's sum-of-per-year-medians workaround. ``*_total_p50/p5/p95``
        are true quantiles of the per-trial totals; ``cgt_floor_extra_total_*``
        is the per-trial difference between CGT with and without the 30%
        minimum-rate floor (the number the UI labels "Extra CGT").
    """

    # ── M1 — bootstrap percentile intervals ──────────────────────────────
    bridge_median_ci_lo: float | None = None
    bridge_median_ci_hi: float | None = None
    bridge_p5_ci_lo: float | None = None
    bridge_p5_ci_hi: float | None = None
    bridge_p95_ci_lo: float | None = None
    bridge_p95_ci_hi: float | None = None
    super_median_ci_lo: float | None = None
    super_median_ci_hi: float | None = None

    # ── M4a — per-trial drawdown totals (today's dollars) ────────────────
    offset_drawn_total_p5: float = 0.0
    offset_drawn_total_p50: float = 0.0
    offset_drawn_total_p95: float = 0.0
    non_offset_drawn_total_p5: float = 0.0
    non_offset_drawn_total_p50: float = 0.0
    non_offset_drawn_total_p95: float = 0.0
    cgt_paid_total_p5: float = 0.0
    cgt_paid_total_p50: float = 0.0
    cgt_paid_total_p95: float = 0.0
    cgt_without_floor_total_p5: float = 0.0
    cgt_without_floor_total_p50: float = 0.0
    cgt_without_floor_total_p95: float = 0.0
    cgt_floor_extra_total_p5: float = 0.0
    """Per-trial total CGT attributable to the 30% minimum-rate floor
    (CGT paid minus CGT without the floor), P5."""
    cgt_floor_extra_total_p50: float = 0.0
    """Median per-trial extra CGT from the 30% minimum-rate floor.

    This is a true quantile of the per-trial difference, not a difference of
    two sums of per-year medians.
    """
    cgt_floor_extra_total_p95: float = 0.0

    def summary_dict(self) -> dict[str, object]:
        """Flat dict for serialisation, including the extended fields."""
        data: dict[str, object] = dict(super().summary_dict())
        data.update(
            {
                "bridge_median_ci_lo": self.bridge_median_ci_lo,
                "bridge_median_ci_hi": self.bridge_median_ci_hi,
                "bridge_p5_ci_lo": self.bridge_p5_ci_lo,
                "bridge_p5_ci_hi": self.bridge_p5_ci_hi,
                "bridge_p95_ci_lo": self.bridge_p95_ci_lo,
                "bridge_p95_ci_hi": self.bridge_p95_ci_hi,
                "super_median_ci_lo": self.super_median_ci_lo,
                "super_median_ci_hi": self.super_median_ci_hi,
                "offset_drawn_total_p5": self.offset_drawn_total_p5,
                "offset_drawn_total_p50": self.offset_drawn_total_p50,
                "offset_drawn_total_p95": self.offset_drawn_total_p95,
                "non_offset_drawn_total_p5": self.non_offset_drawn_total_p5,
                "non_offset_drawn_total_p50": self.non_offset_drawn_total_p50,
                "non_offset_drawn_total_p95": self.non_offset_drawn_total_p95,
                "cgt_paid_total_p5": self.cgt_paid_total_p5,
                "cgt_paid_total_p50": self.cgt_paid_total_p50,
                "cgt_paid_total_p95": self.cgt_paid_total_p95,
                "cgt_without_floor_total_p5": self.cgt_without_floor_total_p5,
                "cgt_without_floor_total_p50": self.cgt_without_floor_total_p50,
                "cgt_without_floor_total_p95": self.cgt_without_floor_total_p95,
                "cgt_floor_extra_total_p5": self.cgt_floor_extra_total_p5,
                "cgt_floor_extra_total_p50": self.cgt_floor_extra_total_p50,
                "cgt_floor_extra_total_p95": self.cgt_floor_extra_total_p95,
            }
        )
        return data


def run_monte_carlo(
    household: Household,
    inputs: SimulationInputs,
    seed: int | None = None,
    near_miss_threshold: float = 0.0,
) -> MonteCarloResults:
    """Run a full Monte Carlo simulation with ``n_iterations`` independent paths.

    Each trial generates a correlated equity/super return series using
    Cholesky decomposition, then runs a single trial through the time
    horizon.

    Args:
        household: The household to simulate.
        inputs: Simulation parameters.
        seed: Optional random seed for reproducibility. If None, uses
              the system random state.
        near_miss_threshold: Dollar buffer (today's dollars) used to count
            near misses — trials whose worst bridge balance fell at or below
            this level. ``0.0`` (default) counts only outright failures.
            Exposed on the result as ``near_miss_threshold``.

    Returns:
        ``MonteCarloResults`` (a ``SimulationResults``) with sorted
        percentile statistics, bootstrap standard errors and percentile
        intervals for key percentiles, and per-trial drawdown totals.

    Raises:
        ValueError: If ``inputs.sell_strategy`` is not ``"waterfall"`` (the
            only currently implemented strategy).

    """
    if near_miss_threshold < 0:
        raise ValueError(f"near_miss_threshold must be >= 0 (got {near_miss_threshold})")
    if inputs.sell_strategy != "waterfall":
        raise ValueError(
            f"Unsupported sell_strategy '{inputs.sell_strategy}'. "
            f"Only 'waterfall' is currently implemented."
        )
    # ── RNG isolation: equity/inflation series use a dedicated RNG so that
    # scenario comparisons with the same seed share identical equity paths
    # regardless of which per-trial stochastic subsystems are active (Work
    # Item 6 / Finding F3).
    series_rng = _series_rng(seed)
    if seed is not None:
        # Per-trial processing gets a derived seed (offset from series RNG)
        random.seed(seed + 1 if isinstance(seed, int) else hash(str(seed) + "trial"))

    n_years = min(e.super_access_age for e in household.earners) - inputs.simulation_start_age
    bridge_values: list[float] = []
    min_bridge_values: list[float] = []
    super_values: list[float] = []
    per_earner_supers: list[list[float]] = []
    per_mortgage_remaining: list[list[float]] = []
    per_mortgage_term_cleared: list[list[bool | None]] = []
    # M4a — per-trial drawdown totals (today's dollars), one tuple per trial:
    # (offset drawn, non-offset drawn, CGT paid, CGT without floor).
    per_trial_totals: list[tuple[float, float, float, float]] = []

    # Track the trial with the worst running-minimum bridge (for diagnostics)
    global_floor_value: float = float("inf")
    global_floor_age: int = inputs.simulation_start_age
    global_floor_end_bridge: float = 0.0

    # Failure age collection (Work Item 2)
    failure_ages: list[int] = []

    # Per-year trajectory accumulators (Work Items 1, 10)
    # Each outer list is one year; inner lists grow to n_trials entries
    bridge_by_age_accum: list[list[float]] = [[] for _ in range(n_years)]
    mortgage_by_age_accum: list[list[list[float]]] = [
        [[] for _ in range(len(household.mortgages))] for _ in range(n_years)
    ]
    offset_by_age_accum: list[list[list[float]]] = [
        [[] for _ in range(_num_offset_accounts(household))] for _ in range(n_years)
    ]
    # Per-year drawdown accumulators (Work Items 4, 5)
    offset_drawn_accum: list[list[float]] = [[] for _ in range(n_years)]
    non_offset_drawn_accum: list[list[float]] = [[] for _ in range(n_years)]
    cgt_paid_accum: list[list[float]] = [[] for _ in range(n_years)]
    cgt_without_floor_accum: list[list[float]] = [[] for _ in range(n_years)]
    # Per-year mortgage rate accumulators (stochastic rates)
    mortgage_rate_accum: list[list[list[float]]] = [
        [[] for _ in range(len(household.mortgages))] for _ in range(n_years)
    ]

    # ── Phase 1: Generate equity/inflation series upfront using a dedicated
    # series RNG. This isolates equity paths from per-trial stochastic
    # subsystems (mortgage rates, per-earner super overrides), ensuring
    # scenario comparisons with the same seed share identical equity paths
    # (Work Item 6 / Finding F3).
    #
    # M12: the generators also return a correlated super leg. It is
    # deliberately NOT used here — super is modelled per earner inside
    # ``simulate_working_year`` as an equity/bond blend (so per-earner
    # ``super_mean_override`` / ``super_std_override`` and the glide path are
    # honoured), which gives an effective equity correlation of
    # ``growth_pct + (1 - growth_pct) * BOND_EQ_CORR`` rather than
    # ``SUPER_EQ_CORR``. The super leg is discarded (bound to ``_super_*``).
    all_series: list[tuple[list[float], list[float], list[float]]] = []
    for _ in range(inputs.n_iterations):
        eq_returns: list[float] = []
        eq_zs: list[float] = []
        inf_returns: list[float] = []
        for _y in range(n_years):
            if inputs.stochastic_inflation:
                # Generate triplet WITHOUT uplift, then uplift using realised inflation
                (eq_r, eq_z), (_super_r, _super_z), inf_r = generate_correlated_triplet(
                    rng=series_rng
                )
                # Uplift equity return using the realised inflation for this year
                eq_r_nominal = (1.0 + eq_r) * (1.0 + inf_r) - 1.0
                eq_returns.append(eq_r_nominal)
                eq_zs.append(eq_z)
                inf_returns.append(inf_r)
            else:
                eq_r, _super_r, eq_z = generate_correlated_returns(
                    rho=SUPER_EQ_CORR, return_z=True, rng=series_rng, inflation=inputs.inflation
                )
                eq_returns.append(eq_r)
                eq_zs.append(eq_z)
        all_series.append((eq_returns, eq_zs, inf_returns))

    # ── Phase 2: Run trials using pre-generated series
    for eq_returns, eq_zs, inf_returns in all_series:
        # (per-earner super returns, mortgage rates, etc. are generated
        #  inside run_single_trial using the module-level random RNG)

        result = run_single_trial(
            household=household,
            inputs=inputs,
            eq_returns=eq_returns,
            eq_zs=eq_zs,
            inf_returns=inf_returns if inputs.stochastic_inflation else None,
        )
        # M3 — convert each trial's nominal balances to today's dollars using
        # that trial's own deflators (realised inflation when stochastic).
        real_bridge = result.bridge / result.horizon_deflator
        real_min_bridge = result.min_bridge / result.floor_deflator
        bridge_values.append(real_bridge)
        min_bridge_values.append(real_min_bridge)
        super_values.append(result.total_super / result.horizon_deflator)
        per_earner_supers.append([s / result.horizon_deflator for s in result.super_balances])
        per_mortgage_remaining.append(result.mortgage_principals)
        per_mortgage_term_cleared.append(result.term_cleared)
        per_trial_totals.append(
            (
                result.total_offset_drawn,
                result.total_non_offset_drawn,
                result.total_cgt_paid,
                result.total_cgt_without_floor,
            )
        )

        # Accumulate per-year trajectory (Work Items 1, 10) — already real
        for y in range(n_years):
            if y < len(result.bridge_by_age):
                bridge_by_age_accum[y].append(result.bridge_by_age[y])
            if y < len(result.mortgage_by_age):
                for mi in range(len(result.mortgage_by_age[y])):
                    if mi < len(mortgage_by_age_accum[y]):
                        mortgage_by_age_accum[y][mi].append(result.mortgage_by_age[y][mi])
            if y < len(result.offset_by_age):
                for oi in range(len(result.offset_by_age[y])):
                    if oi < len(offset_by_age_accum[y]):
                        offset_by_age_accum[y][oi].append(result.offset_by_age[y][oi])

        # Accumulate per-year drawdown (Work Items 4, 5) — already real
        for y in range(n_years):
            if y < len(result.offset_drawn_by_age):
                offset_drawn_accum[y].append(result.offset_drawn_by_age[y])
                non_offset_drawn_accum[y].append(result.non_offset_drawn_by_age[y])
                cgt_paid_accum[y].append(result.cgt_paid_by_age[y])
                cgt_without_floor_accum[y].append(result.cgt_without_floor_by_age[y])

        # Accumulate per-year mortgage rates
        for y in range(n_years):
            if y < len(result.mortgage_rate_by_age):
                for mi in range(len(result.mortgage_rate_by_age[y])):
                    if mi < len(mortgage_rate_accum[y]):
                        mortgage_rate_accum[y][mi].append(result.mortgage_rate_by_age[y][mi])

        # Track worst running-minimum across all trials (real dollars)
        if real_min_bridge < global_floor_value:
            global_floor_value = real_min_bridge
            global_floor_age = result.floor_age
            global_floor_end_bridge = real_bridge

        # Collect failure ages for near-miss analysis (Work Item 2)
        if real_min_bridge <= 0 and result.first_failure_age is not None:
            failure_ages.append(result.first_failure_age)

    # ── Deflation to today's dollars is done per trial above (M3) ──────
    # Each trial uses its own deflator: balances at the horizon by the
    # realised horizon factor, the floor by the factor prevailing in the
    # year it occurred. This makes the fixed-inflation case bit-identical to
    # before and makes stochastic inflation properly neutral.
    bridge_values.sort()
    min_bridge_values.sort()
    super_values.sort()

    # ── Compute per-year trajectory percentiles (Work Item 1) ────────
    bridge_by_age_ages: list[int] = [inputs.simulation_start_age + y for y in range(n_years)]
    bridge_by_age_p5: list[float] = []
    bridge_by_age_p10: list[float] = []
    bridge_by_age_p25: list[float] = []
    bridge_by_age_p50: list[float] = []
    bridge_by_age_p75: list[float] = []
    bridge_by_age_p90: list[float] = []
    bridge_by_age_p95: list[float] = []

    for y in range(n_years):
        # Values are already deflated to today's dollars per trial (M3).
        vals = sorted(bridge_by_age_accum[y])

        bridge_by_age_p5.append(_percentile(vals, 5))
        bridge_by_age_p10.append(_percentile(vals, 10))
        bridge_by_age_p25.append(_percentile(vals, 25))
        bridge_by_age_p50.append(_percentile(vals, 50))
        bridge_by_age_p75.append(_percentile(vals, 75))
        bridge_by_age_p90.append(_percentile(vals, 90))
        bridge_by_age_p95.append(_percentile(vals, 95))

    # ── Compute per-year mortgage percentiles (Work Item 10) ──────────
    mortgage_by_age_ages = list(bridge_by_age_ages)
    mortgage_by_age_out: dict[str, dict[str, list[float]]] = {}
    offset_by_age_out: dict[str, dict[str, list[float]]] = {}

    for mi, mortgage in enumerate(household.mortgages):
        p5_list: list[float] = []
        p50_list: list[float] = []
        p95_list: list[float] = []
        for y in range(n_years):
            vals = (
                sorted(mortgage_by_age_accum[y][mi]) if mi < len(mortgage_by_age_accum[y]) else []
            )
            p50_list.append(_percentile(vals, 50))
            p5_list.append(_percentile(vals, 5))
            p95_list.append(_percentile(vals, 95))
        mortgage_by_age_out[mortgage.label] = {
            "p50": p50_list,
            "p5": p5_list,
            "p95": p95_list,
        }

    # ── Compute per-year offset percentiles (Work Item 10) ────────────
    offset_labels = [a.label for a in household.investment_accounts if a.is_offset]
    for oi, olabel in enumerate(offset_labels):
        p5_list = []
        p50_list = []
        p95_list = []
        for y in range(n_years):
            vals = sorted(offset_by_age_accum[y][oi]) if oi < len(offset_by_age_accum[y]) else []
            p50_list.append(_percentile(vals, 50))
            p5_list.append(_percentile(vals, 5))
            p95_list.append(_percentile(vals, 95))
        offset_by_age_out[olabel] = {
            "p50": p50_list,
            "p5": p5_list,
            "p95": p95_list,
        }

    # ── Compute per-year mortgage rate percentiles (stochastic rates) ──
    mortgage_rate_by_age_out: dict[str, dict[str, list[float]]] = {}
    for mi, mortgage in enumerate(household.mortgages):
        p5_list = []
        p50_list = []
        p95_list = []
        for y in range(n_years):
            vals = sorted(mortgage_rate_accum[y][mi]) if mi < len(mortgage_rate_accum[y]) else []
            p50_list.append(_percentile(vals, 50))
            p5_list.append(_percentile(vals, 5))
            p95_list.append(_percentile(vals, 95))
        mortgage_rate_by_age_out[mortgage.label] = {
            "p50": p50_list,
            "p5": p5_list,
            "p95": p95_list,
        }

    # ── Compute per-year drawdown composition percentiles (Items 4, 5) ──
    def _percentile_series(accum: list[list[float]], pct: int) -> list[float]:
        result_list: list[float] = []
        for y in range(n_years):
            result_list.append(_percentile(sorted(accum[y]), float(pct)))
        return result_list

    offset_drawn_p50 = _percentile_series(offset_drawn_accum, 50)
    non_offset_drawn_p50 = _percentile_series(non_offset_drawn_accum, 50)
    cgt_paid_p50 = _percentile_series(cgt_paid_accum, 50)
    cgt_without_floor_p50 = _percentile_series(cgt_without_floor_accum, 50)
    offset_drawn_p5 = _percentile_series(offset_drawn_accum, 5)
    offset_drawn_p95 = _percentile_series(offset_drawn_accum, 95)
    cgt_paid_p5 = _percentile_series(cgt_paid_accum, 5)
    cgt_paid_p95 = _percentile_series(cgt_paid_accum, 95)

    n = len(bridge_values)
    p_success = sum(1 for b in min_bridge_values if b > 0) / n

    # ── Compute per-trial drawdown totals (M4a) ─────────────────────────
    # Each component is a true quantile of the per-trial total, not a sum of
    # per-year medians. ``cgt_floor_extra`` is the per-trial difference
    # between CGT with and without the 30% minimum-rate floor.
    offset_drawn_totals = sorted(t[0] for t in per_trial_totals)
    non_offset_drawn_totals = sorted(t[1] for t in per_trial_totals)
    cgt_paid_totals = sorted(t[2] for t in per_trial_totals)
    cgt_without_floor_totals = sorted(t[3] for t in per_trial_totals)
    cgt_floor_extra_totals = sorted(t[2] - t[3] for t in per_trial_totals)

    # Per-earner median super at horizon
    per_earner_super_p50: dict[str, float] = {}
    if per_earner_supers:
        for ei, earner in enumerate(household.earners):
            values = sorted(t[ei] for t in per_earner_supers)
            per_earner_super_p50[earner.label] = _percentile(values, 50)

    # Per-mortgage median remaining at horizon
    remaining_mortgage_p50: dict[str, float] = {}
    if per_mortgage_remaining:
        for mi, mortgage in enumerate(household.mortgages):
            values = sorted(t[mi] for t in per_mortgage_remaining)
            remaining_mortgage_p50[mortgage.label] = _percentile(values, 50)

    # Per-mortgage term-clearance rates
    per_mortgage_term_cleared_pct: dict[str, float] = {}
    # Tracks mortgages with term check but horizon too short
    per_mortgage_not_checkable: dict[str, int] = {}
    horizon_age = min(e.super_access_age for e in household.earners)
    if per_mortgage_term_cleared and household.mortgages:
        for mi, mortgage in enumerate(household.mortgages):
            if mortgage.loan_term_end_age is not None and mortgage.loan_term_end_age <= horizon_age:
                # Only count trials where the check was actually possible
                checkable = [t[mi] for t in per_mortgage_term_cleared if t[mi] is not None]
                if checkable:
                    cleared = sum(1 for c in checkable if c is True)
                    per_mortgage_term_cleared_pct[mortgage.label] = cleared / len(checkable)
                else:
                    # Should not happen: term end is within horizon but no checkable trials
                    per_mortgage_term_cleared_pct[mortgage.label] = 1.0
            elif mortgage.loan_term_end_age is not None:
                # Term end is beyond horizon — entire mortgage is not checkable
                per_mortgage_not_checkable[mortgage.label] = n
        # Overall: min across mortgages that had checkable results
        if per_mortgage_term_cleared_pct:
            mortgage_term_clearance_rate = min(per_mortgage_term_cleared_pct.values())
        else:
            mortgage_term_clearance_rate = 1.0
    else:
        mortgage_term_clearance_rate = 1.0

    # ── Bootstrap standard errors and percentile intervals (M1) ─────────
    # Seeded, isolated RNG; >=1000 resamples; one percentile definition.
    bridge_mean_se = statistics.stdev(bridge_values) / math.sqrt(n)
    bridge_median_se, bridge_median_lo, bridge_median_hi = _bootstrap_percentile(
        bridge_values, 50.0, seed=seed
    )
    bridge_p5_se, bridge_p5_lo, bridge_p5_hi = _bootstrap_percentile(bridge_values, 5.0, seed=seed)
    bridge_p95_se, bridge_p95_lo, bridge_p95_hi = _bootstrap_percentile(
        bridge_values, 95.0, seed=seed
    )
    super_median_se, super_median_lo, super_median_hi = _bootstrap_percentile(
        super_values, 50.0, seed=seed
    )

    # ── Near-miss / failure depth analysis (Work Item 2, M5) ────────────
    # A near miss is a trial whose worst bridge balance fell at or below the
    # configured threshold. With the default threshold of 0.0 this counts
    # outright failures (the complement of ``p_success``); a positive
    # threshold additionally counts trials that came within that buffer.
    near_miss_count = sum(1 for b in min_bridge_values if b <= near_miss_threshold)
    near_miss_rate = near_miss_count / n
    failure_age_distribution: dict[int, int] = {}
    for age in failure_ages:
        failure_age_distribution[age] = failure_age_distribution.get(age, 0) + 1

    def p(idx: int) -> float:
        return bridge_values[min(idx, n - 1)]

    # ── Low-variance guardrail (Finding F1) ─────────────────────────
    # Warn if Monte Carlo bridge variance is suspiciously low, which may
    # indicate a near-deterministic configuration (e.g. fixed return
    # overrides silencing stochasticity on all investment accounts).
    import warnings

    bridge_cv = (
        (p(int(round(n * 0.95))) - p(int(round(n * 0.05)))) / bridge_values[n // 2]
        if bridge_values[n // 2] > 0
        else 0.0
    )
    if bridge_cv < 0.02 and n >= 100:
        warnings.warn(
            f"Monte Carlo bridge variance is extremely low "
            f"(CV={bridge_cv:.4%}). "
            f"This may indicate a near-deterministic configuration. "
            f"Check that investment accounts are using asset class returns "
            f"(interest_rate=0.0) rather than fixed-rate overrides.",
            stacklevel=2,
        )

    return MonteCarloResults(
        trials=n,
        p_success=p_success,
        bridge_mean=statistics.mean(bridge_values),
        bridge_median=_percentile(bridge_values, 50),
        bridge_p5=_percentile(bridge_values, 5),
        bridge_p10=_percentile(bridge_values, 10),
        bridge_p25=_percentile(bridge_values, 25),
        bridge_p75=_percentile(bridge_values, 75),
        bridge_p90=_percentile(bridge_values, 90),
        bridge_p95=_percentile(bridge_values, 95),
        bridge_min=bridge_values[0],
        bridge_floor=min_bridge_values[0],
        floor_age=global_floor_age,
        floor_end_bridge=global_floor_end_bridge,
        super_median=_percentile(super_values, 50),
        horizon_age=min(e.super_access_age for e in household.earners),
        per_earner_super_p50=per_earner_super_p50,
        remaining_mortgage_p50=remaining_mortgage_p50,
        mortgage_term_clearance_rate=mortgage_term_clearance_rate,
        per_mortgage_term_cleared_pct=per_mortgage_term_cleared_pct,
        per_mortgage_not_checkable=per_mortgage_not_checkable,
        seed=seed,
        bridge_mean_se=bridge_mean_se,
        bridge_median_se=bridge_median_se,
        bridge_p5_se=bridge_p5_se,
        bridge_p95_se=bridge_p95_se,
        super_median_se=super_median_se,
        # M1 — bootstrap percentile intervals
        bridge_median_ci_lo=bridge_median_lo,
        bridge_median_ci_hi=bridge_median_hi,
        bridge_p5_ci_lo=bridge_p5_lo,
        bridge_p5_ci_hi=bridge_p5_hi,
        bridge_p95_ci_lo=bridge_p95_lo,
        bridge_p95_ci_hi=bridge_p95_hi,
        super_median_ci_lo=super_median_lo,
        super_median_ci_hi=super_median_hi,
        # Work Item 2 / M5 — threshold-based near-miss / failure depth
        near_miss_rate=near_miss_rate,
        near_miss_threshold=near_miss_threshold,
        failure_age_distribution=failure_age_distribution,
        # Work Items 4, 5 — Per-year drawdown composition (today's dollars)
        offset_drawn_p50=offset_drawn_p50,
        non_offset_drawn_p50=non_offset_drawn_p50,
        offset_drawn_p5=offset_drawn_p5,
        offset_drawn_p95=offset_drawn_p95,
        cgt_paid_p50=cgt_paid_p50,
        cgt_paid_p5=cgt_paid_p5,
        cgt_paid_p95=cgt_paid_p95,
        cgt_without_floor_p50=cgt_without_floor_p50,
        # M4a — per-trial drawdown total quantiles (today's dollars)
        offset_drawn_total_p5=_percentile(offset_drawn_totals, 5),
        offset_drawn_total_p50=_percentile(offset_drawn_totals, 50),
        offset_drawn_total_p95=_percentile(offset_drawn_totals, 95),
        non_offset_drawn_total_p5=_percentile(non_offset_drawn_totals, 5),
        non_offset_drawn_total_p50=_percentile(non_offset_drawn_totals, 50),
        non_offset_drawn_total_p95=_percentile(non_offset_drawn_totals, 95),
        cgt_paid_total_p5=_percentile(cgt_paid_totals, 5),
        cgt_paid_total_p50=_percentile(cgt_paid_totals, 50),
        cgt_paid_total_p95=_percentile(cgt_paid_totals, 95),
        cgt_without_floor_total_p5=_percentile(cgt_without_floor_totals, 5),
        cgt_without_floor_total_p50=_percentile(cgt_without_floor_totals, 50),
        cgt_without_floor_total_p95=_percentile(cgt_without_floor_totals, 95),
        cgt_floor_extra_total_p5=_percentile(cgt_floor_extra_totals, 5),
        cgt_floor_extra_total_p50=_percentile(cgt_floor_extra_totals, 50),
        cgt_floor_extra_total_p95=_percentile(cgt_floor_extra_totals, 95),
        # Work Items 1, 10 — Per-year trajectory
        bridge_by_age_ages=bridge_by_age_ages,
        bridge_by_age_p5=bridge_by_age_p5,
        bridge_by_age_p10=bridge_by_age_p10,
        bridge_by_age_p25=bridge_by_age_p25,
        bridge_by_age_p50=bridge_by_age_p50,
        bridge_by_age_p75=bridge_by_age_p75,
        bridge_by_age_p90=bridge_by_age_p90,
        bridge_by_age_p95=bridge_by_age_p95,
        mortgage_by_age_ages=mortgage_by_age_ages,
        mortgage_by_age=mortgage_by_age_out,
        offset_by_age=offset_by_age_out,
        mortgage_rate_by_age=mortgage_rate_by_age_out,
    )


# =============================================================================
# SEQUENCING RISK ANALYSIS  (Work Item 3, opt-in)
# =============================================================================


@dataclass
class SequencingRiskResult:
    """Results of a sequencing risk analysis."""

    worst_first_p_success: float
    worst_first_p5: float
    best_first_p_success: float
    best_first_p5: float
    original_p_success: float
    original_p5: float

    @property
    def spread(self) -> float:
        """Difference between best-first and worst-first p_success."""
        return self.best_first_p_success - self.worst_first_p_success


def run_sequencing_analysis(
    household: Household,
    inputs: SimulationInputs,
    original_results: SimulationResults,
    seed: int | None = None,
) -> SequencingRiskResult:
    """Assess sequencing risk by reordering drawdown-period returns.

    Runs two additional passes: worst-first (reordering drawdown-period
    equity returns ascending so the worst drawdown years hit earliest)
    and best-first (descending). Working years (before the earliest
    retirement age) retain their original random order — only the
    decumulation-phase returns are reordered, since that is where
    sequencing risk matters.

    Uses **common random numbers** (H3): one seed-locked set of per-trial
    return series is generated up front, and both orderings are applied to
    that same set. The two orderings therefore differ only by the reordering
    effect, not by sampling error, and the analysis runs at the base run's
    ``n_iterations`` (no cap) so the comparison against the base is valid.

    Within-year pairing (equity return, its z-score, and that year's
    inflation) is preserved by reordering the three series together.

    Note:
        Mortgage rates are regenerated from the reordered ``eq_z`` inside
        ``run_single_trial``, so reordering also changes the joint
        equity/rate path. That is intended: the analysis asks what happens
        if a bad market sequence arrives early, and the mortgage innovation
        is correlated with equity by construction (``BK_RHO``).

    Args:
        household: The household definition.
        inputs: Simulation parameters.
        original_results: Results from the original (random order) run.
        seed: Random seed for reproducibility.

    Returns:
        SequencingRiskResult with p_success and p5 for all three orderings.

    Note:
        This is an opt-in analysis (~3× compute of a normal run).

    """
    n_years = min(e.super_access_age for e in household.earners) - inputs.simulation_start_age
    n_trials = inputs.n_iterations  # match the base run (H3)

    # ── Common random numbers (H3): one dedicated series RNG, both orderings
    # applied to the same set of per-trial series.
    series_rng = _series_rng(seed)
    # Per-trial subsystems (super overrides, mortgage rates) use the module
    # RNG; seed it consistently with the base run, and re-seed before each
    # ordering below so both orderings share identical per-trial draws too.
    trial_seed = (
        (seed + 1 if isinstance(seed, int) else hash(str(seed) + "trial"))
        if seed is not None
        else None
    )
    if trial_seed is not None:
        random.seed(trial_seed)

    base_series: list[tuple[list[float], list[float], list[float]]] = []
    for _ in range(n_trials):
        eq_returns: list[float] = []
        eq_zs: list[float] = []
        inf_returns: list[float] = []
        for _y in range(n_years):
            if inputs.stochastic_inflation:
                # M12: the correlated super leg is intentionally discarded —
                # super is rebuilt per earner as an equity/bond blend.
                (eq_r, eq_z), (_super_r, _super_z), inf_r = generate_correlated_triplet(
                    rng=series_rng
                )
                eq_r = (1.0 + eq_r) * (1.0 + inf_r) - 1.0
                inf_returns.append(inf_r)
            else:
                eq_r, _super_r, eq_z = generate_correlated_returns(
                    rho=SUPER_EQ_CORR,
                    return_z=True,
                    rng=series_rng,
                    inflation=inputs.inflation,
                )
            eq_returns.append(eq_r)
            eq_zs.append(eq_z)
        base_series.append((eq_returns, eq_zs, inf_returns))

    # Compute the drawdown start year (earliest retirement age)
    employed_retirements = [e.retirement_age for e in household.earners if e.retirement_age < 999]
    retire_age = min(employed_retirements) if employed_retirements else inputs.simulation_start_age
    drawdown_start_year = retire_age - inputs.simulation_start_age

    def _run_with_order(direction: str) -> SimulationResults:
        """Run a simulation with returns reordered worst-first or best-first."""
        # Re-seed the per-trial RNG so both orderings see identical
        # mortgage-rate / override draws (paired design, H3).
        if trial_seed is not None:
            random.seed(trial_seed)
        bridge_vals: list[float] = []
        min_vals: list[float] = []

        for eq_returns, eq_zs, inf_returns in base_series:
            # Reorder only the drawdown-period returns. Working years stay in
            # their original (random) order — the sequencing risk that matters
            # is the order of returns during the decumulation phase, when
            # assets are being sold. Reorder (equity, z, inflation) together
            # to preserve within-year correlation.
            zipped = list(
                zip(
                    eq_returns,
                    eq_zs,
                    inf_returns if inputs.stochastic_inflation else [0.0] * n_years,
                )
            )
            working = zipped[:drawdown_start_year]
            drawdown = list(zipped[drawdown_start_year:])
            if direction == "worst_first":
                drawdown.sort(key=lambda x: x[0])  # ascending: worst returns hit drawdown first
            else:
                drawdown.sort(
                    key=lambda x: x[0], reverse=True
                )  # descending: best returns hit drawdown first
            zipped = working + drawdown

            re_eq, re_z, re_inf = zip(*zipped) if zipped else ([], [], [])

            result = run_single_trial(
                household=household,
                inputs=inputs,
                eq_returns=list(re_eq),
                eq_zs=list(re_z),
                inf_returns=list(re_inf) if inputs.stochastic_inflation else None,
            )
            # Real (today's dollars) using each trial's own deflators (M3)
            bridge_vals.append(result.bridge / result.horizon_deflator)
            min_vals.append(result.min_bridge / result.floor_deflator)

        bridge_vals.sort()
        min_vals.sort()

        n = len(bridge_vals)
        p_succ = sum(1 for b in min_vals if b > 0) / n

        return SimulationResults(
            trials=n,
            p_success=p_succ,
            bridge_mean=sum(bridge_vals) / n,
            bridge_median=_percentile(bridge_vals, 50),
            bridge_p5=_percentile(bridge_vals, 5),
            bridge_p10=_percentile(bridge_vals, 10),
            bridge_p25=_percentile(bridge_vals, 25),
            bridge_p75=_percentile(bridge_vals, 75),
            bridge_p90=_percentile(bridge_vals, 90),
            bridge_p95=_percentile(bridge_vals, 95),
            bridge_min=bridge_vals[0],
            bridge_floor=min_vals[0],
            floor_age=0,
            floor_end_bridge=0.0,
            super_median=0.0,
            horizon_age=min(e.super_access_age for e in household.earners),
            seed=seed,
        )

    worst = _run_with_order("worst_first")
    best = _run_with_order("best_first")

    return SequencingRiskResult(
        worst_first_p_success=worst.p_success,
        worst_first_p5=worst.bridge_p5,
        best_first_p_success=best.p_success,
        best_first_p5=best.bridge_p5,
        original_p_success=original_results.p_success,
        original_p5=original_results.bridge_p5,
    )


# =============================================================================
# SCENARIO COMPARISON  (Work Item 6, opt-in)
# =============================================================================


@dataclass
class ScenarioComparisonResult:
    """A single scenario's result for comparison display."""

    label: str
    p_success: float
    bridge_p5: float
    bridge_median: float


def run_scenario_comparison(
    household: Household,
    inputs: SimulationInputs,
    seed: int | None = None,
    n_trials: int = 10_000,
) -> dict[str, ScenarioComparisonResult]:
    """Run dynamic scenario comparisons against the base case.

    Scenarios are built from the household's actual configuration:
    - "No PT income": only when at least one earner has PT work
    - "{label} stops working": one per earner, sets them to not_employed
    - "Expenses 10% higher": increases both base + target expenses
    - "Expenses 10% lower": decreases both

    Args:
        household: Base household definition.
        inputs: Base simulation parameters.
        seed: Random seed.
        n_trials: Trials per scenario (default 10,000).

    Returns:
        Dict mapping scenario name to ScenarioComparisonResult.

    """
    import warnings as _warnings
    from dataclasses import replace as dc_replace

    results: dict[str, ScenarioComparisonResult] = {}

    def _run(hh: Household, inp: SimulationInputs, label: str) -> None:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            r = run_monte_carlo(hh, inp, seed=seed)
        results[label] = ScenarioComparisonResult(
            label=label,
            p_success=r.p_success,
            bridge_p5=r.bridge_p5,
            bridge_median=r.bridge_median,
        )

    # ── No PT income (only if at least one earner has PT configured) ─
    if any(e.pt_days_per_week > 0 for e in household.earners):
        new_earners = tuple(dc_replace(e, pt_days_per_week=0.0) for e in household.earners)
        hh_no_pt = dc_replace(household, earners=new_earners)
        inp_no_pt = dc_replace(inputs, household=hh_no_pt, n_iterations=n_trials)
        _run(hh_no_pt, inp_no_pt, "No PT income")

    # ── One earner stops working (per earner) ───────────────────────
    for i, earner in enumerate(household.earners):
        label = f"{earner.label} stops working"
        if label in results:
            continue  # deduplicate labels
        new_e = dc_replace(
            earner,
            salary=0.0,
            sg_rate=0.0,
            employment_type="not_employed",
            self_employed_income=0.0,
        )
        earners_list = list(household.earners)
        earners_list[i] = new_e
        hh_stop = dc_replace(household, earners=tuple(earners_list))
        inp_stop = dc_replace(inputs, household=hh_stop, n_iterations=n_trials)
        _run(hh_stop, inp_stop, label)

    # ── Expenses 10% higher ─────────────────────────────────────────
    hh_high = dc_replace(
        household,
        base_living_expenses=household.base_living_expenses * 1.1,
        retirement_target=household.retirement_target * 1.1,
    )
    inp_high = dc_replace(inputs, household=hh_high, n_iterations=n_trials)
    _run(hh_high, inp_high, "Expenses 10% higher")

    # ── Expenses 10% lower ──────────────────────────────────────────
    hh_low = dc_replace(
        household,
        base_living_expenses=household.base_living_expenses * 0.9,
        retirement_target=household.retirement_target * 0.9,
    )
    inp_low = dc_replace(inputs, household=hh_low, n_iterations=n_trials)
    _run(hh_low, inp_low, "Expenses 10% lower")

    return results


# =============================================================================
# EARLIEST FEASIBLE RETIREMENT AGE  (Work Item 9, opt-in, single-earner only)
# =============================================================================

_NEVER_RETIRE = 999
"""Retirement-age sentinel meaning "never retires"."""


@dataclass
class RetirementSearchResult:
    """Result of an earliest-feasible-retirement-age search."""

    entered_age: int
    entered_p_success: float
    earliest_age: int
    earliest_p_success: float
    floor_age: int
    threshold: float
    mode: str = "both_together"
    """Search mode: "both_together" or "per_earner"."""
    target_earner_label: str | None = None
    """Label of the earner optimised (only set for per_earner mode)."""
    entered_ages_by_earner: dict[str, int] = field(default_factory=dict)
    """All earners' current retirement ages at time of search."""

    # ── H4 — failure-mode reporting and non-monotonicity detection ──────
    entered_plan_feasible: bool = True
    """True if the household's entered plan already meets the threshold
    (within the Monte Carlo tolerance)."""
    feasible: bool = True
    """True if at least one scanned age met the threshold. When False, no age
    in ``[floor_age, entered_age]`` was feasible and ``earliest_age`` is a
    sentinel (``floor_age - 1``) rather than the entered age."""
    earliest_feasible_age: int | None = None
    """Lowest scanned age meeting the threshold, or ``None`` if none did."""
    non_monotonic: bool = False
    """True if success probability was not monotone across the scanned range
    (a lower age passed while a higher age failed), so the scan cannot be
    trusted to have stopped at a clean boundary."""
    non_monotonic_ages: list[int] = field(default_factory=list)
    """The (lower, higher) age pair of the first detected inversion."""
    evaluated_ages: dict[int, float] = field(default_factory=dict)
    """age -> p_success for every age evaluated, including the entered age."""
    tolerance: float = 0.0
    """Monte Carlo noise tolerance used when classifying pass/fail."""


def run_retirement_search(
    household: Household,
    inputs: SimulationInputs,
    seed: int | None = None,
    n_trials: int = 10_000,
    min_search_age: int = 40,
    success_threshold: float | None = None,
    mode: str = "both_together",
    target_earner_index: int = 0,
) -> RetirementSearchResult:
    """Scan for the earliest retirement age meeting the success threshold.

    This is a **descending linear scan**, not a binary search: it evaluates
    every age from ``entered_age`` down to ``search_floor`` at the fixed
    ``n_trials`` sample size. The full range is evaluated (no early break)
    so a non-monotone success curve can be detected and reported rather than
    silently trusted.

    Two modes:

    ``"both_together"`` (default): the earliest age ALL earners can retire
    at simultaneously. Earners with a ``999`` (never-retire) sentinel are
    left unchanged.

    ``"per_earner"``: how early a single earner could retire while holding
    all other earners at their current ages.

    The objective is a finite-sample success *rate*, which is not guaranteed
    monotone in age (CGT and other policy effects are recomputed each year),
    so ages are classified against ``threshold - tolerance`` where
    ``tolerance`` is 1.5 binomial standard errors of the estimate.

    Args:
        household: Household definition.
        inputs: Base simulation parameters.
        seed: Random seed.
        n_trials: Trials per age evaluated (default 10,000).
        min_search_age: Lowest age to test.
        success_threshold: Minimum acceptable success probability. If
            ``None`` (default), ``inputs.success_threshold`` is used (H8a);
            passing an explicit value overrides the inputs.
        mode: "both_together" or "per_earner".
        target_earner_index: Index of earner to optimise (per_earner mode only).

    Returns:
        ``RetirementSearchResult``. ``entered_plan_feasible`` distinguishes
        "the entered plan already fails" from ``feasible`` = "no age in range
        meets the threshold"; ``non_monotonic`` flags an inverted pass/fail
        boundary. When no age is feasible, ``earliest_feasible_age`` is
        ``None`` and ``earliest_age`` is the sentinel ``floor_age - 1``.

    """
    from dataclasses import replace as dc_replace

    # H8a — honour the configured threshold unless explicitly overridden.
    if success_threshold is None:
        success_threshold = inputs.success_threshold

    # Common: capture all earners' current ages for display context
    entered_ages_by_earner = {e.label: e.retirement_age for e in household.earners}

    search_floor = max(min_search_age, inputs.simulation_start_age + 5)

    # Monte Carlo noise tolerance for classifying pass/fail (1.5 binomial SE).
    std_err = math.sqrt(max(success_threshold * (1.0 - success_threshold), 1e-12) / n_trials)
    tolerance = 1.5 * std_err
    pass_floor = success_threshold - tolerance

    def _evaluate(hh: Household) -> float:
        inp = dc_replace(inputs, household=hh, n_iterations=n_trials)
        return run_monte_carlo(hh, inp, seed=seed).p_success

    def _candidate(test_age: int) -> Household:
        if mode == "per_earner":
            new_earners = list(household.earners)
            new_earners[target_earner_index] = dc_replace(
                household.earners[target_earner_index], retirement_age=test_age
            )
            return dc_replace(household, earners=tuple(new_earners))
        # both_together: never overwrite the 999 never-retire sentinel
        updated = tuple(
            dc_replace(e, retirement_age=test_age) if e.retirement_age < _NEVER_RETIRE else e
            for e in household.earners
        )
        return dc_replace(household, earners=updated)

    if mode == "per_earner":
        target = household.earners[target_earner_index]
        entered_age = target.retirement_age
        target_label: str | None = target.label
    else:
        non_never = [
            e.retirement_age for e in household.earners if e.retirement_age < _NEVER_RETIRE
        ]
        entered_age = max(non_never) if non_never else inputs.simulation_start_age
        target_label = None

    # Baseline: the household as entered (all earners at their current ages)
    entered_p = _evaluate(household)

    # Scan the full range (no early break) to detect non-monotonicity (H4b)
    evaluated_ages: dict[int, float] = {entered_age: entered_p}
    for test_age in range(entered_age - 1, search_floor - 1, -1):
        evaluated_ages[test_age] = _evaluate(_candidate(test_age))

    # Classify pass/fail and find inversions
    ages_asc = sorted(evaluated_ages)
    passes = [evaluated_ages[a] >= pass_floor for a in ages_asc]
    passing = [a for a, ok in zip(ages_asc, passes) if ok]
    feasible = bool(passing)
    earliest_feasible: int | None = min(passing) if passing else None

    non_monotonic_ages: list[int] = []
    for i, lower_age in enumerate(ages_asc):
        if not passes[i]:
            continue
        for j in range(i + 1, len(ages_asc)):
            if not passes[j]:
                non_monotonic_ages = [lower_age, ages_asc[j]]
                break
        if non_monotonic_ages:
            break

    if earliest_feasible is not None:
        earliest_age = earliest_feasible
        earliest_p = evaluated_ages[earliest_feasible]
    else:
        # Sentinel: never report the entered age as "feasible" when it isn't
        earliest_age = search_floor - 1
        earliest_p = entered_p

    return RetirementSearchResult(
        mode=mode,
        entered_age=entered_age,
        entered_p_success=entered_p,
        earliest_age=earliest_age,
        earliest_p_success=earliest_p,
        target_earner_label=target_label,
        entered_ages_by_earner=entered_ages_by_earner,
        floor_age=search_floor,
        threshold=success_threshold,
        entered_plan_feasible=entered_p >= pass_floor,
        feasible=feasible,
        earliest_feasible_age=earliest_feasible,
        non_monotonic=bool(non_monotonic_ages),
        non_monotonic_ages=non_monotonic_ages,
        evaluated_ages=evaluated_ages,
        tolerance=tolerance,
    )
