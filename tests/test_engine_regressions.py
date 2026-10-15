"""Regression tests for engine defects found in the review (C1, C5, H2, H1, H3, H5, H6, M7).

Each test uses only public APIs and deterministic draws, and avoids CGT and investment
income. Cash accounts carry ``interest_rate=0.0`` so the hand calculations are exact.
Each test names the pre-fix behaviour it guards against.
"""

import dataclasses

import pytest

from donebyfifty.engine import _super_growth_pct, run_single_trial
from donebyfifty.models import (
    Earner,
    Household,
    InvestmentAccount,
    MortgageAccount,
    SimulationInputs,
)
from donebyfifty.montecarlo import generate_draws
from donebyfifty.state import DrawsBatch, horizon_years


def _inputs(inflation: float = 0.0) -> SimulationInputs:
    return SimulationInputs(
        n_iterations=2,
        simulation_start_age=60,
        inflation=inflation,
        cgt_on_drawdowns=False,
        super_fee_rate=0.0,
        bracket_growth_rate=0.0,
        conc_cap_growth_rate=0.0,
        sg_max_base_growth_rate=0.0,
        mls_enabled=False,
        simulation_start_year=2026,
    )


def _cash(value: float, label: str = "Cash", *, is_offset: bool = False) -> InvestmentAccount:
    return InvestmentAccount(
        label=label,
        market_value=value,
        cost_basis=value,
        asset_class="cash",
        is_offset=is_offset,
        interest_rate=0.0,
    )


def _idle_earner(super_access_age: int = 65) -> Earner:
    return Earner(
        label="Idle",
        salary=0.0,
        super_balance=0.0,
        salary_growth_rate=0.0,
        employment_type="not_employed",
        retirement_age=999,
        super_access_age=super_access_age,
        sg_rate=0.0,
        personal_super_contributions_total_p_a=0.0,
    )


def test_interest_only_mortgage_interest_paid_once() -> None:
    """Pre-fix: IO interest was paid in cash AND capitalised onto the balance (C1).

    Hand calc: 500,000 x 6% = 30,000 interest per year, paid in cash. Principal stays at
    500,000, so after year i (0-indexed) the cumulative cash cost is 30,000 x (i + 1).
    """
    inputs = _inputs()
    plain = Household(
        earners=(_idle_earner(),),
        investment_accounts=(_cash(1_000_000.0),),
        base_living_expenses=0.0,
        retirement_target=0.0,
    )
    io = dataclasses.replace(
        plain,
        mortgages=(
            MortgageAccount(
                label="IO", principal=500_000.0, interest_rate=0.06, monthly_payment=0.0
            ),
        ),
    )
    base = run_single_trial(plain, inputs)
    with_io = run_single_trial(io, inputs)

    assert len(with_io.bridge_by_age[0]) >= 3
    for i, (b, w) in enumerate(zip(base.bridge_by_age[0], with_io.bridge_by_age[0], strict=True)):
        assert w - b == pytest.approx(-30_000.0 * (i + 1), abs=0.01)
    for balances in with_io.mortgage_by_age[0]:
        assert balances == pytest.approx([500_000.0], abs=0.01)


def test_payoff_year_pays_only_the_remaining_balance() -> None:
    """Pre-fix: the payoff year charged 12 full monthly payments (H2).

    Hand calc (6% p.a., 0.5% per month, payment 4,000, principal 10,000):
      Month 1: interest 50.00, pays 4,000, balance 10,000 + 50 - 4,000 = 6,050.00
      Month 2: interest 30.25, pays 4,000, balance 6,050 + 30.25 - 4,000 = 2,080.25
      Month 3: interest 10.40, owed 2,080.25 + 10.40 = 2,090.65, pays all of it.
    Cash paid in year 0 = 4,000 + 4,000 + 2,090.65 = 10,090.65. No further cash is paid.
    """
    inputs = _inputs()
    plain = Household(
        earners=(_idle_earner(),),
        investment_accounts=(_cash(1_000_000.0),),
        base_living_expenses=0.0,
        retirement_target=0.0,
    )
    loan = dataclasses.replace(
        plain,
        mortgages=(
            MortgageAccount(
                label="P&I", principal=10_000.0, interest_rate=0.06, monthly_payment=4_000.0
            ),
        ),
    )
    base = run_single_trial(plain, inputs)
    with_loan = run_single_trial(loan, inputs)

    assert with_loan.bridge_by_age[0, 0] - base.bridge_by_age[0, 0] == pytest.approx(
        -10_090.65, abs=0.01
    )
    # bridge_by_age is cumulative cash, so the 10,090.65 cost persists in every later year.
    for b, w in zip(base.bridge_by_age[0], with_loan.bridge_by_age[0], strict=True):
        assert w - b == pytest.approx(-10_090.65, abs=0.01)
    assert with_loan.mortgage_by_age[0, 0] == pytest.approx([0.0], abs=0.01)


def test_offset_locked_by_reserve_floor_is_not_spendable() -> None:
    """Pre-fix: success tested total_bridge - unmet > 0, which counted the reserve-locked
    offset as spendable, so this trial succeeded (C5).

    Hand calc: no income, living expenses 80,000 in year 0. The 200,000 offset is locked by a
    200,000 reserve floor, so nothing can be drawn from it. Unmet spending is about 80,000
    plus interest, which is far above the 1e-6 tolerance, so the trial fails in year 0.
    """
    offset = _cash(200_000.0, "Offset", is_offset=True)
    mortgage = MortgageAccount(
        label="Home",
        principal=500_000.0,
        interest_rate=0.06,
        monthly_payment=0.0,
        offset_accounts=("Offset",),
        offset_reserve_mode="fixed",
        offset_reserve_floor=200_000.0,
    )
    household = Household(
        earners=(_idle_earner(super_access_age=61),),
        mortgages=(mortgage,),
        investment_accounts=(offset,),
        base_living_expenses=80_000.0,
        retirement_target=80_000.0,
    )
    result = run_single_trial(household, _inputs())

    assert result.unmet_spending[0] > 0.0
    assert not result.succeeded[0]
    assert result.first_failure_age[0] >= 0


def test_median_path_uses_inputs_inflation() -> None:
    """Pre-fix: the median path hard-coded MU_INFLATION (2.5%) and ignored inputs.inflation (H1).

    With inputs.inflation = 0.03, the default path (draws=None) must match explicit zero-shock
    draws built at 0.03. A 0.0 inflation run must give a different path, because living
    expenses and the deflator both depend on inflation.
    """
    inputs = _inputs(inflation=0.03)
    household = Household(
        earners=(_idle_earner(),),
        investment_accounts=(_cash(1_000_000.0),),
        base_living_expenses=60_000.0,
        retirement_target=60_000.0,
    )
    n_years = horizon_years(household, inputs)
    draws = DrawsBatch.median(n_years=n_years, n_earners=1, n_mortgages=0, inflation=0.03)

    implicit = run_single_trial(household, inputs)
    explicit = run_single_trial(household, inputs, draws=draws)
    zero_inflation = run_single_trial(household, _inputs(inflation=0.0))

    assert list(implicit.bridge_by_age[0]) == pytest.approx(
        list(explicit.bridge_by_age[0]), rel=1e-12
    )
    assert list(implicit.bridge_by_age[0]) != pytest.approx(
        list(zero_inflation.bridge_by_age[0]), rel=1e-9
    )


def test_non_working_partner_spending_reaches_retirement_target() -> None:
    """Pre-fix: a non-working earner (retirement_age 999) was never counted as retired, so
    spending stayed at base_living_expenses (100,000) instead of the target (H5).

    Hand calc: the only working earner retires at 60, which is the start age, so the household
    is retired from year 0. Spending is the target, 70,000. Cash starts at 1,000,000, so
    bridge_by_age[0] = 1,000,000 - 70,000 = 930,000 and each later year costs another 70,000.
    """
    partner = Earner(
        label="Partner",
        salary=100_000.0,
        super_balance=0.0,
        salary_growth_rate=0.0,
        retirement_age=60,
        super_access_age=65,
        sg_rate=0.0,
        personal_super_contributions_total_p_a=0.0,
    )
    household = Household(
        earners=(_idle_earner(), partner),
        investment_accounts=(_cash(1_000_000.0),),
        base_living_expenses=100_000.0,
        retirement_target=70_000.0,
    )
    result = run_single_trial(household, _inputs())

    assert result.bridge_by_age[0, 0] == pytest.approx(930_000.0, abs=0.01)
    assert result.bridge_by_age[0, 1] - result.bridge_by_age[0, 0] == pytest.approx(
        -70_000.0, abs=0.01
    )


def test_super_glide_path_holds_target_after_end_year() -> None:
    """Pre-fix: the glide snapped back to super_growth_pct (70) after the end year (H6).

    Hand calc: linear from 70 at year 0 to 30 at year 10, so year 5 is 70 - 40 x 5/10 = 50.
    After year 10 the target 30 holds.
    """
    earner = Earner(
        label="Glide",
        salary=0.0,
        super_balance=0.0,
        salary_growth_rate=0.0,
        super_access_age=65,
        super_growth_pct=70.0,
        super_glide_target_pct=30.0,
        super_glide_end_year=10,
    )
    assert _super_growth_pct(earner, 0) == pytest.approx(70.0)
    assert _super_growth_pct(earner, 5) == pytest.approx(50.0)
    assert _super_growth_pct(earner, 10) == pytest.approx(30.0)
    assert _super_growth_pct(earner, 15) == pytest.approx(30.0)


def test_zero_principal_mortgage_does_not_shift_equity_draws() -> None:
    """Pre-fix: mortgage shocks came from the same RNG as equity, so adding even a zero-principal
    mortgage changed the equity path (M7). Streams are now isolated.

    Hand calc: principal 0 means zero interest and zero balance, so the two runs have identical
    cash flows. Only the RNG streams can make them differ.
    """
    inputs = _inputs()
    equity = InvestmentAccount(
        label="Shares", market_value=300_000.0, cost_basis=300_000.0, asset_class="equity"
    )
    without = Household(
        earners=(_idle_earner(),),
        investment_accounts=(equity,),
        base_living_expenses=20_000.0,
        retirement_target=20_000.0,
    )
    with_mortgage = dataclasses.replace(
        without,
        mortgages=(
            MortgageAccount(label="Zero", principal=0.0, interest_rate=0.06, monthly_payment=0.0),
        ),
    )
    n_years = horizon_years(without, inputs)
    draws_without = generate_draws(without, inputs, 1, n_years, seed=42)
    draws_with = generate_draws(with_mortgage, inputs, 1, n_years, seed=42)

    a = run_single_trial(without, inputs, draws=draws_without)
    b = run_single_trial(with_mortgage, inputs, draws=draws_with)

    assert list(a.bridge_by_age[0]) == pytest.approx(list(b.bridge_by_age[0]), rel=1e-12)
    assert a.bridge[0] == pytest.approx(b.bridge[0], rel=1e-12)


def test_partner_works_on_own_age_clock() -> None:
    """Pre-fix: one shared age clock (the primary's) decided every earner's retirement, so the
    partner, who is 5 years younger, retired at year 0 instead of year 5 (H3).

    Hand calc: the partner starts at 55 and retires at 60, so years 0-4 earn a salary and year 5
    onward does not. Working years leave a surplus (take-home about 77k against 60k of spending).
    Retired years cost 60,000 each, and the cash is not interest-bearing.
    """
    partner = Earner(
        label="Partner",
        salary=100_000.0,
        super_balance=0.0,
        salary_growth_rate=0.0,
        retirement_age=60,
        super_access_age=65,
        start_age=55,
        sg_rate=0.0,
        personal_super_contributions_total_p_a=0.0,
    )
    household = Household(
        earners=(_idle_earner(super_access_age=70), partner),
        investment_accounts=(_cash(1_000_000.0),),
        base_living_expenses=60_000.0,
        retirement_target=60_000.0,
    )
    result = run_single_trial(household, _inputs())

    assert result.bridge_by_age[0, 4] - result.bridge_by_age[0, 3] > 0.0
    assert result.bridge_by_age[0, 5] - result.bridge_by_age[0, 4] == pytest.approx(
        -60_000.0, abs=0.01
    )


def test_division_293_income_includes_part_time_gross() -> None:
    """Pre-fix: Division 293 income was taxable employment income plus contributions, so part-time
    gross was left out. A self-employed earner whose part-time income pushed the total over the
    threshold paid no Division 293 tax.

    Hand calc: business income 210,000 with a fixed 30,000 personal contribution (no SG), plus
    part-time gross 5 days x 10 weeks x 1,000 = 50,000.
      Division 293 income = taxable (210,000 - 30,000) + part-time 50,000 + contributions 30,000
                          = 260,000, which is 10,000 over the 250,000 threshold.
      Excess contributions taxed = min(30,000, 10,000) = 10,000, so the tax is 15% x 10,000 = 1,500.
    Raising the threshold to 1e9 leaves tax and every other flow unchanged, so the year-0 bridge
    difference between the two runs is exactly -1,500. Pre-fix the income was 210,000, so 0.
    """
    earner = Earner(
        label="Trader",
        salary=210_000.0,
        super_balance=0.0,
        salary_growth_rate=0.0,
        employment_type="self_employed",
        retirement_age=65,
        super_access_age=65,
        pt_start_age=60,
        sg_rate=0.0,
        personal_super_contributions_total_p_a=30_000.0,
        pt_days_per_week=5.0,
        pt_weeks_per_year=10.0,
        pt_daily_rate=1_000.0,
        pt_rate_mode="daily_rate",
    )
    household = Household(
        earners=(earner,),
        investment_accounts=(_cash(1_000_000.0),),
        base_living_expenses=0.0,
        retirement_target=0.0,
    )
    inputs = _inputs()
    taxed = run_single_trial(household, inputs)
    untaxed = run_single_trial(household, dataclasses.replace(inputs, div293_threshold=1e9))

    assert taxed.bridge_by_age[0, 0] - untaxed.bridge_by_age[0, 0] == pytest.approx(
        -1_500.0, abs=0.01
    )
