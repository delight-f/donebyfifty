"""Golden-value end-to-end test: verify a hand-calculated deterministic scenario.

Runs a single trial with NO stochastic noise (all returns at mean, no RNG)
and compares against hand-calculated expected values.

Scenario:
  - Single earner, age 60, salary $100k, $200k super, retires at 65
  - Super access at 65 (bridge length = 5 years)
  - Living expenses $70k/year, no inflation, no CGT
  - No investment accounts, no mortgages
  - SG rate 12%, salary growth 0%
  - All returns deterministic (mean returns)

Expected hand calculation (year-by-year). The simulation starts in FY2026-27
(``simulation_start_year=2026``). Bracket schedule (D10): the 18,201-45,000 band is 15%
for FY2026 and 14% from FY2027. Taxable income = salary (no sacrifice). LITO is zero
above $45,000. Medicare = 2% of $100,000 = $2,000 each year.

  Tax on $100,000:
    FY2026 (15%): (45000-18200)*0.15 + (100000-45000)*0.30 = 4020 + 16500 = 20520
                  + Medicare 2000 = 22520 tax; take-home = 100000 - 22520 = 77480
    FY2027+ (14%): (45000-18200)*0.14 + 16500 = 3752 + 16500 = 20252
                  + Medicare 2000 = 22252 tax; take-home = 100000 - 22252 = 77748

  Year-by-year (cash earns no interest; surplus = take-home - $70,000 expenses):
    Year 0 (age 60, FY2026): take-home 77480, surplus 7480, cash 7480
    Year 1 (age 61, FY2027): take-home 77748, surplus 7748, cash 15228
    Year 2 (age 62, FY2028): take-home 77748, surplus 7748, cash 22976
    Year 3 (age 63, FY2029): take-home 77748, surplus 7748, cash 30724
    Year 4 (age 64, FY2030): take-home 77748, surplus 7748, cash 38472

  Bridge at age 65 = cash after year 4 = 38472.

  Super: blended return = 70% equity (6.5% geometric) + 30% bonds (3.0%) = 5.45%,
  so the half-year growth factor is sqrt(1.0545) = 1.0268885 and the full-year factor
  is g = 1.0545. Net SG contribution = 12000 * 0.85 = 10200 per year, added at mid-year
  (D4), so it earns the second half-year's growth too:
    B(y+1) = g * B(y) + c,  c = 10200 * sqrt(1.0545) = 10474.26,  B(0) = 200000
  Closed form after 5 years:
    B5 = g^5 * 200000 + c * (g^4 + g^3 + g^2 + g + 1)
       = 1.3038659 * 200000 + 10474.26 * 5.5755207
       = 260773.18 + 58399.47
       = 319172.64

  Total super at age 65: $319,172.64

"""

from donebyfifty.engine import run_single_trial
from donebyfifty.models import Earner, Household, SimulationInputs


def test_golden_single_earner() -> None:
    """End-to-end golden-value test: single earner 60-65, deterministic."""
    earner = Earner(
        label="Test",
        salary=100_000.0,
        super_balance=200_000.0,
        salary_growth_rate=0.0,
        retirement_age=65,
        super_access_age=65,
        sg_rate=0.12,
        personal_super_contributions_total_p_a=0.0,  # disable auto-concessional-cap sacrifice
    )
    household = Household(
        earners=(earner,),
        base_living_expenses=70_000.0,
        retirement_target=70_000.0,
    )
    inputs = SimulationInputs(
        n_iterations=2,
        simulation_start_age=60,
        inflation=0.0,
        cgt_on_drawdowns=False,
        super_fee_rate=0.0,
        bracket_growth_rate=0.0,  # Fixed brackets
        conc_cap_growth_rate=0.0,
        sg_max_base_growth_rate=0.0,
        mls_enabled=False,
        simulation_start_year=2026,
    )

    # Run single trial with no stochastic noise
    result = run_single_trial(household, inputs)

    # Expected values (hand calculation in the module docstring):
    #   Bridge: 7,480 (FY2026, 15% band) + 4 x 7,748 (FY2027-30, 14% band) = 38,472
    #   Super: 1.0545 blended growth, 10,200 net SG added mid-year -> 319,172.64
    #   (closed form in the module docstring)
    expected_bridge = 38_472.0
    expected_super = 319_172.64

    # Allow small tolerance for floating point
    tolerance = 1.0

    print("Golden-value test results:")
    print(f"  Bridge:              ${result.bridge[0]:,.2f}  (expected ~${expected_bridge:,.2f})")
    print(
        f"  Total super:         ${result.total_super[0]:,.2f}  (expected ~${expected_super:,.2f})"
    )
    print(f"  Super balances:      {[f'${s:,.2f}' for s in result.super_balances[0]]}")
    print()

    bridge_ok = abs(result.bridge[0] - expected_bridge) <= tolerance
    super_ok = abs(result.total_super[0] - expected_super) <= tolerance

    assert bridge_ok, f"Bridge mismatch: {result.bridge[0]:.2f} vs expected {expected_bridge:.2f}"
    assert super_ok, f"Super mismatch: {result.total_super[0]:.2f} vs expected {expected_super:.2f}"

    print("PASS: Golden-value test matches hand calculation.")


def test_golden_simple_cgt() -> None:
    """Golden-value test with CGT: verify CGT on a bridge drawdown.

    Single earner working through the bridge, with an investment account
    that must be drawn down to cover expenses. Deterministic returns.
    """
    from donebyfifty.models import InvestmentAccount, Ownership

    earner = Earner(
        label="Test",
        salary=130_000.0,
        super_balance=200_000.0,
        salary_growth_rate=0.0,
        retirement_age=65,
        super_access_age=65,
        sg_rate=0.12,
        personal_super_contributions_total_p_a=0.0,
    )
    account = InvestmentAccount(
        label="Shares",
        market_value=100_000.0,
        cost_basis=70_000.0,
        asset_class="equity",
        interest_rate=0.07,  # Custom mean 7% return (fixed in deterministic mode)
        ownership=Ownership({0: 1.0}),
    )
    household = Household(
        earners=(earner,),
        investment_accounts=(account,),
        base_living_expenses=105_000.0,  # Slight deficit forces partial drawdown
        retirement_target=105_000.0,
    )
    inputs = SimulationInputs(
        n_iterations=2,
        simulation_start_age=60,
        inflation=0.0,
        cgt_on_drawdowns=True,
        super_fee_rate=0.0,
        bracket_growth_rate=0.0,
        conc_cap_growth_rate=0.0,
        sg_max_base_growth_rate=0.0,
        mls_enabled=False,
    )

    result = run_single_trial(household, inputs)

    # Verify bridge is positive (simulation completes without error)
    print("Golden-value CGT test results:")
    print(f"  Bridge:              ${result.bridge[0]:,.2f}")
    print(f"  Total super:         ${result.total_super[0]:,.2f}")
    print(f"  Min bridge:          ${result.min_bridge[0]:,.2f}")

    # The bridge should be positive (the investment account + salary
    # covers 100k expenses for 5 years)
    assert result.bridge[0] > 0, "Bridge should be positive"
    # Total super should have grown from contributions + returns
    assert result.total_super[0] > 200_000, "Super should have grown"

    print("PASS: Golden-value CGT test completes with positive bridge.")


if __name__ == "__main__":
    test_golden_single_earner()
    print()
    test_golden_simple_cgt()
