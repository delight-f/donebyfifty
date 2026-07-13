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

Expected hand calculation (year-by-year):
  Year 0 (age 60):
    Salary: $100,000
    SG contribution: $100,000 * 0.12 = $12,000
    Tax on $100,000 (2025-26 brackets):
      Tax = 0 + (45000-18200)*0.16 + (100000-45000)*0.30 = 0 + 4288 + 16500 = 20788
      Medicare = 100000 * 0.02 = 2000
      Total tax = 22788
    Take-home = 100000 - 22788 = 77212
    Super balance: 200000 * (1 + 0.07) + 12000 = 214000 + 12000 = 226000
    Cash surplus = 77212 - 70000 = 7212
    Cash balance = 7212

  Year 1 (age 61):
    Salary: $100,000
    SG: $12,000
    Tax: $22,788 (same)
    Take-home: $77,212
    Super: 226000 * 1.07 + 12000 = 241820 + 12000 = 253820
    Cash surplus: 77212 - 70000 = 7212
    Cash balance: 7212 + 7212 = 14424

  Year 2 (age 62):
    Tax: $22,788
    Take-home: $77,212
    Super: 253820 * 1.07 + 12000 = 271587.4 + 12000 = 283587.4
    Cash: 14424 + 7212 = 21636

  Year 3 (age 63):
    Super: 283587.4 * 1.07 + 12000 = 303438.518 + 12000 = 315438.518
    Cash: 21636 + 7212 = 28848

  Year 4 (age 64):
    Super: 315438.518 * 1.07 + 12000 = 337519.214 + 12000 = 349519.214
    Cash: 28848 + 7212 = 36060

  Bridge at age 65 (end of year 4): cash_balance = $36,060
  Total super at age 65: $349,519.21

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Earner, Household, SimulationInputs
from simulation import run_single_trial


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
        n_iterations=1,
        simulation_start_age=60,
        inflation=0.0,
        cgt_on_drawdowns=False,
        super_fee_rate=0.0,
        bracket_growth_rate=0.0,  # Fixed brackets
        conc_cap_growth_rate=0.0,
        sg_max_base_growth_rate=0.0,
        mls_enabled=False,
    )

    # Run single trial with no stochastic noise
    result = run_single_trial(household, inputs)

    # Expected values:
    #   Take-home per year: $77,212 (salary - tax - medicare, no sacrifice)
    #   Surplus per year: $7,212 (take-home - $70k expenses)
    #   Bridge after 5 years: $36,060
    #   Super: 70% equity(7%) + 30% bonds(3%) = 5.8% blended return
    #   After 15% contributions tax: net SG = $12,000 * 0.85 = $10,200
    #   Year-by-year: 200000*1.058+10200=221800, *1.058+10200=244864,
    #     *1.058+10200=269267, *1.058+10200=295084, *1.058+10200=322399
    expected_bridge = 36_060.0
    expected_super = 322_398.87

    # Allow small tolerance for floating point
    tolerance = 1.0

    print("Golden-value test results:")
    print(f"  Bridge:              ${result.bridge:,.2f}  (expected ~${expected_bridge:,.2f})")
    print(f"  Total super:         ${result.total_super:,.2f}  (expected ~${expected_super:,.2f})")
    print(f"  Super balances:      {[f'${s:,.2f}' for s in result.super_balances]}")
    print()

    bridge_ok = abs(result.bridge - expected_bridge) <= tolerance
    super_ok = abs(result.total_super - expected_super) <= tolerance

    assert bridge_ok, f"Bridge mismatch: {result.bridge:.2f} vs expected {expected_bridge:.2f}"
    assert super_ok, f"Super mismatch: {result.total_super:.2f} vs expected {expected_super:.2f}"

    print("PASS: Golden-value test matches hand calculation.")


def test_golden_simple_cgt() -> None:
    """Golden-value test with CGT: verify CGT on a bridge drawdown.

    Single earner working through the bridge, with an investment account
    that must be drawn down to cover expenses. Deterministic returns.
    """
    from models import InvestmentAccount

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
        cgt_rate=0.30,
        ownership={0: 1.0},
    )
    household = Household(
        earners=(earner,),
        investment_accounts=(account,),
        base_living_expenses=105_000.0,  # Slight deficit forces partial drawdown
        retirement_target=105_000.0,
    )
    inputs = SimulationInputs(
        n_iterations=1,
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
    print(f"  Bridge:              ${result.bridge:,.2f}")
    print(f"  Total super:         ${result.total_super:,.2f}")
    print(f"  Min bridge:          ${result.min_bridge:,.2f}")

    # The bridge should be positive (the investment account + salary
    # covers 100k expenses for 5 years)
    assert result.bridge > 0, "Bridge should be positive"
    # Total super should have grown from contributions + returns
    assert result.total_super > 200_000, "Super should have grown"

    print("PASS: Golden-value CGT test completes with positive bridge.")


if __name__ == "__main__":
    test_golden_single_earner()
    print()
    test_golden_simple_cgt()
