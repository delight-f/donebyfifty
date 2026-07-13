#!/usr/bin/env python3
"""Reference regression test — compares new engine against original output.

Run from the project root:
    python -m tests.regression

This script:
1. Loads the reference family inputs through the new engine
2. Runs a multi-trial Monte Carlo simulation
3. Reports summary statistics
4. Compares against the original published results (if available)

The original deterministic single-trial values (verified by Professor of Finance):
    bridge_p50  ≈ $12,415,095.37
    super_median ≈ $10,792,545.64
    mortgage_remaining ≈ $0.00

The new engine will not match these exactly because:
- The general model applies SG (Super Guarantee) to ALL employed earners
  (the original only applied SG to Earner 1; Earner 2 was treated as
   a contractor without SG)
- The original has several implicit reference-specific assumptions
  baked into the simulation loop that the general engine deliberately
  does not replicate

The purpose of this regression test is to ensure the results remain
within a plausible range, not to pin exact values.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure the project is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    Child,
    Earner,
    Household,
    InvestmentAccount,
    MortgageAccount,
    SimulationInputs,
)
from primitives import EDU_SCHEDULE_TODAY
from simulation import run_monte_carlo, run_single_trial


def build_reference_household(
    retire_age: int = 50,
    sell_uk: bool = True,
) -> Household:
    """Replicate the original reference family's financial inputs."""
    e1 = Earner(
        label="Earner 1",
        salary=320_000.0,
        super_balance=310_720.71,
        salary_growth_rate=0.03,
        retirement_age=retire_age,
        super_access_age=60,
        sg_rate=0.12,
    )
    e2 = Earner(
        label="Earner 2",
        salary=200_000.0,
        super_balance=238_380.63,
        salary_growth_rate=0.025,
        retirement_age=retire_age,
        super_access_age=60,
        sg_rate=0.0,
        personal_super_contributions_total_p_a=32_500.0,
    )

    c1 = Child(
        label="Child 1",
        age=2,
        education_schedule=tuple(EDU_SCHEDULE_TODAY.items()),
    )

    m1 = MortgageAccount(
        label="Mortgage 1",
        principal=820_000.0,
        interest_rate=0.0605,
        monthly_payment=5_491.43,
        offset_accounts=("Offset 1",),
    )

    if sell_uk:
        offset_bal = 200_000.0 + (512_000.0 - 73_085.0)  # ~638,915
        accounts = [
            InvestmentAccount(
                label="Offset 1",
                market_value=offset_bal,
                cost_basis=offset_bal,
                asset_class="cash",
                is_offset=True,
                cgt_rate=0.0,
            ),
        ]
    else:
        accounts = [
            InvestmentAccount(
                label="Offset 1",
                market_value=200_000.0,
                cost_basis=200_000.0,
                asset_class="cash",
                is_offset=True,
                cgt_rate=0.0,
            ),
            InvestmentAccount(
                label="UK ETFs",
                market_value=512_000.0,
                cost_basis=201_000.0,
                asset_class="equity",
                tax_jurisdiction="uk",
                cgt_rate=0.30,
            ),
        ]

    accounts.append(
        InvestmentAccount(
            label="AU ETFs",
            market_value=63_000.0,
            cost_basis=63_000.0,
            asset_class="equity",
            tax_jurisdiction="au",
            cgt_rate=0.30,
        ),
    )

    return Household(
        earners=(e1, e2),
        children=(c1,),
        mortgages=(m1,),
        investment_accounts=tuple(accounts),
        base_living_expenses=75_000.0,
        retirement_target=100_000.0,
    )


def run_deterministic() -> dict:
    """Run a single deterministic trial and return key metrics."""
    h = build_reference_household(retire_age=50, sell_uk=True)
    bridge_end_age = min(e.super_access_age for e in h.earners)
    inputs = SimulationInputs(
        simulation_start_age=37,
        cgt_on_drawdowns=True,
        sell_order=("AU ETFs",),
    )

    n_years = bridge_end_age - inputs.simulation_start_age
    eq = [0.07] * n_years  # EQ_MEAN

    r = run_single_trial(h, inputs, eq_returns=eq)

    return {
        "bridge": r.bridge,
        "total_super": r.total_super,
        "total_mortgage": r.total_mortgage,
    }


def run_monte_carlo_report(n_iterations: int = 5_000) -> dict:
    """Run a full Monte Carlo simulation and return summary stats."""
    h = build_reference_household(retire_age=50, sell_uk=True)
    inputs = SimulationInputs(
        n_iterations=n_iterations,
        simulation_start_age=37,
        cgt_on_drawdowns=True,
    )

    result = run_monte_carlo(h, inputs)

    return {
        "trials": result.trials,
        "p_success": result.p_success,
        "bridge_mean": result.bridge_mean,
        "bridge_median": result.bridge_median,
        "bridge_p5": result.bridge_p5,
        "bridge_p10": result.bridge_p10,
        "bridge_p25": result.bridge_p25,
        "bridge_p75": result.bridge_p75,
        "bridge_p90": result.bridge_p90,
        "bridge_p95": result.bridge_p95,
        "bridge_min": result.bridge_min,
        "super_median": result.super_median,
        "per_earner_super": result.per_earner_super_p50,
    }


def main() -> int:
    """Run regression tests and print a report."""
    print("=" * 60)
    print("  Reference Regression Test — Monte Carlo CLI")
    print("=" * 60)

    # ── Deterministic ────────────────────────────────────────────────
    print("\n[1/2] Running deterministic single trial...")
    t0 = time.perf_counter()
    det = run_deterministic()
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s\n")

    print(f"  Bridge assets:     ${det['bridge']:>14,.2f}")
    print(f"  Total super:       ${det['total_super']:>14,.2f}")
    print(f"  Remaining mortgage: ${det['total_mortgage']:>13,.2f}")
    print()

    # Original values for reference
    print(f"  Original (ref):    ${12_415_095.37:>14,.2f}  (bridge)")
    print(f"  Original (ref):    ${10_792_545.64:>14,.2f}  (super)")
    bridge_ratio = det["bridge"] / 12_415_095.37
    super_ratio = det["total_super"] / 10_792_545.64
    print(f"  Bridge ratio:      {bridge_ratio:.2f}x  (target: 0.25–2.0x)")
    print(f"  Super ratio:       {super_ratio:.2f}x  (target: 0.25–2.0x)")

    # Bounds widened after adding super fees, blended growth/defensive super
    # allocation, tax bracket indexation, and per-earner super returns, all of
    # which systematically reduce deterministic values below the original
    # fee-free, equity-only benchmark. Super now uses 70/30 equity/bonds blend
    # (~5.8% mean) instead of 100% equity (7% mean).
    assert 0.25 <= bridge_ratio <= 2.0, f"Bridge ratio {bridge_ratio:.2f} outside [0.25, 2.0]"
    assert 0.25 <= super_ratio <= 2.0, f"Super ratio {super_ratio:.2f} outside [0.25, 2.0]"

    # ── Monte Carlo ─────────────────────────────────────────────────
    print("\n[2/2] Running Monte Carlo simulation (1,000 trials)...")
    t0 = time.perf_counter()
    mc = run_monte_carlo_report(1_000)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    print(f"  Trials:          {mc['trials']:>8,}")
    print(f"  Success rate:    {mc['p_success'] * 100:>7.1f}%")
    print()
    print("  Bridge assets:")
    print(f"    Mean:          ${mc['bridge_mean']:>14,.2f}")
    print(f"    Median:        ${mc['bridge_median']:>14,.2f}")
    print(f"    P5:            ${mc['bridge_p5']:>14,.2f}")
    print(f"    P10:           ${mc['bridge_p10']:>14,.2f}")
    print(f"    P25:           ${mc['bridge_p25']:>14,.2f}")
    print(f"    P75:           ${mc['bridge_p75']:>14,.2f}")
    print(f"    P90:           ${mc['bridge_p90']:>14,.2f}")
    print(f"    P95:           ${mc['bridge_p95']:>14,.2f}")
    print(f"    Min:           ${mc['bridge_min']:>14,.2f}")
    print()
    print(f"  Super (median):  ${mc['super_median']:>14,.2f}")
    print()
    if mc["per_earner_super"]:
        print("  Per-earner super (median):")
        for label, val in mc["per_earner_super"].items():
            print(f"    {label:15s}: ${val:>14,.2f}")

    # Sanity checks
    assert 0 < mc["p_success"] <= 1.0, f"p_success={mc['p_success']} outside (0, 1]"
    assert mc["bridge_median"] > 0, "Median bridge should be positive"
    assert mc["bridge_p5"] <= mc["bridge_median"], "P5 should be <= median"
    assert mc["bridge_median"] <= mc["bridge_p95"], "Median should be <= P95"

    print("\n" + "=" * 60)
    print("  All regression checks passed.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
