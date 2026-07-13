"""Reproduce the p_success / bridge_min divergence.

Demonstrates that bridge_min (table row) comes from bridge_values[0]
(year-end balance) while p_success uses min_bridge_values (running minimum).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Earner, Household, SimulationInputs
from simulation import run_monte_carlo


def test_divergence():
    """Show that bridge_min != bridge_floor when a trial dips mid-bridge."""
    e1 = Earner(
        label="Earner 1",
        salary=300_000.0,
        super_balance=300_000.0,
        salary_growth_rate=0.02,
        retirement_age=50,
        super_access_age=60,
        sg_rate=0.12,
    )
    household = Household(earners=(e1,))
    inputs = SimulationInputs(
        n_iterations=50_000,
        simulation_start_age=50,
        simulation_end_age=60,
        inflation=0.025,
        cgt_on_drawdowns=True,
    )

    results = run_monte_carlo(household, inputs, seed=42)

    print("=== Divergence Demonstration ===")
    print(f"Seed:               {results.seed}")
    print(f"Trials:             {results.trials}")
    print(f"Horizon age:        {results.horizon_age}")
    print()

    # p_success (uses running minimum -- correct)
    rounded = results.p_success * 100
    print(f"p_success:           {results.p_success:.10f}  ({rounded:.10f}%)")
    print(f"Display format:      {rounded:.1f}%")
    print()

    # The two "minimum" sources
    print(f"bridge_min  (table): {results.bridge_min:>12,.2f}  (from bridge_values[0] = year-end)")
    print(f"bridge_floor (panel):{results.bridge_floor:>12,.2f}  (from min_bridge_values[0] = running-min)")
    print()

    # Critical check
    if results.bridge_floor < 0 and results.bridge_min >= 0:
        print("*** DIVERGENCE CONFIRMED ***")
        print(f"The table shows 'Minimum' = ${results.bridge_min:,.2f} (year-end)")
        print(f"But the worst-outcome panel shows negative: ${results.bridge_floor:,.2f} (running-min)")
        print(f"p_success = {results.p_success*100:.10f}% -- the failing trial IS counted")
        print(f"  ({(1-results.p_success)*results.trials:.0f} trial(s) failed with negative running minimum)")
        print(f"Display rounds {results.p_success*100:.4f}% -> {rounded:.1f}%")
        print()
        print("ROOT CAUSE: ui.py line 1108 displays results.bridge_min")
        print("  which = bridge_values[0] (year-end minimum).")
        print("  It should display results.bridge_floor instead,")
        print("  which = min_bridge_values[0] (running-minimum minimum).")
        print("  These were split when bridge_floor was added to")
        print("  SimulationResults, but the table row was never updated.")
    print()

    # Worst-outcome details (uses bridge_floor = correct data source)
    recovered = results.floor_end_bridge > results.bridge_floor
    print(f"Floor age:          {results.floor_age}")
    print(f"Floor end bridge:   {results.floor_end_bridge:>12,.2f}")
    print(f"Floor value:        {results.bridge_floor:>12,.2f}")
    print(f"Recovered:          {recovered}")
    print(f"Horizon age:        {results.horizon_age}")
    print()

    # Verify p_success internal consistency
    print("=== Consistency check ===")
    print(f"  IF bridge_floor < 0  ({results.bridge_floor:,.2f} < 0)")
    print(f"  THEN at least 1 trial in min_bridge_values has negative running min")
    print(f"  THEN p_success = ({results.trials} - >=1) / {results.trials} = at most {((results.trials-1)/results.trials)*100:.4f}%")
    print(f"  Reported p_success: {results.p_success*100:.4f}%")
    print(f"  Consistent?         {'YES' if results.p_success < 1.0 else 'NO -- BUG'}")


if __name__ == "__main__":
    test_divergence()
