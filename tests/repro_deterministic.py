"""Run the de Belder profile in deterministic mode - year by year."""
import sys, json
sys.path.insert(0, ".")

from models import _deserialise_inputs, _deserialise_child, _deserialise_mortgage, _deserialise_account
from models import Household, Earner, SimulationInputs
import simulation as simmod

# ── Load profile ──────────────────────────────────────────────────────
with open("dist/profiles/de belder.json") as f:
    prof = json.load(f)
d = prof["inputs"]

# ── Use proper deserialisation ────────────────────────────────────────
inputs = _deserialise_inputs(d)

# Build household
earners = []
for ed in d["household"]["earners"]:
    # Map from profile dict to Earner using same pattern as _deserialise_earner
    from models import _deserialise_earner
    e = _deserialise_earner(ed)
    earners.append(e)

children = [_deserialise_child(c) for c in d["household"]["children"]]
mortgages = [_deserialise_mortgage(m) for m in d["household"]["mortgages"]]
accounts = [_deserialise_account(a) for a in d["household"]["investment_accounts"]]

household = Household(
    earners=earners,
    mortgages=mortgages,
    investment_accounts=accounts,
    children=children,
    base_living_expenses=d["household"]["base_living_expenses"],
    retirement_target=d["household"]["retirement_target"],
)

# Check ownership keys
for i, a in enumerate(household.investment_accounts):
    print(f"Account {i}: ownership keys = {list(a.ownership.keys())}")

# ── Run deterministic single trial ────────────────────────────────────
result = simmod.run_single_trial(household, inputs)
print()
print("===== DETERMINISTIC SINGLE TRIAL =====")
print(f"Bridge end balance: ${result.bridge:,.2f}")
print(f"Min bridge:         ${result.min_bridge:,.2f}")
print(f"Total super:        ${sum(result.super_balances):,.2f}")
print(f"Earner 1 super:     ${result.super_balances[0]:,.2f}")
print(f"Earner 2 super:     ${result.super_balances[1]:,.2f}")
print(f"Mortgage remaining: {result.mortgage_principals}")
print(f"Account values:     {[f'${v:,.2f}' for v in result.account_values]}")
print()

# ── Year-by-year ──────────────────────────────────────────────────────
bridge_end = min(e.super_access_age for e in household.earners)
n_years = bridge_end - inputs.simulation_start_age

state = simmod.init_state(household, inputs.simulation_start_age)
offset_idxs = [i for i, a in enumerate(household.investment_accounts) if a.is_offset]
non_offset_idxs = [i for i, a in enumerate(household.investment_accounts) if not a.is_offset]

cumulative_inflation = 1.0

print(f"Bridge: age {inputs.simulation_start_age} -> {bridge_end} ({n_years} years)")
print(f"Accounts initial: {[f'${v:,.0f}' for v in state.account_values]}")
print(f"Mortgage initial: ${state.mortgage_principals[0]:,.0f}")
print(f"Total bridge:    ${state.total_bridge:,.0f}")
print(f"Super:           {[f'${v:,.0f}' for v in state.super_balances]}")
print()

for y in range(n_years):
    age = inputs.simulation_start_age + y
    eq_ret = 0.07
    
    shortfall = simmod.simulate_working_year(
        state, household, inputs,
        eq_return=eq_ret, eq_z=0.0,
        deterministic=True,
        cumulative_inflation=cumulative_inflation,
        offset_idxs=offset_idxs, non_offset_idxs=non_offset_idxs,
    )
    cumulative_inflation *= (1 + inputs.inflation)
    
    # Summarise key years
    if age <= 50 or age >= 58 or age % 5 == 0:
        acct_vals = ", ".join(f"${v:,.0f}" for v in state.account_values)
        print(f"Age {age:>2}: bridge=${state.total_bridge:>10,.0f}  shortfall={shortfall:>8,.0f}  "
              f"accts=[{acct_vals}]  mtg=${state.mortgage_principals[0]:>8,.0f}  "
              f"super=[${state.super_balances[0]:>8,.0f}, ${state.super_balances[1]:>8,.0f}]")

print(f"\n=== FINAL ===")
print(f"Bridge: ${state.total_bridge:,.2f}")
print(f"Matches run_single_trial: ${result.bridge:,.2f}")
