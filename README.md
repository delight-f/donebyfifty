# DoneByFifty

Monte Carlo simulator for the retirement *bridge*: the gap between leaving work
and reaching super access age, when superannuation becomes accessible.

It answers one question: can your non-super assets fund the bridge without
running dry? It runs thousands of simulated futures and reports how often the
plan fails, when it tends to break, and what drives the risk. The tax and super
rules are Australian (income tax, Medicare levy, Division 293, CGT). The
interface is a terminal UI built on Rich. It runs fully offline, with no network
access and no telemetry.

## Features

**Household**

- One or two earners, each with their own current age, employment type
  (employed, self-employed, both, or not employed), salary path, part-time phases
  with a daily rate and an age window, and an independent retirement age
- Per-earner super: asset allocation, optional age-based glidepath, and optional
  non-concessional contributions
- Children with education cost schedules
- Multiple mortgages (principal and interest, or interest-only), each linked to
  offset accounts with a selectable offset reserve mode
- Investment accounts across equity, international equity, property, bonds, and
  cash. Accounts can be jointly held, with income and capital gains split by
  ownership share
- Presets: single, couple (both working), couple (single income), family, and
  custom

**Tax and super**

- Resident income tax brackets, indexed each year, with LITO and the Medicare
  levy (including the low-income reduction for singles and families)
- Medicare levy surcharge, tiered by household income, with an option to turn it
  off
- Division 293 on concessional contributions, deducted from take-home pay
  (worst case)
- Super guarantee (12%) up to the SG maximum base, concessional cap with
  auto-sacrifice, 15% contributions tax, and a configurable fund fee
- CGT, including the 1 July 2027 reform: holdings re-based to market value at the
  reform, a 50% discount on pre-reform gains, CPI-indexed cost base with a 30%
  minimum rate on post-reform real gains, cost bases tracked by acquisition year
  until the reform, and per-owner bracket stacking
- Investment income tax: franked and unfranked dividends, with franking credits
  refundable. Income is reinvested and taxed the following year

**Stochastic model**

- Each asset class gets one shock per year, correlated with equity; equity and
  inflation are drawn jointly through a Cholesky factor
- Optional stochastic inflation, centred on your inflation input
- Black-Karasinski mortgage rates: log-normal and mean-reverting, with the
  long-run *mean* (not median) at the target rate. Optional repayment reset
  when the rate changes
- NumPy batch engine: all trials are simulated together as arrays. Each
  subsystem (equity and inflation, asset classes, super, mortgages) has its own
  seeded random stream, so enabling one feature does not change another's draws.
  A seed is shown with each result for reproducibility

**Analysis**

- Success probability, with tiered risk messaging
- Bootstrap standard errors and 95% intervals (1,000 resamples)
- Age-by-age bridge trajectory
- Near-miss and failure timing: trials whose worst real bridge balance fell
  below one year of retirement spending, and the age failures occurred
- Sequencing risk: original order vs worst-first vs best-first return ordering
- Drawdown source composition (offset vs non-offset funding, CGT, running totals)
- CGT breakdown, with and without the 30% floor
- Scenario comparison built from the household's own configuration, reported as
  paired differences with 95% confidence intervals
- Earliest feasible retirement age search
- Mortgage amortisation schedule

**Profiles**

- One JSON file per profile, written atomically, with schema versioning (v1 and
  v2 profiles upgrade to v3 on load; corrupt files are listed but not loaded)
- Save, load, tweak and re-run, and manage (list and delete). The last result is
  stored with the inputs

## Install and run

Requires Python 3.12 or newer.

```bash
git clone https://github.com/delight-f/donebyfifty.git
cd donebyfifty
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
donebyfifty            # or: python -m donebyfifty
```

Runtime dependencies are NumPy and Rich.

Profiles are stored in `$XDG_DATA_HOME/donebyfifty/profiles` (default
`~/.local/share/donebyfifty/profiles`). On Windows they are stored in
`%APPDATA%\donebyfifty\profiles`.

### Upgrading from 0.4.x

- Python 3.12 or newer is required (NumPy 2.5).
- Profiles moved out of the program directory. Copy your old
  `src/donebyfifty/profiles/*.json` (or the `profiles/` folder next to the
  executable) into the folder above; they upgrade to schema v3 on load.
- Results change materially: CGT, interest-only mortgages, inflation timing,
  return calibration and investment-income tax were corrected, so re-run saved
  plans rather than comparing against old results.

## Usage

Main menu: New simulation, Load profile, Re-run last simulation with a seed,
Manage profiles, Exit.

After each run, the results menu offers eight views:

1. Age-by-age bridge trajectory
2. Near misses and failure timing
3. Sequencing risk (opt-in, runs two extra simulations)
4. Drawdown source composition
5. CGT breakdown
6. Scenario comparison (opt-in; one extra simulation per scenario)
7. Earliest feasible retirement age (opt-in; the search runs several simulations)
8. Mortgage amortisation

Opt-in views ask for confirmation before running.

## Build a standalone executable

```bash
./build.sh      # Linux / macOS: builds dist/montecarlo-cli with PyInstaller
./build.ps1     # Windows build script (untested in CI)
```

No Python install is needed to run the built executable.

## Model notes

These are the modelling decisions the results depend on.

- **Time.** Simulation year `y` is the Australian financial year starting 1 July
  of the start year plus `y`. The start year defaults to the current financial
  year.
- **Prices.** Year-`y` expenses and flows are priced at the cumulative price
  index. Reported real figures use that index, so they are in today's dollars.
- **Returns.** All return inputs are real and geometric (compound median
  returns), with volatilities on the log-return scale.

  | Asset class | Real geometric return | Volatility |
  | --- | --- | --- |
  | Australian equity | 6.5% | 15% |
  | International equity | 6.0% | 16% |
  | Property | 5.0% | 12% |
  | Bonds | 3.0% | 6% |
  | Cash | 2.5% | 1% |

  Super returns blend Australian equity and bonds by each earner's growth
  allocation (optionally gliding to a target). A per-earner override uses its
  own mean with 12% volatility by default. The volatilities are unsourced
  placeholders. Correlations with equity are applied to log-returns: super
  override 0.80, property 0.60, international equity 0.70, bonds 0.10, cash 0.0,
  inflation −0.15.
- **Per-earner ages.** Each earner's retirement, part-time window and super
  access use that earner's own age. Ages in reports follow Earner 1. The horizon
  ends when the first earner reaches super access age.
- **Investment income.** Australian equity pays a 4.0% yield, 75% franked.
  International equity pays 2.0% and property 3.5%, both unfranked. Bonds and
  cash earn their whole positive nominal return as income. Investment income tax
  for year `y` is paid in year `y+1`.
- **Mortgage rates.** Black-Karasinski with a long-run mean of 6.5%, mean-reversion
  speed 0.20, and yearly volatility 0.18 (moderate) or 0.25 (stress). Rate shocks
  are correlated with equity at 0.20 by default.
- **Cash-flow timing.** Each year runs in order: half a year of growth, income
  and expenses, mortgage interest, offset overflow, then the second half-year of
  growth.
- **Success.** A year fails if spending goes unmet in that year, or if the
  year-end net bridge (bridge assets less cumulative unmet spending) is zero or
  below. A trial succeeds if no year fails.
- **Expenses.** The spending target blends linearly from the base figure to the
  retirement figure, weighted by the fraction of employed earners who have
  retired.
- **Retirement search.** The search scans down from the entered age and reports
  the earliest age from which every evaluated age passes the threshold. The
  per-earner mode requires the target earner to be employed.
- **Horizon.** The simulation covers the bridge only, up to the earliest super
  access age. It does not model post-bridge drawdown, the Age Pension, or
  minimum pension rules.

## Known limitations

- Division 293 is taken from take-home pay as a worst case. Super guarantee
  contributions from part-time work are not counted towards the Division 293
  test.
- When the start year is after 2027, the whole opening unrealised gain is
  treated as accrued before the CGT reform.
- Mortgage repayments stay fixed when rates move unless the repayment reset
  option is on.
- Historical means are not forecasts. Test the plan against different return and
  expense assumptions.
- Tax rules are as legislated at the time of writing. The CGT reform is modelled
  as legislated, with no allowance for further policy changes.

This is a planning tool, not a substitute for advice from a licensed adviser.

## Development

```bash
pip install -e ".[dev]"
.venv/bin/mypy src --strict && .venv/bin/ruff check . && .venv/bin/black --check . && .venv/bin/pytest tests -q
```

Coverage: `pytest tests -q --cov --cov-report=term-missing`. The coverage gate
(`fail_under`) is set in `pyproject.toml`.

The test suite covers golden values, financial primitives, tax and CGT
calculations, engine integration, profile storage, the user interface, and
regression invariants.

CI (`.github/workflows/ci.yml`) runs on pushes to `main` and on pull requests. It
runs lint (ruff and black), types (mypy strict), tests with coverage on Ubuntu
(Python 3.12 and 3.13) and Windows (Python 3.12), and a Linux executable build.

## License

See [LICENSE](LICENSE).
