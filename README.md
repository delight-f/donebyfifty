# DoneByFifty — Monte Carlo Retirement Bridge Simulator

> A Monte Carlo simulator for the years between early retirement and superannuation access, with full Australian tax modelling.

DoneByFifty answers one question: can your non-super assets carry you from retirement to preservation age without running dry? It runs the bridge years, the gap between leaving work and accessing superannuation, through thousands of Monte Carlo trials to give you the odds.

---

## Motivation

Most Australian retirement calculators answer "will I have enough super?" or "can I retire at 65?" Neither is much help to someone retiring at 45 with a 15-year gap before they can touch their super. That gap has to be funded entirely from non-super savings, and there is no Age Pension safety net if the numbers are wrong.

DoneByFifty runs thousands of simulated futures. The output isn't just whether a bridge plan works. It shows how often it fails, when it tends to break, and what is driving the risk.

## Key Features

**Household modelling.**
- Up to two earners, each with an independent salary trajectory, employment type (employed, self-employed, or both), part-time phases with custom daily rates and date ranges, and a staggered retirement age
- Independent super per earner, with configurable asset allocations, growth/defensive glide paths, and optional non-concessional contributions
- Children with education cost schedules
- Multiple mortgages (P&I or interest-only) with linked offset accounts
- Investment accounts across asset classes and tax jurisdictions (AU/UK)
- Jointly held accounts split taxable income by ownership share for CGT purposes

**Tax and CGT.**
- Full Australian personal tax: bracket progression, Medicare Levy Surcharge, Division 293
- Post-1-July-2027 CGT reform: CPI-indexed cost basis so only real gains attract tax, plus a 30% minimum rate floor per owner
- See Mathematical Soundness below for how the reform is implemented

**Risk analysis.**
- Sequencing risk analysis re-orders return histories (worst-first versus best-first) to isolate how much return order, as distinct from return level, affects bridge viability
- Scenario comparison runs the same household under alternative assumptions, such as no part-time income or full offset drawdown, and sets the results out side by side
- Bootstrap standard errors use 200 resamples with colour-coded relative SE, so you can tell when a result needs more trials before you trust it
- Earliest-feasible-retirement-age search: binary-searches over retirement age to find the youngest age that still meets your success threshold
- Drawdown composition tracking: a per-year breakdown of offset versus non-offset funding and CGT paid, with running totals across the bridge

---

## Quick Start

```bash
# Clone and enter the repo
git clone <repo-url>
cd montecarlo-cli

# Create a venv and install (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate   # or source .venv/bin/activate on Unix
pip install -e .

# Run the interactive TUI
python main.py
```

The menu walks you through:

1. **New simulation** — enter household details, run Monte Carlo, explore results
2. **Load profile** — pick up where you left off
3. **Manage profiles** — list, view, or delete saved households

---

## Architecture

```
main.py          — Entry point, menu flow, profile orchestration
ui.py            — Rich TUI: prompts, result display, probability charts
simulation.py    — Core engine: Monte Carlo loop, working-years simulation, bridge phase
primitives.py    — Financial leaf functions: tax, CGT, mortgage amortisation, correlated returns
models.py        — Dataclass graph: Household, Earner, Account, Mortgage, SimulationInputs
config.py        — Constants: concessional cap, Div 293 threshold, colour theming
profiles.py      — Profile load/save/delete (JSON on disk)
tests/           — pytest suite: golden values, property tests, integration tests
```

### Simulation Engine

1. Pre-generate correlated asset returns for all working years via a Cholesky-decomposed covariance matrix
2. For each trial, simulate each working year: salary, tax, concessional contributions, surplus allocation, mortgage amortisation, investment growth
3. At retirement, snapshot the bridge portfolio
4. In bridge-only mode, project drawdown through to super access age
5. Score success or failure: did the bridge assets survive to preservation age?

Results are seed-locked, so a given set of inputs and seed will always reproduce the same output.

---

## Asset Return Assumptions

All returns are real (inflation-adjusted). The simulation compounds in real terms and applies a deflator at output.

| Asset Class | Real Return (μ) | Volatility (σ) | Confidence |
|---|---|---|---|
| Australian Equity | 7.0% | 15.0% | Sourced: Credit Suisse/UBS Global Investment Returns Yearbook (since 1900 ≈ 6.4–6.7%) |
| International Equity | 7.0% | 17.0% | Assumed|
| Bonds | 3.0% | 5.0% | Sourced: RBA Bulletin (Fraser 1991, ~1.5% real to 1990); current figure sits above historical |
| Cash | 2.5% | 2.0% | Derived estimate; sits at upper edge of 1–2% range implied by sources |
| Property | 5.0% | 12.0% | Derived: CoreLogic capital growth ~3.5–4% real plus estimated net rental yield ~2–3% |
| Super (equity-like) | 7.0% | 15.0% | Assumed |

All volatility figures are unsourced placeholders. The correlation matrix lives in `primitives.py`: equity/property 0.7, equity/bonds −0.2, and so on.

---

## Mathematical Soundness

### Correlated returns via Cholesky decomposition

In reality equity, super, inflation, and mortgage-rate shocks move together to some degree, and modelling them as independent understates tail risk. The correlation matrix is factorised once via Cholesky decomposition, and each year's independent standard normals are transformed through that factor to produce correlated z-scores. This preserves the equity-super correlation, the equity-inflation correlation, and, when stochastic mortgage rates are enabled, the equity-mortgage-rate correlation, all from a single decomposition rather than three separate approximations. Each year's shocks are still drawn independently of other years. The only multi-year dependency comes from the mortgage rate's own AR(1) structure, covered below.

### Black-Karasinski mortgage rates

Mortgage rates are modelled with a discrete-time Black-Karasinski process rather than a flat assumption or a simple random walk. Rates evolve log-normally and mean-revert toward a long-run level (default 6.5%, with a ~95% interval of roughly 3.7%–11.4% under moderate volatility), with mean-reversion strength κ = 0.20/year and a half-life of around 3.5 years. Log-normal dynamics rule out negative rates by construction, which a naive Gaussian model doesn't.

The implementation uses the exact discretisation of the underlying Ornstein-Uhlenbeck process in log-space, not the more common Euler approximation. Over a 10-year horizon the exact and Euler discretisations diverge by about 25.6% in persistence. That gap compounds meaningfully over multi-decade bridge simulations, which is why the exact form was used.

### CGT: indexation and the 30% floor, not indexation and the discount

The Treasury reform replaces the old 50% CGT discount with CPI-indexed cost basis, and the two mechanisms aren't applied together. Applying both would understate tax for high earners, so the model applies indexation only: earners in the 37% and 45% brackets pay their full marginal rate (subject to the 30% floor) on real gains, with no residual discount. This was verified line-by-line against the policy text rather than assumed. The CGT algorithm also handles cost-basis proportioning (only the gain fraction of each sale is taxed), gross-up for tax (solving for the pre-tax sale amount needed to net the required after-tax spending, rather than treating the after-tax figure as the sale amount), and per-owner weighted averaging for jointly held accounts.

### RNG isolation

Each stochastic subsystem draws from its own seeded generator, with a separate seed-space region. Equity and inflation series are pre-generated per trial from a dedicated generator, so switching on stochastic mortgage rates or stochastic inflation doesn't change the equity path for a given seed. A same-seed comparison with and without a given stochastic feature therefore isolates the effect of that feature, rather than conflating it with an unrelated change in the random path.

### Conservative defaults

Where policy is ambiguous or the real-world outcome depends on taxpayer behaviour, the model defaults conservatively: Division 293 tax is deducted from take-home pay rather than paid from super (the worse case for bridge viability); a bridge failure lets the simulation continue with unmet spending accumulating, rather than assuming a default or restructure; success requires bridge assets to stay above zero at every timestep, not just at the end of the horizon; and all bridge values are reported in today's dollars to avoid nominal-dollar illusion.

### Golden-value verification

Two deterministic end-to-end tests check the model against hand calculations: a single-earner scenario whose bridge and super values match independently computed figures to the dollar, and a CGT drawdown scenario that confirms the bridge stays positive through a full drawdown under known tax parameters.

---

## Tax Model

| Mechanism | Implementation |
|---|---|
| **Marginal rates** | 5-bracket schedule (0%, 16%, 30%, 37%, 45%), indexed annually at configured rate |
| **Medicare Levy** | 2% on taxable income, with low-income phase-in thresholds |
| **Medicare Levy Surcharge** | Tiered 1.0–1.5% on singles/couples by income band |
| **Division 293** | 15% additional tax on concessional contributions above $250k combined income (statutory, not indexed) |
| **Concessional cap** | $30,000 base, optionally indexed, with auto-sacrifice logic |
| **CGT (post-2027)** | 30% floor rate on real gains, tax withheld at disposal, cost-basis tracking |
| **SG** | 12% mandatory employer contribution on salary up to max base |

---

## Development

### Toolchain

```bash
pip install -e ".[dev]"
```

| Tool | Purpose | Command |
|---|---|---|
| **mypy** | Static type checking (strict mode) | `mypy .` |
| **ruff** | Linting + import sorting | `ruff check .` |
| **black** | Code formatting (line width 100) | `black .` |
| **pytest** | Test suite | `pytest tests/` |

### Testing

```bash
# Full suite (~170 tests)
pytest tests/ -q

# Exclude slow integration tests
pytest tests/ -q -k "not slow"

# Run with coverage
pytest --cov=. tests/
```

Test categories:

- **`test_primitives.py`** — tax calculations, CGT, Medicare, mortgage amortisation
- **`test_models.py`** — dataclass construction, serialisation, backward compatibility
- **`test_simulation.py`** — engine integration, deterministic output, scenario analysis
- **`test_golden_values.py`** — hand-calculated reference cases that lock in numerical correctness
- **`test_ui.py`** — input validation, warning generation
- **`regression.py`** — diff-based regression harness for output stability

### Code Quality Gates

All gates must pass before committing:

```bash
mypy . --strict && ruff check . && black --check . && pytest tests/ -q
```

- `mypy --strict`: zero errors on all production modules
- `ruff`: zero violations
- `black`: all files conform
- `pytest`: 170/171 passing (1 pre-existing flaky test unrelated to core engine)

---

## Profiles

Household configurations are saved as versioned JSON in a user-configured profiles directory. Profiles carry:

- Full household state (earners, accounts, mortgages, children)
- Simulation parameters (iterations, seed, inflation, CGT mode)
- Last result summary (success rate, bridge percentiles, failure ages)
- Version header for forward compatibility

Default profiles directory: `./profiles/` (gitignored).

## Caveats

- Return assumptions are real, not nominal. Input your expected real returns accordingly.
- Historical means are not forecasts. The 7% equity assumption may be optimistic relative to current valuation-adjusted forward estimates, so sensitivity-test your plan.
- Tax rules are current as at July 2026, and the CGT reform is modelled as legislated for 1 July 2027 with no allowance for further policy changes.
- This is a planning tool. It does not replace advice from a licensed financial adviser.
