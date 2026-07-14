# DoneByFifty &mdash; Monte Carlo Retirement Bridge Simulator

> Australian tax, modelled. Financial-grade reproducibility. Built for the bridge years.

DoneByFifty answers one question: *can your non-super assets carry you from retirement to preservation age without running dry?* It simulates the most exposed period in early retirement &mdash; the **bridge years** between leaving work and accessing superannuation &mdash; thousands of times over, with confidence bands.

---

## Motivation

Australian retirement calculators answer either *"will I have enough super?"* or *"can I retire at 65?"* Neither helps someone retiring at 45 with a 15-year gap to super access.

- Models **bridge-to-super** as a distinct phase with its own drawdown dynamics
- Full **Australian personal tax** with bracket progression, Medicare Levy Surcharge, and Division&nbsp;293
- **Post-1-July-2027 CGT reform** with the 30% floor rate, tax withheld on real gains only
- Household-level modelling: two earners, staggered retirements, part-time work phases, children with education costs
- Multiple mortgages (P&amp;I or interest-only), linked offset accounts, multi-class investment accounts

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

1. **New simulation** &mdash; enter household details, run Monte Carlo, explore results
2. **Load profile** &mdash; pick up where you left off
3. **Manage profiles** &mdash; list, view, or delete saved households

---

## Architecture

```
main.py          — Entry point, menu flow, profile orchestration
ui.py            — Rich TUI: prompts, result display, probability charts
simulation.py    — Core engine: Monte Carlo loop, working-years simulation, bridge phase
primitives.py    — Financial leaf functions: tax, CGT, mortgage amortization, correlated returns
models.py        — Dataclass graph: Household, Earner, Account, Mortgage, SimulationInputs
config.py        — Constants: concessional cap, Div 293 threshold, colour theming
profiles.py      — Profile load/save/delete (JSON on disk)
tests/           — pytest suite: golden values, property tests, integration tests
```

### Simulation Engine

1. **Pre-generate correlated asset returns** for all working years via Cholesky-decomposed covariance matrix
2. For each trial: simulate each working year (salary, tax, concessional contributions, surplus allocation, mortgage amortization, investment growth)
3. At retirement: snapshot the bridge portfolio
4. For bridge-only mode: project drawdown through to super access age
5. Score success/failure: did bridge assets survive to preservation age?

Results are **seed-locked** &mdash; same inputs plus same seed gives the same output, guaranteed.

---

## Asset Return Assumptions

All returns are **real** (inflation-adjusted). The simulation compounds in real terms and applies a deflator at output.

| Asset Class | Real Return (μ) | Volatility (σ) | Confidence |
|---|---|---|---|
| Australian Equity | 7.0% | 15.0% | Sourced: Credit Suisse/UBS Global Investment Returns Yearbook (since 1900 ≈ 6.4&ndash;6.7%) |
| International Equity | 7.0% | 17.0% | Unsourced placeholder |
| Bonds | 3.0% | 5.0% | Sourced: RBA Bulletin (Fraser 1991, ∼1.5% real to 1990); current figure sits above historical |
| Cash | 2.5% | 2.0% | Derived estimate; sits at upper edge of 1&ndash;2% range implied by sources |
| Property | 5.0% | 12.0% | Derived: CoreLogic capital growth ∼3.5&ndash;4% real plus estimated net rental yield ∼2&ndash;3% |
| Super (equity-like) | 7.0% | 15.0% | Unsourced placeholder |

All standard-deviation figures are unsourced placeholders. Correlation matrix is in `primitives.py` &ndash; equity/property 0.7, equity/bonds &minus;0.2, etc.

---

## Tax Model

Australian personal tax model:

| Mechanism | Implementation |
|---|---|
| **Marginal rates** | 5-bracket schedule (0%, 16%, 30%, 37%, 45%), indexed annually at configured rate |
| **Medicare Levy** | 2% on taxable income, with low-income phase-in thresholds |
| **Medicare Levy Surcharge** | Tiered 1.0%&ndash;1.5% on singles/couples by income band |
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

- **`test_primitives.py`** &mdash; tax calculations, CGT, Medicare, mortgage amortization
- **`test_models.py`** &mdash; dataclass construction, serialization, backward compatibility
- **`test_simulation.py`** &mdash; engine integration, deterministic output, scenario analysis
- **`test_golden_values.py`** &mdash; hand-calculated reference cases that lock in numerical correctness
- **`test_ui.py`** &mdash; input validation, warning generation
- **`regression.py`** &mdash; diff-based regression harness for output stability

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

- **Return assumptions are real, not nominal.** Input your expected real returns accordingly.
- **Historical means are not forecasts.** The 7% equity assumption may be optimistic relative to current valuation-adjusted forward estimates. Sensitivity-test your plan.
- **Tax rules are current as at July 2026.** The CGT reform is modelled as legislated for 1&nbsp;July&nbsp;2027. No future policy changes are anticipated.
- **This is a planning tool, not financial advice.** Consult a licensed adviser before making decisions.
