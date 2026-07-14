# DoneByFifty — Monte Carlo Retirement Bridge Simulator

> A Monte Carlo simulator for the years between early retirement and superannuation access, with full Australian tax modelling.

DoneByFifty answers one question: can your non-super assets carry you from retirement to preservation age without running dry? It runs the bridge years — the gap between leaving work and accessing superannuation — through thousands of Monte Carlo trials and reports the odds.

---

## Motivation

Most Australian retirement calculators answer "will I have enough super?" or "can I retire at 65?" Neither is much use to someone retiring at 45 with a 15-year gap before they can touch their super.

- Treats bridge-to-super as its own phase, with its own drawdown dynamics
- Full Australian personal tax: bracket progression, Medicare Levy Surcharge, Division 293
- Post-1-July-2027 CGT reform: 30% floor rate, tax withheld on real gains only
- Household-level modelling: two earners, staggered retirements, part-time phases, kids and education costs
- Multiple mortgages (P&I or interest-only), linked offset accounts, multi-class investment accounts

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
5. Score success or failure — did the bridge assets survive to preservation age?

Results are seed-locked. Same inputs, same seed, same output, every time.

---

## Asset Return Assumptions

All returns are real (inflation-adjusted). The simulation compounds in real terms and applies a deflator at output.

| Asset Class | Real Return (μ) | Volatility (σ) | Confidence |
|---|---|---|---|
| Australian Equity | 7.0% | 15.0% | Sourced: Credit Suisse/UBS Global Investment Returns Yearbook (since 1900 ≈ 6.4–6.7%) |
| International Equity | 7.0% | 17.0% | Unsourced placeholder |
| Bonds | 3.0% | 5.0% | Sourced: RBA Bulletin (Fraser 1991, ~1.5% real to 1990); current figure sits above historical |
| Cash | 2.5% | 2.0% | Derived estimate; sits at upper edge of 1–2% range implied by sources |
| Property | 5.0% | 12.0% | Derived: CoreLogic capital growth ~3.5–4% real plus estimated net rental yield ~2–3% |
| Super (equity-like) | 7.0% | 15.0% | Unsourced placeholder |

All volatility figures are unsourced placeholders. The correlation matrix lives in `primitives.py` — equity/property 0.7, equity/bonds −0.2, etc.

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
- Historical means are not forecasts. The 7% equity assumption may be optimistic relative to current valuation-adjusted forward estimates — sensitivity-test your plan.
- Tax rules are current as at July 2026. The CGT reform is modelled as legislated for 1 July 2027. No future policy changes are assumed.
- This is a planning tool, not financial advice. Consult a licensed adviser before making decisions.
