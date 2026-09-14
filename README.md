# DoneByFifty

Monte Carlo simulator for the retirement *bridge* — the gap between leaving
work and reaching preservation age, when superannuation becomes accessible.

It answers one question: can your non-super assets fund the bridge without
running dry? It runs thousands of simulated futures and reports how often the
plan fails, when it tends to break, and what drives the risk. Built for the
Australian system (super, income tax, CGT). Interactive terminal UI (Rich).
Fully offline — no network, no telemetry.

## Features

**Household**

- Up to two earners, each with an independent salary trajectory, employment
  type (employed / self-employed / both), part-time phases with custom daily
  rates and date ranges, and a staggered retirement age
- Independent super per earner: asset allocation, growth/defensive glide path,
  optional non-concessional contributions
- Children with education cost schedules
- Multiple mortgages (P&I or interest-only) with linked offset accounts and
  selectable offset reserve modes
- Investment accounts across asset classes and AU/UK tax jurisdictions
- Jointly held accounts split taxable income by ownership share (per-owner CGT)

**Australian tax and super**

- 5-bracket personal rates with annual indexation, Medicare Levy with the
  low-income reduction, LITO, tiered Medicare Levy Surcharge, Division 293
- Super: 12% SG on salary up to the max base, concessional cap with
  auto-sacrifice, 15% contributions tax, configurable fund fee drag
- CGT post-1-July-2027 reform: CPI-indexed cost basis so only real gains are
  taxed, with a 30% per-owner minimum rate floor
- CGT across the reform line: 30 Jun 2027 deemed-disposal split (50% discount
  pre-reform, indexation + 30% floor post-reform), acquisition-dated cost-base
  lots, gross-up for tax, per-owner bracket stacking

**Stochastic model**

- Correlated returns (equity, super, inflation, mortgage rate) from a single
  Cholesky decomposition
- Black–Karasinski mortgage rates: log-normal, mean-reverting, exact
  Ornstein–Uhlenbeck discretisation
- Real return inputs uplifted to nominal per year, so results are invariant to
  the inflation assumption apart from statutory nominal thresholds
- Per-subsystem seeded RNG, so enabling a stochastic feature does not change
  the equity path for a fixed seed

**Analysis**

- Success probability with a 0–100% chart and tiered risk messaging
- Bridge asset distribution: mean, median, P5–P95, running minimum
- Bootstrap standard errors (≥ 1,000 resamples) with colour-coded relative SE
- Age-by-age bridge trajectory
- Near-miss / failure-depth analysis (first age each trial crosses zero)
- Sequencing risk: worst-first vs best-first return ordering
- Drawdown source composition (offset vs non-offset funding, CGT paid, running
  totals)
- CGT breakdown
- Scenario comparison, built dynamically from the household's own config
- Earliest feasible retirement age search (scans every age; flags
  non-monotonicity and distinguishes "no feasible age" from "infeasible plan")
- Mortgage amortisation schedule

**Profiles**

- Versioned JSON, one file per profile, with atomic writes and a v1 upgrade path
- Save, load, tweak, list, delete; last-result summary stored alongside inputs

## Run on Linux Mint

Option A — prebuilt binary (no Python required). Download `montecarlo-cli` from
the latest release, then:

```bash
chmod +x montecarlo-cli
./montecarlo-cli
```

Option B — from source (Python 3.11+):

```bash
git clone git@github.com:delight-f/donebyfifty.git
cd donebyfifty
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python src/donebyfifty/main.py
```

The only runtime dependency is Rich. Modules use flat imports, so run
`src/donebyfifty/main.py` directly (or use the built binary) — do not invoke
`python -m donebyfifty`.

## Usage

Main menu: New simulation · Load profile · Manage profiles · Re-run last with
seed · Exit.

Household presets: single · couple (both working) · couple (single income) ·
family · custom.

After each run a results menu exposes the eight drill-down views above. The
expensive views (sequencing risk, scenario comparison, retirement search) are
opt-in and cached. A seed is auto-generated per run for reproducibility.

## Build

```bash
./build.sh      # Linux / macOS  -> dist/montecarlo-cli
./build.ps1     # Windows        -> dist/montecarlo-cli.exe
```

## Model notes

- All return assumptions are real (inflation-adjusted). Defaults: AU equity
  7.0% μ / 15% σ; international equity 6.5 / 16; bonds 3.0 / 6; cash 2.5 / 1;
  property 5.0 / 12; super 7.0 / 12. Volatilities are placeholders.
- Asset correlations live in `primitives.py` (equity/property 0.60,
  equity/bonds 0.10, equity/international 0.70, equity/cash 0.0) and are
  applied to log-returns, not raw returns.
- Conservative defaults: Division 293 is deducted from take-home pay; a bridge
  failure lets the simulation continue with unmet spending accumulating;
  success requires assets to stay above zero at every timestep; all values are
  reported in today's dollars.
- Seeded runs are reproducible; golden-value tests pin the engine against a
  hand calculation.

## Development

```bash
pip install -e ".[dev]"
mypy . --strict && ruff check . && black --check . && pytest tests/ -q
```

The suite (~200 tests) covers golden values, financial primitives, property
behaviour, engine integration, and regression invariants.

## Caveats

- Historical means are not forecasts. Sensitivity-test the plan.
- Tax rules are current as at July 2026; the CGT reform is modelled as
  legislated for 1 July 2027 with no allowance for further policy changes.
- Planning tool only — not a substitute for advice from a licensed adviser.

## License

See [LICENSE](LICENSE).
