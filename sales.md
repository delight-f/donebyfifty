# BridgeCheck — Monte Carlo Retirement Bridge Simulator

## Overview

BridgeCheck models the most financially fragile period in early retirement: the **bridge years** between leaving full-time work and gaining access to superannuation. For Australians retiring before preservation age, this gap — typically 5 to 20 years — must be funded entirely from non-super savings. Getting the numbers wrong means running out of money before super unlocks, with no Age Pension safety net during the bridge.

BridgeCheck runs thousands of simulated futures to show not just whether your bridge plan works, but how often it fails, when it breaks, and what drives the risk.

---

## Key Features

### Realistic Household Modelling

Model up to two earners with their own salary trajectories, employment types (employed, self-employed, or both), part-time work phases with custom daily rates and date ranges, and staggered retirement ages. Each earner has independent superannuation with configurable asset allocations, growth-to-defensive glide paths, and optional non-concessional contributions.

Add children with education cost schedules, multiple mortgages (principal-and-interest or interest-only) with linked offset accounts, and investment accounts across different asset classes and tax jurisdictions (AU/UK). Jointly-held accounts split CGT liability by ownership share per earner.

### Full Australian Tax Model

Per-earner tax is computed against the current marginal rate bracket system, including:

- **Medicare Levy Surcharge:** progressive tiered rates (1.0% / 1.25% / 1.5%) based on singles and couple income thresholds
- **Division 293 tax:** additional 15% on concessional contributions above the indexed threshold
- **Concessional cap tracking:** auto-sacrifice logic with annually-indexed caps and salary-growth-aware SG computation
- **Bracket creep:** tax thresholds and policy caps index annually to preserve real tax burden

### Post-2027 CGT Reform

BridgeCheck implements the Treasury Laws Amendment (Tax Reform No. 1) Act 2026:

- **CPI-indexed cost basis:** only real (above-inflation) gains attract CGT — untaxed return of capital is correctly excluded. Cumulative inflation is tracked per-year and applied to every sale transaction.
- **30% minimum rate floor:** the effective CGT rate per owner is `max(marginal_rate, 0.30)`, weighted by ownership share. This prevents high earners from deferring capital gains into low-income years.
- **Correct gross-up:** the model solves for the gross sale amount needed to net a given after-tax spending requirement, rather than naively treating the after-tax need as the pre-tax sale amount.

### Stochastic Financial Modelling

Correlated stochastic processes capture the uncertainty that matters:

- **Equity returns:** log-normal, configurable mean and volatility, correlated across asset classes
- **Super returns:** per-earner — each earner's super can be allocated across equity, bonds, cash, property, and international equity, each with its own risk/return profile
- **Mortgage rates:** Black-Karasinski mean-reverting model producing plausible rate paths that stay positive even in extreme scenarios
- **Stochastic inflation:** optionally modelled as a correlated process alongside equity and super returns, so inflation surprises are consistent with the asset return environment

### Risk Analysis Suite

- **Sequencing risk:** reorders return histories into "worst returns first" and "best returns first" orderings to isolate the impact of return *sequence* on bridge viability
- **Scenario comparison:** runs the same household under alternative assumptions (e.g., no part-time income, full offset depletion) and compares results side-by-side
- **Bootstrap standard errors:** 200-resample bootstrap with colour-coded relative standard errors so you can judge result stability: green under 2%, yellow 2–4%, orange 5–9%, red above 10%
- **Earliest feasible retirement age:** binary search over retirement age to find the youngest age that meets your success threshold
- **Drawdown composition:** per-year breakdown of offset vs non-offset funding and CGT paid, with cumulative totals

---

## Mathematical Approach

### Correlated Returns

Multi-asset correlated returns use Cholesky decomposition, the standard technique in financial simulation. A correlation matrix is factorised as LLᵀ, and correlated z-scores are computed as L·z where z is a vector of independent standard normals.

The model preserves the equity-super correlation (approximately 0.80), equity-inflation correlation, and optionally equity-mortgage-rate correlation (0.20), all from a single decomposition. Each year's shocks are drawn independently; year-to-year dependency exists only through the mortgage rate's autoregressive structure.

### Mortgage Rate Model

Mortgage interest rates follow a discrete-time Black-Karasinski model — the standard for interest rate derivatives where rates must remain positive under all realisations. The log-rate evolves as a mean-reverting process around a long-run mean θ:

> log(r_t₊₁) = log(θ) + φ · (log(r_t) − log(θ)) + σ_ε · ε_t

where φ = exp(−κ) is the persistence coefficient (κ = 0.20/year gives a half-life of approximately 3.5 years), and σ_ε is the discrete shock volatility calibrated from the continuous-time parameter.

The model uses exact discretisation (φ = e^(−κΔt)) rather than the Euler approximation (φ ≈ 1 − κΔt). This difference compounds over multi-decade bridges. Under default parameters the rates stay within a ~95% band of roughly 3.7% to 11.4%, mean-reverting to a 6.5% long-run average.

### CGT Cost-Basis Indexation

When assets are sold, the taxable gain is the sale proceeds minus the CPI-indexed cost basis — not the full proceeds. The model tracks cost bases per account and applies the cumulative inflation factor at sale time. Only the real gain fraction attracts CGT, consistent with the policy goal of neutral investment incentives.

For jointly-held accounts, the effective CGT rate is the ownership-weighted average of each earner's rate after the 30% floor: `Σ(shareᵢ · max(marginal_rateᵢ, 0.30))`.

### RNG Isolation

Equity and inflation return series are pre-generated per trial with a dedicated random number generator separate from the per-trial simulation RNG. Changing a stochastic setting (such as enabling mortgage rate volatility or stochastic inflation) does not alter equity paths for a given seed. This means "what if?" comparisons — same seed, different mortgage rate assumption — produce genuinely comparable results rather than conflating model changes with path changes.

### Conservative Defaults

Where policy is ambiguous or outcomes depend on taxpayer behaviour, the model defaults conservatively:

- **Division 293 tax** is paid from take-home income rather than released from super, maximising the cash-flow burden on the bridge
- **Bridge failure** continues the simulation rather than halting — unmet spending accumulates, producing a clean lower bound rather than assuming the household restructures or defaults
- **Success** requires bridge assets stay above zero at every timestep (running-minimum check), not just at the final horizon
- **Bridge values** are deflated to simulation-start purchasing power, avoiding nominal-dollar illusion

### Validation

Deterministic end-to-end tests verify the model against hand calculations — bridge and super values match independently-computed figures to the dollar. A CGT drawdown test confirms the drawdown completes with a positive bridge under known tax parameters. A suite of 177 automated tests covers edge cases across earner counts, mortgage types, account configurations, and tax scenarios.
