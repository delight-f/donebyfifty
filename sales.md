# DoneByFifty — Monte Carlo Retirement Bridge Simulator

## Overview

DoneByFifty models the most financially fragile period in early retirement: the **bridge years** between leaving full-time work and gaining access to superannuation. For Australians retiring before preservation age, this gap — typically 5 to 20 years — must be funded entirely from non-super savings. Getting the numbers wrong means running out of money before super unlocks, with no Age Pension safety net during the bridge.

DoneByFifty runs thousands of simulated futures to tell you not just *whether* your bridge plan works, but *how often* it fails, *when* it breaks, and *what drives* the risk.


## Key Features

### Realistic Household Modelling

Model up to two earners with distinct salary trajectories, employment types (employed, self-employed, or both), part-time work phases with custom daily rates and date ranges, and staggered retirement ages. Each earner has independent superannuation — with configurable asset allocations, growth/defensive glide paths, and optional non-concessional contributions.

Add children with education cost schedules, multiple mortgages (P&I or interest-only) with linked offset accounts, and multiple investment accounts across different asset classes and tax jurisdictions (AU/UK). Accounts can be jointly held with per-owner taxable-income splitting for CGT.

### Full Australian Tax Model

The simulator computes per-earner tax using the current marginal rate bracket system, including:

- **Progressive Medicare Levy Surcharge:** tiered rates (1.0% / 1.25% / 1.5%) based on singles or couple income thresholds

- **Division 293 tax:** additional 15% on concessional contributions above $250,000, with annual indexation

- **Concessional cap tracking:** auto-sacrifice logic with indexed caps and salary-growth-aware SG computation

- **Bracket creep:** tax thresholds index annually to preserve real tax burden

### Post-2027 CGT Reform (Fully Implemented)

DoneByFifty models the 2026 Treasury Laws Amendment reforms:

- **CPI-indexed cost basis:** only real (above-inflation) gains attract CGT — untaxed return of capital is correctly excluded, matching the policy intent of neutral investment incentives

- **30% minimum rate floor:** `max(marginal\_rate, 0.30)` per owner, weighted by ownership share, correctly preventing high earners from deferring gains into low-income years

- **Per-year inflation tracking:** cumulative inflation factor applied to every sell transaction, not just annualised

### Stochastic Financial Modelling

DoneByFifty uses correlated stochastic processes to capture real-world uncertainty:

- **Equity returns:** log-normal with configurable mean (7% default) and volatility (15% default), correlated across asset classes via Cholesky decomposition

- **Super returns:** per-earner asset class returns (equity, bonds, cash, property, international equity), each with its own risk/return profile, computed from a shared equity z-score to preserve the equity-super correlation structure

- **Mortgage rates:** Black-Karasinski mean-reverting log-normal model (see Mathematical Soundness section) for stochastic interest rate paths

- **Inflation:** configurable as either a flat assumption or a correlated stochastic process (3-way Cholesky with equity and super), so inflation surprises are not independent of asset returns

### Risk Analysis Suite

- **Sequencing risk analysis:** re-orders return histories to simulate "worst returns first" vs "best returns first" scenarios, isolating the impact of return order on the bridge's viability independent of the return *level*

- **Scenario comparison:** runs the same household under alternative assumptions (e.g., no part-time income, full offset drawdown) and compares key metrics side-by-side

- **Bootstrap standard errors:** 200-resample bootstrap with color-coded relative SE (%): green \<2%, yellow 2–4%, orange 5–9%, red ≥10% — so you know when to increase trial count or report results with caution

- **Earliest feasible retirement age:** binary search over retirement age to find the youngest age that achieves a user-defined success threshold

- **Drawdown composition tracking:** per-year breakdown of offset vs non-offset funding sources and CGT paid, with cumulative totals across the bridge


## Mathematical Soundness

### Correlated Return Generation

DoneByFifty generates multi-asset correlated returns using Cholesky decomposition, the standard approach in financial engineering for producing correlated Gaussian draws from independent normals. The correlation matrix is factorised once as L L^T $L L^T$, and correlated z-scores are obtained as $L \\cdot \\vec\{z\}$ where $\\vec\{z\}$ is a vector of independent standard normals.

The model preserves the equity-super correlation ($\\rho \\approx 0.80$ by default), the equity-inflation correlation, and optionally the equity-mortgage-rate correlation ($\\rho = 0.20$ by default), all from a single Cholesky decomposition. Each year's shocks are drawn independently; multi-year dependency arises only through the mortgage rate's AR(1) structure.

### Black-Karasinski Mortgage Rate Model

Mortgage interest rates follow a discrete-time Black-Karasinski (BK) model, a standard in interest rate derivatives for its ability to prevent negative rates via log-normal dynamics:

ln(r\_(t+1)) = ln(θ) + φ(ln(r\_t) − ln(θ)) + σ\_ε · ε\_t 

where φ = e^(−κ)  is the exact AR(1) coefficient (mean-reversion strength $\\kappa$, default 0.20/year, half-life ≈ 3.5 years), and σ\_ε = σ̃ √((1 − φ²) / (2κ)) is the discrete shock volatility calibrated to produce a stationary distribution consistent with the continuous-time parameter $\\tilde\{\\sigma\}$.

The model uses the **exact discretisation** of the OU process in log-space, not the Euler approximation ($\\phi \\approx 1 - \\kappa$). Over a 10-year horizon, exact persistence is $e^\{-10\\kappa\} = 0.135$ vs Euler $0.107$ — a 25.6% difference that compounds over multi-decade bridges. The stationary distribution has mean $\\theta$ (default 6.5%) with a ~95% interval of approximately \[3.7%, 11.4%\] under moderate volatility.

### CGT: Indexation, Floor, and the No-Discount-Doubling Error

Prior versions of the code incorrectly applied both CPI indexation AND the 50% CGT discount, computing `max(marginal\_rate × 0.5, 0.30)`. The Treasury reform *replaces* the discount with indexation — they do not stack. The corrected formula is `max(marginal\_rate, 0.30)`, which means earners in the 37% and 45% brackets pay at their full marginal rate on real gains. The model was corrected after a line-by-line audit confirmed the error against the policy text.

The CGT algorithm correctly handles:

- **Cost-basis proportioning:** only the gain fraction of each sale ($1 - \\text\{basis\} / \\text\{market\_value\}$) is taxed

- **Gross-up for tax:** the model solves for the gross sale amount needed to net the required after-tax spending, avoiding the common mistake of treating after-tax needs as the pre-tax sale amount

- **Per-owner weighted averaging:** for jointly-held accounts, each owner's rate is computed independently and weighted by their ownership share

### RNG Isolation

All stochastic subsystems use isolated random number generators with separate seed-space regions. Equity and inflation return series are pre-generated at the trial level using a dedicated `series\_rng`, ensuring that changing one subsystem's stochastic settings (e.g., enabling stochastic mortgage rates or inflation) does not alter the equity paths for a given seed. This makes "what-if" comparisons — running the same seed with and without stochastic mortgage rates — produce genuinely comparable results rather than conflating model changes with path changes.

### Conservative Defaults

The model defaults to conservative assumptions where the policy is ambiguous or the real-world outcome depends on taxpayer behaviour:

- **Division 293 tax** is deducted from take-home pay, not paid from super (worst case for bridge viability)

- **Bridge failure** continues the simulation rather than halting — unmet spending accumulates, so the model does not assume a borrower defaults or restructures, producing a clean lower bound

- **Success** is defined as bridge assets staying above zero at every timestep (running minimum check), not just at the final horizon

- **Bridge values** are reported in today's dollars (deflated to simulation-start purchasing power), preventing nominal-dollar illusion

### Golden-Value Verification

Two deterministic end-to-end tests verify the model against hand calculations: a single-earner scenario produces bridge and super values matching independently-computed figures to the dollar, and a CGT drawdown scenario confirms the bridge stays positive through the full drawdown with known tax parameters.

### Audit Trail

The codebase underwent a structured 13-finding audit covering financial model trace, Monte Carlo mechanics, edge-case handling, and UI consistency. All findings rated "major" or above were addressed and verified by the test suite. Outstanding items rated "minor" (birth-date-granular preservation age, back-to-main-menu UX flow) are documented and deferred by informed choice, not oversight.

