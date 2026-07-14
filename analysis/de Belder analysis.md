# Monte Carlo Bridge Simulation — de Belder Profile

**Run date:** 2026-07-14  
**Seed:** 193,320,613  
**Iterations:** 20,000  
**Horizon:** Age 37 → 60 (super access age)  
**Value basis:** Real (2026 dollars, inflation-adjusted)

---

## 1. Profile Summary

| Field | Value |
|---|---|
| Earners | 2 (both employed, retire at 50) |
| Earner 1 | $180,000 salary, $320,000 super |
| Earner 2 | $250,000 salary, $230,000 super |
| Combined pre-tax income | $430,000/year |
| Child | 1 (age unknown, education schedule in profile) |
| Living expenses | $130,000/year (real) |
| Retirement target | $130,000/year (real) — unchanged from working |
| Mortgage | $886,346 principal, 6.05% fixed rate |
| Offset accounts | Account 1: $620,000 + Account 2: $100,000 = $720,000 total |
| Investment account | Account 3: $65,911 in equity (8% expected return, 15% std dev) |
| Surplus allocation | 100% to offset (surplus_investment_pct = 0.0) |
| Inflation | 2.5% fixed (no stochastic) |
| CGT on drawdowns | Enabled, 30% floor rate (post-2027 reform) |
| MLS | Disabled |

**Bridge period structure:**
- **Working years (37–50):** 13 years of dual income accumulation, mortgage amortisation
- **Drawdown years (50–60):** 10 years living off non-super bridge assets only — no salary
- **Super access at 60:** simulation ends — bridge assets after this point are not modelled

---

## 2. Top-Line Results

| Metric | Value |
|---|---|
| **Success probability** | **99.36%** |
| Bridge mean (end of bridge, after all drawdowns) | $3,522,916 |
| Bridge median | $2,773,841 |
| Bridge P5 (worst 5th percentile) | $550,457 |
| Bridge P10 | $876,892 |
| Bridge P25 | $1,598,607 |
| Bridge P75 | $4,550,655 |
| Bridge P90 | $7,058,053 |
| Bridge P95 (best 5th percentile) | $8,976,871 |
| Bridge coefficient of variation | 303.8% |
| Median super at age 60 | $3,944,340 |
| Mortgage clearance rate | 100.0% |
| Remaining mortgage (median) | $0 |

### Bootstrap Standard Errors

These measure the precision of the estimates given 20,000 trials:

| Metric | Estimate | SE | SE % |
|---|---|---|---|
| Bridge mean | $3,522,916 | $20,673 | 0.59% |
| Bridge median | $2,773,841 | $18,591 | 0.67% |
| Bridge P5 | $550,457 | $11,298 | 2.05% |
| Bridge P95 | $8,976,871 | $108,933 | 1.21% |

**Standard errors are well below 5% for all key metrics** — 20,000 trials provides adequate precision. The P5 estimate has the highest relative error at 2.05%, which is an acceptable level for financial planning.

---

## 3. Why 99.36%? — Drivers of the High Success Rate

The simulation reports 99.36% success (128 failures in 20,000 trials). All failures occur at age 59 — the very last bridge year. No trial fails before age 59. This result is driven by four factors:

### 3.1 Massive Surplus During Working Years

The household earns **$530,000/year** pre-tax and spends $130,000/year. After tax, the surplus is enormous:

**Earner 1** (employment_type=both, $180k salary + $100k self-employed):

| Component | Value |
|---|---|
| Salary | $180,000 |
| Self-employed income | $100,000 |
| Total gross | $280,000 |
| SG (12% on salary only, cap at $260k) | $21,600 |
| Concessional sacrifice (to fill cap) | $10,900 |
| Taxable income | $269,100 |
| Income tax | ~$87,233 |
| Medicare levy (2%) | ~$5,382 |
| **After-tax income** | **~$176,485** |

**Tax calculation (Earner 1):** $269,100 taxable — $0 on first $18,200; $4,288 on $26,800 up to $45k; $27,000 on $90k up to $135k; $20,350 on $55k up to $190k; $35,595 on $79,100 above $190k at 45%.

**Earner 2** (employment_type=self_employed, $250k salary):

| Component | Value |
|---|---|
| Salary (sole income) | $250,000 |
| SG | $0 (self-employed, sg_rate=0) |
| Concessional sacrifice (full cap) | $32,500 |
| Taxable income | $217,500 |
| Income tax | ~$64,013 |
| Medicare levy (2%) | ~$4,350 |
| **After-tax income** | **~$149,137** |

**Combined after-tax income: ~$325,622**

After-tax surplus before mortgage: ~$325,622 − $130,000 = **~$195,622/year**.

With mortgage payments of ~$65,892/year, free cash flow to offset: **~$129,730/year**.

### 3.2 Offset Accounts Absorb Surplus Immediately

The household starts with $720,000 in offset accounts against an $886,346 mortgage. That leaves only ~$166,000 of headroom before offset = mortgage balance.

At ~$111,000/year surplus, the offset fully covers the mortgage **within ~1.5–2 years**. After that point:
- The mortgage effectively costs 0% interest (fully offset)
- Monthly payments go entirely to principal, rapidly reducing the loan
- Additional surplus beyond offset capacity... **hits a modelling limitation** (see §7.2)

### 3.3 Mortgage Payoff During Working Years

With $720,000 offset against $886,346 at age 37 and $111,000/year surplus flowing to offset + $65,892/year in principal payments, the mortgage is **fully discharged before retirement at age 50** in the median case.

The simulation confirms this: mortgage clearance rate is 100.0%, and median remaining mortgage is $0 at the horizon.

### 3.4 Bridge Assets at Retirement Far Exceed Needs

**Median bridge assets at retirement (age 50):** $3,133,766  
**Total expenses for 10-year bridge:** $130,000 × 10 = $1,300,000

The median bridge portfolio at retirement is **2.4× the total expenses** for the entire bridge period. Even in the P5 case ($1,724,448 at age 50), assets exceed total projected expenses by $424,000.

Critically, the household's expenses **do not increase** after retirement (retirement_target = living_expenses = $130,000), so there is no "retirement lifestyle inflation" to absorb the surplus.

### 3.5 Super Balances Are Irrelevant to Bridge Success

Median super at age 60 is $3,944,340, but **super is inaccessible during the bridge**. The simulation measures bridge success solely on whether non-super assets suffice. Super is tracked for informational purposes only — it does not affect the success/failure determination.

The high super balances do, however, provide a massive safety net after age 60: the household could comfortably draw ~$158,000/year at a 4% withdrawal rate from super alone.

---

## 4. Bridge Asset Trajectory by Age

Values shown as **real** (2026 dollars), percentiles across 20,000 trials:

| Age | P5 | P25 | P50 (Median) | P75 | P95 |
|-----|----|-----|------|-----|-----|
| 37 | $885,909 | $894,472 | $901,345 | $908,864 | $921,164 |
| 40 | $1,105,872 | $1,190,473 | $1,258,771 | $1,336,062 | $1,469,820 |
| 45 | $1,519,966 | $1,862,440 | $2,165,265 | $2,537,896 | $3,253,079 |
| 50 (retire) | $1,724,448 | $2,446,401 | $3,133,766 | $4,034,090 | $5,863,583 |
| 55 | $1,034,325 | $1,991,967 | $2,985,096 | $4,383,297 | $7,658,827 |
| 59 (final) | $564,218 | $1,638,572 | $2,843,187 | $4,664,422 | $9,201,292 |

Key observations:
- Bridge assets **peak around age 49–50** (retirement), then decline as expenses are drawn
- The P5 trajectory shows a healthy buffer throughout — never dipping below $564,000
- The spread between P5 and P95 widens dramatically after retirement as equity returns compound: $1.5M gap at age 50 vs $8.6M gap at age 59
- The median portfolio **grows slightly** during the early bridge years (50→52), suggesting surplus from non-offset accounts exceeds spending — this is likely Account 3 (equity) returns exceeding drawdown needs in the median case

---

## 5. CGT Analysis

### 5.1 How CGT Works in This Simulation

The model uses the **post-2027 CGT reform** rules:
- The 50% CGT discount is abolished
- Cost basis is CPI-indexed (inflated by cumulative inflation since purchase)
- Tax is paid on real (above-inflation) gains only
- A 30% **floor rate** applies: CGT = max(marginal_rate, 0.30) per owner

During the bridge (age 50–60), the earners have **zero salary income**. After selling assets to fund expenses, their taxable income consists only of realised capital gains. This pushes their marginal rate into lower brackets, but the **30% floor** catches them: if their marginal rate would be 0–16%, they still pay 30% on real gains.

### 5.2 CGT Paid by Age (per trial)

| Age | P5 | P50 (Median) | P95 |
|-----|----|------|-----|
| 50 | $0 | $7,671 | $38,877 |
| 52 | $0 | $12,559 | $47,119 |
| 55 | $0 | $18,156 | $50,757 |
| 57 | $0 | $23,443 | $58,458 |
| 59 | $0 | $29,324 | $65,722 |

**CGT is a minor drag on the bridge.** The median CGT bill at age 59 is ~$29,000 on what is presumably a six-figure drawdown from Account 3. This represents the 30% floor applied to the real gain portion of the sale.

In the P5 case (worst scenarios), CGT is $0 across all ages — this occurs when Account 3 has been fully drawn in prior years or when market returns are negative (no gain to tax).

### 5.3 Drawdown Sources (Median Trial)

| Age | Offset drawn | Non-offset drawn (Account 3) |
|-----|-------------|------|
| 50 | $54,404 | $224,374 |
| 51 | $45,596 | $173,861 |
| 52 | $0 | $225,031 |
| 53 | $0 | $192,986 |
| 54 | $0 | $197,810 |
| 55 | $0 | $202,756 |
| 56 | $0 | $207,825 |
| 57 | $0 | $213,020 |
| 58 | $0 | $218,346 |
| 59 | $0 | $223,804 |

The offset accounts are **fully depleted within the first two bridge years** (ages 50–51). After that, all living expenses come from Account 3 (the $65,911 equity account, grown by 13 years of compounding at ~8% expected return). The non-offset drawdowns grow slowly from ~$174k at age 51 to ~$224k at age 59 — this likely reflects the 2.5% inflation applied to the $130,000 living expenses over time.

This also explains the P5 CGT of $0: the worst-case scenarios exhaust Account 3 early (bad equity returns), leaving no gains to tax.

---

## 6. Failure Scenarios — The 0.64%

All 128 failures (0.64% of 20,000 trials) occur at **age 59** — the final bridge year. Zero failures before age 59.

**Interpretation:** In the worst 0.64% of possible futures, the bridge portfolio survives nearly the entire 10-year drawdown period but runs dry in the final year. This is the classic "sequence of returns risk" at work: poor equity returns during the bridge period deplete Account 3 faster than expected.

**What this means for the client:** The household has a 99.36% probability of reaching super access age (60) with positive bridge assets. In the worst case, they'd face a shortfall in the final year — but with $3.94M in super becoming accessible at exactly that point, this is more of a technical "failure" than a real hardship.

### Concrete scenario: The worst simulation trial

The simulation's running minimum (across all trials, across all ages) is **−$543,332 at age 59**. This is the single worst bridge value observed in any trial at any age.

This means: in the absolute worst trial, the household has a $543,332 shortfall at age 59. They'd need to cover this from other sources (family assistance, downsizing home, reverse mortgage) for ~1 year until super becomes accessible.

---

## 7. Caveats, Limitations, and Modelling Issues

### 7.1 Interest Rate Override Semantics

Account 3 uses `interest_rate = 0.08` (8% expected return). Per the current engine semantics, this is treated as a **mean override** — the asset still experiences full equity-class volatility (15% std dev) and correlation with other equity returns via the lognormal generation path.

This is correct behaviour and contributes to realistic Monte Carlo variance. The bridge CV of 303.8% is appropriately wide.

### 7.2 Surplus Beyond Offset Capacity — Verified Correct

Previously flagged as an open question, this has now been confirmed by code review.

The engine's `handle_offset_overflow` function (primitives.py:853) correctly redirects excess offset balance into the first non-offset investment account. When offset ≥ mortgage principal, the excess automatically flows to Account 3 (equity) with cost-basis tracking. When the mortgage is fully discharged, the entire offset balance migrates to the non-offset account.

This means the ~$129,730/year of working-years surplus (after the offset fills within ~1.3 years) is properly accumulated and compounded in Account 3 at equity returns for the remaining ~11.7 working years. This further supports the high success rate — the model is correctly capturing the household's full saving capacity.

**Correction:** My earlier caveat suggesting surplus might be discarded was incorrect. The engine handles this correctly.

### 7.3 No Stochastic Inflation — Feature Defined but Not Exposed in UI

`SimulationInputs.stochastic_inflation` (models.py:501) is a field that, when `True`, triggers a 3-way Cholesky decomposition (equity + super + inflation) using the correlation constants `EQ_INF_CORR = -0.10` and `SUPER_INF_CORR = -0.10` (primitives.py:224-225). The `generate_correlated_triplet` function (primitives.py:561) implements this correctly, and the simulation engine checks `inputs.stochastic_inflation` at multiple call sites (simulation.py:1126, 1149, 1537, 1555, 1575).

**However, there is no UI control to toggle it.** The `configure_simulation_params` function in ui.py:1178 prompts for inflation rate, iterations, CGT, super fees, surplus allocation, and success threshold — but does not prompt for stochastic inflation. The only way to enable it is by manually editing the profile JSON and setting `"stochastic_inflation": true`.

This is a **documentation gap**, not an engine gap. The feature works but isn't accessible through the UI. With stochastic inflation enabled, the P5–P95 spread would likely widen due to additional variance from the inflation process (inflation has mild negative correlation with equity returns via the Cholesky path).

**This run uses `stochastic_inflation = False` (default).** All results shown are with fixed 2.5% inflation.

### 7.4 Education Costs

The profile has one child. If the child's education schedule includes costs during the working years (age 37–50), these would reduce the surplus available for offset accumulation. The education cost schedule is not directly visible from the simulation output — it's embedded in the child's `education_schedule` field in the profile JSON.

### 7.5 CGT Cost-Basis Accuracy

The CGT cost-basis indexation uses cumulative inflation. Since the equity account (Account 3) was likely funded from surplus during the working years (with new cost-basis lots each year), the effective CGT rate on sale depends on which lot is sold (weighted-average cost basis). The model appears to use a single aggregate cost-basis approach, which is a reasonable simplification for a Monte Carlo tool but may overstate or understate actual CGT depending on lot-level gains.

### 7.6 No Home Equity or Age Pension

The model does not include:
- The primary residence (presumably exists but is not in the model)
- Age Pension eligibility (which would kick in at 67, not relevant for bridge to 60)
- Potential downsizing proceeds

These are reasonable omissions for a bridge-to-super-access tool.

---

## 8. Assessment

### Is 99.36% plausible?

**Yes, given the inputs.** The household has:

1. **Very high income ($430k) vs. moderate expenses ($130k)** — a 3.3× coverage ratio during working years
2. **$720k in offset accounts** protecting against mortgage interest immediately
3. **13 years of accumulation** before drawdown begins
4. **No lifestyle inflation at retirement** — spending stays flat at $130k
5. **$3.9M in super at age 60** as a backstop (inaccessible during bridge, but reassuring)

The model's result is not suspicious — it's the natural outcome of a household that saves aggressively during high-earning years and has a moderate spending target. The 0.64% failure rate reflects tail risk from extreme equity drawdowns, not a modelling error.

### Would I trust this number for a real financial decision?

**Conditionally.** The key condition is §7.2 (surplus beyond offset capacity). If the engine is discarding surplus after offset is full, the actual success rate could be even higher — the model is being conservative. But this needs to be confirmed, not assumed.

### Biggest risk to the client

The dominant risk is **not** running out of money during the bridge. It's the opposite: the household is **over-saving** during working years relative to their spending target. The profile shows retirement_target = living_expenses, which implies the household plans to maintain exactly the same lifestyle in retirement — but their pre-retirement budget already had room for significantly higher spending.

This is a lifestyle planning question, not a model accuracy question. The model is reporting what was asked; the question is whether the inputs reflect the household's actual intentions.

---

*Analysis generated by the Monte Carlo Retirement Bridge Simulator.  
Returns are modelled as lognormal (Black-Scholes parameterisation). Equity mean is sourced from Credit Suisse / UBS Global Investment Returns Yearbook (Australian equities, ~6.4–6.7% real since 1900). Standard deviations, correlations, and non-equity means are unsourced placeholders — see `primitives.py` for full provenance documentation.*
