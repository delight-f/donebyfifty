"""Data models for the Monte Carlo CLI tool.

Pure dataclasses with no logic, I/O, or UI dependencies. All models are
frozen (immutable) to prevent accidental mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simulation import ScenarioComparisonResult

# =============================================================================
# AUSTRALIAN PRESERVATION AGE LOOKUP
# =============================================================================


def preservation_age(birth_year: int) -> int:
    """Return the Australian superannuation preservation age for a birth year.

    Preservation age is the minimum age at which a person can access their
    superannuation. It depends on date of birth, rising from 55 (born before
    1 July 1960) to 60 (born after 30 June 1964).

    Args:
        birth_year: Calendar year of birth (e.g. 1970).

    Returns:
        Preservation age (55–60).

    Raises:
        ValueError: If birth_year is before 1900 or after 2100.

    Note:
        Preservation age depends on **date** of birth, not just year. The
        function uses a calendar-year approximation: everyone born in year
        *X* is assigned the preservation age that applies from 1 July of
        that year. A person born in December 1960 (actual preservation age
        56) would be reported as 55. This is a known simplification — see
        O4 in the audit findings.

    """
    if birth_year < 1900 or birth_year > 2100:
        raise ValueError(f"birth_year {birth_year} out of supported range (1900–2100)")
    # Calendar-year approximation of the fiscal-year preservation age table.
    # Treats the whole of year X as if born before 1 July of year X+1,
    # which is the standard approximation when month of birth is unknown.
    if birth_year < 1960:
        return 55
    if birth_year < 1961:
        return 55
    if birth_year < 1962:
        return 56
    if birth_year < 1963:
        return 57
    if birth_year < 1964:
        return 58
    if birth_year < 1965:
        return 59
    return 60


# =============================================================================
# HOUSEHOLD COMPONENTS (continued)
# =============================================================================


@dataclass(frozen=True)
class Earner:
    """An individual income earner in the household.

    Each earner is self-contained with their own salary, super balance,
    retirement age, and part-time work parameters. No gendered fields.
    """

    label: str = "Earner 1"
    salary: float = 100_000.0
    super_balance: float = 100_000.0
    salary_growth_rate: float = 0.005
    """Real salary growth rate (above inflation).

    Default 0.5% real (compounded with inflation to get effective nominal).
    Set to 0 for salary that just tracks inflation (flat in real terms).
    """
    retirement_age: int = 50
    birth_year: int | None = None
    """Birth year for preservation-age lookup.

    If set, ``super_access_age`` is auto-computed from Australian
    preservation-age rules (55–60 depending on birth year). When
    ``None`` (default), ``super_access_age`` must be set manually.
    """
    super_access_age: int = 60
    sg_rate: float = 0.12
    employment_type: str = "employed"
    """Employment type for income/super modelling.

    ``"employed"`` (default): salary income, SG applied at ``sg_rate``,
    salary-sacrifice via ``personal_super_contributions_total_p_a``.

    ``"self_employed"``: business income (modelled via ``salary``), SG is
    forced to 0 regardless of ``sg_rate`` (sole traders don't receive SG
    from themselves). Personal deductible contributions use the
    ``personal_super_contributions_total_p_a`` field.

    ``"both"``: earner has BOTH salary income (``salary``, SG at ``sg_rate``)
    AND self-employed/business income (``self_employed_income``, SG if
    ``self_employed_sg_applies``). Both income streams stop at retirement
    age and grow independently. Total taxable income is the sum of both.

    ``"not_employed"``: no salary income, no SG. Still eligible for
    part-time income and non-concessional contributions.

    """
    self_employed_income: float = 0.0
    """Annual self-employed / business income in today's dollars.

    Only meaningful when ``employment_type == "both"``.
    Grows independently via ``self_employed_growth_rate``.
    Default 0.0 (no self-employed income).
    """
    self_employed_growth_rate: float = 0.005
    """Real growth rate for self-employed income (above inflation).

    Only meaningful when ``employment_type == "both"``.
    Default 0.5% real (compounded with inflation for nominal growth).
    """
    self_employed_sg_applies: bool = False
    """Whether Super Guarantee applies to the self-employed income.

    Only meaningful when ``employment_type == "both"``.
    Some business structures (e.g. company directors paying SG to
    themselves via payroll) may have SG on both income streams.
    Default ``False`` (no SG on self-employed income).
    """
    pt_days_per_week: float = 0.0
    pt_start_age: int = -1
    """Age at which part-time income begins. -1 (default) means use ``retirement_age``."""
    pt_end_age: int = 65
    pt_daily_rate: float = 3_000.0
    pt_weeks_per_year: float = 48.0
    pt_rate_mode: str = "daily_rate"
    """How the part-time income rate is entered.

    ``"daily_rate"`` (default): user provides an explicit daily rate
    (e.g., $500/day). Annual PT income is:
        days_per_week * weeks_per_year * daily_rate

    ``"salary_pct"``: user provides a percentage of the earner's initial
    full-time salary. Annual PT income is:
        salary \u00d7 pt_salary_pct / 100

    Old profiles without this field default to ``"daily_rate"`` for
    backward compatibility.
    """
    pt_salary_pct: float = 0.0
    """Percentage of initial salary used as part-time income base.

    Only meaningful when ``pt_rate_mode == \"salary_pct\"``.
    E.g. 40 means PT income = 40% of the earner's full-time salary.
    Default 0.0 (no percentage basis).
    """
    personal_super_contributions_total_p_a: float | None = None
    """Fixed annual salary-sacrifice/concessional contribution amount.

    If ``None`` (default), the engine auto-calculates the top-up needed
    to reach the concessional cap after the Super Guarantee:
    ``max(0, CONC_CAP - SG)``.

    Set to a specific dollar amount (e.g. 32_500) to override the
    auto-calc and use a fixed personal deductible contribution instead.
    This matches the original reference case where Earner 2 uses a full
    $32,500 personal deductible regardless of SG.
    """
    non_concessional_contributions_p_a: float = 0.0
    """Annual non-concessional (after-tax) super contributions.

    These are added directly to the earner's super balance each year
    without any tax deduction. They do not count toward the concessional
    cap or Division 293 threshold.
    Default 0.0 (no after-tax contributions).
    """
    super_growth_pct: float = 70.0
    """Percentage allocated to growth assets (equity) within super, 0\u2013100.

    The remainder is allocated to defensive assets (bonds).
    The blended annual return is:
        (growth_pct / 100) \u00d7 equity_return + (1 \u2212 growth_pct / 100) \u00d7 bond_return

    Used only when ``super_mean_override`` is None. If an override is set
    (via ``super_mean_override``), the raw override return is used instead
    of the blended calculation.
    """
    super_glide_end_year: int | None = None
    """Simulation year (0-indexed) when the glidepath reaches its target.

    If set, the growth percentage glides linearly from ``super_growth_pct``
    at year 0 to ``super_glide_target_pct`` at this year. After this year,
    the target percentage is used.

    Example: start at 70% growth, glide to 30% by year 15:
        super_growth_pct = 70, super_glide_end_year = 15, super_glide_target_pct = 30

    ``None`` (default) means no glidepath — ``super_growth_pct`` is fixed.
    """
    super_glide_target_pct: float = 30.0
    """Target growth percentage at the end of the glidepath.

    Only meaningful when ``super_glide_end_year`` is set.
    Default 30% (conservative, suitable for someone approaching preservation age).
    """
    super_asset_class: str = "equity"
    """Asset class for this earner's superannuation returns.
    Used when ``super_mean_override`` is not set AND growth blending is
    not applicable (kept for backward compatibility).
    """
    super_mean_override: float | None = None
    """Override mean return for this earner's super. If None, uses the
    asset class lookup from ``super_asset_class``.

    Note: The defaults these overrides fall back to (``EQ_MEAN``, ``EQ_STD``,
    etc. in ``primitives.py``) have mixed provenance. Equity mean is sourced
    (Credit Suisse / UBS Yearbook); all standard deviations and most other
    means are unsourced placeholders. See ``primitives.py`` module docstring
    and per-constant comments for full provenance details.
    """
    super_std_override: float | None = None
    """Override std deviation for this earner's super. Only used when
    ``super_mean_override`` is also set.

    Same provenance caveat as ``super_mean_override`` — default fallback
    standard deviations are unsourced placeholders.
    """

    def __post_init__(self) -> None:
        """Auto-compute super_access_age from birth_year; resolve pt_start_age sentinel."""
        if self.birth_year is not None:
            computed = preservation_age(self.birth_year)
            if self.super_access_age == 60 and computed != 60:
                # User didn't override super_access_age — use computed value
                object.__setattr__(self, "super_access_age", computed)
        # Resolve pt_start_age sentinel: -1 means "follow retirement_age"
        if self.pt_start_age == -1:
            if self.retirement_age >= 999:
                # Non-employed or no retirement set — use old default of 60
                object.__setattr__(self, "pt_start_age", 60)
            else:
                object.__setattr__(self, "pt_start_age", self.retirement_age)


@dataclass(frozen=True)
class Child:
    """A child in the household with optional education cost schedule.

    An empty ``education_schedule`` means no education costs.
    Schedule is a tuple of (age, annual_cost_in_today_dollars) pairs.
    """

    label: str = "Child 1"
    age: int = 5
    education_schedule: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class MortgageAccount:
    r"""A mortgage or other amortising debt with optional offset linkage.

    ``principal`` is the **outstanding balance** at the start of the simulation
    (not necessarily the original loan amount).
    If ``monthly_payment`` is 0, only interest is paid (interest-only loan).
    ``offset_accounts`` contains labels of linked ``InvestmentAccount`` s.
    """

    label: str = "Mortgage 1"
    principal: float = 500_000.0
    """Outstanding balance at simulation start, not original loan amount."""
    interest_rate: float = 0.0605
    monthly_payment: float = 0.0
    offset_accounts: tuple[str, ...] = ()
    offset_reserve_mode: str = "fixed"
    """How the offset reserve floor is determined each period.

    ``"fixed"`` (default): the floor is the static ``offset_reserve_floor``
    value (dollar amount, often 0 for no reserve).

    ``"stall_prevention"``: the floor is dynamically computed each period as
    the minimum offset needed so that the mortgage payment fully covers the
    interest charge on the net debt.  Calculated as:

        required = max(0, mortgage_balance - monthly_payment / monthly_rate)

    This prevents **negative amortisation** – the loan balance will not grow,
    but it will not reduce either.  ``monthly_payment - interest = 0``.
    The floor is recalculated each period as the mortgage amortises.
    If the client lacks enough total offset to meet this requirement,
    whatever offset exists is fully preserved and the remaining shortfall
    is sourced from non-offset accounts.

    ``"interest_cancelling"``: the floor is set to the full mortgage
    principal each period.  This preserves enough offset to keep the
    effective debt at zero, so **no interest is charged** and 100% of the
    payment goes to principal.  This mode effectively requires offset
    balance >= mortgage balance; if the client has less offset than the
    mortgage, all offset is preserved and the mortgage partly amortises
    on the un-offset portion.
    """
    offset_reserve_floor: float = 0.0
    """Static offset reserve floor (used only when
    ``offset_reserve_mode == "fixed"``).

    Minimum offset balance to preserve for this mortgage.
    The drawdown logic will not draw offset balances linked to this
    mortgage below this floor. Once reached, any further shortfall is
    sourced from non-offset accounts instead.

    Default 0.0 (no reserve — offset drains fully).
    """
    loan_term_end_age: int | None = None
    """Age by which this mortgage must be fully repaid.

    If set, each simulation path checks whether the principal reaches
    zero by this age. Paths where it does not are flagged as a
    term-clearance failure independently of the bridge-survival check.

    ``None`` (default) means no term check is applied.
    """

    # ── Stochastic mortgage rate parameters (Black-Karasinski) ───────
    interest_rate_stochastic: bool = False
    """If True, the mortgage interest rate varies each year following a
    Black-Karasinski mean-reverting lognormal process (discrete AR(1)
    exact solution of the continuous-time BK SDE).

    If False (default), ``interest_rate`` is fixed for all years.
    """
    interest_rate_vol: float | None = None
    """Continuous-time diffusion coefficient sigma_tilde for the BK
    log-rate process.  ``None`` (default) uses ``BK_SIGMA_MODERATE``
    (0.18 = moderate) at runtime.  See config.py for documented
    convention (this is NOT the stationary log-rate std dev).
    """
    interest_rate_kappa: float | None = None
    """Mean-reversion speed per year for the BK log-rate process.
    ``None`` (default) uses ``BK_KAPPA`` (0.20).
    """
    interest_rate_theta: float | None = None
    """Long-run mean mortgage rate (decimal) for the BK process.
    ``None`` (default) uses ``BK_THETA`` (0.065 = 6.5%).
    """
    interest_rate_corr: float | None = None
    """Target Pearson correlation between the mortgage rate innovation
    and the equity return innovation (eq_z) for each simulation year.
    ``None`` (default) uses ``BK_RHO`` (0.20).

    Use 0.0 for genuinely uncorrelated rates (``is not None`` resolution
    distinguishes this from the default).
    """


@dataclass(frozen=True)
class InvestmentAccount:
    """An investment account (equity, cash, property, etc.) outside super.

    ``asset_class`` describes the type of asset for return modelling.
    ``tax_jurisdiction`` determines CGT treatment on drawdowns.
    ``is_offset`` marks this as a mortgage offset account (treated as cash).
    ``interest_rate`` allows users to specify a custom fixed return rate (e.g.
    0.01 for 1% cash accounts). When 0.0 (default), the asset class return
    model is used instead (e.g. ~7% for equity, ~3% for bonds).
    """

    label: str = "Account 1"
    market_value: float = 0.0
    cost_basis: float = 0.0
    asset_class: str = "equity"
    tax_jurisdiction: str = "au"
    cgt_rate: float = 0.30
    is_offset: bool = False
    fee_rate: float = 0.0
    interest_rate: float = 0.0
    """Expected annual return override for this account (decimal).

    0.0 (default) means use the asset class's default mean and volatility.
    When set, the return is lognormal with this user-specified mean and the
    asset class's standard deviation and equity correlation — NOT a fixed
    deterministic rate.  For example, an equity account with
    ``interest_rate=0.08`` gets ~8% expected return with ~15% volatility.
    """
    ownership: dict[int, float] = field(default_factory=lambda: {0: 1.0})
    """Fractional ownership per earner index (matches ``Household.earners`` tuple).

    Key = earner index in ``Household.earners``, value = ownership share (0.0–1.0).
    Must sum to 1.0.  Default ``{0: 1.0}`` = 100% to Earner 1, which preserves
    pre-Phase-2 behaviour and provides a safe default for single-earner households.

    Ownership indices are stable across a simulation trial — the earner tuple
    order is invariant, matching the indexing of ``super_balances[]``,
    ``earner_taxable_incomes[]``, etc.
    """


@dataclass(frozen=True)
class Household:
    """Complete household definition for a simulation run.

    Default is a single-earner household with no children, no mortgage,
    and no investment accounts.
    """

    earners: tuple[Earner, ...] = (Earner(),)
    children: tuple[Child, ...] = ()
    mortgages: tuple[MortgageAccount, ...] = ()
    investment_accounts: tuple[InvestmentAccount, ...] = ()
    base_living_expenses: float = 60_000.0
    retirement_target: float = 80_000.0

    @property
    def num_earners(self) -> int:
        """Number of earners in the household."""
        return len(self.earners)

    @property
    def num_children(self) -> int:
        """Number of children in the household."""
        return len(self.children)

    @property
    def num_mortgages(self) -> int:
        """Number of mortgages."""
        return len(self.mortgages)

    @property
    def total_super(self) -> float:
        """Sum of all earners' super balances."""
        return sum(e.super_balance for e in self.earners)


# =============================================================================
# SIMULATION INPUTS / RESULTS
# =============================================================================


@dataclass(frozen=True)
class SimulationInputs:
    """All user-configurable inputs for a Monte Carlo simulation run."""

    household: Household = field(default_factory=Household)
    n_iterations: int = 5_000
    inflation: float = 0.025
    simulation_start_age: int = 37
    cgt_on_drawdowns: bool = True
    sell_strategy: str = "waterfall"
    sell_order: tuple[str, ...] = ()
    conc_cap_growth_rate: float = 0.03
    """Annual growth rate for the concessional contributions cap."""
    sg_max_base_growth_rate: float = 0.03
    """Annual growth rate for the Super Guarantee maximum salary base."""
    seed: int | None = None
    """Random seed for reproducible Monte Carlo runs. If None, system entropy used."""
    super_fee_rate: float = 0.0085
    """Annual superannuation fund fee rate (admin + investment combined).

    Default 0.85% p.a. is the median Australian super fund fee.
    Applied as a drag on super balances before each year's returns.
    """
    mls_enabled: bool = False
    """Enable Medicare Levy Surcharge for high-income earners without private hospital cover."""
    mls_tiered: bool = True
    """Use progressive MLS tiers (1.0% / 1.25% / 1.5%) based on income.

    When True (default), the MLS rate is computed from ``mls_rate_for_income()``
    using singles or couple tiers depending on household composition.
    When False, the flat ``mls_rate`` is used for all incomes above threshold.
    """
    mls_threshold: float = 90_000.0
    """Income threshold for MLS (singles). Couples threshold is 2x this."""
    mls_rate: float = 0.01
    """MLS rate: Tier 1 = 1%, Tier 2 = 1.25%, Tier 3 = 1.5%.
    Default 1% for simplicity; users can override.
    """
    bracket_growth_rate: float = 0.025
    """Annual growth rate for tax bracket thresholds (bracket creep indexation).

    Default 2.5% matches long-run CPI. Set to 0 for frozen brackets.
    """
    div293_threshold: float = 250_000.0
    """Division 293 income threshold for additional tax on concessional contributions."""
    div293_rate: float = 0.15
    """Division 293 additional tax rate on concessional contributions above threshold."""
    div293_growth_rate: float = 0.025
    """DEPRECATED — Division 293 threshold is statutory at $250,000 since
    1 July 2017 and is not indexed. This field is retained for backward
    compatibility with existing profiles but is no longer used in the engine.
    """
    surplus_investment_pct: float = 0.0
    """Percentage of annual surplus to allocate to non-offset investment accounts
    (the remainder goes to offset accounts or cash).
    0.0 (default) = all surplus to offset/cash.
    50.0 = half to investments, half to offset/cash.
    100.0 = all to investments.
    """
    stochastic_inflation: bool = False
    """If True, inflation is drawn from a correlated stochastic process
    (3-way Cholesky with equity and super). If False, uses fixed ``inflation``.
    """
    success_threshold: float = 0.95
    """Target success probability (e.g. 0.90 for 90%). Used to highlight
    results as meeting/not-meeting the user's risk tolerance."""


@dataclass(frozen=True)
class SimulationResults:
    """Aggregate statistics from a completed Monte Carlo run."""

    trials: int
    p_success: float
    bridge_mean: float
    bridge_median: float
    bridge_p5: float
    bridge_p10: float
    bridge_p25: float
    bridge_p75: float
    bridge_p90: float
    bridge_p95: float
    bridge_min: float
    super_median: float
    bridge_floor: float = 0.0
    """Worst-case minimum bridge balance across all paths (the lowest
    value any path ever reached, not just at the horizon)."""
    floor_age: int = 0
    """Age at which ``bridge_floor`` occurred in the worst-path trial."""
    floor_end_bridge: float = 0.0
    """End-state bridge of the same trial that produced ``bridge_floor``.
    If equal to ``bridge_floor``, the worst path declined monotonically;
    if higher, the portfolio recovered after the floor."""
    horizon_age: int = 60
    per_earner_super_p50: dict[str, float] = field(default_factory=dict)
    remaining_mortgage_p50: dict[str, float] = field(default_factory=dict)
    mortgage_term_clearance_rate: float = 1.0
    """Proportion of trials where ALL mortgages with a term end cleared by that term."""
    per_mortgage_term_cleared_pct: dict[str, float] = field(default_factory=dict)
    """Per-mortgage term-clearance rate across all checkable trials."""
    per_mortgage_not_checkable: dict[str, int] = field(default_factory=dict)
    """Mortgages whose term end is beyond the simulation horizon.
    Maps label -> number of trials (should equal total trials). Not an
    error, but means the term-clearance question can't be answered yet.
    """
    seed: int | None = None
    """Random seed used for this run (None = system entropy).

    Stored so the exact run can be reproduced from a saved profile.
    """
    bridge_mean_se: float | None = None

    # Work Item 1 — Age-by-age bridge trajectory (percentiles per year)
    bridge_by_age_ages: list[int] = field(default_factory=list)
    """Ages corresponding to the bridge trajectory rows. Populated when
    per-year trajectory capture is enabled."""
    bridge_by_age_p5: list[float] = field(default_factory=list)
    bridge_by_age_p10: list[float] = field(default_factory=list)
    bridge_by_age_p25: list[float] = field(default_factory=list)
    bridge_by_age_p50: list[float] = field(default_factory=list)
    bridge_by_age_p75: list[float] = field(default_factory=list)
    bridge_by_age_p90: list[float] = field(default_factory=list)
    bridge_by_age_p95: list[float] = field(default_factory=list)

    # Work Item 2 — Near-miss / failure depth analysis
    near_miss_rate: float = 1.0
    """Proportion of trials that never crossed the near-miss threshold.
    Only populated when capture is enabled (same as bridge_by_age).
    """
    near_miss_threshold: float = 0.0
    """Dollar threshold below which a trial is considered a near-miss.
    0.0 (default) = same as failure.
    """
    failure_age_distribution: dict[int, int] = field(default_factory=dict)
    """Ages at which failures occurred, one entry per failing trial.
    Key = age, value = count of failing trials at that age.
    Empty dict when no failures.
    """

    # Work Items 4, 5 — Per-year drawdown composition
    offset_drawn_p50: list[float] = field(default_factory=list)
    """Median offset draws by year (per-year breakdown, Work Item 4)."""
    non_offset_drawn_p50: list[float] = field(default_factory=list)
    """Median non-offset draws by year (Work Item 4)."""
    offset_drawn_p5: list[float] = field(default_factory=list)
    offset_drawn_p95: list[float] = field(default_factory=list)
    cgt_paid_p50: list[float] = field(default_factory=list)
    """Median CGT paid by year (with 30% floor, Work Item 5)."""
    cgt_paid_p5: list[float] = field(default_factory=list)
    cgt_paid_p95: list[float] = field(default_factory=list)
    cgt_without_floor_p50: list[float] = field(default_factory=list)
    """Median CGT without 30% floor by year (counterfactual, Work Item 5)."""

    # Work Item 10 — Mortgage amortisation by age (P5/P50/P95 per mortgage)
    mortgage_by_age_ages: list[int] = field(default_factory=list)
    """Ages for the mortgage amortisation rows."""
    mortgage_by_age: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    """Per-mortgage balances by age with P5/P50/P95 percentiles.
    Structure: {label: {"p50": [y0, y1, ...], "p5": [...], "p95": [...]}}
    Nominal dollars (debt doesn't inflate).
    """
    offset_by_age: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    """Per-offset-account balances by age with P5/P50/P95 percentiles.
    Same structure as mortgage_by_age. Nominal dollars.
    """

    # Work Item 10 — Mortgage rate trajectory (P5/P50/P95 per mortgage)
    mortgage_rate_by_age: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    """Per-mortgage annual interest rate by age with P5/P50/P95 percentiles.
    Same dict structure as mortgage_by_age. Decimal rates (e.g. 0.065).
    Only populated when stochastic rates are enabled.
    """
    bridge_median_se: float | None = None
    """Bootstrap standard error of the median bridge balance."""
    bridge_p5_se: float | None = None
    """Bootstrap standard error of the 5th percentile bridge balance."""
    bridge_p95_se: float | None = None
    """Bootstrap standard error of the 95th percentile bridge balance."""
    super_median_se: float | None = None
    """Bootstrap standard error of the median super balance."""
    value_basis: str = "real"
    """Convention for all dollar values in this result.

    "real" = today's dollars (deflated by cumulative inflation).
    "nominal" = then-year dollars (not deflated).

    As of the results enhancement (July 2026), all output uses real
    (today's) dollars. Old saved profiles may have "nominal" values.
    """

    def summary_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for JSON serialisation."""
        return {
            "trials": self.trials,
            "p_success": self.p_success,
            "bridge_mean": self.bridge_mean,
            "bridge_median": self.bridge_median,
            "bridge_p5": self.bridge_p5,
            "bridge_p10": self.bridge_p10,
            "bridge_p25": self.bridge_p25,
            "bridge_p75": self.bridge_p75,
            "bridge_p90": self.bridge_p90,
            "bridge_p95": self.bridge_p95,
            "bridge_min": self.bridge_min,
            "bridge_floor": self.bridge_floor,
            "floor_age": self.floor_age,
            "floor_end_bridge": self.floor_end_bridge,
            "super_median": self.super_median,
            "horizon_age": self.horizon_age,
            "per_earner_super_p50": dict(self.per_earner_super_p50),
            "remaining_mortgage_p50": dict(self.remaining_mortgage_p50),
            "mortgage_term_clearance_rate": self.mortgage_term_clearance_rate,
            "per_mortgage_term_cleared_pct": dict(self.per_mortgage_term_cleared_pct),
            "per_mortgage_not_checkable": dict(self.per_mortgage_not_checkable),
            "seed": self.seed,
            "bridge_mean_se": self.bridge_mean_se,
            "bridge_median_se": self.bridge_median_se,
            "bridge_p5_se": self.bridge_p5_se,
            "bridge_p95_se": self.bridge_p95_se,
            "super_median_se": self.super_median_se,
            "value_basis": self.value_basis,
            "near_miss_rate": self.near_miss_rate,
            "near_miss_threshold": self.near_miss_threshold,
            "failure_age_distribution": dict(self.failure_age_distribution),
            # Trajectory (Work Items 1, 10)
            # Drawdown composition (Work Items 4, 5)
            "offset_drawn_p50": list(self.offset_drawn_p50),
            "non_offset_drawn_p50": list(self.non_offset_drawn_p50),
            "offset_drawn_p5": list(self.offset_drawn_p5),
            "offset_drawn_p95": list(self.offset_drawn_p95),
            "cgt_paid_p50": list(self.cgt_paid_p50),
            "cgt_paid_p5": list(self.cgt_paid_p5),
            "cgt_paid_p95": list(self.cgt_paid_p95),
            "cgt_without_floor_p50": list(self.cgt_without_floor_p50),
            # Trajectory (Work Items 1, 10)
            "bridge_by_age_ages": list(self.bridge_by_age_ages),
            "bridge_by_age_p5": list(self.bridge_by_age_p5),
            "bridge_by_age_p10": list(self.bridge_by_age_p10),
            "bridge_by_age_p25": list(self.bridge_by_age_p25),
            "bridge_by_age_p50": list(self.bridge_by_age_p50),
            "bridge_by_age_p75": list(self.bridge_by_age_p75),
            "bridge_by_age_p90": list(self.bridge_by_age_p90),
            "bridge_by_age_p95": list(self.bridge_by_age_p95),
            "mortgage_by_age_ages": list(self.mortgage_by_age_ages),
            "mortgage_by_age": {
                k: {pk: list(pv) for pk, pv in v.items()} for k, v in self.mortgage_by_age.items()
            },
            "offset_by_age": {
                k: {pk: list(pv) for pk, pv in v.items()} for k, v in self.offset_by_age.items()
            },
            "mortgage_rate_by_age": {
                k: {pk: list(pv) for pk, pv in v.items()}
                for k, v in self.mortgage_rate_by_age.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationResults:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            trials=data["trials"],
            p_success=data["p_success"],
            bridge_mean=data["bridge_mean"],
            bridge_median=data["bridge_median"],
            bridge_p5=data["bridge_p5"],
            bridge_p10=data["bridge_p10"],
            bridge_p25=data["bridge_p25"],
            bridge_p75=data["bridge_p75"],
            bridge_p90=data["bridge_p90"],
            bridge_p95=data["bridge_p95"],
            bridge_min=data["bridge_min"],
            bridge_floor=data.get("bridge_floor", 0.0),
            floor_age=data.get("floor_age", 0),
            floor_end_bridge=data.get("floor_end_bridge", 0.0),
            super_median=data["super_median"],
            horizon_age=data.get("horizon_age", 60),
            per_earner_super_p50=data.get("per_earner_super_p50", {}),
            remaining_mortgage_p50=data.get("remaining_mortgage_p50", {}),
            mortgage_term_clearance_rate=data.get("mortgage_term_clearance_rate", 1.0),
            per_mortgage_term_cleared_pct=data.get("per_mortgage_term_cleared_pct", {}),
            per_mortgage_not_checkable=data.get("per_mortgage_not_checkable", {}),
            seed=data.get("seed"),
            bridge_mean_se=data.get("bridge_mean_se"),
            bridge_median_se=data.get("bridge_median_se"),
            bridge_p5_se=data.get("bridge_p5_se"),
            bridge_p95_se=data.get("bridge_p95_se"),
            super_median_se=data.get("super_median_se"),
            value_basis=data.get("value_basis", "real"),
            near_miss_rate=data.get("near_miss_rate", 1.0),
            near_miss_threshold=data.get("near_miss_threshold", 0.0),
            failure_age_distribution={
                int(k): v for k, v in data.get("failure_age_distribution", {}).items()
            },
            # Drawdown composition (Work Items 4, 5)
            offset_drawn_p50=list(data.get("offset_drawn_p50", [])),
            non_offset_drawn_p50=list(data.get("non_offset_drawn_p50", [])),
            offset_drawn_p5=list(data.get("offset_drawn_p5", [])),
            offset_drawn_p95=list(data.get("offset_drawn_p95", [])),
            cgt_paid_p50=list(data.get("cgt_paid_p50", [])),
            cgt_paid_p5=list(data.get("cgt_paid_p5", [])),
            cgt_paid_p95=list(data.get("cgt_paid_p95", [])),
            cgt_without_floor_p50=list(data.get("cgt_without_floor_p50", [])),
            # Trajectory (Work Items 1, 10)
            bridge_by_age_ages=list(data.get("bridge_by_age_ages", [])),
            bridge_by_age_p5=list(data.get("bridge_by_age_p5", [])),
            bridge_by_age_p10=list(data.get("bridge_by_age_p10", [])),
            bridge_by_age_p25=list(data.get("bridge_by_age_p25", [])),
            bridge_by_age_p50=list(data.get("bridge_by_age_p50", [])),
            bridge_by_age_p75=list(data.get("bridge_by_age_p75", [])),
            bridge_by_age_p90=list(data.get("bridge_by_age_p90", [])),
            bridge_by_age_p95=list(data.get("bridge_by_age_p95", [])),
            mortgage_by_age_ages=list(data.get("mortgage_by_age_ages", [])),
            mortgage_by_age=data.get("mortgage_by_age", {}),
            offset_by_age=data.get("offset_by_age", {}),
            mortgage_rate_by_age=data.get("mortgage_rate_by_age", {}),
        )


# =============================================================================
# PROFILE
# =============================================================================


@dataclass
class ResultsSession:
    """In-memory session holding results and cached detail computations.

    Created after each simulation run and passed to the results menu loop.
    Detail views populate the optional fields on demand (lazy / opt-in).
    """

    results: SimulationResults
    household: Household
    inputs: SimulationInputs

    # Work Item 6 — Scenario comparisons (populated on demand)
    scenarios: dict[str, ScenarioComparisonResult] | None = None

    # Work Item 3 — Sequencing risk analysis (populated on demand)
    sequencing: Any | None = None

    # Work Item 9 — Earliest retirement search (populated on demand)
    retirement_search: Any | None = None


@dataclass
class Profile:
    """A saved user profile containing inputs and last results.

    Mutable because ``updated_at`` and ``last_results`` change on each save.
    """

    profile_name: str
    created_at: str = ""
    updated_at: str = ""
    inputs: SimulationInputs = field(default_factory=SimulationInputs)
    last_results: SimulationResults | None = None

    # ── Serialisation ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise profile to a JSON-compatible dict."""
        return {
            "profile_name": self.profile_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "inputs": _serialise_inputs(self.inputs),
            "last_results": self.last_results.summary_dict() if self.last_results else None,
            "_version": 2,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        """Deserialise profile from a JSON dict.

        Handles backward compatibility with v1 (legacy-specific) profiles.
        """
        # Upgrade v1 if needed
        data = _upgrade_v1_profile(data)

        inputs_data = data.get("inputs", {})
        inputs = _deserialise_inputs(inputs_data) if inputs_data else SimulationInputs()

        results_data = data.get("last_results")
        last_results = SimulationResults.from_dict(results_data) if results_data else None

        return cls(
            profile_name=data.get("profile_name", "Unnamed"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            inputs=inputs,
            last_results=last_results,
        )


# =============================================================================
# SERIALISATION HELPERS
# =============================================================================


def _serialise_earner(earner: Earner) -> dict[str, Any]:
    return {
        "label": earner.label,
        "salary": earner.salary,
        "super_balance": earner.super_balance,
        "salary_growth_rate": earner.salary_growth_rate,
        "retirement_age": earner.retirement_age,
        "birth_year": earner.birth_year,
        "super_access_age": earner.super_access_age,
        "sg_rate": earner.sg_rate,
        "employment_type": earner.employment_type,
        "pt_days_per_week": earner.pt_days_per_week,
        "pt_rate_mode": earner.pt_rate_mode,
        "pt_salary_pct": earner.pt_salary_pct,
        "pt_start_age": earner.pt_start_age,
        "pt_end_age": earner.pt_end_age,
        "pt_daily_rate": earner.pt_daily_rate,
        "pt_weeks_per_year": earner.pt_weeks_per_year,
        "personal_super_contributions_total_p_a": earner.personal_super_contributions_total_p_a,
        "non_concessional_contributions_p_a": earner.non_concessional_contributions_p_a,
        "super_growth_pct": earner.super_growth_pct,
        "super_glide_end_year": earner.super_glide_end_year,
        "super_glide_target_pct": earner.super_glide_target_pct,
        "super_asset_class": earner.super_asset_class,
        "super_mean_override": earner.super_mean_override,
        "super_std_override": earner.super_std_override,
        "self_employed_income": earner.self_employed_income,
        "self_employed_growth_rate": earner.self_employed_growth_rate,
        "self_employed_sg_applies": earner.self_employed_sg_applies,
    }


def _deserialise_earner(data: dict[str, Any]) -> Earner:
    sacrifice = data.get("personal_super_contributions_total_p_a")
    # Backward compatibility: map legacy is_employed to employment_type
    # Only use is_employed fallback when employment_type is NOT present
    if "employment_type" in data:
        emp_type = data["employment_type"]
    else:
        # Legacy profile: map is_employed bool -> employment_type
        is_emp = data.get("is_employed", True)
        emp_type = "employed" if is_emp else "not_employed"
    return Earner(
        label=data.get("label", "Earner 1"),
        salary=data.get("salary", 100_000.0),
        super_balance=data.get("super_balance", 100_000.0),
        salary_growth_rate=data.get("salary_growth_rate", 0.03),
        retirement_age=data.get("retirement_age", 50),
        birth_year=data.get("birth_year"),
        super_access_age=data.get("super_access_age", 60),
        sg_rate=data.get("sg_rate", 0.12),
        employment_type=emp_type,
        pt_days_per_week=data.get("pt_days_per_week", 0.0),
        pt_start_age=data.get("pt_start_age", -1),
        pt_end_age=data.get("pt_end_age", 65),
        pt_daily_rate=data.get("pt_daily_rate", 3_000.0),
        pt_weeks_per_year=data.get("pt_weeks_per_year", 48.0),
        pt_rate_mode=data.get("pt_rate_mode", "daily_rate"),
        pt_salary_pct=data.get("pt_salary_pct", 0.0),
        personal_super_contributions_total_p_a=sacrifice if sacrifice is not None else None,
        non_concessional_contributions_p_a=data.get("non_concessional_contributions_p_a", 0.0),
        super_growth_pct=data.get("super_growth_pct", 70.0),
        super_glide_end_year=data.get("super_glide_end_year"),
        super_glide_target_pct=data.get("super_glide_target_pct", 30.0),
        super_asset_class=data.get("super_asset_class", "equity"),
        super_mean_override=data.get("super_mean_override"),
        super_std_override=data.get("super_std_override"),
        self_employed_income=data.get("self_employed_income", 0.0),
        self_employed_growth_rate=data.get("self_employed_growth_rate", 0.005),
        self_employed_sg_applies=data.get("self_employed_sg_applies", False),
    )


def _serialise_child(child: Child) -> dict[str, Any]:
    return {
        "label": child.label,
        "age": child.age,
        "education_schedule": [[age, cost] for age, cost in child.education_schedule],
    }


def _deserialise_child(data: dict[str, Any]) -> Child:
    schedule_raw = data.get("education_schedule", [])
    return Child(
        label=data.get("label", "Child 1"),
        age=data.get("age", 5),
        education_schedule=tuple(tuple(pair) for pair in schedule_raw),
    )


def _serialise_mortgage(mortgage: MortgageAccount) -> dict[str, Any]:
    return {
        "label": mortgage.label,
        "principal": mortgage.principal,
        "interest_rate": mortgage.interest_rate,
        "monthly_payment": mortgage.monthly_payment,
        "offset_accounts": list(mortgage.offset_accounts),
        "offset_reserve_mode": mortgage.offset_reserve_mode,
        "offset_reserve_floor": mortgage.offset_reserve_floor,
        "loan_term_end_age": mortgage.loan_term_end_age,
        "interest_rate_stochastic": mortgage.interest_rate_stochastic,
        "interest_rate_vol": mortgage.interest_rate_vol,
        "interest_rate_kappa": mortgage.interest_rate_kappa,
        "interest_rate_theta": mortgage.interest_rate_theta,
        "interest_rate_corr": mortgage.interest_rate_corr,
    }


def _deserialise_mortgage(data: dict[str, Any]) -> MortgageAccount:
    return MortgageAccount(
        label=data.get("label", "Mortgage 1"),
        principal=data.get("principal", 500_000.0),
        interest_rate=data.get("interest_rate", 0.0605),
        monthly_payment=data.get("monthly_payment", 0.0),
        offset_accounts=tuple(data.get("offset_accounts", [])),
        offset_reserve_mode=data.get("offset_reserve_mode", "fixed"),
        offset_reserve_floor=data.get("offset_reserve_floor", 0.0),
        loan_term_end_age=data.get("loan_term_end_age"),
        interest_rate_stochastic=data.get("interest_rate_stochastic", False),
        interest_rate_vol=data.get("interest_rate_vol"),
        interest_rate_kappa=data.get("interest_rate_kappa"),
        interest_rate_theta=data.get("interest_rate_theta"),
        interest_rate_corr=data.get("interest_rate_corr"),
    )


def _serialise_account(account: InvestmentAccount) -> dict[str, Any]:
    return {
        "label": account.label,
        "market_value": account.market_value,
        "cost_basis": account.cost_basis,
        "asset_class": account.asset_class,
        "tax_jurisdiction": account.tax_jurisdiction,
        "cgt_rate": account.cgt_rate,
        "is_offset": account.is_offset,
        "fee_rate": account.fee_rate,
        "interest_rate": account.interest_rate,
        "ownership": {str(k): v for k, v in account.ownership.items()},
    }


def _deserialise_account(data: dict[str, Any]) -> InvestmentAccount:
    raw_ownership = data.get("ownership", {"0": 1.0})
    ownership = {int(k): v for k, v in raw_ownership.items()}
    # Validate ownership sums to ~1.0 — hard-block if malformed
    total = sum(ownership.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"Account '{data.get('label', '?')}' ownership sums to {total:.4f}, "
            f"expected 1.0.  Each account's ownership shares must sum to exactly "
            f"100% before the simulation can proceed."
        )
    return InvestmentAccount(
        label=data.get("label", "Account 1"),
        market_value=data.get("market_value", 0.0),
        cost_basis=data.get("cost_basis", 0.0),
        asset_class=data.get("asset_class", "equity"),
        tax_jurisdiction=data.get("tax_jurisdiction", "au"),
        cgt_rate=data.get("cgt_rate", 0.30),
        is_offset=data.get("is_offset", False),
        fee_rate=data.get("fee_rate", 0.0),
        interest_rate=data.get("interest_rate", 0.0),
        ownership=ownership,
    )


def _serialise_inputs(inputs: SimulationInputs) -> dict[str, Any]:
    return {
        "n_iterations": inputs.n_iterations,
        "inflation": inputs.inflation,
        "simulation_start_age": inputs.simulation_start_age,
        "cgt_on_drawdowns": inputs.cgt_on_drawdowns,
        "sell_strategy": inputs.sell_strategy,
        "sell_order": list(inputs.sell_order),
        "conc_cap_growth_rate": inputs.conc_cap_growth_rate,
        "sg_max_base_growth_rate": inputs.sg_max_base_growth_rate,
        "seed": inputs.seed,
        "super_fee_rate": inputs.super_fee_rate,
        "mls_enabled": inputs.mls_enabled,
        "mls_tiered": inputs.mls_tiered,
        "mls_threshold": inputs.mls_threshold,
        "mls_rate": inputs.mls_rate,
        "bracket_growth_rate": inputs.bracket_growth_rate,
        "div293_threshold": inputs.div293_threshold,
        "div293_rate": inputs.div293_rate,
        "div293_growth_rate": inputs.div293_growth_rate,
        "surplus_investment_pct": inputs.surplus_investment_pct,
        "stochastic_inflation": inputs.stochastic_inflation,
        "success_threshold": inputs.success_threshold,
        "household": {
            "earners": [_serialise_earner(e) for e in inputs.household.earners],
            "children": [_serialise_child(c) for c in inputs.household.children],
            "mortgages": [_serialise_mortgage(m) for m in inputs.household.mortgages],
            "investment_accounts": [
                _serialise_account(a) for a in inputs.household.investment_accounts
            ],
            "base_living_expenses": inputs.household.base_living_expenses,
            "retirement_target": inputs.household.retirement_target,
        },
    }


def _repair_offset_links(
    mortgages: tuple[MortgageAccount, ...],
    accounts: tuple[InvestmentAccount, ...],
) -> tuple[MortgageAccount, ...]:
    """Rebuild mortgage→offset linkages for loaded profiles.

    Called at deserialisation time to repair profiles that were saved
    before ``_link_offsets_to_mortgages()`` populated ``offset_accounts``
    on each mortgage.  Without this step the simulation ignores offset
    balances during mortgage amortisation, leaving a large fraction of
    the loan outstanding at horizon.

    Strategy:
    * Single-mortgage households: link all ``is_offset`` accounts to that
      mortgage unconditionally (the common case).
    * Multi-mortgage households: only rebuild links when all mortgages
      have an empty ``offset_accounts`` AND there is exactly one offset
      account — link it to the first mortgage.  In all other multi-
      mortgage cases we cannot guess the intended mapping and leave the
      data as-is (the interactive wizard is required to set up the links).
    """
    if not mortgages or not accounts:
        return mortgages

    offset_accounts = [a for a in accounts if a.is_offset]
    if not offset_accounts:
        return mortgages  # nothing to link

    offset_labels = tuple(a.label for a in offset_accounts)

    if len(mortgages) == 1:
        # Single mortgage: all offsets belong to it
        m = mortgages[0]
        if set(m.offset_accounts) != set(offset_labels):
            repaired = MortgageAccount(
                label=m.label,
                principal=m.principal,
                interest_rate=m.interest_rate,
                monthly_payment=m.monthly_payment,
                offset_accounts=offset_labels,
                offset_reserve_mode=m.offset_reserve_mode,
                offset_reserve_floor=m.offset_reserve_floor,
                loan_term_end_age=m.loan_term_end_age,
                interest_rate_stochastic=m.interest_rate_stochastic,
                interest_rate_vol=m.interest_rate_vol,
                interest_rate_kappa=m.interest_rate_kappa,
                interest_rate_theta=m.interest_rate_theta,
                interest_rate_corr=m.interest_rate_corr,
            )
            return (repaired,)
        return mortgages

    # Multi-mortgage: attempt repair only when simple enough to guess
    all_empty = all(len(m.offset_accounts) == 0 for m in mortgages)
    if all_empty and len(offset_accounts) == 1:
        # One offset, multiple mortgages — link to first as best guess
        result: list[MortgageAccount] = []
        for i, m in enumerate(mortgages):
            if i == 0:
                result.append(
                    MortgageAccount(
                        label=m.label,
                        principal=m.principal,
                        interest_rate=m.interest_rate,
                        monthly_payment=m.monthly_payment,
                        offset_accounts=(offset_labels[0],),
                        offset_reserve_mode=m.offset_reserve_mode,
                        offset_reserve_floor=m.offset_reserve_floor,
                        loan_term_end_age=m.loan_term_end_age,
                        interest_rate_stochastic=m.interest_rate_stochastic,
                        interest_rate_vol=m.interest_rate_vol,
                        interest_rate_kappa=m.interest_rate_kappa,
                        interest_rate_theta=m.interest_rate_theta,
                        interest_rate_corr=m.interest_rate_corr,
                    )
                )
            else:
                result.append(m)
        return tuple(result)

    return mortgages  # too ambiguous; leave for interactive repair


def _deserialise_inputs(data: dict[str, Any]) -> SimulationInputs:
    household_data = data.get("household", {})

    accounts = tuple(_deserialise_account(a) for a in household_data.get("investment_accounts", []))
    mortgages_raw = tuple(_deserialise_mortgage(m) for m in household_data.get("mortgages", []))
    # Repair offset→mortgage links that may be missing in older profiles
    mortgages = _repair_offset_links(mortgages_raw, accounts)

    household = Household(
        earners=tuple(_deserialise_earner(e) for e in household_data.get("earners", [])),
        children=tuple(_deserialise_child(c) for c in household_data.get("children", [])),
        mortgages=mortgages,
        investment_accounts=accounts,
        base_living_expenses=household_data.get("base_living_expenses", 60_000.0),
        retirement_target=household_data.get("retirement_target", 80_000.0),
    )

    return SimulationInputs(
        household=household,
        n_iterations=data.get("n_iterations", 5_000),
        inflation=data.get("inflation", 0.025),
        simulation_start_age=data.get("simulation_start_age", 37),
        cgt_on_drawdowns=data.get("cgt_on_drawdowns", True),
        sell_strategy=data.get("sell_strategy", "waterfall"),
        sell_order=tuple(data.get("sell_order", [])),
        conc_cap_growth_rate=data.get("conc_cap_growth_rate", 0.03),
        sg_max_base_growth_rate=data.get("sg_max_base_growth_rate", 0.03),
        seed=data.get("seed"),
        super_fee_rate=data.get("super_fee_rate", 0.0085),
        mls_enabled=data.get("mls_enabled", False),
        mls_tiered=data.get("mls_tiered", True),
        mls_threshold=data.get("mls_threshold", 90_000.0),
        mls_rate=data.get("mls_rate", 0.01),
        bracket_growth_rate=data.get("bracket_growth_rate", 0.025),
        div293_threshold=data.get("div293_threshold", 250_000.0),
        div293_rate=data.get("div293_rate", 0.15),
        div293_growth_rate=data.get("div293_growth_rate", 0.025),
        surplus_investment_pct=data.get("surplus_investment_pct", 0.0),
        stochastic_inflation=data.get("stochastic_inflation", False),
        success_threshold=data.get("success_threshold", 0.95),
    )


# =============================================================================
# BACKWARD COMPATIBILITY (v1 → v2 profile upgrade)
# =============================================================================


def _upgrade_v1_profile(data: dict[str, Any]) -> dict[str, Any]:
    """Detect and convert old-format v1 profiles to v2.

    v1 profiles stored data under ``inputs.finances.salary_h`` etc.
    v2 uses ``inputs.household.earners``, ``inputs.household.children``, etc.

    Args:
        data: Raw profile dict (possibly v1 format).

    Returns:
        Upgraded dict in v2 format, or unchanged data if already v2.

    """
    if "_version" in data and data["_version"] >= 2:
        return data  # Already v2+

    inputs = data.get("inputs", {})
    finances = inputs.get("finances", {})

    # Heuristic: v1 has salary_h in finances
    if "salary_h" not in finances:
        return data  # Not a v1 profile (or no data at all)

    salary_h = finances.get("salary_h", 320_000.0)
    salary_w = finances.get("salary_w", 200_000.0)
    super_h = finances.get("super_h", 310_720.71)
    super_w = finances.get("super_w", 238_380.63)
    child_age = finances.get("child_age", 2)

    # Build earners
    earner1 = Earner(
        label="Earner 1",
        salary=salary_h,
        super_balance=super_h,
        salary_growth_rate=0.03,
        retirement_age=inputs.get("retire_age", 50),
        super_access_age=60,
        pt_days_per_week=inputs.get("pt_days_per_week", 0.0),
        pt_rate_mode=inputs.get("pt_rate_mode", "daily_rate"),
        pt_salary_pct=inputs.get("pt_salary_pct", 0.0),
        pt_start_age=inputs.get("pt_start_age", 50),
        pt_end_age=inputs.get("pt_end_age", 52),
    )
    earner2 = Earner(
        label="Earner 2",
        salary=salary_w,
        super_balance=super_w,
        salary_growth_rate=0.025,
        retirement_age=inputs.get("retire_age", 50),
        super_access_age=60,
    )

    # Build child
    children: list[Child] = []
    if child_age >= 0:
        from primitives import EDU_SCHEDULE_TODAY

        children.append(
            Child(
                label="Child 1",
                age=child_age,
                education_schedule=tuple((age, cost) for age, cost in EDU_SCHEDULE_TODAY.items()),
            )
        )

    # Build mortgage
    mortgages: list[MortgageAccount] = []
    mortgage_principal = finances.get("mortgage", 0.0)
    if mortgage_principal > 0:
        mortgages.append(
            MortgageAccount(
                label="Mortgage 1",
                principal=mortgage_principal,
                interest_rate=0.0605,
                monthly_payment=5491.43,
                offset_accounts=("Offset 1",),
            )
        )

    # Build investment accounts
    accounts: list[InvestmentAccount] = []
    offset_bal = finances.get("offset", 0.0)
    if offset_bal > 0:
        accounts.append(
            InvestmentAccount(
                label="Offset 1",
                market_value=offset_bal,
                cost_basis=offset_bal,
                asset_class="cash",
                tax_jurisdiction="au",
                cgt_rate=0.0,
                is_offset=True,
            )
        )
    uk_etfs = finances.get("uk_etfs", 0.0)
    uk_basis = finances.get("uk_basis", 0.0)
    if uk_etfs > 0:
        accounts.append(
            InvestmentAccount(
                label="UK ETFs",
                market_value=uk_etfs,
                cost_basis=uk_basis,
                asset_class="equity",
                tax_jurisdiction="uk",
                cgt_rate=0.30,
            )
        )
    au_etfs = finances.get("au_etfs", 0.0)
    au_basis = finances.get("au_basis", 0.0)
    if au_etfs > 0:
        accounts.append(
            InvestmentAccount(
                label="AU ETFs",
                market_value=au_etfs,
                cost_basis=au_basis,
                asset_class="equity",
                tax_jurisdiction="au",
                cgt_rate=0.30,
            )
        )

    household = Household(
        earners=(earner1, earner2),
        children=tuple(children),
        mortgages=tuple(mortgages),
        investment_accounts=tuple(accounts),
        base_living_expenses=finances.get("living_expenses", 75_000.0),
        retirement_target=100_000.0,
    )

    upgraded: dict[str, Any] = {
        "_version": 2,
        "profile_name": data.get("profile_name", "Upgraded (v1)"),
        "created_at": data.get("created_at", ""),
        "updated_at": data.get("updated_at", ""),
        "inputs": {
            "household": {
                "earners": [_serialise_earner(e) for e in household.earners],
                "children": [_serialise_child(c) for c in household.children],
                "mortgages": [_serialise_mortgage(m) for m in household.mortgages],
                "investment_accounts": [
                    _serialise_account(a) for a in household.investment_accounts
                ],
                "base_living_expenses": household.base_living_expenses,
                "retirement_target": household.retirement_target,
            },
            "n_iterations": inputs.get("n_iterations", 5_000),
            "inflation": inputs.get("inflation", 0.025),
            "simulation_start_age": inputs.get("simulation_start_age", 37),
            "cgt_on_drawdowns": inputs.get("cgt_on_drawdowns", True),
            "sell_strategy": "waterfall",
            "sell_order": [],
        },
        "last_results": data.get("last_results"),
    }

    return upgraded
