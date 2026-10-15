"""Data models for the Monte Carlo CLI tool.

Pure dataclasses with no I/O or UI dependencies. Scenario inputs validate
themselves in ``__post_init__``. ``SimulationResults`` is immutable (tuples
and read-only mappings). ``Profile`` and ``ResultsSession`` are mutable
session records.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import date
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Literal,
    NewType,
    TypeVar,
    get_args,
    overload,
)

from .config import DEFAULT_EDU_SCHEDULE, DEFAULT_SIM_START_AGE

_T = TypeVar("_T")

if TYPE_CHECKING:
    from .analysis import ScenarioComparisonResult

# =============================================================================
# UNIT TYPES — documentation-only NewTypes for clarity
# =============================================================================

Percent = NewType("Percent", float)
"""A percentage stored on a 0–100 scale (e.g. 70.0 means 70%)."""

Rate = NewType("Rate", float)
"""A rate stored as a decimal (e.g. 0.005 means 0.5%)."""

EmploymentType = Literal["employed", "self_employed", "both", "not_employed"]
"""Income/super modelling mode for an ``Earner``."""
PtRateMode = Literal["daily_rate", "salary_pct"]
"""How a part-time income rate is entered."""
AssetClass = Literal["equity", "intl_equity", "bonds", "cash", "property"]
"""Asset class key into ``returns.ASSET_CLASS_PARAMS``."""
OffsetReserveMode = Literal["fixed", "stall_prevention", "interest_cancelling"]
"""How the offset reserve is determined for a mortgage offset account."""
SellStrategy = Literal["waterfall"]
"""Order in which investment accounts are sold to fund drawdowns."""

EMPLOYMENT_TYPES: tuple[str, ...] = get_args(EmploymentType)
PT_RATE_MODES: tuple[str, ...] = get_args(PtRateMode)
ASSET_CLASSES: tuple[str, ...] = get_args(AssetClass)
OFFSET_RESERVE_MODES: tuple[str, ...] = get_args(OffsetReserveMode)
SELL_STRATEGIES: tuple[str, ...] = get_args(SellStrategy)


def _check_choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    """Raise ``ValueError`` unless ``value`` is one of ``choices``."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices} (got {value!r})")


def _check_finite_nonneg(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is a finite number >= 0."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0 (got {value})")


def _check_age(name: str, value: int | None) -> None:
    """Raise ``ValueError`` unless ``value`` is None or an age in 0..120."""
    if value is not None and not 0 <= value <= 120:
        raise ValueError(f"{name} must be an age in 0..120 (got {value})")


# =============================================================================
# OWNERSHIP — immutable, hashable mapping of earner index → share
# =============================================================================


class Ownership(Mapping[int, float]):
    """Immutable, hashable fractional ownership of an investment account.

    A ``Mapping[int, float]`` that cannot be mutated after construction and
    whose ``__hash__`` is well-defined (the ``frozen`` dataclass generator
    raises ``TypeError`` on a plain ``dict`` field, which previously made
    ``InvestmentAccount`` and ``Household`` unhashable).

    All read paths used elsewhere — ``.get(key, default)``, ``.items()``,
    ``[key]``, iteration, ``len`` — work unchanged.
    """

    __slots__ = ("_data", "_hash")

    def __init__(self, data: Mapping[int, float] | dict[int, float]) -> None:
        # Store as a tuple of sorted (key, value) pairs so hashing is stable.
        self._data: tuple[tuple[int, float], ...] = tuple(sorted(data.items()))
        self._hash: int = hash(self._data)

    def __getitem__(self, key: int) -> float:
        """Return the share for earner ``key``."""
        for k, v in self._data:
            if k == key:
                return v
        raise KeyError(key)

    @overload
    def get(self, key: int) -> float | None: ...

    @overload
    def get(self, key: int, default: float) -> float: ...

    @overload
    def get(self, key: int, default: _T) -> float | _T: ...

    def get(self, key: int, default: float | _T | None = None) -> float | _T | None:
        """Return the share for earner ``key``, or ``default`` if absent."""
        for k, v in self._data:
            if k == key:
                return v
        return default

    def __iter__(self) -> Iterator[int]:
        """Iterate over earner indices."""
        return iter(k for k, _ in self._data)

    def __len__(self) -> int:
        """Return number of earners with ownership shares."""
        return len(self._data)

    def __hash__(self) -> int:
        """Hash based on the sorted (key, value) pairs."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Equality compares the underlying (key, value) pairs."""
        if isinstance(other, Ownership):
            return self._data == other._data
        if isinstance(other, Mapping):
            return dict(self._data) == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        """Return repr string."""
        return f"Ownership({dict(self._data)})"

    def to_dict(self) -> dict[int, float]:
        """Return a plain dict copy for serialisation."""
        return dict(self._data)


# =============================================================================
# AUSTRALIAN PRESERVATION AGE LOOKUP
# =============================================================================


def preservation_age(birth_year: int, birth_month: int | None = None) -> int:
    """Return the Australian superannuation preservation age for a birth year.

    Preservation age is the minimum age at which a person can access their
    superannuation. It depends on date of birth, rising from 55 (born before
    1 July 1960) to 60 (born after 30 June 1964).

    Args:
        birth_year: Calendar year of birth (e.g. 1970).
        birth_month: Month of birth (1–12).  If ``None`` (default), the
            function returns the **conservative** (higher) preservation age
            for the ambiguous 1960–1964 cohorts, so the model errs in its
            documented conservative direction.  When supplied, the exact
            preservation age for that birth date is returned.

    Returns:
        Preservation age (55–60).

    Raises:
        ValueError: If ``birth_year`` is before 1900 or after 2100, or if
            ``birth_month`` is not in 1–12.

    Note:
        The statutory table keys off the fiscal year: a person born on
        1 July 1960 has preservation age 56, while one born on 30 June
        1960 has preservation age 55.  When the month is unknown, the
        default is the **higher** (more conservative) of the two possible
        ages for that calendar year, so the model does not silently bias
        ``p_success`` optimisticly for July–December births in the
        1960–1964 transition window.

    """
    if birth_year < 1900 or birth_year > 2100:
        raise ValueError(f"birth_year {birth_year} out of supported range (1900–2100)")
    if birth_month is not None and (birth_month < 1 or birth_month > 12):
        raise ValueError(f"birth_month {birth_month} out of range (1–12)")

    if birth_month is not None:
        # Exact calculation: preservation age changes at 1 July of each year.
        # Born before 1 July 1960 → 55; 1 Jul 1960 – 30 Jun 1961 → 56; etc.
        fiscal_year_start = birth_year if birth_month >= 7 else birth_year - 1
        if fiscal_year_start < 1960:
            return 55
        if fiscal_year_start < 1961:
            return 56
        if fiscal_year_start < 1962:
            return 57
        if fiscal_year_start < 1963:
            return 58
        if fiscal_year_start < 1964:
            return 59
        return 60

    # Month unknown: default to the conservative (higher) age for the
    # ambiguous 1960–1964 cohorts.  A July–December birth in year X has
    # the same preservation age as a January–June birth in year X+1.
    if birth_year < 1960:
        return 55
    if birth_year < 1961:
        return 56  # Was 55: Jul–Dec 1960 is actually 56
    if birth_year < 1962:
        return 57  # Was 56: Jul–Dec 1961 is actually 57
    if birth_year < 1963:
        return 58  # Was 57: Jul–Dec 1962 is actually 58
    if birth_year < 1964:
        return 59  # Was 58: Jul–Dec 1963 is actually 59
    return 60  # >= 1964: Jul–Dec 1964 is 60, and all later years are 60


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
    start_age: int | None = None
    """Age at simulation start for this earner (D15). ``None`` means use
    ``SimulationInputs.simulation_start_age``. Validated 0..120."""
    birth_year: int | None = None
    """Birth year for preservation-age lookup. Used by
    ``effective_super_access_age`` when ``super_access_age`` is None."""
    super_access_age: int | None = None
    """Age at which super is accessible. ``None`` means use
    ``preservation_age(birth_year)``. At least one of ``super_access_age``
    and ``birth_year`` must be set. Read via ``effective_super_access_age``."""
    sg_rate: float = 0.12
    employment_type: EmploymentType = "employed"
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
    pt_start_age: int | None = None
    """Age at which part-time income begins. ``None`` (default) means use
    ``retirement_age``. Read via ``effective_pt_start_age``."""
    pt_end_age: int | None = None
    """Age at which part-time income ends. ``None`` (default) means use the
    effective super access age (bridge horizon). Read via ``effective_pt_end_age``."""
    pt_daily_rate: float = 3_000.0
    pt_weeks_per_year: float = 48.0
    pt_rate_mode: PtRateMode = "daily_rate"
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
    super_asset_class: AssetClass = "equity"
    """Asset class for this earner's superannuation returns.
    Used when ``super_mean_override`` is not set AND growth blending is
    not applicable (kept for backward compatibility).
    """
    super_mean_override: float | None = None
    """Override mean return for this earner's super. If None, uses the
    asset class lookup from ``super_asset_class``.

    Note: The defaults these overrides fall back to (``EQ_MEAN``, ``EQ_STD``,
    etc. in ``returns.py``) have mixed provenance. Equity mean is sourced
    (Credit Suisse / UBS Yearbook); all standard deviations and most other
    means are unsourced placeholders. See ``returns.py`` module docstring
    and per-constant comments for full provenance details.
    """
    super_std_override: float | None = None
    """Override std deviation for this earner's super. Only used when
    ``super_mean_override`` is also set.

    Same provenance caveat as ``super_mean_override`` — default fallback
    standard deviations are unsourced placeholders.
    """

    def __post_init__(self) -> None:
        """Validate fields. Sentinels are never overwritten here (D16)."""
        if self.super_access_age is None and self.birth_year is None:
            raise ValueError(f"Earner {self.label!r}: set super_access_age or birth_year")
        if self.birth_year is not None:
            preservation_age(self.birth_year)  # raises ValueError outside 1900–2100
        _check_choice("employment_type", self.employment_type, EMPLOYMENT_TYPES)
        _check_choice("pt_rate_mode", self.pt_rate_mode, PT_RATE_MODES)
        _check_choice("super_asset_class", self.super_asset_class, ASSET_CLASSES)
        for name in ("start_age", "super_access_age", "pt_start_age", "pt_end_age"):
            _check_age(name, getattr(self, name))
        if self.retirement_age < 0:
            raise ValueError(f"retirement_age must be >= 0 (got {self.retirement_age})")
        for name in (
            "salary",
            "super_balance",
            "self_employed_income",
            "pt_daily_rate",
            "pt_salary_pct",
            "non_concessional_contributions_p_a",
        ):
            _check_finite_nonneg(name, getattr(self, name))
        if self.personal_super_contributions_total_p_a is not None:
            _check_finite_nonneg(
                "personal_super_contributions_total_p_a",
                self.personal_super_contributions_total_p_a,
            )
        if not 0.0 <= self.pt_days_per_week <= 7.0:
            raise ValueError(f"pt_days_per_week must be 0–7 (got {self.pt_days_per_week})")
        if not 0.0 <= self.pt_weeks_per_year <= 52.0:
            raise ValueError(f"pt_weeks_per_year must be 0–52 (got {self.pt_weeks_per_year})")
        # Unit validation: _pct fields are 0–100, _rate fields are ≥ -1.0.
        # Bounds are chosen to accept every value the UI can produce; see
        # ui_wizard.py prompt ranges (salary_growth -5..15%, sg 0..20%, etc.).
        if not 0.0 <= self.super_growth_pct <= 100.0:
            raise ValueError(f"super_growth_pct must be 0–100 (got {self.super_growth_pct})")
        if not 0.0 <= self.super_glide_target_pct <= 100.0:
            raise ValueError(
                f"super_glide_target_pct must be 0–100 (got {self.super_glide_target_pct})"
            )
        if self.salary_growth_rate < -1.0:
            raise ValueError(f"salary_growth_rate must be ≥ -1.0 (got {self.salary_growth_rate})")
        if self.self_employed_growth_rate < -1.0:
            raise ValueError(
                f"self_employed_growth_rate must be ≥ -1.0 (got {self.self_employed_growth_rate})"
            )
        if not 0.0 <= self.sg_rate <= 1.0:
            raise ValueError(f"sg_rate must be 0–1 (got {self.sg_rate})")

    @property
    def effective_pt_start_age(self) -> int:
        """Part-time start age: ``pt_start_age``, else ``retirement_age``."""
        return self.retirement_age if self.pt_start_age is None else self.pt_start_age

    @property
    def effective_super_access_age(self) -> int:
        """Super access age: ``super_access_age``, else preservation age from ``birth_year``."""
        if self.super_access_age is not None:
            return self.super_access_age
        if self.birth_year is None:  # unreachable: __post_init__ enforces one of the two
            raise ValueError(f"Earner {self.label!r}: set super_access_age or birth_year")
        return preservation_age(self.birth_year)

    @property
    def effective_pt_end_age(self) -> int:
        """Part-time end age: ``pt_end_age``, else the effective super access age."""
        return self.effective_super_access_age if self.pt_end_age is None else self.pt_end_age

    def resolved_start_age(self, reference_start_age: int) -> int:
        """Age at simulation start: ``start_age``, else ``reference_start_age`` (D15)."""
        return reference_start_age if self.start_age is None else self.start_age


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
    reprice_payment: bool = False
    """If True, the rate changes, and ``loan_term_end_age`` is set, the P&I
    payment is recomputed for the remaining term each year. Default False
    keeps ``monthly_payment`` fixed."""
    offset_accounts: tuple[str, ...] = ()
    offset_reserve_mode: OffsetReserveMode = "fixed"
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

    def __post_init__(self) -> None:
        """Validate amounts, rates, and stochastic-rate parameters."""
        _check_finite_nonneg("principal", self.principal)
        _check_finite_nonneg("monthly_payment", self.monthly_payment)
        _check_finite_nonneg("offset_reserve_floor", self.offset_reserve_floor)
        _check_choice("offset_reserve_mode", self.offset_reserve_mode, OFFSET_RESERVE_MODES)
        _check_age("loan_term_end_age", self.loan_term_end_age)
        if not 0.0 <= self.interest_rate <= 1.0:
            raise ValueError(f"interest_rate must be 0–1 (got {self.interest_rate})")
        if self.interest_rate_vol is not None and not self.interest_rate_vol >= 0.0:
            raise ValueError(f"interest_rate_vol must be >= 0 (got {self.interest_rate_vol})")
        if self.interest_rate_kappa is not None and not self.interest_rate_kappa > 0.0:
            raise ValueError(f"interest_rate_kappa must be > 0 (got {self.interest_rate_kappa})")
        if self.interest_rate_theta is not None and not self.interest_rate_theta > 0.0:
            raise ValueError(f"interest_rate_theta must be > 0 (got {self.interest_rate_theta})")
        if self.interest_rate_corr is not None and not -1.0 <= self.interest_rate_corr <= 1.0:
            raise ValueError(
                f"interest_rate_corr must be in [-1, 1] (got {self.interest_rate_corr})"
            )


@dataclass(frozen=True)
class InvestmentAccount:
    """An investment account (equity, cash, property, etc.) outside super.

    ``asset_class`` describes the type of asset for return modelling.
    ``is_offset`` marks this as a mortgage offset account (treated as cash).
    ``interest_rate`` is an optional fixed-return override (e.g. 0.01 for 1%
    cash accounts). ``None`` (default) uses the asset class return model
    instead (e.g. ~7% for equity, ~3% for bonds); 0.0 is a real 0% return.
    """

    label: str = "Account 1"
    market_value: float = 0.0
    cost_basis: float = 0.0
    asset_class: AssetClass = "equity"
    is_offset: bool = False
    fee_rate: float = 0.0
    interest_rate: float | None = None
    """Expected annual return override for this account (decimal).

    ``None`` (default) means use the asset class's default mean and volatility.
    When set, the return is lognormal with this user-specified mean and the
    asset class's standard deviation and equity correlation — NOT a fixed
    deterministic rate.  For example, an equity account with
    ``interest_rate=0.08`` gets ~8% expected return with ~15% volatility.
    """
    ownership: Ownership = field(default_factory=lambda: Ownership({0: 1.0}))
    """Fractional ownership per earner index (matches ``Household.earners`` tuple).

    An immutable, hashable :class:`Ownership` mapping of earner index → share.
    Shares must sum to 1.0 (±0.001).  Default ``{0: 1.0}`` = 100% to Earner 1,
    which preserves pre-Phase-2 behaviour and provides a safe default for
    single-earner households.

    Ownership indices are stable across a simulation trial — the earner tuple
    order is invariant, matching the indexing of ``super_balances[]``,
    the per-owner tax positions, etc.
    """

    def __post_init__(self) -> None:
        """Validate ownership invariants and unit ranges at construction."""
        # Coerce a plain dict to Ownership (e.g. from deserialisation).
        if not isinstance(self.ownership, Ownership):
            object.__setattr__(self, "ownership", Ownership(self.ownership))
        _check_choice("asset_class", self.asset_class, ASSET_CLASSES)
        _check_finite_nonneg("market_value", self.market_value)
        _check_finite_nonneg("cost_basis", self.cost_basis)
        total = sum(self.ownership.values())
        if not math.isfinite(total) or abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Account '{self.label}' ownership shares sum to {total:.4f}, "
                f"expected 1.0.  Each account's ownership shares must sum to "
                f"exactly 100%."
            )
        for ei, share in self.ownership.items():
            if ei < 0:
                raise ValueError(f"Account '{self.label}' has negative earner index {ei}.")
            if share < 0:
                raise ValueError(
                    f"Account '{self.label}' has negative share {share} for earner {ei}."
                )
        if not 0.0 <= self.fee_rate <= 1.0:
            raise ValueError(f"fee_rate must be 0–1 (got {self.fee_rate})")
        if self.interest_rate is not None and not 0.0 <= self.interest_rate <= 1.0:
            raise ValueError(f"interest_rate must be 0–1 or None (got {self.interest_rate})")


@dataclass(frozen=True)
class Household:
    """Complete household definition for a simulation run.

    Default is a single-earner household with no children, no mortgage,
    and no investment accounts.
    """

    earners: tuple[Earner, ...] = (Earner(super_access_age=60),)
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

    def __post_init__(self) -> None:
        """Cross-field validation: earners, unique labels, ownership and offset links."""
        if not self.earners:
            raise ValueError("Household needs at least one earner")
        _check_finite_nonneg("base_living_expenses", self.base_living_expenses)
        _check_finite_nonneg("retirement_target", self.retirement_target)
        for kind, items in (
            ("earner", self.earners),
            ("child", self.children),
            ("mortgage", self.mortgages),
            ("investment account", self.investment_accounts),
        ):
            labels = [item.label for item in items]
            if len(set(labels)) != len(labels):
                raise ValueError(f"duplicate {kind} labels: {labels}")
        for acct in self.investment_accounts:
            for ei, _share in acct.ownership.items():
                if ei >= len(self.earners):
                    raise ValueError(
                        f"Account '{acct.label}' is owned by earner index {ei}, "
                        f"but the household has {len(self.earners)} earner(s)"
                    )
        offset_accounts = {a.label: a for a in self.investment_accounts}
        for mort in self.mortgages:
            for label in mort.offset_accounts:
                offset = offset_accounts.get(label)
                if offset is None:
                    raise ValueError(
                        f"Mortgage '{mort.label}' links to unknown offset account '{label}'"
                    )
                if not offset.is_offset:
                    raise ValueError(
                        f"Mortgage '{mort.label}' links to '{label}', "
                        "which is not an offset account"
                    )


# =============================================================================
# SIMULATION INPUTS / RESULTS
# =============================================================================


def _current_fy_start_year() -> int:
    """Start year of the current Australian financial year (1 July start)."""
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


@dataclass(frozen=True)
class SimulationInputs:
    """All user-configurable inputs for a Monte Carlo simulation run."""

    household: Household = field(default_factory=Household)
    n_iterations: int = 5_000
    inflation: float = 0.025
    simulation_start_age: int = DEFAULT_SIM_START_AGE
    simulation_start_year: int = field(default_factory=_current_fy_start_year)
    """Calendar year in which simulation year 0 falls.

    Anchors the model to the real calendar so date-based tax rules (the
    30 June 2027 CGT reform line) can be located in simulation time.
    """
    cgt_on_drawdowns: bool = True
    sell_strategy: SellStrategy = "waterfall"
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

    When True (default), the MLS rate is computed from ``tax.mls_rate()``
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

    def __post_init__(self) -> None:
        """Validate unit ranges on percent and rate fields."""
        # _pct fields: 0–100.  _rate fields: 0–1.0.  Bounds accept every
        # value the UI can produce.
        if not 0.0 <= self.surplus_investment_pct <= 100.0:
            raise ValueError(
                f"surplus_investment_pct must be 0–100 (got {self.surplus_investment_pct})"
            )
        if not 0.0 <= self.inflation <= 1.0:
            raise ValueError(f"inflation must be 0–1 (got {self.inflation})")
        if not 0.0 <= self.super_fee_rate <= 1.0:
            raise ValueError(f"super_fee_rate must be 0–1 (got {self.super_fee_rate})")
        if not 0.0 <= self.mls_rate <= 1.0:
            raise ValueError(f"mls_rate must be 0–1 (got {self.mls_rate})")
        if not 0.0 <= self.bracket_growth_rate <= 1.0:
            raise ValueError(f"bracket_growth_rate must be 0–1 (got {self.bracket_growth_rate})")
        if not 0.0 <= self.div293_rate <= 1.0:
            raise ValueError(f"div293_rate must be 0–1 (got {self.div293_rate})")
        if not 0.0 <= self.conc_cap_growth_rate <= 1.0:
            raise ValueError(f"conc_cap_growth_rate must be 0–1 (got {self.conc_cap_growth_rate})")
        if not 0.0 <= self.sg_max_base_growth_rate <= 1.0:
            raise ValueError(
                f"sg_max_base_growth_rate must be 0–1 (got {self.sg_max_base_growth_rate})"
            )

        _check_age("simulation_start_age", self.simulation_start_age)
        if self.n_iterations < 2:
            raise ValueError(f"n_iterations must be >= 2 (got {self.n_iterations})")
        if not 0.0 < self.success_threshold <= 1.0:
            raise ValueError(f"success_threshold must be in (0, 1] (got {self.success_threshold})")
        _check_choice("sell_strategy", self.sell_strategy, SELL_STRATEGIES)


def validate_horizon(household: Household, inputs: SimulationInputs) -> None:
    """Raise ``ValueError`` unless every earner has at least one simulated year before super access.

    Simulation year ``y`` falls at ``resolved_start_age + y`` for each earner (D15), so the
    horizon is ``min over earners of (effective_super_access_age - resolved_start_age)``;
    it must be positive.
    """
    for earner in household.earners:
        start = earner.resolved_start_age(inputs.simulation_start_age)
        if earner.effective_super_access_age <= start:
            raise ValueError(
                f"Earner {earner.label!r}: super access age "
                f"{earner.effective_super_access_age} must be above start age {start}"
            )


def _freeze_value(value: object) -> object:
    """Recursively turn lists into tuples and dicts into read-only mappings (L5)."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    return value


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
    per_earner_super_p50: Mapping[str, float] = field(default_factory=dict)
    remaining_mortgage_p50: Mapping[str, float] = field(default_factory=dict)
    mortgage_term_clearance_rate: float = 1.0
    """Proportion of trials where ALL mortgages with a term end cleared by that term."""
    per_mortgage_term_cleared_pct: Mapping[str, float] = field(default_factory=dict)
    """Per-mortgage term-clearance rate across all checkable trials."""
    per_mortgage_not_checkable: Mapping[str, int] = field(default_factory=dict)
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
    bridge_by_age_ages: Sequence[int] = field(default_factory=list)
    """Ages corresponding to the bridge trajectory rows. Populated when
    per-year trajectory capture is enabled."""
    bridge_by_age_p5: Sequence[float] = field(default_factory=list)
    bridge_by_age_p10: Sequence[float] = field(default_factory=list)
    bridge_by_age_p25: Sequence[float] = field(default_factory=list)
    bridge_by_age_p50: Sequence[float] = field(default_factory=list)
    bridge_by_age_p75: Sequence[float] = field(default_factory=list)
    bridge_by_age_p90: Sequence[float] = field(default_factory=list)
    bridge_by_age_p95: Sequence[float] = field(default_factory=list)

    # Work Item 2 — Near-miss / failure depth analysis
    near_miss_rate: float = 1.0
    """Fraction of trials that failed OR whose real floor fell at or below
    ``near_miss_threshold``. Failures are included. Populated by the Monte Carlo run."""
    near_miss_only_rate: float = 0.0
    """Fraction of trials with ``0 < floor <= near_miss_threshold``, whether or not
    they failed. Not cumulative with failures."""
    near_miss_threshold: float = 0.0
    """Real floor (today's dollars) at or below which a trial counts as a near miss.
    The Monte Carlo default is one year of ``household.retirement_target``."""
    failure_age_distribution: Mapping[int, int] = field(default_factory=dict)
    """Ages at which failures occurred, one entry per failing trial.
    Key = age, value = count of failing trials at that age.
    Empty dict when no failures.
    """
    warnings: tuple[str, ...] = ()
    """Structured run warnings (for example, low Monte Carlo variance). Empty when none."""

    # Work Items 4, 5 — Per-year drawdown composition
    offset_drawn_p50: Sequence[float] = field(default_factory=list)
    """Median offset draws by year (per-year breakdown, Work Item 4)."""
    non_offset_drawn_p50: Sequence[float] = field(default_factory=list)
    """Median non-offset draws by year (Work Item 4)."""
    offset_drawn_p5: Sequence[float] = field(default_factory=list)
    offset_drawn_p95: Sequence[float] = field(default_factory=list)
    cgt_paid_p50: Sequence[float] = field(default_factory=list)
    """Median CGT paid by year (with 30% floor, Work Item 5)."""
    cgt_paid_p5: Sequence[float] = field(default_factory=list)
    cgt_paid_p95: Sequence[float] = field(default_factory=list)
    cgt_without_floor_p50: Sequence[float] = field(default_factory=list)
    """Median CGT without 30% floor by year (counterfactual, Work Item 5)."""
    # Investment-income tax (D14), today's dollars: median by year, per-trial totals.
    income_tax_p50: Sequence[float] = field(default_factory=list)
    """Median investment-income tax paid by year (today's dollars)."""
    income_tax_total_p5: float = 0.0
    income_tax_total_p50: float = 0.0
    income_tax_total_p95: float = 0.0

    # Per-trial drawdown totals (today's dollars). True quantiles of the per-trial
    # totals, not sums of per-year medians (M4a).
    offset_drawn_total_p5: float = 0.0
    offset_drawn_total_p50: float = 0.0
    offset_drawn_total_p95: float = 0.0
    non_offset_drawn_total_p5: float = 0.0
    non_offset_drawn_total_p50: float = 0.0
    non_offset_drawn_total_p95: float = 0.0
    cgt_paid_total_p5: float = 0.0
    cgt_paid_total_p50: float = 0.0
    cgt_paid_total_p95: float = 0.0
    cgt_without_floor_total_p5: float = 0.0
    cgt_without_floor_total_p50: float = 0.0
    cgt_without_floor_total_p95: float = 0.0
    cgt_floor_extra_total_p5: float = 0.0
    """Per-trial total CGT attributable to the 30% minimum-rate floor
    (CGT paid minus CGT without the floor), P5."""
    cgt_floor_extra_total_p50: float = 0.0
    """Median per-trial extra CGT from the 30% minimum-rate floor. A true quantile
    of the per-trial difference, not a difference of two sums of per-year medians."""
    cgt_floor_extra_total_p95: float = 0.0

    # Work Item 10 — Mortgage amortisation by age (P5/P50/P95 per mortgage)
    mortgage_by_age_ages: Sequence[int] = field(default_factory=list)
    """Ages for the mortgage amortisation rows."""
    mortgage_by_age: Mapping[str, Mapping[str, Sequence[float]]] = field(default_factory=dict)
    """Per-mortgage balances by age with P5/P50/P95 percentiles.
    Structure: {label: {"p50": [y0, y1, ...], "p5": [...], "p95": [...]}}
    Nominal dollars (debt doesn't inflate).
    """
    offset_by_age: Mapping[str, Mapping[str, Sequence[float]]] = field(default_factory=dict)
    """Per-offset-account balances by age with P5/P50/P95 percentiles.
    Same structure as mortgage_by_age. Nominal dollars.
    """

    # Work Item 10 — Mortgage rate trajectory (P5/P50/P95 per mortgage)
    mortgage_rate_by_age: Mapping[str, Mapping[str, Sequence[float]]] = field(default_factory=dict)
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

    def __post_init__(self) -> None:
        """Freeze containers so results cannot be mutated after the run (L5).

        Lists become tuples and dicts become read-only mappings, copying
        them so later changes to the caller's objects cannot leak in.
        """
        for f in fields(self):
            object.__setattr__(self, f.name, _freeze_value(getattr(self, f.name)))

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
            "near_miss_only_rate": self.near_miss_only_rate,
            "warnings": list(self.warnings),
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
            "income_tax_p50": list(self.income_tax_p50),
            "income_tax_total_p5": self.income_tax_total_p5,
            "income_tax_total_p50": self.income_tax_total_p50,
            "income_tax_total_p95": self.income_tax_total_p95,
            "offset_drawn_total_p5": self.offset_drawn_total_p5,
            "offset_drawn_total_p50": self.offset_drawn_total_p50,
            "offset_drawn_total_p95": self.offset_drawn_total_p95,
            "non_offset_drawn_total_p5": self.non_offset_drawn_total_p5,
            "non_offset_drawn_total_p50": self.non_offset_drawn_total_p50,
            "non_offset_drawn_total_p95": self.non_offset_drawn_total_p95,
            "cgt_paid_total_p5": self.cgt_paid_total_p5,
            "cgt_paid_total_p50": self.cgt_paid_total_p50,
            "cgt_paid_total_p95": self.cgt_paid_total_p95,
            "cgt_without_floor_total_p5": self.cgt_without_floor_total_p5,
            "cgt_without_floor_total_p50": self.cgt_without_floor_total_p50,
            "cgt_without_floor_total_p95": self.cgt_without_floor_total_p95,
            "cgt_floor_extra_total_p5": self.cgt_floor_extra_total_p5,
            "cgt_floor_extra_total_p50": self.cgt_floor_extra_total_p50,
            "cgt_floor_extra_total_p95": self.cgt_floor_extra_total_p95,
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
            near_miss_only_rate=data.get("near_miss_only_rate", 0.0),
            warnings=tuple(data.get("warnings", ())),
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
            income_tax_p50=list(data.get("income_tax_p50", [])),
            income_tax_total_p5=data.get("income_tax_total_p5", 0.0),
            income_tax_total_p50=data.get("income_tax_total_p50", 0.0),
            income_tax_total_p95=data.get("income_tax_total_p95", 0.0),
            offset_drawn_total_p5=data.get("offset_drawn_total_p5", 0.0),
            offset_drawn_total_p50=data.get("offset_drawn_total_p50", 0.0),
            offset_drawn_total_p95=data.get("offset_drawn_total_p95", 0.0),
            non_offset_drawn_total_p5=data.get("non_offset_drawn_total_p5", 0.0),
            non_offset_drawn_total_p50=data.get("non_offset_drawn_total_p50", 0.0),
            non_offset_drawn_total_p95=data.get("non_offset_drawn_total_p95", 0.0),
            cgt_paid_total_p5=data.get("cgt_paid_total_p5", 0.0),
            cgt_paid_total_p50=data.get("cgt_paid_total_p50", 0.0),
            cgt_paid_total_p95=data.get("cgt_paid_total_p95", 0.0),
            cgt_without_floor_total_p5=data.get("cgt_without_floor_total_p5", 0.0),
            cgt_without_floor_total_p50=data.get("cgt_without_floor_total_p50", 0.0),
            cgt_without_floor_total_p95=data.get("cgt_without_floor_total_p95", 0.0),
            cgt_floor_extra_total_p5=data.get("cgt_floor_extra_total_p5", 0.0),
            cgt_floor_extra_total_p50=data.get("cgt_floor_extra_total_p50", 0.0),
            cgt_floor_extra_total_p95=data.get("cgt_floor_extra_total_p95", 0.0),
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
    retirement_search_all: Any | None = None
    """Cached result for 'both together' mode."""
    retirement_search_per_earner: dict[str, Any] = field(default_factory=dict)
    """Cached results for 'per earner' mode, keyed by earner label."""


class ProfileError(ValueError):
    """A saved profile is unreadable, malformed, or from an unsupported schema version."""


PROFILE_VERSION = 3
"""Current on-disk profile schema version. Older versions are upgraded on load."""


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
            "_version": PROFILE_VERSION,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        """Deserialise a profile dict, upgrading older schema versions first.

        Raises:
            ProfileError: If the data is malformed, or was written by a newer
                schema version than this application supports.
        """
        name = data.get("profile_name", "?")
        version = data.get("_version", 1)
        if not isinstance(version, int) or not 1 <= version <= PROFILE_VERSION:
            raise ProfileError(
                f"Profile '{name}' uses schema version {version!r}; this version of "
                f"DoneByFifty supports versions 1 to {PROFILE_VERSION}."
            )
        try:
            data = _upgrade_v1_profile(data)
            if data.get("_version", 2) == 2:
                data = _upgrade_v2_profile(data)
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
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfileError(f"Profile '{name}' is invalid: {exc}") from exc


# =============================================================================
# SERIALISATION HELPERS
# =============================================================================


def _kwargs_for(cls: type[Any], data: dict[str, Any]) -> dict[str, Any]:
    """Return the entries of ``data`` that are fields of dataclass ``cls``.

    Absent fields keep their dataclass defaults; unknown keys (such as fields
    removed by a schema upgrade) are ignored.
    """
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def _serialise_earner(earner: Earner) -> dict[str, Any]:
    return {f.name: getattr(earner, f.name) for f in fields(earner)}


def _deserialise_earner(data: dict[str, Any]) -> Earner:
    kwargs = _kwargs_for(Earner, data)
    # Legacy profiles stored is_employed instead of employment_type
    if "employment_type" not in data and "is_employed" in data:
        kwargs["employment_type"] = "employed" if data["is_employed"] else "not_employed"
    return Earner(**kwargs)


def _serialise_child(child: Child) -> dict[str, Any]:
    data: dict[str, Any] = {f.name: getattr(child, f.name) for f in fields(child)}
    data["education_schedule"] = [[age, cost] for age, cost in child.education_schedule]
    return data


def _deserialise_child(data: dict[str, Any]) -> Child:
    kwargs = _kwargs_for(Child, data)
    if "education_schedule" in kwargs:
        kwargs["education_schedule"] = tuple(tuple(pair) for pair in kwargs["education_schedule"])
    return Child(**kwargs)


def _serialise_mortgage(mortgage: MortgageAccount) -> dict[str, Any]:
    data: dict[str, Any] = {f.name: getattr(mortgage, f.name) for f in fields(mortgage)}
    data["offset_accounts"] = list(mortgage.offset_accounts)
    return data


def _deserialise_mortgage(data: dict[str, Any]) -> MortgageAccount:
    kwargs = _kwargs_for(MortgageAccount, data)
    if "offset_accounts" in kwargs:
        kwargs["offset_accounts"] = tuple(kwargs["offset_accounts"])
    return MortgageAccount(**kwargs)


def _serialise_account(account: InvestmentAccount) -> dict[str, Any]:
    data: dict[str, Any] = {f.name: getattr(account, f.name) for f in fields(account)}
    # JSON object keys are strings, so ownership is stored as str(earner_index) -> share
    data["ownership"] = {str(k): v for k, v in account.ownership.items()}
    return data


def _deserialise_account(data: dict[str, Any]) -> InvestmentAccount:
    kwargs = _kwargs_for(InvestmentAccount, data)
    if "ownership" in kwargs:
        kwargs["ownership"] = Ownership({int(k): v for k, v in kwargs["ownership"].items()})
    return InvestmentAccount(**kwargs)


def _serialise_household(household: Household) -> dict[str, Any]:
    return {
        "earners": [_serialise_earner(e) for e in household.earners],
        "children": [_serialise_child(c) for c in household.children],
        "mortgages": [_serialise_mortgage(m) for m in household.mortgages],
        "investment_accounts": [_serialise_account(a) for a in household.investment_accounts],
        "base_living_expenses": household.base_living_expenses,
        "retirement_target": household.retirement_target,
    }


def _serialise_inputs(inputs: SimulationInputs) -> dict[str, Any]:
    data: dict[str, Any] = {f.name: getattr(inputs, f.name) for f in fields(inputs)}
    data["sell_order"] = list(inputs.sell_order)
    data["household"] = _serialise_household(inputs.household)
    return data


def _repair_offset_links(
    mortgages: tuple[MortgageAccount, ...],
    accounts: tuple[InvestmentAccount, ...],
) -> tuple[MortgageAccount, ...]:
    """Rebuild mortgage→offset linkages for loaded profiles.

    Called at deserialisation time to repair profiles that were saved
    before ``offset_accounts`` was populated on each mortgage.  Without this
    step the simulation ignores offset balances during mortgage amortisation.

    Strategy:
    * Single-mortgage households: link all ``is_offset`` accounts to that
      mortgage unconditionally (the common case).
    * Multi-mortgage households: only rebuild links when no mortgage has any
      offset account and there is exactly one offset account — link it to the
      first mortgage.  In all other multi-mortgage cases the intended mapping
      cannot be inferred, so the data is left as-is for the wizard to repair.
    """
    offset_labels = tuple(a.label for a in accounts if a.is_offset)
    if not mortgages or not offset_labels:
        return mortgages

    if len(mortgages) == 1:
        if set(mortgages[0].offset_accounts) != set(offset_labels):
            return (replace(mortgages[0], offset_accounts=offset_labels),)
        return mortgages

    if len(offset_labels) == 1 and all(not m.offset_accounts for m in mortgages):
        return (replace(mortgages[0], offset_accounts=offset_labels), *mortgages[1:])
    return mortgages


def _deserialise_household(data: dict[str, Any]) -> Household:
    accounts = tuple(_deserialise_account(a) for a in data.get("investment_accounts", []))
    mortgages = _repair_offset_links(
        tuple(_deserialise_mortgage(m) for m in data.get("mortgages", [])),
        accounts,
    )
    kwargs = _kwargs_for(Household, data)
    kwargs.update(
        earners=tuple(_deserialise_earner(e) for e in data.get("earners", [])),
        children=tuple(_deserialise_child(c) for c in data.get("children", [])),
        mortgages=mortgages,
        investment_accounts=accounts,
    )
    return Household(**kwargs)


def _deserialise_inputs(data: dict[str, Any]) -> SimulationInputs:
    kwargs = _kwargs_for(SimulationInputs, data)
    kwargs["household"] = _deserialise_household(data.get("household", {}))
    if "sell_order" in kwargs:
        kwargs["sell_order"] = tuple(kwargs["sell_order"])
    return SimulationInputs(**kwargs)


# =============================================================================
# BACKWARD COMPATIBILITY (profile schema upgrades)
# =============================================================================


def _given(**values: Any) -> dict[str, Any]:
    """Keep only the keyword arguments whose value was present (not None)."""
    return {k: v for k, v in values.items() if v is not None}


def _v1_age(value: object) -> int | None:
    """v1 used 0 or a negative number for "no PT age set"; map those to None."""
    return value if isinstance(value, int) and value > 0 else None


def _upgrade_v1_profile(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 profile (``inputs.finances`` layout) to the v2 layout.

    Returns ``data`` unchanged when it is not a v1 profile.  Only values present
    in the v1 file are carried over; anything missing takes the model default,
    so no household-specific figures are invented.
    """
    if "_version" in data and data["_version"] >= 2:
        return data  # Already v2+

    inputs = data.get("inputs", {})
    finances = inputs.get("finances", {})

    # Require several v1-only fields: a single marker could be absent from a
    # genuine v1 file and the data would then be misparsed as v2.
    v1_markers = ("salary_h", "salary_w", "super_h", "super_w", "living_expenses")
    if not any(m in finances for m in v1_markers):
        return data  # Not a v1 profile (or no data at all)

    # v1 files carry no birth year, so super access age is set explicitly.
    earner1 = Earner(
        super_access_age=60,
        **_given(
            label="Earner 1",
            salary=finances.get("salary_h"),
            super_balance=finances.get("super_h"),
            retirement_age=inputs.get("retire_age"),
            pt_days_per_week=inputs.get("pt_days_per_week"),
            pt_rate_mode=inputs.get("pt_rate_mode"),
            pt_salary_pct=inputs.get("pt_salary_pct"),
            pt_start_age=_v1_age(inputs.get("pt_start_age")),
            pt_end_age=_v1_age(inputs.get("pt_end_age")),
        ),
    )
    earner2 = Earner(
        super_access_age=60,
        **_given(
            label="Earner 2",
            salary=finances.get("salary_w"),
            super_balance=finances.get("super_w"),
            retirement_age=inputs.get("retire_age"),
        ),
    )

    children: list[Child] = []
    child_age = finances.get("child_age")
    if child_age is not None and child_age >= 0:
        children.append(
            Child(
                label="Child 1",
                age=child_age,
                education_schedule=DEFAULT_EDU_SCHEDULE,
            )
        )

    offset_bal = finances.get("offset", 0.0)
    mortgages: list[MortgageAccount] = []
    mortgage_principal = finances.get("mortgage", 0.0)
    if mortgage_principal > 0:
        mortgages.append(
            MortgageAccount(
                label="Mortgage 1",
                principal=mortgage_principal,
                offset_accounts=("Offset 1",) if offset_bal > 0 else (),
            )
        )

    accounts: list[InvestmentAccount] = []
    if offset_bal > 0:
        accounts.append(
            InvestmentAccount(
                label="Offset 1",
                market_value=offset_bal,
                cost_basis=offset_bal,
                asset_class="cash",
                is_offset=True,
            )
        )
    for label, value_key, basis_key in (
        ("UK ETFs", "uk_etfs", "uk_basis"),
        ("AU ETFs", "au_etfs", "au_basis"),
    ):
        value = finances.get(value_key, 0.0)
        if value > 0:
            accounts.append(
                InvestmentAccount(
                    label=label,
                    market_value=value,
                    cost_basis=finances.get(basis_key, 0.0),
                    asset_class="equity",
                )
            )

    household = Household(
        earners=(earner1, earner2),
        children=tuple(children),
        mortgages=tuple(mortgages),
        investment_accounts=tuple(accounts),
        **_given(base_living_expenses=finances.get("living_expenses")),
    )

    return {
        "_version": 2,
        "profile_name": data.get("profile_name", "Upgraded (v1)"),
        "created_at": data.get("created_at", ""),
        "updated_at": data.get("updated_at", ""),
        "inputs": {
            "household": _serialise_household(household),
            **_given(
                n_iterations=inputs.get("n_iterations"),
                inflation=inputs.get("inflation"),
                simulation_start_age=inputs.get("simulation_start_age"),
                cgt_on_drawdowns=inputs.get("cgt_on_drawdowns"),
            ),
            "sell_strategy": "waterfall",
            "sell_order": [],
        },
        "last_results": data.get("last_results"),
    }


def _upgrade_v2_profile(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a v2 profile dict to v3.

    v2 used sentinel values for "not set": ``pt_start_age`` -1, ``pt_end_age``
    0 and an account ``interest_rate`` of 0.0 (which meant "asset-class
    default").  v3 uses None for all three.  Fields removed in v3
    (``tax_jurisdiction``, ``cgt_rate``, ``div293_growth_rate``) are ignored
    by the loaders, so they need no explicit removal here.
    """
    inputs_data = data.get("inputs")
    if not inputs_data:
        return {**data, "_version": PROFILE_VERSION}

    household = dict(inputs_data.get("household", {}))
    household["earners"] = [_upgrade_v2_earner(e) for e in household.get("earners", [])]
    household["investment_accounts"] = [
        _upgrade_v2_account(a) for a in household.get("investment_accounts", [])
    ]
    return {**data, "_version": PROFILE_VERSION, "inputs": {**inputs_data, "household": household}}


def _upgrade_v2_earner(earner: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(earner)
    if upgraded.get("pt_start_age") == -1:
        upgraded["pt_start_age"] = None
    if upgraded.get("pt_end_age") == 0:
        upgraded["pt_end_age"] = None
    return upgraded


def _upgrade_v2_account(account: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(account)
    if upgraded.get("interest_rate") == 0.0:
        upgraded["interest_rate"] = None
    return upgraded
