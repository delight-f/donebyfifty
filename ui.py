"""Rich-based terminal UI for the Monte Carlo retirement simulator.

Retro terminal aesthetic: green-on-black, ASCII boxes, limited palette.
Uses ``rich.prompt`` for validated input, ``rich.table`` for results,
and ``rich.panel`` for layout.
"""

from __future__ import annotations

import math
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from primitives import ASSET_CLASS_PARAMS

from config import (
    CONC_CAP,
    DEFAULT_CONC_CAP_GROWTH_RATE,
    DEFAULT_DIV293_GROWTH_RATE,
    DEFAULT_SIM_START_AGE,
    DIV293_RATE,
    DIV293_THRESHOLD,
    SG_MAX_BASE,
    THEME_COLOR,
    THEME_COLOR_ACCENT,
    THEME_COLOR_BRIGHT,
    THEME_COLOR_ERROR,
    THEME_COLOR_WARN,
    BK_SIGMA_MODERATE,
    BK_SIGMA_STRESS,
    BK_KAPPA,
    BK_THETA,
    BK_RHO,
)
from models import (
    Child,
    Earner,
    Household,
    InvestmentAccount,
    MortgageAccount,
    ResultsSession,
    SimulationInputs,
    SimulationResults,
)

# =============================================================================
# CONSOLE
# =============================================================================

console = Console(style=THEME_COLOR)

# =============================================================================
# ASCII BANNER
# =============================================================================

BANNER = r"""    ____                   ____        __________
   / __ \____  ____  ___  / __ )__  __/ ____/ __ \
  / / / / __ \/ __ \/ _ \/ __  / / / /___ \/ / / /
 / /_/ / /_/ / / / /  __/ /_/ / /_/ /___/ / /_/ /
/_____/\____/_/ /_/\___/_____/\__, /_____/\____/
                             /____/

            DoneBy50 Retirement Simulator"""


# =============================================================================
# ASCII CHART UTILITIES
# =============================================================================


def _ascii_probability_chart(
    p_success: float,
    time_periods: list[int] | None = None,
    trials: int | None = None,
    success_threshold: float = 0.95,
) -> str:
    """Return a simple text line showing success probability vs threshold.

    Shows the actual failure count alongside the percentage when trials
    data is available, so 99.998% (1 failure in 50,000) is not misleadingly
    displayed as 100.0%.

    Args:
        p_success: Overall success probability (0-1).
        time_periods: Ignored (kept for backward compatibility).
        trials: Total number of simulation trials (for failure count).
        success_threshold: Target success probability (default 0.95).

    Returns:
        String with probability status.
    """
    pct = p_success * 100.0
    threshold_pct = success_threshold * 100.0
    failure_note = ""
    if trials is not None and trials > 0:
        failures = round(trials * (1.0 - p_success))
        if failures > 0:
            failure_note = f"  ({failures} of {trials} paths failed)"
        else:
            failure_note = f"  (0 of {trials} paths failed)"
    if pct >= threshold_pct:
        return f"  Success probability: {pct:.2f}%  ({threshold_pct:.0f}% threshold: reached){failure_note}"
    return f"  Success probability: {pct:.2f}%  [red]({threshold_pct:.0f}% threshold: not reached)[/]{failure_note}"


def print_banner() -> None:
    """Print the ASCII art banner."""
    banner_text = Text(BANNER, style=THEME_COLOR_BRIGHT)
    console.print(banner_text)
    console.print()


# =============================================================================
# HELPERS
# =============================================================================


def _validate_range(value: Any, lo: float, hi: float, name: str) -> float | None:
    """Validate a numeric value is within [lo, hi]. Returns error string or None."""
    try:
        v = float(value)
        if v < lo or v > hi:
            console.print(
                f"  [bold {THEME_COLOR_ERROR}]  {name} must be between {lo:.0f} and {hi:.0f}[/]"
            )
            return None
        return v
    except (ValueError, TypeError):
        console.print(f"  [bold {THEME_COLOR_ERROR}]  Invalid number for {name}[/]")
        return None


def _parse_monetary(value: str) -> float | None:
    """Parse monetary input with shorthand notation (k for thousands, M for millions).

    Supports both shorthand and comma-separated formats:
    '300k' -> 300000, '1.5M' -> 1500000, '500,000' -> 500000.0, '500' -> 500.0

    Args:
        value: The monetary input string to parse.

    Returns:
        Float value if parsing succeeds, None otherwise.
    """
    if not value:
        return None

    value = value.strip().lower()
    # Strip thousand-separator commas (e.g. "500,000" -> "500000")
    # But be careful not to strip the only decimal point if present
    if ',' in value:
        if '.' in value:
            # e.g. "1,500.50" -> strip all commas: "1500.50"
            value = value.replace(',', '')
        else:
            # e.g. "500,000" -> strip commas: "500000"
            value = value.replace(',', '')
    try:
        if value.endswith('k'):
            return float(value[:-1]) * 1000.0
        elif value.endswith('m'):
            return float(value[:-1]) * 1_000_000.0
        elif value.endswith('b'):
            return float(value[:-1]) * 1_000_000_000.0
        else:
            return float(value)
    except ValueError:
        return None


def _prompt_int(
    prompt_text: str,
    default: int | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> int:
    """Prompt for an integer with optional default and range validation."""
    while True:
        raw = Prompt.ask(
            f"[{THEME_COLOR_BRIGHT}]{prompt_text}[/]",
            default=str(default) if default is not None else None,
        )
        if raw is None or raw.strip() == "":
            if default is not None:
                return default
            continue
        try:
            val = int(_parse_monetary(raw))
            if lo is not None and val < lo:
                console.print(f"  [bold {THEME_COLOR_ERROR}]Minimum is {lo:.0f}[/]")
                continue
            if hi is not None and val > hi:
                console.print(f"  [bold {THEME_COLOR_ERROR}]Maximum is {hi:.0f}[/]")
                continue
            return val
        except ValueError:
            console.print(f"  [bold {THEME_COLOR_ERROR}]Enter a whole number (try 300k for 300,000)[/]")


def _prompt_float(
    prompt_text: str,
    default: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    """Prompt for a float with optional default and range validation."""
    while True:
        raw = Prompt.ask(
            f"[{THEME_COLOR_BRIGHT}]{prompt_text}[/]",
            default=f"{default:g}" if default is not None else None,
        )
        if raw is None or raw.strip() == "":
            if default is not None:
                return default
            continue
        try:
            val = float(_parse_monetary(raw))
            if lo is not None and val < lo:
                console.print(f"  [bold {THEME_COLOR_ERROR}]Minimum is {lo}[/]")
                continue
            if hi is not None and val > hi:
                console.print(f"  [bold {THEME_COLOR_ERROR}]Maximum is {hi}[/]")
                continue
            return val
        except ValueError:
            console.print(f"  [bold {THEME_COLOR_ERROR}]Enter a number [dim](e.g. 300k for 300,000)[/][/]")


def _prompt_yn(prompt_text: str, default: bool = True) -> bool:
    """Prompt for yes/no."""
    default_str = "Y/n" if default else "y/N"
    raw = Prompt.ask(
        f"[{THEME_COLOR_BRIGHT}]{prompt_text}[/] [{default_str}]",
        default="y" if default else "n",
    )
    return raw.strip().lower() in ("y", "yes", "true")


# =============================================================================
# PRESET SELECTION
# =============================================================================


def choose_preset() -> str:
    """Let the user pick a household preset.

    Returns:
        One of ``"single"``, ``"couple"``, ``"family"``, ``"custom"``.

    """
    console.print(
        Panel(
            "[bold]Select a household preset:[/]\n\n"
            "  1. Single (1 earner, no children)\n"
            "  2. Couple, both working (2 earners, no children)\n"
            "  3. Couple, single income (1 working, 1 not)\n"
            "  4. Family (2 earners, children)\n"
            "  5. Custom (configure everything)\n",
            title="Household Setup",
            border_style=THEME_COLOR_ACCENT,
        )
    )
    choice = _prompt_int("Choice", default=1, lo=1, hi=5)
    mapping = {1: "single", 2: "couple", 3: "single_income", 4: "family", 5: "custom"}
    return mapping[choice]


# =============================================================================
# HOUSEHOLD CONFIGURATION
# =============================================================================


def _configure_earner(label: str, defaults: Earner | None = None, start_age: int = DEFAULT_SIM_START_AGE) -> Earner:
    """Prompt for one earner's details.

    Args:
        label: Earner display label.
        defaults: Pre-existing earner to edit.
        start_age: Simulation start age (for accurate warning projections).

    """
    d = defaults or Earner(label=label)
    console.print(f"\n[bold {THEME_COLOR_ACCENT}]  --- {label} ---[/]")

    # ── Employment type ─────────────────────────────────────────────
    emp_default = defaults.employment_type if defaults else "employed"
    emp_map = {"employed": 1, "self_employed": 2, "both": 3, "not_employed": 4}
    emp_default_num = emp_map.get(emp_default, 1)
    console.print(
        f"  [{THEME_COLOR_BRIGHT}]Employment type:[/]"
        f"\n    [{THEME_COLOR}]1.[/] Employed (salary + SG)"
        f"\n    [{THEME_COLOR}]2.[/] Self-employed (business income, no SG)"
        f"\n    [{THEME_COLOR}]3.[/] Both (salaried + self-employed)"
        f"\n    [{THEME_COLOR}]4.[/] Not employed"
    )
    emp_choice = _prompt_int("  Choice", default=emp_default_num, lo=1, hi=4)
    emp_type = {1: "employed", 2: "self_employed", 3: "both", 4: "not_employed"}[emp_choice]
    employed = (emp_type == "employed")

    if emp_type == "self_employed":
        console.print(
            f"  [dim]Super Guarantee does not apply to self-employed income —"
            f" use personal contributions below to model voluntary super contributions.[/]"
        )

    # ── Birth year (for preservation-age lookup) ─────────────────
    birth_year = None
    if _prompt_yn("  Auto-calculate super access age from birth year?",
                  default=defaults is not None and defaults.birth_year is not None or defaults is None):
        from models import preservation_age as _preservation_age
        birth_year = _prompt_int("  Birth year", d.birth_year or 1980, lo=1950, hi=2010)
        super_access = _preservation_age(birth_year)
        console.print(
            f"  [dim]Your preservation age: {super_access}[/]"
        )
        console.print(
            f"  [dim]Note: calculated from birth year only (not month)."
            f" Actual age may differ by a year if born in a transition year.[/]"
        )
    else:
        super_access = _prompt_int(
            "  Super access age (preservation age — check with your super fund)",
            d.super_access_age, lo=55, hi=70,
        )

    if emp_type in ("employed", "self_employed", "both"):
        salary = _prompt_float("  Gross annual salary [$]", d.salary, lo=0, hi=5_000_000)
        sal_growth = _prompt_float("  Real salary growth rate [%] (above inflation)", d.salary_growth_rate * 100, lo=-5, hi=15)
        retire_age = _prompt_int("  Retirement age", d.retirement_age, lo=30, hi=75)
        if emp_type in ("employed", "both"):
            sg = _prompt_float("  Super Guarantee rate [%] (as at 2025\u201326)", d.sg_rate * 100, lo=0, hi=20)
        else:
            sg = 0.0
            console.print(
                f"  [dim]SG rate set to 0% for self-employed — "
                f"employer SG obligations do not apply to sole traders.[/]"
            )
        super_bal = _prompt_float("  Current super balance [$]", d.super_balance, lo=0, hi=10_000_000)

        # ── Self-employed income (for "both" type) ──────────────────
        if emp_type == "both":
            se_income = _prompt_float(
                "  Self-employed / business income [$]",
                d.self_employed_income if d.self_employed_income > 0 else 50_000.0,
                lo=0, hi=5_000_000,
            )
            se_growth = _prompt_float(
                "  Real business income growth rate [%] (above inflation)",
                d.self_employed_growth_rate * 100,
                lo=-5, hi=15,
            )
            se_sg = _prompt_yn(
                "  Does Super Guarantee apply to business income?",
                default=d.self_employed_sg_applies,
            )
        else:
            se_income = 0.0
            se_growth = 0.0
            se_sg = False
    else:
        # Not employed
        salary = 0.0
        sal_growth = 0.0
        retire_age = 999
        sg = 0.0
        se_income = 0.0
        se_growth = 0.0
        se_sg = False
        super_bal = _prompt_float("  Current super balance [$]", d.super_balance, lo=0, hi=10_000_000)

    # ── Bridge-duration check ───────────────────────────────────────
    if emp_type in ("employed", "self_employed", "both") and retire_age < 999:
        bridge_years = super_access - retire_age
        if bridge_years < 3:
            console.print(
                f"  [bold {THEME_COLOR_WARN}]Bridge period: {bridge_years} years"
                f" (retire at {retire_age}, access super at {super_access})."
                f" This is very short — confirm this is intentional.[/]"
            )
        elif bridge_years > 15:
            console.print(
                f"  [bold {THEME_COLOR_WARN}]Bridge period: {bridge_years} years"
                f" (retire at {retire_age}, access super at {super_access})."
                f" This is a long bridge — your non-super assets need to cover"
                f" {bridge_years} years of expenses.[/]"
            )
        elif bridge_years < 5:
            console.print(
                f"  [dim]Bridge period: {bridge_years} years (retire at {retire_age},"
                f" access super at {super_access}).[/]"
            )
        else:
            console.print(
                f"  [dim]Bridge period: {bridge_years} years (retire at {retire_age},"
                f" access super at {super_access}).[/]"
            )

    # ── Super asset allocation ────────────────────────────────────
    gl = defaults
    super_growth = _prompt_float(
        "  Super growth asset allocation [%] (70 = typical balanced growth)",
        gl.super_growth_pct if gl else 70.0,
        lo=0, hi=100,
    )
    glide_enabled = _prompt_yn(
        "  Enable age-based glidepath (reduce growth near access age)?",
        default=gl is not None and gl.super_glide_end_year is not None,
    )
    if glide_enabled:
        glide_years = _prompt_int(
            "  Glide duration in years (e.g. 15 for a 15-year gradual transition)",
            gl.super_glide_end_year if gl else 15, lo=1, hi=50,
        )
        glide_target = _prompt_float(
            "  Target growth allocation at glide end [%] (30 = conservative)",
            gl.super_glide_target_pct if gl else 30.0, lo=0, hi=100,
        )
    else:
        glide_years = None
        glide_target = 30.0

    sacrifice: float | None = None
    console.print(
        f"  [{THEME_COLOR_BRIGHT}]Super contribution strategy:[/]"
    )
    console.print(
        f"    [{THEME_COLOR}]1. Auto-max to concessional cap (${CONC_CAP:,.0f} as at 2025\u201326)[/]"
    )
    console.print(
        f"    [{THEME_COLOR}]2. Set a custom annual amount[/]"
    )
    strat = _prompt_int("  Choice", default=1, lo=1, hi=2)
    if strat == 2:
        sacrifice = _prompt_float("  Annual pre-tax contribution [$]", lo=0, hi=100_000)

    non_conc = _prompt_float(
        "  Annual non-concessional (after-tax) contribution [$] (0=none)",
        defaults.non_concessional_contributions_p_a if defaults else 0.0,
        lo=0, hi=360_000,
    )

    # ── Part-time work post retirement ────────────────────────────
    pt_days = 0.0
    pt_start = 0
    pt_end = 65
    pt_rate_mode = "daily_rate"
    pt_daily_rate = 3_000.0
    pt_salary_pct = 0.0
    has_pt = _prompt_yn("  Part-time work after retirement?", default=d.pt_days_per_week > 0)
    if has_pt:
        pt_days = _prompt_float("  Days per week", d.pt_days_per_week if d.pt_days_per_week > 0 else 3.0, lo=0.5, hi=7)
        default_start = d.pt_start_age if d.pt_start_age > 0 else (retire_age if employed and retire_age < 999 else 60)
        pt_start = _prompt_int("  Start age", default_start, lo=30, hi=75)
        pt_end = _prompt_int("  End age", d.pt_end_age if d.pt_end_age > pt_start else pt_start + 5, lo=pt_start + 1, hi=80)
        # Ask how to value the part-time work
        rate_mode_raw = Prompt.ask(
            "  Calculate income as",
            choices=["daily_rate", "salary_pct"],
            default=d.pt_rate_mode if d.pt_rate_mode in ("daily_rate", "salary_pct") else "daily_rate",
        )
        pt_rate_mode = rate_mode_raw.strip()
        if pt_rate_mode == "daily_rate":
            pt_daily_rate = _prompt_float(
                "  Daily rate [$/day]",
                d.pt_daily_rate if d.pt_daily_rate > 0 else 500.0,
                lo=0,
                hi=20_000,
            )
        else:
            pt_salary_pct = _prompt_float(
                "  % of full-time salary",
                d.pt_salary_pct if d.pt_salary_pct > 0 else 40.0,
                lo=0,
                hi=100,
            )
        # Overlap check: PT window must overlap with simulated years
        effective_start = max(pt_start, retire_age if emp_type in ("employed", "self_employed", "both") and retire_age < 999 else start_age)
        effective_end = min(pt_end, super_access)
        if effective_start >= effective_end:
            console.print(
                f"  [bold {THEME_COLOR_WARN}]PT window ({pt_start}–{pt_end}) does not overlap"
                f" with bridge period"
                f" ({retire_age if emp_type in ('employed', 'self_employed', 'both') and retire_age < 999 else start_age}"
                f"→{super_access})."
                f" Income will not be applied.[/]"
            )

    # Display super contribution warnings using actual start age
    _display_earner_super_warnings(label, salary, sg / 100, retire_age, start_age)

    return Earner(
        label=label,
        salary=salary,
        super_balance=super_bal,
        salary_growth_rate=sal_growth / 100,
        retirement_age=retire_age,
        birth_year=birth_year,
        super_access_age=super_access,
        sg_rate=sg / 100,
        is_employed=employed,
        employment_type=emp_type,
        pt_days_per_week=pt_days,
        pt_start_age=pt_start,
        pt_end_age=pt_end,
        pt_daily_rate=pt_daily_rate,
        pt_salary_pct=pt_salary_pct,
        pt_rate_mode=pt_rate_mode,
        non_concessional_contributions_p_a=non_conc,
        super_growth_pct=super_growth,
        super_glide_end_year=glide_years,
        super_glide_target_pct=glide_target,
        personal_super_contributions_total_p_a=sacrifice,
        self_employed_income=se_income,
        self_employed_growth_rate=se_growth / 100,
        self_employed_sg_applies=se_sg,
    )


def _configure_child(label: str, defaults: Child | None = None) -> Child:
    """Prompt for one child's details."""
    d = defaults or Child(label=label)
    console.print(f"\n[bold {THEME_COLOR_ACCENT}]  --- {label} ---[/]")
    age = _prompt_int("  Age", d.age, lo=0, hi=25)

    # Three-way education cost choice
    if d.education_schedule:
        edu_default = 2  # existing custom schedule → default to "keep custom"
    else:
        edu_default = 1  # default to "use default schedule"

    console.print(f"  [{THEME_COLOR_BRIGHT}]Education costs:[/]")
    console.print(f"    [{THEME_COLOR}]1. Use default schedule (~$292k total, age 0-17)[/]")
    console.print(f"    [{THEME_COLOR}]2. Enter custom costs manually[/]")
    console.print(f"    [{THEME_COLOR}]3. No education costs[/]")
    edu_choice = _prompt_int("  Choice", default=edu_default, lo=1, hi=3)

    schedule: tuple[tuple[int, float], ...] = ()
    if edu_choice == 2:
        console.print("  Enter (age, cost) pairs. Press Enter on a blank Age to finish.")
        pairs: list[tuple[int, float]] = []
        while True:
            age_raw = Prompt.ask(
                f"    [{THEME_COLOR_BRIGHT}]Age[/]",
                default="",
            )
            if age_raw.strip() == "":
                break
            try:
                a = int(age_raw)
            except ValueError:
                console.print(f"    [{THEME_COLOR_ERROR}]Invalid age[/]")
                continue
            cost = _prompt_float(f"    Cost at age {a} [$]", lo=0)
            pairs.append((a, cost))
        schedule = tuple(pairs)
        if schedule:
            console.print(f"  [dim]Entered {len(schedule)} age/cost pairs.[/]")
    elif edu_choice == 3:
        # Non-empty but unusable schedule signals "no education costs"
        # (empty tuple = use default education schedule)
        schedule = ((-1, 0.0),)

    return Child(label=label, age=age, education_schedule=schedule)


def _configure_mortgage(label: str, defaults: MortgageAccount | None = None) -> MortgageAccount:
    """Prompt for one mortgage's details."""
    d = defaults or MortgageAccount(label=label)
    console.print(f"\n[bold {THEME_COLOR_ACCENT}]  --- {label} ---[/]")
    principal = _prompt_float("  Remaining balance (outstanding) [$]", d.principal, lo=0, hi=10_000_000)
    rate = _prompt_float("  Interest rate [%]", d.interest_rate * 100, lo=0, hi=15)
    pmt = _prompt_float("  Monthly payment [$] (0=interest-only)", d.monthly_payment, lo=0)
    if pmt > 0 and pmt * 12 < principal * (rate / 100):
        console.print(
            f"  [bold {THEME_COLOR_WARN}]\u26a0\ufe0f  Monthly payment is less than the interest"
            f" charge \u2014 the mortgage balance will grow over time.[/]"
        )
    mode_default = {
        "fixed": 1,
        "stall_prevention": 2,
        "interest_cancelling": 3,
    }.get(d.offset_reserve_mode, 1)
    console.print(
        "  Offset reserve mode:"
        f"\n    [{THEME_COLOR}]1[/] No reserve (offset drains fully)"
        "\n    "
        f"[{THEME_COLOR}]2[/] Stall prevention (keeps enough offset that"
        " payment covers interest \u2014"
        " loan doesn't grow, but doesn't reduce)"
        "\n    "
        f"[{THEME_COLOR}]3[/] Interest cancelling (keeps enough offset to"
        " eliminate all interest \u2014"
        " 100% of payment goes to principal)"
    )
    mode_choice = _prompt_int(
        "  Choice", default=mode_default, lo=1, hi=3,
    )
    if mode_choice == 1:
        mode = "fixed"
        floor = 0.0
    elif mode_choice == 2:
        mode = "stall_prevention"
        floor = 0.0  # unused
        console.print(
            f"  [dim]Floor recalculated each period: max(0, balance \u2014"
            f" payment / monthly rate). Loan will not grow but will not"
            f" reduce from interest savings alone.[/]"
        )
    else:
        mode = "interest_cancelling"
        floor = 0.0  # unused
        console.print(
            f"  [dim]Floor set to full mortgage principal each period."
            f" All offset balance is preserved for this mortgage, making"
            f" 100% of every payment go to principal. No offset available"
            f" for other expenses until mortgage clears.[/]"
        )

    # ── Stochastic mortgage rate prompts ─────────────────────────────
    stoch = _prompt_yn(
        "  Apply stochastic rate variation (Black-Karasinski process)?",
        default=d.interest_rate_stochastic,
    )
    vol = None
    kappa = None
    theta = None
    corr = None
    if stoch:
        console.print(
            f"  [dim]Enter values in % (e.g. 18 = 18%). Blank = use recommended default.[/]"
        )
        vol = _prompt_float(
            f"  How much can the rate vary each year? (default {BK_SIGMA_MODERATE*100:.0f}% = moderate, {BK_SIGMA_STRESS*100:.0f}% = stress)",
            default=BK_SIGMA_MODERATE * 100,
            lo=0, hi=50,
        )
        kappa = _prompt_float(
            f"  How quickly does the rate revert to its long-run average? (default {BK_KAPPA*100:.0f}%, higher = faster reversion)",
            default=BK_KAPPA * 100,
            lo=1, hi=100,
        )
        theta_v = _prompt_float(
            f"  Long-run average mortgage rate in % (default {BK_THETA*100:.1f}% = historical average)",
            default=BK_THETA * 100,
            lo=1.0, hi=15.0,
        )
        corr = _prompt_float(
            f"  Correlation with sharemarket returns in % (default {BK_RHO*100:.0f}% = moderate positive, 0% = uncorrelated)",
            default=BK_RHO * 100,
            lo=-50, hi=50,
        )

        # Convert percentage inputs to decimals for storage
        # Also handles None (blank = use config default at runtime)
        if vol is not None and vol != BK_SIGMA_MODERATE * 100:
            vol = vol / 100
        else:
            vol = None  # use config default
        if kappa is not None and kappa != BK_KAPPA * 100:
            kappa = kappa / 100
        else:
            kappa = None
        if theta_v is not None and theta_v != BK_THETA * 100:
            theta_v = theta_v / 100
        else:
            theta_v = None
        if corr is not None and corr != BK_RHO * 100:
            corr = corr / 100
        else:
            corr = None

    term = _prompt_int(
        "  Loan must be repaid by age (0=no term check)",
        d.loan_term_end_age or 0, lo=0, hi=100,
    )
    return MortgageAccount(
        label=label,
        principal=principal,
        interest_rate=rate / 100,
        monthly_payment=pmt,
        offset_reserve_mode=mode,
        offset_reserve_floor=floor,
        loan_term_end_age=term if term > 0 else None,
        interest_rate_stochastic=stoch,
        interest_rate_vol=vol,
        interest_rate_kappa=kappa,
        interest_rate_theta=theta_v,
        interest_rate_corr=corr,
    )


def _configure_account(label: str, defaults: InvestmentAccount | None = None,
                        *, num_earners: int = 1,
                        earner_labels: list[str] | None = None) -> InvestmentAccount:
    """Prompt for one investment account's details."""
    d = defaults or InvestmentAccount(label=label)
    console.print(f"\n[bold {THEME_COLOR_ACCENT}]  --- {label} ---[/]")
    value = _prompt_float("  Current market value [$]", d.market_value, lo=0, hi=10_000_000)
    is_offset = _prompt_yn("  Is this a mortgage offset account?", d.is_offset)
    
    # Only ask for cost basis if NOT an offset account (no CGT on offsets)
    basis = 0.0
    if not is_offset:
        basis = _prompt_float("  Original purchase value [$]", d.cost_basis, lo=0, hi=10_000_000)

    asset_class = "cash" if is_offset else "equity"
    # Only ask for asset class and return rate if not an offset account
    interest_rate = d.interest_rate
    if not is_offset:
        # ── Asset class selection ───────────────────────────────────
        console.print(f"  [{THEME_COLOR_BRIGHT}]Asset class:[/]")
        classes = list(ASSET_CLASS_PARAMS.keys())
        for i, ac in enumerate(classes, 1):
            params = ASSET_CLASS_PARAMS[ac]
            console.print(
                f"    [{THEME_COLOR}]{i}. {ac:<15} ~{params['mean'] * 100:.0f}% return, {params['std'] * 100:.0f}% risk[/]"
            )
        ac_choice = _prompt_int("  Choice", default=1, lo=1, hi=len(classes))
        asset_class = classes[ac_choice - 1]
        # Custom return rate (0 = use the selected asset class default)
        interest_rate = _prompt_float(
            "  Custom annual return [%] (0 = use asset class default above)",
            d.interest_rate * 100 if d.interest_rate > 0 else 0,
            lo=0,
            hi=20,
        ) / 100
    
    # ── Ownership split prompt (multi-earner, non-offset only) ──────
    ownership: dict[int, float] = {0: 1.0}
    if num_earners > 1 and not is_offset:
        el = earner_labels or [f"Earner {i + 1}" for i in range(num_earners)]
        console.print(f"  [{THEME_COLOR_BRIGHT}]Ownership split:[/]")
        shares: dict[int, float] = {}
        remaining = 100
        for ei in range(num_earners):
            default_pct = int(d.ownership.get(ei, 0.0) * 100) if defaults else 0
            if ei == num_earners - 1:
                pct = remaining
            else:
                pct = _prompt_int(
                    f"  {el[ei]} ownership [%]",
                    default=default_pct if default_pct > 0 else (100 // num_earners),
                    lo=0, hi=remaining,
                )
            shares[ei] = pct / 100
            remaining -= pct
        ownership = shares

    return InvestmentAccount(
        label=label,
        market_value=value,
        cost_basis=basis,
        asset_class=asset_class,
        is_offset=is_offset,
        cgt_rate=0.0 if is_offset else 0.30,
        interest_rate=interest_rate,
        ownership=ownership,
    )


# =============================================================================
# FULL HOUSEHOLD CONFIGURATION
# =============================================================================


def _configure_earners_section(
    preset: str,
    existing: Household | None,
    start_age: int = DEFAULT_SIM_START_AGE,
) -> tuple[Earner, ...]:
    """Configure all earners for a household, reusing existing values as defaults."""
    num_earners = 1
    if preset in ("couple", "family"):
        num_earners = 2
    elif preset == "single_income":
        num_earners = 2
    elif preset == "custom":
        num_earners = _prompt_int(
            "How many earners? [1-10]", existing.num_earners if existing else 1, lo=1, hi=10
        )

    earners: list[Earner] = []
    for i in range(num_earners):
        label = f"Earner {i + 1}"
        existing_e = existing.earners[i] if existing and i < len(existing.earners) else None
        if preset == "single_income" and i == 1:
            existing_e = existing_e or Earner(label=label, salary=0, is_employed=False, employment_type="not_employed")
            e = _configure_earner(label, existing_e, start_age)
            e = Earner(
                label=e.label,
                salary=0.0,
                super_balance=e.super_balance,
                is_employed=False,
                employment_type="not_employed",
                sg_rate=e.sg_rate,
                pt_days_per_week=e.pt_days_per_week,
                pt_start_age=e.pt_start_age,
                pt_end_age=e.pt_end_age,
                personal_super_contributions_total_p_a=e.personal_super_contributions_total_p_a,
            )
            earners.append(e)
        else:
            earners.append(_configure_earner(label, existing_e, start_age))

    for e in earners:
        if e.employment_type == "not_employed" and e.salary > 0:
            console.print(
                f"[yellow]Warning: Earner '{e.label}' has "
                f"employment_type='not_employed' but salary > 0 ({e.salary:,.0f}). "
                f"PT income will be double-counted with salary income. "
                f"Set salary to 0 or change employment type.[/]"
            )
    return tuple(earners)


def _configure_mortgages_section(
    existing: Household | None,
) -> tuple[MortgageAccount, ...]:
    """Configure all mortgages, reusing existing values as defaults."""
    mortgages: list[MortgageAccount] = []
    has_mortgage = _prompt_yn("Any mortgages?", bool(existing and existing.mortgages) if existing else True)
    if has_mortgage or (existing and existing.mortgages):
        num_m = _prompt_int(
            "How many mortgages?",
            len(existing.mortgages) if existing and existing.mortgages else 1,
            lo=0, hi=5,
        )
        for i in range(num_m):
            label = f"Mortgage {i + 1}"
            existing_m = existing.mortgages[i] if existing and i < len(existing.mortgages) else None
            mortgages.append(_configure_mortgage(label, existing_m))
    return tuple(mortgages)


def _configure_children_section(
    preset: str,
    existing: Household | None,
) -> tuple[Child, ...]:
    """Configure all children, reusing existing values as defaults."""
    children: list[Child] = []
    has_kids = preset == "family" or (
        preset == "custom" and _prompt_yn("Any children?", bool(existing and existing.children))
    )
    if has_kids or (existing and existing.children):
        num_kids = _prompt_int(
            "How many children?",
            len(existing.children) if existing and existing.children else 1,
            lo=0, hi=20,
        )
        for i in range(num_kids):
            label = f"Child {i + 1}"
            existing_c = existing.children[i] if existing and i < len(existing.children) else None
            children.append(_configure_child(label, existing_c))
    return tuple(children)


def _configure_accounts_section(
    existing: Household | None,
) -> tuple[InvestmentAccount, ...]:
    """Configure all investment accounts, reusing existing values as defaults."""
    # Determine earner count and labels for ownership split prompt
    n_earners = len(existing.earners) if existing and existing.earners else 1
    earner_labels = [e.label for e in existing.earners] if existing and existing.earners else ["Earner 1"]

    accounts: list[InvestmentAccount] = []
    has_accounts = _prompt_yn(
        "Any investment accounts?", bool(existing and existing.investment_accounts)
    )
    if has_accounts or (existing and existing.investment_accounts):
        num_a = _prompt_int(
            "How many accounts?",
            len(existing.investment_accounts) if existing and existing.investment_accounts else 1,
            lo=0, hi=10,
        )
        for i in range(num_a):
            label = f"Account {i + 1}"
            existing_a = (
                existing.investment_accounts[i]
                if existing and i < len(existing.investment_accounts)
                else None
            )
            accounts.append(_configure_account(label, existing_a,
                                                num_earners=n_earners,
                                                earner_labels=earner_labels))
    return tuple(accounts)


def _link_offsets_to_mortgages(
    accounts: list[InvestmentAccount],
    mortgages: list[MortgageAccount],
) -> list[MortgageAccount]:
    """Link offset accounts to their target mortgages."""
    offset_labels = [a.label for a in accounts if a.is_offset]
    result = list(mortgages)
    if offset_labels and result:
        if len(result) == 1:
            m = result[0]
            result[0] = MortgageAccount(
                label=m.label, principal=m.principal,
                interest_rate=m.interest_rate, monthly_payment=m.monthly_payment,
                offset_accounts=tuple(offset_labels),
                offset_reserve_mode=m.offset_reserve_mode,
                offset_reserve_floor=m.offset_reserve_floor,
                loan_term_end_age=m.loan_term_end_age,
                interest_rate_stochastic=m.interest_rate_stochastic,
                interest_rate_vol=m.interest_rate_vol,
                interest_rate_kappa=m.interest_rate_kappa,
                interest_rate_theta=m.interest_rate_theta,
                interest_rate_corr=m.interest_rate_corr,
            )
        else:
            for ol in offset_labels:
                console.print(f"  [dim]Which mortgage does '{ol}' offset?[/]")
                for mi, m in enumerate(result, 1):
                    console.print(f"    [{mi}]. {m.label}[/]")
                choice = _prompt_int("  Choice", lo=1, hi=len(result))
                idx = choice - 1
                m = result[idx]
                new_offsets = m.offset_accounts + (ol,)
                result[idx] = MortgageAccount(
                    label=m.label, principal=m.principal,
                    interest_rate=m.interest_rate, monthly_payment=m.monthly_payment,
                    offset_accounts=new_offsets,
                    offset_reserve_mode=m.offset_reserve_mode,
                    offset_reserve_floor=m.offset_reserve_floor,
                    loan_term_end_age=m.loan_term_end_age,
                    interest_rate_stochastic=m.interest_rate_stochastic,
                    interest_rate_vol=m.interest_rate_vol,
                    interest_rate_kappa=m.interest_rate_kappa,
                    interest_rate_theta=m.interest_rate_theta,
                    interest_rate_corr=m.interest_rate_corr,
                )
    return result


def _configure_expenses_section(
    existing: Household | None,
) -> tuple[float, float]:
    """Configure household living expenses."""
    console.print(f"[bold {THEME_COLOR_ACCENT}]  --- Household Expenses ---[/]")
    living = _prompt_float(
        "  Base annual living expenses [$]",
        existing.base_living_expenses if existing else 60_000.0,
        lo=10_000, hi=1_000_000,
    )
    target = _prompt_float(
        "  Spending in retirement [$]",
        existing.retirement_target if existing else 80_000.0,
        lo=10_000, hi=1_000_000,
    )
    return living, target


def configure_household(
    preset: str,
    existing: Household | None = None,
    start_age: int = DEFAULT_SIM_START_AGE,
) -> Household:
    """Interactively configure a household from scratch.

    Args:
        preset: One of ``"single"``, ``"couple"``, ``"single_income"``,
                ``"family"``, ``"custom"``.
        existing: Pre-populate from an existing household (when editing a profile).
        start_age: Simulation start age (for accurate warning projections).

    Returns:
        A configured ``Household``.

    """
    earners = _configure_earners_section(preset, existing, start_age)
    mortgages = _configure_mortgages_section(existing)
    children = _configure_children_section(preset, existing)
    accounts = _configure_accounts_section(existing)
    mortgages = tuple(_link_offsets_to_mortgages(list(accounts), list(mortgages)))
    living, target = _configure_expenses_section(existing)
    return Household(
        earners=earners,
        children=children,
        mortgages=mortgages,
        investment_accounts=accounts,
        base_living_expenses=living,
        retirement_target=target,
    )


def edit_household(
    existing: Household,
    start_age: int = DEFAULT_SIM_START_AGE,
) -> Household:
    """Edit sections of an existing household via a section menu.

    Presents a menu of sections to edit rather than re-running the full
    configuration wizard. Each section reuses existing values as defaults.

    Args:
        existing: The household to edit.
        start_age: Simulation start age.

    Returns:
        The modified ``Household``.

    """
    earners = list(existing.earners)
    mortgages = list(existing.mortgages)
    children = list(existing.children)
    accounts = list(existing.investment_accounts)
    living = existing.base_living_expenses
    target = existing.retirement_target

    while True:
        console.print()
        choices_text = (
            f"[bold]Edit which section?[/]\n\n"
            f"  [{THEME_COLOR}]1[/] Earners ({len(earners)} configured)\n"
            f"  [{THEME_COLOR}]2[/] Mortgages ({len(mortgages)} configured)\n"
            f"  [{THEME_COLOR}]3[/] Children ({len(children)} configured)\n"
            f"  [{THEME_COLOR}]4[/] Investment accounts ({len(accounts)} configured)\n"
            f"  [{THEME_COLOR}]5[/] Living expenses (${living:,.0f} / ${target:,.0f})\n"
            f"  [{THEME_COLOR}]6[/] Done editing — run simulation"
        )
        console.print(Panel(
            choices_text,
            title="Edit Profile",
            border_style=THEME_COLOR_ACCENT,
        ))
        choice = _prompt_int("Choice", default=6, lo=1, hi=6)
        if choice == 1:
            earners = list(_configure_earners_section("custom", _make_dummy_household(earners, mortgages, children, accounts, living, target), start_age))
        elif choice == 2:
            mortgages = list(_configure_mortgages_section(_make_dummy_household(earners, mortgages, children, accounts, living, target)))
        elif choice == 3:
            children = list(_configure_children_section("custom", _make_dummy_household(earners, mortgages, children, accounts, living, target)))
        elif choice == 4:
            accounts = list(_configure_accounts_section(_make_dummy_household(earners, mortgages, children, accounts, living, target)))
            mortgages = _link_offsets_to_mortgages(accounts, mortgages)
        elif choice == 5:
            living, target = _configure_expenses_section(_make_dummy_household(earners, mortgages, children, accounts, living, target))
        else:
            break

    return Household(
        earners=tuple(earners),
        children=tuple(children),
        mortgages=tuple(mortgages),
        investment_accounts=tuple(accounts),
        base_living_expenses=living,
        retirement_target=target,
    )


def _make_dummy_household(
    earners: list,
    mortgages: list,
    children: list,
    accounts: list,
    living: float,
    target: float,
) -> Household:
    """Build a throwaway Household from current in-progress values for passing as defaults."""
    from models import Household
    return Household(
        earners=tuple(earners),
        mortgages=tuple(mortgages),
        children=tuple(children),
        investment_accounts=tuple(accounts),
        base_living_expenses=living,
        retirement_target=target,
    )
# =============================================================================


def configure_simulation_params(
    existing: SimulationInputs | None = None,
    household: Household | None = None,
    start_age: int | None = None,
) -> SimulationInputs:
    """Configure simulation parameters (iterations, inflation, etc.).

    Start age is collected *before* household configuration (passed in)
    so that earner super warnings can use the accurate start age.

    Args:
        existing: Pre-existing inputs to edit (when loading a profile).
        household: The household definition (for surplus allocation prompts).
        start_age: Pre-collected simulation start age. If provided, the
                   start age prompt is skipped. Falls back to existing
                   value or config default.

    Returns:
        A new ``SimulationInputs`` with user-provided values.

    """
    # Determine effective start age: parameter > existing > default
    if start_age is None:
        start_age = existing.simulation_start_age if existing else DEFAULT_SIM_START_AGE
    console.print(f"\n[bold {THEME_COLOR_ACCENT}]--- Simulation Parameters ---[/]")
    iters = _prompt_int(
        "  Number of simulation runs",
        existing.n_iterations if existing else 5_000,
        lo=100,
        hi=100_000,
    )
    infl = _prompt_float(
        "  Inflation rate [%]",
        (existing.inflation * 100 if existing else 2.5),
        lo=0,
        hi=10,
    )
    cgt = _prompt_yn(
        "  Apply CGT when selling investments?",
        existing.cgt_on_drawdowns if existing else True,
    )

    # ── Super fee rate ───────────────────────────────────────────
    super_fee = _prompt_float(
        "  Superannuation fee rate [%] (0.85 = median AusSuper fund)",
        existing.super_fee_rate * 100 if existing else 0.85,
        lo=0.0, hi=5.0,
    ) / 100

    # ── Surplus allocation ──────────────────────────────────────────
    # Only ask when there are both offset and non-offset accounts
    default_pct = existing.surplus_investment_pct if existing else 0.0
    surplus_pct = default_pct
    if household:
        has_offset = any(a.is_offset for a in household.investment_accounts)
        has_investment = any(not a.is_offset for a in household.investment_accounts)
        if has_offset and has_investment:
            console.print()
            surplus_pct = _prompt_float(
                "  % of surplus to investments each year, even if offset isn't full yet (rest to offset/cash)",
                default_pct,
                lo=0,
                hi=100,
            )
        elif not has_offset and has_investment:
            # No offset accounts — all surplus naturally goes to investments
            surplus_pct = 100.0

    # ── Success threshold ───────────────────────────────
    success_th = _prompt_float(
        "  Target success probability [%] (e.g. 90 = 90%)",
        (existing.success_threshold * 100 if existing else 95.0),
        lo=50,
        hi=100,
    )

    return SimulationInputs(
        n_iterations=iters,
        inflation=infl / 100,
        simulation_start_age=start_age,
        simulation_end_age=existing.simulation_end_age if existing else 72,
        cgt_on_drawdowns=cgt,
        surplus_investment_pct=surplus_pct,
        super_fee_rate=super_fee,
        success_threshold=success_th / 100,
    )


# =============================================================================
# RESULTS DISPLAY
# =============================================================================


def display_results(results: SimulationResults, household: Household | None = None, start_age: int = 37, success_threshold: float = 0.95) -> None:
    """Display simulation results in a Rich table.

    Args:
        results: The completed simulation results.
        household: The household definition (for contribution warnings).
        start_age: Simulation start age (default 37).

    """
    console.print()
    console.print(
        Panel(
            f"[bold {THEME_COLOR_BRIGHT}]Simulation Complete — {results.trials:,} trials[/]",
            border_style=THEME_COLOR_ACCENT,
        )
    )

    # Show success probability warning with actionable tiered messaging
    if results.p_success == 0.0:
        console.print(
            f"  ⚠️  [bold {THEME_COLOR_ERROR}]Every simulation ran out of money before super became"
            f" accessible. Your bridge assets cannot cover expenses for the full bridge"
            f" period. You need more non-super savings, lower spending, or a shorter"
            f" bridge (earlier super access age / later retirement).[/]"
        )
        if results.bridge_median < 0:
            console.print(
                f"  [dim]Median shortfall: {_fmt_dollar(abs(results.bridge_median))}[/]"
            )
    elif results.p_success < 0.30:
        console.print(
            f"  ⚠️  [bold {THEME_COLOR_ERROR}]Severe risk: success in only {results.p_success * 100:.2f}% of trials."
            f" Your bridge is far too tight. Consider more non-super savings, lower"
            f" spending, or a shorter bridge period.[/]"
        )
    elif results.p_success < 0.50:
        console.print(
            f"  ⚠️  [bold {THEME_COLOR_ERROR}]High risk: success in {results.p_success * 100:.2f}% of trials."
            f" More than half of possible futures run out early. Increase bridge assets"
            f" or reduce expenses.[/]"
        )
    elif results.p_success < 0.70:
        console.print(
            f"  ⚠️  [bold {THEME_COLOR_WARN}]Moderate risk: success in {results.p_success * 100:.2f}% of trials."
            f" Consider boosting non-super savings or trimming expenses to improve odds.[/]"
        )
    elif results.p_success < success_threshold:
        console.print(
            f"  ⚠️  [bold {THEME_COLOR_WARN}]Success probability {results.p_success * 100:.2f}% is below"
            f" the {success_threshold*100:.0f}% threshold. A small buffer would improve confidence.[/]"
        )

    # Display honest success probability chart (absolute 0-100% scale)
    console.print()
    chart = _ascii_probability_chart(results.p_success, trials=results.trials, success_threshold=success_threshold)
    console.print(chart)

    # Bridge assets table
    table = Table(
        title="Bridge Assets at Horizon (today's dollars)",
        border_style=THEME_COLOR_ACCENT,
        box=None,
    )
    table.add_column("Metric", style=THEME_COLOR_BRIGHT)
    table.add_column("Value", justify="right", style=THEME_COLOR)

    rows = [
        ("Mean", results.bridge_mean),
        ("Median", results.bridge_median),
        ("P5", results.bridge_p5),
        ("P10", results.bridge_p10),
        ("P25", results.bridge_p25),
        ("P75", results.bridge_p75),
        ("P90", results.bridge_p90),
        ("P95", results.bridge_p95),
        ("Running minimum", results.bridge_floor),
    ]
    for label, val in rows:
        table.add_row(label, _fmt_dollar(val))

    console.print(table)

    # ── Bootstrap standard errors (when available) ──────────────────
    if results.bridge_mean_se is not None:
        console.print()
        se_table = Table(
            title="Uncertainty (bootstrap SE, 200 resamples)",
            border_style="dim",
            box=None,
        )
        se_table.add_column("Metric", style=THEME_COLOR_BRIGHT)
        se_table.add_column("Estimate", justify="right", style=THEME_COLOR)
        se_table.add_column("SE", justify="right", style=THEME_COLOR)
        se_table.add_column("SE %", justify="right", style=THEME_COLOR)

        se_rows = [
            ("Mean",  results.bridge_mean,  results.bridge_mean_se),
            ("Median", results.bridge_median, results.bridge_median_se),
            ("P5",     results.bridge_p5,     results.bridge_p5_se),
            ("P95",    results.bridge_p95,    results.bridge_p95_se),
        ]
        for label, est, se in se_rows:
            if se is not None:
                if est != 0:
                    se_pct = se / est * 100
                else:
                    se_pct = float('inf') if se > 0 else 0.0

                # Color-code relative SE: green <2%, yellow 2-4%, orange 5-9%, red >=10%
                if se_pct < 2.0:
                    se_color = THEME_COLOR          # green
                elif se_pct < 5.0:
                    se_color = THEME_COLOR_WARN     # yellow
                elif se_pct < 10.0:
                    se_color = "dark_orange"
                else:
                    se_color = THEME_COLOR_ERROR    # red

                se_pct_str = f"{se_pct:.1f}%"
                se_table.add_row(
                    label, _fmt_dollar(est), _fmt_dollar(se),
                    f"[{se_color}]{se_pct_str}[/]"
                )
        console.print(se_table)

    # Worst simulated outcome: plain-English diagnostic
    if results.trials >= 1000:
        console.print()
        recovery_note = (
            f"[{THEME_COLOR_WARN}]Lowest at age {results.floor_age} ({_fmt_dollar(results.bridge_floor)})[/]"
        )

        panel_text = (
            f"[bold]Worst simulated outcome[/]\n\n"
            f"{recovery_note}"
        )
        console.print(Panel(
            panel_text,
            border_style="dim",
        ))

    # Super summary
    console.print(f"\n  Median super balance at end of scenario: [bold]{_fmt_dollar(results.super_median)}[/]")

    # Per-earner super
    if results.per_earner_super_p50:
        super_table = Table(
            title="Median Super per Earner",
            border_style=THEME_COLOR_ACCENT,
            box=None,
        )
        super_table.add_column("Earner", style=THEME_COLOR_BRIGHT)
        super_table.add_column("Median Super", justify="right", style=THEME_COLOR)
        for label, val in results.per_earner_super_p50.items():
            super_table.add_row(label, _fmt_dollar(val))
        console.print(super_table)

    # Remaining mortgage
    if results.remaining_mortgage_p50:
        mort_table = Table(
            title="Remaining Mortgage (Median)",
            border_style=THEME_COLOR_ACCENT,
            box=None,
        )
        mort_table.add_column("Mortgage", style=THEME_COLOR_BRIGHT)
        mort_table.add_column("Remaining", justify="right", style=THEME_COLOR)
        for label, val in results.remaining_mortgage_p50.items():
            mort_table.add_row(label, _fmt_dollar(val))
        console.print(mort_table)

    # Term-clearance check (headline when checkable mortgages exist)
    if results.per_mortgage_term_cleared_pct:
        tc_rate = results.mortgage_term_clearance_rate
        tc_color = (
            THEME_COLOR_BRIGHT
            if tc_rate >= 0.9
            else THEME_COLOR_ERROR if tc_rate < 0.5
            else THEME_COLOR_WARN
        )
        console.print(
            f"  Mortgage term clearance: [bold {tc_color}]{tc_rate * 100:.1f}%[/]"
        )
        if tc_rate < 1.0:
            if results.per_mortgage_term_cleared_pct:
                for label, pct in results.per_mortgage_term_cleared_pct.items():
                    if pct < 1.0:
                        console.print(
                            f"    [dim]{label}: cleared in {pct * 100:.1f}% of paths[/]"
                        )
            console.print(
                f"  [dim]A mortgage not cleared by term end may force"
                f" refinancing, asset sales, or reduced living standards.[/]"
            )

    # Show mortgages whose term end is beyond the simulation horizon
    if results.per_mortgage_not_checkable:
        for label, count in results.per_mortgage_not_checkable.items():
            console.print(
                f"  [dim]\u26a0\ufe0f  {label}: term end is beyond the simulation"
                f" horizon (age {results.horizon_age}). Clearance cannot"
                f" be evaluated in this projection. Re-run with a longer"
                f" horizon or check manually.[/]"
            )
    elif results.remaining_mortgage_p50 and any(
        v > 0 for v in results.remaining_mortgage_p50.values()
    ) and not results.per_mortgage_term_cleared_pct:
        console.print(
            f"  [dim]Mortgage term clearance: no term-end check was configured.[/]"
        )

    # ── Super and Tax Warnings ─────────────────────────────────────────
    if household:
        _display_contribution_warnings(household, start_age)

    console.print()

    # ── Scope disclosure ──────────────────────────────────────────────
    disclosures = [
        f"All bridge asset figures are in today's dollars"
        f" (deflated to start-of-simulation purchasing power)."
        f" Super and mortgage figures are in nominal (then-year) dollars.",
        f"This simulation models only the bridge period "
        f"(up to age {results.horizon_age}). It does not model post-bridge "
        f"retirement drawdown, Age Pension, or minimum pension rules.",
    ]
    if results.remaining_mortgage_p50 and any(
        v > 0 for v in results.remaining_mortgage_p50.values()
    ):
        # Check if any mortgage has stochastic rates enabled
        has_stochastic = (
            household is not None
            and any(m.interest_rate_stochastic for m in household.mortgages)
        )
        if has_stochastic:
            theta = BK_THETA
            kappa = BK_KAPPA
            sigma_mod = BK_SIGMA_MODERATE
            sigma_str = BK_SIGMA_STRESS
            rho = BK_RHO
            lo = math.exp(math.log(theta) - 1.96 * sigma_mod / math.sqrt(2 * kappa)) * 100
            hi = math.exp(math.log(theta) + 1.96 * sigma_mod / math.sqrt(2 * kappa)) * 100
            disclosures.append(
                f"Mortgage interest rates follow a Black-Karasinski "
                f"mean-reverting lognormal process (discrete AR(1), "
                f"kappa={kappa:.2f}/yr). The long-run mean is theta={theta*100:.1f}% "
                f"with log-rate volatility sigma={sigma_mod:.2f}/\u221ayr "
                f"(stress sigma={sigma_str:.2f}/\u221ayr for sensitivity), "
                f"producing a steady-state ~95% interval of approximately "
                f"[{lo:.1f}%, {hi:.1f}%]. Innovations are correlated "
                f"with equity returns (rho={rho:.2f}). This replaces the "
                f"static-rate assumption used in previous runs."
            )
        else:
            disclosures.append(
                f"Mortgage interest rates are held constant for the full "
                f"simulation. Real rate volatility \u2014 including the scenario "
                f"where rates rise while offset is being drawn down \u2014 "
                f"is not captured. This may understate the stress on repayment "
                f"affordability in rising-rate environments."
            )
    for i, d in enumerate(disclosures):
        if i > 0:
            console.print()
        console.print(f"  [dim]Note: {d}[/]")

    console.print()


# =============================================================================
# SUPER/TAX WARNINGS
# =============================================================================


def _display_earner_super_warnings(
    earner_label: str,
    salary: float,
    sg_rate: float,
    retirement_age: int,
    start_age: int = DEFAULT_SIM_START_AGE,
) -> None:
    """Display real-time warnings about superannuation caps and Division 293.

    Uses config-level default indexation rates for projections. The
    projection is approximate — actual simulation uses the user's
    configured growth rates which may differ.

    Args:
        earner_label: The earner's label.
        salary: Annual salary.
        sg_rate: Super Guarantee rate.
        retirement_age: Earner's retirement age.
        start_age: Simulation start age (default: ``DEFAULT_SIM_START_AGE``).
    """
    warnings = _super_warning_messages(
        earner_label=earner_label,
        salary=salary,
        sg_rate=sg_rate,
        years=retirement_age - start_age,
    )
    if warnings:
        console.print()
        console.print(Panel(
            "\n".join(warnings),
            title="Superannuation & Tax Alerts",
            border_style=THEME_COLOR_WARN,
        ))


def _super_warning_messages(
    earner_label: str,
    salary: float,
    sg_rate: float,
    years: int,
) -> list[str]:
    """Build warning messages about super caps and Div 293 for an earner.

    Uses config-default indexation rates for cap and threshold
    projections. Called from both the earner config UI and the
    post-simulation results display.

    Args:
        earner_label: Earner display name.
        salary: Current annual salary.
        sg_rate: Super Guarantee rate (decimal).
        years: Years between start age and retirement (for indexation).

    Returns:
        List of warning strings (empty if none triggered).
    """
    warnings: list[str] = []

    years = max(0, years)

    # Projected values using config-default growth rates
    indexed_conc_cap = CONC_CAP * (1 + DEFAULT_CONC_CAP_GROWTH_RATE) ** years
    indexed_salary = salary * (1 + DEFAULT_CONC_CAP_GROWTH_RATE) ** years
    sg_contribution = min(indexed_salary, SG_MAX_BASE) * sg_rate

    if sg_contribution > indexed_conc_cap:
        warnings.append(
            f"  ⚠️  [bold {THEME_COLOR_WARN}]SG contribution ({sg_contribution:,.0f}) "
            f"exceeds concessional cap ({indexed_conc_cap:,.0f}). "
            f"Excess may be subject to excess contributions tax.[/]"
        )

    indexed_div293_threshold = DIV293_THRESHOLD * (1 + DEFAULT_DIV293_GROWTH_RATE) ** years
    if indexed_salary > indexed_div293_threshold and sg_contribution > 0:
        warnings.append(
            f"  ⚠️  [bold {THEME_COLOR_WARN}]Division 293 applies for {earner_label} "
            f"(income ${salary:,.0f} > projected threshold ${indexed_div293_threshold:,.0f}). "
            f"An additional {DIV293_RATE * 100:.0f}% tax on concessional contributions is deducted"
            f" from take-home pay in the simulation (worst-case assumption).[/]"
        )

    return warnings


def _display_contribution_warnings(household: Household, start_age: int) -> None:
    """Display post-run warnings about superannuation caps and Div 293.

    Args:
        household: The household definition.
        start_age: Simulation start age.
    """
    warnings: list[str] = []

    for earner in household.earners:
        warnings.extend(_super_warning_messages(
            earner_label=earner.label,
            salary=earner.salary,
            sg_rate=earner.sg_rate,
            years=earner.retirement_age - start_age,
        ))

    if warnings:
        console.print()
        console.print(Panel(
            "\n".join(warnings),
            title="Superannuation & Tax Alerts",
            border_style=THEME_COLOR_WARN,
        ))


# =============================================================================
# REVIEW BEFORE RUN
# =============================================================================


def review_before_run(
    household: Household,
    sim_inputs: SimulationInputs,
) -> bool:
    """Show a summary of the configured household and simulation params.

    Returns True if the user wants to proceed, False to abort.

    """
    bridge_end = min(e.super_access_age for e in household.earners)
    employed_retirements = [e.retirement_age for e in household.earners if e.retirement_age < 999]
    bridge_start = min(employed_retirements) if employed_retirements else sim_inputs.simulation_start_age
    bridge_years = bridge_end - bridge_start

    console.print()

    # ── Household summary ───────────────────────────────────────────
    hh_lines: list[str] = []
    hh_lines.append(f"[bold]Earners ({household.num_earners}):[/]")
    for e in household.earners:
        emp_labels = {"employed": "Employed", "self_employed": "Self-employed", "both": "Salary + self-employed", "not_employed": "Not employed"}
        emp = emp_labels.get(e.employment_type, "Employed")
        pt_info = ""
        if e.pt_days_per_week > 0:
            pt_info = (
                f", PT {e.pt_days_per_week:.1f}d/wk "
                f"({e.pt_start_age}–{e.pt_end_age})"
            )
        hh_lines.append(
            f"  {e.label}: {emp}, salary ${e.salary:,.0f}, "
            f"retire at {e.retirement_age}, super access at {e.super_access_age}"
            f"{pt_info}"
        )
    if household.children:
        hh_lines.append(f"[bold]Children ({household.num_children}):[/]")
        for c in household.children:
            edu = "custom schedule" if c.education_schedule else "default schedule"
            hh_lines.append(f"  {c.label}: age {c.age}, {edu}")
    if household.mortgages:
        hh_lines.append(f"[bold]Mortgages ({household.num_mortgages}):[/]")
        for m in household.mortgages:
            stoch_tag = " [dim](stochastic rates)[/dim]" if m.interest_rate_stochastic else ""
            hh_lines.append(
                f"  {m.label}: ${m.principal:,.0f} at {m.interest_rate * 100:.2f}%"
                f"{stoch_tag}, "
                f"${m.monthly_payment:,.0f}/month"
            )
    if household.investment_accounts:
        hh_lines.append(f"[bold]Investment accounts ({len(household.investment_accounts)}):[/]")
        for a in household.investment_accounts:
            tag = " (offset)" if a.is_offset else ""
            # Show ownership split for multi-earner, non-offset accounts
            owner_parts = []
            if len(household.earners) > 1 and not a.is_offset and len(a.ownership) > 1:
                for ei, share in sorted(a.ownership.items()):
                    if share > 0 and ei < len(household.earners):
                        owner_parts.append(f"{household.earners[ei].label} {share * 100:.0f}%")
            owner_str = f" — {' / '.join(owner_parts)}" if owner_parts else ""
            hh_lines.append(f"  {a.label}: ${a.market_value:,.0f}{tag}{owner_str}")
    hh_lines.append(
        f"[bold]Expenses:[/] ${household.base_living_expenses:,.0f}/yr base annual, "
        f"${household.retirement_target:,.0f}/yr retirement target "
        "(blended proportionally as each earner retires)"
    )
    if household.num_earners > 1:
        hh_lines.append(
            "  [dim]Expenses blend linearly between the two figures based on "
            "the fraction of earners retired each year.[/]"
        )

    console.print(Panel(
        "\n".join(hh_lines),
        title="Household Summary",
        border_style=THEME_COLOR_ACCENT,
    ))

    # ── Simulation summary ──────────────────────────────────────────
    console.print(Panel(
        f"Trials: {sim_inputs.n_iterations:,}\n"
        f"Inflation: {sim_inputs.inflation * 100:.1f}%\n"
        f"Super fee rate: {sim_inputs.super_fee_rate * 100:.2f}%/yr\n"
        f"Bridge period: age {bridge_start} \u2192 {bridge_end} "
        f"({bridge_years} years)\n"
        f"CGT on withdrawals: {'Yes' if sim_inputs.cgt_on_drawdowns else 'No'}"
        f"\n"
        f"Stochastic mortgage rates: "
        f"{'Yes' if any(m.interest_rate_stochastic for m in household.mortgages) else 'No'}",
        title="Simulation Parameters",
        border_style=THEME_COLOR_ACCENT,
    ))

    return _prompt_yn("Run simulation with these settings?", default=True)


# =============================================================================
# PROFILE LISTING
# =============================================================================


def display_profile_list(profiles: list[dict[str, Any]]) -> None:
    """Display a list of saved profiles."""
    if not profiles:
        console.print("[yellow]No saved profiles.[/]")
        return

    table = Table(
        title="Saved Profiles",
        border_style=THEME_COLOR_ACCENT,
        box=None,
    )
    table.add_column("#", style=THEME_COLOR_BRIGHT)
    table.add_column("Name", style=THEME_COLOR)
    table.add_column("Earners", justify="right")
    table.add_column("Children", justify="right")
    table.add_column("Last Success", justify="right")

    for i, p in enumerate(profiles, 1):
        success_str = (
            f"{p['p_success'] * 100:.1f}%"
            if p.get("p_success") is not None
            else ("[red]corrupt[/]" if p.get("corrupt") else "—")
        )
        table.add_row(
            str(i),
            p["name"],
            str(p["num_earners"]),
            str(p["num_children"]),
            success_str,
        )

    console.print(table)


# =============================================================================
# RESULTS NAVIGATION MENU  (Work Item 8)
# =============================================================================


def show_results_menu(session: ResultsSession) -> None:
    """Interactive post-run menu for drill-down views.

    Loops until the user chooses 0 (back to main menu) or q (quit).
    Expensive computations (scenario comparison, sequencing risk,
    retirement search) are opt-in and cached in the session.
    """
    while True:
        console.print()
        console.print(
            Panel(
                f"Simulation complete \u2014 {session.results.trials:,} trials\n"
                f"Success: {session.results.p_success * 100:.2f}%"
                + (f"\nSeed: {session.inputs.seed}" if session.inputs.seed is not None else ""),
                border_style=THEME_COLOR_ACCENT,
            )
        )
        console.print(f"  [{THEME_COLOR_BRIGHT}]View additional detail:[/]")
        console.print()
        options = [
            ("1",  "Age-by-age bridge trajectory"),
            ("2",  "Near-miss / failure depth analysis"),
            ("3",  "Sequencing risk comparison [dim](opt-in, ~2 min)[/]"),
            ("4",  "Drawdown source composition"),
            ("5",  "Tax (CGT) breakdown"),
            ("6",  "Scenario comparison [dim](opt-in, ~1 min)[/]"),
            ("7",  "Earliest feasible retirement age [dim](opt-in, ~2 min, single-earner only)[/]"),
            ("8",  "Mortgage amortisation schedule"),
        ]
        for num, desc in options:
            console.print(f"    [{THEME_COLOR}]{num}[/]  {desc}")
        console.print(f"    [{THEME_COLOR}]0[/]   Back to main menu")
        console.print(f"    [dim]q[/]   Quit")
        console.print()

        raw = Prompt.ask(
            f"[{THEME_COLOR_BRIGHT}]Choice[/]",
            choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "q"],
            default="0",
        )
        choice = raw.strip().lower()

        if choice == "q":
            import sys
            sys.exit(0)
        elif choice == "0":
            break
        elif choice == "1":
            _view_trajectory(session)
        elif choice == "2":
            _view_near_miss(session)
        elif choice == "3":
            _view_sequencing_risk(session)
        elif choice == "4":
            _view_drawdown_composition(session)
        elif choice == "5":
            _view_cgt_breakdown(session)
        elif choice == "6":
            _view_scenario_comparison(session)
        elif choice == "7":
            _view_retirement_search(session)
        elif choice == "8":
            _view_mortgage_amortisation(session)


# =============================================================================
# DETAIL VIEW PLACEHOLDERS  (filled in by later work items)
# =============================================================================


def _view_trajectory(session: ResultsSession) -> None:
    """Display age-by-age bridge trajectory (Work Item 1).

    Shows P5/P10/P25/Median/P75/P90/P95 bridge balance at each age
    during the bridge period, in today's dollars.
    """
    r = session.results
    if not r.bridge_by_age_ages:
        console.print(
            f"  [bold {THEME_COLOR_WARN}]No trajectory data available."
            f" This result was computed without per-year capture.[/]"
        )
        return

    ages = r.bridge_by_age_ages
    table = Table(
        title="Bridge trajectory (today's dollars)",
        border_style=THEME_COLOR_ACCENT,
        box=None,
    )
    table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
    table.add_column("P5", justify="right", style=THEME_COLOR)
    table.add_column("P10", justify="right", style=THEME_COLOR)
    table.add_column("P25", justify="right", style=THEME_COLOR)
    table.add_column("Median", justify="right", style=THEME_COLOR_BRIGHT)
    table.add_column("P75", justify="right", style=THEME_COLOR)
    table.add_column("P90", justify="right", style=THEME_COLOR)
    table.add_column("P95", justify="right", style=THEME_COLOR)

    for i, age in enumerate(ages):
        table.add_row(
            str(age),
            _fmt_dollar(r.bridge_by_age_p5[i]) if i < len(r.bridge_by_age_p5) else "\u2014",
            _fmt_dollar(r.bridge_by_age_p10[i]) if i < len(r.bridge_by_age_p10) else "\u2014",
            _fmt_dollar(r.bridge_by_age_p25[i]) if i < len(r.bridge_by_age_p25) else "\u2014",
            _fmt_dollar(r.bridge_by_age_p50[i]) if i < len(r.bridge_by_age_p50) else "\u2014",
            _fmt_dollar(r.bridge_by_age_p75[i]) if i < len(r.bridge_by_age_p75) else "\u2014",
            _fmt_dollar(r.bridge_by_age_p90[i]) if i < len(r.bridge_by_age_p90) else "\u2014",
            _fmt_dollar(r.bridge_by_age_p95[i]) if i < len(r.bridge_by_age_p95) else "\u2014",
        )

    console.print(table)

    # Export hint
    console.print()
    console.print(
        f"  [dim]Copy this table for Excel: select and copy the terminal output.[/]"
    )


def _view_near_miss(session: ResultsSession) -> None:
    """Display near-miss / failure depth analysis (Work Item 2).

    Shows the proportion of trials that approached failure, and the
    age distribution of failures if any occurred.
    """
    r = session.results

    # Near-miss rate
    console.print()
    near_miss_pct = (1.0 - r.near_miss_rate) * 100
    console.print(
        f"  [bold]Near-miss analysis[/] (threshold: \u2264 {_fmt_dollar(r.near_miss_threshold)})"
    )
    console.print(
        f"  Trials that crossed below threshold: [bold]{near_miss_pct:.2f}%[/]"
    )

    # Failure age distribution
    if r.failure_age_distribution:
        total_failures = sum(r.failure_age_distribution.values())
        console.print()
        console.print(f"  [bold]Failure age distribution:[/] ({total_failures} failing trials)")

        table = Table(
            border_style=THEME_COLOR_ACCENT,
            box=None,
        )
        table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
        table.add_column("Failures", justify="right", style=THEME_COLOR)
        table.add_column("% of total", justify="right", style=THEME_COLOR)

        for age in sorted(r.failure_age_distribution.keys()):
            count = r.failure_age_distribution[age]
            pct = count / total_failures * 100
            table.add_row(str(age), str(count), f"{pct:.1f}%")

        console.print(table)
    else:
        console.print(f"  [dim]No failures occurred in this simulation run.[/]")


def _view_sequencing_risk(session: ResultsSession) -> None:
    """Display sequencing risk comparison (Work Item 3, opt-in, expensive)."""
    from simulation import run_sequencing_analysis

    if session.sequencing is not None:
        seq = session.sequencing
    else:
        if not _prompt_yn(
            "Run sequencing risk analysis? This will run 2 additional "
            "full simulations and may take 2-3 minutes.",
            default=False,
        ):
            return

        console.print(f"  [dim]Running sequencing risk analysis (2 passes)...[/]")
        seq = run_sequencing_analysis(
            household=session.household,
            inputs=session.inputs,
            original_results=session.results,
            seed=session.inputs.seed,
        )
        session.sequencing = seq

    console.print()
    console.print(Panel(
        f"[bold {THEME_COLOR_BRIGHT}]Sequencing risk analysis[/]",
        border_style=THEME_COLOR_ACCENT,
    ))
    console.print(
        f"  Original (random order):  {seq.original_p_success * 100:.2f}% success,"
        f"  P5 = {_fmt_dollar(seq.original_p5)}"
    )
    console.print(
        f"  Worst returns first:      {seq.worst_first_p_success * 100:.2f}% success,"
        f"  P5 = {_fmt_dollar(seq.worst_first_p5)}"
    )
    console.print(
        f"  Best returns first:       {seq.best_first_p_success * 100:.2f}% success,"
        f"  P5 = {_fmt_dollar(seq.best_first_p5)}"
    )
    console.print(
        f"  [bold]Sequencing risk spread:[/] {seq.spread * 100:.2f} percentage points"
    )
    console.print()
    console.print(
        f"  [dim]What this means:[/]"
    )
    console.print(
        f"  [dim]• Worst returns first: the historical return sequence is rearranged so"
        f" the worst-performing years happen earliest in your bridge period."
        f" This tests the scenario where a market downturn hits just as you start drawing down.[/]"
    )
    console.print(
        f"  [dim]• Best returns first: the opposite — best-case timing of returns,"
        f" with strong early years and weaker ones later.[/]"
    )
    console.print(
        f"  [dim]• The spread = worst-first success minus best-first success."
        f" A negative spread means poor returns early in retirement reduce your"
        f" success probability more than having them later. This is the classic"
        f" 'sequencing risk' problem: it matters most during the bridge phase when"
        f" balances are being drawn down, not accumulated.[/]"
    )


def _view_drawdown_composition(session: ResultsSession) -> None:
    """Display drawdown source composition (Work Item 4).

    Shows median offset draws and non-offset (investment sale) draws
    per year, in nominal dollars (drawn amounts are inherently nominal).
    """
    r = session.results
    ages = r.bridge_by_age_ages

    if not r.offset_drawn_p50 or not ages:
        console.print(
            f"  [bold {THEME_COLOR_WARN}]No drawdown composition data available.[/]"
        )
        return

    # Compute total (across all years) for summary
    total_offset = sum(r.offset_drawn_p50)
    total_non_offset = sum(r.non_offset_drawn_p50)
    total_cgt = sum(r.cgt_paid_p50)

    # Per-year table
    table = Table(
        title="Drawdown source composition (median per year, nominal)",
        border_style=THEME_COLOR_ACCENT,
        box=None,
    )
    table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
    table.add_column("Offset drawn", justify="right", style=THEME_COLOR)
    table.add_column("Non-offset drawn", justify="right", style=THEME_COLOR)
    table.add_column("CGT paid", justify="right", style=THEME_COLOR)

    for i, age in enumerate(ages):
        if i >= len(r.offset_drawn_p50):
            break
        off = r.offset_drawn_p50[i]
        non = r.non_offset_drawn_p50[i]
        cgt = r.cgt_paid_p50[i]
        if off == 0 and non == 0 and cgt == 0 and i < len(ages) - 1:
            # Skip years with no drawdown activity (after early mortgage paydown etc.)
            # Still show the last year for completeness
            continue
        table.add_row(
            str(age),
            _fmt_dollar(off),
            _fmt_dollar(non),
            _fmt_dollar(cgt),
        )

    console.print(table)

    # Summary totals
    console.print()
    console.print(
        f"  [bold]Total (median):[/] Offset: {_fmt_dollar(total_offset)}"
        f"  |  Non-offset: {_fmt_dollar(total_non_offset)}"
        f"  |  CGT paid: {_fmt_dollar(total_cgt)}"
    )
    console.print()
    console.print(
        f"  [dim]CGT is charged only on real (CPI-indexed) gains, not on the full"
        f" withdrawal amount. Each sale returns your own capital untaxed — only the"
        f" portion above your indexed cost basis attracts the 30% minimum rate.[/]"
    )


def _view_cgt_breakdown(session: ResultsSession) -> None:
    """Display CGT breakdown (Work Item 5).

    Shows CGT paid with the 30% minimum rate floor vs the counterfactual
    without the floor, by year. Quantifies the dollar impact of the
    Phase 2 CGT reform.
    """
    r = session.results
    ages = r.bridge_by_age_ages

    if not r.cgt_paid_p50 or not ages:
        console.print(
            f"  [bold {THEME_COLOR_WARN}]No CGT breakdown data available.[/]"
        )
        return

    total_with_floor = sum(r.cgt_paid_p50)
    total_without_floor = sum(r.cgt_without_floor_p50)
    floor_cost = total_with_floor - total_without_floor

    # Per-year table
    table = Table(
        title="CGT breakdown (median per year, nominal)",
        border_style=THEME_COLOR_ACCENT,
        box=None,
    )
    table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
    table.add_column("With 30% floor", justify="right", style=THEME_COLOR)
    table.add_column("Without floor", justify="right", style=THEME_COLOR)
    table.add_column("Floor cost", justify="right", style=THEME_COLOR)

    for i, age in enumerate(ages):
        if i >= len(r.cgt_paid_p50):
            break
        floor_cgt = r.cgt_paid_p50[i]
        no_floor_cgt = r.cgt_without_floor_p50[i]
        diff = floor_cgt - no_floor_cgt
        # Only show years with nonzero CGT
        if floor_cgt == 0 and no_floor_cgt == 0:
            continue
        table.add_row(
            str(age),
            _fmt_dollar(floor_cgt),
            _fmt_dollar(no_floor_cgt),
            _fmt_dollar(diff),
        )

    console.print(table)

    # Summary
    console.print()
    if floor_cost > 0:
        pct_increase = (floor_cost / total_without_floor * 100) if total_without_floor > 0 else 0.0
        console.print(
            f"  [bold]Total CGT with 30% floor:[/] {_fmt_dollar(total_with_floor)}"
        )
        console.print(
            f"  [bold]Total CGT without floor:[/] {_fmt_dollar(total_without_floor)}"
        )
        console.print(
            f"  [bold]Extra CGT from 30% minimum rate:[/] {_fmt_dollar(floor_cost)}"
            f"  ({pct_increase:.1f}% increase)"
        )
        console.print(
            f"  [dim](vs what you'd pay at your standard marginal rate,"
            f" without the 30% minimum floor)[/]"
        )
    else:
        console.print(
            f"  [dim]The 30% CGT floor had no effect in this scenario"
            f" (marginal rates were already above 30% or no CGT was triggered).[/]"
        )


def _view_scenario_comparison(session: ResultsSession) -> None:
    """Display scenario comparison table (Work Item 6, opt-in, expensive)."""
    from simulation import run_scenario_comparison

    if session.scenarios is not None:
        scens = session.scenarios
    else:
        if not _prompt_yn(
            "Run scenario comparison? This will run 2 additional "
            "simulations and may take 1-2 minutes.",
            default=False,
        ):
            return

        console.print(f"  [dim]Running scenario comparison...[/]")
        scens = run_scenario_comparison(
            household=session.household,
            inputs=session.inputs,
            seed=session.inputs.seed,
        )
        session.scenarios = scens

    base_p = session.results.p_success
    base_p5 = session.results.bridge_p5

    console.print()
    table = Table(
        title=f"Scenario comparison ({session.inputs.n_iterations:,} trials base)",
        border_style=THEME_COLOR_ACCENT,
        box=None,
    )
    table.add_column("Scenario", style=THEME_COLOR_BRIGHT)
    table.add_column("Success", justify="right", style=THEME_COLOR)
    table.add_column("P5 bridge", justify="right", style=THEME_COLOR)
    table.add_column("vs base", justify="right", style=THEME_COLOR)

    # Base row
    table.add_row(
        "Current plan",
        f"{base_p * 100:.2f}%",
        _fmt_dollar(base_p5),
        "\u2014",
    )

    for label, r in scens.items():
        diff = (r.p_success - base_p) * 100
        diff_str = f"{diff:+.2f}pp" if abs(diff) > 0.01 else "\u2014"
        table.add_row(
            label,
            f"{r.p_success * 100:.2f}%",
            _fmt_dollar(r.bridge_p5),
            diff_str,
        )

    console.print(table)
    console.print()
    console.print(f"  [dim]Note: scenarios run at {10_000:,} trials each for speed.[/]")


def _view_retirement_search(session: ResultsSession) -> None:
    """Display earliest feasible retirement age (Work Item 9, opt-in, expensive)."""
    from simulation import run_retirement_search

    if session.retirement_search is not None:
        rs = session.retirement_search
    else:
        # Single-earner check
        if len(session.household.earners) > 1:
            console.print(
                f"  [bold {THEME_COLOR_ERROR}]Earliest-retirement search is only available"
                f" for single-earner households. Multi-earner search is"
                f" deferred pending a firm decision on scope.[/]"
            )
            return

        if not _prompt_yn(
            "Search for earliest feasible retirement age? This will run "
            "several full simulations and may take 2-3 minutes.",
            default=False,
        ):
            return

        console.print(f"  [dim]Searching for earliest feasible retirement age...[/]")
        rs = run_retirement_search(
            household=session.household,
            inputs=session.inputs,
            seed=session.inputs.seed,
        )
        session.retirement_search = rs

    console.print()
    console.print(Panel(
        f"[bold {THEME_COLOR_BRIGHT}]Earliest feasible retirement age[/]",
        border_style=THEME_COLOR_ACCENT,
    ))
    console.print(
        f"  Your entered retirement age: {rs.entered_age}"
        f" ({rs.entered_p_success * 100:.2f}% success)"
    )

    if rs.earliest_age < rs.entered_age:
        console.print(
            f"  Earliest age with \u2265{rs.threshold * 100:.0f}% success:"
            f" [bold]{rs.earliest_age}[/] ({rs.earliest_p_success * 100:.2f}% success)"
        )
        console.print(
            f"  Age {rs.earliest_age - 1} or earlier falls below"
            f" the {rs.threshold * 100:.0f}% threshold."
        )
    else:
        console.print(
            f"  No earlier age meets the {rs.threshold * 100:.0f}% threshold"
            f" (searched down to age {rs.floor_age})."
        )
    console.print()
    console.print(
        f"  [dim]Note: this search is specific to your current plan configuration."
        f" Changing any input will change this result.[/]"
    )


def _view_mortgage_amortisation(session: ResultsSession) -> None:
    """Display mortgage amortisation schedule (Work Item 10).

    Shows per-year median mortgage principal with P5/P95 bands,
    plus offset balance and mortgage-neutral year detection.
    All figures in nominal dollars (debt doesn't inflate).
    """
    r = session.results
    household = session.household

    if not r.mortgage_by_age and not household.mortgages:
        console.print(
            f"  [dim]No mortgages configured in this scenario.[/]"
        )
        return

    if not r.mortgage_by_age or not r.mortgage_by_age_ages:
        console.print(
            f"  [bold {THEME_COLOR_WARN}]No mortgage trajectory data available."
            f" This result was computed without per-year capture.[/]"
        )
        return

    ages = r.mortgage_by_age_ages

    # Build a set of offset account labels for quick lookup
    offset_labels = [a.label for a in household.investment_accounts if a.is_offset]
    # Map each mortgage to the offset labels linked to it
    mortgage_to_offset: dict[str, list[str]] = {}
    has_offset_links = False
    for m in household.mortgages:
        linked = [ol for ol in offset_labels if ol in m.offset_accounts]
        if linked:
            mortgage_to_offset[m.label] = linked
            has_offset_links = True

    for mi, mortgage in enumerate(household.mortgages):
        mdata = r.mortgage_by_age.get(mortgage.label)
        if mdata is None:
            continue

        p50 = mdata["p50"]
        p5 = mdata.get("p5", [])
        p95 = mdata.get("p95", [])

        # Get rate data (only populated when stochastic rates are enabled)
        ratedata = r.mortgage_rate_by_age.get(mortgage.label)
        has_rates = ratedata is not None and ratedata.get("p50")

        # Get offset data for linked accounts
        linked_offsets = mortgage_to_offset.get(mortgage.label, [])
        offset_p50: list[float] | None = None
        if linked_offsets and r.offset_by_age:
            # Sum linked offset accounts by year
            first_ol = linked_offsets[0]
            odata = r.offset_by_age.get(first_ol)
            if odata:
                offset_p50 = odata["p50"]

        console.print()
        table = Table(
            title=f"Mortgage amortisation \u2014 {mortgage.label} (nominal dollars)",
            border_style=THEME_COLOR_ACCENT,
            box=None,
        )
        table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
        table.add_column("P5", justify="right", style=THEME_COLOR)
        table.add_column("Median", justify="right", style=THEME_COLOR_BRIGHT)
        table.add_column("P95", justify="right", style=THEME_COLOR)
        if has_rates:
            table.add_column("Rate P5", justify="right", style=THEME_COLOR)
            table.add_column("Rate P50", justify="right", style=THEME_COLOR_BRIGHT)
            table.add_column("Rate P95", justify="right", style=THEME_COLOR)
        if offset_p50 is not None:
            table.add_column("Offset (med)", justify="right", style=THEME_COLOR)
        table.add_column("", style="dim")  # neutral marker column

        for i, age in enumerate(ages):
            if i >= len(p50):
                break
            bal = p50[i]
            bal_p5 = p5[i] if i < len(p5) else bal
            bal_p95 = p95[i] if i < len(p95) else bal
            offset_bal = offset_p50[i] if offset_p50 and i < len(offset_p50) else None

            # Mortgage-neutral detection: offset balance >= mortgage principal
            neutral_marker = ""
            if offset_bal is not None and offset_bal >= bal > 0:
                neutral_marker = "\u2190 neutral"
            elif bal <= 0:
                neutral_marker = "\u2713 cleared"

            row = [str(age), _fmt_dollar(bal_p5), _fmt_dollar(bal), _fmt_dollar(bal_p95)]
            if has_rates:
                rp5 = _fmt_pct(ratedata["p5"][i]) if i < len(ratedata["p5"]) else "\u2014"
                rp50 = _fmt_pct(ratedata["p50"][i]) if i < len(ratedata["p50"]) else "\u2014"
                rp95 = _fmt_pct(ratedata["p95"][i]) if i < len(ratedata["p95"]) else "\u2014"
                row.extend([rp5, rp50, rp95])
            if offset_bal is not None:
                row.append(_fmt_dollar(offset_bal))
            row.append(neutral_marker)
            table.add_row(*row)

        console.print(table)

    if has_offset_links:
        console.print()
        console.print(
            f"  [dim]\u2190 neutral: offset balance \u2265 mortgage principal at year end.[/]"
        )
        console.print(
            f"  [dim]Offset reserve mode interactions are reflected in the account balances.[/]"
        )
    console.print(
        f"  [dim]Mortgage figures are in nominal (then-year) dollars.[/]"
    )


# =============================================================================
# INTERNAL FORMATTERS
# =============================================================================


def _fmt_dollar(x: float) -> str:
    if x >= 1e6:
        return f"${x / 1e6:,.2f}M"
    return f"${x:,.0f}"


def _fmt_pct(x: float) -> str:
    """Format a decimal rate as a percentage string."""
    return f"{x * 100:.2f}%"
