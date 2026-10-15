"""Interactive household and simulation parameter configuration, plus review."""

from __future__ import annotations

import dataclasses
from typing import (
    Any,
    Literal,
    cast,
)

from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt

from .config import (
    BK_KAPPA,
    BK_RHO,
    BK_SIGMA_MODERATE,
    BK_SIGMA_STRESS,
    BK_THETA,
    CONC_CAP,
    DEFAULT_CONC_CAP_GROWTH_RATE,
    DEFAULT_EDU_SCHEDULE,
    DEFAULT_SIM_START_AGE,
    DIV293_RATE,
    DIV293_THRESHOLD,
    SG_MAX_BASE,
    THEME_COLOR,
    THEME_COLOR_ACCENT,
    THEME_COLOR_BRIGHT,
    THEME_COLOR_ERROR,
    THEME_COLOR_WARN,
)
from .models import (
    AssetClass,
    Child,
    Earner,
    EmploymentType,
    Household,
    InvestmentAccount,
    MortgageAccount,
    OffsetReserveMode,
    Ownership,
    PtRateMode,
    SimulationInputs,
)
from .mortgage import pi_payment
from .returns import ASSET_CLASS_PARAMS
from .ui_common import _prompt_float, _prompt_int, _prompt_pct, _prompt_yn, console

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


def _configure_earner(
    label: str, defaults: Earner | None = None, start_age: int = DEFAULT_SIM_START_AGE
) -> Earner:
    """Prompt for one earner's details.

    Args:
        label: Earner display label.
        defaults: Pre-existing earner to edit.
        start_age: Simulation start age (for accurate warning projections).

    """
    d = defaults or Earner(label=label, super_access_age=60)
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
    emp_types: dict[int, EmploymentType] = {
        1: "employed",
        2: "self_employed",
        3: "both",
        4: "not_employed",
    }
    emp_type = emp_types[emp_choice]

    if emp_type == "self_employed":
        console.print(
            "  [dim]Super Guarantee does not apply to self-employed income —"
            " use personal contributions below to model voluntary super contributions.[/]"
        )

    # ── Birth year (for preservation-age lookup) ─────────────────
    birth_year = None
    if _prompt_yn(
        "  Auto-calculate super access age from birth year?",
        default=defaults is not None and defaults.birth_year is not None or defaults is None,
    ):
        from .models import preservation_age as _preservation_age

        birth_year = _prompt_int("  Birth year", d.birth_year or 1980, lo=1950, hi=2010)
        super_access = _preservation_age(birth_year)
        console.print(f"  [dim]Your preservation age: {super_access}[/]")
        console.print(
            "  [dim]Note: calculated from birth year only (not month)."
            " Actual age may differ by a year if born in a transition year.[/]"
        )
    else:
        super_access = _prompt_int(
            "  Super access age (preservation age — check with your super fund)",
            d.effective_super_access_age,
            lo=55,
            hi=70,
        )

    if emp_type in ("employed", "self_employed", "both"):
        salary = _prompt_float("  Gross annual salary [$]", d.salary, lo=0, hi=5_000_000)
        sal_growth = _prompt_pct(
            "  Real salary growth rate [%] (above inflation)",
            d.salary_growth_rate * 100,
            lo=-5,
            hi=15,
        )
        retire_age = _prompt_int(
            "  Retirement age",
            max(d.retirement_age, start_age + 1),
            lo=start_age + 1,
            hi=75,
        )
        if emp_type in ("employed", "both"):
            sg = _prompt_pct(
                "  Super Guarantee rate [%] (as at 2025\u201326)",
                d.sg_rate * 100,
                lo=0,
                hi=20,
            )
        else:
            sg = 0.0
            console.print(
                "  [dim]SG rate set to 0% for self-employed — "
                "employer SG obligations do not apply to sole traders.[/]"
            )
        super_bal = _prompt_float(
            "  Current super balance [$]", d.super_balance, lo=0, hi=10_000_000
        )

        # ── Self-employed income (for "both" type) ──────────────────
        if emp_type == "both":
            se_income = _prompt_float(
                "  Self-employed / business income [$]",
                d.self_employed_income if d.self_employed_income > 0 else 50_000.0,
                lo=0,
                hi=5_000_000,
            )
            se_growth = _prompt_pct(
                "  Real business income growth rate [%] (above inflation)",
                d.self_employed_growth_rate * 100,
                lo=-5,
                hi=15,
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
        super_bal = _prompt_float(
            "  Current super balance [$]", d.super_balance, lo=0, hi=10_000_000
        )

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
        lo=0,
        hi=100,
    )
    glide_enabled = _prompt_yn(
        "  Enable age-based glidepath (reduce growth near access age)?",
        default=gl is not None and gl.super_glide_end_year is not None,
    )
    if glide_enabled:
        glide_years = _prompt_int(
            "  Glide ends at simulation year (years from simulation start, e.g. 15)",
            gl.super_glide_end_year if gl and gl.super_glide_end_year is not None else 15,
            lo=1,
            hi=50,
        )
        glide_target = _prompt_float(
            "  Target growth allocation at glide end [%] (30 = conservative)",
            gl.super_glide_target_pct if gl else 30.0,
            lo=0,
            hi=100,
        )
    else:
        glide_years = None
        glide_target = 30.0

    sacrifice: float | None = None
    console.print(f"  [{THEME_COLOR_BRIGHT}]Super contribution strategy:[/]")
    console.print(
        f"    [{THEME_COLOR}]1. Auto-max to concessional cap (${CONC_CAP:,.0f} as at 2025\u201326)[/]"
    )
    console.print(f"    [{THEME_COLOR}]2. Set a custom annual amount[/]")
    default_strat = 2 if d.personal_super_contributions_total_p_a is not None else 1
    strat = _prompt_int("  Choice", default=default_strat, lo=1, hi=2)
    if strat == 2:
        sacrifice = _prompt_float(
            "  Annual pre-tax contribution [$]",
            d.personal_super_contributions_total_p_a,
            lo=0,
            hi=100_000,
        )

    non_conc = _prompt_float(
        "  Annual non-concessional (after-tax) contribution [$] (0=none)",
        defaults.non_concessional_contributions_p_a if defaults else 0.0,
        lo=0,
        hi=360_000,
    )

    # ── Part-time work post retirement ────────────────────────────
    pt_days = 0.0
    pt_start: int | None = None
    pt_end: int | None = None
    pt_rate_mode: PtRateMode = "daily_rate"
    pt_daily_rate = 3_000.0
    pt_salary_pct = 0.0
    has_pt = _prompt_yn("  Part-time work after retirement?", default=d.pt_days_per_week > 0)
    if has_pt:
        pt_days = _prompt_float(
            "  Days per week",
            d.pt_days_per_week if d.pt_days_per_week > 0 else 3.0,
            lo=0.5,
            hi=7,
        )
        default_start = (
            retire_age
            if emp_type in ("employed", "self_employed", "both") and retire_age < 999
            else 60
        )
        pt_start = _prompt_int("  Start age", default_start, lo=30, hi=75)
        pt_end = _prompt_int(
            "  End age",
            d.pt_end_age if d.pt_end_age is not None and d.pt_end_age > pt_start else pt_start + 5,
            lo=pt_start + 1,
            hi=80,
        )
        # Ask how to value the part-time work
        rate_mode_raw = Prompt.ask(
            "  Calculate income as",
            choices=["daily_rate", "salary_pct"],
            default=(
                d.pt_rate_mode if d.pt_rate_mode in ("daily_rate", "salary_pct") else "daily_rate"
            ),
        )
        pt_rate_mode = "salary_pct" if rate_mode_raw.strip() == "salary_pct" else "daily_rate"
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
        effective_start = max(
            pt_start,
            (
                retire_age
                if emp_type in ("employed", "self_employed", "both") and retire_age < 999
                else start_age
            ),
        )
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
    _display_earner_super_warnings(
        label,
        salary,
        sg,
        retire_age,
        start_age,
        salary_growth_rate=sal_growth,
        self_employed_income=se_income,
        self_employed_growth_rate=se_growth,
        sacrifice=sacrifice,
    )

    return dataclasses.replace(
        d,
        label=label,
        salary=salary,
        super_balance=super_bal,
        salary_growth_rate=sal_growth,
        retirement_age=retire_age,
        birth_year=birth_year,
        super_access_age=super_access,
        sg_rate=sg,
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
        self_employed_growth_rate=se_growth,
        self_employed_sg_applies=se_sg,
    )


def _configure_child(label: str, defaults: Child | None = None) -> Child:
    """Prompt for one child's details."""
    d = defaults or Child(label=label)
    console.print(f"\n[bold {THEME_COLOR_ACCENT}]  --- {label} ---[/]")
    age = _prompt_int("  Age", d.age, lo=0, hi=25)

    # Three-way education cost choice. A new child (no defaults) gets the default schedule.
    if defaults is None or d.education_schedule == DEFAULT_EDU_SCHEDULE:
        edu_default = 1  # use default schedule
    elif d.education_schedule:
        edu_default = 2  # existing custom schedule → keep custom
    else:
        edu_default = 3  # existing empty schedule → no education costs

    console.print(f"  [{THEME_COLOR_BRIGHT}]Education costs:[/]")
    console.print(f"    [{THEME_COLOR}]1. Use default schedule (~$292k total, age 0-17)[/]")
    console.print(f"    [{THEME_COLOR}]2. Enter custom costs manually[/]")
    console.print(f"    [{THEME_COLOR}]3. No education costs[/]")
    edu_choice = _prompt_int("  Choice", default=edu_default, lo=1, hi=3)

    schedule: tuple[tuple[int, float], ...] = ()
    if edu_choice == 1:
        schedule = DEFAULT_EDU_SCHEDULE
    elif edu_choice == 2:
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
        else:
            console.print("  [dim]No pairs entered: no education costs.[/]")
    # Choice 3 keeps the empty schedule, which the engine reads as no costs.

    return Child(label=label, age=age, education_schedule=schedule)


def _configure_mortgage(label: str, defaults: MortgageAccount | None = None) -> MortgageAccount:
    """Prompt for one mortgage's details."""
    d = defaults or MortgageAccount(label=label)
    console.print(f"\n[bold {THEME_COLOR_ACCENT}]  --- {label} ---[/]")
    principal = _prompt_float(
        "  Remaining balance (outstanding) [$]", d.principal, lo=0, hi=10_000_000
    )
    rate = _prompt_pct("  Interest rate [%]", d.interest_rate * 100, lo=0, hi=15)
    # Interest-only is stored as a zero monthly payment (the engine reads it that way).
    interest_only = _prompt_yn(
        "  Interest-only loan (interest only, no principal repaid)?",
        default=d.monthly_payment == 0,
    )
    pmt = 0.0
    if not interest_only:
        # Suggested payment: principal and interest repaid over 25 years at this rate.
        suggested = float(pi_payment(principal, rate, 25 * 12))
        pmt = _prompt_float(
            f"  Monthly payment [$] (principal + interest; 25-year suggestion ${suggested:,.0f})",
            d.monthly_payment if d.monthly_payment > 0 else suggested,
            lo=1,
            hi=1_000_000,
        )
        if pmt * 12 < principal * rate:
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
        "  Choice",
        default=mode_default,
        lo=1,
        hi=3,
    )
    if mode_choice == 1:
        mode: OffsetReserveMode = "fixed"
        floor = 0.0
    elif mode_choice == 2:
        mode = "stall_prevention"
        floor = 0.0  # unused
        console.print(
            "  [dim]Floor recalculated each period: max(0, balance \u2014"
            " payment / monthly rate). Loan will not grow but will not"
            " reduce from interest savings alone.[/]"
        )
    else:
        mode = "interest_cancelling"
        floor = 0.0  # unused
        console.print(
            "  [dim]Floor set to full mortgage principal each period."
            " All offset balance is preserved for this mortgage, making"
            " 100% of every payment go to principal. No offset available"
            " for other expenses until mortgage clears.[/]"
        )

    # ── Stochastic mortgage rate prompts ─────────────────────────────
    stoch = _prompt_yn(
        "  Apply stochastic rate variation (Black-Karasinski process)?",
        default=d.interest_rate_stochastic,
    )
    vol = None
    kappa = None
    corr = None
    theta_v = None
    if stoch:
        console.print("  [dim]Press Enter to accept each recommended default.[/]")
        vol = _prompt_pct(
            "  Yearly swing in the rate [%] "
            f"(default {BK_SIGMA_MODERATE * 100:.0f}% = moderate, {BK_SIGMA_STRESS * 100:.0f}% = stress)",
            (d.interest_rate_vol if d.interest_rate_vol is not None else BK_SIGMA_MODERATE) * 100,
            lo=0,
            hi=50,
        )
        kappa = _prompt_pct(
            "  How fast the rate returns to its average [% of the gap closed each year] "
            f"(default {BK_KAPPA * 100:.0f}%)",
            (d.interest_rate_kappa if d.interest_rate_kappa is not None else BK_KAPPA) * 100,
            lo=1,
            hi=100,
        )
        theta_v = _prompt_pct(
            f"  Long-run average mortgage rate [%] (default {BK_THETA * 100:.1f}%)",
            (d.interest_rate_theta if d.interest_rate_theta is not None else BK_THETA) * 100,
            lo=1.0,
            hi=15.0,
        )
        corr = _prompt_pct(
            "  Correlation with sharemarket returns [%] "
            f"(default {BK_RHO * 100:.0f}%, 0% = uncorrelated)",
            (d.interest_rate_corr if d.interest_rate_corr is not None else BK_RHO) * 100,
            lo=-50,
            hi=50,
        )

    term = _prompt_int(
        "  Loan must be repaid by age (0 = no term check)",
        d.loan_term_end_age or 0,
        lo=0,
        hi=100,
    )
    reprice = False
    if term > 0 and not interest_only:
        reprice = _prompt_yn(
            "  Recalculate the payment over the remaining term when the rate changes?",
            default=d.reprice_payment,
        )
    return dataclasses.replace(
        d,
        label=label,
        principal=principal,
        interest_rate=rate,
        monthly_payment=pmt,
        offset_reserve_mode=mode,
        offset_reserve_floor=floor,
        loan_term_end_age=term if term > 0 else None,
        reprice_payment=reprice,
        interest_rate_stochastic=stoch,
        interest_rate_vol=vol,
        interest_rate_kappa=kappa,
        interest_rate_theta=theta_v,
        interest_rate_corr=corr,
    )


def _configure_account(
    label: str,
    defaults: InvestmentAccount | None = None,
    *,
    num_earners: int = 1,
    earner_labels: list[str] | None = None,
) -> InvestmentAccount:
    """Prompt for one investment account's details."""
    d = defaults or InvestmentAccount(label=label)
    console.print(f"\n[bold {THEME_COLOR_ACCENT}]  --- {label} ---[/]")
    value = _prompt_float("  Current market value [$]", d.market_value, lo=0, hi=10_000_000)
    is_offset = _prompt_yn("  Is this a mortgage offset account?", d.is_offset)

    # Only ask for cost basis if NOT an offset account (no CGT on offsets)
    basis = 0.0
    if not is_offset:
        basis = _prompt_float("  Original purchase value [$]", d.cost_basis, lo=0, hi=10_000_000)

    asset_class: AssetClass = "cash" if is_offset else "equity"
    # Only ask for asset class and return rate if not an offset account
    interest_rate = d.interest_rate
    if not is_offset:
        # ── Asset class selection ───────────────────────────────────
        console.print(f"  [{THEME_COLOR_BRIGHT}]Asset class:[/]")
        classes = list(ASSET_CLASS_PARAMS.keys())
        for i, ac in enumerate(classes, 1):
            params = ASSET_CLASS_PARAMS[ac]
            console.print(
                f"    [{THEME_COLOR}]{i}. {ac:<15} ~{params['geo_mean'] * 100:.0f}% return, {params['std'] * 100:.0f}% risk[/]"
            )
        ac_default = classes.index(d.asset_class) + 1 if d.asset_class in classes else 1
        ac_choice = _prompt_int("  Choice", default=ac_default, lo=1, hi=len(classes))
        asset_class = cast(AssetClass, classes[ac_choice - 1])
        # Optional return override; "no" keeps the asset class's own expected return.
        if _prompt_yn(
            "  Override the asset class expected return?", default=d.interest_rate is not None
        ):
            default_return = (
                d.interest_rate
                if d.interest_rate is not None
                else ASSET_CLASS_PARAMS[asset_class]["geo_mean"]
            )
            interest_rate = _prompt_pct(
                "  Expected annual return [%] (real, after inflation)",
                default_return * 100,
                lo=0,
                hi=20,
            )
        else:
            interest_rate = None

    # ── Ownership split prompt (multi-earner, non-offset only) ──────
    ownership: Ownership = Ownership({0: 1.0})
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
                    lo=0,
                    hi=remaining,
                )
            shares[ei] = pct / 100
            remaining -= pct
        ownership = Ownership(shares)

    return dataclasses.replace(
        d,
        label=label,
        market_value=value,
        cost_basis=basis,
        asset_class=asset_class,
        is_offset=is_offset,
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
            "How many earners? [1-10]",
            existing.num_earners if existing else 1,
            lo=1,
            hi=10,
        )

    earners: list[Earner] = []
    for i in range(num_earners):
        label = f"Earner {i + 1}"
        existing_e = existing.earners[i] if existing and i < len(existing.earners) else None
        # Earner 1 follows the simulation start age; later earners have their own current age.
        earner_age = start_age
        if i > 0:
            earner_age = _prompt_int(
                f"  {label} current age",
                existing_e.resolved_start_age(start_age) if existing_e else start_age,
                lo=0,
                hi=100,
            )
        if preset == "single_income" and i == 1:
            existing_e = existing_e or Earner(
                label=label,
                salary=0,
                employment_type="not_employed",
                super_access_age=60,
                retirement_age=999,
            )
        e = _configure_earner(label, existing_e, earner_age)
        earners.append(dataclasses.replace(e, start_age=earner_age if i > 0 else None))

    for e in earners:
        if e.employment_type == "not_employed" and e.salary > 0:
            console.print(
                f"[yellow]Warning: Earner '{escape(e.label)}' has "
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
    has_mortgage = _prompt_yn(
        "Any mortgages?", bool(existing and existing.mortgages) if existing else True
    )
    if not has_mortgage:
        return ()
    num_m = _prompt_int(
        "How many mortgages?",
        len(existing.mortgages) if existing and existing.mortgages else 1,
        lo=1,
        hi=5,
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
    if preset == "custom":
        has_kids = _prompt_yn("Any children?", bool(existing and existing.children))
    else:
        has_kids = preset == "family"
    if not has_kids:
        return ()
    num_kids = _prompt_int(
        "How many children?",
        len(existing.children) if existing and existing.children else 1,
        lo=1,
        hi=20,
    )
    for i in range(num_kids):
        label = f"Child {i + 1}"
        existing_c = existing.children[i] if existing and i < len(existing.children) else None
        children.append(_configure_child(label, existing_c))
    return tuple(children)


def _configure_accounts_section(
    existing: Household | None,
    *,
    earners: tuple[Earner, ...] | None = None,
) -> tuple[InvestmentAccount, ...]:
    """Configure all investment accounts, reusing existing values as defaults.

    Args:
        existing: Pre-existing household (for account defaults).
        earners: The freshly configured earners.  When ``None`` (legacy
            callers), falls back to ``existing.earners``.  This parameter
            is required so that the ownership-split prompt sees the
            *new* earners rather than the pre-edit ones.

    """
    # Determine earner count and labels for ownership split prompt.
    # Use the freshly built earners when provided (H1 fix); fall back to
    # the existing household for backward compatibility with edit_household.
    active_earners = (
        earners
        if earners is not None
        else (existing.earners if existing and existing.earners else (Earner(super_access_age=60),))
    )
    n_earners = len(active_earners)
    earner_labels = [e.label for e in active_earners]

    accounts: list[InvestmentAccount] = []
    has_accounts = _prompt_yn(
        "Any investment accounts?", bool(existing and existing.investment_accounts)
    )
    if not has_accounts:
        return ()
    num_a = _prompt_int(
        "How many accounts?",
        len(existing.investment_accounts) if existing and existing.investment_accounts else 1,
        lo=1,
        hi=10,
    )
    for i in range(num_a):
        label = f"Account {i + 1}"
        existing_a = (
            existing.investment_accounts[i]
            if existing and i < len(existing.investment_accounts)
            else None
        )
        accounts.append(
            _configure_account(
                label,
                existing_a,
                num_earners=n_earners,
                earner_labels=earner_labels,
            )
        )
    return tuple(accounts)


def _link_offsets_to_mortgages(
    accounts: list[InvestmentAccount],
    mortgages: list[MortgageAccount],
) -> list[MortgageAccount]:
    """Link offset accounts to their target mortgages."""
    offset_labels = [a.label for a in accounts if a.is_offset]
    # Keep only links to offset accounts that still exist, so repeated edits never duplicate.
    result = [
        dataclasses.replace(
            m, offset_accounts=tuple(o for o in m.offset_accounts if o in offset_labels)
        )
        for m in mortgages
    ]
    if offset_labels and result:
        if len(result) == 1:
            m = result[0]
            result[0] = dataclasses.replace(m, offset_accounts=tuple(offset_labels))
        else:
            assigned = {o for m in result for o in m.offset_accounts}
            for ol in offset_labels:
                if ol in assigned:
                    continue
                console.print(f"  [dim]Which mortgage does '{escape(ol)}' offset?[/]")
                for mi, m in enumerate(result, 1):
                    console.print(f"    {mi}. {escape(m.label)}")
                choice = _prompt_int("  Choice", default=1, lo=1, hi=len(result))
                idx = choice - 1
                m = result[idx]
                result[idx] = dataclasses.replace(m, offset_accounts=(*m.offset_accounts, ol))
    return result


def _configure_expenses_section(
    existing: Household | None,
) -> tuple[float, float]:
    """Configure household living expenses."""
    console.print(f"[bold {THEME_COLOR_ACCENT}]  --- Household Expenses ---[/]")
    living = _prompt_float(
        "  Base annual living expenses [$]",
        existing.base_living_expenses if existing else 60_000.0,
        lo=10_000,
        hi=1_000_000,
    )
    target = _prompt_float(
        "  Spending in retirement [$]",
        existing.retirement_target if existing else 80_000.0,
        lo=10_000,
        hi=1_000_000,
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
    accounts = _configure_accounts_section(existing, earners=earners)
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
        console.print(
            Panel(
                choices_text,
                title="Edit Profile",
                border_style=THEME_COLOR_ACCENT,
            )
        )
        choice = _prompt_int("Choice", default=6, lo=1, hi=6)
        if choice == 1:
            earners = list(
                _configure_earners_section(
                    "custom",
                    _make_dummy_household(earners, mortgages, children, accounts, living, target),
                    start_age,
                )
            )
        elif choice == 2:
            mortgages = list(
                _configure_mortgages_section(
                    _make_dummy_household(earners, mortgages, children, accounts, living, target)
                )
            )
            # Re-link offset accounts after mortgage changes (silent if only one mortgage)
            mortgages = _link_offsets_to_mortgages(accounts, mortgages)
        elif choice == 3:
            children = list(
                _configure_children_section(
                    "custom",
                    _make_dummy_household(earners, mortgages, children, accounts, living, target),
                )
            )
        elif choice == 4:
            accounts = list(
                _configure_accounts_section(
                    _make_dummy_household(earners, mortgages, children, accounts, living, target),
                    earners=tuple(earners),
                )
            )
            mortgages = _link_offsets_to_mortgages(accounts, mortgages)
        elif choice == 5:
            living, target = _configure_expenses_section(
                _make_dummy_household(earners, mortgages, children, accounts, living, target)
            )
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
    earners: list[Earner],
    mortgages: list[MortgageAccount],
    children: list[Child],
    accounts: list[InvestmentAccount],
    living: float,
    target: float,
) -> Household:
    """Build a throwaway Household from current in-progress values for passing as defaults."""
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
        existing: Pre-existing inputs to edit (when loading a profile). Settings
                  not asked here (seed, sell order, ...) are preserved from it.
        household: The household definition. When given it replaces
                   ``existing.household``, so an edited household is never lost.
        start_age: Pre-collected simulation start age. If provided, the
                   start age prompt is skipped. Falls back to existing
                   value or config default.

    Returns:
        A new ``SimulationInputs`` with user-provided values.

    """
    base = existing if existing is not None else SimulationInputs()
    # Determine effective start age: parameter > existing > default
    if start_age is None:
        start_age = base.simulation_start_age
    if household is None:
        household = base.household

    console.print(f"\n[bold {THEME_COLOR_ACCENT}]--- Simulation Parameters ---[/]")
    iters = _prompt_int(
        "  Number of simulation runs",
        base.n_iterations,
        lo=100,
        hi=100_000,
    )
    infl = _prompt_pct(
        "  Inflation rate [%]",
        base.inflation * 100,
        lo=0,
        hi=10,
    )
    cgt = _prompt_yn(
        "  Apply CGT when selling investments?",
        base.cgt_on_drawdowns,
    )

    # ── Super fee rate ───────────────────────────────────────────
    super_fee = _prompt_pct(
        "  Superannuation fee rate [%] (0.85 = median AusSuper fund)",
        existing.super_fee_rate * 100 if existing else 0.85,
        lo=0.0,
        hi=5.0,
    )

    # ── Surplus allocation ──────────────────────────────────────────
    # Only ask when there are both offset and non-offset accounts
    surplus_pct = base.surplus_investment_pct
    has_offset = any(a.is_offset for a in household.investment_accounts)
    has_investment = any(not a.is_offset for a in household.investment_accounts)
    if has_offset and has_investment:
        console.print()
        surplus_pct = _prompt_float(
            "  % of surplus to investments each year, even if offset isn't full yet (rest to offset/cash)",
            base.surplus_investment_pct,
            lo=0,
            hi=100,
        )
    elif not has_offset and has_investment:
        # No offset accounts — all surplus naturally goes to investments
        surplus_pct = 100.0

    # ── Success threshold ───────────────────────────────
    success_th = _prompt_pct(
        "  Target success probability [%] (e.g. 90 = 90%)",
        base.success_threshold * 100,
        lo=50,
        hi=100,
    )

    # ── Advanced assumptions (optional) ─────────────────────────────
    start_year = base.simulation_start_year
    stoch_infl = base.stochastic_inflation
    mls_enabled = base.mls_enabled
    mls_tiered = base.mls_tiered
    mls_threshold = base.mls_threshold
    mls_rate = base.mls_rate
    bracket_growth = base.bracket_growth_rate
    if _prompt_yn(
        "  Change advanced assumptions (start year, inflation model, Medicare levy surcharge,"
        " tax bracket indexation)?",
        default=False,
    ):
        start_year = _prompt_int(
            "  Calendar year of simulation year 0 (start of financial year)",
            base.simulation_start_year,
            lo=2000,
            hi=2200,
        )
        stoch_infl = _prompt_yn(
            "  Use stochastic inflation (random yearly inflation, correlated with markets)?",
            default=base.stochastic_inflation,
        )
        mls_enabled = _prompt_yn(
            "  Model the Medicare levy surcharge (applies without private hospital cover)?",
            default=base.mls_enabled,
        )
        if mls_enabled:
            mls_tiered = _prompt_yn(
                "  Use the progressive MLS tiers (No = one flat rate)?",
                default=base.mls_tiered,
            )
            mls_threshold = _prompt_float(
                "  MLS income threshold, singles [$]",
                base.mls_threshold,
                lo=0,
                hi=1_000_000,
            )
            if not mls_tiered:
                mls_rate = _prompt_pct("  Flat MLS rate [%]", base.mls_rate * 100, lo=0, hi=5)
        bracket_growth = _prompt_pct(
            "  Tax bracket indexation [%] per year (0 = frozen brackets)",
            base.bracket_growth_rate * 100,
            lo=0,
            hi=10,
        )

    # Build overrides from user input
    overrides: dict[str, Any] = dict(
        household=household,
        n_iterations=iters,
        inflation=infl,
        simulation_start_age=start_age,
        simulation_start_year=start_year,
        stochastic_inflation=stoch_infl,
        mls_enabled=mls_enabled,
        mls_tiered=mls_tiered,
        mls_threshold=mls_threshold,
        mls_rate=mls_rate,
        bracket_growth_rate=bracket_growth,
        cgt_on_drawdowns=cgt,
        surplus_investment_pct=surplus_pct,
        super_fee_rate=super_fee,
        success_threshold=success_th,
    )

    if existing is None:
        return SimulationInputs(**overrides)

    # Preserve ALL existing fields, overriding only what the user changed
    # dataclasses.replace keeps sell_order, seed, and anything not asked above.
    return dataclasses.replace(existing, **overrides)


# =============================================================================
# SUPER/TAX WARNINGS
# =============================================================================


def _display_earner_super_warnings(
    earner_label: str,
    salary: float,
    sg_rate: float,
    retirement_age: int,
    start_age: int = DEFAULT_SIM_START_AGE,
    salary_growth_rate: float = 0.0,
    *,
    self_employed_income: float = 0.0,
    self_employed_growth_rate: float = 0.0,
    sacrifice: float | None = None,
) -> None:
    """Display real-time warnings about superannuation caps and Division 293.

    Uses the earner's own growth rates and the config-default cap indexation
    rates, matching the engine's projection rules.

    Args:
        earner_label: The earner's label.
        salary: Annual salary.
        sg_rate: Super Guarantee rate (decimal).
        retirement_age: Earner's retirement age.
        start_age: Simulation start age (default: ``DEFAULT_SIM_START_AGE``).
        salary_growth_rate: Earner's annual salary growth (decimal).
        self_employed_income: Current annual business income.
        self_employed_growth_rate: Business income growth (decimal).
        sacrifice: Pre-tax contribution per year, or ``None`` for auto-max.

    """
    warnings = _super_warning_messages(
        earner_label=earner_label,
        salary=salary,
        sg_rate=sg_rate,
        years=retirement_age - start_age,
        salary_growth_rate=salary_growth_rate,
        self_employed_income=self_employed_income,
        self_employed_growth_rate=self_employed_growth_rate,
        sacrifice=sacrifice,
    )
    if warnings:
        console.print()
        console.print(
            Panel(
                "\n".join(warnings),
                title="Superannuation & Tax Alerts",
                border_style=THEME_COLOR_WARN,
            )
        )


def _super_warning_messages(
    earner_label: str,
    salary: float,
    sg_rate: float,
    years: int,
    salary_growth_rate: float = 0.0,
    conc_cap_growth_rate: float = DEFAULT_CONC_CAP_GROWTH_RATE,
    sg_max_base_growth_rate: float = DEFAULT_CONC_CAP_GROWTH_RATE,
    *,
    self_employed_income: float = 0.0,
    self_employed_growth_rate: float = 0.0,
    sacrifice: float | None = None,
) -> list[str]:
    """Build warning messages about super caps and Div 293 for an earner.

    Projects with the same rules the engine uses: salary and business income at
    their own growth rates, the concessional cap and SG maximum base at their
    configured indexation rates, and the Division 293 threshold left statutory
    (the engine never indexes it).

    Args:
        earner_label: Earner display name.
        salary: Current annual salary.
        sg_rate: Super Guarantee rate (decimal).
        years: Years between start age and retirement (for indexation).
        salary_growth_rate: Earner's annual salary growth (decimal).
        conc_cap_growth_rate: Concessional cap indexation rate.
        sg_max_base_growth_rate: SG maximum salary base indexation rate.
        self_employed_income: Current annual business income (no SG).
        self_employed_growth_rate: Business income growth (decimal).
        sacrifice: Pre-tax contribution per year. ``None`` means auto-max,
            topped up to the indexed concessional cap as the engine does.

    Returns:
        List of warning strings (empty if none triggered).

    """
    # An earner with no income has nothing to project.
    if salary <= 0 and self_employed_income <= 0:
        return []

    warnings: list[str] = []

    years = max(0, years)

    # Projected values using the engine's indexation bases.
    indexed_conc_cap = CONC_CAP * (1 + conc_cap_growth_rate) ** years
    indexed_sg_max = SG_MAX_BASE * (1 + sg_max_base_growth_rate) ** years
    indexed_salary = salary * (1 + salary_growth_rate) ** years
    indexed_se = self_employed_income * (1 + self_employed_growth_rate) ** years
    sg_contribution = min(indexed_salary, indexed_sg_max) * sg_rate
    if sacrifice is None:
        sacrifice = max(0.0, indexed_conc_cap - sg_contribution)

    if sg_contribution > indexed_conc_cap:
        warnings.append(
            f"  ⚠️  [bold {THEME_COLOR_WARN}]SG contribution ({sg_contribution:,.0f}) "
            f"exceeds concessional cap ({indexed_conc_cap:,.0f}). "
            f"Excess may be subject to excess contributions tax.[/]"
        )

    # Division 293 mirrors the engine: taxable income is after salary sacrifice,
    # and Div 293 income adds the concessional contributions back.
    # Division 293 threshold is statutory at $250,000 — it is never indexed.
    taxable = indexed_salary + indexed_se - sacrifice
    div293_income = taxable + sg_contribution + sacrifice
    if div293_income > DIV293_THRESHOLD and sg_contribution + sacrifice > 0:
        warnings.append(
            f"  ⚠️  [bold {THEME_COLOR_WARN}]Division 293 applies for {escape(earner_label)} "
            f"(income ${div293_income:,.0f} > threshold ${DIV293_THRESHOLD:,.0f}). "
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
        warnings.extend(
            _super_warning_messages(
                earner_label=earner.label,
                salary=earner.salary,
                sg_rate=earner.sg_rate,
                years=earner.retirement_age - earner.resolved_start_age(start_age),
                salary_growth_rate=earner.salary_growth_rate,
                self_employed_income=earner.self_employed_income,
                self_employed_growth_rate=earner.self_employed_growth_rate,
                sacrifice=earner.personal_super_contributions_total_p_a,
            )
        )

    if warnings:
        console.print()
        console.print(
            Panel(
                "\n".join(warnings),
                title="Superannuation & Tax Alerts",
                border_style=THEME_COLOR_WARN,
            )
        )


# =============================================================================
# REVIEW BEFORE RUN
# =============================================================================


ReviewAction = Literal["run", "household", "simulation", "cancel"]


def review_before_run(
    household: Household,
    sim_inputs: SimulationInputs,
) -> ReviewAction:
    """Show every assumption that changes the result, then ask what to do next.

    Returns:
        ``"run"`` to simulate, ``"household"`` or ``"simulation"`` to edit that
        part and review again, or ``"cancel"`` to return to the main menu.

    """
    start_age = sim_inputs.simulation_start_age
    bridge_end = min(e.effective_super_access_age for e in household.earners)
    employed_retirements = [e.retirement_age for e in household.earners if e.retirement_age < 999]
    bridge_start = min(employed_retirements) if employed_retirements else start_age
    bridge_years = bridge_end - bridge_start

    console.print()

    # ── Household summary ───────────────────────────────────────────
    emp_labels = {
        "employed": "Employed",
        "self_employed": "Self-employed",
        "both": "Salary + self-employed",
        "not_employed": "Not employed",
    }
    hh_lines: list[str] = []
    hh_lines.append(f"[bold]Earners ({household.num_earners}):[/]")
    for e in household.earners:
        emp = emp_labels.get(e.employment_type, "Employed")
        pt_info = ""
        if e.pt_days_per_week > 0:
            pt_info = (
                f", PT {e.pt_days_per_week:.1f}d/wk "
                f"({e.effective_pt_start_age}–{e.effective_pt_end_age})"
            )
        income_parts = [f"salary ${e.salary:,.0f}"]
        if e.employment_type == "both" and e.self_employed_income > 0:
            income_parts.append(f"self-employed ${e.self_employed_income:,.0f}")
        elif e.employment_type == "self_employed":
            income_parts = [f"self-employed ${e.salary:,.0f}"]
        income_str = " + ".join(income_parts)
        hh_lines.append(
            f"  {escape(e.label)}: {emp}, {income_str}, "
            f"current age {e.resolved_start_age(start_age)}, "
            f"retire at {e.retirement_age}, super access at {e.effective_super_access_age}"
            f"{pt_info}"
        )
        contrib = (
            "auto-max to concessional cap"
            if e.personal_super_contributions_total_p_a is None
            else f"pre-tax ${e.personal_super_contributions_total_p_a:,.0f}/yr"
        )
        glide = (
            f"glide to {e.super_glide_target_pct:.0f}% by simulation year {e.super_glide_end_year}"
            if e.super_glide_end_year is not None
            else "no glidepath"
        )
        hh_lines.append(
            f"    Super: {contrib}, non-concessional ${e.non_concessional_contributions_p_a:,.0f}/yr, "
            f"{e.super_growth_pct:.0f}% growth assets, {glide}"
        )
    if household.children:
        hh_lines.append(f"[bold]Children ({household.num_children}):[/]")
        for c in household.children:
            if not c.education_schedule:
                edu = "no education costs"
            elif c.education_schedule == DEFAULT_EDU_SCHEDULE:
                edu = "default schedule"
            else:
                edu = "custom schedule"
            hh_lines.append(f"  {escape(c.label)}: age {c.age}, {edu}")
    if household.mortgages:
        hh_lines.append(f"[bold]Mortgages ({household.num_mortgages}):[/]")
        for m in household.mortgages:
            stoch_tag = " [dim](stochastic rates)[/dim]" if m.interest_rate_stochastic else ""
            if m.monthly_payment == 0:
                payment = "interest-only"
            else:
                payment = f"${m.monthly_payment:,.0f}/month"
                if m.reprice_payment:
                    payment += ", recalculated on rate change"
            term = f", repaid by age {m.loan_term_end_age}" if m.loan_term_end_age else ""
            offsets = (
                f", offset: {escape(', '.join(m.offset_accounts))}" if m.offset_accounts else ""
            )
            hh_lines.append(
                f"  {escape(m.label)}: ${m.principal:,.0f} at {m.interest_rate * 100:.2f}%"
                f"{stoch_tag}, {payment}{term}{offsets}"
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
                        owner_parts.append(
                            f"{escape(household.earners[ei].label)} {share * 100:.0f}%"
                        )
            owner_str = f" — {' / '.join(owner_parts)}" if owner_parts else ""
            if a.is_offset:
                returns = ""
            elif a.interest_rate is None:
                returns = f", {a.asset_class}, asset-class return"
            else:
                returns = f", {a.asset_class}, return {a.interest_rate * 100:.1f}% real"
            hh_lines.append(f"  {escape(a.label)}: ${a.market_value:,.0f}{tag}{returns}{owner_str}")
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

    console.print(
        Panel(
            "\n".join(hh_lines),
            title="Household Summary",
            border_style=THEME_COLOR_ACCENT,
        )
    )

    # ── Simulation summary ──────────────────────────────────────────
    if not sim_inputs.mls_enabled:
        mls = "off"
    elif sim_inputs.mls_tiered:
        mls = f"progressive tiers, threshold ${sim_inputs.mls_threshold:,.0f} (singles)"
    else:
        mls = (
            f"flat {sim_inputs.mls_rate * 100:.2f}%, "
            f"threshold ${sim_inputs.mls_threshold:,.0f} (singles)"
        )
    console.print(
        Panel(
            f"Trials: {sim_inputs.n_iterations:,}\n"
            f"Calendar year of simulation year 0: {sim_inputs.simulation_start_year}\n"
            f"Start age: {start_age}; bridge period: age {bridge_start} \u2192 {bridge_end} "
            f"({bridge_years} years)\n"
            f"Inflation: {sim_inputs.inflation * 100:.1f}% "
            f"({'stochastic' if sim_inputs.stochastic_inflation else 'fixed'})\n"
            f"Tax bracket indexation: {sim_inputs.bracket_growth_rate * 100:.1f}%/yr\n"
            f"Medicare levy surcharge: {mls}\n"
            f"Super fee rate: {sim_inputs.super_fee_rate * 100:.2f}%/yr\n"
            f"Surplus to investments: {sim_inputs.surplus_investment_pct:.0f}% "
            "(when offsets are not yet full)\n"
            f"CGT on withdrawals: {'Yes' if sim_inputs.cgt_on_drawdowns else 'No'}\n"
            f"Stochastic mortgage rates: "
            f"{'Yes' if any(m.interest_rate_stochastic for m in household.mortgages) else 'No'}\n"
            f"Target success probability: {sim_inputs.success_threshold * 100:.1f}%",
            title="Simulation Parameters",
            border_style=THEME_COLOR_ACCENT,
        )
    )

    console.print(
        Panel(
            "[bold]What next?[/]\n\n"
            "  1. Run simulation\n"
            "  2. Edit household (people, mortgages, children, accounts, expenses)\n"
            "  3. Edit simulation settings\n"
            "  4. Cancel (back to main menu)\n",
            border_style=THEME_COLOR_ACCENT,
        )
    )
    choice = _prompt_int("Choice", default=1, lo=1, hi=4)
    actions: dict[int, ReviewAction] = {1: "run", 2: "household", 3: "simulation", 4: "cancel"}
    return actions[choice]
